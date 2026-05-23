#!/usr/bin/env python3
"""Focused signed relative-movement feasibility check for EURUSD/GBPUSD.

Descriptive-only study:
- event/context statistics
- signed relative movement and convergence direction
- first-touch outcomes
- friction-adjusted behavior
- sequence stress
- OOS stability
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


WINDOW = 50
HORIZON_BARS = 40
FIXED_SESSION_BUCKET = "New York open"
FIXED_HOUR = 15
FIXED_ABS_ZSCORE_BUCKET = "1_2"

SPREAD_A_PIPS = 1.2
SPREAD_B_PIPS = 1.8
SLIPPAGE_A_PIPS = 0.3
SLIPPAGE_B_PIPS = 0.5
COMMISSION_EQUIVALENT_PIPS_TOTAL = 0.6
ESTIMATED_ROUND_TRIP_FRICTION_PIPS = (
    2 * (SPREAD_A_PIPS + SPREAD_B_PIPS)
    + 2 * (SLIPPAGE_A_PIPS + SLIPPAGE_B_PIPS)
    + COMMISSION_EQUIVALENT_PIPS_TOTAL
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-context signed relative-movement feasibility check."
    )
    parser.add_argument("--csv-a", default="data/EURUSD_M15_MT5_5Y.csv")
    parser.add_argument("--csv-b", default="data/GBPUSD_M15_MT5_5Y.csv")
    parser.add_argument("--pair-name", default="EURUSD_GBPUSD")
    parser.add_argument("--output-dir", default="signed_relative_micro_reports")
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    datetime_col = cols.get("datetime") or cols.get("time") or cols.get("date")
    close_col = cols.get("close")
    if datetime_col is None or close_col is None:
        raise ValueError(f"{prefix}: required columns missing. Need datetime/time/date and close.")

    out = df[[datetime_col, close_col]].copy()
    out.columns = ["datetime", f"close_{prefix}"]
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    out = out.dropna(subset=["datetime", f"close_{prefix}"])
    out = out.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)
    return out


def session_bucket_from_hour(hour: int) -> str:
    if 0 <= hour <= 3:
        return "Asia early"
    if 4 <= hour <= 6:
        return "Asia late"
    if 7 <= hour <= 9:
        return "London open"
    if 10 <= hour <= 12:
        return "London mid"
    if 13 <= hour <= 15:
        return "New York open"
    if 16 <= hour <= 18:
        return "New York mid"
    return "Late session"


def abs_bucket(z: float) -> str:
    az = abs(z)
    if az < 1:
        return "lt_1"
    if az < 2:
        return "1_2"
    if az < 3:
        return "2_3"
    return "ge_3"


def safe_median(series: pd.Series) -> float:
    return float(series.median()) if not series.empty else np.nan


def first_index_true(mask: np.ndarray) -> Optional[int]:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_a = pd.read_csv(args.csv_a)
    raw_b = pd.read_csv(args.csv_b)

    a = normalize_columns(raw_a, "a")
    b = normalize_columns(raw_b, "b")

    df = a.merge(b, on="datetime", how="inner")

    df["log_price_a"] = np.log(df["close_a"])
    df["log_price_b"] = np.log(df["close_b"])
    df["return_a"] = df["log_price_a"].diff()
    df["return_b"] = df["log_price_b"].diff()

    cov_ab = df["log_price_a"].rolling(WINDOW).cov(df["log_price_b"])
    var_b = df["log_price_b"].rolling(WINDOW).var()
    df["rolling_beta_50"] = cov_ab / var_b.replace(0, np.nan)

    df["relative_spread_50"] = df["log_price_a"] - df["rolling_beta_50"] * df["log_price_b"]
    df["spread_mean_50"] = df["relative_spread_50"].rolling(WINDOW).mean()
    df["spread_std_50"] = df["relative_spread_50"].rolling(WINDOW).std()
    df["spread_zscore_50"] = (
        (df["relative_spread_50"] - df["spread_mean_50"]) / df["spread_std_50"].replace(0, np.nan)
    )

    df["hour"] = df["datetime"].dt.hour
    df["session_bucket"] = df["hour"].map(session_bucket_from_hour)
    df["year"] = df["datetime"].dt.year
    df["abs_zscore_bucket_50"] = df["spread_zscore_50"].map(abs_bucket)

    context_df = df[
        (df["session_bucket"] == FIXED_SESSION_BUCKET)
        & (df["hour"] == FIXED_HOUR)
        & (df["abs_zscore_bucket_50"] == FIXED_ABS_ZSCORE_BUCKET)
        & df["spread_zscore_50"].notna()
    ].copy()

    records = []
    z = df["spread_zscore_50"].to_numpy()
    rs = df["relative_spread_50"].to_numpy()
    ca = df["close_a"].to_numpy()
    cb = df["close_b"].to_numpy()

    for idx in context_df.index:
        current_z = z[idx]
        if np.isnan(current_z):
            continue

        f_start = idx + 1
        f_end = min(len(df), idx + 1 + HORIZON_BARS)
        future = z[f_start:f_end]
        if future.size == 0:
            continue

        if current_z > 0:
            norm_mask = future <= 0.5
        elif current_z < 0:
            norm_mask = future >= -0.5
        else:
            norm_mask = np.ones_like(future, dtype=bool)

        div_mask = np.abs(future) >= 3.0

        norm_i = first_index_true(norm_mask)
        div_i = first_index_true(div_mask)

        if norm_i is not None and div_i is not None:
            if norm_i < div_i:
                label = "normalized_0_5_first"
            elif div_i < norm_i:
                label = "moved_further_to_3_first"
            else:
                label = "both_same_bar"
        elif norm_i is not None:
            label = "normalized_0_5_first"
        elif div_i is not None:
            label = "moved_further_to_3_first"
        else:
            label = "neither"

        bars_to_first_touch = np.nan
        direction = "not_applicable"
        signed_log_move = np.nan
        signed_pips_proxy = np.nan
        signed_after_friction = np.nan

        if label == "normalized_0_5_first" and norm_i is not None:
            touch_idx = f_start + norm_i
            bars_to_first_touch = norm_i + 1

            cur_rs = rs[idx]
            touch_rs = rs[touch_idx]

            if current_z > 0:
                signed_log_move = cur_rs - touch_rs
            else:
                signed_log_move = touch_rs - cur_rs

            if signed_log_move <= 0:
                direction = "adverse_or_flat"
            else:
                direction = "favorable"

            signed_a = signed_log_move * ca[idx] / 0.0001
            signed_b = signed_log_move * cb[idx] / 0.0001
            signed_pips_proxy = min(abs(signed_a), abs(signed_b))
            signed_after_friction = signed_pips_proxy - ESTIMATED_ROUND_TRIP_FRICTION_PIPS

        records.append(
            {
                "datetime": df.at[idx, "datetime"],
                "year": int(df.at[idx, "year"]),
                "current_z": current_z,
                "first_touch_0_5_vs_3": label,
                "bars_to_first_touch": bars_to_first_touch,
                "signed_convergence_direction": direction,
                "signed_convergence_log_move": signed_log_move,
                "signed_convergence_pips_proxy": signed_pips_proxy,
                "estimated_round_trip_friction_pips": ESTIMATED_ROUND_TRIP_FRICTION_PIPS,
                "signed_convergence_after_friction": signed_after_friction,
            }
        )

    events = pd.DataFrame(records).sort_values("datetime").reset_index(drop=True)

    observations = len(events)
    normalized_mask = events["first_touch_0_5_vs_3"] == "normalized_0_5_first" if observations else pd.Series(dtype=bool)
    moved_mask = events["first_touch_0_5_vs_3"] == "moved_further_to_3_first" if observations else pd.Series(dtype=bool)
    neither_mask = events["first_touch_0_5_vs_3"] == "neither" if observations else pd.Series(dtype=bool)

    norm_events = events[normalized_mask].copy() if observations else pd.DataFrame()

    summary = {
        "observations": observations,
        "normalization_0_5_first_rate": float(normalized_mask.mean()) if observations else np.nan,
        "moved_further_to_3_first_rate": float(moved_mask.mean()) if observations else np.nan,
        "neither_rate": float(neither_mask.mean()) if observations else np.nan,
        "median_bars_to_first_touch": safe_median(norm_events["bars_to_first_touch"]) if not norm_events.empty else np.nan,
        "favorable_signed_convergence_rate": float((norm_events["signed_convergence_direction"] == "favorable").mean()) if not norm_events.empty else np.nan,
        "median_signed_convergence_pips_proxy": safe_median(norm_events["signed_convergence_pips_proxy"]) if not norm_events.empty else np.nan,
        "p25_signed_convergence_pips_proxy": float(norm_events["signed_convergence_pips_proxy"].quantile(0.25)) if not norm_events.empty else np.nan,
        "p75_signed_convergence_pips_proxy": float(norm_events["signed_convergence_pips_proxy"].quantile(0.75)) if not norm_events.empty else np.nan,
        "p90_signed_convergence_pips_proxy": float(norm_events["signed_convergence_pips_proxy"].quantile(0.90)) if not norm_events.empty else np.nan,
        "median_signed_convergence_after_friction": safe_median(norm_events["signed_convergence_after_friction"]) if not norm_events.empty else np.nan,
        "positive_signed_convergence_after_friction_rate": float((norm_events["signed_convergence_after_friction"] > 0).mean()) if not norm_events.empty else np.nan,
    }

    split_ix = int(observations * 0.7)
    is_df = events.iloc[:split_ix].copy()
    oos_df = events.iloc[split_ix:].copy()

    def split_row(name: str, part: pd.DataFrame) -> dict:
        part_norm = part[part["first_touch_0_5_vs_3"] == "normalized_0_5_first"] if not part.empty else pd.DataFrame()
        return {
            "split": name,
            "observations": len(part),
            "normalization_0_5_first_rate": float((part["first_touch_0_5_vs_3"] == "normalized_0_5_first").mean()) if not part.empty else np.nan,
            "median_signed_convergence_after_friction": safe_median(part_norm["signed_convergence_after_friction"]) if not part_norm.empty else np.nan,
            "positive_signed_convergence_after_friction_rate": float((part_norm["signed_convergence_after_friction"] > 0).mean()) if not part_norm.empty else np.nan,
        }

    is_oos = pd.DataFrame([split_row("IS", is_df), split_row("OOS", oos_df)])

    yearly_rows = []
    for year, g in events.groupby("year"):
        gn = g[g["first_touch_0_5_vs_3"] == "normalized_0_5_first"]
        yearly_rows.append(
            {
                "year": int(year),
                "year_observations": len(g),
                "year_normalization_0_5_first_rate": float((g["first_touch_0_5_vs_3"] == "normalized_0_5_first").mean()),
                "year_median_signed_convergence_after_friction": safe_median(gn["signed_convergence_after_friction"]) if not gn.empty else np.nan,
                "year_positive_signed_convergence_after_friction_rate": float((gn["signed_convergence_after_friction"] > 0).mean()) if not gn.empty else np.nan,
            }
        )
    yearly = pd.DataFrame(yearly_rows).sort_values("year") if yearly_rows else pd.DataFrame(
        columns=[
            "year",
            "year_observations",
            "year_normalization_0_5_first_rate",
            "year_median_signed_convergence_after_friction",
            "year_positive_signed_convergence_after_friction_rate",
        ]
    )

    non_norm = (events["first_touch_0_5_vs_3"] != "normalized_0_5_first").astype(int) if observations else pd.Series(dtype=int)
    neg_signed = (norm_events["signed_convergence_after_friction"] <= 0).astype(int) if not norm_events.empty else pd.Series(dtype=int)

    def longest_run(series: pd.Series, value: int = 1) -> int:
        run = best = 0
        for x in series.tolist():
            if x == value:
                run += 1
                best = max(best, run)
            else:
                run = 0
        return int(best)

    sequence_stress = {
        "longest_consecutive_non_normalization_first": longest_run(non_norm, 1) if observations else 0,
        "longest_consecutive_negative_signed_after_friction": longest_run(neg_signed, 1) if not norm_events.empty else 0,
        "max_rolling_20_negative_signed_rate": float(neg_signed.rolling(20).mean().max()) if len(neg_signed) >= 20 else np.nan,
        "max_rolling_50_negative_signed_rate": float(neg_signed.rolling(50).mean().max()) if len(neg_signed) >= 50 else np.nan,
    }

    events_path = output_dir / f"{args.pair_name}_signed_relative_events.csv"
    summary_path = output_dir / f"{args.pair_name}_signed_relative_summary.csv"
    md_path = output_dir / f"{args.pair_name}_signed_relative_summary.md"

    events.to_csv(events_path, index=False)

    summary_df = pd.DataFrame([summary])
    for k, v in sequence_stress.items():
        summary_df[k] = v

    prefixed = is_oos.copy()
    prefixed.columns = [
        "split",
        "IS_OOS_observations",
        "IS_OOS_normalization_0_5_first_rate",
        "IS_OOS_median_signed_convergence_after_friction",
        "IS_OOS_positive_signed_convergence_after_friction_rate",
    ]
    summary_wide = summary_df.copy()
    for _, row in prefixed.iterrows():
        split = row["split"]
        for c in prefixed.columns[1:]:
            summary_wide[f"{split}_{c}"] = row[c]

    summary_wide.to_csv(summary_path, index=False)

    md_lines = [
        f"# Signed Relative-Movement Feasibility Check: {args.pair_name}",
        "",
        "**Warning:** Descriptive feasibility only. This is not profitability analysis and not a trading system.",
        "",
        "## Fixed Context Tested",
        f"- window: {WINDOW}",
        f"- horizon_bars: {HORIZON_BARS}",
        f"- session_bucket: {FIXED_SESSION_BUCKET}",
        f"- hour: {FIXED_HOUR}",
        f"- abs_zscore_bucket: {FIXED_ABS_ZSCORE_BUCKET}",
        f"- estimated_round_trip_friction_pips: {ESTIMATED_ROUND_TRIP_FRICTION_PIPS:.1f}",
        "",
        "## Overall Summary",
    ]
    for k, v in summary.items():
        md_lines.append(f"- {k}: {v}")

    md_lines.extend(["", "## IS/OOS (Chronological 70/30)", is_oos.to_markdown(index=False), ""])

    md_lines.extend(["## Yearly Summary", yearly.to_markdown(index=False) if not yearly.empty else "No yearly observations.", ""])

    seq_df = pd.DataFrame([sequence_stress])
    md_lines.extend(["## Sequence Stress", seq_df.to_markdown(index=False), ""])

    oos_median = is_oos.loc[is_oos["split"] == "OOS", "median_signed_convergence_after_friction"].iloc[0] if not is_oos.empty else np.nan
    oos_positive = is_oos.loc[is_oos["split"] == "OOS", "positive_signed_convergence_after_friction_rate"].iloc[0] if not is_oos.empty else np.nan
    seq_fragile = sequence_stress["longest_consecutive_negative_signed_after_friction"] >= 5

    md_lines.extend(
        [
            "## Final Interpretation",
            f"- Signed relative convergence after friction median is {summary['median_signed_convergence_after_friction']}; positive-after-friction rate is {summary['positive_signed_convergence_after_friction_rate']}.",
            f"- OOS stability reference: OOS median signed convergence after friction is {oos_median}; OOS positive-after-friction rate is {oos_positive}.",
            f"- Sequence stress fragility flag (heuristic): {seq_fragile} based on longest consecutive negative signed-after-friction sequence.",
        ]
    )

    md_path.write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
