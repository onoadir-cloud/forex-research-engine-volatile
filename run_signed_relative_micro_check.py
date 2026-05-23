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


def z_convergence_stats(valid_z: pd.Series) -> dict:
    sample_count = int(len(valid_z))
    positive_count = int((valid_z > 0).sum()) if sample_count > 0 else 0
    non_positive_count = int((valid_z <= 0).sum()) if sample_count > 0 else 0
    favorable_rate = float((valid_z > 0).mean()) if sample_count > 0 else np.nan
    median_signed = float(valid_z.median()) if sample_count > 0 else np.nan
    min_signed = float(valid_z.min()) if sample_count > 0 else np.nan
    max_signed = float(valid_z.max()) if sample_count > 0 else np.nan

    if sample_count > 0:
        expected = positive_count / sample_count
        assert np.isclose(favorable_rate, expected, atol=1e-12), (
            "favorable_z_convergence_rate mismatch: "
            f"{favorable_rate} != {expected}"
        )

    return {
        "z_convergence_sample_count": sample_count,
        "z_convergence_positive_count": positive_count,
        "z_convergence_non_positive_count": non_positive_count,
        "favorable_z_convergence_rate": favorable_rate,
        "median_signed_z_convergence": median_signed,
        "min_signed_z_convergence": min_signed,
        "max_signed_z_convergence": max_signed,
    }


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
        deprecated_log_spread_pips_proxy = np.nan
        deprecated_signed_convergence_after_friction = np.nan
        current_abs_z = abs(current_z)
        touch_abs_z = np.nan
        signed_z_convergence = np.nan
        a_move_pips = np.nan
        b_move_pips = np.nan
        basket_move_pips_proxy = np.nan
        sum_leg_move_pips_proxy = np.nan
        conservative_behavior_after_friction = np.nan
        sum_leg_behavior_after_friction = np.nan

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

            touch_z = z[touch_idx]
            touch_abs_z = abs(touch_z)
            signed_z_convergence = current_abs_z - touch_abs_z

            pip_size_a = 0.0001
            pip_size_b = 0.0001
            a_move_pips = abs(ca[touch_idx] - ca[idx]) / pip_size_a
            b_move_pips = abs(cb[touch_idx] - cb[idx]) / pip_size_b
            basket_move_pips_proxy = min(a_move_pips, b_move_pips)
            sum_leg_move_pips_proxy = a_move_pips + b_move_pips
            conservative_behavior_after_friction = (
                basket_move_pips_proxy - ESTIMATED_ROUND_TRIP_FRICTION_PIPS
            )
            sum_leg_behavior_after_friction = (
                sum_leg_move_pips_proxy - ESTIMATED_ROUND_TRIP_FRICTION_PIPS
            )

            # Deprecated: beta-log spread is not a direct pip/price unit.
            signed_a = signed_log_move * ca[idx] / pip_size_a
            signed_b = signed_log_move * cb[idx] / pip_size_b
            deprecated_log_spread_pips_proxy = min(abs(signed_a), abs(signed_b))
            deprecated_signed_convergence_after_friction = (
                deprecated_log_spread_pips_proxy - ESTIMATED_ROUND_TRIP_FRICTION_PIPS
            )

        records.append(
            {
                "datetime": df.at[idx, "datetime"],
                "year": int(df.at[idx, "year"]),
                "current_z": current_z,
                "first_touch_0_5_vs_3": label,
                "bars_to_first_touch": bars_to_first_touch,
                "signed_convergence_direction": direction,
                "signed_convergence_log_move": signed_log_move,
                "current_abs_z": current_abs_z,
                "touch_abs_z": touch_abs_z,
                "signed_z_convergence": signed_z_convergence,
                "a_move_pips": a_move_pips,
                "b_move_pips": b_move_pips,
                "basket_move_pips_proxy": basket_move_pips_proxy,
                "sum_leg_move_pips_proxy": sum_leg_move_pips_proxy,
                "estimated_round_trip_friction_pips": ESTIMATED_ROUND_TRIP_FRICTION_PIPS,
                "conservative_behavior_after_friction": conservative_behavior_after_friction,
                "sum_leg_behavior_after_friction": sum_leg_behavior_after_friction,
                "deprecated_log_spread_pips_proxy": deprecated_log_spread_pips_proxy,
                "deprecated_signed_convergence_after_friction": deprecated_signed_convergence_after_friction,
            }
        )

    events = pd.DataFrame(records).sort_values("datetime").reset_index(drop=True)

    observations = len(events)
    normalized_mask = events["first_touch_0_5_vs_3"] == "normalized_0_5_first" if observations else pd.Series(dtype=bool)
    moved_mask = events["first_touch_0_5_vs_3"] == "moved_further_to_3_first" if observations else pd.Series(dtype=bool)
    neither_mask = events["first_touch_0_5_vs_3"] == "neither" if observations else pd.Series(dtype=bool)

    norm_events = events[normalized_mask].copy() if observations else pd.DataFrame()
    overall_valid_z = norm_events["signed_z_convergence"].dropna() if not norm_events.empty else pd.Series(dtype=float)
    overall_z_stats = z_convergence_stats(overall_valid_z)

    summary = {
        "observations": observations,
        "normalization_0_5_first_rate": float(normalized_mask.mean()) if observations else np.nan,
        "moved_further_to_3_first_rate": float(moved_mask.mean()) if observations else np.nan,
        "neither_rate": float(neither_mask.mean()) if observations else np.nan,
        "median_bars_to_first_touch": safe_median(norm_events["bars_to_first_touch"]) if not norm_events.empty else np.nan,
        "favorable_signed_convergence_rate": float((norm_events["signed_convergence_direction"] == "favorable").mean()) if not norm_events.empty else np.nan,
        **overall_z_stats,
        "median_basket_move_pips_proxy": safe_median(norm_events["basket_move_pips_proxy"]) if not norm_events.empty else np.nan,
        "median_sum_leg_move_pips_proxy": safe_median(norm_events["sum_leg_move_pips_proxy"]) if not norm_events.empty else np.nan,
        "median_conservative_behavior_after_friction": safe_median(norm_events["conservative_behavior_after_friction"]) if not norm_events.empty else np.nan,
        "positive_conservative_after_friction_rate": float((norm_events["conservative_behavior_after_friction"] > 0).mean()) if not norm_events.empty else np.nan,
        "median_sum_leg_behavior_after_friction": safe_median(norm_events["sum_leg_behavior_after_friction"]) if not norm_events.empty else np.nan,
        "positive_sum_leg_after_friction_rate": float((norm_events["sum_leg_behavior_after_friction"] > 0).mean()) if not norm_events.empty else np.nan,
    }

    split_ix = int(observations * 0.7)
    is_df = events.iloc[:split_ix].copy()
    oos_df = events.iloc[split_ix:].copy()

    def split_row(name: str, part: pd.DataFrame) -> dict:
        part_norm = part[part["first_touch_0_5_vs_3"] == "normalized_0_5_first"] if not part.empty else pd.DataFrame()
        part_valid_z = part_norm["signed_z_convergence"].dropna() if not part_norm.empty else pd.Series(dtype=float)
        part_z_stats = z_convergence_stats(part_valid_z)
        return {
            "split": name,
            "observations": len(part),
            "normalization_0_5_first_rate": float((part["first_touch_0_5_vs_3"] == "normalized_0_5_first").mean()) if not part.empty else np.nan,
            "favorable_signed_convergence_rate": float((part_norm["signed_convergence_direction"] == "favorable").mean()) if not part_norm.empty else np.nan,
            **part_z_stats,
            "median_basket_move_pips_proxy": safe_median(part_norm["basket_move_pips_proxy"]) if not part_norm.empty else np.nan,
            "median_conservative_behavior_after_friction": safe_median(part_norm["conservative_behavior_after_friction"]) if not part_norm.empty else np.nan,
            "positive_conservative_after_friction_rate": float((part_norm["conservative_behavior_after_friction"] > 0).mean()) if not part_norm.empty else np.nan,
        }

    is_oos = pd.DataFrame([split_row("IS", is_df), split_row("OOS", oos_df)])

    yearly_rows = []
    for year, g in events.groupby("year"):
        gn = g[g["first_touch_0_5_vs_3"] == "normalized_0_5_first"]
        year_valid_z = gn["signed_z_convergence"].dropna() if not gn.empty else pd.Series(dtype=float)
        year_z_stats = z_convergence_stats(year_valid_z)
        yearly_rows.append(
            {
                "year": int(year),
                "year_observations": len(g),
                "year_normalization_0_5_first_rate": float((g["first_touch_0_5_vs_3"] == "normalized_0_5_first").mean()),
                "year_favorable_z_convergence_rate": year_z_stats["favorable_z_convergence_rate"],
                "year_median_signed_z_convergence": year_z_stats["median_signed_z_convergence"],
                "year_z_convergence_sample_count": year_z_stats["z_convergence_sample_count"],
                "year_z_convergence_positive_count": year_z_stats["z_convergence_positive_count"],
                "year_z_convergence_non_positive_count": year_z_stats["z_convergence_non_positive_count"],
                "year_min_signed_z_convergence": year_z_stats["min_signed_z_convergence"],
                "year_max_signed_z_convergence": year_z_stats["max_signed_z_convergence"],
                "year_median_basket_move_pips_proxy": safe_median(gn["basket_move_pips_proxy"]) if not gn.empty else np.nan,
                "year_median_conservative_behavior_after_friction": safe_median(gn["conservative_behavior_after_friction"]) if not gn.empty else np.nan,
                "year_positive_conservative_after_friction_rate": float((gn["conservative_behavior_after_friction"] > 0).mean()) if not gn.empty else np.nan,
            }
        )
    yearly = pd.DataFrame(yearly_rows).sort_values("year") if yearly_rows else pd.DataFrame(
        columns=[
            "year",
            "year_observations",
            "year_normalization_0_5_first_rate",
            "year_favorable_z_convergence_rate",
            "year_median_signed_z_convergence",
            "year_z_convergence_sample_count",
            "year_z_convergence_positive_count",
            "year_z_convergence_non_positive_count",
            "year_min_signed_z_convergence",
            "year_max_signed_z_convergence",
            "year_median_basket_move_pips_proxy",
            "year_median_conservative_behavior_after_friction",
            "year_positive_conservative_after_friction_rate",
        ]
    )

    non_norm = (events["first_touch_0_5_vs_3"] != "normalized_0_5_first").astype(int) if observations else pd.Series(dtype=int)
    neg_conservative = (norm_events["conservative_behavior_after_friction"] <= 0).astype(int) if not norm_events.empty else pd.Series(dtype=int)

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
        "longest_consecutive_negative_conservative_after_friction": longest_run(neg_conservative, 1) if not norm_events.empty else 0,
        "max_rolling_20_negative_conservative_rate": float(neg_conservative.rolling(20).mean().max()) if len(neg_conservative) >= 20 else np.nan,
        "max_rolling_50_negative_conservative_rate": float(neg_conservative.rolling(50).mean().max()) if len(neg_conservative) >= 50 else np.nan,
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
        "IS_OOS_favorable_signed_convergence_rate",
        "IS_OOS_favorable_z_convergence_rate",
        "IS_OOS_median_signed_z_convergence",
        "IS_OOS_z_convergence_sample_count",
        "IS_OOS_z_convergence_positive_count",
        "IS_OOS_z_convergence_non_positive_count",
        "IS_OOS_min_signed_z_convergence",
        "IS_OOS_max_signed_z_convergence",
        "IS_OOS_median_basket_move_pips_proxy",
        "IS_OOS_median_conservative_behavior_after_friction",
        "IS_OOS_positive_conservative_after_friction_rate",
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
        "## Measurement Notes",
        "- Signed relative movement confirms direction of convergence.",
        "- Z-convergence is evaluated only on `normalized_0_5_first` events.",
        "- Non-normalization events are excluded from z-convergence rate.",
        "- Pip feasibility is measured only using actual close-to-close leg movement.",
        "- Direct beta-log-spread-to-pips conversion is not used for feasibility.",
        "- Deprecated columns retained for reference only: `deprecated_log_spread_pips_proxy`, `deprecated_signed_convergence_after_friction`.",
        "",
        "## Overall Summary",
    ]
    for k, v in summary.items():
        md_lines.append(f"- {k}: {v}")

    warnings = []
    if (
        pd.notna(summary["normalization_0_5_first_rate"])
        and pd.notna(summary["favorable_z_convergence_rate"])
        and summary["normalization_0_5_first_rate"] > 0.50
        and summary["favorable_z_convergence_rate"] < 0.50
    ):
        warnings.append("z convergence rate unexpectedly low; inspect first-touch z calculation.")
    if (
        pd.notna(summary["median_signed_z_convergence"])
        and pd.notna(summary["favorable_z_convergence_rate"])
        and summary["median_signed_z_convergence"] > 0
        and summary["favorable_z_convergence_rate"] < 0.50
    ):
        warnings.append("median z convergence positive but favorable rate low; inspect aggregation.")

    if warnings:
        md_lines.extend(["", "## Validation Warnings"])
        md_lines.extend([f"- {w}" for w in warnings])

    md_lines.extend(["", "## IS/OOS (Chronological 70/30)", is_oos.to_markdown(index=False), ""])

    md_lines.extend(["## Yearly Summary", yearly.to_markdown(index=False) if not yearly.empty else "No yearly observations.", ""])

    seq_df = pd.DataFrame([sequence_stress])
    md_lines.extend(["## Sequence Stress", seq_df.to_markdown(index=False), ""])

    oos_median = is_oos.loc[is_oos["split"] == "OOS", "median_conservative_behavior_after_friction"].iloc[0] if not is_oos.empty else np.nan
    oos_positive = is_oos.loc[is_oos["split"] == "OOS", "positive_conservative_after_friction_rate"].iloc[0] if not is_oos.empty else np.nan
    seq_fragile = sequence_stress["longest_consecutive_negative_conservative_after_friction"] >= 5

    md_lines.extend(
        [
            "## Final Interpretation",
            f"- Signed convergence remains a direction/stability diagnostic: favorable_signed_convergence_rate is {summary['favorable_signed_convergence_rate']} and favorable_z_convergence_rate is {summary['favorable_z_convergence_rate']}.",
            f"- Feasibility after friction uses close-to-close leg movement: median conservative behavior after friction is {summary['median_conservative_behavior_after_friction']}; positive rate is {summary['positive_conservative_after_friction_rate']}.",
            f"- OOS stability reference: OOS median conservative behavior after friction is {oos_median}; OOS positive-after-friction rate is {oos_positive}.",
            f"- Sequence stress fragility flag (heuristic): {seq_fragile} based on longest consecutive negative conservative-after-friction sequence.",
        ]
    )

    md_path.write_text("\n".join(md_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
