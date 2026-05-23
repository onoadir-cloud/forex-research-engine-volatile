#!/usr/bin/env python3
"""Focused descriptive micro-feasibility check for fixed survivor contexts only."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


FIXED_CONTEXTS: List[Dict[str, object]] = [
    {
        "window": 50,
        "horizon_bars": 40,
        "session_bucket": "New York open",
        "hour": 14,
        "abs_zscore_bucket": "1_2",
    },
    {
        "window": 50,
        "horizon_bars": 40,
        "session_bucket": "New York open",
        "hour": 15,
        "abs_zscore_bucket": "1_2",
    },
]

SPREAD_A_PIPS = 1.2
SPREAD_B_PIPS = 1.8
SLIPPAGE_A_PIPS = 0.3
SLIPPAGE_B_PIPS = 0.5
COMMISSION_EQUIV_PIPS_TOTAL = 0.6
ESTIMATED_FRICTION_PIPS = (
    2 * (SPREAD_A_PIPS + SPREAD_B_PIPS)
    + 2 * (SLIPPAGE_A_PIPS + SLIPPAGE_B_PIPS)
    + COMMISSION_EQUIV_PIPS_TOTAL
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run descriptive micro-feasibility check.")
    p.add_argument("--csv-a", default="data/EURUSD_M15_MT5_5Y.csv")
    p.add_argument("--csv-b", default="data/GBPUSD_M15_MT5_5Y.csv")
    p.add_argument("--pair-name", default="EURUSD_GBPUSD")
    p.add_argument("--output-dir", default="micro_feasibility_reports")
    return p.parse_args()


def normalize_price_df(path: str, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    col_map = {c.lower().strip(): c for c in df.columns}

    dt_col = None
    for k in ["datetime", "date", "time", "timestamp"]:
        if k in col_map:
            dt_col = col_map[k]
            break
    if dt_col is None:
        raise ValueError(f"Could not find datetime column in {path}")

    close_col = None
    for k in ["close", "closeprice", "last", "settle"]:
        if k in col_map:
            close_col = col_map[k]
            break
    if close_col is None:
        raise ValueError(f"Could not find close column in {path}")

    out = pd.DataFrame()
    out["datetime"] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
    out[f"close_{prefix}"] = pd.to_numeric(df[close_col], errors="coerce")
    out = out.dropna(subset=["datetime", f"close_{prefix}"]).drop_duplicates("datetime")
    return out.sort_values("datetime")


def session_bucket_from_hour(h: int) -> str:
    if 0 <= h <= 3:
        return "Asia early"
    if 4 <= h <= 6:
        return "Asia late"
    if 7 <= h <= 9:
        return "London open"
    if 10 <= h <= 12:
        return "London mid"
    if 13 <= h <= 15:
        return "New York open"
    if 16 <= h <= 18:
        return "New York mid"
    return "Late session"


def abs_z_bucket(z: float) -> Optional[str]:
    az = abs(z)
    if 1 <= az < 2:
        return "1_2"
    return None


def first_index_true(mask: np.ndarray) -> Optional[int]:
    idx = np.where(mask)[0]
    return int(idx[0]) if idx.size else None


def summarize_context(events: pd.DataFrame, context_key: str) -> Dict[str, object]:
    label = events["first_touch_0_5_vs_3"]
    norm = events["basket_move_pips_proxy"].dropna()
    cons = events["conservative_behavior_after_friction"].dropna()
    sum_leg = events["sum_leg_behavior_after_friction"].dropna()
    return {
        "context_key": context_key,
        "observations": int(len(events)),
        "normalization_0_5_first_rate": float((label == "normalized_0_5_first").mean()) if len(events) else np.nan,
        "moved_further_to_3_first_rate": float((label == "moved_further_to_3_first").mean()) if len(events) else np.nan,
        "both_same_bar_rate": float((label == "both_same_bar").mean()) if len(events) else np.nan,
        "neither_rate": float((label == "neither").mean()) if len(events) else np.nan,
        "median_bars_to_first_touch": float(events["bars_to_first_touch_0_5_vs_3"].median()) if len(events) else np.nan,
        "median_basket_move_pips_proxy": float(norm.median()) if len(norm) else np.nan,
        "p25_basket_move_pips_proxy": float(norm.quantile(0.25)) if len(norm) else np.nan,
        "p75_basket_move_pips_proxy": float(norm.quantile(0.75)) if len(norm) else np.nan,
        "p90_basket_move_pips_proxy": float(norm.quantile(0.90)) if len(norm) else np.nan,
        "median_sum_leg_move_pips_proxy": float(events["sum_leg_move_pips_proxy"].dropna().median()) if events["sum_leg_move_pips_proxy"].notna().any() else np.nan,
        "median_conservative_behavior_after_friction": float(cons.median()) if len(cons) else np.nan,
        "median_sum_leg_behavior_after_friction": float(sum_leg.median()) if len(sum_leg) else np.nan,
        "positive_conservative_after_friction_rate": float((cons > 0).mean()) if len(cons) else np.nan,
        "positive_sum_leg_after_friction_rate": float((sum_leg > 0).mean()) if len(sum_leg) else np.nan,
        "estimated_round_trip_friction_pips": ESTIMATED_FRICTION_PIPS,
    }


def build_markdown(pair: str, contexts_df: pd.DataFrame, isoos_df: pd.DataFrame, yearly_df: pd.DataFrame, stress_df: pd.DataFrame) -> str:
    parts = [
        f"# {pair} Descriptive Micro-Feasibility Check",
        "",
        "**Warning:** This is a descriptive micro-feasibility report only. It is not a profitability claim, not a trading system, and not production execution guidance.",
        "",
        "## Tested Fixed Contexts",
        "- window=50, horizon_bars=40, session_bucket=New York open, hour=14, abs_zscore_bucket=1_2",
        "- window=50, horizon_bars=40, session_bucket=New York open, hour=15, abs_zscore_bucket=1_2",
        "",
        "## Context Summary",
        contexts_df.to_markdown(index=False),
        "",
        "## IS/OOS",
        isoos_df.to_markdown(index=False),
        "",
        "## Yearly Stability",
        yearly_df.to_markdown(index=False),
        "",
        "## Sequence Stress",
        stress_df.to_markdown(index=False),
        "",
        "## Final Interpretation",
        "- Evaluate whether normalization-first event behavior remains stable across contexts, yearly slices, and IS/OOS.",
        "- Evaluate whether conservative friction-adjusted behavior remains mostly positive in descriptive terms.",
        "- If sequence stress and friction-adjusted behavior are unstable, treat the context as fragile.",
    ]
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a = normalize_price_df(args.csv_a, "a")
    b = normalize_price_df(args.csv_b, "b")
    df = a.merge(b, on="datetime", how="inner")

    df["log_price_a"] = np.log(df["close_a"])
    df["log_price_b"] = np.log(df["close_b"])
    df["return_a"] = df["log_price_a"].diff()
    df["return_b"] = df["log_price_b"].diff()

    window = 50
    cov = df["log_price_a"].rolling(window).cov(df["log_price_b"])
    var = df["log_price_b"].rolling(window).var()
    df["rolling_beta_50"] = cov / var.replace(0, np.nan)
    df["relative_spread_50"] = df["log_price_a"] - df["rolling_beta_50"] * df["log_price_b"]
    df["spread_mean_50"] = df["relative_spread_50"].rolling(window).mean()
    df["spread_std_50"] = df["relative_spread_50"].rolling(window).std()
    df["spread_zscore_50"] = (df["relative_spread_50"] - df["spread_mean_50"]) / df["spread_std_50"].replace(0, np.nan)

    df["hour"] = df["datetime"].dt.hour
    df["session_bucket"] = df["hour"].map(session_bucket_from_hour)
    df["year"] = df["datetime"].dt.year
    df["abs_zscore_bucket_50"] = df["spread_zscore_50"].apply(abs_z_bucket)

    event_rows: List[Dict[str, object]] = []

    for ctx in FIXED_CONTEXTS:
        context_key = (
            f"window={ctx['window']}|horizon_bars={ctx['horizon_bars']}|"
            f"session_bucket={ctx['session_bucket']}|hour={ctx['hour']}|abs_zscore_bucket={ctx['abs_zscore_bucket']}"
        )
        subset = df[
            (df["session_bucket"] == ctx["session_bucket"])
            & (df["hour"] == ctx["hour"])
            & (df["abs_zscore_bucket_50"] == ctx["abs_zscore_bucket"])
            & df["spread_zscore_50"].notna()
        ]

        for idx in subset.index:
            z_now = float(df.at[idx, "spread_zscore_50"])
            horizon = int(ctx["horizon_bars"])
            future = df.iloc[idx + 1 : idx + 1 + horizon]
            future_z = future["spread_zscore_50"].to_numpy(dtype=float)

            if future.empty:
                continue

            if z_now > 0:
                n05_idx = first_index_true(future_z <= 0.5)
                n0_idx = first_index_true(future_z <= 0.0)
            else:
                n05_idx = first_index_true(future_z >= -0.5)
                n0_idx = first_index_true(future_z >= 0.0)

            d3_idx = first_index_true(np.abs(future_z) >= 3.0)
            _d4_idx = first_index_true(np.abs(future_z) >= 4.0)

            if n05_idx is None and d3_idx is None:
                label = "neither"
                bars = np.nan
            elif n05_idx is not None and d3_idx is None:
                label = "normalized_0_5_first"
                bars = n05_idx + 1
            elif n05_idx is None and d3_idx is not None:
                label = "moved_further_to_3_first"
                bars = d3_idx + 1
            elif n05_idx == d3_idx:
                label = "both_same_bar"
                bars = n05_idx + 1
            elif n05_idx < d3_idx:
                label = "normalized_0_5_first"
                bars = n05_idx + 1
            else:
                label = "moved_further_to_3_first"
                bars = d3_idx + 1

            row: Dict[str, object] = {
                "datetime": df.at[idx, "datetime"],
                "year": int(df.at[idx, "year"]),
                "context_key": context_key,
                "pair_name": args.pair_name,
                "window": ctx["window"],
                "horizon_bars": ctx["horizon_bars"],
                "session_bucket": ctx["session_bucket"],
                "hour": ctx["hour"],
                "abs_zscore_bucket": ctx["abs_zscore_bucket"],
                "current_z": z_now,
                "first_touch_0_5_vs_3": label,
                "bars_to_first_touch_0_5_vs_3": bars,
                "a_move_pips": np.nan,
                "b_move_pips": np.nan,
                "basket_move_pips_proxy": np.nan,
                "sum_leg_move_pips_proxy": np.nan,
                "estimated_round_trip_friction_pips": ESTIMATED_FRICTION_PIPS,
                "conservative_behavior_after_friction": np.nan,
                "sum_leg_behavior_after_friction": np.nan,
            }

            if label == "normalized_0_5_first" and n05_idx is not None:
                first_touch_row = future.iloc[n05_idx]
                a_now = float(df.at[idx, "close_a"])
                b_now = float(df.at[idx, "close_b"])
                a_then = float(first_touch_row["close_a"])
                b_then = float(first_touch_row["close_b"])
                a_move = abs(a_then - a_now) / 0.0001
                b_move = abs(b_then - b_now) / 0.0001
                basket = min(a_move, b_move)
                sum_leg = a_move + b_move
                row.update(
                    {
                        "a_move_pips": a_move,
                        "b_move_pips": b_move,
                        "basket_move_pips_proxy": basket,
                        "sum_leg_move_pips_proxy": sum_leg,
                        "conservative_behavior_after_friction": basket - ESTIMATED_FRICTION_PIPS,
                        "sum_leg_behavior_after_friction": sum_leg - ESTIMATED_FRICTION_PIPS,
                    }
                )
            event_rows.append(row)

    events = pd.DataFrame(event_rows).sort_values(["context_key", "datetime"])

    context_summary = []
    yearly_rows = []
    isoos_rows = []
    stress_rows = []

    for context_key, g in events.groupby("context_key", sort=False):
        g = g.sort_values("datetime").reset_index(drop=True)
        context_summary.append(summarize_context(g, context_key))

        for year, yg in g.groupby("year"):
            cons = yg["conservative_behavior_after_friction"].dropna()
            yearly_rows.append(
                {
                    "context_key": context_key,
                    "year": int(year),
                    "year_observations": int(len(yg)),
                    "year_normalization_0_5_first_rate": float((yg["first_touch_0_5_vs_3"] == "normalized_0_5_first").mean()),
                    "year_median_basket_move_pips_proxy": float(yg["basket_move_pips_proxy"].dropna().median()) if yg["basket_move_pips_proxy"].notna().any() else np.nan,
                    "year_median_conservative_behavior_after_friction": float(cons.median()) if len(cons) else np.nan,
                    "year_positive_conservative_after_friction_rate": float((cons > 0).mean()) if len(cons) else np.nan,
                }
            )

        split_idx = int(len(g) * 0.7)
        is_g = g.iloc[:split_idx]
        oos_g = g.iloc[split_idx:]
        is_cons = is_g["conservative_behavior_after_friction"].dropna()
        oos_cons = oos_g["conservative_behavior_after_friction"].dropna()
        isoos_rows.append(
            {
                "context_key": context_key,
                "IS_observations": int(len(is_g)),
                "OOS_observations": int(len(oos_g)),
                "IS_normalization_0_5_first_rate": float((is_g["first_touch_0_5_vs_3"] == "normalized_0_5_first").mean()) if len(is_g) else np.nan,
                "OOS_normalization_0_5_first_rate": float((oos_g["first_touch_0_5_vs_3"] == "normalized_0_5_first").mean()) if len(oos_g) else np.nan,
                "IS_median_conservative_behavior_after_friction": float(is_cons.median()) if len(is_cons) else np.nan,
                "OOS_median_conservative_behavior_after_friction": float(oos_cons.median()) if len(oos_cons) else np.nan,
                "IS_positive_conservative_after_friction_rate": float((is_cons > 0).mean()) if len(is_cons) else np.nan,
                "OOS_positive_conservative_after_friction_rate": float((oos_cons > 0).mean()) if len(oos_cons) else np.nan,
            }
        )

        non_norm = (g["first_touch_0_5_vs_3"] != "normalized_0_5_first").astype(int)
        cons_neg = (g["conservative_behavior_after_friction"].fillna(-np.inf) < 0).astype(int)

        longest_non_norm = int(non_norm.groupby((non_norm != non_norm.shift()).cumsum()).sum().max()) if len(non_norm) else 0
        longest_neg = int(cons_neg.groupby((cons_neg != cons_neg.shift()).cumsum()).sum().max()) if len(cons_neg) else 0

        rolling20 = cons_neg.rolling(20).mean()
        rolling50 = cons_neg.rolling(50).mean()

        stress_rows.append(
            {
                "context_key": context_key,
                "longest_consecutive_non_normalization_first": longest_non_norm,
                "longest_consecutive_negative_conservative_after_friction": longest_neg,
                "max_rolling_20_negative_conservative_rate": float(rolling20.max()) if rolling20.notna().any() else np.nan,
                "max_rolling_50_negative_conservative_rate": float(rolling50.max()) if rolling50.notna().any() else np.nan,
            }
        )

    context_df = pd.DataFrame(context_summary)
    yearly_df = pd.DataFrame(yearly_rows)
    isoos_df = pd.DataFrame(isoos_rows)
    stress_df = pd.DataFrame(stress_rows)

    events_path = out_dir / f"{args.pair_name}_micro_feasibility_events.csv"
    summary_csv_path = out_dir / f"{args.pair_name}_micro_feasibility_summary.csv"
    summary_md_path = out_dir / f"{args.pair_name}_micro_feasibility_summary.md"

    events.to_csv(events_path, index=False)
    combined_summary = context_df.merge(isoos_df, on="context_key", how="left")
    combined_summary.to_csv(summary_csv_path, index=False)

    markdown = build_markdown(args.pair_name, context_df, isoos_df, yearly_df, stress_df)
    summary_md_path.write_text(markdown, encoding="utf-8")

    print(f"Wrote: {events_path}")
    print(f"Wrote: {summary_csv_path}")
    print(f"Wrote: {summary_md_path}")
    print("Descriptive micro-feasibility only. No optimization, no strategy rules, no equity curves.")


if __name__ == "__main__":
    main()
