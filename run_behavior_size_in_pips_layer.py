#!/usr/bin/env python3
"""Descriptive behavior-size and execution-feasibility screening for pair first-touch movement."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLS = ["datetime", "open", "high", "low", "close"]
DEFAULT_PAIRS = ["EURUSD_GBPUSD", "GBPAUD_GBPNZD"]
DEFAULT_CSV_A = ["data/EURUSD_M15_MT5_5Y.csv", "data/GBPAUD_M15_MT5_5Y.csv"]
DEFAULT_CSV_B = ["data/GBPUSD_M15_MT5_5Y.csv", "data/GBPNZD_M15_MT5_5Y.csv"]
WINDOWS = [20, 50, 100]
HORIZONS = [5, 10, 20, 40]
DEFAULT_OUTPUT_DIR = "behavior_size_reports"

COST_PROFILES: Dict[str, Dict[str, Dict[str, float]]] = {
    "low": {
        "EURUSD_GBPUSD": {"spread_a_pips": 0.8, "spread_b_pips": 1.2, "slippage_a_pips": 0.2, "slippage_b_pips": 0.3, "commission_equivalent_pips_total": 0.4},
        "GBPAUD_GBPNZD": {"spread_a_pips": 2.5, "spread_b_pips": 3.5, "slippage_a_pips": 0.5, "slippage_b_pips": 0.7, "commission_equivalent_pips_total": 0.6},
    },
    "conservative": {
        "EURUSD_GBPUSD": {"spread_a_pips": 1.2, "spread_b_pips": 1.8, "slippage_a_pips": 0.3, "slippage_b_pips": 0.5, "commission_equivalent_pips_total": 0.6},
        "GBPAUD_GBPNZD": {"spread_a_pips": 3.5, "spread_b_pips": 5.0, "slippage_a_pips": 0.8, "slippage_b_pips": 1.0, "commission_equivalent_pips_total": 0.8},
    },
    "high": {
        "EURUSD_GBPUSD": {"spread_a_pips": 2.0, "spread_b_pips": 2.5, "slippage_a_pips": 0.6, "slippage_b_pips": 0.8, "commission_equivalent_pips_total": 1.0},
        "GBPAUD_GBPNZD": {"spread_a_pips": 5.0, "spread_b_pips": 7.0, "slippage_a_pips": 1.2, "slippage_b_pips": 1.5, "commission_equivalent_pips_total": 1.2},
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate behavior-size proxy and cost-share feasibility from first-touch movement.")
    p.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    p.add_argument("--csv-a-list", default=",".join(DEFAULT_CSV_A))
    p.add_argument("--csv-b-list", default=",".join(DEFAULT_CSV_B))
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--cost-profile", choices=["low", "conservative", "high"], default="conservative")
    p.add_argument("--min-observations", type=int, default=200)
    return p.parse_args()


def parse_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def pip_size(symbol: str) -> float:
    s = symbol.upper()
    return 0.01 if s.endswith("JPY") else 0.0001


def load_side(path: Path, suffix: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {path} missing columns: {missing}")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).sort_values("datetime").reset_index(drop=True)
    return df[["datetime", "open", "high", "low", "close"]].rename(columns={c: f"{c}_{suffix}" for c in ["open", "high", "low", "close"]})


def session_bucket_from_hour(hour: pd.Series) -> pd.Series:
    return pd.cut(hour, bins=[-1, 3, 6, 9, 12, 15, 18, 23], labels=["Asia early", "Asia late", "London open", "London mid", "New York open", "New York mid", "Late session"]).astype(str)


def abs_zscore_bucket(s: pd.Series) -> pd.Series:
    out = pd.cut(s, bins=[0, 1, 2, 3, np.inf], labels=["0_1", "1_2", "2_3", "3_plus"], include_lowest=True)
    return out.astype("object").where(~out.isna(), "unknown")


def build_future_arrays(series: pd.Series, horizon: int) -> np.ndarray:
    vals = series.to_numpy()
    n = len(vals)
    arr = np.full((n, horizon), np.nan)
    for k in range(1, horizon + 1):
        shifted = np.roll(vals, -k)
        shifted[n - k :] = np.nan
        arr[:, k - 1] = shifted
    return arr


def first_touch_labels(z_values: np.ndarray, current_z: np.ndarray, norm_level: float, div_level: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(current_z)
    labels = np.full(n, "unknown", dtype=object)
    bars = np.full(n, np.nan)
    for i in range(n):
        z_now = current_z[i]
        row = z_values[i]
        if np.isnan(z_now):
            continue
        idx_norm = None
        idx_div = None
        for j, val in enumerate(row):
            if np.isnan(val):
                continue
            if idx_norm is None:
                if z_now >= 0 and val <= norm_level:
                    idx_norm = j
                elif z_now < 0 and val >= -norm_level:
                    idx_norm = j
            if idx_div is None and abs(val) >= div_level:
                idx_div = j
            if idx_norm is not None and idx_div is not None:
                break
        if idx_norm is None and idx_div is None:
            labels[i] = "neither"
        elif idx_norm is None:
            labels[i] = f"moved_further_to_{int(div_level)}_first"
            bars[i] = idx_div + 1
        elif idx_div is None or idx_norm < idx_div:
            labels[i] = f"normalization_{str(norm_level).replace('.', '_')}_first"
            bars[i] = idx_norm + 1
        elif idx_norm == idx_div:
            labels[i] = "both_same_bar"
            bars[i] = idx_norm + 1
        else:
            labels[i] = f"moved_further_to_{int(div_level)}_first"
            bars[i] = idx_div + 1
    return labels, bars


def round_trip_friction(cost_cfg: Dict[str, float]) -> float:
    return 2 * (cost_cfg["spread_a_pips"] + cost_cfg["spread_b_pips"]) + 2 * (cost_cfg["slippage_a_pips"] + cost_cfg["slippage_b_pips"]) + cost_cfg["commission_equivalent_pips_total"]


def feasibility_band(v: float) -> str:
    if v <= 0.20:
        return "low_cost_share"
    if v <= 0.40:
        return "moderate_cost_share"
    if v <= 0.70:
        return "high_cost_share"
    return "very_high_cost_share"


def main() -> None:
    args = parse_args()
    pairs = parse_list(args.pairs)
    csv_as = parse_list(args.csv_a_list)
    csv_bs = parse_list(args.csv_b_list)
    if not (len(pairs) == len(csv_as) == len(csv_bs)):
        raise ValueError("pairs, csv-a-list, csv-b-list must have equal lengths")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[pd.DataFrame] = []

    for pair_name, csv_a, csv_b in zip(pairs, csv_as, csv_bs):
        print(f"[progress] pair={pair_name} loading")
        symbol_a, symbol_b = pair_name.split("_", 1)
        df_a = load_side(Path(csv_a), "a")
        df_b = load_side(Path(csv_b), "b")
        df = df_a.merge(df_b, on="datetime", how="inner")
        if df.empty:
            raise ValueError(f"No synchronized rows for {pair_name}")

        df["hour"] = df["datetime"].dt.hour
        df["session_bucket"] = session_bucket_from_hour(df["hour"])
        df["log_price_a"] = np.log(df["close_a"])
        df["log_price_b"] = np.log(df["close_b"])
        df["returns_a"] = df["log_price_a"].diff()
        df["returns_b"] = df["log_price_b"].diff()

        for window in WINDOWS:
            print(f"[progress] pair={pair_name} window={window}")
            beta = df["returns_a"].rolling(window).cov(df["returns_b"]) / df["returns_b"].rolling(window).var()
            rel_spread = df["log_price_a"] - beta * df["log_price_b"]
            spread_mean = rel_spread.rolling(window).mean()
            spread_std = rel_spread.rolling(window).std()
            z = (rel_spread - spread_mean) / spread_std

            work = df[["datetime", "hour", "session_bucket", "close_a", "close_b"]].copy()
            work["pair_name"] = pair_name
            work["window"] = window
            work["rolling_beta"] = beta
            work["spread_std"] = spread_std
            work["spread_zscore"] = z
            work["current_abs_z"] = work["spread_zscore"].abs()
            work["abs_zscore_bucket"] = abs_zscore_bucket(work["current_abs_z"])

            for horizon in HORIZONS:
                print(f"[progress] pair={pair_name} window={window} horizon={horizon}")
                z_future = build_future_arrays(work["spread_zscore"], horizon)
                close_a_future = build_future_arrays(work["close_a"], horizon)
                close_b_future = build_future_arrays(work["close_b"], horizon)
                valid = (~work["spread_zscore"].isna()) & (~work["spread_std"].isna()) & (~pd.isna(z_future).all(axis=1))
                if not valid.any():
                    continue

                labels_05_3, bars_05_3 = first_touch_labels(z_future, work["spread_zscore"].to_numpy(), 0.5, 3.0)
                labels_0_3, bars_0_3 = first_touch_labels(z_future, work["spread_zscore"].to_numpy(), 0.0, 3.0)
                labels_05_4, bars_05_4 = first_touch_labels(z_future, work["spread_zscore"].to_numpy(), 0.5, 4.0)

                temp = work.loc[valid].copy()
                idx = valid.to_numpy()
                temp["horizon_bars"] = horizon
                eligible_05 = temp["current_abs_z"] > 0.5
                eligible_0 = temp["current_abs_z"] > 0.0
                temp["normalization_0_5_first_vs_3"] = np.where(eligible_05, (labels_05_3[idx] == "normalization_0_5_first").astype(float), np.nan)
                temp["moved_further_to_3_first_vs_0_5"] = np.where(eligible_05, (labels_05_3[idx] == "moved_further_to_3_first").astype(float), np.nan)
                temp["normalization_0_first_vs_3"] = np.where(eligible_0, (labels_0_3[idx] == "normalization_0_0_first").astype(float), np.nan)
                temp["moved_further_to_3_first_vs_0"] = np.where(eligible_0, (labels_0_3[idx] == "moved_further_to_3_first").astype(float), np.nan)
                temp["normalization_0_5_first_vs_4"] = (labels_05_4[idx] == "normalization_0_5_first").astype(float)
                temp["moved_further_to_4_first_vs_0_5"] = (labels_05_4[idx] == "moved_further_to_4_first").astype(float)
                temp["bars_to_first_touch_0_5_vs_3"] = bars_05_3[idx]
                temp["bars_to_first_touch_0_vs_3"] = bars_0_3[idx]
                temp["bars_to_first_touch_0_5_vs_4"] = bars_05_4[idx]

                pipa = pip_size(symbol_a)
                pipb = pip_size(symbol_b)

                temp["a_move_pips_to_first_touch"] = np.nan
                temp["b_move_pips_to_first_touch"] = np.nan
                temp["basket_move_pips_proxy"] = np.nan
                temp["sum_leg_move_pips_proxy"] = np.nan

                norm_mask_05_3 = (labels_05_3[idx] == "normalization_0_5_first") & (temp["current_abs_z"] > 0.5)
                if norm_mask_05_3.any():
                    future_step = temp.loc[norm_mask_05_3, "bars_to_first_touch_0_5_vs_3"].astype("Int64")
                    base_close_a = temp.loc[norm_mask_05_3, "close_a"].to_numpy()
                    base_close_b = temp.loc[norm_mask_05_3, "close_b"].to_numpy()
                    row_positions = np.where(idx)[0][norm_mask_05_3.to_numpy()]
                    future_close_a = np.array([close_a_future[i_row, int(step - 1)] for i_row, step in zip(row_positions, future_step.to_numpy())], dtype=float)
                    future_close_b = np.array([close_b_future[i_row, int(step - 1)] for i_row, step in zip(row_positions, future_step.to_numpy())], dtype=float)
                    a_move = np.abs(future_close_a - base_close_a) / pipa
                    b_move = np.abs(future_close_b - base_close_b) / pipb
                    basket_move = np.minimum(a_move, b_move)
                    sum_move = a_move + b_move
                    temp.loc[norm_mask_05_3, "a_move_pips_to_first_touch"] = a_move
                    temp.loc[norm_mask_05_3, "b_move_pips_to_first_touch"] = b_move
                    temp.loc[norm_mask_05_3, "basket_move_pips_proxy"] = basket_move
                    temp.loc[norm_mask_05_3, "sum_leg_move_pips_proxy"] = sum_move

                temp["behavior_size_pips_proxy"] = temp["basket_move_pips_proxy"]

                all_rows.append(temp)

    if not all_rows:
        raise ValueError("No valid rows generated.")

    full = pd.concat(all_rows, ignore_index=True)
    keys = ["pair_name", "window", "horizon_bars", "session_bucket", "hour", "abs_zscore_bucket"]
    valid_size = full["behavior_size_pips_proxy"].where((full["behavior_size_pips_proxy"] > 0) & (~full["behavior_size_pips_proxy"].isna()))
    valid_a_move = full["a_move_pips_to_first_touch"].where((full["a_move_pips_to_first_touch"] > 0) & (~full["a_move_pips_to_first_touch"].isna()))
    valid_b_move = full["b_move_pips_to_first_touch"].where((full["b_move_pips_to_first_touch"] > 0) & (~full["b_move_pips_to_first_touch"].isna()))
    valid_sum_move = full["sum_leg_move_pips_proxy"].where((full["sum_leg_move_pips_proxy"] > 0) & (~full["sum_leg_move_pips_proxy"].isna()))
    full["behavior_size_pips_proxy_valid"] = valid_size
    full["a_move_pips_valid"] = valid_a_move
    full["b_move_pips_valid"] = valid_b_move
    full["sum_leg_move_pips_proxy_valid"] = valid_sum_move

    summary = full.groupby(keys, dropna=False).agg(
        observations=("spread_zscore", "size"),
        normalization_0_5_first_vs_3_rate=("normalization_0_5_first_vs_3", "mean"),
        moved_further_to_3_first_vs_0_5_rate=("moved_further_to_3_first_vs_0_5", "mean"),
        normalization_0_first_vs_3_rate=("normalization_0_first_vs_3", "mean"),
        moved_further_to_3_first_vs_0_rate=("moved_further_to_3_first_vs_0", "mean"),
        normalization_0_5_first_vs_4_rate=("normalization_0_5_first_vs_4", "mean"),
        moved_further_to_4_first_vs_0_5_rate=("moved_further_to_4_first_vs_0_5", "mean"),
        median_bars_to_first_touch_0_5_vs_3=("bars_to_first_touch_0_5_vs_3", "median"),
        behavior_size_pips_proxy_median=("behavior_size_pips_proxy_valid", "median"),
        behavior_size_pips_proxy_p25=("behavior_size_pips_proxy_valid", lambda s: np.nanpercentile(s.dropna(), 25) if s.notna().any() else np.nan),
        behavior_size_pips_proxy_p75=("behavior_size_pips_proxy_valid", lambda s: np.nanpercentile(s.dropna(), 75) if s.notna().any() else np.nan),
        behavior_size_pips_proxy_p90=("behavior_size_pips_proxy_valid", lambda s: np.nanpercentile(s.dropna(), 90) if s.notna().any() else np.nan),
        a_move_pips_median=("a_move_pips_valid", "median"),
        b_move_pips_median=("b_move_pips_valid", "median"),
        sum_leg_move_pips_proxy_median=("sum_leg_move_pips_proxy_valid", "median"),
    ).reset_index()

    summary = summary[summary["observations"] >= args.min_observations].copy()
    summary["grouping"] = "pair_window_horizon_session_hour_abs_z"

    friction_vals = []
    for p in summary["pair_name"]:
        cfg = COST_PROFILES[args.cost_profile].get(p)
        if cfg is None:
            raise ValueError(f"No cost profile for pair={p} profile={args.cost_profile}")
        friction_vals.append(round_trip_friction(cfg))
    summary["estimated_round_trip_friction_pips"] = friction_vals
    summary["cost_share_of_median_behavior"] = summary["estimated_round_trip_friction_pips"] / summary["behavior_size_pips_proxy_median"].clip(lower=0.01)
    summary["feasibility_band"] = np.where(
        summary["behavior_size_pips_proxy_median"].isna(),
        "insufficient_size_data",
        summary["cost_share_of_median_behavior"].apply(feasibility_band),
    )

    csv_out = out_dir / "behavior_size_in_pips_summary.csv"
    summary.to_csv(csv_out, index=False)

    md_out = out_dir / "behavior_size_in_pips_summary.md"
    lines: List[str] = []
    lines.append("# Behavior Size in Pips - First-Touch Feasibility Screening")
    lines.append("")
    lines.append("**Warning:** This report is descriptive feasibility screening only. It is not profitability analysis and not a strategy tester.")
    lines.append("")
    lines.append(f"- Cost profile: **{args.cost_profile}**")
    lines.append(f"- Minimum observations per context: **{args.min_observations}**")
    lines.append("")
    lines.append("## Cost assumptions")
    lines.append("")
    lines.append("| pair_name | spread_a_pips | spread_b_pips | slippage_a_pips | slippage_b_pips | commission_equivalent_pips_total | estimated_round_trip_friction_pips |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for pair in pairs:
        cfg = COST_PROFILES[args.cost_profile].get(pair)
        if cfg is None:
            continue
        fr = round_trip_friction(cfg)
        lines.append(f"| {pair} | {cfg['spread_a_pips']:.2f} | {cfg['spread_b_pips']:.2f} | {cfg['slippage_a_pips']:.2f} | {cfg['slippage_b_pips']:.2f} | {cfg['commission_equivalent_pips_total']:.2f} | {fr:.2f} |")
    lines.append("")

    lines.append("## Summary by pair/window/horizon/abs_zscore_bucket")
    cols = ["pair_name", "window", "horizon_bars", "abs_zscore_bucket", "observations", "normalization_0_5_first_vs_3_rate", "behavior_size_pips_proxy_median", "estimated_round_trip_friction_pips", "cost_share_of_median_behavior", "feasibility_band"]
    lines.append(summary[cols].sort_values(["pair_name", "window", "horizon_bars", "abs_zscore_bucket"]).head(50).to_markdown(index=False))
    lines.append("")

    best = summary.sort_values(["cost_share_of_median_behavior", "normalization_0_5_first_vs_3_rate"], ascending=[True, False]).head(15)
    worst = summary.sort_values(["cost_share_of_median_behavior", "normalization_0_5_first_vs_3_rate"], ascending=[False, True]).head(15)
    lines.append("## Best low-cost-share normalization-first contexts")
    lines.append(best[["pair_name", "window", "horizon_bars", "session_bucket", "hour", "abs_zscore_bucket", "cost_share_of_median_behavior", "normalization_0_5_first_vs_3_rate", "behavior_size_pips_proxy_median"]].to_markdown(index=False))
    lines.append("")
    lines.append("## Worst high-cost-share contexts")
    lines.append(worst[["pair_name", "window", "horizon_bars", "session_bucket", "hour", "abs_zscore_bucket", "cost_share_of_median_behavior", "normalization_0_5_first_vs_3_rate", "behavior_size_pips_proxy_median"]].to_markdown(index=False))
    lines.append("")

    lines.append("## Comparison: EURUSD_GBPUSD vs GBPAUD_GBPNZD")
    comp = summary.groupby("pair_name", as_index=False).agg(
        rows=("pair_name", "size"),
        median_behavior_size_pips=("behavior_size_pips_proxy_median", "median"),
        median_cost_share=("cost_share_of_median_behavior", "median"),
        low_cost_share_rate=("feasibility_band", lambda s: (s == "low_cost_share").mean()),
    )
    lines.append(comp.to_markdown(index=False))
    lines.append("")

    lines.append("## Notes")
    lines.append("- behavior size is measured from actual current close to first-touch close (normalization-first events only).")
    lines.append("- This remains descriptive feasibility screening and is not profitability.")
    lines.append("- This still ignores live spread series and broker execution.")

    if (summary["behavior_size_pips_proxy_median"] > 500).any():
        lines.append('- Large movement proxy detected; inspect pair/time context.')
    missing_size_ratio = summary["behavior_size_pips_proxy_median"].isna().mean() if len(summary) else 0.0
    if missing_size_ratio > 0.25:
        lines.append(f"- Warning: many contexts have missing size data ({missing_size_ratio:.1%}).")

    md_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] wrote {csv_out}")
    print(f"[done] wrote {md_out}")


if __name__ == "__main__":
    main()
