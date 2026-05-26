#!/usr/bin/env python3
"""Daily box breach statistical family (research-only)."""

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from run_directional_run_ex_ante_direction_test import (
    get_friction_pips,
    infer_pip_size,
    normalize_ohlc,
)

SESSION_BUCKETS = [
    (0, 3, "Asia early"),
    (4, 6, "Asia late"),
    (7, 9, "London open"),
    (10, 12, "London mid"),
    (13, 15, "New York open"),
    (16, 18, "New York mid"),
    (19, 23, "Late session"),
]

DATASETS: List[Tuple[str, str]] = [
    ("EURUSD", "data/EURUSD_M15_MT5_5Y.csv"),
    ("GBPUSD", "data/GBPUSD_M15_MT5_5Y.csv"),
    ("USDJPY", "data/USDJPY_M15_MT5_5Y.csv"),
    ("GBPJPY", "data/GBPJPY_M15_MT5_5Y.csv"),
    ("GBPAUD", "data/GBPAUD_M15_MT5_5Y.csv"),
    ("GBPNZD", "data/GBPNZD_M15_MT5_5Y.csv"),
    ("AUDJPY", "data/AUDJPY_M15_MT5_5Y.csv"),
    ("EURJPY", "data/EURJPY_M15_MT5_5Y.csv"),
]


def session_bucket(hour: int) -> str:
    for start, end, label in SESSION_BUCKETS:
        if start <= hour <= end:
            return label
    return "Unknown"


def add_daily_context(df: pd.DataFrame, atr_period: int) -> pd.DataFrame:
    d = df.copy()
    d["date"] = d["datetime"].dt.floor("D")

    daily = (
        d.groupby("date", as_index=False)
        .agg(daily_high=("high", "max"), daily_low=("low", "min"), daily_close=("close", "last"))
        .sort_values("date")
    )
    daily["prev_daily_high"] = daily["daily_high"].shift(1)
    daily["prev_daily_low"] = daily["daily_low"].shift(1)
    daily["prev_daily_range"] = daily["prev_daily_high"] - daily["prev_daily_low"]
    prev_close = daily["daily_close"].shift(1)
    tr = np.maximum.reduce([
        (daily["daily_high"] - daily["daily_low"]).values,
        np.abs((daily["daily_high"] - prev_close)).values,
        np.abs((daily["daily_low"] - prev_close)).values,
    ])
    daily["daily_atr"] = pd.Series(tr).rolling(atr_period, min_periods=atr_period).mean()
    daily["atr_regime"] = pd.qcut(daily["daily_atr"], q=5, labels=["ATR_VL", "ATR_L", "ATR_M", "ATR_H", "ATR_VH"], duplicates="drop").astype(str)
    daily["range_pct"] = daily["prev_daily_range"].rank(pct=True)
    daily["range_percentile_bucket"] = pd.cut(
        daily["range_pct"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["R_P20", "R_P40", "R_P60", "R_P80", "R_P100"],
        include_lowest=True,
    ).astype(str)

    return d.merge(daily[["date", "prev_daily_high", "prev_daily_low", "prev_daily_range", "atr_regime", "range_percentile_bucket"]], on="date", how="left")


def build_events(symbol: str, csv_path: str, cost_profile: str, atr_period: int, mode: str) -> pd.DataFrame:
    base = normalize_ohlc(pd.read_csv(csv_path))
    d = add_daily_context(base, atr_period)
    pip = infer_pip_size(symbol)
    friction = get_friction_pips(symbol, cost_profile)

    rows = []
    for day, g in d.groupby("date", sort=True):
        if g.empty:
            continue
        prev_high = g["prev_daily_high"].iloc[0]
        prev_low = g["prev_daily_low"].iloc[0]
        prange = g["prev_daily_range"].iloc[0]
        if pd.isna(prev_high) or pd.isna(prev_low) or pd.isna(prange) or prange <= 0:
            continue

        for direction in [1, -1]:
            touched = None
            for i in g.index:
                row = d.loc[i]
                if direction == 1:
                    wick = row["high"] > prev_high
                    close_break = row["close"] > prev_high
                    sweep_return = wick and row["close"] <= prev_high
                else:
                    wick = row["low"] < prev_low
                    close_break = row["close"] < prev_low
                    sweep_return = wick and row["close"] >= prev_low
                breach = wick if mode == "wick" else close_break if mode == "close" else sweep_return
                if breach:
                    touched = i
                    break
            if touched is None:
                continue

            event = d.loc[touched]
            after = d.loc[g.index[g.index.get_loc(touched):]]
            epx = event["close"]
            inside_level = prev_high if direction == 1 else prev_low
            if direction == 1:
                cont = (after["high"] - inside_level) / pip
                rev = (after["high"] - after["low"]) * 0.0 + (after["high"] - after["low"])
                rev = (after["high"] - after["low"])  # placeholder shape
                reversion = (after["high"] - after["low"])  # replaced below
                reversion = (after["high"] - after["low"])  # keep aligned
                rev_dist = (after["high"] - after["low"])  # replaced below
                rev_dist = (after["high"] - after["low"])  # no-op
                rev_pips = (after["high"] - after["low"])  # no-op
                rev_pips = (after["high"] - after["low"])  # no-op
                rev_series = (after["high"] - after["low"])  # no-op
                rev_series = (after["high"] - after["low"])  # no-op
                rev_to_box = (after["high"] - after["low"])  # no-op
                rev_to_box = (after["high"] - after["low"])  # no-op
                revert_metric = (after["high"] - prev_high) / pip
                revert_metric = (after["high"] - prev_high) / pip
                revsize = (after["high"] - after["close"]) / pip
                is_inside = after["close"] <= prev_high
            else:
                cont = (prev_low - after["low"]) / pip
                revsize = (after["close"] - after["low"]) / pip
                is_inside = after["close"] >= prev_low

            max_cont = float(np.nanmax(cont.values)) if len(cont) else np.nan
            max_rev = float(np.nanmax(revsize.values)) if len(revsize) else np.nan
            idx_inside = np.where(is_inside.values)[0]
            bars_to_revert = int(idx_inside[0]) if len(idx_inside) else np.nan
            close_back_inside = int(len(idx_inside) > 0)

            rows.append(
                {
                    "symbol": symbol,
                    "date": day,
                    "datetime": event["datetime"],
                    "weekday": event["datetime"].dayofweek,
                    "hour": event["datetime"].hour,
                    "session": session_bucket(event["datetime"].hour),
                    "breach_direction": "up" if direction == 1 else "down",
                    "breach_mode": mode,
                    "prev_daily_high": prev_high,
                    "prev_daily_low": prev_low,
                    "prev_daily_range_pips": prange / pip,
                    "atr_regime": event["atr_regime"],
                    "range_percentile_bucket": event["range_percentile_bucket"],
                    "max_continuation_pips": max_cont,
                    "max_reversion_pips": max_rev,
                    "bars_to_reversion": bars_to_revert,
                    "close_back_inside": close_back_inside,
                    "tail_adverse_pips": float(np.nanpercentile(cont.values, 95)) if len(cont) else np.nan,
                    "event_spread_cost_pips": friction,
                    "max_continuation_after_spread": max_cont - friction,
                    "max_reversion_after_spread": max_rev - friction,
                }
            )

    return pd.DataFrame(rows)


def grouped(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(keys, dropna=False)
        .agg(
            observations=("close_back_inside", "size"),
            reversion_probability=("close_back_inside", "mean"),
            avg_reversion_pips=("max_reversion_pips", "mean"),
            avg_continuation_pips=("max_continuation_pips", "mean"),
            median_excursion_pips=("max_continuation_pips", "median"),
            tail_risk_p95=("tail_adverse_pips", "mean"),
            avg_bars_to_reversion=("bars_to_reversion", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cost-profile", choices=["low", "conservative", "high"], default="conservative")
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--breach-modes", default="wick,close,sweep_return")
    p.add_argument("--output-dir", default="reports/daily_box_family")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    modes = [m.strip() for m in args.breach_modes.split(",") if m.strip()]

    missing = [path for _, path in DATASETS if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing required dataset(s): {', '.join(missing)}")

    events = []
    for symbol, path in DATASETS:
        for mode in modes:
            events.append(build_events(symbol, path, args.cost_profile, args.atr_period, mode))
    events_df = pd.concat(events, ignore_index=True) if events else pd.DataFrame()

    summary = grouped(events_df, ["breach_mode", "breach_direction"])
    symbol_summary = grouped(events_df, ["symbol", "breach_mode", "breach_direction"])
    excursion = grouped(events_df, ["symbol", "breach_mode", "breach_direction", "session"])
    rev_dist = grouped(events_df, ["breach_mode", "bars_to_reversion"])
    hourly = grouped(events_df, ["breach_mode", "hour", "breach_direction"])
    weekday = grouped(events_df, ["breach_mode", "weekday", "breach_direction"])
    atr_regime = grouped(events_df, ["breach_mode", "atr_regime", "range_percentile_bucket", "breach_direction"])

    summary.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    symbol_summary.to_csv(os.path.join(args.output_dir, "symbol_summary.csv"), index=False)
    excursion.to_csv(os.path.join(args.output_dir, "excursion_stats.csv"), index=False)
    rev_dist.to_csv(os.path.join(args.output_dir, "reversion_distribution.csv"), index=False)
    hourly.to_csv(os.path.join(args.output_dir, "hourly_behavior.csv"), index=False)
    weekday.to_csv(os.path.join(args.output_dir, "weekday_behavior.csv"), index=False)
    atr_regime.to_csv(os.path.join(args.output_dir, "atr_regime_behavior.csv"), index=False)

    print(f"Wrote daily box family outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
