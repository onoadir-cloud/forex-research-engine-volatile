#!/usr/bin/env python3
"""Daily box breach behavior family (research-only, no trading logic)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_loader import load_ohlc_csv
from src.sessions import add_session_features

PIP_SIZE_FALLBACK = 0.0001
PIP_SIZE_JPY = 0.01


def infer_pip_size(symbol: str) -> float:
    return PIP_SIZE_JPY if "JPY" in symbol.upper() else PIP_SIZE_FALLBACK


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="symbols_config.json")
    p.add_argument("--data-template", default="data/{symbol}_M15_MT5_5Y.csv")
    p.add_argument("--base-timeframe", default="M15")
    p.add_argument("--output-dir", default="reports/daily_box_family")
    p.add_argument("--range-percentile-buckets", type=int, default=5)
    return p.parse_args()


def load_symbols(config_path: Path) -> list[str]:
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    symbols = payload.get("symbols", [])
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("symbols_config.json must contain non-empty 'symbols' list")
    return [str(s) for s in symbols]




def candidate_priority(csv_path: Path) -> tuple[int, str]:
    name = csv_path.name.upper()
    if name.endswith("_CLEAN.CSV"):
        return (0, name)
    if name.endswith("_MT5_5Y.CSV"):
        return (1, name)
    return (2, name)


def discover_symbol_csv_files(data_dir: Path, configured_symbols: list[str], base_timeframe: str) -> dict[str, Path]:
    csv_files = sorted(data_dir.glob("*.csv"))
    print(f"[info] scanning data folder: {data_dir}")
    if csv_files:
        print("[info] discovered CSV files:")
        for csv_file in csv_files:
            print(f"  - {csv_file}")
    else:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'. Expected symbol files such as "
            "'EURUSD_M15_MT5_5Y.csv' or 'EURUSD_M15.csv'."
        )

    tf = base_timeframe.upper()
    configured_set = {s.upper() for s in configured_symbols}
    candidates: dict[str, list[Path]] = {s: [] for s in configured_set}

    for csv_file in csv_files:
        stem_upper = csv_file.stem.upper()
        if tf not in stem_upper:
            continue
        for symbol in configured_set:
            if symbol in stem_upper:
                candidates[symbol].append(csv_file)

    print(f"[info] discovered symbols with timeframe {tf}:")
    discovered = sorted([s for s, files in candidates.items() if files])
    if discovered:
        print("  - " + ", ".join(discovered))
    else:
        print("  - none")

    selected: dict[str, Path] = {}
    for symbol in sorted(configured_set):
        symbol_candidates = candidates[symbol]
        if not symbol_candidates:
            continue
        ranked = sorted(symbol_candidates, key=candidate_priority)
        selected[symbol] = ranked[0]

        if len(ranked) > 1:
            print(f"[info] duplicate candidates for {symbol}; selected: {ranked[0]}")
            for option in ranked:
                print(f"    candidate: {option}")

    print("[info] selected symbol files:")
    for symbol in sorted(configured_symbols, key=str.upper):
        csv_path = selected.get(symbol.upper())
        if csv_path is not None:
            print(f"  - {symbol}: {csv_path}")

    return selected

def pct_rank_bucket(series: pd.Series, q: int) -> pd.Series:
    ranks = series.rank(method="first", pct=True)
    edges = np.linspace(0.0, 1.0, q + 1)
    labels = [f"Q{i+1}" for i in range(q)]
    return pd.cut(ranks, bins=edges, labels=labels, include_lowest=True)


def extract_symbol_events(symbol: str, csv_path: Path, percentile_buckets: int) -> pd.DataFrame:
    raw, _ = load_ohlc_csv(str(csv_path))
    df = add_session_features(raw)
    df["trading_date"] = df["datetime"].dt.floor("D")

    daily = (
        df.groupby("trading_date", as_index=False)
        .agg(daily_high=("high", "max"), daily_low=("low", "min"))
        .sort_values("trading_date")
    )
    daily["prev_daily_high"] = daily["daily_high"].shift(1)
    daily["prev_daily_low"] = daily["daily_low"].shift(1)
    daily["prev_daily_range"] = daily["prev_daily_high"] - daily["prev_daily_low"]
    daily["range_bucket"] = pct_rank_bucket(daily["prev_daily_range"], percentile_buckets)

    day_meta = daily[["trading_date", "prev_daily_high", "prev_daily_low", "prev_daily_range", "range_bucket"]].dropna()
    intraday = df.merge(day_meta, on="trading_date", how="inner").sort_values("datetime").reset_index(drop=True)

    events: list[dict[str, Any]] = []
    pip_size = infer_pip_size(symbol)

    for day, day_df in intraday.groupby("trading_date", sort=True):
        row0 = day_df.iloc[0]
        day_end_close = float(day_df.iloc[-1]["close"])
        box_high = float(row0["prev_daily_high"])
        box_low = float(row0["prev_daily_low"])
        box_range = float(row0["prev_daily_range"])

        for side in ("upside", "downside"):
            if side == "upside":
                breach_mask = day_df["high"] >= box_high
            else:
                breach_mask = day_df["low"] <= box_low

            if not breach_mask.any():
                continue

            breach_idx = int(np.flatnonzero(breach_mask.to_numpy())[0])
            after = day_df.iloc[breach_idx:].copy()
            breach_bar = after.iloc[0]

            inside_mask = (after["close"] <= box_high) & (after["close"] >= box_low)
            reverted = bool(inside_mask.any())

            if reverted:
                rev_pos = int(np.flatnonzero(inside_mask.to_numpy())[0])
                rev_bar = after.iloc[rev_pos]
                mins_to_reversion = int((rev_bar["datetime"] - breach_bar["datetime"]).total_seconds() // 60)
                bars_to_reversion = rev_pos
            else:
                mins_to_reversion = np.nan
                bars_to_reversion = np.nan

            if side == "upside":
                continuation_px = float(after["high"].max() - box_high)
                reversion_px = float(max(0.0, box_high - after["low"].min()))
                breach_distance_px = float(max(0.0, breach_bar["high"] - box_high))
            else:
                continuation_px = float(box_low - after["low"].min())
                reversion_px = float(max(0.0, after["high"].max() - box_low))
                breach_distance_px = float(max(0.0, box_low - breach_bar["low"]))

            eod_location = (
                "above_range" if day_end_close > box_high else "below_range" if day_end_close < box_low else "inside_range"
            )

            events.append(
                {
                    "symbol": symbol,
                    "trading_date": day.date().isoformat(),
                    "breach_side": side,
                    "breach_datetime": breach_bar["datetime"],
                    "breach_hour": int(breach_bar["hour"]),
                    "weekday": int(breach_bar["day_of_week"]),
                    "session": str(breach_bar["session"]),
                    "prev_daily_high": box_high,
                    "prev_daily_low": box_low,
                    "prev_daily_range": box_range,
                    "range_bucket": str(breach_bar["range_bucket"]),
                    "breach_distance_pips": breach_distance_px / pip_size,
                    "reverted_inside_range": reverted,
                    "minutes_to_reversion": mins_to_reversion,
                    "bars_to_reversion": bars_to_reversion,
                    "max_continuation_pips": continuation_px / pip_size,
                    "max_reversion_pips": reversion_px / pip_size,
                    "worst_continuation_minus_reversion_pips": (continuation_px - reversion_px) / pip_size,
                    "eod_location": eod_location,
                }
            )

    return pd.DataFrame(events)


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(group_cols, dropna=False)
    out = grouped.agg(
        total_breach_events=("breach_side", "size"),
        reversion_probability=("reverted_inside_range", "mean"),
        avg_time_to_reversion_minutes=("minutes_to_reversion", "mean"),
        median_time_to_reversion_minutes=("minutes_to_reversion", "median"),
        avg_continuation_pips=("max_continuation_pips", "mean"),
        median_continuation_pips=("max_continuation_pips", "median"),
        avg_reversion_pips=("max_reversion_pips", "mean"),
        median_reversion_pips=("max_reversion_pips", "median"),
        worst_continuation_against_reversion_pips=("worst_continuation_minus_reversion_pips", "max"),
    ).reset_index()
    return out.sort_values(group_cols).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols = load_symbols(Path(args.config))

    data_template = Path(args.data_template)
    data_dir = data_template.parent if str(data_template.parent) != "" else Path("data")
    symbol_to_csv = discover_symbol_csv_files(data_dir, symbols, args.base_timeframe)

    missing_symbols = [s for s in symbols if s.upper() not in symbol_to_csv]
    if missing_symbols:
        raise FileNotFoundError(
            "Missing required symbol CSV files (must contain symbol and timeframe token): "
            + ", ".join(missing_symbols)
        )

    all_events = [
        extract_symbol_events(symbol, symbol_to_csv[symbol.upper()], args.range_percentile_buckets)
        for symbol in symbols
    ]

    breach_events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()

    summary = summarize(breach_events, ["breach_side", "range_bucket"])
    symbol_summary = summarize(breach_events, ["symbol", "breach_side"])
    hourly_behavior = summarize(breach_events, ["symbol", "breach_side", "breach_hour", "session", "range_bucket"])
    weekday_behavior = summarize(breach_events, ["symbol", "breach_side", "weekday", "range_bucket"])

    breach_events.to_csv(output_dir / "breach_events.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    symbol_summary.to_csv(output_dir / "symbol_summary.csv", index=False)
    hourly_behavior.to_csv(output_dir / "hourly_behavior.csv", index=False)
    weekday_behavior.to_csv(output_dir / "weekday_behavior.csv", index=False)

    print(f"events: {len(breach_events)}")
    print(f"output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
