#!/usr/bin/env python3
"""Standalone EURUSD M15 pip reversion probability lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PIP_SIZE = 0.0001
ANCHORS = ["rolling_16_close", "rolling_32_close", "daily_open"]
DIRECTIONS = ["LONG", "SHORT"]
MOVE_X_PIPS = [5, 7, 10, 12, 15, 20, 25, 30]
CORRECTION_PIPS = [5, 6, 7, 8, 9, 10]
ADVERSE_PIPS = [10, 15, 20, 25, 30, 40, 50]
MAX_HOLD_BARS = [10, 20, 40, 80]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EURUSD M15 pip reversion probability lab")
    parser.add_argument("--csv", default="data/EURUSD_M15_MT5_5Y.csv")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--base-timeframe", default="M15")
    parser.add_argument("--spread-pips", type=float, default=1.0)
    parser.add_argument("--slippage-pips", type=float, default=0.3)
    parser.add_argument("--output-dir", default="pip_reversion_probability_reports")
    return parser.parse_args()


def session_bucket(hour: int) -> str:
    if 0 <= hour <= 3:
        return "00-03 Asia early"
    if 4 <= hour <= 6:
        return "04-06 Asia late"
    if 7 <= hour <= 9:
        return "07-09 London open"
    if 10 <= hour <= 12:
        return "10-12 London mid"
    if 13 <= hour <= 15:
        return "13-15 New York open"
    if 16 <= hour <= 18:
        return "16-18 New York mid"
    return "19-23 Late session"


def load_data(csv_path: str, symbol: str, timeframe: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols_lower = {c.lower(): c for c in df.columns}
    rename_map = {}
    for expected in ["datetime", "open", "high", "low", "close", "symbol", "timeframe"]:
        if expected in cols_lower:
            rename_map[cols_lower[expected]] = expected
    df = df.rename(columns=rename_map)

    required = ["datetime", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).copy()

    if "symbol" in df.columns:
        df = df[df["symbol"].astype(str).str.upper() == symbol.upper()].copy()
    if "timeframe" in df.columns:
        df = df[df["timeframe"].astype(str).str.upper() == timeframe.upper()].copy()

    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["session_bucket"] = df["hour"].map(session_bucket)
    df["rolling_16_close"] = df["close"].shift(16)
    df["rolling_32_close"] = df["close"].shift(32)
    df["daily_open"] = df.groupby("date")["open"].transform("first")
    return df


def simulate_events(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    records: List[Dict] = []
    cost_pips = args.spread_pips + args.slippage_pips

    for anchor_type in ANCHORS:
        for direction in DIRECTIONS:
            for move_x in MOVE_X_PIPS:
                move_px = move_x * PIP_SIZE
                for i in range(len(df) - 1):
                    anchor = df.at[i, anchor_type]
                    if pd.isna(anchor):
                        continue
                    signal_close = df.at[i, "close"]
                    is_signal = (
                        signal_close <= anchor - move_px
                        if direction == "LONG"
                        else signal_close >= anchor + move_px
                    )
                    if not is_signal:
                        continue

                    entry_idx = i + 1
                    entry_price = df.at[entry_idx, "open"]
                    entry_dt = df.at[entry_idx, "datetime"]

                    for correction_pips in CORRECTION_PIPS:
                        for adverse_pips in ADVERSE_PIPS:
                            for max_hold in MAX_HOLD_BARS:
                                target_px = correction_pips * PIP_SIZE
                                adverse_px = adverse_pips * PIP_SIZE
                                target = entry_price + target_px if direction == "LONG" else entry_price - target_px
                                adverse = entry_price - adverse_px if direction == "LONG" else entry_price + adverse_px

                                outcome = "timeout"
                                bars_to_outcome = max_hold
                                exit_close = df.at[min(entry_idx + max_hold, len(df) - 1), "close"]
                                max_fav = 0.0
                                max_adv = 0.0

                                end_idx = min(entry_idx + max_hold, len(df) - 1)
                                for j in range(entry_idx, end_idx + 1):
                                    hi = df.at[j, "high"]
                                    lo = df.at[j, "low"]
                                    if direction == "LONG":
                                        max_fav = max(max_fav, (hi - entry_price) / PIP_SIZE)
                                        max_adv = max(max_adv, (entry_price - lo) / PIP_SIZE)
                                        hit_target = hi >= target
                                        hit_adverse = lo <= adverse
                                    else:
                                        max_fav = max(max_fav, (entry_price - lo) / PIP_SIZE)
                                        max_adv = max(max_adv, (hi - entry_price) / PIP_SIZE)
                                        hit_target = lo <= target
                                        hit_adverse = hi >= adverse

                                    if hit_adverse and hit_target:
                                        outcome = "adverse_failure"
                                        bars_to_outcome = j - entry_idx + 1
                                        break
                                    if hit_adverse:
                                        outcome = "adverse_failure"
                                        bars_to_outcome = j - entry_idx + 1
                                        break
                                    if hit_target:
                                        outcome = "hit"
                                        bars_to_outcome = j - entry_idx + 1
                                        break

                                if outcome == "hit":
                                    gross = float(correction_pips)
                                elif outcome == "adverse_failure":
                                    gross = float(-adverse_pips)
                                else:
                                    if direction == "LONG":
                                        gross = float((exit_close - entry_price) / PIP_SIZE)
                                    else:
                                        gross = float((entry_price - exit_close) / PIP_SIZE)

                                records.append(
                                    {
                                        "symbol": args.symbol,
                                        "anchor_type": anchor_type,
                                        "direction": direction,
                                        "signal_datetime": df.at[i, "datetime"],
                                        "entry_datetime": entry_dt,
                                        "hour": int(df.at[i, "hour"]),
                                        "session_bucket": df.at[i, "session_bucket"],
                                        "anchor_price": float(anchor),
                                        "signal_close": float(signal_close),
                                        "entry_price": float(entry_price),
                                        "move_x_pips": move_x,
                                        "correction_pips": correction_pips,
                                        "adverse_pips": adverse_pips,
                                        "max_hold_bars": max_hold,
                                        "outcome": outcome,
                                        "bars_to_outcome": int(bars_to_outcome),
                                        "max_favorable_pips": float(max_fav),
                                        "max_adverse_pips_seen": float(max_adv),
                                        "gross_pips_if_traded": gross,
                                        "cost_pips": float(cost_pips),
                                        "net_pips_after_costs": gross - cost_pips,
                                    }
                                )

    ev = pd.DataFrame.from_records(records)
    if ev.empty:
        return ev
    ev = ev.sort_values("signal_datetime").reset_index(drop=True)
    n = len(ev)
    ev["split"] = np.where(np.arange(n) < int(0.7 * n), "IS", "OOS")
    ev["wf_window"] = pd.cut(np.arange(n), bins=3, labels=[1, 2, 3], include_lowest=True).astype(int)
    return ev


def aggregate(events: pd.DataFrame) -> pd.DataFrame:
    base_keys = ["anchor_type", "direction", "move_x_pips", "correction_pips", "adverse_pips", "max_hold_bars", "hour", "session_bucket"]
    variants = [
        ("full", events.copy()),
        ("hour_all", events.assign(hour="ALL")),
        ("session_all", events.assign(session_bucket="ALL")),
        ("hour_session_all", events.assign(hour="ALL", session_bucket="ALL")),
        ("direction_all", events.assign(direction="ALL")),
        ("dir_hour_all", events.assign(direction="ALL", hour="ALL")),
        ("dir_session_all", events.assign(direction="ALL", session_bucket="ALL")),
        ("dir_hour_session_all", events.assign(direction="ALL", hour="ALL", session_bucket="ALL")),
    ]

    rows = []
    for _, dfv in variants:
        for keys, g in dfv.groupby(base_keys, dropna=False):
            row = dict(zip(base_keys, keys))
            row["events"] = len(g)
            row["hit_rate"] = (g["outcome"] == "hit").mean()
            row["adverse_failure_rate"] = (g["outcome"] == "adverse_failure").mean()
            row["timeout_rate"] = (g["outcome"] == "timeout").mean()
            hit_bars = g.loc[g["outcome"] == "hit", "bars_to_outcome"]
            row["avg_bars_to_hit"] = float(hit_bars.mean()) if len(hit_bars) else np.nan
            row["median_bars_to_hit"] = float(hit_bars.median()) if len(hit_bars) else np.nan
            row["avg_net_pips_after_costs"] = g["net_pips_after_costs"].mean()
            row["total_net_pips_after_costs"] = g["net_pips_after_costs"].sum()
            pos = g.loc[g["net_pips_after_costs"] > 0, "net_pips_after_costs"].sum()
            neg = -g.loc[g["net_pips_after_costs"] < 0, "net_pips_after_costs"].sum()
            row["profit_factor"] = float(pos / neg) if neg > 0 else np.nan
            row["avg_max_adverse_pips_seen"] = g["max_adverse_pips_seen"].mean()
            row["p95_max_adverse_pips_seen"] = g["max_adverse_pips_seen"].quantile(0.95)
            row["max_adverse_pips_seen"] = g["max_adverse_pips_seen"].max()

            is_g = g[g["split"] == "IS"]
            oos_g = g[g["split"] == "OOS"]
            row["IS_events"] = len(is_g)
            row["OOS_events"] = len(oos_g)
            row["IS_hit_rate"] = (is_g["outcome"] == "hit").mean() if len(is_g) else np.nan
            row["OOS_hit_rate"] = (oos_g["outcome"] == "hit").mean() if len(oos_g) else np.nan
            row["IS_avg_net"] = is_g["net_pips_after_costs"].mean() if len(is_g) else np.nan
            row["OOS_avg_net"] = oos_g["net_pips_after_costs"].mean() if len(oos_g) else np.nan
            row["OOS_agrees_with_IS"] = bool((row["IS_avg_net"] > 0) and (row["OOS_avg_net"] > 0)) if pd.notna(row["IS_avg_net"]) and pd.notna(row["OOS_avg_net"]) else False

            wf = g.groupby("wf_window").agg(hit_rate=("outcome", lambda s: (s == "hit").mean()), total_net=("net_pips_after_costs", "sum"))
            row["wf_positive_windows"] = int((wf["total_net"] > 0).sum()) if len(wf) else 0
            row["wf_total_windows"] = int(len(wf))
            rows.append(row)

    res = pd.DataFrame(rows)
    res = res.sort_values(
        ["events", "OOS_hit_rate", "OOS_avg_net", "wf_positive_windows", "p95_max_adverse_pips_seen", "max_adverse_pips_seen", "hit_rate"],
        ascending=[False, False, False, False, True, True, False],
    ).reset_index(drop=True)
    return res


def df_to_md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    return df.to_markdown(index=False)


def write_summary(results: pd.DataFrame, out_path: Path) -> None:
    eligible = results[results["events"] >= 200].copy()
    top_oos_hit = eligible.sort_values(["OOS_hit_rate", "events"], ascending=[False, False]).head(30)
    top_total_net = eligible.sort_values(["total_net_pips_after_costs", "events"], ascending=[False, False]).head(30)
    top_avg_net = eligible.sort_values(["avg_net_pips_after_costs", "events"], ascending=[False, False]).head(30)

    corr = eligible.groupby("correction_pips", as_index=False).agg(events=("events", "sum"), avg_oos_hit=("OOS_hit_rate", "mean"), avg_oos_net=("OOS_avg_net", "mean")).sort_values("avg_oos_net", ascending=False)
    hours = eligible[eligible["hour"].astype(str) != "ALL"].groupby("hour", as_index=False).agg(events=("events", "sum"), avg_oos_hit=("OOS_hit_rate", "mean"), avg_oos_net=("OOS_avg_net", "mean")).sort_values("avg_oos_net", ascending=False).head(10)
    sessions = eligible[eligible["session_bucket"] != "ALL"].groupby("session_bucket", as_index=False).agg(events=("events", "sum"), avg_oos_hit=("OOS_hit_rate", "mean"), avg_oos_net=("OOS_avg_net", "mean")).sort_values("avg_oos_net", ascending=False)
    long_short = eligible[eligible["direction"].isin(["LONG", "SHORT"])].groupby("direction", as_index=False).agg(events=("events", "sum"), avg_oos_hit=("OOS_hit_rate", "mean"), avg_oos_net=("OOS_avg_net", "mean"))
    anchors = eligible.groupby("anchor_type", as_index=False).agg(events=("events", "sum"), avg_oos_hit=("OOS_hit_rate", "mean"), avg_oos_net=("OOS_avg_net", "mean")).sort_values("avg_oos_net", ascending=False)

    md = []
    md.append("# EURUSD Pip Reversion Probability Lab Summary")
    md.append("\n## Top 30 by OOS hit rate (events >= 200)\n")
    md.append(df_to_md_table(top_oos_hit))
    md.append("\n## Top 30 by total net pips (events >= 200)\n")
    md.append(df_to_md_table(top_total_net))
    md.append("\n## Top 30 by avg net pips (events >= 200)\n")
    md.append(df_to_md_table(top_avg_net))
    md.append("\n## Best correction_pips comparison (5 to 10)\n")
    md.append(df_to_md_table(corr))
    md.append("\n## Best hours\n")
    md.append(df_to_md_table(hours))
    md.append("\n## Best session buckets\n")
    md.append(df_to_md_table(sessions))
    md.append("\n## LONG vs SHORT comparison\n")
    md.append(df_to_md_table(long_short))
    md.append("\n## Anchor comparison\n")
    md.append(df_to_md_table(anchors))
    md.append("\n## Warning\n")
    md.append("High hit rate alone is not enough; adverse excursion size and trading costs (spread/slippage) can eliminate edge.")

    out_path.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = load_data(args.csv, args.symbol, args.base_timeframe)
    events = simulate_events(data, args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / f"{args.symbol}_pip_reversion_events.csv"
    results_path = out_dir / f"{args.symbol}_pip_reversion_results.csv"
    best_path = out_dir / f"{args.symbol}_pip_reversion_best.json"
    summary_path = out_dir / f"{args.symbol}_pip_reversion_summary.md"

    if events.empty:
        events.to_csv(events_path, index=False)
        pd.DataFrame().to_csv(results_path, index=False)
        best_path.write_text(json.dumps({"warning": "No events generated."}, indent=2), encoding="utf-8")
        summary_path.write_text("# EURUSD Pip Reversion Probability Lab Summary\n\nNo events generated.", encoding="utf-8")
        return

    results = aggregate(events)
    events.to_csv(events_path, index=False)
    results.to_csv(results_path, index=False)

    best_eligible = results[results["events"] >= 200].copy()
    best = best_eligible.head(30) if not best_eligible.empty else results.head(30)
    best_path.write_text(best.to_json(orient="records", indent=2), encoding="utf-8")
    write_summary(results, summary_path)


if __name__ == "__main__":
    main()
