#!/usr/bin/env python3
"""Standalone lab: EURUSD M15 candle behavior statistics."""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

PIP_SIZE_EURUSD = 0.0001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EURUSD candle behavior lab")
    parser.add_argument("--csv", default="data/EURUSD_M15_MT5_5Y.csv")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--base-timeframe", default="M15")
    parser.add_argument("--spread-pips", type=float, default=1.0)
    parser.add_argument("--slippage-pips", type=float, default=0.3)
    parser.add_argument("--output-dir", default="candle_behavior_reports")
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--write-events", action="store_true", default=False)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-scenarios", type=int, default=0)
    return parser.parse_args()


def session_bucket(hour: int) -> str:
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


def _first_present(columns: Dict[str, str], choices: Iterable[str]) -> str | None:
    for c in choices:
        if c in columns:
            return columns[c]
    return None


def load_data(csv_path: str, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    original = {c.lower().strip(): c for c in df.columns}
    datetime_col = _first_present(original, ["datetime", "time", "date", "timestamp"])
    open_col = _first_present(original, ["open", "o"])
    high_col = _first_present(original, ["high", "h"])
    low_col = _first_present(original, ["low", "l"])
    close_col = _first_present(original, ["close", "c"])
    required = [datetime_col, open_col, high_col, low_col, close_col]
    if any(v is None for v in required):
        raise ValueError("CSV must include datetime/time and OHLC columns")

    df = df.rename(
        columns={
            datetime_col: "datetime",
            open_col: "open",
            high_col: "high",
            low_col: "low",
            close_col: "close",
        }
    )
    df.columns = [c.lower().strip() for c in df.columns]
    df = df[["datetime", "open", "high", "low", "close"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).copy()
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("datetime").reset_index(drop=True)
    df["symbol"] = symbol
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["body_pips"] = (out["close"] - out["open"]).abs() / PIP_SIZE_EURUSD
    out["range_pips"] = (out["high"] - out["low"]) / PIP_SIZE_EURUSD
    out["upper_wick_pips"] = (out["high"] - out[["open", "close"]].max(axis=1)) / PIP_SIZE_EURUSD
    out["lower_wick_pips"] = (out[["open", "close"]].min(axis=1) - out["low"]) / PIP_SIZE_EURUSD

    nonzero_range = out["range_pips"].replace(0, np.nan)
    out["body_to_range"] = (out["body_pips"] / nonzero_range).fillna(0.0)
    out["upper_wick_to_range"] = (out["upper_wick_pips"] / nonzero_range).fillna(0.0)
    out["lower_wick_to_range"] = (out["lower_wick_pips"] / nonzero_range).fillna(0.0)
    out["close_position"] = ((out["close"] - out["low"]) / (out["high"] - out["low"]).replace(0, np.nan)).fillna(0.5)

    out["candle_direction"] = np.where(out["close"] > out["open"], "bull", np.where(out["close"] < out["open"], "bear", "doji"))
    out["dir_code"] = out["candle_direction"].map({"bull": "B", "bear": "R", "doji": "D"})

    bull = out["candle_direction"].eq("bull")
    bear = out["candle_direction"].eq("bear")
    out["consecutive_bull_count"] = bull.groupby((~bull).cumsum()).cumcount() + 1
    out.loc[~bull, "consecutive_bull_count"] = 0
    out["consecutive_bear_count"] = bear.groupby((~bear).cumsum()).cumcount() + 1
    out.loc[~bear, "consecutive_bear_count"] = 0

    out["previous_3_direction"] = (
        out["dir_code"].shift(3).fillna("X") + out["dir_code"].shift(2).fillna("X") + out["dir_code"].shift(1).fillna("X")
    )
    out["sum_body_last_3_pips"] = out["body_pips"].shift(1).rolling(3).sum()
    out["sum_range_last_3_pips"] = out["range_pips"].shift(1).rolling(3).sum()

    out["rolling_16_close"] = out["close"].shift(16)
    out["rolling_32_close"] = out["close"].shift(32)
    out["trading_date"] = out["datetime"].dt.date
    out["daily_open"] = out.groupby("trading_date")["open"].transform("first")

    out["distance_from_rolling_16_pips"] = (out["close"] - out["rolling_16_close"]) / PIP_SIZE_EURUSD
    out["distance_from_rolling_32_pips"] = (out["close"] - out["rolling_32_close"]) / PIP_SIZE_EURUSD
    out["distance_from_daily_open_pips"] = (out["close"] - out["daily_open"]) / PIP_SIZE_EURUSD

    out["hour"] = out["datetime"].dt.hour
    out["day_of_week"] = out["datetime"].dt.dayofweek
    out["session_bucket"] = out["hour"].map(session_bucket)

    out["body_size_bucket"] = pd.cut(out["body_pips"], [-np.inf, 2, 5, 10, 20, np.inf], labels=["tiny", "small", "medium", "large", "extreme"], right=False).astype(str)
    out["range_size_bucket"] = pd.cut(out["range_pips"], [-np.inf, 5, 10, 20, 35, np.inf], labels=["tiny", "small", "medium", "large", "extreme"], right=False).astype(str)
    out["close_position_bucket"] = pd.cut(out["close_position"], [-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf], labels=["bottom_20", "lower_mid", "middle", "upper_mid", "top_20"]).astype(str)

    out["wick_signal_bucket"] = "indecision"
    out.loc[out["lower_wick_to_range"] >= 0.45, "wick_signal_bucket"] = "long_lower_wick"
    out.loc[out["upper_wick_to_range"] >= 0.45, "wick_signal_bucket"] = "long_upper_wick"
    out.loc[out["body_to_range"] >= 0.70, "wick_signal_bucket"] = "full_body"

    out["streak_bucket"] = "mixed"
    out.loc[out["consecutive_bear_count"] >= 3, "streak_bucket"] = "bear_3_plus"
    out.loc[out["consecutive_bull_count"] >= 3, "streak_bucket"] = "bull_3_plus"

    bucket_edges = [-np.inf, 5, 10, 15, 25, np.inf]
    bucket_labels = ["0_5", "5_10", "10_15", "15_25", "25_plus"]
    out["distance_from_rolling_16_abs_bucket"] = pd.cut(out["distance_from_rolling_16_pips"].abs(), bucket_edges, labels=bucket_labels, right=False).astype(str)
    out["distance_from_daily_open_abs_bucket"] = pd.cut(out["distance_from_daily_open_pips"].abs(), bucket_edges, labels=bucket_labels, right=False).astype(str)
    return out


def evaluate_signal(df: pd.DataFrame, signal_idx: int, direction: str, target_pips: int, adverse_pips: int, max_hold_bars: int, cost_pips: float):
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None
    entry = float(df.at[entry_idx, "open"])
    pip = PIP_SIZE_EURUSD
    if direction == "LONG":
        target_px = entry + target_pips * pip
        adverse_px = entry - adverse_pips * pip
    else:
        target_px = entry - target_pips * pip
        adverse_px = entry + adverse_pips * pip

    last_idx = min(len(df) - 1, entry_idx + max_hold_bars - 1)
    max_fav = 0.0
    max_adv = 0.0
    for i in range(entry_idx, last_idx + 1):
        row = df.iloc[i]
        hi = float(row["high"])
        lo = float(row["low"])
        if direction == "LONG":
            fav = max(0.0, (hi - entry) / pip)
            adv = max(0.0, (entry - lo) / pip)
            hit_target = hi >= target_px
            hit_adverse = lo <= adverse_px
        else:
            fav = max(0.0, (entry - lo) / pip)
            adv = max(0.0, (hi - entry) / pip)
            hit_target = lo <= target_px
            hit_adverse = hi >= adverse_px
        max_fav = max(max_fav, fav)
        max_adv = max(max_adv, adv)

        if hit_target and hit_adverse:
            return "adverse", i - entry_idx + 1, -float(adverse_pips), max_fav, max_adv
        if hit_adverse:
            return "adverse", i - entry_idx + 1, -float(adverse_pips), max_fav, max_adv
        if hit_target:
            return "hit", i - entry_idx + 1, float(target_pips), max_fav, max_adv

    close_px = float(df.at[last_idx, "close"])
    gross = (close_px - entry) / pip if direction == "LONG" else (entry - close_px) / pip
    return "timeout", last_idx - entry_idx + 1, float(gross), max_fav, max_adv


def aggregate(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    split_idx = int(len(df) * 0.7)
    df = df.sort_values("signal_datetime").reset_index(drop=True)
    df["is_oos"] = np.arange(len(df)) >= split_idx
    wf_bins = pd.qcut(np.arange(len(df)), q=3, labels=False, duplicates="drop")
    df["wf_window"] = wf_bins

    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        d = dict(zip(group_cols, keys))
        n = len(g)
        hits = (g["outcome"] == "hit")
        adverse = (g["outcome"] == "adverse")
        timeout = (g["outcome"] == "timeout")
        net = g["net_pips_after_costs"]
        gp = net[net > 0].sum()
        gl = -net[net < 0].sum()
        pf = float(gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)

        isg = g[~g["is_oos"]]
        oosg = g[g["is_oos"]]
        wf_positive = 0
        wf_total = 0
        for _, w in g.groupby("wf_window"):
            if len(w) == 0:
                continue
            wf_total += 1
            if w["net_pips_after_costs"].mean() > 0:
                wf_positive += 1

        d.update({
            "events": n,
            "hit_rate": hits.mean() if n else 0.0,
            "adverse_failure_rate": adverse.mean() if n else 0.0,
            "timeout_rate": timeout.mean() if n else 0.0,
            "avg_bars_to_hit": g.loc[hits, "bars_to_outcome"].mean() if hits.any() else np.nan,
            "median_bars_to_hit": g.loc[hits, "bars_to_outcome"].median() if hits.any() else np.nan,
            "avg_net_pips_after_costs": net.mean() if n else 0.0,
            "total_net_pips_after_costs": net.sum(),
            "profit_factor": pf,
            "avg_max_adverse_pips_seen": g["max_adverse_pips_seen"].mean() if n else 0.0,
            "p95_max_adverse_pips_seen": g["max_adverse_pips_seen"].quantile(0.95) if n else 0.0,
            "max_adverse_pips_seen": g["max_adverse_pips_seen"].max() if n else 0.0,
            "IS_events": len(isg),
            "OOS_events": len(oosg),
            "IS_hit_rate": (isg["outcome"] == "hit").mean() if len(isg) else np.nan,
            "OOS_hit_rate": (oosg["outcome"] == "hit").mean() if len(oosg) else np.nan,
            "IS_avg_net": isg["net_pips_after_costs"].mean() if len(isg) else np.nan,
            "OOS_avg_net": oosg["net_pips_after_costs"].mean() if len(oosg) else np.nan,
            "OOS_agrees_with_IS": bool((isg["net_pips_after_costs"].mean() > 0) and (oosg["net_pips_after_costs"].mean() > 0)) if len(isg) and len(oosg) else False,
            "wf_positive_windows": wf_positive,
            "wf_total_windows": wf_total,
        })
        rows.append(d)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.symbol.upper() != "EURUSD":
        raise ValueError("This lab is restricted to EURUSD only")
    if args.base_timeframe.upper() != "M15":
        raise ValueError("This lab is restricted to M15 timeframe")

    grids = {
        "quick": {"target": [5, 6, 7, 8, 9, 10], "adverse": [20, 30, 40, 50], "hold": [20, 40, 80]},
        "full": {"target": [5, 6, 7, 8, 9, 10, 12, 15], "adverse": [15, 20, 25, 30, 40, 50, 75], "hold": [10, 20, 40, 80]},
    }
    grid = grids[args.preset]
    total_scenarios = len(grid["target"]) * len(grid["adverse"]) * len(grid["hold"]) * 2
    scenarios = total_scenarios if args.max_scenarios <= 0 else min(total_scenarios, args.max_scenarios)
    print(f"Preset={args.preset} | parameter scenarios={scenarios}")

    start_ts = time.perf_counter()
    source_df = load_data(args.csv, args.symbol.upper())
    if args.max_rows > 0:
        source_df = source_df.head(args.max_rows).copy()
    df = build_features(source_df)
    cost_pips = args.spread_pips + args.slippage_pips

    base_cols = [
        "symbol", "datetime", "hour", "day_of_week", "session_bucket", "open", "high", "low", "close",
        "body_pips", "range_pips", "upper_wick_pips", "lower_wick_pips", "body_to_range", "upper_wick_to_range",
        "lower_wick_to_range", "close_position", "candle_direction", "body_size_bucket", "range_size_bucket",
        "close_position_bucket", "wick_signal_bucket", "consecutive_bull_count", "consecutive_bear_count", "streak_bucket",
        "previous_3_direction", "distance_from_rolling_16_pips", "distance_from_rolling_32_pips", "distance_from_daily_open_pips",
        "distance_from_rolling_16_abs_bucket", "distance_from_daily_open_abs_bucket",
    ]

    records = []
    scenario_idx = 0
    for target, adverse, hold in product(grid["target"], grid["adverse"], grid["hold"]):
        for direction in ["LONG", "SHORT"]:
            if args.max_scenarios > 0 and scenario_idx >= scenarios:
                break
            scenario_idx += 1
            log_every = 5 if args.preset == "quick" else 20
            if scenario_idx == 1 or scenario_idx % log_every == 0 or scenario_idx == scenarios:
                print(f"Processing scenario {scenario_idx}/{scenarios}")
            for i in range(len(df) - 1):
                ev = evaluate_signal(df, i, direction, target, adverse, hold, cost_pips)
                if ev is None:
                    continue
                outcome, bars_to_outcome, gross, max_fav, max_adv = ev
                row = df.iloc[i]
                entry_dt = df.iloc[i + 1]["datetime"]
                rec = {k: row[k] for k in base_cols}
                rec.update({
                    "signal_datetime": row["datetime"],
                    "entry_datetime": entry_dt,
                    "direction_test": direction,
                    "target_pips": target,
                    "adverse_pips": adverse,
                    "max_hold_bars": hold,
                    "outcome": outcome,
                    "bars_to_outcome": bars_to_outcome,
                    "gross_pips_if_traded": gross,
                    "cost_pips": cost_pips,
                    "net_pips_after_costs": gross - cost_pips,
                    "max_favorable_pips": max_fav,
                    "max_adverse_pips_seen": max_adv,
                })
                records.append(rec)
        if args.max_scenarios > 0 and scenario_idx >= scenarios:
            break

    events = pd.DataFrame.from_records(records)
    events = events[[
        "symbol", "signal_datetime", "entry_datetime", "hour", "day_of_week", "session_bucket", "direction_test", "target_pips",
        "adverse_pips", "max_hold_bars", "open", "high", "low", "close", "body_pips", "range_pips", "upper_wick_pips",
        "lower_wick_pips", "body_to_range", "upper_wick_to_range", "lower_wick_to_range", "close_position", "candle_direction",
        "body_size_bucket", "range_size_bucket", "close_position_bucket", "wick_signal_bucket", "consecutive_bull_count",
        "consecutive_bear_count", "streak_bucket", "previous_3_direction", "distance_from_rolling_16_pips",
        "distance_from_rolling_32_pips", "distance_from_daily_open_pips", "distance_from_rolling_16_abs_bucket",
        "distance_from_daily_open_abs_bucket", "outcome", "bars_to_outcome", "gross_pips_if_traded", "cost_pips",
        "net_pips_after_costs", "max_favorable_pips", "max_adverse_pips_seen",
    ]]

    group_sets = [
        ["candle_direction", "body_size_bucket", "target_pips", "adverse_pips", "max_hold_bars", "direction_test"],
        ["wick_signal_bucket", "close_position_bucket", "target_pips", "adverse_pips", "max_hold_bars", "direction_test"],
        ["streak_bucket", "previous_3_direction", "target_pips", "adverse_pips", "max_hold_bars", "direction_test"],
        ["session_bucket", "hour", "wick_signal_bucket", "target_pips", "adverse_pips", "max_hold_bars", "direction_test"],
        ["distance_from_rolling_16_abs_bucket", "candle_direction", "target_pips", "adverse_pips", "max_hold_bars", "direction_test"],
        ["distance_from_daily_open_abs_bucket", "candle_direction", "target_pips", "adverse_pips", "max_hold_bars", "direction_test"],
    ]
    results = pd.concat([aggregate(events, cols).assign(table_id=i + 1) for i, cols in enumerate(group_sets)], ignore_index=True)

    clean = results[
        (results["events"] >= 200)
        & (results["avg_net_pips_after_costs"] > 0)
        & (results["OOS_avg_net"] > 0)
        & (results["profit_factor"] > 1)
        & (results["wf_positive_windows"] >= 2)
    ].copy()

    output_dir = args.output_dir
    if args.preset == "quick" and args.output_dir == "candle_behavior_reports":
        output_dir = "candle_behavior_reports_quick"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / f"{args.symbol.upper()}_candle_behavior_events.csv"
    results_path = out_dir / f"{args.symbol.upper()}_candle_behavior_results.csv"
    best_path = out_dir / f"{args.symbol.upper()}_candle_behavior_best.json"
    summary_path = out_dir / f"{args.symbol.upper()}_candle_behavior_summary.md"

    if args.write_events:
        events.to_csv(events_path, index=False)
    results.to_csv(results_path, index=False)

    top_oos = clean.sort_values("OOS_avg_net", ascending=False).head(30)
    top_pf = clean.sort_values("profit_factor", ascending=False).head(30)
    top_oos_hit = clean.sort_values("OOS_hit_rate", ascending=False).head(30)

    best = {
        "preset": args.preset,
        "scenario_count": scenarios,
        "clean_candidates": int(len(clean)),
        "top_by_oos_avg_net": top_oos.head(10).to_dict(orient="records"),
    }
    best_path.write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")

    def section(df_in: pd.DataFrame, title: str, cols: List[str]) -> str:
        if df_in.empty:
            return f"## {title}\n\n_No rows._\n"
        return f"## {title}\n\n" + df_in[cols].to_markdown(index=False) + "\n"

    lines = ["# EURUSD Candle Behavior Summary", "", f"Preset: **{args.preset}**", f"Scenarios: **{scenarios}**", ""]
    lines.append(section(top_oos, "Top 30 clean candidates by OOS_avg_net", list(top_oos.columns[:12])))
    lines.append(section(top_pf, "Top 30 clean candidates by profit_factor", list(top_pf.columns[:12])))
    lines.append(section(top_oos_hit, "Top 30 clean candidates by OOS_hit_rate", list(top_oos_hit.columns[:12])))

    comparisons = [
        ("Candle direction comparison", ["candle_direction"]),
        ("Wick bucket comparison", ["wick_signal_bucket"]),
        ("Streak bucket comparison", ["streak_bucket"]),
        ("Hour/session comparison", ["session_bucket", "hour"]),
        ("Distance from rolling_16 comparison", ["distance_from_rolling_16_abs_bucket"]),
        ("Distance from daily_open comparison", ["distance_from_daily_open_abs_bucket"]),
    ]
    for title, cols in comparisons:
        present = [c for c in cols if c in results.columns]
        if not present:
            continue
        comp = results.groupby(present, dropna=False).agg(events=("events", "sum"), avg_net=("avg_net_pips_after_costs", "mean"), hit_rate=("hit_rate", "mean")).reset_index()
        lines.append(section(comp.sort_values("avg_net", ascending=False).head(30), title, comp.columns.tolist()))

    if clean.empty:
        lines.append("## Warning\n\nNo clean candidates were found under current criteria.\n")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    if args.write_events:
        print(f"Wrote: {events_path}")
    else:
        print(f"Skipped events CSV (use --write-events to enable): {events_path}")
    print(f"Wrote: {results_path}")
    print(f"Wrote: {best_path}")
    print(f"Wrote: {summary_path}")
    print(f"Total runtime seconds: {time.perf_counter() - start_ts:.2f}")


if __name__ == "__main__":
    main()
