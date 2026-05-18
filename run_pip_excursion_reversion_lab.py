#!/usr/bin/env python3
"""Standalone lab: USDJPY pip excursion reversion by hour/session."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PIP_SIZE_USDJPY = 0.01
MOVE_X_PIPS = [20, 30, 40, 50, 70]
TP_PIPS = [5, 10, 15, 20]
EMERGENCY_SL_PIPS = [100, 150, 200, 300]
MAX_HOLD_BARS = [20, 40, 80]
ANCHORS = ["rolling_16_close", "daily_open"]


@dataclass
class TradeResult:
    symbol: str
    anchor_type: str
    signal_datetime: pd.Timestamp
    entry_datetime: pd.Timestamp
    hour: int
    session_bucket: str
    signal_direction: str
    anchor_price: float
    signal_close: float
    entry_price: float
    move_x_pips: int
    actual_distance_pips: float
    tp_pips: int
    emergency_sl_pips: int
    max_hold_bars: int
    exit_datetime: pd.Timestamp
    exit_reason: str
    gross_pips: float
    net_pips_after_costs: float
    holding_bars: int
    max_adverse_pips: float
    max_favorable_pips: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="USDJPY Pip Excursion Reversion by Hour lab")
    parser.add_argument("--csv", default="data/USDJPY_M15_MT5_5Y.csv")
    parser.add_argument("--symbol", default="USDJPY")
    parser.add_argument("--base-timeframe", default="M15")
    parser.add_argument("--spread-pips", type=float, default=1.2)
    parser.add_argument("--slippage-pips", type=float, default=0.2)
    parser.add_argument("--output-dir", default="pip_excursion_reports")
    parser.add_argument("--focused-only", action="store_true")
    parser.add_argument("--focused-hour", type=int)
    parser.add_argument("--focused-move-x-pips", type=int)
    parser.add_argument("--focused-tp-pips", type=int)
    parser.add_argument("--focused-emergency-sl-pips", type=int)
    parser.add_argument("--focused-max-hold-bars", type=int)
    parser.add_argument("--focused-direction", choices=["LONG", "SHORT"])
    parser.add_argument("--entry-mode", choices=["continuous", "cross_only"], default="continuous")
    parser.add_argument("--no-overlap", action="store_true")
    parser.add_argument("--cooldown-until-anchor-reset", action="store_true")
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


def load_data(csv_path: str, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    lower_cols = {c.lower(): c for c in df.columns}
    datetime_col = lower_cols.get("datetime") or lower_cols.get("time")
    if datetime_col is None:
        raise ValueError("CSV must include datetime or time column")

    df = df.rename(columns={lower_cols["open"]: "open", lower_cols["high"]: "high", lower_cols["low"]: "low", lower_cols["close"]: "close", datetime_col: "datetime"})

    df["datetime"] = pd.to_datetime(df["datetime"], utc=False)
    df = df.sort_values("datetime").reset_index(drop=True)
    df["symbol"] = symbol
    df["hour"] = df["datetime"].dt.hour
    df["session_bucket"] = df["hour"].map(session_bucket)
    df["trading_date"] = df["datetime"].dt.date
    df["rolling_16_close"] = df["close"].shift(16)
    df["daily_open"] = df.groupby("trading_date")["open"].transform("first")
    return df


def simulate_trade(
    df: pd.DataFrame,
    signal_idx: int,
    direction: str,
    tp_pips: int,
    emergency_sl_pips: int,
    max_hold_bars: int,
    cost_pips: float,
) -> Optional[Tuple[str, pd.Timestamp, float, int, float, float]]:
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None

    entry_price = float(df.at[entry_idx, "open"])
    if direction == "LONG":
        tp_price = entry_price + tp_pips * PIP_SIZE_USDJPY
        sl_price = entry_price - emergency_sl_pips * PIP_SIZE_USDJPY
    else:
        tp_price = entry_price - tp_pips * PIP_SIZE_USDJPY
        sl_price = entry_price + emergency_sl_pips * PIP_SIZE_USDJPY

    last_idx = min(len(df) - 1, entry_idx + max_hold_bars - 1)
    window = df.iloc[entry_idx : last_idx + 1]

    max_adverse = 0.0
    max_favorable = 0.0

    for idx, row in window.iterrows():
        high = float(row["high"])
        low = float(row["low"])

        if direction == "LONG":
            adverse = max(0.0, (entry_price - low) / PIP_SIZE_USDJPY)
            favorable = max(0.0, (high - entry_price) / PIP_SIZE_USDJPY)
            sl_hit = low <= sl_price
            tp_hit = high >= tp_price
        else:
            adverse = max(0.0, (high - entry_price) / PIP_SIZE_USDJPY)
            favorable = max(0.0, (entry_price - low) / PIP_SIZE_USDJPY)
            sl_hit = high >= sl_price
            tp_hit = low <= tp_price

        max_adverse = max(max_adverse, adverse)
        max_favorable = max(max_favorable, favorable)

        if sl_hit and tp_hit:
            exit_reason = "emergency_sl"
            gross = -float(emergency_sl_pips)
            return exit_reason, row["datetime"], gross - cost_pips, idx - entry_idx + 1, max_adverse, max_favorable
        if sl_hit:
            exit_reason = "emergency_sl"
            gross = -float(emergency_sl_pips)
            return exit_reason, row["datetime"], gross - cost_pips, idx - entry_idx + 1, max_adverse, max_favorable
        if tp_hit:
            exit_reason = "tp"
            gross = float(tp_pips)
            return exit_reason, row["datetime"], gross - cost_pips, idx - entry_idx + 1, max_adverse, max_favorable

    timeout_row = df.iloc[last_idx]
    close_price = float(timeout_row["close"])
    if direction == "LONG":
        gross = (close_price - entry_price) / PIP_SIZE_USDJPY
    else:
        gross = (entry_price - close_price) / PIP_SIZE_USDJPY
    return "timeout", timeout_row["datetime"], gross - cost_pips, last_idx - entry_idx + 1, max_adverse, max_favorable


def max_losing_streak(values: pd.Series) -> int:
    streak = 0
    max_streak = 0
    for v in values:
        if v < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def aggregate_metrics(group: pd.DataFrame) -> Dict[str, object]:
    net = group["net_pips_after_costs"]
    wins = (net > 0).sum()
    losses = (net < 0)
    gross_profit = net[net > 0].sum()
    gross_loss_abs = -net[net < 0].sum()
    pf = float(gross_profit / gross_loss_abs) if gross_loss_abs > 0 else (float("inf") if gross_profit > 0 else 0.0)

    n = len(group)
    return {
        "trades": n,
        "win_rate": wins / n if n else 0.0,
        "emergency_sl_rate": (group["exit_reason"] == "emergency_sl").mean() if n else 0.0,
        "timeout_rate": (group["exit_reason"] == "timeout").mean() if n else 0.0,
        "total_net_pips": net.sum(),
        "avg_net_pips": net.mean() if n else 0.0,
        "median_net_pips": net.median() if n else 0.0,
        "profit_factor": pf,
        "max_losing_streak": max_losing_streak(net),
        "avg_holding_bars": group["holding_bars"].mean() if n else 0.0,
        "max_holding_bars": group["holding_bars"].max() if n else 0,
        "avg_max_adverse_pips": group["max_adverse_pips"].mean() if n else 0.0,
        "p95_max_adverse_pips": group["max_adverse_pips"].quantile(0.95) if n else 0.0,
        "max_adverse_pips": group["max_adverse_pips"].max() if n else 0.0,
        "avg_max_favorable_pips": group["max_favorable_pips"].mean() if n else 0.0,
    }


def verdict(row: pd.Series) -> str:
    max_adv_ok = row["max_adverse_pips"] <= row["emergency_sl_pips"] * 0.9
    strong = (
        row["trades"] >= 300
        and row["total_net_pips"] > 0
        and row["avg_net_pips"] > 0
        and row["profit_factor"] >= 1.15
        and row["oos_avg_net_pips"] > 0
        and bool(row["oos_agrees_with_is"])
        and row["walk_forward_positive_windows"] >= 2
        and max_adv_ok
    )
    if strong:
        return "Strong Candidate"
    candidate = (
        row["trades"] >= 200
        and row["total_net_pips"] > 0
        and row["avg_net_pips"] > 0
        and row["oos_avg_net_pips"] > 0
        and row["walk_forward_positive_windows"] >= 2
    )
    return "Candidate" if candidate else "Reject"


def build_summary(results: pd.DataFrame, output_md: Path) -> None:
    lines: List[str] = []
    lines.append("# USDJPY Pip Excursion Reversion by Hour Summary")
    lines.append("")
    lines.append("This lab uses no regular SL. Trades exit on TP, emergency SL, or time stop.")
    lines.append("")

    top_total = results.sort_values("total_net_pips", ascending=False).head(20)
    lines.append("## Top 20 overall candidates by total_net_pips")
    lines.append(top_total.to_markdown(index=False))
    lines.append("")

    top_avg = results[results["trades"] >= 300].sort_values("avg_net_pips", ascending=False).head(20)
    lines.append("## Top 20 by avg_net_pips (trades >= 300)")
    lines.append(top_avg.to_markdown(index=False) if not top_avg.empty else "No rows met trades >= 300.")
    lines.append("")

    hour_best = results.sort_values("total_net_pips", ascending=False).groupby("hour", dropna=False).head(1).sort_values("hour")
    lines.append("## Best candidates by exact hour")
    lines.append(hour_best.to_markdown(index=False))
    lines.append("")

    session_best = results.sort_values("total_net_pips", ascending=False).groupby("session_bucket", dropna=False).head(1)
    lines.append("## Best candidates by session bucket")
    lines.append(session_best.to_markdown(index=False))
    lines.append("")

    anchor_cmp = results.groupby("anchor_type", as_index=False)[["trades", "total_net_pips", "avg_net_pips"]].mean(numeric_only=True)
    lines.append("## rolling_16_close vs daily_open")
    lines.append(anchor_cmp.to_markdown(index=False))
    lines.append("")

    dir_cmp = results[results["signal_direction"] != "ALL"].groupby("signal_direction", as_index=False)[["trades", "total_net_pips", "avg_net_pips"]].mean(numeric_only=True)
    lines.append("## LONG vs SHORT")
    lines.append(dir_cmp.to_markdown(index=False))

    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csv, args.symbol)
    cost_pips = args.spread_pips + args.slippage_pips

    trades: List[TradeResult] = []

    focused_anchor_type = "daily_open"
    focused_move_x_pips = args.focused_move_x_pips if args.focused_move_x_pips is not None else 30
    focused_tp_pips = args.focused_tp_pips if args.focused_tp_pips is not None else 15
    focused_emergency_sl_pips = args.focused_emergency_sl_pips if args.focused_emergency_sl_pips is not None else 300
    focused_max_hold_bars = args.focused_max_hold_bars if args.focused_max_hold_bars is not None else 40
    focused_hour = args.focused_hour if args.focused_hour is not None else 22
    focused_direction = args.focused_direction if args.focused_direction is not None else "SHORT"

    anchor_types = [focused_anchor_type] if args.focused_only else ANCHORS
    move_x_values = [focused_move_x_pips] if args.focused_only else MOVE_X_PIPS
    tp_values = [focused_tp_pips] if args.focused_only else TP_PIPS
    emergency_sl_values = [focused_emergency_sl_pips] if args.focused_only else EMERGENCY_SL_PIPS
    max_hold_values = [focused_max_hold_bars] if args.focused_only else MAX_HOLD_BARS

    for anchor_type in anchor_types:
        anchor_values = df[anchor_type].values
        for move_x in move_x_values:
            for tp in tp_values:
                for emergency_sl in emergency_sl_values:
                    for max_hold in max_hold_values:
                        open_until_idx_by_direction: Dict[str, int] = {}
                        cooldown_active_by_direction: Dict[str, bool] = {}
                        prev_trading_date = None
                        for i in range(len(df) - 1):
                            if args.focused_only and int(df.at[i, "hour"]) != focused_hour:
                                continue
                            trading_date = df.at[i, "trading_date"]
                            if (
                                args.cooldown_until_anchor_reset
                                and anchor_type == "daily_open"
                                and prev_trading_date is not None
                                and trading_date != prev_trading_date
                            ):
                                cooldown_active_by_direction.clear()
                            prev_trading_date = trading_date
                            anchor_price = anchor_values[i]
                            if pd.isna(anchor_price):
                                continue
                            close = float(df.at[i, "close"])
                            distance_pips = (close - float(anchor_price)) / PIP_SIZE_USDJPY
                            prev_distance_pips = None
                            if i > 0:
                                prev_anchor = anchor_values[i - 1]
                                if not pd.isna(prev_anchor):
                                    prev_close = float(df.at[i - 1, "close"])
                                    prev_distance_pips = (prev_close - float(prev_anchor)) / PIP_SIZE_USDJPY
                            direction = None
                            if args.entry_mode == "continuous":
                                if close >= float(anchor_price) + move_x * PIP_SIZE_USDJPY:
                                    direction = "SHORT"
                                elif close <= float(anchor_price) - move_x * PIP_SIZE_USDJPY:
                                    direction = "LONG"
                            else:
                                if prev_distance_pips is not None and prev_distance_pips < move_x <= distance_pips:
                                    direction = "SHORT"
                                elif prev_distance_pips is not None and prev_distance_pips > -move_x >= distance_pips:
                                    direction = "LONG"
                            if direction is None:
                                continue
                            if args.focused_only and direction != focused_direction:
                                continue

                            # strict validation controls (opt-in via flags only)
                            if args.no_overlap or args.cooldown_until_anchor_reset:
                                open_until_idx = open_until_idx_by_direction.get(direction, -1)
                                if args.no_overlap and i <= open_until_idx:
                                    continue
                                if args.cooldown_until_anchor_reset and cooldown_active_by_direction.get(direction, False):
                                    if direction == "SHORT":
                                        if close <= float(anchor_price):
                                            cooldown_active_by_direction[direction] = False
                                        else:
                                            continue
                                    else:
                                        if close >= float(anchor_price):
                                            cooldown_active_by_direction[direction] = False
                                        else:
                                            continue

                            outcome = simulate_trade(df, i, direction, tp, emergency_sl, max_hold, cost_pips)
                            if outcome is None:
                                continue
                            exit_reason, exit_dt, net_pips, holding_bars, max_adv, max_fav = outcome
                            gross_pips = net_pips + cost_pips
                            entry_i = i + 1
                            exit_i = entry_i + holding_bars - 1
                            if args.no_overlap:
                                open_until_idx_by_direction[direction] = max(open_until_idx_by_direction.get(direction, -1), exit_i)
                            if args.cooldown_until_anchor_reset:
                                cooldown_active_by_direction[direction] = True
                            trades.append(
                                TradeResult(
                                    symbol=args.symbol,
                                    anchor_type=anchor_type,
                                    signal_datetime=df.at[i, "datetime"],
                                    entry_datetime=df.at[entry_i, "datetime"],
                                    hour=int(df.at[i, "hour"]),
                                    session_bucket=df.at[i, "session_bucket"],
                                    signal_direction=direction,
                                    anchor_price=float(anchor_price),
                                    signal_close=close,
                                    entry_price=float(df.at[entry_i, "open"]),
                                    move_x_pips=move_x,
                                    actual_distance_pips=distance_pips,
                                    tp_pips=tp,
                                    emergency_sl_pips=emergency_sl,
                                    max_hold_bars=max_hold,
                                    exit_datetime=exit_dt,
                                    exit_reason=exit_reason,
                                    gross_pips=gross_pips,
                                    net_pips_after_costs=net_pips,
                                    holding_bars=holding_bars,
                                    max_adverse_pips=max_adv,
                                    max_favorable_pips=max_fav,
                                )
                            )

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    trades_path = output_dir / "USDJPY_pip_excursion_trades.csv"
    trades_df.to_csv(trades_path, index=False)

    if trades_df.empty:
        raise RuntimeError("No trades generated for the configured dataset/parameters")

    trades_df = trades_df.sort_values("signal_datetime").reset_index(drop=True)
    n = len(trades_df)
    is_cut = int(n * 0.7)
    trades_df["is_oos"] = np.where(np.arange(n) < is_cut, "IS", "OOS")
    trades_df["wf_window"] = pd.cut(np.arange(n), bins=3, labels=[1, 2, 3], include_lowest=True)

    grouping_cols = ["anchor_type", "move_x_pips", "tp_pips", "emergency_sl_pips", "max_hold_bars", "hour", "session_bucket"]
    rows: List[Dict[str, object]] = []

    for keys, group in trades_df.groupby(grouping_cols + ["signal_direction"], dropna=False):
        rec = dict(zip(grouping_cols + ["signal_direction"], keys))
        rec.update(aggregate_metrics(group))
        is_g = group[group["is_oos"] == "IS"]["net_pips_after_costs"]
        oos_g = group[group["is_oos"] == "OOS"]["net_pips_after_costs"]
        rec.update(
            {
                "is_trades": len(is_g),
                "oos_trades": len(oos_g),
                "is_total_net_pips": is_g.sum(),
                "oos_total_net_pips": oos_g.sum(),
                "is_avg_net_pips": is_g.mean() if len(is_g) else 0.0,
                "oos_avg_net_pips": oos_g.mean() if len(oos_g) else 0.0,
                "oos_agrees_with_is": bool((is_g.mean() if len(is_g) else 0.0) > 0 and (oos_g.mean() if len(oos_g) else 0.0) > 0),
            }
        )
        wf = group.groupby("wf_window")["net_pips_after_costs"].sum()
        rec["walk_forward_positive_windows"] = int((wf > 0).sum())
        rec["walk_forward_total_windows"] = int(len(wf))
        rec["verdict"] = verdict(pd.Series(rec))
        rows.append(rec)

        all_rec = dict(zip(grouping_cols, keys[:-1]))
        all_rec["signal_direction"] = "ALL"
        all_rec.update(aggregate_metrics(group))
        all_rec.update(
            {
                "is_trades": len(is_g),
                "oos_trades": len(oos_g),
                "is_total_net_pips": is_g.sum(),
                "oos_total_net_pips": oos_g.sum(),
                "is_avg_net_pips": is_g.mean() if len(is_g) else 0.0,
                "oos_avg_net_pips": oos_g.mean() if len(oos_g) else 0.0,
                "oos_agrees_with_is": bool((is_g.mean() if len(is_g) else 0.0) > 0 and (oos_g.mean() if len(oos_g) else 0.0) > 0),
            }
        )
        all_wf = group.groupby("wf_window")["net_pips_after_costs"].sum()
        all_rec["walk_forward_positive_windows"] = int((all_wf > 0).sum())
        all_rec["walk_forward_total_windows"] = int(len(all_wf))
        all_rec["verdict"] = verdict(pd.Series(all_rec))
        rows.append(all_rec)

    results_df = pd.DataFrame(rows).drop_duplicates(
        subset=grouping_cols + ["signal_direction"], keep="first"
    )
    results_path = output_dir / "USDJPY_pip_excursion_results.csv"
    results_df.to_csv(results_path, index=False)

    best_row = results_df.sort_values("total_net_pips", ascending=False).iloc[0].to_dict()
    best_path = output_dir / "USDJPY_pip_excursion_best.json"
    best_path.write_text(json.dumps(best_row, indent=2, default=str), encoding="utf-8")

    summary_path = output_dir / "USDJPY_pip_excursion_summary.md"
    build_summary(results_df, summary_path)

    print(f"Saved trades: {trades_path}")
    print(f"Saved results: {results_path}")
    print(f"Saved best json: {best_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
