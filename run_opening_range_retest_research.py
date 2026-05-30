#!/usr/bin/env python3
"""Research-only Opening Range Breakout + Retest study for FX M15 data.

This script is deliberately offline and research-only. It reads historical M15 CSV
files, builds session opening ranges, waits for a breakout, then enters only after
a valid retest candle. It writes neutral research artifacts comparing the tested
strategy with opposite-direction and deterministic-random baselines.

It contains no EA, live trading, MT4/MT5 execution, lot sizing, account-equity
calculation, or single-row optimizer.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

PIP_SIZE_FALLBACK = 0.0001
PIP_SIZE_JPY = 0.01

DATASETS = {
    "EURUSD": "EURUSD_M15_MT5_5Y.csv",
    "GBPUSD": "GBPUSD_M15_MT5_5Y.csv",
    "USDJPY": "USDJPY_M15_MT5_5Y.csv",
    "GBPJPY": "GBPJPY_M15_MT5_5Y.csv",
    "GBPAUD": "GBPAUD_M15_MT5_5Y.csv",
    "GBPNZD": "GBPNZD_M15_MT5_5Y.csv",
    "AUDJPY": "AUDJPY_M15_MT5_5Y.csv",
    "EURJPY": "EURJPY_M15_MT5_5Y.csv",
}

SESSION_OPEN_HOURS = {
    "Tokyo_Open": 0,
    "London_Open": 7,
    "NewYork_Open": 13,
    "Sydney_Open": 22,
}

SYMBOL_SESSION_MAP = {
    "EURUSD": ["London_Open", "NewYork_Open"],
    "GBPUSD": ["London_Open", "NewYork_Open"],
    "USDJPY": ["Tokyo_Open", "NewYork_Open"],
    "EURJPY": ["London_Open", "Tokyo_Open"],
    "GBPJPY": ["London_Open", "Tokyo_Open"],
    "AUDJPY": ["Tokyo_Open", "Sydney_Open"],
    "GBPAUD": ["London_Open", "Sydney_Open"],
    "GBPNZD": ["London_Open", "Sydney_Open"],
}

OPENING_RANGE_BARS = (1, 2, 4)
BREAKOUT_MONITOR_BARS = (8, 16, 24)
BREAKOUT_BUFFER_PIPS = (0, 2)
RETEST_WINDOW_BARS = (4, 8, 12)
RETEST_TOLERANCE_PIPS = (0, 2, 4)
CANDLE_FILTERS = (False, True)
STOP_MODES = ("retest_candle", "OR_mid", "opposite_edge")
TARGET_MODES = ("fixed_pips", "rr_multiple")
FIXED_TARGET_PIPS = (10, 15, 20, 30)
RR_MULTIPLES = (1.0, 1.5, 2.0)
TRADE_HORIZON_BARS = (8, 16, 24)

EVENT_COLUMNS = [
    "symbol",
    "date",
    "year",
    "month",
    "day_of_week",
    "session_name",
    "session_open_hour",
    "session_open_datetime",
    "opening_range_bars",
    "breakout_monitor_bars",
    "breakout_buffer_pips",
    "retest_window_bars",
    "retest_tolerance_pips",
    "candle_filter",
    "OR_high",
    "OR_low",
    "OR_mid",
    "OR_range_pips",
    "breakout_direction",
    "breakout_datetime",
    "retest_datetime",
    "entry_datetime",
    "entry_price",
    "direction",
    "stop_mode",
    "target_mode",
    "target_value",
    "trade_horizon_bars",
    "stop_loss",
    "take_profit",
    "risk_pips",
    "reward_pips",
    "rr_ratio",
    "friction_pips",
    "outcome",
    "ambiguous",
    "gross_pnl_pips",
    "net_pnl_pips",
    "opposite_pnl_after_friction",
    "random_pnl_after_friction",
    "edge_vs_opposite",
    "edge_vs_random",
    "bars_in_trade",
]

PARAMETER_COLUMNS = [
    "opening_range_bars",
    "breakout_monitor_bars",
    "breakout_buffer_pips",
    "retest_window_bars",
    "retest_tolerance_pips",
    "candle_filter",
    "stop_mode",
    "target_mode",
    "target_value",
    "trade_horizon_bars",
]

OUTPUT_FILES = [
    "opening_range_retest_events.csv",
    "opening_range_retest_by_symbol_session_params.csv",
    "opening_range_retest_by_symbol_session.csv",
    "opening_range_retest_by_symbol.csv",
    "opening_range_retest_by_session.csv",
    "opening_range_retest_by_year.csv",
    "opening_range_retest_by_month.csv",
    "opening_range_retest_walkforward.csv",
    "opening_range_retest_verdict.csv",
    "opening_range_retest_summary.md",
]


@dataclass(frozen=True)
class Config:
    data_dir: Path
    output_dir: Path
    min_oos_observations: int


def infer_pip_size(symbol: str) -> float:
    return PIP_SIZE_JPY if "JPY" in symbol.upper() else PIP_SIZE_FALLBACK


def friction_pips(symbol: str) -> float:
    if "JPY" in symbol.upper():
        return 1.5 + 0.5 + 0.5
    return 1.2 + 0.4 + 0.4


def or_bounds_pips(symbol: str) -> tuple[float, float]:
    if "JPY" in symbol.upper():
        return 5.0, 80.0
    return 4.0, 60.0


def normalize_ohlc_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if "datetime" in raw.columns:
        dt = pd.to_datetime(raw["datetime"], errors="raise", utc=False)
    elif {"date", "time"}.issubset(raw.columns):
        dt = pd.to_datetime(raw["date"].astype(str) + " " + raw["time"].astype(str), errors="raise", utc=False)
    else:
        raise ValueError(f"{path} must include datetime or date/time columns")

    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    raw["datetime"] = dt
    for col in ("open", "high", "low", "close"):
        if col not in raw.columns:
            raise ValueError(f"{path} missing required column: {col}")
        raw[col] = pd.to_numeric(raw[col], errors="raise")

    invalid_high = raw["high"] < raw[["open", "close", "low"]].max(axis=1)
    invalid_low = raw["low"] > raw[["open", "close", "high"]].min(axis=1)
    if invalid_high.any() or invalid_low.any():
        raise ValueError(f"{path} has invalid OHLC relationships")

    df = raw[["datetime", "open", "high", "low", "close"]].drop_duplicates("datetime")
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date.astype(str)
    return df


def pips_between(direction: str, start: float, end: float, pip_size: float) -> float:
    return (end - start) / pip_size if direction == "LONG" else (start - end) / pip_size


def first_index_at_or_after(datetimes: pd.Series, timestamp: pd.Timestamp) -> int | None:
    positions = datetimes.searchsorted(timestamp, side="left")
    if positions >= len(datetimes):
        return None
    return int(positions)


def build_opening_range(df: pd.DataFrame, start_idx: int, opening_range_bars: int, pip_size: float) -> dict[str, Any] | None:
    end_idx = start_idx + opening_range_bars
    if end_idx > len(df):
        return None
    or_df = df.iloc[start_idx:end_idx]
    or_high = float(or_df["high"].max())
    or_low = float(or_df["low"].min())
    or_mid = (or_high + or_low) / 2.0
    return {
        "OR_high": or_high,
        "OR_low": or_low,
        "OR_mid": or_mid,
        "OR_range_pips": (or_high - or_low) / pip_size,
        "monitor_start_idx": end_idx,
        "session_open_datetime": pd.Timestamp(df.iloc[start_idx]["datetime"]),
    }


def find_breakout(
    df: pd.DataFrame,
    start_idx: int,
    monitor_bars: int,
    or_box: dict[str, Any],
    buffer_pips: float,
    pip_size: float,
) -> tuple[int, str] | None:
    upper = or_box["OR_high"] + buffer_pips * pip_size
    lower = or_box["OR_low"] - buffer_pips * pip_size
    for idx in range(start_idx, min(start_idx + monitor_bars, len(df))):
        close = float(df.iloc[idx]["close"])
        if close > upper:
            return idx, "upside"
        if close < lower:
            return idx, "downside"
    return None


def find_retest(
    df: pd.DataFrame,
    breakout_idx: int,
    breakout_direction: str,
    retest_window_bars: int,
    tolerance_pips: float,
    candle_filter: bool,
    or_box: dict[str, Any],
    pip_size: float,
) -> int | None:
    tolerance = tolerance_pips * pip_size
    for idx in range(breakout_idx + 1, min(breakout_idx + 1 + retest_window_bars, len(df))):
        row = df.iloc[idx]
        open_price = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        if breakout_direction == "upside":
            touched = low <= or_box["OR_high"] + tolerance
            held_range = close >= or_box["OR_high"]
            candle_ok = (close > open_price) if candle_filter else True
            if touched and held_range and candle_ok:
                return idx
        else:
            touched = high >= or_box["OR_low"] - tolerance
            held_range = close <= or_box["OR_low"]
            candle_ok = (close < open_price) if candle_filter else True
            if touched and held_range and candle_ok:
                return idx
    return None


def make_order(
    direction: str,
    entry_price: float,
    retest_row: pd.Series,
    or_box: dict[str, Any],
    stop_mode: str,
    target_mode: str,
    target_value: float,
    buffer_pips: float,
    pip_size: float,
) -> tuple[float, float, float, float, float] | None:
    if direction == "LONG":
        if stop_mode == "retest_candle":
            stop_loss = float(retest_row["low"]) - buffer_pips * pip_size
        elif stop_mode == "OR_mid":
            stop_loss = float(or_box["OR_mid"])
        else:
            stop_loss = float(or_box["OR_low"])
        risk_pips = (entry_price - stop_loss) / pip_size
        reward_pips = target_value if target_mode == "fixed_pips" else risk_pips * target_value
        take_profit = entry_price + reward_pips * pip_size
    else:
        if stop_mode == "retest_candle":
            stop_loss = float(retest_row["high"]) + buffer_pips * pip_size
        elif stop_mode == "OR_mid":
            stop_loss = float(or_box["OR_mid"])
        else:
            stop_loss = float(or_box["OR_high"])
        risk_pips = (stop_loss - entry_price) / pip_size
        reward_pips = target_value if target_mode == "fixed_pips" else risk_pips * target_value
        take_profit = entry_price - reward_pips * pip_size

    if risk_pips <= 0 or reward_pips <= 0:
        return None
    rr_ratio = reward_pips / risk_pips
    if rr_ratio < 1.0:
        return None
    return float(stop_loss), float(take_profit), float(risk_pips), float(reward_pips), float(rr_ratio)


def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    horizon_bars: int,
    friction: float,
    pip_size: float,
) -> dict[str, Any]:
    last_idx = min(entry_idx + horizon_bars, len(df) - 1)
    ambiguous = False
    exit_price = float(df.iloc[last_idx]["close"])
    outcome = "horizon"
    bars_in_trade = max(0, last_idx - entry_idx)

    for idx in range(entry_idx + 1, last_idx + 1):
        row = df.iloc[idx]
        high = float(row["high"])
        low = float(row["low"])
        if direction == "LONG":
            hit_tp = high >= take_profit
            hit_sl = low <= stop_loss
        else:
            hit_tp = low <= take_profit
            hit_sl = high >= stop_loss
        if hit_tp and hit_sl:
            ambiguous = True
            outcome = "loss_ambiguous_sl_first"
            exit_price = stop_loss
            bars_in_trade = idx - entry_idx
            break
        if hit_sl:
            outcome = "loss"
            exit_price = stop_loss
            bars_in_trade = idx - entry_idx
            break
        if hit_tp:
            outcome = "win"
            exit_price = take_profit
            bars_in_trade = idx - entry_idx
            break

    gross_pnl = pips_between(direction, entry_price, exit_price, pip_size)
    return {
        "outcome": outcome,
        "ambiguous": ambiguous,
        "gross_pnl_pips": gross_pnl,
        "net_pnl_pips": gross_pnl - friction,
        "bars_in_trade": bars_in_trade,
    }


def stable_random_uses_strategy(parts: Iterable[Any]) -> bool:
    key = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2 == 0


def target_values() -> Iterable[tuple[str, float]]:
    for pips in FIXED_TARGET_PIPS:
        yield "fixed_pips", float(pips)
    for multiple in RR_MULTIPLES:
        yield "rr_multiple", float(multiple)


def research_symbol(symbol: str, path: Path) -> list[dict[str, Any]]:
    df = normalize_ohlc_csv(path)
    pip_size = infer_pip_size(symbol)
    min_or, max_or = or_bounds_pips(symbol)
    friction = friction_pips(symbol)
    records: list[dict[str, Any]] = []
    datetimes = df["datetime"]

    for date_text in sorted(df["date"].unique()):
        session_date = pd.Timestamp(date_text)
        for session_name in SYMBOL_SESSION_MAP[symbol]:
            session_hour = SESSION_OPEN_HOURS[session_name]
            session_dt = session_date + pd.Timedelta(hours=session_hour)
            session_idx = first_index_at_or_after(datetimes, session_dt)
            if session_idx is None:
                continue
            if pd.Timestamp(df.iloc[session_idx]["datetime"]) >= session_dt + pd.Timedelta(hours=6):
                continue

            for opening_range_bars in OPENING_RANGE_BARS:
                or_box = build_opening_range(df, session_idx, opening_range_bars, pip_size)
                if or_box is None:
                    continue
                or_range = float(or_box["OR_range_pips"])
                if or_range < min_or or or_range > max_or:
                    continue

                for monitor_bars in BREAKOUT_MONITOR_BARS:
                    for breakout_buffer_pips in BREAKOUT_BUFFER_PIPS:
                        breakout = find_breakout(
                            df,
                            int(or_box["monitor_start_idx"]),
                            monitor_bars,
                            or_box,
                            float(breakout_buffer_pips),
                            pip_size,
                        )
                        if breakout is None:
                            continue
                        breakout_idx, breakout_direction = breakout
                        direction = "LONG" if breakout_direction == "upside" else "SHORT"
                        opposite_direction = "SHORT" if direction == "LONG" else "LONG"

                        for retest_window_bars in RETEST_WINDOW_BARS:
                            for retest_tolerance_pips in RETEST_TOLERANCE_PIPS:
                                for candle_filter in CANDLE_FILTERS:
                                    retest_idx = find_retest(
                                        df,
                                        breakout_idx,
                                        breakout_direction,
                                        retest_window_bars,
                                        float(retest_tolerance_pips),
                                        candle_filter,
                                        or_box,
                                        pip_size,
                                    )
                                    if retest_idx is None:
                                        continue

                                    retest_row = df.iloc[retest_idx]
                                    entry_price = float(retest_row["close"])
                                    for stop_mode in STOP_MODES:
                                        for target_mode, target_value in target_values():
                                            order = make_order(
                                                direction,
                                                entry_price,
                                                retest_row,
                                                or_box,
                                                stop_mode,
                                                target_mode,
                                                target_value,
                                                float(breakout_buffer_pips),
                                                pip_size,
                                            )
                                            if order is None:
                                                continue
                                            stop_loss, take_profit, risk_pips, reward_pips, rr_ratio = order
                                            for horizon_bars in TRADE_HORIZON_BARS:
                                                result = simulate_trade(
                                                    df,
                                                    retest_idx,
                                                    direction,
                                                    entry_price,
                                                    stop_loss,
                                                    take_profit,
                                                    horizon_bars,
                                                    friction,
                                                    pip_size,
                                                )
                                                opposite_stop = entry_price + risk_pips * pip_size if opposite_direction == "SHORT" else entry_price - risk_pips * pip_size
                                                opposite_tp = entry_price - reward_pips * pip_size if opposite_direction == "SHORT" else entry_price + reward_pips * pip_size
                                                opposite = simulate_trade(
                                                    df,
                                                    retest_idx,
                                                    opposite_direction,
                                                    entry_price,
                                                    opposite_stop,
                                                    opposite_tp,
                                                    horizon_bars,
                                                    friction,
                                                    pip_size,
                                                )
                                                hash_parts = (
                                                    symbol,
                                                    date_text,
                                                    session_name,
                                                    opening_range_bars,
                                                    monitor_bars,
                                                    breakout_buffer_pips,
                                                    retest_window_bars,
                                                    retest_tolerance_pips,
                                                    candle_filter,
                                                    stop_mode,
                                                    target_mode,
                                                    target_value,
                                                    horizon_bars,
                                                )
                                                random_pnl = result["net_pnl_pips"] if stable_random_uses_strategy(hash_parts) else opposite["net_pnl_pips"]
                                                entry_dt = pd.Timestamp(retest_row["datetime"])
                                                records.append(
                                                    {
                                                        "symbol": symbol,
                                                        "date": date_text,
                                                        "year": entry_dt.year,
                                                        "month": entry_dt.strftime("%Y-%m"),
                                                        "day_of_week": entry_dt.day_name(),
                                                        "session_name": session_name,
                                                        "session_open_hour": session_hour,
                                                        "session_open_datetime": or_box["session_open_datetime"],
                                                        "opening_range_bars": opening_range_bars,
                                                        "breakout_monitor_bars": monitor_bars,
                                                        "breakout_buffer_pips": breakout_buffer_pips,
                                                        "retest_window_bars": retest_window_bars,
                                                        "retest_tolerance_pips": retest_tolerance_pips,
                                                        "candle_filter": candle_filter,
                                                        "OR_high": or_box["OR_high"],
                                                        "OR_low": or_box["OR_low"],
                                                        "OR_mid": or_box["OR_mid"],
                                                        "OR_range_pips": or_range,
                                                        "breakout_direction": breakout_direction,
                                                        "breakout_datetime": pd.Timestamp(df.iloc[breakout_idx]["datetime"]),
                                                        "retest_datetime": entry_dt,
                                                        "entry_datetime": entry_dt,
                                                        "entry_price": entry_price,
                                                        "direction": direction,
                                                        "stop_mode": stop_mode,
                                                        "target_mode": target_mode,
                                                        "target_value": target_value,
                                                        "trade_horizon_bars": horizon_bars,
                                                        "stop_loss": stop_loss,
                                                        "take_profit": take_profit,
                                                        "risk_pips": risk_pips,
                                                        "reward_pips": reward_pips,
                                                        "rr_ratio": rr_ratio,
                                                        "friction_pips": friction,
                                                        "opposite_pnl_after_friction": opposite["net_pnl_pips"],
                                                        "random_pnl_after_friction": random_pnl,
                                                        "edge_vs_opposite": result["net_pnl_pips"] - opposite["net_pnl_pips"],
                                                        "edge_vs_random": result["net_pnl_pips"] - random_pnl,
                                                        **result,
                                                    }
                                                )
    return records


def profit_factor(values: pd.Series) -> float:
    gains = float(values[values > 0].sum())
    losses = abs(float(values[values < 0].sum()))
    if losses == 0:
        return math.inf if gains > 0 else 0.0
    return gains / losses


def summarize_group(group: pd.DataFrame) -> pd.Series:
    pnl = group["net_pnl_pips"]
    observations = int(len(group))
    wins = int((pnl > 0).sum())
    return pd.Series(
        {
            "observations": observations,
            "total_net_pnl": float(pnl.sum()),
            "expectancy": float(pnl.mean()) if observations else 0.0,
            "median_net_pnl": float(pnl.median()) if observations else 0.0,
            "win_rate": float(wins / observations) if observations else 0.0,
            "profit_factor": profit_factor(pnl),
            "avg_risk_pips": float(group["risk_pips"].mean()) if observations else 0.0,
            "avg_reward_pips": float(group["reward_pips"].mean()) if observations else 0.0,
            "ambiguous_rate": float(group["ambiguous"].mean()) if observations else 0.0,
            "opposite_total_net_pnl": float(group["opposite_pnl_after_friction"].sum()),
            "random_total_net_pnl": float(group["random_pnl_after_friction"].sum()),
            "edge_vs_opposite": float(group["edge_vs_opposite"].sum()),
            "edge_vs_random": float(group["edge_vs_random"].sum()),
        }
    )


def grouped_summary(events: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=group_cols + list(summarize_group(events).index))
    return events.groupby(group_cols, dropna=False).apply(summarize_group, include_groups=False).reset_index()


def walkforward_bucket(year: int) -> str:
    if year in (2022, 2023):
        return "reference_2022_2023"
    if year == 2024:
        return "validation_2024"
    if year in (2025, 2026):
        return "test_oos_2025_2026"
    return "outside_walkforward"


def dependency_checks(oos: pd.DataFrame) -> dict[str, Any]:
    if oos.empty:
        return {
            "top_month_profit_share": 1.0,
            "one_month_dependency": True,
            "positive_symbols": 0,
            "positive_sessions": 0,
            "broad_based": False,
        }
    positive_total = float(oos["net_pnl_pips"].sum())
    month_pnl = oos.groupby("month")["net_pnl_pips"].sum().sort_values(ascending=False)
    top_month = float(month_pnl.iloc[0]) if len(month_pnl) else 0.0
    top_share = top_month / positive_total if positive_total > 0 and top_month > 0 else 1.0
    positive_symbols = int((oos.groupby("symbol")["net_pnl_pips"].sum() > 0).sum())
    positive_sessions = int((oos.groupby("session_name")["net_pnl_pips"].sum() > 0).sum())
    return {
        "top_month_profit_share": top_share,
        "one_month_dependency": bool(top_share >= 0.60),
        "positive_symbols": positive_symbols,
        "positive_sessions": positive_sessions,
        "broad_based": bool(positive_symbols > 1 or positive_sessions > 1),
    }


def verdict(events: pd.DataFrame, min_oos_observations: int) -> tuple[pd.DataFrame, str, list[str]]:
    oos = events[events["walkforward"] == "test_oos_2025_2026"] if not events.empty else events
    stats = summarize_group(oos)
    checks = dependency_checks(oos)
    row = {**stats.to_dict(), **checks}
    row["walkforward"] = "test_oos_2025_2026"

    failures: list[str] = []
    if row["total_net_pnl"] <= 0:
        failures.append("OOS total_net_pnl <= 0")
    if row["expectancy"] <= 0:
        failures.append("OOS expectancy <= 0")
    if row["profit_factor"] <= 1:
        failures.append("OOS profit_factor <= 1")
    if row["profit_factor"] <= 1.15:
        failures.append("OOS profit_factor <= 1.15 PASS threshold")
    if row["edge_vs_random"] <= 0:
        failures.append("OOS edge_vs_random <= 0")
    if row["edge_vs_opposite"] <= 0:
        failures.append("OOS edge_vs_opposite <= 0")
    if row["observations"] < min_oos_observations:
        failures.append("OOS observations insufficient")
    if row["one_month_dependency"]:
        failures.append("one month explains most profit")
    if not row["broad_based"]:
        failures.append("result does not appear across more than one symbol or more than one session")

    pass_conditions = (
        row["total_net_pnl"] > 0
        and row["expectancy"] > 0
        and row["profit_factor"] > 1.15
        and row["edge_vs_random"] > 0
        and row["edge_vs_opposite"] > 0
        and row["observations"] >= min_oos_observations
        and not row["one_month_dependency"]
        and row["broad_based"]
    )
    weak_positive = row["total_net_pnl"] > 0 and row["expectancy"] > 0 and row["profit_factor"] > 1
    if pass_conditions:
        final = "PASS"
        recommendation = "continue_research"
    elif weak_positive:
        final = "WARN"
        recommendation = "do_not_deploy_under_sampled_or_weak_edge"
    else:
        final = "FAIL"
        recommendation = "reject"

    row["final_verdict"] = final
    row["recommendation"] = recommendation
    row["failure_reasons"] = "; ".join(failures) if failures else "none"
    columns = ["final_verdict", "recommendation", "walkforward"] + [c for c in row if c not in {"final_verdict", "recommendation", "walkforward"}]
    return pd.DataFrame([{c: row[c] for c in columns}]), final, failures


def write_summary(output_dir: Path, events: pd.DataFrame, verdict_df: pd.DataFrame, final: str, failures: list[str]) -> None:
    total_obs = len(events)
    verdict_row = verdict_df.iloc[0].to_dict()
    lines = [
        "# Opening Range Breakout + Retest Research Summary",
        "",
        "Research-only test of a pure Opening Range Breakout + Retest continuation strategy.",
        "The script does not enter on the breakout candle; entry is only at the close of a valid retest candle.",
        "No EA, live trading, MT4/MT5 execution, lot sizing, or account-equity logic is included.",
        "",
        "## Verdict",
        "",
        f"- Final verdict: **{final}**",
        f"- Recommendation: **{verdict_row['recommendation']}**",
        f"- OOS observations: {int(verdict_row['observations'])}",
        f"- OOS total_net_pnl: {verdict_row['total_net_pnl']:.2f} pips",
        f"- OOS expectancy: {verdict_row['expectancy']:.4f} pips/trade",
        f"- OOS profit_factor: {verdict_row['profit_factor']:.4f}",
        f"- OOS edge_vs_random: {verdict_row['edge_vs_random']:.2f} pips",
        f"- OOS edge_vs_opposite: {verdict_row['edge_vs_opposite']:.2f} pips",
        "",
        "## Failure / Warning Reasons",
        "",
    ]
    if failures:
        lines.extend(f"- {reason}" for reason in failures)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            f"- Total generated events: {total_obs}",
            "- Walk-forward buckets: 2022-2023 reference, 2024 validation, 2025-2026 test/OOS.",
            "- Costs: conservative spread + slippage + commission-equivalent pips by JPY/non-JPY pair type.",
            "- Ambiguous TP/SL same-candle cases are marked ambiguous and treated as SL first.",
            "",
            "## Output Files",
            "",
        ]
    )
    lines.extend(f"- `{name}`" for name in OUTPUT_FILES)
    (output_dir / "opening_range_retest_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(events: pd.DataFrame, config: Config) -> None:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if events.empty:
        events = pd.DataFrame(columns=EVENT_COLUMNS)
    else:
        events = events.copy()
        events["walkforward"] = events["year"].map(walkforward_bucket)
        for col in EVENT_COLUMNS:
            if col not in events.columns:
                events[col] = pd.NA
        events = events[EVENT_COLUMNS + ["walkforward"]]

    events.to_csv(output_dir / "opening_range_retest_events.csv", index=False)
    grouped_summary(events, ["symbol", "session_name"] + PARAMETER_COLUMNS).to_csv(
        output_dir / "opening_range_retest_by_symbol_session_params.csv", index=False
    )
    grouped_summary(events, ["symbol", "session_name"]).to_csv(output_dir / "opening_range_retest_by_symbol_session.csv", index=False)
    grouped_summary(events, ["symbol"]).to_csv(output_dir / "opening_range_retest_by_symbol.csv", index=False)
    grouped_summary(events, ["session_name"]).to_csv(output_dir / "opening_range_retest_by_session.csv", index=False)
    grouped_summary(events, ["year"]).to_csv(output_dir / "opening_range_retest_by_year.csv", index=False)
    grouped_summary(events, ["month"]).to_csv(output_dir / "opening_range_retest_by_month.csv", index=False)
    grouped_summary(events, ["walkforward"]).to_csv(output_dir / "opening_range_retest_walkforward.csv", index=False)
    verdict_df, final, failures = verdict(events, config.min_oos_observations)
    verdict_df.to_csv(output_dir / "opening_range_retest_verdict.csv", index=False)
    write_summary(output_dir, events, verdict_df, final, failures)


def run(config: Config) -> pd.DataFrame:
    all_records: list[dict[str, Any]] = []
    missing: list[Path] = []
    for symbol, filename in DATASETS.items():
        path = config.data_dir / filename
        if not path.exists():
            missing.append(path)
            continue
        all_records.extend(research_symbol(symbol, path))
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required dataset files:\n{missing_text}")
    events = pd.DataFrame(all_records)
    write_outputs(events, config)
    return events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research Opening Range Breakout + Retest strategy on FX M15 data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing M15 OHLC CSV files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/opening_range_retest"),
        help="Directory for research CSV/Markdown outputs.",
    )
    parser.add_argument("--min-oos-observations", type=int, default=50, help="Minimum OOS observations required for PASS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(data_dir=args.data_dir, output_dir=args.output_dir, min_oos_observations=args.min_oos_observations)
    events = run(config)
    print(f"Generated {len(events)} opening-range retest events")
    print(f"Wrote outputs to {config.output_dir}")


if __name__ == "__main__":
    main()
