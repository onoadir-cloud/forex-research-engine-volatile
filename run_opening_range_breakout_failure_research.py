#!/usr/bin/env python3
"""Research-only Opening Range Breakout/Failure study for FX M15 data.

This script is deliberately offline and research-only. It reads historical CSV data,
builds session opening ranges, evaluates breakout-continuation and breakout-failure
hypotheses, compares them with opposite/random baselines, and writes CSV/Markdown
research artifacts. It contains no EA, live trading, MT4/MT5 execution, lot sizing,
or account-equity logic.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
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

COST_PROFILES = {
    "JPY": {
        "low": {"spread_pips": 1.0, "slippage_pips": 0.3, "commission_equivalent_pips": 0.3},
        "conservative": {"spread_pips": 1.5, "slippage_pips": 0.5, "commission_equivalent_pips": 0.5},
        "high": {"spread_pips": 2.2, "slippage_pips": 0.8, "commission_equivalent_pips": 0.8},
    },
    "NON_JPY": {
        "low": {"spread_pips": 0.8, "slippage_pips": 0.2, "commission_equivalent_pips": 0.2},
        "conservative": {"spread_pips": 1.2, "slippage_pips": 0.4, "commission_equivalent_pips": 0.4},
        "high": {"spread_pips": 2.0, "slippage_pips": 0.8, "commission_equivalent_pips": 0.8},
    },
}

OPENING_RANGE_BARS = (2, 4)
MONITOR_BARS = (8, 16, 24)
FAILURE_CONFIRM_BARS = (2, 4)
CONTINUATION_STOP_MODES = ("OR_mid", "opposite_edge")
CONTINUATION_TARGET_PIPS = (10, 15, 20, 30)
FAILURE_TARGET_MODES = ("mid", "opposite_edge")
BUFFER_PIPS = (0, 2)
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
    "OR_high",
    "OR_low",
    "OR_mid",
    "OR_range_pips",
    "strategy_mode",
    "breakout_direction",
    "entry_datetime",
    "entry_price",
    "direction",
    "stop_loss",
    "take_profit",
    "stop_mode",
    "target_mode",
    "target_pips",
    "buffer_pips",
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
    "max_favorable_pips",
    "max_adverse_pips",
]

PARAMETER_COLUMNS = [
    "opening_range_bars",
    "monitor_bars",
    "failure_confirm_bars",
    "stop_mode",
    "target_mode",
    "target_pips",
    "buffer_pips",
    "trade_horizon_bars",
]


@dataclass(frozen=True)
class Config:
    data_dir: Path
    output_dir: Path
    cost_profile: str
    min_oos_observations: int


def infer_pip_size(symbol: str) -> float:
    return PIP_SIZE_JPY if "JPY" in symbol.upper() else PIP_SIZE_FALLBACK


def friction_pips(symbol: str, cost_profile: str) -> float:
    family = "JPY" if "JPY" in symbol.upper() else "NON_JPY"
    return float(sum(COST_PROFILES[family][cost_profile].values()))


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
    df["date"] = df["datetime"].dt.date
    return df


def find_session_open(day_df: pd.DataFrame, session_open_hour: int) -> tuple[int | None, pd.Timestamp | None]:
    if day_df.empty:
        return None, None
    date0 = pd.Timestamp(day_df["datetime"].iloc[0]).normalize()
    session_dt = date0 + pd.Timedelta(hours=session_open_hour)
    candidates = day_df.index[day_df["datetime"] >= session_dt].tolist()
    if not candidates:
        return None, None
    idx = int(candidates[0])
    return idx, pd.Timestamp(day_df.loc[idx, "datetime"])


def build_opening_range(day_df: pd.DataFrame, session_open_idx: int, opening_range_bars: int, pip_size: float) -> dict[str, Any] | None:
    or_df = day_df.loc[session_open_idx : session_open_idx + opening_range_bars - 1]
    if len(or_df) < opening_range_bars:
        return None
    or_high = float(or_df["high"].max())
    or_low = float(or_df["low"].min())
    or_mid = (or_high + or_low) / 2.0
    return {
        "OR_high": or_high,
        "OR_low": or_low,
        "OR_mid": or_mid,
        "OR_range_pips": (or_high - or_low) / pip_size,
        "monitor_start_idx": session_open_idx + opening_range_bars,
        "session_open_datetime": pd.Timestamp(day_df.loc[session_open_idx, "datetime"]),
    }


def pips_between(direction: str, start: float, end: float, pip_size: float) -> float:
    if direction == "LONG":
        return (end - start) / pip_size
    return (start - end) / pip_size


def continuation_order(
    breakout_direction: str,
    entry_price: float,
    or_box: dict[str, Any],
    stop_mode: str,
    target_pips: float,
    pip_size: float,
) -> tuple[str, float, float, float, float]:
    direction = "LONG" if breakout_direction == "upside" else "SHORT"
    if direction == "LONG":
        stop_loss = or_box["OR_mid"] if stop_mode == "OR_mid" else or_box["OR_low"]
        take_profit = entry_price + target_pips * pip_size
    else:
        stop_loss = or_box["OR_mid"] if stop_mode == "OR_mid" else or_box["OR_high"]
        take_profit = entry_price - target_pips * pip_size
    risk = abs(entry_price - stop_loss) / pip_size
    reward = abs(take_profit - entry_price) / pip_size
    return direction, float(stop_loss), float(take_profit), float(risk), float(reward)


def failure_order(
    breakout_direction: str,
    entry_price: float,
    breakout_extreme: float,
    or_box: dict[str, Any],
    target_mode: str,
    buffer_pips: float,
    pip_size: float,
) -> tuple[str, float, float, float, float]:
    if breakout_direction == "upside":
        direction = "SHORT"
        stop_loss = breakout_extreme + buffer_pips * pip_size
        take_profit = or_box["OR_mid"] if target_mode == "mid" else or_box["OR_low"]
    else:
        direction = "LONG"
        stop_loss = breakout_extreme - buffer_pips * pip_size
        take_profit = or_box["OR_mid"] if target_mode == "mid" else or_box["OR_high"]
    risk = abs(entry_price - stop_loss) / pip_size
    reward = abs(take_profit - entry_price) / pip_size
    return direction, float(stop_loss), float(take_profit), float(risk), float(reward)


def manage_trade(
    day_df: pd.DataFrame,
    entry_idx: int,
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    pip_size: float,
    friction: float,
    trade_horizon_bars: int,
) -> dict[str, Any]:
    future = day_df.loc[entry_idx + 1 : entry_idx + trade_horizon_bars]
    max_fav = 0.0
    max_adv = 0.0
    outcome = "timeout"
    ambiguous = False
    gross = 0.0
    bars_in_trade = 0

    for bars_in_trade, (_, row) in enumerate(future.iterrows(), start=1):
        high = float(row["high"])
        low = float(row["low"])
        if direction == "LONG":
            max_fav = max(max_fav, (high - entry_price) / pip_size)
            max_adv = min(max_adv, (low - entry_price) / pip_size)
            tp_hit = high >= take_profit
            sl_hit = low <= stop_loss
            if tp_hit and sl_hit:
                outcome = "ambiguous"
                ambiguous = True
                gross = (stop_loss - entry_price) / pip_size
                break
            if sl_hit:
                outcome = "loss"
                gross = (stop_loss - entry_price) / pip_size
                break
            if tp_hit:
                outcome = "win"
                gross = (take_profit - entry_price) / pip_size
                break
        else:
            max_fav = max(max_fav, (entry_price - low) / pip_size)
            max_adv = min(max_adv, (entry_price - high) / pip_size)
            tp_hit = low <= take_profit
            sl_hit = high >= stop_loss
            if tp_hit and sl_hit:
                outcome = "ambiguous"
                ambiguous = True
                gross = (entry_price - stop_loss) / pip_size
                break
            if sl_hit:
                outcome = "loss"
                gross = (entry_price - stop_loss) / pip_size
                break
            if tp_hit:
                outcome = "win"
                gross = (entry_price - take_profit) / pip_size
                break
    else:
        if not future.empty:
            bars_in_trade = len(future)
            close_price = float(future.iloc[-1]["close"])
            gross = pips_between(direction, entry_price, close_price, pip_size)
        else:
            bars_in_trade = 0
            gross = 0.0

    return {
        "outcome": outcome,
        "ambiguous": bool(ambiguous),
        "gross_pnl_pips": float(gross),
        "net_pnl_pips": float(gross - friction),
        "bars_in_trade": int(bars_in_trade),
        "max_favorable_pips": float(max_fav),
        "max_adverse_pips": float(max_adv),
    }


def baseline_prices(direction: str, entry_price: float, risk_pips: float, reward_pips: float, pip_size: float) -> tuple[str, float, float]:
    if direction == "LONG":
        return "SHORT", entry_price + risk_pips * pip_size, entry_price - reward_pips * pip_size
    return "LONG", entry_price - risk_pips * pip_size, entry_price + reward_pips * pip_size


def deterministic_random_uses_strategy(*parts: Any) -> bool:
    key = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2 == 0


def add_baselines(
    event: dict[str, Any],
    day_df: pd.DataFrame,
    entry_idx: int,
    pip_size: float,
    trade_horizon_bars: int,
    params_key: str,
) -> dict[str, Any]:
    friction = float(event["friction_pips"])
    opposite_direction, opposite_sl, opposite_tp = baseline_prices(
        event["direction"], event["entry_price"], event["risk_pips"], event["reward_pips"], pip_size
    )
    opposite = manage_trade(
        day_df, entry_idx, opposite_direction, event["entry_price"], opposite_sl, opposite_tp, pip_size, friction, trade_horizon_bars
    )
    use_strategy = deterministic_random_uses_strategy(
        event["symbol"], event["date"], event["session_name"], event["strategy_mode"], params_key
    )
    if use_strategy:
        random_pnl = event["net_pnl_pips"]
    else:
        random_pnl = opposite["net_pnl_pips"]
    event["opposite_pnl_after_friction"] = float(opposite["net_pnl_pips"])
    event["random_pnl_after_friction"] = float(random_pnl)
    event["edge_vs_opposite"] = float(event["net_pnl_pips"] - event["opposite_pnl_after_friction"])
    event["edge_vs_random"] = float(event["net_pnl_pips"] - event["random_pnl_after_friction"])
    return event


def event_base(symbol: str, day: Any, session_name: str, session_open_hour: int, or_box: dict[str, Any]) -> dict[str, Any]:
    day_ts = pd.Timestamp(day)
    return {
        "symbol": symbol,
        "date": str(day),
        "year": int(day_ts.year),
        "month": day_ts.strftime("%Y-%m"),
        "day_of_week": day_ts.day_name(),
        "session_name": session_name,
        "session_open_hour": session_open_hour,
        "session_open_datetime": or_box["session_open_datetime"],
        "OR_high": or_box["OR_high"],
        "OR_low": or_box["OR_low"],
        "OR_mid": or_box["OR_mid"],
        "OR_range_pips": or_box["OR_range_pips"],
    }


def continuation_events_for_breakout(
    day_df: pd.DataFrame,
    entry_idx: int,
    breakout_direction: str,
    base: dict[str, Any],
    or_box: dict[str, Any],
    opening_range_bars: int,
    monitor_bars: int,
    pip_size: float,
    friction: float,
) -> list[dict[str, Any]]:
    entry_price = float(day_df.loc[entry_idx, "close"])
    events = []
    for stop_mode in CONTINUATION_STOP_MODES:
        for target_pips in CONTINUATION_TARGET_PIPS:
            direction, sl, tp, risk, reward = continuation_order(
                breakout_direction, entry_price, or_box, stop_mode, target_pips, pip_size
            )
            rr = reward / risk if risk > 0 else math.nan
            if risk <= 0 or reward <= 0 or rr < 1.0:
                continue
            for horizon in TRADE_HORIZON_BARS:
                event = dict(base)
                event.update(
                    {
                        "opening_range_bars": opening_range_bars,
                        "monitor_bars": monitor_bars,
                        "failure_confirm_bars": 0,
                        "trade_horizon_bars": horizon,
                        "strategy_mode": "breakout_continuation",
                        "breakout_direction": breakout_direction,
                        "entry_datetime": pd.Timestamp(day_df.loc[entry_idx, "datetime"]),
                        "entry_price": entry_price,
                        "direction": direction,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "stop_mode": stop_mode,
                        "target_mode": "fixed_pips",
                        "target_pips": float(target_pips),
                        "buffer_pips": 0.0,
                        "risk_pips": risk,
                        "reward_pips": reward,
                        "rr_ratio": rr,
                        "friction_pips": friction,
                    }
                )
                event.update(manage_trade(day_df, entry_idx, direction, entry_price, sl, tp, pip_size, friction, horizon))
                key = f"OR={opening_range_bars};monitor={monitor_bars};stop={stop_mode};target={target_pips};horizon={horizon}"
                events.append(add_baselines(event, day_df, entry_idx, pip_size, horizon, key))
    return events


def failure_events_for_breakout(
    day_df: pd.DataFrame,
    breakout_idx: int,
    breakout_direction: str,
    base: dict[str, Any],
    or_box: dict[str, Any],
    opening_range_bars: int,
    monitor_bars: int,
    pip_size: float,
    friction: float,
) -> list[dict[str, Any]]:
    events = []
    for confirm_bars in FAILURE_CONFIRM_BARS:
        confirm_slice = day_df.loc[breakout_idx + 1 : breakout_idx + confirm_bars]
        if confirm_slice.empty:
            continue
        extreme = float(day_df.loc[breakout_idx, "high"] if breakout_direction == "upside" else day_df.loc[breakout_idx, "low"])
        for entry_idx, row in confirm_slice.iterrows():
            if breakout_direction == "upside":
                extreme = max(extreme, float(row["high"]))
                inside = float(row["close"]) < or_box["OR_high"] and float(row["close"]) > or_box["OR_low"]
            else:
                extreme = min(extreme, float(row["low"]))
                inside = float(row["close"]) > or_box["OR_low"] and float(row["close"]) < or_box["OR_high"]
            if not inside:
                continue
            entry_price = float(row["close"])
            for target_mode in FAILURE_TARGET_MODES:
                for buffer_pips in BUFFER_PIPS:
                    direction, sl, tp, risk, reward = failure_order(
                        breakout_direction, entry_price, extreme, or_box, target_mode, buffer_pips, pip_size
                    )
                    rr = reward / risk if risk > 0 else math.nan
                    if risk <= 0 or reward <= 0 or rr < 1.0:
                        continue
                    for horizon in TRADE_HORIZON_BARS:
                        event = dict(base)
                        event.update(
                            {
                                "opening_range_bars": opening_range_bars,
                                "monitor_bars": monitor_bars,
                                "failure_confirm_bars": confirm_bars,
                                "trade_horizon_bars": horizon,
                                "strategy_mode": "breakout_failure_reversion",
                                "breakout_direction": breakout_direction,
                                "entry_datetime": pd.Timestamp(row["datetime"]),
                                "entry_price": entry_price,
                                "direction": direction,
                                "stop_loss": sl,
                                "take_profit": tp,
                                "stop_mode": "breakout_extreme_buffer",
                                "target_mode": target_mode,
                                "target_pips": np.nan,
                                "buffer_pips": float(buffer_pips),
                                "risk_pips": risk,
                                "reward_pips": reward,
                                "rr_ratio": rr,
                                "friction_pips": friction,
                            }
                        )
                        event.update(manage_trade(day_df, int(entry_idx), direction, entry_price, sl, tp, pip_size, friction, horizon))
                        key = (
                            f"OR={opening_range_bars};monitor={monitor_bars};confirm={confirm_bars};"
                            f"target={target_mode};buffer={buffer_pips};horizon={horizon}"
                        )
                        events.append(add_baselines(event, day_df, int(entry_idx), pip_size, horizon, key))
            break
    return events


def generate_events_for_symbol(symbol: str, df: pd.DataFrame, cost_profile: str) -> list[dict[str, Any]]:
    pip_size = infer_pip_size(symbol)
    friction = friction_pips(symbol, cost_profile)
    events: list[dict[str, Any]] = []
    for day, day_df in df.groupby("date", sort=True):
        day_df = day_df.reset_index(drop=True)
        for session_name in SYMBOL_SESSION_MAP.get(symbol, []):
            session_open_hour = SESSION_OPEN_HOURS[session_name]
            session_idx, _ = find_session_open(day_df, session_open_hour)
            if session_idx is None:
                continue
            for opening_range_bars in OPENING_RANGE_BARS:
                or_box = build_opening_range(day_df, session_idx, opening_range_bars, pip_size)
                if not or_box or or_box["OR_range_pips"] <= 0:
                    continue
                base = event_base(symbol, day, session_name, session_open_hour, or_box)
                for monitor_bars in MONITOR_BARS:
                    start_idx = int(or_box["monitor_start_idx"])
                    monitor_df = day_df.loc[start_idx : start_idx + monitor_bars - 1]
                    for breakout_idx, row in monitor_df.iterrows():
                        close = float(row["close"])
                        if close > or_box["OR_high"]:
                            breakout_direction = "upside"
                        elif close < or_box["OR_low"]:
                            breakout_direction = "downside"
                        else:
                            continue
                        events.extend(
                            continuation_events_for_breakout(
                                day_df,
                                int(breakout_idx),
                                breakout_direction,
                                base,
                                or_box,
                                opening_range_bars,
                                monitor_bars,
                                pip_size,
                                friction,
                            )
                        )
                        events.extend(
                            failure_events_for_breakout(
                                day_df,
                                int(breakout_idx),
                                breakout_direction,
                                base,
                                or_box,
                                opening_range_bars,
                                monitor_bars,
                                pip_size,
                                friction,
                            )
                        )
                        break
    return events


def profit_factor(values: pd.Series) -> float:
    gains = float(values[values > 0].sum())
    losses = float(values[values < 0].sum())
    if losses < 0:
        return gains / abs(losses)
    return math.inf if gains > 0 else 0.0


def longest_loss_streak(outcomes: Iterable[float]) -> int:
    longest = current = 0
    for value in outcomes:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def max_drawdown(values: Iterable[float]) -> float:
    equity = np.cumsum(list(values))
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    return float(np.max(drawdown))


def aggregate(events: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = [
        *group_cols,
        "trades",
        "win_rate",
        "loss_rate",
        "timeout_rate",
        "ambiguous_rate",
        "mean_net_pnl",
        "median_net_pnl",
        "total_net_pnl",
        "profit_factor",
        "expectancy",
        "mean_opposite_pnl",
        "mean_random_pnl",
        "mean_edge_vs_opposite",
        "mean_edge_vs_random",
        "median_edge_vs_opposite",
        "median_edge_vs_random",
        "p05_net_pnl",
        "worst_trade",
        "best_trade",
        "average_rr",
        "longest_loss_streak",
        "max_drawdown_pips",
    ]
    if events.empty:
        return pd.DataFrame(columns=metric_cols)
    rows = []
    for keys, grp in events.sort_values("entry_datetime").groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        pnl = grp["net_pnl_pips"].astype(float)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            {
                "trades": int(len(grp)),
                "win_rate": float((grp["outcome"] == "win").mean()),
                "loss_rate": float((grp["outcome"] == "loss").mean()),
                "timeout_rate": float((grp["outcome"] == "timeout").mean()),
                "ambiguous_rate": float(grp["ambiguous"].astype(bool).mean()),
                "mean_net_pnl": float(pnl.mean()),
                "median_net_pnl": float(pnl.median()),
                "total_net_pnl": float(pnl.sum()),
                "profit_factor": float(profit_factor(pnl)),
                "expectancy": float(pnl.mean()),
                "mean_opposite_pnl": float(grp["opposite_pnl_after_friction"].mean()),
                "mean_random_pnl": float(grp["random_pnl_after_friction"].mean()),
                "mean_edge_vs_opposite": float(grp["edge_vs_opposite"].mean()),
                "mean_edge_vs_random": float(grp["edge_vs_random"].mean()),
                "median_edge_vs_opposite": float(grp["edge_vs_opposite"].median()),
                "median_edge_vs_random": float(grp["edge_vs_random"].median()),
                "p05_net_pnl": float(pnl.quantile(0.05)),
                "worst_trade": float(pnl.min()),
                "best_trade": float(pnl.max()),
                "average_rr": float(grp["rr_ratio"].mean()),
                "longest_loss_streak": longest_loss_streak(pnl),
                "max_drawdown_pips": max_drawdown(pnl),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=metric_cols)


def walkforward(events: pd.DataFrame) -> pd.DataFrame:
    cols = ["segment", *PARAMETER_COLUMNS, "trades", "total_net_pnl", "expectancy", "profit_factor", "mean_edge_vs_random", "mean_edge_vs_opposite"]
    if events.empty:
        return pd.DataFrame(columns=cols)
    segments = {
        "reference_2022_2023": events[(events["year"] >= 2022) & (events["year"] <= 2023)],
        "validation_2024": events[events["year"] == 2024],
        "test_oos_2025_2026": events[(events["year"] >= 2025) & (events["year"] <= 2026)],
    }
    frames = []
    for segment, segment_df in segments.items():
        agg = aggregate(segment_df, PARAMETER_COLUMNS)
        if agg.empty:
            continue
        slim = agg[[*PARAMETER_COLUMNS, "trades", "total_net_pnl", "expectancy", "profit_factor", "mean_edge_vs_random", "mean_edge_vs_opposite"]].copy()
        slim.insert(0, "segment", segment)
        frames.append(slim)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def assess_verdict(events: pd.DataFrame, by_params: pd.DataFrame, min_oos_observations: int) -> pd.DataFrame:
    rows = []
    if events.empty or by_params.empty:
        return pd.DataFrame([{"verdict": "FAIL", "reason": "No events generated.", "recommendation": "reject"}])
    oos = events[(events["year"] >= 2025) & (events["year"] <= 2026)]
    if oos.empty:
        return pd.DataFrame([{"verdict": "FAIL", "reason": "No 2025-2026 OOS observations.", "recommendation": "reject"}])
    for keys, grp in oos.groupby(PARAMETER_COLUMNS, dropna=False, sort=True):
        pnl = grp["net_pnl_pips"].astype(float)
        total = float(pnl.sum())
        exp = float(pnl.mean())
        pf = float(profit_factor(pnl))
        edge_random = float(grp["edge_vs_random"].mean())
        edge_opposite = float(grp["edge_vs_opposite"].mean())
        obs = int(len(grp))
        ambiguous_rate = float(grp["ambiguous"].astype(bool).mean())
        by_month = grp.groupby("month")["net_pnl_pips"].sum()
        top_month_share = float(by_month.max() / total) if total > 0 and not by_month.empty else math.inf
        positive_symbols = int((grp.groupby("symbol")["net_pnl_pips"].sum() > 0).sum())
        positive_sessions = int((grp.groupby("session_name")["net_pnl_pips"].sum() > 0).sum())
        loss_streak = longest_loss_streak(pnl)
        dd = max_drawdown(pnl)
        reasons = []
        if total <= 0:
            reasons.append("OOS total_net_pnl <= 0")
        if exp <= 0:
            reasons.append("OOS expectancy <= 0")
        if pf <= 1:
            reasons.append("OOS profit_factor <= 1")
        if edge_random <= 0:
            reasons.append("OOS edge_vs_random <= 0")
        if edge_opposite <= 0:
            reasons.append("OOS edge_vs_opposite <= 0")
        if obs < min_oos_observations:
            reasons.append("OOS observations are insufficient")
        if top_month_share > 0.5:
            reasons.append("one month explains most profit")
        if ambiguous_rate > 0.15:
            reasons.append("ambiguous_rate is excessive")
        broad = positive_symbols > 1 or positive_sessions > 1
        if not broad:
            reasons.append("result does not appear across more than one symbol or session")
        if not reasons and pf > 1.15:
            verdict = "PASS"
            recommendation = "proceed to locked robustness"
        elif total > 0 and exp > 0 and pf > 1 and obs >= max(10, min_oos_observations // 2):
            verdict = "WARN"
            recommendation = "needs more research"
            if edge_random <= 0 or edge_opposite <= 0:
                reasons.append("edge vs random/opposite is weak or mixed")
            if obs < min_oos_observations:
                reasons.append("observations are borderline")
            if loss_streak >= 8 or dd > max(50.0, abs(total)):
                reasons.append("drawdown or loss streak is high")
        else:
            verdict = "FAIL"
            recommendation = "reject"
        row = {col: key for col, key in zip(PARAMETER_COLUMNS, keys)}
        row.update(
            {
                "verdict": verdict,
                "recommendation": recommendation,
                "reason": "; ".join(reasons) if reasons else "All OOS quality checks passed.",
                "oos_observations": obs,
                "oos_total_net_pnl": total,
                "oos_expectancy": exp,
                "oos_profit_factor": pf,
                "oos_mean_edge_vs_random": edge_random,
                "oos_mean_edge_vs_opposite": edge_opposite,
                "oos_ambiguous_rate": ambiguous_rate,
                "oos_top_month_profit_share": top_month_share,
                "oos_positive_symbols": positive_symbols,
                "oos_positive_sessions": positive_sessions,
                "oos_longest_loss_streak": loss_streak,
                "oos_max_drawdown_pips": dd,
            }
        )
        rows.append(row)
    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    return pd.DataFrame(rows).sort_values(
        by=["verdict", "oos_expectancy", "oos_mean_edge_vs_random"],
        key=lambda s: s.map(order) if s.name == "verdict" else s,
        ascending=[True, False, False],
    )


def markdown_table(df: pd.DataFrame, columns: list[str], n: int = 10) -> str:
    if df.empty:
        return "No rows."
    view = df.loc[:, [c for c in columns if c in df.columns]].head(n).copy()
    if view.empty:
        return "No rows."
    string_view = view.fillna("").astype(str)
    header = "| " + " | ".join(string_view.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(string_view.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in string_view.to_numpy()]
    return "\n".join([header, separator, *rows])


def write_summary(
    path: Path,
    events: pd.DataFrame,
    by_params: pd.DataFrame,
    by_symbol_session_strategy: pd.DataFrame,
    wf: pd.DataFrame,
    verdict: pd.DataFrame,
    cost_profile: str,
) -> None:
    oos_params = aggregate(events[(events["year"] >= 2025) & (events["year"] <= 2026)], ["strategy_mode", *PARAMETER_COLUMNS])
    best_expectancy = oos_params.sort_values("expectancy", ascending=False) if not oos_params.empty else oos_params
    best_random = oos_params.sort_values("mean_edge_vs_random", ascending=False) if not oos_params.empty else oos_params
    best_opposite = oos_params.sort_values("mean_edge_vs_opposite", ascending=False) if not oos_params.empty else oos_params
    worst_tail = by_params.sort_values(["p05_net_pnl", "max_drawdown_pips"], ascending=[True, False]) if not by_params.empty else by_params
    top_verdict = verdict.iloc[0].to_dict() if not verdict.empty else {"verdict": "FAIL", "recommendation": "reject", "reason": "No verdict rows."}
    lines = [
        "# Opening Range Breakout/Failure Research Summary",
        "",
        "## 1. Research warning",
        "This is research only. It is not an EA, not live trading, not MT4/MT5 execution, and it performs no lot sizing or account-equity calculation. Results are filtered by OOS behavior, expectancy, profit factor, and edge vs random/opposite baselines rather than by a single best in-sample row.",
        "",
        "## 2. Opening Range construction",
        "For each symbol/session/date, the script finds the first available M15 candle at or after the configured server-hour session open. The opening range is the high/low of the first 2 or 4 M15 bars (30 or 60 minutes). Breakouts are monitored only after that range is complete.",
        "",
        "## 3. Breakout continuation vs failure/reversion",
        "Continuation enters in the breakout direction at the breakout candle close. Failure/reversion first requires a close outside the range, then a later close back inside the range within the confirmation window; it enters opposite the failed breakout and targets the midpoint or opposite edge.",
        "",
        "## 4. Session mapping",
        "Tokyo_Open=0, London_Open=7, NewYork_Open=13, Sydney_Open=22 (data/server hour). Symbol mappings: "
        + "; ".join(f"{symbol}: {', '.join(sessions)}" for symbol, sessions in SYMBOL_SESSION_MAP.items()),
        "",
        "## 5. Cost assumptions",
        f"Cost profile: `{cost_profile}`. Friction is spread + slippage + commission-equivalent pips, with separate JPY and non-JPY assumptions.",
        "",
        "## 6. Parameter grid",
        f"opening_range_bars={OPENING_RANGE_BARS}; monitor_bars={MONITOR_BARS}; failure_confirm_bars={FAILURE_CONFIRM_BARS}; continuation stop_mode={CONTINUATION_STOP_MODES}; continuation target_pips={CONTINUATION_TARGET_PIPS}; failure target_mode={FAILURE_TARGET_MODES}; buffer_pips={BUFFER_PIPS}; trade_horizon_bars={TRADE_HORIZON_BARS}.",
        "",
        "## 7. Best rows by OOS expectancy",
        markdown_table(best_expectancy, ["strategy_mode", *PARAMETER_COLUMNS, "trades", "expectancy", "profit_factor", "mean_edge_vs_random", "mean_edge_vs_opposite"]),
        "",
        "## 8. Best rows by OOS edge vs random",
        markdown_table(best_random, ["strategy_mode", *PARAMETER_COLUMNS, "trades", "expectancy", "profit_factor", "mean_edge_vs_random", "mean_edge_vs_opposite"]),
        "",
        "## 9. Best rows by OOS edge vs opposite",
        markdown_table(best_opposite, ["strategy_mode", *PARAMETER_COLUMNS, "trades", "expectancy", "profit_factor", "mean_edge_vs_random", "mean_edge_vs_opposite"]),
        "",
        "## 10. Worst tail-risk rows",
        markdown_table(worst_tail, ["symbol", "session_name", "strategy_mode", *PARAMETER_COLUMNS, "trades", "p05_net_pnl", "worst_trade", "longest_loss_streak", "max_drawdown_pips"]),
        "",
        "## 11. Symbol/session/strategy summary",
        markdown_table(by_symbol_session_strategy.sort_values("expectancy", ascending=False) if not by_symbol_session_strategy.empty else by_symbol_session_strategy, ["symbol", "session_name", "strategy_mode", "trades", "expectancy", "profit_factor", "mean_edge_vs_random", "mean_edge_vs_opposite"]),
        "",
        "## 12. Walk-forward interpretation",
        "Reference=2022-2023, validation=2024, test/OOS=2025-2026. Prefer parameter families that stay directionally consistent across segments rather than one-month or one-symbol spikes.",
        markdown_table(wf, ["segment", *PARAMETER_COLUMNS, "trades", "total_net_pnl", "expectancy", "profit_factor", "mean_edge_vs_random", "mean_edge_vs_opposite"]),
        "",
        "## 13. Final PASS/WARN/FAIL conclusion",
        f"Final conclusion: **{top_verdict.get('verdict', 'FAIL')}**. Reason: {top_verdict.get('reason', '')}",
        "",
        "## 14. Recommendation",
        f"Recommendation: **{top_verdict.get('recommendation', 'reject')}**.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_research(cfg: Config) -> dict[str, Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    all_events: list[dict[str, Any]] = []
    for symbol, filename in DATASETS.items():
        path = cfg.data_dir / filename
        if not path.exists():
            continue
        df = normalize_ohlc_csv(path)
        all_events.extend(generate_events_for_symbol(symbol, df, cfg.cost_profile))
    events = pd.DataFrame(all_events)
    if events.empty:
        events = pd.DataFrame(columns=[*EVENT_COLUMNS, *PARAMETER_COLUMNS])
    else:
        events = events[[*EVENT_COLUMNS, *[c for c in PARAMETER_COLUMNS if c not in EVENT_COLUMNS]]]
        events = events.sort_values(["entry_datetime", "symbol", "session_name", "strategy_mode"]).reset_index(drop=True)

    by_params = aggregate(events, ["symbol", "session_name", "strategy_mode", *PARAMETER_COLUMNS])
    by_symbol_session_strategy = aggregate(events, ["symbol", "session_name", "strategy_mode"])
    by_symbol = aggregate(events, ["symbol"])
    by_session = aggregate(events, ["session_name"])
    by_strategy_mode = aggregate(events, ["strategy_mode"])
    by_year = aggregate(events, ["year"])
    by_month = aggregate(events, ["month"])
    wf = walkforward(events)
    verdict = assess_verdict(events, by_params, cfg.min_oos_observations)

    outputs = {
        "events": cfg.output_dir / "opening_range_events.csv",
        "by_params": cfg.output_dir / "opening_range_by_symbol_session_strategy_params.csv",
        "by_symbol_session_strategy": cfg.output_dir / "opening_range_by_symbol_session_strategy.csv",
        "by_symbol": cfg.output_dir / "opening_range_by_symbol.csv",
        "by_session": cfg.output_dir / "opening_range_by_session.csv",
        "by_strategy_mode": cfg.output_dir / "opening_range_by_strategy_mode.csv",
        "by_year": cfg.output_dir / "opening_range_by_year.csv",
        "by_month": cfg.output_dir / "opening_range_by_month.csv",
        "walkforward": cfg.output_dir / "opening_range_walkforward.csv",
        "verdict": cfg.output_dir / "opening_range_verdict.csv",
        "summary": cfg.output_dir / "opening_range_summary.md",
    }
    events.to_csv(outputs["events"], index=False)
    by_params.to_csv(outputs["by_params"], index=False)
    by_symbol_session_strategy.to_csv(outputs["by_symbol_session_strategy"], index=False)
    by_symbol.to_csv(outputs["by_symbol"], index=False)
    by_session.to_csv(outputs["by_session"], index=False)
    by_strategy_mode.to_csv(outputs["by_strategy_mode"], index=False)
    by_year.to_csv(outputs["by_year"], index=False)
    by_month.to_csv(outputs["by_month"], index=False)
    wf.to_csv(outputs["walkforward"], index=False)
    verdict.to_csv(outputs["verdict"], index=False)
    write_summary(outputs["summary"], events, by_params, by_symbol_session_strategy, wf, verdict, cfg.cost_profile)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research-only Opening Range Breakout/Failure study.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/opening_range_research"))
    parser.add_argument("--cost-profile", choices=("low", "conservative", "high"), default="conservative")
    parser.add_argument("--min-oos-observations", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(args.data_dir, args.output_dir, args.cost_profile, args.min_oos_observations)
    outputs = run_research(cfg)
    print("Opening Range research complete. Outputs:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
