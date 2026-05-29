#!/usr/bin/env python3
"""Research-only Daily Box Session-Open Sweep/Reclaim study.

This script is deliberately offline/research-only: it reads historical M15 OHLC CSVs,
constructs each day's box from the previous completed trading date only, scans from
configured session opens, and writes CSV/Markdown research outputs. It contains no
broker connectivity, live trading, position sizing, account equity, EA, MT4, or MT5
execution logic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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

DEFAULT_SESSION_OPEN_HOURS = {
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

TARGET_MODES = ("mid", "far_quartile", "opposite_edge")
EVENT_COLUMNS = [
    "symbol", "date", "year", "month", "day_of_week", "session_name", "session_open_hour",
    "session_open_datetime", "box_source_date", "box_top", "box_bottom", "box_mid", "box_q1", "box_q3",
    "box_range_pips", "setup_direction", "sweep_datetime", "confirmation_datetime", "entry_price",
    "stop_loss", "take_profit", "target_mode", "buffer_pips", "risk_pips", "reward_pips", "rr_ratio",
    "friction_pips", "outcome", "ambiguous", "gross_pnl_pips", "net_pnl_pips",
    "continuation_pnl_after_friction", "random_pnl_after_friction", "edge_vs_continuation",
    "edge_vs_random", "bars_in_trade", "force_closed", "current_day_first_candle_time",
    "box_source_candles_count", "skipped_competing_setups",
]


@dataclass(frozen=True)
class Config:
    data_dir: Path
    output_dir: Path
    cost_profile: str
    scan_window_bars: int
    force_close_hour: int
    force_close_minute: int
    buffer_pips: float
    risk_reward_min: float
    session_open_hours: dict[str, int]
    min_oos_observations: int


def infer_pip_size(symbol: str) -> float:
    return PIP_SIZE_JPY if "JPY" in symbol.upper() else PIP_SIZE_FALLBACK


def friction_pips(symbol: str, cost_profile: str) -> float:
    family = "JPY" if "JPY" in symbol.upper() else "NON_JPY"
    return float(sum(COST_PROFILES[family][cost_profile].values()))


def normalize_ohlc_csv(path: Path) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
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
    duplicate_count = int(raw["datetime"].duplicated().sum())
    if duplicate_count:
        warnings.append(f"{path.name}: dropped {duplicate_count} duplicate datetime row(s).")
    df = raw[["datetime", "open", "high", "low", "close"]].drop_duplicates("datetime")
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date
    return df, warnings


def build_daily_boxes(df: pd.DataFrame, pip_size: float) -> tuple[dict[object, dict], list[str]]:
    warnings: list[str] = []
    day_groups = {day: group.dropna(subset=["open", "high", "low", "close"]).copy() for day, group in df.groupby("date", sort=True)}
    dates = sorted(day_groups)
    boxes: dict[object, dict] = {}
    for i, day in enumerate(dates):
        if i == 0:
            warnings.append(f"{day}: skipped because no previous completed trading date exists for the box.")
            continue
        prev_day = dates[i - 1]
        prev_df = day_groups.get(prev_day, pd.DataFrame())
        if prev_df.empty:
            warnings.append(f"{day}: skipped because previous trading date {prev_day} has insufficient OHLC data.")
            continue
        top = float(prev_df["high"].max())
        bottom = float(prev_df["low"].min())
        if not top > bottom:
            warnings.append(f"{day}: skipped because previous trading date {prev_day} has invalid/flat box range.")
            continue
        mid = (top + bottom) / 2.0
        boxes[day] = {
            "box_source_date": str(prev_day),
            "box_top": top,
            "box_bottom": bottom,
            "box_mid": mid,
            "box_q1": bottom + 0.25 * (top - bottom),
            "box_q3": bottom + 0.75 * (top - bottom),
            "box_range_pips": (top - bottom) / pip_size,
            "box_source_candles_count": int(len(prev_df)),
        }
    return boxes, warnings


def find_session_open(day_df: pd.DataFrame, session_hour: int) -> tuple[int | None, pd.Timestamp | None]:
    day = day_df.iloc[0]["date"]
    threshold = pd.Timestamp(f"{day} {session_hour:02d}:00:00")
    matches = day_df.index[day_df["datetime"] >= threshold].tolist()
    if not matches:
        return None, None
    idx = int(matches[0])
    return idx, pd.Timestamp(day_df.loc[idx, "datetime"])


def _inside_box(row: pd.Series, box_bottom: float, box_top: float) -> bool:
    return bool(row["low"] >= box_bottom and row["high"] <= box_top)


def detect_setup(scan_df: pd.DataFrame, direction: str, box: dict) -> dict | None:
    box_bottom = float(box["box_bottom"])
    box_top = float(box["box_top"])
    if scan_df.empty:
        return None

    sweep_pos: int | None = None
    if direction == "long":
        for pos, (_, row) in enumerate(scan_df.iterrows()):
            if row["low"] <= box_bottom:
                sweep_pos = pos
                break
    else:
        for pos, (_, row) in enumerate(scan_df.iterrows()):
            if row["high"] >= box_top:
                sweep_pos = pos
                break
    if sweep_pos is None:
        return None

    pre_sweep = scan_df.iloc[:sweep_pos]
    if direction == "long":
        refs = pre_sweep[(pre_sweep["close"] > pre_sweep["open"]) & pre_sweep.apply(lambda r: _inside_box(r, box_bottom, box_top), axis=1)]
        if refs.empty:
            return None
        ref_value = float(refs.iloc[-1]["high"])
        for pos in range(sweep_pos + 1, len(scan_df)):
            row = scan_df.iloc[pos]
            if box_bottom < row["close"] < box_top and row["close"] > ref_value:
                return {
                    "setup_direction": "long",
                    "sweep_datetime": pd.Timestamp(scan_df.iloc[sweep_pos]["datetime"]),
                    "confirmation_datetime": pd.Timestamp(row["datetime"]),
                    "confirmation_pos": pos,
                    "entry_price": float(row["close"]),
                }
    else:
        refs = pre_sweep[(pre_sweep["close"] < pre_sweep["open"]) & pre_sweep.apply(lambda r: _inside_box(r, box_bottom, box_top), axis=1)]
        if refs.empty:
            return None
        ref_value = float(refs.iloc[-1]["low"])
        for pos in range(sweep_pos + 1, len(scan_df)):
            row = scan_df.iloc[pos]
            if box_bottom < row["close"] < box_top and row["close"] < ref_value:
                return {
                    "setup_direction": "short",
                    "sweep_datetime": pd.Timestamp(scan_df.iloc[sweep_pos]["datetime"]),
                    "confirmation_datetime": pd.Timestamp(row["datetime"]),
                    "confirmation_pos": pos,
                    "entry_price": float(row["close"]),
                }
    return None


def target_price(direction: str, target_mode: str, box: dict) -> float:
    if direction == "long":
        return float({"mid": box["box_mid"], "far_quartile": box["box_q3"], "opposite_edge": box["box_top"]}[target_mode])
    return float({"mid": box["box_mid"], "far_quartile": box["box_q1"], "opposite_edge": box["box_bottom"]}[target_mode])


def stop_price(direction: str, box: dict, buffer_pips: float, pip_size: float) -> float:
    buffer_px = buffer_pips * pip_size
    return float(box["box_bottom"] - buffer_px) if direction == "long" else float(box["box_top"] + buffer_px)


def risk_reward(direction: str, entry: float, sl: float, tp: float, pip_size: float) -> tuple[float, float, float]:
    if direction == "long":
        risk = (entry - sl) / pip_size
        reward = (tp - entry) / pip_size
    else:
        risk = (sl - entry) / pip_size
        reward = (entry - tp) / pip_size
    rr = reward / risk if risk > 0 else math.nan
    return float(risk), float(reward), float(rr)


def manage_trade(
    day_df: pd.DataFrame,
    entry_datetime: pd.Timestamp,
    direction: str,
    entry: float,
    sl: float,
    tp: float,
    pip_size: float,
    friction: float,
    force_close_hour: int,
    force_close_minute: int,
) -> dict:
    day = day_df.iloc[0]["date"]
    cutoff = pd.Timestamp(f"{day} {force_close_hour:02d}:{force_close_minute:02d}:00")
    post = day_df[day_df["datetime"] > entry_datetime].copy()
    outcome = "force_close"
    exit_price = entry
    ambiguous = False
    force_closed = True
    bars_in_trade = 0

    for _, row in post.iterrows():
        if row["datetime"] > cutoff:
            break
        bars_in_trade += 1
        if direction == "long":
            hit_sl = row["low"] <= sl
            hit_tp = row["high"] >= tp
        else:
            hit_sl = row["high"] >= sl
            hit_tp = row["low"] <= tp
        if hit_sl and hit_tp:
            outcome = "ambiguous"
            exit_price = sl
            ambiguous = True
            force_closed = False
            break
        if hit_sl:
            outcome = "loss"
            exit_price = sl
            force_closed = False
            break
        if hit_tp:
            outcome = "win"
            exit_price = tp
            force_closed = False
            break

    if outcome == "force_close":
        eligible = post[post["datetime"] <= cutoff]
        if eligible.empty:
            eligible = post
        if not eligible.empty:
            exit_price = float(eligible.iloc[-1]["close"])
            bars_in_trade = max(bars_in_trade, int(len(eligible)))

    gross = (exit_price - entry) / pip_size if direction == "long" else (entry - exit_price) / pip_size
    return {
        "outcome": outcome,
        "ambiguous": bool(ambiguous),
        "gross_pnl_pips": float(gross),
        "net_pnl_pips": float(gross - friction),
        "bars_in_trade": int(bars_in_trade),
        "force_closed": bool(force_closed),
    }


def opposite_direction(direction: str) -> str:
    return "short" if direction == "long" else "long"


def baseline_prices(direction: str, entry: float, risk_pips: float, reward_pips: float, pip_size: float) -> tuple[str, float, float]:
    base_direction = opposite_direction(direction)
    if base_direction == "long":
        return base_direction, entry - risk_pips * pip_size, entry + reward_pips * pip_size
    return base_direction, entry + risk_pips * pip_size, entry - reward_pips * pip_size


def deterministic_random_uses_strategy(symbol: str, date: str, session_name: str, target_mode: str) -> bool:
    key = f"{symbol}|{date}|{session_name}|{target_mode}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest(), 16) % 2 == 0


def scan_symbol(symbol: str, path: Path, cfg: Config) -> tuple[pd.DataFrame, list[str]]:
    df, warnings = normalize_ohlc_csv(path)
    pip_size = infer_pip_size(symbol)
    friction = friction_pips(symbol, cfg.cost_profile)
    boxes, box_warnings = build_daily_boxes(df, pip_size)
    warnings.extend([f"{symbol}: {w}" for w in box_warnings[:25]])
    if len(box_warnings) > 25:
        warnings.append(f"{symbol}: {len(box_warnings) - 25} additional box warnings suppressed.")

    events: list[dict] = []
    for day, day_df in df.groupby("date", sort=True):
        if day not in boxes:
            continue
        day_df = day_df.sort_values("datetime").reset_index(drop=True)
        first_candle = pd.Timestamp(day_df.iloc[0]["datetime"])
        box = boxes[day]
        for session_name in SYMBOL_SESSION_MAP[symbol]:
            session_hour = int(cfg.session_open_hours[session_name])
            open_idx, session_dt = find_session_open(day_df, session_hour)
            if open_idx is None or session_dt is None:
                warnings.append(f"{symbol} {day} {session_name}: no candle at/after configured session open hour {session_hour}.")
                continue
            scan_df = day_df.iloc[open_idx : open_idx + cfg.scan_window_bars].reset_index(drop=True)
            candidates = [c for c in (detect_setup(scan_df, "long", box), detect_setup(scan_df, "short", box)) if c is not None]
            if not candidates:
                continue
            for target_mode in TARGET_MODES:
                valid_candidates: list[dict] = []
                for candidate in candidates:
                    direction = candidate["setup_direction"]
                    entry = float(candidate["entry_price"])
                    sl = stop_price(direction, box, cfg.buffer_pips, pip_size)
                    tp = target_price(direction, target_mode, box)
                    risk, reward, rr = risk_reward(direction, entry, sl, tp, pip_size)
                    if direction == "long" and tp <= entry:
                        continue
                    if direction == "short" and tp >= entry:
                        continue
                    if risk <= 0 or reward <= 0 or rr < cfg.risk_reward_min:
                        continue
                    item = dict(candidate)
                    item.update({"stop_loss": sl, "take_profit": tp, "risk_pips": risk, "reward_pips": reward, "rr_ratio": rr})
                    valid_candidates.append(item)
                if not valid_candidates:
                    continue
                valid_candidates.sort(key=lambda c: c["confirmation_datetime"])
                chosen = valid_candidates[0]
                skipped_competing = max(0, len(valid_candidates) - 1)
                direction = chosen["setup_direction"]
                entry = float(chosen["entry_price"])
                sl = float(chosen["stop_loss"])
                tp = float(chosen["take_profit"])
                management = manage_trade(
                    day_df, chosen["confirmation_datetime"], direction, entry, sl, tp, pip_size, friction,
                    cfg.force_close_hour, cfg.force_close_minute,
                )
                cont_direction, cont_sl, cont_tp = baseline_prices(direction, entry, chosen["risk_pips"], chosen["reward_pips"], pip_size)
                cont = manage_trade(
                    day_df, chosen["confirmation_datetime"], cont_direction, entry, cont_sl, cont_tp, pip_size, friction,
                    cfg.force_close_hour, cfg.force_close_minute,
                )
                if deterministic_random_uses_strategy(symbol, str(day), session_name, target_mode):
                    random_pnl = management["net_pnl_pips"]
                else:
                    random_pnl = cont["net_pnl_pips"]
                net_pnl = management["net_pnl_pips"]
                events.append({
                    "symbol": symbol,
                    "date": str(day),
                    "year": int(pd.Timestamp(day).year),
                    "month": pd.Timestamp(day).strftime("%Y-%m"),
                    "day_of_week": pd.Timestamp(day).day_name(),
                    "session_name": session_name,
                    "session_open_hour": session_hour,
                    "session_open_datetime": session_dt,
                    "box_source_date": box["box_source_date"],
                    "box_top": box["box_top"],
                    "box_bottom": box["box_bottom"],
                    "box_mid": box["box_mid"],
                    "box_q1": box["box_q1"],
                    "box_q3": box["box_q3"],
                    "box_range_pips": box["box_range_pips"],
                    "setup_direction": direction,
                    "sweep_datetime": chosen["sweep_datetime"],
                    "confirmation_datetime": chosen["confirmation_datetime"],
                    "entry_price": entry,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "target_mode": target_mode,
                    "buffer_pips": cfg.buffer_pips,
                    "risk_pips": chosen["risk_pips"],
                    "reward_pips": chosen["reward_pips"],
                    "rr_ratio": chosen["rr_ratio"],
                    "friction_pips": friction,
                    "outcome": management["outcome"],
                    "ambiguous": management["ambiguous"],
                    "gross_pnl_pips": management["gross_pnl_pips"],
                    "net_pnl_pips": net_pnl,
                    "continuation_pnl_after_friction": cont["net_pnl_pips"],
                    "random_pnl_after_friction": random_pnl,
                    "edge_vs_continuation": net_pnl - cont["net_pnl_pips"],
                    "edge_vs_random": net_pnl - random_pnl,
                    "bars_in_trade": management["bars_in_trade"],
                    "force_closed": management["force_closed"],
                    "current_day_first_candle_time": first_candle.strftime("%H:%M:%S"),
                    "box_source_candles_count": box["box_source_candles_count"],
                    "skipped_competing_setups": skipped_competing,
                })
    return pd.DataFrame(events, columns=EVENT_COLUMNS), warnings


def profit_factor(net: pd.Series) -> float:
    wins = float(net[net > 0].sum())
    losses = float(net[net < 0].sum())
    if losses < 0:
        return wins / abs(losses)
    return math.inf if wins > 0 else 0.0


def longest_loss_streak(outcomes: Iterable[str], net_values: Iterable[float] | None = None) -> int:
    streak = 0
    longest = 0
    nets = list(net_values) if net_values is not None else []
    for idx, outcome in enumerate(outcomes):
        losing = outcome in {"loss", "ambiguous"} or (idx < len(nets) and nets[idx] < 0)
        if losing:
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0
    return int(longest)


def max_drawdown(net: pd.Series) -> float:
    if net.empty:
        return 0.0
    equity = net.cumsum()
    return float((equity - equity.cummax()).min())


def aggregate(events: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    metric_cols = [
        "trades", "win_rate", "loss_rate", "force_close_rate", "ambiguous_rate", "mean_net_pnl",
        "median_net_pnl", "total_net_pnl", "profit_factor", "expectancy", "mean_continuation_pnl",
        "mean_random_pnl", "mean_edge_vs_continuation", "mean_edge_vs_random", "median_edge_vs_continuation",
        "median_edge_vs_random", "p05_net_pnl", "worst_trade", "best_trade", "average_rr",
        "longest_loss_streak", "max_drawdown_pips",
    ]
    if events.empty:
        return pd.DataFrame(columns=(group_cols or []) + metric_cols)

    def metrics(g: pd.DataFrame) -> pd.Series:
        ordered = g.sort_values("confirmation_datetime")
        net = ordered["net_pnl_pips"].astype(float)
        return pd.Series({
            "trades": int(len(ordered)),
            "win_rate": float((ordered["outcome"] == "win").mean()),
            "loss_rate": float((ordered["outcome"].isin(["loss", "ambiguous"])).mean()),
            "force_close_rate": float((ordered["force_closed"] == True).mean()),
            "ambiguous_rate": float((ordered["ambiguous"] == True).mean()),
            "mean_net_pnl": float(net.mean()),
            "median_net_pnl": float(net.median()),
            "total_net_pnl": float(net.sum()),
            "profit_factor": float(profit_factor(net)),
            "expectancy": float(net.mean()),
            "mean_continuation_pnl": float(ordered["continuation_pnl_after_friction"].mean()),
            "mean_random_pnl": float(ordered["random_pnl_after_friction"].mean()),
            "mean_edge_vs_continuation": float(ordered["edge_vs_continuation"].mean()),
            "mean_edge_vs_random": float(ordered["edge_vs_random"].mean()),
            "median_edge_vs_continuation": float(ordered["edge_vs_continuation"].median()),
            "median_edge_vs_random": float(ordered["edge_vs_random"].median()),
            "p05_net_pnl": float(net.quantile(0.05)),
            "worst_trade": float(net.min()),
            "best_trade": float(net.max()),
            "average_rr": float(ordered["rr_ratio"].mean()),
            "longest_loss_streak": longest_loss_streak(ordered["outcome"].tolist(), net.tolist()),
            "max_drawdown_pips": max_drawdown(net),
        })

    if not group_cols:
        return pd.DataFrame([metrics(events)])
    return events.groupby(group_cols, dropna=False, sort=True).apply(metrics, include_groups=False).reset_index()


def walkforward(events: pd.DataFrame) -> pd.DataFrame:
    segments = [
        ("reference", 2022, 2023),
        ("validation", 2024, 2024),
        ("test_oos", 2025, 2026),
    ]
    rows: list[pd.DataFrame] = []
    for segment, start_year, end_year in segments:
        seg = events[(events["year"] >= start_year) & (events["year"] <= end_year)] if not events.empty else events
        agg = aggregate(seg, ["symbol", "session_name", "target_mode"])
        if agg.empty:
            agg = pd.DataFrame(columns=["symbol", "session_name", "target_mode"] + aggregate(events.iloc[0:0]).columns.tolist())
        overall = aggregate(seg)
        if not overall.empty:
            overall.insert(0, "target_mode", "ALL")
            overall.insert(0, "session_name", "ALL")
            overall.insert(0, "symbol", "ALL")
            agg = pd.concat([agg, overall], ignore_index=True)
        agg.insert(0, "to_year", end_year)
        agg.insert(0, "from_year", start_year)
        agg.insert(0, "segment", segment)
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def verdict(wf: pd.DataFrame, by_month: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, str, str]:
    oos = wf[(wf["segment"] == "test_oos") & (wf["symbol"] == "ALL")]
    if oos.empty:
        return pd.DataFrame([{"verdict": "FAIL", "recommendation": "reject", "reason": "No test/OOS observations were generated."}]), "FAIL", "reject"
    row = oos.iloc[0]
    oos_trades = int(row["trades"])
    month_oos = by_month[by_month["month"].astype(str).str[:4].astype(int).between(2025, 2026)] if not by_month.empty else by_month
    total_oos = float(row["total_net_pnl"])
    one_month_dominant = False
    if not month_oos.empty and total_oos > 0:
        one_month_dominant = float(month_oos["total_net_pnl"].max()) > 0.6 * total_oos
    ambiguous_high = float(row["ambiguous_rate"]) > 0.10
    dd_too_large = total_oos > 0 and abs(float(row["max_drawdown_pips"])) > max(250.0, total_oos * 1.5)
    streak_too_large = int(row["longest_loss_streak"]) > 10

    fail_reasons: list[str] = []
    if total_oos <= 0:
        fail_reasons.append("OOS total_net_pnl <= 0")
    if float(row["expectancy"]) <= 0:
        fail_reasons.append("OOS expectancy <= 0")
    if float(row["profit_factor"]) <= 1:
        fail_reasons.append("OOS profit_factor <= 1")
    if float(row["mean_edge_vs_random"]) <= 0:
        fail_reasons.append("OOS edge vs random <= 0")
    if float(row["mean_edge_vs_continuation"]) <= 0:
        fail_reasons.append("OOS edge vs continuation <= 0")
    if oos_trades < cfg.min_oos_observations:
        fail_reasons.append("OOS observations are insufficient")
    if one_month_dominant:
        fail_reasons.append("one month explains most OOS profit")
    if ambiguous_high:
        fail_reasons.append("ambiguous rate is too high")

    pass_core = (
        total_oos > 0
        and float(row["expectancy"]) > 0
        and float(row["profit_factor"]) > 1.15
        and float(row["mean_edge_vs_random"]) > 0
        and float(row["mean_edge_vs_continuation"]) > 0
        and oos_trades >= cfg.min_oos_observations
        and not one_month_dominant
        and not ambiguous_high
        and not dd_too_large
        and not streak_too_large
    )
    warn_reasons: list[str] = []
    if total_oos > 0 and not pass_core:
        warn_reasons.append("OOS is positive but weak or unstable")
    if float(row["profit_factor"]) <= 1.15:
        warn_reasons.append("profit factor is not above the 1.15 PASS threshold")
    if abs(float(row["mean_edge_vs_random"])) < 1.0:
        warn_reasons.append("edge vs random is borderline")
    if abs(float(row["mean_edge_vs_continuation"])) < 1.0:
        warn_reasons.append("edge vs continuation is borderline or mixed")
    if oos_trades < cfg.min_oos_observations * 1.5:
        warn_reasons.append("observations are borderline")
    if dd_too_large or streak_too_large:
        warn_reasons.append("drawdown or loss streak is large relative to total pnl")

    if pass_core:
        label, recommendation, reasons = "PASS", "proceed to locked robustness", ["OOS passes profitability, edge, observation, ambiguity, concentration, and risk filters."]
    elif fail_reasons:
        label, recommendation, reasons = "FAIL", "reject", fail_reasons
    else:
        label, recommendation, reasons = "WARN", "needs more research", warn_reasons or ["OOS is mixed."]
    verdict_df = pd.DataFrame([{
        "verdict": label,
        "recommendation": recommendation,
        "reason": "; ".join(reasons),
        "oos_trades": oos_trades,
        "oos_total_net_pnl": total_oos,
        "oos_expectancy": float(row["expectancy"]),
        "oos_profit_factor": float(row["profit_factor"]),
        "oos_mean_edge_vs_random": float(row["mean_edge_vs_random"]),
        "oos_mean_edge_vs_continuation": float(row["mean_edge_vs_continuation"]),
        "oos_ambiguous_rate": float(row["ambiguous_rate"]),
        "one_month_dominant": bool(one_month_dominant),
    }])
    return verdict_df, label, recommendation


def markdown_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df.empty:
        return "No rows.\n"
    rows = df.head(max_rows).copy()
    cols = list(rows.columns)
    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(fmt(row[col]) for col in cols) + " |" for _, row in rows.iterrows()]
    return "\n".join([header, separator, *body]) + "\n"


def write_summary(
    path: Path,
    cfg: Config,
    warnings: list[str],
    by_symbol_session_target: pd.DataFrame,
    by_symbol: pd.DataFrame,
    by_session: pd.DataFrame,
    by_target: pd.DataFrame,
    by_month: pd.DataFrame,
    wf: pd.DataFrame,
    verdict_df: pd.DataFrame,
) -> None:
    oos = wf[wf["segment"] == "test_oos"].copy()
    candidates = oos[oos["symbol"] != "ALL"].copy()
    best_expectancy = candidates.sort_values(["expectancy", "trades"], ascending=[False, False]) if not candidates.empty else candidates
    best_random = candidates.sort_values(["mean_edge_vs_random", "trades"], ascending=[False, False]) if not candidates.empty else candidates
    best_cont = candidates.sort_values(["mean_edge_vs_continuation", "trades"], ascending=[False, False]) if not candidates.empty else candidates
    worst_tail = by_symbol_session_target.sort_values(["p05_net_pnl", "max_drawdown_pips"], ascending=[True, True]) if not by_symbol_session_target.empty else by_symbol_session_target

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Daily Box Session-Open Sweep/Reclaim Research Summary\n\n")
        handle.write("## Research warning\n")
        handle.write("Research only. This is not trading guidance and contains no EA, live trading, MT4/MT5 execution, lot sizing, or account-equity calculations.\n\n")
        handle.write("## Method definition\n")
        handle.write("- The box is built from the previous completed trading date only: previous-day high, low, midpoint, Q1, and Q3.\n")
        handle.write("- The scan starts only from the relevant configured session open for each symbol/session.\n")
        handle.write("- Execution uses M15 candles only for sweep/reclaim detection and trade management.\n")
        handle.write("- This is previous-daily-box + session-open scan research, not daily-candle reversion and not all-hours pip reversion.\n\n")
        handle.write("## Session mapping\n")
        session_rows = [{"symbol": symbol, "sessions": ", ".join(sessions)} for symbol, sessions in SYMBOL_SESSION_MAP.items()]
        handle.write(markdown_table(pd.DataFrame(session_rows), 20) + "\n")
        handle.write("## Session open hours (server/data hour)\n")
        handle.write(markdown_table(pd.DataFrame([{"session_name": k, "hour": v} for k, v in cfg.session_open_hours.items()]), 10) + "\n")
        handle.write("## Cost assumptions\n")
        cost_rows = []
        for family, profiles in COST_PROFILES.items():
            values = profiles[cfg.cost_profile]
            cost_rows.append({"pair_family": family, "cost_profile": cfg.cost_profile, **values, "friction_pips": sum(values.values())})
        handle.write(markdown_table(pd.DataFrame(cost_rows), 10) + "\n")
        handle.write("## Target modes\n")
        handle.write("| target_mode | Long TP | Short TP |\n|---|---|---|\n| mid | Box_Mid | Box_Mid |\n| far_quartile | Box_Q3 | Box_Q1 |\n| opposite_edge | Box_Top | Box_Bottom |\n\n")
        display_cols = ["symbol", "session_name", "target_mode", "trades", "expectancy", "profit_factor", "total_net_pnl", "mean_edge_vs_random", "mean_edge_vs_continuation", "max_drawdown_pips"]
        handle.write("## Best candidate rows by OOS expectancy\n")
        handle.write(markdown_table(best_expectancy[[c for c in display_cols if c in best_expectancy.columns]], 10) + "\n")
        handle.write("## Best candidate rows by OOS edge vs random\n")
        handle.write(markdown_table(best_random[[c for c in display_cols if c in best_random.columns]], 10) + "\n")
        handle.write("## Best candidate rows by OOS edge vs continuation\n")
        handle.write(markdown_table(best_cont[[c for c in display_cols if c in best_cont.columns]], 10) + "\n")
        handle.write("## Worst tail-risk rows\n")
        tail_cols = ["symbol", "session_name", "target_mode", "trades", "p05_net_pnl", "worst_trade", "max_drawdown_pips", "longest_loss_streak", "ambiguous_rate"]
        handle.write(markdown_table(worst_tail[[c for c in tail_cols if c in worst_tail.columns]], 10) + "\n")
        handle.write("## Symbol/session summary\n")
        handle.write(markdown_table(by_symbol_session_target, 20) + "\n")
        handle.write("## Symbol summary\n")
        handle.write(markdown_table(by_symbol, 20) + "\n")
        handle.write("## Session summary\n")
        handle.write(markdown_table(by_session, 20) + "\n")
        handle.write("## Target-mode summary\n")
        handle.write(markdown_table(by_target, 20) + "\n")
        handle.write("## Walk-forward interpretation\n")
        wf_cols = ["segment", "from_year", "to_year", "symbol", "session_name", "target_mode", "trades", "expectancy", "profit_factor", "total_net_pnl", "mean_edge_vs_random", "mean_edge_vs_continuation"]
        handle.write(markdown_table(wf[[c for c in wf_cols if c in wf.columns]], 30) + "\n")
        handle.write("## Final conclusion\n")
        verdict_row = verdict_df.iloc[0].to_dict()
        handle.write(f"**{verdict_row['verdict']}** — {verdict_row['reason']}\n\n")
        handle.write(f"Recommendation: **{verdict_row['recommendation']}**.\n\n")
        if warnings:
            handle.write("## Warnings / audit notes\n")
            for warning in warnings[:100]:
                handle.write(f"- {warning}\n")
            if len(warnings) > 100:
                handle.write(f"- {len(warnings) - 100} additional warnings suppressed.\n")


def parse_session_open_hours(value: str | None) -> dict[str, int]:
    hours = dict(DEFAULT_SESSION_OPEN_HOURS)
    if not value:
        return hours
    candidate = Path(value)
    if candidate.exists():
        loaded = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        loaded = json.loads(value)
    for key, hour in loaded.items():
        if key not in hours:
            raise ValueError(f"Unknown session name in session-open-hours: {key}")
        hour_int = int(hour)
        if hour_int < 0 or hour_int > 23:
            raise ValueError(f"Invalid hour for {key}: {hour}")
        hours[key] = hour_int
    return hours


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Research-only Daily Box Session-Open Sweep/Reclaim strategy study.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="reports/daily_box_session_open")
    parser.add_argument("--cost-profile", choices=["low", "conservative", "high"], default="conservative")
    parser.add_argument("--scan-window-bars", type=int, default=16)
    parser.add_argument("--force-close-hour", type=int, default=23)
    parser.add_argument("--force-close-minute", type=int, default=45)
    parser.add_argument("--buffer-pips", type=float, default=0)
    parser.add_argument("--risk-reward-min", type=float, default=1.0)
    parser.add_argument("--session-open-hours", default=None, help="JSON string or JSON file path overriding default session open hours.")
    parser.add_argument("--min-oos-observations", type=int, default=50)
    args = parser.parse_args()
    if args.scan_window_bars <= 0:
        raise ValueError("--scan-window-bars must be positive")
    return Config(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        cost_profile=args.cost_profile,
        scan_window_bars=args.scan_window_bars,
        force_close_hour=args.force_close_hour,
        force_close_minute=args.force_close_minute,
        buffer_pips=args.buffer_pips,
        risk_reward_min=args.risk_reward_min,
        session_open_hours=parse_session_open_hours(args.session_open_hours),
        min_oos_observations=args.min_oos_observations,
    )


def run_research(cfg: Config) -> dict[str, Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    all_events: list[pd.DataFrame] = []
    for symbol, filename in DATASETS.items():
        path = cfg.data_dir / filename
        if not path.exists():
            warnings.append(f"{symbol}: missing dataset {path}; skipped.")
            continue
        events, symbol_warnings = scan_symbol(symbol, path, cfg)
        all_events.append(events)
        warnings.extend(symbol_warnings)
    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame(columns=EVENT_COLUMNS)
    if not events_df.empty:
        events_df = events_df.sort_values(["confirmation_datetime", "symbol", "session_name", "target_mode"]).reset_index(drop=True)

    by_symbol_session_target = aggregate(events_df, ["symbol", "session_name", "target_mode"])
    by_symbol = aggregate(events_df, ["symbol"])
    by_session = aggregate(events_df, ["session_name"])
    by_target = aggregate(events_df, ["target_mode"])
    by_year = aggregate(events_df, ["year"])
    by_month = aggregate(events_df, ["month"])
    wf = walkforward(events_df)
    verdict_df, _, _ = verdict(wf, by_month, cfg)

    outputs = {
        "events": cfg.output_dir / "daily_box_session_events.csv",
        "by_symbol_session_target": cfg.output_dir / "daily_box_session_by_symbol_session_target.csv",
        "by_symbol": cfg.output_dir / "daily_box_session_by_symbol.csv",
        "by_session": cfg.output_dir / "daily_box_session_by_session.csv",
        "by_target": cfg.output_dir / "daily_box_session_by_target.csv",
        "by_year": cfg.output_dir / "daily_box_session_by_year.csv",
        "by_month": cfg.output_dir / "daily_box_session_by_month.csv",
        "walkforward": cfg.output_dir / "daily_box_session_walkforward.csv",
        "verdict": cfg.output_dir / "daily_box_session_verdict.csv",
        "summary": cfg.output_dir / "daily_box_session_summary.md",
    }
    events_df.to_csv(outputs["events"], index=False)
    by_symbol_session_target.to_csv(outputs["by_symbol_session_target"], index=False)
    by_symbol.to_csv(outputs["by_symbol"], index=False)
    by_session.to_csv(outputs["by_session"], index=False)
    by_target.to_csv(outputs["by_target"], index=False)
    by_year.to_csv(outputs["by_year"], index=False)
    by_month.to_csv(outputs["by_month"], index=False)
    wf.to_csv(outputs["walkforward"], index=False)
    verdict_df.to_csv(outputs["verdict"], index=False)
    write_summary(outputs["summary"], cfg, warnings, by_symbol_session_target, by_symbol, by_session, by_target, by_month, wf, verdict_df)
    return outputs


def main() -> None:
    cfg = parse_args()
    outputs = run_research(cfg)
    print("Research-only Daily Box Session-Open study complete.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
