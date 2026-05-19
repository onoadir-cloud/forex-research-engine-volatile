#!/usr/bin/env python3
"""Standalone EURUSD controlled basket mean-reversion research lab.

This script intentionally does not import or modify the existing pattern research
engine. It runs an independent parameter sweep for EURUSD M15 basket/layering
mean reversion and writes standalone reports under the requested output folder.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PIP_SIZE = 0.0001
BASE_SIZE = 1.0

MOVE_X_PIPS = [10, 15, 20, 25, 30]
LAYER_DISTANCE_PIPS = [10, 15, 20]
MAX_LAYERS = [2, 3, 4, 5]
LOT_MULTIPLIERS = [1.0, 1.2, 1.3]
GROUP_TP_PIPS = [5, 8, 10, 12]
MAX_HOLD_BARS = [40, 80, 160]
MAX_BASKET_ADVERSE_PIPS = [50, 75, 100, 150]
ANCHORS = ["rolling_16_close", "rolling_32_close", "daily_open"]
DIRECTIONS = ["LONG", "SHORT"]
PARAM_COLUMNS = [
    "anchor_type",
    "move_x_pips",
    "layer_distance_pips",
    "max_layers",
    "lot_multiplier",
    "group_tp_pips",
    "max_hold_bars",
    "max_basket_adverse_pips",
]
RESULT_COLUMNS = PARAM_COLUMNS + [
    "hour",
    "session_bucket",
    "direction",
    "baskets",
    "win_rate",
    "failure_rate",
    "timeout_rate",
    "total_net_weighted_pips",
    "avg_net_weighted_pips",
    "median_net_weighted_pips",
    "profit_factor",
    "max_losing_streak",
    "avg_layers_opened",
    "max_layers_opened",
    "avg_holding_bars",
    "max_holding_bars",
    "avg_max_floating_adverse_pips",
    "p95_max_floating_adverse_pips",
    "max_floating_adverse_pips",
    "avg_total_logical_size",
    "max_total_logical_size",
    "IS baskets",
    "OOS baskets",
    "IS total_net_weighted_pips",
    "OOS total_net_weighted_pips",
    "IS avg_net_weighted_pips",
    "OOS avg_net_weighted_pips",
    "OOS agrees with IS",
    "walk_forward_positive_windows",
    "walk_forward_total_windows",
    "verdict",
]
TRADE_COLUMNS = [
    "symbol",
    "anchor_type",
    "signal_datetime",
    "first_entry_datetime",
    "hour",
    "session_bucket",
    "direction",
    "anchor_price",
    "signal_close",
    "move_x_pips",
    "layer_distance_pips",
    "max_layers",
    "lot_multiplier",
    "group_tp_pips",
    "max_hold_bars",
    "max_basket_adverse_pips",
    "layers_opened",
    "total_logical_size",
    "weighted_avg_entry",
    "exit_datetime",
    "exit_reason",
    "gross_weighted_pips",
    "total_cost_pips",
    "net_weighted_pips_after_costs",
    "holding_bars",
    "max_floating_adverse_pips",
    "max_floating_favorable_pips",
    "max_total_logical_size",
]


@dataclass(frozen=True)
class BasketParams:
    anchor_type: str
    move_x_pips: int
    layer_distance_pips: int
    max_layers: int
    lot_multiplier: float
    group_tp_pips: int
    max_hold_bars: int
    max_basket_adverse_pips: int


@dataclass
class BasketTrade:
    symbol: str
    anchor_type: str
    signal_datetime: pd.Timestamp
    first_entry_datetime: pd.Timestamp
    hour: int
    session_bucket: str
    direction: str
    anchor_price: float
    signal_close: float
    move_x_pips: int
    layer_distance_pips: int
    max_layers: int
    lot_multiplier: float
    group_tp_pips: int
    max_hold_bars: int
    max_basket_adverse_pips: int
    layers_opened: int
    total_logical_size: float
    weighted_avg_entry: float
    exit_datetime: pd.Timestamp
    exit_reason: str
    gross_weighted_pips: float
    total_cost_pips: float
    net_weighted_pips_after_costs: float
    holding_bars: int
    max_floating_adverse_pips: float
    max_floating_favorable_pips: float
    max_total_logical_size: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EURUSD controlled basket mean-reversion lab")
    parser.add_argument("--csv", default="data/EURUSD_M15_MT5_5Y.csv")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--base-timeframe", default="M15")
    parser.add_argument("--spread-pips", type=float, default=1.0)
    parser.add_argument("--slippage-pips", type=float, default=0.3)
    parser.add_argument("--output-dir", default="eurusd_basket_reports")
    parser.add_argument("--preset", choices=["full", "quick"], default="full")
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


def load_data(csv_path: str, symbol: str, base_timeframe: str) -> pd.DataFrame:
    if symbol.upper() != "EURUSD":
        raise ValueError("This standalone lab is intentionally restricted to EURUSD only.")
    if base_timeframe.upper() != "M15":
        raise ValueError("This standalone lab is intentionally restricted to M15 data only.")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="raise", utc=False)
    elif "date" in df.columns and "time" in df.columns:
        dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="raise", utc=False)
    elif "time" in df.columns:
        dt = pd.to_datetime(df["time"], errors="raise", utc=False)
    else:
        raise ValueError("CSV must include datetime, time, or date+time columns")

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing required OHLC column: {col}")
        df[col] = pd.to_numeric(df[col], errors="raise")

    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)

    df["datetime"] = dt
    df = df[["datetime", "open", "high", "low", "close"]].sort_values("datetime").reset_index(drop=True)
    if df["datetime"].duplicated().any():
        raise ValueError("Duplicate timestamps detected in CSV")

    invalid_high = df["high"] < df[["open", "close", "low"]].max(axis=1)
    invalid_low = df["low"] > df[["open", "close", "high"]].min(axis=1)
    if invalid_high.any() or invalid_low.any():
        raise ValueError("Invalid OHLC relationships detected")

    df["symbol"] = symbol.upper()
    df["hour"] = df["datetime"].dt.hour
    df["session_bucket"] = df["hour"].map(session_bucket)
    df["trading_date"] = df["datetime"].dt.date
    df["rolling_16_close"] = df["close"].shift(16)
    df["rolling_32_close"] = df["close"].shift(32)
    df["daily_open"] = df.groupby("trading_date")["open"].transform("first")
    return df


def iter_params(preset: str = "full") -> Iterable[BasketParams]:
    if preset == "quick":
        anchors = ["daily_open", "rolling_16_close"]
        move_x_pips = [10, 15, 20]
        layer_distance_pips = [10, 15]
        max_layers = [2, 3]
        lot_multipliers = [1.0, 1.2]
        group_tp_pips = [5, 8, 10]
        max_hold_bars = [40, 80]
        max_basket_adverse_pips = [50, 75]
    else:
        anchors = ANCHORS
        move_x_pips = MOVE_X_PIPS
        layer_distance_pips = LAYER_DISTANCE_PIPS
        max_layers = MAX_LAYERS
        lot_multipliers = LOT_MULTIPLIERS
        group_tp_pips = GROUP_TP_PIPS
        max_hold_bars = MAX_HOLD_BARS
        max_basket_adverse_pips = MAX_BASKET_ADVERSE_PIPS

    for values in product(
        anchors,
        move_x_pips,
        layer_distance_pips,
        max_layers,
        lot_multipliers,
        group_tp_pips,
        max_hold_bars,
        max_basket_adverse_pips,
    ):
        yield BasketParams(*values)


def layer_size(lot_multiplier: float, layer_index: int) -> float:
    return BASE_SIZE * (lot_multiplier**layer_index)


def weighted_average(entries: Sequence[float], sizes: Sequence[float]) -> float:
    return float(np.average(np.asarray(entries, dtype=float), weights=np.asarray(sizes, dtype=float)))


def basket_pips(direction: str, exit_price: float, entries: Sequence[float], sizes: Sequence[float]) -> float:
    weighted_pips = 0.0
    for entry, size in zip(entries, sizes):
        if direction == "LONG":
            layer_pips = (exit_price - entry) / PIP_SIZE
        else:
            layer_pips = (entry - exit_price) / PIP_SIZE
        weighted_pips += float(size) * float(layer_pips)
    return weighted_pips


def floating_excursions(direction: str, high: float, low: float, entries: Sequence[float], sizes: Sequence[float]) -> Tuple[float, float]:
    avg_entry = weighted_average(entries, sizes)
    if direction == "LONG":
        adverse = max(0.0, (avg_entry - low) / PIP_SIZE)
        favorable = max(0.0, (high - avg_entry) / PIP_SIZE)
    else:
        adverse = max(0.0, (high - avg_entry) / PIP_SIZE)
        favorable = max(0.0, (avg_entry - low) / PIP_SIZE)
    return adverse, favorable


def simulate_basket(
    df: pd.DataFrame,
    signal_idx: int,
    direction: str,
    params: BasketParams,
    symbol: str,
    cost_pips: float,
) -> Optional[Tuple[BasketTrade, int]]:
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None

    signal_row = df.iloc[signal_idx]
    entry_row = df.iloc[entry_idx]
    entries = [float(entry_row["open"])]
    sizes = [layer_size(params.lot_multiplier, 0)]
    max_total_size = sum(sizes)
    max_adverse = 0.0
    max_favorable = 0.0
    exit_idx = entry_idx
    exit_price = entries[0]
    exit_reason = "timeout"

    last_idx = min(len(df) - 1, entry_idx + params.max_hold_bars - 1)
    for idx in range(entry_idx, last_idx + 1):
        row = df.iloc[idx]
        high = float(row["high"])
        low = float(row["low"])

        while len(entries) < params.max_layers:
            next_layer_price = entries[-1] - params.layer_distance_pips * PIP_SIZE if direction == "LONG" else entries[-1] + params.layer_distance_pips * PIP_SIZE
            layer_hit = low <= next_layer_price if direction == "LONG" else high >= next_layer_price
            if not layer_hit:
                break
            entries.append(float(next_layer_price))
            sizes.append(layer_size(params.lot_multiplier, len(entries) - 1))
            max_total_size = max(max_total_size, sum(sizes))

        avg_entry = weighted_average(entries, sizes)
        adverse, favorable = floating_excursions(direction, high, low, entries, sizes)
        max_adverse = max(max_adverse, adverse)
        max_favorable = max(max_favorable, favorable)

        if direction == "LONG":
            tp_price = avg_entry + params.group_tp_pips * PIP_SIZE
            failure_price = avg_entry - params.max_basket_adverse_pips * PIP_SIZE
            tp_hit = high >= tp_price
            failure_hit = low <= failure_price
        else:
            tp_price = avg_entry - params.group_tp_pips * PIP_SIZE
            failure_price = avg_entry + params.max_basket_adverse_pips * PIP_SIZE
            tp_hit = low <= tp_price
            failure_hit = high >= failure_price

        # Conservative same-bar handling: if the basket TP and adverse failure are
        # both reachable in one candle after the current basket exposure is set,
        # the adverse failure is booked first.
        if failure_hit:
            exit_reason = "adverse_failure"
            exit_price = failure_price
            exit_idx = idx
            break
        if tp_hit:
            exit_reason = "group_tp"
            exit_price = tp_price
            exit_idx = idx
            break

        if idx == last_idx:
            exit_reason = "timeout"
            exit_price = float(row["close"])
            exit_idx = idx

    total_size = float(sum(sizes))
    gross = basket_pips(direction, exit_price, entries, sizes)
    total_cost = float(cost_pips * total_size)
    net = gross - total_cost
    holding_bars = int(exit_idx - entry_idx + 1)
    trade = BasketTrade(
        symbol=symbol.upper(),
        anchor_type=params.anchor_type,
        signal_datetime=signal_row["datetime"],
        first_entry_datetime=entry_row["datetime"],
        hour=int(signal_row["hour"]),
        session_bucket=str(signal_row["session_bucket"]),
        direction=direction,
        anchor_price=float(signal_row[params.anchor_type]),
        signal_close=float(signal_row["close"]),
        move_x_pips=params.move_x_pips,
        layer_distance_pips=params.layer_distance_pips,
        max_layers=params.max_layers,
        lot_multiplier=params.lot_multiplier,
        group_tp_pips=params.group_tp_pips,
        max_hold_bars=params.max_hold_bars,
        max_basket_adverse_pips=params.max_basket_adverse_pips,
        layers_opened=len(entries),
        total_logical_size=total_size,
        weighted_avg_entry=weighted_average(entries, sizes),
        exit_datetime=df.iloc[exit_idx]["datetime"],
        exit_reason=exit_reason,
        gross_weighted_pips=gross,
        total_cost_pips=total_cost,
        net_weighted_pips_after_costs=net,
        holding_bars=holding_bars,
        max_floating_adverse_pips=max_adverse,
        max_floating_favorable_pips=max_favorable,
        max_total_logical_size=max_total_size,
    )
    return trade, exit_idx


def signal_direction(close: float, anchor: float, move_x_pips: int) -> Optional[str]:
    if not math.isfinite(anchor):
        return None
    distance_pips = (close - anchor) / PIP_SIZE
    if distance_pips >= move_x_pips:
        return "SHORT"
    if distance_pips <= -move_x_pips:
        return "LONG"
    return None


def reset_reached(direction: str, close: float, anchor: float) -> bool:
    if not math.isfinite(anchor):
        return False
    if direction == "SHORT":
        return close <= anchor
    return close >= anchor


def run_parameter_set(df: pd.DataFrame, params: BasketParams, symbol: str, cost_pips: float) -> List[BasketTrade]:
    trades: List[BasketTrade] = []

    # Symbol + anchor + direction groups are scanned independently. That enforces
    # no overlap within the same direction group while still allowing an opposite
    # direction basket from the same anchor to exist if its own signal/reset state
    # permits it.
    for target_direction in DIRECTIONS:
        allowed = True
        i = 0
        while i < len(df) - 1:
            row = df.iloc[i]
            anchor = float(row[params.anchor_type]) if pd.notna(row[params.anchor_type]) else np.nan
            close = float(row["close"])

            if not allowed and reset_reached(target_direction, close, anchor):
                allowed = True

            direction = signal_direction(close, anchor, params.move_x_pips)
            if direction != target_direction or not allowed:
                i += 1
                continue

            result = simulate_basket(df, i, target_direction, params, symbol, cost_pips)
            if result is None:
                break
            trade, exit_idx = result
            trades.append(trade)
            allowed = False
            i = max(exit_idx + 1, i + 1)

    return trades


def assign_splits(trades: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        trades["sample_split"] = pd.Series(dtype=str)
        trades["wf_window"] = pd.Series(dtype="Int64")
        return trades

    split_idx = int(len(df) * 0.70)
    split_ts = df.iloc[min(split_idx, len(df) - 1)]["datetime"]
    trades["sample_split"] = np.where(trades["signal_datetime"] < split_ts, "IS", "OOS")

    n = len(df)
    boundaries = [df.iloc[int(n / 3)]["datetime"], df.iloc[int(2 * n / 3)]["datetime"]]
    trades["wf_window"] = np.select(
        [trades["signal_datetime"] < boundaries[0], trades["signal_datetime"] < boundaries[1]],
        [1, 2],
        default=3,
    )
    return trades


def max_losing_streak(values: Sequence[float]) -> int:
    best = 0
    current = 0
    for value in values:
        if value <= 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def profit_factor(values: pd.Series) -> float:
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    if losses == 0.0:
        return float("inf") if gains > 0.0 else 0.0
    return gains / losses


def verdict(row: Dict[str, object]) -> str:
    max_size_reasonable = float(row["max_total_logical_size"]) <= 10.0
    p95_not_extreme = float(row["p95_max_floating_adverse_pips"]) <= 200.0
    strong = (
        int(row["baskets"]) >= 300
        and float(row["total_net_weighted_pips"]) > 0
        and float(row["avg_net_weighted_pips"]) > 0
        and float(row["profit_factor"]) >= 1.15
        and float(row["OOS avg_net_weighted_pips"]) > 0
        and bool(row["OOS agrees with IS"])
        and int(row["walk_forward_positive_windows"]) >= 2
        and max_size_reasonable
        and p95_not_extreme
    )
    if strong:
        return "Strong Candidate"
    candidate = (
        int(row["baskets"]) >= 200
        and float(row["total_net_weighted_pips"]) > 0
        and float(row["avg_net_weighted_pips"]) > 0
        and float(row["OOS avg_net_weighted_pips"]) > 0
        and int(row["walk_forward_positive_windows"]) >= 2
    )
    return "Candidate" if candidate else "Reject"


def summarize_group(group: pd.DataFrame, keys: Dict[str, object]) -> Dict[str, object]:
    values = group["net_weighted_pips_after_costs"]
    is_group = group[group["sample_split"] == "IS"]
    oos_group = group[group["sample_split"] == "OOS"]
    wf = group.groupby("wf_window")["net_weighted_pips_after_costs"].sum()
    is_avg = float(is_group["net_weighted_pips_after_costs"].mean()) if not is_group.empty else 0.0
    oos_avg = float(oos_group["net_weighted_pips_after_costs"].mean()) if not oos_group.empty else 0.0
    row: Dict[str, object] = dict(keys)
    row.update(
        {
            "baskets": int(len(group)),
            "win_rate": float((values > 0).mean()) if len(group) else 0.0,
            "failure_rate": float((group["exit_reason"] == "adverse_failure").mean()) if len(group) else 0.0,
            "timeout_rate": float((group["exit_reason"] == "timeout").mean()) if len(group) else 0.0,
            "total_net_weighted_pips": float(values.sum()),
            "avg_net_weighted_pips": float(values.mean()) if len(group) else 0.0,
            "median_net_weighted_pips": float(values.median()) if len(group) else 0.0,
            "profit_factor": profit_factor(values),
            "max_losing_streak": max_losing_streak(values.tolist()),
            "avg_layers_opened": float(group["layers_opened"].mean()) if len(group) else 0.0,
            "max_layers_opened": int(group["layers_opened"].max()) if len(group) else 0,
            "avg_holding_bars": float(group["holding_bars"].mean()) if len(group) else 0.0,
            "max_holding_bars": int(group["holding_bars"].max()) if len(group) else 0,
            "avg_max_floating_adverse_pips": float(group["max_floating_adverse_pips"].mean()) if len(group) else 0.0,
            "p95_max_floating_adverse_pips": float(group["max_floating_adverse_pips"].quantile(0.95)) if len(group) else 0.0,
            "max_floating_adverse_pips": float(group["max_floating_adverse_pips"].max()) if len(group) else 0.0,
            "avg_total_logical_size": float(group["total_logical_size"].mean()) if len(group) else 0.0,
            "max_total_logical_size": float(group["max_total_logical_size"].max()) if len(group) else 0.0,
            "IS baskets": int(len(is_group)),
            "OOS baskets": int(len(oos_group)),
            "IS total_net_weighted_pips": float(is_group["net_weighted_pips_after_costs"].sum()) if not is_group.empty else 0.0,
            "OOS total_net_weighted_pips": float(oos_group["net_weighted_pips_after_costs"].sum()) if not oos_group.empty else 0.0,
            "IS avg_net_weighted_pips": is_avg,
            "OOS avg_net_weighted_pips": oos_avg,
            "OOS agrees with IS": bool(is_avg > 0 and oos_avg > 0),
            "walk_forward_positive_windows": int((wf > 0).sum()),
            "walk_forward_total_windows": int(group["wf_window"].nunique()),
        }
    )
    row["verdict"] = verdict(row)
    return row


def verdict_rank(value: str) -> int:
    return {"Strong Candidate": 0, "Candidate": 1, "Reject": 2}.get(value, 3)


def aggregate_results(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    rows: List[Dict[str, object]] = []
    group_cols = PARAM_COLUMNS + ["hour", "session_bucket", "direction"]
    for keys, group in trades.groupby(group_cols, sort=False):
        rows.append(summarize_group(group, dict(zip(group_cols, keys))))

    combined_cols = PARAM_COLUMNS + ["hour", "session_bucket"]
    for keys, group in trades.groupby(combined_cols, sort=False):
        row_keys = dict(zip(combined_cols, keys))
        row_keys["direction"] = "ALL"
        rows.append(summarize_group(group, row_keys))

    results = pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
    results["_verdict_rank"] = results["verdict"].map(verdict_rank)
    results = results.sort_values(["_verdict_rank", "total_net_weighted_pips", "avg_net_weighted_pips"], ascending=[True, False, False])
    return results.drop(columns=["_verdict_rank"])


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 20) -> str:
    if df.empty:
        return "No rows.\n"
    clipped = df.loc[:, columns].head(max_rows).copy()
    formatted = clipped.astype(object)
    for col in clipped.columns:
        if pd.api.types.is_float_dtype(clipped[col]):
            formatted[col] = clipped[col].map(lambda x: "inf" if np.isinf(x) else f"{x:.2f}")
        else:
            formatted[col] = clipped[col].map(str)
    widths = {col: max(len(str(col)), *(len(str(v)) for v in formatted[col].tolist())) for col in formatted.columns}
    header = "| " + " | ".join(str(col).ljust(widths[col]) for col in formatted.columns) + " |"
    separator = "| " + " | ".join("-" * widths[col] for col in formatted.columns) + " |"
    body = ["| " + " | ".join(str(row[col]).ljust(widths[col]) for col in formatted.columns) + " |" for _, row in formatted.iterrows()]
    return "\n".join([header, separator, *body]) + "\n"


def best_by(results: pd.DataFrame, by_col: str) -> pd.DataFrame:
    if results.empty:
        return results
    base = results[results["direction"] == "ALL"].copy()
    if base.empty:
        base = results.copy()
    idx = base.sort_values([by_col, "total_net_weighted_pips"], ascending=[True, False]).groupby(by_col, sort=False).head(1).index
    return base.loc[idx].sort_values(by_col)


def comparison_table(trades: pd.DataFrame, col: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for value, group in trades.groupby(col, sort=False):
        rows.append(
            {
                col: value,
                "baskets": len(group),
                "total_net_weighted_pips": group["net_weighted_pips_after_costs"].sum(),
                "avg_net_weighted_pips": group["net_weighted_pips_after_costs"].mean(),
                "profit_factor": profit_factor(group["net_weighted_pips_after_costs"]),
                "avg_max_floating_adverse_pips": group["max_floating_adverse_pips"].mean(),
                "max_total_logical_size": group["max_total_logical_size"].max(),
            }
        )
    return pd.DataFrame(rows).sort_values("total_net_weighted_pips", ascending=False)


def write_best_json(results: pd.DataFrame, path: Path) -> None:
    usable = results[results["verdict"].isin(["Strong Candidate", "Candidate"])].copy()
    if usable.empty:
        usable = results.copy()
    usable["_verdict_rank"] = usable["verdict"].map(verdict_rank)
    best = usable.sort_values(["_verdict_rank", "total_net_weighted_pips", "avg_net_weighted_pips"], ascending=[True, False, False]).head(1).drop(columns=["_verdict_rank"], errors="ignore")
    payload = {
        "note": "Best row is selected from non-rejected candidates when available; otherwise from all tested rows. Results are empirical and not fabricated.",
        "best": {} if best.empty else best.replace({np.inf: "Infinity", -np.inf: "-Infinity"}).iloc[0].to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_summary(trades: pd.DataFrame, results: pd.DataFrame, path: Path, output_paths: Dict[str, Path], args: argparse.Namespace) -> None:
    key_cols = [
        "anchor_type",
        "move_x_pips",
        "layer_distance_pips",
        "max_layers",
        "lot_multiplier",
        "group_tp_pips",
        "max_hold_bars",
        "max_basket_adverse_pips",
        "hour",
        "session_bucket",
        "direction",
        "baskets",
        "total_net_weighted_pips",
        "avg_net_weighted_pips",
        "profit_factor",
        "OOS avg_net_weighted_pips",
        "walk_forward_positive_windows",
        "p95_max_floating_adverse_pips",
        "max_total_logical_size",
        "verdict",
    ]
    lines = [
        f"# {args.symbol.upper()} Controlled Basket Mean-Reversion Lab",
        "",
        "Standalone EURUSD M15 basket/layering research output. This lab does not use the existing pattern research engine.",
        "",
        "## Run Configuration",
        f"- CSV: `{args.csv}`",
        f"- Symbol: `{args.symbol.upper()}`",
        f"- Base timeframe: `{args.base_timeframe.upper()}`",
        f"- Spread pips: `{args.spread_pips}`",
        f"- Slippage pips: `{args.slippage_pips}`",
        f"- Total baskets simulated: `{len(trades)}`",
        "",
        "## Output Files",
        *[f"- `{p}`" for p in output_paths.values()],
        "",
        "## Top 20 Overall Candidates by Total Net Weighted Pips",
        markdown_table(results.sort_values("total_net_weighted_pips", ascending=False), key_cols, 20),
        "## Top 20 by Average Net Weighted Pips (baskets >= 300)",
        markdown_table(results[results["baskets"] >= 300].sort_values("avg_net_weighted_pips", ascending=False), key_cols, 20),
        "## Best Candidates by Exact Hour",
        markdown_table(best_by(results, "hour"), key_cols, 24),
        "## Best Candidates by Session Bucket",
        markdown_table(best_by(results, "session_bucket"), key_cols, 20),
        "## Comparison of Anchors",
        markdown_table(comparison_table(trades, "anchor_type"), ["anchor_type", "baskets", "total_net_weighted_pips", "avg_net_weighted_pips", "profit_factor", "avg_max_floating_adverse_pips", "max_total_logical_size"], 20),
        "## Comparison of LONG vs SHORT",
        markdown_table(comparison_table(trades, "direction"), ["direction", "baskets", "total_net_weighted_pips", "avg_net_weighted_pips", "profit_factor", "avg_max_floating_adverse_pips", "max_total_logical_size"], 20),
        "## Basket Exposure Warning",
        "This is a basket/layering strategy. Closed-profit metrics alone are not enough: each candidate must be judged by floating adverse exposure, layer count, total logical size, p95 adverse excursion, and worst observed adverse excursion. Conservative same-bar handling books adverse failure before group TP when both are touched in the same candle.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cost_pips = float(args.spread_pips + args.slippage_pips)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csv, args.symbol, args.base_timeframe)
    all_trades: List[BasketTrade] = []
    params_list = list(iter_params(args.preset))
    print(f"Selected preset: {args.preset}")
    print(f"Total parameter sets: {len(params_list)}")
    for n, params in enumerate(params_list, start=1):
        if n == 1 or n % 1000 == 0:
            print(f"Simulating parameter set {n}/{len(params_list)}...")
        all_trades.extend(run_parameter_set(df, params, args.symbol, cost_pips))

    trades_df = pd.DataFrame([asdict(t) for t in all_trades], columns=TRADE_COLUMNS)
    if not trades_df.empty:
        trades_df = assign_splits(trades_df, df)
        trades_df = trades_df.sort_values(["signal_datetime", "anchor_type", "direction"]).reset_index(drop=True)
    else:
        trades_df["sample_split"] = pd.Series(dtype=str)
        trades_df["wf_window"] = pd.Series(dtype="Int64")

    results_df = aggregate_results(trades_df)

    symbol = args.symbol.upper()
    output_paths = {
        "trades": output_dir / f"{symbol}_basket_trades.csv",
        "results": output_dir / f"{symbol}_basket_results.csv",
        "best": output_dir / f"{symbol}_basket_best.json",
        "summary": output_dir / f"{symbol}_basket_summary.md",
    }
    trades_df.to_csv(output_paths["trades"], index=False)
    results_df.to_csv(output_paths["results"], index=False)
    write_best_json(results_df, output_paths["best"])
    write_summary(trades_df, results_df, output_paths["summary"], output_paths, args)

    print("Wrote outputs:")
    for path in output_paths.values():
        print(f"- {path}")


if __name__ == "__main__":
    main()
