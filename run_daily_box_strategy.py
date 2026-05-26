#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data_loader import load_ohlc_csv


PIP_SIZE_FALLBACK = 0.0001
PIP_SIZE_JPY = 0.01


@dataclass(frozen=True)
class VariantConfig:
    name: str
    require_sweep_close: bool
    require_ema200: bool


VARIANTS = (
    VariantConfig("basic", require_sweep_close=False, require_ema200=False),
    VariantConfig("sweep", require_sweep_close=True, require_ema200=False),
    VariantConfig("sweep_ema200", require_sweep_close=True, require_ema200=True),
)


@dataclass
class Config:
    data_dir: Path
    output_dir: Path
    buffer_pips: float = 1.0
    spread_pips: float = 0.5
    slippage_pips: float = 0.1
    max_confirmation_bars: int = 6
    rr_min: float = 1.5


def infer_pip_size(symbol: str) -> float:
    return PIP_SIZE_JPY if "JPY" in symbol.upper() else PIP_SIZE_FALLBACK


def prepare_frames(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.rename(columns={"datetime": "timestamp"}).copy().set_index("timestamp")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    m5 = data.resample("5min").agg(agg).dropna().reset_index()
    daily = data.resample("1D").agg(agg).dropna().reset_index()
    daily["prev_high"] = daily["high"].shift(1)
    daily["prev_low"] = daily["low"].shift(1)
    daily = daily.dropna(subset=["prev_high", "prev_low"]).copy()

    m5["date"] = m5["timestamp"].dt.date
    daily["date"] = daily["timestamp"].dt.date

    out = m5.merge(daily[["date", "prev_high", "prev_low"]], on="date", how="left")
    out = out.dropna(subset=["prev_high", "prev_low"]).copy()
    out["box_top"] = out["prev_high"]
    out["box_bottom"] = out["prev_low"]
    out["box_mid"] = (out["box_top"] + out["box_bottom"]) / 2.0
    out["box_range"] = out["box_top"] - out["box_bottom"]
    out["ema200"] = out["close"].ewm(span=200, adjust=False).mean()
    return out.reset_index(drop=True)


def find_last_inside_candle(df: pd.DataFrame, idx: int, side: str) -> Optional[pd.Series]:
    if idx <= 0:
        return None
    pre = df.iloc[:idx]
    inside = (pre["close"] > pre["box_bottom"]) & (pre["close"] < pre["box_top"])
    if side == "long":
        mask = inside & (pre["close"] > pre["open"])
    else:
        mask = inside & (pre["close"] < pre["open"])
    rows = pre[mask]
    return None if rows.empty else rows.iloc[-1]


def backtest_symbol_variant(df: pd.DataFrame, symbol: str, cfg: Config, variant: VariantConfig) -> list[dict]:
    pip_size = infer_pip_size(symbol)
    buffer_px = cfg.buffer_pips * pip_size
    spread_px = cfg.spread_pips * pip_size
    slip_px = cfg.slippage_pips * pip_size

    trades: list[dict] = []
    traded_dates: set = set()

    i = 0
    n = len(df)
    while i < n:
        row = df.iloc[i]
        day = row["date"]
        if day in traded_dates:
            i += 1
            continue

        long_trigger = row["low"] <= row["box_bottom"]
        short_trigger = row["high"] >= row["box_top"]

        if variant.require_sweep_close:
            long_trigger = long_trigger and (row["close"] > row["box_bottom"])
            short_trigger = short_trigger and (row["close"] < row["box_top"])

        if not (long_trigger or short_trigger):
            i += 1
            continue

        side = "long" if long_trigger else "short"
        ref = find_last_inside_candle(df, i, side)
        if ref is None:
            i += 1
            continue

        conf_end = min(i + cfg.max_confirmation_bars, n - 1)
        entry_idx = None
        for j in range(i, conf_end + 1):
            c = df.iloc[j]
            if side == "long" and c["close"] > ref["high"]:
                entry_idx = j
                break
            if side == "short" and c["close"] < ref["low"]:
                entry_idx = j
                break

        if entry_idx is None:
            i = conf_end + 1
            continue

        e = df.iloc[entry_idx]
        if variant.require_ema200:
            if side == "long" and not (e["close"] > e["ema200"]):
                i = entry_idx + 1
                continue
            if side == "short" and not (e["close"] < e["ema200"]):
                i = entry_idx + 1
                continue

        if side == "long":
            entry_fill = e["close"] + (spread_px / 2.0) + slip_px
            sl = e["box_bottom"] - buffer_px
            tp = e["box_top"]
            risk = entry_fill - sl
            reward = tp - entry_fill
        else:
            entry_fill = e["close"] - (spread_px / 2.0) - slip_px
            sl = e["box_top"] + buffer_px
            tp = e["box_bottom"]
            risk = sl - entry_fill
            reward = entry_fill - tp

        if risk <= 0 or reward <= 0 or (reward / risk) < cfg.rr_min:
            i = entry_idx + 1
            continue

        day_slice = df[(df["date"] == day) & (df.index >= entry_idx)]
        force_cutoff = pd.Timestamp(f"{day} 23:45:00")
        exit_time = None
        exit_price = None
        exit_reason = None

        for _, bar in day_slice.iterrows():
            if bar["timestamp"] > force_cutoff:
                break
            hit_sl = bar["low"] <= sl if side == "long" else bar["high"] >= sl
            hit_tp = bar["high"] >= tp if side == "long" else bar["low"] <= tp
            if hit_sl and hit_tp:
                hit_tp = False
            if hit_sl:
                exit_time = bar["timestamp"]
                exit_price = sl - slip_px if side == "long" else sl + slip_px
                exit_reason = "SL"
                break
            if hit_tp:
                exit_time = bar["timestamp"]
                exit_price = tp - slip_px if side == "long" else tp + slip_px
                exit_reason = "TP"
                break

        if exit_time is None:
            cutoff_rows = day_slice[day_slice["timestamp"] <= force_cutoff]
            if cutoff_rows.empty:
                i = entry_idx + 1
                continue
            last = cutoff_rows.iloc[-1]
            exit_time = last["timestamp"]
            exit_price = last["close"] - (spread_px / 2.0) - slip_px if side == "long" else last["close"] + (spread_px / 2.0) + slip_px
            exit_reason = "EOD_2345"

        r_result = ((exit_price - entry_fill) if side == "long" else (entry_fill - exit_price)) / risk

        trades.append(
            {
                "variant": variant.name,
                "symbol": symbol,
                "date": str(day),
                "side": side,
                "entry_time": e["timestamp"],
                "entry_price": round(float(entry_fill), 8),
                "sl": round(float(sl), 8),
                "tp": round(float(tp), 8),
                "exit_time": exit_time,
                "exit_price": round(float(exit_price), 8),
                "exit_reason": exit_reason,
                "rr_at_entry": round(float(reward / risk), 4),
                "r_result": round(float(r_result), 4),
            }
        )
        traded_dates.add(day)
        i = entry_idx + 1

    return trades


def compute_summary_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "net_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "average_r": 0.0,
            "max_consecutive_losses": 0,
        }

    wins = df[df["r_result"] > 0]["r_result"].sum()
    losses = df[df["r_result"] < 0]["r_result"].sum()
    profit_factor = wins / abs(losses) if losses != 0 else np.inf

    streak = 0
    max_streak = 0
    for r in df.sort_values("entry_time")["r_result"]:
        if r < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    eq = df.sort_values("entry_time")["r_result"].cumsum()
    dd = eq - eq.cummax()

    return {
        "total_trades": int(len(df)),
        "win_rate": round(float((df["r_result"] > 0).mean()), 4),
        "net_r": round(float(df["r_result"].sum()), 4),
        "profit_factor": round(float(profit_factor), 4) if np.isfinite(profit_factor) else np.inf,
        "max_drawdown_r": round(float(dd.min()), 4),
        "average_r": round(float(df["r_result"].mean()), 4),
        "max_consecutive_losses": int(max_streak),
    }


def run(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    trades: list[dict] = []

    for csv_path in sorted(cfg.data_dir.glob("*.csv")):
        symbol = csv_path.stem
        raw, _ = load_ohlc_csv(str(csv_path))
        prepared = prepare_frames(raw)
        if prepared.empty:
            continue
        for variant in VARIANTS:
            trades.extend(backtest_symbol_variant(prepared, symbol, cfg, variant))

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=["variant", "symbol", "date", "side", "entry_time", "entry_price", "sl", "tp", "exit_time", "exit_price", "exit_reason", "rr_at_entry", "r_result"])

    trades_df = trades_df.sort_values(["entry_time", "symbol", "variant"]).reset_index(drop=True)
    trades_df.to_csv(cfg.output_dir / "trades.csv", index=False)

    metrics_overall = compute_summary_metrics(trades_df)
    summary_rows = [{"scope": "overall", "name": "all", **metrics_overall}]

    if not trades_df.empty:
        for variant, grp in trades_df.groupby("variant"):
            summary_rows.append({"scope": "variant", "name": variant, **compute_summary_metrics(grp)})
        for symbol, grp in trades_df.groupby("symbol"):
            summary_rows.append({"scope": "symbol", "name": symbol, **compute_summary_metrics(grp)})

    pd.DataFrame(summary_rows).to_csv(cfg.output_dir / "summary.csv", index=False)

    if trades_df.empty:
        monthly = pd.DataFrame(columns=["month", "variant", "symbol", "total_trades", "win_rate", "net_r", "average_r"])
        eq = pd.DataFrame(columns=["entry_time", "variant", "symbol", "r_result", "equity_r"])
    else:
        trades_df["month"] = pd.to_datetime(trades_df["entry_time"]).dt.to_period("M").astype(str)
        monthly = (
            trades_df.groupby(["month", "variant", "symbol"], as_index=False)
            .agg(total_trades=("r_result", "count"), win_rate=("r_result", lambda s: (s > 0).mean()), net_r=("r_result", "sum"), average_r=("r_result", "mean"))
        )
        monthly[["win_rate", "net_r", "average_r"]] = monthly[["win_rate", "net_r", "average_r"]].round(4)

        eq = trades_df[["entry_time", "variant", "symbol", "r_result"]].copy()
        eq["equity_r"] = eq["r_result"].cumsum()

    monthly.to_csv(cfg.output_dir / "monthly_results.csv", index=False)
    eq.to_csv(cfg.output_dir / "equity_curve.csv", index=False)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Daily Box Strategy isolated backtest")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("reports/daily_box_strategy"))
    p.add_argument("--buffer-pips", type=float, default=1.0)
    p.add_argument("--spread-pips", type=float, default=0.5)
    p.add_argument("--slippage-pips", type=float, default=0.1)
    p.add_argument("--max-confirmation-bars", type=int, default=6)
    p.add_argument("--rr-min", type=float, default=1.5)
    a = p.parse_args()
    return Config(
        data_dir=a.data_dir,
        output_dir=a.output_dir,
        buffer_pips=a.buffer_pips,
        spread_pips=a.spread_pips,
        slippage_pips=a.slippage_pips,
        max_confirmation_bars=a.max_confirmation_bars,
        rr_min=a.rr_min,
    )


if __name__ == "__main__":
    run(parse_args())
