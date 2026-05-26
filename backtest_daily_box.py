#!/usr/bin/env python3
"""Simple CSV-based backtester for the Daily Box Strategy.

Expected input CSV columns (case-insensitive):
- timestamp (or time/date/datetime)
- open, high, low, close
- optional: volume

Timestamp should be parseable by pandas and represent server time.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PIP_SIZE_FALLBACK = 0.0001
PIP_SIZE_JPY = 0.01


@dataclass
class Config:
    input_dir: Path
    output_dir: Path
    buffer_pips: float = 1.0
    spread_pips: float = 0.5
    slippage_pips: float = 0.1
    commission_per_lot: float = 0.0
    max_confirmation_bars: int = 6
    rr_min: float = 1.5
    use_ema_filter: bool = True
    use_liquidity_sweep_filter: bool = True


def infer_pip_size(symbol: str) -> float:
    return PIP_SIZE_JPY if "JPY" in symbol.upper() else PIP_SIZE_FALLBACK


def load_ohlcv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rename_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=rename_map)

    ts_col_candidates = ["timestamp", "time", "date", "datetime"]
    ts_col = next((c for c in ts_col_candidates if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"No timestamp column found in {csv_path}")

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns in {csv_path}: {missing}")

    df[ts_col] = pd.to_datetime(df[ts_col], utc=False)
    df = df.sort_values(ts_col).reset_index(drop=True)
    df = df.rename(columns={ts_col: "timestamp"})
    df = df[["timestamp", "open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])]
    return df


def to_5m_and_daily(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy().set_index("timestamp")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in data.columns:
        agg["volume"] = "sum"

    m5 = data.resample("5min").agg(agg).dropna().reset_index()
    daily = data.resample("1D").agg(agg).dropna().reset_index()
    daily["prev_high"] = daily["high"].shift(1)
    daily["prev_low"] = daily["low"].shift(1)
    daily = daily.dropna(subset=["prev_high", "prev_low"]).copy()
    daily["date"] = daily["timestamp"].dt.date
    m5["date"] = m5["timestamp"].dt.date
    return m5, daily[["date", "prev_high", "prev_low"]]


def build_daily_box_on_5m(m5: pd.DataFrame, daily_prev: pd.DataFrame) -> pd.DataFrame:
    out = m5.merge(daily_prev, on="date", how="left")
    out = out.dropna(subset=["prev_high", "prev_low"]).copy()
    out["box_top"] = out["prev_high"]
    out["box_bottom"] = out["prev_low"]
    out["box_mid"] = (out["box_top"] + out["box_bottom"]) / 2.0
    out["box_range"] = out["box_top"] - out["box_bottom"]
    out["ema200"] = out["close"].ewm(span=200, adjust=False).mean()
    return out


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
    if rows.empty:
        return None
    return rows.iloc[-1]


def backtest_symbol(df: pd.DataFrame, symbol: str, cfg: Config) -> list[dict]:
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
        d = row["date"]
        if d in traded_dates:
            i += 1
            continue

        long_trigger = row["low"] <= row["box_bottom"]
        short_trigger = row["high"] >= row["box_top"]

        if cfg.use_liquidity_sweep_filter:
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

        if cfg.use_ema_filter:
            if side == "long" and not (e["close"] > e["ema200"]):
                i = entry_idx + 1
                continue
            if side == "short" and not (e["close"] < e["ema200"]):
                i = entry_idx + 1
                continue

        if side == "long":
            entry_raw = e["close"]
            entry_fill = entry_raw + (spread_px / 2.0) + slip_px
            sl = e["box_bottom"] - buffer_px
            tp = e["box_top"]
            risk = entry_fill - sl
            reward = tp - entry_fill
        else:
            entry_raw = e["close"]
            entry_fill = entry_raw - (spread_px / 2.0) - slip_px
            sl = e["box_top"] + buffer_px
            tp = e["box_bottom"]
            risk = sl - entry_fill
            reward = entry_fill - tp

        if risk <= 0 or reward <= 0:
            i = entry_idx + 1
            continue

        rr = reward / risk
        if rr < cfg.rr_min:
            i = entry_idx + 1
            continue

        exit_time = None
        exit_price = None
        exit_reason = None
        r_multiple = None

        day_slice = df[(df["date"] == d) & (df.index >= entry_idx)]
        force_cutoff = pd.Timestamp(f"{d} 23:45:00")

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
            mid_close = last["close"]
            exit_price = mid_close - (spread_px / 2.0) - slip_px if side == "long" else mid_close + (spread_px / 2.0) + slip_px
            exit_reason = "EOD_2345"

        pnl = (exit_price - entry_fill) if side == "long" else (entry_fill - exit_price)
        r_multiple = pnl / risk

        if cfg.commission_per_lot:
            commission_in_price = cfg.commission_per_lot * pip_size
            r_multiple -= (2.0 * commission_in_price) / risk

        trades.append(
            {
                "symbol": symbol,
                "date": str(d),
                "side": side,
                "entry_time": e["timestamp"],
                "entry_price": round(float(entry_fill), 8),
                "sl": round(float(sl), 8),
                "tp": round(float(tp), 8),
                "exit_time": exit_time,
                "exit_price": round(float(exit_price), 8),
                "exit_reason": exit_reason,
                "risk_price": round(float(risk), 8),
                "reward_price": round(float(reward), 8),
                "rr_at_entry": round(float(rr), 4),
                "r_result": round(float(r_multiple), 4),
            }
        )
        traded_dates.add(d)
        i = entry_idx + 1

    return trades


def compute_outputs(trades_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if trades_df.empty:
        summary = pd.DataFrame(
            [{
                "total_trades": 0,
                "win_rate": 0.0,
                "net_profit_r": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_r": 0.0,
                "average_r": 0.0,
                "max_consecutive_losses": 0,
            }]
        )
        return summary, pd.DataFrame(), pd.DataFrame()

    trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)
    wins = trades_df[trades_df["r_result"] > 0]
    losses = trades_df[trades_df["r_result"] < 0]
    gross_profit = wins["r_result"].sum()
    gross_loss_abs = abs(losses["r_result"].sum())
    profit_factor = gross_profit / gross_loss_abs if gross_loss_abs > 0 else np.inf

    consec_losses = 0
    max_consec_losses = 0
    for r in trades_df["r_result"]:
        if r < 0:
            consec_losses += 1
            max_consec_losses = max(max_consec_losses, consec_losses)
        else:
            consec_losses = 0

    equity = trades_df["r_result"].cumsum()
    peak = equity.cummax()
    dd = equity - peak
    max_dd = dd.min()

    summary = pd.DataFrame(
        [{
            "total_trades": int(len(trades_df)),
            "win_rate": round(float((trades_df["r_result"] > 0).mean()), 4),
            "net_profit_r": round(float(trades_df["r_result"].sum()), 4),
            "profit_factor": round(float(profit_factor), 4) if np.isfinite(profit_factor) else np.inf,
            "max_drawdown_r": round(float(max_dd), 4),
            "average_r": round(float(trades_df["r_result"].mean()), 4),
            "max_consecutive_losses": int(max_consec_losses),
        }]
    )

    by_symbol = (
        trades_df.groupby("symbol", as_index=False)
        .agg(total_trades=("r_result", "count"), win_rate=("r_result", lambda s: (s > 0).mean()), net_profit_r=("r_result", "sum"), average_r=("r_result", "mean"))
    )
    by_symbol["win_rate"] = by_symbol["win_rate"].round(4)
    by_symbol["net_profit_r"] = by_symbol["net_profit_r"].round(4)
    by_symbol["average_r"] = by_symbol["average_r"].round(4)

    trades_df["month"] = pd.to_datetime(trades_df["entry_time"]).dt.to_period("M").astype(str)
    by_month = (
        trades_df.groupby(["month", "symbol"], as_index=False)
        .agg(total_trades=("r_result", "count"), win_rate=("r_result", lambda s: (s > 0).mean()), net_profit_r=("r_result", "sum"), average_r=("r_result", "mean"))
    )
    by_month["win_rate"] = by_month["win_rate"].round(4)
    by_month["net_profit_r"] = by_month["net_profit_r"].round(4)
    by_month["average_r"] = by_month["average_r"].round(4)

    equity_curve = trades_df[["entry_time", "symbol", "r_result"]].copy()
    equity_curve["equity_r"] = equity_curve["r_result"].cumsum()
    return summary, by_symbol, by_month, equity_curve


def run(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    trade_rows: list[dict] = []

    for csv_path in sorted(cfg.input_dir.glob("*.csv")):
        symbol = csv_path.stem
        raw = load_ohlcv(csv_path)
        m5, daily_prev = to_5m_and_daily(raw)
        if m5.empty or daily_prev.empty:
            continue
        prepared = build_daily_box_on_5m(m5, daily_prev)
        if prepared.empty:
            continue
        trade_rows.extend(backtest_symbol(prepared, symbol, cfg))

    trades = pd.DataFrame(trade_rows)
    if trades.empty:
        trades = pd.DataFrame(columns=["symbol", "date", "side", "entry_time", "entry_price", "sl", "tp", "exit_time", "exit_price", "exit_reason", "risk_price", "reward_price", "rr_at_entry", "r_result"])

    summary, by_symbol, by_month, equity_curve = compute_outputs(trades)

    trades.to_csv(cfg.output_dir / "trades.csv", index=False)
    summary.to_csv(cfg.output_dir / "summary.csv", index=False)
    by_month.to_csv(cfg.output_dir / "monthly_results.csv", index=False)
    equity_curve.to_csv(cfg.output_dir / "equity_curve.csv", index=False)

    if not by_symbol.empty:
        by_symbol.to_csv(cfg.output_dir / "results_by_symbol.csv", index=False)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Daily Box Strategy backtest from local OHLCV CSV files")
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--output-dir", default=Path("outputs/daily_box_backtest"), type=Path)
    p.add_argument("--buffer-pips", type=float, default=1.0)
    p.add_argument("--spread-pips", type=float, default=0.5)
    p.add_argument("--slippage-pips", type=float, default=0.1)
    p.add_argument("--commission-per-lot", type=float, default=0.0)
    p.add_argument("--max-confirmation-bars", type=int, default=6)
    p.add_argument("--rr-min", type=float, default=1.5)
    p.add_argument("--use-ema-filter", action="store_true")
    p.add_argument("--no-use-ema-filter", action="store_true")
    p.add_argument("--use-liquidity-sweep-filter", action="store_true")
    p.add_argument("--no-use-liquidity-sweep-filter", action="store_true")
    a = p.parse_args()

    use_ema = True
    if a.no_use_ema_filter:
        use_ema = False
    elif a.use_ema_filter:
        use_ema = True

    use_ls = True
    if a.no_use_liquidity_sweep_filter:
        use_ls = False
    elif a.use_liquidity_sweep_filter:
        use_ls = True

    return Config(
        input_dir=a.input_dir,
        output_dir=a.output_dir,
        buffer_pips=a.buffer_pips,
        spread_pips=a.spread_pips,
        slippage_pips=a.slippage_pips,
        commission_per_lot=a.commission_per_lot,
        max_confirmation_bars=a.max_confirmation_bars,
        rr_min=a.rr_min,
        use_ema_filter=use_ema,
        use_liquidity_sweep_filter=use_ls,
    )


if __name__ == "__main__":
    run(parse_args())
