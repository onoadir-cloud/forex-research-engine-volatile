#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_ohlc_csv

PIP_SIZE_FALLBACK = 0.0001
PIP_SIZE_JPY = 0.01

COST_PROFILES = {
    "low": {"spread_pips": 1.0, "slippage_pips": 0.3, "commission_equivalent_pips": 0.3},
    "conservative": {"spread_pips": 1.5, "slippage_pips": 0.5, "commission_equivalent_pips": 0.5},
    "high": {"spread_pips": 2.2, "slippage_pips": 0.8, "commission_equivalent_pips": 0.8},
}


@dataclass
class DBSConfig:
    csv: str
    symbol: str
    output_dir: Path
    buffer_pips: float
    max_trades_per_day: int
    force_close_hour: int
    force_close_minute: int
    cost_profile: str
    use_ema_filter: bool
    ema_period: int
    use_sweep_filter: bool


def infer_pip_size(symbol: str) -> float:
    return PIP_SIZE_JPY if "JPY" in symbol.upper() else PIP_SIZE_FALLBACK


def bool_arg(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def prepare_data(raw: pd.DataFrame, ema_period: int) -> pd.DataFrame:
    data = raw.copy().sort_values("datetime").reset_index(drop=True)
    data["date"] = data["datetime"].dt.date
    daily = data.groupby("date", as_index=False).agg({"high": "max", "low": "min"})
    daily["box_top"] = daily["high"].shift(1)
    daily["box_bottom"] = daily["low"].shift(1)
    daily = daily.dropna(subset=["box_top", "box_bottom"]).copy()
    daily["box_mid"] = (daily["box_top"] + daily["box_bottom"]) / 2.0
    daily["box_q1"] = daily["box_bottom"] + 0.25 * (daily["box_top"] - daily["box_bottom"])
    daily["box_q3"] = daily["box_bottom"] + 0.75 * (daily["box_top"] - daily["box_bottom"])

    merged = data.merge(daily[["date", "box_top", "box_bottom", "box_mid", "box_q1", "box_q3"]], on="date", how="left")
    merged = merged.dropna(subset=["box_top", "box_bottom"]).copy()
    merged["ema"] = merged["close"].ewm(span=ema_period, adjust=False).mean()
    merged = merged.reset_index(drop=True)
    return merged


def _inside_box(bar: pd.Series, box_bottom: float, box_top: float) -> bool:
    return bar["low"] >= box_bottom and bar["high"] <= box_top


def _find_entry(day_df: pd.DataFrame, direction: str, use_sweep_filter: bool) -> tuple[int | None, bool]:
    box_top = float(day_df.iloc[0]["box_top"])
    box_bottom = float(day_df.iloc[0]["box_bottom"])
    touch_idx = None
    sweep_pass = False

    if direction == "long":
        for i in range(len(day_df)):
            if day_df.iloc[i]["low"] <= box_bottom:
                touch_idx = i
                if day_df.iloc[i]["low"] < box_bottom and day_df.iloc[i]["close"] > box_bottom and day_df.iloc[i]["close"] < box_top:
                    sweep_pass = True
                break
    else:
        for i in range(len(day_df)):
            if day_df.iloc[i]["high"] >= box_top:
                touch_idx = i
                if day_df.iloc[i]["high"] > box_top and day_df.iloc[i]["close"] > box_bottom and day_df.iloc[i]["close"] < box_top:
                    sweep_pass = True
                break

    if touch_idx is None:
        return None, False

    pre = day_df.iloc[:touch_idx]
    if direction == "long":
        refs = pre[(pre["close"] > pre["open"]) & pre.apply(lambda r: _inside_box(r, box_bottom, box_top), axis=1)]
    else:
        refs = pre[(pre["close"] < pre["open"]) & pre.apply(lambda r: _inside_box(r, box_bottom, box_top), axis=1)]
    if refs.empty:
        return None, False

    ref = refs.iloc[-1]
    for j in range(touch_idx + 1, len(day_df)):
        c = day_df.iloc[j]
        back_inside = c["close"] > box_bottom and c["close"] < box_top
        if not back_inside:
            continue
        if direction == "long" and c["close"] > ref["high"]:
            return j, (sweep_pass if use_sweep_filter else True)
        if direction == "short" and c["close"] < ref["low"]:
            return j, (sweep_pass if use_sweep_filter else True)

    return None, False


def run_backtest(data: pd.DataFrame, cfg: DBSConfig) -> pd.DataFrame:
    pip_size = infer_pip_size(cfg.symbol)
    friction_pips = sum(COST_PROFILES[cfg.cost_profile].values())
    buffer_px = cfg.buffer_pips * pip_size

    trades: list[dict] = []
    for day, day_df in data.groupby("date", sort=True):
        if cfg.max_trades_per_day <= 0:
            continue
        day_df = day_df.reset_index(drop=True)
        candidates: list[tuple[pd.Timestamp, str, int, bool]] = []

        for direction in ("long", "short"):
            idx, sweep_pass = _find_entry(day_df, direction, cfg.use_sweep_filter)
            if idx is not None:
                candidates.append((day_df.iloc[idx]["datetime"], direction, idx, sweep_pass))

        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])

        taken = 0
        for _, direction, entry_idx, sweep_pass in candidates:
            if taken >= cfg.max_trades_per_day:
                break
            row = day_df.iloc[entry_idx]
            entry = float(row["close"])
            box_top = float(row["box_top"])
            box_bottom = float(row["box_bottom"])
            if direction == "long":
                sl = box_bottom - buffer_px
                tp = box_top
                risk_pips = (entry - sl) / pip_size
                reward_pips = (tp - entry) / pip_size
            else:
                sl = box_top + buffer_px
                tp = box_bottom
                risk_pips = (sl - entry) / pip_size
                reward_pips = (entry - tp) / pip_size
            if risk_pips <= 0 or reward_pips <= 0:
                continue
            rr = reward_pips / risk_pips
            if rr < 1.5:
                continue

            ema_pass = True
            if cfg.use_ema_filter:
                ema_pass = bool(entry > row["ema"] if direction == "long" else entry < row["ema"])
                if not ema_pass:
                    continue
            if cfg.use_sweep_filter and not sweep_pass:
                continue

            cutoff = pd.Timestamp(f"{day} {cfg.force_close_hour:02d}:{cfg.force_close_minute:02d}:00")
            post = day_df.iloc[entry_idx + 1 :]
            outcome = "force_close"
            exit_price = entry
            bars_in_trade = 0

            for k, bar in post.iterrows():
                if bar["datetime"] > cutoff:
                    break
                bars_in_trade += 1
                if direction == "long":
                    hit_sl = bar["low"] <= sl
                    hit_tp = bar["high"] >= tp
                else:
                    hit_sl = bar["high"] >= sl
                    hit_tp = bar["low"] <= tp
                if hit_sl and hit_tp:
                    outcome = "ambiguous"
                    exit_price = sl
                    break
                if hit_sl:
                    outcome = "loss"
                    exit_price = sl
                    break
                if hit_tp:
                    outcome = "win"
                    exit_price = tp
                    break

            if outcome == "force_close":
                eligible = day_df[day_df["datetime"] <= cutoff]
                if eligible.empty:
                    continue
                exit_price = float(eligible.iloc[-1]["close"])

            gross_pnl_pips = ((exit_price - entry) / pip_size) if direction == "long" else ((entry - exit_price) / pip_size)
            net_pnl_pips = gross_pnl_pips - friction_pips

            trades.append({
                "symbol": cfg.symbol,
                "date": str(day),
                "entry_datetime": row["datetime"],
                "direction": direction,
                "box_top": box_top,
                "box_bottom": box_bottom,
                "box_mid": float(row["box_mid"]),
                "box_q1": float(row["box_q1"]),
                "box_q3": float(row["box_q3"]),
                "entry_price": entry,
                "stop_loss": sl,
                "take_profit": tp,
                "buffer_pips": cfg.buffer_pips,
                "risk_pips": risk_pips,
                "reward_pips": reward_pips,
                "rr_ratio": rr,
                "outcome": outcome,
                "gross_pnl_pips": gross_pnl_pips,
                "friction_pips": friction_pips,
                "net_pnl_pips": net_pnl_pips,
                "bars_in_trade": bars_in_trade,
                "hour": int(row["datetime"].hour),
                "day_of_week": row["datetime"].day_name(),
                "ema_filter_pass": ema_pass,
                "sweep_filter_pass": sweep_pass,
            })
            taken += 1

    return pd.DataFrame(trades)


def compute_summary(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {k: 0.0 for k in ["trades", "win_rate", "avg_net_pnl", "median_net_pnl", "total_net_pnl", "profit_factor", "expectancy", "max_drawdown_pips", "longest_loss_streak", "average_rr", "p05_net_pnl", "worst_trade", "best_trade"]}
    net = trades["net_pnl_pips"]
    wins = net[net > 0].sum()
    losses = net[net < 0].sum()
    profit_factor = wins / abs(losses) if losses < 0 else np.inf
    eq = net.cumsum()
    dd = (eq - eq.cummax()).min()
    streak = max_streak = 0
    for outcome in trades["outcome"]:
        if outcome in {"loss", "ambiguous"}:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return {
        "trades": int(len(trades)),
        "win_rate": float((trades["outcome"] == "win").mean()),
        "avg_net_pnl": float(net.mean()),
        "median_net_pnl": float(net.median()),
        "total_net_pnl": float(net.sum()),
        "profit_factor": float(profit_factor),
        "expectancy": float(net.mean()),
        "max_drawdown_pips": float(dd),
        "longest_loss_streak": int(max_streak),
        "average_rr": float(trades["rr_ratio"].mean()),
        "p05_net_pnl": float(net.quantile(0.05)),
        "worst_trade": float(net.min()),
        "best_trade": float(net.max()),
    }


def parse_args() -> DBSConfig:
    p = argparse.ArgumentParser(description="Daily Box Strategy research-only backtest")
    p.add_argument("--csv", default="data/USDJPY_M15_MT5_5Y.csv")
    p.add_argument("--symbol", default="USDJPY")
    p.add_argument("--output-dir", default="reports/daily_box_strategy")
    p.add_argument("--buffer-pips", type=float, default=0)
    p.add_argument("--max-trades-per-day", type=int, default=1)
    p.add_argument("--force-close-hour", type=int, default=23)
    p.add_argument("--force-close-minute", type=int, default=45)
    p.add_argument("--cost-profile", choices=["low", "conservative", "high"], default="conservative")
    p.add_argument("--use-ema-filter", default="false")
    p.add_argument("--ema-period", type=int, default=200)
    p.add_argument("--use-sweep-filter", default="true")
    a = p.parse_args()
    return DBSConfig(a.csv, a.symbol, Path(a.output_dir), a.buffer_pips, a.max_trades_per_day, a.force_close_hour, a.force_close_minute, a.cost_profile, bool_arg(a.use_ema_filter), a.ema_period, bool_arg(a.use_sweep_filter))


def verdict_text(wf: pd.DataFrame, by_year: pd.DataFrame, by_month: pd.DataFrame, summary: dict) -> tuple[str, str]:
    oos = wf[wf["segment"] == "test_oos"]
    if oos.empty:
        return "FAIL", "No OOS trades available"
    row = oos.iloc[0]
    one_month_heavy = False if by_month.empty else abs(by_month["total_net_pnl"]).max() > 0.6 * abs(summary["total_net_pnl"]) if summary["total_net_pnl"] != 0 else False
    multi_year = (by_year["total_net_pnl"] > 0).sum() >= 2 if not by_year.empty else False
    if row["total_net_pnl"] > 0 and row["expectancy"] > 0 and row["profit_factor"] > 1.15 and multi_year and not one_month_heavy and summary["max_drawdown_pips"] > -500 and summary["longest_loss_streak"] <= 8:
        return "PASS", "OOS metrics and stability checks passed"
    if row["total_net_pnl"] > 0 and row["expectancy"] > 0 and row["profit_factor"] > 1:
        return "WARN", "OOS positive but weak/unstable"
    return "FAIL", "OOS metrics failed or concentration/drawdown too high"


def main() -> None:
    cfg = parse_args()
    raw, warnings = load_ohlc_csv(cfg.csv)
    data = prepare_data(raw, cfg.ema_period)
    trades = run_backtest(data, cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    trades_path = cfg.output_dir / "dbs_trades.csv"
    trades.to_csv(trades_path, index=False)

    summary = compute_summary(trades)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(cfg.output_dir / "dbs_summary.csv", index=False)

    if trades.empty:
        by_year = pd.DataFrame(columns=["year", "trades", "total_net_pnl", "expectancy", "profit_factor"])
        by_month = pd.DataFrame(columns=["month", "trades", "total_net_pnl", "expectancy", "profit_factor"])
        by_hour = pd.DataFrame(columns=["hour", "trades", "total_net_pnl", "expectancy"])
    else:
        t = trades.copy()
        t["entry_datetime"] = pd.to_datetime(t["entry_datetime"])
        t["year"] = t["entry_datetime"].dt.year
        t["month"] = t["entry_datetime"].dt.to_period("M").astype(str)
        by_year = t.groupby("year").apply(lambda g: pd.Series({"trades": len(g), "total_net_pnl": g["net_pnl_pips"].sum(), "expectancy": g["net_pnl_pips"].mean(), "profit_factor": g.loc[g["net_pnl_pips"] > 0, "net_pnl_pips"].sum() / abs(g.loc[g["net_pnl_pips"] < 0, "net_pnl_pips"].sum()) if (g["net_pnl_pips"] < 0).any() else np.inf})).reset_index()
        by_month = t.groupby("month").apply(lambda g: pd.Series({"trades": len(g), "total_net_pnl": g["net_pnl_pips"].sum(), "expectancy": g["net_pnl_pips"].mean(), "profit_factor": g.loc[g["net_pnl_pips"] > 0, "net_pnl_pips"].sum() / abs(g.loc[g["net_pnl_pips"] < 0, "net_pnl_pips"].sum()) if (g["net_pnl_pips"] < 0).any() else np.inf})).reset_index()
        by_hour = t.groupby("hour").agg(trades=("net_pnl_pips", "count"), total_net_pnl=("net_pnl_pips", "sum"), expectancy=("net_pnl_pips", "mean")).reset_index()

    by_year.to_csv(cfg.output_dir / "dbs_by_year.csv", index=False)
    by_month.to_csv(cfg.output_dir / "dbs_by_month.csv", index=False)
    by_hour.to_csv(cfg.output_dir / "dbs_by_hour.csv", index=False)

    segments = {
        "reference": (2022, 2023),
        "validation": (2024, 2024),
        "test_oos": (2025, 2026),
    }
    wf_rows = []
    for name, (y0, y1) in segments.items():
        if trades.empty:
            s = {"trades": 0, "total_net_pnl": 0.0, "expectancy": 0.0, "profit_factor": 0.0}
        else:
            ts = pd.to_datetime(trades["entry_datetime"])
            seg = trades[(ts.dt.year >= y0) & (ts.dt.year <= y1)]
            if seg.empty:
                s = {"trades": 0, "total_net_pnl": 0.0, "expectancy": 0.0, "profit_factor": 0.0}
            else:
                sn = compute_summary(seg)
                s = {k: sn[k] for k in ["trades", "total_net_pnl", "expectancy", "profit_factor"]}
        s.update({"segment": name, "from_year": y0, "to_year": y1})
        wf_rows.append(s)
    wf = pd.DataFrame(wf_rows)
    wf.to_csv(cfg.output_dir / "dbs_walkforward.csv", index=False)

    verdict, reason = verdict_text(wf, by_year, by_month, summary)
    with (cfg.output_dir / "dbs_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Daily Box Strategy Research Summary\n\n")
        f.write("Research-only backtest. No live trading, no EA execution, no lot sizing, no equity curve optimization.\n\n")
        f.write(f"- Symbol: {cfg.symbol}\n- CSV: {cfg.csv}\n- Cost profile: {cfg.cost_profile}\n")
        f.write(f"- Trades: {summary['trades']}\n- Win rate: {summary['win_rate']:.2%}\n- Total net pnl (pips): {summary['total_net_pnl']:.2f}\n")
        f.write(f"- Profit factor: {summary['profit_factor']:.3f}\n- Expectancy (pips): {summary['expectancy']:.3f}\n")
        f.write(f"\n## Verdict: {verdict}\n{reason}\n")
        if warnings:
            f.write("\n## Loader warnings\n")
            for w in warnings:
                f.write(f"- {w}\n")


if __name__ == "__main__":
    main()
