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
    risk_reward_min: float
    max_trades_per_day: int
    force_close_hour: int
    force_close_minute: int
    cost_profile: str
    use_ema_filter: bool
    ema_period: int


def bool_arg(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def infer_pip_size(symbol: str) -> float:
    return PIP_SIZE_JPY if "JPY" in symbol.upper() else PIP_SIZE_FALLBACK


def prepare_data(raw: pd.DataFrame, ema_period: int) -> pd.DataFrame:
    data = raw.copy().sort_values("datetime").reset_index(drop=True)
    data["date"] = data["datetime"].dt.date

    daily = data.groupby("date", as_index=False).agg(day_high=("high", "max"), day_low=("low", "min"))
    daily["box_top"] = daily["day_high"].shift(1)
    daily["box_bottom"] = daily["day_low"].shift(1)
    daily = daily.dropna(subset=["box_top", "box_bottom"]).copy()
    daily["box_mid"] = (daily["box_top"] + daily["box_bottom"]) / 2.0

    merged = data.merge(daily[["date", "box_top", "box_bottom", "box_mid"]], on="date", how="left")
    merged = merged.dropna(subset=["box_top", "box_bottom"]).copy()
    merged["ema"] = merged["close"].ewm(span=ema_period, adjust=False).mean()
    return merged.reset_index(drop=True)


def _inside_box(row: pd.Series, box_bottom: float, box_top: float) -> bool:
    return row["low"] >= box_bottom and row["high"] <= box_top


def _find_setup(day_df: pd.DataFrame, direction: str) -> tuple[int | None, float | None]:
    box_top = float(day_df.iloc[0]["box_top"])
    box_bottom = float(day_df.iloc[0]["box_bottom"])

    touch_idx = None
    if direction == "long":
        for i in range(len(day_df)):
            if day_df.iloc[i]["low"] <= box_bottom:
                touch_idx = i
                break
    else:
        for i in range(len(day_df)):
            if day_df.iloc[i]["high"] >= box_top:
                touch_idx = i
                break

    if touch_idx is None:
        return None, None

    pre_touch = day_df.iloc[:touch_idx]
    if direction == "long":
        ref = pre_touch[(pre_touch["close"] > pre_touch["open"]) & pre_touch.apply(lambda r: _inside_box(r, box_bottom, box_top), axis=1)]
        if ref.empty:
            return None, None
        prev_green_high = float(ref.iloc[-1]["high"])
        for i in range(touch_idx + 1, len(day_df)):
            bar = day_df.iloc[i]
            closes_inside = box_bottom < bar["close"] < box_top
            if closes_inside and bar["close"] > prev_green_high:
                return i, prev_green_high
    else:
        ref = pre_touch[(pre_touch["close"] < pre_touch["open"]) & pre_touch.apply(lambda r: _inside_box(r, box_bottom, box_top), axis=1)]
        if ref.empty:
            return None, None
        prev_red_low = float(ref.iloc[-1]["low"])
        for i in range(touch_idx + 1, len(day_df)):
            bar = day_df.iloc[i]
            closes_inside = box_bottom < bar["close"] < box_top
            if closes_inside and bar["close"] < prev_red_low:
                return i, prev_red_low

    return None, None


def compute_summary(trades: pd.DataFrame) -> dict:
    keys = [
        "trades", "win_rate", "avg_net_pnl", "median_net_pnl", "total_net_pnl", "profit_factor", "expectancy",
        "max_drawdown_pips", "longest_loss_streak", "average_rr", "p05_net_pnl", "worst_trade", "best_trade",
    ]
    if trades.empty:
        return {k: 0.0 for k in keys}

    net = trades["net_pnl_pips"]
    gross_wins = net[net > 0].sum()
    gross_losses = net[net < 0].sum()
    profit_factor = gross_wins / abs(gross_losses) if gross_losses < 0 else np.inf

    equity = net.cumsum()
    max_drawdown = float((equity - equity.cummax()).min())

    streak = 0
    longest_loss_streak = 0
    for outcome in trades["outcome"]:
        if outcome in {"loss", "ambiguous"}:
            streak += 1
            longest_loss_streak = max(longest_loss_streak, streak)
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
        "max_drawdown_pips": max_drawdown,
        "longest_loss_streak": int(longest_loss_streak),
        "average_rr": float(trades["rr_ratio"].mean()),
        "p05_net_pnl": float(net.quantile(0.05)),
        "worst_trade": float(net.min()),
        "best_trade": float(net.max()),
    }


def run_backtest(data: pd.DataFrame, cfg: DBSConfig) -> pd.DataFrame:
    pip_size = infer_pip_size(cfg.symbol)
    buffer_px = cfg.buffer_pips * pip_size
    friction_pips = sum(COST_PROFILES[cfg.cost_profile].values())

    trades: list[dict] = []
    for day, day_df in data.groupby("date", sort=True):
        day_df = day_df.reset_index(drop=True)
        candidates: list[tuple[pd.Timestamp, str, int]] = []
        for direction in ("long", "short"):
            entry_idx, _ = _find_setup(day_df, direction)
            if entry_idx is not None:
                candidates.append((day_df.iloc[entry_idx]["datetime"], direction, entry_idx))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0])
        taken = 0
        for _, direction, entry_idx in candidates:
            if taken >= cfg.max_trades_per_day:
                break

            bar = day_df.iloc[entry_idx]
            entry = float(bar["close"])
            box_top = float(bar["box_top"])
            box_bottom = float(bar["box_bottom"])

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
            if rr < cfg.risk_reward_min:
                continue

            if cfg.use_ema_filter:
                if direction == "long" and not (entry > bar["ema"]):
                    continue
                if direction == "short" and not (entry < bar["ema"]):
                    continue

            cutoff = pd.Timestamp(f"{day} {cfg.force_close_hour:02d}:{cfg.force_close_minute:02d}:00")
            post = day_df.iloc[entry_idx + 1 :]
            outcome = "force_close"
            exit_price = entry
            bars_in_trade = 0

            for _, post_bar in post.iterrows():
                if post_bar["datetime"] > cutoff:
                    break
                bars_in_trade += 1
                if direction == "long":
                    hit_sl = post_bar["low"] <= sl
                    hit_tp = post_bar["high"] >= tp
                else:
                    hit_sl = post_bar["high"] >= sl
                    hit_tp = post_bar["low"] <= tp

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
                in_window = day_df[day_df["datetime"] <= cutoff]
                if in_window.empty:
                    continue
                exit_price = float(in_window.iloc[-1]["close"])

            gross_pnl_pips = (exit_price - entry) / pip_size if direction == "long" else (entry - exit_price) / pip_size
            net_pnl_pips = gross_pnl_pips - friction_pips

            trades.append({
                "symbol": cfg.symbol,
                "date": str(day),
                "entry_datetime": bar["datetime"],
                "direction": direction,
                "box_top": box_top,
                "box_bottom": box_bottom,
                "box_mid": float(bar["box_mid"]),
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
                "hour": int(bar["datetime"].hour),
            })
            taken += 1

    return pd.DataFrame(trades)


def verdict_text(wf: pd.DataFrame, by_year: pd.DataFrame, by_month: pd.DataFrame, summary: dict) -> tuple[str, str]:
    oos = wf[wf["segment"] == "test_oos"]
    if oos.empty:
        return "FAIL", "No OOS segment available."

    oos_row = oos.iloc[0]
    year_positive_count = int((by_year["total_net_pnl"] > 0).sum()) if not by_year.empty else 0
    one_month_dominant = False
    if not by_month.empty:
        top_month_abs = abs(float(by_month["total_net_pnl"].max()))
        total_abs = abs(float(summary["total_net_pnl"]))
        one_month_dominant = total_abs > 0 and top_month_abs > 0.6 * total_abs

    dd_ok = summary["max_drawdown_pips"] > -500
    streak_ok = summary["longest_loss_streak"] <= 8

    pass_core = (
        oos_row["total_net_pnl"] > 0
        and oos_row["expectancy"] > 0
        and oos_row["profit_factor"] > 1.15
        and year_positive_count >= 2
        and not one_month_dominant
        and dd_ok
        and streak_ok
    )

    if pass_core:
        return "PASS", "OOS passes profitability and stability checks."

    weak_positive = oos_row["total_net_pnl"] > 0 and oos_row["expectancy"] > 0 and oos_row["profit_factor"] > 1
    if weak_positive:
        return "WARN", "OOS is positive but weak or unstable."

    return "FAIL", "OOS expectancy/total/profit factor or stability checks failed."


def parse_args() -> DBSConfig:
    parser = argparse.ArgumentParser(description="Daily Box Strategy research script (no live trading).")
    parser.add_argument("--csv", default="data/USDJPY_M15_MT5_5Y.csv")
    parser.add_argument("--symbol", default="USDJPY")
    parser.add_argument("--output-dir", default="reports/daily_box_strategy")
    parser.add_argument("--buffer-pips", type=float, default=0)
    parser.add_argument("--risk-reward-min", type=float, default=1.5)
    parser.add_argument("--max-trades-per-day", type=int, default=1)
    parser.add_argument("--force-close-hour", type=int, default=23)
    parser.add_argument("--force-close-minute", type=int, default=45)
    parser.add_argument("--cost-profile", choices=["low", "conservative", "high"], default="conservative")
    parser.add_argument("--use-ema-filter", default="false")
    parser.add_argument("--ema-period", type=int, default=200)

    args = parser.parse_args()
    return DBSConfig(
        csv=args.csv,
        symbol=args.symbol,
        output_dir=Path(args.output_dir),
        buffer_pips=args.buffer_pips,
        risk_reward_min=args.risk_reward_min,
        max_trades_per_day=args.max_trades_per_day,
        force_close_hour=args.force_close_hour,
        force_close_minute=args.force_close_minute,
        cost_profile=args.cost_profile,
        use_ema_filter=bool_arg(args.use_ema_filter),
        ema_period=args.ema_period,
    )


def main() -> None:
    cfg = parse_args()
    raw, warnings = load_ohlc_csv(cfg.csv)
    data = prepare_data(raw, cfg.ema_period)
    trades = run_backtest(data, cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(cfg.output_dir / "dbs_trades.csv", index=False)

    summary = compute_summary(trades)
    pd.DataFrame([summary]).to_csv(cfg.output_dir / "dbs_summary.csv", index=False)

    if trades.empty:
        by_year = pd.DataFrame(columns=["year", "trades", "total_net_pnl", "expectancy", "profit_factor"])
        by_month = pd.DataFrame(columns=["month", "trades", "total_net_pnl", "expectancy", "profit_factor"])
        by_hour = pd.DataFrame(columns=["hour", "trades", "total_net_pnl", "expectancy"])
    else:
        t = trades.copy()
        t["entry_datetime"] = pd.to_datetime(t["entry_datetime"])
        t["year"] = t["entry_datetime"].dt.year
        t["month"] = t["entry_datetime"].dt.to_period("M").astype(str)

        def pf(group: pd.DataFrame) -> float:
            wins = group.loc[group["net_pnl_pips"] > 0, "net_pnl_pips"].sum()
            losses = group.loc[group["net_pnl_pips"] < 0, "net_pnl_pips"].sum()
            return wins / abs(losses) if losses < 0 else np.inf

        by_year = t.groupby("year").apply(lambda g: pd.Series({"trades": len(g), "total_net_pnl": g["net_pnl_pips"].sum(), "expectancy": g["net_pnl_pips"].mean(), "profit_factor": pf(g)})).reset_index()
        by_month = t.groupby("month").apply(lambda g: pd.Series({"trades": len(g), "total_net_pnl": g["net_pnl_pips"].sum(), "expectancy": g["net_pnl_pips"].mean(), "profit_factor": pf(g)})).reset_index()
        by_hour = t.groupby("hour").agg(trades=("net_pnl_pips", "count"), total_net_pnl=("net_pnl_pips", "sum"), expectancy=("net_pnl_pips", "mean")).reset_index()

    by_year.to_csv(cfg.output_dir / "dbs_by_year.csv", index=False)
    by_month.to_csv(cfg.output_dir / "dbs_by_month.csv", index=False)
    by_hour.to_csv(cfg.output_dir / "dbs_by_hour.csv", index=False)

    segs = {"reference": (2022, 2023), "validation": (2024, 2024), "test_oos": (2025, 2026)}
    wf_rows: list[dict] = []
    for name, (from_y, to_y) in segs.items():
        if trades.empty:
            m = {"trades": 0, "total_net_pnl": 0.0, "expectancy": 0.0, "profit_factor": 0.0}
        else:
            years = pd.to_datetime(trades["entry_datetime"]).dt.year
            seg_trades = trades[(years >= from_y) & (years <= to_y)]
            if seg_trades.empty:
                m = {"trades": 0, "total_net_pnl": 0.0, "expectancy": 0.0, "profit_factor": 0.0}
            else:
                seg_summary = compute_summary(seg_trades)
                m = {k: seg_summary[k] for k in ["trades", "total_net_pnl", "expectancy", "profit_factor"]}
        m.update({"segment": name, "from_year": from_y, "to_year": to_y})
        wf_rows.append(m)

    wf = pd.DataFrame(wf_rows)
    wf.to_csv(cfg.output_dir / "dbs_walkforward.csv", index=False)

    verdict, reason = verdict_text(wf, by_year, by_month, summary)
    with (cfg.output_dir / "dbs_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# Daily Box Strategy Research Summary\n\n")
        handle.write("Research-only script. No EA, no MT4/MT5 execution, no live trading, no lot sizing, and no equity/account modeling.\n\n")
        handle.write(f"- Symbol: {cfg.symbol}\n")
        handle.write(f"- CSV: {cfg.csv}\n")
        handle.write(f"- Cost profile: {cfg.cost_profile}\n")
        handle.write(f"- Trades: {summary['trades']}\n")
        handle.write(f"- Win rate: {summary['win_rate']:.2%}\n")
        handle.write(f"- Total net pnl (pips): {summary['total_net_pnl']:.2f}\n")
        handle.write(f"- Profit factor: {summary['profit_factor']:.3f}\n")
        handle.write(f"- Expectancy (pips): {summary['expectancy']:.3f}\n\n")
        handle.write(f"## Verdict: {verdict}\n{reason}\n")
        if warnings:
            handle.write("\n## Loader warnings\n")
            for warning in warnings:
                handle.write(f"- {warning}\n")


if __name__ == "__main__":
    main()
