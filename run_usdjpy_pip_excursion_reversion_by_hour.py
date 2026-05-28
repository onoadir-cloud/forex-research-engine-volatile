#!/usr/bin/env python3
"""Research-only USDJPY intraday pip-excursion reversion study by hour.

This script tests whether the first M15-confirmed excursion away from an hour's
opening price tends to revert, continue, or behave no better than a deterministic
random choice. It intentionally contains no MT4/MT5 execution, live trading, lot
sizing, or account-equity logic.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SYMBOL = "USDJPY"
PIP_SIZE_USDJPY = 0.01
EXCURSION_PIPS = [10, 15, 20, 25, 30]
REVERSION_TARGET_PIPS = [5, 8, 10, 12, 15]
ADVERSE_CONTINUATION_PIPS = [10, 15, 20, 25, 30]
FUTURE_HORIZON_BARS = [4, 8, 16]
COST_PROFILES: Dict[str, Dict[str, float]] = {
    "low": {"spread_pips": 1.0, "slippage_pips": 0.3, "commission_equivalent_pips": 0.3},
    "conservative": {"spread_pips": 1.5, "slippage_pips": 0.5, "commission_equivalent_pips": 0.5},
    "high": {"spread_pips": 2.2, "slippage_pips": 0.8, "commission_equivalent_pips": 0.8},
}
EVENT_COLUMNS = [
    "symbol",
    "datetime",
    "date",
    "year",
    "month",
    "day_of_week",
    "hour",
    "hour_open",
    "entry_price",
    "stretch_direction",
    "excursion_pips",
    "reversion_target_pips",
    "adverse_continuation_pips",
    "future_horizon_bars",
    "friction_pips",
    "outcome",
    "contrarian_pnl_after_friction",
    "continuation_pnl_after_friction",
    "random_pnl_after_friction",
    "edge_vs_continuation",
    "edge_vs_random",
    "bars_to_outcome",
    "max_favorable_pips",
    "max_adverse_pips",
]


@dataclass(frozen=True)
class BaseEvent:
    symbol: str
    datetime: pd.Timestamp
    date: object
    year: int
    month: int
    day_of_week: int
    hour: int
    hour_open: float
    entry_price: float
    stretch_direction: str
    excursion_pips: int
    event_idx: int


@dataclass(frozen=True)
class OutcomeResult:
    outcome: str
    pnl_after_friction: float
    bars_to_outcome: int
    max_favorable_pips: float
    max_adverse_pips: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research-only USDJPY M15 pip-excursion reversion by hour."
    )
    parser.add_argument("--csv", default="data/USDJPY_M15_MT5_5Y.csv")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument(
        "--cost-profile",
        choices=sorted(COST_PROFILES),
        default="conservative",
        help="Round-trip friction profile for spread, slippage, and commission-equivalent pips.",
    )
    parser.add_argument("--output-dir", default="reports/usdjpy_pip_reversion_by_hour")
    return parser.parse_args()


def friction_for_profile(profile_name: str) -> float:
    profile = COST_PROFILES[profile_name]
    return (
        2 * profile["spread_pips"]
        + 2 * profile["slippage_pips"]
        + profile["commission_equivalent_pips"]
    )


def load_ohlc(csv_path: str | Path, symbol: str = SYMBOL) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    lower_cols = {str(col).lower().strip(): col for col in df.columns}
    datetime_col = lower_cols.get("datetime") or lower_cols.get("time") or lower_cols.get("date")
    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in lower_cols]
    if datetime_col is None or missing:
        raise ValueError(
            "CSV must include datetime/time/date plus open, high, low, close columns; "
            f"missing={missing}, has_datetime={datetime_col is not None}"
        )

    df = df.rename(
        columns={
            datetime_col: "datetime",
            lower_cols["open"]: "open",
            lower_cols["high"]: "high",
            lower_cols["low"]: "low",
            lower_cols["close"]: "close",
        }
    )[["datetime", "open", "high", "low", "close"]]
    df["datetime"] = pd.to_datetime(df["datetime"], utc=False)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="first").reset_index(drop=True)
    df["symbol"] = symbol
    df["date"] = df["datetime"].dt.date
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    return df


def resolve_stretch_direction(row: pd.Series, hour_open: float, excursion_pips: int) -> Optional[str]:
    up_pips = (float(row["high"]) - hour_open) / PIP_SIZE_USDJPY
    down_pips = (hour_open - float(row["low"])) / PIP_SIZE_USDJPY
    up_hit = up_pips >= excursion_pips
    down_hit = down_pips >= excursion_pips
    if up_hit and not down_hit:
        return "UP"
    if down_hit and not up_hit:
        return "DOWN"
    if not up_hit and not down_hit:
        return None

    close_delta = float(row["close"]) - hour_open
    if close_delta > 0:
        return "UP"
    if close_delta < 0:
        return "DOWN"
    return "UP" if up_pips >= down_pips else "DOWN"


def find_base_events(df: pd.DataFrame, excursion_values: Sequence[int]) -> List[BaseEvent]:
    events: List[BaseEvent] = []
    for excursion_pips in excursion_values:
        for (_, _), hour_df in df.groupby(["date", "hour"], sort=True):
            if hour_df.empty:
                continue
            hour_open = float(hour_df.iloc[0]["open"])
            for event_idx, row in hour_df.iterrows():
                stretch_direction = resolve_stretch_direction(row, hour_open, excursion_pips)
                if stretch_direction is None:
                    continue
                event_time = row["datetime"]
                events.append(
                    BaseEvent(
                        symbol=str(row.get("symbol", SYMBOL)),
                        datetime=event_time,
                        date=row["date"],
                        year=int(row["year"]),
                        month=int(row["month"]),
                        day_of_week=int(row["day_of_week"]),
                        hour=int(row["hour"]),
                        hour_open=hour_open,
                        entry_price=float(row["close"]),
                        stretch_direction=stretch_direction,
                        excursion_pips=int(excursion_pips),
                        event_idx=int(event_idx),
                    )
                )
                break
    return events


def direction_for_event(stretch_direction: str, contrarian: bool) -> str:
    if contrarian:
        return "SHORT" if stretch_direction == "UP" else "LONG"
    return "LONG" if stretch_direction == "UP" else "SHORT"


def simulate_direction(
    df: pd.DataFrame,
    event_idx: int,
    entry_price: float,
    direction: str,
    target_pips: int,
    adverse_pips: int,
    horizon_bars: int,
    friction_pips: float,
) -> OutcomeResult:
    start_idx = event_idx + 1
    end_idx = min(len(df), start_idx + horizon_bars)
    future = df.iloc[start_idx:end_idx]
    if future.empty:
        return OutcomeResult("timeout", -friction_pips, 0, 0.0, 0.0)

    max_favorable = 0.0
    max_adverse = 0.0
    for bar_number, (_, row) in enumerate(future.iterrows(), start=1):
        high = float(row["high"])
        low = float(row["low"])
        if direction == "LONG":
            favorable = max(0.0, (high - entry_price) / PIP_SIZE_USDJPY)
            adverse = max(0.0, (entry_price - low) / PIP_SIZE_USDJPY)
            target_hit = high >= entry_price + target_pips * PIP_SIZE_USDJPY
            adverse_hit = low <= entry_price - adverse_pips * PIP_SIZE_USDJPY
        else:
            favorable = max(0.0, (entry_price - low) / PIP_SIZE_USDJPY)
            adverse = max(0.0, (high - entry_price) / PIP_SIZE_USDJPY)
            target_hit = low <= entry_price - target_pips * PIP_SIZE_USDJPY
            adverse_hit = high >= entry_price + adverse_pips * PIP_SIZE_USDJPY

        max_favorable = max(max_favorable, favorable)
        max_adverse = max(max_adverse, adverse)
        if adverse_hit:
            return OutcomeResult(
                "failure", -float(adverse_pips) - friction_pips, bar_number, max_favorable, max_adverse
            )
        if target_hit:
            return OutcomeResult(
                "success", float(target_pips) - friction_pips, bar_number, max_favorable, max_adverse
            )

    horizon_close = float(future.iloc[-1]["close"])
    if direction == "LONG":
        mark_to_market = (horizon_close - entry_price) / PIP_SIZE_USDJPY
    else:
        mark_to_market = (entry_price - horizon_close) / PIP_SIZE_USDJPY
    return OutcomeResult(
        "timeout",
        mark_to_market - friction_pips,
        len(future),
        max_favorable,
        max_adverse,
    )


def deterministic_random_uses_contrarian(event: BaseEvent) -> bool:
    key = f"{event.symbol}|{pd.Timestamp(event.datetime).isoformat()}|{event.date}|{event.hour}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2 == 0


def build_event_rows(df: pd.DataFrame, base_events: Iterable[BaseEvent], friction_pips: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for event in base_events:
        contrarian_direction = direction_for_event(event.stretch_direction, contrarian=True)
        continuation_direction = direction_for_event(event.stretch_direction, contrarian=False)
        for target_pips in REVERSION_TARGET_PIPS:
            for adverse_pips in ADVERSE_CONTINUATION_PIPS:
                for horizon_bars in FUTURE_HORIZON_BARS:
                    contrarian = simulate_direction(
                        df,
                        event.event_idx,
                        event.entry_price,
                        contrarian_direction,
                        target_pips,
                        adverse_pips,
                        horizon_bars,
                        friction_pips,
                    )
                    continuation = simulate_direction(
                        df,
                        event.event_idx,
                        event.entry_price,
                        continuation_direction,
                        target_pips,
                        adverse_pips,
                        horizon_bars,
                        friction_pips,
                    )
                    random_pnl = (
                        contrarian.pnl_after_friction
                        if deterministic_random_uses_contrarian(event)
                        else continuation.pnl_after_friction
                    )
                    rows.append(
                        {
                            "symbol": event.symbol,
                            "datetime": event.datetime,
                            "date": event.date,
                            "year": event.year,
                            "month": event.month,
                            "day_of_week": event.day_of_week,
                            "hour": event.hour,
                            "hour_open": event.hour_open,
                            "entry_price": event.entry_price,
                            "stretch_direction": event.stretch_direction,
                            "excursion_pips": event.excursion_pips,
                            "reversion_target_pips": target_pips,
                            "adverse_continuation_pips": adverse_pips,
                            "future_horizon_bars": horizon_bars,
                            "friction_pips": friction_pips,
                            "outcome": contrarian.outcome,
                            "contrarian_pnl_after_friction": contrarian.pnl_after_friction,
                            "continuation_pnl_after_friction": continuation.pnl_after_friction,
                            "random_pnl_after_friction": random_pnl,
                            "edge_vs_continuation": contrarian.pnl_after_friction
                            - continuation.pnl_after_friction,
                            "edge_vs_random": contrarian.pnl_after_friction - random_pnl,
                            "bars_to_outcome": contrarian.bars_to_outcome,
                            "max_favorable_pips": contrarian.max_favorable_pips,
                            "max_adverse_pips": contrarian.max_adverse_pips,
                        }
                    )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def profit_factor(values: pd.Series) -> float:
    wins = values[values > 0].sum()
    losses = -values[values < 0].sum()
    if losses > 0:
        return float(wins / losses)
    return float("inf") if wins > 0 else 0.0


def max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    equity = values.cumsum()
    drawdown = equity - equity.cummax()
    return float(drawdown.min())


def aggregate_group(group: pd.DataFrame) -> Dict[str, object]:
    pnl = group["contrarian_pnl_after_friction"]
    n = len(group)
    return {
        "observations": int(n),
        "mean_contrarian_pnl": float(pnl.mean()) if n else 0.0,
        "median_contrarian_pnl": float(pnl.median()) if n else 0.0,
        "total_contrarian_pnl": float(pnl.sum()) if n else 0.0,
        "win_rate": float((pnl > 0).mean()) if n else 0.0,
        "success_rate": float((group["outcome"] == "success").mean()) if n else 0.0,
        "failure_rate": float((group["outcome"] == "failure").mean()) if n else 0.0,
        "timeout_rate": float((group["outcome"] == "timeout").mean()) if n else 0.0,
        "profit_factor": profit_factor(pnl),
        "mean_continuation_pnl": float(group["continuation_pnl_after_friction"].mean()) if n else 0.0,
        "mean_random_pnl": float(group["random_pnl_after_friction"].mean()) if n else 0.0,
        "mean_edge_vs_continuation": float(group["edge_vs_continuation"].mean()) if n else 0.0,
        "mean_edge_vs_random": float(group["edge_vs_random"].mean()) if n else 0.0,
        "p05_contrarian_pnl": float(pnl.quantile(0.05)) if n else 0.0,
        "p95_contrarian_pnl": float(pnl.quantile(0.95)) if n else 0.0,
        "min_contrarian_pnl": float(pnl.min()) if n else 0.0,
        "max_contrarian_pnl": float(pnl.max()) if n else 0.0,
        "max_drawdown_pips": max_drawdown(pnl),
        "mean_max_favorable_pips": float(group["max_favorable_pips"].mean()) if n else 0.0,
        "mean_max_adverse_pips": float(group["max_adverse_pips"].mean()) if n else 0.0,
        "p95_max_adverse_pips": float(group["max_adverse_pips"].quantile(0.95)) if n else 0.0,
    }


def aggregate_by(events_df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, group in events_df.groupby(list(group_cols), dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        rec.update(aggregate_group(group))
        rows.append(rec)
    return pd.DataFrame(rows)


def walkforward_label(year: int) -> str:
    if year in (2022, 2023):
        return "2022-2023_reference"
    if year == 2024:
        return "2024_validation"
    if year in (2025, 2026):
        return "2025-2026_test_oos"
    return "outside_walkforward"


def build_walkforward(events_df: pd.DataFrame) -> pd.DataFrame:
    wf_df = events_df.copy()
    wf_df["walkforward_window"] = wf_df["year"].map(walkforward_label)
    wf_df = wf_df[wf_df["walkforward_window"] != "outside_walkforward"]
    group_cols = [
        "walkforward_window",
        "hour",
        "excursion_pips",
        "reversion_target_pips",
        "adverse_continuation_pips",
        "future_horizon_bars",
    ]
    return aggregate_by(wf_df, group_cols)


def verdict_for_group(group: pd.DataFrame) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    oos = group[group["year"].isin([2025, 2026])]
    if oos.empty:
        return "FAIL", ["No 2025-2026 test/OOS observations."]
    oos_metrics = aggregate_group(oos)
    year_metrics = aggregate_by(group, ["year"])
    month_metrics = aggregate_by(group, ["year", "month"])
    positive_years = int((year_metrics["mean_contrarian_pnl"] > 0).sum()) if not year_metrics.empty else 0
    oos_total_abs = abs(float(oos_metrics["total_contrarian_pnl"]))
    month_dominance = 0.0
    if not month_metrics.empty and oos_total_abs > 0:
        month_dominance = float(month_metrics["total_contrarian_pnl"].abs().max() / oos_total_abs)
    tail_ratio = (
        abs(float(oos_metrics["p05_contrarian_pnl"])) / max(abs(float(oos_metrics["mean_contrarian_pnl"])), 1e-9)
        if oos_metrics["mean_contrarian_pnl"] != 0
        else float("inf")
    )

    fail_checks = [
        (oos_metrics["mean_contrarian_pnl"] <= 0, "OOS pnl after friction is not positive."),
        (oos_metrics["mean_edge_vs_random"] <= 0, "OOS edge vs random is not positive."),
        (oos_metrics["mean_edge_vs_continuation"] <= 0, "OOS edge vs continuation is not positive."),
        (oos_metrics["observations"] < 50, "OOS observations are insufficient (<50)."),
    ]
    failed = [reason for condition, reason in fail_checks if condition]
    if failed:
        return "FAIL", failed

    warn_checks = [
        (oos_metrics["mean_contrarian_pnl"] < 0.5, "OOS positive expectancy is weak (<0.5 pip/event)."),
        (oos_metrics["mean_edge_vs_random"] < 0.25, "OOS edge vs random is borderline (<0.25 pip/event)."),
        (oos_metrics["mean_edge_vs_continuation"] < 0.25, "OOS edge vs continuation is borderline (<0.25 pip/event)."),
        (positive_years < 3, "Positive performance is not stable across at least three years."),
        (month_dominance > 0.60, "Result may be concentrated in one month/year."),
        (tail_ratio > 20, "Tail risk is large relative to mean edge."),
        (oos_metrics["p95_max_adverse_pips"] > 2 * group["adverse_continuation_pips"].iloc[0], "Adverse tail is large."),
    ]
    warnings = [reason for condition, reason in warn_checks if condition]
    if warnings:
        return "WARN", warnings
    return "PASS", ["OOS expectancy and baseline edges are positive with acceptable stability and tail risk."]


def build_verdict(events_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["hour", "excursion_pips", "reversion_target_pips", "adverse_continuation_pips", "future_horizon_bars"]
    rows: List[Dict[str, object]] = []
    for keys, group in events_df.groupby(group_cols, dropna=False, sort=True):
        rec = dict(zip(group_cols, keys))
        verdict, reasons = verdict_for_group(group)
        oos = group[group["year"].isin([2025, 2026])]
        rec.update({f"oos_{k}": v for k, v in aggregate_group(oos).items()})
        rec["positive_years"] = int((aggregate_by(group, ["year"])["mean_contrarian_pnl"] > 0).sum())
        rec["verdict"] = verdict
        rec["reasons"] = " | ".join(reasons)
        rows.append(rec)
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "No rows."
    return df.head(max_rows).to_markdown(index=False)


def build_summary(
    output_path: Path,
    events_df: pd.DataFrame,
    by_hour_params: pd.DataFrame,
    by_hour: pd.DataFrame,
    by_year: pd.DataFrame,
    walkforward: pd.DataFrame,
    verdict: pd.DataFrame,
    cost_profile: str,
    friction_pips: float,
) -> None:
    lines = [
        "# USDJPY Pip Excursion Reversion by Hour",
        "",
        "Research-only study. No MT4/MT5 execution, live trading, lot sizing, or account-equity logic is included.",
        "",
        f"Cost profile: `{cost_profile}`; friction_pips = `{friction_pips:.2f}`.",
        f"Total parameterized events: `{len(events_df)}`.",
        "",
        "## Verdict distribution",
        markdown_table(verdict["verdict"].value_counts().rename_axis("verdict").reset_index(name="rows")),
        "",
        "## Best OOS rows by mean_contrarian_pnl (not an optimization instruction)",
        markdown_table(verdict.sort_values("oos_mean_contrarian_pnl", ascending=False)),
        "",
        "## By hour",
        markdown_table(by_hour.sort_values("mean_contrarian_pnl", ascending=False)),
        "",
        "## By year",
        markdown_table(by_year.sort_values("year")),
        "",
        "## Walk-forward aggregate by window",
        markdown_table(aggregate_by(events_df.assign(walkforward_window=events_df["year"].map(walkforward_label)).query("walkforward_window != 'outside_walkforward'"), ["walkforward_window"])),
        "",
        "## Notes",
        "- Events are selected using the first M15 candle inside each date/hour whose high/low confirms the excursion from the first open of that hour.",
        "- Entry is the close of the event candle; outcome simulation starts on subsequent future bars only.",
        "- If target and adverse thresholds are both reachable inside the same future M15 candle, the adverse threshold is counted first as a conservative no-tick-order assumption.",
        "- Deterministic random baseline is reproducible from a SHA-256 hash of symbol and event datetime metadata.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    friction_pips = friction_for_profile(args.cost_profile)

    df = load_ohlc(args.csv, args.symbol)
    base_events = find_base_events(df, EXCURSION_PIPS)
    events_df = build_event_rows(df, base_events, friction_pips)
    if events_df.empty:
        raise RuntimeError("No excursion events generated; check dataset and parameters.")

    by_hour_params = aggregate_by(
        events_df,
        ["hour", "excursion_pips", "reversion_target_pips", "adverse_continuation_pips", "future_horizon_bars"],
    )
    by_hour = aggregate_by(events_df, ["hour"])
    by_year = aggregate_by(events_df, ["year"])
    walkforward = build_walkforward(events_df)
    verdict = build_verdict(events_df)

    paths = {
        "events": output_dir / "usdjpy_reversion_events.csv",
        "by_hour_params": output_dir / "usdjpy_reversion_by_hour_params.csv",
        "by_hour": output_dir / "usdjpy_reversion_by_hour.csv",
        "by_year": output_dir / "usdjpy_reversion_by_year.csv",
        "walkforward": output_dir / "usdjpy_reversion_walkforward.csv",
        "verdict": output_dir / "usdjpy_reversion_verdict.csv",
        "summary": output_dir / "usdjpy_reversion_summary.md",
    }
    events_df.to_csv(paths["events"], index=False)
    by_hour_params.to_csv(paths["by_hour_params"], index=False)
    by_hour.to_csv(paths["by_hour"], index=False)
    by_year.to_csv(paths["by_year"], index=False)
    walkforward.to_csv(paths["walkforward"], index=False)
    verdict.to_csv(paths["verdict"], index=False)
    build_summary(
        paths["summary"],
        events_df,
        by_hour_params,
        by_hour,
        by_year,
        walkforward,
        verdict,
        args.cost_profile,
        friction_pips,
    )

    for label, path in paths.items():
        print(f"Saved {label}: {path}")


if __name__ == "__main__":
    main()
