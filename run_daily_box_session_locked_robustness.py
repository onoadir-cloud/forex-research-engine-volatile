#!/usr/bin/env python3
"""Locked-candidate robustness check for Daily Box Session-Open Sweep/Reclaim research.

This script is deliberately offline/research-only. It does not search for new best
rows, optimize parameters, alter session mappings, alter target modes, build an EA,
or add live-trading behavior. It reuses the existing Daily Box Session-Open
sweep/reclaim research logic and evaluates only the three post-hoc locked
candidates requested for robustness checking.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import pandas as pd

from run_daily_box_session_open_research import (
    DATASETS,
    EVENT_COLUMNS,
    Config,
    aggregate,
    baseline_prices,
    build_daily_boxes,
    detect_setup,
    deterministic_random_uses_strategy,
    find_session_open,
    friction_pips,
    infer_pip_size,
    manage_trade,
    markdown_table,
    normalize_ohlc_csv,
    parse_session_open_hours,
    risk_reward,
    stop_price,
    target_price,
)

LOCKED_CANDIDATES = [
    {
        "candidate_id": "Candidate A",
        "symbol": "USDJPY",
        "session_name": "NewYork_Open",
        "target_mode": "opposite_edge",
    },
    {
        "candidate_id": "Candidate B",
        "symbol": "USDJPY",
        "session_name": "Tokyo_Open",
        "target_mode": "mid",
    },
    {
        "candidate_id": "Candidate C",
        "symbol": "GBPJPY",
        "session_name": "London_Open",
        "target_mode": "mid",
    },
]

CANDIDATE_COLUMNS = ["candidate_id", "symbol", "session_name", "target_mode"]
YEARLY_COLUMNS = CANDIDATE_COLUMNS + [
    "year",
    "trades",
    "total_net_pnl",
    "expectancy",
    "profit_factor",
    "mean_edge_vs_random",
    "mean_edge_vs_continuation",
    "win_rate",
    "p05_net_pnl",
    "worst_trade",
    "max_drawdown_pips",
    "longest_loss_streak",
]
MONTHLY_COLUMNS = CANDIDATE_COLUMNS + [
    "year_month",
    "trades",
    "total_net_pnl",
    "expectancy",
    "profit_factor",
    "mean_edge_vs_random",
    "mean_edge_vs_continuation",
    "win_rate",
    "p05_net_pnl",
    "worst_trade",
]
WALKFORWARD_COLUMNS = CANDIDATE_COLUMNS + [
    "split",
    "from_year",
    "to_year",
    "trades",
    "total_net_pnl",
    "expectancy",
    "profit_factor",
    "mean_edge_vs_random",
    "mean_edge_vs_continuation",
    "win_rate",
    "p05_net_pnl",
    "worst_trade",
    "max_drawdown_pips",
    "longest_loss_streak",
]


def candidate_key(candidate: dict[str, str]) -> tuple[str, str, str]:
    return (candidate["symbol"], candidate["session_name"], candidate["target_mode"])


def scan_locked_candidate(candidate: dict[str, str], path: Path, cfg: Config) -> tuple[pd.DataFrame, list[str]]:
    """Scan exactly one locked symbol/session/target tuple using the existing rules."""
    symbol = candidate["symbol"]
    session_name = candidate["session_name"]
    target_mode = candidate["target_mode"]
    df, warnings = normalize_ohlc_csv(path)
    pip_size = infer_pip_size(symbol)
    friction = friction_pips(symbol, cfg.cost_profile)
    boxes, box_warnings = build_daily_boxes(df, pip_size)
    warnings.extend([f"{candidate['candidate_id']} {symbol}: {w}" for w in box_warnings[:25]])
    if len(box_warnings) > 25:
        warnings.append(f"{candidate['candidate_id']} {symbol}: {len(box_warnings) - 25} additional box warnings suppressed.")

    events: list[dict] = []
    session_hour = int(cfg.session_open_hours[session_name])
    for day, day_df in df.groupby("date", sort=True):
        if day not in boxes:
            continue
        day_df = day_df.sort_values("datetime").reset_index(drop=True)
        first_candle = pd.Timestamp(day_df.iloc[0]["datetime"])
        box = boxes[day]
        open_idx, session_dt = find_session_open(day_df, session_hour)
        if open_idx is None or session_dt is None:
            warnings.append(
                f"{candidate['candidate_id']} {symbol} {day} {session_name}: "
                f"no candle at/after configured session open hour {session_hour}."
            )
            continue

        scan_df = day_df.iloc[open_idx : open_idx + cfg.scan_window_bars].reset_index(drop=True)
        candidates = [setup for setup in (detect_setup(scan_df, "long", box), detect_setup(scan_df, "short", box)) if setup is not None]
        if not candidates:
            continue

        valid_candidates: list[dict] = []
        for setup in candidates:
            direction = setup["setup_direction"]
            entry = float(setup["entry_price"])
            sl = stop_price(direction, box, cfg.buffer_pips, pip_size)
            tp = target_price(direction, target_mode, box)
            risk, reward, rr = risk_reward(direction, entry, sl, tp, pip_size)
            if direction == "long" and tp <= entry:
                continue
            if direction == "short" and tp >= entry:
                continue
            if risk <= 0 or reward <= 0 or rr < cfg.risk_reward_min:
                continue
            item = dict(setup)
            item.update({"stop_loss": sl, "take_profit": tp, "risk_pips": risk, "reward_pips": reward, "rr_ratio": rr})
            valid_candidates.append(item)
        if not valid_candidates:
            continue

        valid_candidates.sort(key=lambda setup: setup["confirmation_datetime"])
        chosen = valid_candidates[0]
        skipped_competing = max(0, len(valid_candidates) - 1)
        direction = chosen["setup_direction"]
        entry = float(chosen["entry_price"])
        sl = float(chosen["stop_loss"])
        tp = float(chosen["take_profit"])
        management = manage_trade(
            day_df,
            chosen["confirmation_datetime"],
            direction,
            entry,
            sl,
            tp,
            pip_size,
            friction,
            cfg.force_close_hour,
            cfg.force_close_minute,
        )
        cont_direction, cont_sl, cont_tp = baseline_prices(direction, entry, chosen["risk_pips"], chosen["reward_pips"], pip_size)
        cont = manage_trade(
            day_df,
            chosen["confirmation_datetime"],
            cont_direction,
            entry,
            cont_sl,
            cont_tp,
            pip_size,
            friction,
            cfg.force_close_hour,
            cfg.force_close_minute,
        )
        if deterministic_random_uses_strategy(symbol, str(day), session_name, target_mode):
            random_pnl = management["net_pnl_pips"]
        else:
            random_pnl = cont["net_pnl_pips"]
        net_pnl = management["net_pnl_pips"]
        events.append(
            {
                "candidate_id": candidate["candidate_id"],
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
            }
        )
    return pd.DataFrame(events, columns=["candidate_id", *EVENT_COLUMNS]), warnings


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    return result[columns]


def aggregate_required(events: pd.DataFrame, group_cols: list[str], columns: list[str]) -> pd.DataFrame:
    grouped = aggregate(events, group_cols)
    return ensure_columns(grouped, columns)


def yearly_robustness(events: pd.DataFrame) -> pd.DataFrame:
    return aggregate_required(events, [*CANDIDATE_COLUMNS, "year"], YEARLY_COLUMNS)


def monthly_robustness(events: pd.DataFrame) -> pd.DataFrame:
    monthly = aggregate_required(events, [*CANDIDATE_COLUMNS, "month"], [*CANDIDATE_COLUMNS, "month", *MONTHLY_COLUMNS[len(CANDIDATE_COLUMNS) + 1 :]])
    if monthly.empty:
        return pd.DataFrame(columns=MONTHLY_COLUMNS)
    return monthly.rename(columns={"month": "year_month"})[MONTHLY_COLUMNS]


def walkforward_robustness(events: pd.DataFrame) -> pd.DataFrame:
    splits = [
        ("2022-2023 reference", 2022, 2023),
        ("2024 validation", 2024, 2024),
        ("2025-2026 test/OOS", 2025, 2026),
    ]
    rows: list[pd.DataFrame] = []
    for split, start_year, end_year in splits:
        segment = events[(events["year"] >= start_year) & (events["year"] <= end_year)] if not events.empty else events
        grouped = aggregate_required(segment, CANDIDATE_COLUMNS, CANDIDATE_COLUMNS + WALKFORWARD_COLUMNS[len(CANDIDATE_COLUMNS) + 3 :])
        if grouped.empty:
            grouped = pd.DataFrame(columns=CANDIDATE_COLUMNS + WALKFORWARD_COLUMNS[len(CANDIDATE_COLUMNS) + 3 :])
        grouped.insert(len(CANDIDATE_COLUMNS), "split", split)
        grouped.insert(len(CANDIDATE_COLUMNS) + 1, "from_year", start_year)
        grouped.insert(len(CANDIDATE_COLUMNS) + 2, "to_year", end_year)
        rows.append(grouped)
    wf = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=WALKFORWARD_COLUMNS)
    return ensure_columns(wf, WALKFORWARD_COLUMNS)


def _ratio_best_positive_month(oos_months: pd.DataFrame, total_oos: float) -> float:
    if oos_months.empty or total_oos <= 0:
        return 0.0
    return float(oos_months["total_net_pnl"].clip(lower=0).max() / total_oos)


def verdicts(wf: pd.DataFrame, yearly: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for candidate in LOCKED_CANDIDATES:
        key = candidate_key(candidate)
        mask = (
            (wf["candidate_id"] == candidate["candidate_id"])
            & (wf["symbol"] == key[0])
            & (wf["session_name"] == key[1])
            & (wf["target_mode"] == key[2])
            & (wf["split"] == "2025-2026 test/OOS")
        )
        oos = wf[mask]
        candidate_months = monthly[
            (monthly["candidate_id"] == candidate["candidate_id"])
            & (monthly["symbol"] == key[0])
            & (monthly["session_name"] == key[1])
            & (monthly["target_mode"] == key[2])
        ].copy()
        if not candidate_months.empty:
            candidate_months = candidate_months[candidate_months["year_month"].astype(str).str[:4].astype(int).between(2025, 2026)]
        candidate_years = yearly[
            (yearly["candidate_id"] == candidate["candidate_id"])
            & (yearly["symbol"] == key[0])
            & (yearly["session_name"] == key[1])
            & (yearly["target_mode"] == key[2])
        ].copy()
        if not candidate_years.empty:
            candidate_years = candidate_years[candidate_years["trades"].fillna(0).astype(int) > 0]

        if oos.empty:
            rows.append({**candidate, "verdict": "FAIL", "recommendation": "reject", "reason": "No OOS observations were generated."})
            continue

        row = oos.iloc[0]
        trades = int(row["trades"])
        total = float(row["total_net_pnl"])
        expectancy = float(row["expectancy"])
        pf = float(row["profit_factor"])
        edge_random = float(row["mean_edge_vs_random"])
        edge_cont = float(row["mean_edge_vs_continuation"])
        dd = float(row["max_drawdown_pips"])
        streak = int(row["longest_loss_streak"])
        month_profit_ratio = _ratio_best_positive_month(candidate_months, total)
        one_month_explains_most = month_profit_ratio > 0.50
        one_month_dominates = month_profit_ratio > 0.35
        oos_years = candidate_years[candidate_years["year"].astype(int).between(2025, 2026)] if not candidate_years.empty else candidate_years
        positive_oos_years = int((oos_years["total_net_pnl"].astype(float) > 0).sum()) if not oos_years.empty else 0
        required_positive_years = max(1, math.ceil(len(oos_years) / 2)) if not oos_years.empty else 1
        positive_or_acceptable_years = positive_oos_years >= required_positive_years
        drawdown_high = total > 0 and abs(dd) > max(250.0, total * 1.5)
        loss_streak_high = streak > 10
        borderline_edge = 0 < edge_random < 1.0 or 0 < edge_cont < 1.0

        fail_reasons: list[str] = []
        if trades < 30:
            fail_reasons.append("OOS trades < 30")
        if total <= 0:
            fail_reasons.append("OOS total_net_pnl <= 0")
        if expectancy <= 0:
            fail_reasons.append("OOS expectancy <= 0")
        if pf <= 1:
            fail_reasons.append("OOS profit_factor <= 1")
        if edge_random <= 0:
            fail_reasons.append("OOS edge vs random <= 0")
        if edge_cont <= 0:
            fail_reasons.append("OOS edge vs continuation <= 0")
        if one_month_explains_most:
            fail_reasons.append("one month explains most OOS profit")

        pass_core = (
            trades >= 50
            and total > 0
            and expectancy > 0
            and pf > 1.15
            and edge_random > 0
            and edge_cont > 0
            and positive_or_acceptable_years
            and not one_month_dominates
            and not drawdown_high
            and not loss_streak_high
        )
        warn_reasons: list[str] = []
        if total > 0 and trades < 50:
            warn_reasons.append("OOS is positive but trades < 50")
        if total > 0 and one_month_dominates:
            warn_reasons.append("OOS is positive but one month dominates")
        if borderline_edge:
            warn_reasons.append("edge vs random or continuation is borderline")
        if drawdown_high:
            warn_reasons.append("drawdown is high")
        if loss_streak_high:
            warn_reasons.append("loss streak is high")
        if not positive_or_acceptable_years:
            warn_reasons.append("not positive or acceptable in most OOS years")
        if pf <= 1.15:
            warn_reasons.append("profit factor is not above the 1.15 PASS threshold")

        if pass_core:
            verdict = "PASS"
            recommendation = "robustness passed; still post-hoc research only"
            reasons = ["OOS passes trade-count, pnl, expectancy, profit-factor, edge, yearly stability, concentration, and risk checks."]
        elif fail_reasons:
            verdict = "FAIL"
            recommendation = "reject locked candidate"
            reasons = fail_reasons
        else:
            verdict = "WARN"
            recommendation = "do not approve; review robustness concerns"
            reasons = warn_reasons or ["OOS is positive but robustness is mixed."]

        rows.append(
            {
                **candidate,
                "verdict": verdict,
                "recommendation": recommendation,
                "reason": "; ".join(reasons),
                "oos_trades": trades,
                "oos_total_net_pnl": total,
                "oos_expectancy": expectancy,
                "oos_profit_factor": pf,
                "oos_mean_edge_vs_random": edge_random,
                "oos_mean_edge_vs_continuation": edge_cont,
                "oos_win_rate": float(row["win_rate"]),
                "oos_p05_net_pnl": float(row["p05_net_pnl"]),
                "oos_worst_trade": float(row["worst_trade"]),
                "oos_max_drawdown_pips": dd,
                "oos_longest_loss_streak": streak,
                "positive_oos_years": positive_oos_years,
                "oos_years_with_trades": int(len(oos_years)),
                "largest_positive_month_share_of_oos_profit": month_profit_ratio,
                "one_month_explains_most_profit": bool(one_month_explains_most),
                "one_month_dominates_profit": bool(one_month_dominates),
                "drawdown_high": bool(drawdown_high),
                "loss_streak_high": bool(loss_streak_high),
                "post_hoc_only": True,
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    path: Path,
    cfg: Config,
    warnings: list[str],
    events: pd.DataFrame,
    yearly: pd.DataFrame,
    monthly: pd.DataFrame,
    wf: pd.DataFrame,
    verdict_df: pd.DataFrame,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Locked Daily Box Session-Open Sweep/Reclaim robustness\n\n")
        handle.write("## Scope and warning\n")
        handle.write(
            "The broad Daily Box Session-Open family failed overall. These rows are post-hoc locked "
            "candidates for robustness checking only, not approved strategies. This script does not build "
            "an EA, add live trading, optimize parameters, search for new best rows, alter strategy rules, "
            "change session mapping, or change target modes.\n\n"
        )
        handle.write("## Locked candidates\n")
        handle.write(markdown_table(pd.DataFrame(LOCKED_CANDIDATES), 10) + "\n")
        handle.write("## Locked configuration\n")
        config_rows = pd.DataFrame(
            [
                {"setting": "cost_profile", "value": cfg.cost_profile},
                {"setting": "scan_window_bars", "value": cfg.scan_window_bars},
                {"setting": "buffer_pips", "value": cfg.buffer_pips},
                {"setting": "risk_reward_min", "value": cfg.risk_reward_min},
                {"setting": "force_close_hour", "value": cfg.force_close_hour},
                {"setting": "force_close_minute", "value": cfg.force_close_minute},
                {"setting": "events", "value": len(events)},
            ]
        )
        handle.write(markdown_table(config_rows, 20) + "\n")
        handle.write("## Walk-forward splits\n")
        split_rows = pd.DataFrame(
            [
                {"split": "2022-2023 reference", "from_year": 2022, "to_year": 2023},
                {"split": "2024 validation", "from_year": 2024, "to_year": 2024},
                {"split": "2025-2026 test/OOS", "from_year": 2025, "to_year": 2026},
            ]
        )
        handle.write(markdown_table(split_rows, 10) + "\n")
        handle.write("## Verdict rules\n")
        handle.write(
            "PASS requires OOS trades >= 50, positive OOS net pnl and expectancy, profit factor > 1.15, "
            "positive edge versus random and continuation baselines, positive/acceptable OOS years, no one-month "
            "profit dependence, and acceptable drawdown/loss streak. FAIL is assigned for the requested hard "
            "failure conditions, including trades < 30, non-positive OOS pnl/expectancy/edges, profit factor <= 1, "
            "or one month explaining most profit. WARN covers positive but under-sampled, concentrated, borderline, "
            "or high-risk OOS results.\n\n"
        )
        handle.write("## Candidate verdicts\n")
        verdict_cols = [
            "candidate_id",
            "symbol",
            "session_name",
            "target_mode",
            "verdict",
            "reason",
            "oos_trades",
            "oos_total_net_pnl",
            "oos_expectancy",
            "oos_profit_factor",
            "oos_mean_edge_vs_random",
            "oos_mean_edge_vs_continuation",
            "largest_positive_month_share_of_oos_profit",
        ]
        handle.write(markdown_table(verdict_df[[c for c in verdict_cols if c in verdict_df.columns]], 10) + "\n")
        handle.write("## Walk-forward robustness\n")
        handle.write(markdown_table(wf, 30) + "\n")
        handle.write("## Yearly robustness\n")
        handle.write(markdown_table(yearly, 30) + "\n")
        handle.write("## Monthly robustness (first 30 rows)\n")
        handle.write(markdown_table(monthly, 30) + "\n")
        if warnings:
            handle.write("## Warnings / audit notes\n")
            for warning in warnings[:100]:
                handle.write(f"- {warning}\n")
            if len(warnings) > 100:
                handle.write(f"- {len(warnings) - 100} additional warnings suppressed.\n")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Locked robustness check for Daily Box Session-Open Sweep/Reclaim candidates.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="reports/daily_box_session_locked_robustness")
    parser.add_argument("--scan-window-bars", type=int, default=16)
    parser.add_argument("--force-close-hour", type=int, default=23)
    parser.add_argument("--force-close-minute", type=int, default=45)
    parser.add_argument("--buffer-pips", type=float, default=0)
    parser.add_argument("--risk-reward-min", type=float, default=1.0)
    parser.add_argument("--session-open-hours", default=None, help="JSON string or JSON file path overriding default session open hours.")
    args = parser.parse_args()
    if args.scan_window_bars <= 0:
        raise ValueError("--scan-window-bars must be positive")
    return Config(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        cost_profile="conservative",
        scan_window_bars=args.scan_window_bars,
        force_close_hour=args.force_close_hour,
        force_close_minute=args.force_close_minute,
        buffer_pips=args.buffer_pips,
        risk_reward_min=args.risk_reward_min,
        session_open_hours=parse_session_open_hours(args.session_open_hours),
        min_oos_observations=50,
    )


def run_locked_robustness(cfg: Config) -> dict[str, Path]:
    cfg = replace(cfg, cost_profile="conservative", min_oos_observations=50)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    all_events: list[pd.DataFrame] = []

    for candidate in LOCKED_CANDIDATES:
        symbol = candidate["symbol"]
        path = cfg.data_dir / DATASETS[symbol]
        if not path.exists():
            warnings.append(f"{candidate['candidate_id']} {symbol}: missing dataset {path}; skipped.")
            continue
        events, candidate_warnings = scan_locked_candidate(candidate, path, cfg)
        all_events.append(events)
        warnings.extend(candidate_warnings)

    event_columns = ["candidate_id", *EVENT_COLUMNS]
    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame(columns=event_columns)
    if not events_df.empty:
        events_df = events_df.sort_values(["confirmation_datetime", "candidate_id", "symbol", "session_name", "target_mode"]).reset_index(drop=True)

    yearly = yearly_robustness(events_df)
    monthly = monthly_robustness(events_df)
    wf = walkforward_robustness(events_df)
    verdict_df = verdicts(wf, yearly, monthly)

    outputs = {
        "events": cfg.output_dir / "locked_daily_box_events.csv",
        "yearly": cfg.output_dir / "locked_daily_box_yearly.csv",
        "monthly": cfg.output_dir / "locked_daily_box_monthly.csv",
        "walkforward": cfg.output_dir / "locked_daily_box_walkforward.csv",
        "verdict": cfg.output_dir / "locked_daily_box_verdict.csv",
        "summary": cfg.output_dir / "locked_daily_box_summary.md",
    }
    events_df.to_csv(outputs["events"], index=False)
    yearly.to_csv(outputs["yearly"], index=False)
    monthly.to_csv(outputs["monthly"], index=False)
    wf.to_csv(outputs["walkforward"], index=False)
    verdict_df.to_csv(outputs["verdict"], index=False)
    write_summary(outputs["summary"], cfg, warnings, events_df, yearly, monthly, wf, verdict_df)
    return outputs


def main() -> None:
    cfg = parse_args()
    outputs = run_locked_robustness(cfg)
    print("Locked Daily Box Session-Open robustness check complete.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
