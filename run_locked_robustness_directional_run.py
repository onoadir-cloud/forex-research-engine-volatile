#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from run_directional_run_ex_ante_direction_test import (
    add_indicators,
    calc_directional_run,
    get_friction_pips,
    get_method_direction,
    infer_pip_size,
    normalize_ohlc,
    resample_tf,
    session_bucket,
)


@dataclass(frozen=True)
class LockedCandidate:
    name: str
    symbol: str
    timeframe: str
    direction_method: str
    session_bucket: str
    hour: int
    allowed_pullback_atr: float
    future_horizon_bars: int


SPLITS: List[Tuple[str, str, str]] = [
    ("train_reference", "2022-01-01", "2023-12-31"),
    ("validation", "2024-01-01", "2024-12-31"),
    ("test_oos", "2025-01-01", "2026-12-31"),
]


def build_candidate_events(base: pd.DataFrame, c: LockedCandidate, atr_period: int, cost_profile: str) -> pd.DataFrame:
    pip_size = infer_pip_size(c.symbol)
    friction = get_friction_pips(c.symbol, cost_profile)
    d = add_indicators(resample_tf(base, c.timeframe), atr_period, pip_size)

    rows = []
    for i in range(len(d)):
        atrp = d.at[i, "atr_pips"]
        if pd.isna(atrp) or atrp <= 0:
            continue
        dt = d.at[i, "datetime"]
        if dt.hour != c.hour:
            continue
        if session_bucket(dt.hour) != c.session_bucket:
            continue

        sel = get_method_direction(d.iloc[i], c.direction_method, c.timeframe)
        if sel is None:
            continue

        m_sel = calc_directional_run(d, i, c.future_horizon_bars, sel, c.allowed_pullback_atr, pip_size)
        m_opp = calc_directional_run(d, i, c.future_horizon_bars, -sel, c.allowed_pullback_atr, pip_size)
        rand_dir = get_method_direction(d.iloc[i], "random_baseline_deterministic", c.timeframe)
        m_rand = calc_directional_run(d, i, c.future_horizon_bars, rand_dir, c.allowed_pullback_atr, pip_size)
        if m_sel is None or m_opp is None or m_rand is None:
            continue

        sel_run = m_sel[0] - friction
        opp_run = m_opp[0] - friction
        rand_run = m_rand[0] - friction
        rows.append(
            {
                "candidate": c.name,
                "symbol": c.symbol,
                "timeframe": c.timeframe,
                "direction_method": c.direction_method,
                "datetime": dt,
                "year": dt.year,
                "year_month": dt.strftime("%Y-%m"),
                "hour": dt.hour,
                "session_bucket": c.session_bucket,
                "allowed_pullback_atr": c.allowed_pullback_atr,
                "future_horizon_bars": c.future_horizon_bars,
                "selected_run_after_friction": sel_run,
                "opposite_run_after_friction": opp_run,
                "random_mean_after_friction": rand_run,
                "edge_vs_opposite": sel_run - opp_run,
                "edge_vs_random": sel_run - rand_run,
            }
        )

    return pd.DataFrame(rows)


def summarize_period(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby(["candidate", key], dropna=False)
        .agg(
            observations=("selected_run_after_friction", "size"),
            selected_mean_after_friction=("selected_run_after_friction", "mean"),
            opposite_mean_after_friction=("opposite_run_after_friction", "mean"),
            random_mean_after_friction=("random_mean_after_friction", "mean"),
            edge_vs_opposite=("edge_vs_opposite", "mean"),
            edge_vs_random=("edge_vs_random", "mean"),
            p05_selected_run_after_friction=("selected_run_after_friction", lambda s: s.quantile(0.05)),
            worst_selected_run_after_friction=("selected_run_after_friction", "min"),
        )
        .reset_index()
        .sort_values(["candidate", key])
    )
    return out


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cand, g in df.groupby("candidate"):
        for split_name, start, end in SPLITS:
            mask = (g["datetime"] >= pd.Timestamp(start, tz="UTC")) & (g["datetime"] <= pd.Timestamp(end, tz="UTC"))
            s = g.loc[mask].copy()
            if s.empty:
                rows.append({"candidate": cand, "split": split_name, "observations": 0})
                continue
            rows.append(
                {
                    "candidate": cand,
                    "split": split_name,
                    "observations": len(s),
                    "selected_mean_after_friction": s["selected_run_after_friction"].mean(),
                    "edge_vs_random_mean": s["edge_vs_random"].mean(),
                    "edge_vs_random_median": s["edge_vs_random"].median(),
                    "pct_selected_beats_random": (s["edge_vs_random"] > 0).mean(),
                    "edge_vs_opposite_mean": s["edge_vs_opposite"].mean(),
                    "edge_vs_opposite_median": s["edge_vs_opposite"].median(),
                    "pct_selected_beats_opposite": (s["edge_vs_opposite"] > 0).mean(),
                }
            )
    return pd.DataFrame(rows)


def verdict(candidate_df: pd.DataFrame, yearly_df: pd.DataFrame, split_df: pd.DataFrame) -> str:
    y = yearly_df[yearly_df["candidate"] == candidate_df["candidate"].iloc[0]]
    s = split_df[split_df["candidate"] == candidate_df["candidate"].iloc[0]]
    if y.empty or s.empty:
        return "FAIL"
    pos_rand_years = (y["edge_vs_random"] > 0).mean()
    pos_opp_years = (y["edge_vs_opposite"] > 0).mean()
    oos = s[s["split"] == "test_oos"]
    oos_rand = float(oos["edge_vs_random_mean"].iloc[0]) if not oos.empty and "edge_vs_random_mean" in oos else np.nan
    oos_opp = float(oos["edge_vs_opposite_mean"].iloc[0]) if not oos.empty and "edge_vs_opposite_mean" in oos else np.nan
    year_concentration = y["observations"].max() / y["observations"].sum() if y["observations"].sum() else 1.0
    tail = y["worst_selected_run_after_friction"].min()
    mean_sel = y["selected_mean_after_friction"].mean()

    if (pd.isna(oos_rand) or pd.isna(oos_opp) or oos_rand <= 0 or oos_opp <= 0 or pos_rand_years < 0.5 or pos_opp_years < 0.5):
        return "FAIL"
    if pos_rand_years < 0.67 or pos_opp_years < 0.67 or year_concentration > 0.5 or abs(tail) > abs(mean_sel) * 20:
        return "WARN"
    return "PASS"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/EURUSD_M15_MT5_5Y.csv")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--cost-profile", choices=["low", "conservative", "high"], default="conservative")
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--output-dir", default="reports/locked_robustness")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base = normalize_ohlc(pd.read_csv(args.csv))

    candidates = [
        LockedCandidate("A_H1_momentum50_h20", args.symbol, "H1", "momentum_50", "Asia early", 0, 1.5, 20),
        LockedCandidate("A_H1_momentum50_h40", args.symbol, "H1", "momentum_50", "Asia early", 0, 1.5, 40),
        LockedCandidate("A_H1_momentum50_h80", args.symbol, "H1", "momentum_50", "Asia early", 0, 1.5, 80),
        LockedCandidate("B_H4_mapos50_h80", args.symbol, "H4", "ma_position_50", "New York mid", 16, 1.5, 80),
    ]

    all_events = [build_candidate_events(base, c, args.atr_period, args.cost_profile) for c in candidates]
    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    year_df = summarize_period(events, "year")
    month_df = summarize_period(events, "year_month")
    split_df = split_summary(events)

    events.to_csv(os.path.join(args.output_dir, "locked_candidates_events.csv"), index=False)
    year_df.to_csv(os.path.join(args.output_dir, "locked_candidates_yearly.csv"), index=False)
    month_df.to_csv(os.path.join(args.output_dir, "locked_candidates_monthly.csv"), index=False)
    split_df.to_csv(os.path.join(args.output_dir, "locked_candidates_walkforward.csv"), index=False)

    verdict_rows = []
    for cand in events["candidate"].unique():
        cdf = events[events["candidate"] == cand]
        v = verdict(cdf, year_df, split_df)
        verdict_rows.append({"candidate": cand, "verdict": v})
    verdict_df = pd.DataFrame(verdict_rows)
    verdict_df.to_csv(os.path.join(args.output_dir, "locked_candidates_verdict.csv"), index=False)

    md_path = os.path.join(args.output_dir, "locked_robustness_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Locked Method Robustness Summary\n\n")
        f.write("Research-only robustness validation (no execution logic, no optimization).\n\n")
        f.write("## Year-by-year stability\n\n")
        f.write(year_df.to_markdown(index=False) if not year_df.empty else "No yearly rows")
        f.write("\n\n## Month-by-month stability\n\n")
        f.write(month_df.to_markdown(index=False) if not month_df.empty else "No monthly rows")
        f.write("\n\n## Walk-forward splits\n\n")
        f.write(split_df.to_markdown(index=False) if not split_df.empty else "No split rows")
        f.write("\n\n## Final verdicts\n\n")
        f.write(verdict_df.to_markdown(index=False) if not verdict_df.empty else "No verdict rows")

    concise = "; ".join([f"{r['candidate']}={r['verdict']}" for r in verdict_rows])
    print(f"Locked robustness verdict: {concise}")


if __name__ == "__main__":
    main()
