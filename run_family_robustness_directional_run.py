#!/usr/bin/env python3
import argparse
import os
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
)

METHODS = [
    "momentum_20",
    "momentum_50",
    "ma_position_50",
    "ma_slope_20",
    "recent_range_position",
]
TIMEFRAMES = ["H1", "H4"]
HORIZONS = [20, 40, 80]
PULLBACK_ATR = 1.5

SPLITS: List[Tuple[str, str, str]] = [
    ("reference_2022_2023", "2022-01-01", "2023-12-31"),
    ("validation_2024", "2024-01-01", "2024-12-31"),
    ("test_oos_2025_2026", "2025-01-01", "2026-12-31"),
]

DATASETS = [
    ("EURUSD", "data/EURUSD_M15_MT5_5Y.csv"),
    ("GBPUSD", "data/GBPUSD_M15_MT5_5Y.csv"),
    ("USDJPY", "data/USDJPY_M15_MT5_5Y.csv"),
    ("GBPJPY", "data/GBPJPY_M15_MT5_5Y.csv"),
    ("GBPAUD", "data/GBPAUD_M15_MT5_5Y.csv"),
    ("GBPNZD", "data/GBPNZD_M15_MT5_5Y.csv"),
    ("AUDJPY", "data/AUDJPY_M15_MT5_5Y.csv"),
    ("EURJPY", "data/EURJPY_M15_MT5_5Y.csv"),
]


def build_events_for_symbol(symbol: str, csv_path: str, atr_period: int, cost_profile: str) -> pd.DataFrame:
    base = normalize_ohlc(pd.read_csv(csv_path))
    pip_size = infer_pip_size(symbol)
    friction = get_friction_pips(symbol, cost_profile)

    rows = []
    for tf in TIMEFRAMES:
        d = add_indicators(resample_tf(base, tf), atr_period, pip_size)
        for i in range(len(d)):
            atrp = d.at[i, "atr_pips"]
            if pd.isna(atrp) or atrp <= 0:
                continue
            dt = d.at[i, "datetime"]

            for method in METHODS:
                sel = get_method_direction(d.iloc[i], method, tf)
                if sel is None:
                    continue

                for horizon in HORIZONS:
                    m_sel = calc_directional_run(d, i, horizon, sel, PULLBACK_ATR, pip_size)
                    m_opp = calc_directional_run(d, i, horizon, -sel, PULLBACK_ATR, pip_size)
                    rand_dir = get_method_direction(d.iloc[i], "random_baseline_deterministic", tf)
                    m_rand = calc_directional_run(d, i, horizon, rand_dir, PULLBACK_ATR, pip_size)
                    if m_sel is None or m_opp is None or m_rand is None:
                        continue

                    sel_run = m_sel[0] - friction
                    opp_run = m_opp[0] - friction
                    rand_run = m_rand[0] - friction
                    rows.append(
                        {
                            "symbol": symbol,
                            "timeframe": tf,
                            "direction_method": method,
                            "future_horizon_bars": horizon,
                            "allowed_pullback_atr": PULLBACK_ATR,
                            "datetime": dt,
                            "year": dt.year,
                            "selected_run_after_friction": sel_run,
                            "opposite_run_after_friction": opp_run,
                            "random_run_after_friction": rand_run,
                            "edge_vs_opposite": sel_run - opp_run,
                            "edge_vs_random": sel_run - rand_run,
                        }
                    )
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby(keys, dropna=False)
        .agg(
            observations=("selected_run_after_friction", "size"),
            selected_mean_after_friction=("selected_run_after_friction", "mean"),
            opposite_mean_after_friction=("opposite_run_after_friction", "mean"),
            random_mean_after_friction=("random_run_after_friction", "mean"),
            edge_vs_opposite_mean=("edge_vs_opposite", "mean"),
            edge_vs_random_mean=("edge_vs_random", "mean"),
            pct_selected_beats_opposite=("edge_vs_opposite", lambda s: (s > 0).mean()),
            pct_selected_beats_random=("edge_vs_random", lambda s: (s > 0).mean()),
            p05_selected_run_after_friction=("selected_run_after_friction", lambda s: s.quantile(0.05)),
            worst_selected_run_after_friction=("selected_run_after_friction", "min"),
        )
        .reset_index()
    )
    return out


def walkforward(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()

    group_cols = ["direction_method", "timeframe", "future_horizon_bars"]
    for key, g in df.groupby(group_cols, dropna=False):
        for split_name, start, end in SPLITS:
            mask = (g["datetime"] >= pd.Timestamp(start, tz="UTC")) & (g["datetime"] <= pd.Timestamp(end, tz="UTC"))
            s = g.loc[mask]
            row = {
                "direction_method": key[0],
                "timeframe": key[1],
                "future_horizon_bars": key[2],
                "split": split_name,
                "observations": len(s),
            }
            if len(s):
                row.update(
                    {
                        "symbols_covered": s["symbol"].nunique(),
                        "selected_mean_after_friction": s["selected_run_after_friction"].mean(),
                        "edge_vs_opposite_mean": s["edge_vs_opposite"].mean(),
                        "edge_vs_random_mean": s["edge_vs_random"].mean(),
                        "pct_selected_beats_opposite": (s["edge_vs_opposite"] > 0).mean(),
                        "pct_selected_beats_random": (s["edge_vs_random"] > 0).mean(),
                    }
                )
            rows.append(row)

    return pd.DataFrame(rows)


def build_verdict(events: pd.DataFrame, method_df: pd.DataFrame, symbol_df: pd.DataFrame, year_df: pd.DataFrame, wf_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, m in method_df.iterrows():
        method = m["direction_method"]
        tf = m["timeframe"]
        h = m["future_horizon_bars"]

        w = wf_df[
            (wf_df["direction_method"] == method)
            & (wf_df["timeframe"] == tf)
            & (wf_df["future_horizon_bars"] == h)
        ]
        oos = w[w["split"] == "test_oos_2025_2026"]

        sy = symbol_df[
            (symbol_df["direction_method"] == method)
            & (symbol_df["timeframe"] == tf)
            & (symbol_df["future_horizon_bars"] == h)
            & (symbol_df["edge_vs_random_mean"] > 0)
            & (symbol_df["edge_vs_opposite_mean"] > 0)
        ]
        yr = year_df[
            (year_df["direction_method"] == method)
            & (year_df["timeframe"] == tf)
            & (year_df["future_horizon_bars"] == h)
        ]
        pos_years_rand = (yr["edge_vs_random_mean"] > 0).mean() if len(yr) else 0.0
        pos_years_opp = (yr["edge_vs_opposite_mean"] > 0).mean() if len(yr) else 0.0

        oos_edge_rand = float(oos["edge_vs_random_mean"].iloc[0]) if len(oos) else np.nan
        oos_edge_opp = float(oos["edge_vs_opposite_mean"].iloc[0]) if len(oos) else np.nan
        oos_symbols = int(oos["symbols_covered"].iloc[0]) if len(oos) and "symbols_covered" in oos else 0

        pass_cond = (
            pd.notna(oos_edge_rand)
            and pd.notna(oos_edge_opp)
            and oos_edge_rand > 0
            and oos_edge_opp > 0
            and oos_symbols >= 4
            and sy["symbol"].nunique() >= 4
            and pos_years_rand >= 0.60
            and pos_years_opp >= 0.60
        )

        verdict = "PASS" if pass_cond else "FAIL"
        reason = (
            "OOS beats random+opposite with multi-symbol and year stability"
            if pass_cond
            else "Random/opposite not beaten robustly in OOS, or concentrated by symbol/year"
        )

        rows.append(
            {
                "direction_method": method,
                "timeframe": tf,
                "future_horizon_bars": h,
                "verdict": verdict,
                "oos_edge_vs_random_mean": oos_edge_rand,
                "oos_edge_vs_opposite_mean": oos_edge_opp,
                "oos_symbols_covered": oos_symbols,
                "symbols_positive_both": sy["symbol"].nunique(),
                "positive_year_rate_vs_random": pos_years_rand,
                "positive_year_rate_vs_opposite": pos_years_opp,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cost-profile", choices=["low", "conservative", "high"], default="conservative")
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--output-dir", default="reports/family_robustness")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    events_all = []
    for symbol, path in DATASETS:
        events_all.append(build_events_for_symbol(symbol, path, args.atr_period, args.cost_profile))
    events = pd.concat(events_all, ignore_index=True) if events_all else pd.DataFrame()

    by_method = aggregate(events, ["direction_method", "timeframe", "future_horizon_bars", "allowed_pullback_atr"])
    by_symbol = aggregate(events, ["direction_method", "timeframe", "future_horizon_bars", "symbol"])
    by_year = aggregate(events, ["direction_method", "timeframe", "future_horizon_bars", "year"])
    wf = walkforward(events)
    verdict = build_verdict(events, by_method, by_symbol, by_year, wf)

    events.to_csv(os.path.join(args.output_dir, "family_events.csv"), index=False)
    by_method.to_csv(os.path.join(args.output_dir, "family_by_method.csv"), index=False)
    by_symbol.to_csv(os.path.join(args.output_dir, "family_by_symbol.csv"), index=False)
    by_year.to_csv(os.path.join(args.output_dir, "family_by_year.csv"), index=False)
    wf.to_csv(os.path.join(args.output_dir, "family_walkforward.csv"), index=False)
    verdict.to_csv(os.path.join(args.output_dir, "family_verdict.csv"), index=False)

    md = os.path.join(args.output_dir, "family_robustness_summary.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Family Robustness Directional-Run Summary\n\n")
        f.write("Research-only study. No EA, no live trading, no optimization, no top-context search.\n\n")
        f.write("## Scope\n")
        f.write(f"- Methods: {', '.join(METHODS)}\n")
        f.write(f"- Timeframes: {', '.join(TIMEFRAMES)}\n")
        f.write(f"- Horizons: {', '.join(map(str, HORIZONS))}\n")
        f.write(f"- Allowed pullback ATR: {PULLBACK_ATR}\n\n")

        f.write("## By method\n\n")
        f.write(by_method.to_markdown(index=False) if not by_method.empty else "No rows")
        f.write("\n\n## By symbol\n\n")
        f.write(by_symbol.to_markdown(index=False) if not by_symbol.empty else "No rows")
        f.write("\n\n## By year\n\n")
        f.write(by_year.to_markdown(index=False) if not by_year.empty else "No rows")
        f.write("\n\n## Walk-forward\n\n")
        f.write(wf.to_markdown(index=False) if not wf.empty else "No rows")
        f.write("\n\n## Verdict\n\n")
        f.write(verdict.to_markdown(index=False) if not verdict.empty else "No rows")

    print(f"Wrote family robustness outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
