#!/usr/bin/env python3
"""Deep descriptive stability check for strict survivor contexts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REQUIRED_COLS = ["datetime", "open", "high", "low", "close"]

COST_PROFILES: Dict[str, Dict[str, Dict[str, float]]] = {
    "low": {
        "EURUSD_GBPUSD": {
            "spread_a_pips": 0.8,
            "spread_b_pips": 1.2,
            "slippage_a_pips": 0.2,
            "slippage_b_pips": 0.3,
            "commission_equivalent_pips_total": 0.4,
        },
        "GBPAUD_GBPNZD": {
            "spread_a_pips": 2.5,
            "spread_b_pips": 3.5,
            "slippage_a_pips": 0.5,
            "slippage_b_pips": 0.7,
            "commission_equivalent_pips_total": 0.6,
        },
    },
    "conservative": {
        "EURUSD_GBPUSD": {
            "spread_a_pips": 1.2,
            "spread_b_pips": 1.8,
            "slippage_a_pips": 0.3,
            "slippage_b_pips": 0.5,
            "commission_equivalent_pips_total": 0.6,
        },
        "GBPAUD_GBPNZD": {
            "spread_a_pips": 3.5,
            "spread_b_pips": 5.0,
            "slippage_a_pips": 0.8,
            "slippage_b_pips": 1.0,
            "commission_equivalent_pips_total": 0.8,
        },
    },
    "high": {
        "EURUSD_GBPUSD": {
            "spread_a_pips": 2.0,
            "spread_b_pips": 2.5,
            "slippage_a_pips": 0.6,
            "slippage_b_pips": 0.8,
            "commission_equivalent_pips_total": 1.0,
        },
        "GBPAUD_GBPNZD": {
            "spread_a_pips": 5.0,
            "spread_b_pips": 7.0,
            "slippage_a_pips": 1.2,
            "slippage_b_pips": 1.5,
            "commission_equivalent_pips_total": 1.2,
        },
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep-check strict survivor context stability.")
    p.add_argument("--survivors", default="behavior_size_reports/survivor_contexts_strict.csv")
    p.add_argument("--csv-a", default="data/EURUSD_M15_MT5_5Y.csv")
    p.add_argument("--csv-b", default="data/GBPUSD_M15_MT5_5Y.csv")
    p.add_argument("--pair-name", default="EURUSD_GBPUSD")
    p.add_argument("--output-dir", default="survivor_context_reports")
    p.add_argument("--cost-profile", choices=["low", "conservative", "high"], default="conservative")
    return p.parse_args()


def pip_size(symbol: str) -> float:
    return 0.01 if symbol.upper().endswith("JPY") else 0.0001


def load_side(path: Path, suffix: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {path} missing columns: {missing}")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).sort_values("datetime").reset_index(drop=True)
    return df[["datetime", "open", "high", "low", "close"]].rename(columns={c: f"{c}_{suffix}" for c in ["open", "high", "low", "close"]})


def session_bucket_from_hour(hour: pd.Series) -> pd.Series:
    return pd.cut(hour, bins=[-1, 3, 6, 9, 12, 15, 18, 23], labels=["Asia early", "Asia late", "London open", "London mid", "New York open", "New York mid", "Late session"]).astype(str)


def abs_zscore_bucket(s: pd.Series) -> pd.Series:
    out = pd.cut(s, bins=[0, 1, 2, 3, np.inf], labels=["0_1", "1_2", "2_3", "3_plus"], include_lowest=True)
    return out.astype("object").where(~out.isna(), "unknown")


def build_future_arrays(series: pd.Series, horizon: int) -> np.ndarray:
    vals = series.to_numpy()
    n = len(vals)
    arr = np.full((n, horizon), np.nan)
    for k in range(1, horizon + 1):
        shifted = np.roll(vals, -k)
        shifted[n - k :] = np.nan
        arr[:, k - 1] = shifted
    return arr


def first_touch(z_row: np.ndarray, z_now: float, norm_level: float = 0.5, div_level: float = 3.0) -> tuple[str, float]:
    idx_norm = None
    idx_div = None
    for j, val in enumerate(z_row):
        if np.isnan(val):
            continue
        if idx_norm is None:
            if z_now >= 0 and val <= norm_level:
                idx_norm = j
            elif z_now < 0 and val >= -norm_level:
                idx_norm = j
        if idx_div is None and abs(val) >= div_level:
            idx_div = j
        if idx_norm is not None and idx_div is not None:
            break

    if idx_norm is None and idx_div is None:
        return "neither", np.nan
    if idx_norm is None:
        return "moved_further_to_3_first_vs_0_5", float(idx_div + 1)
    if idx_div is None or idx_norm < idx_div:
        return "normalized_0_5_first_vs_3", float(idx_norm + 1)
    if idx_norm == idx_div:
        return "both_same_bar", float(idx_norm + 1)
    return "moved_further_to_3_first_vs_0_5", float(idx_div + 1)


def round_trip_friction(cost_cfg: Dict[str, float]) -> float:
    return 2 * (cost_cfg["spread_a_pips"] + cost_cfg["spread_b_pips"]) + 2 * (cost_cfg["slippage_a_pips"] + cost_cfg["slippage_b_pips"]) + cost_cfg["commission_equivalent_pips_total"]


def longest_true_streak(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def max_rolling_rate(mask: np.ndarray, win: int) -> float:
    if len(mask) == 0:
        return np.nan
    s = pd.Series(mask.astype(float))
    return float(s.rolling(win, min_periods=min(win, len(mask))).mean().max())


def analyze_context(ctx: pd.Series, base: pd.DataFrame, symbol_a: str, symbol_b: str, friction: float) -> tuple[Dict[str, object], List[Dict[str, object]], str]:
    filt = (
        (base["pair_name"] == str(ctx["pair_name"]))
        & (base["window"] == int(ctx["window"]))
        & (base["horizon_bars"] == int(ctx["horizon_bars"]))
        & (base["session_bucket"] == str(ctx["session_bucket"]))
        & (base["hour"] == int(ctx["hour"]))
        & (base["abs_zscore_bucket"] == str(ctx["abs_zscore_bucket"]))
    )
    w = base.loc[filt].copy().sort_values("datetime")
    context_key = f"{ctx['pair_name']}|w{int(ctx['window'])}|h{int(ctx['horizon_bars'])}|{ctx['session_bucket']}|hr{int(ctx['hour'])}|{ctx['abs_zscore_bucket']}"
    if w.empty:
        row = {
            "pair_name": ctx["pair_name"], "window": int(ctx["window"]), "horizon_bars": int(ctx["horizon_bars"]),
            "session_bucket": ctx["session_bucket"], "hour": int(ctx["hour"]), "abs_zscore_bucket": ctx["abs_zscore_bucket"],
            "observations": 0, "normalization_0_5_first_vs_3_rate": np.nan, "moved_further_to_3_first_vs_0_5_rate": np.nan,
            "both_same_bar_rate": np.nan, "neither_rate": np.nan, "median_bars_to_first_touch": np.nan,
            "behavior_size_pips_proxy_median": np.nan, "behavior_size_pips_proxy_p25": np.nan, "behavior_size_pips_proxy_p75": np.nan,
            "behavior_size_pips_proxy_p90": np.nan, "estimated_round_trip_friction_pips": friction,
            "cost_share_of_median_behavior": np.nan, "years_count": 0, "positive_years_count": 0, "strong_years_count": 0,
            "worst_year_normalization_rate": np.nan, "best_year_normalization_rate": np.nan, "yearly_normalization_rate_std": np.nan,
            "worst_year_cost_share": np.nan, "best_year_cost_share": np.nan, "IS_observations": 0, "OOS_observations": 0,
            "IS_normalization_0_5_first_vs_3_rate": np.nan, "OOS_normalization_0_5_first_vs_3_rate": np.nan,
            "IS_behavior_size_pips_proxy_median": np.nan, "OOS_behavior_size_pips_proxy_median": np.nan,
            "IS_cost_share_of_median_behavior": np.nan, "OOS_cost_share_of_median_behavior": np.nan,
            "longest_consecutive_non_normalization_first": 0, "longest_consecutive_divergence_first": 0,
            "max_rolling_20_non_normalization_rate": np.nan, "max_rolling_50_non_normalization_rate": np.nan,
            "warning": "no matching bars"
        }
        return row, [], context_key

    pipa = pip_size(symbol_a)
    pipb = pip_size(symbol_b)
    events = []
    for _, r in w.iterrows():
        label, bars = first_touch(r["z_future"], r["spread_zscore"])
        ev = {"datetime": r["datetime"], "year": int(r["year"]), "label": label, "bars_to_first_touch_0_5_vs_3": bars}
        if label == "normalized_0_5_first_vs_3" and not np.isnan(bars):
            k = int(bars - 1)
            a_move = abs(r["close_a_future"][k] - r["close_a"]) / pipa
            b_move = abs(r["close_b_future"][k] - r["close_b"]) / pipb
            ev["behavior_size_pips_proxy"] = min(a_move, b_move)
            ev["sum_leg_move_pips_proxy"] = a_move + b_move
        else:
            ev["behavior_size_pips_proxy"] = np.nan
            ev["sum_leg_move_pips_proxy"] = np.nan
        events.append(ev)
    evdf = pd.DataFrame(events)

    obs = len(evdf)
    norm_rate = (evdf["label"] == "normalized_0_5_first_vs_3").mean()
    div_rate = (evdf["label"] == "moved_further_to_3_first_vs_0_5").mean()
    both_rate = (evdf["label"] == "both_same_bar").mean()
    neither_rate = (evdf["label"] == "neither").mean()
    med_beh = float(evdf["behavior_size_pips_proxy"].median()) if evdf["behavior_size_pips_proxy"].notna().any() else np.nan
    cost_share = friction / max(med_beh, 0.01) if not np.isnan(med_beh) else np.nan

    yearly_rows = []
    for y, g in evdf.groupby("year"):
        ym = float(g["behavior_size_pips_proxy"].median()) if g["behavior_size_pips_proxy"].notna().any() else np.nan
        ycs = friction / max(ym, 0.01) if not np.isnan(ym) else np.nan
        yearly_rows.append({
            "context_key": context_key, "year": int(y), "year_observations": int(len(g)),
            "year_normalization_0_5_first_vs_3_rate": float((g["label"] == "normalized_0_5_first_vs_3").mean()),
            "year_moved_further_to_3_first_vs_0_5_rate": float((g["label"] == "moved_further_to_3_first_vs_0_5").mean()),
            "year_behavior_size_pips_proxy_median": ym, "year_cost_share_of_median_behavior": ycs,
        })
    ydf = pd.DataFrame(yearly_rows)

    split = int(np.floor(obs * 0.7))
    is_df = evdf.iloc[:split]
    oos_df = evdf.iloc[split:]

    non_norm = (evdf["label"] != "normalized_0_5_first_vs_3").to_numpy()
    div_first = (evdf["label"] == "moved_further_to_3_first_vs_0_5").to_numpy()

    row = {
        "pair_name": ctx["pair_name"], "window": int(ctx["window"]), "horizon_bars": int(ctx["horizon_bars"]),
        "session_bucket": ctx["session_bucket"], "hour": int(ctx["hour"]), "abs_zscore_bucket": ctx["abs_zscore_bucket"],
        "observations": int(obs), "normalization_0_5_first_vs_3_rate": float(norm_rate), "moved_further_to_3_first_vs_0_5_rate": float(div_rate),
        "both_same_bar_rate": float(both_rate), "neither_rate": float(neither_rate),
        "median_bars_to_first_touch": float(evdf["bars_to_first_touch_0_5_vs_3"].median()),
        "behavior_size_pips_proxy_median": med_beh,
        "behavior_size_pips_proxy_p25": float(evdf["behavior_size_pips_proxy"].quantile(0.25)) if evdf["behavior_size_pips_proxy"].notna().any() else np.nan,
        "behavior_size_pips_proxy_p75": float(evdf["behavior_size_pips_proxy"].quantile(0.75)) if evdf["behavior_size_pips_proxy"].notna().any() else np.nan,
        "behavior_size_pips_proxy_p90": float(evdf["behavior_size_pips_proxy"].quantile(0.90)) if evdf["behavior_size_pips_proxy"].notna().any() else np.nan,
        "estimated_round_trip_friction_pips": friction,
        "cost_share_of_median_behavior": cost_share,
        "years_count": int(len(ydf)),
        "positive_years_count": int(((ydf["year_normalization_0_5_first_vs_3_rate"] >= 0.65) & (ydf["year_cost_share_of_median_behavior"] <= 1.00)).sum()) if not ydf.empty else 0,
        "strong_years_count": int(((ydf["year_normalization_0_5_first_vs_3_rate"] >= 0.75) & (ydf["year_cost_share_of_median_behavior"] <= 0.70)).sum()) if not ydf.empty else 0,
        "worst_year_normalization_rate": float(ydf["year_normalization_0_5_first_vs_3_rate"].min()) if not ydf.empty else np.nan,
        "best_year_normalization_rate": float(ydf["year_normalization_0_5_first_vs_3_rate"].max()) if not ydf.empty else np.nan,
        "yearly_normalization_rate_std": float(ydf["year_normalization_0_5_first_vs_3_rate"].std(ddof=0)) if not ydf.empty else np.nan,
        "worst_year_cost_share": float(ydf["year_cost_share_of_median_behavior"].max()) if not ydf.empty else np.nan,
        "best_year_cost_share": float(ydf["year_cost_share_of_median_behavior"].min()) if not ydf.empty else np.nan,
        "IS_observations": int(len(is_df)), "OOS_observations": int(len(oos_df)),
        "IS_normalization_0_5_first_vs_3_rate": float((is_df["label"] == "normalized_0_5_first_vs_3").mean()) if len(is_df) else np.nan,
        "OOS_normalization_0_5_first_vs_3_rate": float((oos_df["label"] == "normalized_0_5_first_vs_3").mean()) if len(oos_df) else np.nan,
        "IS_behavior_size_pips_proxy_median": float(is_df["behavior_size_pips_proxy"].median()) if is_df["behavior_size_pips_proxy"].notna().any() else np.nan,
        "OOS_behavior_size_pips_proxy_median": float(oos_df["behavior_size_pips_proxy"].median()) if oos_df["behavior_size_pips_proxy"].notna().any() else np.nan,
        "IS_cost_share_of_median_behavior": friction / max(float(is_df["behavior_size_pips_proxy"].median()), 0.01) if is_df["behavior_size_pips_proxy"].notna().any() else np.nan,
        "OOS_cost_share_of_median_behavior": friction / max(float(oos_df["behavior_size_pips_proxy"].median()), 0.01) if oos_df["behavior_size_pips_proxy"].notna().any() else np.nan,
        "longest_consecutive_non_normalization_first": longest_true_streak(non_norm),
        "longest_consecutive_divergence_first": longest_true_streak(div_first),
        "max_rolling_20_non_normalization_rate": max_rolling_rate(non_norm, 20),
        "max_rolling_50_non_normalization_rate": max_rolling_rate(non_norm, 50),
        "warning": ""
    }
    return row, yearly_rows, context_key


def main() -> None:
    args = parse_args()
    survivors_path = Path(args.survivors)
    if not survivors_path.exists():
        raise FileNotFoundError(f"Missing survivor contexts file: {survivors_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_out = out_dir / f"{args.pair_name}_survivor_deep_check.csv"
    md_out = out_dir / f"{args.pair_name}_survivor_deep_check.md"

    survivors = pd.read_csv(survivors_path)
    if survivors.empty:
        pd.DataFrame().to_csv(csv_out, index=False)
        md_out.write_text("# Survivor Context Deep Check\n\nWarning: descriptive survivor-context stability only, not strategy/profitability.\n\nInput survivor contexts count: 0\n\nNo survivor contexts found.\n", encoding="utf-8")
        print("No survivor contexts found. Wrote empty report outputs.")
        return

    required_survivor_cols = ["pair_name", "window", "horizon_bars", "session_bucket", "hour", "abs_zscore_bucket"]
    miss = [c for c in required_survivor_cols if c not in survivors.columns]
    if miss:
        raise ValueError(f"Survivor file missing required columns: {miss}")

    symbol_a, symbol_b = args.pair_name.split("_", 1)
    df_a = load_side(Path(args.csv_a), "a")
    df_b = load_side(Path(args.csv_b), "b")
    df = df_a.merge(df_b, on="datetime", how="inner")
    if df.empty:
        raise ValueError("No synchronized rows between csv-a and csv-b.")

    windows = sorted({int(w) for w in survivors["window"].dropna().astype(int).tolist()})
    horizons = sorted({int(h) for h in survivors["horizon_bars"].dropna().astype(int).tolist()})

    df["hour"] = df["datetime"].dt.hour
    df["year"] = df["datetime"].dt.year
    df["session_bucket"] = session_bucket_from_hour(df["hour"])
    df["log_price_a"] = np.log(df["close_a"])
    df["log_price_b"] = np.log(df["close_b"])
    df["returns_a"] = df["log_price_a"].diff()
    df["returns_b"] = df["log_price_b"].diff()
    df["pair_name"] = args.pair_name

    rows: List[pd.DataFrame] = []
    for w in windows:
        beta = df["returns_a"].rolling(w).cov(df["returns_b"]) / df["returns_b"].rolling(w).var()
        rel = df["log_price_a"] - beta * df["log_price_b"]
        mean = rel.rolling(w).mean()
        std = rel.rolling(w).std()
        z = (rel - mean) / std
        base = df[["datetime", "pair_name", "hour", "year", "session_bucket", "close_a", "close_b"]].copy()
        base["window"] = w
        base["rolling_beta"] = beta
        base["relative_spread"] = rel
        base["spread_mean"] = mean
        base["spread_std"] = std
        base["spread_zscore"] = z
        base["abs_zscore_bucket"] = abs_zscore_bucket(base["spread_zscore"].abs())

        for h in horizons:
            t = base.copy()
            t["horizon_bars"] = h
            t["z_future"] = list(build_future_arrays(t["spread_zscore"], h))
            t["close_a_future"] = list(build_future_arrays(t["close_a"], h))
            t["close_b_future"] = list(build_future_arrays(t["close_b"], h))
            rows.append(t)

    full = pd.concat(rows, ignore_index=True)
    full = full[~full["spread_zscore"].isna()].copy()

    cfg = COST_PROFILES[args.cost_profile].get(args.pair_name)
    if cfg is None:
        raise ValueError(f"No cost profile mapping for pair_name={args.pair_name} under cost_profile={args.cost_profile}")
    friction = round_trip_friction(cfg)

    summary_rows = []
    yearly_rows_all = []
    for _, ctx in survivors.iterrows():
        r, yrows, ckey = analyze_context(ctx, full, symbol_a, symbol_b, friction)
        r["context_key"] = ckey
        summary_rows.append(r)
        yearly_rows_all.extend(yrows)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(csv_out, index=False)
    yearly_df = pd.DataFrame(yearly_rows_all)

    lines = []
    lines.append("# Survivor Context Deep Check")
    lines.append("")
    lines.append("Warning: descriptive survivor-context stability only, not strategy/profitability.")
    lines.append("")
    lines.append(f"Input survivor contexts count: {len(survivors)}")
    lines.append("")
    lines.append("## Summary table of all survivor contexts")
    lines.append(summary_df.to_markdown(index=False) if not summary_df.empty else "No rows.")
    lines.append("")
    lines.append("## IS/OOS table")
    is_oos_cols = ["context_key", "IS_observations", "OOS_observations", "IS_normalization_0_5_first_vs_3_rate", "OOS_normalization_0_5_first_vs_3_rate", "IS_behavior_size_pips_proxy_median", "OOS_behavior_size_pips_proxy_median", "IS_cost_share_of_median_behavior", "OOS_cost_share_of_median_behavior"]
    lines.append(summary_df[is_oos_cols].to_markdown(index=False) if not summary_df.empty else "No rows.")
    lines.append("")
    lines.append("## Yearly stability table")
    lines.append(yearly_df.to_markdown(index=False) if not yearly_df.empty else "No yearly rows.")
    lines.append("")
    lines.append("## Sequence stress table")
    seq_cols = ["context_key", "longest_consecutive_non_normalization_first", "longest_consecutive_divergence_first", "max_rolling_20_non_normalization_rate", "max_rolling_50_non_normalization_rate", "warning"]
    lines.append(summary_df[seq_cols].to_markdown(index=False) if not summary_df.empty else "No rows.")
    lines.append("")
    lines.append("## Final interpretation")
    stable = summary_df[(summary_df["normalization_0_5_first_vs_3_rate"] >= 0.65) & (summary_df["cost_share_of_median_behavior"] <= 1.0)]
    degraded = summary_df[(summary_df["OOS_normalization_0_5_first_vs_3_rate"] < summary_df["IS_normalization_0_5_first_vs_3_rate"]) | (summary_df["worst_year_normalization_rate"] < 0.60)]
    friction_survive = summary_df[summary_df["cost_share_of_median_behavior"] <= 1.0]
    lines.append(f"- Contexts that remain stable: {len(stable)} of {len(summary_df)}.")
    lines.append(f"- Contexts that degrade in IS/OOS or yearly stability: {len(degraded)} of {len(summary_df)}.")
    lines.append(f"- Contexts passing conservative friction screening descriptively: {len(friction_survive)} of {len(summary_df)}.")

    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {csv_out}")
    print(f"Wrote: {md_out}")


if __name__ == "__main__":
    main()
