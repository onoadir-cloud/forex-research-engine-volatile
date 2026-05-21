#!/usr/bin/env python3
"""Standalone Pairs / Relative Behavior Atlas.

Descriptive relative-behavior analysis for synchronized instrument pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PIP_SIZES = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "AUDUSD": 0.0001,
    "NZDUSD": 0.0001,
    "USDJPY": 0.01,
    "EURJPY": 0.01,
    "GBPJPY": 0.01,
}

REQUIRED_COLS = ["datetime", "open", "high", "low", "close"]

GROUPINGS = [
    ["hour"],
    ["session_bucket"],
    ["day_of_week"],
    ["correlation_regime"],
    ["beta_stability"],
    ["zscore_bucket"],
    ["abs_zscore_bucket"],
    ["session_bucket", "abs_zscore_bucket"],
    ["hour", "abs_zscore_bucket"],
    ["session_bucket", "correlation_regime"],
    ["session_bucket", "beta_stability"],
    ["correlation_regime", "abs_zscore_bucket"],
    ["beta_stability", "abs_zscore_bucket"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pairs relative behavior atlas.")
    parser.add_argument("--csv-a", default="data/EURUSD_M15_MT5_5Y.csv")
    parser.add_argument("--csv-b", default="data/GBPUSD_M15_MT5_5Y.csv")
    parser.add_argument("--symbol-a", default="EURUSD")
    parser.add_argument("--symbol-b", default="GBPUSD")
    parser.add_argument("--base-timeframe", default="M15")
    parser.add_argument("--output-dir", default="pairs_behavior_atlas_reports")
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--write-row-level", action="store_true", default=False)
    return parser.parse_args()


def get_pip_size(symbol: str) -> float:
    s = symbol.upper()
    if s in PIP_SIZES:
        return PIP_SIZES[s]
    print(f"[warning] Unknown symbol {symbol}; using fallback pip size 0.0001")
    return 0.0001


def load_single_csv(path: Path, suffix: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")

    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {path} is missing required columns: {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    return df[["datetime", "open", "high", "low", "close"]].rename(
        columns={c: f"{c}_{suffix}" for c in ["open", "high", "low", "close"]}
    )


def session_bucket(hour: pd.Series) -> pd.Series:
    return pd.cut(
        hour,
        bins=[-1, 3, 6, 9, 12, 15, 18, 23],
        labels=[
            "Asia early",
            "Asia late",
            "London open",
            "London mid",
            "New York open",
            "New York mid",
            "Late session",
        ],
    ).astype(str)


def correlation_regime(series: pd.Series) -> pd.Series:
    out = np.select(
        [
            series >= 0.70,
            (series >= 0.40) & (series < 0.70),
            (series > -0.20) & (series < 0.40),
            series <= -0.20,
        ],
        ["high_positive", "medium_positive", "low_or_broken", "negative"],
        default=np.nan,
    )
    return pd.Series(out, index=series.index, dtype="object")


def zscore_bucket(series: pd.Series) -> pd.Series:
    bins = [-np.inf, -3, -2, -1, 1, 2, 3, np.inf]
    labels = [
        "below_minus_3",
        "minus_3_to_minus_2",
        "minus_2_to_minus_1",
        "minus_1_to_1",
        "plus_1_to_2",
        "plus_2_to_3",
        "above_plus_3",
    ]
    return pd.cut(series, bins=bins, labels=labels).astype(str)


def abs_zscore_bucket(series: pd.Series) -> pd.Series:
    bins = [0, 1, 2, 3, np.inf]
    labels = ["0_1", "1_2", "2_3", "3_plus"]
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True).astype(str)


def first_touch_metrics(z_values: np.ndarray, current_z: np.ndarray, level: float):
    n, h = z_values.shape
    bars = np.full(n, np.nan)
    max_abs_before = np.full(n, np.nan)

    if level == 0.0:
        cond = np.where(
            current_z[:, None] >= 0,
            z_values <= level,
            z_values >= level,
        )
    else:
        cond = np.where(
            current_z[:, None] >= 0,
            z_values <= level,
            z_values >= -level,
        )

    touched = cond.any(axis=1)
    first = np.argmax(cond, axis=1)
    bars[touched] = first[touched] + 1

    for i in np.where(touched)[0]:
        end_idx = int(first[i]) + 1
        max_abs_before[i] = np.nanmax(np.abs(z_values[i, :end_idx]))

    return bars, max_abs_before


def build_future_arrays(series: pd.Series, horizon: int) -> np.ndarray:
    vals = series.to_numpy()
    n = len(vals)
    arr = np.full((n, horizon), np.nan)
    for k in range(1, horizon + 1):
        shifted = np.roll(vals, -k)
        shifted[n - k :] = np.nan
        arr[:, k - 1] = shifted
    return arr


def summarize_group(base: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    gb = base.groupby(keys, dropna=False)
    out = gb.agg(
        observations=("valid_row", "sum"),
        mean_corr=("rolling_corr", "mean"),
        median_corr=("rolling_corr", "median"),
        mean_beta=("rolling_beta", "mean"),
        median_beta=("rolling_beta", "median"),
        mean_abs_zscore=("zscore_abs", "mean"),
        median_abs_zscore=("zscore_abs", "median"),
        p75_abs_zscore=("zscore_abs", lambda s: np.nanpercentile(s, 75)),
        p90_abs_zscore=("zscore_abs", lambda s: np.nanpercentile(s, 90)),
        p95_abs_zscore=("zscore_abs", lambda s: np.nanpercentile(s, 95)),
        normalized_to_1_rate=("normalized_to_1", "mean"),
        normalized_to_0_5_rate=("normalized_to_0_5", "mean"),
        normalized_to_0_rate=("normalized_to_0", "mean"),
        moved_further_to_abs_z_3_rate=("moved_further_to_abs_z_3", "mean"),
        moved_further_to_abs_z_4_rate=("moved_further_to_abs_z_4", "mean"),
        median_bars_to_normalized_1=("bars_to_normalized_1", "median"),
        median_bars_to_normalized_0_5=("bars_to_normalized_0_5", "median"),
        median_bars_to_normalized_0=("bars_to_normalized_0", "median"),
        mean_future_abs_zscore_min=("future_abs_zscore_min", "mean"),
        mean_future_abs_zscore_max=("future_abs_zscore_max", "mean"),
    ).reset_index()

    is_stats = gb.apply(
        lambda g: pd.Series(
            {
                "IS_observations": int((g["split"] == "IS").sum()),
                "OOS_observations": int((g["split"] == "OOS").sum()),
                "IS_normalized_to_0_5_rate": g.loc[g["split"] == "IS", "normalized_to_0_5"].mean(),
                "OOS_normalized_to_0_5_rate": g.loc[g["split"] == "OOS", "normalized_to_0_5"].mean(),
                "IS_moved_further_to_abs_z_3_rate": g.loc[g["split"] == "IS", "moved_further_to_abs_z_3"].mean(),
                "OOS_moved_further_to_abs_z_3_rate": g.loc[g["split"] == "OOS", "moved_further_to_abs_z_3"].mean(),
                "WF_normalized_to_0_5_std": g.groupby("wf_split")["normalized_to_0_5"].mean().std(),
                "WF_moved_further_to_abs_z_3_std": g.groupby("wf_split")["moved_further_to_abs_z_3"].mean().std(),
            }
        )
    ).reset_index()

    return out.merge(is_stats, on=keys, how="left")


def main() -> int:
    start = time.time()
    args = parse_args()

    windows = [20, 50, 100] if args.preset == "quick" else [20, 50, 100, 200]
    horizons = [5, 10, 20, 40] if args.preset == "quick" else [5, 10, 20, 40, 80]

    try:
        df_a = load_single_csv(Path(args.csv_a), "a")
        df_b = load_single_csv(Path(args.csv_b), "b")
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}")
        return 1

    df = df_a.merge(df_b, on="datetime", how="inner")
    if args.max_rows and args.max_rows > 0:
        df = df.tail(args.max_rows).copy()
    df = df.reset_index(drop=True)

    if df.empty:
        print("[error] No synchronized rows after inner join on datetime.")
        return 1

    _ = get_pip_size(args.symbol_a)
    _ = get_pip_size(args.symbol_b)

    df["log_price_a"] = np.log(df["close_a"])
    df["log_price_b"] = np.log(df["close_b"])
    df["return_a"] = df["log_price_a"].diff()
    df["return_b"] = df["log_price_b"].diff()

    dt = df["datetime"]
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["session_bucket"] = session_bucket(df["hour"])

    split_idx = int(len(df) * 0.7)
    df["split"] = np.where(df.index < split_idx, "IS", "OOS")
    wf_size = len(df) // 3
    df["wf_split"] = np.select(
        [df.index < wf_size, (df.index >= wf_size) & (df.index < 2 * wf_size), df.index >= 2 * wf_size],
        ["WF1", "WF2", "WF3"],
        default="WF3",
    )

    grouped_frames = []

    for w in windows:
        print(f"[progress] computing features for window={w}")
        corr_col = f"rolling_corr_returns_{w}"
        beta_col = f"rolling_beta_{w}"
        beta_change_col = f"beta_change_{w}"
        beta_abs_change_col = f"beta_abs_change_{w}"
        rel_spread_col = f"relative_spread_{w}"
        spread_mean_col = f"spread_mean_{w}"
        spread_std_col = f"spread_std_{w}"
        z_col = f"spread_zscore_{w}"

        df[corr_col] = df["return_a"].rolling(w).corr(df["return_b"])
        cov = df["return_a"].rolling(w).cov(df["return_b"])
        var_b = df["return_b"].rolling(w).var()
        df[beta_col] = cov / var_b.replace(0, np.nan)
        df[beta_change_col] = df[beta_col].diff()
        df[beta_abs_change_col] = df[beta_change_col].abs()
        df[rel_spread_col] = df["log_price_a"] - df[beta_col] * df["log_price_b"]
        df[spread_mean_col] = df[rel_spread_col].rolling(w).mean()
        df[spread_std_col] = df[rel_spread_col].rolling(w).std()
        df[z_col] = (df[rel_spread_col] - df[spread_mean_col]) / df[spread_std_col].replace(0, np.nan)
        df[f"zscore_abs_{w}"] = df[z_col].abs()
        df[f"zscore_change_{w}"] = df[z_col].diff()

        df[f"correlation_regime_{w}"] = correlation_regime(df[corr_col])

        beta_change_roll = df[beta_abs_change_col].rolling(w).mean()
        q1 = beta_change_roll.quantile(1 / 3)
        q2 = beta_change_roll.quantile(2 / 3)
        df[f"beta_stability_{w}"] = np.select(
            [beta_change_roll <= q1, (beta_change_roll > q1) & (beta_change_roll <= q2), beta_change_roll > q2],
            ["low_change", "medium_change", "high_change"],
            default=np.nan,
        )

        df[f"zscore_bucket_{w}"] = zscore_bucket(df[z_col])
        df[f"abs_zscore_bucket_{w}"] = abs_zscore_bucket(df[f"zscore_abs_{w}"])

        lead_a = {}
        lead_b = {}
        for lag in [1, 2, 4]:
            lead_a[lag] = df["return_a"].corr(df["return_b"].shift(-lag))
            lead_b[lag] = df["return_b"].corr(df["return_a"].shift(-lag))

        for h in horizons:
            print(f"[progress] window={w}, horizon={h}")
            z_future = build_future_arrays(df[z_col], h)
            z_now = df[z_col].to_numpy()
            abs_future = np.abs(z_future)

            base = f"_{w}_{h}"
            df[f"future_zscore_change{base}"] = np.nanmean(z_future - z_now[:, None], axis=1)
            df[f"future_abs_zscore_min{base}"] = np.nanmin(abs_future, axis=1)
            df[f"future_abs_zscore_max{base}"] = np.nanmax(abs_future, axis=1)
            df[f"future_zscore_max_up{base}"] = np.nanmax(z_future - z_now[:, None], axis=1)
            df[f"future_zscore_max_down{base}"] = np.nanmin(z_future - z_now[:, None], axis=1)

            pos = z_now >= 0
            df[f"normalized_to_1{base}"] = np.where(pos, np.nanmin(z_future, axis=1) <= 1.0, np.nanmax(z_future, axis=1) >= -1.0)
            df[f"normalized_to_0_5{base}"] = np.where(pos, np.nanmin(z_future, axis=1) <= 0.5, np.nanmax(z_future, axis=1) >= -0.5)
            df[f"normalized_to_0{base}"] = np.where(pos, np.nanmin(z_future, axis=1) <= 0.0, np.nanmax(z_future, axis=1) >= 0.0)

            df[f"moved_further_to_abs_z_3{base}"] = np.nanmax(abs_future, axis=1) >= 3.0
            df[f"moved_further_to_abs_z_4{base}"] = np.nanmax(abs_future, axis=1) >= 4.0

            b1, max_before_1 = first_touch_metrics(z_future, z_now, 1.0)
            b05, _ = first_touch_metrics(z_future, z_now, 0.5)
            b0, _ = first_touch_metrics(z_future, z_now, 0.0)

            df[f"bars_to_normalized_1{base}"] = b1
            df[f"bars_to_normalized_0_5{base}"] = b05
            df[f"bars_to_normalized_0{base}"] = b0
            df[f"max_abs_zscore_before_normalization{base}"] = max_before_1

            analysis = pd.DataFrame(
                {
                    "datetime": df["datetime"],
                    "hour": df["hour"],
                    "day_of_week": df["day_of_week"],
                    "session_bucket": df["session_bucket"],
                    "correlation_regime": df[f"correlation_regime_{w}"],
                    "beta_stability": df[f"beta_stability_{w}"],
                    "zscore_bucket": df[f"zscore_bucket_{w}"],
                    "abs_zscore_bucket": df[f"abs_zscore_bucket_{w}"],
                    "rolling_corr": df[corr_col],
                    "rolling_beta": df[beta_col],
                    "zscore_abs": df[f"zscore_abs_{w}"],
                    "normalized_to_1": df[f"normalized_to_1{base}"].astype(float),
                    "normalized_to_0_5": df[f"normalized_to_0_5{base}"].astype(float),
                    "normalized_to_0": df[f"normalized_to_0{base}"].astype(float),
                    "moved_further_to_abs_z_3": df[f"moved_further_to_abs_z_3{base}"].astype(float),
                    "moved_further_to_abs_z_4": df[f"moved_further_to_abs_z_4{base}"].astype(float),
                    "bars_to_normalized_1": df[f"bars_to_normalized_1{base}"],
                    "bars_to_normalized_0_5": df[f"bars_to_normalized_0_5{base}"],
                    "bars_to_normalized_0": df[f"bars_to_normalized_0{base}"],
                    "future_abs_zscore_min": df[f"future_abs_zscore_min{base}"],
                    "future_abs_zscore_max": df[f"future_abs_zscore_max{base}"],
                    "split": df["split"],
                    "wf_split": df["wf_split"],
                    "valid_row": 1,
                }
            )
            analysis = analysis.dropna(subset=["rolling_corr", "rolling_beta", "zscore_abs"])

            for grp in GROUPINGS:
                print(f"[progress] grouping={'+'.join(grp)} window={w} horizon={h}")
                gdf = summarize_group(analysis, grp)
                gdf["window"] = w
                gdf["horizon_bars"] = h
                gdf["grouping"] = "+".join(grp)
                gdf["lead_lag_summary"] = (
                    f"A_to_B_lag1={lead_a[1]:.4f};A_to_B_lag2={lead_a[2]:.4f};A_to_B_lag4={lead_a[4]:.4f};"
                    f"B_to_A_lag1={lead_b[1]:.4f};B_to_A_lag2={lead_b[2]:.4f};B_to_A_lag4={lead_b[4]:.4f}"
                )
                grouped_frames.append(gdf)

    grouped = pd.concat(grouped_frames, ignore_index=True)

    assert all(col in grouped.columns for col in ["window", "horizon_bars", "grouping"])
    if args.preset == "quick":
        assert set([20, 50, 100]).issubset(set(grouped["window"].unique()))
        assert set([5, 10, 20, 40]).issubset(set(grouped["horizon_bars"].unique()))

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.symbol_a.upper()}_{args.symbol_b.upper()}_pairs"

    overview = pd.DataFrame(
        [
            {
                "symbol_a": args.symbol_a.upper(),
                "symbol_b": args.symbol_b.upper(),
                "base_timeframe": args.base_timeframe,
                "preset": args.preset,
                "rows_synchronized": len(df),
                "start_datetime": df["datetime"].min(),
                "end_datetime": df["datetime"].max(),
                "windows": ",".join(map(str, windows)),
                "horizons": ",".join(map(str, horizons)),
                "pip_size_a": get_pip_size(args.symbol_a),
                "pip_size_b": get_pip_size(args.symbol_b),
            }
        ]
    )

    feature_dict = {
        "description": "Descriptive relative behavior atlas for synchronized pair data.",
        "window_features": [
            "rolling_corr_returns_w",
            "rolling_beta_w",
            "beta_change_w",
            "beta_abs_change_w",
            "relative_spread_w",
            "spread_mean_w",
            "spread_std_w",
            "spread_zscore_w",
            "zscore_abs_w",
            "zscore_change_w",
            "correlation_regime_w",
            "beta_stability_w",
            "zscore_bucket_w",
            "abs_zscore_bucket_w",
        ],
        "future_features": [
            "future_zscore_change_w_h",
            "future_abs_zscore_min_w_h",
            "future_abs_zscore_max_w_h",
            "future_zscore_max_up_w_h",
            "future_zscore_max_down_w_h",
            "normalized_to_1_w_h",
            "normalized_to_0_5_w_h",
            "normalized_to_0_w_h",
            "moved_further_to_abs_z_3_w_h",
            "moved_further_to_abs_z_4_w_h",
            "bars_to_normalized_1_w_h",
            "bars_to_normalized_0_5_w_h",
            "bars_to_normalized_0_w_h",
            "max_abs_zscore_before_normalization_w_h",
        ],
    }

    corr_summary = grouped.groupby("grouping")["mean_corr"].mean().sort_values(ascending=False).head(5)
    beta_summary = grouped.groupby("grouping")["mean_beta"].mean().sort_values(ascending=False).head(5)
    z_summary = grouped["mean_abs_zscore"].describe()

    md = [
        f"# Pairs Relative Behavior Atlas: {args.symbol_a.upper()} vs {args.symbol_b.upper()}",
        "",
        f"- Dataset range: {df['datetime'].min()} to {df['datetime'].max()}",
        f"- Synchronized row count: {len(df)}",
        "",
        "This report is descriptive relative behavior analysis and is not a strategy test.",
        "",
        "## Correlation regime summary",
        grouped["grouping"].value_counts().head(10).to_string(),
        "",
        "## Beta stability summary",
        beta_summary.to_string(),
        "",
        "## Z-score excursion summary",
        z_summary.to_string(),
        "",
        "## Z-score normalization behavior",
        grouped[["normalized_to_1_rate", "normalized_to_0_5_rate", "normalized_to_0_rate"]].mean().to_string(),
        "",
        "## Session comparison",
        grouped[grouped["grouping"] == "session_bucket"][
            ["normalized_to_0_5_rate", "moved_further_to_abs_z_3_rate"]
        ].describe().to_string(),
        "",
        "## Time-of-day comparison",
        grouped[grouped["grouping"] == "hour"][["mean_abs_zscore", "normalized_to_0_5_rate"]].describe().to_string(),
        "",
        "## Notes on IS/OOS stability",
        grouped[["IS_normalized_to_0_5_rate", "OOS_normalized_to_0_5_rate", "WF_normalized_to_0_5_std"]]
        .describe()
        .to_string(),
        "",
        "## Warning",
        "Overfitting risk exists in any segmented descriptive analysis. No decision or execution implication is provided.",
    ]

    overview_path = outdir / f"{prefix}_overview.csv"
    grouped_path = outdir / f"{prefix}_grouped_behavior.csv"
    dict_path = outdir / f"{prefix}_feature_dictionary.json"
    summary_path = outdir / f"{prefix}_summary.md"

    overview.to_csv(overview_path, index=False)
    grouped.to_csv(grouped_path, index=False)
    dict_path.write_text(json.dumps(feature_dict, indent=2), encoding="utf-8")
    summary_path.write_text("\n".join(md), encoding="utf-8")

    if args.write_row_level:
        row_path = outdir / f"{prefix}_row_level_features.csv"
        df.to_csv(row_path, index=False)

    print(f"[done] overview: {overview_path}")
    print(f"[done] grouped behavior: {grouped_path}")
    print(f"[done] feature dictionary: {dict_path}")
    print(f"[done] summary markdown: {summary_path}")
    if args.write_row_level:
        print(f"[done] row-level features: {row_path}")

    print(f"[done] runtime_seconds={time.time() - start:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
