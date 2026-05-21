#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

PIP_SIZE = 0.0001


def parse_args():
    p = argparse.ArgumentParser(description="EURUSD M15 descriptive market behavior atlas")
    p.add_argument("--csv", default="data/EURUSD_M15_MT5_5Y.csv")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--base-timeframe", default="M15")
    p.add_argument("--output-dir", default="market_behavior_atlas_reports")
    p.add_argument("--preset", choices=["quick", "full"], default="quick")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--write-row-level", action="store_true", default=False)
    return p.parse_args()


def load_data(csv_path: str, max_rows: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = ["datetime", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=False)
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).copy()
    df = df.sort_values("datetime").reset_index(drop=True)
    if max_rows > 0:
        df = df.iloc[:max_rows].copy()
    return df


def session_bucket(hour: pd.Series) -> pd.Series:
    conds = [
        hour.between(0, 3), hour.between(4, 6), hour.between(7, 9),
        hour.between(10, 12), hour.between(13, 15), hour.between(16, 18), hour.between(19, 23)
    ]
    vals = ["asia_early", "asia_late", "london_open", "london_mid", "newyork_open", "newyork_mid", "late_session"]
    return pd.Series(np.select(conds, vals, default="late_session"), index=hour.index)


def bucket_abs_distance(s: pd.Series) -> pd.Series:
    a = s.abs()
    conds = [a < 5, (a >= 5) & (a < 10), (a >= 10) & (a < 15), (a >= 15) & (a < 25), (a >= 25) & (a < 40), a >= 40]
    vals = ["0_5", "5_10", "10_15", "15_25", "25_40", "40_plus"]
    return pd.Series(np.select(conds, vals, default="40_plus"), index=s.index)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    df["session_bucket"] = session_bucket(df["hour"])

    body = (df["close"] - df["open"]).abs() / PIP_SIZE
    rng = (df["high"] - df["low"]).clip(lower=0) / PIP_SIZE
    upper = (df["high"] - df[["open", "close"]].max(axis=1)).clip(lower=0) / PIP_SIZE
    lower = (df[["open", "close"]].min(axis=1) - df["low"]).clip(lower=0) / PIP_SIZE
    denom = rng.replace(0, np.nan)
    df["body_pips"], df["range_pips"], df["upper_wick_pips"], df["lower_wick_pips"] = body, rng, upper, lower
    df["body_to_range"], df["upper_wick_to_range"], df["lower_wick_to_range"] = body / denom, upper / denom, lower / denom
    df["close_position"] = ((df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)).fillna(0.5)
    df["candle_direction"] = np.where(df["close"] > df["open"], "bull", np.where(df["close"] < df["open"], "bear", "doji"))

    df["body_size_bucket"] = np.select([body < 2, body < 5, body < 10, body < 20, body >= 20], ["tiny", "small", "medium", "large", "extreme"], default="tiny")
    df["range_size_bucket"] = np.select([rng < 5, rng < 10, rng < 20, rng < 35, rng >= 35], ["tiny", "small", "medium", "large", "extreme"], default="tiny")
    df["close_position_bucket"] = np.select([df["close_position"] < 0.2, df["close_position"] < 0.4, df["close_position"] < 0.6, df["close_position"] < 0.8, df["close_position"] >= 0.8], ["bottom_20", "lower_mid", "middle", "upper_mid", "top_20"], default="middle")
    df["wick_structure_bucket"] = np.select(
        [df["lower_wick_to_range"] >= 0.45, df["upper_wick_to_range"] >= 0.45, df["body_to_range"] >= 0.70],
        ["long_lower_wick", "long_upper_wick", "full_body"],
        default="indecision",
    )

    dmap = {"bull": "B", "bear": "R", "doji": "D"}
    seq = df["candle_direction"].map(dmap).fillna("D")
    df["consecutive_bull_count"] = (df["candle_direction"].eq("bull").groupby(df["candle_direction"].ne("bull").cumsum()).cumcount() + 1).where(df["candle_direction"].eq("bull"), 0)
    df["consecutive_bear_count"] = (df["candle_direction"].eq("bear").groupby(df["candle_direction"].ne("bear").cumsum()).cumcount() + 1).where(df["candle_direction"].eq("bear"), 0)
    df["previous_3_direction"] = seq.shift(1).fillna("D") + seq.shift(2).fillna("D") + seq.shift(3).fillna("D")
    df["previous_5_direction"] = seq.shift(1).fillna("D") + seq.shift(2).fillna("D") + seq.shift(3).fillna("D") + seq.shift(4).fillna("D") + seq.shift(5).fillna("D")
    df["sum_body_last_3_pips"] = body.shift(1).rolling(3).sum()
    df["sum_range_last_3_pips"] = rng.shift(1).rolling(3).sum()
    df["sum_body_last_5_pips"] = body.shift(1).rolling(5).sum()
    df["sum_range_last_5_pips"] = rng.shift(1).rolling(5).sum()
    df["streak_bucket"] = np.where(df["consecutive_bull_count"] >= 3, "bull_3_plus", np.where(df["consecutive_bear_count"] >= 3, "bear_3_plus", "mixed"))

    prev_close = df["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]), (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1) / PIP_SIZE
    df["atr_14_pips"] = tr.rolling(14).mean()
    df["atr_50_pips"] = tr.rolling(50).mean()
    df["atr_ratio_14_to_50"] = df["atr_14_pips"] / df["atr_50_pips"].replace(0, np.nan)
    df["rolling_range_10_pips"] = rng.rolling(10).mean()
    df["rolling_range_20_pips"] = rng.rolling(20).mean()
    df["compression_score"] = df["rolling_range_10_pips"] / df["rolling_range_20_pips"].replace(0, np.nan)
    df["expansion_score"] = rng / df["atr_14_pips"].replace(0, np.nan)
    q25, q75, q90 = df["atr_14_pips"].quantile([0.25, 0.75, 0.90])
    df["atr_percentile_bucket"] = np.select([df["atr_14_pips"] < q25, df["atr_14_pips"] < q75, df["atr_14_pips"] < q90, df["atr_14_pips"] >= q90], ["low_0_25", "mid_25_75", "high_75_90", "extreme_90_plus"], default="mid_25_75")
    df["compression_bucket"] = np.select([df["compression_score"] < 0.9, df["compression_score"] > 1.1], ["compressed", "expanded"], default="normal")

    df["rolling_16_close"] = df["close"].rolling(16).mean()
    df["rolling_32_close"] = df["close"].rolling(32).mean()
    day = df["datetime"].dt.floor("D")
    df["daily_open"] = df.groupby(day)["open"].transform("first")
    day_h = df.groupby(day)["high"].max()
    day_l = df.groupby(day)["low"].min()
    df["previous_day_high"] = day.map(day_h.shift(1))
    df["previous_day_low"] = day.map(day_l.shift(1))
    asia = df[df["hour"].between(0, 6)].groupby(day).agg(asia_high=("high", "max"), asia_low=("low", "min"))
    df["asian_session_high"] = day.map(asia["asia_high"])
    df["asian_session_low"] = day.map(asia["asia_low"])

    anchors = ["rolling_16_close", "rolling_32_close", "daily_open", "previous_day_high", "previous_day_low", "asian_session_high", "asian_session_low"]
    for a in anchors:
        df[f"distance_from_{a.replace('_close', '').replace('session_', '')}_pips"] = (df["close"] - df[a]) / PIP_SIZE

    df["distance_from_rolling_16_abs_bucket"] = bucket_abs_distance(df["distance_from_rolling_16_pips"])
    df["distance_from_daily_open_abs_bucket"] = bucket_abs_distance(df["distance_from_daily_open_pips"])
    return df


def first_touch_and_bars(up_arr, down_arr, threshold):
    up_hit = up_arr >= threshold
    down_hit = down_arr >= threshold
    n, h = up_arr.shape
    up_idx = np.where(up_hit, np.arange(1, h + 1), h + 1).min(axis=1)
    down_idx = np.where(down_hit, np.arange(1, h + 1), h + 1).min(axis=1)
    up_bars = np.where(up_idx <= h, up_idx, np.nan)
    down_bars = np.where(down_idx <= h, down_idx, np.nan)
    first = np.where((up_idx > h) & (down_idx > h), "none", np.where(up_idx < down_idx, "up_first", np.where(down_idx < up_idx, "down_first", "both_same_bar")))
    return up_hit.any(axis=1), down_hit.any(axis=1), first, up_bars, down_bars


def add_future_labels(df, horizons, thresholds):
    n = len(df)
    close, high, low = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
    for h in horizons:
        print(f"Processing horizon {h}")
        up_m = np.full(n, np.nan)
        down_m = np.full(n, np.nan)
        fut_rng = np.full(n, np.nan)
        fut_close = np.full(n, np.nan)
        for i in range(n - h):
            hc = high[i + 1 : i + h + 1]
            lc = low[i + 1 : i + h + 1]
            up_m[i] = (hc.max() - close[i]) / PIP_SIZE
            down_m[i] = (close[i] - lc.min()) / PIP_SIZE
            fut_rng[i] = (hc.max() - lc.min()) / PIP_SIZE
            fut_close[i] = (close[i + h] - close[i]) / PIP_SIZE
        df[f"future_close_change_pips_{h}"] = fut_close
        df[f"future_max_up_pips_{h}"] = up_m
        df[f"future_max_down_pips_{h}"] = down_m
        df[f"future_range_pips_{h}"] = fut_rng
        df[f"future_close_direction_{h}"] = np.where(fut_close > 0, "up", np.where(fut_close < 0, "down", "flat"))

        up_steps = np.vstack([np.r_[((high[j + 1 : j + h + 1] - close[j]) / PIP_SIZE), np.full(max(0, h - len(high[j + 1 : j + h + 1])), np.nan)] for j in range(n)])
        down_steps = np.vstack([np.r_[((close[j] - low[j + 1 : j + h + 1]) / PIP_SIZE), np.full(max(0, h - len(low[j + 1 : j + h + 1])), np.nan)] for j in range(n)])
        for t in thresholds:
            hit_u, hit_d, first, b_u, b_d = first_touch_and_bars(up_steps, down_steps, t)
            df[f"hit_up_{t}_{h}"] = hit_u.astype(float)
            df[f"hit_down_{t}_{h}"] = hit_d.astype(float)
            df[f"first_touch_{t}_{h}"] = first
            df[f"bars_to_up_{t}_{h}"] = b_u
            df[f"bars_to_down_{t}_{h}"] = b_d
    return df


def aggregate_group(df, keys, horizon):
    rows = []
    split = int(len(df) * 0.7)
    wf = np.array_split(np.arange(len(df)), 3)
    for g, gdf in df.groupby(keys, dropna=False):
        gdict = {keys[i]: g[i] for i in range(len(keys))} if isinstance(g, tuple) else {keys[0]: g}
        up = gdf[f"future_max_up_pips_{horizon}"]
        down = gdf[f"future_max_down_pips_{horizon}"]
        r = {
            **gdict,
            "horizon_bars": horizon,
            "observations": len(gdf),
            "mean_future_max_up_pips": up.mean(), "median_future_max_up_pips": up.median(),
            "p75_future_max_up_pips": up.quantile(0.75), "p90_future_max_up_pips": up.quantile(0.90), "p95_future_max_up_pips": up.quantile(0.95),
            "mean_future_max_down_pips": down.mean(), "median_future_max_down_pips": down.median(),
            "p75_future_max_down_pips": down.quantile(0.75), "p90_future_max_down_pips": down.quantile(0.90), "p95_future_max_down_pips": down.quantile(0.95),
        }
        for t in [5, 8, 10, 15, 20]:
            uh = f"hit_up_{t}_{horizon}"; dh = f"hit_down_{t}_{horizon}"
            if uh in gdf:
                r[f"hit_up_{t}_rate"] = gdf[uh].mean()
                r[f"hit_down_{t}_rate"] = gdf[dh].mean()
        for t in [8, 10]:
            ft = f"first_touch_{t}_{horizon}"
            if ft in gdf:
                r[f"up_first_{t}_rate"] = (gdf[ft] == "up_first").mean()
                r[f"down_first_{t}_rate"] = (gdf[ft] == "down_first").mean()
                r[f"median_bars_to_up_{t}"] = gdf[f"bars_to_up_{t}_{horizon}"].median()
                r[f"median_bars_to_down_{t}"] = gdf[f"bars_to_down_{t}_{horizon}"].median()
        is_df, oos_df = gdf[gdf.index < split], gdf[gdf.index >= split]
        r["IS_observations"], r["OOS_observations"] = len(is_df), len(oos_df)
        for side in ["up", "down"]:
            col = f"hit_{side}_8_{horizon}"
            if col in gdf:
                r[f"IS_hit_{side}_8_rate"] = is_df[col].mean()
                r[f"OOS_hit_{side}_8_rate"] = oos_df[col].mean()
        wf_rates = []
        for w in wf:
            wdf = gdf.loc[gdf.index.intersection(w)]
            wf_rates.append(wdf[f"hit_up_8_{horizon}"].mean() if len(wdf) else np.nan)
        r["WF_hit_up_8_std"] = float(np.nanstd(wf_rates))
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    args = parse_args()
    horizons = [5, 10, 20, 40] if args.preset == "quick" else [5, 10, 20, 40, 80]
    thresholds = [5, 8, 10, 15] if args.preset == "quick" else [5, 8, 10, 15, 20, 30]

    df = load_data(args.csv, args.max_rows)
    df = compute_features(df)
    df = add_future_labels(df, horizons, thresholds)

    grouping_specs = [
        ["hour"], ["session_bucket"], ["day_of_week"], ["candle_direction"], ["body_size_bucket"], ["range_size_bucket"],
        ["wick_structure_bucket"], ["close_position_bucket"], ["streak_bucket"], ["previous_3_direction"], ["atr_percentile_bucket"],
        ["compression_bucket"], ["distance_from_rolling_16_abs_bucket"], ["distance_from_daily_open_abs_bucket"],
        ["session_bucket", "wick_structure_bucket"], ["session_bucket", "streak_bucket"], ["hour", "wick_structure_bucket"],
        ["hour", "compression_bucket"], ["wick_structure_bucket", "close_position_bucket"], ["session_bucket", "distance_from_rolling_16_abs_bucket"],
    ]

    grouped = []
    for g in grouping_specs:
        for h in horizons:
            print(f"Aggregating group: {g} @ horizon {h}")
            tmp = aggregate_group(df, g, h)
            tmp["grouping"] = "+".join(g)
            grouped.append(tmp)
    grouped_df = pd.concat(grouped, ignore_index=True, sort=False)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    overview = pd.DataFrame([{
        "symbol": args.symbol, "base_timeframe": args.base_timeframe, "preset": args.preset,
        "rows": len(df), "start_datetime": df["datetime"].min(), "end_datetime": df["datetime"].max(),
        "horizons": ",".join(map(str, horizons)), "thresholds": ",".join(map(str, thresholds)),
    }])

    assert "horizon_bars" in grouped_df.columns, "Grouped output must include horizon_bars."
    if args.preset == "quick":
        expected_quick_horizons = {5, 10, 20, 40}
        present_horizons = set(pd.to_numeric(grouped_df["horizon_bars"], errors="coerce").dropna().astype(int).unique())
        missing_quick_horizons = expected_quick_horizons - present_horizons
        assert not missing_quick_horizons, f"Grouped output missing quick horizons: {sorted(missing_quick_horizons)}"

    overview.to_csv(out / "EURUSD_atlas_overview.csv", index=False)
    grouped_df.to_csv(out / "EURUSD_atlas_grouped_behavior.csv", index=False)

    feature_dict = {
        "purpose": "Descriptive market behavior atlas for EURUSD M15 based on future movement distributions.",
        "time_features": ["hour", "day_of_week", "month", "year", "session_bucket"],
        "candle_features": ["body_pips", "range_pips", "upper_wick_pips", "lower_wick_pips", "body_to_range", "upper_wick_to_range", "lower_wick_to_range", "close_position", "candle_direction"],
        "future_horizons": horizons,
        "thresholds": thresholds,
    }
    (out / "EURUSD_atlas_feature_dictionary.json").write_text(json.dumps(feature_dict, indent=2))

    top_up = grouped_df.sort_values("mean_future_max_up_pips", ascending=False).head(10)
    top_down = grouped_df.sort_values("mean_future_max_down_pips", ascending=False).head(10)
    md = [
        "# EURUSD M15 Market Behavior Atlas Summary",
        f"- Dataset range: {df['datetime'].min()} to {df['datetime'].max()} ({len(df)} rows)",
        f"- Horizons included in grouped output: {', '.join(map(str, horizons))} bars.",
        "- This report is descriptive market behavior analysis based on future movement distributions, not a strategy test.",
        "## Top behavior groups by strong future up movement tendency",
        top_up[["grouping", "mean_future_max_up_pips", "observations"]].to_markdown(index=False),
        "## Top behavior groups by strong future down movement tendency",
        top_down[["grouping", "mean_future_max_down_pips", "observations"]].to_markdown(index=False),
        "## Session comparison\nSee grouped behavior table for `session_bucket`.",
        "## Candle structure comparison\nSee grouped behavior table for `candle_direction`, `body_size_bucket`, and `range_size_bucket`.",
        "## Wick structure comparison\nSee grouped behavior table for `wick_structure_bucket`.",
        "## Streak comparison\nSee grouped behavior table for `streak_bucket` and `previous_3_direction`.",
        "## Volatility/compression comparison\nSee grouped behavior table for `atr_percentile_bucket` and `compression_bucket`.",
        "## Anchor-distance comparison\nSee grouped behavior table for `distance_from_rolling_16_abs_bucket` and `distance_from_daily_open_abs_bucket`.",
        "## Notes on IS/OOS stability\nUse IS/OOS rates and WF consistency columns from grouped behavior output.",
        "## Warning\nOverfitting risk exists in conditional grouping analysis. No trading decision is implied.",
    ]
    (out / "EURUSD_atlas_summary.md").write_text("\n\n".join(md))

    if args.write_row_level:
        df.to_csv(out / "EURUSD_atlas_row_level_features.csv", index=False)

    print(f"Total runtime seconds: {time.time() - t0:.2f}")


if __name__ == "__main__":
    main()
