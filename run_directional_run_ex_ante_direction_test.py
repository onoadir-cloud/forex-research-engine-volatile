#!/usr/bin/env python3
import argparse
import hashlib
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SESSION_BUCKETS = [
    (0, 3, "Asia early"),
    (4, 6, "Asia late"),
    (7, 9, "London open"),
    (10, 12, "London mid"),
    (13, 15, "New York open"),
    (16, 18, "New York mid"),
    (19, 23, "Late session"),
]

COST_PROFILES = {
    "EURUSD": {
        "low": {"spread_pips": 0.8, "slippage_pips": 0.2, "commission_equivalent_pips": 0.2},
        "conservative": {"spread_pips": 1.2, "slippage_pips": 0.4, "commission_equivalent_pips": 0.4},
        "high": {"spread_pips": 2.0, "slippage_pips": 0.8, "commission_equivalent_pips": 0.8},
    }
}


def parse_list(s: str, cast=float) -> List:
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def session_bucket(hour: int) -> str:
    for start, end, label in SESSION_BUCKETS:
        if start <= hour <= end:
            return label
    return "Unknown"


def infer_pip_size(symbol: str) -> float:
    return 0.01 if "JPY" in symbol.upper() else 0.0001


def get_friction_pips(symbol: str, profile: str) -> float:
    sym = symbol.upper()
    source = COST_PROFILES.get(sym, COST_PROFILES["EURUSD"])
    c = source[profile]
    return 2 * c["spread_pips"] + 2 * c["slippage_pips"] + c["commission_equivalent_pips"]


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    dt_col = cols.get("datetime") or cols.get("time") or cols.get("date")
    open_col = cols.get("open")
    high_col = cols.get("high")
    low_col = cols.get("low")
    close_col = cols.get("close")
    if not all([dt_col, open_col, high_col, low_col, close_col]):
        raise ValueError("Input CSV must include datetime/open/high/low/close (case-insensitive).")
    out = df[[dt_col, open_col, high_col, low_col, close_col]].copy()
    out.columns = ["datetime", "open", "high", "low", "close"]
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    out = out.dropna(subset=["datetime", "open", "high", "low", "close"]).sort_values("datetime")
    out = out.drop_duplicates(subset=["datetime"]).reset_index(drop=True)
    return out


def resample_tf(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    m = {"H1": "1h", "H4": "4h", "D1": "1D"}
    if tf not in m:
        raise ValueError(f"Unsupported timeframe: {tf}")
    x = df.set_index("datetime").resample(m[tf]).agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    return x.dropna(subset=["open", "high", "low", "close"]).reset_index()


def add_indicators(df: pd.DataFrame, atr_period: int, pip_size: float) -> pd.DataFrame:
    d = df.copy()
    prev_close = d["close"].shift(1)
    tr = np.maximum.reduce([
        (d["high"] - d["low"]).values,
        np.abs((d["high"] - prev_close)).values,
        np.abs((d["low"] - prev_close)).values,
    ])
    d["atr"] = pd.Series(tr, index=d.index).rolling(atr_period, min_periods=atr_period).mean()
    d["atr_pips"] = d["atr"] / pip_size

    d["close_20_ago"] = d["close"].shift(20)
    d["close_50_ago"] = d["close"].shift(50)
    d["sma20"] = d["close"].rolling(20, min_periods=20).mean()
    d["sma20_10_ago"] = d["sma20"].shift(10)
    d["sma50"] = d["close"].rolling(50, min_periods=50).mean()
    d["prior_high_20"] = d["high"].shift(1).rolling(20, min_periods=20).max()
    d["prior_low_20"] = d["low"].shift(1).rolling(20, min_periods=20).min()
    d["prior_high_50"] = d["high"].shift(1).rolling(50, min_periods=50).max()
    d["prior_low_50"] = d["low"].shift(1).rolling(50, min_periods=50).min()
    den = (d["prior_high_50"] - d["prior_low_50"])
    d["range_position"] = np.where(den > 0, (d["close"] - d["prior_low_50"]) / den, np.nan)
    d["momentum_20_pips"] = (d["close"] - d["close_20_ago"]) / pip_size
    d["momentum_50_pips"] = (d["close"] - d["close_50_ago"]) / pip_size
    return d


def deterministic_random_direction(dt: pd.Timestamp, tf: str) -> int:
    seed = f"{dt.isoformat()}|{tf}|random_baseline_deterministic"
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return 1 if (int(h[:8], 16) % 2 == 0) else -1


def get_method_direction(row: pd.Series, method: str, tf: str) -> Optional[int]:
    c = row["close"]
    if method == "momentum_20":
        return None if pd.isna(row["close_20_ago"]) else (1 if c > row["close_20_ago"] else -1)
    if method == "momentum_50":
        return None if pd.isna(row["close_50_ago"]) else (1 if c > row["close_50_ago"] else -1)
    if method == "ma_slope_20":
        return None if pd.isna(row["sma20"]) or pd.isna(row["sma20_10_ago"]) else (1 if row["sma20"] > row["sma20_10_ago"] else -1)
    if method == "ma_position_50":
        return None if pd.isna(row["sma50"]) else (1 if c > row["sma50"] else -1)
    if method == "breakout_state_20":
        if pd.isna(row["prior_high_20"]) or pd.isna(row["prior_low_20"]):
            return None
        if c > row["prior_high_20"]:
            return 1
        if c < row["prior_low_20"]:
            return -1
        return None
    if method == "breakout_state_50":
        if pd.isna(row["prior_high_50"]) or pd.isna(row["prior_low_50"]):
            return None
        if c > row["prior_high_50"]:
            return 1
        if c < row["prior_low_50"]:
            return -1
        return None
    if method == "recent_range_position":
        rp = row["range_position"]
        if pd.isna(rp):
            return None
        if rp >= 0.70:
            return 1
        if rp <= 0.30:
            return -1
        return None
    if method == "random_baseline_deterministic":
        return deterministic_random_direction(row["datetime"], tf)
    raise ValueError(f"Unknown method: {method}")


def calc_directional_run(df: pd.DataFrame, i: int, h: int, direction: int, pullback_atr: float, pip_size: float) -> Optional[Tuple[float, float, int, int]]:
    end_i = i + h
    if end_i >= len(df):
        return None
    atr_pips = df.at[i, "atr_pips"]
    if pd.isna(atr_pips) or atr_pips <= 0:
        return None
    threshold = pullback_atr * atr_pips
    close0 = df.at[i, "close"]

    best = -np.inf
    bars_to_best = np.nan
    best_idx = None
    pullback_trigger = 0
    running_extreme = None

    for j in range(i + 1, end_i + 1):
        high_j = df.at[j, "high"]
        low_j = df.at[j, "low"]
        if direction == 1:
            if running_extreme is None or high_j > running_extreme:
                running_extreme = high_j
            fav = (running_extreme - close0) / pip_size
            pullback = (running_extreme - low_j) / pip_size
        else:
            if running_extreme is None or low_j < running_extreme:
                running_extreme = low_j
            fav = (close0 - running_extreme) / pip_size
            pullback = (high_j - running_extreme) / pip_size

        if fav > best:
            best = fav
            bars_to_best = j - i
            best_idx = j
        if pullback >= threshold:
            pullback_trigger = 1
            break

    if np.isinf(best):
        return None

    window = df.iloc[i + 1 : best_idx + 1]
    if direction == 1:
        adverse = float(((window["low"] - close0) / pip_size).min()) * -1.0
    else:
        adverse = float(((window["high"] - close0) / pip_size).max())
    return float(best), float(adverse), int(bars_to_best), int(pullback_trigger)


def expectancy_components(s: pd.Series) -> float:
    s = s.dropna()
    if s.empty:
        return np.nan
    pos = s[s > 0]
    neg = s[s <= 0]
    p_pos = len(pos) / len(s)
    p_neg = len(neg) / len(s)
    avg_pos = pos.mean() if len(pos) else 0.0
    avg_neg = neg.mean() if len(neg) else 0.0
    return p_pos * avg_pos + p_neg * avg_neg


def longest_negative_streak(vals: pd.Series) -> int:
    m = c = 0
    for v in vals:
        if pd.notna(v) and v < 0:
            c += 1
            m = max(m, c)
        else:
            c = 0
    return m


def rolling_negative_rate(vals: pd.Series, win: int) -> float:
    if len(vals) < win:
        return np.nan
    neg = (vals < 0).astype(float)
    return float(neg.rolling(win).mean().max())


def summarize(events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gcols = ["timeframe", "direction_method", "future_horizon_bars", "allowed_pullback_atr", "session_bucket", "hour"]
    ycols = ["timeframe", "direction_method", "future_horizon_bars", "allowed_pullback_atr", "year"]
    rows = []
    yearly = []
    for key, g in events.sort_values("datetime").groupby(gcols, dropna=False):
        s = g["selected_run_after_friction"]
        pos = s[s > 0]
        neg = s[s <= 0]
        pr = float(len(pos) / len(s)) if len(s) else np.nan
        nr = float(len(neg) / len(s)) if len(s) else np.nan
        avg_pos = float(pos.mean()) if len(pos) else np.nan
        avg_neg = float(neg.mean()) if len(neg) else np.nan
        payoff = (avg_pos / abs(avg_neg)) if (pd.notna(avg_pos) and pd.notna(avg_neg) and avg_neg != 0) else np.nan
        exp = expectancy_components(s)

        cut = int(len(g) * 0.7)
        g_is = g.iloc[:cut]
        g_oos = g.iloc[cut:]

        row = dict(zip(gcols, key))
        row.update({
            "observations": len(g),
            "mean_selected_run_after_friction": s.mean(),
            "median_selected_run_after_friction": s.median(),
            "avg_positive_selected_run": avg_pos,
            "avg_negative_selected_run": avg_neg,
            "positive_rate": pr,
            "negative_rate": nr,
            "payoff_ratio": payoff,
            "expectancy_from_components": exp,
            "p05_selected_run_after_friction": s.quantile(0.05),
            "p10_selected_run_after_friction": s.quantile(0.10),
            "p90_selected_run_after_friction": s.quantile(0.90),
            "p95_selected_run_after_friction": s.quantile(0.95),
            "worst_selected_run_after_friction": s.min(),
            "best_selected_run_after_friction": s.max(),
            "mean_opposite_run_after_friction": g["opposite_run_after_friction"].mean(),
            "median_opposite_run_after_friction": g["opposite_run_after_friction"].median(),
            "mean_edge_vs_opposite": g["edge_vs_opposite"].mean(),
            "median_edge_vs_opposite": g["edge_vs_opposite"].median(),
            "mean_edge_vs_random": g["edge_vs_random"].mean(),
            "median_edge_vs_random": g["edge_vs_random"].median(),
            "median_selected_max_adverse_pips_before_run": g["selected_max_adverse_pips_before_run"].median(),
            "p90_selected_max_adverse_pips_before_run": g["selected_max_adverse_pips_before_run"].quantile(0.90),
            "worst_selected_max_adverse_pips_before_run": g["selected_max_adverse_pips_before_run"].max(),
            "median_bars_to_max_favorable": g["selected_bars_to_max_favorable"].median(),
            "pullback_trigger_rate": g["selected_did_allowed_pullback_trigger"].mean(),
            "IS_observations": len(g_is),
            "OOS_observations": len(g_oos),
            "IS_mean_selected_run_after_friction": g_is["selected_run_after_friction"].mean(),
            "OOS_mean_selected_run_after_friction": g_oos["selected_run_after_friction"].mean(),
            "IS_expectancy_from_components": expectancy_components(g_is["selected_run_after_friction"]),
            "OOS_expectancy_from_components": expectancy_components(g_oos["selected_run_after_friction"]),
            "IS_mean_edge_vs_opposite": g_is["edge_vs_opposite"].mean(),
            "OOS_mean_edge_vs_opposite": g_oos["edge_vs_opposite"].mean(),
            "IS_mean_edge_vs_random": g_is["edge_vs_random"].mean(),
            "OOS_mean_edge_vs_random": g_oos["edge_vs_random"].mean(),
            "IS_p05_selected_run_after_friction": g_is["selected_run_after_friction"].quantile(0.05) if len(g_is) else np.nan,
            "OOS_p05_selected_run_after_friction": g_oos["selected_run_after_friction"].quantile(0.05) if len(g_oos) else np.nan,
            "longest_consecutive_negative_selected_runs": longest_negative_streak(s),
            "max_rolling_20_negative_selected_rate": rolling_negative_rate(s, 20),
            "max_rolling_50_negative_selected_rate": rolling_negative_rate(s, 50),
            "max_rolling_100_negative_selected_rate": rolling_negative_rate(s, 100),
        })
        rows.append(row)

    for key, g in events.groupby(ycols, dropna=False):
        s = g["selected_run_after_friction"]
        yearly.append({
            **dict(zip(ycols, key)),
            "year_observations": len(g),
            "year_mean_selected_run_after_friction": s.mean(),
            "year_expectancy_from_components": expectancy_components(s),
            "year_mean_edge_vs_opposite": g["edge_vs_opposite"].mean(),
            "year_mean_edge_vs_random": g["edge_vs_random"].mean(),
            "year_p05_selected_run_after_friction": s.quantile(0.05),
            "year_worst_selected_run_after_friction": s.min(),
        })

    return pd.DataFrame(rows), pd.DataFrame(yearly)


def markdown_report(symbol: str, friction: float, summary: pd.DataFrame, yearly: pd.DataFrame, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# {symbol} Ex-Ante Directional-Run Direction Test\n\n")
        f.write("**Warning:** descriptive ex-ante directional-run research only, not trading guidance.\n\n")
        f.write("## Explanation\n")
        f.write("- Family C looked strong descriptively.\n")
        f.write("- This test checks whether direction can be selected using event-time data only.\n")
        f.write("- Expectancy is primary; positive rate is secondary.\n")
        f.write("- Selected direction should beat opposite direction and random baseline.\n\n")
        f.write("## Cost assumptions\n")
        f.write(f"- Friction pips per directional run: {friction:.2f}.\n")
        f.write("- Formula: 2*spread + 2*slippage + commission_equivalent.\n\n")
        f.write("## Direction methods tested\n")
        methods = summary["direction_method"].dropna().unique().tolist() if not summary.empty else []
        for m in methods:
            f.write(f"- {m}\n")
        f.write("\n")
        if summary.empty:
            f.write("## No rows available\nNo valid directional-run rows were produced after indicator and horizon filtering.\n")
            return

        interesting = summary[(summary["observations"] >= 200) & (summary["OOS_mean_selected_run_after_friction"] > 0) &
                              (summary["OOS_mean_edge_vs_opposite"] > 0) & (summary["OOS_mean_edge_vs_random"] > 0) &
                              (summary["OOS_expectancy_from_components"] > 0)]

        def write_top(title: str, df: pd.DataFrame, col: str, asc: bool = False):
            f.write(f"## {title}\n")
            if df.empty:
                f.write("No qualifying contexts.\n\n")
                return
            cols = ["timeframe", "direction_method", "future_horizon_bars", "allowed_pullback_atr", "session_bucket", "hour", "observations", col]
            f.write(df.sort_values(col, ascending=asc).head(20)[cols].to_markdown(index=False))
            f.write("\n\n")

        write_top("Top 20 contexts by OOS selected expectancy", interesting, "OOS_expectancy_from_components")
        write_top("Top 20 contexts by OOS edge vs opposite", interesting, "OOS_mean_edge_vs_opposite")
        write_top("Top 20 contexts by OOS edge vs random", interesting, "OOS_mean_edge_vs_random")

        write_top("Worst tail-risk contexts", summary, "p05_selected_run_after_friction", asc=True)

        f.write("## Method comparison\n")
        f.write(summary.groupby("direction_method", dropna=False)[["mean_selected_run_after_friction", "OOS_expectancy_from_components", "mean_edge_vs_opposite", "mean_edge_vs_random"]].mean().reset_index().sort_values("OOS_expectancy_from_components", ascending=False).to_markdown(index=False))
        f.write("\n\n## Timeframe comparison\n")
        f.write(summary.groupby("timeframe", dropna=False)[["mean_selected_run_after_friction", "OOS_expectancy_from_components", "mean_edge_vs_opposite", "mean_edge_vs_random"]].mean().reset_index().sort_values("OOS_expectancy_from_components", ascending=False).to_markdown(index=False))
        f.write("\n\n## Session comparison\n")
        f.write(summary.groupby("session_bucket", dropna=False)[["mean_selected_run_after_friction", "OOS_expectancy_from_components", "mean_edge_vs_opposite", "mean_edge_vs_random"]].mean().reset_index().sort_values("OOS_expectancy_from_components", ascending=False).to_markdown(index=False))
        f.write("\n\n## Final interpretation\n")
        f.write("- Evaluate whether any ex-ante direction method shows real directional-run edge.\n")
        f.write("- Evaluate whether edge survives OOS stability checks.\n")
        f.write("- Evaluate whether selected direction beats opposite direction.\n")
        f.write("- Evaluate whether selected direction beats random baseline.\n")
        f.write("- Evaluate whether adverse movement is too large relative to directional run.\n")
        f.write("- Next step is locked-method robustness research, not EA development.\n\n")
        if not yearly.empty:
            f.write("## Yearly summary\n")
            f.write(yearly.sort_values(["timeframe", "direction_method", "future_horizon_bars", "allowed_pullback_atr", "year"]).to_markdown(index=False))
            f.write("\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/EURUSD_M15_MT5_5Y.csv")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--output-dir", default="directional_run_ex_ante_reports")
    p.add_argument("--timeframes", default="H1,H4,D1")
    p.add_argument("--future-horizons", default="20,40,80")
    p.add_argument("--allowed-pullback-atr", default="1.0,1.5")
    p.add_argument("--cost-profile", choices=["low", "conservative", "high"], default="conservative")
    p.add_argument("--atr-period", type=int, default=14)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tfs = parse_list(args.timeframes, str)
    horizons = parse_list(args.future_horizons, int)
    pullbacks = parse_list(args.allowed_pullback_atr, float)
    methods = [
        "momentum_20", "momentum_50", "ma_slope_20", "ma_position_50",
        "breakout_state_20", "breakout_state_50", "recent_range_position", "random_baseline_deterministic",
    ]

    pip_size = infer_pip_size(args.symbol)
    friction = get_friction_pips(args.symbol, args.cost_profile)

    base = normalize_ohlc(pd.read_csv(args.csv))

    events = []
    for tf in tfs:
        print(f"[Progress] Timeframe={tf}")
        d = add_indicators(resample_tf(base, tf), args.atr_period, pip_size)
        for method in methods:
            print(f"[Progress] Timeframe={tf}, Method={method}")
            random_cache: Dict[Tuple[int, int, float], float] = {}
            for i in range(len(d)):
                atrp = d.at[i, "atr_pips"]
                if pd.isna(atrp) or atrp <= 0:
                    continue
                sel = get_method_direction(d.iloc[i], method, tf)
                if sel is None:
                    continue
                for h in horizons:
                    for pb in pullbacks:
                        m_sel = calc_directional_run(d, i, h, sel, pb, pip_size)
                        m_opp = calc_directional_run(d, i, h, -sel, pb, pip_size)
                        if m_sel is None or m_opp is None:
                            continue
                        sel_run = m_sel[0] - friction
                        opp_run = m_opp[0] - friction
                        dt = d.at[i, "datetime"]
                        key = (i, h, pb)
                        if key not in random_cache:
                            rand_dir = get_method_direction(d.iloc[i], "random_baseline_deterministic", tf)
                            m_rand = calc_directional_run(d, i, h, rand_dir, pb, pip_size)
                            random_cache[key] = (m_rand[0] - friction) if m_rand is not None else np.nan
                        rand_run = random_cache[key]
                        edge_rand = sel_run - rand_run if pd.notna(rand_run) else np.nan
                        if method == "random_baseline_deterministic":
                            edge_rand = np.nan
                        events.append({
                            "symbol": args.symbol,
                            "timeframe": tf,
                            "datetime": dt,
                            "year": dt.year,
                            "month": dt.month,
                            "day_of_week": dt.dayofweek,
                            "hour": dt.hour,
                            "session_bucket": session_bucket(dt.hour),
                            "direction_method": method,
                            "selected_direction": sel,
                            "future_horizon_bars": h,
                            "allowed_pullback_atr": pb,
                            "atr_pips": atrp,
                            "friction_pips": friction,
                            "selected_max_favorable_pips_before_pullback": m_sel[0],
                            "selected_run_after_friction": sel_run,
                            "selected_max_adverse_pips_before_run": m_sel[1],
                            "selected_bars_to_max_favorable": m_sel[2],
                            "selected_did_allowed_pullback_trigger": m_sel[3],
                            "opposite_run_after_friction": opp_run,
                            "edge_vs_opposite": sel_run - opp_run,
                            "random_baseline_run_after_friction": rand_run,
                            "edge_vs_random": edge_rand,
                            "range_position": d.at[i, "range_position"],
                            "momentum_20_pips": d.at[i, "momentum_20_pips"],
                            "momentum_50_pips": d.at[i, "momentum_50_pips"],
                        })

    events_df = pd.DataFrame(events)
    event_csv = os.path.join(args.output_dir, f"{args.symbol}_directional_run_ex_ante_events.csv")
    summary_csv = os.path.join(args.output_dir, f"{args.symbol}_directional_run_ex_ante_summary.csv")
    summary_md = os.path.join(args.output_dir, f"{args.symbol}_directional_run_ex_ante_summary.md")

    if events_df.empty:
        events_df = pd.DataFrame(columns=[
            "symbol", "timeframe", "datetime", "year", "month", "day_of_week", "hour", "session_bucket",
            "direction_method", "selected_direction", "future_horizon_bars", "allowed_pullback_atr", "atr_pips", "friction_pips",
            "selected_max_favorable_pips_before_pullback", "selected_run_after_friction", "selected_max_adverse_pips_before_run",
            "selected_bars_to_max_favorable", "selected_did_allowed_pullback_trigger", "opposite_run_after_friction", "edge_vs_opposite",
            "random_baseline_run_after_friction", "edge_vs_random", "range_position", "momentum_20_pips", "momentum_50_pips",
        ])
        events_df.to_csv(event_csv, index=False)
        pd.DataFrame().to_csv(summary_csv, index=False)
        markdown_report(args.symbol, friction, pd.DataFrame(), pd.DataFrame(), summary_md)
        print("No rows produced. Empty outputs written.")
        return

    events_df = events_df.sort_values(["timeframe", "datetime", "direction_method", "future_horizon_bars", "allowed_pullback_atr"]).reset_index(drop=True)
    summary_df, yearly_df = summarize(events_df)
    if not yearly_df.empty:
        summary_df = summary_df.merge(
            yearly_df.groupby(["timeframe", "direction_method", "future_horizon_bars", "allowed_pullback_atr"], dropna=False)
            .agg(
                yearly_mean_selected_run_after_friction=("year_mean_selected_run_after_friction", "mean"),
                yearly_mean_edge_vs_opposite=("year_mean_edge_vs_opposite", "mean"),
                yearly_mean_edge_vs_random=("year_mean_edge_vs_random", "mean"),
            ).reset_index(),
            on=["timeframe", "direction_method", "future_horizon_bars", "allowed_pullback_atr"],
            how="left",
        )

    events_df.to_csv(event_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    markdown_report(args.symbol, friction, summary_df, yearly_df, summary_md)

    print(f"Wrote: {event_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {summary_md}")


if __name__ == "__main__":
    main()
