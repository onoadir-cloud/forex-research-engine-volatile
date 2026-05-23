#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
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


@dataclass
class Metrics:
    raw_outcome_pips: float
    outcome_after_friction: float
    max_favorable_pips: float
    max_adverse_pips: float
    max_favorable_atr: float
    max_adverse_atr: float
    bars_to_max_favorable: int
    bars_to_max_adverse: int


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
    if sym in COST_PROFILES:
        c = COST_PROFILES[sym][profile]
    else:
        c = COST_PROFILES["EURUSD"][profile]
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
    m = {"M30": "30min", "H1": "1h", "H4": "4h", "D1": "1D"}
    if tf not in m:
        raise ValueError(f"Unsupported timeframe: {tf}")
    x = df.set_index("datetime").resample(m[tf]).agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    x = x.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return x


def add_atr(df: pd.DataFrame, atr_period: int, pip_size: float) -> pd.DataFrame:
    d = df.copy()
    prev_close = d["close"].shift(1)
    tr = np.maximum.reduce([
        (d["high"] - d["low"]).values,
        np.abs((d["high"] - prev_close)).values,
        np.abs((d["low"] - prev_close)).values,
    ])
    d["atr"] = pd.Series(tr, index=d.index).rolling(atr_period, min_periods=atr_period).mean()
    d["atr_pips"] = d["atr"] / pip_size
    return d


def calc_path_metrics(df: pd.DataFrame, i: int, h: int, direction: int, pip_size: float, atr_pips_event: float) -> Optional[Metrics]:
    end_i = i + h
    if end_i >= len(df):
        return None
    event_close = df.at[i, "close"]
    window = df.iloc[i + 1 : end_i + 1]
    if window.empty:
        return None
    future_close = df.at[end_i, "close"]
    raw = direction * (future_close - event_close) / pip_size

    if direction == 1:
        favorable = (window["high"] - event_close) / pip_size
        adverse = (window["low"] - event_close) / pip_size
    else:
        favorable = (event_close - window["low"]) / pip_size
        adverse = (event_close - window["high"]) / pip_size

    max_fav = float(favorable.max())
    max_adv = float(adverse.min())
    b_fav = int(np.argmax(favorable.values) + 1)
    b_adv = int(np.argmin(adverse.values) + 1)

    fav_atr = max_fav / atr_pips_event if pd.notna(atr_pips_event) and atr_pips_event > 0 else np.nan
    adv_atr = abs(max_adv) / atr_pips_event if pd.notna(atr_pips_event) and atr_pips_event > 0 else np.nan

    return Metrics(raw, np.nan, max_fav, max_adv, fav_atr, adv_atr, b_fav, b_adv)


def build_base_event(df: pd.DataFrame, i: int, symbol: str, tf: str, family: str, lookback: Optional[int], h: int, direction: int) -> Dict:
    dt = df.at[i, "datetime"]
    return {
        "symbol": symbol,
        "timeframe": tf,
        "family": family,
        "datetime": dt,
        "year": dt.year,
        "month": dt.month,
        "day_of_week": dt.dayofweek,
        "hour": dt.hour,
        "session_bucket": session_bucket(dt.hour),
        "lookback_bars": lookback,
        "future_horizon_bars": h,
        "direction": direction,
        "event_type": None,
        "atr_pips": df.at[i, "atr_pips"],
    }


def family_a(df, symbol, tf, lookbacks, horizons, pip_size, friction):
    events = []
    for n in lookbacks:
        ph = df["high"].shift(1).rolling(n, min_periods=n).max()
        pl = df["low"].shift(1).rolling(n, min_periods=n).min()
        for i in range(len(df)):
            c = df.at[i, "close"]
            if pd.isna(ph.iat[i]) or pd.isna(pl.iat[i]) or pd.isna(df.at[i, "atr_pips"]):
                continue
            typ = None
            direction = 0
            if c > ph.iat[i]:
                typ, direction = "up_breakout", 1
            elif c < pl.iat[i]:
                typ, direction = "down_breakout", -1
            if direction == 0:
                continue
            for h in horizons:
                m = calc_path_metrics(df, i, h, direction, pip_size, df.at[i, "atr_pips"])
                if m is None:
                    continue
                row = build_base_event(df, i, symbol, tf, "Family_A_Breakout_FollowThrough", n, h, direction)
                row.update({
                    "event_type": typ,
                    "friction_pips": friction,
                    "raw_outcome_pips": m.raw_outcome_pips,
                    "outcome_after_friction": m.raw_outcome_pips - friction,
                    "max_favorable_pips": m.max_favorable_pips,
                    "max_adverse_pips": m.max_adverse_pips,
                    "max_favorable_atr": m.max_favorable_atr,
                    "max_adverse_atr": m.max_adverse_atr,
                    "bars_to_max_favorable": m.bars_to_max_favorable,
                    "bars_to_max_adverse": m.bars_to_max_adverse,
                    "pullback_bucket": np.nan,
                    "pullback_atr": np.nan,
                    "leg_atr_multiple": np.nan,
                    "allowed_pullback_atr": np.nan,
                    "run_after_friction": np.nan,
                    "did_allowed_pullback_trigger": np.nan,
                })
                events.append(row)
    return events


def bucket_pullback(x: float) -> str:
    if x < 0.5:
        return "0_0.5_ATR"
    if x < 1.0:
        return "0.5_1.0_ATR"
    if x < 1.5:
        return "1.0_1.5_ATR"
    if x < 2.0:
        return "1.5_2.0_ATR"
    return "2.0_plus_ATR"


def family_b(df, symbol, tf, lookbacks, horizons, pip_size, friction):
    events = []
    for n in lookbacks:
        rlow = df["low"].rolling(n, min_periods=n).min()
        rhigh = df["high"].rolling(n, min_periods=n).max()
        for i in range(len(df)):
            atrp = df.at[i, "atr_pips"]
            if pd.isna(atrp) or atrp <= 0 or pd.isna(rlow.iat[i]) or pd.isna(rhigh.iat[i]):
                continue
            c = df.at[i, "close"]
            up_leg_pips = (c - rlow.iat[i]) / pip_size
            down_leg_pips = (rhigh.iat[i] - c) / pip_size
            up_leg_atr = up_leg_pips / atrp
            down_leg_atr = down_leg_pips / atrp

            candidates = []
            if up_leg_atr >= 2.0:
                pb_pips = (rhigh.iat[i] - c) / pip_size
                pb_atr = pb_pips / atrp
                if pb_atr >= 0:
                    candidates.append((1, "pullback_continuation_up", pb_atr, up_leg_atr))
            if down_leg_atr >= 2.0:
                pb_pips = (c - rlow.iat[i]) / pip_size
                pb_atr = pb_pips / atrp
                if pb_atr >= 0:
                    candidates.append((-1, "pullback_continuation_down", pb_atr, down_leg_atr))

            for direction, etype, pb_atr, leg_atr in candidates:
                pb_bucket = bucket_pullback(pb_atr)
                for h in horizons:
                    m = calc_path_metrics(df, i, h, direction, pip_size, atrp)
                    if m is None:
                        continue
                    row = build_base_event(df, i, symbol, tf, "Family_B_Pullback_Continuation", n, h, direction)
                    row.update({
                        "event_type": etype,
                        "friction_pips": friction,
                        "raw_outcome_pips": m.raw_outcome_pips,
                        "outcome_after_friction": m.raw_outcome_pips - friction,
                        "max_favorable_pips": m.max_favorable_pips,
                        "max_adverse_pips": m.max_adverse_pips,
                        "max_favorable_atr": m.max_favorable_atr,
                        "max_adverse_atr": m.max_adverse_atr,
                        "bars_to_max_favorable": m.bars_to_max_favorable,
                        "bars_to_max_adverse": m.bars_to_max_adverse,
                        "pullback_bucket": pb_bucket,
                        "pullback_atr": pb_atr,
                        "leg_atr_multiple": leg_atr,
                        "allowed_pullback_atr": np.nan,
                        "run_after_friction": np.nan,
                        "did_allowed_pullback_trigger": np.nan,
                    })
                    events.append(row)
    return events


def family_c(df, symbol, tf, horizons, pullback_thresholds, pip_size, friction):
    events = []
    for i in range(len(df)):
        atrp = df.at[i, "atr_pips"]
        if pd.isna(atrp) or atrp <= 0:
            continue
        c0 = df.at[i, "close"]
        for direction in (1, -1):
            for h in horizons:
                end_i = i + h
                if end_i >= len(df):
                    continue
                window = df.iloc[i + 1 : end_i + 1]
                if window.empty:
                    continue
                for x in pullback_thresholds:
                    allowed = x * atrp
                    run_max = 0.0
                    max_adv = 0.0
                    bars_to_pb = np.nan
                    bars_to_max = 0
                    triggered = False
                    if direction == 1:
                        running_high = c0
                        for j, (_, r) in enumerate(window.iterrows(), start=1):
                            running_high = max(running_high, r["high"])
                            fav = (running_high - c0) / pip_size
                            adv = (r["low"] - c0) / pip_size
                            if fav > run_max:
                                run_max = fav
                                bars_to_max = j
                            max_adv = min(max_adv, adv)
                            pb = (running_high - r["low"]) / pip_size
                            if pb >= allowed:
                                bars_to_pb = j
                                triggered = True
                                break
                    else:
                        running_low = c0
                        for j, (_, r) in enumerate(window.iterrows(), start=1):
                            running_low = min(running_low, r["low"])
                            fav = (c0 - running_low) / pip_size
                            adv = (c0 - r["high"]) / pip_size
                            if fav > run_max:
                                run_max = fav
                                bars_to_max = j
                            max_adv = min(max_adv, adv)
                            pb = (r["high"] - running_low) / pip_size
                            if pb >= allowed:
                                bars_to_pb = j
                                triggered = True
                                break

                    row = build_base_event(df, i, symbol, tf, "Family_C_Directional_Run_Before_Allowed_Pullback", None, h, direction)
                    row.update({
                        "event_type": "directional_run_probe",
                        "friction_pips": friction,
                        "raw_outcome_pips": run_max,
                        "outcome_after_friction": run_max - friction,
                        "max_favorable_pips": run_max,
                        "max_adverse_pips": max_adv,
                        "max_favorable_atr": run_max / atrp,
                        "max_adverse_atr": abs(max_adv) / atrp,
                        "bars_to_max_favorable": bars_to_max,
                        "bars_to_max_adverse": np.nan,
                        "pullback_bucket": np.nan,
                        "pullback_atr": np.nan,
                        "leg_atr_multiple": np.nan,
                        "allowed_pullback_atr": x,
                        "run_after_friction": run_max - friction,
                        "did_allowed_pullback_trigger": int(triggered),
                    })
                    events.append(row)
    return events


def expectancy_components(s: pd.Series) -> Dict[str, float]:
    pos = s[s > 0]
    neg = s[s < 0]
    avg_pos = pos.mean() if len(pos) else np.nan
    avg_neg = neg.mean() if len(neg) else np.nan
    pr = len(pos) / len(s) if len(s) else np.nan
    nr = len(neg) / len(s) if len(s) else np.nan
    payoff = avg_pos / abs(avg_neg) if pd.notna(avg_pos) and pd.notna(avg_neg) and avg_neg != 0 else np.nan
    exp = pr * avg_pos - nr * abs(avg_neg) if pd.notna(pr) and pd.notna(avg_pos) and pd.notna(nr) and pd.notna(avg_neg) else np.nan
    return {
        "avg_positive_outcome": avg_pos,
        "avg_negative_outcome": avg_neg,
        "positive_rate": pr,
        "negative_rate": nr,
        "payoff_ratio": payoff,
        "expectancy_from_components": exp,
    }


def seq_stress(s: pd.Series) -> Dict[str, float]:
    is_neg = (s < 0).astype(int)
    longest = 0
    cur = 0
    for v in is_neg:
        cur = cur + 1 if v == 1 else 0
        longest = max(longest, cur)
    out = {"longest_consecutive_negative_outcomes": longest}
    for w in (20, 50, 100):
        out[f"max_rolling_{w}_negative_outcome_rate"] = is_neg.rolling(w, min_periods=min(w, len(is_neg))).mean().max() if len(is_neg) else np.nan
    return out


def is_oos_metrics(g: pd.DataFrame) -> Dict[str, float]:
    g = g.sort_values("datetime")
    cut = int(np.floor(len(g) * 0.7))
    is_df = g.iloc[:cut]
    oos_df = g.iloc[cut:]
    out = {"IS_observations": len(is_df), "OOS_observations": len(oos_df)}
    for tag, sub in (("IS", is_df), ("OOS", oos_df)):
        s = sub["outcome_after_friction"].dropna()
        out[f"{tag}_mean_outcome_after_friction"] = s.mean() if len(s) else np.nan
        comps = expectancy_components(s) if len(s) else {"expectancy_from_components": np.nan}
        out[f"{tag}_expectancy_from_components"] = comps["expectancy_from_components"]
        out[f"{tag}_p05_outcome_after_friction"] = s.quantile(0.05) if len(s) else np.nan
        out[f"{tag}_worst_single_outcome"] = s.min() if len(s) else np.nan
    return out


def summarize(events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    gcols = ["family", "timeframe", "lookback_bars", "future_horizon_bars", "direction", "session_bucket", "hour", "allowed_pullback_atr"]
    rows = []
    for key, g in events.groupby(gcols, dropna=False):
        s = g["outcome_after_friction"].dropna()
        if len(s) == 0:
            continue
        row = dict(zip(gcols, key))
        row["observations"] = len(s)
        row["mean_outcome_after_friction"] = s.mean()
        row["median_outcome_after_friction"] = s.median()
        row.update(expectancy_components(s))
        for q, n in ((0.01, "p01"), (0.05, "p05"), (0.10, "p10"), (0.90, "p90"), (0.95, "p95"), (0.99, "p99")):
            row[f"{n}_outcome_after_friction"] = s.quantile(q)
        row["worst_single_outcome"] = s.min()
        row["best_single_outcome"] = s.max()
        row["median_max_favorable_pips"] = g["max_favorable_pips"].median()
        row["median_max_adverse_pips"] = g["max_adverse_pips"].median()
        row["p90_max_adverse_pips"] = g["max_adverse_pips"].quantile(0.9)
        row["worst_max_adverse_pips"] = g["max_adverse_pips"].min()
        row["median_max_adverse_atr"] = g["max_adverse_atr"].median()
        row["p90_max_adverse_atr"] = g["max_adverse_atr"].quantile(0.9)
        row["median_bars_to_max_favorable"] = g["bars_to_max_favorable"].median()
        row["median_bars_to_max_adverse"] = g["bars_to_max_adverse"].median()
        row.update(is_oos_metrics(g))
        row.update(seq_stress(g.sort_values("datetime")["outcome_after_friction"]))
        rows.append(row)
    summary = pd.DataFrame(rows)

    ycols = ["family", "timeframe", "lookback_bars", "future_horizon_bars", "direction", "year"]
    yrows = []
    for key, g in events.groupby(ycols, dropna=False):
        s = g["outcome_after_friction"].dropna()
        if len(s) == 0:
            continue
        row = dict(zip(ycols, key))
        row["year_observations"] = len(s)
        row["year_mean_outcome_after_friction"] = s.mean()
        row["year_median_outcome_after_friction"] = s.median()
        c = expectancy_components(s)
        row["year_expectancy_from_components"] = c["expectancy_from_components"]
        row["year_positive_rate"] = c["positive_rate"]
        row["year_avg_positive_outcome"] = c["avg_positive_outcome"]
        row["year_avg_negative_outcome"] = c["avg_negative_outcome"]
        row["year_p05_outcome_after_friction"] = s.quantile(0.05)
        row["year_worst_single_outcome"] = s.min()
        yrows.append(row)
    return summary, pd.DataFrame(yrows)


def markdown_report(symbol, tfs, friction, summary, yearly, out_md):
    def safe_table(df, requested_cols, n=20):
        if df.empty:
            return "No rows available for this section."
        safe_cols = [c for c in requested_cols if c in df.columns]
        if not safe_cols:
            return "No rows available for this section."
        trimmed = df[safe_cols].head(n)
        if trimmed.empty:
            return "No rows available for this section."
        return trimmed.to_markdown(index=False)

    top_oos = summary.sort_values("OOS_expectancy_from_components", ascending=False) if not summary.empty else pd.DataFrame()
    top_payoff = summary[summary["observations"] >= 200].sort_values("payoff_ratio", ascending=False) if not summary.empty else pd.DataFrame()
    worst_tail = summary.sort_values("p05_outcome_after_friction") if not summary.empty else pd.DataFrame()

    yr_pos = pd.DataFrame()
    if not yearly.empty:
        y = yearly.copy()
        y["year_pos"] = y["year_expectancy_from_components"] > 0
        yr_pos = y.groupby(["family", "timeframe", "lookback_bars", "future_horizon_bars", "direction"], dropna=False)["year_pos"].sum().reset_index(name="yearly_positive_expectancy_count")

    interesting = summary.copy()
    if not interesting.empty:
        interesting = interesting.merge(yr_pos, on=["family", "timeframe", "lookback_bars", "future_horizon_bars", "direction"], how="left")
        interesting["yearly_positive_expectancy_count"] = interesting["yearly_positive_expectancy_count"].fillna(0)
        enough_years = interesting["yearly_positive_expectancy_count"] >= 3
        interesting = interesting[(interesting["observations"] >= 200) & (interesting["OOS_expectancy_from_components"] > 0) & (interesting["mean_outcome_after_friction"] > 0) & (interesting["payoff_ratio"] > 1.2) & (enough_years | (interesting["observations"] < 3))]

    tf_comp = summary.groupby("timeframe", dropna=False)[["mean_outcome_after_friction", "OOS_expectancy_from_components", "payoff_ratio"]].mean().reset_index() if not summary.empty else pd.DataFrame()
    ses_comp = summary.groupby("session_bucket", dropna=False)[["mean_outcome_after_friction", "OOS_expectancy_from_components", "payoff_ratio"]].mean().reset_index() if not summary.empty else pd.DataFrame()
    fam_comp = summary.groupby("family", dropna=False)[["mean_outcome_after_friction", "OOS_expectancy_from_components", "payoff_ratio"]].mean().reset_index() if not summary.empty else pd.DataFrame()

    interesting_cols = [
        "family", "timeframe", "lookback_bars", "future_horizon_bars", "direction", "session_bucket", "hour",
        "observations", "mean_outcome_after_friction", "OOS_expectancy_from_components", "payoff_ratio",
        "yearly_positive_expectancy_count",
    ]
    top_oos_cols = [
        "family", "timeframe", "lookback_bars", "future_horizon_bars", "direction", "session_bucket", "hour",
        "observations", "OOS_expectancy_from_components", "mean_outcome_after_friction", "payoff_ratio",
        "p05_outcome_after_friction", "worst_single_outcome", "yearly_positive_expectancy_count",
    ]
    top_payoff_cols = [
        "family", "timeframe", "lookback_bars", "future_horizon_bars", "direction", "session_bucket", "hour",
        "observations", "payoff_ratio", "OOS_expectancy_from_components", "mean_outcome_after_friction",
        "p05_outcome_after_friction", "worst_single_outcome", "yearly_positive_expectancy_count",
    ]
    worst_tail_cols = [
        "family", "timeframe", "lookback_bars", "future_horizon_bars", "direction", "session_bucket", "hour",
        "observations", "p05_outcome_after_friction", "worst_single_outcome",
        "OOS_expectancy_from_components", "payoff_ratio", "mean_outcome_after_friction",
    ]
    comp_cols = ["timeframe", "session_bucket", "family", "mean_outcome_after_friction", "OOS_expectancy_from_components", "payoff_ratio"]

    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    sections = {
        "interesting": safe_table(interesting, interesting_cols, 20),
        "top_oos": safe_table(top_oos, top_oos_cols, 20),
        "top_payoff": safe_table(top_payoff, top_payoff_cols, 20),
        "worst_tail": safe_table(worst_tail, worst_tail_cols, 20),
        "tf_comp": safe_table(tf_comp, comp_cols, 20),
        "ses_comp": safe_table(ses_comp, comp_cols, 20),
        "fam_comp": safe_table(fam_comp, comp_cols, 20),
    }
    lines = [
        f"# {symbol} Trend/Swing Expectancy Atlas",
        "",
        "**Warning:** Descriptive trend/swing expectancy research only. This is not trading guidance.",
        "",
        "- Expectancy is primary; positive rate is secondary.",
        "- Filters use event-time known information only.",
        "- No EA, execution rules, account equity, or monetary PnL are produced.",
        "",
        "## Cost assumptions",
        f"- Friction (round-trip equivalent): **{friction:.2f} pips**.",
        "",
        "## Timeframes tested",
        f"- {', '.join(tfs)}",
        "",
        "## Executive summary",
        "- Strongest family/contexts are ranked by OOS expectancy table below.",
        "- Weak/noisy areas are visible in tail-risk and low OOS expectancy rows.",
        "- Compare H1/H4/D1 vs lower timeframes in timeframe comparison for friction sensitivity.",
        "- Payoff ratio progression is summarized in payoff and family/timeframe tables.",
        "",
        "## Interesting contexts (research-only filters)",
        sections["interesting"],
        "",
        "## Top 20 contexts by OOS expectancy",
        sections["top_oos"],
        "",
        "## Top 20 contexts by payoff ratio (observations >= 200)",
        sections["top_payoff"],
        "",
        "## Worst tail-risk contexts",
        sections["worst_tail"],
        "",
        "## Timeframe comparison",
        sections["tf_comp"],
        "",
        "## Session comparison",
        sections["ses_comp"],
        "",
        "## Family comparison",
        sections["fam_comp"],
        "",
        "## Final interpretation",
        "- Any promising trend leg, breakout, pullback continuation, or directional run context should be taken to locked context deep-check research.",
        "- Results should be treated as descriptive expectancy evidence and OOS stability evidence, not as direct trading rules.",
        "- Next step is locked context robustness review, not EA creation.",
    ]
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description="Descriptive trend/swing expectancy atlas builder.")
    p.add_argument("--csv", default="data/EURUSD_M15_MT5_5Y.csv")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--output-dir", default="trend_swing_expectancy_reports")
    p.add_argument("--timeframes", default="M30,H1,H4,D1")
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--lookback-highlow", default="20,50,100")
    p.add_argument("--future-horizons", default="10,20,40,80")
    p.add_argument("--pullback-atr-thresholds", default="0.5,1.0,1.5,2.0")
    p.add_argument("--cost-profile", choices=["low", "conservative", "high"], default="conservative")
    args = p.parse_args()

    timeframes = [x.strip() for x in args.timeframes.split(",") if x.strip()]
    lookbacks = parse_list(args.lookback_highlow, int)
    horizons = parse_list(args.future_horizons, int)
    pullbacks = parse_list(args.pullback_atr_thresholds, float)

    os.makedirs(args.output_dir, exist_ok=True)
    pip_size = infer_pip_size(args.symbol)
    friction = get_friction_pips(args.symbol, args.cost_profile)

    raw = pd.read_csv(args.csv)
    base = normalize_ohlc(raw)

    all_events = []
    for tf in timeframes:
        print(f"[Progress] Timeframe={tf}: resampling and ATR computation")
        d = resample_tf(base, tf)
        if len(d) < max(max(lookbacks), args.atr_period) + max(horizons) + 5:
            print(f"[Warning] Timeframe={tf} skipped (too few rows: {len(d)})")
            continue
        d = add_atr(d, args.atr_period, pip_size)
        print(f"[Progress] Timeframe={tf}: Family A breakout follow-through")
        all_events.extend(family_a(d, args.symbol, tf, lookbacks, horizons, pip_size, friction))
        print(f"[Progress] Timeframe={tf}: Family B pullback continuation")
        all_events.extend(family_b(d, args.symbol, tf, lookbacks, horizons, pip_size, friction))
        print(f"[Progress] Timeframe={tf}: Family C directional run")
        all_events.extend(family_c(d, args.symbol, tf, horizons, pullbacks, pip_size, friction))

    events = pd.DataFrame(all_events)
    cols = [
        "symbol", "timeframe", "family", "datetime", "year", "month", "day_of_week", "hour", "session_bucket",
        "lookback_bars", "future_horizon_bars", "direction", "event_type", "atr_pips", "friction_pips", "raw_outcome_pips",
        "outcome_after_friction", "max_favorable_pips", "max_adverse_pips", "max_favorable_atr", "max_adverse_atr",
        "bars_to_max_favorable", "bars_to_max_adverse", "pullback_bucket", "pullback_atr", "leg_atr_multiple",
        "allowed_pullback_atr", "run_after_friction", "did_allowed_pullback_trigger"
    ]
    if events.empty:
        events = pd.DataFrame(columns=cols)
    else:
        for c in cols:
            if c not in events.columns:
                events[c] = np.nan
        events = events[cols].sort_values(["datetime", "family", "timeframe"]).reset_index(drop=True)

    summary, yearly = summarize(events)
    events_path = os.path.join(args.output_dir, f"{args.symbol}_trend_swing_expectancy_events.csv")
    summary_path = os.path.join(args.output_dir, f"{args.symbol}_trend_swing_expectancy_summary.csv")
    md_path = os.path.join(args.output_dir, f"{args.symbol}_trend_swing_expectancy_summary.md")

    events.to_csv(events_path, index=False)
    if summary.empty:
        summary = pd.DataFrame()
    if not yearly.empty:
        summary = summary.merge(
            yearly.groupby(["family", "timeframe", "lookback_bars", "future_horizon_bars", "direction"], dropna=False)["year_expectancy_from_components"]
            .apply(lambda x: int((x > 0).sum()))
            .reset_index(name="yearly_positive_expectancy_count"),
            on=["family", "timeframe", "lookback_bars", "future_horizon_bars", "direction"],
            how="left",
        )
    summary.to_csv(summary_path, index=False)
    markdown_report(args.symbol, timeframes, friction, summary, yearly, md_path)

    print(f"[Done] Events: {events_path}")
    print(f"[Done] Summary CSV: {summary_path}")
    print(f"[Done] Summary MD: {md_path}")


if __name__ == "__main__":
    main()
