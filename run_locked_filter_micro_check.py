#!/usr/bin/env python3
"""Run a locked descriptive micro-check for EURUSD/GBPUSD event-time filtered context."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


INPUT_PATH = Path("signed_relative_micro_reports/EURUSD_GBPUSD_signed_relative_events.csv")
OUTPUT_DIR = Path("locked_filter_micro_reports")
OUTPUT_EVENTS = OUTPUT_DIR / "EURUSD_GBPUSD_locked_filter_events.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "EURUSD_GBPUSD_locked_filter_summary.csv"
OUTPUT_SUMMARY_MD = OUTPUT_DIR / "EURUSD_GBPUSD_locked_filter_summary.md"

REQUIRED_RAW_COLUMNS = [
    "datetime",
    "year",
    "current_z",
    "first_touch_0_5_vs_3",
    "bars_to_first_touch",
    "basket_move_pips_proxy",
    "sum_leg_move_pips_proxy",
    "estimated_round_trip_friction_pips",
    "conservative_behavior_after_friction",
    "sum_leg_behavior_after_friction",
]


def _rate(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.mean())


def _q(series: pd.Series, q: float) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.quantile(q))


def _safe_median(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.median())


def _fmt(v: float, digits: int = 4) -> str:
    if pd.isna(v):
        return "nan"
    return f"{v:.{digits}f}"


def _table_md(df: pd.DataFrame, float_digits: int = 4) -> str:
    if df.empty:
        return "(no rows)"
    render = df.copy()
    for col in render.columns:
        if pd.api.types.is_float_dtype(render[col]):
            render[col] = render[col].map(lambda x: _fmt(x, float_digits))
    return render.to_markdown(index=False)


def rolling_negative_rate(labels: pd.Series, window: int) -> float:
    if len(labels) < window:
        return float("nan")
    neg = (labels == "negative_after_friction").astype(int)
    return float(neg.rolling(window).mean().max())


def longest_negative_streak(labels: Iterable[str]) -> int:
    longest = 0
    current = 0
    for label in labels:
        if label == "negative_after_friction":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def extract_negative_clusters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "cluster_start_datetime",
                "cluster_end_datetime",
                "cluster_length",
                "cluster_year",
                "cluster_month",
                "median_current_abs_z",
                "median_bars_to_first_touch",
                "median_basket_move_pips_proxy",
                "median_conservative_behavior_after_friction",
            ]
        )

    negative_mask = df["conservative_behavior_after_friction"] <= 0
    cluster_rows = []
    i = 0
    while i < len(df):
        if not negative_mask.iloc[i]:
            i += 1
            continue
        start_i = i
        while i < len(df) and negative_mask.iloc[i]:
            i += 1
        end_i = i - 1
        chunk = df.iloc[start_i : end_i + 1]
        first_dt = chunk["datetime"].iloc[0]
        cluster_rows.append(
            {
                "cluster_start_datetime": first_dt,
                "cluster_end_datetime": chunk["datetime"].iloc[-1],
                "cluster_length": int(len(chunk)),
                "cluster_year": int(pd.Timestamp(first_dt).year),
                "cluster_month": int(pd.Timestamp(first_dt).month),
                "median_current_abs_z": _safe_median(chunk["current_abs_z"]),
                "median_bars_to_first_touch": _safe_median(chunk["bars_to_first_touch"]),
                "median_basket_move_pips_proxy": _safe_median(chunk["basket_move_pips_proxy"]),
                "median_conservative_behavior_after_friction": _safe_median(
                    chunk["conservative_behavior_after_friction"]
                ),
            }
        )

    clusters = pd.DataFrame(cluster_rows)
    return clusters.sort_values(
        ["cluster_length", "cluster_start_datetime"], ascending=[False, True]
    ).head(20)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns and c != "year"]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    selected_columns = [c for c in REQUIRED_RAW_COLUMNS if c in df.columns]
    df = df[selected_columns].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if "hour" not in df.columns:
        df["hour"] = df["datetime"].dt.hour
    if "year" not in df.columns:
        df["year"] = df["datetime"].dt.year
    if "current_abs_z" not in df.columns:
        df["current_abs_z"] = df["current_z"].abs()
    if "z_sign" not in df.columns:
        df["z_sign"] = np.where(df["current_z"] > 0, "positive_z", "negative_z")

    numeric_cols = [
        "year",
        "current_z",
        "bars_to_first_touch",
        "basket_move_pips_proxy",
        "sum_leg_move_pips_proxy",
        "estimated_round_trip_friction_pips",
        "conservative_behavior_after_friction",
        "sum_leg_behavior_after_friction",
        "hour",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["year"].isna().any():
        df.loc[df["year"].isna(), "year"] = df.loc[df["year"].isna(), "datetime"].dt.year

    df = df.dropna(
        subset=[
            "datetime",
            "year",
            "current_z",
            "first_touch_0_5_vs_3",
            "hour",
            "basket_move_pips_proxy",
            "conservative_behavior_after_friction",
            "bars_to_first_touch",
        ]
    ).copy()

    if "current_abs_z" not in df.columns:
        df["current_abs_z"] = df["current_z"].abs()
    if "z_sign" not in df.columns:
        df["z_sign"] = np.where(df["current_z"] > 0, "positive_z", "negative_z")
    df["after_friction_label"] = np.where(
        df["conservative_behavior_after_friction"] > 0,
        "positive_after_friction",
        "negative_after_friction",
    )

    locked = df.loc[
        (df["first_touch_0_5_vs_3"] == "normalized_0_5_first")
        & (df["hour"] == 15)
        & (df["current_abs_z"] >= 1.8)
        & (df["current_abs_z"] < 2.0)
    ].copy()

    locked = locked.sort_values("datetime").reset_index(drop=True)
    if locked.empty:
        locked.to_csv(OUTPUT_EVENTS, index=False)
        pd.DataFrame(
            columns=[
                "section",
                "metric",
                "value",
            ]
        ).to_csv(OUTPUT_SUMMARY_CSV, index=False)
        OUTPUT_SUMMARY_MD.write_text(
            "\n".join(
                [
                    "# EURUSD/GBPUSD Locked Filter Micro Check",
                    "",
                    "**Warning:** descriptive locked-filter diagnostics only, not trading guidance.",
                    "",
                    "## Locked Filter Definition",
                    "- first_touch_0_5_vs_3 == `normalized_0_5_first`",
                    "- hour == 15",
                    "- current_abs_z >= 1.8",
                    "- abs(current_z) < 2.0",
                    "",
                    "## Result",
                    "No rows matched the locked filter; outputs were written with headers only.",
                ]
            ),
            encoding="utf-8",
        )
        return

    locked["year"] = locked["year"].astype(int)
    locked["month"] = locked["datetime"].dt.month.astype(int)

    locked.to_csv(OUTPUT_EVENTS, index=False)

    overall = {
        "observations": int(len(locked)),
        "positive_after_friction_rate": _rate(
            locked["after_friction_label"] == "positive_after_friction"
        ),
        "negative_after_friction_rate": _rate(
            locked["after_friction_label"] == "negative_after_friction"
        ),
        "median_conservative_behavior_after_friction": _safe_median(
            locked["conservative_behavior_after_friction"]
        ),
        "p25_conservative_behavior_after_friction": _q(
            locked["conservative_behavior_after_friction"], 0.25
        ),
        "p75_conservative_behavior_after_friction": _q(
            locked["conservative_behavior_after_friction"], 0.75
        ),
        "median_basket_move_pips_proxy": _safe_median(locked["basket_move_pips_proxy"]),
        "p25_basket_move_pips_proxy": _q(locked["basket_move_pips_proxy"], 0.25),
        "p75_basket_move_pips_proxy": _q(locked["basket_move_pips_proxy"], 0.75),
        "p90_basket_move_pips_proxy": _q(locked["basket_move_pips_proxy"], 0.90),
        "median_sum_leg_behavior_after_friction": _safe_median(
            locked["sum_leg_behavior_after_friction"]
        ),
        "positive_sum_leg_after_friction_rate": _rate(
            locked["sum_leg_behavior_after_friction"] > 0
        ),
        "median_bars_to_first_touch": _safe_median(locked["bars_to_first_touch"]),
        "median_current_abs_z": _safe_median(locked["current_abs_z"]),
        "positive_z_rate": _rate(locked["z_sign"] == "positive_z"),
        "negative_z_rate": _rate(locked["z_sign"] == "negative_z"),
    }

    split_idx = int(len(locked) * 0.7)
    is_df = locked.iloc[:split_idx]
    oos_df = locked.iloc[split_idx:]
    is_oos = {
        "IS_observations": int(len(is_df)),
        "OOS_observations": int(len(oos_df)),
        "IS_positive_after_friction_rate": _rate(
            is_df["after_friction_label"] == "positive_after_friction"
        ),
        "OOS_positive_after_friction_rate": _rate(
            oos_df["after_friction_label"] == "positive_after_friction"
        ),
        "IS_median_conservative_behavior_after_friction": _safe_median(
            is_df["conservative_behavior_after_friction"]
        ),
        "OOS_median_conservative_behavior_after_friction": _safe_median(
            oos_df["conservative_behavior_after_friction"]
        ),
        "IS_median_basket_move_pips_proxy": _safe_median(is_df["basket_move_pips_proxy"]),
        "OOS_median_basket_move_pips_proxy": _safe_median(oos_df["basket_move_pips_proxy"]),
    }

    yearly = (
        locked.groupby("year", dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "year_observations": int(len(g)),
                    "year_positive_after_friction_rate": _rate(
                        g["after_friction_label"] == "positive_after_friction"
                    ),
                    "year_negative_after_friction_rate": _rate(
                        g["after_friction_label"] == "negative_after_friction"
                    ),
                    "year_median_conservative_behavior_after_friction": _safe_median(
                        g["conservative_behavior_after_friction"]
                    ),
                    "year_median_basket_move_pips_proxy": _safe_median(
                        g["basket_move_pips_proxy"]
                    ),
                    "year_median_bars_to_first_touch": _safe_median(
                        g["bars_to_first_touch"]
                    ),
                    "year_positive_z_rate": _rate(g["z_sign"] == "positive_z"),
                    "year_negative_z_rate": _rate(g["z_sign"] == "negative_z"),
                }
            )
        )
        .reset_index()
        .sort_values("year")
    )

    monthly = (
        locked.groupby(["year", "month"], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "month_observations": int(len(g)),
                    "month_positive_after_friction_rate": _rate(
                        g["after_friction_label"] == "positive_after_friction"
                    ),
                    "month_median_conservative_behavior_after_friction": _safe_median(
                        g["conservative_behavior_after_friction"]
                    ),
                }
            )
        )
        .reset_index()
        .sort_values(["year", "month"])
    )

    z_sign_summary = (
        locked.groupby("z_sign", dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "observations": int(len(g)),
                    "positive_after_friction_rate": _rate(
                        g["after_friction_label"] == "positive_after_friction"
                    ),
                    "median_conservative_behavior_after_friction": _safe_median(
                        g["conservative_behavior_after_friction"]
                    ),
                    "median_basket_move_pips_proxy": _safe_median(
                        g["basket_move_pips_proxy"]
                    ),
                    "median_bars_to_first_touch": _safe_median(g["bars_to_first_touch"]),
                }
            )
        )
        .reset_index()
        .sort_values("z_sign")
    )

    stress = {
        "longest_consecutive_negative_after_friction": longest_negative_streak(
            locked["after_friction_label"]
        ),
        "max_rolling_20_negative_after_friction_rate": rolling_negative_rate(
            locked["after_friction_label"], 20
        ),
        "max_rolling_50_negative_after_friction_rate": rolling_negative_rate(
            locked["after_friction_label"], 50
        ),
        "max_rolling_100_negative_after_friction_rate": rolling_negative_rate(
            locked["after_friction_label"], 100
        ),
    }

    clusters = extract_negative_clusters(locked)

    summary_rows = []
    summary_rows.extend(
        {"section": "overall", "metric": k, "value": v} for k, v in overall.items()
    )
    summary_rows.extend(
        {"section": "is_oos", "metric": k, "value": v} for k, v in is_oos.items()
    )
    summary_rows.extend(
        {"section": "sequence_stress", "metric": k, "value": v}
        for k, v in stress.items()
    )
    pd.DataFrame(summary_rows).to_csv(OUTPUT_SUMMARY_CSV, index=False)

    interpretation_lines = [
        "- locked filter improves positive-after-friction behavior: "
        + (
            "likely acceptable"
            if overall["positive_after_friction_rate"] >= 0.5
            else "likely fragile"
        ),
        "- OOS remains acceptable: "
        + (
            "likely acceptable"
            if pd.notna(is_oos["OOS_positive_after_friction_rate"])
            and is_oos["OOS_positive_after_friction_rate"] >= 0.45
            else "potentially fragile"
        ),
        "- yearly stability remains acceptable: "
        + (
            "mixed but acceptable"
            if not yearly.empty and yearly["year_positive_after_friction_rate"].min() >= 0.4
            else "fragile"
        ),
        "- sequence stress remains too fragile: "
        + (
            "yes"
            if (
                pd.notna(stress["max_rolling_20_negative_after_friction_rate"])
                and stress["max_rolling_20_negative_after_friction_rate"] > 0.65
            )
            else "not obviously"
        ),
        "- ready only for theoretical basket simulation, not EA: yes.",
    ]

    md = "\n".join(
        [
            "# EURUSD/GBPUSD Locked Filter Micro Check",
            "",
            "**Warning:** descriptive locked-filter diagnostics only, not trading guidance.",
            "",
            "## Locked Filter Definition",
            "- first_touch_0_5_vs_3 == `normalized_0_5_first`",
            "- hour == 15",
            "- current_abs_z >= 1.8",
            "- abs(current_z) < 2.0",
            "- rows require valid basket_move_pips_proxy, conservative_behavior_after_friction, and bars_to_first_touch.",
            "",
            "## Overall Summary",
            _table_md(pd.DataFrame([overall])),
            "",
            "## IS/OOS Table",
            _table_md(pd.DataFrame([is_oos])),
            "",
            "## Yearly Table",
            _table_md(yearly),
            "",
            "## z_sign Table",
            _table_md(z_sign_summary),
            "",
            "## Sequence Stress",
            _table_md(pd.DataFrame([stress])),
            "",
            "## Top Negative Clusters",
            _table_md(clusters),
            "",
            "## Final Interpretation",
            *interpretation_lines,
        ]
    )
    OUTPUT_SUMMARY_MD.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
