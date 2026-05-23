#!/usr/bin/env python3
"""Descriptive event-time filter diagnostics for EURUSD/GBPUSD signed-relative events."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
import argparse

import numpy as np
import pandas as pd

INPUT_PATH = Path("signed_relative_micro_reports/EURUSD_GBPUSD_signed_relative_events.csv")
OUTPUT_DIR = Path("ex_ante_filter_reports")
OUTPUT_CSV = OUTPUT_DIR / "EURUSD_GBPUSD_ex_ante_filter_diagnostics.csv"
OUTPUT_MD = OUTPUT_DIR / "EURUSD_GBPUSD_ex_ante_filter_diagnostics.md"

REQUIRED_COLUMNS = [
    "datetime",
    "year",
    "current_z",
    "first_touch_0_5_vs_3",
    "basket_move_pips_proxy",
    "conservative_behavior_after_friction",
    "bars_to_first_touch",
]


SubsetRule = tuple[str, Callable[[pd.DataFrame], pd.Series]]


SUBSET_RULES: list[SubsetRule] = [
    ("all_normalized_first", lambda d: pd.Series(True, index=d.index)),
    ("positive_z_only", lambda d: d["z_sign"] == "positive_z"),
    ("negative_z_only", lambda d: d["z_sign"] == "negative_z"),
    ("current_abs_z_1_6_plus", lambda d: d["current_abs_z"] >= 1.6),
    ("current_abs_z_1_8_plus", lambda d: d["current_abs_z"] >= 1.8),
    (
        "positive_z_and_abs_z_1_6_plus",
        lambda d: (d["z_sign"] == "positive_z") & (d["current_abs_z"] >= 1.6),
    ),
    (
        "positive_z_and_abs_z_1_8_plus",
        lambda d: (d["z_sign"] == "positive_z") & (d["current_abs_z"] >= 1.8),
    ),
    (
        "negative_z_and_abs_z_1_6_plus",
        lambda d: (d["z_sign"] == "negative_z") & (d["current_abs_z"] >= 1.6),
    ),
    (
        "negative_z_and_abs_z_1_8_plus",
        lambda d: (d["z_sign"] == "negative_z") & (d["current_abs_z"] >= 1.8),
    ),
]


def _safe_rate(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.mean())


def _rolling_negative_rate(values: pd.Series, window: int) -> float:
    if len(values) < window:
        return float("nan")
    return float(values.rolling(window=window).mean().max())


def _build_event_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    out["current_abs_z"] = out["current_z"].abs()
    out["z_sign"] = np.where(out["current_z"] > 0, "positive_z", "negative_z")

    bins = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    labels = ["1.0_1.2", "1.2_1.4", "1.4_1.6", "1.6_1.8", "1.8_2.0"]
    out["current_abs_z_subbucket"] = pd.cut(
        out["current_abs_z"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    ).astype("string")
    out["current_abs_z_subbucket"] = out["current_abs_z_subbucket"].fillna("other")

    out["after_friction_label"] = np.where(
        out["conservative_behavior_after_friction"] > 0,
        "positive_after_friction",
        "negative_after_friction",
    )
    out["after_friction_positive_flag"] = (
        out["after_friction_label"] == "positive_after_friction"
    ).astype(int)
    out["after_friction_negative_flag"] = (
        out["after_friction_label"] == "negative_after_friction"
    ).astype(int)

    return out


def _subset_metrics(subset_name: str, sdf: pd.DataFrame) -> dict:
    sdf = sdf.sort_values("datetime").reset_index(drop=True)
    n = len(sdf)

    split_idx = int(np.floor(n * 0.7))
    is_df = sdf.iloc[:split_idx]
    oos_df = sdf.iloc[split_idx:]

    neg_seq = sdf["after_friction_negative_flag"].to_numpy(dtype=int)
    longest_neg = 0
    cur = 0
    for v in neg_seq:
        if v == 1:
            cur += 1
            longest_neg = max(longest_neg, cur)
        else:
            cur = 0

    return {
        "subset_name": subset_name,
        "observations": n,
        "positive_after_friction_rate": _safe_rate(sdf["after_friction_positive_flag"]),
        "negative_after_friction_rate": _safe_rate(sdf["after_friction_negative_flag"]),
        "median_conservative_behavior_after_friction": sdf[
            "conservative_behavior_after_friction"
        ].median(),
        "p25_conservative_behavior_after_friction": sdf[
            "conservative_behavior_after_friction"
        ].quantile(0.25),
        "p75_conservative_behavior_after_friction": sdf[
            "conservative_behavior_after_friction"
        ].quantile(0.75),
        "median_basket_move_pips_proxy": sdf["basket_move_pips_proxy"].median(),
        "median_current_abs_z": sdf["current_abs_z"].median(),
        "median_bars_to_first_touch": sdf["bars_to_first_touch"].median(),
        "IS_observations": len(is_df),
        "OOS_observations": len(oos_df),
        "IS_positive_after_friction_rate": _safe_rate(is_df["after_friction_positive_flag"]),
        "OOS_positive_after_friction_rate": _safe_rate(oos_df["after_friction_positive_flag"]),
        "IS_median_conservative_behavior_after_friction": is_df[
            "conservative_behavior_after_friction"
        ].median(),
        "OOS_median_conservative_behavior_after_friction": oos_df[
            "conservative_behavior_after_friction"
        ].median(),
        "longest_consecutive_negative_after_friction": longest_neg,
        "max_rolling_20_negative_after_friction_rate": _rolling_negative_rate(
            sdf["after_friction_negative_flag"], 20
        ),
        "max_rolling_50_negative_after_friction_rate": _rolling_negative_rate(
            sdf["after_friction_negative_flag"], 50
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_csv = output_dir / OUTPUT_CSV.name
    output_md = output_dir / OUTPUT_MD.name

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    normalized_df = df[df["first_touch_0_5_vs_3"] == "normalized_0_5_first"].copy()
    prepared = _build_event_time_fields(normalized_df)

    subset_rows: list[dict] = []
    yearly_rows: list[dict] = []

    for subset_name, rule in SUBSET_RULES:
        sdf = prepared[rule(prepared)].copy().sort_values("datetime")
        subset_rows.append(_subset_metrics(subset_name, sdf))

        for year, ydf in sdf.groupby("year", dropna=False):
            yearly_rows.append(
                {
                    "subset_name": subset_name,
                    "year": year,
                    "year_observations": len(ydf),
                    "year_positive_after_friction_rate": _safe_rate(
                        ydf["after_friction_positive_flag"]
                    ),
                    "year_median_conservative_behavior_after_friction": ydf[
                        "conservative_behavior_after_friction"
                    ].median(),
                }
            )

    diagnostics_df = pd.DataFrame(subset_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_df.to_csv(output_csv, index=False)

    yearly_df = pd.DataFrame(yearly_rows).sort_values(["subset_name", "year"])

    overall_cols = [
        "subset_name",
        "observations",
        "positive_after_friction_rate",
        "negative_after_friction_rate",
        "median_conservative_behavior_after_friction",
        "p25_conservative_behavior_after_friction",
        "p75_conservative_behavior_after_friction",
        "median_basket_move_pips_proxy",
        "median_current_abs_z",
        "median_bars_to_first_touch",
    ]

    isoos_cols = [
        "subset_name",
        "IS_observations",
        "OOS_observations",
        "IS_positive_after_friction_rate",
        "OOS_positive_after_friction_rate",
        "IS_median_conservative_behavior_after_friction",
        "OOS_median_conservative_behavior_after_friction",
    ]

    stress_cols = [
        "subset_name",
        "longest_consecutive_negative_after_friction",
        "max_rolling_20_negative_after_friction_rate",
        "max_rolling_50_negative_after_friction_rate",
    ]

    lines: list[str] = []
    lines.append("# EURUSD/GBPUSD Ex-Ante Filter Diagnostics")
    lines.append("")
    lines.append(
        "**Warning:** Descriptive event-time filter diagnostics only, not trading guidance."
    )
    lines.append("")
    lines.append(
        "This diagnostic uses known-at-event features only. `movement_bucket` is intentionally excluded because it is future-known (available only after first-touch)."
    )
    lines.append("")
    lines.append("## Overall subset comparison")
    lines.append(diagnostics_df[overall_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## IS/OOS subset comparison (chronological 70/30)")
    lines.append(diagnostics_df[isoos_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Yearly subset comparison")
    lines.append(yearly_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Sequence stress")
    lines.append(diagnostics_df[stress_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Final interpretation")
    lines.append(
        "- Evaluate whether raising the `current_abs_z` threshold is associated with improved positive-after-friction behavior and reduced fragility."
    )
    lines.append(
        "- Compare `positive_z` vs `negative_z` diagnostic subsets to assess whether one side is materially more stable after friction."
    )
    lines.append(
        "- Check whether any subset preserves positive-after-friction and median conservative behavior in OOS, indicating stability."
    )
    lines.append(
        "- If OOS deterioration and sequence stress remain elevated, interpret this as persistent fragility despite event-time filtering."
    )

    output_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {output_csv}")
    print(f"Wrote: {output_md}")


if __name__ == "__main__":
    main()
