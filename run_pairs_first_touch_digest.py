#!/usr/bin/env python3
"""Summarize first-touch pairs behavior atlas metrics into compact CSV + Markdown digests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

INPUT_DEFAULT = "pairs_behavior_atlas_reports_quick/EURUSD_GBPUSD_pairs_grouped_behavior.csv"
OUTPUT_DIR_DEFAULT = "pairs_behavior_atlas_reports_quick/readouts"

BASE_OUTPUT_COLUMNS = [
    "section",
    "window",
    "horizon_bars",
    "grouping",
    "observations",
    "session_bucket",
    "hour",
    "correlation_regime",
    "beta_stability",
    "zscore_bucket",
    "abs_zscore_bucket",
    "normalized_0_5_first_vs_3_rate",
    "moved_further_to_3_first_vs_0_5_rate",
    "normalized_0_first_vs_3_rate",
    "moved_further_to_3_first_vs_0_rate",
    "normalized_0_5_first_vs_4_rate",
    "moved_further_to_4_first_vs_0_5_rate",
    "median_bars_to_first_touch_0_5_vs_3",
    "median_bars_to_first_touch_0_vs_3",
    "median_bars_to_first_touch_0_5_vs_4",
    "IS_normalized_0_5_first_vs_3_rate",
    "OOS_normalized_0_5_first_vs_3_rate",
    "IS_moved_further_to_3_first_vs_0_5_rate",
    "OOS_moved_further_to_3_first_vs_0_5_rate",
    "WF_normalized_0_5_first_vs_3_std",
    "WF_moved_further_to_3_first_vs_0_5_std",
    "mean_corr",
    "median_corr",
    "mean_beta",
    "median_beta",
    "mean_abs_zscore",
    "median_abs_zscore",
    "p90_abs_zscore",
    "p95_abs_zscore",
    "norm_first_gap",
    "div_first_gap",
]

OPTIONAL_COLUMNS = [
    "both_same_bar_0_5_vs_3_rate",
    "neither_0_5_vs_3_rate",
    "both_same_bar_0_vs_3_rate",
    "neither_0_vs_3_rate",
    "both_same_bar_0_5_vs_4_rate",
    "neither_0_5_vs_4_rate",
]

SECTION_SPECS = [
    {"name": "highest_normalized_0_5_before_abs_z_3", "sort_metric": "normalized_0_5_first_vs_3_rate"},
    {
        "name": "highest_moved_further_to_abs_z_3_before_normalized_0_5",
        "sort_metric": "moved_further_to_3_first_vs_0_5_rate",
    },
    {"name": "highest_normalized_0_before_abs_z_3", "sort_metric": "normalized_0_first_vs_3_rate"},
    {
        "name": "highest_moved_further_to_abs_z_3_before_normalized_0",
        "sort_metric": "moved_further_to_3_first_vs_0_rate",
    },
    {"name": "highest_normalized_0_5_before_abs_z_4", "sort_metric": "normalized_0_5_first_vs_4_rate"},
    {
        "name": "highest_moved_further_to_abs_z_4_before_normalized_0_5",
        "sort_metric": "moved_further_to_4_first_vs_0_5_rate",
    },
    {
        "name": "stable_normalization_first_contexts",
        "sort_metric": "normalized_0_5_first_vs_3_rate",
        "stable_gap": "norm_first_gap",
    },
    {
        "name": "stable_divergence_first_contexts",
        "sort_metric": "moved_further_to_3_first_vs_0_5_rate",
        "stable_gap": "div_first_gap",
    },
    {
        "name": "window_comparison",
        "groupby": ["window"],
        "sort_metric": "normalized_0_5_first_vs_3_rate",
    },
    {
        "name": "session_comparison",
        "groupby": ["session_bucket"],
        "sort_metric": "normalized_0_5_first_vs_3_rate",
    },
    {
        "name": "abs_zscore_bucket_comparison",
        "groupby": ["abs_zscore_bucket"],
        "sort_metric": "normalized_0_5_first_vs_3_rate",
    },
    {
        "name": "correlation_regime_abs_zscore_bucket_comparison",
        "groupby": ["correlation_regime", "abs_zscore_bucket"],
        "sort_metric": "normalized_0_5_first_vs_3_rate",
    },
]

REQUIRED_COLUMNS = [
    "window",
    "horizon_bars",
    "grouping",
    "observations",
    "normalized_0_5_first_vs_3_rate",
    "moved_further_to_3_first_vs_0_5_rate",
    "normalized_0_first_vs_3_rate",
    "moved_further_to_3_first_vs_0_rate",
    "normalized_0_5_first_vs_4_rate",
    "moved_further_to_4_first_vs_0_5_rate",
]

METRIC_COLUMNS = [
    "normalized_0_5_first_vs_3_rate",
    "moved_further_to_3_first_vs_0_5_rate",
    "normalized_0_first_vs_3_rate",
    "moved_further_to_3_first_vs_0_rate",
    "normalized_0_5_first_vs_4_rate",
    "moved_further_to_4_first_vs_0_5_rate",
]


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_text}")
    return list(required)


def _build_grouped_section(df: pd.DataFrame, section: dict) -> pd.DataFrame:
    group_cols = section["groupby"]
    numeric_cols = [c for c in BASE_OUTPUT_COLUMNS + OPTIONAL_COLUMNS if c in df.columns and c != "section"]
    grouped = df.groupby(group_cols, dropna=False)
    out = grouped[numeric_cols].mean(numeric_only=True).reset_index()
    for c in group_cols:
        if c not in out.columns:
            out[c] = pd.NA
    out["grouping"] = " + ".join(group_cols)
    return out


def build_digest(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    warnings: dict[str, str] = {}
    frames: list[pd.DataFrame] = []

    has_norm_stability = {"IS_normalized_0_5_first_vs_3_rate", "OOS_normalized_0_5_first_vs_3_rate"}.issubset(df.columns)
    has_div_stability = {"IS_moved_further_to_3_first_vs_0_5_rate", "OOS_moved_further_to_3_first_vs_0_5_rate"}.issubset(df.columns)

    if has_norm_stability:
        df["norm_first_gap"] = (df["IS_normalized_0_5_first_vs_3_rate"] - df["OOS_normalized_0_5_first_vs_3_rate"]).abs()
    else:
        df["norm_first_gap"] = pd.NA

    if has_div_stability:
        df["div_first_gap"] = (
            df["IS_moved_further_to_3_first_vs_0_5_rate"] - df["OOS_moved_further_to_3_first_vs_0_5_rate"]
        ).abs()
    else:
        df["div_first_gap"] = pd.NA

    for section in SECTION_SPECS:
        sort_metric = section["sort_metric"]
        part = _build_grouped_section(df, section) if "groupby" in section else df.copy()

        stability_gap_col = section.get("stable_gap")
        if stability_gap_col:
            if part[stability_gap_col].notna().any():
                part = part.sort_values(by=[stability_gap_col, sort_metric], ascending=[True, False])
            else:
                warnings[section["name"]] = (
                    f"Warning: stability gap column {stability_gap_col} is unavailable or all NaN; "
                    "stability filter skipped."
                )

        part = part[part[sort_metric].notna()].sort_values(by=sort_metric, ascending=False).head(10).copy()
        part.insert(0, "section", section["name"])
        frames.append(part)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BASE_OUTPUT_COLUMNS)
    output_cols = BASE_OUTPUT_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in out.columns]
    for c in output_cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[output_cols], warnings


def build_markdown(
    digest_df: pd.DataFrame,
    md_path: Path,
    source_path: Path,
    source_row_count: int,
    filtered_row_count: int,
    found_required_columns: list[str],
    section_warnings: dict[str, str],
) -> None:
    lines = [
        "# First-Touch Digest",
        "",
        f"- Source file path: `{source_path}`",
        f"- Source row count: {source_row_count}",
        f"- Filtered row count (observations >= threshold): {filtered_row_count}",
        f"- Required columns found: {', '.join(found_required_columns)}",
        "",
    ]

    for section in SECTION_SPECS:
        name = section["name"]
        rows = digest_df[digest_df["section"] == name].head(10)
        lines.append(f"## {name}")
        if name in section_warnings:
            lines.append(section_warnings[name])
            lines.append("")
        if rows.empty:
            lines.append("No rows for this section.")
            lines.append("")
            continue
        table = rows.to_markdown(index=False)
        lines.append(table)
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create first-touch digest files from grouped pairs behavior CSV.")
    parser.add_argument("--input", default=INPUT_DEFAULT, help="Input grouped behavior CSV path")
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT, help="Output directory for digest files")
    parser.add_argument("--min-observations", type=int, default=200, help="Minimum observations filter")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    found_required_columns = _require_columns(df, REQUIRED_COLUMNS)

    for metric in METRIC_COLUMNS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df["observations"] = pd.to_numeric(df["observations"], errors="coerce")

    source_row_count = len(df)
    filtered_df = df[df["observations"] >= args.min_observations].copy()
    filtered_row_count = len(filtered_df)

    digest_df, section_warnings = build_digest(filtered_df)

    csv_path = out_dir / "EURUSD_GBPUSD_first_touch_digest.csv"
    md_path = out_dir / "EURUSD_GBPUSD_first_touch_digest.md"

    digest_df.to_csv(csv_path, index=False)
    build_markdown(
        digest_df,
        md_path,
        in_path,
        source_row_count,
        filtered_row_count,
        found_required_columns,
        section_warnings,
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
