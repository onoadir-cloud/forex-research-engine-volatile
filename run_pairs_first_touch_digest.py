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
    {
        "name": "1. Highest normalized 0.5 before abs z 3",
        "sort": ["normalized_0_5_first_vs_3_rate", "observations"],
        "ascending": [False, False],
    },
    {
        "name": "2. Highest moved further to abs z 3 before normalized 0.5",
        "sort": ["moved_further_to_3_first_vs_0_5_rate", "observations"],
        "ascending": [False, False],
    },
    {
        "name": "3. Highest normalized 0 before abs z 3",
        "sort": ["normalized_0_first_vs_3_rate", "observations"],
        "ascending": [False, False],
    },
    {
        "name": "4. Highest moved further to abs z 3 before normalized 0",
        "sort": ["moved_further_to_3_first_vs_0_rate", "observations"],
        "ascending": [False, False],
    },
    {
        "name": "5. Highest normalized 0.5 before abs z 4",
        "sort": ["normalized_0_5_first_vs_4_rate", "observations"],
        "ascending": [False, False],
    },
    {
        "name": "6. Highest moved further to abs z 4 before normalized 0.5",
        "sort": ["moved_further_to_4_first_vs_0_5_rate", "observations"],
        "ascending": [False, False],
    },
    {
        "name": "7. Stable normalization-first contexts",
        "sort": ["normalized_0_5_first_vs_3_rate", "observations"],
        "ascending": [False, False],
        "stable_col_a": "IS_normalized_0_5_first_vs_3_rate",
        "stable_col_b": "OOS_normalized_0_5_first_vs_3_rate",
    },
    {
        "name": "8. Stable divergence-first contexts",
        "sort": ["moved_further_to_3_first_vs_0_5_rate", "observations"],
        "ascending": [False, False],
        "stable_col_a": "IS_moved_further_to_3_first_vs_0_5_rate",
        "stable_col_b": "OOS_moved_further_to_3_first_vs_0_5_rate",
    },
    {
        "name": "9. Window comparison",
        "groupby": ["window", "horizon_bars"],
        "sort": ["normalized_0_5_first_vs_3_rate", "observations"],
        "ascending": [False, False],
    },
    {
        "name": "10. Session comparison",
        "groupby": ["window", "horizon_bars", "session_bucket"],
        "sort": ["normalized_0_5_first_vs_3_rate", "observations"],
        "ascending": [False, False],
    },
    {
        "name": "11. Abs zscore bucket comparison",
        "groupby": ["window", "horizon_bars", "abs_zscore_bucket"],
        "sort": ["normalized_0_5_first_vs_3_rate", "observations"],
        "ascending": [False, False],
    },
    {
        "name": "12. Correlation regime + abs zscore bucket comparison",
        "groupby": ["window", "horizon_bars", "correlation_regime", "abs_zscore_bucket"],
        "sort": ["normalized_0_5_first_vs_3_rate", "observations"],
        "ascending": [False, False],
    },
]


def _format_val(v: object, digits: int = 4) -> str:
    if pd.isna(v):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")


def _avg_existing(grouped: pd.core.groupby.generic.DataFrameGroupBy, cols: list[str]) -> pd.DataFrame:
    use_cols = [c for c in cols if c in grouped.obj.columns]
    if not use_cols:
        return pd.DataFrame(index=grouped.size().index)
    return grouped[use_cols].mean(numeric_only=True)


def _build_grouped_section(df: pd.DataFrame, section: dict) -> pd.DataFrame:
    group_cols = section["groupby"]
    grouped = df.groupby(group_cols, dropna=False)
    rows = grouped.size().rename("observations").to_frame()

    avg_cols = [c for c in BASE_OUTPUT_COLUMNS + OPTIONAL_COLUMNS if c != "section" and c != "observations"]
    avg_df = _avg_existing(grouped, avg_cols)
    out = rows.join(avg_df, how="left").reset_index()
    out["grouping"] = " + ".join(group_cols)

    for c in ["session_bucket", "hour", "correlation_regime", "beta_stability", "zscore_bucket", "abs_zscore_bucket"]:
        if c not in out.columns:
            out[c] = pd.NA

    out.insert(0, "section", section["name"])
    out = out.sort_values(by=section["sort"], ascending=section["ascending"]).head(10)
    return out


def _build_top_section(df: pd.DataFrame, section: dict) -> pd.DataFrame:
    out = df.copy()
    col_a = section.get("stable_col_a")
    col_b = section.get("stable_col_b")
    if col_a and col_b:
        _require_columns(out, [col_a, col_b])
        out = out[(out[col_a] - out[col_b]).abs() <= 0.05]

    out = out.sort_values(by=section["sort"], ascending=section["ascending"]).head(10).copy()
    out.insert(0, "section", section["name"])
    return out


def build_digest(df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for section in SECTION_SPECS:
        part = _build_grouped_section(df, section) if "groupby" in section else _build_top_section(df, section)
        frames.append(part)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    output_cols = BASE_OUTPUT_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in out.columns]
    for c in output_cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[output_cols]


def build_markdown(df: pd.DataFrame, md_path: Path, source_name: str) -> None:
    lines = [
        f"# First-Touch Digest: {source_name}",
        "",
        "Descriptive first-touch relative behavior only across relative spread and z-score context.",
        "",
    ]

    for section in SECTION_SPECS:
        name = section["name"]
        rows = df[df["section"] == name].head(10)
        lines.append(f"## {name}")
        if rows.empty:
            lines.append("No rows met filters.")
            lines.append("")
            continue

        for _, row in rows.iterrows():
            compact_context = (
                f"session={_format_val(row.get('session_bucket'))}, "
                f"hour={_format_val(row.get('hour'))}, "
                f"corr_regime={_format_val(row.get('correlation_regime'))}, "
                f"beta_stability={_format_val(row.get('beta_stability'))}, "
                f"abs_z_bucket={_format_val(row.get('abs_zscore_bucket'))}"
            )
            line = (
                f"- window={_format_val(row.get('window'))} | "
                f"horizon={_format_val(row.get('horizon_bars'))} | "
                f"grouping={_format_val(row.get('grouping'))} | "
                f"observations={_format_val(row.get('observations'))} | "
                f"context={compact_context} | "
                f"normalization-first={_format_val(row.get('normalized_0_5_first_vs_3_rate'))} | "
                f"divergence-first={_format_val(row.get('moved_further_to_3_first_vs_0_5_rate'))} | "
                f"IS/OOS norm={_format_val(row.get('IS_normalized_0_5_first_vs_3_rate'))}/"
                f"{_format_val(row.get('OOS_normalized_0_5_first_vs_3_rate'))} | "
                f"IS/OOS div={_format_val(row.get('IS_moved_further_to_3_first_vs_0_5_rate'))}/"
                f"{_format_val(row.get('OOS_moved_further_to_3_first_vs_0_5_rate'))} | "
                f"median bars 0.5vs3={_format_val(row.get('median_bars_to_first_touch_0_5_vs_3'))}, "
                f"0vs3={_format_val(row.get('median_bars_to_first_touch_0_vs_3'))}, "
                f"0.5vs4={_format_val(row.get('median_bars_to_first_touch_0_5_vs_4'))}"
            )
            lines.append(line)
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
    required = [
        "observations",
        "window",
        "horizon_bars",
        "grouping",
        "normalized_0_5_first_vs_3_rate",
        "moved_further_to_3_first_vs_0_5_rate",
        "normalized_0_first_vs_3_rate",
        "moved_further_to_3_first_vs_0_rate",
        "normalized_0_5_first_vs_4_rate",
        "moved_further_to_4_first_vs_0_5_rate",
    ]
    _require_columns(df, required)

    df = df[df["observations"] >= args.min_observations].copy()

    digest_df = build_digest(df)

    csv_path = out_dir / "EURUSD_GBPUSD_first_touch_digest.csv"
    md_path = out_dir / "EURUSD_GBPUSD_first_touch_digest.md"

    digest_df.to_csv(csv_path, index=False)
    build_markdown(digest_df, md_path, in_path.stem)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
