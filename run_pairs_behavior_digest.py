#!/usr/bin/env python3
"""Summarize grouped pairs behavior atlas output into compact CSV + Markdown digests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

OUTPUT_DIR_DEFAULT = "pairs_behavior_atlas_reports_quick/readouts"

OUTPUT_COLUMNS = [
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
    "normalized_to_1_rate",
    "normalized_to_0_5_rate",
    "normalized_to_0_rate",
    "moved_further_to_abs_z_3_rate",
    "moved_further_to_abs_z_4_rate",
    "median_bars_to_normalized_1",
    "median_bars_to_normalized_0_5",
    "median_bars_to_normalized_0",
    "mean_corr",
    "median_corr",
    "mean_beta",
    "median_beta",
    "mean_abs_zscore",
    "median_abs_zscore",
    "p90_abs_zscore",
    "p95_abs_zscore",
    "IS_normalized_to_0_5_rate",
    "OOS_normalized_to_0_5_rate",
    "IS_moved_further_to_abs_z_3_rate",
    "OOS_moved_further_to_abs_z_3_rate",
    "WF_normalized_to_0_5_std",
    "WF_moved_further_to_abs_z_3_std",
]

DIGEST_SECTIONS = [
    ("Highest normalization to 0.5 rate", "normalized_to_0_5_rate", False),
    ("Highest normalization to 0 rate", "normalized_to_0_rate", False),
    ("Highest moved further to abs z 3 rate", "moved_further_to_abs_z_3_rate", False),
    ("Highest moved further to abs z 4 rate", "moved_further_to_abs_z_4_rate", False),
    ("Strongest high-correlation normalization contexts", "normalized_to_0_5_rate", False),
    ("Weak/broken-correlation behavior", "moved_further_to_abs_z_3_rate", False),
    ("Session comparison", "normalized_to_0_5_rate", False),
    ("Hour comparison", "normalized_to_0_5_rate", False),
    ("Abs zscore bucket comparison", "normalized_to_0_5_rate", False),
    ("Correlation regime + abs zscore bucket comparison", "normalized_to_0_5_rate", False),
]


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")


def _format_val(v: object) -> str:
    if pd.isna(v):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_section(df: pd.DataFrame, name: str, sort_col: str, ascending: bool) -> pd.DataFrame:
    section_df = df.copy()

    if name == "Strongest high-correlation normalization contexts":
        section_df = section_df[section_df["correlation_regime"].astype(str).str.lower().str.contains("high", na=False)]
    elif name == "Weak/broken-correlation behavior":
        section_df = section_df[
            section_df["correlation_regime"].astype(str).str.lower().str.contains("weak|broken|low", regex=True, na=False)
        ]
    elif name == "Session comparison":
        section_df = section_df[section_df["grouping"].astype(str).str.contains("session", case=False, na=False)]
    elif name == "Hour comparison":
        section_df = section_df[section_df["grouping"].astype(str).str.contains("hour", case=False, na=False)]
    elif name == "Abs zscore bucket comparison":
        section_df = section_df[section_df["grouping"].astype(str).str.contains("abs_zscore", case=False, na=False)]
    elif name == "Correlation regime + abs zscore bucket comparison":
        g = section_df["grouping"].astype(str)
        section_df = section_df[g.str.contains("correlation", case=False, na=False) & g.str.contains("abs_zscore", case=False, na=False)]

    if section_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    section_df = section_df.sort_values(by=[sort_col, "observations"], ascending=[ascending, False]).head(20).copy()
    section_df.insert(0, "section", name)

    for c in OUTPUT_COLUMNS:
        if c not in section_df.columns:
            section_df[c] = pd.NA

    return section_df[OUTPUT_COLUMNS]


def build_markdown(df: pd.DataFrame, md_path: Path, source_name: str) -> None:
    lines = [
        f"# Pairs Behavior Digest: {source_name}",
        "",
        "Descriptive relative behavior only based on relative spread and z-score normalization context; not a strategy.",
        "",
    ]

    for section in DIGEST_SECTIONS:
        section_name = section[0]
        rows = df[df["section"] == section_name].head(10)
        lines.append(f"## {section_name}")
        if rows.empty:
            lines.append("No rows met the minimum-observation and grouping/context filters.")
            lines.append("")
            continue

        for _, row in rows.iterrows():
            context = (
                f"window={_format_val(row['window'])}, horizon={_format_val(row['horizon_bars'])}, "
                f"grouping={_format_val(row['grouping'])}, obs={_format_val(row['observations'])}, "
                f"session={_format_val(row['session_bucket'])}, hour={_format_val(row['hour'])}, "
                f"corr_regime={_format_val(row['correlation_regime'])}, beta_stability={_format_val(row['beta_stability'])}, "
                f"z_bucket={_format_val(row['zscore_bucket'])}, abs_z_bucket={_format_val(row['abs_zscore_bucket'])}, "
                f"norm0.5={_format_val(row['normalized_to_0_5_rate'])}, norm0={_format_val(row['normalized_to_0_rate'])}, "
                f"to|z|3={_format_val(row['moved_further_to_abs_z_3_rate'])}, to|z|4={_format_val(row['moved_further_to_abs_z_4_rate'])}, "
                f"corr_med={_format_val(row['median_corr'])}, beta_med={_format_val(row['median_beta'])}, "
                f"abs_z_p95={_format_val(row['p95_abs_zscore'])}"
            )
            lines.append(f"- {context}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create behavior digest files from grouped pairs behavior CSV.")
    parser.add_argument("--symbol-a", default="EURUSD", help="First symbol in pair prefix")
    parser.add_argument("--symbol-b", default="GBPUSD", help="Second symbol in pair prefix")
    parser.add_argument("--input", default=None, help="Input grouped behavior CSV path")
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT, help="Output directory for digest files")
    parser.add_argument("--min-observations", type=int, default=200, help="Minimum observations filter")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_prefix = f"{args.symbol_a}_{args.symbol_b}"
    in_path = Path(args.input) if args.input else Path(f"pairs_behavior_atlas_reports_quick/{pair_prefix}_pairs_grouped_behavior.csv")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    _require_columns(df, ["observations", "grouping", "correlation_regime"])

    df = df[df["observations"] >= args.min_observations].copy()

    all_sections = [build_section(df, name, sort_col, ascending) for name, sort_col, ascending in DIGEST_SECTIONS]
    digest_df = pd.concat(all_sections, ignore_index=True) if all_sections else pd.DataFrame(columns=OUTPUT_COLUMNS)

    csv_path = out_dir / f"{pair_prefix}_pairs_digest.csv"
    md_path = out_dir / f"{pair_prefix}_pairs_digest.md"

    digest_df.to_csv(csv_path, index=False)
    build_markdown(digest_df, md_path, in_path.stem)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
