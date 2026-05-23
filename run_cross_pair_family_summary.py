#!/usr/bin/env python3
"""Cross-pair first-touch family behavior summarization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_INPUTS = [
    "pairs_behavior_atlas_reports_quick/readouts/EURUSD_GBPUSD_first_touch_families.csv",
    "pairs_behavior_atlas_reports_GBPAUD_GBPNZD_quick/readouts/GBPAUD_GBPNZD_first_touch_families.csv",
]
DEFAULT_PAIR_NAMES = ["EURUSD_GBPUSD", "GBPAUD_GBPNZD"]

REQUIRED_COLUMNS = ["family_name", "rows_in_family"]
OPTIONAL_COLUMNS = [
    "best_windows",
    "best_horizons",
    "total_observations_sum",
    "max_observations",
    "avg_normalized_0_5_first_vs_3_rate",
    "max_normalized_0_5_first_vs_3_rate",
    "avg_moved_further_to_3_first_vs_0_5_rate",
    "max_moved_further_to_3_first_vs_0_5_rate",
    "avg_normalized_0_first_vs_3_rate",
    "avg_moved_further_to_4_first_vs_0_5_rate",
    "avg_median_bars_to_first_touch_0_5_vs_3",
    "top_context_1",
    "top_context_2",
    "top_context_3",
    "notes",
]


def parse_csv_list(raw: str | None, fallback: list[str]) -> list[str]:
    if not raw:
        return fallback
    return [item.strip() for item in raw.split(",") if item.strip()]


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def first_non_empty(values: Iterable[object]) -> str:
    seen: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            seen.append(text)
    if not seen:
        return ""
    unique = []
    for item in seen:
        if item not in unique:
            unique.append(item)
    return " | ".join(unique)


def common_behavior_label(family_name: str) -> str:
    if "Stable normalization-first" in family_name:
        return "stable_shared_normalization_first_behavior"
    if "Stable divergence-first" in family_name:
        return "stable_shared_divergence_first_behavior"
    if "Strong normalization-to-zero" in family_name:
        return "shared_normalization_to_zero_behavior"
    if "Extreme divergence-first" in family_name:
        return "shared_extreme_divergence_first_behavior"
    if "Severe divergence-first" in family_name:
        return "shared_severe_divergence_first_behavior"
    if "Normalization-first" in family_name:
        return "shared_normalization_first_behavior"
    return "descriptive_family_observed"


def build_confidence_note(confirmed_count: int, total_pairs: int, family_name: str) -> str:
    notes = []
    if confirmed_count == total_pairs:
        notes.append("Confirmed across all tested pairs.")
    elif confirmed_count == 1:
        notes.append("Observed in one tested pair only.")

    if confirmed_count == total_pairs and "stable" in family_name.lower():
        notes.append("Stable behavior family observed across tested pairs; still descriptive only.")

    notes.append("This does not imply a trading decision.")
    return " ".join(notes)


def summarize_family(group: pd.DataFrame, all_pairs: list[str]) -> dict[str, object]:
    family_name = str(group["family_name"].iloc[0])
    present_pairs = sorted(group["pair_name"].dropna().unique().tolist())
    missing_pairs = [pair for pair in all_pairs if pair not in present_pairs]

    rows_numeric = to_float(group["rows_in_family"]) if "rows_in_family" in group.columns else pd.Series(dtype=float)

    summary = {
        "family_name": family_name,
        "pairs_confirmed_count": len(present_pairs),
        "pairs_confirmed": ", ".join(present_pairs),
        "pairs_missing": ", ".join(missing_pairs),
        "total_rows_in_family": float(rows_numeric.sum()) if not rows_numeric.empty else "",
        "avg_rows_in_family": float(rows_numeric.mean()) if not rows_numeric.empty else "",
        "max_rows_in_family": float(rows_numeric.max()) if not rows_numeric.empty else "",
        "common_best_windows": first_non_empty(group.get("best_windows", pd.Series(dtype=object))),
        "common_best_horizons": first_non_empty(group.get("best_horizons", pd.Series(dtype=object))),
        "avg_normalization_first_rate_across_pairs": "",
        "avg_divergence_first_rate_across_pairs": "",
        "avg_normalization_to_zero_rate_across_pairs": "",
        "avg_divergence_to_abs_z_4_rate_across_pairs": "",
        "common_behavior_label": common_behavior_label(family_name),
        "pair_specific_contexts": "",
        "confidence_notes": build_confidence_note(len(present_pairs), len(all_pairs), family_name),
    }

    if "avg_normalized_0_5_first_vs_3_rate" in group.columns:
        summary["avg_normalization_first_rate_across_pairs"] = float(
            to_float(group["avg_normalized_0_5_first_vs_3_rate"]).mean()
        )
    if "avg_moved_further_to_3_first_vs_0_5_rate" in group.columns:
        summary["avg_divergence_first_rate_across_pairs"] = float(
            to_float(group["avg_moved_further_to_3_first_vs_0_5_rate"]).mean()
        )
    if "avg_normalized_0_first_vs_3_rate" in group.columns:
        summary["avg_normalization_to_zero_rate_across_pairs"] = float(
            to_float(group["avg_normalized_0_first_vs_3_rate"]).mean()
        )
    if "avg_moved_further_to_4_first_vs_0_5_rate" in group.columns:
        summary["avg_divergence_to_abs_z_4_rate_across_pairs"] = float(
            to_float(group["avg_moved_further_to_4_first_vs_0_5_rate"]).mean()
        )

    context_lines: list[str] = []
    for pair in all_pairs:
        pair_rows = group[group["pair_name"] == pair]
        if pair_rows.empty:
            context_lines.append(f"{pair}: (not present)")
            continue
        context_1 = first_non_empty(pair_rows.get("top_context_1", pd.Series(dtype=object)))
        context_2 = first_non_empty(pair_rows.get("top_context_2", pd.Series(dtype=object)))
        context_3 = first_non_empty(pair_rows.get("top_context_3", pd.Series(dtype=object)))
        context_lines.append(f"{pair}: {context_1} | {context_2} | {context_3}".strip())

    summary["pair_specific_contexts"] = " || ".join(context_lines)
    return summary


def build_markdown(summary_df: pd.DataFrame, source_files: list[str], all_pairs: list[str]) -> str:
    all_pair_families = summary_df[summary_df["pairs_confirmed_count"] == len(all_pairs)]["family_name"].tolist()
    differing = summary_df[summary_df["pairs_confirmed_count"] < len(all_pairs)]["family_name"].tolist()

    lines = [
        "# Cross-Pair First-Touch Family Behavior Summary",
        "",
        "## Source files",
    ]
    lines.extend([f"- `{src}`" for src in source_files])
    lines.extend(
        [
            "",
            "Descriptive cross-pair relative behavior only, not a strategy.",
            "",
            "## Executive summary",
            f"- Families appearing in all tested pairs: {', '.join(all_pair_families) if all_pair_families else 'None'}",
            f"- Families differing by pair/session/hour: {', '.join(differing) if differing else 'None'}",
            "- Key repeated pattern: intermediate relative z-score regimes tend toward normalization-first, while abs_zscore_bucket=3_plus tends toward divergence-first.",
            "",
            "## Family overview table",
            "| family_name | pairs_confirmed | common_behavior_label | confidence_notes |",
            "|---|---|---|---|",
        ]
    )

    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['family_name']} | {row['pairs_confirmed']} | {row['common_behavior_label']} | {row['confidence_notes']} |"
        )

    lines.append("")
    lines.append("## Detailed family contexts")
    for _, row in summary_df.iterrows():
        lines.extend(
            [
                "",
                f"### {row['family_name']}",
                f"- Pairs confirmed: {row['pairs_confirmed']}",
                f"- Pairs missing: {row['pairs_missing']}",
                f"- Pair-specific contexts: {row['pair_specific_contexts']}",
            ]
        )

    lines.extend(
        [
            "",
            "## Final research notes",
            "- This is still behavior research only.",
            "- Costs/spreads/slippage are not included.",
            "- Next stage can be either more pairs or execution reality layer.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize cross-pair first-touch family behavior")
    parser.add_argument("--inputs", default=",".join(DEFAULT_INPUTS), help="Comma-separated input CSV file paths")
    parser.add_argument("--pair-names", default=",".join(DEFAULT_PAIR_NAMES), help="Comma-separated pair names")
    parser.add_argument("--output-dir", default="cross_pair_behavior_summary", help="Output directory")
    args = parser.parse_args()

    input_files = parse_csv_list(args.inputs, DEFAULT_INPUTS)
    pair_names = parse_csv_list(args.pair_names, DEFAULT_PAIR_NAMES)

    if len(input_files) != len(pair_names):
        raise ValueError("Number of input files must match number of pair names.")

    frames = []
    for csv_path, pair_name in zip(input_files, pair_names):
        frame = pd.read_csv(csv_path)
        missing_required = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
        if missing_required:
            raise ValueError(f"Missing required columns in {csv_path}: {missing_required}")

        frame = frame.copy()
        for col in OPTIONAL_COLUMNS:
            if col not in frame.columns:
                frame[col] = ""
        frame["pair_name"] = pair_name
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    summary_rows = [summarize_family(group, pair_names) for _, group in combined.groupby("family_name", sort=True)]
    summary_df = pd.DataFrame(summary_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_output = output_dir / "cross_pair_first_touch_family_summary.csv"
    md_output = output_dir / "cross_pair_first_touch_family_summary.md"

    summary_df.to_csv(csv_output, index=False)
    md_output.write_text(build_markdown(summary_df, input_files, pair_names), encoding="utf-8")

    print(f"Wrote: {csv_output}")
    print(f"Wrote: {md_output}")


if __name__ == "__main__":
    main()
