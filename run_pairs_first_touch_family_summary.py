#!/usr/bin/env python3
"""Summarize first-touch digest into descriptive relative-behavior families."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


FAMILIES: List[Tuple[str, str]] = [
    (
        "Normalization-first relative behavior",
        "(normalized_0_5_first_vs_3_rate >= @n0 and moved_further_to_3_first_vs_0_5_rate <= @d0 and observations >= @min_obs)",
    ),
    (
        "Strong normalization-to-zero behavior",
        "(normalized_0_first_vs_3_rate >= @n0 and observations >= @min_obs)",
    ),
    (
        "Extreme divergence-first behavior",
        "(moved_further_to_3_first_vs_0_5_rate >= @d1 and observations >= @min_obs)",
    ),
    (
        "Severe divergence-first to abs z 4",
        "(moved_further_to_4_first_vs_0_5_rate >= @d2 and observations >= @min_obs)",
    ),
    (
        "Stable normalization-first behavior",
        "(normalized_0_5_first_vs_3_rate >= @n1 and abs(IS_normalized_0_5_first_vs_3_rate - OOS_normalized_0_5_first_vs_3_rate) <= @st and observations >= @min_obs)",
    ),
    (
        "Stable divergence-first behavior",
        "(moved_further_to_3_first_vs_0_5_rate >= @d1 and abs(IS_moved_further_to_3_first_vs_0_5_rate - OOS_moved_further_to_3_first_vs_0_5_rate) <= @st and observations >= @min_obs)",
    ),
]



CONTEXT_COLUMNS = [
    "window",
    "horizon_bars",
    "grouping",
    "session_bucket",
    "hour",
    "correlation_regime",
    "beta_stability",
    "zscore_bucket",
    "abs_zscore_bucket",
]

VALID_WINDOWS = {20, 50, 100}
VALID_HORIZONS = {5, 10, 20, 40}

FAMILY_SORT_RULES: Dict[str, List[str]] = {
    "Normalization-first relative behavior": ["normalized_0_5_first_vs_3_rate", "observations"],
    "Strong normalization-to-zero behavior": ["normalized_0_first_vs_3_rate", "observations"],
    "Extreme divergence-first behavior": ["moved_further_to_3_first_vs_0_5_rate", "observations"],
    "Severe divergence-first to abs z 4": ["moved_further_to_4_first_vs_0_5_rate", "observations"],
    "Stable normalization-first behavior": [
        "normalized_0_5_first_vs_3_rate",
        "stability_gap",
        "observations",
    ],
    "Stable divergence-first behavior": [
        "moved_further_to_3_first_vs_0_5_rate",
        "stability_gap",
        "observations",
    ],
}


REQUIRED_COLUMNS = {
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
    "median_corr",
    "median_beta",
    "mean_abs_zscore",
    "p95_abs_zscore",
    "normalized_0_5_first_vs_3_rate",
    "moved_further_to_3_first_vs_0_5_rate",
    "normalized_0_first_vs_3_rate",
    "moved_further_to_4_first_vs_0_5_rate",
    "median_bars_to_first_touch_0_5_vs_3",
    "IS_normalized_0_5_first_vs_3_rate",
    "OOS_normalized_0_5_first_vs_3_rate",
    "IS_moved_further_to_3_first_vs_0_5_rate",
    "OOS_moved_further_to_3_first_vs_0_5_rate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol-a", default="EURUSD")
    parser.add_argument("--symbol-b", default="GBPUSD")
    parser.add_argument("--input", default=None)
    parser.add_argument("--fallback-input", default=None)
    parser.add_argument(
        "--output-dir",
        default="pairs_behavior_atlas_reports_quick/readouts",
    )
    parser.add_argument("--min-observations", type=int, default=200)
    return parser.parse_args()


def load_input(primary: Path, fallback: Path) -> Tuple[pd.DataFrame, Path]:
    if primary.exists():
        return pd.read_csv(primary), primary
    if fallback.exists():
        return pd.read_csv(fallback), fallback
    raise FileNotFoundError(
        f"Neither input exists: {primary} nor fallback: {fallback}"
    )


def ensure_columns(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def context_string(row: pd.Series) -> str:
    return (
        f"window={row['window']}, horizon={row['horizon_bars']}, grouping={row['grouping']}, "
        f"session={row['session_bucket']}, hour={row['hour']}, corr_regime={row['correlation_regime']}, "
        f"beta_stability={row['beta_stability']}, z_bucket={row['zscore_bucket']}, abs_z_bucket={row['abs_zscore_bucket']}, "
        f"obs={int(row['observations'])}, median_corr={row['median_corr']:.3f}, median_beta={row['median_beta']:.3f}, "
        f"mean_abs_z={row['mean_abs_zscore']:.3f}, p95_abs_z={row['p95_abs_zscore']:.3f}"
    )


def _validate_context_values(df_contexts: pd.DataFrame, family_name: str) -> None:
    non_null_windows = df_contexts["window"].dropna()
    invalid_windows = sorted({int(v) for v in non_null_windows if int(v) not in VALID_WINDOWS})
    if invalid_windows:
        raise ValueError(
            f"Invalid window values in top contexts for '{family_name}': {invalid_windows}. "
            f"Allowed: {sorted(VALID_WINDOWS)}"
        )

    non_null_horizons = df_contexts["horizon_bars"].dropna()
    invalid_horizons = sorted({int(v) for v in non_null_horizons if int(v) not in VALID_HORIZONS})
    if invalid_horizons:
        raise ValueError(
            f"Invalid horizon_bars values in top contexts for '{family_name}': {invalid_horizons}. "
            f"Allowed: {sorted(VALID_HORIZONS)}"
        )


def top_context_rows(family_name: str, family_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if family_df.empty:
        return family_df.head(0)

    unique_contexts = family_df.drop_duplicates(subset=CONTEXT_COLUMNS, keep="first").copy()

    sort_by = FAMILY_SORT_RULES[family_name]
    ascending = [False if col != "stability_gap" else True for col in sort_by]

    if "stability_gap" in sort_by:
        unique_contexts["stability_gap"] = (
            unique_contexts["IS_normalized_0_5_first_vs_3_rate"]
            - unique_contexts["OOS_normalized_0_5_first_vs_3_rate"]
        ).abs()
        if "moved_further_to_3_first_vs_0_5_rate" in sort_by:
            unique_contexts["stability_gap"] = (
                unique_contexts["IS_moved_further_to_3_first_vs_0_5_rate"]
                - unique_contexts["OOS_moved_further_to_3_first_vs_0_5_rate"]
            ).abs()

    top_rows = unique_contexts.sort_values(by=sort_by, ascending=ascending).head(n)
    _validate_context_values(top_rows, family_name)
    return top_rows


def top_contexts(family_name: str, family_df: pd.DataFrame, n: int = 5) -> List[str]:
    top_rows = top_context_rows(family_name, family_df, n=n)
    return [context_string(r) for _, r in top_rows.iterrows()]


def summarize_family(name: str, family_df: pd.DataFrame) -> Dict[str, object]:
    if family_df.empty:
        return {
            "family_name": name,
            "rows_in_family": 0,
            "best_windows": "",
            "best_horizons": "",
            "total_observations_sum": 0,
            "max_observations": 0,
            "avg_normalized_0_5_first_vs_3_rate": 0.0,
            "max_normalized_0_5_first_vs_3_rate": 0.0,
            "avg_moved_further_to_3_first_vs_0_5_rate": 0.0,
            "max_moved_further_to_3_first_vs_0_5_rate": 0.0,
            "avg_normalized_0_first_vs_3_rate": 0.0,
            "avg_moved_further_to_4_first_vs_0_5_rate": 0.0,
            "avg_median_bars_to_first_touch_0_5_vs_3": 0.0,
            "top_context_1": "",
            "top_context_2": "",
            "top_context_3": "",
            "notes": "No rows met family definition at current thresholds.",
        }

    window_counts = family_df.groupby("window")["observations"].sum().sort_values(ascending=False)
    horizon_counts = family_df.groupby("horizon_bars")["observations"].sum().sort_values(ascending=False)
    contexts = top_contexts(name, family_df, n=5)

    return {
        "family_name": name,
        "rows_in_family": int(len(family_df)),
        "best_windows": "|".join(map(str, window_counts.head(3).index.tolist())),
        "best_horizons": "|".join(map(str, horizon_counts.head(3).index.tolist())),
        "total_observations_sum": int(family_df["observations"].sum()),
        "max_observations": int(family_df["observations"].max()),
        "avg_normalized_0_5_first_vs_3_rate": float(family_df["normalized_0_5_first_vs_3_rate"].mean()),
        "max_normalized_0_5_first_vs_3_rate": float(family_df["normalized_0_5_first_vs_3_rate"].max()),
        "avg_moved_further_to_3_first_vs_0_5_rate": float(family_df["moved_further_to_3_first_vs_0_5_rate"].mean()),
        "max_moved_further_to_3_first_vs_0_5_rate": float(family_df["moved_further_to_3_first_vs_0_5_rate"].max()),
        "avg_normalized_0_first_vs_3_rate": float(family_df["normalized_0_first_vs_3_rate"].mean()),
        "avg_moved_further_to_4_first_vs_0_5_rate": float(family_df["moved_further_to_4_first_vs_0_5_rate"].mean()),
        "avg_median_bars_to_first_touch_0_5_vs_3": float(family_df["median_bars_to_first_touch_0_5_vs_3"].mean()),
        "top_context_1": contexts[0] if len(contexts) > 0 else "",
        "top_context_2": contexts[1] if len(contexts) > 1 else "",
        "top_context_3": contexts[2] if len(contexts) > 2 else "",
        "notes": "Descriptive relative behavior summary under threshold-defined family filter.",
    }


def build_markdown(source_path: Path, source_rows: int, family_frames: Dict[str, pd.DataFrame]) -> str:
    lines: List[str] = [
        "# First-Touch Relative Behavior Families",
        "",
        f"Source: `{source_path}`",
        f"Source row count: **{source_rows}**",
        "",
        "Descriptive behavior only, not a strategy.",
        "",
        "Interpretation notes:",
        "- normalization-first means relative spread reached normalization threshold before divergence threshold.",
        "- divergence-first means relative spread reached wider absolute z-score before normalization.",
        "- high first-touch rate does not imply a trading instruction.",
        "",
    ]

    for family_name, df_family in family_frames.items():
        lines.append(f"## {family_name}")
        lines.append(f"Rows in family: **{len(df_family)}**")
        contexts = top_contexts(family_name, df_family, n=5)
        if contexts:
            lines.append("Top 5 contexts:")
            for idx, ctx in enumerate(contexts, start=1):
                lines.append(f"{idx}. {ctx}")
        else:
            lines.append("Top 5 contexts: none at current thresholds.")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    pair_prefix = f"{args.symbol_a}_{args.symbol_b}"
    primary = Path(args.input) if args.input else Path(f"pairs_behavior_atlas_reports_quick/readouts/{pair_prefix}_first_touch_digest.csv")
    fallback = Path(args.fallback_input) if args.fallback_input else Path(f"pairs_behavior_atlas_reports_quick/{pair_prefix}_pairs_grouped_behavior.csv")
    out_dir = Path(args.output_dir)

    df, source_path = load_input(primary, fallback)
    ensure_columns(df)

    n0, n1, d0, d1, d2, st = 0.90, 0.85, 0.10, 0.50, 0.25, 0.05
    min_obs = int(args.min_observations)

    family_frames: Dict[str, pd.DataFrame] = {}
    summary_rows: List[Dict[str, object]] = []

    for family_name, query_expr in FAMILIES:
        filtered = df.query(query_expr).copy()
        family_frames[family_name] = filtered
        summary_rows.append(summarize_family(family_name, filtered))

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_dir / f"{pair_prefix}_first_touch_families.csv"
    md_out = out_dir / f"{pair_prefix}_first_touch_families.md"

    pd.DataFrame(summary_rows).to_csv(csv_out, index=False)
    md_out.write_text(build_markdown(source_path, len(df), family_frames), encoding="utf-8")

    print(f"Wrote: {csv_out}")
    print(f"Wrote: {md_out}")


if __name__ == "__main__":
    main()
