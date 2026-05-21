#!/usr/bin/env python3
"""Summarize descriptive behavior families from grouped atlas CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_INPUT = "market_behavior_atlas_reports_quick/EURUSD_atlas_grouped_behavior.csv"
DEFAULT_OUTPUT_DIR = "market_behavior_atlas_reports_quick/readouts"
TARGET_HORIZONS = [5, 10, 20, 40]

CONTEXT_COLUMNS = [
    "grouping",
    "hour",
    "session_bucket",
    "range_size_bucket",
    "body_size_bucket",
    "wick_structure_bucket",
    "compression_bucket",
    "distance_from_rolling_16_abs_bucket",
    "distance_from_daily_open_abs_bucket",
    "atr_percentile_bucket",
    "streak_bucket",
    "previous_3_direction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Behavior-family summarization for Market Behavior Atlas.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to grouped behavior CSV.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for readout outputs.")
    parser.add_argument("--min-observations", type=int, default=200, help="Minimum observations per row.")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def context_string(row: pd.Series) -> str:
    parts: list[str] = []
    for col in CONTEXT_COLUMNS:
        if col not in row.index:
            continue
        value = row[col]
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text == "":
            continue
        parts.append(f"{col}={text}")
    return " | ".join(parts) if parts else "context_unavailable"


def select_top_contexts(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["n/a", "n/a", "n/a"]
    ranked = df.sort_values(["observations", "two_sided_movement"], ascending=[False, False]).head(3)
    contexts = [context_string(r) for _, r in ranked.iterrows()]
    while len(contexts) < 3:
        contexts.append("n/a")
    return contexts


def summarize_family(df_all: pd.DataFrame, family_name: str, family_mask: pd.Series) -> pd.DataFrame:
    family_df = df_all[family_mask].copy()
    if family_df.empty:
        return pd.DataFrame()

    rows = []
    for horizon, part in family_df.groupby("horizon_bars", dropna=False):
        top1, top2, top3 = select_top_contexts(part)
        obs_min = int(part["observations"].min())
        obs_max = int(part["observations"].max())
        rows.append(
            {
                "family_name": family_name,
                "horizon_bars": int(horizon),
                "rows_in_family": int(len(part)),
                "total_observations_sum": int(part["observations"].sum()),
                "max_observations": int(part["observations"].max()),
                "avg_two_sided_movement": float(part["two_sided_movement"].mean()),
                "max_two_sided_movement": float(part["two_sided_movement"].max()),
                "avg_hit_up_8_rate": float(part["hit_up_8_rate"].mean()),
                "max_hit_up_8_rate": float(part["hit_up_8_rate"].max()),
                "avg_hit_down_8_rate": float(part["hit_down_8_rate"].mean()),
                "max_hit_down_8_rate": float(part["hit_down_8_rate"].max()),
                "avg_up_8_minus_down_8": float(part["up_8_minus_down_8"].mean()),
                "max_up_8_minus_down_8": float(part["up_8_minus_down_8"].max()),
                "avg_down_8_minus_up_8": float(part["down_8_minus_up_8"].mean()),
                "max_down_8_minus_up_8": float(part["down_8_minus_up_8"].max()),
                "avg_up_8_is_oos_gap": float(part["up_8_is_oos_gap"].mean()),
                "avg_down_8_is_oos_gap": float(part["down_8_is_oos_gap"].mean()),
                "top_context_1": top1,
                "top_context_2": top2,
                "top_context_3": top3,
                "notes": (
                    f"obs_range={obs_min}-{obs_max}; "
                    f"hit_up_8_rate_range={part['hit_up_8_rate'].min():.4f}-{part['hit_up_8_rate'].max():.4f}; "
                    f"hit_down_8_rate_range={part['hit_down_8_rate'].min():.4f}-{part['hit_down_8_rate'].max():.4f}; "
                    f"stability_avg_gaps(up/down)={part['up_8_is_oos_gap'].mean():.4f}/{part['down_8_is_oos_gap'].mean():.4f}"
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(["horizon_bars", "family_name"]).reset_index(drop=True)


def build_markdown(summary_df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Market Behavior Atlas - Behavior Family Summary")
    lines.append("")
    lines.append("This readout is descriptive behavior only and is not a strategy.")
    lines.append("")
    lines.append("## Behavior Family Definitions")
    lines.append("- **Volatility Expansion**: larger range/body conditions, higher ATR percentile, or expanded compression context.")
    lines.append("- **Anchor Distance Expansion**: larger absolute distance from rolling anchor or daily open anchor.")
    lines.append("- **Midnight Upward Skew**: hour 0 contexts with relatively elevated positive up-minus-down frequency difference.")
    lines.append("- **Late Session Downward Skew**: hour 22/23 contexts with relatively elevated positive down-minus-up frequency difference.")
    lines.append("- **London / New York Expansion Windows**: london/newyork open-mid windows with relatively high two-sided movement.")
    lines.append("- **Compression Context**: compressed contexts summarized by horizon for movement and skew observations.")
    lines.append("")

    for horizon in TARGET_HORIZONS:
        lines.append(f"## Horizon {horizon}")
        part = summary_df[summary_df["horizon_bars"] == horizon].copy()
        if part.empty:
            lines.append("No rows available for this horizon after filters.")
            lines.append("")
            continue

        display_cols = [
            "family_name",
            "rows_in_family",
            "total_observations_sum",
            "avg_two_sided_movement",
            "max_two_sided_movement",
            "avg_hit_up_8_rate",
            "avg_hit_down_8_rate",
            "avg_up_8_minus_down_8",
            "avg_down_8_minus_up_8",
            "avg_up_8_is_oos_gap",
            "avg_down_8_is_oos_gap",
            "notes",
        ]
        lines.append(part[display_cols].to_markdown(index=False))
        lines.append("")
        lines.append("Top contexts per family:")
        for _, row in part.iterrows():
            lines.append(f"- **{row['family_name']}**")
            lines.append(f"  1. {row['top_context_1']}")
            lines.append(f"  2. {row['top_context_2']}")
            lines.append(f"  3. {row['top_context_3']}")
        lines.append("")

    lines.append("## Notes")
    lines.append("- Two-sided movement means future movement in both directions was elevated.")
    lines.append("- Upward/downward skew means relative frequency difference, not a trading instruction.")
    lines.append("- IS/OOS gaps should be checked before using any behavior pattern for future research.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    required = [
        "horizon_bars",
        "observations",
        "mean_future_max_up_pips",
        "mean_future_max_down_pips",
        "hit_up_8_rate",
        "hit_down_8_rate",
        "IS_hit_up_8_rate",
        "OOS_hit_up_8_rate",
        "IS_hit_down_8_rate",
        "OOS_hit_down_8_rate",
        "hour",
        "session_bucket",
        "range_size_bucket",
        "body_size_bucket",
        "atr_percentile_bucket",
        "compression_bucket",
        "distance_from_rolling_16_abs_bucket",
        "distance_from_daily_open_abs_bucket",
    ]
    require_columns(df, required)

    df = df[df["observations"] >= args.min_observations].copy()
    if df.empty:
        raise ValueError("No rows remain after applying observation filter.")

    df["two_sided_movement"] = df["mean_future_max_up_pips"] + df["mean_future_max_down_pips"]
    df["up_8_minus_down_8"] = df["hit_up_8_rate"] - df["hit_down_8_rate"]
    df["down_8_minus_up_8"] = df["hit_down_8_rate"] - df["hit_up_8_rate"]
    df["up_8_is_oos_gap"] = (df["IS_hit_up_8_rate"] - df["OOS_hit_up_8_rate"]).abs()
    df["down_8_is_oos_gap"] = (df["IS_hit_down_8_rate"] - df["OOS_hit_down_8_rate"]).abs()

    up_cut = df[df["up_8_minus_down_8"] > 0]["up_8_minus_down_8"].quantile(0.75)
    down_cut = df[df["down_8_minus_up_8"] > 0]["down_8_minus_up_8"].quantile(0.75)
    ts_cut = df["two_sided_movement"].quantile(0.75)

    families = [
        ("Volatility Expansion", (df["range_size_bucket"].isin(["large", "extreme"])) | (df["body_size_bucket"].isin(["large", "extreme"])) | (df["atr_percentile_bucket"].isin(["high_75_90", "extreme_90_plus"])) | (df["compression_bucket"] == "expanded")),
        ("Anchor Distance Expansion", (df["distance_from_rolling_16_abs_bucket"].isin(["25_40", "40_plus"])) | (df["distance_from_daily_open_abs_bucket"].isin(["25_40", "40_plus"]))),
        ("Midnight Upward Skew", (df["hour"] == 0) & (df["up_8_minus_down_8"] > 0) & (df["up_8_minus_down_8"] >= up_cut)),
        ("Late Session Downward Skew", (df["hour"].isin([22, 23])) & (df["down_8_minus_up_8"] > 0) & (df["down_8_minus_up_8"] >= down_cut)),
        ("London / New York Expansion Windows", (df["session_bucket"].isin(["london_open", "london_mid", "newyork_open", "newyork_mid"])) & (df["two_sided_movement"] >= ts_cut)),
        ("Compression Context", (df["compression_bucket"] == "compressed")),
    ]

    summary_frames = [summarize_family(df, name, mask) for name, mask in families]
    summary_df = pd.concat([f for f in summary_frames if not f.empty], ignore_index=True)
    summary_df = summary_df.sort_values(["horizon_bars", "family_name"]).reset_index(drop=True)

    csv_path = output_dir / "behavior_families.csv"
    md_path = output_dir / "behavior_families.md"
    summary_df.to_csv(csv_path, index=False)
    md_path.write_text(build_markdown(summary_df), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
