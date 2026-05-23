#!/usr/bin/env python3
"""Execution-reality screening layer over atlas family CSV outputs.

This script reads existing family summary CSVs and produces a rough
execution-feasibility screening report using hardcoded friction assumptions.
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_INPUTS = [
    "pairs_behavior_atlas_reports_quick/readouts/EURUSD_GBPUSD_first_touch_families.csv",
    "pairs_behavior_atlas_reports_GBPAUD_GBPNZD_quick/readouts/GBPAUD_GBPNZD_first_touch_families.csv",
]
DEFAULT_PAIR_NAMES = ["EURUSD_GBPUSD", "GBPAUD_GBPNZD"]
DEFAULT_OUTPUT_DIR = "execution_reality_reports"


COST_PROFILES: Dict[str, Dict[str, Dict[str, float]]] = {
    "low": {
        "EURUSD_GBPUSD": {
            "spread_a_pips": 0.8,
            "spread_b_pips": 1.2,
            "slippage_a_pips": 0.2,
            "slippage_b_pips": 0.3,
            "commission_equivalent_pips_total": 0.4,
        },
        "GBPAUD_GBPNZD": {
            "spread_a_pips": 2.5,
            "spread_b_pips": 3.5,
            "slippage_a_pips": 0.5,
            "slippage_b_pips": 0.7,
            "commission_equivalent_pips_total": 0.6,
        },
    },
    "conservative": {
        "EURUSD_GBPUSD": {
            "spread_a_pips": 1.2,
            "spread_b_pips": 1.8,
            "slippage_a_pips": 0.3,
            "slippage_b_pips": 0.5,
            "commission_equivalent_pips_total": 0.6,
        },
        "GBPAUD_GBPNZD": {
            "spread_a_pips": 3.5,
            "spread_b_pips": 5.0,
            "slippage_a_pips": 0.8,
            "slippage_b_pips": 1.0,
            "commission_equivalent_pips_total": 0.8,
        },
    },
    "high": {
        "EURUSD_GBPUSD": {
            "spread_a_pips": 2.0,
            "spread_b_pips": 2.5,
            "slippage_a_pips": 0.6,
            "slippage_b_pips": 0.8,
            "commission_equivalent_pips_total": 1.0,
        },
        "GBPAUD_GBPNZD": {
            "spread_a_pips": 5.0,
            "spread_b_pips": 7.0,
            "slippage_a_pips": 1.2,
            "slippage_b_pips": 1.5,
            "commission_equivalent_pips_total": 1.2,
        },
    },
}


BASE_FAMILY_PATTERNS = [
    "Normalization-first",
    "Strong normalization-to-zero",
    "Stable normalization-first",
]
DIVERGENCE_PATTERNS = [
    "Extreme divergence-first",
    "Severe divergence-first",
    "Stable divergence-first",
]


OPTIONAL_METRIC_COLUMNS = [
    "avg_normalized_0_5_first_vs_3_rate",
    "max_normalized_0_5_first_vs_3_rate",
    "avg_moved_further_to_3_first_vs_0_5_rate",
    "max_moved_further_to_3_first_vs_0_5_rate",
    "avg_normalized_0_first_vs_3_rate",
    "avg_moved_further_to_4_first_vs_0_5_rate",
    "avg_median_bars_to_first_touch_0_5_vs_3",
]

OUTPUT_COLUMNS = [
    "pair_name",
    "family_name",
    "cost_profile",
    "estimated_round_trip_friction_pips",
    "rows_in_family",
    "total_observations_sum",
    "max_observations",
    "avg_normalized_0_5_first_vs_3_rate",
    "max_normalized_0_5_first_vs_3_rate",
    "avg_normalized_0_first_vs_3_rate",
    "avg_median_bars_to_first_touch_0_5_vs_3",
    "avg_context_mean_abs_z",
    "median_context_p95_abs_z",
    "max_context_p95_abs_z",
    "rough_cost_pressure_proxy",
    "feasibility_band",
    "top_context_1",
    "top_context_2",
    "top_context_3",
    "notes",
]


@dataclass
class ContextParseResult:
    mean_abs_z: Optional[float]
    p95_abs_z: Optional[float]
    warning: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rough execution-reality summary from family CSV files.")
    parser.add_argument("--inputs", default=",".join(DEFAULT_INPUTS), help="Comma-separated family CSV paths.")
    parser.add_argument("--pair-names", default=",".join(DEFAULT_PAIR_NAMES), help="Comma-separated pair names.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory path.")
    parser.add_argument(
        "--cost-profile",
        default="conservative",
        choices=["low", "conservative", "high"],
        help="Execution friction cost profile.",
    )
    parser.add_argument(
        "--include-divergence-families",
        action="store_true",
        help="Also include divergence-first families in filtering.",
    )
    return parser.parse_args()


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def estimated_round_trip_friction(cost_cfg: Dict[str, float]) -> float:
    return (
        2 * (cost_cfg["spread_a_pips"] + cost_cfg["spread_b_pips"])
        + 2 * (cost_cfg["slippage_a_pips"] + cost_cfg["slippage_b_pips"])
        + cost_cfg["commission_equivalent_pips_total"]
    )


def parse_top_context_value(text: object, key: str) -> Optional[float]:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return None
    s = str(text).strip()
    if not s:
        return None

    try:
        if s.startswith("{") and s.endswith("}"):
            parsed = ast.literal_eval(s)
            if isinstance(parsed, dict) and key in parsed:
                val = parsed.get(key)
                return float(val) if val is not None else None
    except (ValueError, SyntaxError):
        pass

    match = re.search(rf"{re.escape(key)}\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def extract_context_metrics(row: pd.Series) -> Tuple[List[float], List[float], List[str]]:
    mean_abs_z_values: List[float] = []
    p95_values: List[float] = []
    warnings: List[str] = []

    for ctx_col in ["top_context_1", "top_context_2", "top_context_3"]:
        val = row.get(ctx_col)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        mean_v = parse_top_context_value(val, "mean_abs_z")
        p95_v = parse_top_context_value(val, "p95_abs_z")

        if mean_v is None and p95_v is None:
            warnings.append(f"context_parse_warning:{ctx_col}")
        if mean_v is not None and not math.isnan(mean_v):
            mean_abs_z_values.append(mean_v)
        if p95_v is not None and not math.isnan(p95_v):
            p95_values.append(p95_v)

    return mean_abs_z_values, p95_values, warnings


def to_float_or_nan(row: pd.Series, col: str) -> float:
    if col not in row.index:
        return np.nan
    v = row[col]
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan


def feasibility_band(value: float) -> str:
    if np.isnan(value):
        return "very_high_friction_pressure"
    if value <= 1.5:
        return "low_friction_pressure"
    if value <= 3.5:
        return "medium_friction_pressure"
    if value <= 6.0:
        return "high_friction_pressure"
    return "very_high_friction_pressure"


def family_matches(name: str, include_divergence: bool) -> bool:
    pats = list(BASE_FAMILY_PATTERNS)
    if include_divergence:
        pats.extend(DIVERGENCE_PATTERNS)
    return any(p in name for p in pats)


def build_rows(input_path: str, pair_name: str, cost_profile: str, include_divergence: bool) -> List[Dict[str, object]]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing family CSV: {input_path}")

    df = pd.read_csv(input_path)
    if "family_name" not in df.columns:
        raise ValueError(f"Missing required column 'family_name' in {input_path}")

    filtered = df[df["family_name"].astype(str).apply(lambda x: family_matches(x, include_divergence))].copy()
    pair_cost_cfg = COST_PROFILES[cost_profile].get(pair_name)
    if pair_cost_cfg is None:
        raise ValueError(f"No cost profile mapping for pair_name={pair_name} under cost_profile={cost_profile}")

    friction = estimated_round_trip_friction(pair_cost_cfg)
    rows: List[Dict[str, object]] = []

    for _, r in filtered.iterrows():
        mean_abs_z_values, p95_values, warnings = extract_context_metrics(r)
        avg_context_mean_abs_z = float(np.mean(mean_abs_z_values)) if mean_abs_z_values else np.nan
        max_context_p95_abs_z = float(np.max(p95_values)) if p95_values else np.nan
        median_context_p95_abs_z = float(np.median(p95_values)) if p95_values else np.nan

        denom = max(max_context_p95_abs_z if not np.isnan(max_context_p95_abs_z) else np.nan, 0.01)
        if np.isnan(denom):
            denom = 0.01
        rough_cost_pressure_proxy = friction / denom

        note_parts = ["rough screening only; unit mismatch between pips and z-score"]
        if warnings:
            note_parts.append(";".join(sorted(set(warnings))))

        obs_col_candidates = ["obs", "observations", "n_obs", "count"]
        obs_values = []
        for c in obs_col_candidates:
            if c in filtered.columns:
                val = to_float_or_nan(r, c)
                if not np.isnan(val):
                    obs_values.append(val)

        row = {
            "pair_name": pair_name,
            "family_name": r.get("family_name", ""),
            "cost_profile": cost_profile,
            "estimated_round_trip_friction_pips": friction,
            "rows_in_family": 1,
            "total_observations_sum": float(np.sum(obs_values)) if obs_values else np.nan,
            "max_observations": float(np.max(obs_values)) if obs_values else np.nan,
            "avg_normalized_0_5_first_vs_3_rate": to_float_or_nan(r, "avg_normalized_0_5_first_vs_3_rate"),
            "max_normalized_0_5_first_vs_3_rate": to_float_or_nan(r, "max_normalized_0_5_first_vs_3_rate"),
            "avg_normalized_0_first_vs_3_rate": to_float_or_nan(r, "avg_normalized_0_first_vs_3_rate"),
            "avg_median_bars_to_first_touch_0_5_vs_3": to_float_or_nan(r, "avg_median_bars_to_first_touch_0_5_vs_3"),
            "avg_context_mean_abs_z": avg_context_mean_abs_z,
            "median_context_p95_abs_z": median_context_p95_abs_z,
            "max_context_p95_abs_z": max_context_p95_abs_z,
            "rough_cost_pressure_proxy": rough_cost_pressure_proxy,
            "feasibility_band": feasibility_band(rough_cost_pressure_proxy),
            "top_context_1": r.get("top_context_1", np.nan),
            "top_context_2": r.get("top_context_2", np.nan),
            "top_context_3": r.get("top_context_3", np.nan),
            "notes": " | ".join(note_parts),
        }
        rows.append(row)

    return rows


def write_markdown(
    out_path: str,
    summary_df: pd.DataFrame,
    inputs: List[str],
    pair_names: List[str],
    cost_profile: str,
) -> None:
    lines: List[str] = []
    lines.append("# Execution Reality Layer Summary")
    lines.append("")
    lines.append("## Sources and Configuration")
    lines.append(f"- Cost profile: **{cost_profile}**")
    lines.append("- Source files:")
    for p, n in zip(inputs, pair_names):
        lines.append(f"  - `{n}`: `{p}`")
    lines.append("")
    lines.append(
        "**Warning:** This is not a strategy test, not a profitability test, and not a decision trigger."
    )
    lines.append("")

    lines.append("## Cost Assumptions by Pair")
    lines.append("| pair_name | spread_a_pips | spread_b_pips | slippage_a_pips | slippage_b_pips | commission_equivalent_pips_total | estimated_round_trip_friction_pips |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for pair in pair_names:
        cfg = COST_PROFILES[cost_profile].get(pair)
        if cfg is None:
            continue
        friction = estimated_round_trip_friction(cfg)
        lines.append(
            f"| {pair} | {cfg['spread_a_pips']:.2f} | {cfg['spread_b_pips']:.2f} | {cfg['slippage_a_pips']:.2f} | {cfg['slippage_b_pips']:.2f} | {cfg['commission_equivalent_pips_total']:.2f} | {friction:.2f} |"
        )
    lines.append("")

    lines.append("## Summary by Pair and Family")
    lines.append("| pair_name | family_name | estimated_round_trip_friction_pips | max_context_p95_abs_z | rough_cost_pressure_proxy | feasibility_band |")
    lines.append("|---|---|---:|---:|---:|---|")
    for _, r in summary_df.iterrows():
        lines.append(
            "| "
            f"{r['pair_name']} | {r['family_name']} | {r['estimated_round_trip_friction_pips']:.4f} | "
            f"{r['max_context_p95_abs_z'] if pd.notna(r['max_context_p95_abs_z']) else ''} | "
            f"{r['rough_cost_pressure_proxy']:.4f} | {r['feasibility_band']} |"
        )
    lines.append("")

    lines.append("## Feasibility Interpretation")
    lines.append("- EURUSD/GBPUSD is expected to show lower execution friction pressure.")
    lines.append("- GBPAUD/GBPNZD is expected to show materially higher execution friction pressure.")
    lines.append("")

    lines.append("## Notes")
    lines.append("- z-score is unitless, so rough_cost_pressure_proxy is only a rough screening heuristic.")
    lines.append("- True execution modeling requires spread series, commission, slippage, and broker-specific conditions.")
    lines.append("- Next step should be either spread-data ingestion or broader pair coverage.")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    inputs = parse_csv_list(args.inputs)
    pair_names = parse_csv_list(args.pair_names)

    if len(inputs) != len(pair_names):
        raise ValueError("--inputs and --pair-names must contain the same number of entries")

    os.makedirs(args.output_dir, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    for input_path, pair_name in zip(inputs, pair_names):
        rows = build_rows(
            input_path=input_path,
            pair_name=pair_name,
            cost_profile=args.cost_profile,
            include_divergence=args.include_divergence_families,
        )
        all_rows.extend(rows)

    summary_df = pd.DataFrame(all_rows)
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        for c in OUTPUT_COLUMNS:
            if c not in summary_df.columns:
                summary_df[c] = np.nan
        summary_df = summary_df[OUTPUT_COLUMNS]

    csv_out = os.path.join(args.output_dir, "execution_reality_summary.csv")
    md_out = os.path.join(args.output_dir, "execution_reality_summary.md")

    summary_df.to_csv(csv_out, index=False)
    write_markdown(md_out, summary_df, inputs, pair_names, args.cost_profile)

    print(f"Wrote: {csv_out}")
    print(f"Wrote: {md_out}")


if __name__ == "__main__":
    main()
