#!/usr/bin/env python3
"""Descriptive fragility diagnostics for signed-relative micro context events."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


INPUT_PATH = Path("signed_relative_micro_reports/EURUSD_GBPUSD_signed_relative_events.csv")
OUTPUT_DIR = Path("fragility_forensics_reports")
OUTPUT_CSV = OUTPUT_DIR / "EURUSD_GBPUSD_fragility_forensics.csv"
OUTPUT_MD = OUTPUT_DIR / "EURUSD_GBPUSD_fragility_forensics.md"

REQUIRED_COLUMNS = [
    "datetime",
    "year",
    "current_z",
    "first_touch_0_5_vs_3",
    "bars_to_first_touch",
    "basket_move_pips_proxy",
    "estimated_round_trip_friction_pips",
    "conservative_behavior_after_friction",
    "signed_convergence_direction",
    "signed_convergence_log_move",
    "signed_z_convergence",
]


def assign_abs_z_subbucket(v: float) -> str:
    if pd.isna(v):
        return "other"
    if 1.0 <= v < 1.2:
        return "1.0_1.2"
    if 1.2 <= v < 1.4:
        return "1.2_1.4"
    if 1.4 <= v < 1.6:
        return "1.4_1.6"
    if 1.6 <= v < 1.8:
        return "1.6_1.8"
    if 1.8 <= v <= 2.0:
        return "1.8_2.0"
    return "other"


def assign_bars_bucket(v: float) -> str:
    if pd.isna(v):
        return "missing"
    if 1 <= v <= 5:
        return "1_5"
    if 6 <= v <= 10:
        return "6_10"
    if 11 <= v <= 20:
        return "11_20"
    if 21 <= v <= 40:
        return "21_40"
    return "missing"


def assign_movement_bucket(v: float) -> str:
    if pd.isna(v):
        return "missing"
    if 0 <= v < 5:
        return "0_5"
    if 5 <= v < 8.2:
        return "5_8_2"
    if 8.2 <= v < 12:
        return "8_2_12"
    if 12 <= v < 20:
        return "12_20"
    if v >= 20:
        return "20_plus"
    return "missing"


def summarize_group(df: pd.DataFrame, grouping: str, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        group_value = " | ".join(str(x) for x in key_tuple)
        valid_after = g["conservative_behavior_after_friction"].dropna()
        observations = len(g)
        pos_count = (g["after_friction_label"] == "positive_after_friction").sum()
        neg_count = (g["after_friction_label"] == "negative_after_friction").sum()
        rows.append(
            {
                "grouping": grouping,
                "group_value": group_value,
                "observations": observations,
                "positive_after_friction_rate": pos_count / observations if observations else float("nan"),
                "negative_after_friction_rate": neg_count / observations if observations else float("nan"),
                "median_conservative_behavior_after_friction": valid_after.median() if not valid_after.empty else float("nan"),
                "p25_conservative_behavior_after_friction": valid_after.quantile(0.25) if not valid_after.empty else float("nan"),
                "p75_conservative_behavior_after_friction": valid_after.quantile(0.75) if not valid_after.empty else float("nan"),
                "median_basket_move_pips_proxy": g["basket_move_pips_proxy"].median(),
                "median_bars_to_first_touch": g["bars_to_first_touch"].median(),
                "median_current_abs_z": g["current_abs_z"].median(),
            }
        )
    return pd.DataFrame(rows)


def find_negative_clusters(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("datetime").reset_index(drop=True).copy()
    d["is_negative"] = d["conservative_behavior_after_friction"].le(0)

    clusters = []
    in_cluster = False
    start_idx = None

    for i, is_neg in enumerate(d["is_negative"]):
        if is_neg and not in_cluster:
            in_cluster = True
            start_idx = i
        elif not is_neg and in_cluster:
            cluster_df = d.iloc[start_idx:i]
            clusters.append(cluster_df)
            in_cluster = False
            start_idx = None

    if in_cluster and start_idx is not None:
        clusters.append(d.iloc[start_idx:])

    rows = []
    for c in clusters:
        if c.empty:
            continue
        start_dt = c["datetime"].iloc[0]
        rows.append(
            {
                "cluster_start_datetime": start_dt,
                "cluster_end_datetime": c["datetime"].iloc[-1],
                "cluster_length": len(c),
                "cluster_year": int(c["year"].median()) if not c["year"].isna().all() else pd.NA,
                "cluster_month": int(c["month"].median()) if not c["month"].isna().all() else pd.NA,
                "median_current_abs_z": c["current_abs_z"].median(),
                "median_bars_to_first_touch": c["bars_to_first_touch"].median(),
                "median_basket_move_pips_proxy": c["basket_move_pips_proxy"].median(),
                "median_conservative_behavior_after_friction": c["conservative_behavior_after_friction"].median(),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["cluster_length", "cluster_start_datetime"], ascending=[False, True]).head(20)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    return df.to_markdown(index=False)


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"ERROR: Source file not found: {INPUT_PATH}")
        return 1

    df = pd.read_csv(INPUT_PATH)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print("ERROR: Missing required columns:")
        for col in missing:
            print(f"- {col}")
        return 1

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["month"] = df["datetime"].dt.month

    df["current_abs_z"] = df["current_z"].abs()
    df["z_sign"] = df["current_z"].apply(lambda x: "positive_z" if pd.notna(x) and x > 0 else "negative_z")
    df["after_friction_label"] = "no_size_data"
    df.loc[df["conservative_behavior_after_friction"] > 0, "after_friction_label"] = "positive_after_friction"
    df.loc[df["conservative_behavior_after_friction"].le(0), "after_friction_label"] = "negative_after_friction"
    df.loc[df["conservative_behavior_after_friction"].isna(), "after_friction_label"] = "no_size_data"

    nf = df[df["first_touch_0_5_vs_3"] == "normalized_0_5_first"].copy()

    nf["current_abs_z_subbucket"] = nf["current_abs_z"].apply(assign_abs_z_subbucket)
    nf["bars_to_touch_bucket"] = nf["bars_to_first_touch"].apply(assign_bars_bucket)
    nf["movement_bucket"] = nf["basket_move_pips_proxy"].apply(assign_movement_bucket)

    summary_specs = [
        ("year", ["year"]),
        ("month", ["month"]),
        ("z_sign", ["z_sign"]),
        ("current_abs_z_subbucket", ["current_abs_z_subbucket"]),
        ("bars_to_touch_bucket", ["bars_to_touch_bucket"]),
        ("movement_bucket", ["movement_bucket"]),
        ("year+month", ["year", "month"]),
        ("z_sign+current_abs_z_subbucket", ["z_sign", "current_abs_z_subbucket"]),
        ("bars_to_touch_bucket+current_abs_z_subbucket", ["bars_to_touch_bucket", "current_abs_z_subbucket"]),
        ("year+current_abs_z_subbucket", ["year", "current_abs_z_subbucket"]),
    ]

    summaries = [summarize_group(nf, name, cols) for name, cols in summary_specs]
    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()

    clusters_df = find_negative_clusters(nf)
    clusters_out = clusters_df.copy()
    if not clusters_out.empty:
        clusters_out.insert(0, "grouping", "negative_sequence_cluster")
        clusters_out.insert(1, "group_value", clusters_out.index.astype(str))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([summary_df, clusters_out], ignore_index=True, sort=False).to_csv(OUTPUT_CSV, index=False)

    total = len(nf)
    pos = (nf["after_friction_label"] == "positive_after_friction").sum()
    neg = (nf["after_friction_label"] == "negative_after_friction").sum()

    by_year = summary_df[summary_df["grouping"] == "year"].sort_values("group_value")
    by_z = summary_df[summary_df["grouping"] == "current_abs_z_subbucket"]
    by_bars = summary_df[summary_df["grouping"] == "bars_to_touch_bucket"]
    by_move = summary_df[summary_df["grouping"] == "movement_bucket"]
    by_sign = summary_df[summary_df["grouping"] == "z_sign"]

    md = "# EURUSD_GBPUSD Fragility Forensics\n\n"
    md += "**Warning:** Descriptive fragility diagnostics only, not trading guidance.\n\n"
    md += f"**Source file:** `{INPUT_PATH}`\n\n"
    md += "## Overall counts\n"
    md += f"- total normalized-first events: {total}\n"
    md += f"- positive-after-friction count/rate: {pos} / {(pos / total) if total else float('nan'):.4f}\n"
    md += f"- negative-after-friction count/rate: {neg} / {(neg / total) if total else float('nan'):.4f}\n\n"

    md += "## By year\n" + markdown_table(by_year) + "\n\n"
    md += "## By current_abs_z_subbucket\n" + markdown_table(by_z) + "\n\n"
    md += "## By bars_to_touch_bucket\n" + markdown_table(by_bars) + "\n\n"
    md += "## By movement_bucket\n" + markdown_table(by_move) + "\n\n"
    md += "## By z_sign\n" + markdown_table(by_sign) + "\n\n"
    md += "## Top negative clusters\n" + markdown_table(clusters_df) + "\n\n"

    md += "## Final interpretation\n"
    md += "- Small movement size diagnostics: compare negative-after-friction rates in `0_5` and `5_8_2` movement buckets versus larger movement buckets.\n"
    md += "- Year/month diagnostics: review `year` and `year+month` grouping rows for concentration of fragile segments.\n"
    md += "- Lower abs_z within 1_2 diagnostics: compare `1.0_1.2` behavior stability against higher current_abs_z subbuckets.\n"
    md += "- Timing diagnostics: compare `21_40` versus early touch buckets to assess sequence stress from late timing.\n"
    md += "- Behavior stability conclusion: use the above context diagnostics to judge whether the context remains too fragile.\n"

    OUTPUT_MD.write_text(md, encoding="utf-8")

    print(f"Wrote: {OUTPUT_CSV}")
    print(f"Wrote: {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
