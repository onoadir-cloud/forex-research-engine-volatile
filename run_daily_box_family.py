#!/usr/bin/env python3
"""Scan M15 CSV files from data/ and write simple daily-box family reports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_DATA_GLOB = "data/*.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan data CSV files and emit report artifacts to an output directory."
    )
    parser.add_argument(
        "--output-dir",
        default="reports/daily_box_family",
        help="Directory where report files will be written.",
    )
    return parser.parse_args()


def discover_csv_files() -> list[Path]:
    return sorted(Path().glob(DEFAULT_DATA_GLOB))


def count_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        # Try to skip header when present; if not present this still reports sensible count.
        rows = list(reader)
    if not rows:
        return 0
    header_like = any(cell.strip().lower() in {"date", "time", "datetime", "open"} for cell in rows[0])
    return max(len(rows) - 1, 0) if header_like else len(rows)


def write_reports(csv_files: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "symbol", "timeframe", "variant", "rows"])
        for file_path in csv_files:
            stem_parts = file_path.stem.split("_")
            symbol = stem_parts[0] if stem_parts else ""
            timeframe = stem_parts[1] if len(stem_parts) > 1 else ""
            variant = "_".join(stem_parts[2:]) if len(stem_parts) > 2 else ""
            writer.writerow([file_path.name, symbol, timeframe, variant, count_rows(file_path)])

    readme_path = output_dir / "README.txt"
    with readme_path.open("w", encoding="utf-8") as f:
        f.write("Daily Box Family Report\n")
        f.write("=======================\n\n")
        f.write(f"Scanned glob: {DEFAULT_DATA_GLOB}\n")
        f.write(f"CSV files discovered: {len(csv_files)}\n")
        f.write("Generated files:\n")
        f.write("- summary.csv\n")



def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    csv_files = discover_csv_files()
    write_reports(csv_files, output_dir)
    print(f"Scanned {len(csv_files)} CSV file(s) from {DEFAULT_DATA_GLOB}")
    print(f"Reports written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
