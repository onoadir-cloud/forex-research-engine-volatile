#!/usr/bin/env python3
"""Run fixed-parameter local research flow for multiple forex symbols."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List

CONFIG_PATH = Path("symbols_config.json")
SUMMARY_PATH = Path("reports/multisymbol_batch_summary.json")
DATA_TEMPLATE = "data/{symbol}_M15_MT5_5Y.csv"
OUTPUT_TEMPLATE = "reports/{symbol}"


def load_symbols(config_path: Path) -> List[str]:
    with config_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if "symbols" not in data or not isinstance(data["symbols"], list):
        raise ValueError("symbols_config.json must contain a 'symbols' list.")
    return data["symbols"]


def build_command(symbol: str, data_file: Path, output_dir: Path) -> List[str]:
    return [
        "python",
        "run_research.py",
        "--csv",
        str(data_file),
        "--symbol",
        symbol,
        "--base-timeframe",
        "M15",
        "--spread-pips",
        "1.2",
        "--slippage-pips",
        "0.2",
        "--output-dir",
        str(output_dir),
    ]


def run_symbol(symbol: str) -> Dict[str, object]:
    data_file = Path(DATA_TEMPLATE.format(symbol=symbol))
    output_dir = Path(OUTPUT_TEMPLATE.format(symbol=symbol))

    entry: Dict[str, object] = {
        "symbol": symbol,
        "status": "pending",
        "data_file": str(data_file),
        "data_file_found": data_file.exists(),
        "command": None,
        "output_dir": str(output_dir),
        "generated_report_paths": [],
        "return_code": None,
    }

    if not data_file.exists():
        entry["status"] = "missing_data"
        return entry

    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(symbol, data_file, output_dir)
    entry["command"] = " ".join(command)

    print(f"[batch] Running {symbol}: {entry['command']}")
    completed = subprocess.run(command, check=False)

    entry["return_code"] = completed.returncode
    if completed.returncode == 0:
        entry["status"] = "completed"
    else:
        entry["status"] = "failed"

    if output_dir.exists():
        entry["generated_report_paths"] = sorted(
            str(path)
            for path in output_dir.rglob("*")
            if path.is_file()
        )

    return entry


def main() -> int:
    symbols = load_symbols(CONFIG_PATH)

    summary = []
    for symbol in symbols:
        summary.append(run_symbol(symbol))

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbols": summary,
        "total_symbols": len(summary),
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[batch] Summary written: {SUMMARY_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
