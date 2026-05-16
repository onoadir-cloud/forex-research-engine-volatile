#!/usr/bin/env python3
"""Run the existing research/backtest flow for multiple forex symbols.

This script intentionally does not modify strategy logic, walk-forward logic,
cost assumptions, or filters. It only changes the symbol and aggregates outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


METRIC_FIELDS = [
    "oos_expectancy_after_costs",
    "profit_factor",
    "max_drawdown",
    "num_trades",
    "parameter_stability",
    "survives_2x_cost_stress",
]


def load_symbols(config_path: Path) -> Dict[str, List[str]]:
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "baseline_symbol" not in data or "symbols" not in data:
        raise ValueError("Config must include baseline_symbol and symbols.")
    if data["baseline_symbol"] != "EURUSD":
        raise ValueError("Baseline symbol must remain EURUSD.")
    return data


def run_for_symbol(symbol: str, command_template: str) -> int:
    command = command_template.format(symbol=symbol)
    print(f"[batch] Running {symbol}: {command}")
    completed = subprocess.run(command, shell=True)
    return completed.returncode


def load_metrics(metrics_path: Path) -> Dict[str, object]:
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    missing = [f for f in METRIC_FIELDS if f not in metrics]
    if missing:
        raise ValueError(f"Missing required metrics in {metrics_path}: {missing}")
    return metrics


def rank_symbols(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    def rank_for(field: str, reverse: bool) -> Dict[str, int]:
        ordered = sorted(rows, key=lambda r: r[field], reverse=reverse)
        return {row["symbol"]: idx + 1 for idx, row in enumerate(ordered)}

    ranks = {
        "oos_expectancy_after_costs": rank_for("oos_expectancy_after_costs", True),
        "profit_factor": rank_for("profit_factor", True),
        "max_drawdown": rank_for("max_drawdown", False),
        "num_trades": rank_for("num_trades", True),
        "parameter_stability": rank_for("parameter_stability", True),
        "survives_2x_cost_stress": rank_for("survives_2x_cost_stress", True),
    }

    for row in rows:
        row["ranking_score"] = sum(ranks[k][row["symbol"]] for k in ranks)

    return sorted(rows, key=lambda r: r["ranking_score"])


def write_report(ranked_rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Multi-Symbol Forex Strategy Comparison",
        "",
        f"Generated: {timestamp}",
        "",
        "Ranking score = sum of per-metric rank positions (lower is better).",
        "",
        "| Rank | Symbol | OOS Expectancy (After Costs) | Profit Factor | Max Drawdown | Trades | Parameter Stability | Survives 2x Costs | Ranking Score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for idx, row in enumerate(ranked_rows, start=1):
        lines.append(
            f"| {idx} | {row['symbol']} | {row['oos_expectancy_after_costs']:.6f} | {row['profit_factor']:.6f} | "
            f"{row['max_drawdown']:.6f} | {int(row['num_trades'])} | {row['parameter_stability']:.6f} | "
            f"{bool(row['survives_2x_cost_stress'])} | {row['ranking_score']} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="symbols_config.json")
    parser.add_argument(
        "--single-symbol-cmd",
        required=True,
        help="Existing single-symbol command template. Use {symbol} placeholder.",
    )
    parser.add_argument(
        "--metrics-template",
        default="results/{symbol}/metrics.json",
        help="Path template to each symbol metrics JSON.",
    )
    parser.add_argument(
        "--report",
        default="reports/multi_symbol_comparison.md",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one symbol run fails.",
    )
    args = parser.parse_args()

    cfg = load_symbols(Path(args.config))

    failures = []
    rows = []
    for symbol in cfg["symbols"]:
        rc = run_for_symbol(symbol, args.single_symbol_cmd)
        if rc != 0:
            failures.append((symbol, rc))
            if args.stop_on_error:
                break
            continue

        metrics_path = Path(args.metrics_template.format(symbol=symbol))
        if not metrics_path.exists():
            failures.append((symbol, "metrics_missing"))
            continue

        metrics = load_metrics(metrics_path)
        rows.append({"symbol": symbol, **metrics})

    if not rows:
        print("[batch] No successful runs with usable metrics; report not generated.")
        return 1

    ranked = rank_symbols(rows)
    write_report(ranked, Path(args.report))
    print(f"[batch] Report written to {args.report}")

    if failures:
        print(f"[batch] Completed with failures: {failures}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
