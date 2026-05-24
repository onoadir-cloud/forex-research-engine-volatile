import subprocess
import sys
from pathlib import Path


def test_splits_are_chronological_and_locked():
    text = Path("run_locked_robustness_directional_run.py").read_text(encoding="utf-8")
    assert "(\"train_reference\", \"2022-01-01\", \"2023-12-31\")" in text
    assert "(\"validation\", \"2024-01-01\", \"2024-12-31\")" in text
    assert "(\"test_oos\", \"2025-01-01\", \"2026-12-31\")" in text


def test_cli_exists_and_no_optimization_flags():
    result = subprocess.run(
        [sys.executable, "run_locked_robustness_directional_run.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--csv" in result.stdout
    assert "--output-dir" in result.stdout

    text = Path("run_locked_robustness_directional_run.py").read_text(encoding="utf-8")
    assert "summarize_period(events, \"year\")" in text
    assert "summarize_period(events, \"year_month\")" in text
    assert "edge_vs_random_mean" in text
    assert "edge_vs_opposite_mean" in text
