import subprocess
import sys
from pathlib import Path


def test_family_script_scope_and_constraints():
    text = Path("run_family_robustness_directional_run.py").read_text(encoding="utf-8")

    for required in [
        '"momentum_20"',
        '"momentum_50"',
        '"ma_position_50"',
        '"ma_slope_20"',
        '"recent_range_position"',
    ]:
        assert required in text

    assert '"breakout_state_20"' not in text
    assert '"breakout_state_50"' not in text
    assert "PULLBACK_ATR = 1.5" in text
    assert '("reference_2022_2023", "2022-01-01", "2023-12-31")' in text
    assert '("validation_2024", "2024-01-01", "2024-12-31")' in text
    assert '("test_oos_2025_2026", "2025-01-01", "2026-12-31")' in text
    assert '"family_events.csv"' in text
    assert '"family_by_method.csv"' in text
    assert '"family_by_symbol.csv"' in text
    assert '"family_by_year.csv"' in text
    assert '"family_walkforward.csv"' in text
    assert '"family_verdict.csv"' in text
    assert '"family_robustness_summary.md"' in text
    assert '"random_baseline_deterministic"' in text


def test_family_script_cli_help():
    result = subprocess.run(
        [sys.executable, "run_family_robustness_directional_run.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--cost-profile" in result.stdout
    assert "--atr-period" in result.stdout
    assert "--output-dir" in result.stdout
