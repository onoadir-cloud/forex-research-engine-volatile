import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from run_daily_box_strategy_research import _find_setup


def test_find_setup_long_confirmation():
    box_top = 101.0
    box_bottom = 100.0
    rows = [
        {"open": 100.2, "high": 100.6, "low": 100.1, "close": 100.5, "box_top": box_top, "box_bottom": box_bottom},
        {"open": 100.5, "high": 100.7, "low": 99.9, "close": 100.1, "box_top": box_top, "box_bottom": box_bottom},
        {"open": 100.1, "high": 100.9, "low": 100.0, "close": 100.65, "box_top": box_top, "box_bottom": box_bottom},
    ]
    day_df = pd.DataFrame(rows)
    idx, ref = _find_setup(day_df, "long")
    assert idx == 2
    assert ref == 100.6
