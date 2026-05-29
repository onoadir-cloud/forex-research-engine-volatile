import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from run_daily_box_session_open_research import (  # noqa: E402
    Config,
    build_daily_boxes,
    baseline_prices,
    deterministic_random_uses_strategy,
    detect_setup,
    find_session_open,
    manage_trade,
    run_research,
    target_price,
    walkforward,
)


def test_daily_box_built_from_previous_trading_date_only_and_ignores_current_day():
    df = pd.DataFrame(
        [
            {"datetime": "2025-01-01 00:00", "open": 1.10, "high": 1.20, "low": 1.00, "close": 1.15},
            {"datetime": "2025-01-01 00:15", "open": 1.15, "high": 1.25, "low": 1.05, "close": 1.20},
            {"datetime": "2025-01-02 00:00", "open": 1.20, "high": 9.99, "low": 0.01, "close": 1.21},
        ]
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    boxes, _ = build_daily_boxes(df, 0.0001)
    day2 = pd.Timestamp("2025-01-02").date()
    assert boxes[day2]["box_top"] == 1.25
    assert boxes[day2]["box_bottom"] == 1.00
    assert boxes[day2]["box_source_date"] == "2025-01-01"


def test_session_open_mapping_finds_first_available_at_or_after_hour():
    df = pd.DataFrame({"datetime": pd.to_datetime(["2025-01-02 06:45", "2025-01-02 07:15"]), "date": [pd.Timestamp("2025-01-02").date()] * 2})
    idx, dt = find_session_open(df, 7)
    assert idx == 1
    assert dt == pd.Timestamp("2025-01-02 07:15")


def test_long_bottom_sweep_reclaim_detection_works():
    box = {"box_bottom": 100.0, "box_top": 101.0}
    scan = pd.DataFrame(
        [
            {"datetime": "2025-01-02 07:00", "open": 100.2, "high": 100.6, "low": 100.1, "close": 100.5},
            {"datetime": "2025-01-02 07:15", "open": 100.5, "high": 100.7, "low": 99.9, "close": 100.1},
            {"datetime": "2025-01-02 07:30", "open": 100.1, "high": 100.9, "low": 100.0, "close": 100.65},
        ]
    )
    scan["datetime"] = pd.to_datetime(scan["datetime"])
    setup = detect_setup(scan, "long", box)
    assert setup["setup_direction"] == "long"
    assert setup["confirmation_datetime"] == pd.Timestamp("2025-01-02 07:30")


def test_short_top_sweep_reclaim_detection_works():
    box = {"box_bottom": 100.0, "box_top": 101.0}
    scan = pd.DataFrame(
        [
            {"datetime": "2025-01-02 07:00", "open": 100.8, "high": 100.9, "low": 100.4, "close": 100.5},
            {"datetime": "2025-01-02 07:15", "open": 100.5, "high": 101.1, "low": 100.3, "close": 100.9},
            {"datetime": "2025-01-02 07:30", "open": 100.9, "high": 101.0, "low": 100.1, "close": 100.35},
        ]
    )
    scan["datetime"] = pd.to_datetime(scan["datetime"])
    setup = detect_setup(scan, "short", box)
    assert setup["setup_direction"] == "short"
    assert setup["confirmation_datetime"] == pd.Timestamp("2025-01-02 07:30")


def test_target_modes_calculate_correct_tp_for_long_and_short():
    box = {"box_bottom": 100.0, "box_top": 104.0, "box_mid": 102.0, "box_q1": 101.0, "box_q3": 103.0}
    assert target_price("long", "mid", box) == 102.0
    assert target_price("long", "far_quartile", box) == 103.0
    assert target_price("long", "opposite_edge", box) == 104.0
    assert target_price("short", "mid", box) == 102.0
    assert target_price("short", "far_quartile", box) == 101.0
    assert target_price("short", "opposite_edge", box) == 100.0


def test_ambiguous_tp_sl_same_candle_is_treated_as_sl():
    day = pd.DataFrame(
        [
            {"datetime": "2025-01-02 07:30", "open": 100.5, "high": 100.6, "low": 100.4, "close": 100.5},
            {"datetime": "2025-01-02 07:45", "open": 100.5, "high": 101.5, "low": 99.5, "close": 100.7},
        ]
    )
    day["datetime"] = pd.to_datetime(day["datetime"])
    day["date"] = day["datetime"].dt.date
    result = manage_trade(day, pd.Timestamp("2025-01-02 07:30"), "long", 100.5, 100.0, 101.0, 0.01, 0.0, 23, 45)
    assert result["outcome"] == "ambiguous"
    assert result["ambiguous"] is True
    assert result["gross_pnl_pips"] == -50.0


def test_continuation_and_random_baselines_exist():
    direction, sl, tp = baseline_prices("long", 100.0, 10.0, 20.0, 0.01)
    assert direction == "short"
    assert sl == 100.1
    assert tp == 99.8
    assert deterministic_random_uses_strategy("EURUSD", "2025-01-02", "London_Open", "mid") in {True, False}


def test_walkforward_split_is_chronological():
    events = pd.DataFrame(
        {
            "year": [2022, 2024, 2025],
            "symbol": ["EURUSD"] * 3,
            "session_name": ["London_Open"] * 3,
            "target_mode": ["mid"] * 3,
            "confirmation_datetime": pd.to_datetime(["2022-01-01", "2024-01-01", "2025-01-01"]),
            "net_pnl_pips": [1.0, 2.0, 3.0],
            "outcome": ["win"] * 3,
            "force_closed": [False] * 3,
            "ambiguous": [False] * 3,
            "continuation_pnl_after_friction": [0.0] * 3,
            "random_pnl_after_friction": [0.0] * 3,
            "edge_vs_continuation": [1.0, 2.0, 3.0],
            "edge_vs_random": [1.0, 2.0, 3.0],
            "rr_ratio": [1.0] * 3,
        }
    )
    wf = walkforward(events)
    totals = wf[wf["symbol"] == "ALL"].set_index("segment")["total_net_pnl"].to_dict()
    assert totals["reference"] == 1.0
    assert totals["validation"] == 2.0
    assert totals["test_oos"] == 3.0


def test_output_files_are_generated(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "reports"
    data_dir.mkdir()
    for filename in ["EURUSD_M15_MT5_5Y.csv"]:
        pd.DataFrame(
            [
                {"datetime": "2025-01-01 00:00", "open": 1.1000, "high": 1.1100, "low": 1.0900, "close": 1.1050},
                {"datetime": "2025-01-02 07:00", "open": 1.0950, "high": 1.1000, "low": 1.0910, "close": 1.0990},
                {"datetime": "2025-01-02 07:15", "open": 1.0990, "high": 1.1000, "low": 1.0890, "close": 1.0920},
                {"datetime": "2025-01-02 07:30", "open": 1.0920, "high": 1.1040, "low": 1.0910, "close": 1.1010},
                {"datetime": "2025-01-02 07:45", "open": 1.1010, "high": 1.1060, "low": 1.1000, "close": 1.1050},
            ]
        ).to_csv(data_dir / filename, index=False)
    cfg = Config(data_dir, output_dir, "conservative", 16, 23, 45, 0.0, 1.0, {"Tokyo_Open": 0, "London_Open": 7, "NewYork_Open": 13, "Sydney_Open": 22}, 1)
    outputs = run_research(cfg)
    assert outputs["events"].exists()
    assert outputs["summary"].exists()
