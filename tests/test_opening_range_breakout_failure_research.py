import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from run_opening_range_breakout_failure_research import (  # noqa: E402
    Config,
    add_baselines,
    build_opening_range,
    continuation_events_for_breakout,
    deterministic_random_uses_strategy,
    failure_events_for_breakout,
    find_session_open,
    manage_trade,
    run_research,
    walkforward,
)


def _day(rows):
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    return df


def test_opening_range_construction_works():
    df = _day(
        [
            {"datetime": "2025-01-02 07:00", "open": 1.1000, "high": 1.1010, "low": 1.0990, "close": 1.1005},
            {"datetime": "2025-01-02 07:15", "open": 1.1005, "high": 1.1020, "low": 1.0980, "close": 1.1015},
            {"datetime": "2025-01-02 07:30", "open": 1.1015, "high": 1.1030, "low": 1.1010, "close": 1.1025},
        ]
    )
    box = build_opening_range(df, 0, 2, 0.0001)
    assert box["OR_high"] == 1.1020
    assert box["OR_low"] == 1.0980
    assert round(box["OR_mid"], 5) == 1.1000
    assert round(box["OR_range_pips"], 1) == 40.0


def test_session_open_mapping_works():
    df = _day(
        [
            {"datetime": "2025-01-02 06:45", "open": 1.1, "high": 1.1, "low": 1.1, "close": 1.1},
            {"datetime": "2025-01-02 07:15", "open": 1.1, "high": 1.1, "low": 1.1, "close": 1.1},
        ]
    )
    idx, dt = find_session_open(df, 7)
    assert idx == 1
    assert dt == pd.Timestamp("2025-01-02 07:15")


def test_breakout_continuation_event_works():
    df = _day(
        [
            {"datetime": "2025-01-02 07:00", "open": 1.1000, "high": 1.1010, "low": 1.0990, "close": 1.1000},
            {"datetime": "2025-01-02 07:15", "open": 1.1000, "high": 1.1010, "low": 1.0990, "close": 1.1005},
            {"datetime": "2025-01-02 07:30", "open": 1.1005, "high": 1.1040, "low": 1.1000, "close": 1.1025},
            {"datetime": "2025-01-02 07:45", "open": 1.1025, "high": 1.1050, "low": 1.1020, "close": 1.1040},
        ]
    )
    box = build_opening_range(df, 0, 2, 0.0001)
    base = {
        "symbol": "EURUSD",
        "date": "2025-01-02",
        "year": 2025,
        "month": "2025-01",
        "day_of_week": "Thursday",
        "session_name": "London_Open",
        "session_open_hour": 7,
        "session_open_datetime": pd.Timestamp("2025-01-02 07:00"),
        "OR_high": box["OR_high"],
        "OR_low": box["OR_low"],
        "OR_mid": box["OR_mid"],
        "OR_range_pips": box["OR_range_pips"],
    }
    events = continuation_events_for_breakout(df, 2, "upside", base, box, 2, 8, 0.0001, 0.0)
    assert events
    assert {event["strategy_mode"] for event in events} == {"breakout_continuation"}
    assert any(event["direction"] == "LONG" for event in events)


def test_breakout_failure_reversion_event_works():
    df = _day(
        [
            {"datetime": "2025-01-02 07:00", "open": 1.1000, "high": 1.1010, "low": 1.0990, "close": 1.1000},
            {"datetime": "2025-01-02 07:15", "open": 1.1000, "high": 1.1010, "low": 1.0990, "close": 1.1005},
            {"datetime": "2025-01-02 07:30", "open": 1.1005, "high": 1.1014, "low": 1.1000, "close": 1.1012},
            {"datetime": "2025-01-02 07:45", "open": 1.1012, "high": 1.1013, "low": 1.0995, "close": 1.1008},
            {"datetime": "2025-01-02 08:00", "open": 1.1008, "high": 1.1010, "low": 1.0980, "close": 1.0990},
        ]
    )
    box = build_opening_range(df, 0, 2, 0.0001)
    base = {
        "symbol": "EURUSD",
        "date": "2025-01-02",
        "year": 2025,
        "month": "2025-01",
        "day_of_week": "Thursday",
        "session_name": "London_Open",
        "session_open_hour": 7,
        "session_open_datetime": pd.Timestamp("2025-01-02 07:00"),
        "OR_high": box["OR_high"],
        "OR_low": box["OR_low"],
        "OR_mid": box["OR_mid"],
        "OR_range_pips": box["OR_range_pips"],
    }
    events = failure_events_for_breakout(df, 2, "upside", base, box, 2, 8, 0.0001, 0.0)
    assert events
    assert {event["strategy_mode"] for event in events} == {"breakout_failure_reversion"}
    assert any(event["direction"] == "SHORT" for event in events)


def test_opposite_and_random_baselines_exist():
    df = _day(
        [
            {"datetime": "2025-01-02 07:30", "open": 1.1000, "high": 1.1000, "low": 1.1000, "close": 1.1000},
            {"datetime": "2025-01-02 07:45", "open": 1.1000, "high": 1.1030, "low": 1.0990, "close": 1.1020},
        ]
    )
    event = {
        "symbol": "EURUSD",
        "date": "2025-01-02",
        "session_name": "London_Open",
        "strategy_mode": "breakout_continuation",
        "direction": "LONG",
        "entry_price": 1.1000,
        "risk_pips": 10.0,
        "reward_pips": 20.0,
        "friction_pips": 0.0,
        "net_pnl_pips": 20.0,
    }
    enriched = add_baselines(event, df, 0, 0.0001, 1, "params")
    assert "opposite_pnl_after_friction" in enriched
    assert "random_pnl_after_friction" in enriched
    assert deterministic_random_uses_strategy("EURUSD", "2025-01-02", "London_Open", "params") in {True, False}


def test_ambiguous_same_candle_tp_sl_treated_as_sl():
    df = _day(
        [
            {"datetime": "2025-01-02 07:30", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
            {"datetime": "2025-01-02 07:45", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
        ]
    )
    result = manage_trade(df, 0, "LONG", 100.0, 99.5, 100.5, 0.01, 0.0, 1)
    assert result["outcome"] == "ambiguous"
    assert result["ambiguous"] is True
    assert result["gross_pnl_pips"] == -50.0


def test_walkforward_split_is_chronological():
    events = pd.DataFrame(
        {
            "year": [2022, 2024, 2025],
            "opening_range_bars": [2, 2, 2],
            "monitor_bars": [8, 8, 8],
            "failure_confirm_bars": [0, 0, 0],
            "stop_mode": ["OR_mid", "OR_mid", "OR_mid"],
            "target_mode": ["fixed_pips", "fixed_pips", "fixed_pips"],
            "target_pips": [10.0, 10.0, 10.0],
            "buffer_pips": [0.0, 0.0, 0.0],
            "trade_horizon_bars": [8, 8, 8],
            "entry_datetime": pd.to_datetime(["2022-01-01", "2024-01-01", "2025-01-01"]),
            "net_pnl_pips": [1.0, 2.0, 3.0],
            "outcome": ["win", "win", "win"],
            "ambiguous": [False, False, False],
            "opposite_pnl_after_friction": [0.0, 0.0, 0.0],
            "random_pnl_after_friction": [0.0, 0.0, 0.0],
            "edge_vs_opposite": [1.0, 2.0, 3.0],
            "edge_vs_random": [1.0, 2.0, 3.0],
            "rr_ratio": [1.0, 1.0, 1.0],
        }
    )
    wf = walkforward(events)
    totals = wf.set_index("segment")["total_net_pnl"].to_dict()
    assert totals["reference_2022_2023"] == 1.0
    assert totals["validation_2024"] == 2.0
    assert totals["test_oos_2025_2026"] == 3.0


def test_output_files_are_generated(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "reports"
    data_dir.mkdir()
    pd.DataFrame(
        [
            {"datetime": "2025-01-02 07:00", "open": 1.1000, "high": 1.1010, "low": 1.0990, "close": 1.1000},
            {"datetime": "2025-01-02 07:15", "open": 1.1000, "high": 1.1010, "low": 1.0990, "close": 1.1005},
            {"datetime": "2025-01-02 07:30", "open": 1.1005, "high": 1.1040, "low": 1.1000, "close": 1.1025},
            {"datetime": "2025-01-02 07:45", "open": 1.1025, "high": 1.1050, "low": 1.1020, "close": 1.1040},
            {"datetime": "2025-01-02 08:00", "open": 1.1040, "high": 1.1060, "low": 1.1030, "close": 1.1050},
        ]
    ).to_csv(data_dir / "EURUSD_M15_MT5_5Y.csv", index=False)
    outputs = run_research(Config(data_dir, output_dir, "conservative", 1))
    expected = {
        "opening_range_events.csv",
        "opening_range_by_symbol_session_strategy_params.csv",
        "opening_range_by_symbol_session_strategy.csv",
        "opening_range_by_symbol.csv",
        "opening_range_by_session.csv",
        "opening_range_by_strategy_mode.csv",
        "opening_range_by_year.csv",
        "opening_range_by_month.csv",
        "opening_range_walkforward.csv",
        "opening_range_verdict.csv",
        "opening_range_summary.md",
    }
    assert expected == {path.name for path in outputs.values()}
    assert all(path.exists() for path in outputs.values())
