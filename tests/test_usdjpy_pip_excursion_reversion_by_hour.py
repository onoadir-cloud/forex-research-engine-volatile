import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from run_usdjpy_pip_excursion_reversion_by_hour import (  # noqa: E402
    PIP_SIZE_USDJPY,
    build_event_rows,
    deterministic_random_uses_contrarian,
    find_base_events,
    friction_for_profile,
    load_ohlc,
    simulate_direction,
)


def _sample_df() -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-02 09:00:00",
                    "2025-01-02 09:15:00",
                    "2025-01-02 09:30:00",
                    "2025-01-02 09:45:00",
                    "2025-01-02 10:00:00",
                    "2025-01-02 10:15:00",
                ]
            ),
            "open": [150.00, 150.04, 150.11, 150.08, 150.00, 149.90],
            "high": [150.05, 150.09, 150.13, 150.09, 150.02, 149.92],
            "low": [149.98, 150.02, 150.07, 149.99, 149.89, 149.84],
            "close": [150.04, 150.08, 150.12, 150.00, 149.90, 149.86],
        }
    )
    raw["symbol"] = "USDJPY"
    raw["date"] = raw["datetime"].dt.date
    raw["year"] = raw["datetime"].dt.year
    raw["month"] = raw["datetime"].dt.month
    raw["day_of_week"] = raw["datetime"].dt.dayofweek
    raw["hour"] = raw["datetime"].dt.hour
    return raw


def test_load_ohlc_normalizes_sorts_and_deduplicates(tmp_path):
    csv_path = tmp_path / "ohlc.csv"
    pd.DataFrame(
        {
            "Time": ["2025-01-01 00:15", "2025-01-01 00:00", "2025-01-01 00:00"],
            "Open": [2, 1, 99],
            "High": [3, 2, 100],
            "Low": [1, 0, 98],
            "Close": [2.5, 1.5, 99.5],
        }
    ).to_csv(csv_path, index=False)

    df = load_ohlc(csv_path)

    assert list(df["open"]) == [1, 2]
    assert df["datetime"].is_monotonic_increasing
    assert list(df.columns[:5]) == ["datetime", "open", "high", "low", "close"]


def test_find_base_events_uses_first_hour_open_and_one_event_per_hour_threshold():
    df = _sample_df()

    events = find_base_events(df, [10])

    assert len(events) == 2
    first = events[0]
    assert first.hour == 9
    assert first.hour_open == 150.00
    assert first.datetime == pd.Timestamp("2025-01-02 09:30:00")
    assert first.entry_price == 150.12
    assert first.stretch_direction == "UP"


def test_simulate_direction_contrarian_short_success_after_up_stretch():
    df = _sample_df()
    event = find_base_events(df, [10])[0]

    outcome = simulate_direction(
        df,
        event.event_idx,
        event.entry_price,
        direction="SHORT",
        target_pips=10,
        adverse_pips=20,
        horizon_bars=2,
        friction_pips=1.0,
    )

    assert outcome.outcome == "success"
    assert outcome.pnl_after_friction == 9.0
    assert outcome.bars_to_outcome == 1
    assert outcome.max_favorable_pips >= 10


def test_build_event_rows_adds_baselines_and_edges_reproducibly():
    df = _sample_df()
    event = find_base_events(df, [10])[0]

    rows = build_event_rows(df, [event], friction_for_profile("low"))
    rerun_choice = deterministic_random_uses_contrarian(event)

    assert not rows.empty
    row = rows[
        (rows["reversion_target_pips"] == 10)
        & (rows["adverse_continuation_pips"] == 20)
        & (rows["future_horizon_bars"] == 4)
    ].iloc[0]
    expected_random = (
        row["contrarian_pnl_after_friction"] if rerun_choice else row["continuation_pnl_after_friction"]
    )
    assert row["random_pnl_after_friction"] == expected_random
    assert row["edge_vs_continuation"] == row["contrarian_pnl_after_friction"] - row["continuation_pnl_after_friction"]
    assert row["friction_pips"] == 2 * 1.0 + 2 * 0.3 + 0.3
    assert PIP_SIZE_USDJPY == 0.01
