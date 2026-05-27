import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from run_daily_box_strategy_research import DBSConfig, prepare_data, run_backtest


def _cfg(tmp_path: Path) -> DBSConfig:
    return DBSConfig("", "USDJPY", tmp_path, 0, 1, 23, 45, "conservative", False, 5, False)


def test_previous_daily_box_calculation():
    df = pd.DataFrame({"datetime": pd.to_datetime(["2024-01-01 00:00","2024-01-01 00:15","2024-01-02 00:00","2024-01-02 00:15"]),"open":[100,101,102,103],"high":[109,110,107,108],"low":[101,100,101,100],"close":[101,102,103,104]})
    p = prepare_data(df, ema_period=3)
    assert p.iloc[0]["date"].isoformat() == "2024-01-02"
    assert p.iloc[0]["box_top"] == 110
    assert p.iloc[0]["box_bottom"] == 100


def test_long_short_rr_and_one_trade_per_day(tmp_path: Path):
    rows = [
        ("2024-01-01 00:00", 104, 110, 101, 106),
        ("2024-01-01 00:15", 106, 109, 100, 103),
        ("2024-01-02 00:00", 102.0, 102.8, 101.8, 102.5),
        ("2024-01-02 00:15", 103.5, 104.0, 99.5, 100.5),
        ("2024-01-02 00:30", 100.5, 103.2, 100.4, 103.1),
        ("2024-01-02 00:45", 103.1, 111.0, 103.0, 110.2),
        ("2024-01-02 01:00", 110.2, 110.4, 99.8, 100.2),
    ]
    df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    trades = run_backtest(prepare_data(df, ema_period=3), _cfg(tmp_path))
    assert len(trades) == 1
    assert trades.iloc[0]["direction"] == "long"


def test_force_close(tmp_path: Path):
    rows = [
        ("2024-01-01 00:00", 104, 110, 101, 106),
        ("2024-01-01 00:15", 106, 109, 100, 103),
        ("2024-01-02 00:00", 102.0, 102.8, 101.8, 102.5),
        ("2024-01-02 00:15", 103.5, 104.0, 99.5, 100.5),
        ("2024-01-02 00:30", 100.5, 103.2, 100.4, 103.1),
        ("2024-01-02 00:45", 103.1, 103.3, 103.0, 103.2),
    ]
    df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    cfg = _cfg(tmp_path)
    cfg.force_close_hour = 0
    cfg.force_close_minute = 45
    trades = run_backtest(prepare_data(df, ema_period=3), cfg)
    assert len(trades) == 1
    assert trades.iloc[0]["outcome"] == "force_close"
