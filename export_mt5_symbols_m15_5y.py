from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "MetaTrader5 package is not installed. Install it first: pip install MetaTrader5"
    ) from exc


SYMBOLS_CONFIG_PATH = Path("symbols_config.json")
OUTPUT_DIR = Path("data")
TIMEFRAME = mt5.TIMEFRAME_M15
YEARS_BACK = 5


@dataclass
class ExportResult:
    symbol: str
    status: str
    rows_exported: int
    first_datetime: Optional[str]
    last_datetime: Optional[str]
    output_path: Optional[str]


def load_symbols(config_path: Path) -> List[str]:
    if not config_path.exists():
        raise FileNotFoundError(f"Missing symbols config: {config_path}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    symbols = payload.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        raise ValueError("symbols_config.json must include a non-empty 'symbols' list")

    clean = [str(s).strip() for s in symbols if str(s).strip()]
    if not clean:
        raise ValueError("No valid symbols found in symbols_config.json")
    return clean


def broker_symbol_hint(raw_symbol: str) -> str:
    available = mt5.symbols_get() or []
    names = [s.name for s in available]
    partial = [name for name in names if raw_symbol in name]

    if partial:
        preview = ", ".join(partial[:10])
        suffix = "" if len(partial) <= 10 else f" ... (+{len(partial) - 10} more)"
        return (
            f"Broker does not expose exact symbol '{raw_symbol}'. "
            f"Possible prefixed/suffixed variants: {preview}{suffix}."
        )

    return (
        f"Broker does not expose symbol '{raw_symbol}' and no close prefixed/suffixed match was found."
    )


def export_symbol(symbol: str, utc_from: datetime, utc_to: datetime) -> ExportResult:
    if not mt5.symbol_select(symbol, True):
        return ExportResult(
            symbol=symbol,
            status=f"FAIL: {broker_symbol_hint(symbol)}",
            rows_exported=0,
            first_datetime=None,
            last_datetime=None,
            output_path=None,
        )

    rates = mt5.copy_rates_range(symbol, TIMEFRAME, utc_from, utc_to)
    if rates is None:
        code, msg = mt5.last_error()
        return ExportResult(
            symbol=symbol,
            status=f"FAIL: MT5 copy_rates_range error {code}: {msg}",
            rows_exported=0,
            first_datetime=None,
            last_datetime=None,
            output_path=None,
        )

    if len(rates) == 0:
        return ExportResult(
            symbol=symbol,
            status="FAIL: MT5 returned 0 rows",
            rows_exported=0,
            first_datetime=None,
            last_datetime=None,
            output_path=None,
        )

    df = pd.DataFrame(rates)
    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)

    export_cols = ["datetime", "open", "high", "low", "close"]
    if "tick_volume" in df.columns:
        # Kept as an optional extra column only when provided by MT5.
        df = df.rename(columns={"tick_volume": "volume"})
        export_cols.append("volume")

    out_df = df[export_cols].sort_values("datetime").reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{symbol}_M15_MT5_5Y.csv"
    out_df.to_csv(output_path, index=False)

    return ExportResult(
        symbol=symbol,
        status="OK",
        rows_exported=len(out_df),
        first_datetime=out_df["datetime"].iloc[0].isoformat(sep=" "),
        last_datetime=out_df["datetime"].iloc[-1].isoformat(sep=" "),
        output_path=str(output_path),
    )


def print_summary(results: List[ExportResult]) -> None:
    print("\n=== Export summary ===")
    for result in results:
        print(
            " | ".join(
                [
                    f"symbol={result.symbol}",
                    f"status={result.status}",
                    f"rows_exported={result.rows_exported}",
                    f"first_datetime={result.first_datetime or '-'}",
                    f"last_datetime={result.last_datetime or '-'}",
                    f"output_path={result.output_path or '-'}",
                ]
            )
        )


def main() -> None:
    symbols = load_symbols(SYMBOLS_CONFIG_PATH)

    if not mt5.initialize():
        code, msg = mt5.last_error()
        raise SystemExit(f"Failed to initialize MetaTrader5 terminal: {code} {msg}")

    try:
        utc_to = datetime.utcnow()
        utc_from = utc_to - timedelta(days=365 * YEARS_BACK)

        print(
            f"Exporting M15 OHLC for ~{YEARS_BACK} years from {utc_from.isoformat()} to {utc_to.isoformat()}"
        )
        print(f"Symbols loaded from {SYMBOLS_CONFIG_PATH}: {', '.join(symbols)}")

        results: List[ExportResult] = []
        for symbol in symbols:
            print(f"\n[{symbol}] exporting...")
            result = export_symbol(symbol, utc_from=utc_from, utc_to=utc_to)
            results.append(result)
            print(f"[{symbol}] {result.status}; rows={result.rows_exported}")

        print_summary(results)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
