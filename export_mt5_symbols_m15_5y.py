from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
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
TARGET_BARS = 100000
CHUNK_SIZES = [10000, 5000, 2000, 1000, 500, 100]


@dataclass
class ExportResult:
    symbol: str
    status: str
    rows_exported: int
    first_datetime: Optional[str]
    last_datetime: Optional[str]
    output_path: Optional[str]
    chunk_size_used: Optional[int]


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


def _fetch_chunked_rates(symbol: str) -> tuple[Optional[List], Optional[int], Optional[tuple[int, str]], bool]:
    for chunk_size in CHUNK_SIZES:
        chunks = []
        start_pos = 0
        hard_fail = False

        while start_pos < TARGET_BARS:
            bars_to_fetch = min(chunk_size, TARGET_BARS - start_pos)
            rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, start_pos, bars_to_fetch)

            if rates is None:
                code, msg = mt5.last_error()
                if start_pos == 0:
                    hard_fail = True
                    break
                # Data collection was partially successful; stop gracefully.
                break

            rows = len(rates)
            if rows == 0:
                break

            chunks.append(rates)
            start_pos += rows

            # Safety in case MT5 responds with malformed data.
            if rows > bars_to_fetch:
                break

        if hard_fail:
            continue

        if chunks:
            return chunks, chunk_size, None, False

    code, msg = mt5.last_error()
    return None, None, (code, msg), True


def export_symbol(symbol: str) -> ExportResult:
    if not mt5.symbol_select(symbol, True):
        return ExportResult(
            symbol=symbol,
            status=f"FAIL: {broker_symbol_hint(symbol)}",
            rows_exported=0,
            first_datetime=None,
            last_datetime=None,
            output_path=None,
            chunk_size_used=None,
        )

    chunks, chunk_size_used, last_err, hard_fail = _fetch_chunked_rates(symbol)
    if hard_fail or not chunks:
        code, msg = last_err if last_err is not None else mt5.last_error()
        return ExportResult(
            symbol=symbol,
            status=f"FAIL: MT5 copy_rates_from_pos error {code}: {msg}",
            rows_exported=0,
            first_datetime=None,
            last_datetime=None,
            output_path=None,
            chunk_size_used=None,
        )

    df = pd.concat([pd.DataFrame(chunk) for chunk in chunks], ignore_index=True)
    if df.empty:
        code, msg = mt5.last_error()
        return ExportResult(
            symbol=symbol,
            status=f"FAIL: MT5 returned 0 rows; last_error {code}: {msg}",
            rows_exported=0,
            first_datetime=None,
            last_datetime=None,
            output_path=None,
            chunk_size_used=chunk_size_used,
        )

    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)

    export_cols = ["datetime", "open", "high", "low", "close"]
    if "tick_volume" in df.columns:
        # Kept as an optional extra column only when provided by MT5.
        df = df.rename(columns={"tick_volume": "volume"})
        export_cols.append("volume")

    out_df = (
        df[export_cols]
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

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
        chunk_size_used=chunk_size_used,
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
                    f"chunk_size_used={result.chunk_size_used or '-'}",
                ]
            )
        )


def main() -> None:
    symbols = load_symbols(SYMBOLS_CONFIG_PATH)

    if not mt5.initialize():
        code, msg = mt5.last_error()
        raise SystemExit(f"Failed to initialize MetaTrader5 terminal: {code} {msg}")

    try:
        now_utc = datetime.utcnow()

        print(
            f"Exporting M15 OHLC using chunked copy_rates_from_pos target={TARGET_BARS} at {now_utc.isoformat()} UTC"
        )
        print(f"Chunk fallback sizes: {CHUNK_SIZES}")
        print(f"Symbols loaded from {SYMBOLS_CONFIG_PATH}: {', '.join(symbols)}")

        results: List[ExportResult] = []
        for symbol in symbols:
            print(f"\n[{symbol}] exporting...")
            result = export_symbol(symbol)
            results.append(result)
            print(
                f"[{symbol}] {result.status}; rows={result.rows_exported}; chunk_size_used={result.chunk_size_used or '-'}"
            )

        print_summary(results)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
