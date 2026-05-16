# Multi-Symbol Volatility Research (Parallel, No Strategy Changes)

This repository includes a fixed-parameter batch runner that executes the existing single-symbol research flow across the symbol list from `symbols_config.json`.

## Symbols
Configured in `symbols_config.json`.

## Run
```bash
python scripts/run_multisymbol_batch.py
```

For each symbol, the runner:
1. Checks for `data/{symbol}_M15_MT5_5Y.csv`.
2. If present, runs:
   `python run_research.py --csv data/{symbol}_M15_MT5_5Y.csv --symbol {symbol} --base-timeframe M15 --spread-pips 1.2 --slippage-pips 0.2 --output-dir reports/{symbol}`
3. Writes per-symbol status into:
   `reports/multisymbol_batch_summary.json`

If data is missing for a symbol, status is `missing_data` and processing continues.
