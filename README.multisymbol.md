# Multi-Symbol Volatility Research (Parallel, No Strategy Changes)

This repository now includes a minimal batch runner to execute the *existing* single-symbol research flow across a fixed symbol list while keeping strategy logic unchanged.

## Symbols
Configured in `symbols_config.json`:
- EURUSD (baseline)
- GBPUSD
- USDJPY
- EURJPY
- GBPJPY
- AUDJPY
- GBPAUD
- GBPNZD

## Run
```bash
python scripts/run_multisymbol_batch.py \
  --single-symbol-cmd "python YOUR_EXISTING_SINGLE_SYMBOL_SCRIPT.py --symbol {symbol}" \
  --metrics-template "results/{symbol}/metrics.json" \
  --report reports/multi_symbol_comparison.md
```

Expected per-symbol metrics JSON fields:
- `oos_expectancy_after_costs`
- `profit_factor`
- `max_drawdown`
- `num_trades`
- `parameter_stability`
- `survives_2x_cost_stress`

The report ranks symbols by rank-sum (lower is better) over those metrics.
