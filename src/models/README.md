# Models Layer

This package is the new unified model layer. Legacy scripts stay untouched while
their logic is migrated behind common contracts.

## Main Concepts

- `ModelSpec` describes a model once: key, artifact name, feature source, legacy
  entrypoints, and supported modes.
- `ArtifactStore` owns artifact path conventions and JSON persistence.
- `BaseModelRunner` is the temporary runner shell. It delegates to legacy modules
  today and gives us a stable place to move training/backtest logic later.
- `model.py` is the new CLI entrypoint at the repository root.

## Current Commands

```bash
python model.py list
python model.py sync lightgbm
python model.py train lightgbm -- --split-mode monthly
python model.py wfv lstm -- --split-mode monthly --skip-backtest
python model.py backtest lstm
python model.py wfv lstm_candles -- --max-folds 3 --skip-backtest
```

Arguments after the model key are forwarded to the selected legacy runner.

`sync` copies existing flat legacy artifacts into the new per-model folder and
builds `metadata.json` using the unified `ModelMetadata` schema.
