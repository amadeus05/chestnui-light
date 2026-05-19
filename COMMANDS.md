# Commands

Команды ниже запускать из корня проекта:

```bash
cd ~/Desktop/light
```

В Git Bash используй `/` в путях:

```bash
source venv/Scripts/activate
```

В PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

## Install

```bash
pip install -r requirements.txt
```

Проверить PyTorch:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## ETL

Обновить свечи, фичи и target tables:

```bash
python etl.py
```

## LightGBM

Обычное обучение LightGBM с TimeSeriesSplit:

```bash
python train.py --split-mode tscv --n-splits 5 --purge-gap 12
```

Обычное обучение LightGBM с monthly walk-forward:

```bash
python train.py --split-mode monthly --monthly-train-months 6 --monthly-test-months 1 --purge-gap 12
```

Обычный backtest по сохраненной production model:

```bash
python bt.py
```

Walk-forward OOS backtest для LightGBM:

```bash
python bt_walk_forward.py --split-mode monthly --monthly-train-months 6 --monthly-test-months 1 --purge-gap 12
```

Rolling-window walk-forward OOS backtest для LightGBM:

```bash
python bt_walk_forward.py --split-mode monthly --monthly-window-mode rolling --monthly-train-months 12 --monthly-test-months 1 --purge-gap 12
```

Быстрый LightGBM WFV через TimeSeriesSplit:

```bash
python bt_walk_forward.py --split-mode tscv --n-splits 5 --purge-gap 12
```

## LSTM

Smoke test на одном символе, без backtest:

```bash
python lstm/train_lstm_walk_forward.py --symbols BTC/USDT --split-mode tscv --n-splits 2 --sequence-length 12 --epochs 1 --batch-size 256 --skip-backtest
```

Быстрый monthly experiment на первых 3 фолдах, без backtest:

```bash
python lstm/train_lstm_walk_forward.py --split-mode monthly --monthly-train-months 6 --monthly-test-months 1 --purge-gap 12 --sequence-length 24 --epochs 8 --hidden-size 32 --num-layers 1 --dropout 0.4 --lr 0.0003 --weight-decay 0.001 --batch-size 512 --patience 3 --max-folds 3 --skip-backtest
```

Средний monthly experiment на первых 8 фолдах, без backtest:

```bash
python lstm/train_lstm_walk_forward.py --split-mode monthly --monthly-train-months 6 --monthly-test-months 1 --purge-gap 12 --sequence-length 24 --epochs 10 --hidden-size 32 --num-layers 1 --dropout 0.4 --lr 0.0003 --weight-decay 0.001 --batch-size 512 --patience 3 --max-folds 8 --skip-backtest
```

Полный LSTM monthly walk-forward с backtest:

```bash
python lstm/train_lstm_walk_forward.py --split-mode monthly --monthly-train-months 6 --monthly-test-months 1 --purge-gap 12 --sequence-length 24 --epochs 10 --hidden-size 32 --num-layers 1 --dropout 0.4 --lr 0.0003 --weight-decay 0.001 --batch-size 512 --patience 3
```

Полный LSTM rolling-window monthly walk-forward с backtest:

```bash
python lstm/train_lstm_walk_forward.py --split-mode monthly --monthly-window-mode rolling --monthly-train-months 12 --monthly-test-months 1 --purge-gap 12 --sequence-length 24 --epochs 10 --hidden-size 32 --num-layers 1 --dropout 0.4 --lr 0.0003 --weight-decay 0.001 --batch-size 512 --patience 3
```

Полный LSTM monthly walk-forward без backtest:

```bash
python lstm/train_lstm_walk_forward.py --split-mode monthly --monthly-train-months 6 --monthly-test-months 1 --purge-gap 12 --sequence-length 24 --epochs 10 --hidden-size 32 --num-layers 1 --dropout 0.4 --lr 0.0003 --weight-decay 0.001 --batch-size 512 --patience 3 --skip-backtest
```

Запустить backtest без переобучения по уже сохранённым LSTM OOS predictions:

```bash
python lstm/bt_lstm_walk_forward.py
```

Обучить production LSTM для будущего scan/paper inference:

```bash
python lstm/train_lstm_production.py --sequence-length 24 --epochs 20 --hidden-size 32 --num-layers 1 --dropout 0.4 --lr 0.0003 --weight-decay 0.001 --batch-size 256 --patience 3
```

## Reports

OOS confidence report for LightGBM predictions:

```bash
python oos_edge_report.py --predictions models/lightgbm_target_oos_predictions.csv
```

Trade edge report after backtest:

```bash
python trade_edge_report.py
```

## Artifacts

LightGBM artifacts:

```text
models/lightgbm_target.joblib
models/lightgbm_target_metrics.json
models/lightgbm_target_features.json
models/lightgbm_target_oos_predictions.csv
```

LSTM artifacts:

```text
models/lstm_target.pt
models/lstm_target_metrics.json
models/lstm_target_features.json
models/lstm_walk_forward_oos_predictions.csv
```

Backtest charts:

```text
backtest_charts/equity_curve.png
backtest_charts/equity_curve_walk_forward.png
backtest_charts/equity_curve_lstm_walk_forward.png
```



python lstm/train_lstm_walk_forward.py --split-mode monthly --monthly-train-months 6 --monthly-test-months 1 --purge-gap 12 --sequence-length 24 --epochs 20 --hidden-size 32 --num-layers 1 --dropout 0.4 --lr 0.0003 --weight-decay 0.001 --batch-size 256

python lstm/train_lstm_walk_forward.py --split-mode monthly --monthly-train-months 6 --monthly-test-months 1 --purge-gap 24 --sequence-length 24 --epochs 20 --hidden-size 32 --num-layers 1 --dropout 0.4 --lr 0.0003 --weight-decay 0.001 --batch-size 256

python lstm_candles/train_lstm_candles_walk_forward.py --split-mode monthly --monthly-train-months 6 --monthly-test-months 1 --purge-gap 24 --sequence-length 24 --epochs 20 --hidden-size 32 --num-layers 1 --dropout 0.4 --lr 0.0003 --weight-decay 0.001 --batch-size 256