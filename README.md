# Light

`Light` это локальный ML-проект для исследования и бэктеста криптостратегий на данных Bybit.

Проект загружает исторические свечи и рыночный контекст, сохраняет данные в `SQLite`, строит признаки, обучает модель `LightGBM`, а затем прогоняет портфельный бэктест по нескольким монетам.

## Что здесь есть

- загрузка исторических данных с Bybit
- хранение свечей и фичей в `SQLite`
- мульти-таймфрейм feature engineering
- бинарная directional-модель: `short` или `long`
- сохранение метрик, модели и feature importance
- мульти-символьный бэктест с кривой капитала

## Стек

- Python
- pandas / numpy
- pandas-ta
- LightGBM
- scikit-learn
- SQLite
- matplotlib

## Структура проекта

```text
light/
├─ config.py          # Основные настройки проекта
├─ etl.py             # Загрузка данных, расчёт фичей, разметка
├─ train.py           # Обучение и валидация модели
├─ bt.py              # Бэктест портфеля
├─ requirements.txt   # Зависимости Python
├─ models/            # Сохранённые модели и метрики
└─ market_data.db     # Локальная база SQLite
```

## Быстрый старт

### 1. Создать и активировать виртуальное окружение

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Установить зависимости

```powershell
pip install -r requirements.txt
```

### 3. Загрузить данные и построить фичи

```powershell
python etl.py
```

### 4. Обучить модель

```powershell
python train.py
```

### 5. Запустить бэктест

```powershell
python bt.py
```

### LSTM experiment

Отдельная LSTM-ветка не заменяет текущий LightGBM pipeline. Она использует существующие feature tables,
event filter и target, обучает sequence-модель по walk-forward folds, сохраняет OOS probabilities и может
сразу прогнать их через текущий backtest engine.

```powershell
python lstm/train_lstm_walk_forward.py --split-mode monthly --monthly-train-months 6 --monthly-test-months 1 --purge-gap 12
```

Rolling-window вариант обучает каждый monthly fold только на последних `--monthly-train-months` месяцах:

```powershell
python bt_walk_forward.py --split-mode monthly --monthly-window-mode rolling --monthly-train-months 12 --monthly-test-months 1 --purge-gap 12
```

Быстрый smoke run без бектеста:

```powershell
python lstm/train_lstm_walk_forward.py --split-mode tscv --n-splits 2 --sequence-length 24 --epochs 2 --skip-backtest
```

Быстрый monthly experiment на первых 3 фолдах:

```powershell
python lstm/train_lstm_walk_forward.py --split-mode monthly --monthly-train-months 6 --monthly-test-months 1 --purge-gap 12 --sequence-length 24 --epochs 8 --hidden-size 32 --dropout 0.4 --lr 0.0003 --weight-decay 0.001 --batch-size 512 --patience 3 --max-folds 3 --skip-backtest
```

## Как работает пайплайн

### `etl.py`

Скрипт создаёт и обновляет локальную базу:

- загружает свечи с Bybit
- подтягивает рыночный контекст: mark/index/premium, open interest, funding и другие данные
- считает признаки на нескольких таймфреймах
- формирует таблицы `*_features` для обучения

### `train.py`

Скрипт читает подготовленные таблицы из базы и:

- собирает датасет по выбранным символам
- делает хронологический train/validation split
- обучает `LightGBM`
- считает метрики на валидации
- сохраняет артефакты в папку `models/`

Сохраняемые артефакты:

- `models/lightgbm_target.joblib`
- `models/lightgbm_target_metrics.json`
- `models/lightgbm_target_features.json`
- `models/lightgbm_target_feature_importance.csv`

### `bt.py`

Скрипт использует сохранённую модель и метаданные фичей, чтобы:

- подготовить признаки для инференса
- получить long/short сигналы
- симулировать открытие и закрытие позиций
- посчитать PnL, drawdown, Sharpe, Profit Factor и другие метрики
- сохранить график `backtest_charts/equity_curve.png`

## Основные настройки

Все важные параметры находятся в `config.py`.

Ключевые настройки:

- `SYMBOLS` - список монет для ETL, обучения и бэктеста
- `TIMEFRAME` - основной рабочий таймфрейм
- `HTF_TIMEFRAME` - старший таймфрейм для multi-timeframe фичей
- `START_DATE` - дата начала загрузки истории
- `DB_PATH` - путь к базе `SQLite`
- `TP_PCT` и `SL_PCT` - параметры разметки target
- `DIRECTIONAL_PROBA_THRESHOLD` - минимальная уверенность модели для входа
- `MIN_SIGNAL_GAP` - минимальный разрыв между вероятностями long и short
- `RISK_PER_TRADE` - риск на одну сделку в бэктесте
- `BACKTEST_MAX_OPEN_POSITIONS` - максимум одновременно открытых позиций

## Полезно знать

- База данных может быстро разрастаться, потому что все свечи и фичи хранятся локально.
- Перед запуском `train.py` нужно сначала прогнать `etl.py`, чтобы появились таблицы `*_features`.
- `bt.py` по умолчанию ждёт модель и метафайлы в папке `models/`.
- Проект ориентирован на ресерч и бэктест, а не на live-trading исполнение.

## Повторная установка зависимостей

```powershell
.\venv\Scripts\activate
pip install -r requirements.txt
```
!!! WARNING - train data was used to 28-03-26

!!! WARNING - train data was used to 28-03-26


python lstm/train_lstm_walk_forward.py --split-mode monthly --monthly-train-months 6 --monthly-test-months 1 --purge-gap 12 --sequence-length 24 --epochs 20 --hidden-size 32 --num-layers 1 --dropout 0.4 --lr 0.0003 --weight-decay 0.001 --batch-size 256


Монеты, на которых эта модель обучалась, из models/lstm_target_features.json:

BTC/USDT
BNB/USDT
ETH/USDT
SOL/USDT
XRP/USDT
ADA/USDT
AVAX/USDT
DOT/USDT
DOGE/USDT
Фичи этой модели:

atr_ratio_1h
bollinger_bandwidth_atr_1h_20
bollinger_percent_b_1h_20
distance_to_resistance_1h
distance_to_rolling_high_4h
distance_to_support_1h
ema_fast_slow
ema_slope_4h
funding_rate_8h
linear_regression_slope_atr_1h_12
premium_index_change_24h
price_position_1h
price_position_4h
realized_vol_4h_returns_20
return_4h_14
volatility_regime_change_1h
volatility_regime_stability
zscore_vs_vwap_4h
Дополнительно:

sequence_length у этой модели: 24
альтернативная свечная модель из models/lstm_candles_target.pt здесь не участвует, потому что для неё были бы нужны models/lstm_candles_walk_forward_oos_predictions.csv и models/lstm_candles_target_features.json.