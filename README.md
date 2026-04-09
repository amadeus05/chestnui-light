# Light

`Light` это локальный ML-проект для исследования и бэктеста криптостратегий на данных Bybit.

Проект загружает исторические свечи и рыночный контекст, сохраняет данные в `SQLite`, строит признаки, обучает модель `LightGBM`, а затем прогоняет портфельный бэктест по нескольким монетам.

## Что здесь есть

- загрузка исторических данных с Bybit
- хранение свечей и фичей в `SQLite`
- мульти-таймфрейм feature engineering
- бинарная long-only модель: `no_long` или `long`
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
└─ data/
   └─ market_data.db  # Локальная база SQLite
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

- `models/lightgbm_long_only.joblib`
- `models/lightgbm_long_only_metrics.json`
- `models/lightgbm_long_only_features.json`
- `models/lightgbm_long_only_feature_importance.csv`

### `bt.py`

Скрипт использует сохранённую модель и метаданные фичей, чтобы:

- подготовить признаки для инференса
- получить long-only сигналы
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
- `DB_PATH` - путь к базе `SQLite` (по умолчанию `data/market_data.db`)
- `TP_PCT` и `SL_PCT` - параметры разметки target
- `LONG_PROBA_THRESHOLD` - минимальная вероятность long для входа
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

# Обычный режим (как раньше)
python train.py
# Walk-Forward с expanding window (по умолчанию)
python train.py --walk-forward --wf-splits 7
# Rolling window с фиксированным размером обучения
python train.py --walk-forward --wf-strategy rolling --wf-train-window 5000 --wf-splits 5
# С защитой от лика
python train.py --walk-forward --wf-purge-gap 12 --wf-embargo-pct 0.05






entry_features
interaction_features

python train.py --quick --quick-folds 6 --quick-n-estimators 600

--quick
--quick-folds
--quick-n-estimators
--quick-log-eval-period
--wf-start-fold
--wf-max-folds
