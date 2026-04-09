Как я бы усилил модель

Перестроил бы target.
Сильнейший апгрейд: либо 3-class модель short / flat / long, либо двухэтапная схема direction + trade/no-trade. Сейчас именно это, на мой взгляд, режет потолок сильнее всего.

Нормализовал бы barriers под волатильность.
Сейчас TP/SL фиксированные 3% / 1.5% в config.py:55. Для BTC и SUI это разные режимы сложности, что видно даже по доле directional-сэмплов. Лучше перейти на ATR/realized-vol barriers или хотя бы сделать barrier multiplier по символу.

Подключил бы symbol-awareness.
Минимальный шаг: включить USE_SYMBOL_FEATURE=True. Более сильный вариант: отдельные модели по кластерам активов или хотя бы per-symbol calibration/thresholds.

Добрал бы ортогональные фичи, а не ещё один RSI-подобный блок.
Быстрые кандидаты:

Включить и честно прогнать ENABLE_TEST_MARKET_CONTEXT_FEATURES и ENABLE_TEST_PRICE_ACTION_FEATURES из config.py:74.
Добавить реальные crowding-фичи из long_short_ratio: сейчас это поле загружается, но по факту не превращается в отдельный feature-блок в etl.py.
Расширить cross-sectional блок: rank не только по return_4h_3, а по relative strength, vol-adjusted return, volume surprise, breadth рынка.
Улучшил бы evaluation.
Сделал бы walk-forward CV вместо одного split, отдельную калибровку long/short thresholds и selection metric не по общей accuracy, а по expected trade quality / precision on taken trades.


спользовал бы уже посчитанные MFE_long/MFE_short.
Вы их аккуратно считаете в etl.py:1157, но потом специально исключаете из train-пайплайна как reserved в train.py:14. Из этого просится вторая модель: не только “куда”, но и “насколько качественный вход”. Это очень хорошо работает как фильтр и ранжирование сигналов.


 и добавил настройки в config.py:

BACKTEST_BLOCK_SAME_BAR_REENTRY = True
BACKTEST_TP_COOLDOWN_BARS = 2
BACKTEST_SL_COOLDOWN_BARS = 8
Что это дало по факту:

после SL/TP символ больше не может сразу переоткрыться в тот же бар
после SL он уходит на паузу на 8 баров
лог стал заметно чище: такие цепочки, как SEI -> SL -> снова SEI через 2 часа -> снова SL, стали гораздо реже


Что чаще всего «добавляет силы» сверх того, что уже есть
Данные не из OHLCV (если готовы тянуть API)

Funding / OI по каждому символу: для крипты это один из самых сильных сигналов перекоса позиционирования и режима.
Без этого модель видит только цену/объём свечей — часть режима остаётся слепой зоной.
Объёмно-ценовые фичи (на базе тех же свечей)
У вас есть vol_ratio, но часто помогают ещё:

OBV или CMF/MFI (денежный поток / давление покупок-продаж).
Объём как z-score к своей скользящей (всплеск ликвидности/интереса).
Корреляция доходности с объёмом по окну (растущий тренд на падающем объёме vs на растущем).
Форма свечи и микроструктура бара

Доля тела/теней, upper/lower wick ratio, серии из N подряд зелёных/красных баров.
Это слабо дублирует ваши EMA/ADX и хорошо ловит «усталость» импульса и ложные пробои у уровней.
Режим «тренд vs mean-reversion» явно
У вас есть trend_efficiency, adx_4h. Дополняют:

Hurst или автокорреляция доходностей на окне 48–96 баров.
Rolling skew/kurtosis доходностей (хвосты и асимметрия режима).
Помогает не смешивать в одной модели сценарии, где сигнал должен быть разным.
BTC / рынок на других горизонтах
Есть beta_to_btc_24h, relative_strength_vs_btc_24h. Часто усиливает:

те же метрики на 6–12 баров (6h–12h) и/или отставание/перегрев относительно 24h (разница короткого и долгого RS).
Ликвидность и проскальзывание (прокси)

Amihud illiquidity |ret|/volume, spread proxy через (high−low)/close.
Согласуется с вашими динамическими барьерами: в тонком рынке исходы другие.

=========sonet===========
🟡 Autocorrelation returns — режим trending vs reverting
python# Autocorrelation lag-1: > 0 = momentum, < 0 = mean-reversion
log_ret_1h = np.log(close / close.shift(1))

def rolling_autocorr(series, window, lag=1):
    return series.rolling(window).apply(
        lambda x: pd.Series(x).autocorr(lag=lag), raw=False
    )

autocorr_1h_24 = rolling_autocorr(log_ret_1h, window=24, lag=1)
autocorr_1h_48 = rolling_autocorr(log_ret_1h, window=48, lag=1)
Почему: когда autocorr > 0.15 — momentum-сигналы работают лучше, когда < -0.15 — mean-reversion.
Можно использовать как gate для EMA/RSI сигналов.

🟡 Z-score цены на 1H уровне (у тебя есть только на 4H VWAP)
python# VWAP 1H (24-свечное окно)
vwap_24_1h = compute_rolling_vwap(close, high, low, volume, window=24)
vwap_dist_1h = close - vwap_24_1h
vwap_std_1h  = vwap_dist_1h.rolling(24).std().replace(0, np.nan)
zscore_vs_vwap_1h = vwap_dist_1h / vwap_std_1h  # аналог zscore_vs_vwap_4h но на 1H
Это заполняет явную дырку в твоей матрице: есть price_position_1h (range-based), но нет volume-weighted anchor на 1H.

🟠 RSI-based фичи (у тебя их вообще нет)
pythonrsi_14_1h = ta.rsi(close, length=14)

# Не просто RSI, а его положение + динамика
rsi_zscore_1h = (rsi_14_1h - rsi_14_1h.rolling(48).mean()) / rsi_14_1h.rolling(48).std()
rsi_slope_1h = compute_linear_regression_slope(rsi_14_1h, 6) / atr_14  # нормированный

# RSI divergence proxy (цена vs RSI в одном направлении?)
price_higher_high = (close > close.shift(12)) & (close.shift(6) > close.shift(12))
rsi_lower_high    = (rsi_14_1h < rsi_14_1h.shift(12))
bearish_divergence_1h = (price_higher_high & rsi_lower_high).astype(float).rolling(6).

🟠 Candlestick microstructure — почти бесплатно из OHLC
python# Body ratio: насколько "решительна" свеча
body_ratio_1h = (close - open_).abs() / candle_range  # [0, 1]

# Wick asymmetry: давление покупателей/продавцов
upper_wick = high - close.where(close > open_, open_)
lower_wick = close.where(close < open_, open_) - low
wick_imbalance_1h = (lower_wick - upper_wick) / candle_range  # > 0 = bullish pressure

# Rolling версии — среднее за 6/12 свечей
body_ratio_ma_12 = body_ratio.rolling(12).mean()
wick_imbalance_ma_12 = wick_imbalance.rolling(12).mean()
Почему: wick_imbalance хорошо коррелирует с краткосрочным направлением, особенно при высоком ADX.


🟡 Volume delta proxy (вместо disabled volume_imbalance)
У тебя закомментирован volume_imbalance_1h_12 — причина не указана. Предлагаю более чистую версию:
python# Не просто знак close-open, а взвешенный по диапазону
# (Bishop's approximation of delta)
buy_vol_proxy  = volume * (close - low) / candle_range
sell_vol_proxy = volume * (high - close) / candle_range
net_delta_1h   = buy_vol_proxy - sell_vol_proxy

# Кумулятивный delta тренд
cvd_ratio_1h = safe_ratio(
    net_delta_1h.rolling(12).sum(),
    volume.rolling(12).sum()
)  # [-1, 1], лучше чем просто sign(close-open)

🔵 Взаимодействия (interaction terms) — низкозатратно
Из того что уже есть:
python# Momentum в контексте волатильности
return_1h_12_vol_adj = df["return_1h_12"] / df["realized_vol_1h"]

# Позиция в диапазоне × тренд
price_pos_x_ema = df["price_position_1h"] * df["ema_fast_slow"]

# ADX × cross_sectional_rank (сильный тренд + относительная сила)
adx_x_rank_4h = df["adx_4h"] * df["cross_sectional_rank_4h"]

# Volatility regime × trend direction
vol_regime_x_return = df["volatility_regime_change_1h"] * df["return_4h_3"]


=============sonet opus==============
#	Фича	Блок	Суть
1	cvd_ratio_12	CVD	Чистый orderflow [-1,1], Bishop's approximation
2	cvd_price_divergence_12	CVD	Расхождение slope CVD vs slope цены
3	return_1h_24_vol_adj	Vol-adj	24h return нормированный на realized vol
4	vol_ratio_zscore	Z-score	Аномальность текущего vol_ratio относительно скользящей
5	price_position_x_ema_fast_slow	Interaction	Позиция в диапазоне × моментум
6	adx_4h_x_cross_sectional_rank_4h	Interaction	Сила тренда × относительная сила
7	return_4h_3_x_market_breadth	Interaction	Доходность × ширина рынка




vol_ratio_zscore




trend
regime
entry_location
volume/flow
market-context / cross-sectional
funding/OI/sentiment



Все сделки, которые не достигли ни TP, ни SL до вертикального барьера, автоматически получают Target = 0. Это видно в etl.py и etl.py: simulate_trade_outcome() при истечении HORIZON возвращает 0.0, а triple_barrier_labeling() превращает всё <= 0 в ноль.
Это не классический triple-barrier. Обычно на vertical barrier позицию закрывают по цене в конце горизонта и уже по итоговому PnL ставят label. У тебя сейчас любая “не добежала до TP, но осталась в плюсе” всё равно считается негативом. Это системно зажимает позитивный класс.