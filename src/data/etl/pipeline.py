import config as cfg

from src.features import MasterFeatureBuilder
from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository

from .barriers import attach_barrier_columns
from .candle_maps import build_candle_maps
from .constants import logger
from .factory import build_labeling_snapshot, create_exchange_service
from .finalize import finalize_feature_frame
from .labeling import triple_barrier_labeling
from .time_warnings import warn_if_history_starts_late


def main() -> None:
    exchange_service = create_exchange_service()
    repository = HistoricalKlineRepository(exchange_code=exchange_service.get_exchange_code())
    repository.init_schema()
    labeling_snapshot = build_labeling_snapshot()

    logger.info(
        "Experiment=%s | labeling_profile=%s | training_profile=%s",
        labeling_snapshot["experiment"],
        labeling_snapshot["labeling_profile"],
        labeling_snapshot["training_profile"],
    )
    logger.info(
        "ETL labeling config: horizon=%s | dynamic_barriers=%s | stop[min=%.4f max=%.4f] | tp/sl=%.2f",
        labeling_snapshot["horizon"],
        labeling_snapshot["use_dynamic_barriers"],
        labeling_snapshot["barrier_min_pct"],
        labeling_snapshot["barrier_max_pct"],
        labeling_snapshot["barrier_tp_to_sl_ratio"],
    )
    logger.info(
        "Adaptive horizon: enabled=%s | min=%s | max=%s | vol_low=%.4f | vol_high=%.4f",
        labeling_snapshot["adaptive_horizon"],
        labeling_snapshot["adaptive_horizon_min"],
        labeling_snapshot["adaptive_horizon_max"],
        labeling_snapshot["adaptive_horizon_vol_low"],
        labeling_snapshot["adaptive_horizon_vol_high"],
    )

    symbols_to_load = [exchange_service.normalize_symbol(symbol) for symbol in dict.fromkeys(getattr(cfg, "SYMBOLS", []))]

    for symbol in symbols_to_load:
        symbol_name = str(symbol)
        timeframe = str(getattr(cfg, "TIMEFRAME", "1h"))
        htf_timeframe = str(getattr(cfg, "HTF_TIMEFRAME", "4h"))
        start_date = str(getattr(cfg, "START_DATE", "2023-01-01"))
        end_date = getattr(cfg, "END_DATE", None)

        logger.info("Loading %s %s from %s...", symbol_name, timeframe, start_date)
        loaded = repository.sync_candles(exchange_service, symbol, timeframe, start_date, end_date)
        logger.info("%s %s: %s new candles", symbol_name, timeframe, loaded)
        warn_if_history_starts_late(repository, symbol_name, timeframe, start_date)

        logger.info("Loading %s %s from %s...", symbol_name, htf_timeframe, start_date)
        htf_loaded = repository.sync_candles(exchange_service, symbol, htf_timeframe, start_date, end_date)
        logger.info("%s %s: %s new candles", symbol_name, htf_timeframe, htf_loaded)
        warn_if_history_starts_late(repository, symbol_name, htf_timeframe, start_date)

        logger.info("Loading %s funding from %s...", symbol_name, start_date)
        funding_loaded = repository.sync_funding_rates(exchange_service, symbol, start_date, end_date)
        logger.info("%s funding: %s new points", symbol_name, funding_loaded)

        logger.info("Loading %s premium index %s from %s...", symbol_name, timeframe, start_date)
        premium_loaded = repository.sync_premium_index_klines(exchange_service, symbol, timeframe, start_date, end_date)
        logger.info("%s premium index %s: %s new candles", symbol_name, timeframe, premium_loaded)

        logger.info("Loading %s open interest %s from %s...", symbol_name, timeframe, start_date)
        open_interest_loaded = repository.sync_open_interest(exchange_service, symbol, timeframe, start_date, end_date)
        logger.info("%s open interest %s: %s new points", symbol_name, timeframe, open_interest_loaded)

    base_candle_map, htf_candle_map = build_candle_maps(repository, symbols_to_load)
    feature_builder = MasterFeatureBuilder()
    pipeline_result = feature_builder.build(base_candle_map, htf_candle_map)
    logger.info(
        "Feature build request resolved: profile=%s | blocks=%s | features=%s",
        pipeline_result.profile_name,
        ", ".join(pipeline_result.active_blocks) or "none",
        len(pipeline_result.feature_columns),
    )

    for symbol in symbols_to_load:
        symbol_name = str(symbol)
        feature_df = pipeline_result.feature_map.get(symbol_name)
        if feature_df is None or feature_df.empty:
            logger.warning("%s: skipped, missing prepared feature inputs", symbol_name)
            continue

        feature_df = attach_barrier_columns(feature_df)
        feature_df = triple_barrier_labeling(feature_df)
        feature_df = finalize_feature_frame(feature_df, list(pipeline_result.feature_columns))
        repository.save_features(symbol, feature_df)
        logger.info("%s: saved %s rows with %s requested features", symbol_name, len(feature_df), len(pipeline_result.feature_columns))
