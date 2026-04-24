import config as cfg
import pandas as pd

from src.persistence.repositories.historical_kline_repo import HistoricalKlineRepository

from .constants import logger
from .contexts import attach_funding_context, attach_open_interest_context, attach_premium_index_context


def build_candle_maps(
    repository: HistoricalKlineRepository,
    symbols_to_load: list,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    base_candle_map: dict[str, pd.DataFrame] = {}
    htf_candle_map: dict[str, pd.DataFrame] = {}

    for symbol in symbols_to_load:
        symbol_name = str(symbol)
        df = repository.load_candles(symbol, str(getattr(cfg, "TIMEFRAME", "1h")))
        htf_df = repository.load_candles(symbol, str(getattr(cfg, "HTF_TIMEFRAME", "4h")))
        funding_df = repository.load_funding_rates(symbol)
        premium_index_df = repository.load_premium_index_klines(symbol, str(getattr(cfg, "TIMEFRAME", "1h")))
        open_interest_df = repository.load_open_interest(symbol, str(getattr(cfg, "TIMEFRAME", "1h")))
        if df.empty or htf_df.empty:
            logger.warning("%s: no data in DB (main=%s, htf=%s)", symbol_name, len(df), len(htf_df))
            continue

        df = attach_funding_context(df, funding_df)
        df = attach_premium_index_context(df, premium_index_df)
        df = attach_open_interest_context(df, open_interest_df)
        logger.info(
            "%s: main=%s, htf=%s rows, funding=%s points, premium=%s points, open_interest=%s points",
            symbol_name,
            len(df),
            len(htf_df),
            len(funding_df),
            len(premium_index_df),
            len(open_interest_df),
        )
        base_candle_map[symbol_name] = df
        htf_candle_map[symbol_name] = htf_df

    return base_candle_map, htf_candle_map
