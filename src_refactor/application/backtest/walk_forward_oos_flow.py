from __future__ import annotations

from dataclasses import dataclass, replace

from src_refactor.application.backtest.flow import StoredPredictionBacktestFlow, StoredPredictionBacktestRequest
from src_refactor.application.backtest.runner import BacktestRunResult
from src_refactor.application.runtime_builder import RuntimeConfig, TradingMode
from src_refactor.application.training.train_wvf_oss import WvfOosRunConfig, run_wvf_oos
from src_refactor.application.training.training_runner import WalkForwardTrainingResult
from src_refactor.infrastructure.feeds import HistoricalMarketMapLoader


@dataclass(frozen=True, slots=True)
class WalkForwardOosBacktestResult:
    training: WalkForwardTrainingResult
    backtest: BacktestRunResult


@dataclass(frozen=True, slots=True)
class WalkForwardOosBacktestFlow:
    market_loader: HistoricalMarketMapLoader
    htf_timeframe: str = "4h"

    def run(
        self,
        *,
        training_config: WvfOosRunConfig,
        runtime_config: RuntimeConfig,
        close_open_positions: bool = True,
    ) -> WalkForwardOosBacktestResult:
        symbols = training_config.symbols or runtime_config.symbols
        if not symbols:
            raise ValueError("WalkForwardOosBacktestFlow requires symbols.")

        base_candle_map = self.market_loader.load_base_map(symbols, training_config.timeframe)
        htf_candle_map = self.market_loader.load_htf_map(symbols, self.htf_timeframe)
        training_result = run_wvf_oos(
            base_candle_map=base_candle_map,
            htf_candle_map=htf_candle_map,
            config=training_config,
        )

        effective_runtime_config = replace(
            runtime_config,
            mode=TradingMode.BACKTEST,
            model=training_result.config.model,
            symbols=symbols,
        )
        backtest_flow = StoredPredictionBacktestFlow.from_training_result(
            config=effective_runtime_config,
            candle_loader=self.market_loader.candle_loader,
            result=training_result,
        )
        request = StoredPredictionBacktestRequest.from_training_result(
            training_result,
            close_open_positions=close_open_positions,
        )
        return WalkForwardOosBacktestResult(
            training=training_result,
            backtest=backtest_flow.run(request),
        )
