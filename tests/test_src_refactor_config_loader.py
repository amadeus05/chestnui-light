import json
from pathlib import Path
from uuid import uuid4

from src_refactor.cli.backtest import build_parser
from src_refactor.cli.train import build_parser as build_train_parser
from src_refactor.configs import BacktestCliConfig


def test_backtest_cli_config_loads_nested_json():
    path = _write_config(
        {
            "market": {
                "db_path": "data/test.db",
                "exchange_code": "binance",
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "timeframe": "15m",
                "htf_timeframe": "1h",
            },
            "model": {
                "model_type": "lightgbm",
                "profile": "exp_a",
                "metadata": {"seed": 7},
            },
            "runtime": {
                "initial_balance": 250.0,
                "pricing": {"taker_fee": 0.001, "slippage": 0.002},
                "risk": {"risk_per_trade": 0.02, "max_open_positions": 3},
                "signals": {"directional_proba_threshold": 0.57, "allow_shorts": False},
                "trading": {"max_new_positions_per_bar": 2},
            },
            "stored_backtest": {
                "predictions_path": "models/predictions/oos.parquet",
                "model_id": "lightgbm__15m__exp_a",
            },
            "walk_forward": {
                "predictions_path": "models/predictions/wvf.parquet",
                "split_mode": "monthly_rolling",
                "n_splits": 4,
                "feature_request": {"profile": "all"},
                "labeling": {"horizon": 12},
            },
        }
    )
    try:
        config = BacktestCliConfig.from_path(path)
    finally:
        path.unlink(missing_ok=True)

    assert config.market.symbols == ("BTC/USDT", "ETH/USDT")
    assert config.market.timeframe == "15m"
    assert config.model.profile == "exp_a"
    assert config.model.metadata["seed"] == 7
    assert config.runtime.taker_fee == 0.001
    assert config.runtime.risk_per_trade == 0.02
    assert config.runtime.allow_shorts is False
    assert config.runtime.max_new_positions_per_bar == 2
    assert config.stored.predictions_path == "models/predictions/oos.parquet"
    assert config.walk_forward.split_mode == "monthly_rolling"
    assert config.walk_forward.feature_request == {"profile": "all"}


def test_backtest_cli_accepts_config_without_duplicating_required_flags():
    path = _write_config(
        {
            "market": {"symbols": ["BTC/USDT"]},
            "stored_backtest": {"predictions_path": "models/predictions/oos.parquet"},
        }
    )
    try:
        args = build_parser().parse_args(["stored", "--config", str(path)])
        config = BacktestCliConfig.from_path(args.config)
    finally:
        path.unlink(missing_ok=True)

    assert args.symbols is None
    assert args.predictions_path is None
    assert config.market.symbols == ("BTC/USDT",)
    assert config.stored.predictions_path == "models/predictions/oos.parquet"


def test_example_configs_load():
    examples = [
        "lightgbm_wvf_oos_backtest.json",
        "lightgbm_stored_backtest.json",
        "lstm_features_wvf_oos_backtest.json",
        "lightgbm_wvf_oos_train.json",
        "lstm_features_wvf_oos_train.json",
    ]

    for example_name in examples:
        config = BacktestCliConfig.from_path(Path("src_refactor/configs/examples") / example_name)
        assert config.market.symbols
        assert config.market.timeframe


def test_train_cli_accepts_config_without_duplicating_required_flags():
    path = _write_config(
        {
            "market": {"symbols": ["BTC/USDT"]},
            "walk_forward": {"predictions_path": "models/predictions/oos.parquet"},
        }
    )
    try:
        args = build_train_parser().parse_args(["wvf-oos", "--config", str(path)])
        config = BacktestCliConfig.from_path(args.config)
    finally:
        path.unlink(missing_ok=True)

    assert args.symbols is None
    assert args.predictions_path is None
    assert config.market.symbols == ("BTC/USDT",)
    assert config.walk_forward.predictions_path == "models/predictions/oos.parquet"


def _write_config(payload: dict) -> Path:
    directory = Path("src_refactor/.tmp_tests")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"config_{uuid4().hex}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
