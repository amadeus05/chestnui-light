import config as cfg

from src.contracts.exchange_contract import ExchangeContract
from src.exchanges.binance.binance_service import BinanceService
from src.exchanges.bybit.bybit_service import BybitService


def create_exchange_service() -> ExchangeContract:
    exchange_name = str(getattr(cfg, "ACTIVE_EXCHANGE", "bybit")).strip().lower()
    if exchange_name == "bybit":
        return BybitService()
    if exchange_name == "binance":
        return BinanceService()
    raise ValueError(f"Unsupported ACTIVE_EXCHANGE: {exchange_name}")


def build_labeling_snapshot() -> dict:
    return {
        "experiment": str(getattr(cfg, "ACTIVE_EXPERIMENT", "default")),
        "labeling_profile": str(getattr(cfg, "LABELING_PROFILE", "default")),
        "training_profile": str(getattr(cfg, "TRAINING_PROFILE", "default")),
        "horizon": int(getattr(cfg, "HORIZON", 0)),
        "use_dynamic_barriers": bool(getattr(cfg, "USE_DYNAMIC_BARRIERS", False)),
        "barrier_atr_multiplier": float(getattr(cfg, "BARRIER_ATR_MULTIPLIER", 0.0)),
        "barrier_rvol_multiplier": float(getattr(cfg, "BARRIER_RVOL_MULTIPLIER", 0.0)),
        "barrier_tp_to_sl_ratio": float(getattr(cfg, "BARRIER_TP_TO_SL_RATIO", 0.0)),
        "barrier_min_pct": float(getattr(cfg, "BARRIER_MIN_PCT", 0.0)),
        "barrier_max_pct": float(getattr(cfg, "BARRIER_MAX_PCT", 0.0)),
        "adaptive_horizon": bool(getattr(cfg, "ENABLE_ADAPTIVE_HORIZON", False)),
        "adaptive_horizon_min": int(getattr(cfg, "ADAPTIVE_HORIZON_MIN", 0)),
        "adaptive_horizon_max": int(getattr(cfg, "ADAPTIVE_HORIZON_MAX", 0)),
        "adaptive_horizon_vol_low": float(getattr(cfg, "ADAPTIVE_HORIZON_VOL_LOW", 0.0)),
        "adaptive_horizon_vol_high": float(getattr(cfg, "ADAPTIVE_HORIZON_VOL_HIGH", 0.0)),
    }
