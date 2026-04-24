import config as cfg
import pandas as pd

def get_end_date_cutoff():
    end_date = getattr(cfg, "END_DATE", None)
    if not end_date:
        return None
    return pd.to_datetime(end_date, errors="coerce")


def build_experiment_snapshot() -> dict:
    return {
        "experiment": str(getattr(cfg, "ACTIVE_EXPERIMENT", "default")),
        "labeling_profile": str(getattr(cfg, "LABELING_PROFILE", "default")),
        "training_profile": str(getattr(cfg, "TRAINING_PROFILE", "default")),
        "labeling": {
            "horizon": int(getattr(cfg, "HORIZON", 0)),
            "tp_pct": float(getattr(cfg, "TP_PCT", 0.0)),
            "sl_pct": float(getattr(cfg, "SL_PCT", 0.0)),
            "use_dynamic_barriers": bool(getattr(cfg, "USE_DYNAMIC_BARRIERS", False)),
            "barrier_atr_multiplier": float(getattr(cfg, "BARRIER_ATR_MULTIPLIER", 0.0)),
            "barrier_rvol_multiplier": float(getattr(cfg, "BARRIER_RVOL_MULTIPLIER", 0.0)),
            "barrier_tp_to_sl_ratio": float(getattr(cfg, "BARRIER_TP_TO_SL_RATIO", 0.0)),
            "barrier_min_pct": float(getattr(cfg, "BARRIER_MIN_PCT", 0.0)),
            "barrier_max_pct": float(getattr(cfg, "BARRIER_MAX_PCT", 0.0)),
        },
        "training": {
            "disabled_feature_columns": sorted(getattr(cfg, "MANUAL_DISABLED_FEATURE_COLUMNS", [])),
            "feature_clip_enabled": bool(getattr(cfg, "ENABLE_FEATURE_CLIP", False)),
            "feature_clip_lower_q": float(getattr(cfg, "FEATURE_CLIP_LOWER_Q", 0.0)),
            "feature_clip_upper_q": float(getattr(cfg, "FEATURE_CLIP_UPPER_Q", 1.0)),
        },
    }
