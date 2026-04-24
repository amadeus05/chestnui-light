import config as cfg
import importlib.metadata
import pandas as pd
import platform
import sys

def get_end_date_cutoff():
    end_date = getattr(cfg, "END_DATE", None)
    if not end_date:
        return None
    return pd.to_datetime(end_date, errors="coerce")


def build_experiment_snapshot() -> dict:
    def package_version(name: str):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "experiment": str(getattr(cfg, "ACTIVE_EXPERIMENT", "default")),
        "labeling_profile": str(getattr(cfg, "LABELING_PROFILE", "default")),
        "training_profile": str(getattr(cfg, "TRAINING_PROFILE", "default")),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "lightgbm": package_version("lightgbm"),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scikit_learn": package_version("scikit-learn"),
        },
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
            "lgbm_class_weight": getattr(cfg, "LGBM_CLASS_WEIGHT", None),
            "sample_weight_half_life_days": float(getattr(cfg, "SAMPLE_WEIGHT_HALF_LIFE_DAYS", 90.0)),
            "regime_aware_weighting": bool(getattr(cfg, "REGIME_AWARE_WEIGHTING", True)),
        },
    }
