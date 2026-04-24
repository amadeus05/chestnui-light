import lightgbm as lgb
import numpy as np
import pandas as pd
import config as cfg

from .constants import SYMBOL_COLUMN, TIMESTAMP_COLUMN
from .features import resolve_internal_eval_plan

def compute_sample_weights(
    frame_or_timestamps,
    half_life_days: float | None = None,
    regime_aware: bool = True,
) -> np.ndarray:
    """
    Вычисляет веса сэмплов с экспоненциальным затуханием.

    Параметры:
    - half_life_days: период полураспада в днях (default: из конфига или 90)
    - regime_aware: если True, добавляет буст для самых свежих данных
    """
    # Получаем half-life из конфига или используем дефолт 90 дней (было 365)
    if half_life_days is None:
        half_life_days = float(getattr(cfg, "SAMPLE_WEIGHT_HALF_LIFE_DAYS", 90.0))

    if isinstance(frame_or_timestamps, pd.DataFrame):
        frame = frame_or_timestamps
        timestamps = frame[TIMESTAMP_COLUMN]
    else:
        frame = None
        timestamps = frame_or_timestamps

    ts = pd.to_datetime(timestamps)
    days_ago = (ts.max() - ts).dt.total_seconds() / 86400.0
    decay = np.log(2) / half_life_days
    weights = np.exp(-decay * days_ago.values)

    # Regime-aware буст: самые свежие 30 дней получают дополнительный вес
    if regime_aware and getattr(cfg, "REGIME_AWARE_WEIGHTING", True):
        recent_days = float(getattr(cfg, "REGIME_RECENT_DAYS_BOOST", 30.0))
        boost_factor = float(getattr(cfg, "REGIME_RECENT_BOOST_FACTOR", 2.0))

        recent_mask = days_ago <= recent_days
        weights = np.where(recent_mask, weights * boost_factor, weights)

        if frame is not None:
            regime_score = np.zeros(len(frame), dtype=float)
            regime_terms = 0

            if "market_breadth_ema_fast_slow_1h" in frame.columns:
                breadth_1h = pd.to_numeric(frame["market_breadth_ema_fast_slow_1h"], errors="coerce").fillna(0.5)
                regime_score += np.abs((breadth_1h.to_numpy() * 2.0) - 1.0).clip(0.0, 1.0)
                regime_terms += 1

            if "market_breadth_pos_return_4h_3" in frame.columns:
                breadth_4h = pd.to_numeric(frame["market_breadth_pos_return_4h_3"], errors="coerce").fillna(0.5)
                regime_score += np.abs((breadth_4h.to_numpy() * 2.0) - 1.0).clip(0.0, 1.0)
                regime_terms += 1

            if "ema_slope_4h" in frame.columns:
                ema_slope_4h = pd.to_numeric(frame["ema_slope_4h"], errors="coerce").fillna(0.0)
                slope_scale = float(getattr(cfg, "REGIME_WEIGHT_SLOPE_SCALE_4H", 0.08))
                if slope_scale > 0:
                    regime_score += np.tanh(np.abs(ema_slope_4h.to_numpy()) / slope_scale)
                    regime_terms += 1

            if regime_terms > 0:
                regime_score /= float(regime_terms)
                regime_strength = np.clip(
                    regime_score * float(getattr(cfg, "REGIME_WEIGHT_STRENGTH", 0.18)),
                    0.0,
                    float(getattr(cfg, "REGIME_WEIGHT_STRENGTH_CAP", 0.25)),
                )
                weights = weights * (1.0 + regime_strength)

    min_weight = float(getattr(cfg, "SAMPLE_WEIGHT_MIN", 0.8))
    max_weight = float(getattr(cfg, "SAMPLE_WEIGHT_MAX", 1.35))
    if max_weight < min_weight:
        max_weight = min_weight

    return np.clip(weights, min_weight, max_weight)


def build_model(seed, n_estimators=800):
    """Instantiate LightGBM binary classifier."""
    class_weight = getattr(cfg, "LGBM_CLASS_WEIGHT", None)
    if isinstance(class_weight, str) and class_weight.lower() in {"", "none", "null"}:
        class_weight = None
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.005,
        num_leaves=15,
        min_child_samples=150,
        max_depth=5,
        subsample=0.6,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=3.0,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        min_split_gain=0.01,
        subsample_freq=1,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation — accepts raw vectors, not a dataframe
# ═══════════════════════════════════════════════════════════════════════════

def fit_model_with_internal_eval(
    x_train,
    y_train,
    w_train,
    feature_columns,
    seed,
    best_iterations_so_far=None,
):
    model = build_model(seed=seed)
    eval_plan = resolve_internal_eval_plan(y_train.reset_index(drop=True))
    internal_eval_size = int(eval_plan["eval_size"])

    x_fit = x_train.iloc[:-internal_eval_size]
    y_fit = y_train.iloc[:-internal_eval_size]
    w_fit = w_train[:-internal_eval_size]
    x_eval = x_train.iloc[-internal_eval_size:]
    y_eval = y_train.iloc[-internal_eval_size:]

    model.fit(
        x_fit,
        y_fit,
        sample_weight=w_fit,
        eval_set=[(x_eval, y_eval)],
        eval_metric="binary_logloss",
        categorical_feature=[SYMBOL_COLUMN] if SYMBOL_COLUMN in feature_columns else "auto",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=int(getattr(cfg, "EARLY_STOPPING_ROUNDS", 200)),
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )

    best_iter = int(model.best_iteration_ or model.n_estimators_)
    unstable_min_best_iter = int(getattr(cfg, "UNSTABLE_FOLD_MIN_BEST_ITER", 25))
    fallback_used = False
    fallback_n_estimators = None

    if best_iter < unstable_min_best_iter:
        prior_median = int(np.median(best_iterations_so_far)) if best_iterations_so_far else int(
            getattr(cfg, "UNSTABLE_FOLD_FALLBACK_DEFAULT_ESTIMATORS", 250)
        )
        fallback_n_estimators = max(
            int(getattr(cfg, "UNSTABLE_FOLD_FALLBACK_MIN_ESTIMATORS", 150)),
            prior_median,
        )
        model = build_model(seed=seed, n_estimators=fallback_n_estimators)
        model.fit(
            x_fit,
            y_fit,
            sample_weight=w_fit,
            categorical_feature=[SYMBOL_COLUMN] if SYMBOL_COLUMN in feature_columns else "auto",
        )
        best_iter = int(model.n_estimators_)
        fallback_used = True

    fit_metadata = {
        "internal_eval_size": internal_eval_size,
        "eval_plan": eval_plan,
        "best_iter": best_iter,
        "fallback_used": fallback_used,
        "fallback_n_estimators": fallback_n_estimators,
    }
    return model, fit_metadata
