import config as cfg

from .constants import SYMBOL_COLUMN, TARGET_COLUMN
from .log import logger
from .model_lgb import build_model, compute_sample_weights

def train_production_model(dataset, feature_columns, seed, n_estimators):
    """
    Train the final deployment model on the ENTIRE dataset.

    The n_estimators is typically the median best_iteration from WFV,
    so we do NOT use early stopping here — every row is training data,
    and we have no hold-out to compute an eval metric on.
    """
    logger.info(
        "Training production model on 100%% of data (%s rows) with n_estimators=%s",
        len(dataset), n_estimators,
    )
    model = build_model(seed=seed, n_estimators=n_estimators)
    w_prod = compute_sample_weights(dataset)
    model.fit(
        dataset[feature_columns],
        dataset[TARGET_COLUMN],
        sample_weight=w_prod,
        categorical_feature=[SYMBOL_COLUMN] if SYMBOL_COLUMN in feature_columns else "auto",
    )
    return model
