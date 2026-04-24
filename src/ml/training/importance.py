import pandas as pd

def build_fold_importance_frame(model, feature_columns, fold_idx):
    return pd.DataFrame(
        {
            "feature": feature_columns,
            f"fold_{fold_idx}_gain": model.booster_.feature_importance(importance_type="gain"),
            f"fold_{fold_idx}_split": model.booster_.feature_importance(importance_type="split"),
        }
    )


def aggregate_fold_importance(fold_importance_frames, feature_columns):
    importance = pd.DataFrame({"feature": feature_columns})
    if not fold_importance_frames:
        return importance

    for frame in fold_importance_frames:
        importance = importance.merge(frame, on="feature", how="left")

    gain_columns = [column for column in importance.columns if column.endswith("_gain")]
    split_columns = [column for column in importance.columns if column.endswith("_split")]
    importance[gain_columns + split_columns] = importance[gain_columns + split_columns].fillna(0.0)

    gain_ranks = importance[gain_columns].rank(axis=0, ascending=False, method="min")
    importance["mean_gain_by_fold"] = importance[gain_columns].mean(axis=1)
    importance["std_gain_by_fold"] = importance[gain_columns].std(axis=1).fillna(0.0)
    importance["mean_split_by_fold"] = importance[split_columns].mean(axis=1)
    importance["top_10_fold_count"] = (gain_ranks <= 10).sum(axis=1).astype(int)
    importance["top_20_fold_count"] = (gain_ranks <= 20).sum(axis=1).astype(int)
    importance["nonzero_gain_fold_count"] = (importance[gain_columns] > 0).sum(axis=1).astype(int)

    ordered_columns = [
        "feature",
        "mean_gain_by_fold",
        "std_gain_by_fold",
        "mean_split_by_fold",
        "top_10_fold_count",
        "top_20_fold_count",
        "nonzero_gain_fold_count",
        *gain_columns,
        *split_columns,
    ]
    return importance[ordered_columns].sort_values(
        ["mean_gain_by_fold", "top_10_fold_count"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_fold_importance_summary(fold_importance, limit=20):
    if fold_importance is None or fold_importance.empty:
        return []

    summary_columns = [
        "feature",
        "mean_gain_by_fold",
        "std_gain_by_fold",
        "top_10_fold_count",
        "top_20_fold_count",
        "nonzero_gain_fold_count",
    ]
    return fold_importance.head(limit)[summary_columns].to_dict(orient="records")
