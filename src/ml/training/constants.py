TARGET_COLUMN = "Target"
TIMESTAMP_COLUMN = "timestamp"
SYMBOL_COLUMN = "symbol"
RESERVED_COLUMNS = {
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    "barrier_stop_pct",
    "barrier_take_pct",
}
EXCLUDED_RAW_FEATURE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
}
LABEL_TO_CLASS = {-1: 0, 1: 1}
CLASS_TO_LABEL = {0: -1, 1: 1}
