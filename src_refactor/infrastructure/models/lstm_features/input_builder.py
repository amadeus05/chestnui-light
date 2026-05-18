from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src_refactor.core.config import ExperimentConfig
from src_refactor.core.contracts import ModelInputBuilder
from src_refactor.core.types import ModelSpec
from src_refactor.core.types import LstmFeatureSequenceInput


@dataclass(frozen=True, slots=True)
class LstmFeatureInputBuilder(ModelInputBuilder):
    window_size: int = 48
    excluded_columns: tuple[str, ...] = ("timestamp", "symbol", "Target")

    def build_train_input(self, frame: pd.DataFrame, config: ExperimentConfig) -> LstmFeatureSequenceInput:
        window_size = int(config.model.metadata.get("window_size", self.window_size))
        feature_frame = self._feature_frame(frame)
        sequences, targets = build_supervised_windows_by_symbol(
            frame,
            feature_frame,
            frame["Target"].to_numpy(),
            window_size,
        )
        return LstmFeatureSequenceInput(
            sequence=sequences,
            feature_names=tuple(feature_frame.columns),
            window_size=window_size,
            targets=targets,
            metadata={"frame": frame.copy()},
        )

    def build_predict_input(self, frame: pd.DataFrame, spec: ModelSpec) -> LstmFeatureSequenceInput:
        window_size = int(spec.metadata.get("window_size", self.window_size))
        feature_frame = self._feature_frame(frame).tail(window_size)
        sequence = np.nan_to_num(feature_frame.to_numpy(dtype=np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
        if len(sequence) < window_size:
            sequence = left_pad_sequence(sequence, window_size)
        return LstmFeatureSequenceInput(
            sequence=sequence[None, :, :],
            feature_names=tuple(feature_frame.columns),
            window_size=window_size,
            metadata={"frame": frame.copy()},
        )

    def _feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        feature_frame = frame.drop(columns=[col for col in self.excluded_columns if col in frame.columns])
        return feature_frame.select_dtypes(include=["number", "bool"]).copy()


def build_supervised_windows(
    feature_frame: pd.DataFrame,
    targets: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.nan_to_num(feature_frame.to_numpy(dtype=np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    y_raw = pd.Series(targets).map({-1: 0, 1: 1}).to_numpy()
    sequences = []
    labels = []
    for end in range(window_size, len(values) + 1):
        label = y_raw[end - 1]
        if pd.isna(label):
            continue
        sequences.append(values[end - window_size : end])
        labels.append(int(label))
    if not sequences:
        return (
            np.empty((0, window_size, values.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.stack(sequences).astype(np.float32), np.asarray(labels, dtype=np.int64)


def build_supervised_windows_by_symbol(
    frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    targets: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if "symbol" not in frame.columns:
        return build_supervised_windows(feature_frame, targets, window_size)
    sequences = []
    labels = []
    for _, group in frame.assign(_target=targets).groupby("symbol", observed=True, sort=False):
        group = group.sort_values("timestamp") if "timestamp" in group.columns else group
        group_features = feature_frame.loc[group.index]
        group_sequences, group_targets = build_supervised_windows(
            group_features,
            group["_target"].to_numpy(),
            window_size,
        )
        if len(group_sequences) == 0:
            continue
        sequences.append(group_sequences)
        labels.append(group_targets)
    if not sequences:
        return (
            np.empty((0, window_size, feature_frame.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.vstack(sequences).astype(np.float32), np.concatenate(labels).astype(np.int64)


def left_pad_sequence(sequence: np.ndarray, window_size: int) -> np.ndarray:
    if len(sequence) == 0:
        return np.zeros((window_size, 0), dtype=np.float32)
    pad_rows = window_size - len(sequence)
    if pad_rows <= 0:
        return sequence
    padding = np.repeat(sequence[:1], pad_rows, axis=0)
    return np.vstack([padding, sequence]).astype(np.float32)
