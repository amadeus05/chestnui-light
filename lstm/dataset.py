from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import train


@dataclass(frozen=True)
class SequenceSampleIndex:
    row_index: int
    symbol: str
    timestamp: pd.Timestamp


class SequenceStandardizer:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, sequences: np.ndarray) -> "SequenceStandardizer":
        flat = sequences.reshape(-1, sequences.shape[-1])
        self.mean_ = np.nanmean(flat, axis=0).astype(np.float32)
        self.std_ = np.nanstd(flat, axis=0).astype(np.float32)
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_).astype(np.float32)
        return self

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("SequenceStandardizer must be fitted before transform.")
        return ((sequences - self.mean_) / self.std_).astype(np.float32)

    def to_payload(self) -> dict:
        if self.mean_ is None or self.std_ is None:
            return {}
        return {
            "mean": self.mean_.astype(float).tolist(),
            "std": self.std_.astype(float).tolist(),
        }


class SequenceDataset(Dataset):
    def __init__(
        self,
        sample_frame: pd.DataFrame,
        history_by_symbol: dict[str, pd.DataFrame],
        feature_columns: list[str],
        sequence_length: int,
        sample_indices: list[int] | None = None,
        standardizer: SequenceStandardizer | None = None,
    ):
        self.sample_frame = sample_frame.reset_index(drop=True)
        self.history_by_symbol = history_by_symbol
        self.feature_columns = feature_columns
        self.sequence_length = int(sequence_length)
        self.standardizer = standardizer
        self._timestamps_by_symbol = {
            symbol: pd.to_datetime(history[train.TIMESTAMP_COLUMN]).to_numpy()
            for symbol, history in self.history_by_symbol.items()
            if history is not None and not history.empty
        }
        self._values_by_symbol = {
            symbol: history[self.feature_columns].to_numpy(dtype=np.float32, copy=True)
            for symbol, history in self.history_by_symbol.items()
            if history is not None and not history.empty
        }
        base_indices = list(range(len(self.sample_frame))) if sample_indices is None else list(sample_indices)
        self.samples = self._build_available_samples(base_indices)
        self._sequence_cache: dict[int, np.ndarray] = {}

    def _build_available_samples(self, indices: list[int]) -> list[SequenceSampleIndex]:
        samples = []
        for row_index in indices:
            row = self.sample_frame.iloc[row_index]
            symbol = str(row[train.SYMBOL_COLUMN])
            timestamp = pd.to_datetime(row[train.TIMESTAMP_COLUMN])
            timestamps = self._timestamps_by_symbol.get(symbol)
            if timestamps is None or len(timestamps) == 0:
                continue
            eligible_count = int(np.searchsorted(timestamps, np.datetime64(timestamp), side="right"))
            if eligible_count < self.sequence_length:
                continue
            samples.append(SequenceSampleIndex(row_index=row_index, symbol=symbol, timestamp=timestamp))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sequence(self, sample: SequenceSampleIndex) -> np.ndarray:
        timestamps = self._timestamps_by_symbol[sample.symbol]
        values = self._values_by_symbol[sample.symbol]
        end_index = int(np.searchsorted(timestamps, np.datetime64(sample.timestamp), side="right"))
        start_index = end_index - self.sequence_length
        window = values[start_index:end_index].copy()
        return np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

    def sequences_array(self, sample_positions: list[int] | None = None) -> np.ndarray:
        selected_samples = self.samples if sample_positions is None else [self.samples[position] for position in sample_positions]
        sequences = [self._load_sequence(sample) for sample in selected_samples]
        if not sequences:
            return np.empty((0, self.sequence_length, len(self.feature_columns)), dtype=np.float32)
        return np.stack(sequences).astype(np.float32)

    def __getitem__(self, item: int):
        sample = self.samples[item]
        sequence = self._sequence_cache.get(item)
        if sequence is None:
            sequence = self._load_sequence(sample)
            if self.standardizer is not None:
                sequence = self.standardizer.transform(sequence[None, :, :])[0]
            self._sequence_cache[item] = sequence
        target = int(self.sample_frame.iloc[sample.row_index][train.TARGET_COLUMN])
        return torch.from_numpy(sequence), torch.tensor(target, dtype=torch.long)

    def metadata_frame(self) -> pd.DataFrame:
        rows = []
        for item, sample in enumerate(self.samples):
            source_row = self.sample_frame.iloc[sample.row_index]
            rows.append(
                {
                    "dataset_index": item,
                    train.TIMESTAMP_COLUMN: source_row[train.TIMESTAMP_COLUMN],
                    train.SYMBOL_COLUMN: str(source_row[train.SYMBOL_COLUMN]),
                    train.TARGET_COLUMN: int(source_row[train.TARGET_COLUMN]),
                }
            )
        return pd.DataFrame(rows)


def build_history_by_symbol(full_frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, pd.DataFrame]:
    required = [train.TIMESTAMP_COLUMN, train.SYMBOL_COLUMN] + feature_columns
    history = full_frame.dropna(subset=[train.TIMESTAMP_COLUMN, train.SYMBOL_COLUMN]).copy()
    history.replace([np.inf, -np.inf], np.nan, inplace=True)
    by_symbol = {}
    for symbol, symbol_frame in history.groupby(train.SYMBOL_COLUMN, observed=True):
        prepared = symbol_frame[required].copy()
        prepared = prepared.sort_values(train.TIMESTAMP_COLUMN).drop_duplicates(
            subset=[train.TIMESTAMP_COLUMN],
            keep="last",
        )
        by_symbol[str(symbol)] = prepared.reset_index(drop=True)
    return by_symbol
