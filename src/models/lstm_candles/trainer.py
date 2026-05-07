from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

import config as cfg
from lstm_candles import config_lstm_candles as candle_cfg
from src.models.lstm.dataset import SequenceDataset, SequenceStandardizer
from src.models.lstm.network import LSTMClassifier
from src.models.lstm_candles.data_builder import LstmCandlesDataBuilder, LstmCandlesDataRequest


@dataclass(frozen=True, slots=True)
class LstmCandlesTrainRequest:
    db_path: str = cfg.DB_PATH
    symbols: list[str] = field(default_factory=lambda: list(cfg.SYMBOLS))
    seed: int = 42
    sequence_length: int = candle_cfg.SEQUENCE_LENGTH
    batch_size: int = candle_cfg.BATCH_SIZE
    epochs: int = candle_cfg.EPOCHS
    hidden_size: int = candle_cfg.HIDDEN_SIZE
    num_layers: int = candle_cfg.NUM_LAYERS
    dropout: float = candle_cfg.DROPOUT
    lr: float = candle_cfg.LEARNING_RATE
    weight_decay: float = candle_cfg.WEIGHT_DECAY
    patience: int = candle_cfg.EARLY_STOPPING_PATIENCE
    min_epochs_before_early_stop: int = candle_cfg.MIN_EPOCHS_BEFORE_EARLY_STOP
    early_stopping_min_delta: float = candle_cfg.EARLY_STOPPING_MIN_DELTA


@dataclass(slots=True)
class TrainedLstmCandlesModel:
    model: LSTMClassifier
    dataset: SequenceDataset
    standardizer: SequenceStandardizer
    feature_columns: list[str]
    symbols: list[str]
    training_summary: dict[str, Any]
    model_args: dict[str, Any]


class LstmCandlesTrainer:
    def __init__(self, data_builder: LstmCandlesDataBuilder | None = None):
        self.data_builder = data_builder or LstmCandlesDataBuilder()

    def train_production(
        self,
        request: LstmCandlesTrainRequest,
        device: torch.device | None = None,
    ) -> TrainedLstmCandlesModel:
        self.set_seed(request.seed)
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = self.data_builder.load(
            LstmCandlesDataRequest(
                db_path=request.db_path,
                symbols=request.symbols,
            )
        )
        raw_dataset = SequenceDataset(
            data.sample_frame,
            data.history_by_symbol,
            data.feature_columns,
            request.sequence_length,
        )
        if len(raw_dataset) < candle_cfg.MIN_TRAIN_ROWS:
            raise RuntimeError(f"Only {len(raw_dataset)} sequences, need at least {candle_cfg.MIN_TRAIN_ROWS}.")

        train_indices, eval_indices = self.split_train_eval_indices(raw_dataset, candle_cfg.VALIDATION_FRACTION)
        if not train_indices:
            raise RuntimeError("No train sequences available after internal validation split.")
        standardizer = SequenceStandardizer().fit(raw_dataset.sequences_array(train_indices))
        dataset = SequenceDataset(
            data.sample_frame,
            data.history_by_symbol,
            data.feature_columns,
            request.sequence_length,
            standardizer=standardizer,
        )
        model, best_epoch, best_eval_loss = self.fit_model(dataset, train_indices, eval_indices, request, device)

        return TrainedLstmCandlesModel(
            model=model,
            dataset=dataset,
            standardizer=standardizer,
            feature_columns=list(data.feature_columns),
            symbols=list(request.symbols),
            training_summary={
                "rows": int(len(dataset)),
                "fit_rows": int(len(train_indices)),
                "eval_rows": int(len(eval_indices)),
                "best_epoch": int(best_epoch),
                "best_eval_loss": float(best_eval_loss),
                "epochs": int(request.epochs),
                "batch_size": int(request.batch_size),
                "lr": float(request.lr),
                "weight_decay": float(request.weight_decay),
                "patience": int(request.patience),
                "min_epochs_before_early_stop": int(request.min_epochs_before_early_stop),
                "early_stopping_min_delta": float(request.early_stopping_min_delta),
            },
            model_args={
                "input_size": int(len(data.feature_columns)),
                "hidden_size": int(request.hidden_size),
                "num_layers": int(request.num_layers),
                "dropout": float(request.dropout),
            },
        )

    def fit_model(
        self,
        dataset: SequenceDataset,
        train_indices: list[int],
        eval_indices: list[int],
        request: LstmCandlesTrainRequest,
        device: torch.device,
    ) -> tuple[LSTMClassifier, int, float]:
        fit_loader = DataLoader(Subset(dataset, train_indices), batch_size=request.batch_size, shuffle=True)
        eval_loader = DataLoader(Subset(dataset, eval_indices or train_indices), batch_size=request.batch_size, shuffle=False)

        model = LSTMClassifier(
            input_size=len(dataset.feature_columns),
            hidden_size=request.hidden_size,
            num_layers=request.num_layers,
            dropout=request.dropout,
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=request.lr, weight_decay=request.weight_decay)

        best_state = None
        best_eval_loss = float("inf")
        stale_epochs = 0
        best_epoch = 0

        for epoch in range(1, request.epochs + 1):
            train_loss = self.run_epoch(model, fit_loader, criterion, optimizer, device, train_mode=True)
            eval_loss = self.run_epoch(model, eval_loader, criterion, optimizer, device, train_mode=False)
            print(f"epoch={epoch} train_loss={train_loss:.5f} eval_loss={eval_loss:.5f}")
            if eval_loss < (best_eval_loss - request.early_stopping_min_delta):
                best_eval_loss = eval_loss
                best_epoch = epoch
                stale_epochs = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale_epochs += 1

            if epoch >= request.min_epochs_before_early_stop and stale_epochs >= request.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model, best_epoch, best_eval_loss

    @staticmethod
    def run_epoch(model, loader, criterion, optimizer, device, train_mode: bool) -> float:
        model.train(train_mode)
        total_loss = 0.0
        total_rows = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train_mode):
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                if train_mode:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), candle_cfg.GRADIENT_CLIP)
                    optimizer.step()
            total_loss += float(loss.item()) * len(y_batch)
            total_rows += len(y_batch)
        return total_loss / max(total_rows, 1)

    @staticmethod
    def split_train_eval_indices(
        dataset: SequenceDataset,
        validation_fraction: float,
    ) -> tuple[list[int], list[int]]:
        n_items = len(dataset)
        if n_items < 2:
            return list(range(n_items)), []

        sample_timestamps = np.asarray([sample.timestamp for sample in dataset.samples], dtype="datetime64[ns]")
        unique_timestamps = np.unique(sample_timestamps)
        if len(unique_timestamps) < 2:
            return list(range(n_items)), []

        eval_size = max(1, int(len(unique_timestamps) * validation_fraction))
        if eval_size >= len(unique_timestamps):
            eval_size = 1
        train_timestamps = set(unique_timestamps[:-eval_size].tolist())
        eval_timestamps = set(unique_timestamps[-eval_size:].tolist())
        train_indices = [idx for idx, ts in enumerate(sample_timestamps) if ts in train_timestamps]
        eval_indices = [idx for idx, ts in enumerate(sample_timestamps) if ts in eval_timestamps]
        return train_indices, eval_indices

    @staticmethod
    def set_seed(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
