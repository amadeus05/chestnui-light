from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

import config as cfg
from lstm import config_lstm as lstm_cfg
from src.models.lstm.data_builder import LstmDataBuilder, LstmDataRequest
from src.models.lstm.dataset import SequenceDataset, SequenceStandardizer
from src.models.lstm.network import LSTMClassifier


@dataclass(frozen=True, slots=True)
class LstmTrainRequest:
    db_path: str = cfg.DB_PATH
    symbols: list[str] = field(default_factory=lambda: list(cfg.SYMBOLS))
    seed: int = 42
    sequence_length: int = lstm_cfg.SEQUENCE_LENGTH
    batch_size: int = lstm_cfg.BATCH_SIZE
    epochs: int = lstm_cfg.EPOCHS
    hidden_size: int = lstm_cfg.HIDDEN_SIZE
    num_layers: int = lstm_cfg.NUM_LAYERS
    dropout: float = lstm_cfg.DROPOUT
    lr: float = lstm_cfg.LEARNING_RATE
    weight_decay: float = lstm_cfg.WEIGHT_DECAY
    patience: int = lstm_cfg.EARLY_STOPPING_PATIENCE
    min_epochs_before_early_stop: int = lstm_cfg.MIN_EPOCHS_BEFORE_EARLY_STOP
    early_stopping_min_delta: float = lstm_cfg.EARLY_STOPPING_MIN_DELTA


@dataclass(slots=True)
class TrainedLstmModel:
    model: LSTMClassifier
    dataset: SequenceDataset
    standardizer: SequenceStandardizer
    feature_columns: list[str]
    training_summary: dict[str, Any]
    model_args: dict[str, Any]


class LstmTrainer:
    def __init__(self, data_builder: LstmDataBuilder | None = None):
        self.data_builder = data_builder or LstmDataBuilder()

    def train_production(self, request: LstmTrainRequest, device: torch.device | None = None) -> TrainedLstmModel:
        self.set_seed(request.seed)
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = self.data_builder.load(
            LstmDataRequest(
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
        if len(raw_dataset) < lstm_cfg.MIN_TRAIN_ROWS:
            raise RuntimeError(f"Only {len(raw_dataset)} sequences, need at least {lstm_cfg.MIN_TRAIN_ROWS}.")

        train_indices, eval_indices = self.split_train_eval_indices(len(raw_dataset), lstm_cfg.VALIDATION_FRACTION)
        standardizer = SequenceStandardizer().fit(raw_dataset.sequences_array(train_indices))
        dataset = SequenceDataset(
            data.sample_frame,
            data.history_by_symbol,
            data.feature_columns,
            request.sequence_length,
            standardizer=standardizer,
        )
        model, best_epoch, best_eval_loss = self.fit_model(dataset, train_indices, eval_indices, request, device)

        return TrainedLstmModel(
            model=model,
            dataset=dataset,
            standardizer=standardizer,
            feature_columns=list(data.feature_columns),
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
        request: LstmTrainRequest,
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
                    nn.utils.clip_grad_norm_(model.parameters(), lstm_cfg.GRADIENT_CLIP)
                    optimizer.step()
            total_loss += float(loss.item()) * len(y_batch)
            total_rows += len(y_batch)
        return total_loss / max(total_rows, 1)

    @staticmethod
    def split_train_eval_indices(n_items: int, validation_fraction: float) -> tuple[list[int], list[int]]:
        if n_items < 2:
            return list(range(n_items)), []
        eval_size = max(1, int(n_items * validation_fraction))
        if eval_size >= n_items:
            eval_size = 1
        split_at = n_items - eval_size
        return list(range(split_at)), list(range(split_at, n_items))

    @staticmethod
    def set_seed(seed: int) -> None:
        import numpy as np

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
