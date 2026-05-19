from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True, slots=True)
class LstmTrainingConfig:
    sequence_length: int = 48
    batch_size: int = 128
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    early_stopping_patience: int = 8
    min_epochs_before_early_stop: int = 12
    early_stopping_min_delta: float = 5e-4
    gradient_clip: float = 1.0
    validation_fraction: float = 0.15
    min_train_rows: int = 200

    @classmethod
    def from_metadata(cls, metadata: dict) -> "LstmTrainingConfig":
        defaults = cls()
        return cls(
            sequence_length=int(metadata.get("sequence_length", metadata.get("window_size", defaults.sequence_length))),
            batch_size=int(metadata.get("batch_size", defaults.batch_size)),
            hidden_size=int(metadata.get("hidden_size", defaults.hidden_size)),
            num_layers=int(metadata.get("num_layers", defaults.num_layers)),
            dropout=float(metadata.get("dropout", defaults.dropout)),
            learning_rate=float(metadata.get("learning_rate", metadata.get("lr", defaults.learning_rate))),
            weight_decay=float(metadata.get("weight_decay", defaults.weight_decay)),
            epochs=int(metadata.get("epochs", defaults.epochs)),
            early_stopping_patience=int(metadata.get("early_stopping_patience", defaults.early_stopping_patience)),
            min_epochs_before_early_stop=int(
                metadata.get("min_epochs_before_early_stop", defaults.min_epochs_before_early_stop)
            ),
            early_stopping_min_delta=float(metadata.get("early_stopping_min_delta", defaults.early_stopping_min_delta)),
            gradient_clip=float(metadata.get("gradient_clip", defaults.gradient_clip)),
            validation_fraction=float(metadata.get("validation_fraction", defaults.validation_fraction)),
            min_train_rows=int(metadata.get("min_train_rows", defaults.min_train_rows)),
        )

    def to_metadata(self) -> dict[str, float | int]:
        return {
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "min_epochs_before_early_stop": self.min_epochs_before_early_stop,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "gradient_clip": self.gradient_clip,
            "validation_fraction": self.validation_fraction,
            "min_train_rows": self.min_train_rows,
        }


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1])


class SequenceStandardizer:
    def __init__(self, eps: float = 1e-6) -> None:
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

    def to_payload(self) -> dict[str, list[float]]:
        if self.mean_ is None or self.std_ is None:
            return {}
        return {
            "mean": self.mean_.astype(float).tolist(),
            "std": self.std_.astype(float).tolist(),
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "SequenceStandardizer":
        standardizer = cls()
        standardizer.mean_ = np.asarray(payload["mean"], dtype=np.float32)
        standardizer.std_ = np.asarray(payload["std"], dtype=np.float32)
        return standardizer


def split_train_eval_indices(n_items: int, validation_fraction: float) -> tuple[list[int], list[int]]:
    if n_items < 2:
        return list(range(n_items)), []
    eval_size = max(1, int(n_items * validation_fraction))
    if eval_size >= n_items:
        eval_size = 1
    split_at = n_items - eval_size
    return list(range(split_at)), list(range(split_at, n_items))


def run_epoch(
    model: LSTMClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    train_mode: bool,
    gradient_clip: float,
) -> float:
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
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        total_loss += float(loss.item()) * len(y_batch)
        total_rows += len(y_batch)
    return total_loss / max(total_rows, 1)


def train_lstm_classifier(
    sequences: np.ndarray,
    targets: np.ndarray,
    config: LstmTrainingConfig,
    *,
    device: torch.device,
) -> tuple[LSTMClassifier, SequenceStandardizer, dict[str, float | int]]:
    train_indices, eval_indices = split_train_eval_indices(len(sequences), config.validation_fraction)
    standardizer = SequenceStandardizer().fit(sequences[train_indices])
    standardized = standardizer.transform(sequences)

    x_tensor = torch.from_numpy(standardized.astype(np.float32))
    y_tensor = torch.from_numpy(targets.astype(np.int64))
    dataset = TensorDataset(x_tensor, y_tensor)
    fit_loader = DataLoader(torch.utils.data.Subset(dataset, train_indices), batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(
        torch.utils.data.Subset(dataset, eval_indices or train_indices),
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = LSTMClassifier(
        input_size=sequences.shape[-1],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state = None
    best_eval_loss = float("inf")
    stale_epochs = 0
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        run_epoch(
            model,
            fit_loader,
            criterion,
            optimizer,
            device,
            train_mode=True,
            gradient_clip=config.gradient_clip,
        )
        eval_loss = run_epoch(
            model,
            eval_loader,
            criterion,
            optimizer,
            device,
            train_mode=False,
            gradient_clip=config.gradient_clip,
        )
        if eval_loss < (best_eval_loss - config.early_stopping_min_delta):
            best_eval_loss = eval_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        if epoch >= config.min_epochs_before_early_stop and stale_epochs >= config.early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, standardizer, {"best_epoch": int(best_epoch), "best_eval_loss": float(best_eval_loss)}


def predict_lstm_proba(
    model: LSTMClassifier,
    sequences: np.ndarray,
    standardizer: SequenceStandardizer,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    standardized = standardizer.transform(sequences)
    loader = DataLoader(TensorDataset(torch.from_numpy(standardized.astype(np.float32))), batch_size=batch_size)
    model.eval()
    probas = []
    with torch.no_grad():
        for (x_batch,) in loader:
            logits = model(x_batch.to(device))
            probas.append(torch.softmax(logits, dim=1).cpu().numpy())
    if not probas:
        return np.empty((0, 2), dtype=np.float32)
    return np.vstack(probas)
