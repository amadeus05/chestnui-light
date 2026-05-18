from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src_refactor.core.config import ExperimentConfig
from src_refactor.core.contracts import ModelArtifactStore, ModelPredictor, ModelTrainer
from src_refactor.core.types import ModelArtifact, ModelInput, ModelSpec, WalkForwardFold
from src_refactor.core.types import LstmCandleWindowInput
from src_refactor.infrastructure.models.lstm_candles.predictor import LstmCandlePredictor, build_lstm_candle_model
from src_refactor.infrastructure.models.lstm_common import LstmTrainingConfig, SequenceStandardizer, train_lstm_classifier


CANDLE_DEFAULTS = {
    "sequence_length": 64,
    "window_size": 64,
    "batch_size": 128,
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.1,
    "learning_rate": 5e-4,
    "weight_decay": 1e-5,
    "epochs": 30,
    "early_stopping_patience": 8,
    "min_epochs_before_early_stop": 10,
    "early_stopping_min_delta": 5e-4,
    "gradient_clip": 1.0,
    "validation_fraction": 0.15,
    "min_train_rows": 200,
}


@dataclass(frozen=True, slots=True)
class LstmCandleTrainer(ModelTrainer):
    def train(
        self,
        train_input: ModelInput,
        config: ExperimentConfig,
        fold: WalkForwardFold | None = None,
    ) -> ModelArtifact:
        if not isinstance(train_input, LstmCandleWindowInput):
            raise TypeError("LstmCandleTrainer expects LstmCandleWindowInput.")
        if train_input.targets is None:
            raise ValueError("LSTM candle training requires targets.")
        training_config = LstmTrainingConfig.from_metadata({**CANDLE_DEFAULTS, **config.model.metadata})
        if len(train_input.candles) < training_config.min_train_rows:
            raise ValueError("Not enough LSTM candle sequences to train.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, standardizer, training_meta = train_lstm_classifier(
            train_input.candles,
            train_input.targets,
            training_config,
            device=device,
        )
        artifact_uri = _artifact_dir(config.model, fold) / "model.pt"
        return ModelArtifact(
            spec=config.model,
            uri=artifact_uri,
            fold_id=fold.fold_id if fold is not None else None,
            metadata={
                "model_state_dict": model.state_dict(),
                "feature_columns": list(train_input.candle_columns),
                "standardizer": standardizer.to_payload(),
                "model_args": {
                    "input_size": int(train_input.candles.shape[-1]),
                    "hidden_size": training_config.hidden_size,
                    "num_layers": training_config.num_layers,
                    "dropout": training_config.dropout,
                },
                "training": {**training_config.to_metadata(), **training_meta},
                "label_mapping": {"short": 0, "long": 1},
                "inverse_label_mapping": {"0": -1, "1": 1},
                "symbols": list(config.symbols or config.model.symbols),
                "rows": int(len(train_input.candles)),
            },
        )


@dataclass(frozen=True, slots=True)
class LstmCandleArtifactStore(ModelArtifactStore):
    root: Path = Path("models")

    def save(self, artifact: ModelArtifact) -> ModelArtifact:
        artifact_path = Path(artifact.uri)
        if not artifact_path.is_absolute():
            artifact_path = self.root / artifact_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact.metadata, artifact_path)
        return ModelArtifact(
            spec=artifact.spec,
            uri=artifact_path,
            fold_id=artifact.fold_id,
            metadata={key: value for key, value in artifact.metadata.items() if key != "model_state_dict"},
        )

    def load_predictor(self, spec: ModelSpec, fold_id: int | None = None) -> ModelPredictor:
        artifact_path = Path(spec.artifact_uri) if spec.artifact_uri else self.root / _artifact_dir(spec, fold_id) / "model.pt"
        payload = torch.load(artifact_path, map_location="cpu")
        training_config = LstmTrainingConfig.from_metadata({**CANDLE_DEFAULTS, **payload.get("training", {})})
        model_args = payload.get("model_args", {})
        model = build_lstm_candle_model(int(model_args["input_size"]), training_config)
        model.load_state_dict(payload["model_state_dict"])
        effective_spec = ModelSpec(
            model_type=spec.model_type,
            timeframe=spec.timeframe,
            profile=spec.profile,
            symbols=spec.symbols,
            input_profile=spec.input_profile,
            artifact_uri=str(artifact_path),
            metadata={**payload, **spec.metadata},
        )
        return LstmCandlePredictor(
            spec=effective_spec,
            model=model,
            standardizer=SequenceStandardizer.from_payload(payload["standardizer"]),
            training_config=training_config,
        )


def _artifact_dir(spec: ModelSpec, fold: WalkForwardFold | int | None) -> Path:
    base = Path(spec.model_type) / spec.timeframe / spec.profile
    fold_id = fold.fold_id if isinstance(fold, WalkForwardFold) else fold
    return base / f"fold_{fold_id}" if fold_id is not None else base
