from __future__ import annotations

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError


def _activation(name: str) -> nn.Module:
    activations = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation '{name}'. Choose from {sorted(activations)}.")
    return activations[name]()


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    _activation(activation),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LitRegressor(LightningModule):
    def __init__(
        self,
        input_dim: int = 32,
        output_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 3,
        activation: str = "silu",
        dropout: float = 0.1,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 5,
        scheduler_min_lr: float = 3e-6,
        scheduler_monitor: str = "valid/loss",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = MLP(input_dim, output_dim, hidden_dim, num_layers, activation, dropout)
        self.loss_fn = nn.MSELoss()
        self.train_mse = MeanSquaredError()
        self.valid_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()
        self.valid_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, target = self._shared_step(batch)
        self.train_mse.update(preds, target)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, target = self._shared_step(batch)
        self.valid_mse.update(preds, target)
        self.valid_mae.update(preds, target)
        self.log("valid/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("valid/mse", self.valid_mse, on_step=False, on_epoch=True)
        self.log("valid/mae", self.valid_mae, on_step=False, on_epoch=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, target = self._shared_step(batch)
        self.test_mse.update(preds, target)
        self.test_mae.update(preds, target)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True)

    def predict_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        x, _ = batch
        return self(x)

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience,
            min_lr=self.hparams.scheduler_min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.hparams.scheduler_monitor,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, target = batch
        preds = self(x)
        loss = self.loss_fn(preds, target)
        return loss, preds, target
