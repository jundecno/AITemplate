# AI PyTorch Lightning Template

A compact training template for PyTorch Lightning projects. It is designed to be runnable immediately with synthetic regression data, then easy to adapt to real datasets and models.

## Features

- Hydra configuration with composable `train`, `model`, `callbacks`, and `loggers` files.
- Reproducible Lightning training loop with seed control and deterministic mode.
- Example `LightningDataModule` and `LightningModule` that can run without external data.
- Checkpointing, early stopping, LR monitoring, CSV logging, and TensorBoard logging.
- Separate commands for training, testing, and prediction.

## Setup

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version from the official PyTorch selector before installing the rest of the requirements.

## Quick Start

Run a tiny smoke test:

```bash
python scripts/train.py train.trainer.max_epochs=1 train.data.num_workers=0 logger_names=[csv]
```

Run the default training template:

```bash
python scripts/train.py
```

Outputs are written to `results/YYYY-MM-DD/HH-MM-SS_<name>/`, including:

- `resolved_config.yaml`
- `checkpoints/`
- `logs/`

## Common Overrides

```bash
python scripts/train.py name=exp-mlp model.hidden_dim=256 model.num_layers=4 train.optimizer.lr=1e-3
python scripts/train.py train.trainer.accelerator=gpu train.trainer.devices=1
python scripts/train.py logger_names=[csv,tensorboard,wandb] loggers.wandb.offline=false loggers.wandb.project=my-project
```

## Test And Predict

```bash
python scripts/test.py ckpt_path=results/2026-06-23/12-00-00_debug-run/checkpoints/last.ckpt
python scripts/prediction.py ckpt_path=results/2026-06-23/12-00-00_debug-run/checkpoints/last.ckpt
```

Predictions are saved as `predictions.pt` in the Hydra output directory.

## Adapting To A Real Project

1. Replace `RegressionDataModule` in `models/dataloader.py` with your dataset loading logic.
2. Replace `LitRegressor` in `models/model.py` with your model and task-specific steps.
3. Keep metric names stable if callbacks monitor them, for example `valid/loss`.
4. Add dataset paths and task parameters to `configs/train.yaml` or a new config group.
5. Keep experiment-specific changes as command-line overrides or new YAML files, not hardcoded constants.

## Suggested Layout

```text
configs/      Hydra configs
models/       LightningModule, DataModule, model components
scripts/      CLI entry points for train/test/predict
utils/        shared helpers
results/      generated experiment outputs, ignored by git
notebooks/    exploratory work
```
