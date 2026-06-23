from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import lightning as L
import rootutils
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".root", pythonpath=True)

from utils.config import instantiate_selected


def run_training(cfg: DictConfig) -> dict[str, Any]:
    L.seed_everything(cfg.train.seed, workers=True)

    datamodule = instantiate(cfg.train.data)
    model = instantiate(cfg.model)
    if cfg.train.compile:
        model = torch.compile(model)

    callbacks = instantiate_selected(cfg.callbacks, cfg.callback_names)
    loggers = instantiate_selected(cfg.loggers, cfg.logger_names)
    trainer = instantiate(cfg.train.trainer, callbacks=callbacks, logger=loggers)

    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, Path(cfg.paths.output_dir) / "resolved_config.yaml", resolve=True)

    trainer.fit(model=model, datamodule=datamodule)
    best_ckpt = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None
    trainer.test(model=model, datamodule=datamodule, ckpt_path=best_ckpt or None)

    return {
        "best_checkpoint": best_ckpt,
        "callback_metrics": {k: v.item() if hasattr(v, "item") else v for k, v in trainer.callback_metrics.items()},
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    result = run_training(cfg)
    print(OmegaConf.to_yaml(result))


if __name__ == "__main__":
    main()
