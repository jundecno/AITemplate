from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
import rootutils
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if "ckpt_path" not in cfg or cfg.ckpt_path is None:
        raise ValueError("Pass a checkpoint with ckpt_path=/path/to/model.ckpt")

    L.seed_everything(cfg.train.seed, workers=True)
    datamodule = instantiate(cfg.train.data)
    model = instantiate(cfg.model)
    trainer = instantiate(cfg.train.trainer, logger=False, callbacks=[])
    predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    output_path = Path(cfg.paths.output_dir) / "predictions.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.cat(predictions), output_path)
    print(OmegaConf.to_yaml({"prediction_path": str(output_path)}))


if __name__ == "__main__":
    main()
