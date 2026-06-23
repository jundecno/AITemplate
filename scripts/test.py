from __future__ import annotations

import hydra
import lightning as L
import rootutils
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".root", pythonpath=True)

from utils.config import instantiate_selected


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if "ckpt_path" not in cfg or cfg.ckpt_path is None:
        raise ValueError("Pass a checkpoint with ckpt_path=/path/to/model.ckpt")

    L.seed_everything(cfg.train.seed, workers=True)
    datamodule = instantiate(cfg.train.data)
    model = instantiate(cfg.model)
    loggers = instantiate_selected(cfg.loggers, cfg.logger_names)
    trainer = instantiate(cfg.train.trainer, logger=loggers, callbacks=[])
    metrics = trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    print(OmegaConf.to_yaml({"test_metrics": metrics}))


if __name__ == "__main__":
    main()
