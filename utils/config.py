from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig


def instantiate_selected(config: DictConfig, names: Iterable[str]) -> list[Any]:
    """Instantiate named Hydra config entries while allowing null/disabled values."""
    objects: list[Any] = []
    for name in names:
        if name not in config or config[name] is None:
            continue
        objects.append(instantiate(config[name]))
    return objects
