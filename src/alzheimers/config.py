"""Configuration loading.

The whole pipeline is driven by ``configs/default.yaml``. Config objects are
plain nested dicts wrapped in a small accessor so that ``cfg["training.lr"]``
works and typos raise instead of silently returning ``None``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class Config:
    """Dotted-path read access over a nested dict."""

    def __init__(self, data: dict[str, Any], source: Path | None = None):
        self._data = data
        self.source = source

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(data, source=path)

    def __getitem__(self, dotted: str) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"missing config key: {dotted!r} (in {self.source})")
            node = node[part]
        return node

    def get(self, dotted: str, default: Any = None) -> Any:
        try:
            return self[dotted]
        except KeyError:
            return default

    def path(self, dotted: str) -> Path:
        """Resolve a config value as a path relative to the repo root."""
        value = Path(self[dotted])
        return value if value.is_absolute() else (REPO_ROOT / value).resolve()

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config(source={self.source})"


def set_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is a hard dep in practice
        pass
