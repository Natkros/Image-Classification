"""Small config layer: YAML file + dotted-key CLI overrides, with attribute access."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class Config(dict):
    """A dict that also supports attribute access and dotted get/set.

    cfg.data.batch_size  ==  cfg["data"]["batch_size"]  ==  cfg.get_path("data.batch_size")
    """

    def __init__(self, mapping: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        for key, value in (mapping or {}).items():
            self[key] = Config(value) if isinstance(value, Mapping) else value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = Config(value) if isinstance(value, Mapping) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: Config = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], Config):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = value

    def to_dict(self) -> dict:
        out: dict[str, Any] = {}
        for key, value in self.items():
            out[key] = value.to_dict() if isinstance(value, Config) else value
        return out

    def copy(self) -> "Config":
        return Config(copy.deepcopy(self.to_dict()))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")


def _coerce(text: str) -> Any:
    """Turn a CLI string into a Python value using YAML rules ('3' -> 3, 'true' -> True)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Config:
    """Load a YAML config and apply ``key.path=value`` overrides."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = Config(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.path=value, got: {item!r}")
        key, _, raw = item.partition("=")
        cfg.set_path(key.strip(), _coerce(raw.strip()))

    return cfg
