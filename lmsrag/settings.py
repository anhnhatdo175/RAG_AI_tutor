"""Load and validate project configuration files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs"
_PATH_KEYS = {"path", "skill_map", "output_dir", "tables_dir"}
_DATA_REQUIRED_KEYS = {"data_seed", "seeds", "split", "datasets", "output_dir"}


def _config_path(name: str | Path) -> Path:
    path = Path(name)
    if not path.suffix:
        path = path.with_suffix(".yaml")
    if path.is_absolute():
        return path.resolve()
    if path.exists():
        return path.resolve()
    if path.parts and path.parts[0] == "configs":
        return (REPO / path).resolve()
    return (CONFIG_DIR / path).resolve()


def _resolve_paths(value: Any, key: str | None = None) -> Any:
    if key in _PATH_KEYS and isinstance(value, str):
        path = Path(value)
        return (path if path.is_absolute() else REPO / path).resolve()
    if isinstance(value, dict):
        return {child_key: _resolve_paths(child, child_key) for child_key, child in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(child, key) for child in value]
    return value


def _validate_config(config: Any, path: Path) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    if path.name == "data.yaml":
        missing = sorted(_DATA_REQUIRED_KEYS - config.keys())
        if missing:
            raise ValueError(f"Missing required data configuration keys: {missing}")
        if isinstance(config["data_seed"], bool) or not isinstance(config["data_seed"], int):
            raise ValueError("data_seed must be an integer")
        seed_values = config["seeds"]
        if (not isinstance(seed_values, list) or not seed_values
                or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seed_values)):
            raise ValueError("seeds must be a non-empty list of integers")
        if not isinstance(config["split"], dict):
            raise ValueError("split must be a mapping")
        missing_split = sorted({"history", "validation", "test"} - config["split"].keys())
        if missing_split:
            raise ValueError(f"Missing required split configuration keys: {missing_split}")
        if not isinstance(config["datasets"], dict) or not config["datasets"]:
            raise ValueError("datasets must be a non-empty mapping")
        for dataset, settings in config["datasets"].items():
            if not isinstance(settings, dict) or not isinstance(settings.get("path"), str):
                raise ValueError(f"datasets.{dataset}.path must be a string")
    return _resolve_paths(config)


def load(name: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving configured file paths against the repository root."""
    path = _config_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _validate_config(config, path)


def seeds() -> list[int]:
    """Return the configured experiment seeds."""
    return list(load("data")["seeds"])


def data_seed() -> int:
    """Return the configured data sampling seed."""
    return int(load("data")["data_seed"])
