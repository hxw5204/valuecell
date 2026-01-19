"""Load screener configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from valuecell.utils.path import get_python_root_path


def get_screener_config_dir() -> Path:
    """Return the directory containing screener configs."""
    return Path(get_python_root_path()) / "configs" / "screener"


def load_screener_config(name: str) -> dict[str, Any]:
    """Load a screener configuration YAML file by name."""
    config_path = get_screener_config_dir() / f"{name}.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}
