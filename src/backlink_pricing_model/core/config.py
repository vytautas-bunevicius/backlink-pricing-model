"""Load YAML configuration files into typed dicts."""

from pathlib import Path

import yaml

from backlink_pricing_model.core.environment import get_project_root


def load_config(config_path: str | Path) -> dict:
    """Load a YAML configuration file.

    Resolves relative paths against the project root.

    Args:
        config_path: Path to the YAML file (absolute or relative).

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = get_project_root() / path

    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open() as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root.

    Args:
        relative_path: Path string relative to project root.

    Returns:
        Absolute Path object.
    """
    return get_project_root() / relative_path
