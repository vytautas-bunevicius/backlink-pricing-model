"""Runtime environment detection and configuration."""

from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory.

    Walks up from this file until it finds pyproject.toml.

    Returns:
        Path to the project root.

    Raises:
        FileNotFoundError: If pyproject.toml is not found.
    """
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    msg = "Could not find project root (no pyproject.toml found)"
    raise FileNotFoundError(msg)
