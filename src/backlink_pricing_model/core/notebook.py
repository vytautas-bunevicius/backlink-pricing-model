"""Notebook environment helpers."""

from backlink_pricing_model.core.environment import get_project_root
from backlink_pricing_model.core.logging import setup_logging


def init_notebook() -> None:
    """Initialize notebook environment with logging and paths."""
    setup_logging()
    root = get_project_root()
    print(f"Project root: {root}")
