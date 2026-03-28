"""Notebook bootstrap and display helpers."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from backlink_pricing_model.core.environment import get_project_root
from backlink_pricing_model.core.logging import setup_logging


_logger = logging.getLogger(__name__)


class _ImageFactory(Protocol):
    """Callable protocol for constructing IPython image display objects."""

    def __call__(self, *, filename: str) -> Any: ...


def init_notebook() -> None:
    """Initialize notebook environment with logging and paths."""
    setup_logging()
    root = get_project_root()
    print(f"Project root: {root}")


def display_saved_image_or_figure(
    image_path: str | Path,
    figure: Any,
    *,
    display_fn: Callable[[Any], Any] | None = None,
    image_factory: _ImageFactory | None = None,
) -> None:
    """Display a saved image when available, otherwise fall back to a figure."""
    display_callable = display_fn
    if display_callable is None:
        from IPython.display import display  # noqa: PLC0415

        display_callable = display

    image_path_obj = Path(image_path)
    if image_path_obj.exists():
        image_builder = image_factory
        if image_builder is None:
            from IPython.display import Image  # noqa: PLC0415

            image_builder = Image

        display_callable(image_builder(filename=str(image_path_obj)))
        return

    display_callable(figure)
