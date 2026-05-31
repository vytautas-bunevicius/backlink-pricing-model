"""Visualization-related data models and configurations.

This module contains Pydantic models used for plot configuration
and visualization settings.
"""

from typing import Any

from pydantic import BaseModel


class PlotConfig(BaseModel):
    """Configuration for plot styling and output.

    Attributes:
        height: Plot height in pixels (None uses module default).
        width: Plot width in pixels (None uses module default).
        title: Plot title (None uses function default).
        custom_layout: Custom layout options to merge with defaults.
        save_path: Optional path to save the plot.
    """

    height: int | None = None
    width: int | None = None
    title: str | None = None
    custom_layout: dict[str, Any] | None = None
    save_path: str | None = None
