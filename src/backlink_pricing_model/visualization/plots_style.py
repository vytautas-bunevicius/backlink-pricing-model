"""Shared Plotly styling and export helpers."""

import logging
import warnings
from pathlib import Path

import plotly.io as pio
from plotly.graph_objects import Layout


_LOGGER = logging.getLogger(__name__)

# Ensure Kaleido/export warnings are emitted at most once per session.
warnings.filterwarnings("once", category=RuntimeWarning, module=__name__)

# Primary brand colors
PRIMARY_BLUE = "#3A5CED"
LIGHT_BLUE = "#7BC0FF"

# Technical colors
WHITE = "#FFFFFF"
GRAY_LIGHT = "#E5E8EF"
TEXT_DARK = "#1A1E21"
BACKGROUND_TRANSPARENT = "rgba(255, 255, 255, 0.9)"

# Typography specifications
FONT_FAMILY = "Gordita, Figtree, sans-serif"
FONT_SIZE_TITLE = 24
FONT_SIZE_AXIS = 16
FONT_SIZE_TICK = 14
FONT_SIZE_LEGEND = 14

# Default plot dimensions
PLOT_HEIGHT = 600
PLOT_WIDTH_PER_SUBPLOT = 400
PLOT_MARGINS = {"l": 60, "r": 150, "t": 100, "b": 80, "pad": 10}

# Base plot layout configuration
BASE_LAYOUT = {
    "paper_bgcolor": WHITE,
    "plot_bgcolor": WHITE,
    "font": {
        "family": FONT_FAMILY,
        "color": TEXT_DARK,
        "size": FONT_SIZE_AXIS,
    },
    "xaxis": {
        "gridcolor": GRAY_LIGHT,
        "linecolor": GRAY_LIGHT,
        "zerolinecolor": GRAY_LIGHT,
        "showline": True,
        "linewidth": 1,
    },
    "yaxis": {
        "gridcolor": GRAY_LIGHT,
        "linecolor": GRAY_LIGHT,
        "zerolinecolor": GRAY_LIGHT,
        "showline": True,
        "linewidth": 1,
    },
}

# Color palettes
CATEGORICAL_PALETTE: list[str] = [
    PRIMARY_BLUE,
    LIGHT_BLUE,
    "#7E7AE6",
    "#85A2FF",
    "#82E5E8",
    "#C2A9FF",
    "#34D399",
    "#F59E0B",
    "#F87171",
    "#94A3B8",
]


def apply_plotly_defaults(template_name: str = "backlink_pricing") -> None:
    """Register and activate the default Plotly template for this project."""
    template = {"layout": Layout(**BASE_LAYOUT)}
    pio.templates[template_name] = template
    pio.templates.default = template_name


def save_figure_image(fig, save_path: str | Path) -> bool:
    """Persist a Plotly figure when static image export is available."""
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.write_image(output_path)
    except (ValueError, RuntimeError) as exc:
        message = f"Skipping static export: {exc}"
        _LOGGER.warning(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return False

    return True
