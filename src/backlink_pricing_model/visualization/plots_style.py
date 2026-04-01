"""Shared Plotly styling and export helpers."""

import logging
import warnings
from pathlib import Path

import plotly.io as pio
from plotly.graph_objects import Layout


_LOGGER = logging.getLogger(__name__)

warnings.filterwarnings("once", category=RuntimeWarning, module=__name__)

PRIMARY_BLUE = "#3A5CED"
LIGHT_BLUE = "#7BC0FF"
SOFT_BLUE = "#abd9e9"

WHITE = "#FFFFFF"
GRAY_LIGHT = "#E5E8EF"
GRAY_MID = "#94A3B8"
TEXT_DARK = "#1A1E21"
BACKGROUND_TRANSPARENT = "rgba(255, 255, 255, 0.9)"

FONT_FAMILY = "Inter, system-ui, -apple-system, sans-serif"
FONT_SIZE_TITLE = 18
FONT_SIZE_AXIS = 13
FONT_SIZE_TICK = 12
FONT_SIZE_LEGEND = 12

PLOT_HEIGHT = 500
PLOT_WIDTH_PER_SUBPLOT = 420
PLOT_MARGINS = {"l": 70, "r": 30, "t": 60, "b": 60, "pad": 4}

_AXIS_DEFAULTS = {
    "gridcolor": GRAY_LIGHT,
    "linecolor": GRAY_LIGHT,
    "zerolinecolor": GRAY_LIGHT,
    "showline": True,
    "linewidth": 1,
    "automargin": True,
    "title_standoff": 16,
    "title_font_size": FONT_SIZE_AXIS,
    "tickfont_size": FONT_SIZE_TICK,
}

BASE_LAYOUT = {
    "paper_bgcolor": WHITE,
    "plot_bgcolor": WHITE,
    "font": {
        "family": FONT_FAMILY,
        "color": TEXT_DARK,
        "size": FONT_SIZE_TICK,
    },
    "title_font_size": FONT_SIZE_TITLE,
    "title_x": 0.5,
    "title_xanchor": "center",
    "xaxis": {**_AXIS_DEFAULTS},
    "yaxis": {**_AXIS_DEFAULTS},
    "margin": PLOT_MARGINS,
}

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
    GRAY_MID,
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


def save_plot(fig, name: str, output_dir: str | Path) -> bool:
    """Backward-compatible wrapper to save a figure by name and directory."""
    path = Path(output_dir) / f"{name}.png"
    return save_figure_image(fig, path)
