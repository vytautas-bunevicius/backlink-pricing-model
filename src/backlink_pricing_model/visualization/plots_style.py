"""Shared plot styling configuration for consistent visuals."""

import plotly.graph_objects as go
import plotly.io as pio


# Color palette.
COLORS = {
    "primary": "#3A5CED",
    "secondary": "#7E7AE6",
    "accent": "#82E5E8",
    "highlight": "#85A2FF",
    "muted": "#C2A9FF",
    "positive": "#34D399",
    "negative": "#F87171",
    "neutral": "#94A3B8",
}

SEQUENTIAL_PALETTE: list[str] = [
    "#3A5CED",
    "#5B6FF0",
    "#7E7AE6",
    "#85A2FF",
    "#82E5E8",
    "#C2A9FF",
]

CATEGORICAL_PALETTE: list[str] = [
    "#3A5CED",
    "#7E7AE6",
    "#82E5E8",
    "#85A2FF",
    "#C2A9FF",
    "#34D399",
    "#F59E0B",
    "#F87171",
    "#94A3B8",
    "#6366F1",
]


def get_default_layout(**overrides: object) -> dict:
    """Return default Plotly layout configuration.

    Args:
        **overrides: Layout properties to override.

    Returns:
        Layout dict for plotly figures.
    """
    layout = {
        "template": "plotly_white",
        "font": {"family": "Inter, system-ui, sans-serif", "size": 13},
        "title_font_size": 18,
        "margin": {"l": 60, "r": 30, "t": 60, "b": 60},
        "colorway": CATEGORICAL_PALETTE,
        "hoverlabel": {"font_size": 12},
    }
    layout.update(overrides)
    return layout


def apply_default_style(fig: go.Figure, **overrides: object) -> go.Figure:
    """Apply default styling to a Plotly figure.

    Args:
        fig: Plotly figure to style.
        **overrides: Layout properties to override.

    Returns:
        Styled figure.
    """
    fig.update_layout(**get_default_layout(**overrides))
    return fig


def save_plot(
    fig: go.Figure,
    filename: str,
    output_dir: str = "images",
    width: int = 1200,
    height: int = 600,
) -> None:
    """Save a Plotly figure as a static PNG image.

    Args:
        fig: Figure to save.
        filename: Output filename (without extension).
        output_dir: Target directory.
        width: Image width in pixels.
        height: Image height in pixels.
    """
    from pathlib import Path

    from backlink_pricing_model.core.environment import get_project_root

    path = get_project_root() / output_dir / f"{filename}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_image(fig, str(path), width=width, height=height, scale=2)
