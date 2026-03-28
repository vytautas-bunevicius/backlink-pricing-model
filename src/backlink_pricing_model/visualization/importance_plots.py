"""Feature importance and SHAP visualization plots."""

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from backlink_pricing_model.core.models.visualization import PlotConfig
from backlink_pricing_model.visualization.plots_style import (
    BASE_LAYOUT,
    PLOT_HEIGHT,
    PLOT_MARGINS,
    PRIMARY_BLUE,
    save_figure_image,
)

_logger = logging.getLogger(__name__)


def _apply_base_layout(fig: go.Figure, config: PlotConfig | None) -> go.Figure:
    """Apply base layout and optional PlotConfig overrides to a figure."""
    layout_kwargs: dict = {**BASE_LAYOUT, "margin": PLOT_MARGINS}

    if config is not None:
        if config.height is not None:
            layout_kwargs["height"] = config.height
        if config.width is not None:
            layout_kwargs["width"] = config.width
        if config.title is not None:
            layout_kwargs["title"] = config.title
        if config.custom_layout is not None:
            layout_kwargs.update(config.custom_layout)

    fig.update_layout(**layout_kwargs)
    return fig


def _maybe_save(fig: go.Figure, config: PlotConfig | None) -> None:
    """Save figure if config has a save_path set."""
    if config is not None and config.save_path is not None:
        save_figure_image(fig, config.save_path)


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 20,
    title: str = "Feature importance",
    config: PlotConfig | None = None,
) -> go.Figure:
    """Plot feature importance as a horizontal bar chart.

    Args:
        feature_names: List of feature names.
        importances: Array of importance scores.
        top_n: Number of top features to show.
        title: Plot title.
        config: Optional plot configuration.

    Returns:
        Plotly figure.
    """
    df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=True)

    if len(df) > top_n:
        df = df.tail(top_n)

    effective_title = config.title if config and config.title else title
    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title=effective_title,
        labels={"importance": "Importance", "feature": "Feature"},
        color_discrete_sequence=[PRIMARY_BLUE],
    )
    _apply_base_layout(fig, config)
    _maybe_save(fig, config)
    return fig


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Feature correlation matrix",
    config: PlotConfig | None = None,
) -> go.Figure:
    """Plot a correlation heatmap.

    Args:
        corr_matrix: Correlation matrix DataFrame.
        title: Plot title.
        config: Optional plot configuration.

    Returns:
        Plotly figure.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )

    effective_title = config.title if config and config.title else title
    fig.update_layout(title=effective_title)

    # Apply defaults, then size overrides for heatmaps.
    height = config.height if config and config.height else 700
    width = config.width if config and config.width else 800
    _apply_base_layout(
        fig,
        PlotConfig(height=height, width=width) if config is None else config,
    )

    # Ensure heatmap dimensions are applied even if config has no size.
    if config is None or (config.height is None and config.width is None):
        fig.update_layout(height=height, width=width)

    _maybe_save(fig, config)
    return fig
