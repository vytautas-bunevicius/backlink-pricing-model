"""Model evaluation plots: residuals, predictions vs actuals, learning curves."""

import logging

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backlink_pricing_model.core.models.visualization import PlotConfig
from backlink_pricing_model.visualization.plots_style import (
    BASE_LAYOUT,
    CATEGORICAL_PALETTE,
    FONT_SIZE_TICK,
    GRAY_LIGHT,
    LIGHT_BLUE,
    PLOT_HEIGHT,
    PLOT_MARGINS,
    PLOT_WIDTH_PER_SUBPLOT,
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
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def _maybe_save(fig: go.Figure, config: PlotConfig | None) -> None:
    """Save figure if config has a save_path set."""
    if config is not None and config.save_path is not None:
        save_figure_image(fig, config.save_path)


def plot_predictions_vs_actuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs actuals",
    config: PlotConfig | None = None,
) -> go.Figure:
    """Scatter plot of predicted vs actual prices.

    Args:
        y_true: Actual target values.
        y_pred: Predicted values.
        title: Plot title.
        config: Optional plot configuration.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker={"color": PRIMARY_BLUE, "opacity": 0.4, "size": 5},
            name="Predictions",
        )
    )

    # Perfect prediction line.
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line={"color": "#F87171", "dash": "dash"},
            name="Perfect prediction",
        )
    )

    effective_title = config.title if config and config.title else title
    fig.update_layout(
        title=effective_title,
        xaxis_title="Actual price (USD)",
        yaxis_title="Predicted price (USD)",
    )
    _apply_base_layout(fig, config)
    _maybe_save(fig, config)
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual analysis",
    config: PlotConfig | None = None,
) -> go.Figure:
    """Plot residual distribution and residuals vs predicted.

    Args:
        y_true: Actual target values.
        y_pred: Predicted values.
        title: Plot title.
        config: Optional plot configuration.

    Returns:
        Plotly figure with two subplots.
    """
    residuals = y_true - y_pred
    num_cols = 2

    fig = make_subplots(
        rows=1,
        cols=num_cols,
        subplot_titles=["Residual distribution", "Residuals vs predicted"],
    )

    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=60,
            marker_color=PRIMARY_BLUE,
            name="Residuals",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            marker={"color": LIGHT_BLUE, "opacity": 0.3, "size": 4},
            name="Residuals",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_hline(
        y=0, line_dash="dash", line_color="#F87171", row=1, col=2
    )

    for i in range(1, num_cols + 1):
        axis_suffix = "" if i == 1 else str(i)
        fig.update_layout(**{
            f"xaxis{axis_suffix}": {
                "gridcolor": GRAY_LIGHT,
                "linecolor": GRAY_LIGHT,
                "zerolinecolor": GRAY_LIGHT,
                "showline": True,
                "linewidth": 1,
                "tickfont": {"size": FONT_SIZE_TICK},
            },
            f"yaxis{axis_suffix}": {
                "gridcolor": GRAY_LIGHT,
                "linecolor": GRAY_LIGHT,
                "zerolinecolor": GRAY_LIGHT,
                "showline": True,
                "linewidth": 1,
                "tickfont": {"size": FONT_SIZE_TICK},
            },
        })

    effective_title = config.title if config and config.title else title
    fig.update_layout(
        title=effective_title,
        height=config.height if config and config.height else PLOT_HEIGHT,
        width=(
            config.width
            if config and config.width
            else PLOT_WIDTH_PER_SUBPLOT * num_cols
        ),
    )
    _apply_base_layout(fig, config)
    _maybe_save(fig, config)
    return fig


def plot_model_comparison(
    model_metrics: dict[str, dict[str, float]],
    metric_name: str = "rmse",
    config: PlotConfig | None = None,
) -> go.Figure:
    """Compare multiple models on a single metric.

    Args:
        model_metrics: Dict mapping model name to metrics dict.
        metric_name: Which metric to compare.
        config: Optional plot configuration.

    Returns:
        Plotly bar chart figure.
    """
    models = list(model_metrics.keys())
    values = [m[metric_name] for m in model_metrics.values()]

    default_title = f"Model comparison: {metric_name.upper()}"
    fig = px.bar(
        x=models,
        y=values,
        title=config.title if config and config.title else default_title,
        labels={"x": "Model", "y": metric_name.upper()},
        color=models,
        color_discrete_sequence=CATEGORICAL_PALETTE,
    )
    fig.update_layout(showlegend=False)
    _apply_base_layout(fig, config)
    _maybe_save(fig, config)
    return fig
