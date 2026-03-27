"""Model evaluation plots: residuals, predictions vs actuals, learning curves."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backlink_pricing_model.visualization.plots_style import (
    CATEGORICAL_PALETTE,
    COLORS,
    apply_default_style,
)


def plot_predictions_vs_actuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs actuals",
) -> go.Figure:
    """Scatter plot of predicted vs actual prices.

    Args:
        y_true: Actual target values.
        y_pred: Predicted values.
        title: Plot title.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker={"color": COLORS["primary"], "opacity": 0.4, "size": 5},
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
            line={"color": COLORS["negative"], "dash": "dash"},
            name="Perfect prediction",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Actual price (USD)",
        yaxis_title="Predicted price (USD)",
    )
    return apply_default_style(fig)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual analysis",
) -> go.Figure:
    """Plot residual distribution and residuals vs predicted.

    Args:
        y_true: Actual target values.
        y_pred: Predicted values.
        title: Plot title.

    Returns:
        Plotly figure with two subplots.
    """
    residuals = y_true - y_pred
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Residual distribution", "Residuals vs predicted"],
    )

    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=60,
            marker_color=COLORS["primary"],
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
            marker={"color": COLORS["secondary"], "opacity": 0.3, "size": 4},
            name="Residuals",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["negative"], row=1, col=2)

    fig.update_layout(title=title)
    return apply_default_style(fig)


def plot_model_comparison(
    model_metrics: dict[str, dict[str, float]],
    metric_name: str = "rmse",
) -> go.Figure:
    """Compare multiple models on a single metric.

    Args:
        model_metrics: Dict mapping model name to metrics dict.
        metric_name: Which metric to compare.

    Returns:
        Plotly bar chart figure.
    """
    models = list(model_metrics.keys())
    values = [m[metric_name] for m in model_metrics.values()]

    fig = px.bar(
        x=models,
        y=values,
        title=f"Model comparison: {metric_name.upper()}",
        labels={"x": "Model", "y": metric_name.upper()},
        color=models,
        color_discrete_sequence=CATEGORICAL_PALETTE,
    )
    fig.update_layout(showlegend=False)
    return apply_default_style(fig)
