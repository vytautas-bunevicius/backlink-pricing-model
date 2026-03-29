"""Feature importance and SHAP visualization plots."""

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from backlink_pricing_model.core.models.visualization import PlotConfig
from backlink_pricing_model.visualization.plots_style import (
    BASE_LAYOUT,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    PLOT_MARGINS,
    PRIMARY_BLUE,
    SOFT_BLUE,
    TEXT_DARK,
    WHITE,
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


CORRELATION_TEXT_THRESHOLD = 0.8


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Feature correlation matrix",
    threshold: float = 0.7,
    config: PlotConfig | None = None,
) -> go.Figure:
    """Plot a correlation heatmap with blue colorscale and annotations.

    Args:
        corr_matrix: Correlation matrix DataFrame.
        title: Plot title.
        threshold: Absolute correlation threshold for showing annotations.
        config: Optional plot configuration.

    Returns:
        Plotly figure.
    """
    mask = np.abs(corr_matrix.values) > threshold

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale=[
                [0.0, SOFT_BLUE],
                [0.5, WHITE],
                [1.0, PRIMARY_BLUE],
            ],
            zmin=-1,
            zmax=1,
        )
    )

    for row_idx in range(len(corr_matrix)):
        for col_idx in range(len(corr_matrix)):
            if mask[row_idx, col_idx] and row_idx != col_idx:
                corr_val = float(corr_matrix.iloc[row_idx, col_idx])
                fig.add_annotation(
                    x=col_idx,
                    y=row_idx,
                    text=f"{corr_val:.2f}",
                    showarrow=False,
                    font={
                        "size": FONT_SIZE_TICK,
                        "color": TEXT_DARK
                        if abs(corr_val) < CORRELATION_TEXT_THRESHOLD
                        else WHITE,
                    },
                )

    effective_title = config.title if config and config.title else title
    height = config.height if config and config.height else 700
    width = config.width if config and config.width else 800

    fig.update_layout(
        title={
            "text": effective_title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": FONT_SIZE_TITLE},
        },
        height=height,
        width=width,
        xaxis={
            "tickangle": 45,
            "tickfont": {"size": FONT_SIZE_TICK},
        },
        yaxis={
            "tickfont": {"size": FONT_SIZE_TICK},
        },
        coloraxis_colorbar={
            "title": {"text": "Correlation"},
            "tickfont": {"size": FONT_SIZE_TICK},
        },
    )

    _apply_base_layout(
        fig,
        PlotConfig(height=height, width=width) if config is None else config,
    )
    _maybe_save(fig, config)
    return fig


def extract_high_correlations(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.8,
    top_n: int = 20,
) -> pd.DataFrame:
    """Extract highly correlated feature pairs above a threshold.

    Args:
        corr_matrix: Correlation matrix DataFrame.
        threshold: Minimum absolute correlation to include.
        top_n: Maximum number of pairs to return.

    Returns:
        DataFrame with columns feature_1, feature_2, pearson_r sorted by
        absolute correlation descending.

    Raises:
        ValueError: If threshold is invalid or matrix is empty.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    if corr_matrix.empty:
        raise ValueError("Correlation matrix cannot be empty")

    row_indices, col_indices = np.where(
        np.abs(np.triu(corr_matrix.values, k=1)) > threshold
    )
    pairs = [
        {
            "feature_1": corr_matrix.index[r],
            "feature_2": corr_matrix.columns[c],
            "pearson_r": round(float(corr_matrix.iloc[r, c]), 4),
        }
        for r, c in zip(row_indices, col_indices, strict=True)
    ]

    if not pairs:
        return pd.DataFrame(columns=["feature_1", "feature_2", "pearson_r"])

    result = pd.DataFrame(pairs)
    result["abs_r"] = result["pearson_r"].abs()
    result = (
        result.sort_values("abs_r", ascending=False)
        .drop(columns=["abs_r"])
        .head(top_n)
        .reset_index(drop=True)
    )
    return result
