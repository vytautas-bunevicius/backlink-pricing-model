"""Distribution plots for price, quality metrics, and traffic."""

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backlink_pricing_model.core.models.visualization import PlotConfig
from backlink_pricing_model.visualization.plots_style import (
    BASE_LAYOUT,
    CATEGORICAL_PALETTE,
    COLORS,
    FONT_SIZE_TICK,
    GRAY_LIGHT,
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
    return fig


def _maybe_save(fig: go.Figure, config: PlotConfig | None) -> None:
    """Save figure if config has a save_path set."""
    if config is not None and config.save_path is not None:
        save_figure_image(fig, config.save_path)


def plot_price_distribution(
    df: pd.DataFrame,
    log_scale: bool = True,
    config: PlotConfig | None = None,
) -> go.Figure:
    """Plot the distribution of backlink prices.

    Args:
        df: DataFrame with 'final_price' column.
        log_scale: Whether to use log scale on x-axis.
        config: Optional plot configuration.

    Returns:
        Plotly figure.
    """
    col = "log_price" if log_scale and "log_price" in df.columns else "final_price"
    default_title = (
        "Backlink price distribution (log scale)"
        if log_scale
        else "Backlink price distribution"
    )

    fig = px.histogram(
        df,
        x=col,
        nbins=80,
        title=config.title if config and config.title else default_title,
        labels={col: "Price (USD)" if not log_scale else "Log price"},
        color_discrete_sequence=[PRIMARY_BLUE],
    )
    _apply_base_layout(fig, config)
    _maybe_save(fig, config)
    return fig


def plot_metric_distributions(
    df: pd.DataFrame,
    config: PlotConfig | None = None,
) -> go.Figure:
    """Plot distributions of DR, CF, and TF as subplots.

    Args:
        df: DataFrame with dr, cf, tf columns.
        config: Optional plot configuration.

    Returns:
        Plotly figure with three subplots.
    """
    metrics = [m for m in ["dr", "cf", "tf"] if m in df.columns]
    num_cols = len(metrics)

    fig = make_subplots(
        rows=1,
        cols=num_cols,
        subplot_titles=[m.upper() for m in metrics],
    )

    for i, metric in enumerate(metrics, 1):
        data = df[metric].dropna()
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=50,
                marker_color=CATEGORICAL_PALETTE[i - 1],
                name=metric.upper(),
                showlegend=False,
            ),
            row=1,
            col=i,
        )

        # Apply consistent axis styling to each subplot.
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

    default_title = "Quality metric distributions"
    fig.update_layout(
        title=config.title if config and config.title else default_title,
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


def plot_price_by_quality_tier(
    df: pd.DataFrame,
    metric: str = "dr",
    config: PlotConfig | None = None,
) -> go.Figure:
    """Plot price distribution across quality tiers.

    Args:
        df: DataFrame with final_price and the metric column.
        metric: Quality metric to use for tiers (dr, cf, tf).
        config: Optional plot configuration.

    Returns:
        Plotly box plot figure.
    """
    data = df.dropna(subset=[metric, "final_price"]).copy()
    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0-19", "20-39", "40-59", "60-79", "80-100"]
    data["tier"] = pd.cut(data[metric], bins=bins, labels=labels, right=False)

    default_title = f"Price by {metric.upper()} tier"
    fig = px.box(
        data,
        x="tier",
        y="final_price",
        title=config.title if config and config.title else default_title,
        labels={"tier": f"{metric.upper()} tier", "final_price": "Price (USD)"},
        color="tier",
        color_discrete_sequence=CATEGORICAL_PALETTE,
    )
    fig.update_layout(showlegend=False)
    _apply_base_layout(fig, config)
    _maybe_save(fig, config)
    return fig


def plot_tld_distribution(
    df: pd.DataFrame,
    top_n: int = 15,
    config: PlotConfig | None = None,
) -> go.Figure:
    """Plot top TLD distribution by count.

    Args:
        df: DataFrame with 'tld' column.
        top_n: Number of top TLDs to show.
        config: Optional plot configuration.

    Returns:
        Plotly bar chart figure.
    """
    tld_counts = df["tld"].value_counts().head(top_n).reset_index()
    tld_counts.columns = ["tld", "count"]

    default_title = f"Top {top_n} TLD distribution"
    fig = px.bar(
        tld_counts,
        x="tld",
        y="count",
        title=config.title if config and config.title else default_title,
        labels={"tld": "TLD", "count": "Count"},
        color_discrete_sequence=[PRIMARY_BLUE],
    )
    _apply_base_layout(fig, config)
    _maybe_save(fig, config)
    return fig


def plot_price_by_tld(
    df: pd.DataFrame,
    top_n: int = 10,
    config: PlotConfig | None = None,
) -> go.Figure:
    """Plot median price by TLD.

    Args:
        df: DataFrame with 'tld' and 'final_price' columns.
        top_n: Number of top TLDs to include.
        config: Optional plot configuration.

    Returns:
        Plotly bar chart figure.
    """
    top_tlds = df["tld"].value_counts().head(top_n).index
    data = df[df["tld"].isin(top_tlds)]

    medians = (
        data.groupby("tld")["final_price"]
        .median()
        .sort_values(ascending=False)
        .reset_index()
    )
    medians.columns = ["tld", "median_price"]

    default_title = f"Median price by TLD (top {top_n})"
    fig = px.bar(
        medians,
        x="tld",
        y="median_price",
        title=config.title if config and config.title else default_title,
        labels={"tld": "TLD", "median_price": "Median price (USD)"},
        color_discrete_sequence=[COLORS["secondary"]],
    )
    _apply_base_layout(fig, config)
    _maybe_save(fig, config)
    return fig


def plot_country_distribution(
    df: pd.DataFrame,
    top_n: int = 15,
    config: PlotConfig | None = None,
) -> go.Figure:
    """Plot top country distribution by count.

    Args:
        df: DataFrame with 'country' column.
        top_n: Number of top countries to show.
        config: Optional plot configuration.

    Returns:
        Plotly bar chart figure.
    """
    country_counts = (
        df["country"].dropna().value_counts().head(top_n).reset_index()
    )
    country_counts.columns = ["country", "count"]

    default_title = f"Top {top_n} countries by listing count"
    fig = px.bar(
        country_counts,
        x="country",
        y="count",
        title=config.title if config and config.title else default_title,
        labels={"country": "Country", "count": "Count"},
        color_discrete_sequence=[COLORS["accent"]],
    )
    _apply_base_layout(fig, config)
    _maybe_save(fig, config)
    return fig


def plot_missing_values(
    df: pd.DataFrame,
    config: PlotConfig | None = None,
) -> go.Figure:
    """Plot missing value percentages per column.

    Args:
        df: Input DataFrame.
        config: Optional plot configuration.

    Returns:
        Plotly horizontal bar chart figure.
    """
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values()
    missing_pct = missing_pct[missing_pct > 0]

    default_title = "Missing values by column"
    fig = px.bar(
        x=missing_pct.values,
        y=missing_pct.index,
        orientation="h",
        title=config.title if config and config.title else default_title,
        labels={"x": "Missing (%)", "y": "Column"},
        color_discrete_sequence=[COLORS["negative"]],
    )
    _apply_base_layout(fig, config)
    _maybe_save(fig, config)
    return fig
