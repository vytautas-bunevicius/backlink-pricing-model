"""Distribution plots for price, quality metrics, and traffic."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backlink_pricing_model.core.models.visualization import PlotConfig
from backlink_pricing_model.visualization.plots_style import (
    BASE_LAYOUT,
    CATEGORICAL_PALETTE,
    GRAY_LIGHT,
    GRAY_MID,
    LIGHT_BLUE,
    PLOT_HEIGHT,
    PLOT_WIDTH_PER_SUBPLOT,
    PRIMARY_BLUE,
    save_figure_image,
)


def _apply_base_layout(fig: go.Figure, config: PlotConfig | None) -> go.Figure:
    """Apply base layout and optional PlotConfig overrides."""
    layout_kwargs: dict = {**BASE_LAYOUT}

    if config is not None:
        if config.height is not None:
            layout_kwargs["height"] = config.height
        if config.width is not None:
            layout_kwargs["width"] = config.width
        if config.title is not None:
            layout_kwargs["title_text"] = config.title
        if config.custom_layout is not None:
            layout_kwargs.update(config.custom_layout)

    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def _maybe_save(fig: go.Figure, config: PlotConfig | None) -> None:
    if config is not None and config.save_path is not None:
        save_figure_image(fig, config.save_path)


def _title(config: PlotConfig | None, default: str) -> str:
    return config.title if config and config.title else default


def plot_price_distribution(
    df: pd.DataFrame,
    log_scale: bool = True,
    config: PlotConfig | None = None,
) -> go.Figure:
    """Plot the distribution of backlink prices."""
    if log_scale and "log_price" in df.columns:
        col, x_label = "log_price", "Log price"
        default_title = "Backlink price distribution (log scale)"
    else:
        col, x_label = "final_price", "Price (USD)"
        default_title = "Backlink price distribution"

    fig = go.Figure(
        go.Histogram(
            x=df[col].dropna(),
            nbinsx=80,
            marker_color=PRIMARY_BLUE,
            marker_line_width=0,
        )
    )
    fig.update_layout(
        title_text=_title(config, default_title),
        xaxis_title=x_label,
        yaxis_title=None,
        bargap=0.02,
    )
    _apply_base_layout(fig, config)
    _maybe_save(fig, config)
    return fig


def plot_metric_distributions(
    df: pd.DataFrame,
    config: PlotConfig | None = None,
) -> go.Figure:
    """Plot distributions of DR, CF, and TF as subplots."""
    metrics = [m for m in ["dr", "cf", "tf"] if m in df.columns]
    num_cols = len(metrics)

    fig = make_subplots(
        rows=1,
        cols=num_cols,
        subplot_titles=[m.upper() for m in metrics],
        horizontal_spacing=0.08,
    )

    for i, metric in enumerate(metrics, 1):
        fig.add_trace(
            go.Histogram(
                x=df[metric].dropna(),
                nbinsx=50,
                marker_color=CATEGORICAL_PALETTE[i - 1],
                marker_line_width=0,
                showlegend=False,
            ),
            row=1,
            col=i,
        )

        axis_suffix = "" if i == 1 else str(i)
        fig.update_layout(**{
            f"xaxis{axis_suffix}": {
                "title_text": "Score",
                "gridcolor": GRAY_LIGHT,
                "showline": True,
                "linewidth": 1,
                "linecolor": GRAY_LIGHT,
            },
            f"yaxis{axis_suffix}": {
                "title_text": None,
                "gridcolor": GRAY_LIGHT,
                "showline": True,
                "linewidth": 1,
                "linecolor": GRAY_LIGHT,
            },
        })

    fig.update_layout(
        title_text=_title(config, "Quality metric distributions"),
        height=config.height if config and config.height else PLOT_HEIGHT,
        width=(
            config.width
            if config and config.width
            else PLOT_WIDTH_PER_SUBPLOT * num_cols
        ),
        bargap=0.02,
    )
    _apply_base_layout(fig, config)
    _maybe_save(fig, config)
    return fig


def plot_price_by_quality_tier(
    df: pd.DataFrame,
    metric: str = "dr",
    config: PlotConfig | None = None,
) -> go.Figure:
    """Plot price distribution across quality tiers."""
    data = df.dropna(subset=[metric, "final_price"]).copy()
    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0-19", "20-39", "40-59", "60-79", "80-100"]
    data["tier"] = pd.Categorical(
        pd.cut(data[metric], bins=bins, labels=labels, right=False),
        categories=labels,
        ordered=True,
    )
    data = data.sort_values("tier")

    fig = go.Figure()
    for i, tier in enumerate(labels):
        tier_data = data[data["tier"] == tier]["final_price"]
        fig.add_trace(
            go.Box(
                y=tier_data,
                name=tier,
                marker_color=CATEGORICAL_PALETTE[i],
                line_color=CATEGORICAL_PALETTE[i],
                showlegend=False,
            )
        )

    fig.update_layout(
        title_text=_title(config, f"Price by {metric.upper()} tier"),
        xaxis_title=f"{metric.upper()} tier",
        yaxis_title="Price (USD)",
    )
    _apply_base_layout(fig, config)
    _maybe_save(fig, config)
    return fig


def plot_tld_distribution(
    df: pd.DataFrame,
    top_n: int = 10,
    config: PlotConfig | None = None,
) -> go.Figure:
    """Plot TLD market share as horizontal bar chart with percentages."""
    tld_counts = df["tld"].value_counts()
    total = tld_counts.sum()
    top = tld_counts.head(top_n)
    pct = (top / total * 100).round(1)

    fig = go.Figure(
        go.Bar(
            x=pct.values,
            y=pct.index,
            orientation="h",
            marker_color=PRIMARY_BLUE,
            text=[f"{v}%" for v in pct.values],
            textposition="outside",
            textfont_size=11,
        )
    )
    fig.update_layout(
        title_text=_title(config, "TLD market share"),
        xaxis_title="Share (%)",
        yaxis_title=None,
        yaxis={"categoryorder": "total ascending"},
        height=max(350, top_n * 32 + 100),
    )
    _apply_base_layout(fig, config)
    _maybe_save(fig, config)
    return fig


def plot_price_by_tld(
    df: pd.DataFrame,
    top_n: int = 10,
    config: PlotConfig | None = None,
) -> go.Figure:
    """Plot median price by TLD."""
    top_tlds = df["tld"].value_counts().head(top_n).index
    medians = (
        df[df["tld"].isin(top_tlds)]
        .groupby("tld")["final_price"]
        .median()
        .sort_values(ascending=False)
    )

    fig = go.Figure(
        go.Bar(
            x=medians.index,
            y=medians.values,
            marker_color=LIGHT_BLUE,
        )
    )
    fig.update_layout(
        title_text=_title(config, f"Median price by TLD (top {top_n})"),
        xaxis_title=None,
        yaxis_title="Median price (USD)",
        xaxis_tickangle=-45,
    )
    _apply_base_layout(fig, config)
    _maybe_save(fig, config)
    return fig


def plot_country_distribution(
    df: pd.DataFrame,
    top_n: int = 10,
    config: PlotConfig | None = None,
) -> go.Figure:
    """Plot traffic country market share as horizontal bar chart with percentages."""
    country_counts = df["country"].dropna().value_counts()
    total = country_counts.sum()
    top = country_counts.head(top_n)
    pct = (top / total * 100).round(1)

    fig = go.Figure(
        go.Bar(
            x=pct.values,
            y=pct.index,
            orientation="h",
            marker_color=LIGHT_BLUE,
            text=[f"{v}%" for v in pct.values],
            textposition="outside",
            textfont_size=11,
        )
    )
    fig.update_layout(
        title_text=_title(config, "Traffic country market share"),
        xaxis_title="Share (%)",
        yaxis_title=None,
        yaxis={"categoryorder": "total ascending"},
        height=max(350, top_n * 32 + 100),
    )
    _apply_base_layout(fig, config)
    _maybe_save(fig, config)
    return fig


def plot_missing_values(
    df: pd.DataFrame,
    config: PlotConfig | None = None,
) -> go.Figure:
    """Plot missing value percentages per column."""
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values()
    missing_pct = missing_pct[missing_pct > 0]

    fig = go.Figure(
        go.Bar(
            x=missing_pct.values,
            y=missing_pct.index,
            orientation="h",
            marker_color=GRAY_MID,
            text=[f"{v:.1f}%" for v in missing_pct.values],
            textposition="outside",
            textfont_size=10,
        )
    )
    fig.update_layout(
        title_text=_title(config, "Missing values by column"),
        xaxis_title="Missing (%)",
        yaxis_title=None,
        height=max(350, len(missing_pct) * 35 + 100),
    )
    _apply_base_layout(fig, config)
    _maybe_save(fig, config)
    return fig
