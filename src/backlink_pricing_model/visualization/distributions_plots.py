"""Distribution plots for price, quality metrics, and traffic."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backlink_pricing_model.visualization.plots_style import (
    CATEGORICAL_PALETTE,
    COLORS,
    apply_default_style,
)


def plot_price_distribution(
    df: pd.DataFrame, log_scale: bool = True
) -> go.Figure:
    """Plot the distribution of backlink prices.

    Args:
        df: DataFrame with 'final_price' column.
        log_scale: Whether to use log scale on x-axis.

    Returns:
        Plotly figure.
    """
    col = "log_price" if log_scale and "log_price" in df.columns else "final_price"
    title = "Backlink price distribution (log scale)" if log_scale else "Backlink price distribution"

    fig = px.histogram(
        df,
        x=col,
        nbins=80,
        title=title,
        labels={col: "Price (USD)" if not log_scale else "Log price"},
        color_discrete_sequence=[COLORS["primary"]],
    )
    return apply_default_style(fig)


def plot_metric_distributions(df: pd.DataFrame) -> go.Figure:
    """Plot distributions of DR, CF, and TF as subplots.

    Args:
        df: DataFrame with dr, cf, tf columns.

    Returns:
        Plotly figure with three subplots.
    """
    metrics = [m for m in ["dr", "cf", "tf"] if m in df.columns]
    fig = make_subplots(
        rows=1,
        cols=len(metrics),
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

    fig.update_layout(title="Quality metric distributions")
    return apply_default_style(fig)


def plot_price_by_quality_tier(
    df: pd.DataFrame, metric: str = "dr"
) -> go.Figure:
    """Plot price distribution across quality tiers.

    Args:
        df: DataFrame with final_price and the metric column.
        metric: Quality metric to use for tiers (dr, cf, tf).

    Returns:
        Plotly box plot figure.
    """
    data = df.dropna(subset=[metric, "final_price"]).copy()
    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0-19", "20-39", "40-59", "60-79", "80-100"]
    data["tier"] = pd.cut(data[metric], bins=bins, labels=labels, right=False)

    fig = px.box(
        data,
        x="tier",
        y="final_price",
        title=f"Price by {metric.upper()} tier",
        labels={"tier": f"{metric.upper()} tier", "final_price": "Price (USD)"},
        color="tier",
        color_discrete_sequence=CATEGORICAL_PALETTE,
    )
    fig.update_layout(showlegend=False)
    return apply_default_style(fig)


def plot_tld_distribution(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Plot top TLD distribution by count.

    Args:
        df: DataFrame with 'tld' column.
        top_n: Number of top TLDs to show.

    Returns:
        Plotly bar chart figure.
    """
    tld_counts = df["tld"].value_counts().head(top_n).reset_index()
    tld_counts.columns = ["tld", "count"]

    fig = px.bar(
        tld_counts,
        x="tld",
        y="count",
        title=f"Top {top_n} TLD distribution",
        labels={"tld": "TLD", "count": "Count"},
        color_discrete_sequence=[COLORS["primary"]],
    )
    return apply_default_style(fig)


def plot_price_by_tld(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Plot median price by TLD.

    Args:
        df: DataFrame with 'tld' and 'final_price' columns.
        top_n: Number of top TLDs to include.

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

    fig = px.bar(
        medians,
        x="tld",
        y="median_price",
        title=f"Median price by TLD (top {top_n})",
        labels={"tld": "TLD", "median_price": "Median price (USD)"},
        color_discrete_sequence=[COLORS["secondary"]],
    )
    return apply_default_style(fig)


def plot_country_distribution(
    df: pd.DataFrame, top_n: int = 15
) -> go.Figure:
    """Plot top country distribution by count.

    Args:
        df: DataFrame with 'country' column.
        top_n: Number of top countries to show.

    Returns:
        Plotly bar chart figure.
    """
    country_counts = (
        df["country"].dropna().value_counts().head(top_n).reset_index()
    )
    country_counts.columns = ["country", "count"]

    fig = px.bar(
        country_counts,
        x="country",
        y="count",
        title=f"Top {top_n} countries by listing count",
        labels={"country": "Country", "count": "Count"},
        color_discrete_sequence=[COLORS["accent"]],
    )
    return apply_default_style(fig)


def plot_missing_values(df: pd.DataFrame) -> go.Figure:
    """Plot missing value percentages per column.

    Args:
        df: Input DataFrame.

    Returns:
        Plotly horizontal bar chart figure.
    """
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values()
    missing_pct = missing_pct[missing_pct > 0]

    fig = px.bar(
        x=missing_pct.values,
        y=missing_pct.index,
        orientation="h",
        title="Missing values by column",
        labels={"x": "Missing (%)", "y": "Column"},
        color_discrete_sequence=[COLORS["negative"]],
    )
    return apply_default_style(fig)
