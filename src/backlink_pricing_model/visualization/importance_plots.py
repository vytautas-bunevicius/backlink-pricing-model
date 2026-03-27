"""Feature importance and SHAP visualization plots."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from backlink_pricing_model.visualization.plots_style import (
    COLORS,
    apply_default_style,
)


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 20,
    title: str = "Feature importance",
) -> go.Figure:
    """Plot feature importance as a horizontal bar chart.

    Args:
        feature_names: List of feature names.
        importances: Array of importance scores.
        top_n: Number of top features to show.
        title: Plot title.

    Returns:
        Plotly figure.
    """
    df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=True)

    if len(df) > top_n:
        df = df.tail(top_n)

    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title=title,
        labels={"importance": "Importance", "feature": "Feature"},
        color_discrete_sequence=[COLORS["primary"]],
    )
    return apply_default_style(fig)


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame, title: str = "Feature correlation matrix"
) -> go.Figure:
    """Plot a correlation heatmap.

    Args:
        corr_matrix: Correlation matrix DataFrame.
        title: Plot title.

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
    fig.update_layout(title=title)
    return apply_default_style(fig, height=700, width=800)
