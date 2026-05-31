"""Tests for distribution plots."""

import pandas as pd
import plotly.graph_objects as go

from backlink_pricing_model.core.schemas.visualization import PlotConfig
from backlink_pricing_model.visualization import distributions_plots as dp


def test_price_distribution_log_and_raw(eda_df: pd.DataFrame) -> None:
    assert isinstance(dp.plot_price_distribution(eda_df), go.Figure)
    assert isinstance(
        dp.plot_price_distribution(eda_df, log_scale=False), go.Figure
    )


def test_metric_distributions(eda_df: pd.DataFrame) -> None:
    fig = dp.plot_metric_distributions(eda_df)
    assert isinstance(fig, go.Figure)
    # One histogram trace per available metric (dr, cf, tf).
    assert len(fig.data) == 3


def test_price_by_quality_tier(eda_df: pd.DataFrame) -> None:
    fig = dp.plot_price_by_quality_tier(eda_df, metric="dr")
    assert isinstance(fig, go.Figure)


def test_tld_plots(eda_df: pd.DataFrame) -> None:
    assert isinstance(dp.plot_tld_distribution(eda_df), go.Figure)
    assert isinstance(dp.plot_price_by_tld(eda_df), go.Figure)


def test_country_distribution(eda_df: pd.DataFrame) -> None:
    assert isinstance(dp.plot_country_distribution(eda_df), go.Figure)


def test_missing_values() -> None:
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    assert isinstance(dp.plot_missing_values(df), go.Figure)


def test_custom_title_applied(eda_df: pd.DataFrame) -> None:
    fig = dp.plot_price_distribution(
        eda_df, config=PlotConfig(title="My title")
    )
    assert fig.layout.title.text == "My title"
