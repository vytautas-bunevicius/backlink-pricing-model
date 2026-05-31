"""Tests for feature importance plots."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from backlink_pricing_model.visualization import importance_plots as ip


def test_plot_feature_importance_returns_figure() -> None:
    fig = ip.plot_feature_importance(["a", "b", "c"], np.array([0.1, 0.5, 0.4]))
    assert isinstance(fig, go.Figure)


def test_plot_feature_importance_truncates_to_top_n() -> None:
    names = [f"f{i}" for i in range(10)]
    fig = ip.plot_feature_importance(names, np.arange(10.0), top_n=3)
    # Horizontal bar: one y entry per shown feature.
    assert len(fig.data[0].y) == 3


def test_plot_correlation_heatmap_returns_figure(
    corr_matrix: pd.DataFrame,
) -> None:
    assert isinstance(ip.plot_correlation_heatmap(corr_matrix), go.Figure)


def test_extract_high_correlations_finds_pairs(
    corr_matrix: pd.DataFrame,
) -> None:
    out = ip.extract_high_correlations(corr_matrix, threshold=0.8)
    assert list(out.columns) == ["feature_1", "feature_2", "pearson_r"]
    assert len(out) == 1
    assert out.iloc[0]["pearson_r"] == pytest.approx(0.9)


def test_extract_high_correlations_empty_result(
    corr_matrix: pd.DataFrame,
) -> None:
    out = ip.extract_high_correlations(corr_matrix, threshold=0.99)
    assert out.empty
    assert list(out.columns) == ["feature_1", "feature_2", "pearson_r"]


def test_extract_high_correlations_invalid_threshold(
    corr_matrix: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        ip.extract_high_correlations(corr_matrix, threshold=1.5)


def test_extract_high_correlations_empty_matrix() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        ip.extract_high_correlations(pd.DataFrame())


def test_extract_high_correlations_respects_top_n() -> None:
    cols = ["a", "b", "c", "d"]
    mat = pd.DataFrame(np.full((4, 4), 0.95), index=cols, columns=cols)
    out = ip.extract_high_correlations(mat, threshold=0.9, top_n=2)
    assert len(out) == 2
