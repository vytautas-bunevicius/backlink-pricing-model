"""Tests for model evaluation plots."""

import numpy as np
import plotly.graph_objects as go

from backlink_pricing_model.visualization import models_plots as mp


def test_predictions_vs_actuals() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    fig = mp.plot_predictions_vs_actuals(y_true, y_pred)
    assert isinstance(fig, go.Figure)
    # Scatter of predictions plus the perfect-prediction reference line.
    assert len(fig.data) == 2


def test_residuals() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    assert isinstance(mp.plot_residuals(y_true, y_pred), go.Figure)


def test_model_comparison() -> None:
    metrics = {
        "ridge": {"rmse": 1.2, "mae": 0.9},
        "xgboost": {"rmse": 0.8, "mae": 0.6},
    }
    fig = mp.plot_model_comparison(metrics, metric_name="rmse")
    assert isinstance(fig, go.Figure)
