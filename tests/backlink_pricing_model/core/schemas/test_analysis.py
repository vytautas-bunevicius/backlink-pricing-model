"""Tests for analysis Pydantic models."""

import pytest
from pydantic import ValidationError

from backlink_pricing_model.core.schemas.analysis import (
    FeatureImportance,
    ModelMetrics,
)


def test_feature_importance_valid_instance() -> None:
    fi = FeatureImportance(feature="dr", importance=0.42, rank=1)
    assert fi.feature == "dr"
    assert fi.importance == pytest.approx(0.42)
    assert fi.rank == 1


def test_feature_importance_missing_field_raises() -> None:
    with pytest.raises(ValidationError):
        FeatureImportance(feature="dr", importance=0.1)  # type: ignore[call-arg]


def test_feature_importance_importance_coerced_to_float() -> None:
    fi = FeatureImportance(feature="dr", importance=1, rank=2)
    assert isinstance(fi.importance, float)


def test_model_metrics_valid_instance() -> None:
    m = ModelMetrics(mae=1.0, rmse=2.0, r2=0.9, mape=5.0)
    assert m.rmse == pytest.approx(2.0)
    assert m.r2 == pytest.approx(0.9)


def test_model_metrics_non_numeric_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelMetrics(mae="bad", rmse=2.0, r2=0.9, mape=5.0)  # type: ignore[arg-type]
