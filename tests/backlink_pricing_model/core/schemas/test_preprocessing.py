"""Tests for preprocessing Pydantic models."""

import pytest

from backlink_pricing_model.core.schemas.preprocessing import (
    DataLoadingConfig,
    QualityTierConfig,
)


def test_quality_tier_config_defaults() -> None:
    cfg = QualityTierConfig(metric="dr")
    assert cfg.boundaries == [0, 20, 40, 60, 80, 100]
    assert cfg.labels == ["very_low", "low", "medium", "high", "premium"]
    # One fewer label than boundary (bins between boundaries).
    assert len(cfg.labels) == len(cfg.boundaries) - 1


def test_quality_tier_config_override_boundaries() -> None:
    cfg = QualityTierConfig(metric="tf", boundaries=[0, 50, 100])
    assert cfg.boundaries == [0, 50, 100]


def test_data_loading_config_defaults() -> None:
    cfg = DataLoadingConfig()
    assert cfg.min_price == pytest.approx(0.0)
    assert cfg.max_price == pytest.approx(50_000.0)
    assert "final_price" in cfg.required_columns


def test_data_loading_config_override_prices() -> None:
    cfg = DataLoadingConfig(min_price=10.0, max_price=999.0)
    assert cfg.min_price == pytest.approx(10.0)
    assert cfg.max_price == pytest.approx(999.0)
