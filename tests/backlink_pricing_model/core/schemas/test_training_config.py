"""Tests for training config Pydantic models."""

import pytest
from pydantic import ValidationError

from backlink_pricing_model.core.schemas.training_config import TrainingConfig


def test_training_config_defaults() -> None:
    cfg = TrainingConfig()
    assert cfg.test_size == pytest.approx(0.2)
    assert cfg.random_state == 42
    assert cfg.n_optuna_trials == 100
    assert cfg.cv_folds == 5
    assert cfg.early_stopping_rounds == 50


def test_training_config_overrides() -> None:
    cfg = TrainingConfig(test_size=0.3, n_optuna_trials=10)
    assert cfg.test_size == pytest.approx(0.3)
    assert cfg.n_optuna_trials == 10


def test_training_config_type_coercion_failure() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(cv_folds="five")  # type: ignore[arg-type]
