"""Pydantic models for training configuration."""

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    test_size: float = Field(
        default=0.2,
        description="Fraction of data for test set",
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    n_optuna_trials: int = Field(
        default=100,
        description="Number of Optuna hyperparameter search trials",
    )
    cv_folds: int = Field(
        default=5,
        description="Number of cross-validation folds",
    )
    early_stopping_rounds: int = Field(
        default=50,
        description="Early stopping patience for gradient boosting",
    )
