"""Pydantic models for analysis results."""

from pydantic import BaseModel, Field


class FeatureImportance(BaseModel):
    """Feature importance result from model analysis."""

    feature: str = Field(description="Feature name")
    importance: float = Field(description="Importance score")
    rank: int = Field(description="Rank by importance")


class ModelMetrics(BaseModel):
    """Regression model evaluation metrics."""

    mae: float = Field(description="Mean absolute error")
    rmse: float = Field(description="Root mean squared error")
    r2: float = Field(description="R-squared score")
    mape: float = Field(description="Mean absolute percentage error")
