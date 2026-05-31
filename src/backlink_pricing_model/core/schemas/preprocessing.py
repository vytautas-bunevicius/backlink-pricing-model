"""Pydantic models for preprocessing configuration."""

from pydantic import BaseModel, Field


class QualityTierConfig(BaseModel):
    """Configuration for quality metric tier boundaries."""

    metric: str = Field(description="Quality metric name (dr, cf, tf)")
    boundaries: list[int] = Field(
        default=[0, 20, 40, 60, 80, 100],
        description="Tier boundary values",
    )
    labels: list[str] = Field(
        default=["very_low", "low", "medium", "high", "premium"],
        description="Human-readable tier labels",
    )


class DataLoadingConfig(BaseModel):
    """Configuration for data loading from source."""

    min_price: float = Field(
        default=0.0,
        description="Minimum price filter (exclude zero/negative)",
    )
    max_price: float = Field(
        default=50_000.0,
        description="Maximum price filter (exclude outliers)",
    )
    required_columns: list[str] = Field(
        default=[
            "final_price",
            "dr",
            "cf",
            "tf",
            "domain_traffic",
            "country",
            "date_received",
        ],
        description="Columns required for model training",
    )
