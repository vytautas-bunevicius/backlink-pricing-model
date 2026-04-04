"""Pydantic models for the data extraction pipeline."""

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings


class SupabaseConfig(BaseSettings):
    """Supabase connection settings loaded from environment."""

    database_url: str = Field(
        description="Supabase project URL",
        validation_alias=AliasChoices("DATABASE_URL", "SUPABASE_URL"),
    )
    supabase_service_role_key: str = Field(
        description="Supabase service role key"
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


class ExtractionConfig(BaseModel):
    """Configuration for data extraction."""

    table_name: str = Field(
        default="backlinks",
        description="Source table name",
    )
    columns: list[str] = Field(
        default=[
            "id",
            "domain",
            "final_price",
            "dr",
            "cf",
            "tf",
            "domain_traffic",
            "country",
            "link_source_type",
            "date_received",
            "status",
        ],
        description="Columns to extract",
    )
    batch_size: int = Field(
        default=1000,
        description="Rows per batch for paginated extraction",
    )
    output_dir: str = Field(
        default="data/raw",
        description="Output directory for extracted data",
    )
