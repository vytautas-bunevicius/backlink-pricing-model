"""Entry point for the Supabase data extraction pipeline.

Extracts all backlink records from Supabase in batches and saves them
as both CSV and Parquet files in data/raw/.

Usage:
    python -m scripts.data_pipeline.main
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from supabase import Client, create_client

from scripts.data_pipeline.models import ExtractionConfig, SupabaseConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_supabase_client(config: SupabaseConfig) -> Client:
    """Create an authenticated Supabase client.

    Args:
        config: Supabase connection settings.

    Returns:
        Authenticated Supabase client.
    """
    return create_client(config.supabase_url, config.supabase_service_role_key)


def extract_backlinks(
    client: Client, config: ExtractionConfig
) -> pd.DataFrame:
    """Extract backlink data from Supabase in paginated batches.

    Args:
        client: Authenticated Supabase client.
        config: Extraction configuration.

    Returns:
        DataFrame containing all extracted backlink records.
    """
    all_rows: list[dict] = []
    offset = 0
    select_cols = ",".join(config.columns)

    while True:
        response = (
            client.table(config.table_name)
            .select(select_cols)
            .range(offset, offset + config.batch_size - 1)
            .execute()
        )

        batch = response.data
        if not batch:
            break

        all_rows.extend(batch)
        offset += config.batch_size
        logger.info(
            "Extracted %d rows (total: %d)", len(batch), len(all_rows)
        )

    logger.info("Extraction complete: %d total rows", len(all_rows))
    return pd.DataFrame(all_rows)


def save_raw_data(df: pd.DataFrame, output_dir: str) -> None:
    """Save extracted data as CSV and Parquet.

    Args:
        df: DataFrame to save.
        output_dir: Target directory for output files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "backlinks.csv"
    parquet_path = output_path / "backlinks.parquet"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    logger.info("Saved CSV: %s (%d rows)", csv_path, len(df))
    logger.info("Saved Parquet: %s (%d rows)", parquet_path, len(df))


def main() -> None:
    """Run the full extraction pipeline."""
    logger.info("Starting backlink data extraction")

    try:
        supabase_config = SupabaseConfig()
    except Exception:
        logger.exception(
            "Failed to load Supabase config. "
            "Ensure .env file exists with SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY."
        )
        sys.exit(1)

    extraction_config = ExtractionConfig()
    client = create_supabase_client(supabase_config)
    df = extract_backlinks(client, extraction_config)

    if df.empty:
        logger.warning("No data extracted. Exiting.")
        sys.exit(1)

    save_raw_data(df, extraction_config.output_dir)
    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
