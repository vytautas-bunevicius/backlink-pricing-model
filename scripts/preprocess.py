"""CLI script: preprocess raw data into cleaned, feature-engineered datasets.

Usage:
    python -m scripts.preprocess
    python -m scripts.preprocess --config configs/preprocessing.yaml
"""

import logging

import click
import pandas as pd

from backlink_pricing_model.core.config import load_config, resolve_path
from backlink_pricing_model.preprocessing.data_loading import save_processed
from backlink_pricing_model.preprocessing.data_quality import (
    filter_valid_prices,
    missing_value_report,
    validate_metric_ranges,
)
from backlink_pricing_model.preprocessing.feature_engineering import (
    add_log_price,
    add_log_traffic,
    add_tld_feature,
    normalize_country,
    normalize_link_source_type,
    normalize_link_type,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--config",
    "config_path",
    default="configs/preprocessing.yaml",
    help="Path to preprocessing config YAML.",
)
def main(config_path: str) -> None:
    """Preprocess raw backlink data into a cleaned Parquet file."""
    cfg = load_config(config_path)
    logger.info("Loaded config: %s", config_path)

    # Load raw data.
    raw_path = resolve_path(cfg["raw_data"])
    logger.info("Loading raw data from %s", raw_path)
    df = pd.read_parquet(raw_path, engine="pyarrow")
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # Missing value report.
    report = missing_value_report(df)
    logger.info("Missing values:\n%s", report.to_string(index=False))

    # Filter and validate.
    df = filter_valid_prices(
        df,
        min_price=cfg.get("min_price", 0.0),
        max_price=cfg.get("max_price", 50_000.0),
    )
    df = validate_metric_ranges(df)

    # Normalize categoricals.
    df = normalize_country(df)
    df = normalize_link_type(df)
    df = normalize_link_source_type(df)

    # Derive features.
    df = add_tld_feature(df)
    df = add_log_price(df)
    df = add_log_traffic(df)

    # Save.
    output_dir = cfg.get("output_dir", "data/processed")
    filename = cfg.get("output_filename", "backlinks_cleaned")
    path = save_processed(df, filename, subdir=output_dir.split("/")[-1])
    logger.info("Saved %d rows to %s", len(df), path)


if __name__ == "__main__":
    main()
