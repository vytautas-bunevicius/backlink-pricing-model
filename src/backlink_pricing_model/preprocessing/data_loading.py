"""Load raw backlink data from CSV or Parquet sources."""

from pathlib import Path

import pandas as pd

from backlink_pricing_model.core.environment import get_project_root


# Columns used for model training.
TRAINING_COLUMNS: list[str] = [
    "id",
    "domain",
    "final_price",
    "initial_price",
    "dr",
    "cf",
    "tf",
    "domain_traffic",
    "country",
    "link_type",
    "link_source_type",
    "date_received",
    "status",
]

# Dtype mapping for consistent loading.
COLUMN_DTYPES: dict[str, str] = {
    "id": "Int64",
    "final_price": "Float64",
    "initial_price": "Float64",
    "dr": "Float64",
    "cf": "Float64",
    "tf": "Float64",
    "domain_traffic": "Float64",
    "domain": "string",
    "country": "string",
    "link_type": "string",
    "link_source_type": "string",
    "status": "string",
}


def load_raw_parquet(
    filename: str = "backlinks.parquet",
) -> pd.DataFrame:
    """Load raw backlink data from a Parquet file.

    Args:
        filename: Name of the Parquet file in data/raw/.

    Returns:
        DataFrame with typed columns.

    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    path = get_project_root() / "data" / "raw" / filename
    if not path.exists():
        msg = (
            f"Data file not found: {path}. "
            "Run `python -m scripts.data_pipeline.main` first."
        )
        raise FileNotFoundError(msg)
    return pd.read_parquet(path, engine="pyarrow")


def load_raw_csv(
    filename: str = "backlinks.csv",
) -> pd.DataFrame:
    """Load raw backlink data from a CSV file.

    Args:
        filename: Name of the CSV file in data/raw/.

    Returns:
        DataFrame with parsed dates and typed columns.

    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    path = get_project_root() / "data" / "raw" / filename
    if not path.exists():
        msg = (
            f"Data file not found: {path}. "
            "Run `python -m scripts.data_pipeline.main` first."
        )
        raise FileNotFoundError(msg)
    return pd.read_csv(
        path,
        dtype=COLUMN_DTYPES,
        parse_dates=["date_received"],
    )


def save_processed(
    df: pd.DataFrame,
    filename: str,
    subdir: str = "processed",
) -> Path:
    """Save a DataFrame to data/processed/ or data/engineered/.

    Args:
        df: DataFrame to save.
        filename: Output filename (without extension).
        subdir: Subdirectory under data/ (processed or engineered).

    Returns:
        Path to the saved Parquet file.
    """
    output_dir = get_project_root() / "data" / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{filename}.parquet"
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    return parquet_path
