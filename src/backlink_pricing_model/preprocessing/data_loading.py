"""Load raw backlink data from CSV or Parquet sources."""

from pathlib import Path

import pandas as pd

from backlink_pricing_model.core.environment import get_project_root


# Columns used for model training.
TRAINING_COLUMNS: list[str] = [
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
]

# Dtype mapping for consistent loading.
COLUMN_DTYPES: dict[str, str] = {
    "id": "Int64",
    "final_price": "Float64",
    "dr": "Float64",
    "cf": "Float64",
    "tf": "Float64",
    "domain_traffic": "Float64",
    "domain": "string",
    "country": "string",
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
        FileNotFoundError: If neither Parquet nor same-named CSV exists.
    """
    path = get_project_root() / "data" / "raw" / filename
    if path.exists():
        return pd.read_parquet(path, engine="pyarrow")

    # Graceful fallback: if parquet is missing but CSV exists, load CSV.
    csv_fallback = path.with_suffix(".csv")
    if csv_fallback.exists():
        return pd.read_csv(
            csv_fallback,
            dtype=COLUMN_DTYPES,
            parse_dates=["date_received"],
        )

    msg = (
        f"Data file not found: {path} (or {csv_fallback}). "
        "Run `python -m scripts.data_pipeline.main` first, "
        "or place raw backlinks data in data/raw/."
    )
    raise FileNotFoundError(msg)


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


def domain_grouped_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    domain_col: str = "domain",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by domain to prevent leakage.

    All records for a given domain go entirely into one split.
    This prevents the model from memorizing domain-specific pricing
    during training and then being evaluated on the same domain.

    Args:
        df: Input DataFrame.
        test_size: Fraction of domains for test set.
        val_size: Fraction of domains for validation set.
        random_state: Random seed for reproducibility.
        domain_col: Column containing domain identifiers.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    import logging

    import numpy as np

    logger = logging.getLogger(__name__)

    domains = df[domain_col].fillna("unknown_domain").unique()
    rng = np.random.RandomState(random_state)
    rng.shuffle(domains)

    n_test = int(len(domains) * test_size)
    n_val = int(len(domains) * val_size)

    test_domains = set(domains[:n_test])
    val_domains = set(domains[n_test : n_test + n_val])
    train_domains = set(domains[n_test + n_val :])

    domain_values = df[domain_col].fillna("unknown_domain")
    train_df = df[domain_values.isin(train_domains)].copy()
    val_df = df[domain_values.isin(val_domains)].copy()
    test_df = df[domain_values.isin(test_domains)].copy()

    logger.info(
        "Domain-grouped split: %d domains total -> "
        "train %d domains (%d rows) | "
        "val %d domains (%d rows) | "
        "test %d domains (%d rows)",
        len(domains),
        len(train_domains),
        len(train_df),
        len(val_domains),
        len(val_df),
        len(test_domains),
        len(test_df),
    )

    # Verify zero overlap.
    train_d = set(train_df[domain_col].fillna("unknown_domain"))
    val_d = set(val_df[domain_col].fillna("unknown_domain"))
    test_d = set(test_df[domain_col].fillna("unknown_domain"))
    assert len(train_d & test_d) == 0, "Domain leakage: train/test overlap"
    assert len(train_d & val_d) == 0, "Domain leakage: train/val overlap"
    assert len(val_d & test_d) == 0, "Domain leakage: val/test overlap"

    return train_df, val_df, test_df


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
