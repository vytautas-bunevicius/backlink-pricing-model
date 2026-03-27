"""Missing value imputation strategies for backlink features."""

import logging

import pandas as pd


logger = logging.getLogger(__name__)


def impute_metrics_by_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing CF/TF using domain-level averages.

    When a domain has multiple records with known CF/TF values, use
    the domain median to fill gaps.

    Args:
        df: DataFrame with domain, cf, tf columns.

    Returns:
        DataFrame with imputed values where possible.
    """
    result = df.copy()

    for col in ["cf", "tf"]:
        if col not in result.columns:
            continue
        before = result[col].isnull().sum()
        domain_medians = result.groupby("domain")[col].transform("median")
        result[col] = result[col].fillna(domain_medians)
        after = result[col].isnull().sum()
        filled = before - after
        if filled > 0:
            logger.info(
                "Imputed %d missing '%s' values using domain medians",
                filled,
                col,
            )

    return result


def drop_rows_missing_target(
    df: pd.DataFrame, target: str = "final_price"
) -> pd.DataFrame:
    """Drop rows where the target variable is missing.

    Args:
        df: Input DataFrame.
        target: Target column name.

    Returns:
        DataFrame with non-null target values.
    """
    before = len(df)
    result = df.dropna(subset=[target])
    dropped = before - len(result)
    if dropped > 0:
        logger.info("Dropped %d rows with missing '%s'", dropped, target)
    return result


def summarize_imputation(
    before: pd.DataFrame, after: pd.DataFrame
) -> pd.DataFrame:
    """Compare missing value counts before and after imputation.

    Args:
        before: DataFrame before imputation.
        after: DataFrame after imputation.

    Returns:
        Summary DataFrame with before/after/filled counts per column.
    """
    before_missing = before.isnull().sum()
    after_missing = after.isnull().sum()
    return pd.DataFrame(
        {
            "before": before_missing,
            "after": after_missing,
            "filled": before_missing - after_missing,
        }
    ).query("before > 0")
