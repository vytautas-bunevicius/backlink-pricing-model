"""Missing value imputation strategies for backlink features."""

import logging
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


MetricImputer = dict[str, dict[str, Any]]


def fit_domain_metric_imputer(
    train_df: pd.DataFrame,
    columns: tuple[str, ...] = ("cf", "tf"),
    domain_col: str = "domain",
) -> MetricImputer:
    """Fit domain-aware imputation stats on the training split only.

    For each metric column, this stores:
    - per-domain medians from the train split
    - a global median fallback for unseen domains

    Args:
        train_df: Training split only.
        columns: Metric columns to impute.
        domain_col: Domain identifier column.

    Returns:
        Dict with per-column imputation statistics.
    """
    imputer: MetricImputer = {}
    domains = train_df[domain_col].fillna("unknown_domain").astype(str)

    for col in columns:
        if col not in train_df.columns:
            continue

        numeric = pd.to_numeric(train_df[col], errors="coerce")
        domain_medians_series = numeric.groupby(domains).median().dropna()
        domain_medians = {
            str(k): float(v) for k, v in domain_medians_series.items()
        }
        global_median = (
            float(numeric.median()) if numeric.notna().any() else None
        )
        imputer[col] = {
            "domain_medians": domain_medians,
            "global_median": global_median,
        }
        logger.info(
            "Fitted imputer for '%s': %d domain medians, global median=%s",
            col,
            len(domain_medians),
            f"{global_median:.4f}" if global_median is not None else "None",
        )

    return imputer


def apply_domain_metric_imputer(
    df: pd.DataFrame,
    imputer: MetricImputer,
    domain_col: str = "domain",
) -> pd.DataFrame:
    """Apply train-fitted domain metric imputation to a split or inference data.

    Args:
        df: Data to impute.
        imputer: Fitted imputer stats from fit_domain_metric_imputer.
        domain_col: Domain identifier column.

    Returns:
        DataFrame with imputed metric columns.
    """
    result = df.copy()
    domains = result[domain_col].fillna("unknown_domain").astype(str)

    for col, stats in imputer.items():
        if col not in result.columns:
            continue

        before = result[col].isnull().sum()
        series = pd.to_numeric(result[col], errors="coerce")

        domain_map = stats.get("domain_medians", {})
        if domain_map:
            series = series.fillna(domains.map(domain_map))

        global_median = stats.get("global_median")
        if global_median is not None:
            series = series.fillna(float(global_median))

        result[col] = series
        after = result[col].isnull().sum()
        filled = before - after
        if filled > 0:
            logger.info(
                "Imputed %d missing '%s' values (domain+global fallback)",
                filled,
                col,
            )

    return result


def impute_metrics_by_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing CF/TF using domain-level averages.

    When a domain has multiple records with known CF/TF values, use
    the domain median to fill gaps.

    Args:
        df: DataFrame with domain, cf, tf columns.

    Returns:
        DataFrame with imputed values where possible.
    """
    fitted = fit_domain_metric_imputer(df, columns=("cf", "tf"), domain_col="domain")
    return apply_domain_metric_imputer(df, fitted, domain_col="domain")


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
