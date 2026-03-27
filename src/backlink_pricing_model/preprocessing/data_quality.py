"""Data quality checks: missing values, outliers, range validation."""

import logging

import pandas as pd


logger = logging.getLogger(__name__)


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a missing value report for all columns.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with columns: column, missing_count, missing_pct, dtype.
    """
    total = len(df)
    missing = df.isnull().sum()
    report = pd.DataFrame(
        {
            "column": missing.index,
            "missing_count": missing.values,
            "missing_pct": (missing.values / total * 100).round(2),
            "dtype": df.dtypes.values,
        }
    )
    return report.sort_values("missing_pct", ascending=False).reset_index(
        drop=True
    )


def filter_valid_prices(
    df: pd.DataFrame,
    min_price: float = 0.0,
    max_price: float = 50_000.0,
) -> pd.DataFrame:
    """Filter rows to valid price range.

    Args:
        df: Input DataFrame with final_price column.
        min_price: Minimum price (exclusive).
        max_price: Maximum price (inclusive).

    Returns:
        Filtered DataFrame.
    """
    initial_count = len(df)
    filtered = df[
        (df["final_price"] > min_price) & (df["final_price"] <= max_price)
    ].copy()
    removed = initial_count - len(filtered)
    logger.info(
        "Price filter: removed %d rows (%.1f%%), %d remaining",
        removed,
        removed / initial_count * 100 if initial_count > 0 else 0,
        len(filtered),
    )
    return filtered


def validate_metric_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that quality metrics are within expected ranges.

    Clips DR, CF, TF to [0, 100] and logs any out-of-range values.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with clipped metric values.
    """
    result = df.copy()
    for col in ["dr", "cf", "tf"]:
        if col not in result.columns:
            continue
        out_of_range = (
            (result[col] < 0) | (result[col] > 100)
        ) & result[col].notna()
        if out_of_range.any():
            count = out_of_range.sum()
            logger.warning(
                "Column '%s': %d values outside [0, 100], clipping", col, count
            )
            result[col] = result[col].clip(0, 100)
    return result


def detect_outliers_iqr(
    series: pd.Series, multiplier: float = 1.5
) -> pd.Series:
    """Detect outliers using the IQR method.

    Args:
        series: Numeric series to check.
        multiplier: IQR multiplier for fence calculation.

    Returns:
        Boolean series where True indicates an outlier.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (series < lower) | (series > upper)
