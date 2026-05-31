"""Tests for data quality checks."""

import numpy as np
import pandas as pd
import pytest

from backlink_pricing_model.preprocessing import data_quality as dq


def test_missing_value_report_counts_and_pct_sorted_desc() -> None:
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [1, None, None, None],
        "c": [None, None, 3, 4],
    })
    report = dq.missing_value_report(df)
    assert set(report.columns) == {
        "column",
        "missing_count",
        "missing_pct",
        "dtype",
    }
    # Most-missing column first.
    assert report.iloc[0]["column"] == "b"
    assert report.iloc[0]["missing_count"] == 3
    assert report.iloc[0]["missing_pct"] == pytest.approx(75.0)


def test_filter_valid_prices_min_exclusive_max_inclusive() -> None:
    df = pd.DataFrame({"final_price": [0.0, 1.0, 50_000.0, 50_001.0]})
    out = dq.filter_valid_prices(df)
    # 0 excluded (not > 0), 50001 excluded (> max), 1 and 50000 kept.
    assert sorted(out["final_price"]) == [1.0, 50_000.0]


def test_filter_valid_prices_custom_bounds() -> None:
    df = pd.DataFrame({"final_price": [5.0, 10.0, 15.0]})
    out = dq.filter_valid_prices(df, min_price=5.0, max_price=10.0)
    assert sorted(out["final_price"]) == [10.0]


def test_validate_metric_ranges_clips_out_of_range() -> None:
    df = pd.DataFrame({
        "dr": [-5.0, 50.0, 150.0],
        "cf": [10.0, 20.0, 30.0],
        "tf": [np.nan, 0.0, 100.0],
    })
    out = dq.validate_metric_ranges(df)
    assert list(out["dr"]) == [0.0, 50.0, 100.0]
    # In-range columns untouched; NaN preserved.
    assert list(out["cf"]) == [10.0, 20.0, 30.0]
    assert np.isnan(out["tf"].iloc[0])


def test_validate_metric_ranges_skips_missing_columns() -> None:
    df = pd.DataFrame({"dr": [50.0]})
    out = dq.validate_metric_ranges(df)
    assert list(out["dr"]) == [50.0]


def test_validate_metric_ranges_does_not_mutate_input() -> None:
    df = pd.DataFrame({"dr": [150.0]})
    dq.validate_metric_ranges(df)
    assert df["dr"].iloc[0] == pytest.approx(150.0)


def test_detect_outliers_iqr_flags_extreme_value() -> None:
    series = pd.Series([10, 11, 12, 13, 14, 1000])
    mask = dq.detect_outliers_iqr(series)
    assert mask.iloc[-1]
    assert not mask.iloc[:-1].any()


def test_detect_outliers_iqr_multiplier_widens_fence() -> None:
    series = pd.Series([10, 11, 12, 13, 14, 40])
    strict = dq.detect_outliers_iqr(series, multiplier=1.5)
    loose = dq.detect_outliers_iqr(series, multiplier=10.0)
    assert strict.sum() >= loose.sum()
