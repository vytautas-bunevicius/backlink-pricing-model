"""Tests for data imputation strategies."""

import numpy as np
import pandas as pd
import pytest

from backlink_pricing_model.preprocessing import data_imputation as di


def test_fit_records_domain_and_global_medians() -> None:
    train = pd.DataFrame({
        "domain": ["a.com", "a.com", "b.com"],
        "cf": [10.0, 20.0, 40.0],
        "tf": [1.0, 3.0, np.nan],
    })
    imputer = di.fit_domain_metric_imputer(train)
    assert imputer["cf"]["domain_medians"]["a.com"] == pytest.approx(15.0)
    assert imputer["cf"]["global_median"] == pytest.approx(20.0)
    # tf for b.com is NaN -> not stored as a domain median.
    assert "b.com" not in imputer["tf"]["domain_medians"]


def test_apply_uses_domain_then_global_fallback() -> None:
    train = pd.DataFrame({
        "domain": ["a.com", "a.com", "b.com"],
        "cf": [10.0, 20.0, 40.0],
        "tf": [1.0, 3.0, 5.0],
    })
    imputer = di.fit_domain_metric_imputer(train)
    new = pd.DataFrame({
        "domain": ["a.com", "unseen.com"],
        "cf": [np.nan, np.nan],
        "tf": [np.nan, np.nan],
    })
    out = di.apply_domain_metric_imputer(new, imputer)
    # Known domain -> its median; unseen -> global median.
    assert out["cf"].iloc[0] == pytest.approx(15.0)
    assert out["cf"].iloc[1] == pytest.approx(imputer["cf"]["global_median"])
    assert out["cf"].isna().sum() == 0


def test_apply_does_not_mutate_input() -> None:
    train = pd.DataFrame({"domain": ["a.com"], "cf": [10.0], "tf": [2.0]})
    imputer = di.fit_domain_metric_imputer(train)
    new = pd.DataFrame({"domain": ["a.com"], "cf": [np.nan], "tf": [np.nan]})
    di.apply_domain_metric_imputer(new, imputer)
    assert new["cf"].isna().all()


def test_round_trip_via_impute_metrics_by_domain() -> None:
    df = pd.DataFrame({
        "domain": ["a.com", "a.com", "b.com"],
        "cf": [10.0, np.nan, 40.0],
        "tf": [np.nan, 3.0, 5.0],
    })
    out = di.impute_metrics_by_domain(df)
    assert out["cf"].isna().sum() == 0
    assert out["tf"].isna().sum() == 0


def test_drop_rows_missing_target() -> None:
    df = pd.DataFrame({"final_price": [100.0, np.nan, 200.0]})
    out = di.drop_rows_missing_target(df)
    assert len(out) == 2
    assert out["final_price"].notna().all()


def test_summarize_imputation_reports_only_prior_missing() -> None:
    before = pd.DataFrame({"cf": [np.nan, 2.0], "dr": [1.0, 2.0]})
    after = pd.DataFrame({"cf": [3.0, 2.0], "dr": [1.0, 2.0]})
    summary = di.summarize_imputation(before, after)
    assert "cf" in summary.index
    assert "dr" not in summary.index  # never missing
    assert summary.loc["cf", "filled"] == 1
