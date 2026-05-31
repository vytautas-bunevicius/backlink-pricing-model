"""Tests for feature engineering transforms."""

import numpy as np
import pandas as pd
import pytest

from backlink_pricing_model.preprocessing import feature_engineering as fe


@pytest.mark.parametrize(
    ("domain", "expected"),
    [
        ("example.com", "com"),
        ("sub.example.com", "com"),
        ("businesstask.co.uk", "co.uk"),
        ("shop.com.au", "com.au"),
        ("https://example.io/path/page", "io"),
        ("HTTP://Example.IO", "io"),
        ("blog.example.org.uk", "org.uk"),
    ],
)
def test_extract_tld_known_domains(domain: str, expected: str) -> None:
    assert fe.extract_tld(domain) == expected


@pytest.mark.parametrize("value", ["", "   ", "localhost", None, 42, np.nan])
def test_extract_tld_unparseable_returns_unknown(value: object) -> None:
    assert fe.extract_tld(value) == "unknown"  # type: ignore[arg-type]


def test_add_tld_feature_adds_column() -> None:
    df = pd.DataFrame({"domain": ["a.com", "b.co.uk", "c.io"]})
    out = fe.add_tld_feature(df)
    assert list(out["tld"]) == ["com", "co.uk", "io"]


def test_add_tld_feature_does_not_mutate_input() -> None:
    df = pd.DataFrame({"domain": ["a.com"]})
    fe.add_tld_feature(df)
    assert "tld" not in df.columns


def test_normalize_country_maps_aliases_and_iso_codes() -> None:
    df = pd.DataFrame({"country": ["United States", "germany", "GB", "fr"]})
    out = fe.normalize_country(df)
    assert list(out["country"]) == ["US", "DE", "GB", "FR"]


def test_normalize_country_unmappable_becomes_none() -> None:
    df = pd.DataFrame({"country": ["Narnia", "", None]})
    out = fe.normalize_country(df)
    assert out["country"].isna().all()


def test_normalize_link_source_known_alias_and_passthrough() -> None:
    df = pd.DataFrame({
        "link_source_type": ["Agency", "agnecy", "custom_source"]
    })
    out = fe.normalize_link_source_type(df)
    assert list(out["link_source_type"]) == [
        "agency",
        "agency",
        "custom_source",
    ]


def test_normalize_link_source_json_like_and_blank_become_none() -> None:
    df = pd.DataFrame({"link_source_type": ["[1, 2]", "  ", None]})
    out = fe.normalize_link_source_type(df)
    assert out["link_source_type"].isna().all()


def test_add_log_price_is_log1p() -> None:
    df = pd.DataFrame({"final_price": [0.0, 99.0]})
    out = fe.add_log_price(df)
    assert out["log_price"].iloc[0] == pytest.approx(0.0)
    assert out["log_price"].iloc[1] == pytest.approx(np.log1p(99.0))


def test_add_log_traffic_handles_nan_and_negatives() -> None:
    df = pd.DataFrame({"domain_traffic": [np.nan, -5.0, 100.0]})
    out = fe.add_log_traffic(df)
    assert out["log_traffic"].iloc[0] == pytest.approx(0.0)  # nan -> 0
    assert out["log_traffic"].iloc[1] == pytest.approx(0.0)  # neg clipped to 0
    assert out["log_traffic"].iloc[2] == pytest.approx(np.log1p(100.0))


def test_add_temporal_features_extracts_year_month_quarter() -> None:
    df = pd.DataFrame({"date_received": ["2024-07-15", "2023-01-01"]})
    out = fe.add_temporal_features(df)
    assert list(out["year"]) == [2024, 2023]
    assert list(out["month"]) == [7, 1]
    assert list(out["quarter"]) == [3, 1]


def test_add_missingness_flags_only_existing_columns() -> None:
    df = pd.DataFrame({"cf": [1.0, np.nan], "tf": [np.nan, 2.0]})
    out = fe.add_missingness_flags(df, columns=("cf", "tf", "absent"))
    assert list(out["cf_missing_flag"]) == [0, 1]
    assert list(out["tf_missing_flag"]) == [1, 0]
    assert "absent_missing_flag" not in out.columns


def test_add_interaction_features_creates_expected_columns() -> None:
    df = pd.DataFrame({
        "dr": [10.0, np.nan],
        "cf": [4.0, 2.0],
        "tf": [2.0, 0.0],
        "log_traffic": [3.0, 0.0],
    })
    out = fe.add_interaction_features(df)
    assert out["dr_x_cf"].iloc[0] == pytest.approx(40.0)
    assert out["dr_squared"].iloc[0] == pytest.approx(100.0)
    assert out["dr_x_cf"].iloc[1] == pytest.approx(0.0)  # nan dr filled with 0
    # tf == 0 must not divide-by-zero.
    assert out["cf_tf_ratio"].iloc[1] == pytest.approx(0.0)
    assert out["dr_per_log_traffic"].iloc[1] == pytest.approx(0.0)
    assert out["cf_tf_ratio"].iloc[0] == pytest.approx(2.0)


def test_add_domain_frequency_counts_repeat_domains() -> None:
    df = pd.DataFrame({"domain": ["a.com", "a.com", "b.com"]})
    out = fe.add_domain_frequency(df)
    assert list(out["domain_freq"]) == [2, 2, 1]
    assert out["log_domain_freq"].iloc[0] == pytest.approx(np.log1p(2))


def test_group_rare_tld_collapses_below_min_count() -> None:
    df = pd.DataFrame({"tld": ["com"] * 3 + ["io"] + ["xyz"]})
    out = fe.group_rare_tld(df, min_count=2)
    assert set(out["tld_grouped"]) == {"com", "other"}


def test_group_rare_country_uses_unknown_fill() -> None:
    df = pd.DataFrame({"country": ["US"] * 3 + [None, "ZZ"]})
    out = fe.group_rare_country(df, min_count=2)
    # US kept, the None->unknown and ZZ singletons collapse to other.
    assert (out["country_grouped"] == "US").sum() == 3
    assert "other" in set(out["country_grouped"])


def test_normalize_link_source_for_modeling_collapses_rare() -> None:
    df = pd.DataFrame({
        "link_source_type": ["outreach"] * 5 + ["rare1", "rare2"]
    })
    out = fe.normalize_link_source_for_modeling(df, min_count=3)
    assert set(out["link_source_type_clean"]) == {"outreach", "other"}


def test_group_rare_tld_without_column_is_noop() -> None:
    df = pd.DataFrame({"x": [1]})
    out = fe.group_rare_tld(df)
    assert "tld_grouped" not in out.columns
