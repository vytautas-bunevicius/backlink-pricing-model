"""Tests for data loading."""

from pathlib import Path

import pandas as pd
import pytest

from backlink_pricing_model.preprocessing import data_loading as dl


def test_dtypes_are_subset_of_training_columns() -> None:
    assert set(dl.COLUMN_DTYPES).issubset(set(dl.TRAINING_COLUMNS))


def test_target_is_a_training_column() -> None:
    assert "final_price" in dl.TRAINING_COLUMNS


@pytest.fixture
def multi_record_df() -> pd.DataFrame:
    # 10 domains, several with repeat records.
    rows = []
    for d in range(10):
        for _ in range(d % 3 + 1):
            rows.append({"domain": f"d{d}.com", "final_price": 100.0 + d})
    return pd.DataFrame(rows)


def test_split_has_no_domain_overlap(multi_record_df: pd.DataFrame) -> None:
    train, val, test = dl.domain_grouped_split(multi_record_df)
    train_d = set(train["domain"])
    val_d = set(val["domain"])
    test_d = set(test["domain"])
    assert train_d.isdisjoint(test_d)
    assert train_d.isdisjoint(val_d)
    assert val_d.isdisjoint(test_d)


def test_split_partitions_all_rows(multi_record_df: pd.DataFrame) -> None:
    train, val, test = dl.domain_grouped_split(multi_record_df)
    assert len(train) + len(val) + len(test) == len(multi_record_df)


def test_split_reproducible_with_seed(multi_record_df: pd.DataFrame) -> None:
    a = dl.domain_grouped_split(multi_record_df, random_state=7)
    b = dl.domain_grouped_split(multi_record_df, random_state=7)
    for left, right in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(left, right)


def test_load_raw_parquet_missing_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dl, "get_project_root", lambda: tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        dl.load_raw_parquet("nope.parquet")


def test_load_raw_parquet_csv_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame({
        "id": [1],
        "domain": ["a.com"],
        "final_price": [100.0],
        "dr": [50.0],
        "cf": [10.0],
        "tf": [20.0],
        "domain_traffic": [1000.0],
        "country": ["US"],
        "link_source_type": ["outreach"],
        "date_received": ["2024-01-01"],
        "status": ["live"],
    }).to_csv(raw / "backlinks.csv", index=False)
    monkeypatch.setattr(dl, "get_project_root", lambda: tmp_path)
    out = dl.load_raw_parquet("backlinks.parquet")
    assert len(out) == 1
    assert out["domain"].iloc[0] == "a.com"


def test_load_raw_parquet_loads_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame({"domain": ["x.com"]}).to_parquet(raw / "backlinks.parquet")
    monkeypatch.setattr(dl, "get_project_root", lambda: tmp_path)
    out = dl.load_raw_parquet()
    assert out["domain"].iloc[0] == "x.com"


def test_save_processed_writes_parquet_and_returns_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dl, "get_project_root", lambda: tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3]})
    path = dl.save_processed(df, "out", subdir="engineered")
    assert path.exists()
    assert path == tmp_path / "data" / "engineered" / "out.parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(path), df)
