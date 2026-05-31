"""Tests for YAML config loading and path resolution."""

from pathlib import Path

import pytest

from backlink_pricing_model.core import config as cfg


def test_load_config_absolute_path(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("a: 1\nb: [x, y]\n")
    assert cfg.load_config(path) == {"a": 1, "b": ["x", "y"]}


def test_load_config_relative_path_resolved_against_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "t.yaml").write_text("k: v\n")
    monkeypatch.setattr(cfg, "get_project_root", lambda: tmp_path)
    assert cfg.load_config("configs/t.yaml") == {"k": "v"}


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cfg.load_config(tmp_path / "nope.yaml")


def test_resolve_path_joins_relative_to_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cfg, "get_project_root", lambda: tmp_path)
    assert cfg.resolve_path("data/raw") == tmp_path / "data" / "raw"
