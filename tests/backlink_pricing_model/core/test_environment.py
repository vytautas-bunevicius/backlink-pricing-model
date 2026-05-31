"""Tests for runtime environment detection."""

from backlink_pricing_model.core.environment import get_project_root


def test_get_project_root_returns_directory_with_pyproject() -> None:
    root = get_project_root()
    assert (root / "pyproject.toml").exists()


def test_get_project_root_idempotent() -> None:
    assert get_project_root() == get_project_root()
