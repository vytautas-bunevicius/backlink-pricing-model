"""Tests for plot styling configuration."""

from pathlib import Path

import plotly.io as pio
import pytest

from backlink_pricing_model.visualization import plots_style as ps


class _FakeFig:
    """Stand-in for a Plotly figure with a controllable write_image."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.written: Path | None = None

    def write_image(self, path: Path) -> None:
        """Mimic Plotly export: write bytes, or raise when tooling is missing.

        Args:
            path: Destination path for the rendered image.

        Raises:
            RuntimeError: When export tooling is unavailable (``fail=True``).
        """
        if self.fail:
            raise RuntimeError("Kaleido/Chrome unavailable")
        self.written = Path(path)
        Path(path).write_bytes(b"fake-png")


def test_apply_plotly_defaults_registers_and_activates_template() -> None:
    ps.apply_plotly_defaults("test_template")
    assert "test_template" in pio.templates
    assert pio.templates.default == "test_template"


def test_save_figure_image_returns_true_and_writes(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "fig.png"
    assert ps.save_figure_image(_FakeFig(), out) is True
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_figure_image_returns_false_when_unavailable(
    tmp_path: Path,
) -> None:
    # Export tooling missing must degrade gracefully: warn, do not raise.
    with pytest.warns(RuntimeWarning, match="Skipping static export"):
        result = ps.save_figure_image(_FakeFig(fail=True), tmp_path / "x.png")
    assert result is False


def test_save_plot_wrapper_builds_named_path(tmp_path: Path) -> None:
    assert ps.save_plot(_FakeFig(), "myfig", tmp_path) is True
    assert (tmp_path / "myfig.png").exists()
