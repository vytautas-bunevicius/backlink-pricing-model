"""Tests for notebook environment helpers."""

from pathlib import Path

from backlink_pricing_model.core.notebook import display_saved_image_or_figure


def test_displays_image_when_path_exists(tmp_path: Path) -> None:
    image_path = tmp_path / "plot.png"
    image_path.write_bytes(b"fake-png")
    displayed: list[object] = []
    built: list[str] = []

    def fake_factory(*, filename: str) -> str:
        built.append(filename)
        return f"image:{filename}"

    display_saved_image_or_figure(
        image_path,
        figure="the-figure",
        display_fn=displayed.append,
        image_factory=fake_factory,
    )
    assert built == [str(image_path)]
    assert displayed == [f"image:{image_path}"]


def test_falls_back_to_figure_when_missing(tmp_path: Path) -> None:
    displayed: list[object] = []
    display_saved_image_or_figure(
        tmp_path / "missing.png",
        figure="the-figure",
        display_fn=displayed.append,
        image_factory=lambda *, filename: filename,
    )
    assert displayed == ["the-figure"]
