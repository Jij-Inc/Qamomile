"""Tests for scrollable_svg notebook helper."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from qamomile.circuit.visualization import scrollable_svg


@pytest.fixture
def small_fig():
    """A throwaway 1x1-inch figure with one labelled point for the helper."""
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.plot([0, 1], [0, 1])
    ax.set_title("dummy")
    yield fig
    # `scrollable_svg` already closes the figure; this is a safety net for
    # any test path that does not call the helper before failing.
    plt.close(fig)


class TestScrollableSvg:
    """`scrollable_svg` produces a self-contained, scrollable HTML output."""

    def test_returns_html_with_inline_svg(self, small_fig):
        """The returned HTML data embeds the figure as an inline `<svg>` tag."""
        html = scrollable_svg(small_fig)
        from IPython.display import HTML

        assert isinstance(html, HTML)
        assert "<svg" in html.data, "HTML body must contain an <svg> element"

    def test_strips_xml_preamble(self, small_fig):
        """The SVG payload starts with `<svg`, not the XML/DOCTYPE preamble.

        Some notebook viewers refuse to render a leading `<?xml ?>`
        declaration inside an inline HTML cell output, so the helper
        must drop it before embedding.
        """
        html = scrollable_svg(small_fig)
        body = html.data
        # The container `<div ...>` precedes the SVG, so the preamble is gone
        # iff we never see it before the first `<svg` tag.
        assert "<?xml" not in body[: body.index("<svg")]
        assert "<!DOCTYPE" not in body[: body.index("<svg")]

    def test_default_overflow_styling(self, small_fig):
        """The wrapping container enables both scrollbars by default."""
        html = scrollable_svg(small_fig)
        assert "overflow: auto" in html.data
        assert "max-height: 600px" in html.data

    def test_custom_height_and_no_border(self):
        """`border=""` must omit the border CSS entirely."""
        fig, _ = plt.subplots(figsize=(1, 1))
        html = scrollable_svg(fig, max_height_px=320, border="", padding_px=4)
        assert "max-height: 320px" in html.data
        assert "padding: 4px" in html.data
        assert "border:" not in html.data

    def test_closes_figure_to_suppress_duplicate_inline_render(self):
        """The helper must close `fig` so `%matplotlib inline` does not also
        render it as a duplicate PNG below the SVG.
        """
        fig, _ = plt.subplots(figsize=(1, 1))
        scrollable_svg(fig)
        # `plt.close(fig)` removes `fig` from `Gcf` (the figure manager
        # registry), so `plt.fignum_exists` is the public way to confirm it.
        assert not plt.fignum_exists(fig.number)
