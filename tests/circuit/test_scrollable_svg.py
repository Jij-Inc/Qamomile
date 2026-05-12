"""Tests for scrollable_svg notebook helper."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

# IPython is part of the dev/jupyter extras, not a core qamomile dependency.
# Skip cleanly in minimal envs that have matplotlib but not the notebook stack.
pytest.importorskip("IPython")

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

    def test_border_is_html_attribute_escaped(self):
        """A `border` value with attribute-special chars must not break out
        of the `style="..."` attribute.

        Without escaping, a border like `1px"; background: red` would close
        the `style` attribute early and inject extra HTML/CSS into the
        notebook output. The helper escapes the value, so the literal
        `"` becomes the entity `&quot;` inside the attribute and stays
        inert.
        """
        fig, _ = plt.subplots(figsize=(1, 1))
        html = scrollable_svg(fig, border='1px"; background: red')
        body = html.data
        # The raw attribute-breaking quote must NOT appear inside the
        # serialized `style="..."` attribute.
        assert '1px"; background: red' not in body
        # The escaped form must be present, proving the value flowed
        # through `html.escape(quote=True)`.
        assert "1px&quot;; background: red" in body

    def test_svg_carries_inline_max_width_none(self, small_fig):
        """The embedded `<svg>` must carry inline style that opts out of the
        responsive `svg { max-width: 100% }` rule MyST / Jupyter Book apply
        to all cell-output media — otherwise wide circuits get shrunk to
        the column width and the surrounding `overflow: auto` div has
        nothing to overflow, so the horizontal scroll bar never appears
        on the rendered page.
        """
        html = scrollable_svg(small_fig)
        body = html.data
        # The opening `<svg ...>` tag must include our style override.
        opening = body[body.index("<svg") : body.index(">", body.index("<svg")) + 1]
        assert "max-width:none" in opening, (
            f"Expected `max-width:none` in <svg> style; got opening tag: {opening!r}"
        )
        assert "display:block" in opening

    def test_raises_when_savefig_omits_svg_tag(self, monkeypatch):
        """If the SVG backend produces output without `<svg`, raise loudly
        and still close the figure (so the caller is not silently leaked).
        """
        fig, _ = plt.subplots(figsize=(1, 1))

        def fake_savefig(buf, *args, **kwargs):
            buf.write("<?xml version='1.0'?>\n<not-svg/>\n")

        monkeypatch.setattr(fig, "savefig", fake_savefig)

        with pytest.raises(RuntimeError, match="<svg"):
            scrollable_svg(fig)
        # The figure must be released even though the SVG extraction failed.
        assert not plt.fignum_exists(fig.number)
