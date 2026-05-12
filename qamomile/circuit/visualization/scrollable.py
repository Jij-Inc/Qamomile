"""Notebook helper for rendering wide circuits inside a scrollable container.

`MatplotlibDrawer` returns a Matplotlib `Figure`, which Jupyter renders as
an inline raster (PNG) by default. For long unrolled circuits — easily
many tens of inches wide — the inline raster either overflows the page or
gets squeezed into an unreadable thumbnail when published on
ReadTheDocs / Jupyter Book. This module exports `scrollable_svg`, which
serializes the figure as a vector SVG and wraps it in a
`overflow: auto` HTML container so the published notebook can be panned
horizontally / vertically and zoomed via the browser's native
Ctrl/Cmd + scroll without losing crispness.
"""

from __future__ import annotations

import html as _html
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from IPython.display import HTML
    from matplotlib.figure import Figure


def scrollable_svg(
    fig: "Figure",
    *,
    max_height_px: int = 600,
    border: str = "1px solid #ddd",
    padding_px: int = 8,
) -> "HTML":
    """Wrap a Matplotlib figure as a scrollable, zoomable SVG cell output.

    Saves the figure as SVG (vector, so the browser's native zoom keeps
    every gate label sharp), strips the XML preamble that some inline-HTML
    contexts choke on, and embeds it inside a single `<div>` whose CSS
    enables both axis scrollbars while capping the vertical extent. The
    inline Matplotlib backend would otherwise also auto-display the
    figure as a PNG below the SVG, so the helper closes the figure once
    the SVG bytes are captured to suppress that duplicate output. The
    figure is closed inside a ``finally`` so a Matplotlib failure during
    ``savefig`` (e.g., a backend bug, an invalid artist) does not leak
    the figure into the next cell.

    Args:
        fig (Figure): Matplotlib figure to embed. After the helper
            returns, ``fig`` is closed via ``plt.close(fig)`` and should
            not be reused.
        max_height_px (int): Vertical cap on the scroll container, in
            pixels. Tall circuits scroll vertically once they exceed
            this. Defaults to 600.
        border (str): CSS border declaration for the container (e.g.
            ``"1px solid #ddd"``). The value is HTML-attribute-escaped
            before being interpolated into ``style="..."``. Pass an
            empty string to render without a border.
        padding_px (int): CSS padding inside the container, in pixels.
            Defaults to 8.

    Returns:
        HTML: An ``IPython.display.HTML`` object suitable for being the
            last expression of a notebook cell. The wrapped SVG inherits
            the browser's native zoom (Ctrl/Cmd + scroll) and the
            container handles overflow with scroll bars.

    Raises:
        ImportError: If ``IPython`` is not importable in the current
            environment.
        RuntimeError: If Matplotlib's SVG backend produced output that
            does not contain a ``<svg>`` element. This indicates a
            backend or matplotlib version issue rather than user error;
            the figure is still closed before the exception propagates.

    Example:
        >>> import qamomile.circuit as qmc
        >>> from qamomile.circuit.visualization import (
        ...     MatplotlibDrawer,
        ...     scrollable_svg,
        ... )
        >>> # Inside a Jupyter cell:
        >>> # block = transpiler.inline(transpiler.to_block(my_kernel, ...))
        >>> # fig = MatplotlibDrawer(block).draw(fold_loops=False)
        >>> # scrollable_svg(fig)
    """
    import matplotlib.pyplot as plt
    from IPython.display import HTML

    buf = io.StringIO()
    try:
        fig.savefig(buf, format="svg", bbox_inches="tight")
        svg = buf.getvalue()
        svg_start = svg.find("<svg")
        if svg_start < 0:
            raise RuntimeError(
                "Matplotlib SVG backend produced output without an `<svg>` "
                "element; cannot embed as inline HTML."
            )
        # Drop XML / DOCTYPE preamble — inline HTML embedding only needs the
        # `<svg ...>` element.
        svg = svg[svg_start:]
        # Inject inline style on the `<svg>` element to opt out of the
        # responsive `svg, img { max-width: 100%; height: auto; }` rule
        # that MyST / Jupyter Book's Tailwind layer applies to all
        # cell-output media. Without this, a wide circuit gets shrunk to
        # the cell-output column width — at which point the surrounding
        # `overflow: auto` div has nothing to overflow and the scroll bar
        # never appears. Forcing the SVG back to its natural width
        # (carried by the `width="...pt"` attribute Matplotlib emits)
        # makes the wrapper genuinely scroll horizontally on the rendered
        # page.
        svg = svg.replace(
            "<svg ",
            '<svg style="max-width:none;max-height:none;display:block" ',
            1,
        )
    finally:
        # The inline Matplotlib backend would also emit the figure as a
        # PNG cell output otherwise, producing a duplicate (and tiny)
        # raster below the scrollable SVG. Closing inside `finally` also
        # guarantees the figure is released even if `savefig` failed.
        plt.close(fig)
        buf.close()

    # `border` is caller-supplied, so escape the HTML-attribute special
    # characters before splicing it into the `style="..."` attribute. No
    # legitimate CSS value contains `"`/`<`/`>`/`&`, so escaping is a
    # no-op for normal usage and a hard stop for accidental injection
    # (e.g., a stray quote that would otherwise close the attribute).
    style = (
        f"overflow: auto; max-height: {int(max_height_px)}px; "
        f"padding: {int(padding_px)}px;"
    )
    if border:
        style += f" border: {_html.escape(border, quote=True)};"
    return HTML(f'<div style="{style}">{svg}</div>')
