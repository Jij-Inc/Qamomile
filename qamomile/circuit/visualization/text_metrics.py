"""Measure drawing text with the same fonts used by matplotlib rendering."""

from __future__ import annotations

from dataclasses import dataclass
from threading import local

from matplotlib import rcParams
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.cbook import is_math_text
from matplotlib.font_manager import FontProperties

__all__ = ["TextMetrics", "measure_text", "measure_text_width"]

_POINTS_PER_INCH = 72.0
_METRIC_SAFETY_FACTOR = 1.02
_DEFAULT_LINE_SPACING = 1.2
_RENDERER_LOCAL = local()


@dataclass(frozen=True)
class TextMetrics:
    """Contain conservative rendered text dimensions.

    Args:
        width (float): Maximum line width in layout units.
        height (float): Full multiline height in layout units.
    """

    width: float
    height: float


def _metric_renderer() -> RendererAgg:
    """Return one reusable 72-DPI metric renderer per thread.

    Returns:
        RendererAgg: Thread-local renderer whose pixels correspond to points.
    """
    renderer = getattr(_RENDERER_LOCAL, "renderer", None)
    if not isinstance(renderer, RendererAgg):
        renderer = RendererAgg(1, 1, int(_POINTS_PER_INCH))
        _RENDERER_LOCAL.renderer = renderer
    return renderer


def _math_mode(text: str) -> bool | str:
    """Return the math mode matplotlib will use for one text line.

    Args:
        text (str): Text line to inspect.

    Returns:
        bool | str: ``"TeX"`` for external TeX, True for mathtext, or False
            for plain text.
    """
    if rcParams["text.usetex"]:
        return "TeX"
    return bool(rcParams["text.parse_math"] and is_math_text(text))


def _measure_line(
    text: str,
    properties: FontProperties,
    renderer: RendererAgg,
) -> tuple[float, float, float]:
    """Measure one line with matplotlib's Agg renderer.

    Args:
        text (str): Single line of plain text or matplotlib mathtext.
        properties (FontProperties): Resolved font properties.
        renderer (RendererAgg): Renderer configured at 72 dots per inch so
            pixel metrics equal typographic points.

    Returns:
        tuple[float, float, float]: Width, height, and descent in points.
    """
    if not text:
        return 0.0, 0.0, 0.0
    return renderer.get_text_width_height_descent(
        text,
        properties,
        _math_mode(text),
    )


def measure_text(
    text: str,
    *,
    font_size: float,
    font_weight: str = "normal",
    font_family: str | None = None,
    fallback_char_width: float = 0.15,
) -> TextMetrics:
    """Return conservative dimensions for rendered text.

    Measures each line through matplotlib's own glyph and mathtext machinery.
    Invalid user-provided mathtext falls back to a configurable character
    width so layout remains available even when a label cannot be parsed.

    Args:
        text (str): Plain text or matplotlib mathtext, possibly multiline.
        font_size (float): Font size in points.
        font_weight (str): Matplotlib font weight. Defaults to ``"normal"``.
        font_family (str | None): Matplotlib font family. Defaults to None,
            selecting the family configured in matplotlib ``rcParams``.
        fallback_char_width (float): Width per character used if matplotlib
            cannot parse a line. Defaults to 0.15 layout units.

    Returns:
        TextMetrics: Maximum width and full multiline height in layout units,
            including a small metric safety factor.

    Raises:
        ValueError: If ``font_size`` is not positive or
            ``fallback_char_width`` is negative.
    """
    if font_size <= 0:
        raise ValueError("Font size must be positive")
    if fallback_char_width < 0:
        raise ValueError("Fallback character width must be non-negative")
    properties = FontProperties(
        family=font_family,
        weight=font_weight,
        size=font_size,
    )
    renderer = _metric_renderer()
    lines = text.split("\n")
    try:
        _, reference_height, reference_descent = _measure_line(
            "lp",
            properties,
            renderer,
        )
        minimum_step = (reference_height - reference_descent) * _DEFAULT_LINE_SPACING
        widths: list[float] = []
        baselines: list[float] = []
        descents: list[float] = []
        current_y = 0.0
        for index, line in enumerate(lines):
            width, height, descent = _measure_line(line, properties, renderer)
            height = max(height, reference_height)
            descent = max(descent, reference_descent)
            if index == 0:
                current_y = -(height - descent)
            else:
                current_y -= max(
                    minimum_step,
                    (height - descent) * _DEFAULT_LINE_SPACING,
                )
            widths.append(width)
            baselines.append(current_y)
            descents.append(descent)
            current_y -= descent
        width = max(widths, default=0.0) / _POINTS_PER_INCH
        height = -(baselines[-1] - descents[-1]) / _POINTS_PER_INCH
    except (RuntimeError, ValueError):
        width = max((len(line) for line in lines), default=0) * fallback_char_width
        height = (
            max(1, len(lines)) * font_size * _DEFAULT_LINE_SPACING / _POINTS_PER_INCH
        )
    return TextMetrics(
        width * _METRIC_SAFETY_FACTOR,
        height * _METRIC_SAFETY_FACTOR,
    )


def measure_text_width(
    text: str,
    *,
    font_size: float,
    font_weight: str = "normal",
    font_family: str | None = None,
    fallback_char_width: float = 0.15,
) -> float:
    """Return a conservative rendered text width.

    Args:
        text (str): Plain text or matplotlib mathtext, possibly multiline.
        font_size (float): Font size in points.
        font_weight (str): Matplotlib font weight. Defaults to ``"normal"``.
        font_family (str | None): Matplotlib font family. Defaults to None,
            selecting the family configured in matplotlib ``rcParams``.
        fallback_char_width (float): Width per character used if matplotlib
            cannot parse a line. Defaults to 0.15 layout units.

    Returns:
        float: Maximum rendered line width in layout units.

    Raises:
        ValueError: If ``font_size`` is not positive or
            ``fallback_char_width`` is negative.
    """
    return measure_text(
        text,
        font_size=font_size,
        font_weight=font_weight,
        font_family=font_family,
        fallback_char_width=fallback_char_width,
    ).width
