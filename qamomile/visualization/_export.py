"""Optional static conversion of bounded, internally generated SVG diagrams."""

from __future__ import annotations

import math
from xml.etree import ElementTree

from qamomile.errors import CapabilityError

_MAX_PNG_PIXELS = 32_000_000
_MAX_PNG_AXIS = 32_768
_MAX_EXPORT_AXIS = 100_000


def export_static(svg: str, suffix: str, scale: float) -> bytes:
    """Convert the selected SVG without allocating an unbounded raster surface.

    Args:
        svg (str): Bounded SVG produced by the internal diagram renderer.
        suffix (str): .png or .pdf, already selected by Diagram.save.
        scale (float): Positive finite raster/page scale.

    Returns:
        bytes: PNG image or vector PDF document.

    Raises:
        TypeError: If scale is not an int or float, or is a boolean.
        ValueError: If scale is nonpositive or nonfinite.
        CapabilityError: If canvas dimensions or raster allocations exceed
            their bounds, or CairoSVG/system Cairo is unavailable.
    """
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError("scale must be a real number")
    try:
        factor = float(scale)
    except OverflowError as error:
        raise ValueError("scale must be positive and finite") from error
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("scale must be positive and finite")
    root = ElementTree.fromstring(svg)
    width = float(root.attrib["width"]) * factor
    height = float(root.attrib["height"]) * factor
    if not (0 < width <= _MAX_EXPORT_AXIS and 0 < height <= _MAX_EXPORT_AXIS):
        raise CapabilityError("Q7401: static export exceeds its canvas limit")
    if suffix == ".png" and (
        math.ceil(width) > _MAX_PNG_AXIS
        or math.ceil(height) > _MAX_PNG_AXIS
        or math.ceil(width) * math.ceil(height) > _MAX_PNG_PIXELS
        or width < 1
        or height < 1
    ):
        raise CapabilityError(
            "Q7401: PNG exceeds its pixel budget; reduce scale or save SVG/PDF"
        )
    try:
        import cairosvg
    except (ImportError, OSError) as error:
        raise CapabilityError(
            "Q7404: PNG/PDF export requires qamomile[visualization] and system Cairo"
        ) from error
    convert = cairosvg.svg2png if suffix == ".png" else cairosvg.svg2pdf
    return convert(bytestring=svg.encode("utf-8"), scale=factor)
