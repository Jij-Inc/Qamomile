"""Read-only algorithm diagrams over verified QCore and native circuit projections.

Rust owns logical types, staging, structured interfaces, and bounded expansion.
This package arranges those immutable facts into SVG and self-contained HTML,
with optional PNG and PDF conversion of the same SVG presentation. It never
executes a qkernel or supplies alternate compiler semantics. Concrete lanes,
routing work, canvas dimensions, and raster allocations have presentation bounds.
"""

from ._api import Diagram, draw
from ._model import DisplayNode, DisplayValue

__all__ = ["Diagram", "DisplayNode", "DisplayValue", "draw"]
