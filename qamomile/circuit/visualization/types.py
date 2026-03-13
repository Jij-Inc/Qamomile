"""Shared data structures and constants for circuit visualization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

__all__ = [
    "PORDER_GATE",
    "PORDER_LINE",
    "PORDER_TEXT",
    "PORDER_WIRE",
    "LayoutResult",
    "LayoutState",
]

# Z-order for drawing priority (inspired by Qiskit)
PORDER_WIRE = 1  # Wire lines (lowest priority)
PORDER_GATE = 10  # Gate boxes
PORDER_LINE = 11  # Connection lines for multi-qubit gates
PORDER_TEXT = 13  # Text labels (highest priority)

# Known TeX symbol names (Greek letters)
_TEX_SYMBOLS = {
    # Lowercase Greek
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "varepsilon",
    "zeta",
    "eta",
    "theta",
    "vartheta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "rho",
    "varrho",
    "sigma",
    "varsigma",
    "tau",
    "upsilon",
    "phi",
    "varphi",
    "chi",
    "psi",
    "omega",
    # Uppercase Greek
    "Gamma",
    "Delta",
    "Theta",
    "Lambda",
    "Xi",
    "Pi",
    "Sigma",
    "Upsilon",
    "Phi",
    "Psi",
    "Omega",
    # Common math symbols
    "hbar",
    "ell",
    "nabla",
    "partial",
    "infty",
}


def _default_qubit_columns() -> defaultdict[int, int]:
    """Default factory: every qubit starts at column 1."""
    return defaultdict(lambda: 1)


@dataclass
class LayoutState:
    """Mutable state shared across layout handler methods."""

    positions: dict[tuple, int] = field(default_factory=dict)
    block_ranges: list[dict] = field(default_factory=list)
    block_widths: dict[tuple, float] = field(default_factory=dict)
    column: int = 1
    max_depth: int = 0
    actual_width: float = 1.0
    first_gate_x: float | None = None
    first_gate_half_width: float = 0.0
    qubit_columns: dict[int, int] = field(default_factory=_default_qubit_columns)
    qubit_right_edges: dict[int, float] = field(default_factory=dict)
    qubit_end_positions: dict[int, float] = field(default_factory=dict)
    inlined_op_keys: set[tuple] = field(default_factory=set)
    gate_widths: dict[tuple, float] = field(default_factory=dict)
    folded_block_extents: dict[tuple, dict] = field(default_factory=dict)


@dataclass
class LayoutResult:
    """Result of the layout computation."""

    width: int
    positions: dict[tuple, float]
    block_ranges: list[dict]
    max_depth: int
    block_widths: dict[tuple, float]
    actual_width: float
    first_gate_x: float
    first_gate_half_width: float
    qubit_y: list[float] = field(default_factory=list)
    qubit_end_positions: dict[int, float] = field(default_factory=dict)
    inlined_op_keys: set[tuple] = field(default_factory=set)
    gate_widths: dict[tuple, float] = field(default_factory=dict)
    folded_block_extents: dict[tuple, dict] = field(default_factory=dict)
    max_above: dict[int, float] = field(default_factory=dict)
    max_below: dict[int, float] = field(default_factory=dict)
