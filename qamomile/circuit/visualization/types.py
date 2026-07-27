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

# Z-order for drawing priority (inspired by Qiskit).
#
# The two non-trivial slots are ``PORDER_LINE`` (vertical connection
# wires between control dots and the target qubit of a multi-qubit
# gate) and ``PORDER_GATE`` (gate boxes, control dots, target X glyphs).
# ``PORDER_LINE`` sits *below* ``PORDER_GATE`` so a target gate whose
# qubit happens to lie *between* two control qubits visually occludes
# the connecting line passing through it.  When the constants were
# inverted the connection line painted on top of the target's gate
# box, producing a visible artifact where the line appeared to bleed
# through the gate.
PORDER_WIRE = 1  # Qubit horizontal wire lines (lowest priority)
PORDER_LINE = 9  # Multi-qubit gate connection lines (below gates)
PORDER_GATE = 10  # Gate boxes, control dots, target glyphs
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


def _default_qubit_columns() -> defaultdict[int, float]:
    """Default factory: every qubit starts at column 1."""
    return defaultdict(lambda: 1.0)


@dataclass
class LayoutState:
    """Mutable state shared across layout handler methods."""

    positions: dict[tuple, float] = field(default_factory=dict)
    block_ranges: list[dict] = field(default_factory=list)
    block_widths: dict[tuple, float] = field(default_factory=dict)
    column: float = 1.0
    max_depth: int = 0
    actual_width: float = 1.0
    first_gate_x: float | None = None
    first_gate_half_width: float = 0.0
    qubit_columns: dict[int, float] = field(default_factory=_default_qubit_columns)
    qubit_right_edges: dict[int, float] = field(default_factory=dict)
    qubit_end_positions: dict[int, float] = field(default_factory=dict)
    inlined_op_keys: set[tuple] = field(default_factory=set)
    gate_widths: dict[tuple, float] = field(default_factory=dict)
    folded_block_extents: dict[tuple, dict] = field(default_factory=dict)
    # Header labels drawn above a wire that are not backed by a block_ranges
    # entry (unfolded if/else branch boxes). Each entry is
    # ``{"top_qubit": int, "num_lines": int}`` and only contributes vertical
    # clearance above its top qubit; it never draws a border.
    # Folded control-flow blocks use ``folded_block_extents`` instead because
    # they draw a full bordered summary box centered on the affected qubits.
    label_extents: list[dict] = field(default_factory=list)


@dataclass
class LayoutResult:
    """Result of the layout computation."""

    width: float
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
