"""Shared data structures and constants for circuit visualization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Union

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


@dataclass
class GateMeasure:
    """Width measurement for a gate/measurement/block-box operation."""

    estimated_width: float
    half_width: float
    columns_needed: int
    is_block_box: bool = False
    box_width: float | None = None


@dataclass
class BlockMeasure:
    """Width measurement for an inlined block."""

    label: str
    label_width: float
    content_width: float
    final_width: float
    max_gate_width: float
    border_padding: float
    children: list[MeasureNode]
    affected_qubits: list[int]
    depth: int
    control_qubit_indices: list[int] = field(default_factory=list)


@dataclass
class LoopMeasure:
    """Width measurement for a ForOperation."""

    fold: bool
    affected_qubits: list[int]
    folded_width: float | None = None
    iteration_children: list[list[MeasureNode]] | None = None
    iteration_widths: list[float] | None = None
    num_iterations: int = 0
    iter_param_values: list[dict] | None = None


@dataclass
class SkipMeasure:
    """Marker for zero-space operations."""

    pass


@dataclass
class IfMeasure:
    """Width measurement for an IfOperation."""

    fold: bool
    affected_qubits: list[int]
    condition_label: str
    folded_width: float | None = None
    true_children: list["MeasureNode"] | None = None
    false_children: list["MeasureNode"] | None = None
    true_width: float = 0.0
    false_width: float = 0.0


MeasureNode = Union[GateMeasure, BlockMeasure, LoopMeasure, SkipMeasure, IfMeasure]


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
    max_above: dict[int, float] = field(default_factory=dict)
    max_below: dict[int, float] = field(default_factory=dict)
