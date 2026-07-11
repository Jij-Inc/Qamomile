"""Shared data structures and constants for circuit visualization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

__all__ = [
    "ControlFlowBoxLayout",
    "ControlFlowLayout",
    "FoldedBlockLayout",
    "HorizontalSpan",
    "InlineBlockLayout",
    "LineSegment",
    "PORDER_GATE",
    "PORDER_LINE",
    "PORDER_TEXT",
    "PORDER_WIRE",
    "PoweredGateLayout",
    "LayoutResult",
    "LayoutState",
    "Rect",
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


@dataclass(frozen=True)
class HorizontalSpan:
    """Represent an authoritative horizontal drawing extent.

    Args:
        left (float): Inclusive left edge in layout coordinates.
        right (float): Inclusive right edge in layout coordinates.

    Raises:
        ValueError: If ``right`` is smaller than ``left``.
    """

    left: float
    right: float

    def __post_init__(self) -> None:
        """Validate the span ordering.

        Raises:
            ValueError: If the right edge precedes the left edge.
        """
        if self.right < self.left:
            raise ValueError(
                "A horizontal span's right edge must not precede its left edge"
            )

    @property
    def width(self) -> float:
        """Return the span width.

        Returns:
            float: Distance between the right and left edges.
        """
        return self.right - self.left

    @property
    def center(self) -> float:
        """Return the horizontal center.

        Returns:
            float: Midpoint between the left and right edges.
        """
        return (self.left + self.right) / 2

    def expanded(self, padding: float) -> HorizontalSpan:
        """Expand the span by equal padding on both sides.

        Args:
            padding (float): Non-negative amount added to each side.

        Returns:
            HorizontalSpan: Expanded horizontal span.

        Raises:
            ValueError: If ``padding`` is negative.
        """
        if padding < 0:
            raise ValueError("Horizontal span padding must be non-negative")
        return HorizontalSpan(self.left - padding, self.right + padding)

    def union(self, other: HorizontalSpan) -> HorizontalSpan:
        """Return the smallest span containing this span and another.

        Args:
            other (HorizontalSpan): Span to combine with this one.

        Returns:
            HorizontalSpan: Union of both horizontal spans.
        """
        return HorizontalSpan(min(self.left, other.left), max(self.right, other.right))


@dataclass(frozen=True)
class Rect:
    """Represent an immutable axis-aligned layout rectangle.

    Args:
        left (float): Inclusive left edge in layout coordinates.
        bottom (float): Inclusive bottom edge in layout coordinates.
        right (float): Inclusive right edge in layout coordinates.
        top (float): Inclusive top edge in layout coordinates.

    Raises:
        ValueError: If either pair of rectangle edges is reversed.
    """

    left: float
    bottom: float
    right: float
    top: float

    def __post_init__(self) -> None:
        """Validate rectangle edge ordering.

        Raises:
            ValueError: If the right edge precedes the left edge or the top
                edge lies below the bottom edge.
        """
        if self.right < self.left:
            raise ValueError("A rectangle's right edge must not precede its left edge")
        if self.top < self.bottom:
            raise ValueError(
                "A rectangle's top edge must not lie below its bottom edge"
            )

    @property
    def width(self) -> float:
        """Return the rectangle width.

        Returns:
            float: Distance between the right and left edges.
        """
        return self.right - self.left

    @property
    def height(self) -> float:
        """Return the rectangle height.

        Returns:
            float: Distance between the top and bottom edges.
        """
        return self.top - self.bottom

    @property
    def center_x(self) -> float:
        """Return the horizontal center.

        Returns:
            float: Midpoint between the left and right edges.
        """
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        """Return the vertical center.

        Returns:
            float: Midpoint between the bottom and top edges.
        """
        return (self.bottom + self.top) / 2

    @property
    def horizontal_span(self) -> HorizontalSpan:
        """Return the rectangle's horizontal extent.

        Returns:
            HorizontalSpan: Span from the rectangle's left to right edge.
        """
        return HorizontalSpan(self.left, self.right)

    def expanded(self, horizontal: float, vertical: float | None = None) -> Rect:
        """Expand the rectangle by horizontal and vertical padding.

        Args:
            horizontal (float): Non-negative padding added to both horizontal
                sides.
            vertical (float | None): Non-negative padding added to both vertical
                sides. Defaults to ``horizontal`` when omitted.

        Returns:
            Rect: Expanded rectangle.

        Raises:
            ValueError: If either padding value is negative.
        """
        vertical_padding = horizontal if vertical is None else vertical
        if horizontal < 0 or vertical_padding < 0:
            raise ValueError("Rectangle padding must be non-negative")
        return Rect(
            self.left - horizontal,
            self.bottom - vertical_padding,
            self.right + horizontal,
            self.top + vertical_padding,
        )

    def union(self, other: Rect) -> Rect:
        """Return the smallest rectangle containing this rectangle and another.

        Args:
            other (Rect): Rectangle to combine with this one.

        Returns:
            Rect: Union of both rectangles.
        """
        return Rect(
            min(self.left, other.left),
            min(self.bottom, other.bottom),
            max(self.right, other.right),
            max(self.top, other.top),
        )


@dataclass(frozen=True)
class LineSegment:
    """Represent a fully resolved line segment.

    Args:
        start_x (float): X coordinate of the segment start.
        start_y (float): Y coordinate of the segment start.
        end_x (float): X coordinate of the segment end.
        end_y (float): Y coordinate of the segment end.
    """

    start_x: float
    start_y: float
    end_x: float
    end_y: float


@dataclass(frozen=True)
class ControlFlowBoxLayout:
    """Describe one resolved branch box in an unfolded control-flow node.

    Args:
        branch_index (int): Zero-based branch index within the node.
        label (str): Header label drawn inside the branch box.
        rect (Rect): Final branch rectangle including its header band.
    """

    branch_index: int
    label: str
    rect: Rect


@dataclass(frozen=True)
class ControlFlowLayout:
    """Describe final geometry for an unfolded IF or WHILE node.

    Args:
        boxes (tuple[ControlFlowBoxLayout, ...]): Branch boxes in display order.
        outer_rect (Rect): Smallest rectangle containing every branch box.
        connector_segments (tuple[LineSegment, ...]): Orthogonal
            measurement-to-box connector segments. Empty when the condition
            is not backed by one unambiguous measurement.
    """

    boxes: tuple[ControlFlowBoxLayout, ...]
    outer_rect: Rect
    connector_segments: tuple[LineSegment, ...] = ()


@dataclass(frozen=True)
class FoldedBlockLayout:
    """Describe final geometry for a folded control-flow summary.

    Args:
        rect (Rect): Final folded summary rectangle.
        connector_segments (tuple[LineSegment, ...]): Orthogonal
            measurement-to-box connector segments. Empty when no unambiguous
            source measurement exists.
    """

    rect: Rect
    connector_segments: tuple[LineSegment, ...] = ()


@dataclass(frozen=True)
class PoweredGateLayout:
    """Describe target and wrapper geometry for a powered controlled gate.

    Args:
        target_rect (Rect): Inner rectangle containing the target unitary.
        wrapper_rect (Rect): Outer rectangle carrying the power annotation.
    """

    target_rect: Rect
    wrapper_rect: Rect


@dataclass(frozen=True)
class InlineBlockLayout:
    """Describe final geometry for an expanded inline block.

    Args:
        inner_rect (Rect): Rectangle carrying the block label and children.
        outer_rect (Rect): Outermost rectangle, equal to ``inner_rect`` unless
            a power wrapper is present.
        control_segments (tuple[LineSegment, ...]): Vertical connector segments
            joining controlled-block dots to the target rectangle.
    """

    inner_rect: Rect
    outer_rect: Rect
    control_segments: tuple[LineSegment, ...] = ()


def _default_qubit_columns() -> defaultdict[int, float]:
    """Create a per-qubit column map with the historical baseline.

    Returns:
        defaultdict[int, float]: Map that initializes every qubit at column
            ``1.0``.
    """
    return defaultdict(lambda: 1.0)


@dataclass
class LayoutState:
    """Carry mutable data between layout placement phases.

    Args:
        positions (dict[tuple, float]): Compatibility map from node keys to
            horizontal centers.
        block_ranges (list[dict]): Compatibility descriptions of inline blocks.
        block_widths (dict[tuple, float]): Compatibility map of box widths.
        column (float): Furthest legacy column reached during placement.
        max_depth (int): Deepest visual node nesting level.
        actual_width (float): Furthest occupied horizontal coordinate.
        first_gate_x (float | None): Center of the first positioned node.
        first_gate_half_width (float): Half-width of the first positioned node.
        qubit_columns (dict[int, float]): Next legacy center per wire.
        qubit_right_edges (dict[int, float]): Authoritative frontier per wire.
        qubit_end_positions (dict[int, float]): Explicit wire termination edges.
        inlined_op_keys (set[tuple]): Keys of expanded inline blocks.
        gate_widths (dict[tuple, float]): Positioned gate widths.
        folded_block_extents (dict[tuple, dict]): Compatibility folded-box
            vertical metadata.
        node_spans (dict[tuple, HorizontalSpan]): Full horizontal node extents.
        control_flow_branch_spans (dict[tuple, tuple[HorizontalSpan, ...]]):
            Per-branch outer horizontal extents for IF and WHILE nodes.
        inline_inner_spans (dict[tuple, HorizontalSpan]): Horizontal bounds of
            inline blocks before an optional power wrapper is applied.
    """

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
    node_spans: dict[tuple, HorizontalSpan] = field(default_factory=dict)
    control_flow_branch_spans: dict[tuple, tuple[HorizontalSpan, ...]] = field(
        default_factory=dict
    )
    inline_inner_spans: dict[tuple, HorizontalSpan] = field(default_factory=dict)


@dataclass
class LayoutResult:
    """Expose all geometry computed by the layout engine.

    Args:
        width (float): Legacy circuit width in layout coordinates.
        positions (dict[tuple, float]): Compatibility node-center map.
        block_ranges (list[dict]): Compatibility inline-block descriptions.
        max_depth (int): Deepest visual node nesting level.
        block_widths (dict[tuple, float]): Compatibility box-width map.
        actual_width (float): Furthest occupied horizontal coordinate.
        first_gate_x (float): Center of the first positioned node.
        first_gate_half_width (float): Half-width of the first positioned node.
        qubit_y (list[float]): Final y coordinate of every qubit wire.
        qubit_end_positions (dict[int, float]): Explicit wire termination edges.
        inlined_op_keys (set[tuple]): Keys of expanded inline blocks.
        gate_widths (dict[tuple, float]): Positioned gate widths.
        folded_block_extents (dict[tuple, dict]): Compatibility folded-box
            vertical metadata.
        max_above (dict[int, float]): Reserved extent above each qubit.
        max_below (dict[int, float]): Reserved extent below each qubit.
        node_spans (dict[tuple, HorizontalSpan]): Authoritative full horizontal
            extent of every positioned visual node.
        node_rects (dict[tuple, Rect]): Final two-dimensional bounds for nodes
            that draw concrete geometry.
        gate_box_rects (dict[tuple, Rect]): Exact text-bearing gate box bounds,
            excluding controls and optional power wrappers.
        control_flow_layouts (dict[tuple, ControlFlowLayout]): Final unfolded
            IF and WHILE box geometry.
        folded_block_layouts (dict[tuple, FoldedBlockLayout]): Final folded
            control-flow geometry.
        inline_block_layouts (dict[tuple, InlineBlockLayout]): Final expanded
            block geometry.
        powered_gate_layouts (dict[tuple, PoweredGateLayout]): Final target and
            wrapper geometry for powered controlled gates.
        viewport (Rect): Final axes viewport containing every layout primitive.
        wire_bounds (Rect): Smallest rectangle containing all wire segments.
        wire_spans (dict[int, HorizontalSpan]): Horizontal span of every wire.
    """

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
    node_spans: dict[tuple, HorizontalSpan] = field(default_factory=dict)
    node_rects: dict[tuple, Rect] = field(default_factory=dict)
    gate_box_rects: dict[tuple, Rect] = field(default_factory=dict)
    control_flow_layouts: dict[tuple, ControlFlowLayout] = field(default_factory=dict)
    folded_block_layouts: dict[tuple, FoldedBlockLayout] = field(default_factory=dict)
    inline_block_layouts: dict[tuple, InlineBlockLayout] = field(default_factory=dict)
    powered_gate_layouts: dict[tuple, PoweredGateLayout] = field(default_factory=dict)
    viewport: Rect = field(default_factory=lambda: Rect(0.0, 0.0, 1.0, 1.0))
    wire_bounds: Rect = field(default_factory=lambda: Rect(0.0, 0.0, 1.0, 0.0))
    wire_spans: dict[int, HorizontalSpan] = field(default_factory=dict)
