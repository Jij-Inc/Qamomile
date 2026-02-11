"""Matplotlib-based circuit visualization.

This module provides the MatplotlibDrawer facade that orchestrates
circuit analysis, layout computation, and rendering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.figure import Figure

from .analyzer import CircuitAnalyzer
from .layout import CircuitLayoutEngine
from .renderer import MatplotlibRenderer
from .style import DEFAULT_STYLE, CircuitStyle

if TYPE_CHECKING:
    from qamomile.circuit.ir.graph import Graph


class MatplotlibDrawer:
    """Matplotlib-based circuit drawer with Qiskit-style layout.

    This drawer produces static matplotlib figures showing quantum circuits.
    It supports two modes:
    - Block mode (inline=False): Shows CallBlockOperation as boxes
    - Inline mode (inline=True): Expands CallBlockOperation contents
    """

    def __init__(self, graph: Graph, style: CircuitStyle | None = None):
        """Initialize the drawer.

        Args:
            graph: Computation graph to visualize.
            style: Visual style configuration. Uses DEFAULT_STYLE if None.
        """
        self.graph = graph
        self.style = style or DEFAULT_STYLE
        self.inline = False
        self.fold_loops = True
        self.expand_composite = False
        self.inline_depth: int | None = None
        self._last_analyzer: CircuitAnalyzer | None = None

    def draw(
        self,
        inline: bool = False,
        fold_loops: bool = True,
        expand_composite: bool = False,
        inline_depth: int | None = None,
    ) -> Figure:
        """Generate a matplotlib Figure of the circuit.

        Args:
            inline: If True, expand CallBlockOperation. If False, show as boxes.
            fold_loops: If True (default), display ForOperation as blocks instead of unrolling.
                       If False, expand loops and show all iterations.
            expand_composite: If True, expand CompositeGateOperation (QFT, QPE, etc.).
                            If False (default), show as boxes. Independent of inline.
            inline_depth: Maximum nesting depth for inline expansion. None means
                         unlimited (default). 0 means no inlining, 1 means top-level
                         only, etc. Only affects CallBlock/ControlledU, not CompositeGate.

        Returns:
            Figure object.
        """
        self.inline = inline
        self.fold_loops = fold_loops
        self.expand_composite = expand_composite
        self.inline_depth = inline_depth

        analyzer = CircuitAnalyzer(
            self.graph, self.style, inline, fold_loops, expand_composite, inline_depth
        )
        qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(self.graph)

        engine = CircuitLayoutEngine(analyzer, self.style)
        layout = engine.compute_layout(self.graph, qubit_map, num_qubits)

        renderer = MatplotlibRenderer(analyzer, self.style)
        return renderer.render(self.graph, qubit_map, qubit_names, num_qubits, layout)

    # --- Backward-compatible test helpers ---

    def _build_qubit_map(self, graph: Graph) -> dict[str, int]:
        """Backward-compatible wrapper for tests."""
        analyzer = CircuitAnalyzer(
            graph,
            self.style,
            self.inline,
            self.fold_loops,
            self.expand_composite,
            self.inline_depth,
        )
        qubit_map, _, _ = analyzer.build_qubit_map(graph)
        self._last_analyzer = analyzer
        return qubit_map

    def _layout_operations(
        self, graph: Graph, qubit_map: dict[str, int]
    ) -> dict[str, int | dict]:
        """Backward-compatible wrapper returning dict for tests."""
        analyzer = self._last_analyzer
        if analyzer is None:
            analyzer = CircuitAnalyzer(
                graph,
                self.style,
                self.inline,
                self.fold_loops,
                self.expand_composite,
                self.inline_depth,
            )
        engine = CircuitLayoutEngine(analyzer, self.style)
        layout = engine.compute_layout(graph, qubit_map, len(qubit_map))
        return {
            "width": layout.width,
            "positions": layout.positions,
            "block_ranges": layout.block_ranges,
            "max_depth": layout.max_depth,
            "block_widths": layout.block_widths,
            "actual_width": layout.actual_width,
            "first_gate_x": layout.first_gate_x,
            "first_gate_half_width": layout.first_gate_half_width,
        }
