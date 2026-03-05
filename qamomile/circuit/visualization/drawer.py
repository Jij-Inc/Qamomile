"""Matplotlib-based circuit visualization.

This module provides the MatplotlibDrawer facade that orchestrates
circuit analysis, layout computation, and rendering.
"""

from __future__ import annotations

from typing import Any

from matplotlib.figure import Figure

from qamomile.circuit.ir.graph import Graph

from .analyzer import CircuitAnalyzer
from .layout import CircuitLayoutEngine
from .renderer import MatplotlibRenderer
from .style import DEFAULT_STYLE, CircuitStyle


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
            expand_composite: If True, expand CompositeGateOperation (QFT, IQFT, etc.).
                            If False (default), show as boxes. Independent of inline.
            inline_depth: Maximum nesting depth for inline expansion. None means
                         unlimited (default). 0 means no inlining, 1 means top-level
                         only, etc. Only affects CallBlock/ControlledU, not CompositeGate.

        Returns:
            Figure object.
        """
        analyzer = CircuitAnalyzer(
            self.graph, self.style, inline, fold_loops, expand_composite, inline_depth
        )
        qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(self.graph)

        vc = analyzer.build_visual_ir(self.graph, qubit_map, qubit_names, num_qubits)

        engine = CircuitLayoutEngine(self.style)
        layout = engine.compute_layout(vc)

        renderer = MatplotlibRenderer(self.style)
        return renderer.render(vc, layout)

    @classmethod
    def draw_kernel(
        cls,
        kernel: Any,
        *,
        inline: bool = False,
        fold_loops: bool = True,
        expand_composite: bool = False,
        inline_depth: int | None = None,
        style: CircuitStyle | None = None,
        **kwargs: Any,
    ) -> Figure:
        """Draw a QKernel, handling Vector[Qubit] params with integer sizes.

        For kernels with ``Vector[Qubit]`` parameters, pass an integer to
        specify the array size (e.g., ``inputs=3`` for a 3-qubit vector).

        Args:
            kernel: A QKernel instance to visualize.
            inline: If True, expand CallBlockOperation contents.
            fold_loops: If True (default), display ForOperation as blocks.
            expand_composite: If True, expand CompositeGateOperation.
            inline_depth: Maximum nesting depth for inline expansion.
            style: Visual style configuration.
            **kwargs: Concrete values for kernel arguments. For Vector[Qubit]
                     parameters, pass an integer size.

        Returns:
            Figure object.
        """
        graph = kernel._build_graph_for_visualization(**kwargs)
        drawer = cls(graph, style)
        return drawer.draw(
            inline=inline,
            fold_loops=fold_loops,
            expand_composite=expand_composite,
            inline_depth=inline_depth,
        )
