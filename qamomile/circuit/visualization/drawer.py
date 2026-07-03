"""Matplotlib-based circuit visualization.

This module provides the MatplotlibDrawer facade that orchestrates
circuit analysis, layout computation, and rendering.
"""

from __future__ import annotations

from typing import Any

from matplotlib.figure import Figure

from qamomile.circuit.ir.block import Block, BlockKind

from .analyzer import CircuitAnalyzer
from .layout import CircuitLayoutEngine
from .renderer import MatplotlibRenderer
from .style import DEFAULT_STYLE, CircuitStyle


def _prepare_graph_for_visualization(graph: Block) -> Block:
    """Apply visualization-only IR preparation before analysis.

    Args:
        graph (Block): Freshly traced block to visualize.

    Returns:
        Block: For ``TRACED``, ``AFFINE``, and ``HIERARCHICAL`` graphs, a graph
            with compile-time resolvable ``IfOperation`` nodes lowered to their
            selected branch while runtime/symbolic conditions remain available
            for branch-box rendering. ``ANALYZED`` graphs are returned
            unchanged because compile-time lowering must run before dependency
            analysis.

    Raises:
        ValidationError: If compile-time if lowering rejects the input graph.
        ValueError: If the graph has an unknown ``BlockKind``.
    """
    from qamomile.circuit.transpiler.passes.compile_time_if_lowering import (
        CompileTimeIfLoweringPass,
    )

    if graph.kind in (BlockKind.TRACED, BlockKind.AFFINE, BlockKind.HIERARCHICAL):
        return CompileTimeIfLoweringPass().run(graph)
    if graph.kind == BlockKind.ANALYZED:
        return graph
    raise ValueError(f"Unknown block kind for visualization: {graph.kind}")


class MatplotlibDrawer:
    """Matplotlib-based circuit drawer with Qiskit-style layout.

    This drawer produces static matplotlib figures showing quantum circuits.
    It supports two modes:
    - Block mode (inline=False): Shows CallBlockOperation as boxes
    - Inline mode (inline=True): Expands CallBlockOperation contents
    """

    def __init__(self, graph: Block, style: CircuitStyle | None = None):
        """Initialize the drawer.

        Args:
            graph (Block): Computation graph to visualize.
            style (CircuitStyle | None): Visual style configuration. Uses
                DEFAULT_STYLE if None.

        Raises:
            ValidationError: If compile-time if lowering rejects ``graph``.
            ValueError: If ``graph`` has an unknown ``BlockKind``.
        """
        self.graph = _prepare_graph_for_visualization(graph)
        self.style = style or DEFAULT_STYLE

    def draw(
        self,
        inline: bool = False,
        fold_loops: bool = True,
        expand_composite: bool = False,
        inline_depth: int | None = None,
        fold_ifs: bool = False,
    ) -> Figure:
        """Generate a matplotlib Figure of the circuit.

        Args:
            inline (bool): If True, expand CallBlockOperation. If False, show
                calls as boxes.
            fold_loops (bool): If True (default), display ForOperation as
                blocks instead of unrolling. If False, expand loops and show
                all iterations.
            expand_composite (bool): If True, expand CompositeGateOperation
                nodes. If False (default), show them as boxes.
            inline_depth (int | None): Maximum nesting depth for inline
                expansion. None means unlimited. Only affects CallBlock and
                ControlledU nodes, not CompositeGate.
            fold_ifs (bool): If True, display IfOperation as folded summary
                blocks. If False (default), show if/else branches side by side.

        Returns:
            Figure: Matplotlib figure object.
        """
        analyzer = CircuitAnalyzer(
            self.graph,
            self.style,
            inline=inline,
            fold_loops=fold_loops,
            expand_composite=expand_composite,
            inline_depth=inline_depth,
            fold_ifs=fold_ifs,
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
        fold_ifs: bool = False,
        expand_composite: bool = False,
        inline_depth: int | None = None,
        style: CircuitStyle | None = None,
        **kwargs: Any,
    ) -> Figure:
        """Draw a QKernel, handling Vector[Qubit] params with integer sizes.

        For kernels with ``Vector[Qubit]`` parameters, pass an integer to
        specify the array size (e.g., ``inputs=3`` for a 3-qubit vector).

        Args:
            kernel (Any): A QKernel instance to visualize.
            inline (bool): If True, expand CallBlockOperation contents.
            fold_loops (bool): If True (default), display ForOperation as
                blocks.
            fold_ifs (bool): If True, display IfOperation as folded summary
                blocks. If False (default), show if/else branches side by side.
            expand_composite (bool): If True, expand CompositeGateOperation.
            inline_depth (int | None): Maximum nesting depth for inline
                expansion.
            style (CircuitStyle | None): Visual style configuration.
            **kwargs (Any): Concrete values for kernel arguments. For
                Vector[Qubit] parameters, pass an integer size.

        Returns:
            Figure: Matplotlib figure object.
        """
        graph = kernel._build_graph_for_visualization(**kwargs)
        drawer = cls(graph, style)
        return drawer.draw(
            inline=inline,
            fold_loops=fold_loops,
            fold_ifs=fold_ifs,
            expand_composite=expand_composite,
            inline_depth=inline_depth,
        )
