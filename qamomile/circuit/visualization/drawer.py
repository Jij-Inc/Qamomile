"""Matplotlib-based circuit visualization.

This module provides the MatplotlibDrawer facade that orchestrates exact
circuit lowering, layout computation, and rendering.
"""

from __future__ import annotations

from typing import Any

from matplotlib.figure import Figure

from qamomile.circuit.ir.block import Block, BlockKind

from .circuit_adapter import circuit_program_to_visual_ir
from .drawing_compiler import (
    compile_block_for_drawing,
    compile_qkernel_for_drawing,
)
from .layout import CircuitLayoutEngine
from .renderer import MatplotlibRenderer
from .style import DEFAULT_STYLE, CircuitStyle
from .visual_ir import VisualCircuit


def _prepare_graph_for_visualization(graph: Block) -> Block:
    """Apply visualization-only IR preparation before legacy analysis.

    This compatibility helper remains for direct ``CircuitAnalyzer`` tests.
    Public drawing uses the verified circuit-planning pipeline instead.

    Args:
        graph (Block): Freshly traced block to visualize.

    Returns:
        Block: For ``TRACED``, ``AFFINE``, and ``HIERARCHICAL`` graphs, a graph
            with compile-time resolvable ``IfOperation`` nodes lowered to their
            selected branch. ``ANALYZED`` graphs are returned unchanged.

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


def _render_visual_circuit(
    circuit: VisualCircuit,
    style: CircuitStyle,
) -> Figure:
    """Lay out and render a completed visual circuit.

    Args:
        circuit (VisualCircuit): Renderer-ready visual representation.
        style (CircuitStyle): Drawing geometry and color configuration.

    Returns:
        Figure: Rendered Matplotlib figure.
    """
    engine = CircuitLayoutEngine(style)
    layout = engine.compute_layout(circuit)
    renderer = MatplotlibRenderer(style)
    return renderer.render(circuit, layout)


class MatplotlibDrawer:
    """Matplotlib drawer backed by exact target-neutral circuit lowering.

    Both direct ``Block`` input and :meth:`draw_kernel` pass through the same
    verified ``CircuitProgram`` boundary. Source qkernel regions and callable
    invocations are boxed by default; safe bodies can be expanded with
    ``inline=True``.
    """

    def __init__(self, graph: Block, style: CircuitStyle | None = None) -> None:
        """Initialize the drawer.

        Args:
            graph (Block): Computation graph to visualize.
            style (CircuitStyle | None): Visual style configuration. Uses
                DEFAULT_STYLE if None.
        """
        self.graph = graph
        self.style = style or DEFAULT_STYLE

    def draw(
        self,
        inline: bool = False,
        fold_loops: bool = True,
        expand_composite: bool = False,
        inline_depth: int | None = None,
        fold_ifs: bool = False,
        fold_whiles: bool = False,
    ) -> Figure:
        """Generate a matplotlib Figure of the circuit.

        Args:
            inline (bool): Whether retained source qkernel regions and safe
                direct reusable-circuit calls should be expanded. Defaults to
                ``False``.
            fold_loops (bool): Whether concrete for loops should be folded.
                Defaults to ``True``.
            expand_composite (bool): Compatibility alias for ``inline``.
                Defaults to ``False``.
            inline_depth (int | None): Maximum nested source/reusable-call
                expansion depth. ``None`` permits arbitrary finite nesting.
            fold_ifs (bool): Whether runtime if regions should be folded.
                Defaults to ``False``.
            fold_whiles (bool): Whether runtime while regions should be
                folded. Defaults to ``False``.

        Returns:
            Figure: Matplotlib figure object.

        Raises:
            CircuitDrawingError: If structural quantum addressing cannot be
                resolved to one exact verified circuit.
        """
        drawing = compile_block_for_drawing(self.graph)
        visual_circuit = circuit_program_to_visual_ir(
            drawing.circuit,
            trace=drawing.trace,
            style=self.style,
            qubit_names=drawing.qubit_names,
            output_names=drawing.output_names,
            expectation_value_qubits=drawing.expectation_value_qubits,
            expand_calls=inline or expand_composite,
            inline_depth=inline_depth,
            fold_loops=fold_loops,
            fold_ifs=fold_ifs,
            fold_whiles=fold_whiles,
        )
        return _render_visual_circuit(visual_circuit, self.style)

    @classmethod
    def draw_kernel(
        cls,
        kernel: Any,
        *,
        inline: bool = False,
        fold_loops: bool = True,
        fold_ifs: bool = False,
        fold_whiles: bool = False,
        expand_composite: bool = False,
        inline_depth: int | None = None,
        style: CircuitStyle | None = None,
        **kwargs: Any,
    ) -> Figure:
        """Draw a qkernel through verified target-neutral circuit lowering.

        Scalar and vector quantum inputs remain external circuit wires. For a
        ``Vector[Qubit]`` parameter, pass an integer size (for example,
        ``inputs=3`` for a three-qubit vector).

        Args:
            kernel (Any): qkernel object to visualize.
            inline (bool): Whether retained source qkernel regions and safe
                direct reusable-circuit calls should be expanded. Defaults to
                ``False``.
            fold_loops (bool): Whether concrete for loops should be folded.
                Defaults to ``True``.
            fold_ifs (bool): Whether runtime if regions should be folded.
                Defaults to ``False``.
            fold_whiles (bool): Whether runtime while regions should be
                folded. Defaults to ``False``.
            expand_composite (bool): Compatibility alias for ``inline``.
            inline_depth (int | None): Maximum nested source/reusable-call
                expansion depth.
            style (CircuitStyle | None): Visual style configuration.
            **kwargs (Any): Concrete values for qkernel arguments. For
                Vector[Qubit] parameters, pass an integer size.

        Returns:
            Figure: Matplotlib figure object.

        Raises:
            CircuitDrawingError: If structural quantum addressing cannot be
                resolved to one exact verified circuit.
            TypeError: If a draw-time binding has an invalid frontend type.
            ValueError: If a required quantum-register size or classical
                structural binding is missing or invalid.
        """
        drawing = compile_qkernel_for_drawing(kernel, kwargs)
        resolved_style = style or DEFAULT_STYLE
        visual_circuit = circuit_program_to_visual_ir(
            drawing.circuit,
            trace=drawing.trace,
            style=resolved_style,
            qubit_names=drawing.qubit_names,
            output_names=drawing.output_names,
            expectation_value_qubits=drawing.expectation_value_qubits,
            expand_calls=inline or expand_composite,
            inline_depth=inline_depth,
            fold_loops=fold_loops,
            fold_ifs=fold_ifs,
            fold_whiles=fold_whiles,
        )
        return _render_visual_circuit(visual_circuit, resolved_style)
