"""Matplotlib-based circuit visualization.

This module provides the MatplotlibDrawer facade that orchestrates
circuit analysis, layout computation, and rendering.
"""

from __future__ import annotations

import inspect
from typing import Any

from matplotlib.figure import Figure

from qamomile.circuit.frontend.constructors import qubit_array
from qamomile.circuit.frontend.func_to_block import create_dummy_input, is_array_type
from qamomile.circuit.frontend.handle import Observable, Qubit
from qamomile.circuit.frontend.tracer import Tracer, trace
from qamomile.circuit.ir.graph import Graph
from qamomile.circuit.ir.value import Value

from .analyzer import CircuitAnalyzer
from .layout import CircuitLayoutEngine
from .renderer import MatplotlibRenderer
from .style import DEFAULT_STYLE, CircuitStyle


def _get_array_element_type(pt: Any) -> type | None:
    """Extract element type from an array type annotation."""
    if hasattr(pt, "__args__") and pt.__args__:
        return pt.__args__[0]
    return getattr(pt, "element_type", None)


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

    # ------------------------------------------------------------------
    # QKernel integration: build graph with Vector[Qubit] support
    # ------------------------------------------------------------------

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
        if cls._has_qubit_array_params(kernel):
            graph = cls._build_graph_for_draw(kernel, kwargs)
        else:
            graph = kernel.build(parameters=None, **kwargs)
        graph.output_names = kernel._extract_return_names() or []
        drawer = cls(graph, style)
        return drawer.draw(
            inline=inline,
            fold_loops=fold_loops,
            expand_composite=expand_composite,
            inline_depth=inline_depth,
        )

    @staticmethod
    def _has_qubit_array_params(kernel: Any) -> bool:
        """Check if kernel has any Qubit array parameters (Vector[Qubit], etc.)."""
        for param in kernel.signature.parameters.values():
            pt = param.annotation
            if is_array_type(pt) and _get_array_element_type(pt) is Qubit:
                return True
        return False

    @staticmethod
    def _build_graph_for_draw(kernel: Any, kwargs: dict[str, Any]) -> Graph:
        """Build a computation graph with Vector[Qubit] support for visualization.

        Separates integer-valued kwargs for Qubit array parameters (used as
        array sizes via ``qubit_array()``) from other kwargs, then traces the
        kernel to produce a Graph.
        """
        # Separate qubit array sizes from other kwargs
        qubit_sizes: dict[str, int] = {}
        build_kwargs: dict[str, Any] = {}
        for key, val in kwargs.items():
            if key in kernel.signature.parameters:
                pt = kernel.signature.parameters[key].annotation
                if is_array_type(pt):
                    elem = _get_array_element_type(pt)
                    if elem is Qubit and isinstance(val, int):
                        qubit_sizes[key] = val
                        continue
            build_kwargs[key] = val

        # Validate: all Vector[Qubit] params must have sizes
        missing = []
        for name, param in kernel.signature.parameters.items():
            pt = param.annotation
            if is_array_type(pt):
                elem = _get_array_element_type(pt)
                if elem is Qubit and name not in qubit_sizes:
                    missing.append(name)
        if missing:
            names = ", ".join(f"'{n}'" for n in missing)
            raise ValueError(
                f"Vector[Qubit] parameter(s) {names} require an integer size "
                f"for visualization. Example: draw({missing[0]}=3)"
            )

        parameters = kernel._auto_detect_parameters(build_kwargs)
        kernel._validate_parameters(parameters)

        tracer = Tracer()
        tracked_parameters: dict[str, Value] = {}

        with trace(tracer):
            dummy_inputs: dict[str, Any] = {}
            for name, param in kernel.signature.parameters.items():
                param_type = param.annotation
                if param_type is Observable:
                    handle = kernel._create_parameter_input(param_type, name)
                    tracked_parameters[name] = handle.value
                elif name in parameters:
                    if is_array_type(param_type):
                        # Use create_dummy_input for arrays so they get
                        # symbolic shapes (e.g. edges.shape[0] works).
                        # _create_parameter_input creates _shape=() which
                        # causes IndexError on shape access.
                        handle = create_dummy_input(param_type, name)
                    else:
                        handle = kernel._create_parameter_input(param_type, name)
                    tracked_parameters[name] = handle.value
                elif name in qubit_sizes:
                    handle = qubit_array(qubit_sizes[name], name)
                elif name in build_kwargs:
                    handle = kernel._create_bound_input(
                        param_type, name, build_kwargs[name]
                    )
                elif param.default is not inspect.Parameter.empty:
                    handle = kernel._create_bound_input(param_type, name, param.default)
                else:
                    handle = create_dummy_input(param_type, name)
                dummy_inputs[name] = handle
            result = kernel.func(**dummy_inputs)

        input_values = [h.value for h in dummy_inputs.values()]
        output_values: list[Value] = []
        if result is not None:
            if isinstance(result, tuple):
                for r in result:
                    if hasattr(r, "value"):
                        output_values.append(r.value)
            else:
                if hasattr(result, "value"):
                    output_values.append(result.value)

        return Graph(
            operations=tracer.operations,
            input_values=input_values,
            output_values=output_values,
            name=kernel.name,
            parameters=tracked_parameters,
        )
