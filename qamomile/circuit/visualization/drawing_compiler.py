"""Compile qkernels into verified circuit programs for visualization."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator
from typing import Any

from qamomile.circuit.frontend.qkernel_visualization import (
    build_graph_for_circuit_drawing,
)
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.operation.operation import (
    Operation,
    OperationKind,
    QInitOperation,
)
from qamomile.circuit.ir.value import ArrayValue, TupleValue, Value, ValueLike
from qamomile.circuit.transpiler.circuit_ir.lowering import (
    lower_circuit_plan_with_trace,
    resolve_expval_qubit_slots,
)
from qamomile.circuit.transpiler.circuit_ir.model import CircuitBuilder, CircuitProgram
from qamomile.circuit.transpiler.circuit_ir.trace import CircuitProgramTrace
from qamomile.circuit.transpiler.circuit_planner import CircuitPlanningPipeline
from qamomile.circuit.transpiler.compiler import QamomileCompiler
from qamomile.circuit.transpiler.errors import QamomileCompileError
from qamomile.circuit.transpiler.passes.emit_support.qubit_address import (
    QubitAddress,
    QubitMap,
)
from qamomile.circuit.transpiler.passes.inline import InlinePass
from qamomile.circuit.transpiler.prepared import EntrypointMode
from qamomile.circuit.transpiler.segments import (
    ExpvalStep,
    MultipleQuantumSegmentsError,
)

__all__ = [
    "CircuitDrawingError",
    "CompiledDrawing",
    "compile_block_for_drawing",
    "compile_qkernel_for_drawing",
]


class CircuitDrawingError(ValueError):
    """Raised when semantics cannot be lowered to one exact circuit drawing.

    Args:
        message (str): Actionable explanation of the drawing failure.
    """


@dataclasses.dataclass(frozen=True)
class CompiledDrawing:
    """Hold verified circuit IR and its user-facing wire metadata.

    Args:
        circuit (CircuitProgram): Verified target-neutral circuit program.
        qubit_names (dict[int, str]): Display names keyed by physical slot.
        output_names (tuple[str, ...]): qkernel return names for display.
        expectation_value_qubits (tuple[tuple[int, ...], ...]): Exact physical
            slots for symbolic terminal expectation-value operations.
        trace (CircuitProgramTrace | None): Lossless drawing-only source
            provenance aligned with ``circuit``. Defaults to ``None``.
    """

    circuit: CircuitProgram
    qubit_names: dict[int, str]
    output_names: tuple[str, ...]
    expectation_value_qubits: tuple[tuple[int, ...], ...] = ()
    trace: CircuitProgramTrace | None = None


def compile_qkernel_for_drawing(
    kernel: Any,
    bindings: dict[str, Any] | None = None,
) -> CompiledDrawing:
    """Compile one qkernel into exact target-neutral drawing input.

    The shared circuit planning pipeline owns all structural resolution. A
    drawing is produced only after circuit verification succeeds, so symbolic
    quantum addressing can never be approximated with candidate wires.

    Args:
        kernel (Any): qkernel frontend object to compile.
        bindings (dict[str, Any] | None): Concrete draw-time values, including
            integer sizes for ``Vector[Qubit]`` inputs. Defaults to ``None``.

    Returns:
        CompiledDrawing: Verified circuit and deterministic display metadata.

    Raises:
        CircuitDrawingError: If circuit planning or lowering cannot resolve an
            exact circuit, or if it does not produce exactly one circuit
            fragment.
        TypeError: If frontend tracing rejects an argument type.
        ValueError: If frontend argument validation rejects a binding.
    """
    concrete_bindings = dict(bindings or {})
    graph = build_graph_for_circuit_drawing(kernel, **concrete_bindings)
    return compile_block_for_drawing(graph, concrete_bindings)


def compile_block_for_drawing(
    graph: Block,
    bindings: dict[str, Any] | None = None,
) -> CompiledDrawing:
    """Compile one semantic block into exact target-neutral drawing input.

    This is the common correctness boundary for both ``QKernel.draw()`` and
    direct ``MatplotlibDrawer(Block)`` use. Structural quantum addressing must
    resolve through the shared circuit planning pipeline; visualization never
    substitutes candidate wires for an unresolved index, slice, or loop range.

    Args:
        graph (Block): Traced, hierarchical, affine, or analyzed semantic
            block to compile. Later-stage blocks retain completed analysis;
            marker-only slice operations are normalized before planning.
        bindings (dict[str, Any] | None): Concrete structural bindings used by
            preparation and lowering. Defaults to ``None``.

    Returns:
        CompiledDrawing: Verified circuit and deterministic display metadata.

    Raises:
        CircuitDrawingError: If circuit planning or lowering cannot resolve an
            exact circuit, or if it does not produce exactly one circuit
            fragment.
    """
    concrete_bindings = dict(bindings or {})
    if not _has_quantum_semantics(graph):
        return CompiledDrawing(
            circuit=CircuitBuilder(0, 0, name=graph.name).freeze(),
            qubit_names={},
            output_names=tuple(graph.output_names),
        )

    try:
        prepared = QamomileCompiler().prepare_block(
            graph,
            concrete_bindings,
            mode=EntrypointMode.CIRCUIT_FRAGMENT,
        )
        plan = CircuitPlanningPipeline(
            inline_pass=InlinePass(preserve_regions=True)
        ).run(prepared, concrete_bindings)
        expval_steps = [step for step in plan.steps if isinstance(step, ExpvalStep)]
        circuit_plan = dataclasses.replace(
            plan,
            steps=[step for step in plan.steps if not isinstance(step, ExpvalStep)],
        )
        runtime_parameters = [
            name for name in graph.parameters if name not in concrete_bindings
        ]
        executable, traces = lower_circuit_plan_with_trace(
            circuit_plan,
            bindings=concrete_bindings,
            parameters=runtime_parameters,
            preserve_semantic_call_names=True,
        )
        expectation_value_qubits = tuple(
            resolve_expval_qubit_slots(
                step.segment.qubits_value,
                step.quantum_step_index,
                executable.compiled_quantum,
                concrete_bindings,
                runtime_parameters,
            )
            for step in expval_steps
        )
    except (QamomileCompileError, MultipleQuantumSegmentsError, ValueError) as error:
        raise CircuitDrawingError(_format_compilation_error(error)) from error

    if len(executable.compiled_quantum) != 1:
        raise CircuitDrawingError(
            "Circuit drawing requires exactly one quantum circuit fragment; "
            f"planning produced {len(executable.compiled_quantum)}."
        )

    segment = executable.compiled_quantum[0]
    if len(traces) != 1:
        raise CircuitDrawingError(
            "Circuit drawing trace count does not match the single quantum "
            f"fragment ({len(traces)} traces)."
        )
    return CompiledDrawing(
        circuit=segment.circuit,
        qubit_names=_build_qubit_names(
            graph,
            segment.qubit_map,
            segment.circuit.num_qubits,
            segment.segment.operations,
        ),
        output_names=tuple(graph.output_names),
        expectation_value_qubits=expectation_value_qubits,
        trace=traces[0],
    )


def _has_quantum_semantics(graph: Block) -> bool:
    """Return whether a block contributes a quantum interface or operation.

    Args:
        graph (Block): Semantic block to inspect recursively.

    Returns:
        bool: True when drawing requires a quantum circuit; false for a
            classical-only block that should render as an empty circuit.
    """
    if any(
        value.type.is_quantum() for value in (*graph.input_values, *graph.output_values)
    ):
        return True

    def contains_quantum(
        operations: Iterable[Operation],
        visited_blocks: set[int],
    ) -> bool:
        """Inspect one operation region for quantum or hybrid semantics.

        Args:
            operations (Iterable[Operation]): Operations in one region.
            visited_blocks (set[int]): Callable bodies already inspected.

        Returns:
            bool: True when this region or a nested region is quantum.
        """
        for operation in operations:
            if isinstance(operation, InvokeOperation):
                values = [*operation.all_input_values(), *operation.results]
                if any(value.type.is_quantum() for value in values):
                    return True
                body = operation.effective_body()
                if isinstance(body, Block) and id(body) not in visited_blocks:
                    visited_blocks.add(id(body))
                    if any(
                        value.type.is_quantum()
                        for value in (*body.input_values, *body.output_values)
                    ) or contains_quantum(body.operations, visited_blocks):
                        return True
                continue
            if operation.operation_kind in (
                OperationKind.QUANTUM,
                OperationKind.HYBRID,
            ):
                return True
            if isinstance(operation, HasNestedOps) and any(
                contains_quantum(body, visited_blocks)
                for body in operation.nested_op_lists()
            ):
                return True
        return False

    return contains_quantum(graph.operations, set())


def _format_compilation_error(
    error: QamomileCompileError | MultipleQuantumSegmentsError | ValueError,
) -> str:
    """Add drawing-specific remediation to a compiler diagnosis.

    Args:
        error (QamomileCompileError | MultipleQuantumSegmentsError | ValueError):
            Underlying shared-pipeline or circuit-verification failure.

    Returns:
        str: Actionable public drawing error message.
    """
    return (
        f"Cannot draw an exact circuit: {error} "
        "Drawing requires semantics that lower to a verified CircuitProgram. "
        "When the diagnostic concerns unresolved topology, provide concrete "
        "values to draw(...) for every quantum index, slice bound, register "
        "size, and loop range. For MatplotlibDrawer(Block), trace or build the "
        "Block with those structural values resolved before drawing it. "
        "Runtime gate parameters may remain symbolic."
    )


def _build_qubit_names(
    graph: Block,
    qubit_map: QubitMap,
    num_qubits: int,
    planned_operations: Iterable[Operation] | None = None,
) -> dict[int, str]:
    """Build display names from exact source-value allocation addresses.

    Args:
        graph (Block): QInit-free source fragment used for compilation.
        qubit_map (QubitMap): Exact semantic-address to physical-slot map.
        num_qubits (int): Verified circuit width.
        planned_operations (Iterable[Operation] | None): Post-planning semantic
            operations whose cloned allocation UUIDs align with ``qubit_map``.
            Defaults to the source graph operations.

    Returns:
        dict[int, str]: One deterministic label for every physical slot.
    """
    names: dict[int, str] = {}
    for value in graph.input_values:
        _register_value_name(value, qubit_map, names)
    operations = graph.operations if planned_operations is None else planned_operations
    for value in _iter_allocated_values(operations):
        _register_value_name(value, qubit_map, names)
    for index in range(num_qubits):
        names.setdefault(index, f"q{index}")
    return names


def _register_value_name(
    value: ValueLike,
    qubit_map: QubitMap,
    names: dict[int, str],
) -> None:
    """Register labels for one scalar, tuple, or aggregate quantum value.

    Args:
        value (ValueLike): Semantic value whose allocation may be named.
        qubit_map (QubitMap): Exact semantic-address to physical-slot map.
        names (dict[int, str]): Mutable physical-slot label map.

    Returns:
        None: ``names`` is updated in place.
    """
    if isinstance(value, TupleValue):
        for element in value.elements:
            _register_value_name(element, qubit_map, names)
        return
    if not value.type.is_quantum():
        return
    if isinstance(value, ArrayValue):
        addresses = sorted(
            (
                address
                for address in qubit_map
                if address.uuid == value.uuid and address.element_index is not None
            ),
            key=lambda address: (
                address.element_index if address.element_index is not None else -1
            ),
        )
        for address in addresses:
            assert address.element_index is not None
            names.setdefault(
                qubit_map[address],
                f"{value.name}[{address.element_index}]",
            )
        return
    if isinstance(value, Value):
        address = QubitAddress(value.uuid)
        if address in qubit_map:
            names.setdefault(qubit_map[address], value.name)


def _iter_allocated_values(
    operations: Iterable[Operation],
    visited_blocks: set[int] | None = None,
) -> Iterator[ValueLike]:
    """Yield explicit quantum allocations from nested semantic operations.

    Args:
        operations (Iterable[Operation]): Operations to inspect recursively.
        visited_blocks (set[int] | None): Callable bodies already traversed,
            used to avoid recursive-definition cycles. Defaults to ``None``.

    Returns:
        Iterator[ValueLike]: Lazily yielded ``QInitOperation`` results.
    """
    seen = visited_blocks if visited_blocks is not None else set()
    for operation in operations:
        if isinstance(operation, QInitOperation):
            yield from operation.results
        if isinstance(operation, HasNestedOps):
            for nested in operation.nested_op_lists():
                yield from _iter_allocated_values(nested, seen)
        if isinstance(operation, InvokeOperation):
            body = operation.effective_body()
            if isinstance(body, Block) and id(body) not in seen:
                seen.add(id(body))
                yield from _iter_allocated_values(body.operations, seen)
