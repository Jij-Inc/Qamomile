"""Decomposition tracing helpers for class-based composite gates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qamomile.circuit.frontend.handle.primitives import Qubit
from qamomile.circuit.frontend.tracer import Tracer, trace
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.types.primitives import QubitType
from qamomile.circuit.ir.value import Value

if TYPE_CHECKING:
    from qamomile.circuit.frontend.handle.array import Vector


def build_decomposition_block(
    gate: Any,
    target_qubits: "tuple[Qubit, ...] | Vector[Qubit]",
    *,
    strategy_name: str | None = None,
    has_decompose: bool,
) -> Block | None:
    """Trace a composite gate decomposition into a Block.

    Args:
        gate (Any): Composite gate instance that provides strategy lookup,
            arity, name, and decomposition hooks.
        target_qubits (tuple[Qubit, ...] | Vector[Qubit]): Target qubits used
            only to determine decomposition arity and shape.
        strategy_name (str | None): Optional strategy name to use for
            decomposition. Defaults to ``None``.
        has_decompose (bool): Whether ``gate`` overrides the base
            ``_decompose`` method.

    Returns:
        Block | None: Traced decomposition block, or ``None`` when no
        decomposition path is available.
    """
    from qamomile.circuit.frontend.handle.array import Vector

    strategy = gate.get_strategy(strategy_name) if strategy_name else None
    has_strategy = strategy is not None

    if not has_strategy and not has_decompose:
        return None

    decomp_tracer = Tracer()
    input_values: list[Value] = []
    is_vector_input = isinstance(target_qubits, Vector)

    if is_vector_input:
        fresh_qubits = _fresh_qubits(gate.num_target_qubits, input_values)
    else:
        fresh_qubits = _fresh_qubits(len(target_qubits), input_values)  # type: ignore[arg-type]

    with trace(decomp_tracer):
        if has_strategy:
            result = strategy.decompose(tuple(fresh_qubits))  # type: ignore[union-attr]
        else:
            result = gate._decompose(tuple(fresh_qubits))

    if result is None:
        return None

    return_values = _result_values(result)
    return Block(
        operations=decomp_tracer.operations,
        input_values=input_values,
        output_values=return_values,
        name=gate.custom_name or gate.gate_type.value,
        kind=BlockKind.HIERARCHICAL,
    )


def _fresh_qubits(count: int, input_values: list[Value]) -> list[Qubit]:
    """Create fresh parameter qubits without QInit operations.

    Args:
        count (int): Number of fresh qubit handles to create.
        input_values (list[Value]): Output list that receives each fresh
            parameter value.

    Returns:
        list[Qubit]: Fresh qubit handles backed by parameter values.
    """
    fresh_qubits = []
    for i in range(count):
        q_value = Value(type=QubitType(), name=f"_decomp_q{i}")
        fresh_qubits.append(Qubit(value=q_value))
        input_values.append(q_value)
    return fresh_qubits


def _result_values(result: Any) -> list[Value]:
    """Collect IR values returned by a traced decomposition.

    Args:
        result (Any): Decomposition return value.

    Returns:
        list[Value]: Values to use as block outputs.
    """
    from qamomile.circuit.frontend.handle.array import Vector

    if isinstance(result, Vector):
        return [result.value]
    if isinstance(result, tuple):
        return [q.value for q in result]
    return [result.value]
