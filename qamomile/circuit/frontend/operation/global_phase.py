"""Provide the ``qmc.global_phase`` qkernel combinator."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from qamomile.circuit.frontend.handle import Handle
from qamomile.circuit.frontend.handle.primitives import Float
from qamomile.circuit.frontend.operation.control import _qkernel_for_callable
from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.frontend.qkernel_invocation import invoke_validated_qkernel
from qamomile.circuit.frontend.qkernel_utils import is_full_reslice_of_input
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import (
    GlobalPhaseOperation,
    InvokeOperation,
    Operation,
)
from qamomile.circuit.ir.operation.control_flow import HasNestedOps, WhileOperation
from qamomile.circuit.ir.operation.gate import ControlledUOperation, ResetOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import OperationKind, QInitOperation
from qamomile.circuit.ir.types.primitives import FloatType
from qamomile.circuit.ir.value import ArrayValue, Value

# A phase angle may be supplied as a frontend ``Float`` handle (symbolic /
# runtime parameter) or a plain Python number (compile-time constant).
PhaseValue = float | int | Float


def _phase_to_value(phase: PhaseValue) -> Value:
    """Coerce a user-supplied phase angle into a scalar IR value.

    Args:
        phase (float | int | Float): Phase angle in radians, as a Qamomile
            ``Float`` handle or a Python numeric literal.

    Returns:
        Value: Scalar ``FloatType`` phase value.

    Raises:
        TypeError: If ``phase`` is a boolean, a non-``Float`` handle, or a
            non-numeric object.
    """
    if isinstance(phase, Float):
        return phase.value
    if isinstance(phase, Handle):
        raise TypeError(
            "global_phase(): phase must be a Float handle or a number, got "
            f"{type(phase).__name__}. Pass a qmc.Float (or a Python float)."
        )
    if isinstance(phase, bool):
        raise TypeError(
            "global_phase(): phase must be a number, not bool. Pass a numeric "
            "angle in radians (or a qmc.Float handle)."
        )
    if isinstance(phase, (int, float)):
        return Value(type=FloatType(), name="global_phase").with_const(float(phase))
    raise TypeError(
        "global_phase(): phase must be a Float handle or a number, got "
        f"{type(phase).__name__}."
    )


def _operation_owned_blocks(op: Operation) -> tuple[Block, ...]:
    """Return Blocks stored directly on an operation.

    Covers unified invokes through ``effective_body()`` and controlled or
    inverse operations through their established ``block`` / ``source_block`` /
    ``implementation_block`` fields. The field protocol also keeps the
    validator conservative for future operation-owned Block types.

    Args:
        op (Operation): Operation whose nested Block fields are inspected.

    Returns:
        tuple[Block, ...]: Distinct operation-owned Blocks in field order.
    """
    blocks: list[Block] = []
    seen: set[int] = set()
    if isinstance(op, InvokeOperation):
        body = op.effective_body()
        if isinstance(body, Block):
            blocks.append(body)
            seen.add(id(body))

    for field_name in ("block", "source_block", "implementation_block"):
        block = getattr(op, field_name, None)
        if isinstance(block, Block) and id(block) not in seen:
            blocks.append(block)
            seen.add(id(block))
    return tuple(blocks)


def _validate_unitary_operations(
    operations: Sequence[Operation],
    kernel_name: str,
    visited_blocks: set[int],
) -> None:
    """Reject non-unitary operations anywhere in an operation tree.

    Args:
        operations (Sequence[Operation]): Operations to inspect recursively.
        kernel_name (str): Wrapped-kernel name used in diagnostics.
        visited_blocks (set[int]): Block identities already validated, used to
            terminate recursive call graphs.

    Raises:
        TypeError: If a qubit allocation, measurement, expectation value,
            other hybrid operation, or unresolved call block is found.
    """
    for op in operations:
        if isinstance(op, QInitOperation):
            raise TypeError(
                f"global_phase(): kernel {kernel_name!r} must be a "
                "qubit-preserving unitary and cannot allocate qubits "
                "internally (found QInitOperation)."
            )
        if isinstance(op, ResetOperation):
            raise TypeError(
                f"global_phase(): kernel {kernel_name!r} must be unitary; "
                "found non-unitary ResetOperation."
            )
        if op.operation_kind is OperationKind.HYBRID:
            raise TypeError(
                f"global_phase(): kernel {kernel_name!r} must be unitary; "
                f"found non-unitary {type(op).__name__}. Measurements, "
                "expectation values, and other hybrid operations are not "
                "supported."
            )
        if isinstance(op, InvokeOperation) and op.effective_body() is None:
            raise TypeError(
                f"global_phase(): kernel {kernel_name!r} contains an "
                "opaque or unresolved InvokeOperation, so its unitarity "
                "cannot be validated from IR."
            )
        if isinstance(op, ControlledUOperation) and op.block is None:
            raise TypeError(
                f"global_phase(): kernel {kernel_name!r} contains an "
                "unresolved ControlledUOperation, so its unitarity cannot be "
                "validated from IR."
            )
        if (
            isinstance(op, InverseBlockOperation)
            and op.source_block is None
            and op.implementation_block is None
        ):
            raise TypeError(
                f"global_phase(): kernel {kernel_name!r} contains an opaque "
                "InverseBlockOperation, so its unitarity cannot be validated "
                "from IR."
            )

        for block in _operation_owned_blocks(op):
            # Nested qkernels may be purely classical helpers used to compute
            # an angle for the enclosing unitary. Their own I/O therefore need
            # not satisfy the outer ordered quantum-I/O contract; only their
            # operation trees must remain free of allocations, measurements,
            # and other non-unitary effects.
            if id(block) in visited_blocks:
                continue
            has_quantum_ports = any(
                value.type.is_quantum()
                for value in (*block.input_values, *block.output_values)
            )
            if has_quantum_ports:
                _validate_unitary_block(block, kernel_name, visited_blocks)
            else:
                visited_blocks.add(id(block))
                _validate_unitary_operations(
                    block.operations,
                    kernel_name,
                    visited_blocks,
                )

        if isinstance(op, HasNestedOps):
            for nested_operations in op.nested_op_lists():
                _validate_unitary_operations(
                    nested_operations,
                    kernel_name,
                    visited_blocks,
                )
        if isinstance(op, WhileOperation):
            raise TypeError(
                f"global_phase(): kernel {kernel_name!r} must be unitary; "
                "measurement-conditioned WhileOperation cannot be wrapped."
            )


def _validate_unitary_block(
    block: Block,
    kernel_name: str,
    visited_blocks: set[int] | None = None,
) -> None:
    """Validate a Block as a qubit-preserving unitary.

    Quantum outputs must preserve the Block's quantum-input order and identity.
    A concrete full-array re-slice (``qs[:]``) counts as the same input because
    QKernel's canonical call path transfers that view back to its source
    register. The operation tree is scanned recursively through control-flow
    bodies and operation-owned Blocks so a hidden allocation or measurement
    cannot pass as a unitary wrapper.

    Args:
        block (Block): Block to validate.
        kernel_name (str): Wrapped-kernel name used in diagnostics.
        visited_blocks (set[int] | None): Block identities already validated.
            Defaults to a new set for the outermost validation.

    Raises:
        TypeError: If outputs do not preserve the ordered quantum inputs or if
            any recursively reachable operation is non-unitary.
    """
    if visited_blocks is None:
        visited_blocks = set()
    if id(block) in visited_blocks:
        return
    visited_blocks.add(id(block))

    quantum_inputs = [value for value in block.input_values if value.type.is_quantum()]
    outputs = list(block.output_values)
    basic_contract_holds = (
        bool(quantum_inputs)
        and len(outputs) == len(quantum_inputs)
        and all(output.type.is_quantum() for output in outputs)
    )
    matched_input_indices: list[int] = []
    if basic_contract_holds:
        for output in outputs:
            match = next(
                (
                    index
                    for index, input_value in enumerate(quantum_inputs)
                    if output.logical_id == input_value.logical_id
                    or (
                        isinstance(output, ArrayValue)
                        and isinstance(input_value, ArrayValue)
                        and is_full_reslice_of_input(output, input_value)
                    )
                ),
                None,
            )
            if match is None:
                break
            matched_input_indices.append(match)

    if not (
        basic_contract_holds
        and matched_input_indices == list(range(len(quantum_inputs)))
    ):
        raise TypeError(
            f"global_phase(): kernel {kernel_name!r} must be a "
            "qubit-preserving unitary -- its outputs must preserve the order "
            "and identity of its quantum inputs, with no classical outputs "
            "or discarded, replaced, or reordered qubits. "
            f"Got {len(quantum_inputs)} quantum input(s) and "
            f"{len(outputs)} output(s)."
        )

    _validate_unitary_operations(block.operations, kernel_name, visited_blocks)


class GlobalPhaseGate:
    """Apply a wrapped QKernel followed by a zero-qubit global phase.

    Args:
        qkernel (QKernel): Kernel whose unitary receives the global phase.
        phase (float | int | Float): Phase angle in radians.

    Example:
        >>> import qamomile.circuit as qmc
        >>> @qmc.qkernel
        ... def layer(q: qmc.Qubit) -> qmc.Qubit:
        ...     return qmc.h(q)
        >>> @qmc.qkernel
        ... def circuit(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
        ...     return qmc.global_phase(layer, theta)(q)
    """

    def __init__(self, qkernel: QKernel, phase: PhaseValue) -> None:
        """Initialize the global-phase wrapper.

        Args:
            qkernel (QKernel): Kernel whose unitary receives the phase.
            phase (float | int | Float): Phase angle in radians.
        """
        self._qkernel = qkernel
        self._phase = phase

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Apply the wrapped QKernel and append its global phase.

        The wrapped call uses QKernel's canonical call path so argument
        validation, specialization, affine consumption, output permutation,
        and ``VectorView`` borrow transfer remain shared. Its selected block is
        validated once before consumption and then reused for emission.

        Args:
            *args (Any): Positional arguments for the wrapped kernel.
            **kwargs (Any): Keyword arguments for the wrapped kernel.

        Returns:
            Any: The wrapped kernel's outputs, preserving their exact order and
                frontend handle kinds.

        Raises:
            TypeError: If the phase is invalid or the wrapped call is not a
                recursively validated qubit-preserving unitary.
            QubitConsumedError: If a quantum argument was already consumed.
            FrontendTransformError: If the wrapped call cannot be specialized
                or an exact self-recursive call block cannot be validated.
            RuntimeError: If the wrapped kernel's emitted result count differs
                from its declared outputs or a returned slice lacks shape
                metadata.
        """
        phase = _phase_to_value(self._phase)
        outputs = invoke_validated_qkernel(
            self._qkernel,
            lambda block: _validate_unitary_block(block, self._qkernel.name),
            *args,
            **kwargs,
        )
        get_current_tracer().add_operation(
            GlobalPhaseOperation(operands=[phase], results=[])
        )
        return outputs


def global_phase(
    target: QKernel | Callable[..., Any], phase: PhaseValue
) -> GlobalPhaseGate:
    """Multiply a kernel unitary by ``exp(i * phase)``.

    The wrapped body is emitted through an ordinary QKernel call, followed by
    a zero-qubit global-phase operation. Standalone phase is unobservable;
    controlling a kernel that contains it turns the phase into an observable
    relative phase on the control subspace.

    Args:
        target (QKernel | Callable[..., Any]): QKernel or gate-like callable
            whose unitary receives the global phase.
        phase (float | int | Float): Phase angle in radians, supplied as a
            Qamomile ``Float`` handle or Python numeric literal.

    Returns:
        GlobalPhaseGate: Callable wrapper with the target's call interface.

    Raises:
        TypeError: If ``target`` cannot be interpreted as a gate-like callable.

    Example:
        >>> import qamomile.circuit as qmc
        >>> @qmc.qkernel
        ... def step(q: qmc.Qubit) -> qmc.Qubit:
        ...     return qmc.x(q)
        >>> @qmc.qkernel
        ... def phased_step(q: qmc.Qubit) -> qmc.Qubit:
        ...     return qmc.global_phase(step, 0.7)(q)
    """
    qkernel = _qkernel_for_callable(target, caller="global_phase")
    return GlobalPhaseGate(qkernel, phase)
