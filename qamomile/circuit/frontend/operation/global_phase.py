"""Provide the ``qmc.global_phase`` qkernel combinator."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from qamomile.circuit.frontend.handle import Handle
from qamomile.circuit.frontend.handle.primitives import Float
from qamomile.circuit.frontend.operation.control import _qkernel_for_callable
from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.frontend.qkernel_invocation import invoke_validated_qkernel
from qamomile.circuit.frontend.qkernel_utils import (
    array_extents_equal,
    array_resource_identity,
    const_int,
)
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import (
    GlobalPhaseOperation,
    InvokeOperation,
    Operation,
    ReturnOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    WhileOperation,
    validate_region_args,
)
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    ResetOperation,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import OperationKind, QInitOperation
from qamomile.circuit.ir.types.primitives import BitType, FloatType
from qamomile.circuit.ir.value import ArrayValue, Value, ValueBase

# A phase angle may be supplied as a frontend ``Float`` handle (symbolic /
# runtime parameter) or a plain Python number (compile-time constant).
PhaseValue = float | int | Float
_QuantumOrigin = int | None


def _phase_to_value(
    phase: PhaseValue,
    *,
    caller: str = "global_phase()",
) -> Value:
    """Coerce a user-supplied phase angle into a scalar IR value.

    Args:
        phase (float | int | Float): Phase angle in radians, as a Qamomile
            ``Float`` handle or a Python numeric literal.
        caller (str): API context used in validation diagnostics. Defaults to
            ``"global_phase()"``.

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
            f"{caller}: phase must be a Float handle or a number, got "
            f"{type(phase).__name__}. Pass a qmc.Float (or a Python float)."
        )
    if isinstance(phase, bool):
        raise TypeError(
            f"{caller}: phase must be a number, not bool. Pass a numeric "
            "angle in radians (or a qmc.Float handle)."
        )
    if isinstance(phase, (int, float)):
        return Value(type=FloatType(), name="global_phase").with_const(float(phase))
    raise TypeError(
        f"{caller}: phase must be a Float handle or a number, got "
        f"{type(phase).__name__}."
    )


def _effective_outputs(block: Block) -> tuple[ValueBase, ...]:
    """Return the public outputs represented by a block.

    Args:
        block (Block): Block whose terminator or declared outputs are read.

    Returns:
        tuple[ValueBase, ...]: Outputs in public return order.
    """
    if block.operations and isinstance(block.operations[-1], ReturnOperation):
        return tuple(block.operations[-1].operands)
    return tuple(block.output_values)


def _same_resource_shape(left: Value, right: Value) -> bool:
    """Return whether two values have one compatible quantum shape.

    Args:
        left (Value): Formal/source value.
        right (Value): Actual/result value.

    Returns:
        bool: True when scalar/array kind, type, rank, and extents agree.
    """
    if isinstance(left, ArrayValue) or isinstance(right, ArrayValue):
        return (
            isinstance(left, ArrayValue)
            and isinstance(right, ArrayValue)
            and left.type == right.type
            and len(left.shape) == len(right.shape) == 1
            and array_extents_equal(left.shape[0], right.shape[0])
        )
    return left.type == right.type


def _selected_if_branches(
    op: IfOperation,
) -> tuple[bool | None, tuple[Sequence[Operation], ...]]:
    """Return the statically reachable branches of an If operation.

    Args:
        op (IfOperation): Conditional operation to inspect.

    Returns:
        tuple[bool | None, tuple[Sequence[Operation], ...]]: Constant condition
            when known and the branch operation sequences that can execute.

    Raises:
        TypeError: If the condition is not one scalar ``Bit``.
    """
    if len(op.operands) != 1:
        raise TypeError("global_phase(): IfOperation condition must be one scalar Bit.")
    condition: Any = op.operands[0]
    if isinstance(condition, bool):
        selected: bool | None = condition
    elif (
        isinstance(condition, Value)
        and not isinstance(condition, ArrayValue)
        and isinstance(condition.type, BitType)
    ):
        constant = condition.get_const() if condition.is_constant() else None
        selected = constant if isinstance(constant, bool) else None
    else:
        raise TypeError("global_phase(): IfOperation condition must be one scalar Bit.")
    branches: tuple[Sequence[Operation], ...] = (
        (op.true_operations,)
        if selected is True
        else (op.false_operations,)
        if selected is False
        else (op.true_operations, op.false_operations)
    )
    return selected, branches


def _static_loop_is_empty(op: ForOperation | ForItemsOperation) -> bool:
    """Return whether a loop is proven to execute zero iterations.

    Args:
        op (ForOperation | ForItemsOperation): Loop operation to inspect.

    Returns:
        bool: True only for a concrete empty range or bound empty mapping.
    """
    if isinstance(op, ForItemsOperation):
        return (
            bool(op.operands)
            and op.operands[0].metadata.dict_runtime is not None
            and (not op.operands[0].metadata.dict_runtime.bound_data)
        )
    if len(op.operands) != 3:
        return False
    bounds = tuple(const_int(value) for value in op.operands)
    if any(value is None for value in bounds):
        return False
    start, stop, step = bounds
    assert start is not None and stop is not None and step is not None
    if step == 0:
        return False
    return start >= stop if step > 0 else start <= stop


def _quantum_resource_identity(value: Value) -> str | None:
    """Return the canonical logical identity of one quantum value.

    Args:
        value (Value): Scalar or array quantum value to identify.

    Returns:
        str | None: Canonical logical identity, or ``None`` for an invalid
            cyclic array ancestry.
    """
    if isinstance(value, ArrayValue):
        return array_resource_identity(value)
    return value.logical_id


class _StructuredQuantumOrigins:
    """Resolve structured quantum joins to ordered block inputs.

    Ordinary unitary operations preserve ``logical_id`` across SSA versions,
    while full-array slices preserve their canonical array resource identity.
    Only structured joins need explicit edges because hand-built IR may not
    carry the frontend's canonical merge identity.
    """

    def __init__(self, block: Block) -> None:
        """Initialize provenance for one block.

        Args:
            block (Block): Block whose quantum contract is analyzed.
        """
        self._inputs = tuple(
            value
            for value in block.input_values
            if isinstance(value, Value) and value.type.is_quantum()
        )
        self._input_indices: dict[str, int | None] = {}
        for index, value in enumerate(self._inputs):
            identity = _quantum_resource_identity(value)
            if identity is None:
                continue
            self._input_indices[identity] = (
                index if identity not in self._input_indices else None
            )
        self._sources: dict[str, tuple[Value, ...] | None] = {}
        self._origins: dict[str, _QuantumOrigin] = {}

    @property
    def num_inputs(self) -> int:
        """Return the number of ordered quantum inputs.

        Returns:
            int: Number of quantum inputs in block order.
        """
        return len(self._inputs)

    @property
    def inputs(self) -> tuple[Value, ...]:
        """Return the ordered quantum inputs whose origins are tracked.

        Returns:
            tuple[Value, ...]: Quantum inputs in block order.
        """
        return self._inputs

    def _record(self, result: Value, sources: tuple[Value, ...] | None) -> None:
        """Record one uniquely-produced provenance edge.

        Args:
            result (Value): Quantum SSA result.
            sources (tuple[Value, ...] | None): Resource candidates, or None
                when the operation cannot prove preservation.

        Raises:
            TypeError: If one UUID has multiple producers.
        """
        if result.uuid in self._sources:
            raise TypeError(
                "global_phase(): one quantum SSA value has multiple producers."
            )
        self._sources[result.uuid] = sources

    def _collect_if(self, op: IfOperation) -> tuple[Sequence[Operation], ...]:
        """Record one runtime or compile-time branch merge.

        Args:
            op (IfOperation): Branch operation to inspect.

        Returns:
            tuple[Sequence[Operation], ...]: Branch bodies reachable under the
                condition's static value, if known.

        Raises:
            TypeError: If a quantum merge changes resource shape.
        """
        selected, branches = _selected_if_branches(op)
        for merge in op.iter_merges():
            if not merge.result.type.is_quantum():
                continue
            sources = (
                (merge.select(selected),)
                if isinstance(selected, bool)
                else (merge.true_value, merge.false_value)
            )
            if any(
                not source.type.is_quantum()
                or not _same_resource_shape(source, merge.result)
                for source in sources
            ):
                raise TypeError(
                    "global_phase(): IfOperation quantum merges must preserve "
                    "resource shape."
                )
            self._record(merge.result, sources)
        return branches

    def _collect_loop(
        self, op: ForOperation | ForItemsOperation
    ) -> tuple[Sequence[Operation], ...]:
        """Record explicit loop-carried quantum values.

        Args:
            op (ForOperation | ForItemsOperation): Loop to inspect.

        Returns:
            tuple[Sequence[Operation], ...]: Loop body when it may execute,
                otherwise an empty tuple.

        Raises:
            TypeError: If loop region arguments are invalid or change quantum
                resource shape.
        """
        try:
            region_args = validate_region_args(op)
        except ValueError as error:
            raise TypeError(f"global_phase(): {error}") from error
        empty = _static_loop_is_empty(op)
        for region_arg in region_args:
            if not region_arg.result.type.is_quantum():
                continue
            values = (
                region_arg.init,
                region_arg.block_arg,
                region_arg.yielded,
                region_arg.result,
            )
            if any(
                not value.type.is_quantum()
                or not _same_resource_shape(region_arg.init, value)
                for value in values
            ):
                raise TypeError(
                    "global_phase(): loop-carried quantum values must preserve "
                    "resource shape."
                )
            self._record(region_arg.block_arg, (region_arg.init,))
            self._record(
                region_arg.result,
                (region_arg.init,) if empty else (region_arg.init, region_arg.yielded),
            )
        return () if empty else (op.operations,)

    def collect(self, operations: Sequence[Operation]) -> None:
        """Collect provenance edges iteratively in structured order.

        Args:
            operations (Sequence[Operation]): Operations to inspect.
        """
        pending: list[Sequence[Operation]] = [operations]
        while pending:
            current = pending.pop()
            for op in current:
                if isinstance(op, ReturnOperation):
                    continue
                if isinstance(op, IfOperation):
                    pending.extend(reversed(self._collect_if(op)))
                    continue
                if isinstance(op, (ForOperation, ForItemsOperation)):
                    pending.extend(reversed(self._collect_loop(op)))
                    continue
                if isinstance(op, HasNestedOps):
                    pending.extend(reversed(tuple(op.nested_op_lists())))

    def _origin(self, value: Value) -> _QuantumOrigin:
        """Resolve one quantum value to a unique formal input iteratively.

        Args:
            value (Value): Output or intermediate value.

        Returns:
            int | None: Ordered input index, or None when not proven.
        """
        stack: list[tuple[Value, bool]] = [(value, False)]
        active: set[str] = set()
        while stack:
            current, expanded = stack.pop()
            if current.uuid in self._origins:
                continue
            if current.uuid not in self._sources:
                identity = _quantum_resource_identity(current)
                self._origins[current.uuid] = (
                    self._input_indices.get(identity) if identity is not None else None
                )
                continue
            sources = self._sources[current.uuid]
            if sources is None:
                self._origins[current.uuid] = None
                continue
            if not expanded:
                if current.uuid in active:
                    self._origins[current.uuid] = None
                    continue
                active.add(current.uuid)
                stack.append((current, True))
                stack.extend((source, False) for source in sources)
                continue
            active.discard(current.uuid)
            origins = [self._origins.get(source.uuid) for source in sources]
            self._origins[current.uuid] = (
                origins[0]
                if origins
                and origins[0] is not None
                and all(origin == origins[0] for origin in origins[1:])
                else None
            )
        return self._origins.get(value.uuid)

    def output_origins(
        self, outputs: Sequence[ValueBase]
    ) -> tuple[_QuantumOrigin, ...]:
        """Resolve outputs to their unique ordered quantum-input origins.

        Args:
            outputs (Sequence[ValueBase]): Effective block outputs.

        Returns:
            tuple[int | None, ...]: Input origins aligned with outputs.
        """
        return tuple(
            self._origin(output)
            if isinstance(output, Value) and output.type.is_quantum()
            else None
            for output in outputs
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
) -> list[tuple[Block, bool]]:
    """Reject non-unitary operations anywhere in an operation tree.

    Args:
        operations (Sequence[Operation]): Operations to inspect iteratively.
        kernel_name (str): Wrapped-kernel name used in diagnostics.
        visited_blocks (set[int]): Block identities already validated, used to
            terminate recursive call graphs.

    Returns:
        list[tuple[Block, bool]]: Reachable owned blocks paired with whether
        their quantum input/output contract must also be validated.

    Raises:
        TypeError: If a qubit allocation, measurement, expectation value,
            other hybrid operation, or unresolved call block is found.
    """
    pending: list[Sequence[Operation]] = [operations]
    owned_blocks: list[tuple[Block, bool]] = []
    deferred_while: WhileOperation | None = None
    while pending:
        current = pending.pop()
        for op in current:
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
                    f"global_phase(): kernel {kernel_name!r} must be a "
                    "qubit-preserving unitary; "
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
                    "unresolved ControlledUOperation, so its unitarity cannot "
                    "be validated from IR."
                )
            if (
                isinstance(op, InverseBlockOperation)
                and op.source_block is None
                and op.implementation_block is None
            ):
                raise TypeError(
                    f"global_phase(): kernel {kernel_name!r} contains an "
                    "opaque InverseBlockOperation, so its unitarity cannot be "
                    "validated from IR."
                )
            if isinstance(op, WhileOperation):
                deferred_while = deferred_while or op

            for block in _operation_owned_blocks(op):
                # Nested qkernels may be purely classical helpers used to
                # compute an angle for the enclosing unitary. Their own I/O
                # need not satisfy the outer ordered quantum-I/O contract.
                if id(block) in visited_blocks:
                    continue
                has_quantum_ports = any(
                    value.type.is_quantum()
                    for value in (*block.input_values, *_effective_outputs(block))
                )
                owned_blocks.append((block, has_quantum_ports))

            if isinstance(op, IfOperation):
                _, branches = _selected_if_branches(op)
                pending.extend(reversed(branches))
                continue
            if isinstance(op, (ForOperation, ForItemsOperation)):
                if not _static_loop_is_empty(op):
                    pending.append(op.operations)
                continue
            if isinstance(op, HasNestedOps):
                pending.extend(reversed(tuple(op.nested_op_lists())))

    if deferred_while is not None:
        raise TypeError(
            f"global_phase(): kernel {kernel_name!r} must be unitary; "
            "measurement-conditioned WhileOperation cannot be wrapped."
        )
    return owned_blocks


def _validate_unitary_block(
    block: Block,
    kernel_name: str,
    visited_blocks: set[int] | None = None,
) -> None:
    """Validate a Block as a qubit-preserving unitary.

    Quantum outputs must preserve the Block's quantum-input order and identity.
    An exact full-array re-slice (``qs[:]``) counts as the same input because
    QKernel's canonical call path transfers that view back to its source
    register. A runtime ``IfOperation`` is accepted as a branch-wise unitary
    family when every reachable branch preserves that same contract; the
    classical condition may therefore be supplied by a caller measurement.
    Such a family remains valid for standalone phase, while a later quantum
    control must resolve the condition or fail closed in dependency/emission
    validation. The operation tree is scanned recursively through control-flow
    bodies and operation-owned Blocks so a hidden allocation or measurement
    inside the wrapped body cannot pass as a unitary wrapper.

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
    pending_blocks: list[tuple[Block, bool]] = [(block, True)]
    while pending_blocks:
        current, validate_contract = pending_blocks.pop()
        if id(current) in visited_blocks:
            continue
        visited_blocks.add(id(current))
        reachable = _validate_unitary_operations(
            current.operations,
            kernel_name,
            visited_blocks,
        )
        pending_blocks.extend(reversed(reachable))
        if not validate_contract:
            continue

        provenance = _StructuredQuantumOrigins(current)
        outputs = _effective_outputs(current)
        quantum_outputs = tuple(
            output
            for output in outputs
            if isinstance(output, Value) and output.type.is_quantum()
        )
        basic_contract_holds = (
            provenance.num_inputs > 0
            and len(outputs) == provenance.num_inputs
            and len(quantum_outputs) == len(outputs)
            and all(
                _same_resource_shape(source, output)
                for source, output in zip(
                    provenance.inputs,
                    quantum_outputs,
                    strict=True,
                )
            )
        )
        provenance.collect(current.operations)
        origins = provenance.output_origins(outputs)
        if basic_contract_holds and origins == tuple(range(provenance.num_inputs)):
            continue
        raise TypeError(
            f"global_phase(): kernel {kernel_name!r} must be a "
            "qubit-preserving unitary -- its outputs must preserve the order "
            "and identity of its quantum inputs, with no classical outputs "
            "or discarded, replaced, or reordered qubits. "
            f"Got {provenance.num_inputs} quantum input(s) and "
            f"{len(outputs)} output(s)."
        )


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
    relative phase on the control subspace. Runtime ``Bit`` branches are
    treated as a family of branch-wise unitaries only when every branch
    preserves the same ordered quantum resources. If such a condition remains
    measurement-dependent when the family is quantum-controlled, compilation
    fails explicitly instead of emitting an invalid controlled operation.

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
