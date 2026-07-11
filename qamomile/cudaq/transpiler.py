"""CUDA-Q backend transpiler implementation.

This module provides CudaqTranspiler for converting Qamomile QKernels
into CUDA-Q decorator-kernel artifacts, along with CudaqEmitPass and
CudaqExecutor.

All circuits (static and runtime) are emitted through a single
``CudaqKernelEmitter`` codegen path.  The emitter produces
``CudaqKernelArtifact`` instances whose ``execution_mode`` determines
whether ``cudaq.sample()`` / ``cudaq.observe()`` / ``cudaq.get_state()``
(STATIC) or ``cudaq.run()`` (RUNNABLE) is used for execution.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any, Sequence

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    RuntimeClassicalExpr,
)
from qamomile.circuit.ir.operation.callable import (
    CompositeGateType,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
    GateOperationType,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.executable import (
    ParameterMetadata,
    QuantumExecutor,
)
from qamomile.circuit.transpiler.gate_emitter import MeasurementMode
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.emit_support import (
    ClbitMap,
    QubitAddress,
    QubitMap,
    ValueResolver,
    resolve_condition_address,
    resolve_if_condition,
)
from qamomile.circuit.transpiler.passes.emit_support.cast_binop_emission import (
    evaluate_binop,
)
from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
    _bind_block_inputs,
    _bind_quantum_input_shapes,
    _expand_quantum_operands_to_phys,
    _quantum_input_operands,
)
from qamomile.circuit.transpiler.passes.emit_support.gate_emission import (
    reject_duplicate_physical_indices,
)
from qamomile.circuit.transpiler.passes.emit_support.inverse_emission import (
    _map_inverse_block_results,
    _normalize_inverse_block_op,
)
from qamomile.circuit.transpiler.passes.emit_support.pauli_evolve_emission import (
    _resolve_gamma,
    validate_hamiltonian_within_register,
)
from qamomile.circuit.transpiler.passes.separate import SegmentationPass
from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass
from qamomile.circuit.transpiler.transpiler import Transpiler

from .emitter import (
    CudaqKernelArtifact,
    CudaqKernelEmitter,
    ExecutionMode,
)

if TYPE_CHECKING:
    import qamomile.observable as qm_o


CudaqControlledQubitMap = dict[str | QubitAddress, int]


def _cudaq_slice_root_slots_available(
    array_value: ArrayValue,
    vector_slots: dict[str, list[int]],
    qubit_map: CudaqControlledQubitMap,
    emit_pass: Any,
    bindings: dict[str, Any],
) -> bool:
    """Return whether a sliced vector's root physical slots are available.

    Args:
        array_value (ArrayValue): Sliced Vector[Qubit] value whose root
            availability should be checked.
        vector_slots (dict[str, list[int]]): Cached vector UUID to physical
            slot table carried by the CUDA-Q controlled fallback walker.
        qubit_map (CudaqControlledQubitMap): UUID/address to physical qubit map.
        emit_pass (Any): Emit pass whose resolver folds root shape values.
        bindings (dict[str, Any]): Active controlled-body bindings.

    Returns:
        bool: True when the non-sliced root is registered in ``vector_slots`` or
            every root element address can be found in ``qubit_map``.
    """
    root = array_value
    while root.is_slice():
        if root.slice_of is None:
            return False
        root = root.slice_of

    if root.uuid in vector_slots:
        return True
    if not root.shape:
        return False
    size = emit_pass._resolver.resolve_int_value(root.shape[0], bindings)
    if size is None or size < 0:
        return False
    return all(QubitAddress(root.uuid, i) in qubit_map for i in range(size))


def _build_block_qubit_map(
    block_value: Any,
    target_indices: list[int],
    emit_pass: Any,
    bindings: dict[str, Any],
) -> tuple[CudaqControlledQubitMap, dict[str, list[int]]]:
    """Build a UUID-to-physical-target map for a controlled block.

    Seeds the map from block quantum input_values (positionally matching
    ``target_indices``). Gate bodies are deliberately not pre-walked
    here: element operands inside a ``ForOperation`` may use the loop
    variable as their index, and that index is only concrete after the
    loop emission path binds the current iteration value.

    Scalar ``Qubit`` inputs map one UUID -> one physical target;
    ``Vector[Qubit]`` inputs (``ArrayValue`` carrying a quantum element
    type) each absorb their own declared length, resolved from
    ``bindings`` via the emit pass's value resolver.  This mirrors the
    seeding logic in the base emit pass's ``_populate_input_qubit_map``
    so multi-vector inner blocks (Step 2.b of the controlled-API
    redesign) work the same way as single-vector ones did under the
    old single-``Vector`` enforcement.  The element UUIDs themselves
    are unknown at seed time — they are created during the inner
    kernel's tracing — so the controlled body emitter resolves operands
    whose ``parent_array`` matches a registered vector lazily, using the
    current loop-local bindings.

    Args:
        block_value (Any): Inner block whose ``input_values`` and
            shape metadata are inspected. Missing attributes are treated
            as an empty seed source.
        target_indices (list[int]): Physical target indices the inner
            block's quantum inputs map onto, in declaration order.
        emit_pass (Any): The emit pass driving the conversion.  Used
            for its ``_resolver`` to resolve ``Vector[Qubit]`` shapes
            against ``bindings`` when the shape is symbolic.
        bindings (dict[str, Any]): Caller bindings forwarded to the
            resolver for shape resolution.

    Returns:
        tuple[CudaqControlledQubitMap, dict[str, list[int]]]: The scalar/per-
            element UUID map and a Vector UUID -> physical slots table.
            Empty maps are returned when ``block_value`` has no quantum
            inputs.

    Raises:
        EmitError: If a ``Vector[Qubit]`` input length cannot be
            resolved from ``bindings``, if the resolved length is
            negative, or if the total declared quantum input footprint
            exceeds ``len(target_indices)``.
    """
    qubit_map: CudaqControlledQubitMap = {}
    vector_slots: dict[str, list[int]] = {}

    if hasattr(block_value, "input_values"):
        quantum_inputs = [
            iv
            for iv in block_value.input_values
            if hasattr(iv, "type") and iv.type.is_quantum()
        ]

        # Resolve every ``Vector[Qubit]`` input length up-front so we
        # can budget the target indices per input.  Mirrors the base
        # emit pass's ``_populate_input_qubit_map`` strategy.
        resolved_lengths: dict[str, int] = {}
        for iv in quantum_inputs:
            if not isinstance(iv, ArrayValue):
                continue
            length: int | None = None
            if iv.shape:
                size_val = iv.shape[0]
                if size_val.is_constant():
                    length = int(size_val.get_const())
                else:
                    length = emit_pass._resolver.resolve_int_value(size_val, bindings)
            if length is None:
                shape_name = iv.shape[0].name if iv.shape else "<no shape>"
                raise EmitError(
                    f"CUDA-Q controlled helper: cannot resolve "
                    f"Vector[Qubit] input length for {iv.name!r}; "
                    f"bind {shape_name!r} before transpilation.",
                    operation="ControlledUOperation",
                )
            if length < 0:
                raise EmitError(
                    f"CUDA-Q controlled helper: Vector[Qubit] input "
                    f"{iv.name!r} resolved to a negative length ({length}).",
                    operation="ControlledUOperation",
                )
            resolved_lengths[iv.uuid] = length

        scalar_count = sum(1 for iv in quantum_inputs if not isinstance(iv, ArrayValue))
        total = scalar_count + sum(resolved_lengths.values())
        if total > len(target_indices):
            raise EmitError(
                f"CUDA-Q controlled helper inner block requires {total} "
                f"physical qubits (scalars={scalar_count}, vector lengths="
                f"{sorted(resolved_lengths.values())}) but only "
                f"{len(target_indices)} target indices are available.",
                operation="ControlledUOperation",
            )

        qubit_idx = 0
        for iv in quantum_inputs:
            if isinstance(iv, ArrayValue):
                length = resolved_lengths[iv.uuid]
                slots = target_indices[qubit_idx : qubit_idx + length]
                vector_slots[iv.uuid] = slots
                # Also store the base UUID for any downstream lookup
                # that addresses the array as a whole.
                if slots:
                    qubit_map[iv.uuid] = slots[0]
                qubit_idx += length
            else:
                if qubit_idx < len(target_indices):
                    qubit_map[iv.uuid] = target_indices[qubit_idx]
                qubit_idx += 1

    return qubit_map, vector_slots


def _seed_vector_element_uuid(
    operand: Any,
    vector_slots: dict[str, list[int]],
    qubit_map: CudaqControlledQubitMap,
    emit_pass: Any,
    bindings: dict[str, Any],
) -> None:
    """Seed ``qubit_map`` with a Vector[Qubit] element operand's UUID.

    Resolves an ``operand`` that addresses an element of a registered
    ``Vector[Qubit]`` input, including an element of a slice view over a
    registered root vector, to its physical target via ``vector_slots`` and
    stores it under ``operand.uuid``.
    Non-element operands (scalar ``Qubit`` inputs already seeded by
    ``_build_block_qubit_map``) and elements whose parent/root vector is not
    available are skipped silently so direct UUID mappings can still resolve.

    Args:
        operand (Any): A gate operand to seed.
        vector_slots (dict[str, list[int]]): Map from ``Vector[Qubit]``
            UUID to one physical target index per logical element.
        qubit_map (CudaqControlledQubitMap): UUID-to-physical-target map;
            mutated in place when the operand is a recognized Vector[Qubit]
            element.
        emit_pass (Any): Emit pass whose resolver resolves loop-index
            Values against ``bindings``.
        bindings (dict[str, Any]): Current controlled-body bindings,
            including loop-local values when emitting an unrolled loop.

    Raises:
        EmitError: Raised when a recognized Vector[Qubit] element uses an
            index that cannot be resolved under the current bindings, or when
            the resolved index is outside the registered/sliced slot range. A
            slice parent may also propagate ``EmitError`` from slice length or
            slice-bound resolution.
    """
    parent = getattr(operand, "parent_array", None)
    if parent is None:
        return
    indices = tuple(getattr(operand, "element_indices", ()))
    if not indices:
        return

    if (
        isinstance(parent, ArrayValue)
        and parent.uuid in vector_slots
        and not parent.is_slice()
    ):
        slots = vector_slots[parent.uuid]
    elif isinstance(parent, ArrayValue) and parent.is_slice():
        if not _cudaq_slice_root_slots_available(
            parent, vector_slots, qubit_map, emit_pass, bindings
        ):
            return
        slots = _resolve_cudaq_array_slots(
            parent,
            vector_slots,
            qubit_map,
            emit_pass,
            bindings,
            operation="ControlledUOperation",
        )
    else:
        return

    idx_val = indices[0]
    if idx_val.is_constant():
        elem_idx = int(idx_val.get_const())
    else:
        resolved = emit_pass._resolver.resolve_int_value(idx_val, bindings)
        if resolved is None:
            resolved_value = emit_pass._resolver.resolve_classical_value(
                idx_val, bindings
            )
            if resolved_value is not None:
                resolved = int(resolved_value)
        if resolved is None:
            # Silently skipping leaves the element's UUID unmapped in
            # ``qubit_map``, which later trips a hard ``AssertionError``
            # in ``_emit_cudaq_controlled_ops`` ("Missing qubit mapping
            # ...") when the inner block addresses the element with a
            # symbolic loop variable.  Fail loudly here with the actual
            # limitation instead so the user can act on it.
            raise EmitError(
                f"CUDA-Q controlled helper: a ``Vector[Qubit]`` element "
                f"of {parent.name!r} (uuid {parent.uuid[:8]}) is indexed "
                f"by a value that cannot be resolved under the current "
                f"bindings. Bind the loop bounds or index values before "
                f"transpilation.",
                operation="ControlledUOperation",
            )
        elem_idx = int(resolved)
    if elem_idx < 0 or elem_idx >= len(slots):
        # Silently skipping leaves the element's UUID unmapped in
        # ``qubit_map``, which later trips a hard ``AssertionError``
        # in ``_emit_cudaq_controlled_ops`` ("Missing qubit mapping
        # ...") when the inner block addresses an out-of-range slot.
        # Fail loudly here with the actual limitation instead so the
        # user can act on it.
        raise EmitError(
            f"CUDA-Q controlled helper: Vector[Qubit] element index "
            f"{elem_idx} is outside the declared length of "
            f"{parent.name!r} (uuid {parent.uuid[:8]}, [0, "
            f"{len(slots)})).  This usually means the inner controlled "
            f"block addresses an element past the slice the "
            f"controlled-U was wired up with.",
            operation="ControlledUOperation",
        )
    # A loop body reuses the same IR Value UUID for ``q[i]`` while each
    # unrolled iteration binds ``i`` to a different concrete slot.  Use
    # assignment rather than ``setdefault`` so the current iteration's
    # binding controls where the gate is emitted.
    qubit_map[operand.uuid] = slots[elem_idx]


def _resolve_gate_targets(
    op: GateOperation,
    qubit_map: CudaqControlledQubitMap,
) -> list[int]:
    """Resolve physical target indices for an inner gate's operands.

    For each quantum operand of the gate, looks up the corresponding
    physical target via ``qubit_map``.

    Args:
        op (GateOperation): The gate operation whose operand targets are
            being resolved.
        qubit_map (CudaqControlledQubitMap): UUID-to-physical-target map seeded
            by ``_build_block_qubit_map``.

    Returns:
        list[int]: Per-operand physical target indices in
            ``op.qubit_operands`` order.

    Raises:
        EmitError: If any operand is absent from ``qubit_map``. Falling
            back to slot 0 would silently route a multi-target inner gate
            to the wrong physical qubit.
    """
    gate_type = op.gate_type
    gate_name = gate_type.name if gate_type is not None else "<unknown>"
    resolved: list[int] = []
    for operand in op.qubit_operands:
        if operand.uuid in qubit_map:
            resolved.append(qubit_map[operand.uuid])
        else:
            raise EmitError(
                f"CUDA-Q controlled helper cannot resolve operand "
                f"{operand.name!r} (uuid {operand.uuid[:8]}) for "
                f"inner gate {gate_name}.",
                operation="ControlledUOperation",
            )
    return resolved


def _resolve_cudaq_value_index(
    value: Any,
    vector_slots: dict[str, list[int]],
    qubit_map: CudaqControlledQubitMap,
    emit_pass: Any,
    bindings: dict[str, Any],
    *,
    operation: str,
) -> int:
    """Resolve one quantum Value to a CUDA-Q physical qubit index.

    Args:
        value (Any): Scalar or vector-element quantum value.
        vector_slots (dict[str, list[int]]): Vector input/result slot map
            available while walking a controlled block body.
        qubit_map (CudaqControlledQubitMap): UUID-to-physical-qubit map,
            mutated if ``value`` is a vector element that can be seeded lazily.
        emit_pass (Any): Emit pass whose resolver folds element indices.
        bindings (dict[str, Any]): Current controlled-body bindings.
        operation (str): Operation label for diagnostics.

    Returns:
        int: Physical qubit index.

    Raises:
        EmitError: If the value cannot be resolved under the current maps
            and bindings.
    """
    _seed_vector_element_uuid(value, vector_slots, qubit_map, emit_pass, bindings)
    if value.uuid in qubit_map:
        return qubit_map[value.uuid]
    raise EmitError(
        f"CUDA-Q controlled helper cannot resolve quantum value "
        f"{value.name!r} (uuid {value.uuid[:8]}) for {operation}.",
        operation=operation,
    )


def _resolve_cudaq_value_indices(
    values: list[Any],
    vector_slots: dict[str, list[int]],
    qubit_map: CudaqControlledQubitMap,
    emit_pass: Any,
    bindings: dict[str, Any],
    *,
    operation: str,
) -> list[int]:
    """Resolve a list of quantum values to physical CUDA-Q indices.

    Args:
        values (list[Any]): Quantum values to resolve.
        vector_slots (dict[str, list[int]]): Vector slot map.
        qubit_map (CudaqControlledQubitMap): UUID-to-physical-qubit map.
        emit_pass (Any): Emit pass whose resolver folds element indices.
        bindings (dict[str, Any]): Current controlled-body bindings.
        operation (str): Operation label for diagnostics.

    Returns:
        list[int]: Physical qubit indices in input order.

    Raises:
        EmitError: If any value cannot be resolved.
    """
    return [
        _resolve_cudaq_value_index(
            value,
            vector_slots,
            qubit_map,
            emit_pass,
            bindings,
            operation=operation,
        )
        for value in values
    ]


def _resolve_cudaq_array_slots(
    array_value: Any,
    vector_slots: dict[str, list[int]],
    qubit_map: CudaqControlledQubitMap,
    emit_pass: Any,
    bindings: dict[str, Any],
    *,
    operation: str,
) -> list[int]:
    """Resolve a Vector[Qubit] value to physical CUDA-Q slots.

    Args:
        array_value (Any): ArrayValue or sliced ArrayValue to resolve.
        vector_slots (dict[str, list[int]]): Vector slot map carried by
            the controlled fallback walker.
        qubit_map (CudaqControlledQubitMap): UUID-to-physical-qubit map.
        emit_pass (Any): Emit pass whose resolver folds shapes/slices.
        bindings (dict[str, Any]): Current controlled-body bindings.
        operation (str): Operation label for diagnostics.

    Returns:
        list[int]: Physical qubit slots covered by ``array_value``.

    Raises:
        EmitError: If ``array_value`` is not an ArrayValue, its length
            cannot be resolved, or any element has no physical slot.
    """
    if not isinstance(array_value, ArrayValue):
        raise EmitError(
            f"CUDA-Q controlled helper expected Vector[Qubit] for "
            f"{operation}, got {type(array_value).__name__}.",
            operation=operation,
        )
    if not array_value.is_slice() and array_value.uuid in vector_slots:
        return list(vector_slots[array_value.uuid])
    if not array_value.shape:
        raise EmitError(
            f"CUDA-Q controlled helper cannot resolve lengthless "
            f"Vector[Qubit] {array_value.name!r}.",
            operation=operation,
        )
    size = emit_pass._resolver.resolve_int_value(array_value.shape[0], bindings)
    if size is None:
        raise EmitError(
            f"CUDA-Q controlled helper cannot resolve Vector[Qubit] "
            f"length for {array_value.name!r}.",
            operation=operation,
        )
    if array_value.is_slice():
        if array_value.slice_of is None:
            raise EmitError(
                f"CUDA-Q controlled helper found malformed slice "
                f"{array_value.name!r} without a parent array.",
                operation=operation,
            )
        parent_slots = _resolve_cudaq_array_slots(
            array_value.slice_of,
            vector_slots,
            qubit_map,
            emit_pass,
            bindings,
            operation=operation,
        )
        start = emit_pass._resolver.resolve_int_value(array_value.slice_start, bindings)
        step = emit_pass._resolver.resolve_int_value(array_value.slice_step, bindings)
        if start is None or step is None:
            raise EmitError(
                f"CUDA-Q controlled helper cannot resolve slice bounds "
                f"for {array_value.name!r}.",
                operation=operation,
            )
        slice_slots: list[int] = []
        for i in range(size):
            parent_idx = start + step * i
            if parent_idx < 0 or parent_idx >= len(parent_slots):
                raise EmitError(
                    f"CUDA-Q controlled helper slice index {parent_idx} "
                    f"is outside parent length {len(parent_slots)} for "
                    f"{array_value.name!r}.",
                    operation=operation,
                )
            slice_slots.append(parent_slots[parent_idx])
        vector_slots[array_value.uuid] = slice_slots
        if slice_slots:
            qubit_map[array_value.uuid] = slice_slots[0]
        return slice_slots

    slots: list[int] = []
    for i in range(size):
        addr = QubitAddress(array_value.uuid, i)
        if addr in qubit_map:
            slots.append(qubit_map[addr])
            continue
        raise EmitError(
            f"CUDA-Q controlled helper cannot resolve element {i} "
            f"of Vector[Qubit] {array_value.name!r}.",
            operation=operation,
        )
    vector_slots[array_value.uuid] = slots
    if slots:
        qubit_map.setdefault(array_value.uuid, slots[0])
    return slots


@dataclasses.dataclass
class BoundCudaqKernelArtifact:
    """CUDA-Q kernel artifact with bound parameter values.

    Used as the return type of ``CudaqExecutor.bind_parameters``.
    The executor dispatches to the appropriate CUDA-Q runtime API
    based on ``execution_mode``.

    Args:
        kernel_func: The ``@cudaq.kernel`` decorated function.
        num_qubits: Number of qubits in the circuit.
        num_clbits: Number of classical bits in the circuit.
        param_values: Bound parameter values in order.
        execution_mode: Execution mode inherited from the source artifact.
    """

    kernel_func: Any
    num_qubits: int
    num_clbits: int
    param_values: list[float]
    execution_mode: ExecutionMode = ExecutionMode.STATIC


def _has_runtime_control_flow(
    operations: list[Operation],
    bindings: dict[str, Any],
) -> bool:
    """Check whether operations contain runtime measurement-dependent control flow.

    Returns True if any ``IfOperation`` has a condition that cannot be
    resolved at compile time, or if any ``WhileOperation`` is present.
    """
    for op in operations:
        if isinstance(op, IfOperation):
            if resolve_if_condition(op.condition, bindings) is None:
                return True
            # Also check inside branches
            if _has_runtime_control_flow(op.true_operations, bindings):
                return True
            if _has_runtime_control_flow(op.false_operations, bindings):
                return True
        elif isinstance(op, WhileOperation):
            return True
        elif isinstance(op, ForOperation):
            if _has_runtime_control_flow(op.operations, bindings):
                return True
        elif isinstance(op, ForItemsOperation):
            if _has_runtime_control_flow(op.operations, bindings):
                return True
    return False


def _pauli_evolve_constant_is_significant(
    resolver: Any, op: PauliEvolveOp, bindings: dict[str, Any]
) -> bool:
    """Return whether ``op``'s Hamiltonian carries a droppable constant phase.

    Reports ``True`` when the constant (identity) term is provably non-zero,
    and -- conservatively -- also when the observable cannot be resolved to a
    Hamiltonian from ``bindings`` (so an unknown constant is never silently
    dropped).  ``False`` only when the constant is provably negligible.

    Args:
        resolver (Any): The emit pass value resolver.
        op (PauliEvolveOp): The Pauli evolution op to inspect.
        bindings (dict[str, Any]): Bindings used to resolve the observable.

    Returns:
        bool: ``True`` if a non-negligible (or unresolvable) constant exists.
    """
    import qamomile.observable as qm_o
    from qamomile.observable.hamiltonian import PAULI_TERM_ZERO_ATOL

    hamiltonian = resolver.resolve_bound_value(op.observable, bindings)
    if not isinstance(hamiltonian, qm_o.Hamiltonian):
        return True
    try:
        constant = complex(hamiltonian.constant)
    except (TypeError, ValueError):
        return True
    return abs(constant) >= PAULI_TERM_ZERO_ATOL


def _subtree_drops_constant_phase(
    resolver: Any, operations: Sequence[Operation], bindings: dict[str, Any]
) -> bool:
    """Return whether a PauliEvolveOp unreachable by the phase walker drops a constant.

    Walks the same nested-block channels the controlled-helper validator
    descends -- ``HasNestedOps.nested_op_lists()`` (``for`` / ``if`` / item
    loops), an ``InvokeOperation``'s body, an
    ``InverseBlockOperation``'s source / implementation blocks, and a
    ``ControlledUOperation``'s controlled block -- and reports whether any
    reachable ``PauliEvolveOp`` carries a constant phase that the
    constant-phase emitter cannot re-apply through that construct.  Used to
    decide between raising and skipping for nested shapes the emitter does
    not unroll itself; a zero-constant nested ``pauli_evolve`` stays a no-op.

    Args:
        resolver (Any): The emit pass value resolver.
        operations (Sequence[Operation]): Operations to scan.
        bindings (dict[str, Any]): Best-effort bindings for the scan.

    Returns:
        bool: ``True`` if a droppable constant phase is (or may be) present.
    """
    for op in operations:
        if isinstance(op, PauliEvolveOp):
            if _pauli_evolve_constant_is_significant(resolver, op, bindings):
                return True
            continue
        if isinstance(op, HasNestedOps):
            if any(
                _subtree_drops_constant_phase(resolver, nested, bindings)
                for nested in op.nested_op_lists()
            ):
                return True
            continue
        if isinstance(op, InvokeOperation):
            block = op.effective_body()
            if block is not None and _subtree_drops_constant_phase(
                resolver, block.operations, bindings
            ):
                return True
            continue
        if isinstance(op, InverseBlockOperation):
            if any(
                block is not None
                and _subtree_drops_constant_phase(resolver, block.operations, bindings)
                for block in (op.source_block, op.implementation_block)
            ):
                return True
            continue
        if isinstance(op, ControlledUOperation):
            block = op.block
            if (
                block is not None
                and hasattr(block, "operations")
                and _subtree_drops_constant_phase(resolver, block.operations, bindings)
            ):
                return True
            continue
    return False


def _resolve_real_constant(hamiltonian: Any) -> float:
    """Return the real part of a Hamiltonian's constant term, validating it.

    Args:
        hamiltonian (Any): The resolved Hamiltonian whose constant (identity)
            term is read.

    Returns:
        float: The real part of the constant term.

    Raises:
        EmitError: If the constant is non-numeric, or has a non-negligible
            imaginary part (a non-Hermitian Hamiltonian whose ``exp(-i*gamma*H)``
            evolution would not be unitary).
    """
    from qamomile.observable.hamiltonian import HERMITIAN_IMAG_ATOL

    try:
        constant = complex(hamiltonian.constant)
    except (TypeError, ValueError) as exc:
        raise EmitError(
            f"PauliEvolveOp constant term {hamiltonian.constant!r} is not a "
            f"numeric value.",
            operation="PauliEvolveOp",
        ) from exc
    if abs(constant.imag) > HERMITIAN_IMAG_ATOL:
        raise EmitError(
            f"PauliEvolveOp requires a Hermitian Hamiltonian (real constant "
            f"term), but found constant {hamiltonian.constant}.",
            operation="PauliEvolveOp",
        )
    return constant.real


def _validate_controlled_helper_unitary_ops(
    operations: Sequence[Operation],
    bindings: dict[str, Any],
) -> None:
    """Validate that a CUDA-Q controlled helper is unitary-only.

    Args:
        operations (Sequence[Operation]): Operations in the helper block
            or in a nested control-flow body.
        bindings (dict[str, Any]): Emit-time bindings used to decide
            whether ``if`` conditions are compile-time constants.

    Raises:
        EmitError: If the controlled target contains measurement,
            expectation-value computation, runtime classical state, or
            measurement-dependent control flow.
    """
    non_unitary_ops = (
        MeasureOperation,
        MeasureVectorOperation,
        MeasureQFixedOperation,
        ExpvalOp,
        RuntimeClassicalExpr,
    )
    for op in operations:
        if isinstance(op, non_unitary_ops):
            raise EmitError(
                f"CUDA-Q cudaq.control helper kernels must be unitary-only; "
                f"found {type(op).__name__}.",
                operation="ControlledUOperation",
            )
        if isinstance(op, WhileOperation):
            raise EmitError(
                "CUDA-Q cudaq.control helper kernels must be unitary-only; "
                "runtime while-loops are not supported inside a controlled target.",
                operation="ControlledUOperation",
            )
        if isinstance(op, IfOperation):
            resolved_condition = resolve_if_condition(op.condition, bindings)
            if resolved_condition is None:
                raise EmitError(
                    "CUDA-Q cudaq.control helper kernels must be unitary-only; "
                    "measurement-dependent if-statements are not supported inside "
                    "a controlled target.",
                    operation="ControlledUOperation",
                )
            selected_ops = (
                op.true_operations if resolved_condition else op.false_operations
            )
            _validate_controlled_helper_unitary_ops(selected_ops, bindings)
            continue
        if isinstance(op, ControlledUOperation) and op.block is not None:
            _validate_controlled_helper_unitary_ops(op.block.operations, bindings)
        if isinstance(op, InvokeOperation):
            body = op.effective_body()
            if body is not None:
                _validate_controlled_helper_unitary_ops(body.operations, bindings)
        if isinstance(op, HasNestedOps):
            for nested in op.nested_op_lists():
                _validate_controlled_helper_unitary_ops(nested, bindings)


def _validate_adjoint_helper_ops(
    operations: Sequence[Operation],
    bindings: dict[str, Any],
) -> None:
    """Validate that a CUDA-Q adjoint helper can be emitted safely.

    Args:
        operations (Sequence[Operation]): Operations in the helper block
            or in a nested control-flow body.
        bindings (dict[str, Any]): Emit-time bindings used to decide
            whether ``if`` conditions are compile-time constants.

    Raises:
        EmitError: If the adjoint target contains non-unitary operations or
            nested controlled-kernel synthesis that CUDA-Q 0.14.x aborts on
            when wrapped in ``cudaq.adjoint``.
    """
    _validate_controlled_helper_unitary_ops(operations, bindings)
    for op in operations:
        if isinstance(op, ControlledUOperation):
            raise EmitError(
                "CUDA-Q cudaq.adjoint helper kernels cannot contain nested "
                "controlled-kernel synthesis on CUDA-Q 0.14.x; falling back "
                "to Qamomile inverse decomposition.",
                operation="InverseBlockOperation",
            )
        if isinstance(op, SelectOperation):
            # A SELECT lowers to per-case controlled-U (cudaq.control)
            # helpers at emit; wrapping those in cudaq.adjoint makes NVQIR
            # 0.14.x abort the process when it cannot autogenerate the
            # adjoint. Reject here so the caller falls back to the
            # Qamomile inverse decomposition (which inverts each case
            # block and emits the inverted SELECT directly).
            raise EmitError(
                "CUDA-Q cudaq.adjoint helper kernels cannot contain a nested "
                "SELECT (quantum multiplexer) on CUDA-Q 0.14.x; falling back "
                "to Qamomile inverse decomposition.",
                operation="InverseBlockOperation",
            )
        if isinstance(op, InverseBlockOperation):
            raise EmitError(
                "CUDA-Q cudaq.adjoint helper kernels cannot contain nested "
                "inverse-kernel synthesis on CUDA-Q 0.14.x; falling back "
                "to Qamomile inverse decomposition.",
                operation="InverseBlockOperation",
            )
        if isinstance(op, InvokeOperation):
            body = op.effective_body()
            if body is not None:
                _validate_adjoint_helper_ops(body.operations, bindings)
            continue
        if isinstance(op, IfOperation):
            resolved_condition = resolve_if_condition(op.condition, bindings)
            if resolved_condition is None:
                continue
            selected_ops = (
                op.true_operations if resolved_condition else op.false_operations
            )
            _validate_adjoint_helper_ops(selected_ops, bindings)
            continue
        if isinstance(op, HasNestedOps):
            for nested in op.nested_op_lists():
                _validate_adjoint_helper_ops(nested, bindings)


def _build_helper_qubit_map(
    block_value: Any,
    target_slots: list[int],
    emit_pass: Any,
    bindings: dict[str, Any],
) -> QubitMap:
    """Build a helper-local ``QubitMap`` for a controlled block.

    Args:
        block_value (Any): Inner block whose quantum inputs are mapped.
        target_slots (list[int]): Helper-local qubit slots, one per
            flattened target qubit argument.
        emit_pass (Any): Emit pass used to resolve symbolic vector
            shapes.
        bindings (dict[str, Any]): Bindings used for shape resolution.

    Returns:
        QubitMap: Mapping accepted by ``StandardEmitPass._emit_operations``.

    Raises:
        EmitError: If a vector input length cannot be resolved, is
            negative, or requires more target slots than the controlled
            call supplied.
    """
    from qamomile.circuit.ir.value import ArrayValue

    helper_map: QubitMap = {}
    quantum_inputs = [
        iv
        for iv in getattr(block_value, "input_values", [])
        if hasattr(iv, "type") and iv.type.is_quantum()
    ]

    slot = 0
    for input_value in quantum_inputs:
        if isinstance(input_value, ArrayValue):
            length: int | None = None
            if input_value.shape:
                size_value = input_value.shape[0]
                if size_value.is_constant():
                    length = int(size_value.get_const())
                else:
                    length = emit_pass._resolver.resolve_int_value(size_value, bindings)
            if length is None:
                shape_name = (
                    input_value.shape[0].name if input_value.shape else "<no shape>"
                )
                raise EmitError(
                    f"CUDA-Q controlled helper: cannot resolve "
                    f"Vector[Qubit] input length for {input_value.name!r}; "
                    f"bind {shape_name!r} before transpilation.",
                    operation="ControlledUOperation",
                )
            if length < 0:
                raise EmitError(
                    f"CUDA-Q controlled helper: Vector[Qubit] input "
                    f"{input_value.name!r} resolved to a negative length "
                    f"({length}).",
                    operation="ControlledUOperation",
                )
            if slot + length > len(target_slots):
                raise EmitError(
                    f"CUDA-Q controlled helper inner block requires at least "
                    f"{slot + length} target qubit slot(s), but only "
                    f"{len(target_slots)} are available.",
                    operation="ControlledUOperation",
                )
            if length:
                helper_map[QubitAddress(input_value.uuid)] = target_slots[slot]
            for i in range(length):
                helper_map[QubitAddress(input_value.uuid, i)] = target_slots[slot + i]
            slot += length
            continue

        if slot >= len(target_slots):
            raise EmitError(
                f"CUDA-Q controlled helper inner block requires at least "
                f"{slot + 1} target qubit slot(s), but only "
                f"{len(target_slots)} are available.",
                operation="ControlledUOperation",
            )
        helper_map[QubitAddress(input_value.uuid)] = target_slots[slot]
        slot += 1

    return helper_map


def _collect_loop_carried_clbits(
    operations: list[Operation],
    clbit_map: ClbitMap,
    bindings: dict[str, Any],
    resolver: ValueResolver | None,
) -> set[int]:
    """Collect clbit indices used as loop-carried conditions in WhileOperations.

    CUDA-Q's AST compiler rejects reassignment of parent-scope scalar
    locals inside ``while`` bodies.  This pre-scan identifies which
    canonical clbit indices are used as while-loop conditions so the
    emitter can box them as singleton lists (``__b{i} = [False]``,
    accessed via ``__b{i}[0]``).

    Args:
        operations (list[Operation]): Operations to scan (recursively).
        clbit_map (ClbitMap): Map from ``QubitAddress`` to physical clbit
            index, used to translate a resolved condition address into the
            boxed clbit index.
        bindings (dict[str, Any]): Active emit-time bindings, forwarded to
            ``resolve_condition_address`` so that a ``Vector[Bit]`` element
            condition indexed by a transpile-time-bound parameter resolves
            to its ``(parent_array.uuid, index)`` key rather than falling
            back to the scalar UUID.
        resolver (ValueResolver | None): Resolver used to fold non-constant
            indices through ``bindings``. ``None`` restricts resolution to
            constant indices.

    Returns:
        set[int]: Physical clbit indices that back a while-loop condition.
    """
    result: set[int] = set()
    for op in operations:
        if isinstance(op, WhileOperation) and op.operands:
            cond = op.operands[0]
            cond_val = cond.value if hasattr(cond, "value") else cond
            if isinstance(cond_val, Value):
                # Forward the real bindings / resolver so a Vector[Bit]
                # element indexed by a bound parameter resolves to its
                # (parent_array, index) clbit key; an index that is still
                # unresolved here (e.g. an outer loop variable not yet
                # bound at pre-scan time) falls back to the scalar UUID.
                cond_addr = resolve_condition_address(cond_val, bindings, resolver)
            else:
                cond_addr = QubitAddress(str(cond_val))
            if cond_addr in clbit_map:
                result.add(clbit_map[cond_addr])
            # Also scan inside the while body
            result |= _collect_loop_carried_clbits(
                op.operations, clbit_map, bindings, resolver
            )
        elif isinstance(op, IfOperation):
            result |= _collect_loop_carried_clbits(
                op.true_operations, clbit_map, bindings, resolver
            )
            result |= _collect_loop_carried_clbits(
                op.false_operations, clbit_map, bindings, resolver
            )
        elif isinstance(op, ForOperation):
            result |= _collect_loop_carried_clbits(
                op.operations, clbit_map, bindings, resolver
            )
        elif isinstance(op, ForItemsOperation):
            result |= _collect_loop_carried_clbits(
                op.operations, clbit_map, bindings, resolver
            )
    return result


class CudaqEmitPass(StandardEmitPass[CudaqKernelArtifact]):
    """CUDA-Q-specific emission pass.

    Uses a single ``CudaqKernelEmitter`` for all circuits.  The emitter
    generates ``@cudaq.kernel`` decorated Python source for both static
    and runtime control flow circuits.  The execution mode (STATIC or
    RUNNABLE) is determined by a pre-scan of the operations.
    """

    def __init__(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ):
        parametric = bool(parameters)
        emitter = CudaqKernelEmitter(parametric=parametric)
        composite_emitters: list[Any] = []
        super().__init__(
            emitter,
            bindings,
            parameters,
            composite_emitters,
            backend_name="cudaq",
        )

    def _emit_quantum_segment(
        self,
        operations: list[Operation],
        bindings: dict[str, Any],
    ) -> tuple[CudaqKernelArtifact, QubitMap, ClbitMap]:
        """Emit a quantum segment through the unified codegen path.

        Determines the execution mode via ``_has_runtime_control_flow``,
        configures the emitter accordingly, emits all operations, and
        finalizes the artifact.
        """
        emitter: CudaqKernelEmitter = self._emitter  # type: ignore[assignment]
        is_runtime = _has_runtime_control_flow(operations, bindings)
        mode = ExecutionMode.RUNNABLE if is_runtime else ExecutionMode.STATIC

        # Configure emitter for this segment's mode
        emitter.measurement_mode = (
            MeasurementMode.STATIC
            if mode == ExecutionMode.STATIC
            else MeasurementMode.RUNNABLE
        )

        # Allocate resources
        qubit_map, clbit_map = self._allocator.allocate(operations, bindings)
        qubit_count = max(qubit_map.values()) + 1 if qubit_map else 0
        clbit_count = max(clbit_map.values()) + 1 if clbit_map else 0

        # Pre-scan: identify loop-carried clbits that need singleton-list boxing
        emitter._boxed_clbits = _collect_loop_carried_clbits(
            operations, clbit_map, bindings, self._resolver
        )

        # Create circuit and emit operations
        circuit = emitter.create_circuit(qubit_count, clbit_count)
        self._measurement_qubit_map.clear()
        self._emit_operations(circuit, operations, qubit_map, clbit_map, bindings)

        # Late-bind parametricity: if all parameters were eliminated by
        # compile-time dead branch removal, the kernel signature must be
        # parameterless to match the runtime binding contract.
        emitter._parametric = emitter._param_count > 0

        # For STATIC mode, measurement_qubit_map is populated by base class
        # via measurement_mode property.  For RUNNABLE mode, the kernel returns
        # logical clbit values directly via [__b0, __b1, ...], so
        # measurement_qubit_map is not needed (left empty).

        # Finalize: compile source into @cudaq.kernel function
        circuit = emitter.finalize(circuit, mode)

        return circuit, qubit_map, clbit_map

    def _emit_if(
        self,
        circuit: Any,
        op: IfOperation,
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Handle if-operations for both static and runtime paths.

        Compile-time constant conditions are always handled by the
        base class.  Runtime measurement-dependent conditions are
        delegated to ``StandardEmitPass._emit_if`` when the emitter
        is in RUNNABLE mode (``supports_if_else() == True``), or raise
        ``EmitError`` when in STATIC mode.
        """
        condition = op.condition

        # Compile-time constant conditions are handled by the base class.
        if resolve_if_condition(condition, bindings) is not None:
            super()._emit_if(circuit, op, qubit_map, clbit_map, bindings)
            return

        # Runtime condition: delegate to StandardEmitPass if emitter supports it
        if self._emitter.supports_if_else():
            StandardEmitPass._emit_if(self, circuit, op, qubit_map, clbit_map, bindings)
            return

        # The following error should not be reachable
        # because the mode must have be determined by _has_runtime_control_flow,
        # which checks for unresolvable IfOperation conditions.
        # But guard against misconfiguration to find the bug easily.
        raise EmitError(
            "CUDA-Q 0.14.x does not support measurement-dependent conditional "
            "branching via the builder API. Use a circuit with runtime control "
            "flow support (automatically handled when runtime if/while is detected)."
        )

    def _blockvalue_to_gate(
        self,
        block_value: Any,
        num_qubits: int,
        bindings: dict[str, Any],
        input_operands: list[Any] | None = None,
        operation_name: str = "ControlledUOperation",
    ) -> None:
        """No-op: CUDA-Q codegen does not support sub-circuit gate conversion.

        The base implementation calls ``emitter.create_circuit()`` which
        destructively resets the stateful source builder.  Since CUDA-Q's
        ``circuit_to_gate()`` always returns ``None``, skip the probe
        entirely to avoid corrupting the outer kernel source.

        Args:
            block_value (Any): Ignored nested block.
            num_qubits (int): Ignored nested circuit width.
            bindings (dict[str, Any]): Ignored emit bindings.
            input_operands (list[Any] | None): Ignored call-site operands.
                Defaults to None.
            operation_name (str): Ignored diagnostic operation name.
                Defaults to ``"ControlledUOperation"``.
        """
        return None

    def _emit_custom_composite(
        self,
        circuit: CudaqKernelArtifact,
        op: Any,
        impl: Any,
        qubit_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit custom composites through the shared fallback path.

        Args:
            circuit (CudaqKernelArtifact): CUDA-Q kernel artifact being
                emitted.
            op (Any): Composite gate operation.
            impl (Any): Qamomile fallback implementation block.
            qubit_indices (list[int]): Physical target qubit indices.
            bindings (dict[str, Any]): Active emit bindings.
        """
        super()._emit_custom_composite(circuit, op, impl, qubit_indices, bindings)

    def _emit_pauli_evolve(
        self,
        circuit: CudaqKernelArtifact,
        op: PauliEvolveOp,
        qubit_map: QubitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Emit Pauli evolution natively via CUDA-Q ``exp_pauli``.

        Overrides the shared gadget (``h`` / ``sdg`` / ``rz`` + CX-ladder)
        decomposition: each Hamiltonian term ``coeff * P`` becomes a single
        ``exp_pauli(-gamma * coeff, qubits, P)`` call, letting CUDA-Q's
        compiler synthesize and optimize the rotation.  The Hamiltonian's
        constant (identity) term stays an unobservable global phase here and
        is dropped, exactly as the gadget did; under ``qmc.control`` it is
        re-applied by :meth:`_emit_controlled_constant_phases` (CUDA-Q
        forbids ``exp_pauli`` of an all-identity word, so it must never be
        emitted as a term).

        Args:
            circuit (CudaqKernelArtifact): Artifact being built.
            op (PauliEvolveOp): Pauli evolution operation.
            qubit_map (QubitMap): Quantum value to physical qubit map.
            bindings (dict[str, Any]): Active emit bindings.

        Raises:
            EmitError: If the observable is not a Hamiltonian, gamma cannot
                be resolved, the Hamiltonian is larger than the qubit
                register, a term coefficient or the constant (identity)
                term is non-real (non-Hermitian), or a register qubit
                cannot be resolved from the qubit map.
        """
        import qamomile.observable as qm_o
        from qamomile.observable.hamiltonian import (
            HERMITIAN_IMAG_ATOL,
            PAULI_TERM_ZERO_ATOL,
        )

        hamiltonian = self._resolver.resolve_bound_value(op.observable, bindings)
        if not isinstance(hamiltonian, qm_o.Hamiltonian):
            raise EmitError(
                f"PauliEvolveOp requires a Hamiltonian binding. "
                f"Observable '{op.observable.name}' not found or not a Hamiltonian.",
                operation="PauliEvolveOp",
            )

        gamma = _resolve_gamma(self, op, bindings)
        if gamma is None:
            raise EmitError(
                "Cannot resolve gamma parameter for PauliEvolveOp. "
                "gamma must be a concrete float binding or a declared "
                "parameter (scalar or array element).",
                operation="PauliEvolveOp",
            )

        input_array = op.qubits
        assert isinstance(input_array, ArrayValue)
        num_h_qubits = hamiltonian.num_qubits
        register_qubit_count = num_h_qubits
        if input_array.shape:
            n_resolved = self._resolver.resolve_int_value(
                input_array.shape[0], bindings
            )
            if n_resolved is not None:
                register_qubit_count = n_resolved
                validate_hamiltonian_within_register(num_h_qubits, n_resolved)

        for operators, coeff in hamiltonian:
            if abs(coeff.imag) > HERMITIAN_IMAG_ATOL:
                raise EmitError(
                    f"PauliEvolveOp requires a Hermitian Hamiltonian "
                    f"(real coefficients), but found complex coefficient "
                    f"{coeff} on term {operators}.",
                    operation="PauliEvolveOp",
                )
        # Validate the constant (identity) term is real too. It is dropped as
        # a global phase here, but a complex constant means a non-Hermitian
        # Hamiltonian whose evolution is non-unitary, so it must not pass
        # silently (the shared controlled path enforces the same).
        _resolve_real_constant(hamiltonian)

        root_av, slice_start, slice_step = self._resolver.resolve_slice_chain(
            input_array, bindings, operation="PauliEvolveOp"
        )
        register_qubit_indices: list[int] = []
        for i in range(register_qubit_count):
            addr = QubitAddress(root_av.uuid, slice_start + slice_step * i)
            if addr in qubit_map:
                register_qubit_indices.append(qubit_map[addr])
            else:
                raise EmitError(
                    f"Cannot resolve qubit index {i} for PauliEvolveOp. "
                    f"Key '{addr!s}' not found in qubit_map.",
                    operation="PauliEvolveOp",
                )
        qubit_indices = register_qubit_indices[:num_h_qubits]

        emitter: CudaqKernelEmitter = self._emitter  # type: ignore[assignment]
        for operators, coeff in hamiltonian:
            if abs(coeff) < PAULI_TERM_ZERO_ATOL:
                continue
            # Build the term from its non-identity factors only: CUDA-Q
            # rejects an all-identity ``exp_pauli`` word, and identity
            # factors contribute nothing to the rotation.
            term_qubits: list[int] = []
            pauli_letters: list[str] = []
            for op_item in operators:
                if op_item.pauli == qm_o.Pauli.I:  # identity factor -- skip
                    continue
                term_qubits.append(qubit_indices[op_item.index])
                pauli_letters.append(qm_o.PAULI_TO_CHAR[op_item.pauli])
            if not term_qubits:
                continue
            # ``exp_pauli`` realizes exp(+i*theta*P); to get
            # exp(-i*gamma*coeff*P) set theta = -gamma*coeff (no factor of
            # two, unlike the RZ gadget which uses 2*gamma*coeff).
            angle: Any
            if isinstance(gamma, (int, float)):
                angle = -float(coeff.real * gamma)
            else:
                angle = (-float(coeff.real)) * gamma
            emitter.emit_exp_pauli(circuit, term_qubits, "".join(pauli_letters), angle)

        # Map the result array onto the same physical qubits, mirroring the
        # shared ``emit_pauli_evolve`` bookkeeping so downstream lookups via
        # either the result uuid or its slice-chain root resolve.
        result_array = op.evolved_qubits
        assert isinstance(result_array, ArrayValue)
        result_root, result_start, result_step = self._resolver.resolve_slice_chain(
            result_array, bindings, operation="PauliEvolveOp"
        )
        for i, phys_idx in enumerate(register_qubit_indices):
            direct_addr = QubitAddress(result_array.uuid, i)
            if direct_addr not in qubit_map:
                qubit_map[direct_addr] = phys_idx
            root_addr = QubitAddress(result_root.uuid, result_start + result_step * i)
            if root_addr not in qubit_map:
                qubit_map[root_addr] = phys_idx

    def _emit_inverse_block(
        self,
        circuit: CudaqKernelArtifact,
        op: InverseBlockOperation,
        qubit_map: QubitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Emit a first-class inverse block into CUDA-Q.

        Args:
            circuit (CudaqKernelArtifact): CUDA-Q kernel artifact being
                emitted.
            op (InverseBlockOperation): Inverse block operation to emit.
            qubit_map (QubitMap): Current quantum value to physical qubit map.
            bindings (dict[str, Any]): Active emit bindings.

        Raises:
            EmitError: If neither CUDA-Q adjoint nor the shared fallback can
                emit the inverse operation.
            SliceBorrowViolationError: If a nested block's slice usage fails
                the borrow check run by ``_normalize_inverse_block_op``.
        """
        # Strip nested slice markers (after the borrow check) before the
        # adjoint path or the shared fallback walks the blocks' operations
        # directly; see ``_normalize_inverse_block_op``.
        op = _normalize_inverse_block_op(op, bindings)
        if op.num_control_qubits == 0 and op.source_block is not None:
            try:
                target_index_groups = [
                    _expand_quantum_operands_to_phys(
                        self,
                        operand,
                        qubit_map,
                        bindings,
                        operation="InverseBlockOperation",
                    )
                    for operand in op.target_qubits
                ]
                target_indices = [
                    index for group in target_index_groups for index in group
                ]
            except EmitError:
                target_index_groups = []
                target_indices = []
            if len(target_indices) == op.num_target_qubits:
                # CUDA-Q's adjoint fast path builds a helper kernel and never
                # routes through the base ``emit_inverse_block_at_indices``
                # entry check, so run the shared aliasing check here. This path
                # is uncontrolled (num_control_qubits == 0), so the combined set
                # is exactly ``target_indices``: an inverse block applied to
                # aliased targets (``inverse(u)(qs[i], qs[j])`` on the diagonal)
                # would otherwise compile silently and crash the simulator.
                reject_duplicate_physical_indices(
                    "inverse block (CUDA-Q adjoint)", target_indices
                )
                try:
                    self._emit_adjoint_helper(
                        circuit,
                        op.source_block,
                        [*op.target_qubits, *op.parameters],
                        target_indices,
                        bindings,
                    )
                    _map_inverse_block_results(
                        op,
                        [],
                        target_index_groups,
                        qubit_map,
                    )
                    return
                except EmitError:
                    pass

        super()._emit_inverse_block(circuit, op, qubit_map, bindings)

    def _emit_adjoint_helper(
        self,
        circuit: CudaqKernelArtifact,
        block_value: Any,
        input_operands: list[Any] | None,
        target_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit ``cudaq.adjoint`` for a nested source block.

        Args:
            circuit (CudaqKernelArtifact): CUDA-Q kernel artifact being
                emitted.
            block_value (Any): Source block whose inverse should be
                emitted by CUDA-Q.
            input_operands (list[Any] | None): Call-site operands used to
                bind block inputs. Defaults to None.
            target_indices (list[int]): Physical target qubit indices for
                the adjoint call.
            bindings (dict[str, Any]): Active emit bindings.

        Raises:
            EmitError: If the block cannot be emitted as a CUDA-Q pure
                device helper.
        """
        if not hasattr(block_value, "operations"):
            raise EmitError(
                "Cannot emit CUDA-Q adjoint: block has no operations.",
                operation="InverseBlockOperation",
            )

        _validate_adjoint_helper_ops(block_value.operations, bindings)

        local_bindings = _bind_block_inputs(
            self,
            block_value,
            input_operands,
            bindings,
        )
        quantum_operands = _quantum_input_operands(block_value, input_operands)
        _bind_quantum_input_shapes(
            self._resolver,
            block_value,
            quantum_operands,
            bindings,
            local_bindings,
        )

        helper_targets = list(range(len(target_indices)))
        helper_qubit_map = _build_helper_qubit_map(
            block_value,
            helper_targets,
            self,
            local_bindings,
        )
        emitter: CudaqKernelEmitter = self._emitter  # type: ignore[assignment]

        def emit_body() -> None:
            """Emit the adjoint helper body into the helper source."""
            self._emit_operations(
                circuit,
                block_value.operations,
                helper_qubit_map,
                {},
                local_bindings,
                force_unroll=True,
            )

        helper_name, uses_thetas = emitter.build_adjoint_helper(
            len(target_indices),
            emit_body,
        )
        emitter.emit_adjoint_kernel_call(
            circuit,
            helper_name,
            target_indices,
            uses_thetas,
        )

    def _emit_controlled_fallback(
        self,
        circuit: CudaqKernelArtifact,
        block_value: Any,
        num_controls: int,
        control_indices: list[int],
        target_indices: list[int],
        power: int,
        bindings: dict[str, Any],
    ) -> None:
        """Emit controlled-U through a generated CUDA-Q helper kernel.

        Generates a pure-device ``@cudaq.kernel`` for the wrapped
        Qamomile block and emits ``cudaq.control(helper, controls,
        *targets)`` in the outer kernel.  This uses CUDA-Q's native
        controlled-kernel support instead of decomposing the helper
        gate-by-gate.

        Args:
            circuit (CudaqKernelArtifact): The CUDA-Q kernel artifact
                being built.
            block_value (Any): The block value containing operations to
                control.
            num_controls (int): Number of control qubits.
            control_indices (list[int]): Physical indices of control
                qubits.
            target_indices (list[int]): Physical indices of target
                qubits.
            power (int): Number of times to repeat the controlled
                operation.
            bindings (dict[str, Any]): Parameter bindings local to the
                controlled block.

        Raises:
            EmitError: When the controlled block is missing operations or
                contains non-unitary runtime constructs.
        """
        if not hasattr(block_value, "operations"):
            raise EmitError(
                "Cannot emit controlled operation: block has no operations.",
                operation="ControlledUOperation",
            )
        if num_controls <= 0:
            raise EmitError(
                "CUDA-Q cudaq.control requires at least one control qubit.",
                operation="ControlledUOperation",
            )
        if num_controls != len(control_indices):
            raise EmitError(
                f"CUDA-Q controlled fallback received num_controls="
                f"{num_controls} with {len(control_indices)} physical "
                "control indices.",
                operation="ControlledUOperation",
            )

        # CUDA-Q's fallback builds a helper kernel via
        # ``emit_controlled_kernel_call`` and never routes through
        # ``append_gate``, so the shared aliasing check must run here. A
        # control coinciding with a target (or a duplicated target) would make
        # ``cudaq.control`` act twice on one qubit and crash the simulator.
        reject_duplicate_physical_indices(
            "controlled gate (CUDA-Q fallback)", control_indices + target_indices
        )

        _validate_controlled_helper_unitary_ops(block_value.operations, bindings)

        helper_targets = list(range(len(target_indices)))
        helper_qubit_map = _build_helper_qubit_map(
            block_value, helper_targets, self, bindings
        )

        emitter: CudaqKernelEmitter = self._emitter  # type: ignore[assignment]

        def emit_body() -> None:
            """Emit the wrapped block body into the helper source."""
            self._emit_operations(
                circuit,
                block_value.operations,
                helper_qubit_map,
                {},
                bindings,
                force_unroll=True,
            )

        helper_name, uses_thetas = emitter.build_controlled_helper(
            len(target_indices),
            emit_body,
        )

        for _ in range(power):
            emitter.emit_controlled_kernel_call(
                circuit,
                helper_name,
                control_indices,
                target_indices,
                uses_thetas,
            )

        # ``cudaq.control`` controls only the helper's emitted gates.  The
        # helper lowers each PauliEvolveOp through the *uncontrolled* path,
        # which drops the Hamiltonian's constant (identity) term as an
        # unobservable global phase.  Under control that phase is
        # observable, so re-apply it explicitly on the controls.
        self._emit_controlled_constant_phases(
            circuit,
            block_value.operations,
            control_indices,
            power,
            bindings,
            emitter,
        )

    def _emit_controlled_constant_phases(
        self,
        circuit: CudaqKernelArtifact,
        operations: list[Any],
        control_indices: list[int],
        power: int,
        bindings: dict[str, Any],
        emitter: CudaqKernelEmitter,
    ) -> None:
        """Emit the controlled identity-term phases the helper body dropped.

        The native ``cudaq.control(helper)`` path lowers every
        ``PauliEvolveOp`` through the uncontrolled path, which discards each
        Hamiltonian's constant (identity) term ``c`` as an unobservable
        global phase ``exp(-i * gamma * c)``.  ``cudaq.control`` wraps only
        the helper's emitted gates, so that global phase is lost -- yet
        under control it is an observable relative phase between the
        control-on and control-off branches.  For every ``PauliEvolveOp``
        the helper emits, this re-applies the missing phase as a
        multi-controlled ``P(-power * gamma * c)`` on the controls (a plain
        phase gate when there is a single control), mirroring the shared
        :func:`emit_controlled_pauli_evolve` constant-term handling.  The
        ``power`` factor accounts for the controlled-U**power repetition of
        the helper call above.

        Args:
            circuit (CudaqKernelArtifact): Artifact being built.
            operations (list[Any]): Controlled block body operations.
            control_indices (list[int]): Physical outer control qubits.
            power (int): Controlled-U power applied to the helper.
            bindings (dict[str, Any]): Controlled-body bindings.
            emitter (CudaqKernelEmitter): CUDA-Q source emitter.

        Raises:
            EmitError: If a ``PauliEvolveOp`` with a non-zero constant term
                is nested inside a construct whose controlled constant-phase
                contribution cannot be resolved here; raising avoids
                silently dropping an observable phase.
        """
        if power == 0 or not control_indices:
            return
        self._collect_controlled_constant_phases(
            circuit, operations, control_indices, power, bindings, emitter
        )

    def _collect_controlled_constant_phases(
        self,
        circuit: CudaqKernelArtifact,
        operations: list[Any],
        control_indices: list[int],
        power: int,
        bindings: dict[str, Any],
        emitter: CudaqKernelEmitter,
    ) -> None:
        """Walk ``operations`` and emit each PauliEvolveOp's controlled phase.

        Mirrors the compile-time control-flow unrolling the uncontrolled
        helper body performs (``for`` loops and constant ``if`` branches),
        so the emitted phases line up one-for-one with the gadgets the
        helper actually produced.  Any other construct that could hide a
        ``PauliEvolveOp`` -- composite, inverse, nested controlled-U, or an
        item loop -- raises only when that op would drop a non-zero constant
        phase; a zero-constant nested ``pauli_evolve`` stays a no-op.

        Args:
            circuit (CudaqKernelArtifact): Artifact being built.
            operations (list[Any]): Operations to walk.
            control_indices (list[int]): Physical outer control qubits.
            power (int): Controlled-U power applied to the helper.
            bindings (dict[str, Any]): Current body bindings.
            emitter (CudaqKernelEmitter): CUDA-Q source emitter.

        Raises:
            EmitError: On a ``PauliEvolveOp`` with a non-zero constant term
                nested in an unsupported construct, or when ``for`` / ``if``
                control flow guarding such an op cannot be resolved at
                compile time.
        """
        from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (  # noqa: I001
            _bind_loop_var,
            resolve_loop_bounds,
        )

        for op in operations:
            if isinstance(op, PauliEvolveOp):
                self._emit_one_controlled_constant_phase(
                    circuit, op, control_indices, power, bindings, emitter
                )
                continue
            if isinstance(op, ForOperation):
                start, stop, step = resolve_loop_bounds(self._resolver, op, bindings)
                if start is None or stop is None or step is None:
                    if _subtree_drops_constant_phase(
                        self._resolver, op.operations, bindings
                    ):
                        raise EmitError(
                            "Cannot resolve ForOperation bounds while emitting the "
                            "controlled PauliEvolve constant phase; the identity-term "
                            "phase would be dropped under control.",
                            operation="PauliEvolveOp",
                        )
                    continue
                for i in range(start, stop, step):
                    loop_bindings = bindings.copy()
                    _bind_loop_var(loop_bindings, op, i)
                    self._collect_controlled_constant_phases(
                        circuit,
                        op.operations,
                        control_indices,
                        power,
                        loop_bindings,
                        emitter,
                    )
                continue
            if isinstance(op, IfOperation):
                resolved = resolve_if_condition(op.condition, bindings)
                # The controlled-helper validator already rejected
                # measurement-dependent ``if``s, so a reachable IfOperation
                # has a compile-time-constant condition here.
                if resolved is None:
                    if _subtree_drops_constant_phase(
                        self._resolver, op.true_operations, bindings
                    ) or _subtree_drops_constant_phase(
                        self._resolver, op.false_operations, bindings
                    ):
                        raise EmitError(
                            "Cannot resolve IfOperation condition while emitting the "
                            "controlled PauliEvolve constant phase.",
                            operation="PauliEvolveOp",
                        )
                    continue
                selected = op.true_operations if resolved else op.false_operations
                self._collect_controlled_constant_phases(
                    circuit, selected, control_indices, power, bindings, emitter
                )
                continue
            # Any other construct that could transitively hide a
            # PauliEvolveOp (composite / inverse / nested controlled-U /
            # item loop): raise only when a non-zero constant phase would
            # actually be dropped.  A zero-constant nested pauli_evolve is
            # unaffected and stays a no-op here.
            if _subtree_drops_constant_phase(self._resolver, [op], bindings):
                raise EmitError(
                    f"CUDA-Q controlled PauliEvolve constant-phase emission does not "
                    f"support a PauliEvolveOp with a non-zero constant term nested "
                    f"inside {type(op).__name__}; its Hamiltonian's identity-term "
                    f"phase would be silently dropped under control.  Restructure the "
                    f"controlled sub-kernel so the pauli_evolve appears directly "
                    f"(optionally inside a compile-time for-loop or if-branch).",
                    operation="PauliEvolveOp",
                )
            # Plain gates / binops / returns contribute no constant phase.

    def _emit_one_controlled_constant_phase(
        self,
        circuit: CudaqKernelArtifact,
        op: PauliEvolveOp,
        control_indices: list[int],
        power: int,
        bindings: dict[str, Any],
        emitter: CudaqKernelEmitter,
    ) -> None:
        """Emit ``P(-power * gamma * c)`` on the controls for one PauliEvolveOp.

        Resolves the Hamiltonian's constant term ``c`` and ``gamma`` the
        same way the body emit does, then applies the missing identity-term
        phase as a multi-controlled phase gate, matching the shared
        :func:`emit_controlled_pauli_evolve` formula ``P(-gamma * c)``.
        No-op when ``c`` is negligible, so constant-free Hamiltonians keep
        their previous emission byte-for-byte.

        Args:
            circuit (CudaqKernelArtifact): Artifact being built.
            op (PauliEvolveOp): Pauli evolution op in the controlled body.
            control_indices (list[int]): Physical outer control qubits
                (guaranteed non-empty by the caller).
            power (int): Controlled-U power applied to the helper.
            bindings (dict[str, Any]): Current controlled-body bindings.
            emitter (CudaqKernelEmitter): CUDA-Q source emitter.

        Raises:
            EmitError: If the observable is not a Hamiltonian, gamma cannot
                be resolved, or the constant term is non-real (the
                evolution would be non-unitary).
        """
        import qamomile.observable as qm_o
        from qamomile.observable.hamiltonian import PAULI_TERM_ZERO_ATOL

        hamiltonian = self._resolver.resolve_bound_value(op.observable, bindings)
        if not isinstance(hamiltonian, qm_o.Hamiltonian):
            raise EmitError(
                f"PauliEvolveOp requires a Hamiltonian binding. "
                f"Observable '{op.observable.name}' not found or not a Hamiltonian.",
                operation="PauliEvolveOp",
            )

        constant_real = _resolve_real_constant(hamiltonian)
        if abs(constant_real) < PAULI_TERM_ZERO_ATOL:
            return

        gamma = _resolve_gamma(self, op, bindings)
        if gamma is None:
            raise EmitError(
                "Cannot resolve gamma parameter for PauliEvolveOp. "
                "gamma must be a concrete float binding or a declared "
                "parameter (scalar or array element).",
                operation="PauliEvolveOp",
            )

        # exp(-i*gamma*c*I) is a global phase exp(-i*gamma*c) on the target
        # register; controlled, it is P(-gamma*c) on the controls.  Folding
        # the controlled-U power gives P(-power*gamma*c).  emit_multi_-
        # controlled_p reduces to a plain phase gate for a single control.
        scale = -float(power) * constant_real
        phase: Any
        if isinstance(gamma, (int, float)):
            phase = scale * float(gamma)
        else:
            phase = scale * gamma

        emitter.emit_multi_controlled_p(
            circuit, control_indices[:-1], control_indices[-1], phase
        )

    def _emit_cudaq_controlled_ops(
        self,
        circuit: CudaqKernelArtifact,
        operations: list[Any],
        num_controls: int,
        control_indices: list[int],
        target_indices: list[int],
        qubit_map: CudaqControlledQubitMap,
        vector_slots: dict[str, list[int]],
        emitter: CudaqKernelEmitter,
        bindings: dict[str, Any],
    ) -> None:
        """Recursively emit controlled operations with operand-to-target mapping.

        Handles primitive gates via the CUDA-Q multi-control path,
        composite/Pauli evolution operations by controlled decomposition,
        ``ReturnOperation`` by skipping, and ``ForOperation`` by unrolling
        compile-time resolvable loops.

        Args:
            circuit: The CUDA-Q kernel artifact being built.
            operations: List of operations to process.
            num_controls: Number of control qubits.
            control_indices: Physical indices of control qubits.
            target_indices: Physical indices of target qubits.
            qubit_map: Mutable UUID-to-physical-target map for SSA tracking.
            vector_slots: Mutable ArrayValue UUID-to-physical-slots map
                used for Vector[Qubit] operands and results.
            emitter: The CUDA-Q kernel emitter.
            bindings: Parameter bindings.

        Raises:
            EmitError: When an unsupported operation is encountered.
        """
        for op in operations:
            if isinstance(op, ReturnOperation):
                continue
            if isinstance(op, BinOp):
                evaluate_binop(self, op, bindings)
                continue
            if isinstance(op, GateOperation):
                for operand in op.qubit_operands:
                    _seed_vector_element_uuid(
                        operand,
                        vector_slots,
                        qubit_map,
                        self,
                        bindings,
                    )
                gate_target_indices = _resolve_gate_targets(op, qubit_map)
                self._emit_cudaq_multi_controlled_gate(
                    circuit,
                    op,
                    emitter,
                    control_indices,
                    gate_target_indices,
                    bindings,
                )
                self._propagate_cudaq_gate_results(op, qubit_map)
                continue
            if isinstance(op, InvokeOperation):
                self._emit_cudaq_controlled_composite(
                    circuit,
                    op,
                    num_controls,
                    control_indices,
                    target_indices,
                    qubit_map,
                    vector_slots,
                    emitter,
                    bindings,
                )
                continue
            if isinstance(op, InverseBlockOperation):
                self._emit_cudaq_controlled_inverse(
                    circuit,
                    op,
                    num_controls,
                    control_indices,
                    target_indices,
                    qubit_map,
                    vector_slots,
                    emitter,
                    bindings,
                )
                continue
            if isinstance(op, PauliEvolveOp):
                self._emit_cudaq_controlled_pauli_evolve(
                    circuit,
                    op,
                    num_controls,
                    control_indices,
                    target_indices,
                    qubit_map,
                    vector_slots,
                    emitter,
                    bindings,
                )
                continue
            if isinstance(op, ForOperation):
                from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
                    resolve_loop_bounds,
                )

                start, stop, step = resolve_loop_bounds(self._resolver, op, bindings)
                if start is not None and stop is not None and step is not None:
                    from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
                        _bind_loop_var,
                    )

                    for i in range(start, stop, step):
                        loop_bindings = bindings.copy()
                        _bind_loop_var(loop_bindings, op, i)
                        self._emit_cudaq_controlled_ops(
                            circuit,
                            op.operations,
                            num_controls,
                            control_indices,
                            target_indices,
                            qubit_map,
                            vector_slots,
                            emitter,
                            loop_bindings,
                        )
                else:
                    raise EmitError(
                        "Cannot resolve ForOperation bounds in CUDA-Q "
                        "controlled block body.",
                        operation="ControlledUOperation",
                    )
                continue
            raise EmitError(
                f"Unsupported operation {type(op).__name__} in "
                f"CUDA-Q controlled block body.",
                operation="ControlledUOperation",
            )

    def _propagate_cudaq_gate_results(
        self,
        op: GateOperation,
        qubit_map: CudaqControlledQubitMap,
    ) -> None:
        """Propagate physical qubit slots from gate operands to results.

        Args:
            op (GateOperation): Gate whose results become fresh SSA
                versions of its qubit operands.
            qubit_map (CudaqControlledQubitMap): UUID-to-physical-target
                map to update in place.
        """
        qubit_ops = op.qubit_operands
        assert len(op.results) == len(qubit_ops), (
            f"[For DEVELOPER] GateOperation must have equal qubit operands/results, "
            f"got {len(qubit_ops)} qubit operands and {len(op.results)} results."
            f"There must be a bug."
        )
        for operand, result in zip(qubit_ops, op.results, strict=True):
            assert result.type.is_quantum(), (
                "[For DEVELOPER] GateOperation result must be quantum. "
                "There must be a bug."
            )
            assert operand.uuid in qubit_map, (
                f"Missing qubit mapping for operand {operand.uuid} in controlled helper."
            )
            qubit_map[result.uuid] = qubit_map[operand.uuid]

    def _emit_cudaq_controlled_h(
        self,
        circuit: CudaqKernelArtifact,
        emitter: CudaqKernelEmitter,
        control_indices: list[int],
        target_idx: int,
    ) -> None:
        """Emit a controlled Hadamard with one or more controls.

        Args:
            circuit (CudaqKernelArtifact): Artifact being built.
            emitter (CudaqKernelEmitter): CUDA-Q source emitter.
            control_indices (list[int]): Physical control qubit indices.
            target_idx (int): Physical target qubit index.
        """
        if not control_indices:
            emitter.emit_h(circuit, target_idx)
            return
        emitter.emit_ry(circuit, target_idx, math.pi / 4)
        emitter.emit_multi_controlled_x(circuit, control_indices, target_idx)
        emitter.emit_ry(circuit, target_idx, -math.pi / 4)

    def _emit_cudaq_controlled_y(
        self,
        circuit: CudaqKernelArtifact,
        emitter: CudaqKernelEmitter,
        control_indices: list[int],
        target_idx: int,
    ) -> None:
        """Emit a controlled Pauli-Y with one or more controls.

        Args:
            circuit (CudaqKernelArtifact): Artifact being built.
            emitter (CudaqKernelEmitter): CUDA-Q source emitter.
            control_indices (list[int]): Physical control qubit indices.
            target_idx (int): Physical target qubit index.
        """
        if not control_indices:
            emitter.emit_y(circuit, target_idx)
            return
        emitter.emit_sdg(circuit, target_idx)
        emitter.emit_multi_controlled_x(circuit, control_indices, target_idx)
        emitter.emit_s(circuit, target_idx)

    def _emit_cudaq_controlled_z(
        self,
        circuit: CudaqKernelArtifact,
        emitter: CudaqKernelEmitter,
        control_indices: list[int],
        target_idx: int,
    ) -> None:
        """Emit a controlled Pauli-Z with one or more controls.

        Args:
            circuit (CudaqKernelArtifact): Artifact being built.
            emitter (CudaqKernelEmitter): CUDA-Q source emitter.
            control_indices (list[int]): Physical control qubit indices.
            target_idx (int): Physical target qubit index.
        """
        if not control_indices:
            emitter.emit_z(circuit, target_idx)
            return
        emitter.emit_h(circuit, target_idx)
        emitter.emit_multi_controlled_x(circuit, control_indices, target_idx)
        emitter.emit_h(circuit, target_idx)

    def _emit_cudaq_controlled_swap(
        self,
        circuit: CudaqKernelArtifact,
        emitter: CudaqKernelEmitter,
        control_indices: list[int],
        target_indices: list[int],
    ) -> None:
        """Emit a controlled SWAP with one or more controls.

        Args:
            circuit (CudaqKernelArtifact): Artifact being built.
            emitter (CudaqKernelEmitter): CUDA-Q source emitter.
            control_indices (list[int]): Physical control qubit indices.
            target_indices (list[int]): Two physical target qubit indices.

        Raises:
            EmitError: If fewer than two target qubits are supplied.
        """
        if len(target_indices) < 2:
            raise EmitError(
                "Controlled-SWAP requires at least 2 target qubits.",
                operation="ControlledGate",
            )
        tgt_a = target_indices[0]
        tgt_b = target_indices[1]
        if not control_indices:
            emitter.emit_swap(circuit, tgt_a, tgt_b)
            return
        emitter.emit_cx(circuit, tgt_b, tgt_a)
        emitter.emit_multi_controlled_x(circuit, control_indices + [tgt_a], tgt_b)
        emitter.emit_cx(circuit, tgt_b, tgt_a)

    def _emit_cudaq_controlled_cp(
        self,
        circuit: CudaqKernelArtifact,
        emitter: CudaqKernelEmitter,
        control_indices: list[int],
        target_indices: list[int],
        angle: Any,
    ) -> None:
        """Emit a controlled controlled-phase operation.

        Args:
            circuit (CudaqKernelArtifact): Artifact being built.
            emitter (CudaqKernelEmitter): CUDA-Q source emitter.
            control_indices (list[int]): Outer physical controls.
            target_indices (list[int]): Inner CP control and target.
            angle (Any): Phase angle.

        Raises:
            EmitError: If fewer than two target qubits are supplied.
        """
        if len(target_indices) < 2:
            raise EmitError(
                "Controlled-CP requires at least 2 target qubits "
                "(inner control + inner target).",
                operation="ControlledGate",
            )
        inner_control = target_indices[0]
        inner_target = target_indices[1]
        if not control_indices:
            emitter.emit_cp(circuit, inner_control, inner_target, angle)
            return
        emitter.emit_multi_controlled_p(
            circuit,
            control_indices + [inner_control],
            inner_target,
            angle,
        )

    def _emit_cudaq_controlled_rzz(
        self,
        circuit: CudaqKernelArtifact,
        emitter: CudaqKernelEmitter,
        control_indices: list[int],
        target_indices: list[int],
        angle: Any,
    ) -> None:
        """Emit a controlled RZZ rotation.

        Args:
            circuit (CudaqKernelArtifact): Artifact being built.
            emitter (CudaqKernelEmitter): CUDA-Q source emitter.
            control_indices (list[int]): Physical control qubit indices.
            target_indices (list[int]): Two physical RZZ operands.
            angle (Any): Rotation angle.

        Raises:
            EmitError: If fewer than two target qubits are supplied.
        """
        if len(target_indices) < 2:
            raise EmitError(
                "Controlled-RZZ requires at least 2 target qubits.",
                operation="ControlledGate",
            )
        q0 = target_indices[0]
        q1 = target_indices[1]
        if not control_indices:
            emitter.emit_rzz(circuit, q0, q1, angle)
            return
        emitter.emit_cx(circuit, q0, q1)
        emitter.emit_multi_controlled_rz(circuit, control_indices, q1, angle)
        emitter.emit_cx(circuit, q0, q1)

    def _emit_cudaq_controlled_qft(
        self,
        circuit: CudaqKernelArtifact,
        emitter: CudaqKernelEmitter,
        control_indices: list[int],
        qubit_indices: list[int],
    ) -> None:
        """Emit a controlled QFT decomposition.

        Args:
            circuit (CudaqKernelArtifact): Artifact being built.
            emitter (CudaqKernelEmitter): CUDA-Q source emitter.
            control_indices (list[int]): Outer physical controls.
            qubit_indices (list[int]): QFT target qubits.
        """
        n = len(qubit_indices)
        for j in range(n - 1, -1, -1):
            self._emit_cudaq_controlled_h(
                circuit, emitter, control_indices, qubit_indices[j]
            )
            for k in range(j - 1, -1, -1):
                angle = math.pi / (2 ** (j - k))
                self._emit_cudaq_controlled_cp(
                    circuit,
                    emitter,
                    control_indices,
                    [qubit_indices[j], qubit_indices[k]],
                    angle,
                )
        for j in range(n // 2):
            self._emit_cudaq_controlled_swap(
                circuit,
                emitter,
                control_indices,
                [qubit_indices[j], qubit_indices[n - 1 - j]],
            )

    def _emit_cudaq_controlled_iqft(
        self,
        circuit: CudaqKernelArtifact,
        emitter: CudaqKernelEmitter,
        control_indices: list[int],
        qubit_indices: list[int],
    ) -> None:
        """Emit a controlled inverse-QFT decomposition.

        Args:
            circuit (CudaqKernelArtifact): Artifact being built.
            emitter (CudaqKernelEmitter): CUDA-Q source emitter.
            control_indices (list[int]): Outer physical controls.
            qubit_indices (list[int]): IQFT target qubits.
        """
        n = len(qubit_indices)
        for j in range(n // 2):
            self._emit_cudaq_controlled_swap(
                circuit,
                emitter,
                control_indices,
                [qubit_indices[j], qubit_indices[n - 1 - j]],
            )
        for j in range(n):
            for k in range(j):
                angle = -math.pi / (2 ** (j - k))
                self._emit_cudaq_controlled_cp(
                    circuit,
                    emitter,
                    control_indices,
                    [qubit_indices[j], qubit_indices[k]],
                    angle,
                )
            self._emit_cudaq_controlled_h(
                circuit, emitter, control_indices, qubit_indices[j]
            )

    def _emit_cudaq_controlled_composite(
        self,
        circuit: CudaqKernelArtifact,
        op: InvokeOperation,
        num_controls: int,
        control_indices: list[int],
        target_indices: list[int],
        qubit_map: CudaqControlledQubitMap,
        vector_slots: dict[str, list[int]],
        emitter: CudaqKernelEmitter,
        bindings: dict[str, Any],
    ) -> None:
        """Emit an InvokeOperation under outer controls.

        Args:
            circuit (CudaqKernelArtifact): Artifact being built.
            op (InvokeOperation): Composite or oracle invocation inside the
                controlled block.
            num_controls (int): Number of outer controls.
            control_indices (list[int]): Physical outer controls.
            target_indices (list[int]): Fallback target slots from the
                controlled-U call site.
            qubit_map (CudaqControlledQubitMap): UUID-to-physical-qubit map.
            vector_slots (dict[str, list[int]]): Vector slot map.
            emitter (CudaqKernelEmitter): CUDA-Q source emitter.
            bindings (dict[str, Any]): Current controlled-body bindings.

        Raises:
            EmitError: If the composite type cannot be decomposed by the
                CUDA-Q controlled fallback.
        """
        del num_controls, target_indices
        qubit_indices = _resolve_cudaq_value_indices(
            op.control_qubits + op.target_qubits,
            vector_slots,
            qubit_map,
            self,
            bindings,
            operation=f"InvokeOperation[{op.gate_type.name}]",
        )
        body = op.effective_body()
        if body is not None:
            local_bindings = self._resolver.bind_block_params(
                body,
                op.parameters,
                bindings,
            )
            local_qubit_map, local_vector_slots = _build_block_qubit_map(
                body,
                qubit_indices,
                self,
                local_bindings,
            )
            self._emit_cudaq_controlled_ops(
                circuit,
                body.operations,
                len(control_indices),
                control_indices,
                qubit_indices,
                local_qubit_map,
                local_vector_slots,
                emitter,
                local_bindings,
            )
        elif op.gate_type == CompositeGateType.QFT:
            self._emit_cudaq_controlled_qft(
                circuit, emitter, control_indices, qubit_indices
            )
        elif op.gate_type == CompositeGateType.IQFT:
            self._emit_cudaq_controlled_iqft(
                circuit, emitter, control_indices, qubit_indices
            )
        else:
            raise EmitError(
                f"Unsupported composite gate {op.gate_type.name} in "
                f"CUDA-Q controlled block body.",
                operation="ControlledUOperation",
            )
        for result, phys in zip(op.results, qubit_indices, strict=False):
            qubit_map[result.uuid] = phys

    def _emit_cudaq_controlled_inverse(
        self,
        circuit: CudaqKernelArtifact,
        op: InverseBlockOperation,
        num_controls: int,
        control_indices: list[int],
        target_indices: list[int],
        qubit_map: CudaqControlledQubitMap,
        vector_slots: dict[str, list[int]],
        emitter: CudaqKernelEmitter,
        bindings: dict[str, Any],
    ) -> None:
        """Emit an inverse block under outer CUDA-Q controls.

        Args:
            circuit (CudaqKernelArtifact): Artifact being built.
            op (InverseBlockOperation): Inverse op inside the controlled
                block.
            num_controls (int): Number of outer controls.
            control_indices (list[int]): Physical outer controls.
            target_indices (list[int]): Fallback target slots from the
                controlled-U call site.
            qubit_map (CudaqControlledQubitMap): UUID-to-physical-qubit map.
            vector_slots (dict[str, list[int]]): Vector slot map.
            emitter (CudaqKernelEmitter): CUDA-Q source emitter.
            bindings (dict[str, Any]): Current controlled-body bindings.

        Raises:
            EmitError: If the inverse fallback implementation is missing or
                cannot be emitted under controls.
        """
        del num_controls, target_indices
        impl = op.implementation_block
        if impl is None:
            raise EmitError(
                "CUDA-Q controlled fallback cannot emit an inverse block "
                "without an implementation block.",
                operation="InverseBlockOperation",
            )

        control_groups = [
            (
                _resolve_cudaq_array_slots(
                    operand,
                    vector_slots,
                    qubit_map,
                    self,
                    bindings,
                    operation="InverseBlockOperation",
                )
                if isinstance(operand, ArrayValue)
                else [
                    _resolve_cudaq_value_index(
                        operand,
                        vector_slots,
                        qubit_map,
                        self,
                        bindings,
                        operation="InverseBlockOperation",
                    )
                ]
            )
            for operand in op.control_qubits
        ]
        target_groups = [
            (
                _resolve_cudaq_array_slots(
                    operand,
                    vector_slots,
                    qubit_map,
                    self,
                    bindings,
                    operation="InverseBlockOperation",
                )
                if isinstance(operand, ArrayValue)
                else [
                    _resolve_cudaq_value_index(
                        operand,
                        vector_slots,
                        qubit_map,
                        self,
                        bindings,
                        operation="InverseBlockOperation",
                    )
                ]
            )
            for operand in op.target_qubits
        ]
        local_controls = [index for group in control_groups for index in group]
        inverse_targets = [index for group in target_groups for index in group]
        local_bindings = self._resolver.bind_block_params(
            impl,
            op.parameters,
            bindings,
        )
        _bind_quantum_input_shapes(
            self._resolver,
            impl,
            op.target_qubits,
            bindings,
            local_bindings,
        )
        local_qubit_map, local_vector_slots = _build_block_qubit_map(
            impl,
            inverse_targets,
            self,
            local_bindings,
        )
        effective_controls = control_indices + local_controls
        self._emit_cudaq_controlled_ops(
            circuit,
            impl.operations,
            len(effective_controls),
            effective_controls,
            inverse_targets,
            local_qubit_map,
            local_vector_slots,
            emitter,
            local_bindings,
        )
        for result, indices in zip(
            op.results[: op.num_control_qubits],
            control_groups,
            strict=False,
        ):
            if indices:
                qubit_map[result.uuid] = indices[0]

        target_results = [
            result
            for result in op.results[op.num_control_qubits :]
            if result.type.is_quantum()
        ]
        for result, indices in zip(target_results, target_groups, strict=False):
            if isinstance(result, ArrayValue):
                vector_slots[result.uuid] = list(indices)
                for i, phys in enumerate(indices):
                    qubit_map[QubitAddress(result.uuid, i)] = phys
                if indices:
                    qubit_map[result.uuid] = indices[0]
            elif indices:
                qubit_map[result.uuid] = indices[0]

    def _emit_cudaq_controlled_pauli_evolve(
        self,
        circuit: CudaqKernelArtifact,
        op: PauliEvolveOp,
        num_controls: int,
        control_indices: list[int],
        target_indices: list[int],
        qubit_map: CudaqControlledQubitMap,
        vector_slots: dict[str, list[int]],
        emitter: CudaqKernelEmitter,
        bindings: dict[str, Any],
    ) -> None:
        """Emit Pauli evolution under outer controls.

        Args:
            circuit (CudaqKernelArtifact): Artifact being built.
            op (PauliEvolveOp): Pauli evolution op inside the controlled
                block.
            num_controls (int): Number of outer controls.
            control_indices (list[int]): Physical outer controls.
            target_indices (list[int]): Fallback target slots from the
                controlled-U call site.
            qubit_map (CudaqControlledQubitMap): UUID-to-physical-qubit map.
            vector_slots (dict[str, list[int]]): Vector slot map.
            emitter (CudaqKernelEmitter): CUDA-Q source emitter.
            bindings (dict[str, Any]): Current controlled-body bindings.

        Raises:
            EmitError: If the Hamiltonian, gamma, or target vector cannot
                be resolved.
        """
        del num_controls, target_indices
        import qamomile.observable as qm_o

        hamiltonian = self._resolver.resolve_bound_value(op.observable, bindings)
        if not isinstance(hamiltonian, qm_o.Hamiltonian):
            raise EmitError(
                f"PauliEvolveOp requires a Hamiltonian binding. "
                f"Observable '{op.observable.name}' not found or not a Hamiltonian.",
                operation="PauliEvolveOp",
            )
        gamma = _resolve_gamma(self, op, bindings)
        if gamma is None:
            raise EmitError(
                "Cannot resolve gamma parameter for PauliEvolveOp. "
                "gamma must be a concrete float binding or a declared "
                "parameter (scalar or array element).",
                operation="PauliEvolveOp",
            )
        qubit_indices = _resolve_cudaq_array_slots(
            op.qubits,
            vector_slots,
            qubit_map,
            self,
            bindings,
            operation="PauliEvolveOp",
        )
        # A Hamiltonian smaller than the register is embedded by acting
        # only on its declared qubits (the leading ``num_qubits`` slots).
        validate_hamiltonian_within_register(hamiltonian.num_qubits, len(qubit_indices))
        for operators, coeff in hamiltonian:
            if abs(coeff.imag) > 1e-10:
                raise EmitError(
                    f"PauliEvolveOp requires a Hermitian Hamiltonian "
                    f"(real coefficients), but found complex coefficient "
                    f"{coeff} on term {operators}.",
                    operation="PauliEvolveOp",
                )
            if abs(coeff) < 1e-15 or len(operators) == 0:
                continue
            angle: Any
            if isinstance(gamma, (int, float)):
                angle = 2.0 * float(coeff.real * gamma)
            else:
                angle = (2.0 * float(coeff.real)) * gamma
            term_qubits = [qubit_indices[item.index] for item in operators]
            paulis = [item.pauli for item in operators]
            for qi, pauli in zip(term_qubits, paulis):
                if pauli == qm_o.Pauli.X:
                    self._emit_cudaq_controlled_h(circuit, emitter, control_indices, qi)
                elif pauli == qm_o.Pauli.Y:
                    self._emit_cudaq_multi_controlled_gate_type(
                        circuit,
                        GateOperationType.SDG,
                        emitter,
                        control_indices,
                        [qi],
                        None,
                    )
                    self._emit_cudaq_controlled_h(circuit, emitter, control_indices, qi)
            if len(term_qubits) == 1:
                emitter.emit_multi_controlled_rz(
                    circuit, control_indices, term_qubits[0], angle
                )
            else:
                for step in range(len(term_qubits) - 1):
                    self._emit_cudaq_multi_controlled_gate_type(
                        circuit,
                        GateOperationType.CX,
                        emitter,
                        control_indices,
                        [term_qubits[step], term_qubits[step + 1]],
                        None,
                    )
                emitter.emit_multi_controlled_rz(
                    circuit, control_indices, term_qubits[-1], angle
                )
                for step in range(len(term_qubits) - 2, -1, -1):
                    self._emit_cudaq_multi_controlled_gate_type(
                        circuit,
                        GateOperationType.CX,
                        emitter,
                        control_indices,
                        [term_qubits[step], term_qubits[step + 1]],
                        None,
                    )
            for qi, pauli in reversed(list(zip(term_qubits, paulis))):
                if pauli == qm_o.Pauli.X:
                    self._emit_cudaq_controlled_h(circuit, emitter, control_indices, qi)
                elif pauli == qm_o.Pauli.Y:
                    self._emit_cudaq_controlled_h(circuit, emitter, control_indices, qi)
                    self._emit_cudaq_multi_controlled_gate_type(
                        circuit,
                        GateOperationType.S,
                        emitter,
                        control_indices,
                        [qi],
                        None,
                    )
        vector_slots[op.evolved_qubits.uuid] = list(qubit_indices)
        if qubit_indices:
            qubit_map.setdefault(op.evolved_qubits.uuid, qubit_indices[0])

    def _emit_cudaq_multi_controlled_gate_type(
        self,
        circuit: CudaqKernelArtifact,
        gate_type: GateOperationType,
        emitter: CudaqKernelEmitter,
        control_indices: list[int],
        target_indices: list[int],
        angle: Any,
    ) -> None:
        """Emit one primitive gate type under zero or more controls.

        Args:
            circuit (CudaqKernelArtifact): Artifact being built.
            gate_type (GateOperationType): Primitive gate kind to emit.
            emitter (CudaqKernelEmitter): CUDA-Q source emitter.
            control_indices (list[int]): Physical control qubit indices.
            target_indices (list[int]): Physical target qubit indices.
            angle (Any): Rotation angle for parametric gates, or None
                for fixed gates.

        Raises:
            EmitError: When the gate type is unsupported for multi-control.
        """
        if not target_indices:
            return

        target_idx = target_indices[0]

        if gate_type == GateOperationType.H:
            self._emit_cudaq_controlled_h(circuit, emitter, control_indices, target_idx)
            return
        if gate_type == GateOperationType.X:
            if control_indices:
                emitter.emit_multi_controlled_x(circuit, control_indices, target_idx)
            else:
                emitter.emit_x(circuit, target_idx)
            return
        if gate_type == GateOperationType.Y:
            self._emit_cudaq_controlled_y(circuit, emitter, control_indices, target_idx)
            return
        if gate_type == GateOperationType.Z:
            self._emit_cudaq_controlled_z(circuit, emitter, control_indices, target_idx)
            return
        if gate_type == GateOperationType.S:
            emitter.emit_multi_controlled_p(
                circuit, control_indices, target_idx, math.pi / 2
            )
            return
        if gate_type == GateOperationType.SDG:
            emitter.emit_multi_controlled_p(
                circuit, control_indices, target_idx, -math.pi / 2
            )
            return
        if gate_type == GateOperationType.T:
            emitter.emit_multi_controlled_p(
                circuit, control_indices, target_idx, math.pi / 4
            )
            return
        if gate_type == GateOperationType.TDG:
            emitter.emit_multi_controlled_p(
                circuit, control_indices, target_idx, -math.pi / 4
            )
            return
        if gate_type == GateOperationType.P:
            emitter.emit_multi_controlled_p(circuit, control_indices, target_idx, angle)
            return
        if gate_type == GateOperationType.RX:
            emitter.emit_multi_controlled_rx(
                circuit, control_indices, target_idx, angle
            )
            return
        if gate_type == GateOperationType.RY:
            emitter.emit_multi_controlled_ry(
                circuit, control_indices, target_idx, angle
            )
            return
        if gate_type == GateOperationType.RZ:
            emitter.emit_multi_controlled_rz(
                circuit, control_indices, target_idx, angle
            )
            return
        if gate_type == GateOperationType.CX:
            if len(target_indices) < 2:
                raise EmitError(
                    "Controlled-CX requires at least 2 target qubits "
                    "(inner control + inner target).",
                    operation="ControlledGate",
                )
            inner_control = target_indices[0]
            inner_target = target_indices[1]
            emitter.emit_multi_controlled_x(
                circuit,
                control_indices + [inner_control],
                inner_target,
            )
            return
        if gate_type == GateOperationType.CZ:
            if len(target_indices) < 2:
                raise EmitError(
                    "Controlled-CZ requires at least 2 target qubits "
                    "(inner control + inner target).",
                    operation="ControlledGate",
                )
            self._emit_cudaq_controlled_z(
                circuit,
                emitter,
                control_indices + [target_indices[0]],
                target_indices[1],
            )
            return
        if gate_type == GateOperationType.SWAP:
            self._emit_cudaq_controlled_swap(
                circuit, emitter, control_indices, target_indices
            )
            return
        if gate_type == GateOperationType.CP:
            self._emit_cudaq_controlled_cp(
                circuit, emitter, control_indices, target_indices, angle
            )
            return
        if gate_type == GateOperationType.RZZ:
            self._emit_cudaq_controlled_rzz(
                circuit, emitter, control_indices, target_indices, angle
            )
            return
        if gate_type == GateOperationType.TOFFOLI:
            if len(target_indices) < 3:
                raise EmitError(
                    "Controlled-Toffoli requires at least 3 target qubits.",
                    operation="ControlledGate",
                )
            emitter.emit_multi_controlled_x(
                circuit,
                control_indices + [target_indices[0], target_indices[1]],
                target_indices[2],
            )
            return

        raise EmitError(
            f"Unsupported gate type {gate_type!r} in CUDA-Q multi-controlled "
            f"block decomposition.",
            operation="ControlledGate",
        )

    def _emit_cudaq_multi_controlled_gate(
        self,
        circuit: CudaqKernelArtifact,
        op: GateOperation,
        emitter: CudaqKernelEmitter,
        control_indices: list[int],
        target_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit a GateOperation under zero or more outer controls.

        Args:
            circuit (CudaqKernelArtifact): Artifact being built.
            op (GateOperation): Inner primitive gate.
            emitter (CudaqKernelEmitter): CUDA-Q source emitter.
            control_indices (list[int]): Physical outer controls.
            target_indices (list[int]): Resolved physical gate operands.
            bindings (dict[str, Any]): Current controlled-body bindings.

        Raises:
            EmitError: When the gate type is unsupported.
        """
        gate_type = op.gate_type
        if gate_type is None:
            raise EmitError(
                "CUDA-Q controlled helper cannot emit a gate without a gate type.",
                operation="ControlledUOperation",
            )
        angle = self._resolve_angle(op, bindings) if op.theta is not None else None
        self._emit_cudaq_multi_controlled_gate_type(
            circuit,
            gate_type,
            emitter,
            control_indices,
            target_indices,
            angle,
        )


class CudaqExecutor(QuantumExecutor[CudaqKernelArtifact]):
    """CUDA-Q quantum executor.

    Supports sampling via ``cudaq.sample`` / ``cudaq.run`` and expectation
    value estimation via ``cudaq.observe``.  Dispatches to the appropriate
    CUDA-Q runtime API based on the artifact's ``execution_mode``:

    - ``STATIC`` artifacts (no mid-circuit measurement or runtime control
      flow) support both sampling (``cudaq.sample``) and expectation-value
      estimation (``cudaq.observe``).
    - ``RUNNABLE`` artifacts (mid-circuit measurement or measurement-dependent
      control flow such as ``if bit:`` / ``while bit:``) support sampling only,
      via ``cudaq.run``.

    Expectation-value estimation is therefore **static-only**: calling
    :meth:`estimate` on a ``RUNNABLE`` artifact raises ``TypeError`` because
    ``cudaq.observe`` cannot consume a kernel that requires ``cudaq.run``.
    See :meth:`estimate` for the full rationale.

    Args:
        target (str | None): CUDA-Q target name (e.g., ``"qpp-cpu"``). If
            None, uses the default CUDA-Q target.
    """

    def __init__(self, target: str | None = None):
        import cudaq

        self._target = target
        if self._target:
            cudaq.set_target(self._target)  # type: ignore[operator]

    def _ensure_target(self) -> None:
        """Reapply this executor's target before a runtime call.

        CUDA-Q target selection is process-global.  If another executor (or
        any other code) has called ``cudaq.set_target`` since this instance
        was created, the global target may no longer match ``self._target``.
        Calling this method before every ``cudaq.sample`` / ``cudaq.observe``
        guarantees the correct backend is active.
        """
        if self._target:
            import cudaq

            cudaq.set_target(self._target)  # type: ignore[operator]

    def execute(self, circuit: Any, shots: int) -> dict[str, int]:
        """Execute circuit and return canonical big-endian bitstring counts.

        Dispatches based on ``execution_mode``:

        - ``STATIC``: uses ``cudaq.sample()`` on the decorator kernel.
        - ``RUNNABLE``: uses ``cudaq.run()`` on the runnable kernel.

        Both paths return bitstrings in big-endian format (highest qubit
        index = leftmost bit).
        """
        mode = getattr(circuit, "execution_mode", ExecutionMode.STATIC)
        if mode == ExecutionMode.RUNNABLE:
            return self._execute_runtime(circuit, shots)
        return self._execute_sample(circuit, shots)

    def _execute_sample(self, circuit: Any, shots: int) -> dict[str, int]:
        """Execute via ``cudaq.sample()`` for STATIC-mode kernels."""
        import cudaq

        self._ensure_target()

        if isinstance(circuit, BoundCudaqKernelArtifact):
            result = cudaq.sample(  # type: ignore[operator]
                circuit.kernel_func, circuit.param_values, shots_count=shots
            )
        else:
            result = cudaq.sample(circuit.kernel_func, shots_count=shots)  # type: ignore[operator]

        num_qubits = circuit.num_qubits
        counts: dict[str, int] = {}
        for bitstring in result:
            count = result.count(bitstring)
            if len(bitstring) > num_qubits:
                raise ValueError(
                    f"Bitstring '{bitstring}' has length {len(bitstring)} > "
                    f"num_qubits={num_qubits}"
                )
            padded = bitstring.zfill(num_qubits)
            canonical = padded[::-1]  # little-endian (allocation-order) -> big-endian
            counts[canonical] = counts.get(canonical, 0) + count

        return counts

    def _execute_runtime(
        self,
        circuit: CudaqKernelArtifact | BoundCudaqKernelArtifact,
        shots: int,
    ) -> dict[str, int]:
        """Execute via ``cudaq.run()`` for RUNNABLE-mode kernels.

        ``cudaq.run()`` returns a list of per-shot return values.  Each
        return value is ``list[bool]`` (from ``return [__b0, __b1, ...]``),
        with one bool per logical clbit.

        This method aggregates per-shot results into canonical big-endian
        bitstring counts matching the ``QuantumExecutor.execute()`` contract.
        """
        import cudaq

        self._ensure_target()

        if isinstance(circuit, BoundCudaqKernelArtifact):
            results = cudaq.run(  # type: ignore[operator]
                circuit.kernel_func, circuit.param_values, shots_count=shots
            )
        else:
            results = cudaq.run(circuit.kernel_func, shots_count=shots)  # type: ignore[operator]

        num_clbits = circuit.num_clbits
        counts: dict[str, int] = {}
        for shot_result in results:
            # shot_result is list[bool] in clbit order (from [__b0, __b1, ...])
            bits = ["1" if b else "0" for b in shot_result]
            # Pad to num_clbits if shorter
            while len(bits) < num_clbits:
                bits.append("0")
            # Reverse: clbit-order -> big-endian
            canonical = "".join(reversed(bits))
            counts[canonical] = counts.get(canonical, 0) + 1

        return counts

    def bind_parameters(  # type: ignore[override]
        self,
        circuit: Any,
        bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> BoundCudaqKernelArtifact:
        """Bind parameters to a circuit for execution.

        Returns a ``BoundCudaqKernelArtifact`` with the execution mode
        inherited from the source artifact.
        """
        param_values = []
        for param_info in parameter_metadata.parameters:
            if param_info.name in bindings:
                param_values.append(float(bindings[param_info.name]))
            else:
                raise ValueError(
                    f"Missing binding for parameter '{param_info.name}'. "
                    f"Provided bindings: {list(bindings.keys())}. "
                    f"Required parameters: "
                    f"{[p.name for p in parameter_metadata.parameters]}"
                )

        return BoundCudaqKernelArtifact(
            kernel_func=circuit.kernel_func,
            num_qubits=circuit.num_qubits,
            num_clbits=circuit.num_clbits,
            param_values=param_values,
            execution_mode=getattr(circuit, "execution_mode", ExecutionMode.STATIC),
        )

    def estimate(
        self,
        circuit: Any,
        hamiltonian: "qm_o.Hamiltonian",
        params: Sequence[float] | None = None,
    ) -> float:
        """Estimate an expectation value using ``cudaq.observe``.

        Expectation-value estimation on the CUDA-Q backend is **static-only**.
        It is supported exclusively for ``STATIC``-mode artifacts, which are
        evaluated with ``cudaq.observe()``.  ``RUNNABLE``-mode artifacts (a
        kernel containing mid-circuit measurement or measurement-dependent
        control flow such as ``if bit:`` / ``while bit:``) are emitted for
        ``cudaq.run()``, and ``cudaq.observe()`` cannot consume them.  Such
        artifacts therefore raise ``TypeError`` rather than silently returning
        a meaningless value.

        This is a CUDA-Q API limitation, not merely a Qamomile gap:
        ``cudaq.observe()`` evaluates the analytic expectation value of a
        deterministic state-preparation kernel and has no execution path for a
        kernel that requires the per-shot ``cudaq.run()`` runtime.  There is no
        safe alternative that preserves the exact-expectation contract of this
        method, so the ``TypeError`` path is intentional.  Under the current
        affine model a single Qamomile kernel also cannot be both ``RUNNABLE``
        and carry a terminal ``qmc.expval`` (reusing a measured qubit is
        rejected), so the ``ExecutableProgram.run`` expval path does not reach
        this guard; it is reached only when ``estimate`` is called directly on
        a ``RUNNABLE`` artifact.

        Args:
            circuit (Any): A CUDA-Q artifact (``CudaqKernelArtifact`` or
                ``BoundCudaqKernelArtifact``).  Must be ``STATIC``-mode;
                ``RUNNABLE``-mode artifacts are rejected.
            hamiltonian (qm_o.Hamiltonian): The observable to measure.  A
                ``qamomile.observable.Hamiltonian`` is converted to a CUDA-Q
                ``SpinOperator``; an already-converted operator is used as-is.
            params (Sequence[float] | None): Values for the kernel's runtime
                ``parameters`` slots, in declared parameter order.  Compile-
                time-bound parameters are already baked into the artifact and
                are not included here.  Defaults to None for a non-parametric
                or already-bound (``BoundCudaqKernelArtifact``) kernel.

        Returns:
            float: The estimated expectation value ``<psi|H|psi>``.

        Raises:
            TypeError: If ``circuit`` is a ``RUNNABLE``-mode artifact, since
                ``cudaq.observe()`` only accepts static state-preparation
                kernels (see the static-only note above).

        Example:
            >>> transpiler = CudaqTranspiler()
            >>> exe = transpiler.transpile(static_ansatz, bindings={"H": H})
            >>> executor = transpiler.executor()
            >>> value = executor.estimate(exe.get_first_circuit(), H)
        """
        import cudaq
        import qamomile.observable as qm_o
        from qamomile.cudaq.observable import hamiltonian_to_cudaq_spin_op

        mode = getattr(circuit, "execution_mode", ExecutionMode.STATIC)
        if mode == ExecutionMode.RUNNABLE:
            raise TypeError(
                "Expectation-value estimation is not available for this CUDA-Q "
                "circuit. The kernel uses mid-circuit measurement or "
                "measurement-dependent control flow (e.g. `if bit:` / "
                "`while bit:`), so it is emitted in RUNNABLE mode and executed "
                "with `cudaq.run()`. CUDA-Q's expectation-value primitive "
                "`cudaq.observe()` only accepts static state-preparation "
                "kernels and cannot consume a RUNNABLE kernel. To estimate an "
                "expectation value, express the state preparation as a static "
                "kernel without measurement-dependent control flow; "
                "expectation values cannot be computed for measurement-"
                "conditioned circuits on the CUDA-Q backend."
            )

        self._ensure_target()  # type: ignore[unreachable]

        if isinstance(hamiltonian, qm_o.Hamiltonian):  # type: ignore[unreachable]
            spin_op = hamiltonian_to_cudaq_spin_op(hamiltonian)
        else:
            spin_op = hamiltonian  # type: ignore[unreachable]

        if isinstance(circuit, BoundCudaqKernelArtifact):
            result: Any = cudaq.observe(
                circuit.kernel_func, spin_op, circuit.param_values
            )  # type: ignore[operator]
        else:
            if params is not None:
                result = cudaq.observe(circuit.kernel_func, spin_op, list(params))  # type: ignore[operator]
            else:
                result = cudaq.observe(circuit.kernel_func, spin_op)  # type: ignore[operator]

        return result.expectation()


class CudaqTranspiler(Transpiler[CudaqKernelArtifact]):
    """CUDA-Q transpiler for qamomile.circuit module.

    Converts Qamomile QKernels into CUDA-Q decorator-kernel artifacts.

    Example:
        from qamomile.cudaq import CudaqTranspiler
        import qamomile.circuit as qm

        @qm.qkernel
        def bell_state(q0: qm.Qubit, q1: qm.Qubit) -> tuple[qm.Bit, qm.Bit]:
            q0 = qm.h(q0)
            q0, q1 = qm.cx(q0, q1)
            return qm.measure(q0), qm.measure(q1)

        transpiler = CudaqTranspiler()
        executable = transpiler.transpile(bell_state)
    """

    def _create_segmentation_pass(self) -> SegmentationPass:
        return SegmentationPass()

    def _create_emit_pass(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> EmitPass[CudaqKernelArtifact]:
        return CudaqEmitPass(bindings, parameters)

    def executor(  # type: ignore[override]
        self,
        target: str | None = None,
    ) -> CudaqExecutor:
        """Create a CUDA-Q executor.

        Args:
            target: CUDA-Q target name (e.g., ``"qpp-cpu"``).
                If None, uses the default CUDA-Q target.
        """
        return CudaqExecutor(target)
