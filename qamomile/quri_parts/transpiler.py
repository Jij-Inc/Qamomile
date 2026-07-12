"""QURI Parts backend transpiler implementation.

This module provides QuriPartsTranspiler for converting Qamomile QKernels
into QURI Parts quantum circuits.
"""

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING, Any, Sequence, cast

from qamomile.circuit.ir.operation.arithmetic_operations import BinOp
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
    GateOperationType,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.executable import (
    ParameterMetadata,
    QuantumExecutor,
)
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.emit_support import (
    QubitAddress,
    QubitMap,
)
from qamomile.circuit.transpiler.passes.emit_support.cast_binop_emission import (
    evaluate_binop,
)
from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
    _bind_and_populate_block_inputs,
    _bind_quantum_input_shapes,
    _expand_quantum_operands_to_phys,
    _populate_input_qubit_map,
    _prepare_nested_block_for_emit,
    emit_controlled_gate,
    emit_controlled_pauli_evolve,
    emit_multi_controlled_gate,
    map_nested_controlled_u_results,
    replay_controlled_for,
    resolve_controlled_u_call,
    resolve_power,
    try_emit_batched_controlled_operations,
)
from qamomile.circuit.transpiler.passes.emit_support.gate_emission import (
    reject_duplicate_physical_indices,
)
from qamomile.circuit.transpiler.passes.emit_support.inverse_emission import (
    _map_inverse_block_results,
    _normalize_inverse_block_op,
)
from qamomile.circuit.transpiler.passes.separate import SegmentationPass
from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass
from qamomile.circuit.transpiler.transpiler import Transpiler

from .emitter import QuriPartsGateEmitter
from .exceptions import QamomileQuriPartsTranspileError

if TYPE_CHECKING:
    import qamomile.observable as qm_o
    import quri_parts.circuit as qp_c  # type: ignore[import-not-found]
    import quri_parts.core.operator as qp_o  # type: ignore[import-not-found]
    from quri_parts.circuit import (  # type: ignore[import-not-found]
        ImmutableBoundParametricQuantumCircuit,
    )


def _build_quri_controlled_qubit_map(
    emit_pass: StandardEmitPass[Any],
    block_value: Any,
    target_indices: list[int],
    bindings: dict[str, Any],
) -> QubitMap:
    """Build a block-local qubit map backed by actual QURI target slots.

    Args:
        emit_pass (StandardEmitPass[Any]): Emit pass used to resolve
            symbolic vector shapes.
        block_value (Any): Inner block whose quantum inputs are mapped.
        target_indices (list[int]): Physical QURI Parts qubit indices
            supplied as the controlled-U target operands.
        bindings (dict[str, Any]): Bindings used while resolving vector
            input shapes.

    Returns:
        QubitMap: Mapping from the inner block's formal quantum inputs to
            the parent circuit's physical QURI Parts qubit indices.

    Raises:
        EmitError: If the shared input-mapping helper cannot resolve or
            fit the block's quantum input footprint.
    """
    local_map: QubitMap = {}
    _populate_input_qubit_map(
        emit_pass,
        getattr(block_value, "input_values", []),
        len(target_indices),
        bindings,
        local_map,
    )
    return {address: target_indices[slot] for address, slot in local_map.items()}


def _resolve_quri_gate_targets(
    emit_pass: StandardEmitPass[Any],
    op: GateOperation,
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> list[int]:
    """Resolve a gate operation's quantum operands to physical qubits.

    Args:
        emit_pass (StandardEmitPass[Any]): Emit pass whose resolver maps
            IR values to QURI Parts physical qubit indices.
        op (GateOperation): Gate operation being emitted under controls.
        qubit_map (QubitMap): Current block-local qubit map.
        bindings (dict[str, Any]): Bindings used for array/slice
            resolution.

    Returns:
        list[int]: Physical target qubit indices in operand order.

    Raises:
        EmitError: If any gate operand cannot be resolved.
    """
    target_indices: list[int] = []
    for operand in op.qubit_operands:
        index = emit_pass._resolver.resolve_qubit_index(operand, qubit_map, bindings)
        if index is None:
            raise EmitError(
                f"QURI Parts controlled fallback cannot resolve gate "
                f"operand {operand.name!r} to a physical qubit.",
                operation="ControlledUOperation",
            )
        target_indices.append(index)
    return target_indices


def _propagate_quri_gate_results(
    op: GateOperation,
    target_indices: list[int],
    qubit_map: QubitMap,
) -> None:
    """Propagate gate result values to their unchanged physical slots.

    Args:
        op (GateOperation): Gate operation whose results were just
            emitted.
        target_indices (list[int]): Physical qubit indices resolved from
            ``op.qubit_operands``.
        qubit_map (QubitMap): Mutable block-local qubit map to update.
    """
    quantum_results = [result for result in op.results if result.type.is_quantum()]
    for result, index in zip(quantum_results, target_indices):
        qubit_map[QubitAddress(result.uuid)] = index


def _scale_quri_angle(angle: Any, factor: float) -> Any:
    """Scale a QURI Parts concrete or linear-form angle.

    Args:
        angle (Any): Angle resolved by the active QURI Parts emitter.
            Concrete numeric values, linear-combination dictionaries, and
            single QURI Parts parameter atoms are supported.
        factor (float): Scale factor to apply to every angle coefficient.

    Returns:
        Any: Scaled angle in the same representation.
    """
    if isinstance(angle, Real):
        return float(angle) * factor
    if isinstance(angle, dict):
        return {key: value * factor for key, value in angle.items()}
    return {angle: factor}


def _emit_quri_controlled_gate(
    emit_pass: StandardEmitPass[Any],
    circuit: Any,
    op: GateOperation,
    control_indices: list[int],
    target_indices: list[int],
    bindings: dict[str, Any],
) -> None:
    """Emit one QURI Parts gate under accumulated controls.

    Args:
        emit_pass (Any): QURI Parts emit pass.
        circuit (Any): QURI Parts circuit being built.
        op (GateOperation): Primitive gate operation to emit.
        control_indices (list[int]): Accumulated physical control qubits.
        target_indices (list[int]): Physical target qubits for ``op``.
        bindings (dict[str, Any]): Bindings used to resolve gate angles.

    Raises:
        EmitError: If the gate cannot be lowered under the accumulated
            controls (e.g. an unsupported gate kind, or an ancilla-pool
            shortfall in the shared Toffoli-cascade lowering).
    """
    if not control_indices:
        raise EmitError(
            "QURI Parts controlled fallback requires at least one control.",
            operation="ControlledUOperation",
        )

    # QURI Parts always walks controlled-U bodies gate-by-gate here rather than
    # through ``append_gate``, so this is the choke point for inner-gate
    # aliasing. A body ``cx(qs[i], qs[j])`` with i == j at runtime, or an inner
    # control that coincides with a target, is caught here.
    reject_duplicate_physical_indices(
        f"controlled {op.gate_type.name if op.gate_type else 'gate'} (QURI Parts)",
        control_indices + target_indices,
    )

    if len(control_indices) == 1:
        if op.gate_type == GateOperationType.CX:
            if len(target_indices) < 2:
                raise EmitError(
                    "Controlled-CX requires two target qubits.",
                    operation="ControlledUOperation",
                )
            emit_pass._emitter.emit_toffoli(
                circuit,
                control_indices[0],
                target_indices[0],
                target_indices[1],
            )
            return
        if op.gate_type == GateOperationType.CZ:
            if len(target_indices) < 2:
                raise EmitError(
                    "Controlled-CZ requires two target qubits.",
                    operation="ControlledUOperation",
                )
            emit_pass._emitter.emit_h(circuit, target_indices[1])
            emit_pass._emitter.emit_toffoli(
                circuit,
                control_indices[0],
                target_indices[0],
                target_indices[1],
            )
            emit_pass._emitter.emit_h(circuit, target_indices[1])
            return
        if op.gate_type == GateOperationType.CP:
            if len(target_indices) < 2:
                raise EmitError(
                    "Controlled-CP requires two target qubits.",
                    operation="ControlledUOperation",
                )
            angle = emit_pass._resolve_angle(op, bindings)
            half_angle = _scale_quri_angle(angle, 0.5)
            neg_half_angle = _scale_quri_angle(angle, -0.5)
            emit_pass._emitter.emit_cp(
                circuit,
                control_indices[0],
                target_indices[1],
                half_angle,
            )
            emit_pass._emitter.emit_cx(circuit, target_indices[0], control_indices[0])
            emit_pass._emitter.emit_cp(
                circuit,
                control_indices[0],
                target_indices[1],
                neg_half_angle,
            )
            emit_pass._emitter.emit_cx(circuit, target_indices[0], control_indices[0])
            emit_pass._emitter.emit_cp(
                circuit,
                target_indices[0],
                target_indices[1],
                half_angle,
            )
            return
        if op.gate_type == GateOperationType.RZZ:
            if len(target_indices) < 2:
                raise EmitError(
                    "Controlled-RZZ requires two target qubits.",
                    operation="ControlledUOperation",
                )
            angle = emit_pass._resolve_angle(op, bindings)
            emit_pass._emitter.emit_toffoli(
                circuit,
                control_indices[0],
                target_indices[0],
                target_indices[1],
            )
            emit_pass._emitter.emit_crz(
                circuit,
                control_indices[0],
                target_indices[1],
                angle,
            )
            emit_pass._emitter.emit_toffoli(
                circuit,
                control_indices[0],
                target_indices[0],
                target_indices[1],
            )
            return
        if op.gate_type == GateOperationType.TOFFOLI:
            # Absorb the Toffoli's own controls into the control set and
            # lower the resulting three-controlled X through the shared
            # multi-controlled path (Toffoli cascade on clean ancillas).
            emit_multi_controlled_gate(
                emit_pass, circuit, op, control_indices, target_indices, bindings
            )
            return
        emit_controlled_gate(
            emit_pass,
            circuit,
            op,
            control_indices[0],
            target_indices,
            bindings,
        )
        return

    if (
        len(control_indices) == 2
        and op.gate_type == GateOperationType.X
        and len(target_indices) == 1
    ):
        emit_pass._emitter.emit_toffoli(
            circuit,
            control_indices[0],
            control_indices[1],
            target_indices[0],
        )
        return

    emit_multi_controlled_gate(
        emit_pass, circuit, op, control_indices, target_indices, bindings
    )


def _map_quri_composite_results(
    op: InvokeOperation,
    control_indices: list[int],
    target_index_groups: list[list[int]],
    qubit_map: QubitMap,
) -> None:
    """Map composite result values back to their physical qubits.

    Args:
        op (InvokeOperation): Invocation that was emitted.
        control_indices (list[int]): Physical qubits for the composite's
            own control operands.
        target_index_groups (list[list[int]]): Physical qubits for each
            target operand.
        qubit_map (QubitMap): Mutable block-local qubit map.
    """
    for result, index in zip(op.results[: op.num_control_qubits], control_indices):
        qubit_map[QubitAddress(result.uuid)] = index

    target_results = op.results[op.num_control_qubits :]
    target_indices = [index for group in target_index_groups for index in group]
    for result, index in zip(target_results, target_indices):
        qubit_map[QubitAddress(result.uuid)] = index


def _emit_quri_inverse_operation(
    emit_pass: Any,
    circuit: Any,
    op: InverseBlockOperation,
    outer_control_indices: list[int],
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit an inverse operation under accumulated QURI Parts controls.

    Args:
        emit_pass (StandardEmitPass[Any]): QURI Parts emit pass.
        circuit (Any): QURI Parts circuit being built.
        op (InverseBlockOperation): Inverse operation from the current block.
        outer_control_indices (list[int]): Controls accumulated from
            enclosing controlled-U operations.
        qubit_map (QubitMap): Current block-local qubit map.
        bindings (dict[str, Any]): Bindings visible inside the current block.

    Raises:
        EmitError: If the inverse operation has no source/fallback block or
            cannot be emitted by native inverse or recursive fallback.
        SliceBorrowViolationError: If a nested block's slice usage fails
            the borrow check run by ``_normalize_inverse_block_op``.
    """
    # Strip nested slice markers (after the borrow check) before any path
    # below walks the blocks' operations directly; see
    # ``_normalize_inverse_block_op``.
    op = _normalize_inverse_block_op(op, bindings)
    if op.source_block is None or op.implementation_block is None:
        raise EmitError(
            "QURI Parts cannot emit an inverse block without both source and "
            "implementation blocks.",
            operation="InverseBlockOperation",
        )

    control_index_groups = [
        _expand_quantum_operands_to_phys(emit_pass, operand, qubit_map, bindings)
        for operand in op.control_qubits
    ]
    target_index_groups = [
        _expand_quantum_operands_to_phys(emit_pass, operand, qubit_map, bindings)
        for operand in op.target_qubits
    ]
    local_controls = [index for group in control_index_groups for index in group]
    target_indices = [index for group in target_index_groups for index in group]
    input_operands = [*op.target_qubits, *op.parameters]
    effective_controls = outer_control_indices + local_controls

    # An inverse block whose controls/targets alias at runtime is physically
    # ill-defined; all three sub-paths below (backend inverse, controlled walk,
    # inline) act on this combined set, so one check covers them.
    reject_duplicate_physical_indices(
        "inverse block (QURI Parts)", effective_controls + target_indices
    )

    if not effective_controls:
        # The native inverse path re-runs the segment allocator on the
        # source block; snapshot the segment-level analysis state so
        # later iteration replays keep consulting the segment's own
        # taint/allowlist sets.
        with emit_pass._allocator.preserving_analysis_state():
            emitted_native_inverse = emit_pass._try_emit_backend_inverse(
                circuit,
                op.source_block,
                input_operands,
                target_indices,
                bindings,
            )
        if emitted_native_inverse:
            _map_inverse_block_results(
                op, control_index_groups, target_index_groups, qubit_map
            )
            return

    if effective_controls:
        local_qubit_map, _local_clbit_map, local_bindings = (
            emit_pass._prepare_local_block_maps(
                op.implementation_block,
                input_operands,
                len(target_indices),
                bindings,
                parent_qubits=target_indices,
            )
        )
        _emit_quri_controlled_operations(
            emit_pass,
            circuit,
            op.implementation_block.operations,
            effective_controls,
            local_qubit_map,
            local_bindings,
        )
    else:
        emit_pass._emit_block_inline(
            circuit,
            op.implementation_block,
            input_operands,
            target_indices,
            bindings,
        )

    _map_inverse_block_results(op, control_index_groups, target_index_groups, qubit_map)


def _emit_quri_composite_operation(
    emit_pass: StandardEmitPass[Any],
    circuit: Any,
    op: InvokeOperation,
    outer_control_indices: list[int],
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit a boxed invocation under accumulated controls.

    Args:
        emit_pass (StandardEmitPass[Any]): QURI Parts emit pass.
        circuit (Any): QURI Parts circuit being built.
        op (InvokeOperation): Composite or oracle invocation from the block
            currently being walked.
        outer_control_indices (list[int]): Controls accumulated from
            enclosing controlled-U operations.
        qubit_map (QubitMap): Current block-local qubit map.
        bindings (dict[str, Any]): Bindings visible inside the current
            block.

    Raises:
        EmitError: If the invocation has no implementation block or cannot
            be routed through the guarded recursive fallback.
    """
    impl = op.effective_body()
    if impl is None:
        raise EmitError(
            "QURI Parts controlled fallback cannot emit an invocation "
            "without an implementation block.",
            operation="ControlledUOperation",
        )

    composite_controls = [
        index
        for operand in op.control_qubits
        for index in _expand_quantum_operands_to_phys(
            emit_pass, operand, qubit_map, bindings
        )
    ]
    target_index_groups = [
        _expand_quantum_operands_to_phys(emit_pass, operand, qubit_map, bindings)
        for operand in op.target_qubits
    ]
    target_indices = [index for group in target_index_groups for index in group]
    local_bindings = emit_pass._resolver.bind_block_params(
        impl, op.parameters, bindings
    )
    _bind_quantum_input_shapes(
        emit_pass._resolver, impl, op.target_qubits, bindings, local_bindings
    )
    inner_qubit_map = _build_quri_controlled_qubit_map(
        emit_pass, impl, target_indices, local_bindings
    )
    _emit_quri_controlled_operations(
        emit_pass,
        circuit,
        impl.operations,
        outer_control_indices + composite_controls,
        inner_qubit_map,
        local_bindings,
    )
    _map_quri_composite_results(op, composite_controls, target_index_groups, qubit_map)


def _emit_quri_nested_controlled_u(
    emit_pass: StandardEmitPass[Any],
    circuit: Any,
    op: ControlledUOperation,
    outer_control_indices: list[int],
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit a nested controlled-U by flattening controls at emit time.

    Resolves the nested operation through the shared
    ``resolve_controlled_u_call`` helper, which handles concrete and
    symbolic operand layouts alike — including the loop-dependent
    ``qmc.control(qmc.x, num_controls=k)`` shape that stays a
    ``SymbolicControlledU`` until each unrolled iteration binds ``k``.

    Args:
        emit_pass (StandardEmitPass[Any]): QURI Parts emit pass.
        circuit (Any): QURI Parts circuit being built.
        op (ControlledUOperation): Nested controlled-U operation from the
            block currently being walked.
        outer_control_indices (list[int]): Controls accumulated from
            enclosing controlled-U operations.
        qubit_map (QubitMap): Current block-local qubit map.
        bindings (dict[str, Any]): Bindings visible inside the current
            block.

    Raises:
        EmitError: If the nested op lacks a block, its controls or
            targets cannot be resolved, or the inner body cannot be
            lowered by the guarded recursive fallback.
    """
    resolved = resolve_controlled_u_call(emit_pass, op, qubit_map, bindings)
    block = _prepare_nested_block_for_emit(resolved.block, resolved.local_bindings)
    inner_qubit_map = _build_quri_controlled_qubit_map(
        emit_pass, block, resolved.target_phys, resolved.local_bindings
    )
    effective_controls = outer_control_indices + resolved.control_phys
    power = resolve_power(emit_pass, op, bindings)
    for _ in range(power):
        _emit_quri_controlled_operations(
            emit_pass,
            circuit,
            block.operations,
            effective_controls,
            inner_qubit_map,
            resolved.local_bindings,
        )
    map_nested_controlled_u_results(op, resolved, qubit_map)


def _emit_quri_controlled_operations(
    emit_pass: StandardEmitPass[Any],
    circuit: Any,
    operations: Sequence[Any],
    control_indices: list[int],
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Recursively emit QURI Parts operations under accumulated controls.

    Args:
        emit_pass (StandardEmitPass[Any]): QURI Parts emit pass.
        circuit (Any): QURI Parts circuit being built.
        operations (Sequence[Any]): Block operations to walk.
        control_indices (list[int]): Accumulated physical controls.
        qubit_map (QubitMap): Mutable block-local qubit map.
        bindings (dict[str, Any]): Bindings visible in this block.

    Raises:
        EmitError: If an operation cannot be emitted by the guarded QURI
            Parts recursive fallback.
    """
    if try_emit_batched_controlled_operations(
        emit_pass,
        circuit,
        list(operations),
        control_indices,
        qubit_map,
        bindings,
        walker=_emit_quri_controlled_operations,
    ):
        return
    for op in operations:
        if isinstance(op, ReturnOperation):
            continue
        if isinstance(op, BinOp):
            evaluate_binop(emit_pass, op, bindings)
            continue
        if isinstance(op, GateOperation):
            target_indices = _resolve_quri_gate_targets(
                emit_pass, op, qubit_map, bindings
            )
            _emit_quri_controlled_gate(
                emit_pass, circuit, op, control_indices, target_indices, bindings
            )
            _propagate_quri_gate_results(op, target_indices, qubit_map)
            continue
        if isinstance(op, ControlledUOperation):
            _emit_quri_nested_controlled_u(
                emit_pass, circuit, op, control_indices, qubit_map, bindings
            )
            continue
        if isinstance(op, InvokeOperation):
            _emit_quri_composite_operation(
                emit_pass,
                circuit,
                op,
                control_indices,
                qubit_map,
                bindings,
            )
            continue
        if isinstance(op, InverseBlockOperation):
            _emit_quri_inverse_operation(
                emit_pass, circuit, op, control_indices, qubit_map, bindings
            )
            continue
        if isinstance(op, ForOperation):
            replay_controlled_for(
                emit_pass,
                circuit,
                op,
                control_indices,
                qubit_map,
                bindings,
                walker=_emit_quri_controlled_operations,
            )
            continue
        if isinstance(op, PauliEvolveOp):
            # Reuse the shared lowering: it emits the basis-change and CX
            # ladder uncontrolled and routes only the central RZ through
            # the multi-control machinery, which dispatches to ``emit_crz``
            # for one control and to the shared Toffoli-cascade
            # ``_emit_irreducible_multi_controlled_gate`` for two or more.
            emit_controlled_pauli_evolve(
                emit_pass, circuit, op, control_indices, qubit_map, bindings
            )
            continue
        raise EmitError(
            "QURI Parts recursive controlled fallback only supports "
            "primitive gates, nested ControlledUOperation values, "
            "InvokeOperation values, "
            "InverseBlockOperation values, "
            "PauliEvolveOp values, "
            "ReturnOperation, and statically resolved ForOperation bodies. "
            f"Unsupported operation: {type(op).__name__}.",
            operation="ControlledUOperation",
        )


def _create_seeded_qulacs_vector_sampler(seed: int) -> Any:
    """Create a qulacs vector sampler that seeds its measurement RNG.

    The high-level ``create_qulacs_vector_sampler`` exposed by QURI Parts
    does not thread a random seed down to qulacs, so this helper reproduces
    its qulacs state-vector sampling path while forwarding ``seed`` to
    ``qulacs.QuantumState.sampling``. Sampling the same circuit with the
    same seed therefore yields identical measurement counts.

    Unlike the default QURI Parts sampler, this path does not switch to the
    multinomial state-vector fast-path at very large shot counts (that
    branch is unseedable upstream); it always uses ``QuantumState.sampling``.
    The resulting distribution is statistically identical, only potentially
    slower for very large shot counts.

    Args:
        seed (int): Random seed forwarded to ``QuantumState.sampling`` on
            every call, making sampling deterministic.

    Returns:
        Any: A sampler callable taking ``(circuit, shots)`` and returning a
            ``collections.Counter`` mapping basis-state integers to counts.

    Raises:
        ImportError: If quri-parts-qulacs (or qulacs) is not installed.
    """
    from collections import Counter

    import qulacs  # type: ignore[import-not-found]

    from quri_parts.qulacs.circuit import (  # type: ignore[import-not-found]
        convert_circuit,
    )

    def sampler(circuit: Any, shots: int) -> Any:
        """Sample ``circuit`` for ``shots`` shots using the fixed seed.

        Args:
            circuit (Any): The QURI Parts circuit to sample.
            shots (int): Number of measurement shots.

        Returns:
            Any: A ``collections.Counter`` mapping basis-state integers to
                their observed counts.

        Raises:
            Exception: Propagates any qulacs / QURI Parts circuit-conversion
                or sampling error raised for a malformed circuit.
        """
        state = qulacs.QuantumState(circuit.qubit_count)
        convert_circuit(circuit).update_quantum_state(state)
        return Counter(state.sampling(shots, seed))

    return sampler


class QuriPartsEmitPass(
    StandardEmitPass["qp_c.LinearMappedUnboundParametricQuantumCircuit"]
):
    """QURI Parts-specific emission pass.

    Uses StandardEmitPass with QuriPartsGateEmitter for gate emission.
    QURI Parts does not support native control flow, so all loops are unrolled.
    """

    def __init__(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ):
        """Initialize the QURI Parts emit pass.

        Args:
            bindings: Parameter bindings for the circuit
            parameters: List of parameter names to preserve as backend parameters
        """
        emitter = QuriPartsGateEmitter()
        # QURI Parts has no native composite gate emitters
        composite_emitters: list[Any] = []
        super().__init__(
            cast(Any, emitter),
            bindings,
            parameters,
            composite_emitters,
            backend_name="quri_parts",
        )

    def _blockvalue_to_gate(
        self,
        block_value: Any,
        num_qubits: int,
        bindings: dict[str, Any],
        input_operands: list[Any] | None = None,
        operation_name: str = "ControlledUOperation",
    ) -> None:
        """Return no native gate object for QURI Parts controlled blocks.

        QURI Parts' emitter cannot turn a temporary circuit into a gate
        object that can be appended and controlled later.  The shared
        probe would still build a sub-circuit before discovering that
        ``circuit_to_gate()`` returns ``None``, which can pollute the
        parent emitter's current-circuit and parameter state.  Skip that
        probe and let ``_emit_controlled_fallback`` try the shared
        gate-by-gate decomposition directly.

        Args:
            block_value (Any): Ignored inner block value.
            num_qubits (int): Ignored target qubit count.
            bindings (dict[str, Any]): Ignored local bindings.
            input_operands (list[Any] | None): Ignored call-site
                operands. Defaults to None.
            operation_name (str): Ignored diagnostic operation name.
                Defaults to ``"ControlledUOperation"``.

        Returns:
            None: Always signals that the backend-specific fallback must
            handle the controlled block.
        """
        del block_value, num_qubits, bindings, input_operands
        return None

    def _emit_controlled_fallback(
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        block_value: Any,
        num_controls: int,
        control_indices: list[int],
        target_indices: list[int],
        power: int,
        bindings: dict[str, Any],
    ) -> None:
        """Emit controlled-U fallback through QURI Parts-specific paths.

        QURI Parts does not expose a custom-gate object that can be
        returned by ``circuit_to_gate`` and then controlled.  Instead,
        QURI Parts walks the block body, resolves each inner gate's
        call-site qubit operands, and emits the primitive controlled gates
        directly.

        Args:
            circuit (qp_c.LinearMappedUnboundParametricQuantumCircuit):
                Parent QURI Parts circuit being emitted.
            block_value (Any): Inner controlled-U block.
            num_controls (int): Number of physical control qubits.
            control_indices (list[int]): Physical control qubits.
            target_indices (list[int]): Physical target qubits.
            power (int): Positive power to apply to the inner unitary
                before controlling it.
            bindings (dict[str, Any]): Local block bindings.

        Raises:
            EmitError: If safe gate-by-gate controlled decomposition
                cannot emit the block.
        """
        if not target_indices:
            return
        if not hasattr(block_value, "operations"):
            raise EmitError(
                "Cannot emit QURI Parts controlled fallback: block has no operations.",
                operation="ControlledUOperation",
            )
        if num_controls != len(control_indices):
            raise EmitError(
                "QURI Parts controlled fallback received inconsistent "
                f"control metadata: num_controls={num_controls}, "
                f"control_indices={control_indices!r}.",
                operation="ControlledUOperation",
            )
        # QURI Parts walks the block gate-by-gate rather than routing through
        # ``append_gate``, so run the shared aliasing check here. A control that
        # coincides with a target at runtime is physically ill-defined.
        reject_duplicate_physical_indices(
            "controlled gate (QURI Parts fallback)",
            control_indices + target_indices,
        )
        qubit_map = _build_quri_controlled_qubit_map(
            self, block_value, target_indices, bindings
        )
        for _ in range(power):
            _emit_quri_controlled_operations(
                self,
                circuit,
                block_value.operations,
                control_indices,
                qubit_map,
                bindings,
            )

    def _reserves_multi_control_ancillas(self) -> bool:
        """Opt in to the shared clean-ancilla multi-controlled lowering.

        QURI Parts has no native multi-controlled gate object, so
        irreducible multi-controlled gates are lowered through the
        shared Toffoli-cascade decomposition on clean ancilla qubits
        reserved per quantum segment. Unlike the dense
        ``UnitaryMatrix`` lowering it replaces, the cascade scales
        linearly in the control count and keeps rotation angles
        symbolic, so runtime-parametric multi-controlled rotations are
        supported.

        Returns:
            bool: Always True.
        """
        return True

    def _emit_custom_composite(
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        op: Any,
        impl: Any,
        qubit_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit a custom composite operation into a QURI Parts circuit.

        QURI Parts has no reusable gate object with call-site qubit
        remapping, so custom composites are emitted inline.

        Args:
            circuit (qp_c.LinearMappedUnboundParametricQuantumCircuit):
                Parent QURI Parts circuit.
            op (Any): Composite operation being emitted.
            impl (Any): Qamomile fallback implementation block.
            qubit_indices (list[int]): Parent-circuit qubit indices for
                the composite operation.
            bindings (dict[str, Any]): Active compile-time and runtime
                parameter bindings.
        """
        impl = _prepare_nested_block_for_emit(impl, bindings)
        self._emit_block_inline(
            circuit,
            impl,
            getattr(op, "operands", None),
            qubit_indices,
            bindings,
        )

    def _emit_inverse_block(
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        op: InverseBlockOperation,
        qubit_map: QubitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Emit a first-class inverse block into a QURI Parts circuit.

        Args:
            circuit (qp_c.LinearMappedUnboundParametricQuantumCircuit):
                Parent QURI Parts circuit.
            op (InverseBlockOperation): Inverse block operation to emit.
            qubit_map (QubitMap): Current quantum value to physical qubit map.
            bindings (dict[str, Any]): Active emit bindings.

        Raises:
            EmitError: If native inversion and fallback emission both fail.
        """
        _emit_quri_inverse_operation(self, circuit, op, [], qubit_map, bindings)

    def _try_emit_backend_inverse(
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        block_value: Any,
        input_operands: list[Any] | None,
        qubit_indices: list[int],
        bindings: dict[str, Any],
    ) -> bool:
        """Try emitting ``block_value`` via QURI Parts' circuit inverse.

        Args:
            circuit (qp_c.LinearMappedUnboundParametricQuantumCircuit):
                Parent QURI Parts circuit to append into on success.
            block_value (Any): Source block whose inverse should be emitted.
            input_operands (list[Any] | None): Call-site operands used to
                bind the source block inputs.
            qubit_indices (list[int]): Parent-circuit qubits occupied by
                the source block.
            bindings (dict[str, Any]): Active emit bindings.

        Returns:
            bool: True when backend-native inversion was emitted, False
                when the caller should fall back to the Qamomile inverse
                implementation block.
        """
        if not hasattr(block_value, "operations"):
            return False

        try:
            local_qubit_map, local_clbit_map, local_bindings = (
                self._prepare_local_block_maps(
                    block_value,
                    input_operands,
                    len(qubit_indices),
                    bindings,
                )
            )
            local_qubit_map, local_clbit_map = self._allocator.allocate(
                block_value.operations,
                local_bindings,
                initial_qubit_map=local_qubit_map,
                initial_clbit_map=local_clbit_map,
            )
        except (AttributeError, TypeError, ValueError, RuntimeError, EmitError):
            # ``allocate`` raises EmitError on an unresolvable Vector[Qubit]
            # size or a rank>1 qubit array in the source block; as before the
            # reorg, that must decline the native inverse and fall back to
            # the Qamomile inverse decomposition, not abort the transpile.
            return False

        qubit_count = (
            max(local_qubit_map.values()) + 1 if local_qubit_map else len(qubit_indices)
        )
        if qubit_count > len(qubit_indices):
            return False

        if self._counting_emission:
            # A count-only dry run swaps in a no-op emitter that cannot build
            # or invert a real circuit. The native inverse emits the block
            # into a sub-circuit with the segment pool suspended, so it never
            # consumes parent-pool ancillas; reproduce exactly that — emit the
            # block under suspension so both operand resolution and the
            # zero-pool contribution match real emission (a pool-needing
            # multi-controlled gate still raises EmitError and falls back to
            # the parent-pool path) — then skip the real inversion and append
            # that a count does not need.
            try:
                sub_circuit = self._emitter.create_circuit(qubit_count, 0)
                with self._suspended_mc_ancilla_pool():
                    self._emit_operations(
                        sub_circuit,
                        block_value.operations,
                        local_qubit_map,
                        local_clbit_map,
                        local_bindings,
                        force_unroll=True,
                    )
            except (AttributeError, TypeError, ValueError, RuntimeError, EmitError):
                return False
            return True

        emitter = cast(QuriPartsGateEmitter, self._emitter)
        saved_circuit = emitter._current_circuit
        saved_param_map = dict(emitter._param_map)
        try:
            sub_circuit = self._emitter.create_circuit(qubit_count, 0)
            # The segment ancilla pool addresses the parent circuit;
            # suspend it so a multi-controlled gate inside this inverse
            # block cannot index it against the narrower sub-circuit. If
            # one is present, the shared cascade raises EmitError (caught
            # below), and the caller falls back to gate-by-gate inverse
            # emission on the parent circuit.
            with self._suspended_mc_ancilla_pool():
                self._emit_operations(
                    sub_circuit,
                    block_value.operations,
                    local_qubit_map,
                    local_clbit_map,
                    local_bindings,
                    force_unroll=True,
                )
            inverse_circuit = self._emitter.gate_inverse(sub_circuit)
        except (AttributeError, TypeError, ValueError, RuntimeError, EmitError):
            return False
        finally:
            emitter._current_circuit = saved_circuit
            emitter._param_map = saved_param_map

        if inverse_circuit is None:
            return False

        try:
            self._append_remapped_circuit(circuit, inverse_circuit, qubit_indices)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False
        return True

    def _emit_block_inline(
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        block_value: Any,
        input_operands: list[Any] | None,
        qubit_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit a nested implementation block directly into ``circuit``.

        Args:
            circuit (qp_c.LinearMappedUnboundParametricQuantumCircuit):
                Parent QURI Parts circuit.
            block_value (Any): Implementation block to emit.
            input_operands (list[Any] | None): Call-site operands used to
                bind the block inputs.
            qubit_indices (list[int]): Parent-circuit qubits occupied by
                the nested block.
            bindings (dict[str, Any]): Active emit bindings.
        """
        if not hasattr(block_value, "operations"):
            return

        local_qubit_map, local_clbit_map, local_bindings = (
            self._prepare_local_block_maps(
                block_value,
                input_operands,
                len(qubit_indices),
                bindings,
                parent_qubits=qubit_indices,
            )
        )
        self._emit_operations(
            circuit,
            block_value.operations,
            local_qubit_map,
            local_clbit_map,
            local_bindings,
            force_unroll=True,
        )

    def _prepare_local_block_maps(
        self,
        block_value: Any,
        input_operands: list[Any] | None,
        num_qubits: int,
        bindings: dict[str, Any],
        parent_qubits: list[int] | None = None,
    ) -> tuple[dict[Any, int], dict[Any, int], dict[str, Any]]:
        """Prepare local value maps for nested QURI Parts block emission.

        Args:
            block_value (Any): Nested block whose inputs should be mapped.
            input_operands (list[Any] | None): Call-site operands used to
                bind quantum and classical block inputs.
            num_qubits (int): Local qubit width of the nested operation.
            bindings (dict[str, Any]): Active emit bindings.
            parent_qubits (list[int] | None): Optional parent-circuit
                qubit indices used to remap local addresses. Defaults to
                None, leaving addresses in local ``0..num_qubits-1`` form.

        Returns:
            tuple[dict[Any, int], dict[Any, int], dict[str, Any]]: Local
                qubit map, local classical-bit map, and nested bindings.
        """
        local_qubit_map: dict[Any, int] = {}
        local_clbit_map: dict[Any, int] = {}
        local_bindings = _bind_and_populate_block_inputs(
            self,
            block_value,
            input_operands,
            num_qubits,
            bindings,
            local_qubit_map,
            parent_qubits=parent_qubits,
            operation_name="InverseBlockOperation",
        )

        return local_qubit_map, local_clbit_map, local_bindings

    def _append_remapped_circuit(
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        source_circuit: Any,
        qubit_indices: list[int],
    ) -> None:
        """Append ``source_circuit`` gates after remapping their qubits.

        Args:
            circuit (qp_c.LinearMappedUnboundParametricQuantumCircuit):
                Parent QURI Parts circuit.
            source_circuit (Any): Non-parametric QURI Parts circuit whose
                gates should be appended.
            qubit_indices (list[int]): Parent-circuit qubits corresponding
                to local qubit indices in ``source_circuit``.

        Raises:
            AttributeError: If ``source_circuit`` does not expose gates.
            IndexError: If a gate references a local qubit outside
                ``qubit_indices``.
            TypeError: If QURI Parts rejects a rebuilt gate.
            ValueError: If QURI Parts rejects a rebuilt gate.
        """
        remapped_gates = [
            self._remap_gate(gate, qubit_indices) for gate in source_circuit.gates
        ]
        for gate in remapped_gates:
            circuit.add_gate(gate)

    def _remap_gate(self, gate: Any, qubit_indices: list[int]) -> Any:
        """Rebuild a QURI Parts gate with parent-circuit qubit indices.

        Args:
            gate (Any): QURI Parts ``QuantumGate`` from a local circuit.
            qubit_indices (list[int]): Parent-circuit qubits corresponding
                to local qubit indices.

        Returns:
            Any: Rebuilt QURI Parts ``QuantumGate`` with remapped target and
                control indices.

        Raises:
            IndexError: If ``gate`` references a local qubit outside
                ``qubit_indices``.
            TypeError: If QURI Parts rejects the rebuilt gate.
            ValueError: If QURI Parts rejects the rebuilt gate.
        """
        from quri_parts.circuit import QuantumGate  # type: ignore[import-not-found]

        return QuantumGate(
            name=gate.name,
            target_indices=tuple(qubit_indices[index] for index in gate.target_indices),
            control_indices=tuple(
                qubit_indices[index] for index in gate.control_indices
            ),
            classical_indices=gate.classical_indices,
            params=gate.params,
            pauli_ids=gate.pauli_ids,
            unitary_matrix=gate.unitary_matrix,
        )


class QuriPartsExecutor(
    QuantumExecutor["qp_c.LinearMappedUnboundParametricQuantumCircuit"]
):
    """QURI Parts quantum executor.

    Supports both sampling and expectation value estimation.
    Uses Qulacs backend by default for efficient simulation.

    Example:
        executor = QuriPartsExecutor()  # Uses Qulacs by default
        counts = executor.execute(circuit, shots=1000)
        # counts: {"00": 512, "11": 512}
    """

    def __init__(
        self,
        sampler: Any = None,
        estimator: Any = None,
        seed: int | None = None,
    ):
        """Initialize executor with optional sampler, estimator, and seed.

        Args:
            sampler (Any): QURI Parts sampler. Defaults to None, meaning a
                qulacs vector sampler is created lazily on first use.
            estimator (Any): QURI Parts parametric estimator. Defaults to
                None, meaning a qulacs parametric estimator is created
                lazily on first use.
            seed (int | None): Optional random seed forwarded to the qulacs
                vector sampler so that sampling is reproducible. When set,
                two ``execute`` calls with the same seed and circuit return
                identical shot counts, which unblocks reproducible
                tutorials and benchmarks. Defaults to None, meaning
                sampling is non-deterministic. The seed is only applied to
                the default qulacs sampler; it is ignored when a custom
                ``sampler`` is supplied, since an arbitrary sampler has no
                standard seed interface.
        """
        self._sampler = sampler
        self._estimator = estimator
        self._non_parametric_estimator: Any = None
        self._seed = seed

    @property
    def sampler(self) -> Any:
        """Lazy initialization of sampler.

        When a ``seed`` was supplied to the constructor (and no custom
        sampler was given), a seeded qulacs vector sampler is created so
        that sampling is reproducible.

        Returns:
            Any: A QURI Parts sampler callable taking ``(circuit, shots)``
                and returning measurement counts.

        Raises:
            ImportError: If quri-parts-qulacs is not installed.
        """
        if self._sampler is None:
            try:
                if self._seed is None:
                    from quri_parts.qulacs.sampler import (  # type: ignore[import-not-found]
                        create_qulacs_vector_sampler,
                    )

                    self._sampler = create_qulacs_vector_sampler()
                else:
                    self._sampler = _create_seeded_qulacs_vector_sampler(self._seed)
            except ImportError as e:
                raise ImportError(
                    "quri-parts-qulacs is required for QuriPartsExecutor. "
                    "Install with: pip install quri-parts-qulacs"
                ) from e
        return self._sampler

    @property
    def parametric_estimator(self) -> Any:
        """Lazy initialization of parametric estimator for optimization."""
        if self._estimator is None:
            try:
                from quri_parts.qulacs.estimator import (  # type: ignore[import-not-found]
                    create_qulacs_vector_parametric_estimator,
                )

                self._estimator = create_qulacs_vector_parametric_estimator()
            except ImportError as e:
                raise ImportError(
                    "quri-parts-qulacs is required for QuriPartsExecutor. "
                    "Install with: pip install quri-parts-qulacs"
                ) from e
        return self._estimator

    @property
    def non_parametric_estimator(self) -> Any:
        """Lazy initialization of non-parametric estimator.

        Used when the circuit has already been bound (parameters resolved)
        or is a non-parametric circuit. Unlike the parametric estimator,
        this takes (operator, state) without parameter values.
        """
        if self._non_parametric_estimator is None:
            try:
                from quri_parts.qulacs.estimator import (  # type: ignore[import-not-found]
                    create_qulacs_vector_estimator,
                )

                self._non_parametric_estimator = create_qulacs_vector_estimator()
            except ImportError as e:
                raise ImportError(
                    "quri-parts-qulacs is required for QuriPartsExecutor. "
                    "Install with: pip install quri-parts-qulacs"
                ) from e
        return self._non_parametric_estimator

    def execute(self, circuit: Any, shots: int) -> dict[str, int]:
        """Execute circuit and return bitstring counts.

        Args:
            circuit (Any): Bound or unbound QURI Parts circuit to execute.
            shots (int): Number of measurement shots.

        Returns:
            dict[str, int]: Mapping from bitstrings to counts (for example,
                ``{"00": 512, "11": 512}``). A zero-qubit circuit returns
                ``{"": shots}`` without invoking the sampler.

        Raises:
            ImportError: If the default Qulacs sampler dependency is absent.
        """
        num_qubits = circuit.qubit_count
        if num_qubits == 0:
            return {"": shots}

        counter = self.sampler(circuit, shots)
        return {format(k, f"0{num_qubits}b"): v for k, v in counter.items()}

    def bind_parameters(  # type: ignore[override]
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> "ImmutableBoundParametricQuantumCircuit":
        """Bind parameter values to the QURI Parts circuit.

        QURI Parts requires parameter values as a sequence in the order
        parameters were added to the circuit.

        Args:
            circuit: The unbound parametric circuit
            bindings: Dictionary of parameter name to value
            parameter_metadata: Metadata about the parameters

        Returns:
            Bound parametric circuit

        Raises:
            QamomileQuriPartsTranspileError: If a required parameter binding
                is missing from ``bindings``.
        """
        param_values = []
        for param_info in parameter_metadata.parameters:
            if param_info.name in bindings:
                param_values.append(float(bindings[param_info.name]))
            else:
                raise QamomileQuriPartsTranspileError(
                    f"Missing binding for parameter '{param_info.name}'. "
                    f"Provided bindings: {list(bindings.keys())}. "
                    f"Required parameters: "
                    f"{[p.name for p in parameter_metadata.parameters]}"
                )

        return circuit.bind_parameters(param_values)

    def estimate(
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        hamiltonian: "qm_o.Hamiltonian",
        params: Sequence[float] | None = None,
    ) -> float:
        """Estimate expectation value <psi|H|psi>.

        Also accepts a native ``quri_parts.core.operator.Operator`` at
        runtime for convenience (auto-conversion is skipped in that case).

        Args:
            circuit: The unbound parametric circuit (state preparation ansatz)
            hamiltonian: Hamiltonian to measure (qamomile Hamiltonian or
                native QURI Parts Operator at runtime)
            params: Parameter values for the parametric circuit

        Returns:
            Real part of the expectation value
        """
        import qamomile.observable as qm_o

        if isinstance(hamiltonian, qm_o.Hamiltonian):
            from qamomile.quri_parts.observable import hamiltonian_to_quri_operator

            hamiltonian = hamiltonian_to_quri_operator(hamiltonian)  # type: ignore[assignment]

        return self.estimate_expectation(circuit, hamiltonian, params or [])  # type: ignore[arg-type]

    def estimate_expectation(
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        hamiltonian: "qp_o.Operator",
        param_values: Sequence[float],
    ) -> float:
        """Estimate expectation value of hamiltonian for a circuit.

        Handles both unbound parametric circuits (used during optimization)
        and already-bound or non-parametric circuits. When the circuit has
        already been bound (e.g., by ``_run_expval`` in ``ExecutableProgram``),
        ``apply_circuit`` produces a ``GeneralCircuitQuantumState`` instead
        of a ``ParametricCircuitQuantumState``, so we dispatch to the
        non-parametric estimator automatically.

        Args:
            circuit: The quantum circuit (unbound parametric or bound/concrete)
            hamiltonian: QURI Parts Operator representing the Hamiltonian
            param_values: Sequence of parameter values in order (ignored for
                bound/non-parametric circuits)

        Returns:
            Real part of the expectation value
        """
        from quri_parts.core.state import (  # type: ignore[import-not-found]
            apply_circuit,
            quantum_state,
        )

        cb_state = quantum_state(circuit.qubit_count, bits=0)
        circuit_state = apply_circuit(circuit, cb_state)

        # Dispatch based on whether the state is parametric or not.
        # apply_circuit creates ParametricCircuitQuantumState for unbound
        # parametric circuits, and GeneralCircuitQuantumState for bound
        # or non-parametric circuits.
        if hasattr(circuit_state, "parametric_circuit"):
            # Unbound parametric circuit → use parametric estimator
            estimate = self.parametric_estimator(
                hamiltonian, circuit_state, param_values
            )
        else:
            # Bound or non-parametric circuit → use non-parametric estimator
            estimate = self.non_parametric_estimator(hamiltonian, circuit_state)

        return estimate.value.real


class QuriPartsTranspiler(
    Transpiler["qp_c.LinearMappedUnboundParametricQuantumCircuit"]
):
    """QURI Parts transpiler for qamomile.circuit module.

    Converts Qamomile QKernels into QURI Parts quantum circuits.

    Example:
        from qamomile.quri_parts import QuriPartsTranspiler
        import qamomile.circuit as qm

        @qm.qkernel
        def bell_state(q0: qm.Qubit, q1: qm.Qubit) -> tuple[qm.Bit, qm.Bit]:
            q0 = qm.h(q0)
            q0, q1 = qm.cx(q0, q1)
            return qm.measure(q0), qm.measure(q1)

        transpiler = QuriPartsTranspiler()
        circuit = transpiler.to_circuit(bell_state)
    """

    def _create_segmentation_pass(self) -> SegmentationPass:
        """Create default segmentation pass (no backend-specific overrides)."""
        return SegmentationPass()

    def _create_emit_pass(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> EmitPass["qp_c.LinearMappedUnboundParametricQuantumCircuit"]:
        """Create QURI Parts emission pass with gate emitter."""
        return QuriPartsEmitPass(bindings, parameters)

    def executor(  # type: ignore[override]
        self,
        sampler: Any = None,
        estimator: Any = None,
        seed: int | None = None,
    ) -> QuriPartsExecutor:
        """Create a QURI Parts executor.

        Args:
            sampler (Any): Optional custom sampler. Defaults to None,
                meaning the qulacs vector sampler is used.
            estimator (Any): Optional custom estimator. Defaults to None,
                meaning the qulacs parametric estimator is used.
            seed (int | None): Optional random seed forwarded to the qulacs
                vector sampler for reproducible sampling. Defaults to None,
                meaning sampling is non-deterministic. The seed is ignored
                when a custom ``sampler`` is supplied, since an arbitrary
                sampler has no standard seed interface.

        Returns:
            QuriPartsExecutor: Executor configured for this backend, bound
                to the given seed.
        """
        return QuriPartsExecutor(sampler, estimator, seed=seed)
