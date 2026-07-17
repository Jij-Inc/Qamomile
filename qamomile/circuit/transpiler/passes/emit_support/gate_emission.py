"""Gate emission helper extracted from StandardEmitPass.

Provides ``emit_gate`` and ``resolve_angle`` as module-level functions so
they can be tested and reused independently of the class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass

from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType
from qamomile.circuit.transpiler.errors import (
    EmitError,
    OperandResolutionInfo,
    QubitAliasError,
    QubitIndexResolutionError,
    ResolutionFailureReason,
)
from qamomile.circuit.transpiler.value_resolver import (
    ValueResolver as UnifiedValueResolver,
)

from .qubit_address import QubitAddress, QubitMap


def reject_duplicate_physical_indices(
    gate_label: str,
    physical_indices: list[int],
    operand_names: list[str] | None = None,
) -> None:
    """Reject a multi-qubit gate whose qubits resolve to the same physical qubit.

    A multi-qubit gate (``cx`` / ``cz`` / ``swap`` / ``toffoli`` and any
    controlled block such as ``qmc.control(...)``) is physically defined only on
    *distinct* qubits. The frontend's ``_check_qubit_alias`` already rejects the
    scalar ``cx(q, q)`` case by ``logical_id`` at trace time, but symbolic array
    indices — ``cx(qs[i], qs[j])`` where ``i == j`` only at runtime, or two
    loop-variable indices that coincide after unrolling — resolve to the same
    physical qubit only at emit time. Without this check the duplicate reaches
    the backend as a raw, backend-specific failure (Qiskit ``CircuitError:
    'duplicate qubit arguments'``, a CUDA-Q simulator crash, or — on a backend
    that does not validate — a silently ill-defined gate). Raising a Qamomile
    ``QubitAliasError`` gives one actionable, backend-independent diagnostic.

    This is the shared checker used both for native gates (``emit_gate`` via
    ``_reject_aliased_operands``) and for controlled / composite blocks (the
    ``append_gate`` sites in ``controlled_emission``), so the same diagnostic
    covers every multi-qubit emission path on every backend.

    Args:
        gate_label (str): Human-readable name of the gate for the message
            (e.g. ``"CX"`` or ``"controlled gate"``).
        physical_indices (list[int]): The resolved physical qubit indices the
            gate acts on, in operand order.
        operand_names (list[str] | None): Optional display names aligned with
            ``physical_indices`` (e.g. ``["qs[i]", "qs[j]"]``). When absent,
            the message falls back to ``qubit<index>``. Defaults to None.

    Returns:
        None

    Raises:
        QubitAliasError: If any physical qubit index repeats.
    """
    if len(physical_indices) < 2:
        return

    def _name(pos: int, phys: int) -> str:
        """Return the display name for the operand at ``pos``.

        Args:
            pos (int): Index of the operand within the gate.
            phys (int): The operand's resolved physical qubit index, used as
                the ``qubit<index>`` fallback when no name is available.

        Returns:
            str: The supplied operand name if present and non-empty, otherwise
                ``qubit<phys>``.
        """
        if operand_names is not None and pos < len(operand_names):
            return operand_names[pos] or f"qubit<{phys}>"
        return f"qubit<{phys}>"

    seen: dict[int, str] = {}
    for pos, phys in enumerate(physical_indices):
        if phys in seen:
            raise QubitAliasError(
                f"Gate '{gate_label}' resolved two operands, '{seen[phys]}' and "
                f"'{_name(pos, phys)}', to the same physical qubit (index "
                f"{phys}). A multi-qubit gate requires distinct qubits. This "
                f"usually means two array indices that are equal at runtime — "
                f"e.g. `cx(qs[i], qs[j])` on the diagonal where i == j. Guard "
                f"the indices so they differ (skip the diagonal), or use "
                f"distinct registers."
            )
        seen[phys] = _name(pos, phys)


def _reject_aliased_operands(
    op: GateOperation,
    qubit_ops: list[Any],
    qubit_indices: list[int],
) -> None:
    """Reject a native multi-qubit gate whose operands resolve to one qubit.

    Thin adapter over ``reject_duplicate_physical_indices`` for the native-gate
    path: it supplies the gate type name and the operand display names so the
    error can quote ``qs[i]`` / ``qs[j]``.

    Args:
        op (GateOperation): The gate being emitted (for its gate type).
        qubit_ops (list[Any]): The gate's qubit operand Values, in order.
        qubit_indices (list[int]): The resolved physical qubit indices, in the
            same order as ``qubit_ops``.

    Returns:
        None

    Raises:
        QubitAliasError: If two operands of the gate resolve to the same
            physical qubit index.
    """
    gate_name = op.gate_type.name if op.gate_type else "unknown"
    operand_names = [getattr(v, "name", None) or "" for v in qubit_ops]
    reject_duplicate_physical_indices(gate_name, qubit_indices, operand_names)


def emit_gate(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    op: GateOperation,
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit a single gate operation."""
    qubit_indices = []
    failed_operands: list[OperandResolutionInfo] = []
    qubit_ops = op.qubit_operands

    for v in qubit_ops:
        result = emit_pass._resolver.resolve_qubit_index_detailed(
            v, qubit_map, bindings
        )
        if result.success:
            assert result.index is not None
            qubit_indices.append(result.index)
        else:
            # Collect detailed failure info
            element_indices_names = []
            if v.element_indices:
                for idx in v.element_indices:
                    if idx.parent_array is not None:
                        nested_index = idx.element_indices[0]
                        display_index = nested_index.name or (
                            str(nested_index.get_const())
                            if nested_index.is_constant()
                            else "?"
                        )
                        element_indices_names.append(
                            f"{idx.parent_array.name}[{display_index}]"
                        )
                    else:
                        element_indices_names.append(idx.name)

            info = OperandResolutionInfo(
                operand_name=v.name,
                operand_uuid=v.uuid,
                is_array_element=v.parent_array is not None,
                parent_array_name=v.parent_array.name if v.parent_array else None,
                element_indices_names=element_indices_names,
                failure_reason=result.failure_reason or ResolutionFailureReason.UNKNOWN,
                failure_details=result.failure_details,
            )
            failed_operands.append(info)

    if not qubit_indices or failed_operands:
        # If we have no successful resolutions or some failed, raise detailed error
        if not failed_operands:
            # All operands returned None without detailed failure info
            for v in qubit_ops:
                element_indices_names = []
                if v.element_indices:
                    for idx in v.element_indices:
                        if idx.parent_array is not None:
                            nested_index = idx.element_indices[0]
                            display_index = nested_index.name or (
                                str(nested_index.get_const())
                                if nested_index.is_constant()
                                else "?"
                            )
                            element_indices_names.append(
                                f"{idx.parent_array.name}[{display_index}]"
                            )
                        else:
                            element_indices_names.append(idx.name)

                failed_operands.append(
                    OperandResolutionInfo(
                        operand_name=v.name,
                        operand_uuid=v.uuid,
                        is_array_element=v.parent_array is not None,
                        parent_array_name=v.parent_array.name
                        if v.parent_array
                        else None,
                        element_indices_names=element_indices_names,
                        failure_reason=ResolutionFailureReason.UNKNOWN,
                        failure_details="No indices resolved but no specific failure recorded",
                    )
                )

        raise QubitIndexResolutionError(
            gate_type=op.gate_type.name if op.gate_type else "unknown",
            operand_infos=failed_operands,
            available_bindings_keys=list(bindings.keys()),
            available_qubit_map_keys=[str(k) for k in qubit_map.keys()],
        )

    _reject_aliased_operands(op, qubit_ops, qubit_indices)

    match op.gate_type:
        case GateOperationType.H:
            emit_pass._emitter.emit_h(circuit, qubit_indices[0])
        case GateOperationType.X:
            emit_pass._emitter.emit_x(circuit, qubit_indices[0])
        case GateOperationType.Y:
            emit_pass._emitter.emit_y(circuit, qubit_indices[0])
        case GateOperationType.Z:
            emit_pass._emitter.emit_z(circuit, qubit_indices[0])
        case GateOperationType.T:
            emit_pass._emitter.emit_t(circuit, qubit_indices[0])
        case GateOperationType.S:
            emit_pass._emitter.emit_s(circuit, qubit_indices[0])
        case GateOperationType.SDG:
            emit_pass._emitter.emit_sdg(circuit, qubit_indices[0])
        case GateOperationType.TDG:
            emit_pass._emitter.emit_tdg(circuit, qubit_indices[0])
        case GateOperationType.CX:
            emit_pass._emitter.emit_cx(circuit, qubit_indices[0], qubit_indices[1])
        case GateOperationType.CZ:
            emit_pass._emitter.emit_cz(circuit, qubit_indices[0], qubit_indices[1])
        case GateOperationType.SWAP:
            emit_pass._emitter.emit_swap(circuit, qubit_indices[0], qubit_indices[1])
        case GateOperationType.TOFFOLI:
            emit_pass._emitter.emit_toffoli(
                circuit, qubit_indices[0], qubit_indices[1], qubit_indices[2]
            )
        case GateOperationType.P:
            angle = resolve_angle(emit_pass, op, bindings)
            emit_pass._emitter.emit_p(circuit, qubit_indices[0], angle)
        case GateOperationType.RX:
            angle = resolve_angle(emit_pass, op, bindings)
            emit_pass._emitter.emit_rx(circuit, qubit_indices[0], angle)
        case GateOperationType.RY:
            angle = resolve_angle(emit_pass, op, bindings)
            emit_pass._emitter.emit_ry(circuit, qubit_indices[0], angle)
        case GateOperationType.RZ:
            angle = resolve_angle(emit_pass, op, bindings)
            emit_pass._emitter.emit_rz(circuit, qubit_indices[0], angle)
        case GateOperationType.CP:
            angle = resolve_angle(emit_pass, op, bindings)
            emit_pass._emitter.emit_cp(
                circuit, qubit_indices[0], qubit_indices[1], angle
            )
        case GateOperationType.RZZ:
            angle = resolve_angle(emit_pass, op, bindings)
            emit_pass._emitter.emit_rzz(
                circuit, qubit_indices[0], qubit_indices[1], angle
            )
        case _:
            raise RuntimeError(f"Unsupported gate type: {op.gate_type}")

    # Update qubit_map for new versions
    for i, result in enumerate(op.results):
        if i < len(qubit_indices):
            qubit_map[QubitAddress(result.uuid)] = qubit_indices[i]


def _theta_is_param_array_element(
    theta: Any,
    parameters: "set[str]",
) -> bool:
    """True when theta is ``arr[idx]`` and ``arr`` is a declared parameter.

    Used by ``resolve_angle`` to prioritise backend-parameter creation
    over concrete binding lookup — users pass a concrete array for an
    array parameter as a shape hint, and those elements must remain
    symbolic for runtime binding.
    """
    if not hasattr(theta, "parent_array") or theta.parent_array is None:
        return False
    parent_name = theta.parent_array.name
    return parent_name in parameters


def resolve_angle(
    emit_pass: "StandardEmitPass",
    op: GateOperation,
    bindings: dict[str, Any],
) -> float | Any:
    """Resolve angle parameter for rotation gates.

    The angle is always a `Value` stored as the last operand of a rotation
    gate.

    Args:
        emit_pass (StandardEmitPass): Active emit pass providing parameter
            resolution and creation.
        op (GateOperation): Rotation gate whose angle is resolved.
        bindings (dict[str, Any]): Active compile-time bindings.

    Returns:
        float | Any: Concrete angle or backend parameter expression.

    Raises:
        EmitError: If the rotation angle is missing or cannot be resolved.
    """
    return resolve_angle_value(emit_pass, op.theta, bindings)


def resolve_angle_value(
    emit_pass: "StandardEmitPass",
    theta: Any,
    bindings: dict[str, Any],
) -> float | Any:
    """Resolve a bare angle Value to a concrete float or backend parameter.

    Shared core of :func:`resolve_angle` so operations carrying a standalone
    angle Value reuse the exact same resolution order as rotation-gate thetas.

    Args:
        emit_pass (StandardEmitPass): Active emit pass providing the
            resolver and backend-parameter factory.
        theta (Any): Angle ``Value`` to resolve.
        bindings (dict[str, Any]): Active compile-time bindings.

    Returns:
        float | Any: A concrete ``float`` when the angle is bound, or a
            backend parameter expression when it stays symbolic.

    Raises:
        EmitError: If the angle is missing or cannot be resolved to a concrete
            value or backend parameter. Emitting ``0.0`` in this case would
            silently change the requested unitary.
    """
    if theta is None:
        raise EmitError(
            "Cannot emit an angle-dependent operation without an angle Value.",
            operation="AngleResolution",
        )

    # Shape-hint fast path: if theta is an element of a declared
    # parameter array (``gamma[p]`` with ``parameters=['gamma']``),
    # skip bindings lookup and go straight to backend parameter
    # creation. Otherwise the concrete shape-hint binding would
    # short-circuit the symbolic path.
    if _theta_is_param_array_element(theta, emit_pass._resolver.parameters):
        param_key = emit_pass._resolver.get_parameter_key(theta, bindings)
        if param_key:
            return emit_pass._get_or_create_parameter(param_key, theta.uuid)

    # Use unified resolver for value resolution.
    resolved = UnifiedValueResolver(context=bindings, bindings=bindings).resolve(theta)
    if resolved is not None:
        if not isinstance(resolved, (int, float)):
            return resolved
        return float(resolved)

    # Fall back to array element resolution from the emission resolver.
    array_resolved = emit_pass._resolver.resolve_classical_value(theta, bindings)
    if array_resolved is not None:
        if not isinstance(array_resolved, (int, float)):
            return array_resolved
        return float(array_resolved)

    # Fall back to symbolic parameter creation.
    param_key = emit_pass._resolver.get_parameter_key(theta, bindings)
    if param_key:
        return emit_pass._get_or_create_parameter(param_key, theta.uuid)

    name = getattr(theta, "name", None) or "<anonymous>"
    raise EmitError(
        f"Cannot resolve angle value {name!r}. Bind it at transpile time or "
        "declare it as a runtime parameter.",
        operation="AngleResolution",
    )
