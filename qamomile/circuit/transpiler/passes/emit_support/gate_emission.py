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
    OperandResolutionInfo,
    QubitIndexResolutionError,
    ResolutionFailureReason,
)
from qamomile.circuit.transpiler.value_resolver import (
    ValueResolver as UnifiedValueResolver,
)

from .qubit_address import QubitAddress, QubitMap


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
                        element_indices_names.append(
                            f"{idx.parent_array.name}[{idx.element_indices[0].name if idx.element_indices else '?'}]"
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
                            element_indices_names.append(
                                f"{idx.parent_array.name}[{idx.element_indices[0].name if idx.element_indices else '?'}]"
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

    theta is always a Value stored as the last element of operands
    for rotation gates.
    """
    theta = op.theta
    if theta is not None:
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
        resolved = UnifiedValueResolver(context=bindings, bindings=bindings).resolve(
            theta
        )
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

    return 0.0
