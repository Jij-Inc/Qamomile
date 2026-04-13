"""Cast and binary-operation emission helpers for StandardEmitPass.

Extracted from ``standard_emit.py`` to keep the main class focused on
gate-level dispatch.  Each function mirrors the original method but takes
an explicit ``emit_pass`` parameter instead of ``self``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass

from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.transpiler.passes.eval_utils import evaluate_binop_values

from .qubit_address import QubitAddress, QubitMap


def handle_cast(
    emit_pass: "StandardEmitPass",
    op: CastOperation,
    qubit_map: QubitMap,
) -> None:
    """Handle CastOperation - update qubit_map without emitting gates."""
    result = op.results[0]

    resolved = 0
    for i, qubit_uuid in enumerate(op.qubit_mapping):
        qubit_addr = QubitAddress.from_composite_key(qubit_uuid)
        if qubit_addr not in qubit_map:
            # Fallback: try as plain scalar UUID
            qubit_addr = QubitAddress(qubit_uuid)
        if qubit_addr in qubit_map:
            result_element_addr = QubitAddress(result.uuid, i)
            qubit_map[result_element_addr] = qubit_map[qubit_addr]
            resolved += 1

    if op.qubit_mapping:
        first_addr = QubitAddress.from_composite_key(op.qubit_mapping[0])
        if first_addr not in qubit_map:
            first_addr = QubitAddress(op.qubit_mapping[0])
        if first_addr in qubit_map:
            qubit_map[QubitAddress(result.uuid)] = qubit_map[first_addr]

    total = len(op.qubit_mapping)
    if total > 0 and resolved < total:
        import warnings

        warnings.warn(
            f"CastOperation: {total - resolved}/{total} carrier qubits "
            f"unresolved in qubit_map. "
            f"Downstream measurements may be silently dropped.",
            stacklevel=2,
        )


def _is_param_array_element(operand: Any, parameters: "set[str]") -> bool:
    """True when operand is ``arr[idx]`` and ``arr`` is in ``parameters``."""
    if not hasattr(operand, "parent_array"):
        return False
    if operand.parent_array is None:
        return False
    return operand.parent_array.name in parameters


def evaluate_binop(
    emit_pass: "StandardEmitPass",
    op: BinOp,
    bindings: dict[str, Any],
) -> None:
    """Evaluate a BinOp and store the result in bindings.

    Parameter-array elements take priority over their concrete
    bindings: users often pass a concrete array alongside
    ``parameters=[...]`` as a shape hint, and those elements must stay
    symbolic so the emitted circuit carries backend parameters.
    """
    parameters = emit_pass._resolver.parameters
    lhs_is_param_elem = _is_param_array_element(op.lhs, parameters)
    rhs_is_param_elem = _is_param_array_element(op.rhs, parameters)

    if lhs_is_param_elem:
        lhs = None
    else:
        lhs = emit_pass._resolver.resolve_classical_value(op.lhs, bindings)
    if rhs_is_param_elem:
        rhs = None
    else:
        rhs = emit_pass._resolver.resolve_classical_value(op.rhs, bindings)

    lhs_param_key = emit_pass._resolver.get_parameter_key(op.lhs, bindings)
    rhs_param_key = emit_pass._resolver.get_parameter_key(op.rhs, bindings)

    if lhs is None and lhs_param_key:
        lhs = emit_pass._get_or_create_parameter(lhs_param_key, op.lhs.uuid)
    if rhs is None and rhs_param_key:
        rhs = emit_pass._get_or_create_parameter(rhs_param_key, op.rhs.uuid)

    if lhs is None or rhs is None:
        return

    # For concrete numeric operands, use the shared evaluator.
    # When operands may be symbolic backend parameters, fall back to
    # direct arithmetic (which relies on the backend parameter type
    # supporting __add__ etc.) and use 0-valued defaults for division.
    if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        result = evaluate_binop_values(op.kind, lhs, rhs)
    else:
        result = None
        match op.kind:
            case BinOpKind.ADD:
                result = lhs + rhs
            case BinOpKind.SUB:
                result = lhs - rhs
            case BinOpKind.MUL:
                result = lhs * rhs
            case BinOpKind.DIV:
                result = lhs / rhs if rhs != 0 else 0.0
            case BinOpKind.FLOORDIV:
                result = lhs // rhs if rhs != 0 else 0
            case BinOpKind.POW:
                result = lhs**rhs

    if result is not None and op.results:
        output = op.results[0]
        bindings[output.uuid] = result
        bindings[output.name] = result
