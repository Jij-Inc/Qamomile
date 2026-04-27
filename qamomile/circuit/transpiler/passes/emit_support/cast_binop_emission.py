"""Cast and binary-operation emission helpers for StandardEmitPass.

Extracted from ``standard_emit.py`` to keep the main class focused on
gate-level dispatch.  Each function mirrors the original method but takes
an explicit ``emit_pass`` parameter instead of ``self``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass

from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CondOp,
    NotOp,
)
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.transpiler.passes.eval_utils import (
    evaluate_binop_values,
    evaluate_compop_values,
    evaluate_condop_values,
    evaluate_notop_value,
)

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


def _set_emit_value(bindings: dict[str, Any], uuid: str, value: Any) -> None:
    """Write an emit-time intermediate by UUID, using EmitContext slots when available.

    When ``bindings`` is an ``EmitContext`` instance, route through its
    typed ``set_value`` so the ``_values`` slot stays accurate (useful for
    debugging and future structural enforcement). Otherwise fall back to
    plain dict assignment (e.g. when callers haven't migrated yet).
    """
    set_value = getattr(bindings, "set_value", None)
    if callable(set_value):
        set_value(uuid, value)
    else:
        bindings[uuid] = value


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
        # Write by UUID only. Auto-generated tmp values share generic names
        # (e.g. every UInt arithmetic result is named "uint_tmp"), so a
        # name-keyed write would overwrite earlier tmps with the same name
        # and cause stale-value lookups in chained expressions. UUID-keyed
        # writes are unambiguous and the resolvers (value_resolver.py) check
        # UUID before name, so legitimate name-based reads (parameters, loop
        # variables) are unaffected.
        _set_emit_value(bindings, output.uuid, result)


def evaluate_classical_predicate(
    emit_pass: "StandardEmitPass",
    op: "CompOp | CondOp | NotOp",
    bindings: dict[str, Any],
) -> None:
    """Evaluate a CompOp/CondOp/NotOp at emit time and bind its boolean result.

    Mirrors the predicate evaluation already done by
    ``compile_time_if_lowering`` and ``classical_executor`` so that an
    ``IfOperation`` whose condition reaches emit unfolded — typically
    because its operands depend on a loop variable bound only at emit time
    by ``emit_for_items`` / ``emit_for_unrolled`` — can still be resolved
    via ``resolve_if_condition``.

    All three predicate kinds are handled together because the IR contract
    treats them as a single family: ``CompOp`` is what the frontend
    currently produces from ``==`` / ``<`` / ..., while ``CondOp`` and
    ``NotOp`` are produced by other passes (and are reserved for future
    handle overloads of ``&`` / ``|`` / ``~``). Skipping the latter would
    leave emit as the only pass blind to those ops.

    Args:
        emit_pass: The active emit pass (for resolver access).
        op: The predicate op to evaluate.
        bindings: Current parameter/loop-variable bindings; mutated in
            place on successful evaluation. No-op when operands are
            unresolvable (e.g. measurement bits) or evaluation raises.

    Returns:
        None.
    """
    if not op.results:
        return

    # Defense-in-depth: only accept scalar Python values (bool / int /
    # float) as resolved operands. A backend runtime expression
    # (e.g. ``qiskit.circuit.classical.expr.Expr``) is a truthy Python
    # object, and feeding it through ``bool(lhs and rhs)`` would spuriously
    # fold the predicate to ``True``. The primary protection against this
    # class of bug is the UUID-only binding policy (see ``evaluate_binop``
    # comment above and the matching block at the end of this function) —
    # without name-keyed writes, sub-predicate exprs no longer leak into
    # bindings under colliding tmp names. This isinstance check is a belt-
    # and-suspenders guard against future code paths that might inject
    # non-scalar values into bindings by other means.
    if isinstance(op, NotOp):
        operand = emit_pass._resolver.resolve_classical_value(op.operands[0], bindings)
        if not isinstance(operand, (bool, int, float)):
            return
        result = evaluate_notop_value(operand)
    else:
        lhs = emit_pass._resolver.resolve_classical_value(op.operands[0], bindings)
        rhs = emit_pass._resolver.resolve_classical_value(op.operands[1], bindings)
        if not isinstance(lhs, (bool, int, float)) or not isinstance(
            rhs, (bool, int, float)
        ):
            return
        if isinstance(op, CompOp):
            result = evaluate_compop_values(op.kind, lhs, rhs)
        else:  # CondOp
            result = evaluate_condop_values(op.kind, lhs, rhs)

    if result is None:
        return

    output = op.results[0]
    # Write by UUID only — see the matching comment in evaluate_binop above
    # for why name-keyed writes are unsafe for tmp values like "bit_tmp".
    _set_emit_value(bindings, output.uuid, result)
