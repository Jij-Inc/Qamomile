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
    CompOp,
    CondOp,
    NotOp,
)
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.transpiler.gate_emitter import default_combine_symbolic
from qamomile.circuit.transpiler.passes.eval_utils import (
    FoldPolicy,
    fold_classical_op,
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

    Tries the shared ``fold_classical_op`` first for a clean concrete
    fold (which already encapsulates the runtime-parameter guard).
    Falls back to creating backend ``Parameter`` symbols and doing
    symbolic arithmetic when one or both operands are runtime
    parameters — that's the path that lets ``rx(q, gamma * 2)`` produce
    a circuit with a single ``Parameter("gamma") * 2`` expression rather
    than baking in a placeholder.
    """
    parameters = emit_pass._resolver.parameters

    # Phase 1: try the concrete fold path via the shared API.
    folded = fold_classical_op(
        op,
        lambda v: emit_pass._resolver.resolve_classical_value(v, bindings),
        parameters,
        FoldPolicy.EMIT_RESPECT_PARAMS,
    )
    if folded is not None and op.results:
        # Write by UUID only. Auto-generated tmp values share generic names
        # (e.g. every UInt arithmetic result is named "uint_tmp"), so a
        # name-keyed write would overwrite earlier tmps with the same name
        # and cause stale-value lookups in chained expressions. UUID-keyed
        # writes are unambiguous and the resolvers (value_resolver.py) check
        # UUID before name, so legitimate name-based reads (parameters, loop
        # variables) are unaffected.
        _set_emit_value(bindings, op.results[0].uuid, folded)
        return

    # Phase 2: backend Parameter symbolic path. Operands that are either
    # runtime-parameter array elements or top-level parameters become
    # backend Parameter objects; the arithmetic is performed symbolically
    # using the backend Parameter's overloaded ``__add__`` etc.
    lhs = (
        None
        if _is_param_array_element(op.lhs, parameters)
        else emit_pass._resolver.resolve_classical_value(op.lhs, bindings)
    )
    rhs = (
        None
        if _is_param_array_element(op.rhs, parameters)
        else emit_pass._resolver.resolve_classical_value(op.rhs, bindings)
    )

    lhs_param_key = emit_pass._resolver.get_parameter_key(op.lhs, bindings)
    rhs_param_key = emit_pass._resolver.get_parameter_key(op.rhs, bindings)

    if lhs is None and lhs_param_key:
        lhs = emit_pass._get_or_create_parameter(lhs_param_key, op.lhs.uuid)
    if rhs is None and rhs_param_key:
        rhs = emit_pass._get_or_create_parameter(rhs_param_key, op.rhs.uuid)

    if lhs is None or rhs is None:
        return

    # Symbolic arithmetic. Concrete-only operands have already been
    # handled by ``fold_classical_op`` above, so we land here only when
    # at least one operand is a backend Parameter (or a previously
    # combined symbolic value). The actual operator dispatch is delegated
    # to the backend emitter's ``combine_symbolic`` if it provides one,
    # so that backends whose Parameter type lacks Python operator
    # overloads (e.g. QURI Parts' Rust-backed Parameter, which raises
    # ``TypeError`` for ``param * float``) can substitute a backend-native
    # representation such as a linear-combination dict. Backends with
    # arithmetic-capable Parameters (Qiskit ``ParameterExpression``,
    # CUDA-Q parameters) need not implement the hook — we fall back to
    # ``default_combine_symbolic`` which performs the original Python
    # operator dispatch.
    if op.kind is None:
        return
    combine = getattr(emit_pass._emitter, "combine_symbolic", None)
    if callable(combine):
        result = combine(op.kind, lhs, rhs)
    else:
        result = default_combine_symbolic(op.kind, lhs, rhs)

    if result is not None and op.results:
        _set_emit_value(bindings, op.results[0].uuid, result)


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

    Delegates the actual fold (including runtime-parameter guard and
    strict scalar typing) to ``fold_classical_op`` so that all three
    callers of the same kind dispatch share one implementation. There
    is no symbolic-Parameter fallback for predicates: a runtime classical
    expression cannot be expressed as a folded scalar, so the op is left
    unbound and downstream emit handles it as a runtime condition.

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

    result = fold_classical_op(
        op,
        lambda v: emit_pass._resolver.resolve_classical_value(v, bindings),
        emit_pass._resolver.parameters,
        FoldPolicy.EMIT_RESPECT_PARAMS,
    )
    if result is None:
        return

    # Write by UUID only — see the matching comment in evaluate_binop above
    # for why name-keyed writes are unsafe for tmp values like "bit_tmp".
    _set_emit_value(bindings, op.results[0].uuid, result)
