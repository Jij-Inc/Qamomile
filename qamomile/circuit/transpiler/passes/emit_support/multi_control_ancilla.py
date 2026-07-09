"""Clean-ancilla planning for the shared multi-controlled decomposition.

Backends without a native multi-controlled gate primitive lower an
irreducible ``n``-controlled single-qubit gate through the standard
Toffoli-cascade construction (arXiv:2307.07478, Appendix A.3): the
logical AND of all ``n`` controls is accumulated onto ``n - 1`` clean
ancilla qubits with a cascade of Toffoli gates, the gate is applied
once under a single control (the last ancilla), and the cascade is
uncomputed in reverse. Every ancilla therefore returns to ``|0>`` and
the same pool can be reused by every multi-controlled gate in the
segment.

Backend circuits are created with a fixed qubit count before emission
starts, so the pool must be sized up front. This module provides:

* :func:`estimate_multi_control_ancilla_demand` — a static walk over a
  quantum segment's operations that upper-bounds the largest composed
  control count any single gate can reach at emit time, mirroring the
  structural reductions in ``controlled_emission.emit_multi_controlled_gate``.
* :class:`MultiControlAncillaPool` — the reserved block of physical
  qubit indices appended after the segment's data qubits.

The estimate only needs to be an **upper bound**: over-reservation
costs unused circuit qubits, while under-reservation is caught at emit
time by ``StandardEmitPass._emit_irreducible_multi_controlled_gate``
and reported as a compiler bug.
"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Any

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp
from qamomile.circuit.ir.operation.composite_gate import CompositeGateOperation
from qamomile.circuit.ir.operation.control_flow import ForOperation, HasNestedOps
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    ControlledUOperation,
    GateOperation,
    GateOperationType,
    SymbolicControlledU,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.errors import EmitError

if TYPE_CHECKING:
    from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
        ValueResolver,
    )

_MAX_PRECISE_FOR_LOOP_ITERATIONS = 1024


class MultiControlAncillaPool:
    """A reserved block of clean ancilla qubits for multi-control lowering.

    The pool occupies a contiguous range of physical qubit indices
    appended after a quantum segment's data qubits. Because the
    Toffoli-cascade decomposition uncomputes every ancilla back to
    ``|0>`` before it finishes — and never nests inside itself — each
    multi-controlled gate may simply take the first ``k`` indices of
    the pool; no acquire/release bookkeeping is needed.
    """

    def __init__(self, first_index: int, count: int) -> None:
        """Initialize the pool.

        Args:
            first_index (int): Physical index of the first reserved
                ancilla qubit.
            count (int): Number of reserved ancilla qubits. Must be
                non-negative.

        Raises:
            ValueError: If ``count`` is negative.
        """
        if count < 0:
            raise ValueError(f"Ancilla pool count must be non-negative, got {count}.")
        self._first_index = first_index
        self._count = count

    @property
    def count(self) -> int:
        """Return the number of reserved ancilla qubits.

        Returns:
            int: Pool size.
        """
        return self._count

    def take(self, count: int) -> list[int] | None:
        """Return ``count`` clean ancilla indices from the pool.

        Args:
            count (int): Number of ancilla qubits requested.

        Returns:
            list[int] | None: The first ``count`` reserved physical
                indices, or None when the pool is smaller than the
                request (an estimation bug surfaced by the caller).
        """
        if count > self._count:
            return None
        return list(range(self._first_index, self._first_index + count))


def _x_like_demand(num_controls: int) -> int:
    """Return the ancilla demand of an X / Z gate under ``num_controls``.

    Up to two controls are absorbed natively (CX / CZ for one control,
    Toffoli — Hadamard-conjugated for Z — for two), so only three or
    more controls reach the cascade.

    Args:
        num_controls (int): Composed control count on the gate.

    Returns:
        int: ``num_controls - 1`` when the cascade is needed, else 0.
    """
    return num_controls - 1 if num_controls >= 3 else 0


def _rotation_like_demand(num_controls: int) -> int:
    """Return the ancilla demand of a non-X/Z gate under ``num_controls``.

    Single-control versions are emitted natively (``emit_ch`` /
    ``emit_cp`` / ``emit_cr*`` …), so two or more controls reach the
    cascade.

    Args:
        num_controls (int): Composed control count on the gate.

    Returns:
        int: ``num_controls - 1`` when the cascade is needed, else 0.
    """
    return num_controls - 1 if num_controls >= 2 else 0


def _gate_demand(gate_type: GateOperationType | None, num_controls: int) -> int:
    """Upper-bound the ancilla demand of one gate under composed controls.

    Mirrors the structural reductions in
    ``controlled_emission.emit_multi_controlled_gate``: multi-qubit gate
    types first absorb their own control-like operands into the control
    set (``CX -> X``, ``CZ -> Z``, ``TOFFOLI -> X``, ``CP -> P``;
    ``SWAP`` controls the middle CX of its conjugation; ``RZZ`` controls
    only its central RZ), then the single-target residue is lowered.

    Because under-reservation is a real bug (invariant #2 in the module
    docstring), this ``match`` MUST stay in lockstep with that emitter:
    any new reducible gate type added there must be mirrored here, or the
    estimate can fall below the ancillas the emitter actually consumes.

    Args:
        gate_type (GateOperationType | None): Gate kind, or None for a
            malformed operation (treated pessimistically as
            rotation-like).
        num_controls (int): Composed control count accumulated from
            enclosing controlled operations.

    Returns:
        int: Upper bound on clean ancillas the gate's lowering may take
            from the pool.
    """
    match gate_type:
        case GateOperationType.CX:
            return _x_like_demand(num_controls + 1)
        case GateOperationType.CZ:
            return _x_like_demand(num_controls + 1)
        case GateOperationType.TOFFOLI:
            return _x_like_demand(num_controls + 2)
        case GateOperationType.SWAP:
            return _x_like_demand(num_controls + 1)
        case GateOperationType.CP:
            return _rotation_like_demand(num_controls + 1)
        case GateOperationType.RZZ:
            return _rotation_like_demand(num_controls)
        case GateOperationType.X | GateOperationType.Z:
            return _x_like_demand(num_controls)
        case _:
            return _rotation_like_demand(num_controls)


def _resolve_size_value(
    size: Value,
    resolver: "ValueResolver",
    bindings: dict[str, Any],
) -> int | None:
    """Resolve an array-shape Value to a concrete non-negative int.

    Args:
        size (Value): Shape dimension value of a ``Vector[Qubit]``.
        resolver (ValueResolver): Emit value resolver used for bound
            lookups.
        bindings (dict[str, Any]): Active compile-time bindings.

    Returns:
        int | None: The concrete size, or None when it cannot be
            resolved statically.
    """
    if size.is_constant():
        resolved = size.get_const()
    else:
        resolved = resolver.resolve_classical_value(size, bindings)
    if isinstance(resolved, bool) or not isinstance(resolved, numbers.Integral):
        return None
    resolved = int(resolved)
    return resolved if resolved >= 0 else None


def _quantum_width_upper_bound(
    values: list[Value],
    resolver: "ValueResolver",
    bindings: dict[str, Any],
) -> int:
    """Upper-bound the flattened qubit count of quantum operand values.

    Scalar quantum values count as one qubit; ``Vector[Qubit]`` operands
    count as their resolved leading dimension. Unresolvable vector
    dimensions fail fast because emission also requires concrete quantum
    array sizes for allocation.

    Args:
        values (list[Value]): Quantum operand values (controls).
        resolver (ValueResolver): Emit value resolver used for bound
            lookups.
        bindings (dict[str, Any]): Active compile-time bindings.

    Returns:
        int: Estimated total qubit width.

    Raises:
        EmitError: If a ``Vector[Qubit]`` operand has no statically
            resolvable leading dimension.
    """
    total = 0
    for value in values:
        if isinstance(value, ArrayValue):
            if not value.shape:
                raise EmitError(
                    f"Cannot resolve Vector[Qubit] control width for "
                    f"{value.name!r}: missing shape metadata.",
                    operation="multi-control ancilla estimation",
                )
            size = _resolve_size_value(value.shape[0], resolver, bindings)
            if size is None:
                raise EmitError(
                    f"Cannot resolve Vector[Qubit] control width for "
                    f"{value.name!r}. Structural UInt parameters must be "
                    "bound at transpile time.",
                    operation="multi-control ancilla estimation",
                )
            total += size
        else:
            total += 1
    return total


def _controlled_u_num_controls_upper_bound(
    op: ControlledUOperation,
    resolver: "ValueResolver",
    bindings: dict[str, Any],
) -> int:
    """Upper-bound the resolved control count of a controlled-U operation.

    Args:
        op (ControlledUOperation): Concrete or symbolic controlled-U
            operation.
        resolver (ValueResolver): Emit value resolver used for bound
            lookups.
        bindings (dict[str, Any]): Active compile-time bindings.

    Returns:
        int: The concrete ``num_controls`` when statically known,
            otherwise the flattened width of the control operand prefix
            (every pool qubit could act as a control).
    """
    if isinstance(op, ConcreteControlledU):
        return max(op.num_controls, 0)
    assert isinstance(op, SymbolicControlledU)
    if op.control_indices is not None:
        return len(op.control_indices)
    resolved = resolver.resolve_classical_value(op.num_controls, bindings)
    if isinstance(resolved, bool):
        pass
    elif isinstance(resolved, numbers.Integral):
        return max(int(resolved), 0)
    control_operands = [
        v for v in op.operands[: op.num_control_args] if v.type.is_quantum()
    ]
    return _quantum_width_upper_bound(control_operands, resolver, bindings)


def _controlled_block_bindings(
    op: ControlledUOperation,
    resolver: "ValueResolver",
    bindings: dict[str, Any],
) -> dict[str, Any]:
    """Build inner-block bindings for walking a controlled-U's block.

    Mirrors the binding steps of ``resolve_controlled_u_call``: classical
    call-site operands are bound to the block's formal parameters and
    ``Vector[Qubit]`` actual-argument lengths are seeded onto the formal
    shape Values, so loop bounds and control counts inside the block
    (e.g. ``n = q.shape[0]`` followed by ``qmc.range(1, n)``) resolve
    during estimation exactly as they do at emit time.

    Args:
        op (ControlledUOperation): Controlled-U operation whose block is
            about to be walked. ``op.block`` must not be None.
        resolver (ValueResolver): Emit value resolver.
        bindings (dict[str, Any]): Caller-visible bindings.

    Returns:
        dict[str, Any]: Block-local bindings extending ``bindings``.
    """
    from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
        _bind_quantum_input_shapes,
    )

    if isinstance(op, SymbolicControlledU):
        num_control_args = op.num_control_args
    else:
        assert isinstance(op, ConcreteControlledU)
        num_control_args = op.num_controls
    remaining = list(op.operands[num_control_args:])
    target_qubit_operands = [v for v in remaining if v.type.is_quantum()]
    param_operands = [
        v for v in remaining if v.type.is_classical() or v.type.is_object()
    ]
    local_bindings = resolver.bind_block_params(op.block, param_operands, bindings)
    _bind_quantum_input_shapes(
        resolver, op.block, target_qubit_operands, bindings, local_bindings
    )
    return local_bindings


def estimate_multi_control_ancilla_demand(
    operations: list[Operation],
    resolver: "ValueResolver",
    bindings: dict[str, Any],
) -> int:
    """Upper-bound the clean-ancilla demand of a quantum segment.

    Walks the segment recursively, tracking the control count inherited
    from enclosing controlled operations, and returns the largest
    per-gate demand found. Because the Toffoli-cascade decomposition
    uncomputes its ancillas before finishing, gates never hold pool
    ancillas concurrently and the segment-wide demand is the maximum —
    not the sum — of the per-gate demands.

    Args:
        operations (list[Operation]): Segment operations (the same list
            handed to ``ResourceAllocator.allocate``).
        resolver (ValueResolver): Emit value resolver used to resolve
            symbolic control counts and vector shapes.
        bindings (dict[str, Any]): Active compile-time bindings.

    Returns:
        int: Number of ancilla qubits to reserve; zero when the segment
            contains no gate that can reach the cascade path.
    """
    # The walk folds BinOps into its bindings view (mirroring the emit
    # walkers) so control counts derived from arithmetic resolve; work
    # on a copy so the caller's emit bindings stay untouched.
    return _demand_in_operations(operations, 0, resolver, dict(bindings))


def _demand_in_operations(
    operations: list[Operation],
    inherited_controls: int,
    resolver: "ValueResolver",
    bindings: dict[str, Any],
) -> int:
    """Return the maximum per-gate ancilla demand within ``operations``.

    Args:
        operations (list[Operation]): Operations walked in order.
        inherited_controls (int): Control count accumulated from
            enclosing controlled operations.
        resolver (ValueResolver): Emit value resolver.
        bindings (dict[str, Any]): Active compile-time bindings.

    Returns:
        int: Maximum demand over all reachable gates.
    """
    demand = 0
    for op in operations:
        if isinstance(op, BinOp):
            _fold_binop_into_bindings(op, resolver, bindings)
            continue
        demand = max(
            demand,
            _demand_of_operation(op, inherited_controls, resolver, bindings),
        )
    return demand


def _fold_binop_into_bindings(
    op: BinOp,
    resolver: "ValueResolver",
    bindings: dict[str, Any],
) -> None:
    """Fold a concrete BinOp result into the estimation bindings.

    The emit walkers evaluate classical BinOps and record the result
    under the result Value's UUID before downstream operations resolve
    against it; the estimation walk mirrors the concrete-fold half of
    that behaviour so control counts and loop bounds computed from
    arithmetic (e.g. ``num_controls = n - k``) resolve. The backend
    Parameter symbolic half is intentionally skipped: a structural
    value depending on a runtime parameter cannot be resolved at
    estimation time (nor at emit time).

    Args:
        op (BinOp): Classical binary operation to fold.
        resolver (ValueResolver): Emit value resolver.
        bindings (dict[str, Any]): Estimation-local bindings updated in
            place (UUID-keyed).
    """
    from qamomile.circuit.transpiler.passes.eval_utils import (
        FoldPolicy,
        fold_classical_op,
    )

    folded = fold_classical_op(
        op,
        lambda v: resolver.resolve_classical_value(v, bindings),
        resolver.parameters,
        FoldPolicy.EMIT_RESPECT_PARAMS,
    )
    if folded is not None and op.results:
        bindings[op.results[0].uuid] = folded


def _demand_of_operation(
    op: Operation,
    inherited_controls: int,
    resolver: "ValueResolver",
    bindings: dict[str, Any],
) -> int:
    """Return the maximum ancilla demand of one operation subtree.

    Args:
        op (Operation): Operation to inspect.
        inherited_controls (int): Control count accumulated from
            enclosing controlled operations.
        resolver (ValueResolver): Emit value resolver.
        bindings (dict[str, Any]): Active compile-time bindings.

    Returns:
        int: Maximum demand over the operation and its nested blocks.
    """
    if isinstance(op, ControlledUOperation):
        own = _controlled_u_num_controls_upper_bound(op, resolver, bindings)
        if op.block is None:
            return 0
        inner_bindings = _controlled_block_bindings(op, resolver, bindings)
        return _demand_in_operations(
            op.block.operations, inherited_controls + own, resolver, inner_bindings
        )

    if isinstance(op, GateOperation):
        if inherited_controls == 0:
            return 0
        return _gate_demand(op.gate_type, inherited_controls)

    # Composite/inverse blocks are walked with the OUTER bindings, unlike
    # ControlledUOperation above which seeds block-local bindings via
    # _controlled_block_bindings. This is deliberate and cannot
    # under-reserve: a control count depending on an inner block parameter
    # would simply fail to resolve and fall back to the control-operand
    # width, which is still an upper bound. Real composites (QFT/QPE/IQFT)
    # and inverse bodies carry no parameter-dependent internal
    # multi-controls, so the distinction is currently dormant.
    if isinstance(op, CompositeGateOperation):
        own = _quantum_width_upper_bound(list(op.control_qubits), resolver, bindings)
        if op.implementation_block is None:
            return 0
        return _demand_in_operations(
            op.implementation_block.operations,
            inherited_controls + own,
            resolver,
            bindings,
        )

    if isinstance(op, InverseBlockOperation):
        own = _quantum_width_upper_bound(list(op.control_qubits), resolver, bindings)
        blocks = [
            b for b in (op.implementation_block, op.source_block) if b is not None
        ]
        return max(
            (
                _demand_in_operations(
                    block.operations, inherited_controls + own, resolver, bindings
                )
                for block in blocks
            ),
            default=0,
        )

    if isinstance(op, PauliEvolveOp):
        # Under controls only the central RZ of each term carries them;
        # the constant-term phase uses one fewer control and is dominated.
        return _rotation_like_demand(inherited_controls)

    if isinstance(op, ForOperation):
        return _for_operation_demand(op, inherited_controls, resolver, bindings)

    if isinstance(op, HasNestedOps):
        return max(
            (
                _demand_in_operations(body, inherited_controls, resolver, bindings)
                for body in op.nested_op_lists()
            ),
            default=0,
        )

    return 0


def _for_operation_demand(
    op: ForOperation,
    inherited_controls: int,
    resolver: "ValueResolver",
    bindings: dict[str, Any],
) -> int:
    """Return the maximum ancilla demand across a for loop's iterations.

    A ``SymbolicControlledU`` inside a loop body may derive its control
    count from the loop variable (e.g. ``qmc.control(qmc.x,
    num_controls=k)`` inside ``qmc.range(...)``), which only resolves
    with the loop variable bound. Small resolved loops are walked per
    iteration, mirroring emit-time unrolling. Large resolved loops —
    including ones so long that ``len(range(...))`` itself overflows —
    are walked once without the loop variable to keep estimation
    bounded; loop-variable-dependent counts then fall back to their
    control-operand width, which is a conservative upper bound for
    valid controlled calls. When the bounds cannot be resolved, the same
    single symbolic body walk still counts loop-independent demand.

    Args:
        op (ForOperation): Loop operation whose body is inspected.
        inherited_controls (int): Control count accumulated from
            enclosing controlled operations.
        resolver (ValueResolver): Emit value resolver.
        bindings (dict[str, Any]): Active compile-time bindings.

    Returns:
        int: Maximum demand over all (resolvable) iterations.
    """
    from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
        _bind_loop_var,
        resolve_loop_bounds,
    )

    start, stop, step = resolve_loop_bounds(resolver, op, bindings)
    if (
        start is None
        or stop is None
        or step is None
        or step == 0
        or op.loop_var_value is None
    ):
        return _demand_in_operations(
            op.operations, inherited_controls, resolver, bindings
        )
    indexset = range(start, stop, step)
    try:
        iteration_count: int | None = len(indexset)
    except OverflowError:
        # A range longer than ``sys.maxsize`` cannot be counted (nor
        # precisely walked). Treat it like any over-cutoff loop and fall
        # back to a single conservative symbolic body walk.
        iteration_count = None
    if iteration_count == 0:
        return 0
    if iteration_count is None or iteration_count > _MAX_PRECISE_FOR_LOOP_ITERATIONS:
        return _demand_in_operations(
            op.operations, inherited_controls, resolver, bindings
        )

    demand = 0
    for i in indexset:
        loop_bindings = bindings.copy()
        _bind_loop_var(loop_bindings, op, i)
        demand = max(
            demand,
            _demand_in_operations(
                op.operations, inherited_controls, resolver, loop_bindings
            ),
        )
    return demand
