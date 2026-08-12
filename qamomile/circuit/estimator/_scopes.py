"""Build child resolver scopes for nested resource interpretation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, cast

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._constants import (
    _ONE,
    _ZERO,
)
from qamomile.circuit.estimator._quantum_values import _qubit_value_size
from qamomile.circuit.estimator._resolver import (
    ExprResolver,
    UnresolvedValueError,
)
from qamomile.circuit.estimator._resource_base import (
    _SYMPY_SIMPLIFICATION_ERRORS,
    ResourceExpr,
)
from qamomile.circuit.estimator._resource_expressions import (
    _and_conditions,
    _boolean_condition,
    _expr,
)
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import Operation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.types.primitives import (
    BitType,
    FloatType,
    UIntType,
)
from qamomile.circuit.ir.value import (
    ArrayValue,
    Value,
    ValueBase,
)
from qamomile.circuit.transpiler.block_parameter_binding import pair_block_operands


class _LocalBlock:
    """Provide a minimal operation container for nested resource scopes.

    Args:
        operations (list[Operation]): Operations visible in the nested scope.
    """

    __slots__ = ("operations",)

    def __init__(self, operations: list[Operation]) -> None:
        """Initialize a local operation container.

        Args:
            operations (list[Operation]): Operations visible in the nested
                scope.
        """
        self.operations = operations


def _typed_value_symbol(
    value: ValueBase,
    name: str,
    *,
    fresh: bool = False,
) -> sp.Symbol:
    """Create a symbolic scalar matching an IR value's domain.

    Args:
        value (ValueBase): IR value whose type supplies the symbol
            assumptions.
        name (str): Human-readable symbol name.
        fresh (bool): Whether to create an identity-distinct ``Dummy``
            instead of a same-name ``Symbol``. Defaults to False.

    Returns:
        sp.Symbol: A real symbol for Float, a nonnegative integer for
            UInt/Bit, or an unconstrained symbol for other value types.
    """
    factory = sp.Dummy if fresh else sp.Symbol
    if isinstance(value.type, FloatType):
        return factory(name, real=True)
    if isinstance(value.type, (BitType, UIntType)):
        return factory(name, integer=True, nonnegative=True)
    return factory(name)


def build_for_loop_scope(
    operation: ForOperation,
    resolver: ExprResolver,
) -> tuple[ExprResolver, ResourceExpr, ResourceExpr, ResourceExpr, sp.Symbol]:
    """Build resolver and symbolic bounds for a for loop.

    Args:
        operation (ForOperation): Loop operation.
        resolver (ExprResolver): Resolver for the enclosing scope.

    Returns:
        tuple[ExprResolver, ResourceExpr, ResourceExpr, ResourceExpr, sp.Symbol]:
        Child resolver, start, stop, step, and loop-variable symbol.

    Raises:
        UnresolvedValueError: If the loop has no UUID-bearing loop-variable
            value and would require unsafe display-name resolution.
    """
    if operation.loop_var_value is None:
        raise UnresolvedValueError(
            "?",
            "ForOperation is missing loop_var_value identity; display-name "
            "resolution is unsafe for nested loops.",
        )
    loop_symbol = sp.Dummy(operation.loop_var, integer=True)
    child = resolver.child_scope(
        inner_block=_LocalBlock(operation.operations),
        extra_context={operation.loop_var_value.uuid: loop_symbol},
        # Opaque resource models expose loop variables by their source-level
        # names. Expression resolution itself remains UUID-based above.
        extra_loop_vars={operation.loop_var: loop_symbol},
    )
    start = child.resolve(operation.operands[0])
    stop = child.resolve(operation.operands[1])
    step = (
        child.resolve(operation.operands[2]) if len(operation.operands) >= 3 else _ONE
    )
    return child, start, stop, step, loop_symbol


def _solve_affine_recurrence(
    *,
    yielded: sp.Expr,
    carry_symbol: sp.Symbol,
    other_carry_symbols: set[sp.Symbol],
    loop_symbol: sp.Symbol,
    start: ResourceExpr,
    step: ResourceExpr,
    iterations: ResourceExpr,
    init: sp.Expr,
) -> tuple[sp.Expr, sp.Expr] | None:
    """Solve one independent affine loop-carried recurrence.

    The supported recurrence is ``x[k + 1] = a*x[k] + b(k)`` where ``a``
    does not depend on the loop index or another carry. This covers counters,
    arithmetic accumulators, and geometric updates without requiring users to
    write separate resource equations for ordinary classical qkernel code.

    Args:
        yielded (sp.Expr): Expression yielded by one body iteration.
        carry_symbol (sp.Symbol): Symbol representing the incoming carry.
        other_carry_symbols (set[sp.Symbol]): Symbols for simultaneously carried
            values, which are rejected as coupled recurrences.
        loop_symbol (sp.Symbol): Symbol representing the Python loop value.
        start (ResourceExpr): Inclusive Python-range start.
        step (ResourceExpr): Python-range step.
        iterations (ResourceExpr): Symbolic number of loop iterations.
        init (sp.Expr): Carry value before iteration zero.

    Returns:
        tuple[sp.Expr, sp.Expr] | None: Carry at the current loop iteration and
        final carry after all iterations, or ``None`` when the recurrence is
        nonlinear, coupled, or has an index-dependent multiplier.
    """
    try:
        fixed_point_residual = sp.simplify(yielded.subs(carry_symbol, init) - init)
        coefficient = sp.simplify(sp.diff(yielded, carry_symbol))
        remainder = sp.simplify(yielded - coefficient * carry_symbol)
    except _SYMPY_SIMPLIFICATION_ERRORS:
        # Nested symbolic sums and piecewise observation values can exceed
        # SymPy's polynomial reduction domain. Such a failure means only that
        # this optional compact solver cannot prove an affine closed form; the
        # caller already has a conservative symbolic or concrete-replay path.
        return None
    if fixed_point_residual == _ZERO:
        return init, init
    if (
        carry_symbol in coefficient.free_symbols
        or carry_symbol in remainder.free_symbols
    ):
        return None
    if (coefficient.free_symbols | remainder.free_symbols) & other_carry_symbols:
        return None
    if loop_symbol in coefficient.free_symbols:
        return None

    summation_index = sp.Symbol(
        f"{loop_symbol}_previous", integer=True, nonnegative=True
    )
    previous_loop_value = start + summation_index * step
    previous_remainder = remainder.subs(loop_symbol, previous_loop_value)

    def value_after(count: sp.Expr) -> sp.Expr:
        """Return the recurrence value after ``count`` iterations.

        Args:
            count (sp.Expr): Number of completed iterations.

        Returns:
            sp.Expr: Closed-form recurrence value.
        """
        if coefficient == 1:
            accumulated = sp.Sum(
                previous_remainder,
                (summation_index, 0, count - 1),
            ).doit()
            return cast(sp.Expr, sp.simplify(init + accumulated))
        if coefficient == 0:
            last_remainder = remainder.subs(
                loop_symbol,
                start + (count - 1) * step,
            )
            return cast(
                sp.Expr,
                sp.Piecewise((init, sp.Eq(count, 0)), (last_remainder, True)),
            )
        accumulated = sp.Sum(
            coefficient ** (count - 1 - summation_index) * previous_remainder,
            (summation_index, 0, count - 1),
        ).doit()
        return cast(
            sp.Expr,
            sp.simplify(coefficient**count * init + accumulated),
        )

    try:
        completed_at_loop_value = sp.simplify((loop_symbol - start) / step)
        at_iteration = value_after(completed_at_loop_value)
        final = value_after(iterations)
    except _SYMPY_SIMPLIFICATION_ERRORS:
        return None
    nonfinite_atoms = (sp.nan, sp.zoo, sp.oo, -sp.oo)
    if at_iteration.has(*nonfinite_atoms) or final.has(*nonfinite_atoms):
        return None
    return at_iteration, final


def _invariant_identity_branch_guard(
    yielded: sp.Expr,
    *,
    carry_symbol: sp.Symbol,
    invariant_symbols: set[sp.Symbol],
) -> Boolean:
    """Return where an unsupported recurrence provably preserves its carry.

    Only ordered ``Piecewise`` branches whose value is exactly the incoming
    carry are considered. Every symbol in an effective guard must come from an
    enclosing capture or loop bound that the caller proved invariant. This
    positive proof avoids mistaking an unresolved body-local fallback symbol
    for an immutable input. Unresolved functions are rejected as well.

    Args:
        yielded (sp.Expr): Expression yielded by one loop iteration.
        carry_symbol (sp.Symbol): Symbol representing the incoming carry.
        invariant_symbols (set[sp.Symbol]): Symbols proven to come from
            immutable enclosing captures or loop bounds.

    Returns:
        Boolean: Ordered condition under which every iteration is an identity,
        or ``False`` when no such invariant branch can be proved.
    """
    if not isinstance(yielded, sp.Piecewise):
        return sp.false
    branches = cast(tuple[Any, ...], yielded.args)
    if not branches or branches[-1][1] is not sp.true:
        return sp.false
    remaining: Boolean = sp.true
    identity_guards: list[Boolean] = []
    for value, raw_guard in branches:
        branch_guard = _boolean_condition(cast(sp.Basic, raw_guard))
        effective_guard = _and_conditions(remaining, branch_guard)
        guard_symbols = cast(set[sp.Symbol], effective_guard.free_symbols)
        guard_is_invariant = not (
            not guard_symbols <= invariant_symbols
            or any(
                isinstance(node, AppliedUndef)
                for node in sp.preorder_traversal(effective_guard)
            )
        )
        preserves_carry = value == carry_symbol
        if guard_is_invariant and preserves_carry:
            identity_guards.append(effective_guard)
        remaining = _and_conditions(
            remaining,
            cast(Boolean, sp.Not(branch_guard)),
        )
        if remaining is sp.false:
            break
    return cast(Boolean, sp.Or(*identity_guards))


def _loop_invariant_symbols(
    resolver: ExprResolver,
    captures: Iterable[ValueBase],
    *,
    bound_expressions: Iterable[sp.Expr],
) -> set[sp.Symbol]:
    """Collect symbols proven invariant for one loop invocation.

    Region captures are explicit IR references to values defined outside the
    loop body. Scalar captures and captured array dimensions therefore remain
    fixed throughout that invocation, as do the already-resolved loop bounds.
    Container contents are deliberately not guessed from their printed form.

    Args:
        resolver (ExprResolver): Resolver for the enclosing loop scope.
        captures (Iterable[ValueBase]): Explicit outer-scope region captures.
        bound_expressions (Iterable[sp.Expr]): Resolved bounds or cardinality
            expressions fixed before the loop begins.

    Returns:
        set[sp.Symbol]: SymPy identities with an IR-backed invariance proof.
    """
    invariant: set[sp.Symbol] = set()

    def add_expression(expression: sp.Expr) -> None:
        """Add every scalar symbol in one proven-invariant expression.

        Args:
            expression (sp.Expr): Resolved enclosing expression.
        """
        invariant.update(
            symbol
            for symbol in expression.free_symbols
            if isinstance(symbol, sp.Symbol)
        )

    for expression in bound_expressions:
        add_expression(expression)
    for capture in captures:
        if isinstance(capture, ArrayValue):
            for dimension in capture.shape:
                add_expression(resolver.resolve(dimension))
        elif isinstance(capture, Value) and not capture.type.is_quantum():
            add_expression(resolver.resolve(capture))
    return invariant


def build_while_scope(
    operation: WhileOperation,
    resolver: ExprResolver,
    *,
    trip_count_name: str = "|while|",
) -> tuple[ExprResolver, sp.Symbol]:
    """Build resolver and symbolic trip count for a while loop.

    Args:
        operation (WhileOperation): While-loop operation.
        resolver (ExprResolver): Resolver for the enclosing scope.
        trip_count_name (str): Public symbolic trip-count name. Defaults to
            ``"|while|"`` for backward compatibility.

    Returns:
        tuple[ExprResolver, sp.Symbol]: Child resolver and ``|while|`` symbol.
    """
    child = resolver.child_scope(inner_block=_LocalBlock(operation.operations))
    return child, sp.Symbol(trip_count_name, integer=True, nonnegative=True)


def build_if_scopes(
    operation: IfOperation,
    resolver: ExprResolver,
) -> tuple[ExprResolver, ExprResolver]:
    """Build child resolvers for both branches of an if operation.

    Args:
        operation (IfOperation): Conditional operation.
        resolver (ExprResolver): Resolver for the enclosing scope.

    Returns:
        tuple[ExprResolver, ExprResolver]: True-branch and false-branch
        resolvers.
    """
    true_child = resolver.child_scope(
        inner_block=_LocalBlock(operation.true_operations)
    )
    false_child = resolver.child_scope(
        inner_block=_LocalBlock(operation.false_operations)
    )
    true_child.copy_array_context()
    false_child.copy_array_context()
    return true_child, false_child


def build_for_items_scope(
    operation: ForItemsOperation,
    resolver: ExprResolver,
) -> ExprResolver:
    """Build a child resolver for a for-items loop.

    Args:
        operation (ForItemsOperation): For-items operation.
        resolver (ExprResolver): Resolver for the enclosing scope.

    Returns:
        ExprResolver: Child resolver for the loop body.
    """
    return resolver.child_scope(inner_block=_LocalBlock(operation.operations))


def resolve_for_items_cardinality(operation: ForItemsOperation) -> ResourceExpr:
    """Return the symbolic cardinality of a for-items input dictionary.

    Args:
        operation (ForItemsOperation): For-items operation.

    Returns:
        ResourceExpr: Symbol of the form ``|dict_name|``.
    """
    dict_operand = operation.operands[0]
    if hasattr(dict_operand, "is_parameter") and dict_operand.is_parameter():
        dict_name = dict_operand.parameter_name() or dict_operand.name
    else:
        dict_name = dict_operand.name
    return sp.Symbol(f"|{dict_name}|", integer=True, nonnegative=True)


def _resolve_controlled_u(
    operation: ControlledUOperation,
    resolver: ExprResolver,
) -> tuple[ResourceExpr, int]:
    """Resolve controlled-U control and target arity.

    Args:
        operation (ControlledUOperation): Controlled-U operation.
        resolver (ExprResolver): Resolver for symbolic control counts.

    Returns:
        tuple[ResourceExpr, int]: Number of controls and concrete number of
        target values.
    """
    if operation.is_symbolic_num_controls:
        controls = resolver.resolve(operation.num_controls)
    else:
        controls = _expr(cast(int, operation.num_controls))

    if isinstance(operation.block, Block):
        targets = sum(
            1 for value in operation.block.input_values if value.type.is_quantum()
        )
    else:
        target_operands = getattr(operation, "target_operands", [])
        targets = len(target_operands) if target_operands else 1
    return controls, targets


def _controlled_u_child_resolver(
    operation: ControlledUOperation,
    resolver: ExprResolver,
) -> ExprResolver:
    """Build a resolver for a controlled-U body.

    Args:
        operation (ControlledUOperation): Controlled-U operation.
        resolver (ExprResolver): Call-site resolver.

    Returns:
        ExprResolver: Resolver scoped to the controlled body.
    """
    block = operation.block
    if not isinstance(block, Block):
        return resolver.child_scope(block)

    extra: dict[str, ResourceExpr] = {}
    classical_inputs: list[tuple[Value, Value]] = []
    array_inputs: list[tuple[ArrayValue, ArrayValue]] = []
    actual_operands = operation.body_operands
    for formal, actual in pair_block_operands(block, actual_operands):
        extra[formal.uuid] = resolver.resolve(actual)
        if (
            isinstance(formal, Value)
            and isinstance(actual, Value)
            and not formal.type.is_quantum()
            and not isinstance(formal, ArrayValue)
            and not isinstance(actual, ArrayValue)
        ):
            classical_inputs.append((formal, actual))
        if isinstance(formal, ArrayValue) and isinstance(actual, ArrayValue):
            if not formal.type.is_quantum():
                array_inputs.append((formal, actual))
            for formal_dim, actual_dim in zip(
                formal.shape,
                actual.shape,
                strict=True,
            ):
                extra[formal_dim.uuid] = resolver.resolve(actual_dim)

    child = resolver.isolated_scope(
        block,
        extra,
        structural_scope=resolver.call_structural_scope(operation, block),
    )
    for formal, actual in classical_inputs:
        child.bind_classical_fact(
            formal,
            resolver.resolve_classical_fact(actual),
        )
    for formal, actual in array_inputs:
        child.bind_call_array_input(
            block,
            formal,
            resolver.snapshot_array_state(actual),
        )
    return child


def _select_case_child_resolver(
    operation: SelectOperation,
    case_block: Block,
    resolver: ExprResolver,
) -> ExprResolver:
    """Build an independent resolver for one SELECT case body.

    Args:
        operation (SelectOperation): SELECT operation owning the case.
        case_block (Block): Case block whose formal inputs are mapped.
        resolver (ExprResolver): Resolver for the SELECT call site.

    Returns:
        ExprResolver: Resolver scoped exclusively to ``case_block`` with
        target, parameter, and array-shape formals bound to actual operands.
    """
    actual_operands = [*operation.target_operands, *operation.param_operands]
    extra: dict[str, ResourceExpr] = {}
    classical_inputs: list[tuple[Value, Value]] = []
    array_inputs: list[tuple[ArrayValue, ArrayValue]] = []
    for formal, actual in pair_block_operands(case_block, actual_operands):
        extra[formal.uuid] = resolver.resolve(actual)
        if (
            isinstance(formal, Value)
            and isinstance(actual, Value)
            and not formal.type.is_quantum()
            and not isinstance(formal, ArrayValue)
            and not isinstance(actual, ArrayValue)
        ):
            classical_inputs.append((formal, actual))
        if isinstance(formal, ArrayValue) and isinstance(actual, ArrayValue):
            if not formal.type.is_quantum():
                array_inputs.append((formal, actual))
            for formal_dim, actual_dim in zip(
                formal.shape,
                actual.shape,
                strict=True,
            ):
                extra[formal_dim.uuid] = resolver.resolve(actual_dim)

    child = resolver.isolated_scope(
        case_block,
        extra,
        structural_scope=resolver.call_structural_scope(operation, case_block),
    )
    for formal, actual in classical_inputs:
        child.bind_classical_fact(
            formal,
            resolver.resolve_classical_fact(actual),
        )
    for formal, actual in array_inputs:
        child.bind_call_array_input(
            case_block,
            formal,
            resolver.snapshot_array_state(actual),
        )
    return child


def _scalar_target_broadcast_factor(
    body: Block,
    target_operands: Sequence[Value],
    resolver: ExprResolver,
) -> ResourceExpr:
    """Return the broadcast count for a scalar body applied to one vector.

    Args:
        body (Block): Callable body with formal quantum inputs.
        target_operands (Sequence[Value]): Actual quantum targets supplied at
            the call site.
        resolver (ExprResolver): Resolver for symbolic target dimensions.

    Returns:
        ResourceExpr: Vector width when one scalar formal target is applied to
        one vector actual target, otherwise one.
    """
    if len(target_operands) != 1 or not isinstance(target_operands[0], ArrayValue):
        return _ONE
    quantum_inputs = [value for value in body.input_values if value.type.is_quantum()]
    if len(quantum_inputs) != 1 or isinstance(quantum_inputs[0], ArrayValue):
        return _ONE
    return _qubit_value_size(target_operands[0], resolver)


def _inverse_block_child_resolver(
    operation: InverseBlockOperation,
    resolver: ExprResolver,
) -> ExprResolver:
    """Build a resolver for an inverse implementation block.

    The inverse operation stores operands in call-site layout: quantum targets
    first, then classical parameters. The implementation block may declare
    classical formals before quantum formals, so this helper maps by formal
    type rather than by raw position.

    Args:
        operation (InverseBlockOperation): Inverse operation to resolve.
        resolver (ExprResolver): Resolver for the call site.

    Returns:
        ExprResolver: Resolver scoped to the inverse implementation.
    """
    impl = operation.implementation_block
    if not isinstance(impl, Block):
        return resolver.child_scope(impl)

    extra: dict[str, ResourceExpr] = {}
    classical_inputs: list[tuple[Value, Value]] = []
    array_inputs: list[tuple[ArrayValue, ArrayValue]] = []
    operands = [*operation.target_qubits, *operation.parameters]
    for formal, actual in pair_block_operands(impl, operands):
        extra[formal.uuid] = resolver.resolve(actual)
        if (
            isinstance(formal, Value)
            and isinstance(actual, Value)
            and not formal.type.is_quantum()
            and not isinstance(formal, ArrayValue)
            and not isinstance(actual, ArrayValue)
        ):
            classical_inputs.append((formal, actual))
        if isinstance(formal, ArrayValue) and isinstance(actual, ArrayValue):
            if not formal.type.is_quantum():
                array_inputs.append((formal, actual))
            for formal_dim, actual_dim in zip(
                formal.shape,
                actual.shape,
                strict=True,
            ):
                extra[formal_dim.uuid] = resolver.resolve(actual_dim)

    child = resolver.isolated_scope(
        impl,
        extra,
        structural_scope=resolver.call_structural_scope(operation, impl),
    )
    for formal, actual in classical_inputs:
        child.bind_classical_fact(
            formal,
            resolver.resolve_classical_fact(actual),
        )
    for formal, actual in array_inputs:
        child.bind_call_array_input(
            impl,
            formal,
            resolver.snapshot_array_state(actual),
        )
    return child
