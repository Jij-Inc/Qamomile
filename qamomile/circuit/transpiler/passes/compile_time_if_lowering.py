"""Compile-time IfOperation lowering pass.

Lowers compile-time resolvable ``IfOperation``s before the segmentation pass,
replacing them with selected-branch operations and substituting merge outputs
with selected-branch values throughout the block.

This prevents ``SegmentationPass`` from seeing classical-only compile-time
``IfOperation``s that would otherwise split quantum segments.
"""

from __future__ import annotations

import dataclasses
import struct
from collections.abc import Sequence
from typing import Any, cast

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import (
    Operation,
    ReleaseSliceViewOperation,
    SliceArrayOperation,
)
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    CompOp,
    CondOp,
    NotOp,
)
from qamomile.circuit.ir.operation.callable import (
    CallTransform,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    RegionArg,
    WhileOperation,
    genuine_input_values,
    validate_region_args,
)
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    MeasureVectorOperation,
    SymbolicControlledU,
)
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.operation import OperationKind, QInitOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.types.primitives import FloatType, UIntType
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    Value,
    ValueBase,
    ValueLike,
    collect_value_like_uuids,
    resolve_root_array_index,
)
from qamomile.circuit.transpiler.block_parameter_binding import (
    pair_block_parameter_operands,
)
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.value_resolver import (
    ValueResolver as UnifiedValueResolver,
)

from . import Pass
from .emit_support import resolve_if_condition
from .eval_utils import FoldPolicy, fold_classical_op
from .value_mapping import ValueSubstitutor

_MAX_STATIC_CARRY_ITERATIONS = 10_000

# Dead-result pruning is deliberately fail-closed. Operations outside this
# tuple may have observable effects even when none of their SSA results remain
# live (measurement is the canonical example), so only the scalar expression
# nodes whose evaluation is known to be pure may be removed here.
_PURE_CLASSICAL_EXPRESSION_TYPES = (BinOp, CompOp, CondOp, NotOp)


def _same_exact_typed_constant(left: Value, right: Value) -> bool:
    """Return whether two scalar Values carry the same exact typed constant.

    Args:
        left (Value): First scalar Value to compare.
        right (Value): Second scalar Value to compare.

    Returns:
        bool: True only for constants of the same IR type and Python type with
            equal value representations. Floating-point comparison preserves
            the sign of zero and the payload bits of NaNs.
    """
    if isinstance(left, ArrayValue) or isinstance(right, ArrayValue):
        return False
    if left.type != right.type or not left.is_constant() or not right.is_constant():
        return False
    left_value = left.get_const()
    right_value = right.get_const()
    if type(left_value) is not type(right_value):
        return False
    if isinstance(left_value, float):
        return struct.pack("!d", left_value) == struct.pack("!d", right_value)
    return bool(left_value == right_value)


def _is_identity_region_arg(region_arg: RegionArg) -> bool:
    """Return whether a region argument always preserves its initializer.

    Args:
        region_arg (RegionArg): Region argument to classify.

    Returns:
        bool: True when the yielded value is the body formal, the initializer,
            or the same exact typed compile-time scalar as the initializer.
    """
    if region_arg.yielded.uuid in {
        region_arg.block_arg.uuid,
        region_arg.init.uuid,
    }:
        return True
    return _same_exact_typed_constant(region_arg.init, region_arg.yielded)


def resolve_compile_time_condition(
    condition: Any,
    concrete_values: dict[str, Any],
    bindings: dict[str, Any],
) -> bool | None:
    """Resolve an ``IfOperation`` condition to a compile-time bool.

    Single source of truth for classifying an if-condition as
    compile-time taken / dead / runtime.  Used by
    :class:`CompileTimeIfLoweringPass` to decide which branches to lower
    and by ``reject_self_referential_loop_stores`` to prune the same
    branches from its scan — both callers must agree on the
    classification, so they share this function.

    Tries ``resolve_if_condition`` first (plain Python values, constant
    Values, direct UUID / parameter-provenance bindings), then falls back to the
    accumulated ``concrete_values`` map for expression-derived
    conditions (``CompOp`` / ``CondOp`` / ``NotOp`` / ``BinOp`` chains
    evaluated by :func:`evaluate_classical_op_concrete`).

    Args:
        condition (Any): The condition operand.  May be a plain Python
            value or a ``Value``.
        concrete_values (dict[str, Any]): UUID-keyed map of concrete
            classical-op results accumulated in program order.
        bindings (dict[str, Any]): Compile-time parameter bindings.

    Returns:
        bool | None: The condition's compile-time truth value, or
            ``None`` when it is not compile-time resolvable (a runtime
            condition).
    """
    resolved = resolve_if_condition(condition, bindings)
    if resolved is not None:
        return resolved
    if hasattr(condition, "uuid") and condition.uuid in concrete_values:
        return bool(concrete_values[condition.uuid])
    return None


def evaluate_classical_op_concrete(
    op: Operation,
    concrete_values: dict[str, Any],
    bindings: dict[str, Any],
) -> None:
    """Try to evaluate a classical op and record its concrete result.

    Supported operation types:

    - ``CompOp``  — comparison (==, !=, <, <=, >, >=)
    - ``CondOp``  — logical connective (and, or)
    - ``NotOp``   — logical negation
    - ``BinOp``   — arithmetic (+, -, *, /, //, %)

    Delegates the actual fold to ``fold_classical_op`` under the
    ``COMPILE_TIME`` policy, which bypasses the runtime-parameter
    guard: everything in ``bindings`` is treated as a real
    compile-time value.  Other operation types are silently ignored.
    If evaluation fails, nothing is recorded and downstream
    ``IfOperation``s referencing the result remain unresolved.

    Args:
        op (Operation): The operation to evaluate.
        concrete_values (dict[str, Any]): UUID-keyed map of concrete
            results; the op's result is recorded here on success.
            Updated in place.
        bindings (dict[str, Any]): Compile-time parameter bindings used
            to resolve operands.
    """
    if not isinstance(op, (CompOp, CondOp, NotOp, BinOp)):
        return
    if not op.results:
        return
    result = fold_classical_op(
        op,
        lambda v: UnifiedValueResolver(
            context=concrete_values, bindings=bindings
        ).resolve(v),
        parameters=set(),
        policy=FoldPolicy.COMPILE_TIME,
    )
    if result is not None:
        concrete_values[op.results[0].uuid] = result


def _array_carrier_keys(
    source: ValueBase,
    num_bits: int,
) -> tuple[list[str], list[str]] | None:
    """Build root-space carrier keys for a selected array source.

    Delegates the slice-chain folding to
    :func:`~qamomile.circuit.ir.value.resolve_root_array_index` so the keys
    stay consistent with every other carrier-key producer and resolver.

    Args:
        source (ValueBase): Selected source value after merge substitution.
            Array sources may be plain arrays or strided views.
        num_bits (int): Number of QFixed carrier bits to build.

    Returns:
        tuple[list[str], list[str]] | None: Parallel UUID and logical-id
            carrier keys when ``source`` is an array with constant slice
            metadata, otherwise ``None``.
    """
    if not isinstance(source, ArrayValue):
        return None

    uuids: list[str] = []
    logical_ids: list[str] = []
    for i in range(num_bits):
        resolved = resolve_root_array_index(source, i)
        if resolved is None:
            return None
        root, root_index = resolved
        uuids.append(f"{root.uuid}_{root_index}")
        logical_ids.append(f"{root.logical_id}_{root_index}")
    return uuids, logical_ids


class CompileTimeIfLoweringPass(Pass[Block, Block]):
    """Lowers compile-time resolvable IfOperations before separation.

    After constant folding, some ``IfOperation`` conditions are statically
    known but remain as control-flow nodes.  ``SegmentationPass`` treats them
    as segment boundaries, causing ``MultipleQuantumSegmentsError`` for
    classical-only compile-time ``if`` after quantum init.

    This pass:

    1. Evaluates conditions including expression-derived ones
       (``CompOp``, ``CondOp``, ``NotOp`` chains).
    2. Replaces resolved ``IfOperation``s with selected-branch operations.
    3. Substitutes merge output UUIDs with selected-branch values in all
       subsequent operations and block outputs.
    """

    def __init__(
        self,
        bindings: dict[str, Any] | None = None,
        *,
        _under_controlled_unitary: bool = False,
        _active_block_ids: frozenset[int] | None = None,
    ):
        """Initialize compile-time if lowering state.

        Args:
            bindings (dict[str, Any] | None): Compile-time bindings visible in
                the current block. Defaults to no bindings.
            _under_controlled_unitary (bool): Internal context flag indicating
                that boxed callables encountered here will be decomposed by the
                controlled emission walker and therefore need their owned
                bodies lowered too. Defaults to ``False``.
            _active_block_ids (frozenset[int] | None): Internal recursion-path
                guard for operation-owned blocks. Defaults to an empty set.
        """
        self._bindings = bindings or {}
        self._under_controlled_unitary = _under_controlled_unitary
        self._active_block_ids = _active_block_ids or frozenset()
        self._static_replay_remaining = _MAX_STATIC_CARRY_ITERATIONS

    @property
    def name(self) -> str:
        return "compile_time_if_lowering"

    def run(self, input: Block) -> Block:
        """Lower every compile-time resolvable IfOperation in the block.

        Args:
            input (Block): Block to lower. Must be ``TRACED``, ``AFFINE``, or
                ``HIERARCHICAL``. ``TRACED`` is accepted so the circuit drawer
                can resolve bound/constant ``if`` conditions on a freshly
                traced block before the transpiler pipeline runs; ``HIERARCHICAL``
                is accepted during the self-recursion unroll loop. Surviving
                inline callable invocations are passed through untouched in both
                cases.

        Returns:
            Block: New block with compile-time ``if``s replaced by their
                selected-branch operations and merge outputs substituted. The
                input's ``BlockKind`` is preserved.

        Raises:
            ValidationError: If ``input.kind`` is ``ANALYZED`` (the pass must
                run before dependency analysis commits to a representation), or
                if a compile-time branch selects a symbolic-bound slice-view
                cast source whose root index space is not resolvable.
        """
        # The replay limit is scoped to one pass invocation. Public pass
        # instances may be reused, and prior runs must not consume budget from
        # a later, independent block transformation.
        self._static_replay_remaining = _MAX_STATIC_CARRY_ITERATIONS
        if input.kind not in (
            BlockKind.TRACED,
            BlockKind.AFFINE,
            BlockKind.HIERARCHICAL,
        ):
            raise ValidationError(
                f"CompileTimeIfLoweringPass expects TRACED, AFFINE, or "
                f"HIERARCHICAL block, got {input.kind}",
            )
        # Track concrete values produced by classical ops.
        # Maps UUID → concrete Python value (int, float, bool).
        concrete_values: dict[str, Any] = {}

        # Seed concrete_values from UUID-keyed bindings entries. Name-keyed
        # user bindings also land here but are dead entries: lookups against
        # ``concrete_values`` are UUID-keyed, and a display name never
        # collides with a UUID string. Name-based resolution happens only via
        # parameter provenance in ``_try_seed_value`` below.
        for key, val in self._bindings.items():
            concrete_values[key] = val

        # Seed from block input_values that are constants or bound.
        for iv in input.input_values:
            self._try_seed_value(iv, concrete_values)

        # Process operations, lowering compile-time ifs.
        new_ops, merge_subst, dead_uuids = self._lower_operations(
            input.operations, concrete_values
        )

        # Apply merge substitution to block output_values.
        new_outputs = self._substitute_output_values(input.output_values, merge_subst)

        # Remove dead operations whose results are only used by lowered ifs.
        if dead_uuids:
            new_ops = self._eliminate_dead_ops(new_ops, dead_uuids, new_outputs)

        return dataclasses.replace(
            input,
            operations=new_ops,
            output_values=new_outputs,
        )

    # ------------------------------------------------------------------
    # Condition evaluation
    # ------------------------------------------------------------------

    def _try_resolve_condition(
        self,
        condition: Any,
        concrete_values: dict[str, Any],
    ) -> bool | None:
        """Resolve an IfOperation condition to a compile-time bool.

        Thin wrapper for the module-level
        :func:`resolve_compile_time_condition` bound to this pass's
        bindings, so external callers (the self-referential loop-store
        check) classify conditions exactly the way this pass does.

        Args:
            condition (Any): The condition operand to resolve.
            concrete_values (dict[str, Any]): Accumulated concrete
                classical-op results, keyed by UUID.

        Returns:
            bool | None: The compile-time truth value, or ``None`` when
                the condition is runtime.
        """
        return resolve_compile_time_condition(
            condition, concrete_values, self._bindings
        )

    def _try_evaluate_classical_op(
        self,
        op: Operation,
        concrete_values: dict[str, Any],
    ) -> None:
        """Try to evaluate a classical op and record its result.

        Thin wrapper for the module-level
        :func:`evaluate_classical_op_concrete` bound to this pass's
        bindings.  At this stage there are no runtime parameters in the
        concrete values map; everything in ``self._bindings`` is treated
        as a real compile-time value.

        Args:
            op (Operation): The operation to evaluate.
            concrete_values (dict[str, Any]): UUID-keyed concrete-result
                map, updated in place on success.
        """
        evaluate_classical_op_concrete(op, concrete_values, self._bindings)

    # ------------------------------------------------------------------
    # Operation lowering
    # ------------------------------------------------------------------

    def _lower_operations(
        self,
        operations: list[Operation],
        concrete_values: dict[str, Any],
    ) -> tuple[list[Operation], dict[str, ValueBase], set[str]]:
        """Lower compile-time control flow within an operation list.

        Args:
            operations (list[Operation]): Operations to lower in program
                order, including nested control-flow operations.
            concrete_values (dict[str, Any]): Mutable UUID-keyed concrete
                value map used to evaluate conditions and static carries.

        Returns:
            tuple[list[Operation], dict[str, ValueBase], set[str]]: Lowered
                operations, transitive result substitutions, and UUIDs that
                are candidates for dead-operation elimination.

        Raises:
            ValidationError: If substitution selects a slice-view cast whose
                root index space remains symbolic.
        """
        new_ops: list[Operation] = []
        merge_subst: dict[str, ValueBase] = {}
        dead_uuids: set[str] = set()

        for op in operations:
            # First apply any accumulated merge substitutions to this op.
            op = self._apply_substitution(op, merge_subst)
            if isinstance(op, (ForOperation, ForItemsOperation, WhileOperation)):
                try:
                    validate_region_args(op)
                except ValueError as error:
                    raise ValidationError(str(error)) from error

            # Track classical op results for expression-aware evaluation.
            self._try_evaluate_classical_op(op, concrete_values)

            if isinstance(op, IfOperation):
                resolved = self._try_resolve_condition(op.condition, concrete_values)
                if resolved is not None:
                    # Mark the condition value as dead (its producer can
                    # be removed if no other op uses it).
                    if hasattr(op.condition, "uuid"):
                        dead_uuids.add(op.condition.uuid)

                    # Build merge substitution map.
                    for merge in op.iter_merges():
                        merge_subst[merge.result.uuid] = merge.select(resolved)
                        # Degenerate-loop flattening may hoist the losing
                        # source's producer beside this if. Mark it as a dead
                        # candidate; the final use check preserves other reads.
                        dead_uuids.add(merge.select(not resolved).uuid)

                    # Inline selected branch operations.
                    selected_ops = (
                        op.true_operations if resolved else op.false_operations
                    )
                    # Recursively lower any nested compile-time ifs in the
                    # selected branch.
                    lowered, nested_subst, nested_dead = self._lower_operations(
                        selected_ops, concrete_values
                    )
                    merge_subst.update(nested_subst)
                    dead_uuids.update(nested_dead)
                    new_ops.extend(lowered)
                    continue

            # For non-lowered IfOperations, recurse into branches.
            if isinstance(op, IfOperation):
                lowered_true, subst_true, dead_true = self._lower_operations(
                    op.true_operations, dict(concrete_values)
                )
                lowered_false, subst_false, dead_false = self._lower_operations(
                    op.false_operations, dict(concrete_values)
                )
                # Apply nested substitutions to the branch-merge yields:
                # a yield may reference the merged output of a
                # compile-time if that was just lowered inside a branch.
                nested_subst_if = {**subst_true, **subst_false}
                new_true_yields = op.true_yields
                new_false_yields = op.false_yields
                if nested_subst_if:
                    nested_substitutor = ValueSubstitutor(
                        nested_subst_if, transitive=True
                    )
                    new_true_yields = [
                        cast(Value, nested_substitutor.substitute_value(v))
                        for v in op.true_yields
                    ]
                    new_false_yields = [
                        cast(Value, nested_substitutor.substitute_value(v))
                        for v in op.false_yields
                    ]
                op = dataclasses.replace(
                    op,
                    true_operations=lowered_true,
                    false_operations=lowered_false,
                    true_yields=new_true_yields,
                    false_yields=new_false_yields,
                )
                op, identity_subst = self._eliminate_identity_if_merges(op)
                merge_subst.update(identity_subst)
                dead_uuids.update(dead_true)
                dead_uuids.update(dead_false)

            elif isinstance(op, HasNestedOps):
                # Generic recursion for For/ForItems/While bodies.
                new_lists: list[list[Operation]] = []
                for body in op.nested_op_lists():
                    lowered_body, nested_subst, nested_dead = self._lower_operations(
                        body, dict(concrete_values)
                    )
                    new_lists.append(lowered_body)
                    merge_subst.update(nested_subst)
                    dead_uuids.update(nested_dead)
                op = op.rebuild_nested(new_lists)
                # The substitution applied at the top of the loop predates
                # the nested lowering, so merge outputs erased INSIDE this
                # op's body are still referenced by its rebind records
                # (and possibly operands). Re-apply with the updated map.
                op = self._apply_substitution(op, merge_subst)
                if isinstance(op, (ForOperation, ForItemsOperation, WhileOperation)):
                    op, identity_subst = self._eliminate_identity_loop_carries(op)
                    merge_subst.update(identity_subst)
                if isinstance(op, ForOperation):
                    degenerate = self._lower_degenerate_for(op)
                    if degenerate is not None:
                        replacement_ops, result_subst = degenerate
                        merge_subst.update(result_subst)
                        if replacement_ops is None:
                            # A zero-trip quantum loop remains boxed solely
                            # for resource-lineage mapping. Its classical
                            # carry results still have zero-trip SSA semantics:
                            # every post-loop read observes the initializer.
                            new_ops.append(op)
                            continue
                        lowered_body, body_subst, body_dead = self._lower_operations(
                            replacement_ops,
                            concrete_values,
                        )
                        merge_subst.update(body_subst)
                        dead_uuids.update(body_dead)
                        new_ops.extend(lowered_body)
                        continue
                    static_subst, eliminate_loop = self._evaluate_static_for_carries(
                        op, concrete_values
                    )
                    merge_subst.update(static_subst)
                    if eliminate_loop:
                        continue
                if isinstance(op, ForItemsOperation):
                    degenerate_items = self._lower_degenerate_for_items(op)
                    if degenerate_items is not None:
                        replacement_ops, result_subst = degenerate_items
                        merge_subst.update(result_subst)
                        if replacement_ops is None:
                            # Mirror the range-loop path above: keep quantum
                            # resource lineage boxed while aliasing every
                            # zero-entry carry result to its initializer.
                            new_ops.append(op)
                            continue
                        lowered_body, body_subst, body_dead = self._lower_operations(
                            replacement_ops,
                            concrete_values,
                        )
                        merge_subst.update(body_subst)
                        dead_uuids.update(body_dead)
                        new_ops.extend(lowered_body)
                        continue
                    static_subst, eliminate_loop = (
                        self._evaluate_static_for_items_carries(
                            op,
                            concrete_values,
                        )
                    )
                    merge_subst.update(static_subst)
                    if eliminate_loop:
                        continue

            elif isinstance(op, SelectOperation):
                # A SELECT case owns a fresh formal namespace just like a
                # controlled or boxed callable body. Bind only the case's
                # declared classical/object inputs from this SELECT's actual
                # operands; never apply outer UUID/name maps directly to the
                # case Block.
                op = dataclasses.replace(
                    op,
                    case_blocks=[
                        self._lower_operation_owned_block(
                            case_block,
                            op.param_operands,
                            concrete_values,
                        )
                        for case_block in op.case_blocks
                    ],
                )

            elif isinstance(op, ControlledUOperation) and op.block is not None:
                # A controlled-U carries its unitary as a nested ``block``
                # with its OWN value namespace (fresh input-value UUIDs), not
                # as HasNestedOps children, so the generic recursion above
                # never reaches it — and applying the outer merge /
                # concrete-value maps to it would be unsound, since those are
                # keyed by outer-namespace UUIDs. Recurse with a binding scope
                # seeded only from the controlled operands so a compile-time
                # ``if sel == k`` in the body is resolved here, before emit,
                # uniformly for every backend. Without this the comparison
                # survives as an unresolved ``CompOp`` inside the controlled
                # target, which QURI Parts / CUDA-Q reject at emit.
                op = self._lower_controlled_block(op, concrete_values)

            elif isinstance(op, InvokeOperation) and (
                op.transform is CallTransform.CONTROLLED
                or self._under_controlled_unitary
            ):
                # A boxed callable reached under structural control also owns
                # fresh-namespace bodies. Lower every body that controlled
                # emission may select for this invocation's transform.
                op = self._lower_invoke_bodies(op, concrete_values)

            elif (
                isinstance(op, InverseBlockOperation) and self._under_controlled_unitary
            ):
                op = self._lower_inverse_block(op, concrete_values)

            new_ops.append(op)

        return new_ops, merge_subst, dead_uuids

    def _lower_controlled_block(
        self,
        op: ControlledUOperation,
        concrete_values: dict[str, Any],
    ) -> ControlledUOperation:
        """Lower compile-time ifs inside a controlled-U's nested unitary block.

        The block is a self-contained unitary with its own value namespace,
        so it is processed by a fresh pass instance whose bindings are seeded
        ONLY from the controlled operands — never the outer bindings, which
        could collide by parameter name and mis-seed an inner parameter. A
        compile-time ``if`` whose condition depends on a bound classical
        operand (e.g. ``if sel == 0`` with ``sel`` bound) is resolved here so
        every backend sees an already-selected branch rather than an
        unresolved ``CompOp`` inside the controlled target.

        Args:
            op (ControlledUOperation): The controlled-U operation to descend
                into. Its ``block`` must not be ``None``.
            concrete_values (dict[str, Any]): Outer-namespace concrete values
                used to resolve the controlled operands to compile-time
                constants.

        Returns:
            ControlledUOperation: ``op`` with its ``block`` field replaced by
                the recursively lowered block.

        Raises:
            AssertionError: If ``op.block`` is ``None``. The sole caller
                guards this with ``op.block is not None``, so a failure here
                signals a broken caller contract rather than user error.
            ValidationError: If the inner block has an unsupported kind, or
                if lowering selects a ``CastOperation`` source that is a
                symbolic-bound slice view.
        """
        block = op.block
        assert block is not None
        new_block = self._lower_operation_owned_block(
            block,
            op.param_operands,
            concrete_values,
        )
        return dataclasses.replace(op, block=new_block)

    def _lower_invoke_bodies(
        self,
        op: InvokeOperation,
        concrete_values: dict[str, Any],
    ) -> InvokeOperation:
        """Lower compile-time ifs in bodies selected for an invocation.

        A callable definition can provide a default body plus backend- or
        strategy-specific implementations. Emission first looks for a body
        whose transform matches the invocation and otherwise falls back to the
        default body, so both sets must be lowered before backend selection.
        The definition and implementations are copied per call site to avoid
        mutating a shared callable body when the same callable is invoked with
        different compile-time arguments.

        Args:
            op (InvokeOperation): Boxed invocation reached under structural
                control, or a controlled invocation at the outer level.
            concrete_values (dict[str, Any]): Outer-namespace concrete values
                used to resolve the invocation's parameter operands.

        Returns:
            InvokeOperation: Invocation with every selectable IR body replaced
            by an independently lowered copy. Bodyless opaque invocations are
            returned unchanged.

        Raises:
            ValidationError: If a selected nested body has an unsupported
                block kind or contains a symbolic-bound slice-view cast source
                selected by compile-time lowering.
        """
        definition = op.definition
        if definition is None:
            return op

        changed = False
        new_body = definition.body
        if new_body is not None:
            new_body = self._lower_operation_owned_block(
                new_body,
                op.parameters,
                concrete_values,
            )
            changed = True

        new_implementations = []
        for implementation in definition.implementations:
            new_implementation = implementation
            if (
                implementation.transform is op.transform
                and implementation.body is not None
            ):
                new_implementation = dataclasses.replace(
                    implementation,
                    body=self._lower_operation_owned_block(
                        implementation.body,
                        op.parameters,
                        concrete_values,
                    ),
                )
                changed = True
            new_implementations.append(new_implementation)

        if not changed:
            return op
        new_definition = dataclasses.replace(
            definition,
            body=new_body,
            implementations=new_implementations,
        )
        return dataclasses.replace(op, definition=new_definition)

    def _lower_inverse_block(
        self,
        op: InverseBlockOperation,
        concrete_values: dict[str, Any],
    ) -> InverseBlockOperation:
        """Lower owned inverse blocks reached by controlled decomposition.

        Args:
            op (InverseBlockOperation): Inverse operation whose forward and
                fallback bodies may be selected during controlled emission.
            concrete_values (dict[str, Any]): Outer-namespace concrete values
                used to resolve the inverse parameter operands.

        Returns:
            InverseBlockOperation: Copy with each available body independently
            lowered in its own namespace.

        Raises:
            ValidationError: If a nested block has an unsupported kind or
                lowering selects a symbolic-bound slice-view cast source.
        """
        source_block = op.source_block
        if source_block is not None:
            source_block = self._lower_operation_owned_block(
                source_block,
                op.parameters,
                concrete_values,
            )
        implementation_block = op.implementation_block
        if implementation_block is not None:
            implementation_block = self._lower_operation_owned_block(
                implementation_block,
                op.parameters,
                concrete_values,
            )
        return dataclasses.replace(
            op,
            source_block=source_block,
            implementation_block=implementation_block,
        )

    def _lower_operation_owned_block(
        self,
        block: Block,
        param_operands: Sequence[ValueBase],
        concrete_values: dict[str, Any],
    ) -> Block:
        """Lower a fresh-namespace block using its call-site parameters.

        Pairs the block's classical/object inputs with the operation's
        parameter operands in signature order and resolves each operand via
        the outer concrete-value map. Resolved values are keyed by inner input
        UUID only. The fresh map is seeded exclusively from this call's
        operands, so an outer parameter with the same display name cannot
        mis-seed the inner parameter.

        The inner UUID key is what carries the value into the fresh inner
        pass: :meth:`run` copies it into that pass's
        ``concrete_values`` (the UUID-keyed context map :class:`ValueResolver`
        consults first), so a scalar comparison condition resolves by context,
        and the resolver's compile-time-array path keys a bound ``Vector``
        container by the same UUID. Operands that do not resolve are skipped,
        leaving the corresponding inner parameter symbolic so its compile-time
        ``if`` stays unlowered — the pre-existing loud outcome, never a silent
        miscompile.

        Args:
            block (Block): Operation-owned block with a fresh value namespace.
            param_operands (Sequence[ValueBase]): Classical/object call-site
                operands in the wrapped block's signature order.
            concrete_values (dict[str, Any]): Outer-namespace concrete values
                used to resolve the operands to compile-time constants.

        Returns:
            Block: Independently lowered block whose value namespace remains
            local to the operation that owns it.

        Raises:
            ValidationError: If the nested block has an unsupported kind or
                lowering selects a symbolic-bound slice-view cast source.
        """
        block_id = id(block)
        if block_id in self._active_block_ids:
            return block

        inner_bindings: dict[str, Any] = {}
        resolver = UnifiedValueResolver(
            context=concrete_values, bindings=self._bindings
        )
        for inner_iv, operand in pair_block_parameter_operands(block, param_operands):
            resolved = resolver.resolve(operand)
            if resolved is None:
                continue
            inner_bindings[inner_iv.uuid] = resolved
        return CompileTimeIfLoweringPass(
            inner_bindings,
            _under_controlled_unitary=True,
            _active_block_ids=self._active_block_ids | {block_id},
        ).run(block)

    # ------------------------------------------------------------------
    # Substitution
    # ------------------------------------------------------------------

    @staticmethod
    def _eliminate_identity_if_merges(
        op: IfOperation,
    ) -> tuple[IfOperation, dict[str, Value]]:
        """Remove runtime-if merge slots whose branches yield one Value.

        Such a merge is an SSA identity even though the branch may still
        contain quantum control flow. Keeping it can strand a classical
        result inside the quantum segment where no runtime executor
        materializes it.

        Args:
            op (IfOperation): Runtime if after nested substitutions.

        Returns:
            tuple[IfOperation, dict[str, Value]]: The compacted if and
                substitutions from removed results to their shared source.

        Raises:
            RuntimeError: If the if's merge storage is inconsistent.
        """
        merges = list(op.iter_merges())
        identity_indices = {
            merge.index
            for merge in merges
            if (
                merge.true_value.uuid == merge.false_value.uuid
                or _same_exact_typed_constant(
                    merge.true_value,
                    merge.false_value,
                )
            )
            and merge.result.type.is_classical()
        }
        if not identity_indices:
            return op, {}
        keep = [i for i in range(len(merges)) if i not in identity_indices]
        rewritten = dataclasses.replace(
            op,
            true_yields=[op.true_yields[i] for i in keep],
            false_yields=[op.false_yields[i] for i in keep],
            results=[op.results[i] for i in keep],
        )
        substitutions = {
            merge.result.uuid: merge.true_value
            for merge in merges
            if merge.index in identity_indices
        }
        return rewritten, substitutions

    def _eliminate_identity_loop_carries(
        self,
        op: ForOperation | ForItemsOperation | WhileOperation,
    ) -> tuple[ForOperation | ForItemsOperation | WhileOperation, dict[str, Value]]:
        """Remove carry slots whose folded body yields its input unchanged.

        A compile-time-dead update can leave a structural carry whose
        ``yielded`` value aliases its ``block_arg``. Reassigning the same typed
        compile-time scalar as the ``init`` is equivalent even when the
        frontend materializes a fresh constant UUID. Such a slot is
        semantically loop-invariant: body reads can use the entry value
        directly and the post-loop result is the same entry value, including
        on a zero-trip path. Removing it is required for runtime ``while``
        loops, whose backends cannot thread arbitrary classical carry slots.

        Args:
            op (ForOperation | ForItemsOperation | WhileOperation): Rebuilt
                loop after nested compile-time-if substitutions.

        Returns:
            tuple[ForOperation | ForItemsOperation | WhileOperation,
            dict[str, Value]]: The rewritten loop and substitutions from
                removed result UUIDs to their loop-entry Values.
        """
        carries = list(op.region_args)
        identity_indices = {
            index
            for index, carry in enumerate(carries)
            if _is_identity_region_arg(carry)
        }
        if not identity_indices:
            return op, {}

        body_subst: dict[str, ValueBase] = {
            carry.block_arg.uuid: carry.init
            for index, carry in enumerate(carries)
            if index in identity_indices
        }
        rewritten_op = self._apply_substitution(op, body_subst)
        assert isinstance(
            rewritten_op, (ForOperation, ForItemsOperation, WhileOperation)
        )
        keep = [i for i in range(len(carries)) if i not in identity_indices]
        rewritten = dataclasses.replace(
            rewritten_op,
            region_args=tuple(rewritten_op.region_args[i] for i in keep),
            results=[rewritten_op.results[i] for i in keep],
        )
        result_subst = {
            carry.result.uuid: carry.init
            for index, carry in enumerate(carries)
            if index in identity_indices
        }
        return rewritten, result_subst

    def _lower_degenerate_for(
        self,
        op: ForOperation,
    ) -> tuple[list[Operation] | None, dict[str, Value]] | None:
        """Lower a compile-time zero- or one-trip range loop to SSA aliases.

        A one-trip loop has no back-edge: its body formal reads the entry
        value, and its final result is the single body yield. Flattening this
        degenerate case prevents dependency analysis and emit from treating a
        dead final yield as though a second iteration consumed it. A zero-trip
        loop similarly aliases each result directly to its entry value.

        Args:
            op (ForOperation): Loop after nested compile-time lowering.

        Returns:
            tuple[list[Operation] | None, dict[str, Value]] | None:
                Replacement body operations and result substitutions for
                zero/one trip. The operation list is ``None`` when a zero-trip
                quantum loop must remain boxed for resource-lineage mapping;
                its result substitutions still alias carries to their entry
                values. Returns ``None`` when the bounds are unresolved or
                have two or more iterations.

        Raises:
            RuntimeError: If the loop's carry storage is inconsistent.
        """
        if len(op.operands) < 2:
            return None
        bound_values: list[int] = []
        for operand in op.operands[:3]:
            if not operand.is_constant():
                return None
            value = operand.get_const()
            if not isinstance(value, (bool, int)):
                return None
            bound_values.append(int(value))
        start, stop = bound_values[:2]
        step = bound_values[2] if len(bound_values) == 3 else 1
        try:
            sequence = range(start, stop, step)
            trip_count = len(sequence)
        except (OverflowError, ValueError):
            return None
        if trip_count > 1:
            return None

        carries = list(op.region_args)
        if trip_count == 0:
            result_subst = {carry.result.uuid: carry.init for carry in carries}
            if self._has_quantum_effects(op.operations):
                # Keep the boxed loop so resource allocation can map the
                # body-produced qubit SSA versions used after the loop back
                # onto their entry resources, even though emit executes no
                # body operations for the empty index set. Classical carry
                # results nevertheless alias their initializers.
                return None, result_subst
            return [], result_subst

        if self._has_slice_lifetime_effects(op.operations):
            # SliceBorrowCheckPass relies on the control-flow boundary to
            # distinguish a release performed inside the body from a legal
            # top-level release. Flattening a one-trip loop would erase that
            # lifetime boundary and silently accept an outer view released by
            # the body. Keep the boxed loop until the borrow check has run.
            return None

        body_subst: dict[str, ValueBase] = {
            carry.block_arg.uuid: carry.init for carry in carries
        }
        if op.loop_var_value is not None:
            body_subst[op.loop_var_value.uuid] = op.operands[0]
        body_ops = [
            self._apply_substitution(body_op, body_subst) for body_op in op.operations
        ]
        substitutor = ValueSubstitutor(body_subst, transitive=True)
        result_subst: dict[str, Value] = {}
        for carry in carries:
            body_yield = substitutor.substitute_value(carry.yielded)
            if not isinstance(body_yield, Value):
                raise RuntimeError(
                    "[FOR DEVELOPER] A loop carry body yield did not remain "
                    "a scalar Value during one-trip lowering."
                )
            result_subst[carry.result.uuid] = body_yield
        return body_ops, result_subst

    @classmethod
    def _has_quantum_effects(cls, operations: list[Operation]) -> bool:
        """Return whether an operation tree contains quantum/hybrid work.

        Args:
            operations (list[Operation]): Operations to inspect recursively.

        Returns:
            bool: ``True`` when a quantum gate, allocation, measurement, or
                other hybrid operation occurs anywhere in the tree.
        """
        for operation in operations:
            if operation.operation_kind in (
                OperationKind.QUANTUM,
                OperationKind.HYBRID,
            ):
                return True
            if isinstance(operation, HasNestedOps) and any(
                cls._has_quantum_effects(body) for body in operation.nested_op_lists()
            ):
                return True
        return False

    @classmethod
    def _has_slice_lifetime_effects(cls, operations: list[Operation]) -> bool:
        """Return whether a body contains slice lifetime markers.

        Args:
            operations (list[Operation]): Operations to inspect recursively.

        Returns:
            bool: True when flattening would move a slice declaration or
                release across its control-flow boundary.
        """
        for operation in operations:
            if isinstance(operation, (SliceArrayOperation, ReleaseSliceViewOperation)):
                return True
            if isinstance(operation, HasNestedOps) and any(
                cls._has_slice_lifetime_effects(body)
                for body in operation.nested_op_lists()
            ):
                return True
        return False

    @staticmethod
    def _normalize_static_carry_value(
        result: Value,
        value: Any,
    ) -> int | float | None:
        """Normalize a concrete result to its carry's scalar type.

        Args:
            result (Value): Carry result whose type constrains the value.
            value (Any): Candidate concrete value produced by evaluation.

        Returns:
            int | float | None: A normalized scalar, or ``None`` when the
                candidate cannot safely inhabit the carry type.
        """
        if isinstance(result.type, UIntType):
            if isinstance(value, bool) or not isinstance(value, int):
                return None
            return int(value)
        if isinstance(result.type, FloatType):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            return float(value)
        return None

    def _resolve_static_for_indexset(
        self,
        op: ForOperation,
        concrete_values: dict[str, Any],
    ) -> range | None:
        """Resolve a for-loop's complete compile-time index set.

        Args:
            op (ForOperation): Range loop whose bounds should resolve.
            concrete_values (dict[str, Any]): UUID-keyed concrete values
                available before the loop.

        Returns:
            range | None: Concrete Python range, or ``None`` when a bound is
                unresolved, malformed, or too large for safe static replay.
        """
        if len(op.operands) < 2:
            return None
        resolver = UnifiedValueResolver(
            context=concrete_values,
            bindings=self._bindings,
        )
        bounds: list[int] = []
        for operand in op.operands[:3]:
            value = resolver.resolve(operand)
            if isinstance(value, bool) or not isinstance(value, int):
                return None
            bounds.append(int(value))
        start, stop = bounds[:2]
        step = bounds[2] if len(bounds) == 3 else 1
        try:
            indexset = range(start, stop, step)
            trip_count = len(indexset)
        except (OverflowError, ValueError):
            return None
        if trip_count > _MAX_STATIC_CARRY_ITERATIONS:
            return None
        return indexset

    def _reserve_static_replay_iterations(self, count: int) -> bool:
        """Reserve work from the pass-wide static replay budget.

        A per-loop bound alone is insufficient for nested loops: two loops
        of 10,000 trips would permit 100 million body evaluations. Charging
        every actually replayed loop invocation to one shared budget keeps
        nested compilation work bounded and makes exhaustion fail closed.

        Args:
            count (int): Number of loop-body iterations about to be replayed.

        Returns:
            bool: True when the work was reserved, or False when it would
                exceed the remaining pass budget.
        """
        if count < 0 or count > self._static_replay_remaining:
            return False
        self._static_replay_remaining -= count
        return True

    def _replay_static_operations(
        self,
        operations: list[Operation],
        concrete_values: dict[str, Any],
    ) -> tuple[bool, bool]:
        """Replay the statically decidable classical part of an op tree.

        Quantum and hybrid operations are deliberately skipped: replay only
        computes classical SSA values and never simulates quantum state. A
        nested range/items loop or if is handled recursively when its index
        set or condition is concrete. Runtime while loops and unsupported
        classical effects make replay fail closed.

        Args:
            operations (list[Operation]): Operations to interpret in program
                order.
            concrete_values (dict[str, Any]): UUID-keyed concrete environment,
                updated in place.

        Returns:
            tuple[bool, bool]: ``(supported, complete)``. ``supported`` is
                False when control structure cannot be decided safely.
                ``complete`` is False when a supported scalar operation could
                not be evaluated; callers may still use independently known
                carry results but must retain a pure-classical loop.

        Raises:
            RuntimeError: If a nested loop's carry storage is inconsistent.
        """
        complete = True
        supported_ops = (BinOp, CompOp, CondOp, NotOp)
        for operation in operations:
            if isinstance(operation, supported_ops):
                # Body operation UUIDs repeat across loop iterations. Clear
                # the previous trip's result before evaluating so a failed
                # fold cannot masquerade as a fresh value.
                for result in operation.results:
                    concrete_values.pop(result.uuid, None)
                evaluate_classical_op_concrete(
                    operation,
                    concrete_values,
                    self._bindings,
                )
                if any(
                    result.uuid not in concrete_values for result in operation.results
                ):
                    complete = False
                continue
            if isinstance(operation, ForOperation):
                nested_supported, nested_complete = self._replay_static_for(
                    operation,
                    concrete_values,
                )
                if not nested_supported:
                    return False, False
                complete = complete and nested_complete
                continue
            if isinstance(operation, ForItemsOperation):
                nested_supported, nested_complete = self._replay_static_for_items(
                    operation,
                    concrete_values,
                )
                if not nested_supported:
                    return False, False
                complete = complete and nested_complete
                continue
            if isinstance(operation, IfOperation):
                condition = self._try_resolve_condition(
                    operation.condition,
                    concrete_values,
                )
                if condition is None:
                    return False, False
                branch = (
                    operation.true_operations
                    if condition
                    else operation.false_operations
                )
                branch_supported, branch_complete = self._replay_static_operations(
                    branch,
                    concrete_values,
                )
                if not branch_supported:
                    return False, False
                complete = complete and branch_complete
                resolver = UnifiedValueResolver(
                    context=concrete_values,
                    bindings=self._bindings,
                )
                for merge in operation.iter_merges():
                    concrete_values.pop(merge.result.uuid, None)
                    if not merge.result.type.is_classical():
                        continue
                    value = resolver.resolve(merge.select(condition))
                    if value is None:
                        concrete_values.pop(merge.result.uuid, None)
                    else:
                        concrete_values[merge.result.uuid] = value
                continue
            if isinstance(operation, WhileOperation):
                return False, False
            if operation.operation_kind in (
                OperationKind.QUANTUM,
                OperationKind.HYBRID,
            ):
                continue
            return False, False
        return True, complete

    def _replay_static_for(
        self,
        op: ForOperation,
        concrete_values: dict[str, Any],
    ) -> tuple[bool, bool]:
        """Replay one nested statically bounded range loop.

        Args:
            op (ForOperation): Nested range loop to replay.
            concrete_values (dict[str, Any]): Enclosing concrete environment,
                updated with the loop's result values.

        Returns:
            tuple[bool, bool]: Whether replay is supported and whether every
                executed scalar operation evaluated completely.

        Raises:
            RuntimeError: If the loop's carry storage is inconsistent.
        """
        indexset = self._resolve_static_for_indexset(op, concrete_values)
        if indexset is None:
            return False, False
        if not self._reserve_static_replay_iterations(len(indexset)):
            return False, False

        carries = list(op.region_args)
        resolver = UnifiedValueResolver(
            context=concrete_values,
            bindings=self._bindings,
        )
        carried_values: dict[str, int | float] = {}
        for carry in carries:
            value = self._normalize_static_carry_value(
                carry.result,
                resolver.resolve(carry.init),
            )
            if value is not None:
                carried_values[carry.block_arg.uuid] = value

        complete = True
        for loop_index in indexset:
            # Match emit/executor replay by installing carried block
            # arguments before the iteration formal. Their UUIDs are
            # validated as disjoint, so this is a readability invariant, not
            # a semantic dependency on dictionary insertion order.
            for carry in carries:
                if carry.block_arg.uuid in carried_values:
                    concrete_values[carry.block_arg.uuid] = carried_values[
                        carry.block_arg.uuid
                    ]
                else:
                    concrete_values.pop(carry.block_arg.uuid, None)
            if op.loop_var_value is not None:
                concrete_values[op.loop_var_value.uuid] = loop_index
            body_supported, body_complete = self._replay_static_operations(
                op.operations,
                concrete_values,
            )
            if not body_supported:
                return False, False
            complete = complete and body_complete

            yield_resolver = UnifiedValueResolver(
                context=concrete_values,
                bindings=self._bindings,
            )
            next_values: dict[str, int | float] = {}
            for carry in carries:
                value = self._normalize_static_carry_value(
                    carry.result,
                    yield_resolver.resolve(carry.yielded),
                )
                if value is not None:
                    next_values[carry.block_arg.uuid] = value
            carried_values = next_values

        for carry in carries:
            if carry.block_arg.uuid in carried_values:
                concrete_values[carry.result.uuid] = carried_values[
                    carry.block_arg.uuid
                ]
            else:
                concrete_values.pop(carry.result.uuid, None)
        complete = complete and len(carried_values) == len(carries)
        return True, complete

    def _resolve_static_items_entries(
        self,
        op: ForItemsOperation,
        concrete_values: dict[str, Any],
    ) -> list[tuple[Any, Any]] | None:
        """Resolve a nested items loop to a bounded entry list.

        Args:
            op (ForItemsOperation): Items loop whose iterable should resolve.
            concrete_values (dict[str, Any]): Current concrete environment.

        Returns:
            list[tuple[Any, Any]] | None: Concrete entries within the replay
                limit, or None when the iterable is unresolved or too large.
        """
        if not op.operands:
            return None
        iterable = op.operands[0]
        entries: list[tuple[Any, Any]] | None = None
        if isinstance(iterable, DictValue):
            if iterable.metadata.dict_runtime is not None:
                entries = list(iterable.get_bound_data_items())
            else:
                candidate = concrete_values.get(iterable.uuid)
                if candidate is None:
                    parameter_name = iterable.parameter_name()
                    for key in (parameter_name, iterable.uuid):
                        if key and key in self._bindings:
                            candidate = self._bindings[key]
                            break
                if isinstance(candidate, dict):
                    entries = list(candidate.items())
        if entries is None or len(entries) > _MAX_STATIC_CARRY_ITERATIONS:
            return None
        return entries

    @staticmethod
    def _seed_static_items_key(
        op: ForItemsOperation,
        key: Any,
        item_value: Any,
        concrete_values: dict[str, Any],
    ) -> bool:
        """Bind one for-items key/value pair into a replay environment.

        Args:
            op (ForItemsOperation): Items loop whose identities are bound.
            key (Any): Current concrete dictionary key.
            item_value (Any): Current concrete dictionary value.
            concrete_values (dict[str, Any]): Environment updated in place.

        Returns:
            bool: True when the key shape matches the loop identities.
        """
        if op.key_var_values is None or op.value_var_value is None:
            return False
        if op.key_is_vector:
            if (
                len(op.key_var_values) != 1
                or not isinstance(op.key_var_values[0], ArrayValue)
                or not isinstance(key, (tuple, list))
            ):
                return False
            identity = op.key_var_values[0]
            bound_key = identity.with_array_runtime_metadata(const_array=key)
            concrete_values[identity.uuid] = bound_key
            if identity.shape:
                concrete_values[identity.shape[0].uuid] = len(key)
        else:
            key_parts = key if isinstance(key, (tuple, list)) else (key,)
            if len(key_parts) != len(op.key_var_values):
                return False
            for identity, part in zip(
                op.key_var_values,
                key_parts,
                strict=True,
            ):
                concrete_values[identity.uuid] = part
        concrete_values[op.value_var_value.uuid] = item_value
        return True

    def _replay_static_for_items(
        self,
        op: ForItemsOperation,
        concrete_values: dict[str, Any],
    ) -> tuple[bool, bool]:
        """Replay one nested compile-time-bound items loop.

        Args:
            op (ForItemsOperation): Nested items loop to replay.
            concrete_values (dict[str, Any]): Enclosing concrete environment,
                updated with final carry results.

        Returns:
            tuple[bool, bool]: Whether replay is supported and whether every
                executed scalar operation evaluated completely.

        Raises:
            RuntimeError: If the loop's carry storage is inconsistent.
        """
        entries = self._resolve_static_items_entries(op, concrete_values)
        if entries is None:
            return False, False
        if not self._reserve_static_replay_iterations(len(entries)):
            return False, False

        carries = list(op.region_args)
        resolver = UnifiedValueResolver(
            context=concrete_values,
            bindings=self._bindings,
        )
        carried_values: dict[str, int | float] = {}
        for carry in carries:
            value = self._normalize_static_carry_value(
                carry.result,
                resolver.resolve(carry.init),
            )
            if value is not None:
                carried_values[carry.block_arg.uuid] = value

        complete = True
        for key, item_value in entries:
            iteration_bindings: dict[str, Any] = {}
            if not self._seed_static_items_key(
                op,
                key,
                item_value,
                iteration_bindings,
            ):
                return False, False
            # Keep the same carried-before-iteration-formals order used by
            # range replay and emit. Build item bindings separately so a
            # malformed key shape cannot partially mutate concrete_values.
            for carry in carries:
                if carry.block_arg.uuid in carried_values:
                    concrete_values[carry.block_arg.uuid] = carried_values[
                        carry.block_arg.uuid
                    ]
                else:
                    concrete_values.pop(carry.block_arg.uuid, None)
            concrete_values.update(iteration_bindings)
            body_supported, body_complete = self._replay_static_operations(
                op.operations,
                concrete_values,
            )
            if not body_supported:
                return False, False
            complete = complete and body_complete

            yield_resolver = UnifiedValueResolver(
                context=concrete_values,
                bindings=self._bindings,
            )
            next_values: dict[str, int | float] = {}
            for carry in carries:
                value = self._normalize_static_carry_value(
                    carry.result,
                    yield_resolver.resolve(carry.yielded),
                )
                if value is not None:
                    next_values[carry.block_arg.uuid] = value
            carried_values = next_values

        for carry in carries:
            if carry.block_arg.uuid in carried_values:
                concrete_values[carry.result.uuid] = carried_values[
                    carry.block_arg.uuid
                ]
            else:
                concrete_values.pop(carry.result.uuid, None)
        complete = complete and len(carried_values) == len(carries)
        return True, complete

    def _evaluate_static_for_carries(
        self,
        op: ForOperation,
        concrete_values: dict[str, Any],
    ) -> tuple[dict[str, Value], bool]:
        """Replay a bounded loop's pure classical carry recurrence.

        The traced quantum body remains boxed in ``op``. Only straight-line,
        side-effect-free scalar arithmetic is evaluated, once per statically
        known iteration. Unsupported classical/control operations make the
        optimization fail closed. A fully evaluated classical-only loop can
        be removed; a loop containing quantum or hybrid work is retained and
        only its post-loop carry results are substituted.

        Args:
            op (ForOperation): Lowered range loop to inspect.
            concrete_values (dict[str, Any]): UUID-keyed concrete values
                available before the loop.

        Returns:
            tuple[dict[str, Value], bool]: Identity-preserving constant
                substitutions for statically known carry results and whether
                the whole loop is a fully evaluated pure-classical recurrence
                that may be removed.

        Raises:
            RuntimeError: If the loop's carry storage is inconsistent.
        """
        indexset = self._resolve_static_for_indexset(op, concrete_values)
        if indexset is None or len(indexset) < 2:
            return {}, False

        carries = list(op.region_args)
        if not carries:
            return {}, False

        replay_values = dict(concrete_values)
        replay_supported, replay_complete = self._replay_static_for(
            op,
            replay_values,
        )
        if not replay_supported:
            return {}, False

        resolver = UnifiedValueResolver(
            context=replay_values,
            bindings=self._bindings,
        )
        substitutions: dict[str, Value] = {}
        for carry in carries:
            value = self._normalize_static_carry_value(
                carry.result,
                resolver.resolve(carry.result),
            )
            if value is not None:
                substitutions[carry.result.uuid] = carry.result.with_const(value)

        pure_classical = (
            not self._has_quantum_effects(op.operations) and replay_complete
        )
        eliminate_loop = pure_classical and len(substitutions) == len(carries)
        return substitutions, eliminate_loop

    def _evaluate_static_for_items_carries(
        self,
        op: ForItemsOperation,
        concrete_values: dict[str, Any],
    ) -> tuple[dict[str, Value], bool]:
        """Replay a bound multi-entry items loop's carry recurrence.

        Args:
            op (ForItemsOperation): Lowered items loop to inspect.
            concrete_values (dict[str, Any]): UUID-keyed concrete values
                available before the loop.

        Returns:
            tuple[dict[str, Value], bool]: Identity-preserving constant
                substitutions for known final carries and whether a fully
                evaluated pure-classical loop may be removed.

        Raises:
            RuntimeError: If the loop's carry storage is inconsistent.
        """
        entries = self._resolve_static_items_entries(op, concrete_values)
        if entries is None or len(entries) < 2:
            return {}, False
        carries = list(op.region_args)
        if not carries:
            return {}, False

        replay_values = dict(concrete_values)
        replay_supported, replay_complete = self._replay_static_for_items(
            op,
            replay_values,
        )
        if not replay_supported:
            return {}, False

        resolver = UnifiedValueResolver(
            context=replay_values,
            bindings=self._bindings,
        )
        substitutions: dict[str, Value] = {}
        for carry in carries:
            value = self._normalize_static_carry_value(
                carry.result,
                resolver.resolve(carry.result),
            )
            if value is not None:
                substitutions[carry.result.uuid] = carry.result.with_const(value)

        pure_classical = (
            not self._has_quantum_effects(op.operations) and replay_complete
        )
        eliminate_loop = pure_classical and len(substitutions) == len(carries)
        return substitutions, eliminate_loop

    def _lower_degenerate_for_items(
        self,
        op: ForItemsOperation,
    ) -> tuple[list[Operation] | None, dict[str, Value]] | None:
        """Lower a bound zero- or one-entry items loop to SSA aliases.

        Scalar and tuple-scalar keys are substituted directly. A Vector key
        is materialized as a constant ``ArrayValue`` with a concrete length,
        allowing its elements and ``shape[0]`` to participate in downstream
        compile-time structure.

        Args:
            op (ForItemsOperation): Items loop after nested lowering.

        Returns:
            tuple[list[Operation] | None, dict[str, Value]] | None:
                Replacement body operations and result substitutions. The
                operation list is ``None`` when an empty quantum loop must
                remain boxed for resource-lineage mapping; its substitutions
                still alias carries to their initializers. Returns ``None``
                when the iterable is unresolved or has multiple entries.

        Raises:
            RuntimeError: If carry storage is inconsistent or a one-entry
                loop lacks its key/value identity Values.
        """
        if not op.operands or not isinstance(op.operands[0], DictValue):
            return None
        iterable = op.operands[0]
        if iterable.metadata.dict_runtime is None:
            return None
        entries = list(iterable.get_bound_data_items())
        if len(entries) > 1:
            return None

        carries = list(op.region_args)
        if not entries:
            result_subst = {carry.result.uuid: carry.init for carry in carries}
            if self._has_quantum_effects(op.operations):
                # Keep the boxed loop so resource allocation can map the
                # body-produced qubit SSA versions used after the loop back
                # onto their entry resources, even though emit executes no
                # body operations for the empty iterable. This mirrors the
                # zero-trip range-loop handling in ``_lower_degenerate_for``.
                # Classical carry results nevertheless alias initializers.
                return None, result_subst
            return [], result_subst
        if op.key_var_values is None or op.value_var_value is None:
            raise RuntimeError(
                "[FOR DEVELOPER] A one-entry ForItemsOperation lacks key/value "
                "identity Values."
            )

        if self._has_slice_lifetime_effects(op.operations):
            # Preserve the same lifetime boundary as a one-trip range loop.
            # Flattening here would move an inner release to top level before
            # SliceBorrowCheckPass can reject it.
            return None

        key, item_value = entries[0]
        key_parts = key if isinstance(key, tuple) else (key,)
        if not all(isinstance(part, (bool, int, float)) for part in key_parts):
            return None
        if not isinstance(item_value, (bool, int, float)):
            return None

        body_subst: dict[str, ValueBase] = {
            carry.block_arg.uuid: carry.init for carry in carries
        }
        if op.key_is_vector:
            if len(op.key_var_values) != 1:
                return None
            identity = op.key_var_values[0]
            if not isinstance(identity, ArrayValue):
                return None
            if not identity.shape:
                # There is no existing SSA identity on which to publish the
                # concrete vector length. Inventing one here would make the
                # public pass nondeterministic across repeated runs, so retain
                # the boxed loop and let normal emit-time binding handle this
                # malformed or legacy shape-less vector key.
                return None
            length_value = identity.shape[0].with_const(len(key_parts))
            key_shape = (length_value, *identity.shape[1:])
            body_subst[identity.shape[0].uuid] = length_value
            bound_key = dataclasses.replace(identity, shape=key_shape)
            bound_key = bound_key.with_array_runtime_metadata(const_array=key_parts)
            body_subst[identity.uuid] = bound_key
        else:
            if len(key_parts) != len(op.key_var_values):
                return None
            for identity, part in zip(op.key_var_values, key_parts, strict=True):
                body_subst[identity.uuid] = identity.with_const(part)
        body_subst[op.value_var_value.uuid] = op.value_var_value.with_const(item_value)
        body_ops = [
            self._apply_substitution(body_op, body_subst) for body_op in op.operations
        ]
        substitutor = ValueSubstitutor(body_subst, transitive=True)
        result_subst: dict[str, Value] = {}
        for carry in carries:
            body_yield = substitutor.substitute_value(carry.yielded)
            if not isinstance(body_yield, Value):
                raise RuntimeError(
                    "[FOR DEVELOPER] A for-items carry body yield did not "
                    "remain a scalar Value during one-entry lowering."
                )
            result_subst[carry.result.uuid] = body_yield
        return body_ops, result_subst

    @staticmethod
    def _substitute_branch_rebinds(
        rebinds: tuple[Any, ...],
        substitutor: "ValueSubstitutor",
    ) -> tuple[Any, ...]:
        """Rewrite branch rebind record values through the substitutor.

        Keeps ``IfOperation.branch_rebinds`` coherent when merge outputs
        referenced by a record are lowered away, so the control-flow
        discard check's ``AnalyzePass`` safety-net run sees live values.

        Args:
            rebinds (tuple[Any, ...]): ``BranchRebind`` records.
            substitutor (ValueSubstitutor): The active substitution.

        Returns:
            tuple[Any, ...]: Rewritten records; ``rebinds`` itself when
                nothing changed.
        """
        if not rebinds:
            return rebinds
        new_records = []
        changed = False
        for record in rebinds:
            new_before = substitutor.substitute_value(record.before)
            if isinstance(new_before, Value) and new_before is not record.before:
                record = dataclasses.replace(record, before=new_before)
                changed = True
            new_records.append(record)
        return tuple(new_records) if changed else rebinds

    @staticmethod
    def _substitute_loop_rebinds(
        rebinds: tuple[Any, ...],
        substitutor: "ValueSubstitutor",
    ) -> tuple[Any, ...]:
        """Rewrite loop rebind record values through the substitutor.

        Keeps ``loop_carried_rebinds`` coherent when a record's ``after``
        (or ``before``) was a nested compile-time if's merge output that
        the lowering erased — without this, the control-flow discard
        check's ``AnalyzePass`` safety-net run would see a dangling UUID
        and lose the record's carried-forward lineage.

        Args:
            rebinds (tuple[Any, ...]): ``LoopCarriedRebind`` records.
            substitutor (ValueSubstitutor): The active substitution.

        Returns:
            tuple[Any, ...]: Rewritten records; ``rebinds`` itself when
                nothing changed.
        """
        if not rebinds:
            return rebinds
        new_records = []
        changed = False
        for record in rebinds:
            new_before = substitutor.substitute_value(record.before)
            new_after = substitutor.substitute_value(record.after)
            replacements: dict[str, Any] = {}
            if isinstance(new_before, ValueBase) and new_before is not record.before:
                replacements["before"] = new_before
            if isinstance(new_after, ValueBase) and new_after is not record.after:
                replacements["after"] = new_after
            if replacements:
                record = dataclasses.replace(record, **replacements)
                changed = True
            new_records.append(record)
        return tuple(new_records) if changed else rebinds

    @staticmethod
    def _substitute_region_args(
        region_args: tuple[Any, ...],
        substitutor: "ValueSubstitutor",
    ) -> tuple[Any, ...]:
        """Rewrite loop region-argument values through the substitutor.

        Keeps ``region_args`` coherent when a region argument's
        ``yielded`` (or ``init``) was a nested compile-time if's phi
        output that the lowering erased — without this, the executor and
        emit-time threading would chase a dangling phi UUID (``Value
        _phi_N not found``).

        Args:
            region_args (tuple[Any, ...]): ``RegionArg`` records.
            substitutor (ValueSubstitutor): The active substitution.

        Returns:
            tuple[Any, ...]: Rewritten records; ``region_args`` itself
                when nothing changed.
        """
        if not region_args:
            return region_args
        new_records = []
        changed = False
        for record in region_args:
            replacements: dict[str, Any] = {}
            for field_name in ("init", "block_arg", "yielded", "result"):
                current = getattr(record, field_name)
                substituted = substitutor.substitute_value(current)
                if isinstance(substituted, Value) and substituted is not current:
                    replacements[field_name] = substituted
            if replacements:
                record = dataclasses.replace(record, **replacements)
                changed = True
            new_records.append(record)
        return tuple(new_records) if changed else region_args

    def _apply_substitution(
        self,
        op: Operation,
        subst: dict[str, ValueBase],
    ) -> Operation:
        """Apply merge substitution map to an operation's operands and results.

        Args:
            op (Operation): Operation to rewrite through ``subst``.
            subst (dict[str, ValueBase]): Accumulated merge substitution map.
                Mutated in place when a CastOperation result is rebuilt with
                re-synced carrier metadata, so later operations holding the
                same SSA value pick up the rebuilt metadata.

        Returns:
            Operation: The rewritten operation (``op`` itself when nothing
                changed).
        """
        if not subst:
            return op

        substitutor = ValueSubstitutor(subst, transitive=True)

        from qamomile.circuit.ir.operation.cast import CastOperation

        new_operands = cast(
            list[Value],
            [
                substitutor.substitute_value(v) if isinstance(v, ValueBase) else v
                for v in op.operands
            ],
        )

        # Substitution must reach into structural array results without
        # replacing their produced SSA identities. QInit/measurement shapes
        # and SliceArray bounds may reference a lowered merge/carry result.
        # Gate results deliberately stay untouched: rebuilding their
        # parent-array lineages here can change slice-borrow ownership.
        new_results: list[Value] = list(op.results)
        results_changed = False
        if isinstance(
            op, (QInitOperation, SliceArrayOperation, MeasureVectorOperation)
        ):
            for i, result in enumerate(op.results):
                if not isinstance(result, ValueBase):
                    continue
                substituted_result = substitutor.substitute_value(result)
                if (
                    isinstance(substituted_result, Value)
                    and substituted_result is not result
                    and substituted_result.uuid == result.uuid
                ):
                    new_results[i] = substituted_result
                    results_changed = True

        changed = (
            any(
                new_operands[i] is not op.operands[i]
                for i in range(len(op.operands))
                if isinstance(op.operands[i], ValueBase)
            )
            or results_changed
        )

        # Handle control-flow recursion.
        if isinstance(op, IfOperation):
            new_true = [self._apply_substitution(o, subst) for o in op.true_operations]
            new_false = [
                self._apply_substitution(o, subst) for o in op.false_operations
            ]
            # Branch-merge yields may reference outputs of previously
            # lowered ifs; rewrite them through the same substitutor as
            # the operands.
            new_true_yields = [
                cast(Value, substitutor.substitute_value(v)) for v in op.true_yields
            ]
            new_false_yields = [
                cast(Value, substitutor.substitute_value(v)) for v in op.false_yields
            ]
            return dataclasses.replace(
                op,
                operands=new_operands,
                results=new_results,
                true_operations=new_true,
                false_operations=new_false,
                true_yields=new_true_yields,
                false_yields=new_false_yields,
                branch_rebinds=self._substitute_branch_rebinds(
                    op.branch_rebinds, substitutor
                ),
            )

        if isinstance(op, HasNestedOps):
            # Generic recursion for For/ForItems/While bodies.
            new_lists = [
                [self._apply_substitution(o, subst) for o in body]
                for body in op.nested_op_lists()
            ]
            rebuilt = op.rebuild_nested(new_lists)
            rebuilt = dataclasses.replace(
                cast(Any, rebuilt),
                operands=new_operands,
                results=new_results,
            )
            rebinds = getattr(rebuilt, "loop_carried_rebinds", ())
            new_rebinds = self._substitute_loop_rebinds(rebinds, substitutor)
            if new_rebinds is not rebinds:
                rebuilt = dataclasses.replace(
                    cast(Any, rebuilt), loop_carried_rebinds=new_rebinds
                )
            region_args = getattr(rebuilt, "region_args", ())
            new_region_args = self._substitute_region_args(region_args, substitutor)
            if new_region_args is not region_args:
                rebuilt = dataclasses.replace(
                    cast(Any, rebuilt), region_args=new_region_args
                )
            return rebuilt

        result_op = op
        if changed:
            result_op = dataclasses.replace(
                op, operands=new_operands, results=new_results
            )

        # theta is now part of operands, handled by the operands
        # substitution above.

        # Handle ControlledUOperation non-operand fields per subclass.
        if isinstance(result_op, ControlledUOperation):
            extra_kwargs: dict[str, Any] = {}
            # power is shared across all subclasses.
            if isinstance(result_op.power, Value):
                new_power = substitutor.substitute_value(result_op.power)
                if new_power is not result_op.power:
                    extra_kwargs["power"] = new_power
            if isinstance(result_op, SymbolicControlledU):
                new_nc = substitutor.substitute_value(result_op.num_controls)
                if new_nc is not result_op.num_controls:
                    extra_kwargs["num_controls"] = new_nc
                if result_op.control_indices is not None:
                    new_ci = self._substitute_value_list(
                        list(result_op.control_indices), substitutor
                    )
                    if new_ci is not None:
                        extra_kwargs["control_indices"] = tuple(new_ci)
            # ConcreteControlledU: num_controls is int, nothing to substitute.
            if extra_kwargs:
                result_op = dataclasses.replace(result_op, **extra_kwargs)

        # Handle CastOperation source provenance sync.
        if isinstance(result_op, CastOperation) and changed:
            new_source = new_operands[0] if new_operands else None
            if (
                new_source is not None
                and hasattr(new_source, "uuid")
                and result_op.results
            ):
                result_val = result_op.results[0]
                if result_val.is_cast_result():
                    num_bits = result_val.get_qfixed_num_bits()
                    carriers: tuple[list[str], list[str]] | None = None
                    if num_bits is not None:
                        carriers = _array_carrier_keys(new_source, num_bits)
                        if (
                            carriers is None
                            and isinstance(new_source, ArrayValue)
                            and new_source.slice_of is not None
                        ):
                            # The selected branch is a strided view whose root
                            # index space is not compile-time resolvable
                            # (symbolic slice bounds). Synthesizing
                            # ``f"{view.uuid}_{i}"`` below would emit view-local
                            # carrier keys that the allocator never registers
                            # (only root-array addresses are), silently dropping
                            # the QFixed measurement at emit. Fail fast instead,
                            # mirroring the frontend's rejection of symbolic
                            # slice views for casts.
                            raise ValidationError(
                                "Compile-time `if` selected a slice-view cast "
                                "source whose root index space is not "
                                "compile-time resolvable (symbolic slice "
                                "bounds). Bind the slice bounds so the QFixed "
                                "carrier qubits resolve to root-array "
                                "addresses; leaving them symbolic would "
                                "silently drop the measurement at emit time."
                            )
                    if carriers is None and num_bits is not None:
                        source_logical_id = getattr(
                            new_source, "logical_id", new_source.uuid
                        )
                        carriers = (
                            [f"{new_source.uuid}_{i}" for i in range(num_bits)],
                            [f"{source_logical_id}_{i}" for i in range(num_bits)],
                        )
                    carrier_uuids = (
                        carriers[0]
                        if carriers is not None
                        else list(result_val.get_cast_qubit_uuids() or ())
                    )
                    carrier_logical_ids = (
                        carriers[1]
                        if carriers is not None
                        else list(result_val.get_cast_qubit_logical_ids() or ())
                    )
                    new_result = result_val.with_cast_metadata(
                        source_uuid=new_source.uuid,
                        source_logical_id=getattr(
                            new_source, "logical_id", new_source.uuid
                        ),
                        qubit_uuids=carrier_uuids,
                        qubit_logical_ids=carrier_logical_ids,
                    )
                    if num_bits is not None:
                        new_result = new_result.with_qfixed_metadata(
                            qubit_uuids=carrier_uuids,
                            num_bits=num_bits,
                            int_bits=result_val.get_qfixed_int_bits() or 0,
                        )
                    new_mapping = (
                        list(new_result.get_qfixed_qubit_uuids())
                        or result_op.qubit_mapping
                    )
                    result_op = dataclasses.replace(
                        result_op,
                        results=[new_result],
                        qubit_mapping=new_mapping,
                    )
                    # Propagate the rebuilt result to downstream consumers.
                    # The MeasureQFixedOperation operand is the same SSA
                    # value, and plan-time lowering reads carrier keys from
                    # that operand's metadata — without this entry it would
                    # keep the stale trace-time carriers of the unselected
                    # branch. Self-mapping is safe: ``_mapped_value_for_uuid``
                    # seeds its cycle guard with the queried UUID.
                    subst[result_val.uuid] = new_result

        return result_op

    def _substitute_value_list(
        self,
        values: list[Value],
        substitutor: ValueSubstitutor,
    ) -> list[Value] | None:
        """Substitute values in a list, returning new list if changed, None otherwise."""
        new_values: list[Value] = []
        changed = False
        for v in values:
            new_v = substitutor.substitute_value(v)
            if new_v is not v and isinstance(new_v, Value):
                new_values.append(new_v)
                changed = True
            else:
                new_values.append(v)
        return new_values if changed else None

    def _substitute_output_values(
        self,
        output_values: list[ValueLike],
        subst: dict[str, ValueBase],
    ) -> list[ValueLike]:
        """Apply transitive merge substitutions to block outputs.

        Args:
            output_values (list[ValueLike]): Block output values to rewrite.
            subst (dict[str, ValueBase]): UUID-keyed substitution map produced
                while lowering control flow.

        Returns:
            list[ValueLike]: Rewritten output values.
        """
        if not subst:
            return output_values

        substitutor = ValueSubstitutor(subst, transitive=True)
        new_outputs: list[ValueLike] = []
        for ov in output_values:
            substituted = substitutor.substitute_value(ov)
            new_outputs.append(cast(ValueLike, substituted))
        return new_outputs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _eliminate_dead_ops(
        self,
        operations: list[Operation],
        dead_uuids: set[str],
        output_values: list[ValueLike] | None = None,
    ) -> list[Operation]:
        """Remove operations whose results are only consumed by dead UUIDs.

        A known-pure scalar expression is removable if ALL of its result UUIDs
        are in the dead set AND none of those results are used by any remaining
        operation's operands (or block outputs). Operations outside the pure
        expression whitelist remain even when their SSA results are unused.

        This is applied iteratively to propagate: if removing a CompOp
        makes a BinOp's result dead, the BinOp is also removed.

        Args:
            operations (list[Operation]): Operations to prune, including
                control-flow operations whose bodies are processed
                recursively.
            dead_uuids (set[str]): Result UUIDs made dead by compile-time
                control-flow lowering. Updated in place while deadness is
                propagated to producer operands.
            output_values (list[ValueLike] | None): Values that must remain live
                at this operation-list boundary. Defaults to None.

        Returns:
            list[Operation]: Operations with dead producers removed from
                this list and every nested control-flow body.
        """
        recursively_pruned: list[Operation] = []
        for op in operations:
            if not isinstance(op, HasNestedOps):
                recursively_pruned.append(op)
                continue

            nested_outputs: list[list[ValueLike]]
            if isinstance(op, IfOperation):
                nested_outputs = [list(op.true_yields), list(op.false_yields)]
            elif isinstance(op, (ForOperation, ForItemsOperation, WhileOperation)):
                nested_outputs = [[arg.yielded for arg in op.region_args]]
            else:
                nested_outputs = [[] for _ in op.nested_op_lists()]

            new_lists = [
                self._eliminate_dead_ops(body, dead_uuids, protected)
                for body, protected in zip(
                    op.nested_op_lists(), nested_outputs, strict=True
                )
            ]
            recursively_pruned.append(op.rebuild_nested(new_lists))

        operations = recursively_pruned

        # Build use set: UUIDs referenced as operands in remaining ops.
        used_uuids: set[str] = set()
        for op in operations:
            self._collect_used_uuids(op, used_uuids)
        # Also include block output UUIDs as used.
        if output_values:
            for ov in output_values:
                used_uuids.update(collect_value_like_uuids(ov))

        result, removed = self._remove_dead_ops_recursive(
            operations, dead_uuids, used_uuids
        )

        # If we removed anything, do another pass for propagation.
        if removed:
            return self._eliminate_dead_ops(result, dead_uuids, output_values)
        return result

    def _remove_dead_ops_recursive(
        self,
        operations: list[Operation],
        dead_uuids: set[str],
        used_uuids: set[str],
    ) -> tuple[list[Operation], bool]:
        """Remove one wave of dead producers throughout control-flow bodies.

        Args:
            operations (list[Operation]): Operations to filter recursively.
            dead_uuids (set[str]): Result UUIDs eligible for removal. The set
                is extended with inputs of removed operations so the next
                fixed-point iteration can remove newly dead producers.
            used_uuids (set[str]): UUIDs still read anywhere in the current
                operation tree or by the enclosing block outputs.

        Returns:
            tuple[list[Operation], bool]: The rebuilt operation list and
                whether at least one operation was removed at any depth.
        """
        result: list[Operation] = []
        removed = False
        for op in operations:
            if isinstance(op, HasNestedOps):
                new_lists: list[list[Operation]] = []
                for body in op.nested_op_lists():
                    new_body, body_removed = self._remove_dead_ops_recursive(
                        body, dead_uuids, used_uuids
                    )
                    new_lists.append(new_body)
                    removed = removed or body_removed
                op = op.rebuild_nested(new_lists)

            # Only known-pure scalar expressions are removable. Quantum,
            # hybrid, control-flow, and unknown classical operations may have
            # observable effects even when every SSA result is dead (a
            # measurement whose result is discarded is the canonical case).
            if (
                isinstance(op, _PURE_CLASSICAL_EXPRESSION_TYPES)
                and op.results
                and all(
                    result_value.uuid in dead_uuids
                    and result_value.uuid not in used_uuids
                    for result_value in op.results
                )
            ):
                for operand in op.operands:
                    dead_uuids.add(operand.uuid)
                removed = True
                continue
            result.append(op)
        return result, removed

    @staticmethod
    def _collect_used_uuids(op: Operation, used: set[str]) -> None:
        """Collect all UUIDs an operation genuinely reads (recursive).

        Reads are taken from :func:`genuine_input_values` so
        subclass-specific reads (composite-gate power, symbolic control
        counts, ``IfOperation`` branch-merge yields) count as uses while
        rebind-record values — exposed only for cloning — do not.

        Loop region yields are re-added on top of that: a region's
        ``yielded`` may be defined OUTSIDE the loop body (the
        loop-invariant overwrite shape ``x = c``), in which case the
        region back-edge is the producer's only read. That read is
        deliberately hidden from read-based *analyses* by
        ``genuine_input_values``, but liveness must still keep the
        producer alive — the executors resolve the yield by UUID on
        every iteration.

        Args:
            op (Operation): Operation to inspect.
            used (set[str]): Mutable set of used UUIDs, updated in place.
        """
        for operand in genuine_input_values(op):
            used.update(collect_value_like_uuids(cast(ValueLike, operand)))
        if isinstance(op, (ForOperation, ForItemsOperation, WhileOperation)):
            for region_arg in op.region_args:
                used.update(collect_value_like_uuids(region_arg.yielded))

        # Recurse into control flow (For/ForItems/While/If).
        if isinstance(op, HasNestedOps):
            for body in op.nested_op_lists():
                for inner in body:
                    CompileTimeIfLoweringPass._collect_used_uuids(inner, used)

    def _try_seed_value(
        self,
        value: ValueBase,
        concrete_values: dict[str, Any],
    ) -> None:
        """Seed a concrete-value map from a constant or bound input.

        Args:
            value (ValueBase): Block input value to inspect for constant or
                declared-parameter metadata.
            concrete_values (dict[str, Any]): Mutable UUID-keyed concrete
                value map, updated when ``value`` can be resolved.
        """
        if hasattr(value, "is_constant") and value.is_constant():
            const = value.get_const() if hasattr(value, "get_const") else None
            if const is not None:
                concrete_values[value.uuid] = const

        # Seed against ``bindings`` only via the sanctioned parameter-name
        # provenance — never the display ``Value.name``. A bare-name seed would
        # bind an inlined callee-local value that happened to share a name with
        # a caller binding key, letting the pass fold an ``if`` on it and delete
        # a live branch (a silent miscompilation).
        if hasattr(value, "is_parameter") and value.is_parameter():
            param_name = (
                value.parameter_name() if hasattr(value, "parameter_name") else None
            )
            if param_name and param_name in self._bindings:
                concrete_values[value.uuid] = self._bindings[param_name]
