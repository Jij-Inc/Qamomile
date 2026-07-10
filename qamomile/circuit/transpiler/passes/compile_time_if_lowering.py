"""Compile-time IfOperation lowering pass.

Lowers compile-time resolvable ``IfOperation``s before the segmentation pass,
replacing them with selected-branch operations and substituting merge outputs
with selected-branch values throughout the block.

This prevents ``SegmentationPass`` from seeing classical-only compile-time
``IfOperation``s that would otherwise split quantum segments.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    CompOp,
    CondOp,
    NotOp,
)
from qamomile.circuit.ir.operation.control_flow import (
    HasNestedOps,
    IfOperation,
    genuine_input_values,
)
from qamomile.circuit.ir.value import (
    ArrayValue,
    Value,
    ValueBase,
    resolve_root_array_index,
)
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.value_resolver import (
    ValueResolver as UnifiedValueResolver,
)

from . import Pass
from .emit_support import resolve_if_condition
from .eval_utils import FoldPolicy, fold_classical_op
from .value_mapping import ValueSubstitutor


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

    def __init__(self, bindings: dict[str, Any] | None = None):
        self._bindings = bindings or {}

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
        """Process operations, lowering compile-time IfOperations.

        Returns:
            Tuple of (new operations list, merge substitution map,
            dead UUIDs from lowered condition values).
        """
        new_ops: list[Operation] = []
        merge_subst: dict[str, ValueBase] = {}
        dead_uuids: set[str] = set()

        for op in operations:
            # First apply any accumulated merge substitutions to this op.
            op = self._apply_substitution(op, merge_subst)

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

            new_ops.append(op)

        return new_ops, merge_subst, dead_uuids

    # ------------------------------------------------------------------
    # Substitution
    # ------------------------------------------------------------------

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
            if isinstance(new_before, Value) and new_before is not record.before:
                replacements["before"] = new_before
            if isinstance(new_after, Value) and new_after is not record.after:
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

        new_operands = cast(
            list[Value],
            [
                substitutor.substitute_value(v) if isinstance(v, ValueBase) else v
                for v in op.operands
            ],
        )

        # Merge substitution must reach into ``SliceArrayOperation`` result
        # metadata too.  The result is a sliced ``ArrayValue`` whose
        # ``slice_start`` / ``slice_step`` / ``slice_of`` Values may be
        # merge-output references when the slice bounds come from an
        # ``if`` branch.  Without substituting the result fields, the
        # post-fold ``SliceBorrowCheckPass`` sees a still-symbolic
        # ``slice_start`` and silently skips coverage registration,
        # letting aliased direct-parent accesses slip through.
        from qamomile.circuit.ir.operation import SliceArrayOperation

        new_results: list[Value] = list(op.results)
        results_changed = False
        if isinstance(op, SliceArrayOperation):
            for i, r in enumerate(op.results):
                if isinstance(r, ValueBase):
                    sub_r = substitutor.substitute_value(r)
                    if sub_r is not r:
                        assert isinstance(sub_r, Value)
                        new_results[i] = sub_r
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

        from qamomile.circuit.ir.operation.gate import (
            ControlledUOperation,
            SymbolicControlledU,
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
        from qamomile.circuit.ir.operation.cast import CastOperation

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
        output_values: list[Value],
        subst: dict[str, ValueBase],
    ) -> list[Value]:
        """Apply merge substitution to block output values."""
        if not subst:
            return output_values

        substitutor = ValueSubstitutor(subst, transitive=True)
        new_outputs = []
        for ov in output_values:
            substituted = substitutor.substitute_value(ov)
            if isinstance(substituted, Value):
                new_outputs.append(substituted)
            else:
                new_outputs.append(ov)
        return new_outputs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _eliminate_dead_ops(
        self,
        operations: list[Operation],
        dead_uuids: set[str],
        output_values: list[Value] | None = None,
    ) -> list[Operation]:
        """Remove operations whose results are only consumed by dead UUIDs.

        An operation is removable if ALL of its result UUIDs are in the
        dead set AND none of those results are used by any remaining
        operation's operands (or block outputs).

        This is applied iteratively to propagate: if removing a CompOp
        makes a BinOp's result dead, the BinOp is also removed.
        """
        # Build use set: UUIDs referenced as operands in remaining ops.
        used_uuids: set[str] = set()
        for op in operations:
            self._collect_used_uuids(op, used_uuids)
        # Also include block output UUIDs as used.
        if output_values:
            for ov in output_values:
                used_uuids.add(ov.uuid)

        result: list[Operation] = []
        for op in operations:
            # Check if all results are dead and unused.
            if op.results and all(
                r.uuid in dead_uuids and r.uuid not in used_uuids
                for r in op.results
                if hasattr(r, "uuid")
            ):
                # This op can be safely removed. Mark its operands as
                # potentially dead too (for iterative propagation).
                for operand in op.operands:
                    if hasattr(operand, "uuid"):
                        dead_uuids.add(operand.uuid)
                continue
            result.append(op)

        # If we removed anything, do another pass for propagation.
        if len(result) < len(operations):
            return self._eliminate_dead_ops(result, dead_uuids, output_values)
        return result

    @staticmethod
    def _collect_used_uuids(op: Operation, used: set[str]) -> None:
        """Collect all UUIDs an operation genuinely reads (recursive).

        Reads are taken from :func:`genuine_input_values` so
        subclass-specific reads (composite-gate power, symbolic control
        counts, ``IfOperation`` branch-merge yields) count as uses while
        rebind-record values — exposed only for cloning — do not.

        Args:
            op (Operation): Operation to inspect.
            used (set[str]): Mutable set of used UUIDs, updated in place.
        """
        for operand in genuine_input_values(op):
            used.add(operand.uuid)
            # Also collect element_indices and parent_array references.
            if isinstance(operand, Value):
                if operand.parent_array is not None:
                    used.add(operand.parent_array.uuid)
                for idx in operand.element_indices:
                    used.add(idx.uuid)

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
        """Seed concrete_values from a block input value if it's constant or bound."""
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
