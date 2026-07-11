"""Classical-op lowering pass: identify runtime-evaluation classical ops.

Walks the block, identifies ``CompOp`` / ``CondOp`` / ``NotOp`` / ``BinOp``
instances whose operand dataflow traces back to a ``MeasureOperation``
(i.e. cannot be folded at compile-time, by emit-time loop unrolling, or
by ``compile_time_if_lowering``), and replaces them with the equivalent
``RuntimeClassicalExpr``.

It also lowers the scalar classical merge slots of measurement-conditioned
runtime ``IfOperation``s to ``RuntimeClassicalExpr(SELECT)`` expressions
(``result = true if cond else false``), so branch merges ride the same
runtime-expression machinery as every other measurement-derived classical
op: consumer-based segment placement, host-side per-shot evaluation, and
backend runtime-expression emission. See :meth:`_lower_if_merges`.

Why this pass exists:

The pre-``RuntimeClassicalExpr`` design left runtime classical ops in
their compile-time IR form (``CompOp`` etc.) all the way to emit, where
the emit pass had to fold-or-translate via ``evaluate_classical_predicate``
+ ``_build_runtime_predicate_expr``. This put backend-specific lowering
logic inside the emit pass and used the ``bindings`` dict as a polymorphic
slot holding either Python scalars (fold result) or backend ``Expr``
objects.

By identifying runtime classical ops at IR level and giving them their
own node type, we:

- Make "runtime evaluation required" structurally explicit in the IR.
- Move backend lowering to a dedicated emit hook (``_emit_runtime_classical_expr``).
- Preserve the existing fold path for ops that *can* fold at compile or
  emit time (loop-bound or parameter-bound) — those are not measurement-
  derived and stay as ``CompOp``/``CondOp``/``NotOp``/``BinOp``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
    CondOp,
    NotOp,
    RuntimeClassicalExpr,
    RuntimeOpKind,
    runtime_kind_from_binop,
    runtime_kind_from_compop,
    runtime_kind_from_condop,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfMerge,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import MeasureOperation
from qamomile.circuit.ir.operation.operation import OperationKind
from qamomile.circuit.ir.types.primitives import BitType
from qamomile.circuit.ir.value import ArrayValue, Value, ValueBase
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.passes.analyze import (
    build_dependency_graph,
    find_measurement_derived_values,
    find_measurement_results,
)


class ClassicalLoweringPass(Pass[Block, Block]):
    """Lower measurement-derived classical ops to ``RuntimeClassicalExpr``.

    Input: Block with ``BlockKind.ANALYZED``.
    Output: Block with ``BlockKind.ANALYZED`` (same kind; only op rewrites).

    The pass:

    1. Builds a measurement-taint set using the same dataflow utilities
       as ``AnalyzePass`` (forward propagation from ``MeasureOperation``
       results through the dependency graph).
    2. Walks operations recursively (through ``HasNestedOps``).
    3. For each ``CompOp`` / ``CondOp`` / ``NotOp`` / ``BinOp`` whose
       result UUID is in the taint set, replaces it with an equivalent
       ``RuntimeClassicalExpr`` (same operands and result Value, only
       the op type and kind enum change).
    4. Non-tainted classical ops are left unchanged so the existing fold
       paths (compile-time fold in ``compile_time_if_lowering``,
       emit-time fold in ``evaluate_classical_predicate``) continue to
       handle them.

    The dependency graph and taint set are computed once, walked once,
    so the pass is O(N) where N is the number of operations.
    """

    @property
    def name(self) -> str:
        return "classical_lowering"

    def run(self, input: Block) -> Block:
        if input.kind != BlockKind.ANALYZED:
            # Defensive: the pass is designed to run after AnalyzePass.
            # Tolerate other kinds (e.g. tests passing AFFINE blocks
            # directly) by computing taint from scratch.
            pass

        dep_graph = build_dependency_graph(input.operations)
        measurement_uuids = find_measurement_results(input.operations)
        tainted = find_measurement_derived_values(dep_graph, measurement_uuids)

        new_ops = self._rewrite_operations(input.operations, tainted)
        while_protected = _collect_while_operand_uuids(new_ops)
        new_ops = self._lower_if_merges(new_ops, tainted, while_protected)
        return dataclasses.replace(input, operations=new_ops)

    # ------------------------------------------------------------------
    # Rewrite walker
    # ------------------------------------------------------------------

    def _rewrite_operations(
        self,
        operations: list[Operation],
        tainted: set[str],
    ) -> list[Operation]:
        """Walk operations recursively; rewrite tainted classical ops."""
        new_ops: list[Operation] = []
        for op in operations:
            if isinstance(op, (CompOp, CondOp, NotOp, BinOp)) and self._is_tainted(
                op, tainted
            ):
                new_ops.append(self._lower(op))
                continue
            if isinstance(op, HasNestedOps):
                rewritten_lists = [
                    self._rewrite_operations(body, tainted)
                    for body in op.nested_op_lists()
                ]
                new_ops.append(op.rebuild_nested(rewritten_lists))
                continue
            new_ops.append(op)
        return new_ops

    def _is_tainted(self, op: Operation, tainted: set[str]) -> bool:
        """True when any operand or result of ``op`` is measurement-derived.

        We check both operands and results: a result UUID landing in
        ``tainted`` confirms the op participates in the measurement
        dataflow chain. Checking operands is redundant (the dataflow
        propagation derived the result from those operands) but cheap and
        defends against subtle graph-construction edge cases.
        """
        for v in op.results:
            if isinstance(v, ValueBase) and v.uuid in tainted:
                return True
        for v in op.operands:
            if isinstance(v, ValueBase) and v.uuid in tainted:
                return True
        return False

    def _lower(self, op: Operation) -> RuntimeClassicalExpr:
        """Convert a compile-time classical op to ``RuntimeClassicalExpr``.

        Preserves operands and result Values verbatim — only the op type
        and the unified ``kind`` enum change. This means downstream
        consumers reading the result Value see the same UUID/name; only
        the op-type-based dispatch differs.
        """
        if isinstance(op, BinOp):
            kind = runtime_kind_from_binop(cast(Any, op.kind))
        elif isinstance(op, CompOp):
            kind = runtime_kind_from_compop(cast(Any, op.kind))
        elif isinstance(op, CondOp):
            kind = runtime_kind_from_condop(cast(Any, op.kind))
        elif isinstance(op, NotOp):
            from qamomile.circuit.ir.operation.arithmetic_operations import (
                RuntimeOpKind,
            )

            kind = RuntimeOpKind.NOT
        else:  # pragma: no cover — guarded by caller
            raise TypeError(f"Cannot lower {type(op).__name__} to RuntimeClassicalExpr")

        return RuntimeClassicalExpr(
            operands=list(op.operands),
            results=list(op.results),
            kind=kind,
        )

    # ------------------------------------------------------------------
    # Runtime-if merge lowering (merge → SELECT)
    # ------------------------------------------------------------------

    def _lower_if_merges(
        self,
        operations: list[Operation],
        tainted: set[str],
        while_protected: set[str],
    ) -> list[Operation]:
        """Lower runtime-if scalar classical merges to ``SELECT`` expressions.

        A runtime ``IfOperation``'s merge slot means ``result =
        true_value if condition else false_value`` — exactly a ternary
        classical expression over measurement-derived values. Representing
        it as ``RuntimeClassicalExpr(SELECT)`` right after the ``IfOperation``
        lets every existing ``RuntimeClassicalExpr`` mechanism apply
        unchanged: consumer-based segment placement, host-side evaluation by
        ``ClassicalExecutor``, and backend runtime-expression emission.
        Without this, the merge value is only reachable through emit-time
        clbit aliasing, which silently collapses to one branch's bit when
        the branch sources differ.

        Bodies are lowered bottom-up so an outer merge whose branch value is
        an inner merge sees the inner ``SELECT`` (an ordinary pure classical
        producer) instead of an ``IfOperation`` merge. Merges recorded inside ``For`` /
        ``ForItems`` bodies float past the loop when loop-invariant (see
        :meth:`_float_loop_invariant_exprs`); zero-trip semantics make
        ``While`` bodies ineligible for floating.

        Args:
            operations (list[Operation]): Operations of one lexical body.
            tainted (set[str]): Measurement-derived value UUIDs.
            while_protected (set[str]): UUIDs consumed as ``WhileOperation``
                condition operands; merges producing them stay on the if so
                the while-condition clbit machinery stays in charge.

        Returns:
            list[Operation]: The rewritten operation list.
        """
        new_ops: list[Operation] = []
        for op in operations:
            if isinstance(op, IfOperation):
                rewritten_if, appended = self._lower_single_if(
                    op, tainted, while_protected
                )
                new_ops.append(rewritten_if)
                new_ops.extend(appended)
                continue
            if isinstance(op, HasNestedOps):
                rebuilt = op.rebuild_nested(
                    [
                        self._lower_if_merges(body, tainted, while_protected)
                        for body in op.nested_op_lists()
                    ]
                )
                if isinstance(rebuilt, (ForOperation, ForItemsOperation)):
                    rebuilt, floated = self._float_loop_invariant_exprs(rebuilt)
                    new_ops.append(rebuilt)
                    new_ops.extend(floated)
                else:
                    new_ops.append(rebuilt)
                continue
            new_ops.append(op)
        return new_ops

    def _lower_single_if(
        self,
        op: IfOperation,
        tainted: set[str],
        while_protected: set[str],
    ) -> tuple[IfOperation, list[Operation]]:
        """Lower one ``IfOperation``'s eligible merges to ``SELECT`` exprs.

        Args:
            op (IfOperation): The if operation to process. Its bodies are
                lowered recursively first.
            tainted (set[str]): Measurement-derived value UUIDs.
            while_protected (set[str]): While-condition operand UUIDs.

        Returns:
            tuple[IfOperation, list[Operation]]: The if with lowered merges
                removed, and the operations to append right after it
                (fresh copies of branch-local pure producers followed by
                the ``SELECT`` expressions).
        """
        true_ops = self._lower_if_merges(op.true_operations, tainted, while_protected)
        false_ops = self._lower_if_merges(op.false_operations, tainted, while_protected)

        condition = op.operands[0] if op.operands else None
        appended: list[Operation] = []
        lowered_indices: set[int] = set()
        if (
            isinstance(condition, ValueBase)
            and not isinstance(condition, ArrayValue)
            and condition.uuid in tainted
        ):
            for merge in op.iter_merges():
                if not self._is_lowerable_merge(merge, while_protected):
                    continue
                true_group = self._copy_pure_closure([merge.true_value], true_ops)
                false_group = self._copy_pure_closure([merge.false_value], false_ops)
                if true_group is None or false_group is None:
                    continue
                true_copies, true_map = true_group
                false_copies, false_map = false_group
                appended.extend(true_copies)
                appended.extend(false_copies)
                appended.append(
                    RuntimeClassicalExpr(
                        operands=cast(
                            "list[Value]",
                            [
                                condition,
                                true_map.get(merge.true_value.uuid, merge.true_value),
                                false_map.get(
                                    merge.false_value.uuid, merge.false_value
                                ),
                            ],
                        ),
                        results=[merge.result],
                        kind=RuntimeOpKind.SELECT,
                    )
                )
                lowered_indices.add(merge.index)

        rewritten = dataclasses.replace(
            op,
            true_operations=true_ops,
            false_operations=false_ops,
            true_yields=[
                value
                for index, value in enumerate(op.true_yields)
                if index not in lowered_indices
            ],
            false_yields=[
                value
                for index, value in enumerate(op.false_yields)
                if index not in lowered_indices
            ],
            results=[
                value
                for index, value in enumerate(op.results)
                if index not in lowered_indices
            ],
        )
        return rewritten, appended

    def _is_lowerable_merge(self, merge: IfMerge, while_protected: set[str]) -> bool:
        """Return whether a merge slot can lower to a ``SELECT`` expression.

        Args:
            merge (IfMerge): The merge to classify.
            while_protected (set[str]): While-condition operand UUIDs.

        Returns:
            bool: ``True`` for a non-identity scalar classical merge that
                does not carry a while condition. Quantum and array merges
                stay on the if (emit-side physical-resource checks own
                them), identity merges resolve through their common source,
                and while-condition merges stay on the clbit-aliasing path.
        """
        output = merge.result
        if isinstance(output, ArrayValue) or output.type.is_quantum():
            return False
        true_value = merge.true_value
        false_value = merge.false_value
        if isinstance(true_value, ArrayValue) or isinstance(false_value, ArrayValue):
            return False
        if true_value.uuid == false_value.uuid:
            return False
        if output.uuid in while_protected:
            return False
        return True

    def _copy_pure_closure(
        self,
        seeds: list[ValueBase],
        body_ops: list[Operation],
    ) -> tuple[list[Operation], dict[str, ValueBase]] | None:
        """Copy the pure-classical producer closure of ``seeds`` out of a body.

        Branch-local pure classical producers (e.g. the ``~a`` behind
        ``out = ~a``) are duplicated with fresh result UUIDs so the
        ``SELECT`` synthesized after the ``IfOperation`` reads values that
        are visible outside the branch. The originals stay in the branch
        for any in-branch consumers; SSA holds because the copies produce
        new UUIDs. Values without a producer in the body (pre-branch
        values) and direct scalar Bit measurements are left as leaf
        references — the per-shot execution context resolves them. Other
        non-classical producers, such as QFixed measurements and expectation
        values, have no directly resolvable host carrier and make the merge
        ineligible.

        Args:
            seeds (list[ValueBase]): Values the ``SELECT`` needs.
            body_ops (list[Operation]): The branch body to copy from.

        Returns:
            tuple[list[Operation], dict[str, ValueBase]] | None: Ordered
                fresh copies and the old-to-fresh value mapping, or ``None``
                when a needed producer is a nested control-flow op that
                cannot be copied (the merge then stays on the if and
                segmentation reports it if a host consumer needs it).
        """
        producers: dict[str, Operation] = {}
        for body_op in body_ops:
            for result in body_op.results:
                if isinstance(result, ValueBase):
                    producers[result.uuid] = body_op

        needed_ids: set[int] = set()
        visited: set[str] = set()

        def require(value: ValueBase, *, direct_seed: bool) -> bool:
            """Mark ``value``'s in-body pure producer chain as needed.

            Args:
                value (ValueBase): Value the closure must resolve.
                direct_seed (bool): Whether ``value`` is selected directly by
                    the merge. A direct branch-local measurement can remain a
                    lazy SELECT operand; a copied producer may not depend on
                    branch-local quantum work.

            Returns:
                bool: ``False`` when the chain hits an uncopyable producer.
            """
            if value.uuid in visited:
                return True
            visited.add(value.uuid)
            producer = producers.get(value.uuid)
            if producer is None:
                return True
            if isinstance(producer, IfOperation):
                for nested_merge in producer.iter_merges():
                    if nested_merge.result.uuid == value.uuid:
                        nested_true = nested_merge.true_value
                        nested_false = nested_merge.false_value
                        if (
                            isinstance(nested_true, ValueBase)
                            and isinstance(nested_false, ValueBase)
                            and nested_true.uuid == nested_false.uuid
                        ):
                            return require(nested_true, direct_seed=direct_seed)
                        return False
                return False
            if isinstance(producer, HasNestedOps):
                return False
            if producer.operation_kind != OperationKind.CLASSICAL:
                return (
                    direct_seed
                    and isinstance(producer, MeasureOperation)
                    and isinstance(value, Value)
                    and isinstance(value.type, BitType)
                )
            if not self._is_total_hoist_candidate(producer):
                return False
            needed_ids.add(id(producer))
            for operand in producer.all_input_values():
                if isinstance(operand, ValueBase) and not require(
                    operand, direct_seed=False
                ):
                    return False
            return True

        for seed in seeds:
            if not require(seed, direct_seed=True):
                return None

        mapping: dict[str, ValueBase] = {}
        copies: list[Operation] = []
        for body_op in body_ops:
            if id(body_op) not in needed_ids:
                continue
            for result in body_op.results:
                if isinstance(result, ValueBase):
                    mapping[result.uuid] = result.next_version()
            copies.append(body_op.replace_values(mapping))
        return copies, mapping

    @staticmethod
    def _is_total_hoist_candidate(operation: Operation) -> bool:
        """Check whether a branch-local operation is safe to run eagerly.

        Hoisted operations execute before the SELECT chooses a branch, so
        they must be pure and total. Arithmetic that can fail for valid
        runtime inputs (division, modulo, exponentiation) and other
        classical operations with lookup or mutation semantics remain in
        the branch and cause a loud unsupported-merge diagnostic.

        Args:
            operation (Operation): Branch-local classical producer.

        Returns:
            bool: True when eager host-side evaluation preserves semantics.
        """
        if isinstance(operation, (CompOp, CondOp, NotOp)):
            return True
        if isinstance(operation, BinOp):
            return operation.kind in {
                BinOpKind.ADD,
                BinOpKind.SUB,
                BinOpKind.MUL,
            }
        if isinstance(operation, RuntimeClassicalExpr):
            return operation.kind in {
                RuntimeOpKind.EQ,
                RuntimeOpKind.NEQ,
                RuntimeOpKind.LT,
                RuntimeOpKind.LE,
                RuntimeOpKind.GT,
                RuntimeOpKind.GE,
                RuntimeOpKind.AND,
                RuntimeOpKind.OR,
                RuntimeOpKind.NOT,
                RuntimeOpKind.ADD,
                RuntimeOpKind.SUB,
                RuntimeOpKind.MUL,
            }
        return False

    def _float_loop_invariant_exprs(
        self,
        loop_op: Operation,
    ) -> tuple[Operation, list[Operation]]:
        """Float loop-invariant runtime expressions past a ``For`` loop.

        A ``SELECT`` synthesized for an if nested in a loop body reads
        loop-invariant operands in every supported shape (the merge sources
        are defined before the loop), so evaluating it once after the loop
        is equivalent to the traced last-iteration value — and makes the
        expression visible to top-level segment routing. An expression
        stays in the body when any transitive operand is produced inside
        the loop subtree or its result is consumed inside it. ``While``
        loops are never floated: a zero-trip while must leave the pre-loop
        value observable, which an unconditionally evaluated ``SELECT``
        would overwrite.

        Args:
            loop_op (Operation): A ``ForOperation`` / ``ForItemsOperation``
                whose bodies were already merge-lowered.

        Returns:
            tuple[Operation, list[Operation]]: The loop with floated
                expressions removed, and those expressions in body order.
        """
        assert isinstance(loop_op, (ForOperation, ForItemsOperation))
        body_lists = loop_op.nested_op_lists()

        subtree_produced: set[str] = set()

        def collect_produced(operations: list[Operation]) -> None:
            """Collect result UUIDs across the loop subtree.

            Args:
                operations (list[Operation]): Body operations to scan.
            """
            for body_op in operations:
                for result in body_op.results:
                    if isinstance(result, ValueBase):
                        subtree_produced.add(result.uuid)
                if isinstance(body_op, HasNestedOps):
                    for nested in body_op.nested_op_lists():
                        collect_produced(nested)

        for body in body_lists:
            collect_produced(body)

        candidates = [
            body_op
            for body in body_lists
            for body_op in body
            if isinstance(body_op, RuntimeClassicalExpr)
        ]
        floating = {id(body_op) for body_op in candidates}

        def reads(operations: list[Operation], uuids: set[str]) -> bool:
            """Return whether any non-floating subtree op reads ``uuids``.

            Args:
                operations (list[Operation]): Body operations to scan.
                uuids (set[str]): Result UUIDs of the floating set.
            """
            for body_op in operations:
                if id(body_op) not in floating:
                    for operand in body_op.all_input_values():
                        if isinstance(operand, ValueBase) and operand.uuid in uuids:
                            return True
                if isinstance(body_op, HasNestedOps) and any(
                    reads(nested, uuids) for nested in body_op.nested_op_lists()
                ):
                    return True
            return False

        changed = True
        while changed:
            changed = False
            floating_results = {
                result.uuid
                for body_op in candidates
                if id(body_op) in floating
                for result in body_op.results
                if isinstance(result, ValueBase)
            }
            for body_op in candidates:
                if id(body_op) not in floating:
                    continue
                for operand in body_op.all_input_values():
                    if (
                        isinstance(operand, ValueBase)
                        and operand.uuid in subtree_produced
                        and operand.uuid not in floating_results
                    ):
                        floating.discard(id(body_op))
                        changed = True
                        break
            floating_results = {
                result.uuid
                for body_op in candidates
                if id(body_op) in floating
                for result in body_op.results
                if isinstance(result, ValueBase)
            }
            if floating and any(reads(body, floating_results) for body in body_lists):
                # Some in-loop op reads a floating result; conservatively
                # keep every candidate whose result is read in-loop.
                for body_op in candidates:
                    if id(body_op) not in floating:
                        continue
                    result_uuids = {
                        result.uuid
                        for result in body_op.results
                        if isinstance(result, ValueBase)
                    }
                    if any(reads(body, result_uuids) for body in body_lists):
                        floating.discard(id(body_op))
                        changed = True

        if not floating:
            return loop_op, []

        floated = [
            body_op
            for body in body_lists
            for body_op in body
            if id(body_op) in floating
        ]
        rebuilt = loop_op.rebuild_nested(
            [
                [body_op for body_op in body if id(body_op) not in floating]
                for body in body_lists
            ]
        )
        return rebuilt, floated


def _collect_while_operand_uuids(operations: list[Operation]) -> set[str]:
    """Collect UUIDs carrying a ``WhileOperation`` loop-carried condition.

    The loop-carried condition update (``operands[1]``) is resolved by the
    emit-time clbit-aliasing machinery, which follows merge chains down to
    the underlying measurements (``validate_while`` /
    ``ResourceAllocator``). Every merge on such a chain must therefore
    stay on its ``IfOperation`` instead of lowering to a ``SELECT`` expression: the set
    contains the carried operands plus, transitively, the branch sources
    of every merge whose output is already protected. The *initial*
    condition (``operands[0]``) is deliberately not protected — a
    merged initial condition lowers to a ``SELECT`` and reaches the
    backend as an ordinary runtime expression, exactly like an ``if``
    condition.

    Args:
        operations (list[Operation]): Operations to scan recursively.

    Returns:
        set[str]: UUIDs of every value on a carried while-condition chain.
    """
    uuids: set[str] = set()
    merge_sources: dict[str, tuple[str, str]] = {}

    def scan(operations: list[Operation]) -> None:
        """Collect carried while operands and merge source pairs in one walk.

        Args:
            operations (list[Operation]): Operations to scan.
        """
        for op in operations:
            if isinstance(op, WhileOperation):
                for operand in op.operands[1:]:
                    if isinstance(operand, ValueBase):
                        uuids.add(operand.uuid)
            if isinstance(op, IfOperation):
                for merge in op.iter_merges():
                    merge_sources[merge.result.uuid] = (
                        merge.true_value.uuid,
                        merge.false_value.uuid,
                    )
            if isinstance(op, HasNestedOps):
                for body in op.nested_op_lists():
                    scan(body)

    scan(operations)

    worklist = [uuid for uuid in uuids if uuid in merge_sources]
    while worklist:
        current = worklist.pop()
        for source_uuid in merge_sources.get(current, ()):
            if source_uuid not in uuids:
                uuids.add(source_uuid)
                worklist.append(source_uuid)
    return uuids
