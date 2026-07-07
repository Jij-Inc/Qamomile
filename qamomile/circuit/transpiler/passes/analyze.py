"""Analyze pass: Validate and analyze dependencies in an affine block."""

from __future__ import annotations

import dataclasses
import numbers
from typing import Any

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import PhiOp
from qamomile.circuit.ir.operation.classical_ops import StoreArrayElementOperation
from qamomile.circuit.ir.operation.control_flow import (
    BranchRebind,
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.gate import (
    GateOperation,
    MeasureOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.operation.operation import OperationKind, QInitOperation
from qamomile.circuit.ir.value import ArrayValue, Value, ValueBase
from qamomile.circuit.transpiler.errors import (
    DependencyError,
    QubitRebindError,
    ValidationError,
)
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.passes.compile_time_if_lowering import (
    evaluate_classical_op_concrete,
    resolve_compile_time_condition,
)
from qamomile.circuit.transpiler.passes.control_flow_visitor import ControlFlowVisitor
from qamomile.circuit.transpiler.passes.validate_while import build_producer_map

# ---------------------------------------------------------------------------
# Public dataflow utilities
#
# These helpers were originally private methods of ``AnalyzePass`` but are
# exposed at module scope so that other passes (e.g. ``ClassicalLoweringPass``)
# can reuse the same measurement-taint analysis without instantiating
# ``AnalyzePass`` or duplicating the worklist algorithm.
# ---------------------------------------------------------------------------


def build_dependency_graph(operations: list[Operation]) -> dict[str, set[str]]:
    """Build a map from each value UUID to the UUIDs it depends on.

    Walks operations recursively (through ``HasNestedOps``) and records,
    for each result UUID, the set of operand UUIDs that produced it.
    ``IfOperation`` merge outputs get explicit edges to the condition and
    both branch sources via ``iter_merges`` — the builder does not rely
    on merge storage being reachable through the generic nested-list
    walk. Also seeds an edge from each ``ArrayValue`` element (``Value``
    carrying ``parent_array``) to its parent array UUID, and walks the
    parent's ``slice_of`` chain so that a sliced view (e.g. ``s[0:4:2][i]``
    for ``s = qmc.measure(register)``) inherits taint from the root
    measured array. ``StripSliceArrayOpsPass`` removes the explicit
    ``SliceArrayOperation`` boundary before this pass runs, so the
    ``slice_of`` link is the only remaining connection between a view and
    its root in the IR. Used downstream by measurement-taint analysis.

    Args:
        operations (list[Operation]): Top-level operations of the block.

    Returns:
        dict[str, set[str]]: Mapping ``result_uuid -> set(operand_uuid, ...)``.
    """

    class DependencyGraphBuilder(ControlFlowVisitor):
        def __init__(self):
            self.graph: dict[str, set[str]] = {}

        def visit_operation(self, op: Operation) -> None:
            """Record one operation's result-to-operand dependency edges.

            Args:
                op (Operation): The visited operation.
            """
            operand_uuids = {v.uuid for v in op.operands if isinstance(v, ValueBase)}
            for result in op.results:
                if result.uuid not in self.graph:
                    self.graph[result.uuid] = set()
                self.graph[result.uuid].update(operand_uuids)
            for v in op.operands:
                self._seed_structural_edges(v)
            if isinstance(op, IfOperation):
                # Explicit merge edges: each merged output depends on the
                # condition and both branch sources. A compile-time-constant
                # condition may be a raw Python bool (closure constant)
                # with no IR identity — and malformed IR may lack the
                # operand entirely — so it contributes no edge in either
                # case.
                condition = op.operands[0] if op.operands else None
                condition_uuid = (
                    condition.uuid if isinstance(condition, ValueBase) else None
                )
                for merge in op.iter_merges():
                    deps = self.graph.setdefault(merge.result.uuid, set())
                    if condition_uuid is not None:
                        deps.add(condition_uuid)
                    deps.add(merge.true_value.uuid)
                    deps.add(merge.false_value.uuid)
                    self._seed_structural_edges(merge.true_value)
                    self._seed_structural_edges(merge.false_value)

        def _seed_structural_edges(self, value: object) -> None:
            """Seed element-to-parent and slice-chain edges for one value.

            Args:
                value (object): Candidate operand / merge source; ignored
                    unless it is a ``ValueBase`` carrying ``parent_array``.
            """
            if not isinstance(value, ValueBase):
                return
            parent = getattr(value, "parent_array", None)
            if parent is None:
                return
            self.graph.setdefault(value.uuid, set()).add(parent.uuid)
            cur = parent
            while getattr(cur, "slice_of", None) is not None:
                self.graph.setdefault(cur.uuid, set()).add(cur.slice_of.uuid)
                cur = cur.slice_of

    builder = DependencyGraphBuilder()
    builder.visit_operations(operations)
    return builder.graph


def find_measurement_results(operations: list[Operation]) -> set[str]:
    """Find all value UUIDs that are direct results of measurement ops.

    Walks operations recursively (through ``HasNestedOps``) and collects
    every measurement result's UUID, covering both scalar
    ``MeasureOperation`` and vector ``MeasureVectorOperation``. The
    returned set is the seed for taint propagation; ``Vector[Bit]``
    element accesses inherit the parent array's taint through the
    parent-array edges added by ``build_dependency_graph``.

    Args:
        operations (list[Operation]): Top-level operations of the block.

    Returns:
        set[str]: Result UUIDs of every measurement operation.
    """

    class MeasurementResultCollector(ControlFlowVisitor):
        def __init__(self):
            self.result_uuids: set[str] = set()

        def visit_operation(self, op: Operation) -> None:
            if isinstance(op, (MeasureOperation, MeasureVectorOperation)):
                for result in op.results:
                    self.result_uuids.add(result.uuid)

    collector = MeasurementResultCollector()
    collector.visit_operations(operations)
    return collector.result_uuids


def find_measurement_derived_values(
    dependency_graph: dict[str, set[str]],
    measurement_uuids: set[str],
) -> set[str]:
    """Forward-propagate measurement taint through the dependency graph.

    Args:
        dependency_graph (dict[str, set[str]]): ``result_uuid ->
            set(operand_uuid, ...)`` as produced by
            ``build_dependency_graph``, including the parent-array /
            slice-of edges that connect measured-vector elements to the
            root measured array.
        measurement_uuids (set[str]): Seed set (results of
            ``MeasureOperation`` / ``MeasureVectorOperation`` collected by
            ``find_measurement_results``).

    Returns:
        set[str]: The set of all UUIDs transitively derived from a
            measurement, including the seeds themselves.
    """
    # Build a reverse adjacency map once (dependency -> dependents) so taint
    # propagation is a linear worklist traversal instead of rescanning the
    # whole graph for every popped node (which is O(N^2) and shows up as a
    # compile-time cost on large kernels, since this runs on every transpile).
    dependents: dict[str, list[str]] = {}
    for uuid, deps in dependency_graph.items():
        for dep in deps:
            dependents.setdefault(dep, []).append(uuid)

    derived: set[str] = set()
    worklist = list(measurement_uuids)
    while worklist:
        current = worklist.pop()
        if current in derived:
            continue
        derived.add(current)
        for dependent in dependents.get(current, ()):
            if dependent not in derived:
                worklist.append(dependent)
    return derived


@dataclasses.dataclass(frozen=True)
class PrunedIfView:
    """Pruned view of an operation list plus its dead-branch phi aliases.

    Produced by :func:`prune_compile_time_ifs`. A compile-time-resolved
    ``IfOperation`` disappears from ``operations`` (only its taken branch
    survives, inlined); each of its merge outputs is recorded here as a
    ``(result, selected_source)`` alias pair so phi-mediated dataflow out
    of the pruned branch stays visible to dependency scans without any
    synthetic operation in the list.

    Attributes:
        operations (list[Operation]): The pruned view of the input
            operations, in program order.
        phi_aliases (tuple[tuple[Value, Value], ...]): Every
            ``(result, selected_source)`` pair recorded anywhere in the
            walk, in pruning order.
        _loop_aliases (dict[int, tuple[tuple[Value, Value], ...]]): Alias
            pairs recorded inside each rebuilt loop operation's subtree,
            keyed by ``id()`` of the rebuilt loop op. The keyed objects
            are kept alive by ``operations``, so the ids are stable for
            this view's lifetime.
    """

    operations: list[Operation]
    phi_aliases: tuple[tuple[Value, Value], ...]
    _loop_aliases: dict[int, tuple[tuple[Value, Value], ...]]

    def aliases_for_loop(self, loop_op: Operation) -> tuple[tuple[Value, Value], ...]:
        """Return the alias pairs recorded inside one pruned loop's body.

        Args:
            loop_op (Operation): A loop operation taken from
                ``operations`` (or a body nested within it). Loop ops that
                were never walked — e.g. inside a kept runtime-if branch —
                have no recorded aliases.

        Returns:
            tuple[tuple[Value, Value], ...]: ``(result, selected_source)``
                pairs from compile-time ifs pruned anywhere inside the
                loop's body, or an empty tuple.
        """
        return self._loop_aliases.get(id(loop_op), ())


def prune_compile_time_ifs(
    ops: list[Operation],
    concrete_values: dict[str, Any],
    bindings: dict[str, Any],
    *,
    walk_runtime_branches: bool = False,
) -> PrunedIfView:
    """Replace compile-time-decidable ``IfOperation``s by their taken branch.

    Mirrors ``CompileTimeIfLoweringPass``: conditions are resolved with
    the shared ``resolve_compile_time_condition`` /
    ``evaluate_classical_op_concrete`` helpers so the taken / dead /
    runtime classification here cannot disagree with the branch the
    lowering pass will actually keep. For a resolved condition the taken
    branch's operations are inlined (recursively pruned) and each merge
    output is recorded as a ``(result, selected_source)`` alias pair, so
    phi-mediated dataflow out of the branch stays visible to dependency
    scans without dead-branch edges. Runtime ``IfOperation``s are kept by
    default with their branches untouched; with
    ``walk_runtime_branches=True`` their branch bodies are pruned in
    place (each side with its own copy of the accumulated state, exactly
    like the lowering pass) while the if itself and its merges stay.

    Shared by ``reject_self_referential_loop_stores`` and
    ``reject_loop_carried_classical_rebinds`` — both checks must classify
    conditions exactly the way the lowering pass does.

    Args:
        ops (list[Operation]): Operations to prune, in program order.
        concrete_values (dict[str, Any]): UUID-keyed concrete
            classical-op results accumulated along the walk.
            Updated in place (nested non-if bodies get a copy,
            matching the lowering pass's scoping).
        bindings (dict[str, Any]): Compile-time parameter bindings used
            to resolve conditions.
        walk_runtime_branches (bool): When ``True``, descend into kept
            runtime ``IfOperation`` branches so compile-time ifs nested
            inside them are pruned (and their merge aliases recorded)
            too — the lowering pass lowers those, so scans that must
            match its output need this. ``False`` (default) preserves
            the historical view where runtime branches pass through
            verbatim, which the quantum discard checks rely on for
            their own position-aware classification. Defaults to False.

    Returns:
        PrunedIfView: The pruned operations together with the recorded
            dead-branch phi alias pairs (global and per pruned loop op).
    """
    global_aliases: list[tuple[Value, Value]] = []
    loop_aliases: dict[int, tuple[tuple[Value, Value], ...]] = {}

    def walk(
        ops: list[Operation],
        concrete_values: dict[str, Any],
        sink: list[tuple[Value, Value]],
    ) -> list[Operation]:
        """Prune one operation list, recording aliases into ``sink``.

        Args:
            ops (list[Operation]): Operations to prune.
            concrete_values (dict[str, Any]): Concrete-result map for
                this scope; updated in place.
            sink (list[tuple[Value, Value]]): Alias accumulator of the
                nearest enclosing loop (or the global one).

        Returns:
            list[Operation]: The pruned view of ``ops``.
        """
        pruned: list[Operation] = []
        for op in ops:
            evaluate_classical_op_concrete(op, concrete_values, bindings)
            if isinstance(op, IfOperation):
                # A malformed condition-less if resolves as runtime (None
                # coerces to no compile-time value) and passes through.
                taken = resolve_compile_time_condition(
                    op.operands[0] if op.operands else None,
                    concrete_values,
                    bindings,
                )
                if taken is None:
                    if walk_runtime_branches:
                        # Mirror the lowering pass: each runtime branch is
                        # classified with its own copy of the accumulated
                        # state; the if itself and its merges stay.
                        op = dataclasses.replace(
                            op,
                            true_operations=walk(
                                op.true_operations, dict(concrete_values), sink
                            ),
                            false_operations=walk(
                                op.false_operations, dict(concrete_values), sink
                            ),
                        )
                    pruned.append(op)
                    continue
                branch = op.true_operations if taken else op.false_operations
                pruned.extend(walk(branch, concrete_values, sink))
                for merge in op.iter_merges():
                    sink.append((merge.result, merge.select(taken)))
                continue
            if isinstance(op, HasNestedOps):
                if isinstance(op, (ForOperation, ForItemsOperation, WhileOperation)):
                    # Collect the loop subtree's aliases separately so the
                    # loop-scoped checks see exactly the pairs from ifs
                    # inside this body — pre-loop pairs must not leak in.
                    subtree: list[tuple[Value, Value]] = []
                    op = op.rebuild_nested(
                        [
                            walk(body, dict(concrete_values), subtree)
                            for body in op.nested_op_lists()
                        ]
                    )
                    loop_aliases[id(op)] = tuple(subtree)
                    sink.extend(subtree)
                else:
                    op = op.rebuild_nested(
                        [
                            walk(body, dict(concrete_values), sink)
                            for body in op.nested_op_lists()
                        ]
                    )
            pruned.append(op)
        return pruned

    pruned_ops = walk(ops, concrete_values, global_aliases)
    return PrunedIfView(
        operations=pruned_ops,
        phi_aliases=tuple(global_aliases),
        _loop_aliases=loop_aliases,
    )


def flatten_ops(
    ops: list[Operation],
    *,
    into_if_branches: bool = True,
) -> list[Operation]:
    """Flatten operations recursively through nested control flow.

    Args:
        ops (list[Operation]): Operations to flatten.
        into_if_branches (bool): When ``True`` (default), recurse
            into ``IfOperation`` bodies too.  ``False`` skips them —
            used by the self-referential store check, since stores
            inside a (runtime) if branch are rejected by
            ``AnalyzePass._reject_stores_in_if_branches`` instead.

    Returns:
        list[Operation]: All reachable operations, including the
            control flow ops themselves.
    """
    flat: list[Operation] = []
    for op in ops:
        flat.append(op)
        if not into_if_branches and isinstance(op, IfOperation):
            continue
        if isinstance(op, HasNestedOps):
            for body in op.nested_op_lists():
                flat.extend(flatten_ops(body, into_if_branches=into_if_branches))
    return flat


def reject_self_referential_loop_stores(
    operations: list[Operation],
    bindings: dict[str, Any] | None = None,
) -> None:
    """Reject in-loop classical element stores that read the same array.

    A loop body is traced once, so a ``StoreArrayElementOperation``
    inside a loop references a fixed pre-loop version of the array it
    writes.  If the stored value or the store index reads an element of
    that same logical array — directly or through classical arithmetic —
    later iterations would observe stale pre-loop contents instead of
    earlier iterations' writes, silently diverging from Python semantics
    (e.g. ``vals[i] = vals[0] + 1`` would write the same folded value
    every iteration).  Such stores are rejected at compile time.

    ``IfOperation``s are classified with the same condition resolution
    ``CompileTimeIfLoweringPass`` uses (via ``bindings``): a compile-time
    condition contributes only its taken branch to the scan (a dead
    branch is eliminated by the lowering pass, so a self-referential
    store inside it never executes), while a runtime condition's
    branches are skipped entirely — every store inside a runtime
    if branch is rejected by ``AnalyzePass._reject_stores_in_if_branches``
    regardless of self-reference.

    Exposed at module scope because it must run from two passes:
    ``PartialEvaluationPass`` calls it *before* constant folding (folding
    a bound element read to a constant erases the ``parent_array``
    provenance this check relies on — the fold is exactly what bakes the
    stale pre-loop value into the loop body), and ``AnalyzePass`` calls
    it again as a safety net for pipelines that skip ``partial_eval``.

    Args:
        operations (list[Operation]): Operations to scan.  Recurses
            through all control flow; every loop at any nesting depth is
            checked against the stores inside its body (if branches per
            the classification above).
        bindings (dict[str, Any] | None): Compile-time parameter bindings
            used to resolve ``IfOperation`` conditions, matching what
            ``CompileTimeIfLoweringPass`` will later resolve.  Defaults
            to None (no bindings): only constant conditions resolve and
            all others are treated as runtime — correct for the
            ``AnalyzePass`` safety-net call, where compile-time ifs are
            already lowered away and any store left inside an if branch
            was already rejected.

    Raises:
        ValidationError: If a store inside a loop transitively reads an
            element of the same logical array it writes.
    """
    resolved_bindings = bindings or {}

    def register_value(value: ValueBase, table: dict[str, ValueBase]) -> None:
        """Record a value and its structural references in ``table``.

        Recurses through ``element_indices``, ``parent_array``, and
        ``slice_of`` chains so the BFS below can resolve every UUID the
        dependency graph may reach back to a ``Value``.

        Args:
            value (ValueBase): Value to record, keyed by UUID.
            table (dict[str, ValueBase]): Mutable UUID-to-value map.
        """
        if value.uuid in table:
            return
        table[value.uuid] = value
        for index in getattr(value, "element_indices", ()):
            register_value(index, table)
        parent = getattr(value, "parent_array", None)
        if parent is not None:
            register_value(parent, table)
        slice_of = getattr(value, "slice_of", None)
        if slice_of is not None:
            register_value(slice_of, table)

    def reads_written_array(value: ValueBase, written_logical: str) -> bool:
        """Check whether a value is an element read of the written array.

        Walks the ``parent_array`` / ``slice_of`` chain so element reads
        through strided views of the written array are caught too.

        Args:
            value (ValueBase): Value to inspect.
            written_logical (str): ``logical_id`` of the array the store
                writes.

        Returns:
            bool: ``True`` if the value reads an element of the written
                logical array.
        """
        chain = getattr(value, "parent_array", None)
        while chain is not None:
            if chain.logical_id == written_logical:
                return True
            chain = getattr(chain, "slice_of", None)
        return False

    def check_loop_body(
        body_ops: list[Operation],
        phi_aliases: tuple[tuple[Value, Value], ...],
    ) -> None:
        """Reject self-referential stores inside one (pruned) loop body.

        Builds a dependency graph restricted to the loop body (reads
        performed before the loop are loop-invariant and fold correctly,
        so they must not trigger a rejection) and BFS-walks it from each
        store's value and index operands.  The graph and value table
        cover the full body — including surviving runtime-if internals
        and the alias pairs of pruned compile-time ifs, so a store after
        an if still sees phi-mediated reads — while the store scan itself
        skips if branches (those stores are rejected by
        ``AnalyzePass._reject_stores_in_if_branches``).

        Args:
            body_ops (list[Operation]): The loop's top-level body
                operations, already pruned of compile-time-decidable
                if branches.
            phi_aliases (tuple[tuple[Value, Value], ...]):
                ``(result, selected_source)`` pairs of the compile-time
                ifs pruned inside this loop's body; each contributes a
                dataflow edge from the merge output to its surviving
                source.

        Raises:
            ValidationError: If a scanned store's value or index
                transitively reads an element of the array the store
                writes.
        """
        dependency_graph = build_dependency_graph(body_ops)
        for result, source in phi_aliases:
            dependency_graph.setdefault(result.uuid, set()).add(source.uuid)
        flat_ops = flatten_ops(body_ops)

        value_table: dict[str, ValueBase] = {}
        for op in flat_ops:
            for value in (*op.all_input_values(), *op.results):
                register_value(value, value_table)
        for result, source in phi_aliases:
            register_value(result, value_table)
            register_value(source, value_table)

        for op in flatten_ops(body_ops, into_if_branches=False):
            if not isinstance(op, StoreArrayElementOperation):
                continue
            written_logical = op.results[0].logical_id
            worklist = [v.uuid for v in (op.stored_value, *op.index_values)]
            visited: set[str] = set()
            while worklist:
                current = worklist.pop()
                if current in visited:
                    continue
                visited.add(current)
                value = value_table.get(current)
                if value is not None:
                    if reads_written_array(value, written_logical):
                        raise ValidationError(
                            f"Classical array element assignment into "
                            f"'{op.array.name}' inside a loop reads an "
                            f"element of the same array (directly or "
                            f"through classical arithmetic): the loop "
                            f"body references a fixed pre-loop array "
                            f"version, so the read would not observe "
                            f"earlier iterations' writes. Restructure "
                            f"the kernel to read from a different array "
                            f"or perform the update outside the loop."
                        )
                    # Follow structural references so same-array reads
                    # hiding inside an element address are found too.
                    worklist.extend(
                        index.uuid for index in getattr(value, "element_indices", ())
                    )
                    parent = getattr(value, "parent_array", None)
                    if parent is not None:
                        worklist.append(parent.uuid)
                worklist.extend(dependency_graph.get(current, ()))

    pruned = prune_compile_time_ifs(operations, {}, resolved_bindings)
    for op in flatten_ops(pruned.operations, into_if_branches=False):
        if isinstance(op, (ForOperation, ForItemsOperation, WhileOperation)):
            check_loop_body(
                [body_op for body in op.nested_op_lists() for body_op in body],
                pruned.aliases_for_loop(op),
            )


_LOOP_KIND_NAMES: dict[type, str] = {
    ForOperation: "for",
    ForItemsOperation: "for-items",
    WhileOperation: "while",
}


def _loop_carried_rebind_error(var_name: str, loop_kind: str) -> ValidationError:
    """Build the targeted loop-carried rebind rejection error.

    Args:
        var_name (str): Display name of the rebound variable.
        loop_kind (str): Human-readable loop kind ("for" / "while" /
            "for-items").

    Returns:
        ValidationError: The error to raise.
    """
    return ValidationError(
        f"Loop-carried update of classical variable '{var_name}' inside a "
        f"qkernel {loop_kind} loop is not supported: the loop body is "
        f"traced once, so '{var_name}' on the right-hand side is fixed to "
        f"its pre-loop value instead of the previous iteration's value, "
        f"and the compiled program would silently diverge from Python "
        f"semantics. Compute the reduction in ordinary Python instead — "
        f"outside the qkernel or in an undecorated helper function — or "
        f"express each iteration's value directly from the loop index. "
        f"Note: builtin range() inside qkernel is traced exactly like "
        f"qmc.range()."
    )


def _op_read_uuids(op: Operation) -> set[str]:
    """Collect the uuids an operation genuinely reads.

    Loop operations expose their loop-carried rebind records — and
    ``IfOperation``s their branch rebind records — through
    ``all_input_values`` (for cloning); those record values are not
    reads and must not trigger read-based checks, so they are
    subtracted before the operands are re-added.

    Args:
        op (Operation): Operation to inspect.

    Returns:
        set[str]: UUIDs of values the operation reads.
    """
    excluded: set[str] = set()
    if isinstance(op, (ForOperation, ForItemsOperation, WhileOperation)):
        for r in op.loop_carried_rebinds:
            excluded.add(r.before.uuid)
            excluded.add(r.after.uuid)
    if isinstance(op, IfOperation):
        for branch_record in op.branch_rebinds:
            excluded.add(branch_record.before.uuid)
    uuids = {v.uuid for v in op.all_input_values()}
    uuids -= excluded
    for v in op.operands:
        operand_uuid = getattr(v, "uuid", None)
        if operand_uuid is not None:
            uuids.add(operand_uuid)
    return uuids


def _stale_condition_alias_family(
    pruned: PrunedIfView,
    loop_op: WhileOperation,
    initial_uuid: str,
) -> set[str]:
    """Collect the initial-condition UUID plus its pruned-merge aliases.

    A compile-time if whose taken branch passes the initial condition
    through (``y = snapshot``) leaves no operation in the pruned view —
    only a ``(result, selected_source)`` alias pair. Reads of such a
    merge output observe the same physical clbit as the initial
    condition, so they count as reads of the initial value. Aliases
    recorded inside the while's own body are excluded: in-body reads are
    correct under the loop-carried aliasing, matching the body-operation
    exclusion in the caller's scan.

    Args:
        pruned (PrunedIfView): The pruned program view whose alias pairs
            are expanded.
        loop_op (WhileOperation): The loop-carried while being checked.
        initial_uuid (str): UUID of the initial condition value.

    Returns:
        set[str]: ``initial_uuid`` plus every pruned-merge output UUID
            that transitively selects it (body-internal merges excluded).
    """
    body_pairs = {
        (result.uuid, source.uuid)
        for result, source in pruned.aliases_for_loop(loop_op)
    }
    candidates = [
        (result.uuid, source.uuid)
        for result, source in pruned.phi_aliases
        if (result.uuid, source.uuid) not in body_pairs
    ]
    family = {initial_uuid}
    changed = True
    while changed:
        changed = False
        for result_uuid, source_uuid in candidates:
            if source_uuid in family and result_uuid not in family:
                family.add(result_uuid)
                changed = True
    return family


def _reject_stale_while_condition_reads(
    pruned: PrunedIfView,
    output_uuids: set[str],
) -> None:
    """Reject post-loop reads of a while condition's pre-loop value.

    For a loop-carried while condition, the resource allocator aliases
    the initial condition, every in-loop re-measurement, and the merged
    phi outputs onto ONE physical classical bit. Reads inside the loop
    body are correct under that aliasing (they observe the current
    value, matching Python). A read of the *initial* condition value
    AFTER the loop is not: Python semantics promise the pre-loop (or
    entry-of-final-iteration) snapshot that a Python-level alias like
    ``out = bit`` captured, but the shared clbit holds the final in-loop
    measurement by then. Measured divergence: both the body-saved and
    the pre-loop-saved snapshot kernels return 0 where Python gives 1.

    Reads through a pruned compile-time merge count too: a merge output
    that transitively selects the initial condition shares its clbit, so
    an operation (or block output) observing that merge output after the
    loop observes the same stale physical bit (see
    :func:`_stale_condition_alias_family`).

    Args:
        pruned (PrunedIfView): Program operations with compile-time-dead
            if branches already pruned, plus the recorded merge aliases.
        output_uuids (set[str]): UUIDs of the block's output values
            (post-loop reads through the kernel return path).

    Raises:
        ValidationError: If the initial-condition value of a
            loop-carried while — directly or through a pruned-merge
            alias — is read by any operation after the loop or escapes
            through the block outputs.
    """
    flat = flatten_ops(pruned.operations)
    for index, op in enumerate(flat):
        if not isinstance(op, WhileOperation) or len(op.operands) < 2:
            continue
        initial = op.operands[0]
        initial_uuid = getattr(initial, "uuid", None)
        if initial_uuid is None:
            continue
        family = _stale_condition_alias_family(pruned, op, initial_uuid)
        # Pre-order flattening lists the loop body right after the loop
        # op; skip those entries (in-body reads are correct under the
        # aliasing) and scan only what follows the loop.
        body_ids = {
            id(body_op)
            for body in op.nested_op_lists()
            for body_op in flatten_ops(body)
        }
        stale_read = any(
            not _op_read_uuids(later).isdisjoint(family)
            for later in flat[index + 1 :]
            if id(later) not in body_ids
        )
        if stale_read or not family.isdisjoint(output_uuids):
            name = getattr(initial, "name", "") or "<condition>"
            raise ValidationError(
                f"Loop-carried while-condition '{name}': the pre-loop "
                f"value of the condition is read after the loop. The "
                f"initial condition and its in-loop re-measurements are "
                f"aliased onto one classical bit, so a post-loop read "
                f"would observe the final measurement instead of the "
                f"snapshot Python semantics promise. Read the updated "
                f"condition variable itself after the loop, or restructure "
                f"the kernel so the snapshot is not needed."
            )


def _check_loop_carried_rebinds(
    loop_op: ForOperation | ForItemsOperation | WhileOperation,
    body_phi_aliases: tuple[tuple[Value, Value], ...],
) -> None:
    """Reject the loop-carried rebind records of one (pruned) loop op.

    Args:
        loop_op (ForOperation | ForItemsOperation | WhileOperation): Loop
            operation whose body has already been pruned of
            compile-time-decidable if branches.
        body_phi_aliases (tuple[tuple[Value, Value], ...]):
            ``(result, selected_source)`` pairs of the compile-time ifs
            pruned inside this loop's body. Body-local by design: the
            canonical chain below must stop at the pre-loop value, so
            pre-loop merge aliases must not leak in.

    Raises:
        ValidationError: If a recorded rebind survives dead-branch
            canonicalization and either reads its pre-loop value in the
            body, was initialized from a plain Python number, or swaps
            with another rebound variable.
    """
    records = loop_op.loop_carried_rebinds
    if not records:
        return

    loop_kind = _LOOP_KIND_NAMES.get(type(loop_op), "for")
    body_ops = [o for body in loop_op.nested_op_lists() for o in body]
    flat_body = flatten_ops(body_ops)

    # Dead-branch merge aliases recorded by pruning:
    # result uuid -> selected source uuid.
    collapsed: dict[str, str] = {
        result.uuid: source.uuid for result, source in body_phi_aliases
    }
    value_table: dict[str, ValueBase] = {}
    for op in flat_body:
        for v in (*op.all_input_values(), *op.results):
            value_table.setdefault(v.uuid, v)
    for result, source in body_phi_aliases:
        value_table.setdefault(result.uuid, result)
        value_table.setdefault(source.uuid, source)

    def canonical(uuid: str) -> str:
        """Follow dead-branch merge aliases to the underlying source uuid.

        Args:
            uuid (str): Starting value uuid.

        Returns:
            str: The uuid after following pruned-merge alias links.
        """
        seen: set[str] = set()
        while uuid in collapsed and uuid not in seen:
            seen.add(uuid)
            uuid = collapsed[uuid]
        return uuid

    body_read_uuids: set[str] = set()
    for op in flat_body:
        body_read_uuids |= _op_read_uuids(op)
    # A pruned merge reads its selected source exactly like the collapsed
    # phi it replaces (e.g. a dead-branch ``y = x`` emits no operation but
    # still reads the stale pre-loop ``x`` through the merge).
    body_read_uuids.update(source.uuid for _, source in body_phi_aliases)

    before_uuids = {r.before.uuid for r in records}

    for record in records:
        # Quantum records are the loop-body quantum discard check's
        # domain (``_check_loop_quantum_discards``); the staleness rules
        # below model classical traced-once divergence only.
        if record.before.type.is_quantum():
            continue
        # Legal while-loop loop-carried condition: the (initial, updated)
        # condition pair is the ONLY measurement-backed rebind the
        # allocator aliases onto one clbit
        # (``ResourceAllocator._alias_loop_carried_clbits`` fires solely
        # for ``WhileOperation.operands[1]``). Any other measurement-
        # backed Bit rebind — in a ``for`` / ``for-items`` body, or a
        # non-condition variable in a ``while`` body — has no aliasing
        # machinery: reads of the variable keep addressing the pre-loop
        # clbit while the branch measurements write elsewhere, so the
        # compiled program silently diverges from Python semantics.
        # Those rebinds fall through to the rejection rules below.
        if (
            isinstance(loop_op, WhileOperation)
            and len(loop_op.operands) >= 2
            and record.before.uuid == loop_op.operands[0].uuid
            and record.after.uuid == loop_op.operands[1].uuid
        ):
            continue

        canon_uuid = canonical(record.after.uuid)

        # Dead-branch no-op: the surviving branch passes the pre-loop
        # value straight through.
        if canon_uuid == record.before.uuid:
            continue

        canon_value = value_table.get(canon_uuid)
        if canon_value is None and canon_uuid == record.after.uuid:
            canon_value = record.after
        before_const = record.before.get_const()
        canon_const = (
            canon_value.get_const() if isinstance(canon_value, Value) else None
        )
        if before_const is not None and canon_const is not None:
            # Dead-branch constant pass-through: the surviving branch
            # selects a constant equal to the initial value.
            if canon_const == before_const:
                continue
            # Trace-time-folded accumulation: an all-constant update like
            # ``total = total + 1.0`` folds during tracing, so the body
            # carries no BinOp reading the pre-loop value — the changed
            # constant is the only remaining evidence. One folded
            # application can never represent N iterations.
            raise _loop_carried_rebind_error(record.var_name, loop_kind)

        # Swap / rotation staleness: this variable's new value is another
        # rebound variable's pre-loop value.
        if canon_uuid != record.before.uuid and canon_uuid in before_uuids:
            raise _loop_carried_rebind_error(record.var_name, loop_kind)

        # Direct read of the pre-loop value anywhere in the body.
        if record.before.uuid in body_read_uuids:
            raise _loop_carried_rebind_error(record.var_name, loop_kind)

        # Plain-Python-number initialization: the stale read is an
        # embedded constant with no uuid, so the AST-certified
        # read-before-write is the evidence.
        if record.before_synthesized:
            raise _loop_carried_rebind_error(record.var_name, loop_kind)


def reject_loop_carried_classical_rebinds(
    operations: list[Operation],
    bindings: dict[str, Any] | None = None,
    output_values: list[Value] | None = None,
) -> None:
    """Reject in-loop classical scalar rebinds that cannot compile correctly.

    A loop body is traced once, so a Python-level reassignment like
    ``total = total + i`` inside a ``qmc.range`` / ``while`` /
    ``qmc.items`` loop reads a fixed pre-loop value instead of the
    previous iteration's value. Every executor (the classical segment
    interpreter and emit-time unrolling) re-runs the same traced
    operations per iteration, so the program silently diverges from
    Python semantics (e.g. ``total`` ends as ``0 + i_last`` instead of
    the sum). The frontend records candidate rebinds on the loop
    operations (``LoopCarriedRebind``); this check rejects the classical
    ones that survive dead-branch pruning. Records whose ``before`` is
    quantum model state discard, not traced-once divergence — they are
    skipped here and rejected by
    :func:`reject_control_flow_quantum_discard` instead.

    ``IfOperation``s are classified with the same condition resolution
    ``CompileTimeIfLoweringPass`` uses (via ``bindings``): a rebind whose
    only path is a compile-time-dead branch canonicalizes back to the
    pre-loop value and is allowed. Unlike the array-store check, loops
    nested inside *runtime* if branches are scanned too — a loop-carried
    scalar rebind there miscompiles all the same — and the pruning walk
    descends into those branches (``walk_runtime_branches=True``) so
    dead-branch canonicalization applies to them exactly as the lowering
    pass will lower them.

    The one exempted rebind — the while loop-carried condition pair —
    additionally requires that the condition's pre-loop value is not
    read after the loop (see ``_reject_stale_while_condition_reads``):
    the allocator aliases the whole condition series onto one classical
    bit, so a post-loop read of the initial value would observe the
    final in-loop measurement instead of the snapshot Python promises.

    Exposed at module scope because it must run from two passes:
    ``PartialEvaluationPass`` calls it before constant folding (folding
    an all-constant accumulation like ``total = total + 1`` erases the
    dependency evidence while keeping the wrong result), and
    ``AnalyzePass`` calls it again as a safety net for pipelines that
    skip ``partial_eval``.

    Args:
        operations (list[Operation]): Operations to scan. Recurses
            through all control flow.
        bindings (dict[str, Any] | None): Compile-time parameter bindings
            used to resolve ``IfOperation`` conditions, matching what
            ``CompileTimeIfLoweringPass`` will later resolve. Defaults to
            None (no bindings).
        output_values (list[Value] | None): The block's output values;
            a while condition's pre-loop value escaping through them is
            a post-loop read. Defaults to None (no outputs known).

    Raises:
        ValidationError: If a loop body rebinds a classical scalar whose
            pre-loop value the body still reads (directly, through
            classical arithmetic, through a surviving phi, or as an
            embedded constant from a plain-Python initialization), or if
            a while condition's pre-loop value is read after the loop.
    """
    resolved_bindings = bindings or {}
    pruned = prune_compile_time_ifs(
        operations, {}, resolved_bindings, walk_runtime_branches=True
    )
    output_uuids = {
        v.uuid for v in (output_values or []) if getattr(v, "uuid", None) is not None
    }
    _reject_stale_while_condition_reads(pruned, output_uuids)
    for op in flatten_ops(pruned.operations):
        if isinstance(op, (ForOperation, ForItemsOperation, WhileOperation)):
            _check_loop_carried_rebinds(op, pruned.aliases_for_loop(op))


def _branch_quantum_discard_error(
    var_name: str,
    branch_label: str,
) -> QubitRebindError:
    """Build the targeted branch-internal quantum discard rejection error.

    Raises the same ``QubitRebindError`` (an ``AffineTypeError``) that the
    decoration-time analyzer raises for a top-level rebind from a
    different quantum source: this check is the runtime, branch-internal
    manifestation of exactly that affine violation (a quantum value
    silently discarded rather than consumed once), so it shares the class
    rather than raising a generic ``ValidationError``.

    Args:
        var_name (str): Display name of the rebound quantum variable.
        branch_label (str): Human-readable branch side ("true" / "false").

    Returns:
        QubitRebindError: The error to raise, with ``handle_name`` set to
            the rebound variable's display name.
    """
    display_name = var_name or "<anonymous>"
    return QubitRebindError(
        f"Branch-internal quantum rebind of '{display_name}' in the "
        f"{branch_label} branch of a runtime if conditionally discards its "
        f"pre-branch quantum state: when the branch is taken, "
        f"'{display_name}' is rebound to a different quantum value while "
        f"the original state is neither consumed in that branch nor merged "
        f"out of it, so it would be silently dropped at runtime. Consume "
        f"the original state before branching or inside the branch (e.g. "
        f"qmc.measure(...)), or rebind the variable through gates on the "
        f"same qubit(s) instead of substituting a different quantum value.",
        handle_name=display_name,
    )


def _loop_quantum_discard_error(
    var_name: str,
    loop_kind: str,
) -> QubitRebindError:
    """Build the targeted loop-body quantum discard rejection error.

    The loop-body counterpart of :func:`_branch_quantum_discard_error`:
    the same ``QubitRebindError`` (an ``AffineTypeError``), because a
    loop-body rebind that never consumes the incoming value silently
    drops quantum state every iteration the loop runs — the decoration-
    time analyzer's top-level rebind violation surfacing inside a loop.

    Args:
        var_name (str): Display name of the rebound quantum variable.
        loop_kind (str): Human-readable loop kind ("for" / "while" /
            "for-items").

    Returns:
        QubitRebindError: The error to raise, with ``handle_name`` set to
            the rebound variable's display name.
    """
    display_name = var_name or "<anonymous>"
    if loop_kind == "while":
        return QubitRebindError(
            f"Loop-body quantum rebind of '{display_name}' inside a "
            f"qkernel while loop cannot compile correctly: the runtime "
            f"loop re-executes the body on one persistent register "
            f"without reset, so a register allocated in the body is not "
            f"fresh on later iterations and a body read of the pre-loop "
            f"value re-executes on a stale register. Use a body-local "
            f"name for a register the body allocates and measures "
            f"(instead of rebinding '{display_name}'), or rebind "
            f"'{display_name}' through gates on the same qubit(s) "
            f"instead of substituting a different quantum value.",
            handle_name=display_name,
        )
    return QubitRebindError(
        f"Loop-body quantum rebind of '{display_name}' inside a qkernel "
        f"{loop_kind} loop cannot compile correctly: the body is traced "
        f"once and re-instantiated per iteration without carrying the "
        f"rebound register between iterations, so later iterations "
        f"re-read the traced pre-loop register and the replaced state is "
        f"silently dropped or mis-measured at runtime. Rebind "
        f"'{display_name}' through gates on the same qubit(s) instead of "
        f"substituting a different quantum value; per-iteration "
        f"re-allocation is not supported until loops carry values "
        f"formally.",
        handle_name=display_name,
    )


def _loop_nonquantum_overwrite_error(
    var_name: str,
    loop_kind: str,
) -> QubitRebindError:
    """Build the loop-body non-quantum overwrite rejection error.

    Covers both a post-body binding with no IR value at all (an opaque
    classical call result, a plain constant, ``None``) and a classical
    IR value produced by consuming the register in place
    (``q = qmc.measure(q)`` — which additionally re-executes the
    measurement against the traced register every iteration when
    unrolled). Same ``QubitRebindError`` family and message prefix as
    the quantum-rebind rejection so callers and tests match uniformly.

    Args:
        var_name (str): Display name of the overwritten quantum variable.
        loop_kind (str): Human-readable loop kind ("for" / "while" /
            "for-items").

    Returns:
        QubitRebindError: The error to raise, with ``handle_name`` set to
            the overwritten variable's display name.
    """
    display_name = var_name or "<anonymous>"
    return QubitRebindError(
        f"Loop-body quantum rebind of '{display_name}' inside a qkernel "
        f"{loop_kind} loop overwrites the quantum variable with a "
        f"non-quantum value: the incoming register's state is dropped "
        f"(and a consuming overwrite like 'q = qmc.measure(q)' would "
        f"re-execute against the traced pre-loop register on later "
        f"iterations). Keep the classical result under a different "
        f"name, consume the state before the loop, or rebind "
        f"'{display_name}' through gates on the same qubit(s).",
        handle_name=display_name,
    )


def _while_zero_trip_rebind_error(var_name: str) -> QubitRebindError:
    """Build the while-loop zero-trip rebind divergence rejection error.

    A ``while`` loop's trip count is a runtime measurement outcome, so
    the zero-trip path is always live. A post-loop read of a variable
    the body rebinds must observe the pre-loop state on that path, but
    the emitted circuit binds the read to the rebound value
    unconditionally — the pre-loop state is silently ignored (discarded)
    exactly when the body never runs. Same ``QubitRebindError`` family
    as the unconditional discard. Static ``for`` loops do not carry this
    dedicated message: their body-produced rebinds are rejected
    unconditionally anyway, and a zero-trip static loop's post-loop
    reads were measured to resolve to the pre-loop values.

    Args:
        var_name (str): Display name of the rebound quantum variable.

    Returns:
        QubitRebindError: The error to raise, with ``handle_name`` set to
            the rebound variable's display name.
    """
    display_name = var_name or "<anonymous>"
    return QubitRebindError(
        f"Loop-body quantum rebind of '{display_name}' inside a qkernel "
        f"while loop is read after the loop: the loop's trip count is a "
        f"runtime measurement outcome, and when the body never runs the "
        f"post-loop read must observe the pre-loop state — but "
        f"'{display_name}' is rebound to a quantum register allocated "
        f"inside the body, so the compiled circuit would read the body's "
        f"register on every path and silently diverge from Python "
        f"semantics on the zero-trip path. Rebind through gates on the "
        f"same qubit(s), or restructure so nothing reads "
        f"'{display_name}' after the loop (e.g. return the measured "
        f"condition instead).",
        handle_name=display_name,
    )


def _zero_trip_static_loop_rebind_error(
    var_name: str,
    loop_kind: str,
) -> QubitRebindError:
    """Build the statically-zero-trip loop rebind rejection error.

    A loop whose bounds resolve to zero iterations never runs, yet the
    emitted program binds post-loop reads to the traced post-body values
    (rebind records' ``after`` is not restored to ``before``), while
    Python keeps the pre-loop bindings. Any quantum rebind record on
    such a loop therefore diverges — including carried rebinds that
    switch which wires the name denotes.

    Args:
        var_name (str): Display name of the rebound quantum variable.
        loop_kind (str): Human-readable loop kind ("for" / "for-items").

    Returns:
        QubitRebindError: The error to raise, with ``handle_name`` set to
            the rebound variable's display name.
    """
    display_name = var_name or "<anonymous>"
    return QubitRebindError(
        f"Loop-body quantum rebind of '{display_name}' inside a qkernel "
        f"{loop_kind} loop whose bounds resolve to zero iterations: the "
        f"loop never runs, so '{display_name}' must keep its pre-loop "
        f"binding, but the emitted program keeps the traced post-body "
        f"binding and would silently read the wrong register. Make the "
        f"loop bounds cover at least one iteration, or remove the "
        f"rebind from the body.",
        handle_name=display_name,
    )


def _static_loop_trip_count(
    loop_op: "ForOperation | ForItemsOperation | WhileOperation",
    concrete_values: dict[str, Any],
    bindings: dict[str, Any],
) -> int | None:
    """Resolve a static loop's trip count when its cardinality is known.

    ``ForOperation`` bounds resolve through constants, accumulated
    concrete values, and parameter bindings; ``ForItemsOperation``
    cardinality resolves through the dict operand's bound contents.
    ``WhileOperation`` trip counts are runtime measurement outcomes and
    always return None. The frontend's zero-trip trace guards
    (``should_trace_for_loop`` / ``should_trace_items_loop``) normally
    keep zero-trip loops out of the IR entirely; this resolver is the
    check-side defense in depth for IR built without the frontend.

    Args:
        loop_op (ForOperation | ForItemsOperation | WhileOperation): The
            loop operation.
        concrete_values (dict[str, Any]): UUID-keyed concrete classical
            results accumulated along the scan.
        bindings (dict[str, Any]): Compile-time parameter bindings.

    Returns:
        int | None: The trip count when it is statically resolvable,
            otherwise None (symbolic bounds, unbound dicts, while loops).
    """
    if isinstance(loop_op, ForItemsOperation):
        for operand in loop_op.operands:
            dict_runtime = getattr(
                getattr(operand, "metadata", None), "dict_runtime", None
            )
            if dict_runtime is not None:
                return len(dict_runtime.bound_data)
        return None
    if not isinstance(loop_op, ForOperation) or len(loop_op.operands) < 3:
        return None
    resolved: list[int] = []
    for bound in loop_op.operands[:3]:
        value = bound.value if hasattr(bound, "value") else bound
        const: Any = None
        if isinstance(value, Value):
            const = value.get_const()
            if const is None and value.uuid in concrete_values:
                const = concrete_values[value.uuid]
            if const is None:
                scalar = value.metadata.scalar
                parameter_name = scalar.parameter_name if scalar else None
                if parameter_name is not None and parameter_name in bindings:
                    const = bindings[parameter_name]
        else:
            const = value
        # Accept any non-bool Integral (Python int, numpy.int64, ...) and
        # coerce, matching the transpiler's other bound/size resolvers
        # (bool is an Integral subclass but is never a valid loop bound).
        if isinstance(const, bool) or not isinstance(const, numbers.Integral):
            return None
        resolved.append(int(const))
    try:
        return len(range(resolved[0], resolved[1], resolved[2]))
    except ValueError:
        # step == 0; leave it to the emit-time bound validation.
        return None


def _root_wire_family(
    root_uuid: str,
    value_table: dict[str, ValueBase],
) -> set[str]:
    """Return a lineage root's UUID together with its array ancestry.

    Resolves a root UUID (from :func:`_pre_branch_root_candidates`) back
    to its ``Value`` so its ``slice_of`` / ``parent_array`` ancestry can
    be walked — a re-sliced view's root reaches the shared base array,
    which is how a slice-view refresh proves same-wire. A root with no
    entry in ``value_table`` (e.g. a value whose producer was stripped
    and that is not itself a record endpoint) contributes only its own
    UUID.

    Args:
        root_uuid (str): The lineage root UUID.
        value_table (dict[str, ValueBase]): UUID-to-value map over the
            loop body plus the record endpoints.

    Returns:
        set[str]: The root's UUID and its array-ancestry UUIDs.
    """
    family: set[str] = set()
    value = value_table.get(root_uuid)
    if value is not None:
        _add_uuid_with_ancestry(value, family)
    else:
        family.add(root_uuid)
    return family


def _pre_branch_root_candidates(
    value: Value,
    branch_producers: dict[str, Operation],
    phi_aliases: dict[str, Value],
    visiting: set[str] | None = None,
    resolved: dict[str, set[str]] | None = None,
) -> set[str]:
    """Over-approximate the lineage roots a phi input may carry.

    Traces backwards through the branch-local producer map and returns
    the UUIDs of every value the input's lineage *may* start at: values
    with no in-branch producer (pre-branch state), in-branch
    ``QInitOperation`` results (fresh allocations), or — for producers
    the trace has no positional model for (composite gates, controlled
    blocks, casts, ...) — the union over all of the producer's genuine
    quantum inputs, since a quantum operation's outputs can only carry
    wires that flow in. Merges contribute the union of both sides;
    pruned compile-time merges contribute their selected source through
    ``phi_aliases``.

    The result feeds the discard check's *carried* exemption, so
    over-approximating is sound: a larger candidate set can only exempt
    more (allow more), never reject a valid kernel. Conversely, a
    pre-branch value absent from this over-approximation provably does
    not flow out through the phi.

    Caution for future precision work: refining the generic union to a
    positional model (result ``i`` carries operand ``i``) is NOT valid
    for composite gates. A composite whose kernel returns its inputs
    permuted (``return b, a``) makes the frontend swap the *variable*
    bindings via the output permutation, so a naive positional model
    would classify the swap as a discard even though every wire
    survives. Any positional refinement must keep result-permuting
    composites exempt — e.g. by staying at the result-set level (all
    input wires carried by *some* result) for composite producers.

    Results are memoized per UUID in ``resolved`` so a value reachable by
    many producer paths — common in wide loop-body DAGs — is computed
    once, keeping the trace linear in the producer graph rather than
    exponential in its path count. The traced body is SSA (each value
    produced once, producers referencing earlier values), hence acyclic;
    the ``visiting`` set is a defensive cycle breaker that returns an
    empty (uncached) set on the impossible back-edge.

    Args:
        value (Value): The quantum value to trace (a phi branch input).
        branch_producers (dict[str, Operation]): Result-UUID-to-producer map
            restricted to the branch body (nested control flow included).
        phi_aliases (dict[str, Value]): Merge-output aliases of the
            compile-time ifs pruned from the traced scope (result UUID ->
            selected source); empty when the scope was not pruned.
        visiting (set[str] | None): UUIDs on the current DFS path, used to
            break cycles. Defaults to None (fresh traversal).
        resolved (dict[str, set[str]] | None): UUID-to-root-set memo,
            shared across the traversal. Defaults to None (fresh memo).

    Returns:
        set[str]: UUIDs of every possible lineage root of ``value``.
    """
    if visiting is None:
        visiting = set()
    if resolved is None:
        resolved = {}
    cached = resolved.get(value.uuid)
    if cached is not None:
        return cached
    if value.uuid in visiting:
        # Defensive cycle break (an SSA body is acyclic). Not cached: the
        # empty set here is incomplete for a non-cyclic re-entry.
        return set()
    visiting.add(value.uuid)
    try:
        result = _compute_pre_branch_roots(
            value, branch_producers, phi_aliases, visiting, resolved
        )
    finally:
        visiting.discard(value.uuid)
    resolved[value.uuid] = result
    return result


def _compute_pre_branch_roots(
    value: Value,
    branch_producers: dict[str, Operation],
    phi_aliases: dict[str, Value],
    visiting: set[str],
    resolved: dict[str, set[str]],
) -> set[str]:
    """Compute one value's lineage roots (uncached inner step).

    Split out from :func:`_pre_branch_root_candidates` so the caller owns
    the cycle-guard and memoization bookkeeping and this function is a
    pure structural recursion over the producer of ``value``.

    Args:
        value (Value): The quantum value to trace.
        branch_producers (dict[str, Operation]): Result-UUID-to-producer
            map restricted to the branch/loop body.
        phi_aliases (dict[str, Value]): Merge-output aliases of the
            compile-time ifs pruned from the traced scope.
        visiting (set[str]): UUIDs on the current DFS path.
        resolved (dict[str, set[str]]): UUID-to-root-set memo, threaded
            through recursive calls.

    Returns:
        set[str]: UUIDs of every possible lineage root of ``value``.
    """
    alias_source = phi_aliases.get(value.uuid)
    if alias_source is not None:
        # Pruned compile-time merge: the output stands for its selected
        # source (the dead side never executes).
        return _pre_branch_root_candidates(
            alias_source, branch_producers, phi_aliases, visiting, resolved
        )
    producer = branch_producers.get(value.uuid)
    if producer is None:
        return {value.uuid}
    if isinstance(producer, QInitOperation):
        return {value.uuid}
    if isinstance(producer, GateOperation):
        qubit_operands = producer.qubit_operands
        for index, result in enumerate(producer.results):
            if result.uuid == value.uuid and index < len(qubit_operands):
                return _pre_branch_root_candidates(
                    qubit_operands[index],
                    branch_producers,
                    phi_aliases,
                    visiting,
                    resolved,
                )
        # Positional mismatch cannot normally happen; fall through to
        # the generic all-quantum-inputs over-approximation below.
    elif isinstance(producer, IfOperation):
        for merge in producer.iter_merges():
            if merge.result.uuid == value.uuid:
                return _pre_branch_root_candidates(
                    merge.true_value, branch_producers, phi_aliases, visiting, resolved
                ) | _pre_branch_root_candidates(
                    merge.false_value, branch_producers, phi_aliases, visiting, resolved
                )
        # The value is a non-merge result of the if (cannot normally
        # happen); fall through to the generic over-approximation below.
    elif isinstance(producer, PhiOp):
        if len(producer.operands) < 3:
            return set()
        return _pre_branch_root_candidates(
            producer.true_value, branch_producers, phi_aliases, visiting, resolved
        ) | _pre_branch_root_candidates(
            producer.false_value, branch_producers, phi_aliases, visiting, resolved
        )
    # Generic producer (composite gate, controlled block, cast, ...):
    # its outputs may carry any of its genuine quantum inputs.
    roots: set[str] = set()
    for input_value in op_genuine_input_values(producer):
        if not isinstance(input_value, Value):
            continue
        if not input_value.type.is_quantum():
            continue
        roots |= _pre_branch_root_candidates(
            input_value, branch_producers, phi_aliases, visiting, resolved
        )
    return roots


def op_genuine_input_values(op: Operation) -> list[ValueBase]:
    """Return the input values an operation genuinely reads.

    Rebind-record values (loop-carried records on loop operations and
    branch records on ``IfOperation``s) ride along ``all_input_values``
    for cloning/substitution but are not reads, so they are excluded.

    Args:
        op (Operation): Operation to inspect.

    Returns:
        list[ValueBase]: ``all_input_values`` minus rebind-record values.
    """
    excluded: set[str] = set()
    if isinstance(op, (ForOperation, ForItemsOperation, WhileOperation)):
        for loop_record in op.loop_carried_rebinds:
            excluded.add(loop_record.before.uuid)
            excluded.add(loop_record.after.uuid)
    if isinstance(op, IfOperation):
        for branch_record in op.branch_rebinds:
            excluded.add(branch_record.before.uuid)
    return [v for v in op.all_input_values() if v.uuid not in excluded]


def _add_uuid_with_ancestry(value: ValueBase, collected: set[str]) -> None:
    """Record a value's UUID together with its array ancestry.

    Walks ``parent_array`` / ``slice_of`` chains so an element or view
    read of a register counts as touching the register itself.

    Args:
        value (ValueBase): Input value read by an operation.
        collected (set[str]): Mutable UUID set, updated in place.
    """
    if value.uuid in collected:
        return
    collected.add(value.uuid)
    for attr in ("parent_array", "slice_of"):
        ancestor = getattr(value, attr, None)
        if ancestor is not None:
            _add_uuid_with_ancestry(ancestor, collected)


def _op_referenced_uuids_with_ancestry(op: Operation) -> set[str]:
    """Collect the UUIDs one operation genuinely reads, with array ancestry.

    Combines :func:`op_genuine_input_values` (rebind-record values
    excluded) with :func:`_add_uuid_with_ancestry` (element / view reads
    count as touching the register itself).

    Args:
        op (Operation): Operation to inspect.

    Returns:
        set[str]: UUIDs of every genuinely-read input value and its
            array ancestry.
    """
    referenced: set[str] = set()
    for value in op_genuine_input_values(op):
        _add_uuid_with_ancestry(value, referenced)
    return referenced


def _branch_referenced_uuids(branch_ops: list[Operation]) -> set[str]:
    """Collect every value UUID a branch body references as an input.

    Covers all nested operations (via ``flatten_ops``), excluding
    rebind-record values, with array ancestry per
    :func:`_op_referenced_uuids_with_ancestry`. Used as the "did this
    branch consume the original state" evidence — any reference is
    enough to disqualify the discard error, which errs toward allowing.

    Args:
        branch_ops (list[Operation]): The branch body operations, already
            pruned of compile-time-decidable ifs.

    Returns:
        set[str]: UUIDs of every referenced input value and its array
            ancestry.
    """
    referenced: set[str] = set()
    for op in flatten_ops(branch_ops):
        referenced |= _op_referenced_uuids_with_ancestry(op)
    return referenced


def _wire_reader_map(ops: list[Operation]) -> dict[str, Operation]:
    """Map each referenced UUID (ancestry included) to a reading op.

    Affine typing gives a quantum value at most one genuine reader, so a
    plain dict suffices; on ancestry or classical collisions the last
    reader wins, which can only push the terminal chase below toward
    its consumed (allow) outcome.

    Args:
        ops (list[Operation]): Scope operations, already pruned of
            compile-time-decidable ifs.

    Returns:
        dict[str, Operation]: Read-UUID-to-reader map over the flattened
            scope (rebind-record values excluded).
    """
    readers: dict[str, Operation] = {}
    for op in flatten_ops(ops):
        for uuid in _op_referenced_uuids_with_ancestry(op):
            readers[uuid] = op
    return readers


def _wire_terminally_consumed(
    value: Value,
    readers: dict[str, Operation],
    alias_reads: set[str],
) -> bool:
    """Whether a scalar wire's final in-scope version is consumed.

    Chases ``value`` forward through positional gate self-updates
    (result ``i`` continues operand ``i``'s wire) to the wire's last
    version inside the scope. The wire counts as consumed when the
    chase ends at a non-gate reader (a measurement consumes the state;
    a composite, controlled block, cast or slice hands it to a producer
    whose outputs the carried exemption models), a pruned compile-time
    merge (the executing pass-through selects the value, matching the
    non-gate outcome its collapsed form used to get), or at a gate read
    with no positional continuation for this wire (an ancestry-level or
    unmodeled read — over-approximated as consumed, erring toward
    allowing). It is NOT consumed when some version is read by nothing
    in the scope: the gated state is dropped there even though the
    original version was touched
    (``if cond: q = qmc.x(q); q = qmc.qubit("fresh")`` touches ``q``
    but drops the X output).

    Args:
        value (Value): The wire's version at scope entry.
        readers (dict[str, Operation]): Read-UUID-to-reader map from
            :func:`_wire_reader_map`.
        alias_reads (set[str]): UUIDs (ancestry included) read by the
            scope's pruned compile-time merges — selected sources of the
            scope's ``PrunedIfView.phi_aliases``.

    Returns:
        bool: True when the wire's final version has a reader.
    """
    current = value.uuid
    seen: set[str] = set()
    while current not in seen:
        seen.add(current)
        if current in alias_reads:
            return True
        reader = readers.get(current)
        if reader is None:
            return False
        if not isinstance(reader, GateOperation):
            return True
        qubit_operands = reader.qubit_operands
        next_uuid: str | None = None
        for index, operand in enumerate(qubit_operands):
            if operand.uuid == current and index < len(reader.results):
                next_uuid = reader.results[index].uuid
                break
        if next_uuid is None:
            return True
        current = next_uuid
    # An SSA body cannot cycle; treat the impossible back-edge as
    # consumed (allow side).
    return True


def _promoted_branch_records(
    branch_ops: list[Operation],
    concrete_values: dict[str, Any],
    bindings: dict[str, Any],
) -> list[BranchRebind]:
    """Collect rebind records that fire unconditionally within a branch.

    A rebind recorded on a compile-time-TAKEN if nested inside a branch
    executes whenever the enclosing branch runs, so it belongs to the
    enclosing runtime if's check (the taken if itself is not runtime
    control flow and is never checked directly). Walks the branch with
    the same condition classification the lowering pass uses: a taken
    nested if contributes its taken-side records and recurses, a dead
    branch contributes nothing, and a runtime nested if keeps its own
    records — it is checked at its own level. Loop bodies are walked
    too; like the loop-carried rebind check, this is trip-count-agnostic
    (a rebind inside a zero-trip loop is still collected).

    Args:
        branch_ops (list[Operation]): The branch body operations, raw
            (not pruned), in program order.
        concrete_values (dict[str, Any]): UUID-keyed concrete classical-op
            results accumulated up to the branch entry. Updated in place
            for this scope; nested non-if scopes receive copies.
        bindings (dict[str, Any]): Compile-time parameter bindings.

    Returns:
        list[BranchRebind]: Records whose rebind executes unconditionally
            when the branch runs, in program order.
    """
    promoted: list[BranchRebind] = []
    for op in branch_ops:
        evaluate_classical_op_concrete(op, concrete_values, bindings)
        if isinstance(op, IfOperation):
            taken = resolve_compile_time_condition(
                op.condition, concrete_values, bindings
            )
            if taken is None:
                # Runtime nested if: its records are checked at its own
                # level by the scan.
                continue
            for record in op.branch_rebinds:
                rebound = record.rebound_in_true if taken else record.rebound_in_false
                if rebound:
                    promoted.append(record)
            promoted.extend(
                _promoted_branch_records(
                    op.true_operations if taken else op.false_operations,
                    concrete_values,
                    bindings,
                )
            )
            continue
        if isinstance(op, HasNestedOps):
            for body in op.nested_op_lists():
                promoted.extend(
                    _promoted_branch_records(body, dict(concrete_values), bindings)
                )
        # Other operations carry no nested control flow to walk.
    return promoted


def _check_branch_quantum_discard(
    if_op: IfOperation,
    concrete_values: dict[str, Any],
    bindings: dict[str, Any],
    reads_outside: "_PathReads",
    promoted: dict[bool, list[BranchRebind]],
) -> None:
    """Check one runtime ``IfOperation``'s rebind records for discards.

    For each recorded quantum rebind and each rebinding branch side,
    verifies that the pre-branch value survives the path on which that
    branch is taken: it must be consumed inside the taken branch, carried
    out through some phi merge on that side, or referenced by an
    operation outside the if that executes on the paths reaching it. A
    pre-branch value with none of these owners is silently dropped
    exactly when the branch is taken — the discard this check rejects.
    Besides the if's own records, records promoted from
    compile-time-TAKEN ifs nested inside a branch (which fire
    unconditionally on that side; see :func:`_promoted_branch_records`)
    are checked against that side. Branch bodies are pruned of
    compile-time-decidable nested ifs first (each side with its own copy
    of ``concrete_values``, matching ``CompileTimeIfLoweringPass``
    scoping) so dead-branch rebinds and dead-branch consumes do not
    distort the analysis.

    Args:
        if_op (IfOperation): Runtime if whose condition did not resolve at
            compile time.
        concrete_values (dict[str, Any]): UUID-keyed concrete classical-op
            results accumulated up to this operation.
        bindings (dict[str, Any]): Compile-time parameter bindings.
        reads_outside (_PathReads): Membership view of the UUIDs read by
            operations outside this if's subtree on the paths that reach
            it (rebind records excluded, array ancestry included) — the
            values still owned outside the if. Assembled path-sensitively
            by the scan.
        promoted (dict[bool, list[BranchRebind]]): Per-side records
            promoted from compile-time-TAKEN nested ifs.

    Raises:
        QubitRebindError: If a rebinding branch drops the pre-branch
            quantum value with no consumer, no carrying phi, and no
            outside owner.
    """
    side_checks: list[tuple[str, Value, bool]] = []
    for record in if_op.branch_rebinds:
        if not isinstance(record.before, Value) or not record.before.type.is_quantum():
            continue
        if record.rebound_in_true:
            side_checks.append((record.var_name, record.before, True))
        if record.rebound_in_false:
            side_checks.append((record.var_name, record.before, False))
    for side in (True, False):
        for record in promoted[side]:
            if isinstance(record.before, Value) and record.before.type.is_quantum():
                side_checks.append((record.var_name, record.before, side))
    if not side_checks:
        return

    pruned_branches = {
        True: prune_compile_time_ifs(
            if_op.true_operations, dict(concrete_values), bindings
        ),
        False: prune_compile_time_ifs(
            if_op.false_operations, dict(concrete_values), bindings
        ),
    }
    referenced: dict[bool, set[str]] = {}
    readers: dict[bool, dict[str, Operation]] = {}
    alias_reads: dict[bool, set[str]] = {}
    carried: dict[bool, set[str]] = {}
    for side, pruned_branch in pruned_branches.items():
        side_referenced = _branch_referenced_uuids(pruned_branch.operations)
        # A pruned compile-time merge inside the branch reads its selected
        # source exactly like the executing pass-through it models.
        side_alias_reads: set[str] = set()
        for _, alias_source in pruned_branch.phi_aliases:
            _add_uuid_with_ancestry(alias_source, side_alias_reads)
        side_referenced |= side_alias_reads
        referenced[side] = side_referenced
        alias_reads[side] = side_alias_reads
        readers[side] = _wire_reader_map(pruned_branch.operations)
        side_producers: dict[str, Operation] = {}
        build_producer_map(pruned_branch.operations, side_producers)
        side_aliases = {
            result.uuid: source for result, source in pruned_branch.phi_aliases
        }
        carried_roots: set[str] = set()
        for merge in if_op.iter_merges():
            side_value = merge.select(side)
            if not side_value.type.is_quantum():
                continue
            carried_roots |= _pre_branch_root_candidates(
                side_value, side_producers, side_aliases
            )
        carried[side] = carried_roots

    for var_name, before, side in side_checks:
        if isinstance(before, ArrayValue):
            # Whole-register rebinds keep the documented element-read
            # granularity: any touch of the register counts as
            # consumption (LIMITATIONS.md conservative corner).
            if before.uuid in referenced[side]:
                continue
        elif _wire_terminally_consumed(before, readers[side], alias_reads[side]):
            # Scalar consumption is judged at the wire's FINAL in-branch
            # version, so a gate-then-reallocate
            # (``if cond: q = qmc.x(q); q = qmc.qubit("fresh")``) is a
            # discard of the gated state, not a consumption of it.
            continue
        if before.uuid in carried[side]:
            continue
        if before.uuid in reads_outside:
            continue
        raise _branch_quantum_discard_error(var_name, "true" if side else "false")


@dataclasses.dataclass
class _DiscardScanCaches:
    """Shared memoization for one discard-scan invocation.

    Attributes:
        element_reads (dict[int, set[str]]): Per-``id(op)`` pruned-subtree
            read sets (see :func:`_scope_element_reads`).
        scope_counts (dict[int, dict[str, int]]): Per-``id(scope list)``
            element read-count maps (see :func:`_scope_read_counts`).
    """

    element_reads: dict[int, set[str]] = dataclasses.field(default_factory=dict)
    scope_counts: dict[int, dict[str, int]] = dataclasses.field(default_factory=dict)


def _scope_element_reads(
    op: Operation,
    concrete_values: dict[str, Any],
    bindings: dict[str, Any],
    element_reads_cache: dict[int, set[str]],
) -> set[str]:
    """Reads of one scope element over its compile-time-pruned subtree.

    Prunes the element with the accumulated compile-time state first, so
    a compile-time-dead branch inside it contributes no reads while its
    collapsed phis contribute their selected pass-through operands —
    matching what actually executes. Runtime ifs inside the element keep
    both branches (a non-ancestor runtime if's reads count as potential
    ownership; see the scan docstring). Results are cached per element
    identity; the accumulated state at the first computation may differ
    from later call positions, which can only under-prune and therefore
    over-approximate the reads — safe for the ownership evidence, which
    errs toward allowing.

    Args:
        op (Operation): The scope element to collect reads from.
        concrete_values (dict[str, Any]): Accumulated compile-time state
            at the caller's position (copied before pruning).
        bindings (dict[str, Any]): Compile-time parameter bindings.
        element_reads_cache (dict[int, set[str]]): Per-``id(op)`` cache,
            updated in place.

    Returns:
        set[str]: UUIDs read anywhere in the element's pruned subtree
            (rebind records excluded, array ancestry included).
    """
    cached = element_reads_cache.get(id(op))
    if cached is not None:
        return cached
    pruned = prune_compile_time_ifs([op], dict(concrete_values), bindings)
    reads: set[str] = set()
    for sub_op in flatten_ops(pruned.operations):
        reads |= _op_referenced_uuids_with_ancestry(sub_op)
    # Pruned compile-time merges read their selected pass-through
    # source — matching what actually executes.
    for _, alias_source in pruned.phi_aliases:
        _add_uuid_with_ancestry(alias_source, reads)
    element_reads_cache[id(op)] = reads
    return reads


def _scope_read_counts(
    scope_ops: list[Operation],
    concrete_values: dict[str, Any],
    bindings: dict[str, Any],
    caches: _DiscardScanCaches,
) -> dict[str, int]:
    """Per-UUID count of scope elements whose subtree reads it.

    Computed once per scope list (cached by list identity) so that
    excluding any single element later is a constant-time count
    adjustment instead of re-unioning the sibling sets per checked if.

    Args:
        scope_ops (list[Operation]): The scope's operation list.
        concrete_values (dict[str, Any]): Accumulated compile-time state
            for pruning the elements.
        bindings (dict[str, Any]): Compile-time parameter bindings.
        caches (_DiscardScanCaches): Shared memoization, updated in place.

    Returns:
        dict[str, int]: For each UUID, the number of scope elements whose
            pruned subtree reads it.
    """
    cached = caches.scope_counts.get(id(scope_ops))
    if cached is not None:
        return cached
    counts: dict[str, int] = {}
    for element in scope_ops:
        for read_uuid in _scope_element_reads(
            element, concrete_values, bindings, caches.element_reads
        ):
            counts[read_uuid] = counts.get(read_uuid, 0) + 1
    caches.scope_counts[id(scope_ops)] = counts
    return counts


class _PathReads:
    """Membership-only view of the reads that execute on one path.

    Represents, as a linked chain of scope contexts, the set of UUIDs
    read by operations outside a given if subtree on the paths that
    reach it. Each node contributes one scope's element read-counts with
    one element excluded (the if or loop being descended into / checked)
    plus optional extra reads (the entered side's phi operands); parent
    nodes contribute the enclosing scopes the same way. The set is never
    materialized — the discard check only probes membership for the few
    record UUIDs — so building a node is O(1) beyond the per-scope count
    map, which is computed once per scope.

    Attributes:
        _parent (_PathReads | None): The enclosing path context.
        _scope_counts (dict[str, int]): Per-UUID element counts of this
            node's scope.
        _excluded_reads (set[str]): Reads of the excluded element (its
            containment is subtracted from the counts).
        _extra (set[str]): Additional reads carried on this path (the
            entered side's phi operands, with array ancestry).
    """

    __slots__ = ("_parent", "_scope_counts", "_excluded_reads", "_extra")

    def __init__(
        self,
        parent: "_PathReads | None",
        scope_counts: dict[str, int],
        excluded_reads: set[str],
        extra: set[str],
    ) -> None:
        """Build one path-context node.

        Args:
            parent (_PathReads | None): The enclosing path context, or
                None at the block's top level.
            scope_counts (dict[str, int]): Per-UUID element counts of the
                current scope (see :func:`_scope_read_counts`).
            excluded_reads (set[str]): Pruned-subtree reads of the scope
                element being excluded.
            extra (set[str]): Additional reads that execute on this path.
        """
        self._parent = parent
        self._scope_counts = scope_counts
        self._excluded_reads = excluded_reads
        self._extra = extra

    def __contains__(self, uuid: str) -> bool:
        """Whether any operation on this path (outside the subtree) reads
        ``uuid``.

        Args:
            uuid (str): The value UUID to probe.

        Returns:
            bool: True when some non-excluded scope element at any level,
                or an entered-side phi operand, reads the UUID.
        """
        node: _PathReads | None = self
        while node is not None:
            if uuid in node._extra:
                return True
            count = node._scope_counts.get(uuid, 0)
            if uuid in node._excluded_reads:
                count -= 1
            if count > 0:
                return True
            node = node._parent
        return False


def _phi_side_operand_reads(if_op: IfOperation, side: bool) -> set[str]:
    """Phi operands an if's join carries on one branch side.

    On the path that takes ``side``, each phi merge selects — and thereby
    keeps alive past the join — its ``side`` operand; the opposite side's
    operands do not execute on that path and must not count as ownership
    for values checked deeper inside the entered branch.

    Args:
        if_op (IfOperation): The if whose branch is being entered.
        side (bool): True for the true branch, False for the false branch.

    Returns:
        set[str]: UUIDs (with array ancestry) of the entered side's phi
            operands.
    """
    reads: set[str] = set()
    for merge in if_op.iter_merges():
        _add_uuid_with_ancestry(merge.select(side), reads)
    return reads


def _check_loop_quantum_discards(
    loop_op: ForOperation | ForItemsOperation | WhileOperation,
    concrete_values: dict[str, Any],
    bindings: dict[str, Any],
    reads_outside: "_PathReads",
    caches: _DiscardScanCaches,
) -> None:
    """Reject one loop op's quantum rebind records that discard state.

    Neither loop kind can express a per-iteration quantum rebind today.
    An unrolled ``for`` / ``for``-items body is re-instantiated per
    iteration WITHOUT carrying the rebound register between iterations:
    later iterations re-read the traced pre-loop register (measured
    divergence: a measure-then-reallocate body sampled the first
    iteration's result for every ``n``) and the replaced state is
    dropped. A runtime ``while`` body re-executes on the same
    registers: a body read of the pre-loop value re-executes on a stale
    register, and an unconsumed rebound register is dropped at
    re-entry. Like the classical loop-carried check, the rejection is
    trip-count-agnostic.

    A record is exempted when:

    - the post-body value carries the incoming value forward — the
      record's ``after`` family (its own value ancestry via ``slice_of``
      / ``parent_array`` chains, plus its over-approximated lineage
      roots via :func:`_pre_branch_root_candidates` over the body's
      producer map) intersects the incoming value's ancestry family.
      The family expansion is what exempts view refreshes:
      ``evens = evens[:]`` re-slices the *base* array, so the new view
      reaches the base rather than the old view, and the shared base
      proves the wires stay reachable. The after side needs both the
      producer lineage and the value ancestry because
      ``StripSliceArrayOpsPass`` removes slice producers before the
      ``AnalyzePass`` safety-net run while ``slice_of`` survives on the
      value; or
    - the rebind is loop-invariant and the first iteration is covered:
      the post-body value is NOT produced in the body (an outer value
      every iteration rebinds to again, discarding nothing beyond the
      first), the body never reads the pre-loop value (nothing
      re-executes on a stale register), and the pre-loop value is owned
      outside the loop (read on the path around it, per the same
      path-sensitive ``_PathReads`` evidence the branch check uses).
      The invariant arm also keeps the ``AnalyzePass`` safety net exact
      for records whose ``after`` was a compile-time phi that
      if-lowering erased and substituted away. A binding that makes a
      static loop run zero times can only diverge for programs whose
      Python semantics already double-consume the pre-loop value, so no
      well-formed program is miscompiled by this arm.

    Everything else is rejected. In particular, a BODY-PRODUCED rebound
    register is never exempt — not even when the body terminally
    consumes it (the former repeat-until-success exemption): a runtime
    while re-executes its body on one persistent register without
    reset, so "fresh per iteration" is not expressible, and review
    measured an rx-gated repeat-until-success body sampling the
    wire-reuse distribution instead of the fresh-register one. Spell
    such bodies with a body-local register name instead (no
    pre-existing variable rebound, so nothing is discarded and the
    identical circuit stays out of this check's jurisdiction). In-body
    consumption of the incoming value is likewise NOT an exemption: the
    read re-executes against the traced register every iteration, which
    matches Python semantics only for the first one (review-measured
    divergence, see above).

    ``while`` loops report a dedicated error for any non-identical
    rebound value read after the loop — checked BEFORE the carried
    exemption, since a carried-but-not-identical rebind (re-slicing a
    different element range of the same base) still binds the post-loop
    read to different wires: the trip count is a runtime measurement
    outcome, so the zero-trip path is live, and on it the post-loop
    read must observe the pre-loop state (see
    :func:`_while_zero_trip_rebind_error`). For static loops the same
    zero-trip hazard is closed twice over: the frontend prunes a
    statically-zero-trip loop at build time (no loop op, no record), and
    :func:`_static_loop_trip_count` rejects any quantum record on a loop
    whose bounds still resolve to zero trips at check time.

    Args:
        loop_op (ForOperation | ForItemsOperation | WhileOperation): The
            loop operation whose records are checked.
        concrete_values (dict[str, Any]): Accumulated compile-time state
            at the loop's position, used to prune the body.
        bindings (dict[str, Any]): Compile-time parameter bindings.
        reads_outside (_PathReads): Membership view of the reads outside
            the loop subtree on the paths that reach it.
        caches (_DiscardScanCaches): Shared read memoization, updated in
            place.

    Raises:
        QubitRebindError: If a quantum rebind record is neither carried
            forward nor a covered loop-invariant rebind — with a
            dedicated zero-trip message when a while body's non-carried
            rebound value is read after the loop.
    """
    quantum_records = [
        r for r in loop_op.loop_carried_rebinds if r.before.type.is_quantum()
    ]
    if not quantum_records:
        return
    loop_kind = _LOOP_KIND_NAMES.get(type(loop_op), "for")
    if _static_loop_trip_count(loop_op, concrete_values, bindings) == 0:
        # A statically-zero-trip loop never runs, but post-loop reads
        # keep the traced post-body binding (emit does not restore
        # rebind records' after values to before) while Python keeps the
        # pre-loop binding — even a carried rebind that switches which
        # wires the name denotes (e.g. re-slicing a different element
        # range) diverges. With no iteration to justify any rebind,
        # reject outright; bindings resolve for-loop bounds at the
        # pre-fold hook, so this is exact there.
        raise _zero_trip_static_loop_rebind_error(
            quantum_records[0].var_name, loop_kind
        )
    # Build the producer map and value table over the COMPILE-TIME-PRUNED
    # body: a compile-time-dead ``if`` inside the body must contribute no
    # fresh-allocation root (its branch never executes), so its recorded
    # merge alias resolves the post-body value straight through to the
    # incoming wire — while a runtime ``if`` keeps both branches, so a
    # conditional rebind's phi still unions in the fresh root and is
    # rejected. Without pruning here, the two are indistinguishable at
    # the pre-fold hook.
    body_producers: dict[str, Operation] = {}
    body_phi_aliases: dict[str, Value] = {}
    value_table: dict[str, ValueBase] = {}
    for body in loop_op.nested_op_lists():
        pruned_body = prune_compile_time_ifs(body, dict(concrete_values), bindings)
        for alias_result, alias_source in pruned_body.phi_aliases:
            body_phi_aliases[alias_result.uuid] = alias_source
            value_table.setdefault(alias_result.uuid, alias_result)
            value_table.setdefault(alias_source.uuid, alias_source)
        for body_op in flatten_ops(pruned_body.operations):
            for result in body_op.results:
                body_producers[result.uuid] = body_op
            for value in (*body_op.all_input_values(), *body_op.results):
                value_table.setdefault(value.uuid, value)

    def body_reads(uuid: str) -> bool:
        """Whether any pruned body op reads the value (ancestry included).

        Args:
            uuid (str): The value UUID to probe.

        Returns:
            bool: True when some body operation's pruned subtree reads it.
        """
        return any(
            _scope_read_counts(body, concrete_values, bindings, caches).get(uuid, 0) > 0
            for body in loop_op.nested_op_lists()
        )

    for record in quantum_records:
        # The carried exemption is only meaningful for a QUANTUM
        # post-body value: a classical after cannot keep the wires
        # reachable, yet its producer lineage would still reach the
        # incoming value when the overwrite CONSUMES it — a loop body
        # ``q = qmc.measure(q)`` re-executes the measurement against the
        # traced register every iteration and must be rejected, not
        # exempted through the measurement's input lineage.
        carried = False
        if record.after.type.is_quantum():
            before_family: set[str] = set()
            _add_uuid_with_ancestry(record.before, before_family)
            value_table.setdefault(record.after.uuid, record.after)
            # Carried iff EVERY possible root source of the post-body
            # value is same-wire as the incoming value — i.e. each root's
            # own ancestry intersects the incoming value's family. A mere
            # OVERLAP of the unioned root set with the family is unsound:
            # a conditional rebind whose phi unions the incoming wire
            # with a fresh allocation (``if bit: q = qmc.qubit(...)``)
            # produces roots {incoming, fresh}, of which only the
            # incoming root overlaps, yet some paths discard the incoming
            # state. Requiring ALL roots to be same-wire rejects that
            # while still exempting slice-view refreshes, whose single
            # re-sliced root shares the base array.
            roots = _pre_branch_root_candidates(
                record.after, body_producers, body_phi_aliases
            )
            carried = bool(roots) and all(
                bool(_root_wire_family(root, value_table) & before_family)
                for root in roots
            )
        if (
            isinstance(loop_op, WhileOperation)
            and record.after.uuid != record.before.uuid
            and record.after.uuid in reads_outside
        ):
            # Any rebound value read after a while loop hits the
            # always-live zero-trip path, on which the read must observe
            # the pre-loop state instead. This fires BEFORE the carried
            # exemption: a carried-but-not-identical rebind (e.g.
            # re-slicing a different element range of the same base)
            # still binds the post-loop read to different wires. Exempt
            # only the provably SAME-WIRE shapes: the strict-identity
            # record (after IS before, left by if-lowering substituting
            # a pass-through phi away), and a quantum after whose
            # over-approximated lineage roots are EXACTLY the incoming
            # value — every dataflow path (gate chains, both phi sides)
            # leads back to the same register, e.g. a gate self-update
            # under an in-body if, so the post-loop read stays on the
            # pre-loop wire even at zero trips.
            same_wire = record.after.type.is_quantum() and _pre_branch_root_candidates(
                record.after, body_producers, body_phi_aliases
            ) == {record.before.uuid}
            if not same_wire:
                raise _while_zero_trip_rebind_error(record.var_name)
        if carried:
            continue
        # A pruned compile-time merge output is body-produced (its if was
        # in the body) even though no operation remains in the pruned view.
        invariant_after = (
            record.after.uuid not in body_producers
            and record.after.uuid not in body_phi_aliases
        )
        before_read = body_reads(record.before.uuid)
        owned = record.before.uuid in reads_outside
        if invariant_after and not before_read and owned:
            # Loop-invariant rebind: iterations beyond the first rebind
            # to the same outer value and discard nothing; the outside
            # owner covers the first. Also keeps the AnalyzePass safety
            # net exact for records whose after was a compile-time phi
            # that if-lowering erased and substituted away.
            continue
        # Body-produced rebinds are rejected for BOTH loop kinds:
        # unrolled loops re-instantiate the body without carrying the
        # rebound register between iterations, and a runtime while
        # re-executes its body on one persistent register without reset,
        # so "fresh per iteration" is not expressible either way (review
        # measured an rx-gated while repeat-until-success body sampling
        # the wire-reuse distribution, not the fresh-register one).
        if not record.after.type.is_quantum():
            raise _loop_nonquantum_overwrite_error(record.var_name, loop_kind)
        raise _loop_quantum_discard_error(record.var_name, loop_kind)


def _scan_branch_quantum_discards(
    ops: list[Operation],
    concrete_values: dict[str, Any],
    bindings: dict[str, Any],
    measurement_tainted: set[str],
    inherited_reads: "_PathReads | None",
    caches: _DiscardScanCaches,
) -> None:
    """Walk operations like the if-lowering pass, checking each runtime if.

    Mirrors ``CompileTimeIfLoweringPass``'s classification walk:
    classical-op results are accumulated in program order, a
    compile-time-resolvable ``IfOperation`` contributes only its taken
    branch to the scan, and a runtime ``IfOperation`` has both branches
    scanned with their own copy of the accumulated state. This per-position
    classification is what lets a rebind inside a compile-time branch
    *nested within* a runtime branch be classified correctly —
    ``prune_compile_time_ifs`` alone does not descend into runtime-if
    bodies.

    The "referenced outside this if" ownership evidence is assembled
    path-sensitively while descending: ``inherited_reads`` accumulates,
    per enclosing level, the reads of the sibling scope elements around
    the branch being entered plus the enclosing if's phi operands of the
    entered side only. Reads on the sibling branch of an enclosing if —
    and the enclosing phis' opposite-side operands — do not execute on
    the paths that reach this scope, so an ownership claim can no longer
    rest on a branch that is never taken together with the checked one.
    Reads inside non-ancestor runtime ifs elsewhere in a scope still
    count as potential ownership: a value conditionally consumed
    downstream is conditionally owned, matching how values that are
    unused on some paths are tolerated elsewhere.

    Args:
        ops (list[Operation]): Operations to scan, in program order.
        concrete_values (dict[str, Any]): UUID-keyed concrete classical-op
            results accumulated along the walk. Updated in place for the
            current scope; nested scopes receive copies.
        bindings (dict[str, Any]): Compile-time parameter bindings used to
            resolve ``IfOperation`` conditions.
        measurement_tainted (set[str]): UUIDs of values transitively
            derived from measurement results; an unresolvable if whose
            condition is in this set is runtime control flow.
        inherited_reads (_PathReads | None): Membership view of the reads
            outside this scope that execute on every path reaching it,
            per the path-sensitive construction above. None at the
            block's top level.
        caches (_DiscardScanCaches): Shared per-element / per-scope read
            memoization, updated in place.

    Raises:
        QubitRebindError: If a runtime if branch rebinds a quantum
            variable whose pre-branch value has no owner on that path.
    """

    def path_context(exclude_op: Operation, extra: set[str]) -> _PathReads:
        """Build the path view outside one scope element.

        Args:
            exclude_op (Operation): The scope element being checked or
                descended into (its subtree is excluded).
            extra (set[str]): Additional reads carried on the path (the
                entered side's phi operands), empty for checks.

        Returns:
            _PathReads: Membership view chaining to ``inherited_reads``.
        """
        return _PathReads(
            inherited_reads,
            _scope_read_counts(ops, concrete_values, bindings, caches),
            _scope_element_reads(
                exclude_op, concrete_values, bindings, caches.element_reads
            ),
            extra,
        )

    for op in ops:
        evaluate_classical_op_concrete(op, concrete_values, bindings)
        if isinstance(op, IfOperation):
            taken = resolve_compile_time_condition(
                op.condition, concrete_values, bindings
            )
            if taken is not None:
                branch = op.true_operations if taken else op.false_operations
                _scan_branch_quantum_discards(
                    branch,
                    concrete_values,
                    bindings,
                    measurement_tainted,
                    path_context(op, _phi_side_operand_reads(op, taken)),
                    caches,
                )
                continue
            # Only a runtime (measurement-derived condition) if can
            # discard, and only when it carries quantum rebind records —
            # its own, or records promoted from compile-time-TAKEN ifs
            # nested inside its branches, which fire unconditionally on
            # that side. Records exist solely when a branch rebinds a
            # variable to a *different* quantum value (fresh allocation,
            # or substitution of another register); ordinary gate
            # self-updates (``q = qmc.x(q)``) and measurement-conditioned
            # gates on other qubits leave no record, so the common case
            # skips the ownership-evidence assembly below.
            if getattr(op.condition, "uuid", None) in measurement_tainted:
                promoted = {
                    True: _promoted_branch_records(
                        op.true_operations, dict(concrete_values), bindings
                    ),
                    False: _promoted_branch_records(
                        op.false_operations, dict(concrete_values), bindings
                    ),
                }
                if op.branch_rebinds or promoted[True] or promoted[False]:
                    _check_branch_quantum_discard(
                        op,
                        concrete_values,
                        bindings,
                        path_context(op, set()),
                        promoted,
                    )
            for side, branch in (
                (True, op.true_operations),
                (False, op.false_operations),
            ):
                _scan_branch_quantum_discards(
                    branch,
                    dict(concrete_values),
                    bindings,
                    measurement_tainted,
                    path_context(op, _phi_side_operand_reads(op, side)),
                    caches,
                )
            continue
        if isinstance(op, HasNestedOps):
            child_reads = path_context(op, set())
            if isinstance(op, (ForOperation, ForItemsOperation, WhileOperation)):
                _check_loop_quantum_discards(
                    op, concrete_values, bindings, child_reads, caches
                )
            for body in op.nested_op_lists():
                _scan_branch_quantum_discards(
                    body,
                    dict(concrete_values),
                    bindings,
                    measurement_tainted,
                    child_reads,
                    caches,
                )
        # Other operations carry no nested control flow to scan.


def reject_control_flow_quantum_discard(
    operations: list[Operation],
    bindings: dict[str, Any] | None = None,
) -> None:
    """Reject control-flow-internal quantum rebinds that discard state.

    The decoration-time rebind analyzer intentionally suppresses
    branch-internal violations (its snapshot-restore scope truncates them)
    so that compile-time-if branch-selection rebinds stay legal. That
    leaves a runtime hole: rebinding a quantum variable inside a runtime
    branch — to a fresh allocation (``if cond: q = qmc.qubit("fresh")``)
    or to any other quantum value (``if cond: q = other``, including in
    both branches at once) — silently drops the variable's pre-branch
    state exactly when a rebinding branch is taken. The frontend records
    every branch-internal quantum binding change on the ``IfOperation``
    (``BranchRebind``, preserving the pre-branch value even when it no
    longer appears in any phi); this check verifies each record against
    each runtime execution path and raises ``QubitRebindError`` — the
    same ``AffineTypeError`` the decoration-time analyzer raises for a
    top-level rebind from a different quantum source, since this is that
    exact affine violation surfacing at runtime inside a branch — when
    the pre-branch value has no owner on a rebinding path: not consumed
    inside the taken branch, not carried out through any phi merge of
    that side, and not referenced by any operation outside the if.
    Scalar ``Qubit`` and whole-register ``Vector[Qubit]`` rebinds are
    covered alike.

    Loop bodies are covered more strictly, because neither loop kind can
    express a per-iteration quantum rebind today (see
    :func:`_check_loop_quantum_discards`): the frontend records quantum
    rebinds on ``ForOperation`` / ``ForItemsOperation`` /
    ``WhileOperation`` (``LoopCarriedRebind`` entries whose ``before`` is
    quantum), and each record is rejected unless it carries the incoming
    value forward on the same wires, or is a covered loop-invariant
    rebind (post-body value not produced in the body, body never reads
    the pre-loop value, pre-loop value owned outside). Body-produced
    rebinds are never exempt — an unrolled loop does not carry the
    rebound register between iterations and a runtime while re-executes
    its body on one persistent register without reset — and in-body
    consumption of the incoming value is not an exemption either: the
    read re-executes against the traced register every iteration and
    matches Python semantics only for the first one. A loop-body rebind
    needs no runtime/compile-time classification, so loops are checked
    wherever they appear on a live (non-pruned) path,
    trip-count-agnostically, exactly like the classical loop-carried
    check.

    ``IfOperation``s are classified with the same condition resolution
    ``CompileTimeIfLoweringPass`` uses (via ``bindings``), including for
    ifs nested inside runtime branches. A rebind confined to a
    compile-time branch stays legal when the surrounding control flow is
    compile-time too: rebinding to an alternative register under a
    compile-time flag is the documented branch-selection idiom, and a
    dead branch is eliminated entirely. A compile-time-TAKEN rebind
    nested inside a *runtime* branch inherits that branch's runtime
    conditionality and is checked (and rejected when it discards). Only
    ifs whose condition transitively derives from a measurement result
    are checked — the same taint analysis the classical-lowering pipeline
    uses, so expression-derived runtime conditions (``~bit``,
    ``a & b``) are covered; a non-measurement, non-compile-time condition
    cannot drive runtime branching and is rejected at emit by the shared
    condition resolution (though for this discard shape the emit-side phi
    physical-resource check can fire first with its generic message).

    What stays allowed:

    - consuming the original inside the branch before rebinding
      (``if cond: qmc.measure(q); q = qmc.qubit(...)``). Scalar
      consumption is judged at the wire's final in-branch version
      (:func:`_wire_terminally_consumed`), so gating the original and
      then dropping the gated state
      (``if cond: q = qmc.x(q); q = qmc.qubit("fresh")``) is a discard,
      not a consumption; whole-register rebinds keep the coarser
      any-touch granularity, where element or view reads count;
    - ordinary quantum rebinds through gates (``q = qmc.h(q)``) — the
      pre-branch value is carried out through the phi merge;
    - rebinds whose pre-branch value is still owned outside the if (a
      value consumed before the if, or an alias referenced after it);
    - handle exchanges where every pre-branch value is carried by some
      phi of the same side (``q1, q2 = q2, q1``).

    The check is deliberately conservative toward allowing where it
    cannot be exact. Phi lineage is over-approximated: producers without
    a positional qubit model (composite gates, controlled blocks, casts)
    contribute all of their quantum inputs as possible roots, so the
    carried exemption can only grow — a rejection requires the
    pre-branch value to be provably absent from every phi lineage.
    Outside-ownership evidence is path-sensitive with respect to
    enclosing ifs (a read on the sibling branch of an enclosing runtime
    if does not exempt), but path-insensitive for non-ancestor runtime
    ifs elsewhere: a value conditionally consumed downstream counts as
    owned. Rebinds inside compile-time-TAKEN ifs nested in a runtime
    branch are promoted to the enclosing if's check trip-count- and
    loop-agnostically; because the lowering pass erases those nested ifs
    (and their records), the promoted rebinds are only caught by the
    pre-fold ``PartialEvaluationPass`` hook, not by the ``AnalyzePass``
    safety net.

    Scope contract: the scan recurses through control-flow nesting only
    (``IfOperation`` branches and ``HasNestedOps`` bodies). Boxed
    implementation blocks — ``CompositeGateOperation.implementation_block``,
    ``InverseBlockOperation.implementation_block``,
    ``ControlledUOperation.block`` — are NOT descended into: they stay
    HIERARCHICAL recipe blocks outside the entrypoint pipeline, exactly
    like every other transpile-time rebind check
    (``reject_loop_carried_classical_rebinds``, ``AffineValidationPass``,
    both built on the same ``HasNestedOps`` walk). A discard written
    inside a composite's recipe kernel is therefore only covered by the
    decoration-time top-level analyzer, with the same branch/loop
    suppression as everywhere else pre-IR.

    Exposed at module scope because it runs from two passes:
    ``PartialEvaluationPass`` calls it before folding and if-lowering
    (with ``bindings``, so compile-time branches are classified exactly as
    the lowering pass will lower them), and ``AnalyzePass`` calls it again
    as a safety net for pipelines that skip ``partial_eval``.

    Args:
        operations (list[Operation]): Operations to scan. Recurses through
            all control flow; every runtime if and every loop at any
            nesting depth on a live path is checked.
        bindings (dict[str, Any] | None): Compile-time parameter bindings
            used to resolve ``IfOperation`` conditions, matching what
            ``CompileTimeIfLoweringPass`` will later resolve. Defaults to
            None (no bindings).

    Raises:
        QubitRebindError: If a runtime if branch rebinds a quantum variable
            whose pre-branch value has no consumer in that branch, no phi
            carrying it out of that side, and no reference outside the if —
            or a loop body rebinds a quantum variable whose incoming value
            it never consumes and whose pre-loop value has no owner outside
            the loop.
    """
    resolved_bindings = bindings or {}
    dependency_graph = build_dependency_graph(operations)
    measurement_tainted = find_measurement_derived_values(
        dependency_graph, find_measurement_results(operations)
    )
    _scan_branch_quantum_discards(
        operations,
        {},
        resolved_bindings,
        measurement_tainted,
        None,
        _DiscardScanCaches(),
    )


class AnalyzePass(Pass[Block, Block]):
    """Analyze and validate an affine block.

    This pass:
    1. Builds a dependency graph between values (used locally for validation)
    2. Validates that quantum ops don't depend on non-parameter classical results
    3. Checks that block inputs/outputs are classical

    Input: Block with BlockKind.AFFINE
    Output: Block with BlockKind.ANALYZED
    """

    @property
    def name(self) -> str:
        return "analyze"

    def run(self, input: Block) -> Block:
        """Analyze the block and validate dependencies."""
        if input.kind != BlockKind.AFFINE:
            raise ValidationError(f"AnalyzePass expects AFFINE block, got {input.kind}")

        # Check inputs/outputs are classical
        self._validate_io_classical(input)

        # Reject classical element stores inside runtime if/else branches
        self._reject_stores_in_if_branches(input.operations)

        # Reject in-loop classical element stores that read the same array
        self._reject_self_referential_loop_stores(input.operations)

        # Reject in-loop classical scalar rebinds (loop-carried updates)
        self._reject_loop_carried_classical_rebinds(
            input.operations, input.output_values
        )

        # Reject branch-internal and loop-body quantum rebinds that
        # discard quantum state
        self._reject_control_flow_quantum_discard(input.operations)

        # Build dependency graph
        dependency_graph = self._build_dependency_graph(input.operations)

        # Validate quantum-classical dependencies
        self._validate_quantum_dependencies(
            input.operations,
            dependency_graph,
            input.parameters,
        )

        # Validate ControlledUOperation fields (power is outside operands)
        self._validate_controlled_u_fields(input.operations)

        return dataclasses.replace(
            input,
            kind=BlockKind.ANALYZED,
        )

    def _validate_io_classical(self, block: Block) -> None:
        """Ensure all block inputs and outputs are classical types."""
        for value in block.input_values:
            if value.type.is_quantum():
                raise ValidationError(
                    f"Block input '{value.name}' must be classical type, "
                    f"got {value.type.label()}",
                    value_name=value.name,
                )

        for value in block.output_values:
            if value.type.is_quantum():
                raise ValidationError(
                    f"Block output '{value.name}' must be classical type, "
                    f"got {value.type.label()}",
                    value_name=value.name,
                )

    def _reject_stores_in_if_branches(self, operations: list[Operation]) -> None:
        """Reject classical element stores nested under a runtime if/else.

        A ``StoreArrayElementOperation`` produces a fresh SSA version of
        the array, but the frontend's branch tracing has no phi merge for
        array values: post-if reads would reference the branch-local
        version unconditionally, silently diverging from Python semantics
        when the branch is not taken.  Compile-time ``if``s (classical
        conditions resolvable from ``bindings``) were already flattened by
        ``CompileTimeIfLoweringPass`` before this pass runs, so only
        runtime (measurement-backed) branches reach this check.

        Args:
            operations (list[Operation]): Operations to scan.  Recurses
                through all control flow; any store anywhere inside an
                ``IfOperation`` branch (including nested loops) is
                rejected.

        Raises:
            ValidationError: If a classical element store appears inside
                an if/else branch.
        """

        def contains_store(ops: list[Operation]) -> bool:
            """Return whether any (nested) op is a classical element store.

            Args:
                ops (list[Operation]): Operations to scan recursively.

            Returns:
                bool: ``True`` if a ``StoreArrayElementOperation`` is
                    found at any nesting depth.
            """
            for op in ops:
                if isinstance(op, StoreArrayElementOperation):
                    return True
                if isinstance(op, HasNestedOps) and any(
                    contains_store(body) for body in op.nested_op_lists()
                ):
                    return True
            return False

        for op in operations:
            if isinstance(op, IfOperation):
                if any(contains_store(body) for body in op.nested_op_lists()):
                    raise ValidationError(
                        "Classical array element assignment inside an "
                        "if/else branch is not supported: array values "
                        "have no phi merge, so the write would apply "
                        "regardless of the branch taken. Restructure the "
                        "kernel to perform the assignment outside the "
                        "branch."
                    )
            elif isinstance(op, HasNestedOps):
                for body in op.nested_op_lists():
                    self._reject_stores_in_if_branches(body)

    def _reject_self_referential_loop_stores(
        self,
        operations: list[Operation],
    ) -> None:
        """Reject in-loop classical element stores that read the same array.

        Thin wrapper for the module-level
        :func:`reject_self_referential_loop_stores` so
        ``PartialEvaluationPass`` can reuse the same check without
        instantiating ``AnalyzePass``.  Called without bindings: by this
        stage ``CompileTimeIfLoweringPass`` has already eliminated
        compile-time ifs, so any surviving ``IfOperation`` classifies as
        runtime and its branches are skipped — safe, because
        :meth:`_reject_stores_in_if_branches` (run just before this
        check) already rejected every store inside an if branch.

        Args:
            operations (list[Operation]): Operations to scan recursively.

        Raises:
            ValidationError: If a store inside a loop transitively reads
                an element of the same logical array it writes.
        """
        reject_self_referential_loop_stores(operations)

    def _reject_loop_carried_classical_rebinds(
        self,
        operations: list[Operation],
        output_values: list[Value],
    ) -> None:
        """Reject in-loop classical scalar rebinds (loop-carried updates).

        Thin wrapper for the module-level
        :func:`reject_loop_carried_classical_rebinds` so
        ``PartialEvaluationPass`` can reuse the same check without
        instantiating ``AnalyzePass``. Called without bindings: by this
        stage ``CompileTimeIfLoweringPass`` has already eliminated
        compile-time ifs, so any surviving ``IfOperation`` classifies as
        runtime.

        Args:
            operations (list[Operation]): Operations to scan recursively.
            output_values (list[Value]): Block output values, used to
                detect a while condition's pre-loop value escaping the
                loop through the return path.

        Raises:
            ValidationError: If a loop body rebinds a classical scalar
                whose pre-loop value the body still reads, or a while
                condition's pre-loop value is read after the loop.
        """
        reject_loop_carried_classical_rebinds(operations, output_values=output_values)

    def _reject_control_flow_quantum_discard(
        self,
        operations: list[Operation],
    ) -> None:
        """Reject control-flow-internal quantum rebinds that discard state.

        Thin wrapper around the module-level
        :func:`reject_control_flow_quantum_discard`, run by this pass
        as a safety net (the same module-level function also runs earlier
        and directly from ``PartialEvaluationPass``, pre-fold, with
        bindings). Invoked here without bindings, so only already-constant
        conditions resolve as compile-time and every other ``IfOperation``
        is treated as runtime; of those, only the ones whose condition is
        measurement-derived are actually checked. In the normal pipeline
        ``CompileTimeIfLoweringPass`` has already folded away the
        compile-time ifs before ``analyze`` runs, so this bindings-free
        call is exact for the surviving runtime ifs' own records; records
        that lived on compile-time-TAKEN nested ifs were erased together
        with those ifs by the lowering and are only caught by the
        pre-fold hook. When ``AnalyzePass`` runs standalone (no
        ``partial_eval``) an unfolded compile-time if with a
        non-measurement condition simply is not checked, which is safe.
        Loop-body records survive the lowering on their loop operations,
        so the loop side of the check is exact here too.

        Args:
            operations (list[Operation]): Operations to scan recursively.

        Raises:
            QubitRebindError: If a runtime if branch or a loop body
                rebinds a quantum variable whose incoming value has no
                owner on that path (no in-scope consumer, no carrying
                phi, and no reference outside the construct).
        """
        reject_control_flow_quantum_discard(operations)

    def _build_dependency_graph(
        self,
        operations: list[Operation],
    ) -> dict[str, set[str]]:
        """Build a map from each value UUID to the UUIDs it depends on.

        Thin wrapper for the module-level :func:`build_dependency_graph` so
        external passes can reuse the same logic without instantiating
        ``AnalyzePass``.
        """
        return build_dependency_graph(operations)

    def _validate_quantum_dependencies(
        self,
        operations: list[Operation],
        dependency_graph: dict[str, set[str]],
        parameters: dict[str, Value],
    ) -> None:
        """Ensure quantum ops don't depend on non-parameter classical results.

        A quantum operation can depend on:
        - Other quantum values
        - Parameters (will be bound at runtime)
        - Constant classical values

        It cannot depend on:
        - Classical values computed from measurements (runtime-only)
        """
        # Collect UUIDs of parameter values
        parameter_uuids = {v.uuid for v in parameters.values()}

        # Collect UUIDs of measurement results (runtime classical)
        measurement_uuids = self._find_measurement_results(operations)

        # Collect UUIDs of values derived from measurements
        derived_from_measurement: set[str] = set()
        self._find_measurement_derived_values(
            dependency_graph, measurement_uuids, derived_from_measurement
        )

        outer_self = self

        class QuantumDependencyValidator(ControlFlowVisitor):
            def visit_operation(self, op: Operation) -> None:
                if op.operation_kind != OperationKind.QUANTUM:
                    return

                for operand in op.operands:
                    if not isinstance(operand, ValueBase):
                        continue

                    # Quantum-typed operands (e.g. phi-merged qubits) are not
                    # subject to the measurement-dependency ban.  Only classical
                    # operands must be free of measurement derivation.
                    if operand.type.is_quantum():
                        continue

                    if outer_self._depends_on_measurement(
                        operand.uuid,
                        dependency_graph,
                        measurement_uuids,
                        parameter_uuids,
                        derived_from_measurement,
                    ):
                        raise DependencyError(
                            f"Quantum operation '{type(op).__name__}' depends on "
                            f"measurement result via value '{operand.name}'. "
                            f"JIT compilation not supported - classical values "
                            f"used in quantum ops must be parameters or constants.",
                            quantum_op=type(op).__name__,
                            classical_value=operand.name,
                        )

        validator = QuantumDependencyValidator()
        validator.visit_operations(operations)

    def _validate_controlled_u_fields(
        self,
        operations: list[Operation],
    ) -> None:
        """Validate ``ControlledUOperation``-specific dataclass fields.

        ``power`` lives outside ``op.operands``, so the generic
        dependency validation does not cover it.  This method rejects
        statically-decidable invalid concrete values (``<= 0``,
        ``bool``, non-integer) while allowing unresolved symbolic
        ``Value`` instances that will be resolved at emit time.

        Args:
            operations: The affine operation list to validate.

        Raises:
            ValidationError: If a concrete ``power`` value is invalid.
        """
        from qamomile.circuit.ir.operation.gate import ControlledUOperation

        def _validate_concrete_power(value: object, op: ControlledUOperation) -> None:
            if isinstance(value, bool):
                raise ValidationError(
                    f"ControlledU power must be a positive integer, got bool ({value})."
                )
            if not isinstance(value, (int, float)):
                raise ValidationError(
                    f"ControlledU power must be a positive integer, "
                    f"got {type(value).__name__}."
                )
            if isinstance(value, float) and value != int(value):
                raise ValidationError(
                    f"ControlledU power must be an integer, "
                    f"got non-integer float {value}."
                )
            int_val = int(value)
            if int_val <= 0:
                raise ValidationError(
                    f"ControlledU power must be strictly positive, got {int_val}."
                )

        class ControlledUValidator(ControlFlowVisitor):
            def visit_operation(self, op: Operation) -> None:
                if not isinstance(op, ControlledUOperation):
                    return
                power = op.power
                if isinstance(power, Value):
                    if power.is_constant():
                        const = power.get_const()
                        if const is not None:
                            _validate_concrete_power(const, op)
                    return
                _validate_concrete_power(power, op)

        validator = ControlledUValidator()
        validator.visit_operations(operations)

    def _find_measurement_results(
        self,
        operations: list[Operation],
    ) -> set[str]:
        """Wrapper for :func:`find_measurement_results`."""
        return find_measurement_results(operations)

    def _find_measurement_derived_values(
        self,
        dependency_graph: dict[str, set[str]],
        measurement_uuids: set[str],
        derived: set[str],
    ) -> None:
        """Populate ``derived`` with all measurement-derived UUIDs.

        Wrapper for :func:`find_measurement_derived_values` that updates
        the caller's ``derived`` set in place (preserves the original
        method signature).
        """
        derived.update(
            find_measurement_derived_values(dependency_graph, measurement_uuids)
        )

    def _depends_on_measurement(
        self,
        value_uuid: str,
        dependency_graph: dict[str, set[str]],
        measurement_uuids: set[str],
        parameter_uuids: set[str],
        derived_from_measurement: set[str],
    ) -> bool:
        """Check if a value transitively depends on a measurement result.

        Returns True if there's a dependency path to a measurement
        that doesn't go through a parameter.
        """
        # Direct measurement dependency
        if value_uuid in measurement_uuids:
            return True

        # Derived from measurement
        if value_uuid in derived_from_measurement:
            return True

        # Parameters are OK
        if value_uuid in parameter_uuids:
            return False

        # Check transitive dependencies
        visited: set[str] = set()

        def dfs(uuid: str) -> bool:
            if uuid in visited:
                return False
            visited.add(uuid)

            # Direct measurement
            if uuid in measurement_uuids:
                return True

            # Parameters are OK
            if uuid in parameter_uuids:
                return False

            # Check dependencies
            deps = dependency_graph.get(uuid, set())
            return any(dfs(dep) for dep in deps)

        return dfs(value_uuid)
