"""Analyze pass: Validate and analyze dependencies in an affine block."""

from __future__ import annotations

import dataclasses
from typing import Any

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.arithmetic_operations import PhiOp
from qamomile.circuit.ir.operation.classical_ops import StoreArrayElementOperation
from qamomile.circuit.ir.operation.control_flow import (
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
from qamomile.circuit.ir.value import Value, ValueBase
from qamomile.circuit.transpiler.errors import DependencyError, ValidationError
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.passes.compile_time_if_lowering import (
    evaluate_classical_op_concrete,
    resolve_compile_time_condition,
)
from qamomile.circuit.transpiler.passes.control_flow_visitor import ControlFlowVisitor
from qamomile.circuit.transpiler.passes.validate_while import (
    build_producer_map,
    is_measurement_backed,
)

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
    for each result UUID, the set of operand UUIDs that produced it. Also
    seeds an edge from each ``ArrayValue`` element (``Value`` carrying
    ``parent_array``) to its parent array UUID, and walks the parent's
    ``slice_of`` chain so that a sliced view (e.g. ``s[0:4:2][i]`` for
    ``s = qmc.measure(register)``) inherits taint from the root measured
    array. ``StripSliceArrayOpsPass`` removes the explicit
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
            operand_uuids = {v.uuid for v in op.operands if isinstance(v, ValueBase)}
            for result in op.results:
                if result.uuid not in self.graph:
                    self.graph[result.uuid] = set()
                self.graph[result.uuid].update(operand_uuids)
            for v in op.operands:
                if not isinstance(v, ValueBase):
                    continue
                parent = getattr(v, "parent_array", None)
                if parent is None:
                    continue
                self.graph.setdefault(v.uuid, set()).add(parent.uuid)
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


def prune_compile_time_ifs(
    ops: list[Operation],
    concrete_values: dict[str, Any],
    bindings: dict[str, Any],
) -> list[Operation]:
    """Replace compile-time-decidable ``IfOperation``s by their taken branch.

    Mirrors ``CompileTimeIfLoweringPass``: conditions are resolved with
    the shared ``resolve_compile_time_condition`` /
    ``evaluate_classical_op_concrete`` helpers so the taken / dead /
    runtime classification here cannot disagree with the branch the
    lowering pass will actually keep. For a resolved condition the taken
    branch's operations are inlined (recursively pruned) and each
    ``PhiOp`` is reduced to its selected source operand, so phi-mediated
    dataflow out of the branch stays visible to dependency scans without
    dead-branch edges. Runtime ``IfOperation``s are kept intact.

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

    Returns:
        list[Operation]: The pruned view of ``ops``.
    """
    pruned: list[Operation] = []
    for op in ops:
        evaluate_classical_op_concrete(op, concrete_values, bindings)
        if isinstance(op, IfOperation):
            taken = resolve_compile_time_condition(
                op.condition, concrete_values, bindings
            )
            if taken is None:
                pruned.append(op)
                continue
            branch = op.true_operations if taken else op.false_operations
            pruned.extend(prune_compile_time_ifs(branch, concrete_values, bindings))
            for phi in op.phi_ops:
                if isinstance(phi, PhiOp):
                    selected = phi.true_value if taken else phi.false_value
                    pruned.append(dataclasses.replace(phi, operands=[selected]))
            continue
        if isinstance(op, HasNestedOps):
            op = op.rebuild_nested(
                [
                    prune_compile_time_ifs(body, dict(concrete_values), bindings)
                    for body in op.nested_op_lists()
                ]
            )
        pruned.append(op)
    return pruned


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

    def check_loop_body(body_ops: list[Operation]) -> None:
        """Reject self-referential stores inside one (pruned) loop body.

        Builds a dependency graph restricted to the loop body (reads
        performed before the loop are loop-invariant and fold correctly,
        so they must not trigger a rejection) and BFS-walks it from each
        store's value and index operands.  The graph and value table
        cover the full body — including surviving runtime-if internals,
        so a store after an if still sees phi-mediated reads — while the
        store scan itself skips if branches (those stores are rejected
        by ``AnalyzePass._reject_stores_in_if_branches``).

        Args:
            body_ops (list[Operation]): The loop's top-level body
                operations, already pruned of compile-time-decidable
                if branches.

        Raises:
            ValidationError: If a scanned store's value or index
                transitively reads an element of the array the store
                writes.
        """
        dependency_graph = build_dependency_graph(body_ops)
        flat_ops = flatten_ops(body_ops)

        value_table: dict[str, ValueBase] = {}
        for op in flat_ops:
            for value in (*op.all_input_values(), *op.results):
                register_value(value, value_table)

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

    pruned_operations = prune_compile_time_ifs(operations, {}, resolved_bindings)
    for op in flatten_ops(pruned_operations, into_if_branches=False):
        if isinstance(op, (ForOperation, ForItemsOperation, WhileOperation)):
            check_loop_body(
                [body_op for body in op.nested_op_lists() for body_op in body]
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
        f"@qkernel {loop_kind} loop is not supported: the loop body is "
        f"traced once, so '{var_name}' on the right-hand side is fixed to "
        f"its pre-loop value instead of the previous iteration's value, "
        f"and the compiled program would silently diverge from Python "
        f"semantics. Compute the reduction in ordinary Python instead — "
        f"outside the @qkernel or in an undecorated helper function — or "
        f"express each iteration's value directly from the loop index. "
        f"Note: builtin range() inside @qkernel is traced exactly like "
        f"qmc.range()."
    )


def _check_loop_carried_rebinds(
    loop_op: ForOperation | ForItemsOperation | WhileOperation,
    producer_map: dict[str, Operation],
) -> None:
    """Reject the loop-carried rebind records of one (pruned) loop op.

    Args:
        loop_op (ForOperation | ForItemsOperation | WhileOperation): Loop
            operation whose body has already been pruned of
            compile-time-decidable if branches.
        producer_map (dict[str, Operation]): Block-wide map from result
            UUID to producing operation, used to classify
            measurement-backed values (pre-loop producers included).

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

    # Map collapsed (single-operand) PhiOps left by dead-branch pruning:
    # result uuid -> selected source uuid.
    collapsed: dict[str, str] = {}
    value_table: dict[str, ValueBase] = {}
    for op in flat_body:
        if isinstance(op, PhiOp) and len(op.operands) == 1 and op.results:
            collapsed[op.results[0].uuid] = op.operands[0].uuid
        for v in (*op.all_input_values(), *op.results):
            value_table.setdefault(v.uuid, v)

    def canonical(uuid: str) -> str:
        """Follow collapsed-phi links to the underlying source uuid.

        Args:
            uuid (str): Starting value uuid.

        Returns:
            str: The uuid after following single-operand phi links.
        """
        seen: set[str] = set()
        while uuid in collapsed and uuid not in seen:
            seen.add(uuid)
            uuid = collapsed[uuid]
        return uuid

    def op_read_uuids(op: Operation) -> set[str]:
        """Collect the uuids an operation genuinely reads.

        Nested loop operations expose their own rebind records through
        ``all_input_values`` (for cloning); those record values are not
        body reads and must not trigger the outer loop's check, so they
        are subtracted before the operands are re-added.

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
        uuids = {v.uuid for v in op.all_input_values()}
        uuids -= excluded
        for v in op.operands:
            operand_uuid = getattr(v, "uuid", None)
            if operand_uuid is not None:
                uuids.add(operand_uuid)
        return uuids

    body_read_uuids: set[str] = set()
    for op in flat_body:
        body_read_uuids |= op_read_uuids(op)

    before_uuids = {r.before.uuid for r in records}

    for record in records:
        # Legal while-loop loop-carried condition: the (initial, updated)
        # condition pair is aliased onto one clbit by the allocator.
        if (
            isinstance(loop_op, WhileOperation)
            and len(loop_op.operands) >= 2
            and record.before.uuid == loop_op.operands[0].uuid
            and record.after.uuid == loop_op.operands[1].uuid
        ):
            continue

        # Legal measurement-backed Bit rebind: a per-iteration
        # re-measurement (``state = qmc.measure(...)``, possibly merged
        # through an if) lives on a physical clbit that the resource
        # allocator aliases across iterations, so reads of the pre-loop
        # value observe the updated bit at runtime. This requires the
        # pre-loop value to be measurement-backed too — a constant
        # initial Bit would resolve branch conditions at compile time
        # and diverge from Python semantics.
        if is_measurement_backed(record.after, producer_map) and is_measurement_backed(
            record.before, producer_map
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
    operations (``LoopCarriedRebind``); this check rejects the ones that
    survive dead-branch pruning.

    ``IfOperation``s are classified with the same condition resolution
    ``CompileTimeIfLoweringPass`` uses (via ``bindings``): a rebind whose
    only path is a compile-time-dead branch canonicalizes back to the
    pre-loop value and is allowed. Unlike the array-store check, loops
    nested inside *runtime* if branches are scanned too — a loop-carried
    scalar rebind there miscompiles all the same.

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

    Raises:
        ValidationError: If a loop body rebinds a classical scalar whose
            pre-loop value the body still reads (directly, through
            classical arithmetic, through a surviving phi, or as an
            embedded constant from a plain-Python initialization).
    """
    resolved_bindings = bindings or {}
    pruned_operations = prune_compile_time_ifs(operations, {}, resolved_bindings)
    producer_map: dict[str, Operation] = {}
    build_producer_map(pruned_operations, producer_map)
    for op in flatten_ops(pruned_operations):
        if isinstance(op, (ForOperation, ForItemsOperation, WhileOperation)):
            _check_loop_carried_rebinds(op, producer_map)


def _branch_quantum_discard_error(
    fresh_value: Value,
    original_value: Value,
    branch_label: str,
) -> ValidationError:
    """Build the targeted branch-internal quantum discard rejection error.

    Args:
        fresh_value (Value): The freshly allocated quantum value the phi
            selects when the rebinding branch is taken.
        original_value (Value): The pre-branch quantum value whose state
            the taken branch would silently drop.
        branch_label (str): Human-readable branch side ("true" / "false").

    Returns:
        ValidationError: The error to raise.
    """
    fresh_name = fresh_value.name or "<anonymous>"
    original_name = original_value.name or "<anonymous>"
    return ValidationError(
        f"Branch-internal fresh allocation '{fresh_name}' in the "
        f"{branch_label} branch of a runtime if conditionally discards the "
        f"quantum state of '{original_name}': when the branch is taken, the "
        f"freshly allocated qubit replaces '{original_name}' without the "
        f"branch ever consuming the original state, so that state would be "
        f"silently dropped at runtime. Consume the original state before "
        f"branching or inside the branch (e.g. qmc.measure(...)), or apply "
        f"gates to the same qubit in both branches instead of allocating a "
        f"new one."
    )


def _is_branch_fresh_allocation(
    value: Value,
    branch_producers: dict[str, Operation],
    visiting: set[str] | None = None,
) -> bool:
    """Check whether a quantum value originates from an in-branch allocation.

    Traces the value's qubit lineage backwards through the branch-local
    producer map: plain gates map each result to the qubit operand at the
    same position, and phi merges (nested runtime ifs, or single-operand
    phis collapsed by dead-branch pruning) count as fresh only when every
    reachable leaf is fresh. The trace is deliberately conservative: any
    producer it does not understand (composite gates, controlled blocks,
    casts, ...) makes the value count as not-fresh, so the discard check
    errs toward allowing.

    Args:
        value (Value): The quantum value to trace (a phi branch input).
        branch_producers (dict[str, Operation]): Result-UUID-to-producer map
            restricted to the branch body (nested control flow included).
            A value with no producer here comes from outside the branch.
        visiting (set[str] | None): UUIDs on the current DFS path, used to
            break cycles with backtracking (the same value may be reached
            from both sides of a nested phi). Defaults to None (fresh
            traversal).

    Returns:
        bool: True if the value provably originates from a
            ``QInitOperation`` inside the branch.
    """
    if visiting is None:
        visiting = set()
    if value.uuid in visiting:
        return False
    visiting.add(value.uuid)
    try:
        producer = branch_producers.get(value.uuid)
        if producer is None:
            return False
        if isinstance(producer, QInitOperation):
            return True
        if isinstance(producer, GateOperation):
            qubit_operands = producer.qubit_operands
            for index, result in enumerate(producer.results):
                if result.uuid == value.uuid and index < len(qubit_operands):
                    return _is_branch_fresh_allocation(
                        qubit_operands[index], branch_producers, visiting
                    )
            return False
        if isinstance(producer, PhiOp):
            if len(producer.operands) == 1:
                return _is_branch_fresh_allocation(
                    producer.operands[0], branch_producers, visiting
                )
            if len(producer.operands) < 3:
                return False
            return _is_branch_fresh_allocation(
                producer.true_value, branch_producers, visiting
            ) and _is_branch_fresh_allocation(
                producer.false_value, branch_producers, visiting
            )
        return False
    finally:
        visiting.discard(value.uuid)


def _trace_pre_branch_root(
    value: Value,
    branch_producers: dict[str, Operation],
    visiting: set[str] | None = None,
) -> Value | None:
    """Trace a phi input back to the pre-branch value it descends from.

    Follows the same qubit lineage steps as
    :func:`_is_branch_fresh_allocation` (positional gate operands, phi
    merges) through the branch-local producer map until it reaches a value
    with no in-branch producer — the state the variable held before the
    if. A phi merge yields a root only when both sides converge on the
    same pre-branch value.

    Args:
        value (Value): The quantum value to trace (a phi branch input).
        branch_producers (dict[str, Operation]): Result-UUID-to-producer map
            restricted to the branch body (nested control flow included).
        visiting (set[str] | None): UUIDs on the current DFS path, used to
            break cycles with backtracking. Defaults to None (fresh
            traversal).

    Returns:
        Value | None: The pre-branch root value, or None when the lineage
            allocates fresh in-branch (``QInitOperation``), diverges, or
            passes through a producer the trace does not understand.
    """
    if visiting is None:
        visiting = set()
    if value.uuid in visiting:
        return None
    visiting.add(value.uuid)
    try:
        producer = branch_producers.get(value.uuid)
        if producer is None:
            return value
        if isinstance(producer, GateOperation):
            qubit_operands = producer.qubit_operands
            for index, result in enumerate(producer.results):
                if result.uuid == value.uuid and index < len(qubit_operands):
                    return _trace_pre_branch_root(
                        qubit_operands[index], branch_producers, visiting
                    )
            return None
        if isinstance(producer, PhiOp):
            if len(producer.operands) == 1:
                return _trace_pre_branch_root(
                    producer.operands[0], branch_producers, visiting
                )
            if len(producer.operands) < 3:
                return None
            true_root = _trace_pre_branch_root(
                producer.true_value, branch_producers, visiting
            )
            if true_root is None:
                return None
            false_root = _trace_pre_branch_root(
                producer.false_value, branch_producers, visiting
            )
            if false_root is None or false_root.uuid != true_root.uuid:
                return None
            return true_root
        return None
    finally:
        visiting.discard(value.uuid)


def _branch_referenced_uuids(branch_ops: list[Operation]) -> set[str]:
    """Collect every value UUID a branch body references as an input.

    Covers all nested operations (via ``flatten_ops``), and for each input
    value also records its ``parent_array`` / ``slice_of`` ancestry so an
    element or view read of a register counts as touching the register
    itself. Used as the "did this branch consume the original state"
    evidence — any reference is enough to disqualify the discard error,
    which errs toward allowing.

    Args:
        branch_ops (list[Operation]): The branch body operations, already
            pruned of compile-time-decidable ifs.

    Returns:
        set[str]: UUIDs of every referenced input value and its array
            ancestry.
    """
    referenced: set[str] = set()

    def add_with_ancestry(value: ValueBase) -> None:
        """Record a value's UUID together with its array ancestry.

        Args:
            value (ValueBase): Input value read by a branch operation.
        """
        if value.uuid in referenced:
            return
        referenced.add(value.uuid)
        for attr in ("parent_array", "slice_of"):
            ancestor = getattr(value, attr, None)
            if ancestor is not None:
                add_with_ancestry(ancestor)

    for op in flatten_ops(branch_ops):
        for value in op.all_input_values():
            add_with_ancestry(value)
    return referenced


def _check_branch_quantum_discard(
    if_op: IfOperation,
    concrete_values: dict[str, Any],
    bindings: dict[str, Any],
) -> None:
    """Check one runtime ``IfOperation``'s phi merges for conditional discards.

    For each phi merge and each branch side, flags the pairing of an
    in-branch fresh allocation with a pre-branch quantum value that the
    allocating branch never references: taking that branch would replace
    the variable with the fresh qubit and silently drop the original
    state. Branch bodies are pruned of compile-time-decidable nested ifs
    first (each side with its own copy of ``concrete_values``, matching
    ``CompileTimeIfLoweringPass`` scoping) so dead-branch allocations and
    dead-branch consumes do not distort the analysis.

    Args:
        if_op (IfOperation): Runtime if whose condition did not resolve at
            compile time.
        concrete_values (dict[str, Any]): UUID-keyed concrete classical-op
            results accumulated up to this operation.
        bindings (dict[str, Any]): Compile-time parameter bindings.

    Raises:
        ValidationError: If a phi merge pairs an in-branch fresh allocation
            with a pre-branch quantum value the allocating branch never
            consumes.
    """
    pruned_branches = {
        True: prune_compile_time_ifs(
            if_op.true_operations, dict(concrete_values), bindings
        ),
        False: prune_compile_time_ifs(
            if_op.false_operations, dict(concrete_values), bindings
        ),
    }
    producers: dict[bool, dict[str, Operation]] = {}
    referenced: dict[bool, set[str]] = {}
    for side, branch_ops in pruned_branches.items():
        side_producers: dict[str, Operation] = {}
        build_producer_map(branch_ops, side_producers)
        producers[side] = side_producers
        referenced[side] = _branch_referenced_uuids(branch_ops)

    for phi in if_op.phi_ops:
        if len(phi.operands) < 3 or not phi.results:
            continue
        for side, label in ((True, "true"), (False, "false")):
            fresh_candidate = phi.true_value if side else phi.false_value
            other_side_value = phi.false_value if side else phi.true_value
            if not isinstance(fresh_candidate, Value) or not isinstance(
                other_side_value, Value
            ):
                continue
            if not fresh_candidate.type.is_quantum():
                continue
            if not _is_branch_fresh_allocation(fresh_candidate, producers[side]):
                continue
            original = _trace_pre_branch_root(other_side_value, producers[not side])
            if original is None or not original.type.is_quantum():
                continue
            # Defensive: the pre-branch root must not be produced inside
            # either branch (branches are traced from the same pre-if
            # state, so this cannot normally happen).
            if original.uuid in producers[True] or original.uuid in producers[False]:
                continue
            if original.uuid in referenced[side]:
                continue
            raise _branch_quantum_discard_error(fresh_candidate, original, label)


def _scan_branch_quantum_discards(
    ops: list[Operation],
    concrete_values: dict[str, Any],
    bindings: dict[str, Any],
    producer_map: dict[str, Operation],
) -> None:
    """Walk operations like the if-lowering pass, checking each runtime if.

    Mirrors ``CompileTimeIfLoweringPass``'s classification walk:
    classical-op results are accumulated in program order, a
    compile-time-resolvable ``IfOperation`` contributes only its taken
    branch to the scan, and a runtime ``IfOperation`` has both branches
    scanned with their own copy of the accumulated state. This per-position
    classification is what lets a fresh allocation inside a compile-time
    branch *nested within* a runtime branch stay legal —
    ``prune_compile_time_ifs`` alone does not descend into runtime-if
    bodies.

    Args:
        ops (list[Operation]): Operations to scan, in program order.
        concrete_values (dict[str, Any]): UUID-keyed concrete classical-op
            results accumulated along the walk. Updated in place for the
            current scope; nested scopes receive copies.
        bindings (dict[str, Any]): Compile-time parameter bindings used to
            resolve ``IfOperation`` conditions.
        producer_map (dict[str, Operation]): Block-wide result-UUID-to-
            producer map used to classify measurement-backed conditions.

    Raises:
        ValidationError: If a runtime if branch pairs a fresh allocation
            with a never-consumed pre-branch quantum value.
    """
    for op in ops:
        evaluate_classical_op_concrete(op, concrete_values, bindings)
        if isinstance(op, IfOperation):
            taken = resolve_compile_time_condition(
                op.condition, concrete_values, bindings
            )
            if taken is not None:
                branch = op.true_operations if taken else op.false_operations
                _scan_branch_quantum_discards(
                    branch, concrete_values, bindings, producer_map
                )
                continue
            if is_measurement_backed(op.condition, producer_map):
                _check_branch_quantum_discard(op, concrete_values, bindings)
            _scan_branch_quantum_discards(
                op.true_operations, dict(concrete_values), bindings, producer_map
            )
            _scan_branch_quantum_discards(
                op.false_operations, dict(concrete_values), bindings, producer_map
            )
            continue
        if isinstance(op, HasNestedOps):
            for body in op.nested_op_lists():
                _scan_branch_quantum_discards(
                    body, dict(concrete_values), bindings, producer_map
                )
        # Other operations carry no nested control flow to scan.


def reject_branch_internal_quantum_discard(
    operations: list[Operation],
    bindings: dict[str, Any] | None = None,
) -> None:
    """Reject runtime-branch fresh allocations that discard quantum state.

    The decoration-time rebind analyzer intentionally suppresses
    branch-internal violations (its snapshot-restore scope truncates them)
    so that compile-time-if dead-branch rebinds stay legal. That leaves a
    runtime hole: ``if cond: q = qmc.qubit("fresh")`` with a
    measurement-backed ``cond`` silently drops the original state of ``q``
    exactly when the branch is taken. The IR makes the hazard visible as a
    phi merge pairing an in-branch ``QInitOperation``-rooted value with a
    pre-branch quantum value the allocating branch never consumes; this
    check rejects that shape with a targeted error instead of letting it
    surface as an emit-time physical-resource mismatch (or pass silently
    on paths without that guard). Scalar ``Qubit`` and whole-register
    ``Vector[Qubit]`` rebinds are covered alike — both merge through a
    single phi.

    ``IfOperation``s are classified with the same condition resolution
    ``CompileTimeIfLoweringPass`` uses (via ``bindings``), including for
    ifs nested inside runtime branches: a fresh allocation confined to a
    compile-time branch — dead or taken — stays legal, because rebinding
    to an alternative register under a compile-time flag is the documented
    branch-selection idiom. Only ifs whose condition is measurement-backed
    are checked; a non-measurement, non-compile-time condition cannot
    drive runtime branching and is rejected at emit by the shared
    condition resolution (though for this discard shape the emit-side phi
    physical-resource check can fire first with its generic message).

    What stays allowed:

    - consuming the original inside the branch before re-allocating
      (``if cond: qmc.measure(q); q = qmc.qubit(...)``) — any reference to
      the original, including element or view reads of a register, counts;
    - ordinary quantum rebinds through gates (``q = qmc.h(q)``);
    - fresh allocations under compile-time-resolvable conditions;
    - both branches allocating fresh values (the pre-branch value is then
      not part of the phi merge, so it is outside this check's evidence).

    The check is deliberately conservative toward allowing: qubit lineage
    is traced only through plain gates and phi merges, so a fresh value
    routed through composite gates, controlled blocks, or casts is not
    flagged.

    Exposed at module scope because it runs from two passes:
    ``PartialEvaluationPass`` calls it before folding and if-lowering
    (with ``bindings``, so compile-time branches are classified exactly as
    the lowering pass will lower them), and ``AnalyzePass`` calls it again
    as a safety net for pipelines that skip ``partial_eval``.

    Args:
        operations (list[Operation]): Operations to scan. Recurses through
            all control flow; every runtime if at any nesting depth is
            checked.
        bindings (dict[str, Any] | None): Compile-time parameter bindings
            used to resolve ``IfOperation`` conditions, matching what
            ``CompileTimeIfLoweringPass`` will later resolve. Defaults to
            None (no bindings).

    Raises:
        ValidationError: If a runtime if branch rebinds a quantum variable
            to an in-branch fresh allocation while the branch never
            consumes the pre-branch value.
    """
    resolved_bindings = bindings or {}
    producer_map: dict[str, Operation] = {}
    build_producer_map(operations, producer_map)
    _scan_branch_quantum_discards(operations, {}, resolved_bindings, producer_map)


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
        self._reject_loop_carried_classical_rebinds(input.operations)

        # Reject branch-internal fresh allocations that conditionally
        # discard quantum state
        self._reject_branch_internal_quantum_discard(input.operations)

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

        Raises:
            ValidationError: If a loop body rebinds a classical scalar
                whose pre-loop value the body still reads.
        """
        reject_loop_carried_classical_rebinds(operations)

    def _reject_branch_internal_quantum_discard(
        self,
        operations: list[Operation],
    ) -> None:
        """Reject runtime-branch fresh allocations that discard quantum state.

        Thin wrapper for the module-level
        :func:`reject_branch_internal_quantum_discard` so
        ``PartialEvaluationPass`` can reuse the same check without
        instantiating ``AnalyzePass``. Called without bindings: by this
        stage ``CompileTimeIfLoweringPass`` has already eliminated
        compile-time ifs, so any surviving ``IfOperation`` classifies as
        runtime.

        Args:
            operations (list[Operation]): Operations to scan recursively.

        Raises:
            ValidationError: If a runtime if branch rebinds a quantum
                variable to an in-branch fresh allocation while the branch
                never consumes the pre-branch value.
        """
        reject_branch_internal_quantum_discard(operations)

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
