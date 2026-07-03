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


def _reject_stale_while_condition_reads(
    pruned_operations: list[Operation],
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

    Args:
        pruned_operations (list[Operation]): Program operations with
            compile-time-dead if branches already pruned.
        output_uuids (set[str]): UUIDs of the block's output values
            (post-loop reads through the kernel return path).

    Raises:
        ValidationError: If the initial-condition value of a
            loop-carried while is read by any operation after the loop
            or escapes through the block outputs.
    """
    flat = flatten_ops(pruned_operations)
    for index, op in enumerate(flat):
        if not isinstance(op, WhileOperation) or len(op.operands) < 2:
            continue
        initial = op.operands[0]
        initial_uuid = getattr(initial, "uuid", None)
        if initial_uuid is None:
            continue
        # Pre-order flattening lists the loop body right after the loop
        # op; skip those entries (in-body reads are correct under the
        # aliasing) and scan only what follows the loop.
        body_ids = {
            id(body_op)
            for body in op.nested_op_lists()
            for body_op in flatten_ops(body)
        }
        stale_read = any(
            initial_uuid in _op_read_uuids(later)
            for later in flat[index + 1 :]
            if id(later) not in body_ids
        )
        if stale_read or initial_uuid in output_uuids:
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
) -> None:
    """Reject the loop-carried rebind records of one (pruned) loop op.

    Args:
        loop_op (ForOperation | ForItemsOperation | WhileOperation): Loop
            operation whose body has already been pruned of
            compile-time-decidable if branches.

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

    body_read_uuids: set[str] = set()
    for op in flat_body:
        body_read_uuids |= _op_read_uuids(op)

    before_uuids = {r.before.uuid for r in records}

    for record in records:
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
    operations (``LoopCarriedRebind``); this check rejects the ones that
    survive dead-branch pruning.

    ``IfOperation``s are classified with the same condition resolution
    ``CompileTimeIfLoweringPass`` uses (via ``bindings``): a rebind whose
    only path is a compile-time-dead branch canonicalizes back to the
    pre-loop value and is allowed. Unlike the array-store check, loops
    nested inside *runtime* if branches are scanned too — a loop-carried
    scalar rebind there miscompiles all the same.

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
    pruned_operations = prune_compile_time_ifs(operations, {}, resolved_bindings)
    output_uuids = {
        v.uuid for v in (output_values or []) if getattr(v, "uuid", None) is not None
    }
    _reject_stale_while_condition_reads(pruned_operations, output_uuids)
    for op in flatten_ops(pruned_operations):
        if isinstance(op, (ForOperation, ForItemsOperation, WhileOperation)):
            _check_loop_carried_rebinds(op)


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


def _trace_pre_branch_root(
    value: Value,
    branch_producers: dict[str, Operation],
    visiting: set[str] | None = None,
) -> Value | None:
    """Trace a phi input back to the quantum value its lineage starts at.

    Follows positional gate operands and phi merges (nested runtime ifs,
    or single-operand phis collapsed by dead-branch pruning) backwards
    through the branch-local producer map. The root is either a value
    with no in-branch producer (the state the variable held before the
    if) or an in-branch ``QInitOperation`` result (a fresh allocation —
    a valid lineage start that simply never equals a pre-branch value).
    A phi merge yields a root only when both sides converge on the same
    root.

    Args:
        value (Value): The quantum value to trace (a phi branch input).
        branch_producers (dict[str, Operation]): Result-UUID-to-producer map
            restricted to the branch body (nested control flow included).
        visiting (set[str] | None): UUIDs on the current DFS path, used to
            break cycles with backtracking. Defaults to None (fresh
            traversal).

    Returns:
        Value | None: The lineage root value, or None when the lineage
            diverges across a phi or passes through a producer the trace
            does not understand (composite gates, controlled blocks,
            casts, ...).
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
        if isinstance(producer, QInitOperation):
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


def _op_referenced_uuids_with_ancestry(op: Operation) -> set[str]:
    """Collect the UUIDs one operation genuinely reads, with array ancestry.

    Rebind-record values (loop-carried records on loop operations and
    branch records on ``IfOperation``s) ride along ``all_input_values``
    for cloning/substitution but are not reads, so they are excluded.
    Each remaining input value contributes its own UUID plus its
    ``parent_array`` / ``slice_of`` ancestry so an element or view read
    of a register counts as touching the register itself.

    Args:
        op (Operation): Operation to inspect.

    Returns:
        set[str]: UUIDs of every genuinely-read input value and its
            array ancestry.
    """
    excluded: set[str] = set()
    if isinstance(op, (ForOperation, ForItemsOperation, WhileOperation)):
        for loop_record in op.loop_carried_rebinds:
            excluded.add(loop_record.before.uuid)
            excluded.add(loop_record.after.uuid)
    if isinstance(op, IfOperation):
        for branch_record in op.branch_rebinds:
            excluded.add(branch_record.before.uuid)

    referenced: set[str] = set()

    def add_with_ancestry(value: ValueBase) -> None:
        """Record a value's UUID together with its array ancestry.

        Args:
            value (ValueBase): Input value read by the operation.
        """
        if value.uuid in referenced:
            return
        referenced.add(value.uuid)
        for attr in ("parent_array", "slice_of"):
            ancestor = getattr(value, attr, None)
            if ancestor is not None:
                add_with_ancestry(ancestor)

    for value in op.all_input_values():
        if value.uuid in excluded:
            continue
        add_with_ancestry(value)
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


def _check_branch_quantum_discard(
    if_op: IfOperation,
    concrete_values: dict[str, Any],
    bindings: dict[str, Any],
    reads_outside: set[str],
) -> None:
    """Check one runtime ``IfOperation``'s rebind records for discards.

    For each recorded quantum rebind and each rebinding branch side,
    verifies that the pre-branch value survives the path on which that
    branch is taken: it must be consumed inside the taken branch, carried
    out through some phi merge on that side, or referenced by an
    operation outside the if entirely (an alias that still owns it, or a
    consume before the if). A pre-branch value with none of these owners
    is silently dropped exactly when the branch is taken — the discard
    this check rejects. Branch bodies are pruned of
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
        reads_outside (set[str]): Pre-branch record UUIDs that are read by
            at least one operation outside this if's subtree (rebind
            records excluded, array ancestry included) — the values still
            owned outside the if. Precomputed by the caller via block-wide
            vs in-subtree read-count comparison.

    Raises:
        QubitRebindError: If a rebinding branch drops the pre-branch
            quantum value with no consumer, no carrying phi, and no
            outside owner.
    """
    records = [
        record
        for record in if_op.branch_rebinds
        if isinstance(record.before, Value) and record.before.type.is_quantum()
    ]
    if not records:
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
    carried: dict[bool, set[str] | None] = {}
    for side, branch_ops in pruned_branches.items():
        referenced[side] = _branch_referenced_uuids(branch_ops)
        side_producers: dict[str, Operation] = {}
        build_producer_map(branch_ops, side_producers)
        carried_roots: set[str] | None = set()
        for phi in if_op.phi_ops:
            if len(phi.operands) < 3 or not phi.results:
                continue
            side_value = phi.true_value if side else phi.false_value
            if not isinstance(side_value, Value) or not side_value.type.is_quantum():
                continue
            root = _trace_pre_branch_root(side_value, side_producers)
            if root is None:
                # An untraceable lineage may still carry the pre-branch
                # value; skip this side entirely (errs toward allowing).
                carried_roots = None
                break
            assert carried_roots is not None
            carried_roots.add(root.uuid)
        carried[side] = carried_roots

    for record in records:
        for side, label in ((True, "true"), (False, "false")):
            rebound = record.rebound_in_true if side else record.rebound_in_false
            if not rebound:
                continue
            carried_side = carried[side]
            if carried_side is None:
                continue
            before_uuid = record.before.uuid
            if before_uuid in referenced[side]:
                continue
            if before_uuid in carried_side:
                continue
            if before_uuid in reads_outside:
                continue
            raise _branch_quantum_discard_error(record.var_name, label)


def _scan_branch_quantum_discards(
    ops: list[Operation],
    concrete_values: dict[str, Any],
    bindings: dict[str, Any],
    measurement_tainted: set[str],
    op_reads: dict[int, set[str]],
    all_read_counts: dict[str, int],
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

    For each runtime if that actually carries quantum rebind records, the
    "referenced outside this if" evidence for the (few) record pre-branch
    values is derived by comparing a UUID's block-wide read count against
    its in-subtree read count: a value read strictly more times overall
    than inside the subtree is read somewhere outside it. Counting (rather
    than the naive ``all_reads - subtree_reads`` set subtraction) is
    required for correctness — a value read both inside the subtree and
    outside it must still count as owned outside, which plain subtraction
    would drop. The per-if cost is therefore O(subtree size) instead of
    O(total ops).

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
        op_reads (dict[int, set[str]]): Per-operation read sets (rebind
            records excluded, array ancestry included), keyed by ``id()``
            of every operation reachable from the scanned block.
        all_read_counts (dict[str, int]): For each UUID, the number of
            operations in the whole block that read it (each op counted
            once per distinct UUID). Precomputed once from ``op_reads``.

    Raises:
        QubitRebindError: If a runtime if branch rebinds a quantum
            variable whose pre-branch value has no owner on that path.
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
                    branch,
                    concrete_values,
                    bindings,
                    measurement_tainted,
                    op_reads,
                    all_read_counts,
                )
                continue
            # Only a runtime if carrying quantum rebind records can discard.
            # Records are attached solely when a branch rebinds a variable
            # to a *different* quantum value whose pre-branch binding would
            # otherwise vanish from the IfOperation (fresh allocation, or
            # substitution of another register); ordinary gate self-updates
            # (``q = qmc.x(q)``) and measurement-conditioned gates on other
            # qubits leave no record, so the common case skips the scan.
            if (
                op.branch_rebinds
                and getattr(op.condition, "uuid", None) in measurement_tainted
            ):
                subtree_read_counts: dict[str, int] = {}
                for sub_op in flatten_ops([op]):
                    for read_uuid in op_reads.get(id(sub_op), ()):
                        subtree_read_counts[read_uuid] = (
                            subtree_read_counts.get(read_uuid, 0) + 1
                        )
                reads_outside = {
                    record.before.uuid
                    for record in op.branch_rebinds
                    if isinstance(record.before, Value)
                    and all_read_counts.get(record.before.uuid, 0)
                    > subtree_read_counts.get(record.before.uuid, 0)
                }
                _check_branch_quantum_discard(
                    op, concrete_values, bindings, reads_outside
                )
            _scan_branch_quantum_discards(
                op.true_operations,
                dict(concrete_values),
                bindings,
                measurement_tainted,
                op_reads,
                all_read_counts,
            )
            _scan_branch_quantum_discards(
                op.false_operations,
                dict(concrete_values),
                bindings,
                measurement_tainted,
                op_reads,
                all_read_counts,
            )
            continue
        if isinstance(op, HasNestedOps):
            for body in op.nested_op_lists():
                _scan_branch_quantum_discards(
                    body,
                    dict(concrete_values),
                    bindings,
                    measurement_tainted,
                    op_reads,
                    all_read_counts,
                )
        # Other operations carry no nested control flow to scan.


def reject_branch_internal_quantum_discard(
    operations: list[Operation],
    bindings: dict[str, Any] | None = None,
) -> None:
    """Reject runtime-branch quantum rebinds that discard quantum state.

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
      (``if cond: qmc.measure(q); q = qmc.qubit(...)``) — any reference to
      the original, including element or view reads of a register, counts;
    - ordinary quantum rebinds through gates (``q = qmc.h(q)``) — the
      pre-branch value is carried out through the phi merge;
    - rebinds whose pre-branch value is still owned outside the if (a
      value consumed before the if, or an alias referenced after it);
    - handle exchanges where every pre-branch value is carried by some
      phi of the same side (``q1, q2 = q2, q1``).

    The check is deliberately conservative toward allowing: phi lineage
    is traced only through plain gates and phi merges, so a branch side
    whose lineage passes through composite gates, controlled blocks, or
    casts is skipped entirely, and any reference outside the if counts
    as ownership even when it sits on a sibling branch of an enclosing
    runtime if.

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
        QubitRebindError: If a runtime if branch rebinds a quantum variable
            whose pre-branch value has no consumer in that branch, no phi
            carrying it out of that side, and no reference outside the if.
    """
    resolved_bindings = bindings or {}
    dependency_graph = build_dependency_graph(operations)
    measurement_tainted = find_measurement_derived_values(
        dependency_graph, find_measurement_results(operations)
    )
    op_reads = {
        id(op): _op_referenced_uuids_with_ancestry(op) for op in flatten_ops(operations)
    }
    all_read_counts: dict[str, int] = {}
    for read_uuids in op_reads.values():
        for read_uuid in read_uuids:
            all_read_counts[read_uuid] = all_read_counts.get(read_uuid, 0) + 1
    _scan_branch_quantum_discards(
        operations,
        {},
        resolved_bindings,
        measurement_tainted,
        op_reads,
        all_read_counts,
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

        # Reject branch-internal quantum rebinds that conditionally
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

    def _reject_branch_internal_quantum_discard(
        self,
        operations: list[Operation],
    ) -> None:
        """Reject runtime-branch quantum rebinds that discard quantum state.

        Thin wrapper around the module-level
        :func:`reject_branch_internal_quantum_discard`, run by this pass
        as a safety net (the same module-level function also runs earlier
        and directly from ``PartialEvaluationPass``, pre-fold, with
        bindings). Invoked here without bindings, so only already-constant
        conditions resolve as compile-time and every other ``IfOperation``
        is treated as runtime; of those, only the ones whose condition is
        measurement-derived are actually checked. In the normal pipeline
        ``CompileTimeIfLoweringPass`` has already folded away the
        compile-time ifs before ``analyze`` runs, so this bindings-free
        call is exact; when ``AnalyzePass`` runs standalone (no
        ``partial_eval``) an unfolded compile-time if with a
        non-measurement condition simply is not checked, which is safe.

        Args:
            operations (list[Operation]): Operations to scan recursively.

        Raises:
            QubitRebindError: If a runtime if branch rebinds a quantum
                variable whose pre-branch value has no owner on that
                path (no in-branch consumer, no carrying phi, and no
                reference outside the if).
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
