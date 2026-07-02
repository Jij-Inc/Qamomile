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
from qamomile.circuit.ir.operation.gate import MeasureOperation, MeasureVectorOperation
from qamomile.circuit.ir.operation.operation import OperationKind
from qamomile.circuit.ir.value import Value, ValueBase
from qamomile.circuit.transpiler.errors import DependencyError, ValidationError
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.passes.compile_time_if_lowering import (
    evaluate_classical_op_concrete,
    resolve_compile_time_condition,
)
from qamomile.circuit.transpiler.passes.control_flow_visitor import ControlFlowVisitor

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

    def prune_compile_time_ifs(
        ops: list[Operation],
        concrete_values: dict[str, Any],
    ) -> list[Operation]:
        """Replace compile-time-decidable ``IfOperation``s by their taken branch.

        Mirrors ``CompileTimeIfLoweringPass``: conditions are resolved
        with the shared ``resolve_compile_time_condition`` /
        ``evaluate_classical_op_concrete`` helpers so the taken / dead /
        runtime classification here cannot disagree with the branch the
        lowering pass will actually keep.  For a resolved condition the
        taken branch's operations are inlined (recursively pruned) and
        each ``PhiOp`` is reduced to its selected source operand, so
        phi-mediated dataflow out of the branch stays visible to the
        dependency scan without dead-branch edges.  Runtime
        ``IfOperation``s are kept intact: their branch stores are
        excluded from the store scan below, while their dataflow keeps
        feeding the dependency graph exactly as before.

        Args:
            ops (list[Operation]): Operations to prune, in program order.
            concrete_values (dict[str, Any]): UUID-keyed concrete
                classical-op results accumulated along the walk.
                Updated in place (nested non-if bodies get a copy,
                matching the lowering pass's scoping).

        Returns:
            list[Operation]: The pruned view of ``ops``.
        """
        pruned: list[Operation] = []
        for op in ops:
            evaluate_classical_op_concrete(op, concrete_values, resolved_bindings)
            if isinstance(op, IfOperation):
                taken = resolve_compile_time_condition(
                    op.condition, concrete_values, resolved_bindings
                )
                if taken is None:
                    pruned.append(op)
                    continue
                branch = op.true_operations if taken else op.false_operations
                pruned.extend(prune_compile_time_ifs(branch, concrete_values))
                for phi in op.phi_ops:
                    if isinstance(phi, PhiOp):
                        selected = phi.true_value if taken else phi.false_value
                        pruned.append(dataclasses.replace(phi, operands=[selected]))
                continue
            if isinstance(op, HasNestedOps):
                op = op.rebuild_nested(
                    [
                        prune_compile_time_ifs(body, dict(concrete_values))
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
                used to collect the loops and stores this check scans,
                since stores inside a (runtime) if branch are rejected
                by ``AnalyzePass._reject_stores_in_if_branches`` instead.

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

    pruned_operations = prune_compile_time_ifs(operations, {})
    for op in flatten_ops(pruned_operations, into_if_branches=False):
        if isinstance(op, (ForOperation, ForItemsOperation, WhileOperation)):
            check_loop_body(
                [body_op for body in op.nested_op_lists() for body_op in body]
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
