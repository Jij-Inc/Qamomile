"""Validation pass: reject unresolvable ``ForOperation`` loop bounds.

Qamomile circuits are fixed-structure at compile time: loop bounds
determine how many gates are emitted, so every ``ForOperation`` bound
must be resolvable before emit. This pass catches two failure families
early — right before segmentation — with an actionable diagnostic,
instead of letting them surface downstream as inconsistent errors:

1. **Unresolved parameter shape dims** (e.g. ``gamma_dim0``): a loop
   bound reads a parameter array's length, but no concrete binding was
   provided so ``ParameterShapeResolutionPass`` could not fold it.
2. **Runtime-parameter-dependent bounds** (e.g. ``qmc.range(n)`` with
   ``parameters=["n"]``): the bound depends — directly or through a
   chain of classical arithmetic — on a kernel argument left as a
   runtime parameter. Runtime parameters are bound per execution,
   after the circuit structure is frozen, so such a loop can never be
   unrolled. Without this check the failure surfaces inconsistently
   downstream: as an emit-time ``ValueError`` ("Cannot unroll loop")
   when the bound is the parameter itself, or as a misleading
   ``MultipleQuantumSegmentsError`` blaming measurement-dependent
   control flow when a bound *expression* strands its classical op
   between quantum ops at segmentation.

Bounds that depend only on enclosing loop variables (e.g. a nested
``qmc.range(i + 1)`` inside ``qmc.range(p)``) are resolvable during
emit-time unrolling and are left alone, as are measurement-derived
values (handled by the runtime control-flow machinery).

The library QAOA pattern (``p`` bound to an int, ``for layer in
qmc.range(p)``, ``gammas`` kept as runtime parameters for gate angles)
is unaffected — compile-time-bound counters are folded to constants by
``partial_eval`` before this pass runs.
"""

from __future__ import annotations

import re

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
)
from qamomile.circuit.ir.operation.operation import OperationKind
from qamomile.circuit.ir.value import Value, ValueBase
from qamomile.circuit.transpiler.errors import QamomileCompileError
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.passes.control_flow_visitor import ControlFlowVisitor

# Shape dims created by ``func_to_block.create_dummy_input`` follow the
# ``{name}_dim{i}`` naming convention. Matching this name lets us produce
# an actionable diagnostic identifying the parameter array whose shape
# failed to resolve.
_SHAPE_DIM_NAME_PATTERN = re.compile(r"^(?P<array>.+)_dim(?P<index>\d+)$")


def _looks_like_parameter_shape_dim(v: Value) -> tuple[str, int] | None:
    """Classify a Value as a parameter array shape dim by its name.

    Args:
        v (Value): The candidate loop-bound operand.

    Returns:
        tuple[str, int] | None: ``(array_name, dim_index)`` when ``v`` is a
            non-constant Value named ``{array}_dim{i}``, ``None`` otherwise.
    """
    if v.is_constant():
        return None
    if not v.name:
        return None
    match = _SHAPE_DIM_NAME_PATTERN.match(v.name)
    if match is None:
        return None
    return match.group("array"), int(match.group("index"))


def _format_actionable_error(
    array_name: str,
    dim_index: int,
    location_hint: str,
) -> str:
    """Build the diagnostic for an unresolved parameter shape dim.

    Args:
        array_name (str): Name of the parameter array whose shape dim is
            unresolved.
        dim_index (int): Index of the unresolved shape dimension.
        location_hint (str): Human-readable description of where the dim
            is consumed (e.g. ``"a for-loop 'stop' bound"``).

    Returns:
        str: The formatted multi-line error message with concrete fixes.
    """
    return (
        f"Parameter array '{array_name}' has unresolved shape dimension "
        f"{dim_index}: {location_hint} depends on its length at compile "
        f"time, but no concrete value is available.\n\n"
        f"Qamomile circuits are fixed-structure at compile time — loop "
        f"bounds and array lengths must be concrete before emit. "
        f"Pick one of the following fixes:\n"
        f"  1. Bind a concrete array at transpile time so its shape is "
        f"known:\n"
        f"       transpiler.transpile(..., bindings={{'{array_name}': "
        f"[...]}})\n"
        f"     This bakes the array values into the emitted circuit; do "
        f"not also list '{array_name}' in parameters=[...].\n"
        f"  2. If the array values must stay per-execution parameters, "
        f"use a separate compile-time loop counter instead of querying "
        f"the shape:\n"
        f"       def kernel(p: qm.UInt, {array_name}: qm.Vector[qm.Float], ...):\n"
        f"           for layer in qm.range(p):\n"
        f"               ... {array_name}[layer] ...\n"
        f"       transpiler.transpile(..., bindings={{'p': 2}}, "
        f"parameters=['{array_name}'])"
    )


def _build_classical_dependency_graph(
    operations: list[Operation],
) -> dict[str, set[str]]:
    """Build a result→operand dependency graph over classical dataflow only.

    Mirrors ``analyze.build_dependency_graph`` but records result→operand
    edges only for ``OperationKind.CLASSICAL`` operations. Quantum and
    hybrid result edges (e.g. ``rx``'s output qubit depending on its angle
    operand) are deliberately excluded: a loop bound's compile-time
    resolvability is a property of classical dataflow, and following a
    quantum edge could blame an unrelated gate parameter for a
    measurement-derived bound. Array-element metadata edges (the element's
    indices, parent array, and the parent's ``slice_of`` chain including
    slice bounds) are recorded for operands of every operation kind — they
    are type-preserving and are needed when a bound indexes a runtime
    parameter array directly (e.g. ``qmc.range(idxs[0])`` as a
    ``ForOperation`` operand) or uses a runtime parameter as the index into
    a compile-time-bound array (e.g. ``qmc.range(idxs[start])``).

    Args:
        operations (list[Operation]): Top-level operations of the block;
            nested control-flow bodies are walked recursively.

    Returns:
        dict[str, set[str]]: Mapping from a value UUID to the UUIDs it
            depends on through classical operation results or metadata
            references.
    """

    class ClassicalDependencyGraphBuilder(ControlFlowVisitor):
        """Visitor accumulating the classical-only dependency edges."""

        def __init__(self) -> None:
            """Initialize the empty edge accumulator."""
            self.graph: dict[str, set[str]] = {}

        def visit_operation(self, op: Operation) -> None:
            """Record the classical dependency edges contributed by ``op``.

            Args:
                op (Operation): The operation being visited.

            Returns:
                None: Mutates ``self.graph`` in place.
            """
            if op.operation_kind == OperationKind.CLASSICAL:
                operand_uuids = {
                    v.uuid for v in op.operands if isinstance(v, ValueBase)
                }
                for result in op.results:
                    self.graph.setdefault(result.uuid, set()).update(operand_uuids)
            for v in op.operands:
                if not isinstance(v, ValueBase):
                    continue
                self._record_value_reference_edges(v)

        def _record_value_reference_edges(
            self,
            value: ValueBase,
            seen: set[str] | None = None,
        ) -> None:
            """Record dependency edges carried by value metadata.

            Args:
                value (ValueBase): Operand value whose metadata references are
                    part of the classical dataflow for structural bounds.
                seen (set[str] | None): UUIDs already expanded during the
                    current metadata walk.

            Returns:
                None: Mutates ``self.graph`` in place.
            """
            if seen is None:
                seen = set()
            if value.uuid in seen:
                return
            seen.add(value.uuid)

            for idx in getattr(value, "element_indices", ()):
                if isinstance(idx, ValueBase):
                    self.graph.setdefault(value.uuid, set()).add(idx.uuid)
                    self._record_value_reference_edges(idx, seen)

            parent = getattr(value, "parent_array", None)
            if parent is None:
                return
            self.graph.setdefault(value.uuid, set()).add(parent.uuid)

            cur = parent
            while cur is not None:
                for bound in (
                    getattr(cur, "slice_start", None),
                    getattr(cur, "slice_step", None),
                ):
                    if isinstance(bound, ValueBase):
                        self.graph.setdefault(cur.uuid, set()).add(bound.uuid)
                        self._record_value_reference_edges(bound, seen)

                next_parent = getattr(cur, "slice_of", None)
                if next_parent is None:
                    break
                self.graph.setdefault(cur.uuid, set()).add(next_parent.uuid)
                cur = next_parent

    builder = ClassicalDependencyGraphBuilder()
    builder.visit_operations(operations)
    return builder.graph


def _format_runtime_parameter_bound_error(
    param_name: str,
    bound_label: str,
    loop_var: str,
) -> str:
    """Build the diagnostic for a runtime-parameter-dependent loop bound.

    Args:
        param_name (str): Name of the runtime parameter the bound's
            dataflow traces back to.
        bound_label (str): Which bound is affected (``"start"`` /
            ``"stop"`` / ``"step"``).
        loop_var (str): Display name of the loop variable.

    Returns:
        str: The formatted multi-line error message with the concrete fix.
    """
    return (
        f"Cannot unroll loop: bounds could not be resolved at compile time "
        f"(the '{bound_label}' bound of the loop over '{loop_var}' depends "
        f"on runtime parameter '{param_name}').\n\n"
        f"Qamomile circuits are fixed-structure at compile time — loop "
        f"bounds determine the emitted gate count, so they cannot depend on "
        f"a runtime parameter, which is only bound at execution time. "
        f"'{param_name}' became a runtime parameter either because it was "
        f"listed in parameters=[...] or because it had neither a bindings "
        f"entry nor a Python default (auto-detect).\n\n"
        f"Fix: bind it at compile time (and remove it from "
        f"parameters=[...] if present):\n"
        f"    transpiler.transpile(..., bindings={{'{param_name}': <int>}})\n"
        f"Per-execution gate parameters such as rotation angles may stay in "
        f"parameters=[...]; structural values (loop bounds, qubit counts, "
        f"array shapes) must be compile-time bound."
    )


class SymbolicShapeValidationPass(Pass[Block, Block]):
    """Reject unresolvable compile-time-structure values in loop bounds.

    Runs two checks on every ``ForOperation`` bound (start / stop / step),
    walking nested control flow recursively:

    1. Unresolved parameter array shape dims, recognized by the
       ``{array}_dim{i}`` naming convention.
    2. Bounds whose dataflow traces back to a runtime parameter
       (an entry of ``Block.parameters``), directly or through classical
       arithmetic. Dependencies on enclosing loop variables are exempt —
       those resolve during emit-time unrolling.

    Input:  ``BlockKind.ANALYZED`` (runs after ``AnalyzePass``).
    Output: same block unchanged, or raises ``QamomileCompileError``.
    """

    @property
    def name(self) -> str:
        """Return the pass name used in pipeline diagnostics.

        Returns:
            str: The constant pass name ``"symbolic_shape_validation"``.
        """
        return "symbolic_shape_validation"

    def run(self, input: Block) -> Block:
        """Validate every ``ForOperation`` bound in the block.

        Args:
            input (Block): The block to validate. Must be
                ``BlockKind.ANALYZED``; other kinds pass through unchanged
                (the pass is defensive and avoids false positives on
                partially-built IR).

        Returns:
            Block: ``input``, unchanged, when validation succeeds.

        Raises:
            QamomileCompileError: If a loop bound is an unresolved
                parameter shape dim or depends on a runtime parameter.
        """
        if input.kind != BlockKind.ANALYZED:
            # Pass is defensive — only runs on analyzed blocks. Skip
            # otherwise to avoid false positives on partially-built IR.
            return input

        self._runtime_param_names = {
            value.uuid: name
            for name, value in input.parameters.items()
            if isinstance(value, ValueBase)
        }
        self._dependency_graph = _build_classical_dependency_graph(input.operations)
        self._walk(input.operations, frozenset())
        return input

    def _walk(
        self,
        operations: list[Operation],
        enclosing_loop_vars: frozenset[str],
    ) -> None:
        """Check operations recursively, tracking enclosing loop variables.

        Args:
            operations (list[Operation]): Operations to check (a block's
                top-level list or a control-flow op's nested body).
            enclosing_loop_vars (frozenset[str]): UUIDs of loop variables
                declared by the control-flow ops enclosing ``operations``.

        Returns:
            None: Raises on violation, otherwise returns nothing.

        Raises:
            QamomileCompileError: Propagated from :meth:`_check_op`.
        """
        for op in operations:
            self._check_op(op, enclosing_loop_vars)
            if isinstance(op, HasNestedOps):
                nested_scope = enclosing_loop_vars | self._declared_loop_var_uuids(op)
                for nested in op.nested_op_lists():
                    self._walk(nested, nested_scope)

    def _declared_loop_var_uuids(self, op: Operation) -> frozenset[str]:
        """Collect the loop-variable UUIDs declared by a control-flow op.

        Args:
            op (Operation): The control-flow operation whose body is about
                to be walked.

        Returns:
            frozenset[str]: UUIDs of the iteration variables ``op`` binds
                for its body — ``loop_var_value`` for ``ForOperation``,
                key/value variables for ``ForItemsOperation``, empty for
                other control flow.
        """
        uuids: set[str] = set()
        if isinstance(op, ForOperation) and op.loop_var_value is not None:
            uuids.add(op.loop_var_value.uuid)
        if isinstance(op, ForItemsOperation):
            for value in op.key_var_values or ():
                uuids.add(value.uuid)
            if op.value_var_value is not None:
                uuids.add(op.value_var_value.uuid)
        return frozenset(uuids)

    def _check_op(
        self,
        op: Operation,
        enclosing_loop_vars: frozenset[str],
    ) -> None:
        """Dispatch the bound checks for a single operation.

        Args:
            op (Operation): The operation to check.
            enclosing_loop_vars (frozenset[str]): UUIDs of loop variables
                in scope at ``op``.

        Returns:
            None: Raises on violation, otherwise returns nothing.

        Raises:
            QamomileCompileError: Propagated from
                :meth:`_check_for_operation`.
        """
        if isinstance(op, ForOperation):
            self._check_for_operation(op, enclosing_loop_vars)

    def _check_for_operation(
        self,
        op: ForOperation,
        enclosing_loop_vars: frozenset[str],
    ) -> None:
        """Validate the start / stop / step bounds of one ``ForOperation``.

        Args:
            op (ForOperation): The loop whose bounds are checked.
            enclosing_loop_vars (frozenset[str]): UUIDs of loop variables
                declared by loops enclosing ``op`` (not ``op``'s own).

        Returns:
            None: Raises on violation, otherwise returns nothing.

        Raises:
            QamomileCompileError: If a bound is an unresolved parameter
                shape dim, or its dataflow traces back to a runtime
                parameter.
        """
        labels = ("start", "stop", "step")
        for label, operand in zip(labels, op.operands):
            if not isinstance(operand, Value):
                continue
            shape_info = _looks_like_parameter_shape_dim(operand)
            if shape_info is not None:
                array_name, dim_index = shape_info
                location = f"a for-loop '{label}' bound (loop variable '{op.loop_var}')"
                raise QamomileCompileError(
                    _format_actionable_error(array_name, dim_index, location)
                )
            param_name = self._runtime_parameter_source(operand, enclosing_loop_vars)
            if param_name is not None:
                raise QamomileCompileError(
                    _format_runtime_parameter_bound_error(
                        param_name, label, op.loop_var
                    )
                )

    def _runtime_parameter_source(
        self,
        operand: Value,
        enclosing_loop_vars: frozenset[str],
    ) -> str | None:
        """Trace a bound operand's dataflow back to a runtime parameter.

        Walks the classical dependency graph (see
        :func:`_build_classical_dependency_graph`) from ``operand`` toward
        its sources. The walk stops at enclosing loop variables (their
        concrete value is supplied during emit-time unrolling, so anything
        derived only from them is resolvable) and reports the first
        runtime parameter reached. Symbolic values with some other origin
        (e.g. measurement-derived — unreachable here because quantum /
        hybrid result edges are excluded from the graph) yield ``None``
        and are left to the passes that own them.

        Args:
            operand (Value): A start / stop / step operand of a
                ``ForOperation``.
            enclosing_loop_vars (frozenset[str]): UUIDs of loop variables
                in scope at the loop being checked.

        Returns:
            str | None: The runtime parameter's name when the bound
                depends on one, ``None`` otherwise (constant bounds,
                loop-variable-derived bounds, or non-parameter symbolics).
        """
        if operand.is_constant():
            return None
        seen: set[str] = set()
        stack = [operand.uuid]
        while stack:
            uuid = stack.pop()
            if uuid in seen:
                continue
            seen.add(uuid)
            if uuid in enclosing_loop_vars:
                continue
            param_name = self._runtime_param_names.get(uuid)
            if param_name is not None:
                return param_name
            stack.extend(self._dependency_graph.get(uuid, ()))
        return None
