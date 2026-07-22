"""Reject unresolved values that determine compile-time circuit structure.

Qamomile circuits are fixed-structure at compile time: loop bounds determine
how many gates are emitted, and SELECT index widths determine reusable-call
arity, while controlled-unitary widths, powers, and selected control indices
determine coherent-control structure. These values must be resolvable before
emit. This pass catches two failure families early — right before segmentation
— with an actionable diagnostic, instead of letting them surface downstream as
inconsistent errors:

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

The same early diagnostic applies to symbolic controlled-unitary structure
(``num_controls``, ``power``, and ``control_indices``) and to loops stored in
operation-owned SELECT/control blocks.

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
from collections.abc import Sequence

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import (
    ControlledUOperation,
    Operation,
    SymbolicControlledU,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForItemsOperation,
    ForOperation,
    HasNestedOps,
)
from qamomile.circuit.ir.operation.operation import OperationKind
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.types import (
    BitType,
    FloatType,
    ObservableType,
    QubitType,
    UIntType,
    ValueType,
)
from qamomile.circuit.ir.value import ArrayValue, Value, ValueBase
from qamomile.circuit.transpiler.block_parameter_binding import (
    pair_block_operands,
)
from qamomile.circuit.transpiler.errors import QamomileCompileError
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.passes.control_flow_visitor import ControlFlowVisitor

# Shape dims created by ``func_to_block.create_dummy_input`` follow the
# ``{name}_dim{i}`` naming convention. Matching this name lets us produce
# an actionable diagnostic identifying the parameter array whose shape
# failed to resolve.
_SHAPE_DIM_NAME_PATTERN = re.compile(r"^(?P<array>.+)_dim(?P<index>\d+)$")

_FRONTEND_SCALAR_TYPE_NAMES: dict[type[ValueType], str] = {
    BitType: "Bit",
    FloatType: "Float",
    ObservableType: "Observable",
    QubitType: "Qubit",
    UIntType: "UInt",
}


def _frontend_scalar_type_name(value_type: ValueType) -> str | None:
    """Return the frontend spelling of a recognized scalar IR type.

    Args:
        value_type (ValueType): Array element type represented in IR.

    Returns:
        str | None: Frontend scalar name for a supported public handle type,
            otherwise ``None`` so diagnostics remain type-neutral.
    """
    return _FRONTEND_SCALAR_TYPE_NAMES.get(type(value_type))


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
    element_type_name: str | None = None,
) -> str:
    """Build the diagnostic for an unresolved parameter shape dim.

    Args:
        array_name (str): Name of the parameter array whose shape dim is
            unresolved.
        dim_index (int): Index of the unresolved shape dimension.
        location_hint (str): Human-readable description of where the dim
            is consumed (e.g. ``"a for-loop 'stop' bound"``).
        element_type_name (str | None): Frontend scalar type stored by the
            array, such as ``"Float"`` or ``"UInt"``. Unknown types use an
            ellipsis instead of suggesting an incorrect annotation.

    Returns:
        str: The formatted multi-line error message with concrete fixes.
    """
    count_stem = array_name[:-1] if array_name.endswith("s") else array_name
    count_name = f"{count_stem}_count"
    element_type = f"qm.{element_type_name}" if element_type_name else "..."
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
        f"       def kernel({count_name}: qm.UInt, "
        f"{array_name}: qm.Vector[{element_type}], ...):\n"
        f"           for layer in qm.range({count_name}):\n"
        f"               ... {array_name}[layer] ...\n"
        f"       transpiler.transpile(..., bindings={{'{count_name}': 2}}, "
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
    slice bounds) are recorded for every input value, including subclass-only
    structural fields such as a SELECT width or controlled-U power. These
    edges are type-preserving and are needed when a bound indexes a runtime
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
            input_values = op.all_input_values()
            if op.operation_kind == OperationKind.CLASSICAL:
                operand_uuids = {value.uuid for value in input_values}
                for result in op.results:
                    self.graph.setdefault(result.uuid, set()).update(operand_uuids)
            for value in input_values:
                self._record_value_reference_edges(value)

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

            # A symbolic array dimension is derived from the array parameter
            # whose runtime payload determines that dimension. Record the
            # reverse dataflow edge so a structural expression such as
            # ``(angles.shape[0] - 1) // 2`` can still be traced back to the
            # public ``angles`` parameter after arithmetic has replaced the
            # direct ``angles_dim0`` loop bound with anonymous results.
            for dimension in getattr(value, "shape", ()):
                if isinstance(dimension, ValueBase):
                    self.graph.setdefault(dimension.uuid, set()).add(value.uuid)

            parent = getattr(value, "parent_array", None)
            if parent is None:
                return
            self.graph.setdefault(value.uuid, set()).add(parent.uuid)

            cur = parent
            while cur is not None:
                for dimension in getattr(cur, "shape", ()):
                    if isinstance(dimension, ValueBase):
                        self.graph.setdefault(dimension.uuid, set()).add(cur.uuid)
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


def _format_runtime_parameter_select_width_error(param_name: str) -> str:
    """Build the diagnostic for a runtime-parameter SELECT width.

    Args:
        param_name (str): Runtime parameter that determines the index width.

    Returns:
        str: Actionable error explaining the compile-time binding requirement.
    """
    return (
        f"Cannot resolve SELECT index width at compile time because "
        f"num_index_qubits depends on runtime parameter '{param_name}'.\n\n"
        f"Qamomile circuits have fixed wire arity before execution, so a "
        f"SELECT index width cannot be supplied as a runtime parameter. "
        f"Bind it during transpilation (and remove it from parameters=[...] "
        f"if present):\n"
        f"    transpiler.transpile(..., bindings={{'{param_name}': <int>}})"
    )


def _format_runtime_parameter_control_structure_error(
    param_name: str,
    field_label: str,
) -> str:
    """Build the diagnostic for a runtime-dependent controlled-U field.

    Args:
        param_name (str): Runtime parameter that determines controlled-U
            structure.
        field_label (str): Human-readable controlled-U field name, such as
            ``"num_controls"`` or ``"power"``.

    Returns:
        str: Actionable error explaining the compile-time binding requirement.
    """
    return (
        f"Cannot resolve controlled-unitary {field_label} at compile time "
        f"because it depends on runtime parameter '{param_name}'.\n\n"
        f"Qamomile circuits have fixed wire arity and gate structure before "
        f"execution, so controlled-unitary structural values cannot be "
        f"runtime parameters. Bind '{param_name}' during transpilation (and "
        f"remove it from parameters=[...] if present):\n"
        f"    transpiler.transpile(..., bindings={{'{param_name}': <int>}})"
    )


class SymbolicShapeValidationPass(Pass[Block, Block]):
    """Reject unresolvable values that determine emitted circuit structure.

    Runs two checks on every ``ForOperation`` bound, symbolic SELECT index
    width, and symbolic controlled-unitary structural field, walking nested
    control flow and operation-owned SELECT/control blocks recursively:

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
        """Validate compile-time structure throughout the analyzed block.

        Checks loop bounds, SELECT widths, and controlled-unitary structural
        fields at top level, in ordinary control-flow bodies, and in
        operation-owned SELECT/control blocks.

        Args:
            input (Block): The block to validate. Must be
                ``BlockKind.ANALYZED``; other kinds pass through unchanged
                (the pass is defensive and avoids false positives on
                partially-built IR).

        Returns:
            Block: ``input``, unchanged, when validation succeeds.

        Raises:
            QamomileCompileError: If any checked structural value is an
                unresolved parameter shape dim or depends on a runtime
                parameter.
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
        self._input_shape_dimensions = {
            dimension.uuid: (
                value.name,
                index,
                _frontend_scalar_type_name(value.type),
            )
            for value in input.input_values
            if isinstance(value, ArrayValue)
            for index, dimension in enumerate(value.shape)
            if value.name and not dimension.is_constant()
        }
        dependency_graph = _build_classical_dependency_graph(input.operations)
        self._walk(
            input.operations,
            frozenset(),
            dependency_graph,
            frozenset(),
            frozenset(),
        )
        return input

    def _walk(
        self,
        operations: list[Operation],
        enclosing_loop_vars: frozenset[str],
        dependency_graph: dict[str, set[str]],
        aliased_shape_dims: frozenset[str],
        owned_blocks_on_path: frozenset[int],
    ) -> None:
        """Check operations recursively, tracking enclosing loop variables.

        Args:
            operations (list[Operation]): Operations to check (a block's
                top-level list or a control-flow op's nested body).
            enclosing_loop_vars (frozenset[str]): UUIDs of loop variables
                declared by the control-flow ops enclosing ``operations``.
            dependency_graph (dict[str, set[str]]): Classical dependency edges
                visible in the current value scope.
            aliased_shape_dims (frozenset[str]): UUIDs of owned-block formal
                shape dimensions that are paired with a call-site actual.
                Their display names alone do not imply an unresolved public
                parameter shape.
            owned_blocks_on_path (frozenset[int]): Object identities of
                operation-owned Blocks currently being traversed. This stops
                malformed or recursive block graphs from recursing forever.

        Returns:
            None: Raises on violation, otherwise returns nothing.

        Raises:
            QamomileCompileError: Propagated from :meth:`_check_op`.
        """
        for op in operations:
            self._check_op(
                op,
                enclosing_loop_vars,
                dependency_graph,
                aliased_shape_dims,
            )
            if isinstance(op, HasNestedOps):
                nested_scope = enclosing_loop_vars | self._declared_loop_var_uuids(op)
                for nested in op.nested_op_lists():
                    self._walk(
                        nested,
                        nested_scope,
                        dependency_graph,
                        aliased_shape_dims,
                        owned_blocks_on_path,
                    )
            for block, actual_operands in self._owned_blocks(op):
                block_id = id(block)
                if block_id in owned_blocks_on_path:
                    continue
                owned_graph, owned_shape_dims = self._owned_block_dependency_graph(
                    block,
                    actual_operands,
                    dependency_graph,
                )
                self._walk(
                    block.operations,
                    enclosing_loop_vars,
                    owned_graph,
                    aliased_shape_dims | owned_shape_dims,
                    owned_blocks_on_path | {block_id},
                )

    def _owned_blocks(
        self,
        op: Operation,
    ) -> list[tuple[Block, Sequence[ValueBase]]]:
        """Return operation-owned blocks and their call-site actuals.

        Args:
            op (Operation): Operation whose independently-scoped blocks should
                be traversed.

        Returns:
            list[tuple[Block, Sequence[ValueBase]]]: Each owned block paired with
                its target and classical/object operands. Controls external to
                the owned callable are excluded. Ordinary control-flow bodies
                are excluded because :class:`HasNestedOps` handles them in the
                parent value scope.
        """
        if isinstance(op, SelectOperation):
            actuals = [*op.target_operands, *op.param_operands]
            return [(block, actuals) for block in op.case_blocks]
        if isinstance(op, ControlledUOperation) and op.block is not None:
            targets = [value for value in op.target_operands if value.type.is_quantum()]
            return [(op.block, [*targets, *op.param_operands])]
        return []

    def _owned_block_dependency_graph(
        self,
        block: Block,
        actual_operands: Sequence[ValueBase],
        parent_graph: dict[str, set[str]],
    ) -> tuple[dict[str, set[str]], frozenset[str]]:
        """Build dependency edges for one independently-scoped owned block.

        Formal values inside SELECT/control bodies have fresh UUIDs, so their
        runtime provenance cannot be recovered by display name. This method
        joins each formal to its call-site actual using the same positional
        pairing contract as partial evaluation and emission.

        Args:
            block (Block): Operation-owned block to inspect.
            actual_operands (Sequence[ValueBase]): Target and classical/object
                call-site operands supplied by the owning operation, grouped
                by quantum then non-quantum category.
            parent_graph (dict[str, set[str]]): Dependency edges visible at the
                owning operation.

        Returns:
            tuple[dict[str, set[str]], frozenset[str]]: Combined dependency
                graph and the owned formal shape-dimension UUIDs paired to
                call-site actual dimensions.
        """
        graph = {uuid: set(dependencies) for uuid, dependencies in parent_graph.items()}
        for uuid, dependencies in _build_classical_dependency_graph(
            block.operations
        ).items():
            graph.setdefault(uuid, set()).update(dependencies)
        aliased_shape_dims: set[str] = set()
        for formal, actual in pair_block_operands(block, actual_operands):
            graph.setdefault(formal.uuid, set()).add(actual.uuid)
            if isinstance(formal, ArrayValue) and isinstance(actual, ArrayValue):
                for formal_dim, actual_dim in zip(formal.shape, actual.shape):
                    graph.setdefault(formal_dim.uuid, set()).add(actual_dim.uuid)
                    graph.setdefault(actual_dim.uuid, set()).add(actual.uuid)
                    aliased_shape_dims.add(formal_dim.uuid)
        return graph, frozenset(aliased_shape_dims)

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
        dependency_graph: dict[str, set[str]],
        aliased_shape_dims: frozenset[str],
    ) -> None:
        """Dispatch the bound checks for a single operation.

        Args:
            op (Operation): The operation to check.
            enclosing_loop_vars (frozenset[str]): UUIDs of loop variables
                in scope at ``op``.
            dependency_graph (dict[str, set[str]]): Classical dependency edges
                visible at ``op``.
            aliased_shape_dims (frozenset[str]): Owned formal shape dimensions
                paired to call-site actuals.

        Returns:
            None: Raises on violation, otherwise returns nothing.

        Raises:
            QamomileCompileError: Propagated from
                :meth:`_check_for_operation`.
        """
        if isinstance(op, ForOperation):
            self._check_for_operation(
                op,
                enclosing_loop_vars,
                dependency_graph,
                aliased_shape_dims,
            )
        elif isinstance(op, SelectOperation):
            self._check_select_operation(
                op,
                enclosing_loop_vars,
                dependency_graph,
                aliased_shape_dims,
            )
        if isinstance(op, ControlledUOperation):
            self._check_controlled_operation(
                op,
                enclosing_loop_vars,
                dependency_graph,
                aliased_shape_dims,
            )

    def _check_select_operation(
        self,
        op: SelectOperation,
        enclosing_loop_vars: frozenset[str],
        dependency_graph: dict[str, set[str]],
        aliased_shape_dims: frozenset[str],
    ) -> None:
        """Validate one symbolic SELECT index width.

        Args:
            op (SelectOperation): SELECT operation whose width is checked.
            enclosing_loop_vars (frozenset[str]): UUIDs of loop variables in
                scope. Widths derived solely from these remain resolvable
                during emit-time unrolling.
            dependency_graph (dict[str, set[str]]): Classical dependency edges
                visible at the SELECT operation.
            aliased_shape_dims (frozenset[str]): Owned formal shape dimensions
                paired to call-site actuals.

        Returns:
            None: Raises on violation, otherwise returns nothing.

        Raises:
            QamomileCompileError: If the width is an unresolved parameter
                shape or depends on a runtime parameter.
        """
        width = op.num_index_qubits
        if not isinstance(width, Value) or width.is_constant():
            return
        shape_info = _looks_like_parameter_shape_dim(width)
        if shape_info is not None and width.uuid not in aliased_shape_dims:
            array_name, dim_index = shape_info
            input_shape_info = self._input_shape_dimensions.get(width.uuid)
            raise QamomileCompileError(
                _format_actionable_error(
                    array_name,
                    dim_index,
                    "a SELECT num_index_qubits value",
                    None if input_shape_info is None else input_shape_info[2],
                )
            )
        shape_info = self._shape_dimension_source(
            width,
            enclosing_loop_vars,
            dependency_graph,
            aliased_shape_dims,
        )
        if shape_info is not None:
            array_name, dim_index, element_type_name = shape_info
            raise QamomileCompileError(
                _format_actionable_error(
                    array_name,
                    dim_index,
                    "a SELECT num_index_qubits value",
                    element_type_name,
                )
            )
        param_name = self._runtime_parameter_source(
            width,
            enclosing_loop_vars,
            dependency_graph,
        )
        if param_name is not None:
            raise QamomileCompileError(
                _format_runtime_parameter_select_width_error(param_name)
            )

    def _check_controlled_operation(
        self,
        op: ControlledUOperation,
        enclosing_loop_vars: frozenset[str],
        dependency_graph: dict[str, set[str]],
        aliased_shape_dims: frozenset[str],
    ) -> None:
        """Validate symbolic fields that determine controlled-U structure.

        Args:
            op (ControlledUOperation): Controlled operation whose structural
                values should be compile-time resolvable.
            enclosing_loop_vars (frozenset[str]): UUIDs of loop variables in
                scope. Fields derived solely from these remain resolvable
                during emit-time unrolling.
            dependency_graph (dict[str, set[str]]): Classical dependency edges
                visible at the operation.
            aliased_shape_dims (frozenset[str]): Owned formal shape dimensions
                paired to call-site actuals.

        Returns:
            None: Raises on invalid structure; otherwise returns nothing.

        Raises:
            QamomileCompileError: If a structural field is an unresolved
                parameter shape or depends on a runtime parameter.
        """
        fields: list[tuple[str, Value]] = []
        if isinstance(op.power, Value):
            fields.append(("power", op.power))
        if isinstance(op, SymbolicControlledU):
            fields.append(("num_controls", op.num_controls))
            fields.extend(
                (f"control_indices[{index}]", value)
                for index, value in enumerate(op.control_indices or ())
            )
        for field_label, value in fields:
            self._check_controlled_structural_value(
                value,
                field_label,
                enclosing_loop_vars,
                dependency_graph,
                aliased_shape_dims,
            )

    def _check_controlled_structural_value(
        self,
        value: Value,
        field_label: str,
        enclosing_loop_vars: frozenset[str],
        dependency_graph: dict[str, set[str]],
        aliased_shape_dims: frozenset[str],
    ) -> None:
        """Validate one symbolic controlled-U structural value.

        Args:
            value (Value): Candidate structural value.
            field_label (str): Controlled-U field name used in diagnostics.
            enclosing_loop_vars (frozenset[str]): UUIDs of enclosing loop
                variables that emit-time unrolling can resolve.
            dependency_graph (dict[str, set[str]]): Classical dependency edges
                visible at the operation.
            aliased_shape_dims (frozenset[str]): Owned formal shape dimensions
                paired to call-site actuals.

        Returns:
            None: Raises on invalid structure; otherwise returns nothing.

        Raises:
            QamomileCompileError: If ``value`` is an unresolved parameter
                shape or depends on a runtime parameter.
        """
        if value.is_constant():
            return
        shape_info = _looks_like_parameter_shape_dim(value)
        if shape_info is not None and value.uuid not in aliased_shape_dims:
            array_name, dim_index = shape_info
            input_shape_info = self._input_shape_dimensions.get(value.uuid)
            raise QamomileCompileError(
                _format_actionable_error(
                    array_name,
                    dim_index,
                    f"a controlled-unitary {field_label} value",
                    None if input_shape_info is None else input_shape_info[2],
                )
            )
        shape_info = self._shape_dimension_source(
            value,
            enclosing_loop_vars,
            dependency_graph,
            aliased_shape_dims,
        )
        if shape_info is not None:
            array_name, dim_index, element_type_name = shape_info
            raise QamomileCompileError(
                _format_actionable_error(
                    array_name,
                    dim_index,
                    f"a controlled-unitary {field_label} value",
                    element_type_name,
                )
            )
        param_name = self._runtime_parameter_source(
            value,
            enclosing_loop_vars,
            dependency_graph,
        )
        if param_name is not None:
            raise QamomileCompileError(
                _format_runtime_parameter_control_structure_error(
                    param_name,
                    field_label,
                )
            )

    def _check_for_operation(
        self,
        op: ForOperation,
        enclosing_loop_vars: frozenset[str],
        dependency_graph: dict[str, set[str]],
        aliased_shape_dims: frozenset[str],
    ) -> None:
        """Validate the start / stop / step bounds of one ``ForOperation``.

        Args:
            op (ForOperation): The loop whose bounds are checked.
            enclosing_loop_vars (frozenset[str]): UUIDs of loop variables
                declared by loops enclosing ``op`` (not ``op``'s own).
            dependency_graph (dict[str, set[str]]): Classical dependency edges
                visible at the loop.
            aliased_shape_dims (frozenset[str]): Owned formal shape dimensions
                paired to call-site actuals.

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
            if shape_info is not None and operand.uuid not in aliased_shape_dims:
                array_name, dim_index = shape_info
                input_shape_info = self._input_shape_dimensions.get(operand.uuid)
                location = f"a for-loop '{label}' bound (loop variable '{op.loop_var}')"
                raise QamomileCompileError(
                    _format_actionable_error(
                        array_name,
                        dim_index,
                        location,
                        None if input_shape_info is None else input_shape_info[2],
                    )
                )
            shape_info = self._shape_dimension_source(
                operand,
                enclosing_loop_vars,
                dependency_graph,
                aliased_shape_dims,
            )
            if shape_info is not None:
                array_name, dim_index, element_type_name = shape_info
                location = f"a for-loop '{label}' bound (loop variable '{op.loop_var}')"
                raise QamomileCompileError(
                    _format_actionable_error(
                        array_name,
                        dim_index,
                        location,
                        element_type_name,
                    )
                )
            param_name = self._runtime_parameter_source(
                operand,
                enclosing_loop_vars,
                dependency_graph,
            )
            if param_name is not None:
                raise QamomileCompileError(
                    _format_runtime_parameter_bound_error(
                        param_name, label, op.loop_var
                    )
                )

    def _shape_dimension_source(
        self,
        operand: Value,
        enclosing_loop_vars: frozenset[str],
        dependency_graph: dict[str, set[str]],
        aliased_shape_dims: frozenset[str],
    ) -> tuple[str, int, str | None] | None:
        """Trace a structural expression back to an input array dimension.

        Args:
            operand (Value): Structural value whose classical dependencies
                should be inspected.
            enclosing_loop_vars (frozenset[str]): Loop-variable UUIDs that are
                resolved during emit-time unrolling.
            dependency_graph (dict[str, set[str]]): Classical dependency edges
                visible at ``operand``.
            aliased_shape_dims (frozenset[str]): Owned-block formal dimensions
                whose display names do not identify public input arrays.

        Returns:
            tuple[str, int, str | None] | None: Public array name, dimension
            index, and recognized frontend element-type name when one is
            reachable, otherwise ``None``. The element-type entry is ``None``
            for an unrecognized IR type.
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
            shape_info = (
                None
                if uuid in aliased_shape_dims
                else self._input_shape_dimensions.get(uuid)
            )
            if shape_info is not None:
                return shape_info
            stack.extend(dependency_graph.get(uuid, ()))
        return None

    def _runtime_parameter_source(
        self,
        operand: Value,
        enclosing_loop_vars: frozenset[str],
        dependency_graph: dict[str, set[str]],
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
            dependency_graph (dict[str, set[str]]): Classical dependency edges
                visible at the structural value.

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
            stack.extend(dependency_graph.get(uuid, ()))
        return None
