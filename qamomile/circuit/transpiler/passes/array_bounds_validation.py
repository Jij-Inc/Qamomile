"""Reject reachable compile-time array accesses outside resolved extents."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    HasNestedOps,
    IfOperation,
)
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.operation.slice_array import SliceArrayOperation
from qamomile.circuit.ir.value import ArrayValue, Value, ValueBase
from qamomile.circuit.transpiler.block_parameter_binding import pair_block_operands
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.passes.value_mapping import ValueSubstitutor


def _constant_integer(value: ValueBase | None) -> int | None:
    """Return a non-boolean integer constant carried by an IR value.

    Args:
        value (ValueBase | None): Candidate scalar value.

    Returns:
        int | None: Normalized Python integer, or ``None`` when ``value`` is
            absent, symbolic, boolean, or non-integral.
    """
    if not isinstance(value, Value) or not value.is_constant():
        return None
    constant = value.get_const()
    if isinstance(constant, (bool, np.bool_)) or not isinstance(
        constant, (int, np.integer)
    ):
        return None
    return int(constant)


def _root_array(array: ArrayValue) -> ArrayValue:
    """Return the root container beneath a possibly nested array view.

    Args:
        array (ArrayValue): Root array or sliced view.

    Returns:
        ArrayValue: Last acyclic parent in the ``slice_of`` chain.
    """
    current = array
    seen: set[str] = set()
    while current.slice_of is not None and current.uuid not in seen:
        seen.add(current.uuid)
        current = current.slice_of
    return current


def _is_zero_trip_loop(operation: ForOperation) -> bool:
    """Return whether a for-loop has a statically empty iteration range.

    Args:
        operation (ForOperation): Loop whose ``start``, ``stop``, and ``step``
            operands should be inspected.

    Returns:
        bool: ``True`` only when all bounds are constant and their range is
            provably empty. Symbolic or zero-step ranges return ``False`` so
            their existing validators retain ownership of the diagnosis.
    """
    if len(operation.operands) < 3:
        return False
    start, stop, step = (
        _constant_integer(operation.operands[index]) for index in range(3)
    )
    if start is None or stop is None or step in (None, 0):
        return False
    return start >= stop if step > 0 else start <= stop


class ArrayBoundsValidationPass(Pass[Block, Block]):
    """Reject reachable element accesses and views outside array bounds.

    This pass runs after partial evaluation has resolved binding-dependent
    slice extents and before declarative slice operations are stripped. It
    deliberately skips statically zero-trip loop bodies so an unreachable
    access does not become a false-positive compilation error.
    """

    @property
    def name(self) -> str:
        """Return the stable pass identifier.

        Returns:
            str: The name ``"array_bounds_validation"``.
        """
        return "array_bounds_validation"

    def run(self, input: Block) -> Block:
        """Validate reachable array element operands in one semantic block.

        Args:
            input (Block): Post-partial-evaluation affine or hierarchical
                block whose concrete array extents should be checked.

        Returns:
            Block: ``input`` unchanged when every reachable access and view is
                valid or still symbolic.

        Raises:
            ValidationError: If ``input`` has an unsupported block kind, a
                reachable constant index is outside a resolved array extent,
                or a concrete view descriptor exceeds its physical root.
        """
        if input.kind not in (BlockKind.AFFINE, BlockKind.HIERARCHICAL):
            raise ValidationError(
                "ArrayBoundsValidationPass expects AFFINE or HIERARCHICAL "
                f"block, got {input.kind}",
            )
        self._walk_block(
            input,
            owned_blocks_on_path=frozenset({id(input)}),
        )
        return input

    def _walk_block(
        self,
        block: Block,
        *,
        substitutor: ValueSubstitutor | None = None,
        owned_blocks_on_path: frozenset[int],
    ) -> None:
        """Validate reachable operations and public outputs in one block.

        Args:
            block (Block): Entry or operation-owned block to inspect.
            substitutor (ValueSubstitutor | None): Formal-to-actual mapping for
                an operation-owned block. Defaults to ``None`` in the entry
                block.
            owned_blocks_on_path (frozenset[int]): Block object identities on
                the active recursion path.

        Raises:
            ValidationError: If a reachable operation input or block output
                has a concrete out-of-bounds array index.
        """
        self._walk_operations(
            block.operations,
            substitutor=substitutor,
            owned_blocks_on_path=owned_blocks_on_path,
        )
        outputs = (
            block.output_values
            if substitutor is None
            else (substitutor.substitute_value(value) for value in block.output_values)
        )
        self._validate_values(outputs)

    def _walk_operations(
        self,
        operations: list[Operation],
        *,
        substitutor: ValueSubstitutor | None = None,
        owned_blocks_on_path: frozenset[int],
    ) -> None:
        """Walk reachable operations and validate their direct operands.

        Args:
            operations (list[Operation]): Operations in the current reachable
                control-flow region.
            substitutor (ValueSubstitutor | None): Formal-to-actual mapping for
                an operation-owned block. Defaults to ``None`` in the entry
                block.
            owned_blocks_on_path (frozenset[int]): Block object identities on
                the active recursion path.

        Raises:
            ValidationError: If a reachable operand has a concrete
                out-of-bounds array index.
        """
        for operation in operations:
            current = (
                operation
                if substitutor is None
                else substitutor.substitute_operation(operation)
            )
            if isinstance(current, IfOperation):
                self._walk_if_operation(
                    current,
                    substitutor=substitutor,
                    owned_blocks_on_path=owned_blocks_on_path,
                )
                continue
            if isinstance(current, ForOperation):
                self._walk_for_operation(
                    current,
                    substitutor=substitutor,
                    owned_blocks_on_path=owned_blocks_on_path,
                )
                continue

            self._validate_values(current.all_input_values())
            if isinstance(current, SliceArrayOperation):
                for result in current.results:
                    if isinstance(result, ArrayValue):
                        self._validate_array_view(result)
            if isinstance(current, HasNestedOps):
                for nested_operations in current.nested_op_lists():
                    self._walk_operations(
                        nested_operations,
                        substitutor=substitutor,
                        owned_blocks_on_path=owned_blocks_on_path,
                    )
            self._walk_owned_blocks(current, owned_blocks_on_path)

    def _walk_if_operation(
        self,
        operation: IfOperation,
        *,
        substitutor: ValueSubstitutor | None,
        owned_blocks_on_path: frozenset[int],
    ) -> None:
        """Validate only branches and yields reachable from one conditional.

        Args:
            operation (IfOperation): Conditional whose condition, branch
                yields, and nested operations should be checked.
            substitutor (ValueSubstitutor | None): Formal-to-actual mapping
                active for this conditional.
            owned_blocks_on_path (frozenset[int]): Block object identities on
                the active recursion path.

        Raises:
            ValidationError: If a reachable array access is out of bounds.
        """
        self._validate_values(operation.operands)
        self._validate_values(rebind.before for rebind in operation.branch_rebinds)

        branches = operation.nested_op_lists()
        yields = (operation.true_yields, operation.false_yields)
        if operation.condition.is_constant():
            branch_index = 0 if bool(operation.condition.get_const()) else 1
            self._validate_values(yields[branch_index])
            self._walk_operations(
                branches[branch_index],
                substitutor=substitutor,
                owned_blocks_on_path=owned_blocks_on_path,
            )
            return

        for branch_yields, branch_operations in zip(yields, branches, strict=True):
            self._validate_values(branch_yields)
            self._walk_operations(
                branch_operations,
                substitutor=substitutor,
                owned_blocks_on_path=owned_blocks_on_path,
            )

    def _walk_for_operation(
        self,
        operation: ForOperation,
        *,
        substitutor: ValueSubstitutor | None,
        owned_blocks_on_path: frozenset[int],
    ) -> None:
        """Validate a counted loop without inspecting a zero-trip body.

        Loop bounds, carried initializers, and pre-loop rebind values are read
        even when the loop is empty. Body operations and yielded/rebound body
        values are reachable only when the loop may execute.

        Args:
            operation (ForOperation): Counted loop to inspect.
            substitutor (ValueSubstitutor | None): Formal-to-actual mapping
                active for this loop.
            owned_blocks_on_path (frozenset[int]): Block object identities on
                the active recursion path.

        Raises:
            ValidationError: If a reachable array access is out of bounds.
        """
        self._validate_values(operation.operands)
        self._validate_values(region_arg.init for region_arg in operation.region_args)
        self._validate_values(
            rebind.before for rebind in operation.loop_carried_rebinds
        )
        if _is_zero_trip_loop(operation):
            return

        self._validate_values(
            region_arg.yielded for region_arg in operation.region_args
        )
        self._validate_values(rebind.after for rebind in operation.loop_carried_rebinds)
        self._walk_operations(
            operation.operations,
            substitutor=substitutor,
            owned_blocks_on_path=owned_blocks_on_path,
        )

    def _walk_owned_blocks(
        self,
        operation: Operation,
        owned_blocks_on_path: frozenset[int],
    ) -> None:
        """Validate SELECT cases and controlled-unitary bodies at their call site.

        Operation-owned blocks have independent value namespaces. Their formal
        inputs are therefore substituted with the owning operation's actual
        operands before bounds and reachability are inspected. Each shared
        block is revisited for each call site, while the path-local identity
        guard prevents malformed cyclic block graphs from recursing forever.

        Args:
            operation (Operation): Operation that may own nested blocks.
            owned_blocks_on_path (frozenset[int]): Block object identities on
                the active recursion path.

        Raises:
            ValidationError: If a reachable owned-body access is out of bounds.
        """
        owned_blocks: list[tuple[Block, Sequence[ValueBase]]] = []
        if isinstance(operation, SelectOperation):
            actuals = [*operation.target_operands, *operation.param_operands]
            owned_blocks.extend((block, actuals) for block in operation.case_blocks)
        elif (
            isinstance(operation, ControlledUOperation) and operation.block is not None
        ):
            targets = [
                value for value in operation.target_operands if value.type.is_quantum()
            ]
            actuals = [*targets, *operation.param_operands]
            owned_blocks.append((operation.block, actuals))

        for block, actuals in owned_blocks:
            block_id = id(block)
            if block_id in owned_blocks_on_path:
                continue
            mapping: dict[str, ValueBase] = {}
            for formal, actual in pair_block_operands(block, actuals):
                mapping[formal.uuid] = actual
                if isinstance(formal, ArrayValue) and isinstance(actual, ArrayValue):
                    mapping.update(
                        (formal_dimension.uuid, actual_dimension)
                        for formal_dimension, actual_dimension in zip(
                            formal.shape,
                            actual.shape,
                            strict=False,
                        )
                    )
            self._walk_block(
                block,
                substitutor=ValueSubstitutor(mapping, transitive=True),
                owned_blocks_on_path=owned_blocks_on_path | {block_id},
            )

    def _validate_array_view(self, view: ArrayValue) -> None:
        """Validate the concrete physical coverage declared by one view.

        A serialized template can acquire concrete parent and view lengths only
        after binding. When the view descriptor then claims slots beyond its
        parent, a loop-local symbolic index can otherwise hide the invalid
        access until backend emission. Checking both affine endpoints is
        sufficient because supported slice strides are constant and positive;
        checking both also keeps malformed non-positive strides conservative.

        Args:
            view (ArrayValue): Slice result whose declared coverage should fit
                every resolved ancestor extent.

        Raises:
            ValidationError: If a concrete non-empty view extends outside a
                concrete ancestor extent.
        """
        if view.slice_of is None or not view.shape:
            return
        length = _constant_integer(view.shape[0])
        if length is None or length <= 0:
            return
        self._validate_leading_coordinate(0, view)
        if length > 1:
            self._validate_leading_coordinate(length - 1, view)

    def _validate_values(self, values: Iterable[ValueBase]) -> None:
        """Validate array-element accesses in one iterable of IR values.

        Args:
            values (Iterable[ValueBase]): Direct operation inputs or semantic
                control-flow values.

        Raises:
            ValidationError: If a value is a concrete out-of-bounds access.
        """
        for value in values:
            self._validate_element_access(value)

    def _validate_element_access(self, value: ValueBase) -> None:
        """Validate every concrete coordinate of one array element value.

        For the leading coordinate of a sliced vector, each view-local index is
        validated before applying that view's affine map to its parent. This
        prevents a zero-length view from borrowing an otherwise valid slot in
        the root array.

        Args:
            value (ValueBase): Operation operand that may represent an array
                element access.

        Raises:
            ValidationError: If a concrete index falls outside a concrete
                immediate-view or root-array extent.
        """
        if not isinstance(value, Value):
            return
        parent = value.parent_array
        if parent is None or not value.element_indices:
            return

        root = _root_array(parent)
        leading_index = _constant_integer(value.element_indices[0])
        if leading_index is not None:
            self._validate_leading_coordinate(leading_index, parent)

        for dimension, index_value in enumerate(value.element_indices[1:], start=1):
            index = _constant_integer(index_value)
            if index is None:
                continue
            self._validate_coordinate(
                index,
                parent,
                dimension=dimension,
                accessed_parent=parent,
                root=root,
            )

    def _validate_leading_coordinate(
        self,
        index: int,
        parent: ArrayValue,
    ) -> None:
        """Validate and map one leading coordinate through a view chain.

        Args:
            index (int): Concrete coordinate in ``parent`` space.
            parent (ArrayValue): Immediate array or view being accessed.

        Raises:
            ValidationError: If the coordinate falls outside a resolved local
                or ancestor extent.
        """
        root = _root_array(parent)
        current = parent
        current_index = index
        seen: set[str] = set()
        while current.uuid not in seen:
            seen.add(current.uuid)
            self._validate_coordinate(
                current_index,
                current,
                dimension=0,
                accessed_parent=parent,
                root=root,
            )
            if current.slice_of is None:
                break
            start = _constant_integer(current.slice_start)
            step = _constant_integer(current.slice_step)
            if start is None or step is None:
                break
            current_index = start + step * current_index
            current = current.slice_of

    def _validate_coordinate(
        self,
        index: int,
        bounded_array: ArrayValue,
        *,
        dimension: int,
        accessed_parent: ArrayValue,
        root: ArrayValue,
    ) -> None:
        """Reject one concrete coordinate outside its resolved extent.

        Args:
            index (int): Concrete coordinate in ``bounded_array`` space.
            bounded_array (ArrayValue): Array or view whose dimension bounds
                ``index``.
            dimension (int): Zero-based dimension being checked.
            accessed_parent (ArrayValue): Immediate parent named by the source
                element access.
            root (ArrayValue): Root array beneath ``accessed_parent``.

        Raises:
            ValidationError: If ``index`` is negative or no smaller than the
                resolved dimension extent.
        """
        if dimension >= len(bounded_array.shape):
            return

        root_name = root.name or "<anonymous>"
        parent_name = accessed_parent.name or root_name
        if accessed_parent.slice_of is None:
            subject = f"array '{root_name}'"
        else:
            subject = f"array view '{parent_name}' of root array '{root_name}'"
        if index < 0:
            raise ValidationError(
                f"Index {index} is out of range for {subject} at dimension "
                f"{dimension}. Array indices must be non-negative.",
                value_name=root_name,
            )

        extent_value = bounded_array.shape[dimension]
        extent = _constant_integer(extent_value)
        if extent is None or index < extent:
            return

        extent_name = extent_value.name or f"dimension {dimension}"
        view_context = ""
        if (
            accessed_parent.slice_of is not None
            and bounded_array.uuid != accessed_parent.uuid
            and accessed_parent.shape
        ):
            view_extent_value = accessed_parent.shape[0]
            view_extent = _constant_integer(view_extent_value)
            if view_extent is not None:
                view_extent_name = view_extent_value.name or "dimension 0"
                view_context = (
                    f"The view extent '{view_extent_name}' resolved to {view_extent}. "
                )
        requirement = (
            f"The extent '{extent_name}' must resolve to at least "
            f"{index + 1} for this access."
        )
        raise ValidationError(
            f"Index {index} is out of range for {subject} at dimension "
            f"{dimension}: extent '{extent_name}' resolved to {extent}. "
            f"{view_context}{requirement}",
            value_name=root_name,
        )


__all__ = ["ArrayBoundsValidationPass"]
