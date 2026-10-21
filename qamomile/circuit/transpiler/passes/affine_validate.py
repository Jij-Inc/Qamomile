"""Affine type validation pass: Verify quantum resources are used correctly."""

from __future__ import annotations

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import (
    Operation,
    ReleaseSliceViewOperation,
    SliceArrayOperation,
)
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.control_flow import HasNestedOps, IfOperation
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.errors import AffineTypeError, ValidationError
from qamomile.circuit.transpiler.passes import Pass


def operation_owned_blocks(op: Operation) -> list[Block]:
    """Enumerate the implementation blocks an operation privately owns.

    Owned blocks are boxed recipe bodies that survive inlining and reach
    emit as part of the operation itself — unlike control-flow regions
    (``HasNestedOps``), they are separate value namespaces whose
    ``input_values`` form their own ownership boundary. Three operation
    families carry them:

    - ``ControlledUOperation.block`` (including ``ConcreteControlledU``):
      the unitary body applied under control.
    - ``InverseBlockOperation.source_block`` / ``implementation_block``:
      the forward recipe and the pre-built inverse fallback.
    - ``InvokeOperation.definition``: the standard ``body`` plus every
      ``implementations[*].body`` (native / strategy-specific candidates,
      the same both-kinds enumeration ``region_validation`` uses). The
      standard pipeline flattens inlineable invokes before
      ``affine_validate`` runs, but non-inline, hand-built, or
      deserialized IR can still carry them — and emit may select an
      implementation body over the standard one.

    Args:
        op (Operation): Operation to inspect.

    Returns:
        list[Block]: The owned blocks present on ``op`` (empty for the
        overwhelming majority of operations).
    """
    blocks: list[Block] = []
    if isinstance(op, ControlledUOperation):
        if op.block is not None:
            blocks.append(op.block)
    elif isinstance(op, InverseBlockOperation):
        if op.source_block is not None:
            blocks.append(op.source_block)
        if op.implementation_block is not None:
            blocks.append(op.implementation_block)
    elif isinstance(op, InvokeOperation):
        if op.definition is not None:
            if op.definition.body is not None:
                blocks.append(op.definition.body)
            for implementation in op.definition.implementations:
                if implementation.body is not None:
                    blocks.append(implementation.body)
    else:
        # Every other operation owns no implementation block.
        pass
    return blocks


class AffineValidationPass(Pass[Block, Block]):
    """Validate affine type semantics at IR level.

    This pass serves as a safety net to catch affine type violations
    that may have bypassed the frontend checks. It verifies that each
    quantum value is used (consumed) at most once — in the entry block's
    control-flow nesting AND inside every operation-owned implementation
    block (controlled-U bodies, inverse blocks, un-inlined callable
    bodies; see :func:`operation_owned_blocks`), each of which is
    validated as an independent affine scope. It does NOT detect
    "never consumed" / silent-discard patterns; the branch-internal and
    loop-body discard cases are rejected separately by
    ``reject_control_flow_quantum_discard`` in
    ``qamomile.circuit.transpiler.passes.analyze``.

    Input: Block (any kind)
    Output: Same Block (unchanged, validation only)
    """

    @property
    def name(self) -> str:
        return "affine_validate"

    def run(self, input: Block) -> Block:
        """Validate affine type semantics in the block.

        Raises:
            ValidationError: If the block kind is not AFFINE.
            AffineTypeError: If a quantum value is consumed multiple times.
        """
        if input.kind not in (BlockKind.AFFINE,):
            raise ValidationError(
                f"AffineValidationPass expects AFFINE block, got {input.kind}",
            )

        # Track which quantum values have been consumed
        # Maps uuid -> operation name that consumed it
        consumed: dict[str, str] = {}

        self._validate_operations(input.operations, consumed)

        return input  # Pass-through, no modifications

    def _validate_operations(
        self,
        operations: list[Operation],
        consumed: dict[str, str],
    ) -> None:
        """Validate operations for affine type violations.

        Note: This method handles control flow specially due to the need
        for scope management (e.g., loop scopes, branch merging).
        """
        for op in operations:
            op_name = type(op).__name__

            # SliceArrayOperation takes the parent array as an operand
            # but does NOT consume it — it only produces metadata
            # describing a strided view.  ReleaseSliceViewOperation
            # likewise carries a sliced ArrayValue operand without
            # consuming it: the op is a declarative borrow-return
            # marker for SliceBorrowCheckPass and does not
            # contribute to the affine-type consume count.
            #
            # ``affine_validate`` runs *before* ``partial_eval`` (and
            # therefore before ``slice_borrow_check`` /
            # ``strip_slice_ops`` — see the pipeline in
            # ``Transpiler.transpile()``), so both ops are still
            # present in the block at this point and the affine-type
            # walk would otherwise mis-count their array operand as a
            # consume.  Skip them explicitly.
            if isinstance(op, (SliceArrayOperation, ReleaseSliceViewOperation)):
                continue

            # Check each operand
            for operand in op.operands:
                if isinstance(operand, Value) and operand.type.is_quantum():
                    self._check_and_mark_consumed(operand, op_name, consumed)

            # Handle control flow with scope management. IfOperation
            # branch-merge yields are neither operands nor nested
            # operations, so the consume walk sees exactly the branch
            # bodies here and never counts a merge source as a consume.
            if isinstance(op, HasNestedOps):
                scoped_sets: list[dict[str, str]] = []
                for region in op.nested_regions():
                    scoped = consumed.copy()
                    self._validate_operations(list(region.operations), scoped)
                    scoped_sets.append(scoped)
                # For IfOperation, merge all scoped consumed back:
                # anything consumed in either branch is considered consumed.
                # For loops (For/ForItems/While), don't propagate - values
                # consumed in loop may or may not be consumed after.
                if isinstance(op, IfOperation):
                    for scoped in scoped_sets:
                        consumed.update(scoped)

            # Operation-OWNED implementation blocks (controlled-U bodies,
            # inverse blocks, un-inlined callable bodies) are separate
            # value namespaces, not control-flow scopes: their
            # ``input_values`` are the ownership boundary, so each is
            # validated as an independent affine scope with a FRESH
            # consumed map instead of sharing the enclosing walk's state.
            # Boxed blocks are HIERARCHICAL-kind, so this deliberately
            # enters via ``_validate_operations`` rather than ``run()``
            # (whose AFFINE-kind guard would reject them). Recursion
            # covers boxed-in-boxed nesting and control flow inside a
            # boxed body.
            for owned in operation_owned_blocks(op):
                self._validate_operations(owned.operations, {})

    def _check_and_mark_consumed(
        self,
        value: Value,
        operation_name: str,
        consumed: dict[str, str],
    ) -> None:
        """Check if a value was already consumed and mark it as consumed.

        Raises:
            AffineTypeError: If the value was already consumed.
        """
        # Skip if this is a result of the previous operation with same uuid
        # (SSA-style versioning means the same logical value has different
        # versions; physical qubit allocation happens later in emit).
        # We check by uuid which should be unique per value instance

        if value.uuid in consumed:
            first_consumer = consumed[value.uuid]
            raise AffineTypeError(
                f"Quantum value '{value.name}' was already consumed by "
                f"'{first_consumer}' and cannot be used again in '{operation_name}'.\n\n"
                f"This is likely an internal error - if you see this message, "
                f"please report it as a bug.",
                handle_name=value.name,
                operation_name=operation_name,
                first_use_location=first_consumer,
            )

        consumed[value.uuid] = operation_name
