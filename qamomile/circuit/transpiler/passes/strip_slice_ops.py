"""Strip SliceArrayOperation / ReleaseSliceViewOperation after the linearity check.

Both ops are purely declarative — the resulting sliced ``ArrayValue``
carries all the emit-time metadata on its own (``slice_of`` /
``slice_start`` / ``slice_step``), and the release marker only exists
so ``SliceBorrowCheckPass`` can observe explicit slice-assignment
borrow-returns in program order.  ``ConstantFoldingPass`` keeps these
ops around when ``strip_slice_ops=False`` so the downstream linearity
check can see them.  Once the linearity check has run, both ops are
unneeded; segmentation and emission expect a pure quantum-op stream
without classical slice declarations in the middle.  This pass removes
them and leaves the block otherwise unchanged.
"""

from __future__ import annotations

import dataclasses

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import (
    Operation,
    ReleaseSliceViewOperation,
    SliceArrayOperation,
)
from qamomile.circuit.ir.operation.control_flow import ForItemsOperation, ForOperation

from . import Pass
from .control_flow_visitor import OperationTransformer


class StripSliceArrayOpsPass(Pass[Block, Block]):
    """Remove ``SliceArrayOperation`` / ``ReleaseSliceViewOperation`` nodes.

    Both ops are only meaningful to ``SliceBorrowCheckPass`` as
    view-declaration / release markers; downstream passes (analyze,
    plan, emit) neither need nor expect them.  Dropping them here
    keeps the segmentation quantum-op-only.
    """

    @property
    def name(self) -> str:
        return "strip_slice_ops"

    def run(self, input: Block) -> Block:
        """Drop slice marker ops from ``input``.

        Walks the operation tree (including nested control flow) and
        returns a structurally identical block with every
        ``SliceArrayOperation`` and ``ReleaseSliceViewOperation`` node
        removed.  The sliced ``ArrayValue`` results remain reachable
        through later operands' ``parent_array`` chains, so emit-time
        resolution is unaffected.

        Args:
            input: Block that has completed ``SliceBorrowCheckPass``.

        Returns:
            A new block with all slice marker ops removed.
        """

        class Stripper(OperationTransformer):
            def transform_operations(
                self, operations: list[Operation]
            ) -> list[Operation]:
                """Transform operations and drop marker-only loop shells.

                Args:
                    operations (list[Operation]): Operations to rewrite.

                Returns:
                    list[Operation]: Rewritten operations with slice
                    markers and newly-empty ``For`` / ``ForItems``
                    operations removed.
                """
                result: list[Operation] = []
                for op in operations:
                    transformed = self.transform_operation(op)
                    if transformed is None:
                        continue
                    transformed = self._transform_control_flow(transformed)
                    if isinstance(transformed, (ForOperation, ForItemsOperation)) and (
                        not transformed.operations
                    ):
                        continue
                    result.append(transformed)
                return result

            def transform_operation(self, op: Operation) -> Operation | None:
                """Drop slice marker operations.

                Args:
                    op (Operation): Operation to inspect.

                Returns:
                    Operation | None: ``None`` for slice markers, or
                    ``op`` unchanged otherwise.
                """
                if isinstance(op, (SliceArrayOperation, ReleaseSliceViewOperation)):
                    return None
                return op

        new_ops = Stripper().transform_operations(input.operations)
        return dataclasses.replace(input, operations=new_ops)
