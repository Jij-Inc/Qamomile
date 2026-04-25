"""Strip SliceArrayOperation nodes after slice_linearity_check has consumed them.

``SliceArrayOperation`` is purely declarative — the resulting sliced
``ArrayValue`` carries all the emit-time metadata on its own
(``slice_of`` / ``slice_start`` / ``slice_step``).  ``ConstantFoldingPass``
keeps these ops around when ``strip_slice_ops=False`` so the downstream
``SliceLinearityCheckPass`` can observe them in program order and treat
them as view-declaration markers.  Once the linearity check has run,
those ops are unneeded; segmentation and emission expect a pure
quantum-op stream without classical slice declarations in the middle.
This pass removes them and leaves the block otherwise unchanged.
"""

from __future__ import annotations

import dataclasses

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation, SliceArrayOperation

from . import Pass
from .control_flow_visitor import OperationTransformer


class StripSliceArrayOpsPass(Pass[Block, Block]):
    """Remove ``SliceArrayOperation`` nodes from the block.

    The op is only meaningful to ``SliceLinearityCheckPass`` as a
    view-declaration marker; downstream passes (analyze, plan, emit)
    neither need nor expect it.  Dropping the op here keeps the
    segmentation quantum-op-only.
    """

    @property
    def name(self) -> str:
        return "strip_slice_ops"

    def run(self, input: Block) -> Block:
        """Drop ``SliceArrayOperation`` nodes from ``input``.

        Walks the operation tree (including nested control flow) and
        returns a structurally identical block with every
        ``SliceArrayOperation`` node removed.  The sliced
        ``ArrayValue`` results remain reachable through later
        operands' ``parent_array`` chains, so emit-time resolution is
        unaffected.

        Args:
            input: Block that has completed ``SliceLinearityCheckPass``.

        Returns:
            A new block with all ``SliceArrayOperation`` nodes removed.
        """

        class Stripper(OperationTransformer):
            def transform_operation(self, op: Operation) -> Operation | None:
                if isinstance(op, SliceArrayOperation):
                    return None
                return op

        new_ops = Stripper().transform_operations(input.operations)
        return dataclasses.replace(input, operations=new_ops)
