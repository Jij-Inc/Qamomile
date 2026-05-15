"""Slice operation that produces a strided view of an array."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .operation import Operation, OperationKind, ParamHint, Signature

if TYPE_CHECKING:
    from qamomile.circuit.ir.types import ValueType


@dataclass
class SliceArrayOperation(Operation):
    """Construct a strided view of an ``ArrayValue``.

    The op itself performs no quantum action — it records that the
    result ``ArrayValue`` is a strided view of the operand parent
    with the given ``start`` / ``step``.  The result's
    ``slice_of`` / ``slice_start`` / ``slice_step`` fields carry the
    affine map used by the emit-time resolver.

    ``SliceArrayOperation`` is classified as :attr:`OperationKind.CLASSICAL`
    because slicing is pure index selection — no new quantum operation
    is introduced.  ``ConstantFoldingPass`` strips this operation from
    the block before :mod:`~qamomile.circuit.transpiler.passes.separate`
    sees it, so the op serves as a trace-time IR checkpoint (useful for
    debugging and inspection) but never reaches emit.  Reaching emit is
    a compiler-internal invariant violation.

    Attributes:
        operands: ``[parent_array, start, step]`` — the array being
            sliced, the parent-space start index, and the stride.
        results: ``[sliced_array]`` — a new ``ArrayValue`` whose
            ``slice_of`` field points to ``parent_array``.

    Example:
        ``q[1::2]`` on a ``Vector[Qubit]`` emits::

            SliceArrayOperation(
                operands=[q_value, uint_1, uint_2],
                results=[sliced_value],  # slice_of=q_value, slice_start=uint_1, slice_step=uint_2
            )
    """

    @property
    def signature(self) -> Signature:
        """Return the type signature of this slice operation.

        Returns:
            A :class:`Signature` with three operands (parent array,
            start, step) and one result (the sliced array).  Types are
            inferred from the attached ``Value`` instances so the
            signature adapts to quantum vs classical arrays without
            extra configuration.
        """
        from qamomile.circuit.ir.types.primitives import UIntType

        parent_type: "ValueType | None"
        parent_type = self.operands[0].type if self.operands else None
        result_type: "ValueType | None"
        result_type = self.results[0].type if self.results else None

        return Signature(
            operands=[
                ParamHint(name="parent", type=parent_type) if parent_type else None,
                ParamHint(name="start", type=UIntType()),
                ParamHint(name="step", type=UIntType()),
            ],
            results=[ParamHint(name="sliced", type=result_type)] if result_type else [],
        )

    @property
    def operation_kind(self) -> OperationKind:
        """Slice is classical — it selects indices without quantum action.

        Returns:
            :attr:`OperationKind.CLASSICAL`.
        """
        return OperationKind.CLASSICAL


@dataclass
class ReleaseSliceViewOperation(Operation):
    """Mark a slice view's borrow as explicitly returned to its parent.

    Emitted by :meth:`Vector.__setitem__` when used with a slice index
    (``qs[a:b] = qmc.h(qs[a:b])``).  This op tells the post-fold
    linearity checker
    (:class:`~qamomile.circuit.transpiler.passes.slice_linearity_check.SliceLinearityCheckPass`)
    that the view referenced in ``operands[0]`` no longer owns its
    covered parent slots, mirroring the frontend's
    ``VectorView.consume(operation_name="slice assignment")`` borrow
    release.

    Like :class:`SliceArrayOperation`, this op is a declarative
    classical-side marker that does not survive into the emit stream:
    :class:`~qamomile.circuit.transpiler.passes.strip_slice_ops.StripSliceArrayOpsPass`
    removes both :class:`SliceArrayOperation` and
    :class:`ReleaseSliceViewOperation` after
    :class:`SliceLinearityCheckPass` has observed them.  Reaching emit
    is a compiler-internal invariant violation and is rejected with a
    ``RuntimeError`` from :mod:`standard_emit`.

    Within a control-flow body (``ForOperation`` / ``WhileOperation``
    / ``IfOperation``), this op only releases view borrows that were
    *created within the same body*.  Releasing a borrow that the
    enclosing block has registered (an "outer-snapshot" borrow) is
    rejected by ``SliceLinearityCheckPass`` with
    ``SliceLinearityViolationError`` — the loop-merge semantics of the
    pass cannot propagate entry deletions out of the body, so the only
    way to keep the static check consistent is to forbid that pattern.

    Attributes:
        operands: ``[view_av]`` — the sliced :class:`ArrayValue` whose
            borrow is being released.  Must have ``slice_of`` non-None.
        results: ``[]`` — the op produces no new IR values.

    Example:
        ``qs[1:3] = qmc.h(qs[1:3])`` emits, after the broadcast loop::

            ReleaseSliceViewOperation(
                operands=[qmc_h_result_view],  # slice_of=qs_value
                results=[],
            )
    """

    @property
    def signature(self) -> Signature:
        """Return the type signature of this release operation.

        Returns:
            A :class:`Signature` with a single operand (the view to
            release) and no results.  The operand type is inferred
            from the attached :class:`Value`.
        """
        operand_type: "ValueType | None"
        operand_type = self.operands[0].type if self.operands else None
        return Signature(
            operands=[
                ParamHint(name="view", type=operand_type) if operand_type else None,
            ],
            results=[],
        )

    @property
    def operation_kind(self) -> OperationKind:
        """Release is classical — it updates borrow tracking metadata only.

        Returns:
            :attr:`OperationKind.CLASSICAL`.
        """
        return OperationKind.CLASSICAL
