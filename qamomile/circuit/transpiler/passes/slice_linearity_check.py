"""Post-fold block-wide linearity checker.

Mirrors the frontend's ``ArrayBase._borrowed_indices`` borrow-tracker
semantics on the IR after :class:`ConstantFoldingPass` has resolved
slice bounds to concrete values.  Two classes of bugs are caught here
that the trace-time checker cannot see on its own:

1. Aliasing between a slice view with symbolic bounds and a direct
   access to the parent — those bounds are ``UInt`` at trace time, so
   the covered slot set isn't enumerable until bindings are applied
   during :class:`ConstantFoldingPass`.
2. ``return q`` or ``measure(q)`` completing while a borrow is still
   outstanding — the frontend's ``validate_all_returned`` is driven by
   ``consume`` and does not fire when a kernel returns its parent array
   without consuming it first.  The block walker in this pass raises
   :class:`UnreturnedBorrowAtBlockEndError` when the state dict is
   non-empty at completion.

State shape mirrors the frontend: a single polymorphic dict keyed by
``(f"const:<idx>",)`` or ``(f"sym:<uuid>",)`` with values of either
``Value`` (direct borrow, the element Value currently checked out) or
``ArrayValue`` (a slice view that owns that slot).  ``isinstance`` on
the value dispatches the two error messages.
"""

from __future__ import annotations

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import (
    Operation,
    ReturnOperation,
    SliceArrayOperation,
)
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, CompOp
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.control_flow import (
    HasNestedOps,
    IfOperation,
)
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import (
    MeasureOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.value import ArrayValue, Value, ValueBase
from qamomile.circuit.transpiler.errors import (
    SliceLinearityViolationError,
    UnreturnedBorrowAtBlockEndError,
    ValidationError,
)

from . import Pass

BorrowKey = tuple[str, ...]


class _ConsumedSlotMarker:
    """Sentinel marking a physical qubit slot destroyed by a prior destructive op.

    Installed by :meth:`SliceLinearityCheckPass._process_result_releases`
    when a destructive operation (``MeasureOperation``,
    ``MeasureVectorOperation``, ``CastOperation``) consumes a view's
    covered slots.  Subsequent operand access to the same slot is
    treated as a violation, matching the frontend's ``_CONSUMED_SLOT``
    sentinel semantics.  ``ExpvalOp`` does not install this marker —
    its physical qubits survive the expval (it only retires the SSA
    handle).  Only one instance ever exists; identity is the test.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return "<IRConsumedSlot>"


_CONSUMED_SLOT: "_ConsumedSlotMarker" = _ConsumedSlotMarker()


_IR_DESTRUCTIVE_OPS: tuple[type, ...] = (
    MeasureOperation,
    MeasureVectorOperation,
    CastOperation,
)


Owner = Value | ArrayValue | _ConsumedSlotMarker
State = dict[BorrowKey, Owner]


class SliceLinearityCheckPass(Pass[Block, Block]):
    """Post-fold linearity checker for sliced views and borrow state.

    Runs after :class:`ConstantFoldingPass` (so slice bounds are
    concrete where possible) and before segmentation / emit.  Walks
    the operations of the root block in order, maintaining a borrow
    state map modelled on the frontend's
    :attr:`ArrayBase._borrowed_indices` — a single ``dict`` whose
    values discriminate between direct element borrows (``Value``)
    and slice-view owners (``ArrayValue``) by ``isinstance``.

    At the end of the walk, any non-empty entry becomes an
    :class:`UnreturnedBorrowAtBlockEndError` — closing the pre-existing
    silent hole where ``return q`` with an outstanding borrow went
    unnoticed.
    """

    def __init__(self) -> None:
        """Initialize per-run mutable state to safe defaults."""
        self._used_view_uuids: set[str] = set()

    @property
    def name(self) -> str:
        """Return the pass identifier for tracing/logging.

        Returns:
            The short name ``"slice_linearity_check"``.
        """
        return "slice_linearity_check"

    def run(self, input: Block) -> Block:
        """Run the borrow tracker over ``input``.

        Args:
            input: Block to check.  Expected to be in AFFINE or
                HIERARCHICAL kind — post-fold but pre-segmentation.

        Returns:
            The same ``Block`` unchanged (pass-through).

        Raises:
            ValidationError: If called on an unexpected block kind.
            SliceLinearityViolationError: If a slice view and a direct
                access collide, or if two views overlap.
            UnreturnedBorrowAtBlockEndError: If any borrow remains
                outstanding once the root block completes.
        """
        if input.kind not in (BlockKind.AFFINE, BlockKind.HIERARCHICAL):
            raise ValidationError(
                f"SliceLinearityCheckPass expects AFFINE or HIERARCHICAL "
                f"block, got {input.kind}",
            )

        state: State = {}
        # ``used_view_uuids`` tracks slice-view ArrayValue uuids whose
        # elements have been referenced in any operand since the view
        # was registered.  It is consulted by
        # ``_register_slice_bulk_borrow_if_new`` to decide whether an
        # overlapping new view may opportunistically drain an older
        # one — mirroring the frontend's ``VectorView._wrap`` logic
        # that lets ``a = q[0:3] (unused); b = q[1:4]`` succeed when
        # ``a`` has no outstanding element borrows.  A view moves
        # into this set the first time its elements, or the view
        # itself as a whole ArrayValue operand, appears in an
        # operation.
        self._used_view_uuids = set()
        self._walk(input.operations, state)

        # ``_ConsumedSlotMarker`` entries are not outstanding borrows
        # — they record physically-destroyed slots — and must be
        # excluded from the unreturned-borrow check.
        outstanding = {
            k: v
            for k, v in state.items()
            if not isinstance(v, _ConsumedSlotMarker)
        }
        if outstanding:
            raise UnreturnedBorrowAtBlockEndError(
                "Block completed with outstanding borrows:\n"
                + self._format_outstanding(outstanding)
                + "\n\nReturn every borrowed element (via ``q[i] = ...``) or "
                "consume the parent/view (via measure / expval / passing to "
                "another kernel) before the block ends."
            )

        return input

    # ------------------------------------------------------------------
    # Walker
    # ------------------------------------------------------------------

    def _walk(self, ops: list[Operation], state: State) -> None:
        """Walk operations in order, updating ``state`` in place.

        Handles nested control flow (``For``, ``ForItems``, ``While``,
        ``If``) with iteration-scoped state copies so a borrow taken and
        returned inside a loop body does not leak into sibling iterations.

        Args:
            ops: Operation list in execution order.
            state: Shared borrow tracker.  Mutated: slice bulk-borrows
                registered on a view's first use are added here, and
                released when the view is consumed.
        """
        for op in ops:
            if isinstance(op, ReturnOperation):
                # Return itself does not consume anything; the kernel's
                # terminator is handled by the caller (outer block end).
                continue

            if isinstance(op, BinOp) or isinstance(op, CompOp):
                # Pure classical arithmetic on UInts/Floats; no borrow
                # semantics.
                continue

            if isinstance(op, SliceArrayOperation):
                # Declarative; the resulting sliced ArrayValue's
                # bulk-borrow is registered lazily when first observed
                # via _register_slice_bulk_borrow_from_values below.
                # Actual registration happens through result-value
                # observation so that ConstantFoldingPass-stripped
                # runs (where the op is gone) still register sliced
                # arrays encountered via downstream operand fields.
                self._register_slice_bulk_borrow_if_new(
                    op.results[0] if op.results else None, state
                )
                continue

            if isinstance(op, HasNestedOps):
                self._walk_nested(op, state)
                continue

            # For gate / measure / expval / cast / etc., collect
            # element-access operands, resolve to (root_uuid, idx), and
            # apply borrow + release semantics.
            self._process_operand_borrows(op, state)
            self._process_result_releases(op, state)

    def _walk_nested(self, op: HasNestedOps, state: State) -> None:
        """Walk nested bodies with iteration-local state copies.

        For a ``For`` / ``ForItems`` / ``While`` body, each iteration is
        independent — borrows taken and returned inside one iteration
        must not leak to the next.  Slice bulk-borrows registered inside
        a body are intentionally propagated to the outer state when
        they persist past the body (i.e. when the view was created in
        the body but used after the loop); but the MVP here models
        each iteration with a copy and merges only views that still own
        slots at the end.

        For ``If``, walks both branches with independent state copies
        derived from the caller state; if the branches disagree on the
        final state, the more permissive (union of outstanding borrows)
        is adopted so that subsequent ops see all possibly-live
        borrows.  This is over-conservative but safe.

        Args:
            op: The ``HasNestedOps`` operation being dispatched.
            state: Caller's mutable borrow tracker.
        """
        if isinstance(op, IfOperation):
            true_state = dict(state)
            self._walk(op.true_operations, true_state)
            false_state = dict(state)
            self._walk(op.false_operations, false_state)
            # Conservative union: if either branch left a slot held,
            # the post-If state considers it held (so subsequent ops
            # get the stricter treatment).
            merged: State = {}
            for k, v in true_state.items():
                merged[k] = v
            for k, v in false_state.items():
                if k not in merged:
                    merged[k] = v
            state.clear()
            state.update(merged)
            return

        # For/ForItems/While: simulate one iteration.  We use a copy so
        # loop-internal borrows cancel at iteration end; post-loop view
        # ownership is inherited from the body's final state for the
        # merged set.
        for body in op.nested_op_lists():
            body_state = dict(state)
            self._walk(body, body_state)
            # Carry slice-view ownership across iteration boundaries
            # (views defined before the loop remain live after the loop
            # body).  Direct element borrows that persisted are left
            # alone — the outer block end-check will flag them.
            for k, v in body_state.items():
                if k not in state and isinstance(v, ArrayValue):
                    state[k] = v

    # ------------------------------------------------------------------
    # Borrow / release processing
    # ------------------------------------------------------------------

    def _process_operand_borrows(self, op: Operation, state: State) -> None:
        """Detect direct-access-over-view conflicts in ``op``'s operands.

        This pass deliberately does *not* track direct element borrows
        at IR level — those are SSA-versioned and have no meaningful
        "outstanding" concept once lowered.  It only flags the specific
        case where a view owns a slot and a different operation accesses
        the same slot directly (not through the view).  Detection of
        unreturned direct element borrows (``qv = q[0]; return q``)
        belongs to the frontend, whose ``ArrayBase.validate_all_returned``
        fires on consume and would not see IR-level ops anyway since
        element access alone emits nothing to the IR.

        Args:
            op: The operation whose operands are being inspected.
            state: Mutable borrow tracker (slice views only).

        Raises:
            SliceLinearityViolationError: If an operand's resolved slot
                is owned by a view that does not contain the operand.
        """
        # Mark whole-ArrayValue operands as "used" for drain tracking.
        # A whole-view operand (``measure(view)``, ``qft(view)``,
        # ``pauli_evolve(view, H, γ)``) counts as using the view even
        # though no per-element access is recorded.
        #
        # Additionally, for root (non-view) whole-array operands —
        # the most common being ``expval(q, H)`` — check every concrete
        # slot of the array against the consumed-slot markers in state.
        # This catches ``measure(q[1::2]); expval(q, H)`` at IR level,
        # complementing the frontend trace-time guard in ``expval()``.
        for v in op.operands:
            if not isinstance(v, ArrayValue):
                continue
            if v.slice_of is not None:
                self._used_view_uuids.add(v.uuid)
            else:
                # Root array whole operand: check each concrete slot.
                shape = v.shape
                if shape:
                    length = self._const_int(shape[0])
                    if length is not None:
                        for idx in range(length):
                            key = (f"const:{idx}",)
                            if isinstance(state.get(key), _ConsumedSlotMarker):
                                raise SliceLinearityViolationError(
                                    f"Whole-array operand '{v.name}' (slot {idx}) "
                                    f"is accessed after it was consumed by a "
                                    f"destructive view operation "
                                    f"(e.g. measure / cast on a view); the "
                                    f"physical qubit is no longer available."
                                )

        for v in op.operands:
            if not isinstance(v, Value):
                continue
            if v.parent_array is None or not v.element_indices:
                continue
            if not v.type.is_quantum():
                continue

            # Register any sliced ArrayValue seen via parent_array, and
            # mark the immediate sliced parent as "used" so subsequent
            # overlapping slices can't opportunistically drain it.
            if (
                isinstance(v.parent_array, ArrayValue)
                and v.parent_array.slice_of is not None
            ):
                self._used_view_uuids.add(v.parent_array.uuid)
            self._register_slice_bulk_borrow_if_new(v.parent_array, state)

            key = self._resolve_qubit_key(v)
            if key is None:
                continue  # Unresolvable symbolic — skip (safety).

            existing = state.get(key)
            if existing is None:
                continue

            if isinstance(existing, _ConsumedSlotMarker):
                # Slot's physical qubit was already destroyed by an
                # earlier destructive view op (``measure(q[1::2])``).
                # Any subsequent operand access — direct or via
                # another view — is invalid.
                raise SliceLinearityViolationError(
                    f"Operand '{v.name}' accesses slot {key[0]} after it was "
                    f"consumed by a destructive view operation; the physical "
                    f"qubit is no longer available."
                )

            if isinstance(existing, ArrayValue):
                # Slot is held by a view.  The operand is OK only if
                # it traces through the same view (via its parent_array
                # chain).  Otherwise it's a direct access colliding
                # with the view.
                if not self._is_element_of_view(v, existing):
                    raise SliceLinearityViolationError(
                        self._format_view_vs_direct(existing, v, op, key)
                    )
                # Same view — legitimate view-internal access, no state
                # mutation needed (the view still owns the slot).
                continue

    def _process_result_releases(self, op: Operation, state: State) -> None:
        """Release slice-view ownership / install consumed markers on operand-consuming ops.

        Walks each ``ArrayValue`` operand of a consuming operation and
        releases the corresponding state entries.  The semantics split
        by operand shape:

        * **View operand** (``slice_of`` chain is non-empty): release
          only the slots the view covers.  Leaves other slots of the
          same root untouched so a sibling view or direct access on a
          non-overlapping slot remains valid.  For destructive ops
          (``MeasureOperation``, ``MeasureVectorOperation``,
          ``CastOperation``), install ``_CONSUMED_SLOT`` markers on
          the covered slots so any later access — direct or via
          another view — is rejected as use-after-destroy.

        * **Root operand** (no chain): release every entry whose
          owner root matches, matching the prior whole-array consume
          semantics.  Destructive ops additionally install consumed
          markers for every concrete-index slot on that root so
          subsequent operations can't accidentally re-touch the
          collapsed state.

        ``ExpvalOp`` retires the SSA handle but does not physically
        destroy the qubits, so it does not install consumed markers
        (matching the frontend's ``_DESTRUCTIVE_CONSUME_OPS``).

        Args:
            op: The operation potentially consuming operand arrays.
            state: Mutable borrow tracker.
        """
        if not isinstance(
            op,
            (
                MeasureOperation,
                MeasureVectorOperation,
                ExpvalOp,
                CastOperation,
            ),
        ):
            return

        is_destructive = isinstance(op, _IR_DESTRUCTIVE_OPS)

        for v in op.operands:
            if not isinstance(v, ArrayValue):
                continue

            if v.slice_of is not None:
                # View operand: release / consume only its covered slots.
                covered = self._collect_view_coverage(v)
                if covered is None:
                    # Symbolic bounds — release anything owned by this
                    # specific view uuid (no partial consumption
                    # semantic possible without concrete coverage).
                    self._release_by_owner_uuid(v.uuid, state)
                    continue
                for slot in covered:
                    key = (f"const:{slot}",)
                    if is_destructive:
                        state[key] = _CONSUMED_SLOT
                    elif key in state:
                        del state[key]
                continue

            # Root operand: full release.
            root_uuid = v.uuid
            to_remove: list[BorrowKey] = []
            to_consume: list[BorrowKey] = []
            for k, owner in state.items():
                owner_root = self._owner_root_uuid(owner)
                if owner_root is not None and owner_root == root_uuid:
                    to_remove.append(k)
            for k in to_remove:
                del state[k]
            if is_destructive:
                # Install consumed markers on every concrete slot of
                # this root that we can enumerate from the operand's
                # shape.  Leaves the root free for post-block checks
                # to complete.
                shape = v.shape
                if shape:
                    length = self._const_int(shape[0])
                    if length is not None:
                        for idx in range(length):
                            to_consume.append((f"const:{idx}",))
                for k in to_consume:
                    state[k] = _CONSUMED_SLOT

    def _release_by_owner_uuid(self, uuid: str, state: State) -> None:
        """Drop every state entry whose direct owner has the given uuid.

        Used for symbolic-bound view release when we can't enumerate
        coverage.  Only entries whose owner ArrayValue uuid matches
        are removed; unrelated root-space consumed markers stay put.

        Args:
            uuid: The ArrayValue uuid whose ownership should be cleared.
            state: Mutable borrow tracker.
        """
        to_remove = [
            k
            for k, owner in state.items()
            if isinstance(owner, ArrayValue) and owner.uuid == uuid
        ]
        for k in to_remove:
            del state[k]

    # ------------------------------------------------------------------
    # Slice registration
    # ------------------------------------------------------------------

    def _register_slice_bulk_borrow_if_new(
        self, av: ValueBase | None, state: State
    ) -> None:
        """Register covered slots for a sliced ArrayValue if not yet seen.

        Walks ``av`` (and any chain through ``slice_of``) to find slice
        views whose bounds are now concrete; for each such view, bulk
        registers its covered root-parent slots in ``state`` keyed
        under ``(f"const:<idx>",)`` with the view as the owner.  A view
        already present in ``state`` is idempotently skipped.

        Args:
            av: Candidate sliced ``ArrayValue`` (or ``None``).  Silently
                returns on ``None`` or non-sliced values.
            state: Mutable borrow tracker.

        Raises:
            SliceLinearityViolationError: If any covered slot is already
                held by a different owner.
        """
        if av is None or not isinstance(av, ArrayValue):
            return
        if av.slice_of is None:
            return

        # Concrete length required.
        length = self._const_int(av.shape[0]) if av.shape else None
        if length is None:
            return  # Symbolic — skip.

        # Walk ``slice_of`` chain composing the affine map.  Start with
        # the identity map ``(start=0, step=1)`` and apply each parent
        # frame ``(p_start, p_step)`` on the way up:
        #   idx_parent = p_start + p_step * idx_current
        # so ``(start, step)`` accumulate to ``(p_start + p_step * start,
        # p_step * step)``.  Mirrors :meth:`_collect_view_coverage`;
        # starting from ``av.slice_start`` directly would double-apply
        # the innermost frame.
        root = av
        root_start = 0
        root_step = 1
        while root.slice_of is not None:
            parent_start = self._const_int(root.slice_start)
            parent_step = self._const_int(root.slice_step)
            if parent_start is None or parent_step is None:
                return
            root_start = parent_start + parent_step * root_start
            root_step = parent_step * root_step
            root = root.slice_of

        # Enumerate the exact covered-slot set for this view so that
        # "sequential same-range" slicing (``a = q[0::2]; loop(a); b =
        # q[0::2]``) can evict the outgoing view as a whole rather than
        # triggering a partial-overlap false positive.  Two ArrayValues
        # whose covered sets are identical represent the same logical
        # view — the earlier one's lifetime implicitly ended when the
        # later one was constructed.
        new_covered: set[int] = {
            root_start + root_step * j for j in range(length)
        }

        # Collect distinct existing views whose slots intersect this
        # view's coverage, then decide per-view whether to replace
        # (identical coverage = OK), raise (genuine overlap), or treat
        # as idempotent (same uuid).
        touched_views: dict[str, ArrayValue] = {}
        touched_views_coverage: dict[str, set[int]] = {}
        for j in range(length):
            idx = root_start + root_step * j
            key = (f"const:{idx}",)
            existing = state.get(key)
            if existing is None:
                continue
            if isinstance(existing, _ConsumedSlotMarker):
                # Attempting to slice into an already-destroyed slot.
                raise SliceLinearityViolationError(
                    f"Slice view '{av.name}' covers slot {idx} on "
                    f"'{root.name}', but that physical qubit was destroyed "
                    f"by a prior destructive view operation (measure / cast)."
                )
            if isinstance(existing, ArrayValue):
                if existing.uuid == av.uuid:
                    continue  # idempotent re-registration on same uuid
                touched_views.setdefault(existing.uuid, existing)
                touched_views_coverage.setdefault(existing.uuid, set()).add(idx)
            else:
                # Direct element borrow collides with the new view.
                raise SliceLinearityViolationError(
                    self._format_view_registration_conflict(
                        existing, av, idx, root
                    )
                )

        for other_uuid, other_view in touched_views.items():
            other_full = self._collect_view_coverage(other_view)
            if other_full is None:
                # Symbolic / unresolvable bounds on the earlier view —
                # we can't prove coverage equality, so be conservative.
                raise SliceLinearityViolationError(
                    self._format_view_registration_conflict(
                        other_view, av, next(iter(touched_views_coverage[other_uuid])), root
                    )
                )
            if other_full == new_covered:
                # Same logical view — release all of the old view's
                # registrations before installing the new one below.
                for slot in other_full:
                    old_key = (f"const:{slot}",)
                    if state.get(old_key) is other_view:
                        del state[old_key]
            elif new_covered.issubset(other_full):
                # Nested slice: the new view is a strict subset of an
                # existing outer view (``inner = q[0::2][1:3]``).  The
                # outer view's life ends the moment a nested slice is
                # taken from it — the frontend's ``VectorView._wrap``
                # already drains the outer in this case.  Mirror the
                # same semantic here: release every slot of the outer
                # view (including those outside the new view's coverage,
                # since the outer handle is no longer usable) before
                # registering the inner.
                for slot in other_full:
                    old_key = (f"const:{slot}",)
                    if state.get(old_key) is other_view:
                        del state[old_key]
            elif other_uuid not in self._used_view_uuids:
                # Partial overlap with an outer view that was never
                # used (no element or whole-view operand reference
                # since its registration).  The frontend's
                # ``VectorView._wrap`` drains such a view regardless
                # of coverage shape; match that here so the two
                # checkers agree on what programs are legal.  A view
                # that HAS been used gets the strict overlap error
                # below — the permissive drain only applies to
                # sliced-but-never-touched placeholders.
                for slot in other_full:
                    old_key = (f"const:{slot}",)
                    if state.get(old_key) is other_view:
                        del state[old_key]
            else:
                raise SliceLinearityViolationError(
                    self._format_view_registration_conflict(
                        other_view,
                        av,
                        next(iter(touched_views_coverage[other_uuid])),
                        root,
                    )
                )

        for idx in new_covered:
            key = (f"const:{idx}",)
            state[key] = av

    def _collect_view_coverage(self, view: "ArrayValue") -> set[int] | None:
        """Return the full covered-slot set of ``view`` in root coordinates.

        Walks the ``slice_of`` chain composing the affine map, then
        enumerates ``start + step * j`` for ``j`` in ``range(length)``.
        Returns ``None`` if any component along the chain is non-constant
        (e.g. a symbolic ``UInt`` that escaped constant folding) so
        callers can fall back to a conservative overlap response.

        Args:
            view: A sliced ``ArrayValue``.

        Returns:
            The set of physical root-parent slot indices covered by the
            view, or ``None`` when the coverage cannot be determined.
        """
        if view.slice_of is None:
            return None
        if not view.shape:
            return None
        length = self._const_int(view.shape[0])
        if length is None:
            return None

        root = view
        root_start = 0
        root_step = 1
        while root.slice_of is not None:
            parent_start = self._const_int(root.slice_start)
            parent_step = self._const_int(root.slice_step)
            if parent_start is None or parent_step is None:
                return None
            root_start = parent_start + parent_step * root_start
            root_step = parent_step * root_step
            root = root.slice_of
        return {root_start + root_step * j for j in range(length)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_qubit_key(self, v: Value) -> BorrowKey | None:
        """Resolve an element Value to its root-parent ``(idx,)`` key.

        Walks the ``parent_array.slice_of`` chain if present, composing
        the affine maps to translate the view-local index to the root
        parent's coordinate space.  Returns ``None`` if any component
        is symbolic / non-constant — those are outside the scope of
        this pass's guarantees.

        Args:
            v: Qubit element Value with ``parent_array`` and
                ``element_indices``.

        Returns:
            ``(f"const:<idx>",)`` when the physical index is known, else
            ``None``.
        """
        if v.parent_array is None or not v.element_indices:
            return None

        idx_value = v.element_indices[0]
        idx = self._const_int(idx_value)
        if idx is None:
            return None

        parent = v.parent_array
        while parent.slice_of is not None:
            start = self._const_int(parent.slice_start)
            step = self._const_int(parent.slice_step)
            if start is None or step is None:
                return None
            idx = start + step * idx
            parent = parent.slice_of

        return (f"const:{idx}",)

    @staticmethod
    def _const_int(v: ValueBase | None) -> int | None:
        """Extract a concrete integer from ``v`` if it is a constant.

        Args:
            v: A Value, or ``None``.

        Returns:
            The integer value if ``v`` holds a constant, else ``None``.
        """
        if v is None:
            return None
        if isinstance(v, Value) and v.is_constant():
            const = v.get_const()
            if const is not None:
                return int(const)
        return None

    @staticmethod
    def _owner_root_uuid(owner: Owner) -> str | None:
        """Return the root parent uuid for a borrow owner.

        Args:
            owner: Either a direct-borrow Value or a slice-view ArrayValue.

        Returns:
            Root parent uuid; for direct Values this is ``parent_array.uuid``
            at the root of any slice chain.  For view owners, the same
            after walking ``slice_of`` to the root.
        """
        if isinstance(owner, ArrayValue):
            root = owner
            while root.slice_of is not None:
                root = root.slice_of
            return root.uuid
        if isinstance(owner, Value) and owner.parent_array is not None:
            root = owner.parent_array
            while root.slice_of is not None:
                root = root.slice_of
            return root.uuid
        return None

    @staticmethod
    def _is_element_of_view(v: Value, view: ArrayValue) -> bool:
        """Check whether element Value ``v`` is derived from ``view``.

        Comparison is by ``uuid`` rather than object identity because
        substitutors and ``dataclasses.replace`` in earlier passes may
        have produced a rebuilt ArrayValue with identical fields but a
        different Python object; they still represent the same logical
        view.

        Args:
            v: Element Value candidate.
            view: A sliced ``ArrayValue`` (owner in the state dict).

        Returns:
            ``True`` iff ``v.parent_array`` is ``view`` or reaches
            ``view`` via the ``slice_of`` chain (matched by uuid).
        """
        parent = v.parent_array
        while parent is not None:
            if parent.uuid == view.uuid:
                return True
            parent = parent.slice_of
        return False

    # ------------------------------------------------------------------
    # Error message formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_outstanding(state: State) -> str:
        """Render outstanding borrows for the block-end error message.

        Args:
            state: Final borrow tracker.

        Returns:
            A newline-separated list of "array[idx] — held by <owner>".
        """
        lines: list[str] = []
        for key, owner in state.items():
            idx_str = key[0].split(":", 1)[1]
            if isinstance(owner, ArrayValue):
                root = owner
                while root.slice_of is not None:
                    root = root.slice_of
                lines.append(
                    f"  {root.name}[{idx_str}] — held by slice view {owner.name}"
                )
            else:
                holder = owner if isinstance(owner, Value) else None
                name = holder.name if holder is not None else "<direct>"
                lines.append(f"  — element borrow {name} (slot {idx_str})")
        return "\n".join(lines)

    @staticmethod
    def _format_view_vs_direct(
        view: ArrayValue,
        direct: Value,
        op: Operation,
        key: BorrowKey,
    ) -> str:
        """Format the direct-access-over-view conflict message.

        Args:
            view: Slice view currently holding the slot.
            direct: Element Value attempting direct access.
            op: Operation where the conflict manifested.
            key: The colliding ``(f"const:<idx>",)`` key.

        Returns:
            Human-readable error body.
        """
        idx_str = key[0].split(":", 1)[1]
        root = view
        while root.slice_of is not None:
            root = root.slice_of
        return (
            f"Direct access to '{root.name}[{idx_str}]' via "
            f"'{type(op).__name__}' while the slot is held by slice view "
            f"'{view.name}'.\n"
            f"Access it through the view, or let the view finish before "
            f"touching the parent directly."
        )

    @staticmethod
    def _format_direct_conflict(
        existing: Value,
        new_v: Value,
        op: Operation,
        key: BorrowKey,
    ) -> str:
        """Format a direct-borrow double-access message.

        Args:
            existing: The Value currently holding the slot.
            new_v: The Value attempting a fresh borrow.
            op: Operation where the conflict manifested.
            key: The colliding ``(f"const:<idx>",)`` key.

        Returns:
            Human-readable error body.
        """
        idx_str = key[0].split(":", 1)[1]
        return (
            f"Element '{new_v.name}' and '{existing.name}' alias on slot "
            f"{idx_str} at '{type(op).__name__}'.\n"
            f"Return the previously borrowed element (via "
            f"``array[idx] = ...``) before borrowing it again."
        )

    @staticmethod
    def _format_view_registration_conflict(
        existing: Owner,
        new_view: ArrayValue,
        idx: int,
        root: ArrayValue,
    ) -> str:
        """Format a view-registration overlap message.

        Args:
            existing: Current owner of the slot (view or direct).
            new_view: The new sliced ArrayValue being registered.
            idx: Colliding root-parent index.
            root: The root parent ArrayValue.

        Returns:
            Human-readable error body.
        """
        owner_desc = (
            f"slice view '{existing.name}'"
            if isinstance(existing, ArrayValue)
            else f"element borrow '{existing.name}'"
            if isinstance(existing, Value)
            else "<unknown>"
        )
        return (
            f"Slice view '{new_view.name}' covers '{root.name}[{idx}]' "
            f"which is already held by {owner_desc}.\n"
            f"Overlapping slice views (or direct access while a view is "
            f"live) are not supported."
        )
