"""Post-fold block-wide slice-view borrow check.

Mirrors the slice-view subset of the frontend's
``ArrayBase._borrowed_indices`` borrow-tracker semantics on the IR
after :class:`ConstantFoldingPass` has resolved slice bounds to
concrete values.  The pass exists to catch bugs the trace-time
checker cannot see on its own — slices whose bounds were ``UInt``
at trace time so their covered slot set was not enumerable until
bindings were applied:

1. Aliasing between a (now-concrete) slice view and another live
   view of the same root parent (raised as
   :class:`QubitBorrowConflictError` from
   :meth:`_register_slice_bulk_borrow_if_new`).
2. A view whose newly-concrete coverage hits a slot that was
   consumed by a destructive view operation earlier in the block
   (raised as :class:`QubitConsumedError` while registering the view
   in :meth:`_register_slice_bulk_borrow_if_new` or inspecting its
   operands in :meth:`_process_operand_borrows`).
3. A slice ownership release, drain, or refresh that crosses a
   control-flow boundary the current state merge cannot represent
   safely (raised as :class:`ValidationError` from the corresponding
   snapshot guard).

Slice views are otherwise treated as **affine** at the kernel
boundary: a view that goes out of scope without being slice-
assigned back to the parent is no longer flagged here.  This
mirrors how element borrows on locally-allocated registers behave
— natural ancilla / scratch-register patterns such as
Deutsch-Jozsa's ``ancilla = qs[n]`` and Simon's
``qs2 = qs[n:2*n]`` (used by the oracle, then discarded
unmeasured) compile cleanly.  The genuine hazards stay covered:

* ``measure(parent)`` (or any other ``parent.consume()`` site) while
  a view is live raises ``UnreturnedBorrowError`` from the
  frontend's ``ArrayBase.consume`` / ``validate_all_returned``.
* Returning the parent with an outstanding borrow raises
  ``UnreturnedBorrowError`` from
  ``qamomile.circuit.frontend.func_to_block._validate_returned_arrays``.
* Direct ``q[i]`` access on a slot a view currently owns is caught
  at the frontend's element-access path / this pass's
  :meth:`_process_operand_borrows` for symbolic-bound views.
* Overlapping live views and use-after-destroy are caught at
  registration time (see the two error sites above).

The creation of a direct element borrow (``q[i]``) is intentionally
**not** observed here because element access emits no IR operation.
Later uses of that element do appear as operation operands, however,
so :meth:`_process_operand_borrows` can reject a use that collides
with a live slice view.  The frontend trace-time validator
(``qamomile.circuit.frontend.func_to_block._validate_returned_arrays`` and
``ArrayBase.validate_all_returned``) remains the source of truth for
unreturned direct-element borrows that have no observable operand use.

State shape: a single dict keyed by a 2-tuple
``(root_logical_id, slot_descriptor)`` where
``root_logical_id`` is the ``logical_id`` of the **root** parent
ArrayValue (after walking the ``slice_of`` chain) and
``slot_descriptor`` is ``f"const:<idx>"`` for slots covered by
compile-time-constant slices.  Symbolic-bound slices (those whose
bounds remain non-constant post-fold) are not enumerated into
per-qubit slots; the pass records only an exact ``"sym:<descriptor>"``
entry so a later SSA-version refresh of the same symbolic slice
descriptor can update the owner without claiming any new concrete
range.

Namespacing by root logical_id is required so that borrow /
consumed state on one register (``a``) does not block access to
the identically-named slot on another register (``b``); without
it, ``measure(a[1::2])`` would incorrectly mark ``b[1]`` and
``b[3]`` as destroyed.

Values are either an :class:`ArrayValue` (the slice view that owns
the slot) or a ``_ConsumedSlotMarker`` (a slot already destroyed by
a prior destructive view consume).  ``isinstance`` on the value
dispatches the two violation messages.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import (
    Operation,
    ReleaseSliceViewOperation,
    ReturnOperation,
    SliceArrayOperation,
)
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    CompOp,
)
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    HasNestedOps,
    IfOperation,
)
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import (
    MeasureOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.types.primitives import UIntType
from qamomile.circuit.ir.value import ArrayValue, Value, ValueBase
from qamomile.circuit.transpiler.errors import (
    QubitBorrowConflictError,
    QubitConsumedError,
    ValidationError,
)

from . import Pass

BorrowKey = tuple[str, str]
ConstBoundToken: TypeAlias = tuple[Literal["const"], int]
ValueBoundToken: TypeAlias = tuple[Literal["value"], str, int]
MinBoundToken: TypeAlias = tuple[Literal["min"], object, object]
BoundToken: TypeAlias = ConstBoundToken | ValueBoundToken | MinBoundToken


def _root_of(av: ArrayValue) -> ArrayValue:
    """Walk ``av``'s ``slice_of`` chain to its root parent ArrayValue.

    For a non-sliced array returns ``av`` unchanged; for a (possibly
    nested) view returns the underlying root parent that physically
    backs the slots — the only register identity that's stable across
    affine remappings.

    Args:
        av: A possibly-sliced ArrayValue.

    Returns:
        The root ArrayValue at the bottom of the ``slice_of`` chain.
    """
    cur = av
    while cur.slice_of is not None:
        cur = cur.slice_of
    return cur


def _const_key(root_av: ArrayValue, idx: int) -> BorrowKey:
    """Build a borrow key for a compile-time-known slot, namespaced by root.

    The key is ``(root_av.logical_id, f"const:{idx}")``.  ``logical_id``
    (rather than ``uuid``) is used because passes such as
    ``partial_eval`` may version-bump an ``ArrayValue`` between the
    point where a destructive view consume installs the marker and the
    point where a later operand checks it; ``logical_id`` is preserved
    across version bumps, while ``uuid`` is regenerated.  Using the
    logical id keeps the marker addressable for the same logical
    register throughout the block, while still partitioning state per
    register so register ``a``'s consumed slots never alias register
    ``b``'s slots.

    Args:
        root_av: The root parent ArrayValue (after ``_root_of`` walk).
        idx: Concrete root-space slot index.

    Returns:
        Two-element tuple suitable as a key into the ``State`` dict.
    """
    return (root_av.logical_id, f"const:{idx}")


def _slot_descriptor(key: BorrowKey) -> str:
    """Return the human-readable slot descriptor portion of a borrow key.

    For ``(root_uuid, "const:3")`` returns ``"const:3"``.  Used by the
    error formatters which only care about the slot, not the namespace.
    """
    return key[1]


class _ConsumedSlotMarker:
    """Sentinel marking a physical qubit slot destroyed by a prior destructive op.

    Installed by :meth:`SliceBorrowCheckPass._process_result_releases`
    when a destructive operation (``MeasureOperation``,
    ``MeasureVectorOperation``, ``CastOperation``, ``ExpvalOp``)
    consumes a view's covered slots.  Subsequent operand access to
    the same slot is treated as a violation, matching the frontend's
    ``_CONSUMED_SLOT`` sentinel semantics.  Only one instance ever
    exists; identity is the test.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return "<IRConsumedSlot>"


_CONSUMED_SLOT: "_ConsumedSlotMarker" = _ConsumedSlotMarker()


_IR_DESTRUCTIVE_OPS: tuple[type, ...] = (
    MeasureOperation,
    MeasureVectorOperation,
    CastOperation,
    ExpvalOp,
)


Owner = ArrayValue | _ConsumedSlotMarker
"""Borrow-state owner discriminant.

State values are always either a slice-view ``ArrayValue`` (the live
owner of one or more parent slots) or ``_ConsumedSlotMarker`` (a
destroyed slot).  Direct element borrows (``q[i]``) are intentionally
not tracked by this IR-level pass — those are SSA-versioned and have
no IR semantics worth modelling here; the frontend trace-time
validator covers them.
"""
State = dict[BorrowKey, Owner]


class _SnapshotKind:
    """Classify control-flow snapshots for safe view-refresh decisions."""

    IF_BRANCH = "if_branch"
    FOR_STATIC_NONZERO = "for_static_nonzero"
    UNSAFE_CONTROL_BODY = "unsafe_control_body"


_SnapshotFrame = tuple[str, State]


class SliceBorrowCheckPass(Pass[Block, Block]):
    """Post-fold linearity checker for sliced views and borrow state.

    Runs after :class:`ConstantFoldingPass` (so slice bounds are
    concrete where possible) and before segmentation / emit.  Walks
    the operations of the root block in order, maintaining a borrow
    state map modelled on the frontend's
    :attr:`ArrayBase._borrowed_indices` — a single ``dict`` whose
    values are slice-view ``ArrayValue`` owners or the
    ``_ConsumedSlotMarker`` sentinel.  Creating a direct element borrow
    (``q[i]``) emits no IR operation, but later operand uses remain visible
    to this pass; the frontend validator handles an unreturned borrow with
    no observable operand use.

    The pass does **not** flag a leftover slice view at block end —
    slice views are affine at the kernel boundary, mirroring how
    element borrows behave on a locally-allocated register (the
    frontend's
    ``qamomile.circuit.frontend.func_to_block._validate_returned_arrays``
    covers the genuine leak: returning the parent with a live borrow).
    Anything that actually clashes with a live view (direct slot access,
    destructive parent consume, overlapping views, use-after-destroy) is
    rejected at the eager check points listed in the module docstring.
    """

    def __init__(self) -> None:
        """Initialize per-run mutable state to safe defaults."""
        # ``_outer_snapshot_stack`` is a stack of outer ``state``
        # snapshots taken when entering a control-flow body in
        # ``_walk_nested``.  While walking inside the body, every
        # state-mutating helper (slice registration's drain paths,
        # ReleaseSliceViewOperation handling) consults the snapshot
        # to forbid removing entries that the enclosing block had
        # already registered — those changes cannot be propagated
        # out of the body by the current merge logic, so they're
        # rejected up-front.  Empty stack means we're at the top
        # level.  Each frame also records whether the body is a
        # statically non-zero ``For`` loop, which is the only context
        # where same-slice SSA-version refresh of an outer owner can
        # safely be accepted without propagating a deletion.
        self._outer_snapshot_stack: list[_SnapshotFrame] = []
        # Small expression cache for symbolic slice-bound comparisons.
        # ``SliceArrayOperation`` normalizes user slices through BinOps
        # such as ``min(k, n)``, ``x - 0``, ``x + 0`` and ``x // 1``.
        # The borrow checker does not evaluate symbolic arithmetic in
        # general, but it needs structurally equal normalized bounds to
        # compare equal when proving adjacent symbolic intervals
        # disjoint.  Keys are strict ``(logical_id, version)`` value
        # identities, values are canonical ``BoundToken`` tuples.
        self._bound_exprs: dict[tuple[str, int], BoundToken] = {}

    @property
    def name(self) -> str:
        """Return the pass identifier for tracing/logging.

        Returns:
            The short name ``"slice_borrow_check"``.
        """
        return "slice_borrow_check"

    def run(self, input: Block) -> Block:
        """Run the borrow tracker over ``input``.

        Args:
            input (Block): Block to check. Expected to be in AFFINE or
                HIERARCHICAL kind — post-fold but pre-segmentation.

        Returns:
            Block: The same block unchanged after successful validation.

        Raises:
            ValidationError: If called on an unexpected block kind or slice
                ownership cannot be propagated safely across a control-flow
                boundary.
            QubitBorrowConflictError: If a slice view and a direct access
                collide or two live views may overlap.
            QubitConsumedError: If an operand accesses a destroyed slot or a
                stale slice version attempts to replace its live successor.
        """
        if input.kind not in (BlockKind.AFFINE, BlockKind.HIERARCHICAL):
            raise ValidationError(
                f"SliceBorrowCheckPass expects AFFINE or HIERARCHICAL "
                f"block, got {input.kind}",
            )

        self._bound_exprs.clear()
        state: State = {}
        self._walk(input.operations, state)

        # NB: ``state`` may still contain live slice-view ownership
        # entries when the block returns — i.e. the user took a slice
        # view, used it (broadcast / sub-kernel / pauli_evolve /
        # element loop) and let it go out of scope without slice-
        # assigning it back.  We intentionally do NOT flag this here:
        # slice views are treated as *affine* (at most once, discard
        # allowed) at the kernel boundary, mirroring how element
        # borrows behave on locally-allocated registers — see
        # Deutsch-Jozsa ancilla and Simon's scratch-register patterns
        # in ``tests/circuit/test_slice_pattern_correctness.py``.
        #
        # The actual hazards (live view + direct slot access, live
        # view + parent consume, view overlap, use-after-destroy,
        # returning the parent with a live borrow) are all caught
        # eagerly elsewhere — by the trace-time bulk-borrow tracker
        # for concrete-bound views, by ``_register_slice_bulk_borrow_if_new``'s
        # conflict path for symbolic-bound views, by
        # ``_process_operand_borrows`` for use-after-destroy, and by
        # the frontend's ``_validate_returned_arrays`` for returned
        # quantum arrays.  Leaking a view past kernel end without
        # touching the parent is the only situation this method used
        # to flag, and the lower-friction "affine" treatment makes
        # natural ancilla / scratch patterns compile cleanly.
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

            if isinstance(op, BinOp):
                # Pure classical arithmetic has no borrow semantics, but
                # slice construction uses a few BinOps to normalize bounds.
                # Record a tiny canonical expression for later symbolic
                # interval-disjointness checks.
                self._record_bound_expr(op)
                continue

            if isinstance(op, CompOp):
                # Pure classical comparisons; no borrow semantics.
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

            if isinstance(op, ReleaseSliceViewOperation):
                # ``Vector.__setitem__`` slice assignment emits this op
                # immediately after consuming a view handle on the
                # frontend.  Mirror that release in the IR borrow
                # tracker: drop every entry whose owner is the view
                # ArrayValue we are releasing.  See ``_handle_release``
                # for the outer-snapshot guard used when this op
                # appears inside a control-flow body.
                if op.operands:
                    self._handle_release(op.operands[0], state)
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
            # Push outer snapshot for the duration of branch walks.
            # Any release / drain inside either branch that targets
            # an entry from the snapshot is rejected by the helpers
            # — the current merge policy cannot propagate entry
            # deletions out of the body, so cross-body release is
            # unsupported in this revision.
            self._outer_snapshot_stack.append((_SnapshotKind.IF_BRANCH, dict(state)))
            try:
                true_state = dict(state)
                self._walk(op.true_operations, true_state)
                false_state = dict(state)
                self._walk(op.false_operations, false_state)
            finally:
                self._outer_snapshot_stack.pop()
            # Conservative union with a consumption-priority rule.  If
            # either branch destroyed a slot the post-If state must
            # treat it as destroyed; otherwise a downstream operand
            # would be wrongly accepted just because the *other* branch
            # left the slot intact.  For non-destroyed entries we keep
            # whichever side recorded ownership (subsequent ops get
            # the stricter view-held treatment).
            merged: State = {}
            for k, v in true_state.items():
                merged[k] = v
            for k, v in false_state.items():
                existing = merged.get(k)
                if isinstance(v, _ConsumedSlotMarker):
                    merged[k] = _CONSUMED_SLOT  # consumption wins
                elif existing is None:
                    merged[k] = v
            state.clear()
            state.update(merged)
            return

        # For/ForItems/While: simulate one iteration.  We use a copy so
        # loop-internal borrows cancel at iteration end, then merge the
        # body's final state back into the outer state with a
        # conservative policy:
        #
        # 1. ``_ConsumedSlotMarker`` is sticky.  Once a slot is destroyed
        #    inside any iteration it stays destroyed for all subsequent
        #    operations, regardless of what the outer state had — this
        #    is the semantic that prevents the symbolic-view destructive
        #    consume inside a loop from being lost across the boundary.
        # 2. New view ownership recorded inside the loop body (where
        #    the outer state had no entry for the slot) is carried
        #    forward — the view is still alive after the loop.
        # 3. Pre-existing entries that the body did not consume are left
        #    untouched.  The body might not have run on a given input
        #    (zero-trip loop), so dropping or rewriting outer state on
        #    the basis of one simulated iteration would be unsound.
        # 4. A body that writes a *different* ArrayValue owner over a
        #    pre-existing entry is intentionally ignored.  The slot
        #    cannot legitimately switch owners without an explicit
        #    consume, and the body simulation is one iteration, so any
        #    transient ownership swap is dropped to avoid corrupting
        #    the outer state's view of the live slice graph.
        trip_count = (
            self._static_for_trip_count(op) if isinstance(op, ForOperation) else None
        )
        # A same-slice refresh of an outer owner is only safe when this
        # body is guaranteed to execute at least once.  In that case the
        # simulated body state is a faithful update of the live owner for
        # the body-local checks, while the outer merge still keeps the
        # enclosing state's original owner.  Skippable or branch-like
        # bodies must stay conservative because rewriting their snapshot
        # owner would make the post-body state depend on an execution path
        # that this pass does not model.
        if isinstance(op, ForOperation) and trip_count is not None and trip_count > 0:
            snapshot_kind = _SnapshotKind.FOR_STATIC_NONZERO
        else:
            snapshot_kind = _SnapshotKind.UNSAFE_CONTROL_BODY

        self._outer_snapshot_stack.append((snapshot_kind, dict(state)))
        try:
            for body in op.nested_op_lists():
                body_state = dict(state)
                self._walk(body, body_state)
                for k, v in body_state.items():
                    if isinstance(v, _ConsumedSlotMarker):
                        state[k] = _CONSUMED_SLOT
                    elif k not in state and isinstance(v, ArrayValue):
                        state[k] = v
        finally:
            self._outer_snapshot_stack.pop()

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
            op (Operation): Operation whose operands are being inspected.
            state (State): Mutable borrow tracker containing slice views and
                consumed-slot markers.

        Raises:
            QubitBorrowConflictError: If an operand's resolved slot is owned
                by a view that does not contain the operand.
            QubitConsumedError: If an operand's resolved slot has already
                been destroyed by a destructive view operation.
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
                # Slice-view operand: bulk-borrow / aliasing is handled
                # by the element-operand loop below via parent_array.
                continue
            # Root array whole operand: check each concrete slot.
            shape = v.shape
            if shape:
                length = self._const_int(shape[0])
                if length is not None:
                    for idx in range(length):
                        key = _const_key(v, idx)
                        if isinstance(state.get(key), _ConsumedSlotMarker):
                            raise QubitConsumedError(
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

            # Register any sliced ArrayValue seen via parent_array.
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
                raise QubitConsumedError(
                    f"Operand '{v.name}' accesses slot "
                    f"{_slot_descriptor(key)} after it was "
                    f"consumed by a destructive view operation; the physical "
                    f"qubit is no longer available."
                )

            if isinstance(existing, ArrayValue):
                # Slot is held by a view.  The operand is OK only if
                # it traces through the same view (via its parent_array
                # chain).  Otherwise it's a direct access colliding
                # with the view.
                if not self._is_element_of_view(v, existing):
                    raise QubitBorrowConflictError(
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
          ``CastOperation``, ``ExpvalOp``), install ``_CONSUMED_SLOT``
          markers on the covered slots so any later access — direct
          or via another view — is rejected as use-after-destroy.

        * **Root operand** (no chain): release every entry whose
          owner root matches, matching the prior whole-array consume
          semantics.  Destructive ops additionally install consumed
          markers for every concrete-index slot on that root so
          subsequent operations can't accidentally re-touch the
          collapsed state.

        ``ExpvalOp`` is treated as destructive in lock-step with the
        frontend's ``_DESTRUCTIVE_CONSUME_OPS`` set — both layers
        agree that estimating an observable consumes its qubits.

        Args:
            op: The operation potentially consuming operand arrays.
            state: Mutable borrow tracker.
        """
        if not isinstance(op, _IR_DESTRUCTIVE_OPS):
            return

        is_destructive = True

        for v in op.operands:
            if not isinstance(v, ArrayValue):
                continue

            if v.slice_of is not None:
                # View operand: release / consume only its covered slots.
                covered = self._collect_view_coverage(v)
                if covered is None:
                    # Symbolic bounds — release anything owned by this
                    # specific view's logical_id (no partial
                    # consumption semantic possible without concrete
                    # coverage).
                    self._release_by_owner_logical_id(v.logical_id, state)
                    continue
                view_root = _root_of(v)
                for slot in covered:
                    key = _const_key(view_root, slot)
                    if is_destructive:
                        state[key] = _CONSUMED_SLOT
                    elif key in state:
                        del state[key]
                continue

            # Root operand: full release.  Match owners by logical_id
            # to align with ``_owner_root_uuid``'s logical-id semantics
            # (see its docstring on why uuid is unsafe across version
            # bumps).
            root_logical_id = v.logical_id
            to_remove: list[BorrowKey] = []
            to_consume: list[BorrowKey] = []
            for k, owner in state.items():
                owner_root = self._owner_root_uuid(owner)
                if owner_root is not None and owner_root == root_logical_id:
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
                            to_consume.append(_const_key(v, idx))
                for k in to_consume:
                    state[k] = _CONSUMED_SLOT

    def _release_by_owner_logical_id(self, logical_id: str, state: State) -> None:
        """Drop every state entry whose direct owner has the given logical_id.

        Used for slice-view release.  Matching is by ``logical_id``
        (stable across ``next_version`` bumps) rather than ``uuid`` so
        that frontend ops which produce a fresh SSA-versioned
        ``ArrayValue`` for the same logical view (e.g. ``pauli_evolve``
        on a slice, or a sub-kernel call whose result preserves the
        slice chain) still release the original registration.  Only
        entries whose owner is an ``ArrayValue`` are considered;
        unrelated root-space consumed markers stay put.

        Args:
            logical_id: The ArrayValue logical_id whose ownership
                should be cleared.
            state: Mutable borrow tracker.
        """
        to_remove = [
            k
            for k, owner in state.items()
            if isinstance(owner, ArrayValue) and owner.logical_id == logical_id
        ]
        for k in to_remove:
            del state[k]

    def _release_keys_for_view(
        self, view_value: ArrayValue, state: State
    ) -> list[BorrowKey]:
        """Find state entries that a slice-assignment release should drop.

        The normal release key is the view's own ``logical_id``.  A
        nested full-slice handoff may leave the IR borrow state pointing
        at the skipped outer view while the release operand is the inner
        view that fully replaced it.  When concrete coverage proves that
        both owners cover exactly the same root slots, release the stale
        equivalent owner as well.

        Args:
            view_value (ArrayValue): Sliced view operand from
                ``ReleaseSliceViewOperation``.
            state (State): Current borrow-state map.

        Returns:
            list[BorrowKey]: State keys that should be removed.
        """
        target_logical_id = view_value.logical_id
        release_keys = {
            k
            for k, owner in state.items()
            if isinstance(owner, ArrayValue) and owner.logical_id == target_logical_id
        }

        target_coverage = self._collect_view_coverage(view_value)
        if target_coverage is None:
            return list(release_keys)

        target_root = _root_of(view_value)
        equivalent_owner_ids: set[str] = set()
        for slot in target_coverage:
            key = _const_key(target_root, slot)
            owner = state.get(key)
            if not isinstance(owner, ArrayValue):
                continue
            if owner.logical_id == target_logical_id:
                continue
            if _root_of(owner).logical_id != target_root.logical_id:
                continue
            if self._collect_view_coverage(owner) == target_coverage:
                equivalent_owner_ids.add(owner.uuid)

        if not equivalent_owner_ids:
            return list(release_keys)

        for k, owner in state.items():
            if isinstance(owner, ArrayValue) and owner.uuid in equivalent_owner_ids:
                release_keys.add(k)
        return list(release_keys)

    def _handle_release(self, view_value: ValueBase, state: State) -> None:
        """Apply a slice-assignment release to ``state``.

        Mirrors the frontend's ``VectorView.consume(operation_name=
        'slice assignment')``: every state entry whose owner has the
        same logical_id as ``view_value`` is removed.  Matching is by
        ``logical_id`` (not ``uuid``) so a release on a fresh
        SSA-versioned view (the result of ``pauli_evolve`` on a slice,
        a sub-kernel call returning the slice, etc.) still releases the
        original registration.

        When called from inside a control-flow body, refuses to drop
        any entry that any active enclosing snapshot already owned.
        The current loop / branch merge cannot propagate entry deletions
        out of the body, so such cross-body releases would leave the
        outer state inconsistent — they are rejected with
        ``ValidationError`` for predictability.

        Args:
            view_value (ValueBase): Sliced value whose borrow is
                being released.  Must be an ``ArrayValue`` (the
                ``ReleaseSliceViewOperation`` op invariant guarantees
                this).
            state (State): Mutable borrow tracker for the current scope.

        Raises:
            ValidationError: When invoked inside a
                control-flow body and any entry being removed was
                recorded in the outer snapshot (cross-body release).
        """
        if not isinstance(view_value, ArrayValue):
            return

        release_keys = self._release_keys_for_view(view_value, state)
        offenders = [
            k
            for _, snapshot in self._outer_snapshot_stack
            for k in release_keys
            if k in snapshot
        ]
        if offenders:
            raise ValidationError(
                f"Slice assignment inside a control-flow body "
                f"attempted to release view '{view_value.name}', "
                f"which was registered by the enclosing block.  "
                f"Cross-body release is not supported in this "
                f"revision because the loop / branch merge cannot "
                f"propagate the deletion to the outer state.  "
                f"Move the view construction *inside* the body, "
                f"or perform the slice assignment outside the "
                f"control-flow region."
            )

        for key in release_keys:
            state.pop(key, None)

    def _guard_drain_against_outer_snapshot(
        self,
        drained_view: ArrayValue,
        new_view: ArrayValue,
        root: ArrayValue,
        reason: str,
    ) -> None:
        """Reject implicit drain of an outer-registered view from inside a body.

        ``_register_slice_bulk_borrow_if_new`` has two drain paths that
        delete an existing view's state entries before registering a new
        view that overlaps it: same-coverage replacement and
        nested-slice drain of the outer view.  Each is an entry-deletion,
        so the same propagation problem that motivates
        ``_handle_release``'s outer-snapshot guard applies — if we drop
        an entry that the enclosing block had registered, the post-body
        merge cannot carry that deletion outward and the outer state
        ends up holding a stale owner.  Forbid the pattern up-front
        instead.

        Args:
            drained_view (ArrayValue): View whose state entries the caller is
                about to delete.
            new_view (ArrayValue): View being registered, used in the error
                message to point users at the construction site.
            root (ArrayValue): Shared root parent used for the
                ``<root>[idx]`` reference in the error body).
            reason (str): Short descriptor of which drain path triggered the
                guard (``"same-coverage replacement"`` /
                ``"nested-slice drain of outer view"``), surfaced in
                the error message so the user can see *why* the IR
                tried to drop the outer view.

        Raises:
            ValidationError: If any active outer snapshot
                owns ``drained_view`` or an equivalent prior version of
                the same sliced view.
        """
        if not self._outer_snapshot_stack:
            return
        offenders = [
            k
            for _, snapshot in self._outer_snapshot_stack
            for k, owner in snapshot.items()
            if self._snapshot_owner_matches_view(owner, drained_view)
        ]
        if not offenders:
            return
        raise ValidationError(
            f"Slice construction inside a control-flow body would "
            f"implicitly drain view '{drained_view.name}' (registered "
            f"on '{root.name}' by the enclosing block) to make room "
            f"for '{new_view.name}' ({reason}).  Cross-body drain is "
            f"not supported in this revision because the loop / branch "
            f"merge cannot propagate the deletion to the outer state.  "
            f"Move the outer view's construction *inside* the body, "
            f"or rework the slicing so the body does not overlap the "
            f"outer view."
        )

    def _can_body_local_same_coverage_handoff(
        self,
        owner: ArrayValue,
        keys: list[BorrowKey],
    ) -> bool:
        """Check whether a same-coverage handoff may rewrite body state.

        Distinct-lineage same-coverage replacement is normally treated
        as an implicit drain and is rejected inside control-flow bodies.
        A statically non-zero ``For`` body is the narrow exception:
        the body definitely executes at least once, and the rewrite is
        used only while checking that simulated body.  The enclosing
        state remains conservative after merge.

        Args:
            owner (ArrayValue): Existing owner that the body-local
                handoff would replace.
            keys (list[BorrowKey]): Exact state entries covered by the
                owner and candidate.

        Returns:
            bool: ``True`` when every matching active snapshot is a
                statically non-zero ``For`` body.
        """
        matching_kinds: list[str] = []
        for kind, snapshot in self._outer_snapshot_stack:
            for key in keys:
                snapshot_owner = snapshot.get(key)
                if isinstance(snapshot_owner, ArrayValue) and (
                    snapshot_owner.uuid == owner.uuid
                ):
                    matching_kinds.append(kind)
        return bool(matching_kinds) and all(
            kind == _SnapshotKind.FOR_STATIC_NONZERO for kind in matching_kinds
        )

    # ------------------------------------------------------------------
    # Slice registration
    # ------------------------------------------------------------------

    def _register_slice_bulk_borrow_if_new(
        self, av: ValueBase | None, state: State
    ) -> None:
        """Register covered slots for a sliced ArrayValue if not yet seen.

        Walks ``av`` (and any chain through ``slice_of``) to find slice
        views whose bounds are now concrete; for each concrete view,
        bulk-registers its covered root-parent slots in ``state`` keyed
        under ``(root_logical_id, f"const:<idx>")`` with the view as the
        owner.  Symbolic views use one exact descriptor entry and only
        support same-lineage forward SSA-version refreshes.  A view
        already present in ``state`` is idempotently skipped.

        Args:
            av (ValueBase | None): Candidate sliced value. Silently
                returns on ``None`` or non-sliced values.
            state (State): Mutable borrow tracker.

        Raises:
            QubitBorrowConflictError: If any covered slot is already held by
                another live owner.
            QubitConsumedError: If any covered slot was destroyed by a
                previous destructive view operation or a stale slice version
                attempts to replace its live successor.
            ValidationError: If registering the view would mutate ownership
                across an unsupported control-flow boundary.
        """
        if av is None or not isinstance(av, ArrayValue):
            return
        if av.slice_of is None:
            return

        new_covered = self._collect_view_coverage(av)
        if new_covered is None:
            self._register_symbolic_slice_if_new(av, state)
            return

        root = _root_of(av)

        self._guard_against_symbolic_root_conflicts(av, root, state)

        # Enumerate the exact covered-slot set for this view so that
        # "sequential same-range" slicing (``a = q[0::2]; loop(a); b =
        # q[0::2]``) can evict the outgoing view as a whole rather than
        # triggering a partial-overlap false positive.  Two ArrayValues
        # whose covered sets are identical represent the same logical
        # view — the earlier one's lifetime implicitly ended when the
        # later one was constructed.

        # Collect distinct existing views whose slots intersect this
        # view's coverage, then decide per-view whether to replace
        # (identical coverage = OK), raise (genuine overlap), or treat
        # as idempotent (same uuid).
        touched_views: dict[str, ArrayValue] = {}
        touched_views_coverage: dict[str, set[int]] = {}
        for idx in new_covered:
            key = _const_key(root, idx)
            existing = state.get(key)
            if existing is None:
                continue
            if isinstance(existing, _ConsumedSlotMarker):
                # Attempting to slice into an already-destroyed slot.
                raise QubitConsumedError(
                    f"Slice view '{av.name}' covers slot {idx} on "
                    f"'{root.name}', but that physical qubit was destroyed "
                    f"by a prior destructive view operation (measure / cast)."
                )
            if isinstance(existing, ArrayValue):
                if existing.uuid == av.uuid:
                    continue  # idempotent re-registration on same uuid
                touched_views.setdefault(existing.uuid, existing)
                touched_views_coverage.setdefault(existing.uuid, set()).add(idx)

        for other_uuid, other_view in touched_views.items():
            other_full = self._collect_view_coverage(other_view)
            if other_full is None:
                # Symbolic / unresolvable bounds on the earlier view —
                # we can't prove coverage equality, so be conservative.
                raise QubitBorrowConflictError(
                    self._format_view_registration_conflict(
                        other_view,
                        av,
                        min(touched_views_coverage[other_uuid]),
                        root,
                    )
                )
            if other_full == new_covered:
                # Same coverage has two very different meanings:
                # a legitimate SSA refresh of the same sliced view, or
                # a distinct live view that happens to cover the same
                # slots.  Accept only the former early; the latter keeps
                # using the existing drain/replacement behavior below.
                if self._is_forward_same_slice_view_refresh(av, other_view):
                    refresh_keys = [_const_key(root, slot) for slot in other_full]
                    self._guard_refresh_against_unsafe_snapshots(
                        other_view, av, refresh_keys
                    )
                    # Refreshing updates the owner identity in place.
                    # It is not a drain: no slot is released or claimed by
                    # a different lineage, so later view-internal element
                    # operands can still pass the existing UUID-chain check.
                    for old_key in refresh_keys:
                        existing = state.get(old_key)
                        if (
                            isinstance(existing, ArrayValue)
                            and existing.uuid == other_view.uuid
                        ):
                            state[old_key] = av
                    continue

                # If the lineage is the same but the version did not move
                # forward, falling through to the old same-coverage drain
                # would let a stale predecessor replace the current owner.
                # Reject that explicitly before considering distinct-lineage
                # compatibility paths such as nested full-slice handoff.
                if self._same_slice_lineage_and_known_extent(av, other_view):
                    self._raise_stale_same_slice_replacement(av, other_view, root)

                # Same coverage with a distinct lineage: release all of
                # the old view's registrations before installing the new
                # one below.
                #
                # Same-lineage forward version bumps refreshed in place
                # above, and stale same-lineage replacements were
                # rejected before reaching this branch.  What remains is
                # a provenance ambiguity: a nested full-slice handoff and
                # a sibling view whose symbolic bounds folded to the same
                # slots can both appear as distinct ``logical_id``s with
                # identical coverage because ``VectorView._nested_slice``
                # flattens nested ``slice_of`` chains to the root.
                refresh_keys = [_const_key(root, slot) for slot in other_full]
                if self._can_body_local_same_coverage_handoff(other_view, refresh_keys):
                    # In a statically non-zero loop body, a concrete
                    # full-slice handoff is path-insensitive for the body
                    # walk: the inner view fully replaces the outer for
                    # every simulated iteration.  Keep the rewrite local
                    # to ``body_state``; the outer merge stays
                    # conservative, and a later top-level release can
                    # clear the equivalent stale owner by coverage.
                    for old_key in refresh_keys:
                        existing = state.get(old_key)
                        if (
                            isinstance(existing, ArrayValue)
                            and existing.uuid == other_view.uuid
                        ):
                            state[old_key] = av
                    continue
                self._guard_drain_against_outer_snapshot(
                    other_view, av, root, "same-coverage replacement"
                )
                for slot in other_full:
                    old_key = _const_key(root, slot)
                    existing = state.get(old_key)
                    if (
                        isinstance(existing, ArrayValue)
                        and existing.uuid == other_view.uuid
                    ):
                        del state[old_key]
            elif new_covered.issubset(other_full):
                # Nested slice: the new view is a strict subset of an
                # existing outer view (``inner = q[0::2][1:3]``).  Only
                # the overlap slots are handed from the outer view to
                # the inner — the outer keeps its non-overlap slots so
                # it can later be slice-assigned back to its own
                # parent (the frontend's ``_nested_slice`` does the
                # same partial hand-off).  Slots outside ``new_covered``
                # remain registered against the outer.
                #
                # NB: same provenance caveat as the ``==`` branch above
                # — symbolic-bound sibling overlap can take this path
                # too because IR has no provenance to distinguish it
                # from genuine nested handoff.
                self._guard_drain_against_outer_snapshot(
                    other_view, av, root, "nested-slice handoff"
                )
                for slot in new_covered:
                    old_key = _const_key(root, slot)
                    existing = state.get(old_key)
                    if (
                        isinstance(existing, ArrayValue)
                        and existing.uuid == other_view.uuid
                    ):
                        del state[old_key]
            elif other_full.issubset(new_covered):
                # The existing view is a (nested) inner of ``av``.  This
                # path fires when ``av`` is re-touched via element /
                # operand reference after a nested inner was sliced off
                # it (e.g. ``a = q[1:9]; b = a[1:5]; qmc.h(a[0])``).
                # The parent-child relationship is legitimate; leave
                # the inner's registration intact.  The final
                # registration loop below preserves the inner's entries
                # by only writing slots that are currently unowned or
                # owned by ``av`` itself.
                continue
            else:
                # Strict-return: partial overlap with an existing live
                # view is a linearity violation.  Callers must
                # explicitly return / consume the outer view before
                # constructing an overlapping inner view.  Matches the
                # frontend's strict ``VectorView._wrap`` overlap check.
                raise QubitBorrowConflictError(
                    self._format_view_registration_conflict(
                        other_view,
                        av,
                        next(iter(touched_views_coverage[other_uuid])),
                        root,
                    )
                )

        # Preserve nested inner-view registrations: when ``av`` is being
        # re-touched and some of its slots are currently held by a
        # nested inner (parent-child relationship validated above), do
        # NOT overwrite those entries.  Only claim slots that are
        # unowned or already owned by ``av`` itself.
        for idx in new_covered:
            key = _const_key(root, idx)
            existing = state.get(key)
            if existing is None or (
                isinstance(existing, ArrayValue) and existing.uuid == av.uuid
            ):
                state[key] = av

    def _register_symbolic_slice_if_new(self, av: ArrayValue, state: State) -> None:
        """Register or refresh a symbolic slice descriptor exactly.

        Symbolic coverage cannot be enumerated into per-qubit keys, so
        this path deliberately supports only the narrow case where the
        exact same sliced ``ArrayValue`` lineage advances to a newer SSA
        version.  It does not prove arithmetic equivalence between
        different expressions and does not claim additional concrete
        slots.

        Args:
            av (ArrayValue): Sliced view with non-concrete coverage.
            state (State): Mutable borrow tracker.

        Raises:
            QubitBorrowConflictError: If a symbolic descriptor is already
                owned by a different live slice lineage.
            QubitConsumedError: If the symbolic view may cover a slot already
                destroyed by a destructive view operation or a stale version
                attempts to replace the current owner.
            ValidationError: If refreshing the descriptor would rewrite an
                unsafe control-flow snapshot.
        """
        root = _root_of(av)
        key = self._symbolic_view_key(root, av)
        if key is None:
            return

        self._guard_against_concrete_root_conflicts(av, root, state)

        existing = state.get(key)
        # A symbolic descriptor is one coarse owner for an unknown slot
        # set.  Different descriptors on the same root might overlap, so
        # reject them unless a narrow interval check proves they are
        # disjoint.  The exact current descriptor is handled below,
        # where only a forward same-descriptor refresh is accepted.
        for other_key, owner in state.items():
            if (
                other_key == key
                or other_key[0] != root.logical_id
                or not other_key[1].startswith("sym:")
            ):
                continue
            if isinstance(owner, ArrayValue) and owner.uuid != av.uuid:
                if self._symbolic_slices_definitely_disjoint(av, owner):
                    continue
                raise QubitBorrowConflictError(
                    f"Symbolic slice view '{av.name}' on '{root.name}' "
                    f"may overlap live symbolic view '{owner.name}'.  "
                    f"Only exact forward refresh of the same symbolic "
                    f"descriptor or provably disjoint symbolic intervals "
                    f"are supported for symbolic slices."
                )
        if existing is None:
            state[key] = av
            return
        if isinstance(existing, _ConsumedSlotMarker):
            raise QubitConsumedError(
                f"Symbolic slice view '{av.name}' on '{root.name}' reuses a "
                f"slice descriptor whose physical qubits were destroyed by "
                f"a prior destructive view operation."
            )
        if existing.uuid == av.uuid:
            return
        if self._is_forward_same_slice_view_refresh(av, existing):
            # This is the symbolic counterpart of the concrete refresh
            # path above: keep the single descriptor key and replace only
            # the owner version.  No per-qubit slots are inferred.
            self._guard_refresh_against_unsafe_snapshots(existing, av, [key])
            state[key] = av
            return
        if self._same_slice_lineage(av, existing) and av.version <= existing.version:
            self._raise_stale_same_slice_replacement(av, existing, root)
        raise QubitBorrowConflictError(
            f"Symbolic slice view '{av.name}' has the same descriptor as "
            f"live view '{existing.name}' on '{root.name}', but it is not "
            f"a forward SSA-version refresh of that view.  Return or "
            f"consume the existing view before constructing another view "
            f"with the same symbolic descriptor."
        )

    def _symbolic_view_key(
        self, root: ArrayValue, view: ArrayValue
    ) -> BorrowKey | None:
        """Build a borrow key for an exact symbolic slice descriptor.

        Args:
            root (ArrayValue): Root parent of ``view``.
            view (ArrayValue): Sliced view whose descriptor cannot be
                enumerated into concrete slots.

        Returns:
            BorrowKey | None: A root-namespaced symbolic descriptor key,
                or ``None`` when the descriptor shape is incomplete.
        """
        descriptor = self._slice_descriptor_signature(view)
        if descriptor is None:
            return None
        return (root.logical_id, f"sym:{descriptor!r}")

    def _slice_descriptor_signature(
        self, view: ArrayValue
    ) -> (
        tuple[
            tuple[
                tuple[str, int],
                tuple[str, int],
                tuple[tuple[str, int], ...],
            ],
            ...,
        ]
        | None
    ):
        """Return an exact symbolic descriptor signature for ``view``.

        The signature intentionally ignores value names and UUIDs.  It
        records each descriptor operand as ``(logical_id, version)`` so
        cloned representations of the same SSA value compare equal, but
        a later SSA version of the same symbolic variable does not.

        Args:
            view (ArrayValue): Sliced view to describe.

        Returns:
            tuple[tuple[tuple[str, int], tuple[str, int],
            tuple[tuple[str, int], ...]], ...] | None: Per-slice-frame
                descriptor signature, or ``None`` when any required
                descriptor value is absent.
        """
        parts: list[
            tuple[
                tuple[str, int],
                tuple[str, int],
                tuple[tuple[str, int], ...],
            ]
        ] = []
        cur = view
        while cur.slice_of is not None:
            if cur.slice_start is None or cur.slice_step is None:
                return None
            start_id = self._value_signature(cur.slice_start)
            step_id = self._value_signature(cur.slice_step)
            if start_id is None or step_id is None:
                return None
            shape_ids: list[tuple[str, int]] = []
            for dim in cur.shape:
                dim_id = self._value_signature(dim)
                if dim_id is None:
                    return None
                shape_ids.append(dim_id)
            parts.append((start_id, step_id, tuple(shape_ids)))
            cur = cur.slice_of
        return tuple(parts)

    @staticmethod
    def _value_signature(value: ValueBase | None) -> tuple[str, int] | None:
        """Return a strict SSA identity signature for a descriptor value.

        Args:
            value (ValueBase | None): Descriptor value to identify.

        Returns:
            tuple[str, int] | None: ``(logical_id, version)`` for scalar
                ``Value`` instances, else ``None``.
        """
        if not isinstance(value, Value):
            return None
        return (value.logical_id, value.version)

    def _record_bound_expr(self, op: BinOp) -> None:
        """Cache a canonical expression token for a bound-normalizing BinOp.

        The cache is deliberately tiny: it captures identities emitted
        while lowering slices (``x + 0``, ``x - 0``, ``x // 1``) and
        canonical ``min`` expressions, plus constant folding for those
        same operators.  Unsupported arithmetic is left uncached, which
        makes later symbolic disjointness checks return ``False`` and
        preserves the conservative overlap behavior.

        Args:
            op (BinOp): Arithmetic operation encountered while walking
                the block.
        """
        output_sig = self._value_signature(op.output)
        if output_sig is None:
            return
        lhs = self._bound_token(op.lhs)
        rhs = self._bound_token(op.rhs)
        if lhs is None or rhs is None:
            return

        token: BoundToken | None = None
        if op.kind == BinOpKind.ADD:
            token = self._canonical_add(lhs, rhs)
        elif op.kind == BinOpKind.SUB:
            token = self._canonical_sub(lhs, rhs)
        elif op.kind == BinOpKind.FLOORDIV:
            token = self._canonical_floordiv(lhs, rhs)
        elif op.kind == BinOpKind.MIN:
            token = self._canonical_min(lhs, rhs, op.output)

        if token is not None:
            self._bound_exprs[output_sig] = token

    @staticmethod
    def _is_const_token(token: BoundToken, value: int) -> bool:
        """Check whether ``token`` is a specific integer constant.

        Args:
            token (BoundToken): Token to inspect.
            value (int): Expected integer value.

        Returns:
            bool: ``True`` when ``token`` is exactly ``("const", value)``.
        """
        return len(token) == 2 and token[0] == "const" and token[1] == value

    @staticmethod
    def _const_token_value(token: BoundToken) -> int | None:
        """Return the integer value represented by a constant token.

        Args:
            token (BoundToken): Token to inspect.

        Returns:
            int | None: Constant integer payload, or ``None`` for a
                non-constant token.
        """
        if len(token) == 2 and token[0] == "const":
            return int(token[1])
        return None

    def _canonical_add(self, lhs: BoundToken, rhs: BoundToken) -> BoundToken | None:
        """Canonicalize a small ``lhs + rhs`` expression.

        Args:
            lhs (BoundToken): Left operand token.
            rhs (BoundToken): Right operand token.

        Returns:
            BoundToken | None: Simplified token, or ``None`` when this
                expression is outside the supported slice-bound subset.
        """
        lhs_const = self._const_token_value(lhs)
        rhs_const = self._const_token_value(rhs)
        if lhs_const is not None and rhs_const is not None:
            return ("const", lhs_const + rhs_const)
        if self._is_const_token(lhs, 0):
            return rhs
        if self._is_const_token(rhs, 0):
            return lhs
        return None

    def _canonical_sub(self, lhs: BoundToken, rhs: BoundToken) -> BoundToken | None:
        """Canonicalize a small ``lhs - rhs`` expression.

        Args:
            lhs (BoundToken): Left operand token.
            rhs (BoundToken): Right operand token.

        Returns:
            BoundToken | None: Simplified token, or ``None`` when this
                expression is outside the supported slice-bound subset.
        """
        lhs_const = self._const_token_value(lhs)
        rhs_const = self._const_token_value(rhs)
        if lhs_const is not None and rhs_const is not None:
            return ("const", lhs_const - rhs_const)
        if self._is_const_token(rhs, 0):
            return lhs
        return None

    def _canonical_floordiv(
        self, lhs: BoundToken, rhs: BoundToken
    ) -> BoundToken | None:
        """Canonicalize a small ``lhs // rhs`` expression.

        Args:
            lhs (BoundToken): Left operand token.
            rhs (BoundToken): Right operand token.

        Returns:
            BoundToken | None: Simplified token, or ``None`` when this
                expression is outside the supported slice-bound subset.
        """
        lhs_const = self._const_token_value(lhs)
        rhs_const = self._const_token_value(rhs)
        if lhs_const is not None and rhs_const not in (None, 0):
            return ("const", lhs_const // rhs_const)
        if self._is_const_token(rhs, 1):
            return lhs
        return None

    def _canonical_min(
        self, lhs: BoundToken, rhs: BoundToken, output: Value
    ) -> BoundToken | None:
        """Canonicalize a small ``min(lhs, rhs)`` expression.

        Args:
            lhs (BoundToken): Left operand token.
            rhs (BoundToken): Right operand token.
            output (Value): Operation result, used to ensure the
                ``min(0, x) -> 0`` simplification applies only to
                non-negative ``UInt`` bounds.

        Returns:
            BoundToken | None: Simplified token, or ``None`` when this
                expression is outside the supported slice-bound subset.
        """
        lhs_const = self._const_token_value(lhs)
        rhs_const = self._const_token_value(rhs)
        if lhs_const is not None and rhs_const is not None:
            return ("const", min(lhs_const, rhs_const))
        if isinstance(output.type, UIntType) and (
            self._is_const_token(lhs, 0) or self._is_const_token(rhs, 0)
        ):
            return ("const", 0)
        ordered = sorted((lhs, rhs), key=repr)
        return ("min", ordered[0], ordered[1])

    def _symbolic_slices_definitely_disjoint(
        self, left: ArrayValue, right: ArrayValue
    ) -> bool:
        """Check a narrow symbolic proof that two sliced views are disjoint.

        This helper intentionally handles only direct, unit-stride
        root-space intervals.  That covers common partition patterns
        such as ``q[:k]`` and ``q[k:n]`` without turning the borrow
        checker into a symbolic inequality solver.  Any unsupported
        shape returns ``False`` so callers keep the existing
        conservative overlap response.

        Args:
            left (ArrayValue): First symbolic sliced view.
            right (ArrayValue): Second symbolic sliced view.

        Returns:
            bool: ``True`` when the views have the same root and one
                interval's end is provably less than or equal to the
                other's start.
        """
        if _root_of(left).logical_id != _root_of(right).logical_id:
            return False
        left_bounds = self._direct_unit_stride_interval_bounds(left)
        right_bounds = self._direct_unit_stride_interval_bounds(right)
        if left_bounds is None or right_bounds is None:
            return False
        left_start, left_end = left_bounds
        right_start, right_end = right_bounds
        if left_end is not None and self._bound_leq(left_end, right_start):
            return True
        return right_end is not None and self._bound_leq(right_end, left_start)

    def _direct_unit_stride_interval_bounds(
        self, view: ArrayValue
    ) -> tuple[BoundToken, BoundToken | None] | None:
        """Return root-space half-open bounds for a direct unit-stride view.

        The returned pair represents ``[start, end)``.  ``end`` is
        ``None`` when the shape is known to be symbolic but cannot be
        written as a bound token this helper can compare.  Only direct
        slices of the root with ``slice_step == 1`` are accepted; nested
        symbolic affine maps and strided symbolic ranges stay
        conservative.

        Args:
            view (ArrayValue): Sliced view to inspect.

        Returns:
            tuple[BoundToken, BoundToken | None] | None: ``(start, end)``
                bound tokens, or ``None`` when this helper cannot model
                the view.
        """
        if view.slice_of is None or view.slice_of.slice_of is not None:
            return None
        if not view.shape:
            return None
        if self._const_int(view.slice_step) != 1:
            return None
        start = self._bound_token(view.slice_start)
        length = self._bound_token(view.shape[0])
        if start is None or length is None:
            return None
        return start, self._end_bound_token(start, length)

    def _bound_token(self, value: ValueBase | None) -> BoundToken | None:
        """Represent a slice bound by either a constant or SSA identity.

        Args:
            value (ValueBase | None): Bound value to encode.

        Returns:
            BoundToken | None: ``("const", n)`` for integer constants,
                a cached canonical expression token for known BinOp
                results, ``("value", logical_id, version)`` for raw
                symbolic scalar Values, or ``None`` for unsupported
                values.
        """
        const = self._const_int(value)
        if const is not None:
            return ("const", const)
        sig = self._value_signature(value)
        if sig is None:
            return None
        cached = self._bound_exprs.get(sig)
        if cached is not None:
            return cached
        logical_id, version = sig
        return ("value", logical_id, version)

    @staticmethod
    def _end_bound_token(
        start: BoundToken,
        length: BoundToken,
    ) -> BoundToken | None:
        """Return ``start + length`` when it stays in the token language.

        Args:
            start (BoundToken): Interval start token.
            length (BoundToken): Interval length token.

        Returns:
            BoundToken | None: Comparable end token, or ``None`` when
                computing the end would require symbolic arithmetic
                beyond identity / constants.
        """
        if start[0] == "const" and length[0] == "const":
            return ("const", int(start[1]) + int(length[1]))
        if start == ("const", 0):
            return length
        if length == ("const", 0):
            return start
        return None

    @staticmethod
    def _bound_leq(
        left: BoundToken,
        right: BoundToken,
    ) -> bool:
        """Compare two bound tokens when equality or constants prove ``<=``.

        Args:
            left (BoundToken): Left bound token.
            right (BoundToken): Right bound token.

        Returns:
            bool: ``True`` when ``left <= right`` follows from exact
                token equality or from concrete integer comparison.
        """
        if left == right:
            return True
        if left[0] == "const" and right[0] == "const":
            return int(left[1]) <= int(right[1])
        return False

    def _guard_against_symbolic_root_conflicts(
        self,
        av: ArrayValue,
        root: ArrayValue,
        state: State,
    ) -> None:
        """Reject concrete registration under live symbolic root ownership.

        Args:
            av (ArrayValue): Concrete-coverage sliced view being
                registered.
            root (ArrayValue): Root parent of ``av``.
            state (State): Mutable borrow tracker.

        Raises:
            QubitBorrowConflictError: If the same root has a live
                symbolic descriptor owner whose overlap with ``av``
                cannot be disproved.
        """
        for key, owner in state.items():
            if key[0] != root.logical_id or not key[1].startswith("sym:"):
                continue
            if isinstance(owner, ArrayValue) and owner.uuid != av.uuid:
                raise QubitBorrowConflictError(
                    f"Slice view '{av.name}' has concrete coverage on "
                    f"'{root.name}', but live symbolic view "
                    f"'{owner.name}' on the same root may overlap it.  "
                    f"Return or consume the symbolic view before "
                    f"constructing another view on the same root."
                )

    def _guard_against_concrete_root_conflicts(
        self,
        av: ArrayValue,
        root: ArrayValue,
        state: State,
    ) -> None:
        """Reject symbolic registration under live concrete root ownership.

        Args:
            av (ArrayValue): Symbolic-coverage sliced view being
                registered.
            root (ArrayValue): Root parent of ``av``.
            state (State): Mutable borrow tracker.

        Raises:
            QubitBorrowConflictError: If the same root has a live concrete
                view whose overlap with ``av`` cannot be disproved.
            QubitConsumedError: If the symbolic view may cover a slot already
                destroyed by a destructive view operation.
        """
        for key, owner in state.items():
            if key[0] != root.logical_id or key[1].startswith("sym:"):
                continue
            if isinstance(owner, _ConsumedSlotMarker):
                raise QubitConsumedError(
                    f"Slice view '{av.name}' has symbolic coverage on "
                    f"'{root.name}', but concrete slot "
                    f"'{_slot_descriptor(key)}' was destroyed by a prior "
                    f"destructive view operation."
                )
            if isinstance(owner, ArrayValue) and owner.uuid != av.uuid:
                raise QubitBorrowConflictError(
                    f"Slice view '{av.name}' has symbolic coverage on "
                    f"'{root.name}', but live concrete view "
                    f"'{owner.name}' on the same root may overlap it.  "
                    f"Return or consume the concrete view before "
                    f"constructing a symbolic view on the same root."
                )

    def _same_slice_lineage(self, left: ArrayValue, right: ArrayValue) -> bool:
        """Check whether two views are versions of the same slice lineage.

        Args:
            left (ArrayValue): First sliced view.
            right (ArrayValue): Second sliced view.

        Returns:
            bool: ``True`` when both values are sliced views with the
                same view ``logical_id`` and the same root
                ``logical_id``.
        """
        return (
            left.slice_of is not None
            and right.slice_of is not None
            and left.logical_id == right.logical_id
            and _root_of(left).logical_id == _root_of(right).logical_id
        )

    def _same_symbolic_slice_descriptor(
        self, left: ArrayValue, right: ArrayValue
    ) -> bool:
        """Check exact symbolic descriptor identity for two slice views.

        Args:
            left (ArrayValue): First sliced view.
            right (ArrayValue): Second sliced view.

        Returns:
            bool: ``True`` when both views have identical slice-chain
                descriptor signatures based on SSA value identity.
        """
        left_sig = self._slice_descriptor_signature(left)
        right_sig = self._slice_descriptor_signature(right)
        return left_sig is not None and left_sig == right_sig

    def _same_slice_lineage_and_known_extent(
        self, left: ArrayValue, right: ArrayValue
    ) -> bool:
        """Check whether same-lineage views cover the same known extent.

        Concrete coverage equality is authoritative.  Symbolic
        descriptor equality is considered only when neither side can be
        enumerated.

        Args:
            left (ArrayValue): First sliced view.
            right (ArrayValue): Second sliced view.

        Returns:
            bool: ``True`` when the views have the same lineage and
                either equal concrete coverage or identical symbolic
                descriptors.
        """
        if not self._same_slice_lineage(left, right):
            return False
        left_coverage = self._collect_view_coverage(left)
        right_coverage = self._collect_view_coverage(right)
        if left_coverage is not None and right_coverage is not None:
            return left_coverage == right_coverage
        if left_coverage is None and right_coverage is None:
            return self._same_symbolic_slice_descriptor(left, right)
        return False

    def _same_slice_lineage_and_possible_extent(
        self, left: ArrayValue, right: ArrayValue
    ) -> bool:
        """Conservatively match same-lineage views for snapshot guards.

        Args:
            left (ArrayValue): First sliced view.
            right (ArrayValue): Second sliced view.

        Returns:
            bool: ``True`` when exact extent equality is known or one
                side's extent is symbolic and therefore cannot disprove
                equality.
        """
        if self._same_slice_lineage_and_known_extent(left, right):
            return True
        if not self._same_slice_lineage(left, right):
            return False
        return (
            self._collect_view_coverage(left) is None
            or self._collect_view_coverage(right) is None
        )

    def _is_forward_same_slice_view_refresh(
        self, candidate: ArrayValue, owner: ArrayValue
    ) -> bool:
        """Check whether ``candidate`` is a safe forward refresh of ``owner``.

        Args:
            candidate (ArrayValue): New sliced view being registered.
            owner (ArrayValue): Existing sliced view currently owning
                the same state entries.

        Returns:
            bool: ``True`` only for same-lineage views with identical
                concrete coverage or exact symbolic descriptors where
                ``candidate.version`` is strictly newer than
                ``owner.version``.
        """
        return (
            candidate.version > owner.version
            and self._same_slice_lineage_and_known_extent(candidate, owner)
        )

    def _snapshot_owner_matches_view(
        self, owner: Owner | None, view: ArrayValue
    ) -> bool:
        """Check whether a snapshot owner refers to ``view`` or its lineage.

        Args:
            owner (Owner | None): Owner recorded in a control-flow
                snapshot, or ``None`` when the key was absent from that
                snapshot.
            view (ArrayValue): View being refreshed or drained.

        Returns:
            bool: ``True`` when ``owner`` is the same UUID as ``view``,
                or when ``view`` is the same-lineage current/later SSA
                version of ``owner``.
        """
        return isinstance(owner, ArrayValue) and (
            owner.uuid == view.uuid
            or (
                view.version >= owner.version
                and self._same_slice_lineage_and_possible_extent(view, owner)
            )
        )

    def _guard_refresh_against_unsafe_snapshots(
        self,
        owner: ArrayValue,
        candidate: ArrayValue,
        keys: list[BorrowKey],
    ) -> None:
        """Reject refreshes that would rewrite unsafe snapshot owners.

        Args:
            owner (ArrayValue): Existing owner being replaced in
                ``state``.
            candidate (ArrayValue): Newer same-lineage view.
            keys (list[BorrowKey]): Exact state entries that the
                refresh would rewrite.

        Raises:
            ValidationError: If any matching active snapshot
                frame is not a statically non-zero ``For`` body.
        """
        matching_kinds: list[str] = []
        # Scan every active frame, not just the innermost one.  A static
        # inner ``For`` nested under an ``if`` or runtime loop must not
        # launder an outer unsafe snapshot into an allowed refresh.
        for kind, snapshot in self._outer_snapshot_stack:
            for key in keys:
                snapshot_owner = snapshot.get(key)
                if self._snapshot_owner_matches_view(
                    snapshot_owner, owner
                ) or self._snapshot_owner_matches_view(snapshot_owner, candidate):
                    matching_kinds.append(kind)
        if not matching_kinds:
            return
        # Only the all-static-nonzero case is path-insensitive enough to
        # allow this body-local owner rewrite.  One unsafe match is enough
        # to reject because the rewritten owner could otherwise be visible
        # only on some runtime paths.
        if all(kind == _SnapshotKind.FOR_STATIC_NONZERO for kind in matching_kinds):
            return
        root = _root_of(candidate)
        raise ValidationError(
            f"Slice construction inside a control-flow body would "
            f"refresh view '{owner.name}' (registered on '{root.name}' "
            f"by an enclosing block) to '{candidate.name}', but that "
            f"enclosing body may be skipped or branch-dependent.  "
            f"Only statically non-zero for-loop bodies may refresh an "
            f"outer slice view's SSA version without returning it."
        )

    @staticmethod
    def _raise_stale_same_slice_replacement(
        candidate: ArrayValue,
        owner: ArrayValue,
        root: ArrayValue,
    ) -> None:
        """Raise for stale or equal-version same-lineage replacement.

        Args:
            candidate (ArrayValue): New sliced view being registered.
            owner (ArrayValue): Existing same-lineage owner.
            root (ArrayValue): Shared root parent ArrayValue.

        Raises:
            QubitConsumedError: Always raised because ``candidate`` is an
                already-consumed predecessor of the live ``owner`` version.
        """
        raise QubitConsumedError(
            f"Slice view '{candidate.name}' on '{root.name}' attempts "
            f"to replace an equal or newer registered version of the "
            f"same slice view '{owner.name}'.  Only forward SSA-version "
            f"refreshes of the same slice descriptor are allowed."
        )

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
        """Resolve an element Value to its root-namespaced borrow key.

        Walks the ``parent_array.slice_of`` chain if present, composing
        the affine maps to translate the view-local index to the root
        parent's coordinate space.  Returns ``None`` if any component
        is symbolic / non-constant — those are outside the scope of
        this pass's guarantees.

        The returned key is ``(root.uuid, f"const:<idx>")`` so borrow
        state is partitioned per root array; without this, a slot index
        ``i`` on register ``a`` would alias the same index on register
        ``b`` and one register's destructive view consume could spuriously
        block the other.

        Args:
            v: Qubit element Value with ``parent_array`` and
                ``element_indices``.

        Returns:
            ``(root_uuid, f"const:<idx>")`` when the physical index is
            known, else ``None``.
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

        return _const_key(parent, idx)

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

    def _static_for_trip_count(self, op: ForOperation) -> int | None:
        """Return a statically known ``ForOperation`` trip count.

        Args:
            op (ForOperation): The loop whose ``start``, ``stop``, and
                ``step`` operands should be inspected.

        Returns:
            int | None: The Python ``range`` trip count when all three
                loop bounds are concrete and the step is non-zero,
                ``1`` as a positive sentinel when the exact count is
                too large for ``len(range(...))``, otherwise ``None``.
        """
        if len(op.operands) < 3:
            return None
        start = self._const_int(op.operands[0])
        stop = self._const_int(op.operands[1])
        step = self._const_int(op.operands[2])
        if start is None or stop is None or step is None or step == 0:
            return None
        loop_range = range(start, stop, step)
        try:
            return len(loop_range)
        except OverflowError:
            return 1 if loop_range else 0

    @staticmethod
    def _owner_root_uuid(owner: Owner) -> str | None:
        """Return the root parent identity for a borrow owner.

        Despite the historical name, this returns ``logical_id`` rather
        than ``uuid``: ``uuid`` is bumped on every ``next_version`` while
        ``logical_id`` is preserved, and the borrow tracker needs an
        identity that's stable across pipeline-internal version bumps
        so the marker installed by an earlier op still matches a later
        op's operand.

        Args:
            owner (Owner): A slice-view ``ArrayValue`` or the consumed
                slot sentinel.

        Returns:
            str | None: Root parent ``logical_id`` after walking
                ``slice_of`` to the root for ``ArrayValue`` owners.
                Returns ``None`` for ``_ConsumedSlotMarker`` (no owning
                view).
        """
        if isinstance(owner, ArrayValue):
            root = owner
            while root.slice_of is not None:
                root = root.slice_of
            return root.logical_id
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
            key: The colliding ``(root_uuid, f"const:<idx>")`` key.

        Returns:
            Human-readable error body.
        """
        idx_str = _slot_descriptor(key).split(":", 1)[1]
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
    def _format_view_registration_conflict(
        existing: Owner,
        new_view: ArrayValue,
        idx: int,
        root: ArrayValue,
    ) -> str:
        """Format a view-registration overlap message.

        Args:
            existing (Owner): Current owner of the slot — always a
                slice-view ``ArrayValue`` at the call sites that
                invoke this formatter (consumed-slot markers and
                direct borrows take different paths above).
            new_view (ArrayValue): The new sliced ArrayValue being
                registered.
            idx (int): Colliding root-parent index.
            root (ArrayValue): The root parent ArrayValue.

        Returns:
            str: Human-readable error body.
        """
        if isinstance(existing, ArrayValue):
            owner_desc = f"slice view '{existing.name}'"
        else:
            owner_desc = "<unknown>"
        return (
            f"Slice view '{new_view.name}' covers '{root.name}[{idx}]' "
            f"which is already held by {owner_desc}.\n"
            f"Overlapping slice views (or direct access while a view is "
            f"live) are not supported."
        )
