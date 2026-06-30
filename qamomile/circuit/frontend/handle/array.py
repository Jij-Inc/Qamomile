from __future__ import annotations

import dataclasses
import enum
import typing
import uuid
from typing import Generic, Iterator, TypeVar, overload

from qamomile._utils import is_plain_int
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.arithmetic_operations import BinOpKind
from qamomile.circuit.ir.operation.operation import CInitOperation, QInitOperation
from qamomile.circuit.ir.operation.slice_array import (
    ReleaseSliceViewOperation,
    SliceArrayOperation,
)
from qamomile.circuit.ir.types import ValueType
from qamomile.circuit.ir.types.primitives import BitType, FloatType, QubitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.errors import (
    AffineTypeError,
    QubitBorrowConflictError,
    QubitConsumedError,
    UnreturnedBorrowError,
)

from .handle import Handle, _emit_binop
from .primitives import Bit, Float, Qubit, UInt

T = TypeVar("T", bound=Handle)


class ConsumeMode(enum.Enum):
    """Classify how a ``VectorView.consume`` call resolves slice borrows.

    ``ArrayBase.consume`` and ``VectorView.consume`` accept a free-form
    ``operation_name`` string used both for error messages and for
    dispatching how the parent's bulk-borrow table is updated.  The
    string itself is purely cosmetic; the dispatch logic only cares
    about which of three resolution modes applies, captured by this
    enum:

    - ``DESTRUCTIVE``: the consume physically destroys the qubits
      (``measure`` / ``cast``).  The consumed view stays parked in the
      parent's borrow table so any later access to the same slot is
      surfaced as a use-after-destroy.
    - ``RELEASING``: the consume returns the borrow to the parent
      cleanly (``slice assignment``) — the entries are dropped and the
      parent regains free access to the covered slots.
    - ``TRANSFER``: the consume hands ownership forward to a freshly
      built ``VectorView`` (broadcast gates, ``pauli_evolve``,
      ``QKernel.__call__``, controlled-U ``index_spec``).  The covered
      slots stay borrowed under the new handle.
    """

    DESTRUCTIVE = enum.auto()
    RELEASING = enum.auto()
    TRANSFER = enum.auto()


# String-keyed dispatch tables.  ``operation_name`` is also used as
# free-form text in error messages, so the canonical mode for an
# operation is its presence in one of these two sets.  Anything not
# listed defaults to :attr:`ConsumeMode.TRANSFER`.  Treat these as
# implementation detail of :func:`_classify_consume`; new code should
# query ``_classify_consume(...)`` rather than inspect the sets
# directly.
_DESTRUCTIVE_CONSUME_OPS: frozenset[str] = frozenset(
    {
        "measure",
        "cast",
        "expval",
        # ``qkernel call (view dropped)`` is used by ``QKernel.__call__``
        # to consume a ``VectorView`` argument whose corresponding output
        # is *not* a sliced ``ArrayValue`` (e.g. the callee returns a
        # scalar or a different register).  Without this consume the
        # input view would remain live after the call even though it was
        # logically handed to the callee — a use-after-move hole.
        # Classify as destructive so the covered parent slots become
        # consumed-slot markers, matching the semantics that the qubits
        # passed to the callee can no longer be reused by the caller.
        "qkernel call (view dropped)",
    }
)
_BORROW_RELEASING_CONSUME_OPS: frozenset[str] = frozenset({"slice assignment"})


def _classify_consume(operation_name: str | None) -> ConsumeMode:
    """Map a free-form ``operation_name`` to its :class:`ConsumeMode`.

    Args:
        operation_name (str | None): The string handed to
            ``ArrayBase.consume`` / ``VectorView.consume`` (also
            recorded on ``ArrayBase._consumed_by``).  ``None`` is
            treated as a transfer because pre-consume helpers may not
            yet have a recorded operation name.

    Returns:
        ConsumeMode: ``DESTRUCTIVE`` for measure / cast,
            ``RELEASING`` for slice assignment, ``TRANSFER`` otherwise.
    """
    if operation_name in _DESTRUCTIVE_CONSUME_OPS:
        return ConsumeMode.DESTRUCTIVE
    if operation_name in _BORROW_RELEASING_CONSUME_OPS:
        return ConsumeMode.RELEASING
    return ConsumeMode.TRANSFER


def _is_destroyed_slot_owner(owner: object) -> bool:
    """Return ``True`` iff ``owner`` represents a destroyed-qubit slot.

    A destroyed slot is a parent ``_borrowed_indices`` entry whose
    owner is a ``VectorView`` that has been destructively consumed
    (``view._consumed`` is ``True`` and the recorded
    ``_consumed_by`` operation name classifies as
    :attr:`ConsumeMode.DESTRUCTIVE`).  The destructively consumed
    view is parked in the parent's borrow table as the slot's owner
    until it goes out of scope; that persistence is what lets
    subsequent direct parent access at the same slot index fail
    loudly with ``QubitConsumedError`` rather than silently
    overwriting the destroyed qubit.

    Args:
        owner (object): A value from ``ArrayBase._borrowed_indices``
            (any of ``tuple[UInt, ...]`` for direct element borrows,
            ``ArrayBase[T]`` for slice-view ownership, or ``None``
            when the slot is absent).

    Returns:
        bool: ``True`` if the slot's recorded owner is a destructively
            consumed ``ArrayBase``; ``False`` otherwise (including
            for live views, direct element borrows, and ``None``).
    """
    return (
        isinstance(owner, ArrayBase)
        and owner._consumed
        and _classify_consume(owner._consumed_by) is ConsumeMode.DESTRUCTIVE
    )


@dataclasses.dataclass
class ArrayBase(Handle, Generic[T]):
    """Base class for array types (Vector, Matrix, Tensor).

    Provides common functionality for array indexing and element access.
    """

    value: ArrayValue
    element_type: typing.Type[T] = dataclasses.field(init=False)
    _shape: tuple[int | UInt, ...] = dataclasses.field(default_factory=tuple)
    # Maps a borrow key to the current owner of that slot.  Both direct
    # element borrows and slice (view) borrows live here because they
    # encode the same invariant: "this slot has an outstanding return
    # obligation, so the parent can't touch it directly".  The owner
    # has two runtime variants distinguished by ``isinstance``:
    #   * ``tuple[UInt, ...]`` — direct element borrow (the index handles
    #     borrowed).
    #   * ``ArrayBase[T]`` — slice view ownership (the ``VectorView``
    #     itself).  For slice borrows only constant indices are
    #     registered; symbolic slices skip registration (best-effort
    #     linearity).
    #
    # A "destroyed slot" (set by a destructive view consume such as
    # ``measure(view)`` or ``cast(view, ...)``) is encoded by leaving
    # the destructively consumed ``VectorView`` in this dict as the
    # slot's owner — ``view._consumed`` is ``True`` and
    # ``view._consumed_by`` records the destroying operation name.
    # The :func:`_is_destroyed_slot_owner` helper centralises the
    # check.
    _borrowed_indices: dict[
        tuple[str, ...],
        "tuple[UInt, ...] | ArrayBase[T]",
    ] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post-initialization to set up the array."""

        if self.value.type == QubitType():
            qinit_op = QInitOperation(operands=[], results=[self.value])
            tracer = get_current_tracer()
            tracer.add_operation(qinit_op)
        else:
            cinit_op = CInitOperation(operands=[], results=[self.value])
            tracer = get_current_tracer()
            tracer.add_operation(cinit_op)

        type_map: dict[ValueType, typing.Type[Handle]] = {
            QubitType(): Qubit,
            UIntType(): UInt,
            FloatType(): Float,
            BitType(): Bit,
        }
        self.element_type = type_map[self.value.type]  # type: ignore

    @classmethod
    def create(
        cls, shape: tuple[int | UInt, ...], name: str, el_type: typing.Type[T]
    ) -> "ArrayBase[T]":
        """Create an ArrayValue for the given shape and name."""
        shape_values = tuple(
            Value(type=UIntType(), name=f"dim_{i}").with_const(dim)
            if isinstance(dim, int)
            else dim.value
            for i, dim in enumerate(shape)
        )

        type_map: dict[typing.Type[Handle], ValueType] = {
            Qubit: QubitType(),
            UInt: UIntType(),
            Float: FloatType(),
            Bit: BitType(),
        }
        if el_type not in type_map:
            raise TypeError(f"Unsupported element type: {el_type}")
        el_ir_type = type_map[el_type]

        value = ArrayValue(type=el_ir_type, name=name, shape=shape_values)

        return cls(value=value, _shape=shape)

    @classmethod
    def _create_from_value(
        cls,
        value: ArrayValue,
        shape: tuple[int | UInt, ...],
        name: str | None = None,
    ) -> typing.Self:
        """Factory method to create an array instance without calling __init__.

        This is used internally when creating output arrays from operations
        like QFT/iQFT where we need to create a new array from an existing Value.

        Args:
            value: The ArrayValue to wrap.
            shape: The shape of the array.
            name: Optional name for the array.

        Returns:
            A new ArrayBase instance of the appropriate type.
        """
        # ObservableType is a ``@dataclass``; its auto-generated
        # ``__hash__`` is ``None``, so it cannot be a dict key.  Handle
        # it via isinstance before the dict lookup.
        from qamomile.circuit.frontend.handle.hamiltonian import Observable
        from qamomile.circuit.ir.types.hamiltonian import ObservableType

        type_map: dict[ValueType, typing.Type[Handle]] = {
            QubitType(): Qubit,
            UIntType(): UInt,
            FloatType(): Float,
            BitType(): Bit,
        }
        instance = object.__new__(cls)
        instance.value = value
        instance._shape = shape
        instance._borrowed_indices = {}
        instance.parent = None
        instance.indices = ()
        instance.name = name
        instance.id = str(uuid.uuid4())
        instance._consumed = False
        if isinstance(value.type, ObservableType):
            instance.element_type = Observable  # type: ignore[assignment]
        else:
            instance.element_type = type_map[value.type]  # type: ignore[assignment]
        return instance

    def _check_no_consumed_slots(self, operation_name: str) -> None:
        """Raise if any slot has been destroyed by a prior destructive view op.

        Destructive view operations (``measure(q[1::2])``, etc.) park the
        destructively-consumed view in ``_borrowed_indices`` for every
        slot they cover; see :func:`_is_destroyed_slot_owner` for how
        the destroyed-slot signal is encoded on the view's own state.
        Operations that subsequently use the
        whole array (e.g. ``expval(q, H)`` after
        ``measure(q[1::2])``) must call this guard before emitting IR
        so that the frontend detects the invalid reuse immediately —
        at trace time — rather than letting the program reach the
        backend and fail at runtime.

        For a :class:`VectorView`, the consumed-slot markers live on the
        view's *parent*'s borrow table (a destructive view consume marks
        the parent slot, not the view's own dict).  We therefore also
        walk the slice parent and check it against this view's
        compile-time-known coverage indices.  This is what catches
        "two views over the same slots; measure one, expval the other"
        — the second view's own ``_borrowed_indices`` is empty but the
        underlying physical slots have been destroyed.

        Args:
            operation_name: Name of the operation being attempted, used
                in the error message.

        Raises:
            QubitConsumedError: If any slot of this array has a
                destroyed-slot marker (see :func:`_is_destroyed_slot_owner`)
                in ``_borrowed_indices`` (own or parent's, for a view).
        """
        if not self.value.type.is_quantum():
            return
        consumed_keys = [
            k
            for k, owner in self._borrowed_indices.items()
            if _is_destroyed_slot_owner(owner)
        ]
        if consumed_keys:
            slots = sorted(
                int(k[0].split(":", 1)[1])
                for k in consumed_keys
                if k[0].startswith("const:")
            )
            raise QubitConsumedError(
                f"Cannot use '{self.value.name}' in '{operation_name}': "
                f"slot(s) {slots} were already destroyed by a prior "
                f"destructive view operation (e.g. measure/cast on a view).\n\n"
                f"Fix: Do not reuse an array whose physical qubits have been "
                f"consumed.  Disjoint views on non-overlapping slots are "
                f"allowed.",
                handle_name=self.value.name or "array",
                operation_name=operation_name,
            )

        # For a view: check the parent's borrow table at this view's
        # covered slots.  ``_slice_covered_indices`` is None for
        # symbolic-bound views; in that case the IR-level
        # SliceBorrowCheckPass picks up the violation post-fold.
        slice_parent = getattr(self, "_slice_parent", None)
        covered = getattr(self, "_slice_covered_indices", None)
        if slice_parent is not None and covered is not None:
            parent_consumed = sorted(
                idx
                for idx in covered
                if _is_destroyed_slot_owner(
                    slice_parent._borrowed_indices.get((f"const:{idx}",))
                )
            )
            if parent_consumed:
                raise QubitConsumedError(
                    f"Cannot use view of '{slice_parent.value.name}' in "
                    f"'{operation_name}': slot(s) {parent_consumed} were "
                    f"already destroyed by a prior destructive view operation "
                    f"(e.g. measure/cast on an overlapping view).\n\n"
                    f"Fix: Do not slice the same physical qubits twice when "
                    f"one view will be consumed destructively.",
                    handle_name=slice_parent.value.name or "array",
                    operation_name=operation_name,
                )

    def consume(self, operation_name: str = "unknown") -> typing.Self:
        """Consume the array, enforcing borrow-return contract for quantum arrays.

        For quantum arrays, all borrowed elements must be returned before the
        array can be consumed. This ensures that no unreturned borrows are
        silently discarded by operations like qkernel calls or controlled gates.

        When any slot of the array has already been physically consumed
        by an earlier destructive view operation (``measure(q[1::2])``
        then ``measure(q)``), this raises ``QubitConsumedError`` rather
        than silently re-consuming those slots.
        """
        self.validate_all_returned()
        if self.value.type.is_quantum():
            consumed_slots = sorted(
                int(k[0].split(":", 1)[1])
                for k, owner in self._borrowed_indices.items()
                if _is_destroyed_slot_owner(owner)
                and len(k) == 1
                and k[0].startswith("const:")
            )
            if consumed_slots:
                raise QubitConsumedError(
                    f"Cannot consume '{self.value.name}' via '{operation_name}': "
                    f"slot(s) {consumed_slots} were already destroyed by a "
                    f"prior destructive view operation.",
                    handle_name=self.value.name or "array",
                    operation_name=operation_name,
                )
        return super().consume(operation_name)  # type: ignore[return-value]

    @property
    def shape(self) -> tuple[int | UInt, ...]:
        """Return the shape of the array."""
        return self._shape

    def _make_uint_index(self, idx: int) -> UInt:
        """Create a UInt from an integer index.

        Args:
            idx (int): The Python integer index to wrap. A ``bool`` is
                rejected: ``True`` / ``False`` are not valid indices even
                though ``bool`` subclasses ``int`` (a bare
                ``isinstance(idx, int)`` would otherwise let ``arr[True]``
                silently alias ``arr[1]``).

        Returns:
            UInt: A handle whose underlying Value carries ``idx`` as a
                compile-time constant.

        Raises:
            TypeError: If ``idx`` is not a plain ``int`` (e.g. a ``bool``).
        """
        # is_plain_int covers element indices and slice start/step (both reach
        # here via _coerce_index, where the arg is already int-typed). The slice
        # ``stop`` bound never flows through here -- it is guarded in
        # _as_int_const instead.
        if not is_plain_int(idx):
            raise TypeError(
                f"array index must be a plain int, got {type(idx).__name__} "
                f"({idx!r}); bool is not a valid index."
            )
        return UInt(
            value=Value(type=UIntType(), name=f"idx_{idx}").with_const(idx),
            init_value=idx,
        )

    @staticmethod
    def _reject_negative_const_indices(indices: tuple[UInt, ...]) -> None:
        """Reject element indices that carry a negative compile-time constant.

        Python-style negative indexing is not supported for array element
        access. On sliced views the emit-time affine root composition
        ``root = start + step * local`` would silently address the wrong
        root slot (e.g. ``v[-1]`` for ``v = q[1:3]`` resolves to ``q[0]``
        instead of ``q[2]``), so the access is rejected before any IR is
        built — mirroring the negative ``start`` / ``stop`` / ``step``
        rejection in ``_make_slice_view``.

        Args:
            indices (tuple[UInt, ...]): Element indices to validate. Only
                indices whose underlying Value is a compile-time constant
                are checked; symbolic indices are deferred to emit-time
                resolution, which refuses negative resolved values.

        Raises:
            NotImplementedError: If any index is a negative compile-time
                constant.
        """
        for idx in indices:
            idx_const = _as_int_const(idx)
            if idx_const is not None and idx_const < 0:
                raise NotImplementedError(
                    f"Negative index is not supported for array element "
                    f"access (got index={idx_const}).  Use a non-negative "
                    f"value or compute the index explicitly."
                )

    def _format_index(self, indices: tuple[UInt, ...]) -> str:
        """Format indices for element naming and parameter tracking."""
        parts = []
        for idx in indices:
            if idx.value.is_constant():
                parts.append(str(int(idx.value.get_const())))
            else:
                parts.append(idx.value.name)
        return ",".join(parts)

    def _make_indices_key(self, indices: tuple[UInt, ...]) -> tuple[str, ...]:
        """Create a key for tracking borrowed indices.

        Uses the actual index value (for constants) or the value's uuid (for symbolic).
        """
        key_parts = []
        for idx in indices:
            if idx.value.is_constant():
                # Use the constant value as the key
                key_parts.append(f"const:{idx.value.get_const()}")
            else:
                # Use the value's uuid for symbolic indices
                key_parts.append(f"sym:{idx.value.uuid}")
        return tuple(key_parts)

    def _indices_definitely_different(
        self, lhs: tuple[UInt, ...], rhs: tuple[UInt, ...]
    ) -> bool:
        """Return True only when index mismatch can be proven safely.

        We intentionally keep this conservative for symbolic expressions:
        if an index is symbolic, we avoid rejecting because equivalent expressions
        may be recomputed into different UUIDs.
        """
        if len(lhs) != len(rhs):
            return True

        for lhs_idx, rhs_idx in zip(lhs, rhs):
            lhs_const = (
                lhs_idx.value.get_const() if lhs_idx.value.is_constant() else None
            )
            rhs_const = (
                rhs_idx.value.get_const() if rhs_idx.value.is_constant() else None
            )
            if (
                lhs_const is not None
                and rhs_const is not None
                and lhs_const != rhs_const
            ):
                return True
        return False

    def _get_element(self, indices: tuple[UInt, ...]) -> T:
        """Get an element at the given indices.

        Raises:
            NotImplementedError: If any index is a negative compile-time
                constant — Python-style negative indexing is not
                supported (see ``_reject_negative_const_indices``).
            QubitConsumedError: If the array (or the targeted slot) was
                already destroyed by a prior destructive operation
                (``measure`` / ``cast`` / ``expval``).
            QubitBorrowConflictError: If the element is already borrowed,
                or the slot is currently owned by a ``VectorView`` slice
                (whether on ``self`` or a nested outer view).
        """
        # Reject constant negative indices before any borrow-table or IR
        # side effect: a negative index built into the IR would either
        # silently compose to the wrong root slot through a view's affine
        # map or surface much later as an internal allocator assertion.
        self._reject_negative_const_indices(indices)

        indices_key = self._make_indices_key(indices)
        index_str = self._format_index(indices)

        # Bounds-check a constant index that overflows a constant dimension.
        # This stops an out-of-range constant access — e.g. ``empty[0]`` on
        # a length-0 slice view (``empty = s[1:1]``) or ``s[5]`` on a
        # length-3 array — from being built into the IR. For a measured
        # ``Vector[Bit]`` such an element would otherwise compose to a
        # valid-but-wrong root clbit at emit time and be read silently. Only
        # the ``idx >= dim`` overflow is rejected here: negative indices were
        # already rejected above, and symbolic indices / dimensions (loop
        # variables, runtime parameters) carry no provable range and are
        # deferred to emit-time resolution.
        if len(indices) == 1 and self._shape:
            idx_const = _as_int_const(indices[0])
            dim_const = _as_int_const(self._shape[0])
            if (
                idx_const is not None
                and dim_const is not None
                and idx_const >= dim_const
            ):
                raise IndexError(
                    f"Index {idx_const} is out of range for "
                    f"'{self.value.name or 'array'}' of length {dim_const}."
                )

        # Check if the array itself has been consumed (e.g., by cast or measure)
        if self._consumed and self.value.type.is_quantum():
            display_name = self.value.name or "array"
            raise QubitConsumedError(
                f"Array '{display_name}' was already consumed by "
                f"'{self._consumed_by}' and cannot be accessed.",
                handle_name=display_name,
                operation_name="array element access",
                first_use_location=self._consumed_by,
            )

        # Check if already borrowed — same dict covers direct element
        # borrows and slice-held slots; a destructively consumed view
        # also stays in the dict (with ``view._consumed`` and
        # ``view._consumed_by`` set), so a single ``ArrayBase`` branch
        # below covers both "view active" and "view destroyed" by
        # querying the helper.  The owner object tells us which one
        # so we can surface a tailored message.
        if indices_key in self._borrowed_indices and self.value.type.is_quantum():
            owner = self._borrowed_indices[indices_key]
            if _is_destroyed_slot_owner(owner):
                raise QubitConsumedError(
                    f"Physical qubit '{self.value.name}[{index_str}]' was already "
                    f"consumed by a destructive operation (e.g. measure / cast) "
                    f"on a view covering this slot.",
                    handle_name=f"{self.value.name}[{index_str}]",
                    operation_name="array element access",
                )
            if isinstance(owner, ArrayBase):
                raise QubitBorrowConflictError(
                    f"Parent slot '{self.value.name}[{index_str}]' is currently "
                    f"held by a VectorView slice.\n"
                    f"Access it through the view, or let the view finish "
                    f"before touching the parent directly.",
                    handle_name=f"{self.value.name}[{index_str}]",
                    operation_name="array element access",
                )
            raise QubitBorrowConflictError(
                f"Array element '{self.value.name}[{index_str}]' is already borrowed.\n"
                f"Return it before borrowing again.\n\n"
                f"Fix:\n"
                f"  q = {self.value.name}[{index_str}]\n"
                f"  q = qmc.h(q)\n"
                f"  {self.value.name}[{index_str}] = q  # Return the element first",
                handle_name=f"{self.value.name}[{index_str}]",
                operation_name="array element access",
            )

        # Nested-view cross-check: when ``self`` is a ``VectorView`` and
        # we can resolve ``indices`` to a concrete root-space slot,
        # reject access if the root parent's borrow table records that
        # slot as owned by another live ``VectorView`` (a nested inner
        # view sliced off ``self``).  Without this, accessing
        # ``a[overlap_idx]`` after ``b = a[inner_range]`` would silently
        # double-use the qubit ``b`` is also claiming.
        if (
            isinstance(self, VectorView)  # type: ignore[unreachable]
            and self.value.type.is_quantum()
            and self._slice_covered_indices is not None
            and len(indices) == 1
        ):
            local_idx = _as_int_const(indices[0])  # type: ignore[unreachable]
            if local_idx is not None and 0 <= local_idx < len(
                self._slice_covered_indices
            ):
                root_idx = self._slice_covered_indices[local_idx]
                root_key = (f"const:{root_idx}",)
                root_owner = self._slice_parent._borrowed_indices.get(root_key)
                if (
                    isinstance(root_owner, ArrayBase)
                    and root_owner is not self
                    and not _is_destroyed_slot_owner(root_owner)
                ):
                    root_name = self._slice_parent.value.name or "array"
                    raise QubitBorrowConflictError(
                        f"View element '{self.value.name}[{index_str}]' "
                        f"resolves to '{root_name}[{root_idx}]', which is "
                        f"currently held by another slice view "
                        f"'{root_owner.value.name}' (typically a nested "
                        f"inner slice).  Return the inner view via "
                        f"``outer[range] = inner`` before accessing the "
                        f"outer at this slot.",
                        handle_name=f"{self.value.name}[{index_str}]",
                        operation_name="array element access",
                    )

        self._borrowed_indices[indices_key] = indices

        element_value = Value(
            type=self.value.type,
            name=f"{self.value.name}[{index_str}]",
            parent_array=self.value,
            element_indices=tuple(idx.value for idx in indices),
        )
        if self.value.is_parameter():
            param_name = self.value.parameter_name()
            element_param_name = f"{param_name}[{index_str}]"
            element_value = element_value.with_parameter(element_param_name)
        return self.element_type(value=element_value, parent=self, indices=indices)

    def _return_element(self, indices: tuple[UInt, ...], value: T) -> None:
        """Validate, consume, and release a borrowed element.

        Order: validate -> consume -> borrow release (fixed sequence).

        Three paths for quantum arrays:

        1. **Borrowed index, correct parent**: validates identity, consumes
           handle, releases borrow. Normal borrow-return path.
        2. **Borrowed index, wrong parent**: raises ``AffineTypeError``.
           Prevents returning a value from a different array.
        3. **Unborrowed index**: consumes the handle without identity check.
           Allows writing a fresh qubit to an index that was never borrowed.

        Args:
            indices: The indices where the element is being returned.
            value: The handle being written back.

        Raises:
            NotImplementedError: If any index is a negative compile-time
                constant — Python-style negative indexing is not
                supported (see ``_reject_negative_const_indices``).
            QubitConsumedError: If the array was already consumed.
            AffineTypeError: If the index was borrowed **and** the value
                was not borrowed from this array (``value.parent is not self``).

        Notes:
            For computed symbolic indices (e.g. ``i + 1``, ``n - j - 1``),
            each evaluation creates a new UInt with a different uuid, so a
            key derived from the LHS indices may not match the borrow key.
            To handle this, we also derive a *source key* from the value's
            provenance (``value.indices``), which still holds the original
            UInt handles from ``_get_element`` and thus matches the borrow
            key reliably.
        """
        # Mirror the read-side rejection so ``arr[-1] = value`` cannot
        # register a borrow-table entry under a negative key either.
        self._reject_negative_const_indices(indices)

        target_key = self._make_indices_key(indices)
        index_str = self._format_index(indices)

        # Check if the array itself has been consumed (e.g., by cast or measure)
        if self._consumed and self.value.type.is_quantum():
            display_name = self.value.name or "array"
            raise QubitConsumedError(
                f"Array '{display_name}' was already consumed by "
                f"'{self._consumed_by}' and cannot be accessed.",
                handle_name=display_name,
                operation_name="array element return",
                first_use_location=self._consumed_by,
            )

        # Determine the borrow key to release.
        # If the value carries provenance from this array, use its original
        # indices key (stable across re-evaluations of computed indices).
        source_key: tuple[str, ...] | None = None
        if isinstance(value, Handle) and value.parent is self and value.indices:
            source_key = self._make_indices_key(value.indices)

        if (
            self.value.type.is_quantum()
            and source_key is not None
            and source_key in self._borrowed_indices
            and self._indices_definitely_different(indices, value.indices)
        ):
            source_index_str = self._format_index(value.indices)
            raise AffineTypeError(
                f"Cannot return borrowed element '{self.value.name}[{source_index_str}]' "
                f"to '{self.value.name}[{index_str}]'.\n"
                f"Borrowed elements must be returned to the same index.",
                handle_name=self.value.name,
                operation_name="array element return",
            )

        release_key = (
            source_key
            if source_key is not None and source_key in self._borrowed_indices
            else target_key
        )

        if self.value.type.is_quantum():
            if release_key in self._borrowed_indices:
                # Borrow-return path: element was borrowed, validate identity
                if not isinstance(value, Handle) or value.parent is not self:
                    raise AffineTypeError(
                        f"Cannot return a value to '{self.value.name}[{index_str}]' "
                        f"that was not borrowed from this array.",
                        handle_name=self.value.name,
                        operation_name="array element return",
                    )
            else:
                # Non-borrowed index: reject handles actively borrowed from
                # a *different*, still-live array.  Handles whose parent was
                # already consumed (e.g. returned from a @qkernel call) are
                # treated as detached/fresh and allowed.
                if (
                    isinstance(value, Handle)
                    and value.parent is not None
                    and value.parent is not self
                    and not value.parent._consumed
                ):
                    raise AffineTypeError(
                        f"Cannot assign a handle borrowed from another array "
                        f"to '{self.value.name}[{index_str}]' — "
                        f"it was not borrowed from this array.",
                        handle_name=self.value.name,
                        operation_name="array element return",
                    )

            # Consume the handle (prevents reuse of old handle)
            value.consume(operation_name=f"return to {self.value.name}[{index_str}]")

        # Release the borrow.  Destroyed-slot owners (destructively
        # consumed views still parked in the dict) are not outstanding
        # borrows — they record physically-destroyed slots — so they
        # must never be deleted by a non-destructive return path.  In
        # practice ``_get_element`` already rejects access to consumed
        # slots, so a borrow couldn't have been issued for one; the
        # guard here is defense-in-depth against future paths that
        # might bypass element access (e.g. computed-index returns).
        if release_key in self._borrowed_indices:
            current = self._borrowed_indices[release_key]
            if not _is_destroyed_slot_owner(current):
                del self._borrowed_indices[release_key]
        else:
            # Classical types are freely copyable — no linear enforcement needed
            pass

    def _copy_subclass_state_to(self, new_handle: Handle) -> None:
        """Copy ArrayBase-specific state to a new handle created by consume()."""
        assert isinstance(new_handle, ArrayBase)
        new_handle._shape = self._shape
        new_handle._borrowed_indices = dict(self._borrowed_indices)
        new_handle.element_type = self.element_type

    def validate_all_returned(self) -> None:
        """Validate all borrowed elements have been returned.

        Strict-return policy: an active slice view that is still
        registered as the owner of any parent slot is treated as an
        unreturned borrow even if the view itself has no outstanding
        element borrows.  The caller must perform an explicit slice
        assignment (``parent[a:b:c] = view``) to release the view's
        bulk-borrow before consuming the parent.  Destructively
        consumed views (parked in the dict with ``view._consumed`` set
        and ``view._consumed_by`` classified as
        :attr:`ConsumeMode.DESTRUCTIVE`) record physically-destroyed
        slots and are not outstanding borrows; they survive
        end-of-block so a later whole-array consume can detect and
        reject the destroyed slots.

        Raises:
            UnreturnedBorrowError: If any elements are still borrowed,
                either directly or by a slice view that has not been
                explicitly returned via slice assignment.
        """
        if not self.value.type.is_quantum():
            return

        # Destroyed-slot owners (destructively consumed views parked
        # in the dict) are not outstanding borrows — they record
        # physically-destroyed slots — so they must be excluded from
        # the "unreturned borrows" report and from the "any entries
        # left?" check.
        outstanding_entries = {
            k: v
            for k, v in self._borrowed_indices.items()
            if not _is_destroyed_slot_owner(v)
        }
        if not outstanding_entries:
            return

        borrowed_strs: list[str] = []
        view_held: set[str] = set()
        for key, owner in outstanding_entries.items():
            if isinstance(owner, ArrayBase):
                # Slice-held slot: key is a single-element tuple like
                # ``("const:<idx>",)``; surface the slot number directly.
                idx_str = key[0].split(":", 1)[1]
                borrowed_strs.append(
                    f"{self.value.name}[{idx_str}] (held by slice view)"
                )
                view_held.add(owner.value.name or "view")
            else:
                index_str = self._format_index(owner)
                borrowed_strs.append(f"{self.value.name}[{index_str}]")

        if view_held:
            view_names = ", ".join(sorted(view_held))
            raise UnreturnedBorrowError(
                f"Array '{self.value.name}' has unreturned slice-view "
                f"borrows from {view_names}.\n"
                f"Borrowed slots: {', '.join(borrowed_strs)}\n\n"
                f"Fix: Return every active view via slice assignment "
                f"before consuming the parent:\n"
                f"  view = {self.value.name}[a:b:c]\n"
                f"  # ... use view ...\n"
                f"  {self.value.name}[a:b:c] = view  # explicit return",
                handle_name=self.value.name,
            )

        raise UnreturnedBorrowError(
            f"Array '{self.value.name}' has unreturned borrowed elements.\n"
            f"Borrowed elements: {', '.join(borrowed_strs)}\n\n"
            f"Fix: Write back all borrowed elements before using the array:\n"
            f"  q = {self.value.name}[i]\n"
            f"  q = qm.h(q)\n"
            f"  {self.value.name}[i] = q  # Return the element",
            handle_name=self.value.name,
        )


@dataclasses.dataclass
class Vector(ArrayBase[T]):
    """1-dimensional array type.

    Example:
        ```python
        import qamomile.circuit as qmc

        # Create a vector of 3 qubits
        qubits: qmc.Vector[qmc.Qubit] = qmc.qubit_array(3, name="qubits")

        # Access elements
        q0 = qubits[0]
        q0 = qmc.h(q0)
        qubits[0] = q0

        # Apply H gate to all qubits (CORRECT)
        n = qubits.shape[0]
        for i in qmc.range(n):
            qubits[i] = qmc.h(qubits[i])

        # Slicing returns a VectorView over a subset of the parent vector.
        # The view shares borrow tracking with the parent; element access
        # on the view transparently indexes the parent.
        evens = qubits[0::2]
        for i in qmc.range(evens.shape[0]):
            evens[i] = qmc.h(evens[i])
        ```
    """

    value: ArrayValue = dataclasses.field(default=None)  # type: ignore
    _shape: tuple[int | UInt] = dataclasses.field(default=(0,))
    _borrowed_indices: dict[
        tuple[str, ...],
        "tuple[UInt, ...] | ArrayBase[T]",
    ] = dataclasses.field(default_factory=dict)

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: UInt) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> "VectorView[T]": ...

    def __getitem__(self, index: int | UInt | slice) -> "T | VectorView[T]":
        """Get element at the given index, or return a view for a slice.

        Args:
            index: Either a concrete index (``int`` or ``UInt``) selecting
                a single element, or a Python ``slice`` object selecting a
                contiguous or strided subset.  Slices accept ``int``,
                ``UInt``, or ``None`` for each of ``start``/``stop``/``step``;
                ``step`` must be positive.

        Returns:
            A single element handle for integer/UInt indices, or a
            ``VectorView`` that shares borrow tracking with this vector
            for slice indices.

        Raises:
            NotImplementedError: If the slice uses a non-positive step or
                negative start/stop values, or if an element index is a
                negative constant — Python-style negative indexing is not
                yet supported.
        """
        if isinstance(index, slice):
            return self._make_slice_view(index)
        if isinstance(index, int):
            index = self._make_uint_index(index)
        return self._get_element((index,))

    @overload
    def __setitem__(self, index: int | UInt, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: "Vector[T]") -> None: ...
    def __setitem__(self, index: "int | UInt | slice", value: "T | Vector[T]") -> None:
        """Set element at the given index, or return a view via slice assignment.

        For an integer / ``UInt`` index this is the existing element
        borrow-return path: the right-hand side handle is validated
        against the borrow recorded for that index and consumed.

        For a ``slice`` index, the assignment is treated as the
        **explicit borrow-return of a slice view**.  The right-hand
        side must be a ``VectorView`` covering exactly the same
        root-space slot set as ``self[index]`` would build (typically
        produced by a broadcast such as
        ``qmc.h(self[index])``).  Although the static type hint accepts
        any ``Vector[T]``, the runtime rejects a non-``VectorView``
        right-hand side with ``TypeError`` — the wider hint exists so
        that helper kernels which broadcast over a view and annotate
        their return as ``Vector[T]`` (the natural choice when the
        helper is meant to work on either a whole register or a view)
        can be slice-assigned without a cast.  The view's parent
        ownership is released on both sides of the IR boundary:
        ``value.consume(...)`` clears the frontend
        ``_borrowed_indices`` entries, and a
        :class:`ReleaseSliceViewOperation` is emitted so the post-fold
        :class:`SliceBorrowCheckPass` can mirror the release in IR
        state.  See :meth:`_return_slice_view` for the full validation
        sequence.

        Args:
            index: Either a concrete index (``int`` / ``UInt``) selecting
                a single element, or a Python ``slice`` selecting a
                contiguous or strided subset.
            value: The handle being written back.  For integer indices
                this must be the element handle previously borrowed
                from ``self[index]``; for slice indices this must be a
                ``VectorView`` covering the same slot set.

        Raises:
            TypeError: If ``value`` is not a ``VectorView`` when
                ``index`` is a ``slice``.
            AffineTypeError: If ``value`` is a view of a different root
                parent, or its coverage does not match the LHS slice.
            QubitConsumedError: If ``value`` is an already-consumed
                view handle.
            ValueError: If either side has symbolic-bound slice
                metadata (not supported in this revision).
        """
        if isinstance(index, slice):
            return self._return_slice_view(index, value)
        if isinstance(index, int):
            index = self._make_uint_index(index)
        # Non-slice indices must receive a single element handle, not a
        # vector.  The overload set enforces this at the call site; the
        # runtime check raises ``TypeError`` (mirroring the slice
        # branch's existing behaviour for non-``VectorView`` RHS) so
        # the error stays stable even under ``python -O``, and the
        # type checker narrows ``value`` to ``T`` past this point.
        if isinstance(value, Vector):
            display = self.value.name or "qs"
            index_str = self._format_index((index,))
            raise TypeError(
                f"Element assignment to '{display}[{index_str}]' "
                f"expected a single element handle, got "
                f"{type(value).__name__}.  Use ``{display}[a:b] = ...`` "
                "for slice-level assignment."
            )
        self._return_element((index,), value)

    def _return_slice_view(self, s: slice, value: "T | Vector[T]") -> None:
        """Accept a ``VectorView`` as the explicit return of ``self[s]``.

        Validates that ``value`` covers the same root-space slot set
        as ``self[s]`` would build, then consumes the view handle and
        emits a :class:`ReleaseSliceViewOperation` so the post-fold
        IR linearity check can drop the corresponding view ownership
        entries.

        The validation order matters: cheap type / consumed checks
        come before the slot-set comparison helper, so a malformed
        right-hand side cannot allocate or trace anything.  In order:
        (1) RHS is a ``VectorView``; (2) RHS is not already consumed;
        (3) LHS root (and, for a view LHS, the view itself) is still
        live — mirrors the consumed-array guard in
        :meth:`_return_element`; (4) compute LHS coverage; (5) RHS
        root matches LHS root, except for a concrete nested full-slice
        that fully replaces its immediate outer and returns directly
        to the same root; (6) RHS coverage matches LHS coverage; (7)
        every covered parent slot is currently owned by the RHS view
        (catches a silently-drained stale view); (8) consume the RHS
        view; (9) emit
        :class:`ReleaseSliceViewOperation`.

        Args:
            s: The Python ``slice`` object that appeared on the left
                of the assignment.
            value: The right-hand side handle.  Must be a
                ``VectorView``.

        Raises:
            TypeError: ``value`` is not a ``VectorView``.
            QubitConsumedError: ``value`` is already consumed, the LHS
                root (or LHS view) was already consumed, or a covered
                slot has been destroyed by a prior destructive consume.
            AffineTypeError: ``value``'s root parent does not match
                ``self`` (or its root parent for a ``VectorView``
                left-hand side), its covered slot set does not match
                what ``self[s]`` would build, or it no longer owns
                every covered slot in the parent's borrow record.
            ValueError: Either side has symbolic-bound slice metadata
                (literal-bounded slices only in this revision).
            NotImplementedError: For non-positive ``step`` or negative
                ``start`` / ``stop`` values.
        """
        # (1) Type check first — cheap and avoids allocating helpers
        #     on an invalid right-hand side.
        if not isinstance(value, VectorView):
            raise TypeError(
                "Slice assignment expects a VectorView (e.g. the "
                "result of ``qmc.h(qs[a:b])``).  Got "
                f"{type(value).__name__}.  Use ``qs[i] = ...`` for "
                "element-level assignment."
            )

        # (2) Reject already-consumed value before any state mutation.
        if value._consumed:
            display_name = value.value.name or "view"
            raise QubitConsumedError(
                f"Slice assignment got an already-consumed VectorView "
                f"'{display_name}'.",
                handle_name=display_name,
                operation_name="slice assignment",
            )

        # (3) LHS-root liveness check.  ``_return_element`` performs the
        #     same guard before touching ``_borrowed_indices`` (see
        #     ``ArrayBase._return_element``); slice assignment must
        #     mirror it so a pattern like
        #     ``v = qs[0:2]; measure(qs); qs[0:2] = v`` cannot quietly
        #     succeed after the parent has been consumed.  For a
        #     ``VectorView`` LHS the parent identity is captured via
        #     ``_slice_parent``; the view's own ``_consumed`` is also
        #     checked because a view may itself have been retired
        #     (e.g. by ``cast``).
        self_root = self._slice_parent if isinstance(self, VectorView) else self  # type: ignore[attr-defined,unreachable]
        if self_root._consumed and self_root.value.type.is_quantum():
            display_name = self_root.value.name or "array"
            raise QubitConsumedError(
                f"Slice assignment target '{display_name}' was already "
                f"consumed by '{self_root._consumed_by}' and cannot be "
                f"used as the LHS of a slice assignment.",
                handle_name=display_name,
                operation_name="slice assignment",
                first_use_location=self_root._consumed_by,
            )
        if (
            isinstance(self, VectorView)  # type: ignore[unreachable]
            and self._consumed
            and self.value.type.is_quantum()
        ):
            display_name = self.value.name or "view"  # type: ignore[unreachable]
            raise QubitConsumedError(
                f"Slice assignment LHS view '{display_name}' was "
                f"already consumed by '{self._consumed_by}'.",
                handle_name=display_name,
                operation_name="slice assignment",
                first_use_location=self._consumed_by,
            )

        # (4) Compute the would-be LHS coverage side-effect-free.  If
        #     anything along the chain (parent length, slice bound) is
        #     still a symbolic ``UInt`` at trace time, defer the deep
        #     checks (6) and (7) to the post-fold
        #     ``SliceBorrowCheckPass`` — the bindings haven't been
        #     applied yet, so concrete coverage is unknowable here.
        try:
            lhs_covered: tuple[int, ...] | None = self._normalize_slice_to_covered(s)
        except ValueError:
            lhs_covered = None

        rhs_covered: tuple[int, ...] | None = None
        if lhs_covered is not None:
            rhs_covered = value._slice_covered_indices
            if rhs_covered is None:
                rhs_covered = _coverage_from_array_value(value.value)

        skipped_outer_views: list[VectorView[T]] = []

        # (5) Identity check.  Two flavours depending on whether the
        #     RHS came from a top-level slice or a nested slice:
        #
        #     * top-level (``value._slice_outer_view is None``):
        #       ``value._slice_parent`` is the root parent.  Slice
        #       assignment must address the same root.
        #     * nested (``value._slice_outer_view is not None``):
        #       the nested view's IR ``slice_of`` chain still points
        #       at the root, but the user is required to return the
        #       view through its immediate outer first
        #       (``outer[range] = inner``), then return the outer to
        #       the root (``root[range] = outer``).  Allowing
        #       ``root[range] = inner`` would normally skip the outer
        #       view's release and leave the outer in an inconsistent
        #       state.  The one safe exception is ``inner = outer[:]``:
        #       when the nested slice, its outer view, and the root LHS
        #       cover exactly the same concrete slots, the inner has
        #       fully replaced the outer.  In that case root assignment
        #       can release the inner directly, and we retire the
        #       skipped outer below so it cannot be reused afterward.
        if value._slice_outer_view is not None:
            if value._slice_outer_view is not self:
                outer_view = value._slice_outer_view
                outer_chain: list[VectorView[T]] = []
                chain_matches_full_coverage = True
                while outer_view is not None:
                    outer_covered = outer_view._slice_covered_indices
                    if outer_covered is None:
                        outer_covered = _coverage_from_array_value(outer_view.value)
                    if (
                        outer_covered is None
                        or lhs_covered is None
                        or rhs_covered is None
                        or value._slice_parent is not self_root
                        or outer_view._slice_parent is not self_root
                        or outer_view._borrowed_indices
                        or tuple(lhs_covered) != tuple(rhs_covered)
                        or tuple(rhs_covered) != tuple(outer_covered)
                    ):
                        chain_matches_full_coverage = False
                        break
                    outer_chain.append(outer_view)
                    outer_view = outer_view._slice_outer_view
                can_skip_outer = (
                    not hasattr(self, "_slice_parent")
                    and bool(outer_chain)
                    and chain_matches_full_coverage
                )
                if can_skip_outer:
                    skipped_outer_views = outer_chain
                else:
                    raise AffineTypeError(
                        f"Nested slice view '{value.value.name}' must be "
                        f"returned through its immediate outer view first "
                        f"(e.g. ``outer_view[a:b:c] = inner_view``); only "
                        f"then can the outer view be returned to "
                        f"'{self_root.value.name}'.",
                        handle_name=value.value.name or "view",
                        operation_name="slice assignment",
                    )
        elif value._slice_parent is not self_root:
            raise AffineTypeError(
                f"Slice assignment on '{self.value.name}' expects a "
                f"VectorView whose root parent is "
                f"'{self_root.value.name}'; got a view of "
                f"'{value._slice_parent.value.name}'.",
                handle_name=self.value.name,
                operation_name="slice assignment",
            )

        if lhs_covered is not None:
            # (6) Compare with the RHS view's recorded coverage.  The
            #     RHS records concrete coverage only when its own
            #     bounds and parent length were constants at slice
            #     construction time; mismatched concrete-vs-symbolic
            #     sides are accepted here and re-checked in IR.
            #
            #     ``VectorView.consume`` clears ``_slice_covered_indices``
            #     after non-destructively releasing the parent, so a view
            #     that has already passed through a broadcast gate
            #     (``qmc.h(qs[a:b])`` consumes the view internally as of
            #     the broadcast-affine-type fix) arrives with
            #     ``_slice_covered_indices is None`` even when its IR
            #     ``ArrayValue`` carries concrete slice metadata.  Fall
            #     back to recomputing the coverage from the underlying
            #     ``ArrayValue`` so coverage mismatch is still caught
            #     after the broadcast.
            if rhs_covered is not None and tuple(lhs_covered) != tuple(rhs_covered):
                raise AffineTypeError(
                    f"Slice assignment coverage mismatch on "
                    f"'{self.value.name}': LHS covers {list(lhs_covered)}, "
                    f"RHS covers {list(rhs_covered)}.",
                    handle_name=self.value.name,
                    operation_name="slice assignment",
                )

            # (7) RHS-ownership check: ``value`` must currently own
            #     every parent slot it claims to cover.  Without
            #     this, a stale view that was silently drained by a
            #     later overlapping slice
            #     (``a = qs[0:2]; b = qs[0:2]; qs[0:2] = a``) would
            #     pass slice assignment as a no-op release — the
            #     parent's borrow record now points at ``b``, so
            #     ``a.consume`` and the emitted
            #     ``ReleaseSliceViewOperation`` both run without
            #     raising or removing anything, leaving the user
            #     with the false impression that ``a`` was returned.
            #     Slice assignment is the *explicit* borrow-return
            #     path, so we require ownership to be intact.  Only
            #     reachable when LHS coverage is concrete; the
            #     symbolic-bound case is validated by the IR pass.
            # ``VectorView.consume`` clears ``_slice_covered_indices`` to
            # ``None`` after non-destructively releasing the parent's
            # bulk-borrow records.  Broadcast gates
            # (``_broadcast_single_qubit_gate`` and friends) call this
            # consume internally on entry, so the handle returned by
            # ``qmc.h(qs[a:b])`` already has both ``_slice_covered_indices
            # is None`` *and* the parent's entries for the covered slots
            # cleared.  Treat that combination as the broadcast-consumed
            # path: the borrow is provably released, the only thing left
            # is the IR-level release marker we emit below.  This is the
            # distinguishing feature against a *stale* view (drained by
            # an overlapping later slice): the stale view's
            # ``_slice_covered_indices`` is still populated, so it falls
            # through to the "no longer owns" branch and gets rejected.
            rhs_pre_released = value._slice_covered_indices is None
            for idx in lhs_covered:
                key = (f"const:{idx}",)
                owner = self_root._borrowed_indices.get(key)
                if owner is value:
                    continue
                if rhs_pre_released and owner is None:
                    continue
                if _is_destroyed_slot_owner(owner):
                    raise QubitConsumedError(
                        f"Slice assignment RHS view "
                        f"'{value.value.name}' covers "
                        f"'{self_root.value.name}[{idx}]', but that "
                        f"slot was destroyed by a prior destructive "
                        f"consume (measure / cast).",
                        handle_name=value.value.name or "view",
                        operation_name="slice assignment",
                    )
                raise AffineTypeError(
                    f"Slice assignment RHS view '{value.value.name}' "
                    f"no longer owns '{self_root.value.name}[{idx}]'; "
                    f"the parent's borrow record points at "
                    f"{'an unrelated view / element' if owner is not None else 'no live owner'}.  "
                    f"This view was likely drained by a later "
                    f"overlapping slice; reconstruct the view before "
                    f"assigning it back.",
                    handle_name=value.value.name or "view",
                    operation_name="slice assignment",
                )

        # (8) Frontend release: consume the view handle.  ``"slice
        #     assignment"`` classifies as :attr:`ConsumeMode.RELEASING`
        #     — no destroyed-slot signal is recorded; instead the
        #     parent's borrow entry for the covered slots is deleted
        #     by the releasing branch of ``VectorView.consume``.
        value.consume(operation_name="slice assignment")
        for skipped_outer_view in skipped_outer_views:
            skipped_outer_view._consumed = True
            skipped_outer_view._consumed_by = "slice assignment"

        # (8.5) Re-register the LHS as owner of the just-released slots
        #       when slice-assigning onto an outer view (nested return).
        #       ``_nested_slice`` only hands the inner-covered slots
        #       over from the outer view to the inner view, so when
        #       the inner is returned via ``outer[range] = inner``,
        #       the outer must reclaim them so it can still be
        #       returned to its own parent later.
        if (
            isinstance(self, VectorView)  # type: ignore[unreachable]
            and lhs_covered is not None
            and self_root.value.type.is_quantum()
        ):
            for idx in lhs_covered:  # type: ignore[unreachable]
                key = (f"const:{idx}",)
                if self_root._borrowed_indices.get(key) is None:
                    self_root._borrowed_indices[key] = self

        # (9) IR release: emit a marker the post-fold linearity check
        #     can observe.  Without this op, ``SliceBorrowCheckPass``
        #     would still treat the view as a live owner of the
        #     covered slots and reject any subsequent direct access
        #     to those slots.  Stripped after the linearity check by
        #     ``StripSliceArrayOpsPass``.
        release_op = ReleaseSliceViewOperation(
            operands=[value.value],
            results=[],
        )
        get_current_tracer().add_operation(release_op)

    def _normalize_slice_to_covered(self, s: slice) -> tuple[int, ...]:
        """Compute the root-space covered indices for ``self[s]`` without side effects.

        Mirrors the validation, clamping, and length computation done
        by :meth:`_make_slice_view` but performs everything in pure
        Python ``int`` space — no IR op is emitted, no entry is added
        to ``_borrowed_indices``, no ``BinOp`` is produced.

        The clamp behaviour intentionally tracks ``_make_slice_view``'s
        guard at [array.py:1099](array.py:1099): clamp ``stop`` against
        parent length **only** when the parent length is itself a
        compile-time constant.  When the parent length is symbolic, the
        user-provided bounds are trusted (matching what
        ``_make_slice_view`` writes into ``_slice_covered_indices`` for
        the RHS view); a later out-of-range binding manifests at
        emit-time against the resolved length.  This is what lets
        ``q = qubit_array(n, "q"); q[0:2] = h(q[0:2])`` validate
        coverage at trace time even before ``n`` is bound.

        Args:
            s: The Python ``slice`` object whose root-space coverage
                is requested.

        Returns:
            A tuple of integer parent-space indices covered by the
            slice, in iteration order.

        Raises:
            ValueError: If any of ``s.start`` / ``s.stop`` / ``s.step``
                is symbolic (not a compile-time integer constant), or
                if ``s.stop is None`` and the parent length is also
                symbolic (no concrete upper bound to fall back on).
            NotImplementedError: For non-positive ``step`` or negative
                ``start`` / ``stop`` values (same restrictions as
                ``_make_slice_view``).
        """
        parent_length = self._shape[0]
        parent_len_int = _as_int_const(parent_length)

        raw_start = 0 if s.start is None else s.start
        if s.stop is None:
            if parent_len_int is None:
                raise ValueError(
                    "slice assignment with an implicit stop (``q[a:]``) "
                    "requires the parent vector length to be a "
                    "compile-time integer constant; got a symbolic "
                    "length."
                )
            raw_stop = parent_len_int
        else:
            raw_stop = s.stop
        raw_step = 1 if s.step is None else s.step

        start_int = _as_int_const(raw_start)
        stop_int = _as_int_const(raw_stop)
        step_int = _as_int_const(raw_step)
        if start_int is None or stop_int is None or step_int is None:
            raise ValueError(
                "slice assignment with symbolic bounds is not supported "
                "in this revision.  Use literal-bounded slicing."
            )

        for bound_name, bound_value in (
            ("start", start_int),
            ("stop", stop_int),
            ("step", step_int),
        ):
            if bound_value < 0:
                raise NotImplementedError(
                    f"Negative {bound_name} ({bound_value}) is not "
                    f"supported for slice assignment."
                )
        if step_int <= 0:
            raise NotImplementedError(
                f"slice assignment requires a positive step (got {step_int})."
            )

        # Same Python-style clamp as ``_make_slice_view``, but in pure
        # int space (no ``_uint_min`` / ``BinOp`` emit).  Skip when the
        # parent length is symbolic — see the docstring for rationale.
        if parent_len_int is not None:
            stop_int = min(stop_int, parent_len_int)
            start_int = min(start_int, stop_int)

        length = max(0, (stop_int - start_int + step_int - 1) // step_int)
        return tuple(start_int + step_int * i for i in range(length))

    def _make_slice_view(self, s: slice) -> "VectorView[T]":
        """Build a VectorView describing the slice ``s`` of this vector.

        Emits a :class:`SliceArrayOperation` to the current tracer whose
        result is a fresh :class:`ArrayValue` with ``slice_of`` pointing
        to this vector's ``value``.  Bounds are normalized (``None`` ->
        0 / parent length / 1) and converted to ``UInt`` handles.  The
        view's length is a Python ``int`` when all bounds are compile-time
        constants and a symbolic ``UInt`` expression otherwise; the
        latter relies on :class:`ConstantFoldingPass` to resolve it
        against bindings before segmentation.

        Args:
            s: The Python ``slice`` object from ``vec[slice]``.

        Returns:
            A ``VectorView`` wrapping the new sliced ``ArrayValue``.
            Element accesses on the view produce IR element values whose
            ``parent_array`` points to the sliced ``ArrayValue``; the
            emit-time resolver walks ``slice_of`` back to the root
            parent to obtain the physical qubit index.

        Raises:
            NotImplementedError: For non-positive ``step`` or any
                negative ``start``/``stop`` value.  Negative indices and
                reverse strides are not yet supported.
        """
        parent_length = self._shape[0]

        raw_start = 0 if s.start is None else s.start
        raw_stop = parent_length if s.stop is None else s.stop
        raw_step = 1 if s.step is None else s.step

        # Validate bounds against raw ``int`` AND against ``UInt`` handles
        # whose underlying ``Value`` is a compile-time constant — otherwise
        # e.g. a ``UInt`` produced from ``UIntType().with_const(0)`` slips
        # through the ``int``-only check and reaches ``_compute_slice_length``
        # where a zero step triggers ``ZeroDivisionError``.
        for name, value in (("start", s.start), ("stop", s.stop), ("step", s.step)):
            const_val = _as_int_const(value) if value is not None else None
            if const_val is not None and const_val < 0:
                raise NotImplementedError(
                    f"Negative {name} is not supported for Vector slicing "
                    f"(got {name}={const_val}).  Use a non-negative value or "
                    f"compute the index explicitly."
                )
        step_const_for_validate = _as_int_const(raw_step)
        if step_const_for_validate is not None and step_const_for_validate <= 0:
            raise NotImplementedError(
                f"Vector slicing requires a positive step "
                f"(got step={step_const_for_validate}). "
                f"Reverse/zero strides are not yet supported."
            )

        # Clamp ``stop`` (and ``start``) to the parent length, matching
        # Python semantics (``s[3:10]`` with ``len(s)==4`` yields
        # ``s[3:4]``).  Without this, ``measure(q[3:10])`` above a
        # 4-qubit register happily produces a 7-clbit result array with
        # only one actual measurement emitted — a correctness bug.
        #
        # The clamp is expressed via ``_uint_min`` so the same single
        # construction handles both the literal-bound case (folded to a
        # plain ``int`` immediately) and the case where ``raw_stop`` is
        # symbolic (left as a ``BinOp(MIN)`` for ``ConstantFoldingPass``
        # to resolve once parameter bindings are available).  We only
        # apply the clamp when ``parent_length`` is itself a concrete
        # constant; if the parent's length is symbolic too, the slice
        # length must remain a free expression resolvable to whatever
        # the parent eventually binds to.  Clamping ``start`` against
        # the just-clamped ``stop`` additionally guarantees
        # ``stop >= start`` so the downstream length expression
        # ``((stop - start) + (step - 1)) // step`` cannot underflow on
        # ``UInt`` arithmetic.
        if _as_int_const(parent_length) is not None:
            raw_stop = _uint_min(raw_stop, parent_length)
            raw_start = _uint_min(raw_start, raw_stop)

        start_uint = self._coerce_index(raw_start)
        step_uint = self._coerce_index(raw_step)

        length = _compute_slice_length(raw_start, raw_stop, raw_step)
        length_value = self._to_length_value(length)

        # ``covered_indices`` is needed by the bulk-borrow tracker for
        # compile-time-known slices.  Accept both raw ints and
        # constant-wrapping ``UInt`` handles (the common case when the
        # parent length came from ``qubit_array(4, "q")`` — the literal
        # ``4`` is wrapped as ``UInt`` downstream so ``raw_stop`` can be
        # a UInt even though the user passed an int).
        covered_indices: tuple[int, ...] | None = None
        start_const = _as_int_const(raw_start)
        step_const = _as_int_const(raw_step)
        if (
            start_const is not None
            and step_const is not None
            and isinstance(length, int)
        ):
            covered_indices = tuple(start_const + j * step_const for j in range(length))

        # Build the sliced ArrayValue with slice metadata.
        sliced_av: ArrayValue = ArrayValue(
            type=self.value.type,
            name=f"{self.value.name}[slice]",
            shape=(length_value,),
            slice_of=self.value,
            slice_start=start_uint.value,
            slice_step=step_uint.value,
        )

        # Emit SliceArrayOperation for IR visibility.  Stripped by
        # ConstantFoldingPass before segmentation, so the op is
        # trace-time-only; the sliced ArrayValue (carrying slice meta)
        # is what downstream ops actually reference.
        slice_op = SliceArrayOperation(
            operands=[self.value, start_uint.value, step_uint.value],
            results=[sliced_av],
        )
        get_current_tracer().add_operation(slice_op)

        return VectorView._wrap(
            parent=self,
            sliced_av=sliced_av,
            length=length,
            start_uint=start_uint,
            step_uint=step_uint,
            covered_indices=covered_indices,
        )

    def _to_length_value(self, length: "int | UInt") -> Value:
        """Coerce a length (int or UInt) to a ``Value`` for ArrayValue.shape.

        Args:
            length: Constant ``int`` length, or symbolic ``UInt``.

        Returns:
            A ``Value`` suitable as an ``ArrayValue.shape`` component:
            a const ``UInt`` Value for int inputs, or the existing
            ``UInt.value`` for symbolic inputs.
        """
        if isinstance(length, int):
            return Value(type=UIntType(), name=f"slice_len_{length}").with_const(length)
        return length.value

    def _coerce_index(self, value: int | UInt) -> UInt:
        """Coerce an int or UInt to a UInt handle.

        Args:
            value: Python integer literal or existing ``UInt`` handle.

        Returns:
            ``UInt`` handle suitable for downstream arithmetic.
        """
        if isinstance(value, int):
            return self._make_uint_index(value)
        return value

    def __iter__(self) -> Iterator[T]:
        """Iteration over Vector is prohibited to prevent common bugs.

        Direct iteration like 'for item in vector:' doesn't support in-place
        modification in @qkernel contexts. Use explicit index-based loops instead.

        Raises:
            TypeError: Always raised to prevent iteration.

        Example:
            Instead of:
                for qi in qubits:
                    qi = qm.h(qi)  # This doesn't modify qubits!

            Use:
                for i in qmc.range(len(qubits)):
                    qubits[i] = qm.h(qubits[i])  # This works correctly
        """
        raise TypeError(
            "Direct iteration over Vector is not supported in @qkernel functions.\n"
            "Vector iteration cannot modify elements in-place, leading to silent bugs.\n\n"
            "Use explicit index-based iteration instead:\n"
            "  # Incorrect:\n"
            "  for item in vector:\n"
            "      item = qmc.operation(item)\n\n"
            "  # Correct:\n"
            "  for i in qmc.range(len(vector)):\n"
            "      vector[i] = qmc.operation(vector[i])\n"
        )


@dataclasses.dataclass
class Matrix(ArrayBase[T]):
    """2-dimensional array type.

    Example:
        ```python
        import qamomile as qm

        # Create a 3x4 matrix of qubits
        matrix: qm.Matrix[qm.Qubit] = qm.Matrix(shape=(3, 4))

        # Access elements (always requires 2 indices)
        q = matrix[0, 1]
        q = qm.h(q)
        matrix[0, 1] = q
        ```
    """

    value: ArrayValue = dataclasses.field(default=None)  # type: ignore
    _shape: tuple[int | UInt, int | UInt] = dataclasses.field(default=(0, 0))
    _borrowed_indices: dict[
        tuple[str, ...],
        "tuple[UInt, ...] | ArrayBase[T]",
    ] = dataclasses.field(default_factory=dict)

    def __getitem__(self, index: tuple[int | UInt, int | UInt]) -> T:
        """Get element at the given (row, col) index."""
        i, j = index
        if isinstance(i, int):
            i = self._make_uint_index(i)
        if isinstance(j, int):
            j = self._make_uint_index(j)
        return self._get_element((i, j))

    def __setitem__(self, index: tuple[int | UInt, int | UInt], value: T) -> None:
        """Set element at the given (row, col) index."""
        i, j = index
        if isinstance(i, int):
            i = self._make_uint_index(i)
        if isinstance(j, int):
            j = self._make_uint_index(j)
        self._return_element((i, j), value)


@dataclasses.dataclass
class Tensor(ArrayBase[T]):
    """N-dimensional array type (3 or more dimensions).

    Example:
        ```python
        import qamomile as qm

        # Create a 2x3x4 tensor of qubits
        tensor: qm.Tensor[qm.Qubit] = qm.Tensor(shape=(2, 3, 4))

        # Access elements (requires all indices)
        q = tensor[0, 1, 2]
        q = qm.h(q)
        tensor[0, 1, 2] = q
        ```
    """

    value: ArrayValue = dataclasses.field(default=None)  # type: ignore
    _shape: tuple[int | UInt, ...] = dataclasses.field(default_factory=tuple)
    _borrowed_indices: dict[
        tuple[str, ...],
        "tuple[UInt, ...] | ArrayBase[T]",
    ] = dataclasses.field(default_factory=dict)

    def __getitem__(self, index: tuple[int | UInt, ...]) -> T:
        """Get element at the given indices."""
        if len(index) != len(self._shape):
            raise IndexError(f"Expected {len(self._shape)} indices, got {len(index)}")
        converted = tuple(
            self._make_uint_index(i) if isinstance(i, int) else i for i in index
        )
        return self._get_element(converted)

    def __setitem__(self, index: tuple[int | UInt, ...], value: T) -> None:
        """Set element at the given indices."""
        if len(index) != len(self._shape):
            raise IndexError(f"Expected {len(self._shape)} indices, got {len(index)}")
        converted = tuple(
            self._make_uint_index(i) if isinstance(i, int) else i for i in index
        )
        self._return_element(converted, value)


def _compute_slice_length(
    start: int | UInt, stop: int | UInt, step: int | UInt
) -> int | UInt:
    """Compute ``max(0, ceil((stop - start) / step))`` for slice bounds.

    When all three bounds resolve to Python ``int`` constants — either
    directly or as a ``UInt`` whose underlying value is a compile-time
    constant (which is the common case because ``qubit_array(4, "q")``
    wraps its literal length in a ``UInt`` handle) — returns a concrete
    ``int``.  Otherwise falls back to symbolic ``UInt`` arithmetic,
    which emits ``BinOp`` nodes into the current tracer and relies on
    constant folding in later passes to collapse the expression where
    possible.

    Args:
        start: Slice start (inclusive).
        stop: Slice stop (exclusive).
        step: Slice step; must be a positive value.

    Returns:
        ``int`` when all inputs are concrete integers, otherwise a ``UInt``
        handle representing the slice length.
    """
    start_int = _as_int_const(start)
    stop_int = _as_int_const(stop)
    step_int = _as_int_const(step)
    if start_int is not None and stop_int is not None and step_int is not None:
        if stop_int <= start_int:
            return 0
        return (stop_int - start_int + step_int - 1) // step_int

    return ((stop - start) + (step - 1)) // step


def _as_int_const(value: int | UInt) -> int | None:
    """Return the Python ``int`` for ``value`` if it is a compile-time constant.

    Args:
        value (int | UInt): Either a raw Python ``int`` or a ``UInt``
            handle whose backing ``Value`` may or may not carry a
            constant. A ``bool`` is rejected: ``True`` / ``False`` are not
            valid integer constants for a slice bound or length even though
            ``bool`` subclasses ``int`` (this is the slice-bound counterpart
            of the index guard in :meth:`ArrayBase._make_uint_index`, which
            ``_as_int_const`` would otherwise let through — e.g. ``q[0:True]``
            silently becoming ``q[0:1]``).

    Returns:
        int | None: The int when ``value`` is a plain ``int`` directly, or a
            ``UInt`` whose ``.value`` is constant and resolvable to an
            ``int``; ``None`` when ``value`` is a symbolic ``UInt``.

    Raises:
        TypeError: If ``value`` is a ``bool``.
    """
    # Explicit bool guard rather than is_plain_int: a bool routed to the
    # symbolic (``None``) return path below would make a slice bound look
    # symbolic instead of being rejected. This is the slice-bound counterpart
    # to the index guard in ArrayBase._make_uint_index; a slice ``stop`` bound
    # only reaches here, never _make_uint_index (e.g. q[0:True] -> q[0:1]).
    if isinstance(value, bool):
        raise TypeError(
            f"a bool is not a valid integer here (got {value!r}); a slice "
            f"bound or length must be a plain int."
        )
    if isinstance(value, int):
        return value
    if isinstance(value, UInt) and value.value.is_constant():
        const = value.value.get_const()
        if const is not None:
            try:
                return int(const)
            except (TypeError, ValueError):
                return None
    return None


def _coverage_from_array_value(av: ArrayValue) -> tuple[int, ...] | None:
    """Recompute concrete coverage from a sliced ``ArrayValue``'s metadata.

    Used by slice-assignment validation when the wrapping
    :class:`VectorView` has already passed through a broadcast gate
    (or any other non-destructive consume) that cleared its
    ``_slice_covered_indices`` to ``None``.  The underlying
    ``ArrayValue`` still carries the full affine map (``slice_of`` /
    ``slice_start`` / ``slice_step`` / ``shape[0]``); when every
    component is a compile-time constant the covered root-space slot
    set can be reconstructed identically to what ``_make_slice_view``
    would have stored.

    Args:
        av (ArrayValue): The sliced ``ArrayValue`` whose root-space
            coverage is wanted.

    Returns:
        tuple[int, ...] | None: The covered root-space slot indices in
            iteration order, or ``None`` when ``av`` is not a slice
            (``slice_of is None``) or when any of its start / step /
            length components is still symbolic.
    """
    if av.slice_of is None:
        return None
    start_const = av.slice_start.get_const() if av.slice_start is not None else None
    step_const = av.slice_step.get_const() if av.slice_step is not None else None
    if start_const is None or step_const is None or not av.shape:
        return None
    length_value = av.shape[0]
    length_const = length_value.get_const() if length_value is not None else None
    if length_const is None:
        return None
    try:
        start_int = int(start_const)
        step_int = int(step_const)
        length_int = int(length_const)
    except (TypeError, ValueError):
        return None
    return tuple(start_int + step_int * i for i in range(length_int))


def _uint_min(a: int | UInt, b: int | UInt) -> int | UInt:
    """Compute ``min(a, b)`` over int / UInt operands.

    When both operands are compile-time integer constants the result is
    folded eagerly to a Python ``int``.  Otherwise emits a
    :class:`BinOp` of kind :attr:`BinOpKind.MIN` and returns a symbolic
    ``UInt`` handle; :class:`ConstantFoldingPass` resolves the
    expression once parameter bindings make both sides concrete.

    The shared trace-time / fold-time path is what unifies the
    Python-style end-bound clamp on slices (``q[3:10]`` over a 4-qubit
    register collapses to ``q[3:4]`` whether the bounds are literals
    or arithmetic over kernel parameters).

    Args:
        a: First operand; raw ``int`` or ``UInt`` handle.
        b: Second operand; raw ``int`` or ``UInt`` handle.

    Returns:
        Concrete ``int`` when both inputs fold to constants, otherwise
        a symbolic ``UInt`` whose backing ``Value`` is the result of an
        emitted ``BinOp(MIN)``.
    """
    a_const = _as_int_const(a)
    b_const = _as_int_const(b)
    if a_const is not None and b_const is not None:
        return a_const if a_const <= b_const else b_const

    a_uint = (
        a
        if isinstance(a, UInt)
        else UInt(
            value=Value(type=UIntType(), name="uint_const").with_const(int(a)),
            init_value=int(a),
        )
    )
    b_uint = (
        b
        if isinstance(b, UInt)
        else UInt(
            value=Value(type=UIntType(), name="uint_const").with_const(int(b)),
            init_value=int(b),
        )
    )
    result = UInt(value=Value(type=UIntType(), name="uint_min"), init_value=0)
    _emit_binop(a_uint.value, b_uint.value, result, BinOpKind.MIN)
    return result


class VectorView(Vector[T]):
    """Strided view over a parent ``Vector``, backed by a sliced ``ArrayValue``.

    A ``VectorView`` is produced by slicing a ``Vector`` (``q[1::2]``,
    ``q[a:b]``, etc.).  It is a thin ``Vector`` subclass whose ``value``
    is a fresh ``ArrayValue`` with ``slice_of`` / ``slice_start`` /
    ``slice_step`` metadata pointing back to the parent's ``ArrayValue``.
    Element accesses go through ``Vector._get_element`` unchanged — the
    IR element carries ``parent_array = sliced_av``, and the emit-time
    resolver walks the ``slice_of`` chain to produce the physical qubit
    index.  No affine translation happens in the view itself.

    Because the sliced ``ArrayValue`` is a first-class IR ``Value``,
    the view can be passed as an operand of ``CallBlockOperation`` to
    another ``@qkernel`` without the inline-trace special-case path
    that earlier iterations required.  Passing views through
    ``expval`` / ``measure`` likewise operates on the sliced qubit
    subset, not the root parent as a whole.

    Linearity:
        Slicing bulk-borrows the covered parent slots whenever
        ``start``, ``step`` and ``length`` are compile-time ``int``
        constants.  While the view is live, accessing the corresponding
        parent slot directly (``q[0]`` after ``evens = q[0::2]``) raises
        ``QubitConsumedError``.  Under the strict-return policy the
        view's ownership is cleared only by two operations:

        - slice-assigning it back into the parent
          (``parent[a:b:c] = view``) — this is the *only* path that
          fully releases the borrow without destroying the qubits;
        - destructively consuming it (``measure(view)`` /
          ``cast(view, ...)`` / ``expval(view, H)``) — the physical
          slots become consumed markers, no return needed.

        Every other consume (broadcast gates ``h(view)``,
        ``pauli_evolve(view, H, gamma)``, sub-kernel calls
        ``f(view)``, controlled-U ``index_spec``) only *transfers*
        ownership to a freshly-wrapped ``VectorView`` and that new
        view still must be returned via slice assignment.  A view
        left bulk-borrowing at the parent's consume point raises
        ``UnreturnedBorrowError``.

        Symbolic slices (``q[lo:hi]`` with ``lo``/``hi`` ``UInt``) cannot
        enumerate their covered slots at trace time and therefore skip
        the bulk-borrow here; ``SliceBorrowCheckPass`` picks them up
        post-fold after bindings resolve the bounds to concrete values.

    Attributes:
        _slice_parent: The backing ``Vector``; owns the bulk-borrow slots.
        _slice_start: Parent-space index of the first covered element.
        _slice_step: Stride in parent-space; always a positive ``UInt``.
        _slice_covered_indices: Concrete parent indices registered as
            slice-owned, or ``None`` for symbolic slices.

    Example:
        ```python
        @qmc.qkernel
        def alternating_h(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            evens = q[0::2]
            for i in qmc.range(evens.shape[0]):
                evens[i] = qmc.h(evens[i])
            q[0::2] = evens  # explicit return before the parent is used
            return q
        ```
    """

    _slice_parent: "Vector[T]"
    _slice_start: UInt
    _slice_step: UInt
    _slice_covered_indices: tuple[int, ...] | None
    # For nested slice views, the immediate outer ``VectorView`` from
    # which this view was sliced — used by slice assignment to enforce
    # the inner→outer→root return order.  ``None`` for top-level views
    # sliced directly from a ``Vector``.
    _slice_outer_view: "VectorView[T] | None"

    @classmethod
    def _wrap(
        cls,
        parent: "Vector[T]",
        sliced_av: ArrayValue,
        length: int | UInt,
        start_uint: UInt,
        step_uint: UInt,
        covered_indices: tuple[int, ...] | None,
    ) -> "VectorView[T]":
        """Wrap a freshly-constructed sliced ``ArrayValue`` as a view handle.

        Args:
            parent: The ``Vector`` being sliced.
            sliced_av: The sliced ``ArrayValue`` (already carrying
                ``slice_of`` / ``slice_start`` / ``slice_step``).  This
                is the new view's ``value``.
            length: Number of elements — a Python ``int`` when known at
                trace time or a symbolic ``UInt`` otherwise; stored on
                ``_shape`` for frontend-side shape queries.
            start_uint: Parent-space start index as a ``UInt`` handle
                (used when the view is later sliced further).
            step_uint: Parent-space stride as a ``UInt`` handle.
            covered_indices: Concrete parent indices the view covers
                (for constant slices), or ``None`` for symbolic slices.

        Returns:
            A ``VectorView`` whose IR value is ``sliced_av``.

        Raises:
            QubitConsumedError: If any covered parent slot has been
                destroyed by a prior destructive operation
                (``measure`` / ``cast`` / ``expval``) on an
                overlapping view.
            QubitBorrowConflictError: If any covered parent slot is
                already held by another live slice view or currently
                borrowed.
        """
        instance = object.__new__(cls)
        # ``value`` is the sliced ArrayValue itself.  Element accesses
        # on the view create IR values with parent_array=sliced_av; the
        # emit resolver walks slice_of back to the root parent.
        instance.value = sliced_av
        instance._shape = (length,)
        instance._borrowed_indices = {}
        instance.element_type = parent.element_type
        instance.parent = None
        instance.indices = ()
        instance.name = None
        instance.id = str(uuid.uuid4())
        instance._consumed = False
        instance._consumed_by = None
        instance._slice_parent = parent
        instance._slice_start = start_uint
        instance._slice_step = step_uint
        instance._slice_covered_indices = covered_indices
        instance._slice_outer_view = None

        if covered_indices is not None and parent.value.type.is_quantum():
            # Strict no-multi-view: while a view is live, the parent
            # slots it covers are locked.  The user must consume /
            # release the existing view (or let its destructive consume
            # leave a destroyed-slot breadcrumb) before constructing a
            # new view that touches the same slot.  Without this guard,
            # ``a = q[0::2]; b = q[0::2]`` would silently make ``a``
            # dead after ``b``'s construction — a silent
            # ownership-transfer that is hard to debug.
            for idx in covered_indices:
                key = (f"const:{idx}",)
                existing = parent._borrowed_indices.get(key)
                if existing is not None:
                    # A destructively-consumed view stays in
                    # ``_borrowed_indices`` with ``_consumed=True`` to
                    # leave a breadcrumb on its covered slots; touching
                    # those slots later is a permanent-loss event, not
                    # a releasable borrow conflict.  Surface it as
                    # ``QubitConsumedError`` to match the semantic of
                    # "the physical qubit is gone".
                    if _is_destroyed_slot_owner(existing):
                        raise QubitConsumedError(
                            f"Cannot slice across "
                            f"'{parent.value.name}[{idx}]' — it was "
                            f"already destroyed by a prior destructive "
                            f"view operation (e.g. measure / cast / "
                            f"expval) on an overlapping view.",
                            handle_name=f"{parent.value.name}[{idx}]",
                            operation_name="array slicing",
                        )
                    if isinstance(existing, ArrayBase):
                        raise QubitBorrowConflictError(
                            f"Parent slot '{parent.value.name}[{idx}]' is already "
                            f"owned by another slice view '{existing.value.name}' "
                            f"and cannot be re-sliced while that view is live.  "
                            f"Consume / release the existing view before "
                            f"slicing the same range again.",
                            handle_name=f"{parent.value.name}[{idx}]",
                            operation_name="array slicing",
                        )
                    raise QubitBorrowConflictError(
                        f"Cannot slice across '{parent.value.name}[{idx}]' — "
                        f"it is currently borrowed.  Return the borrowed "
                        f"element before slicing.",
                        handle_name=f"{parent.value.name}[{idx}]",
                        operation_name="array slicing",
                    )
                parent._borrowed_indices[key] = instance

        return instance

    @classmethod
    def _wrap_unregistered(
        cls,
        parent: "Vector[T]",
        sliced_av: ArrayValue,
        length: int | UInt,
        start_uint: UInt,
        step_uint: UInt,
    ) -> "VectorView[T]":
        """Construct a ``VectorView`` without touching ``parent``'s borrow table.

        :meth:`_wrap` runs the strict no-multi-view check and installs
        the new view as the owner of its covered slots.  This factory
        skips both — it produces a bare ``VectorView`` instance with
        only the structural fields filled in.  The caller is
        responsible for arranging ownership separately, typically via
        :meth:`_transfer_borrow_to` on the predecessor view (used by
        ``pauli_evolve`` and ``QKernel.__call__`` to carry the slice
        ownership forward onto a freshly-minted
        ``next_version`` / call-result ``ArrayValue``).

        Args:
            parent: The root ``Vector`` the view slices.
            sliced_av: The result ``ArrayValue`` produced by the op
                that's constructing the view (has ``slice_of`` /
                ``slice_start`` / ``slice_step`` already populated).
            length: Length of the view (``int`` or ``UInt``).
            start_uint: Parent-space start as a ``UInt`` handle.
            step_uint: Parent-space stride as a ``UInt`` handle.

        Returns:
            A ``VectorView`` with ``_slice_parent`` / ``_slice_start`` /
                ``_slice_step`` set.  ``_slice_covered_indices`` is
                ``None`` until :meth:`_transfer_borrow_to` populates it
                from the predecessor view.
        """
        instance = object.__new__(cls)
        instance.value = sliced_av
        instance._shape = (length,)
        instance._borrowed_indices = {}
        instance.element_type = parent.element_type
        instance.parent = None
        instance.indices = ()
        instance.name = None
        instance.id = str(uuid.uuid4())
        instance._consumed = False
        instance._consumed_by = None
        instance._slice_parent = parent
        instance._slice_start = start_uint
        instance._slice_step = step_uint
        instance._slice_covered_indices = None
        instance._slice_outer_view = None
        return instance

    def __post_init__(self) -> None:
        """Skip ``ArrayBase.__post_init__``.

        A view wraps an already-constructed sliced ``ArrayValue``; we
        must not re-emit a ``QInitOperation`` / ``CInitOperation`` for
        it.  Construction goes through :meth:`_wrap`, not the dataclass
        ``__init__``, so this hook is a no-op.
        """
        return

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: UInt) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> "VectorView[T]": ...

    def __getitem__(self, index: int | UInt | slice) -> "T | VectorView[T]":
        """Get element at the given view-local index, or nest a slice.

        Args:
            index: View-local index, or a ``slice`` that composes with
                this view's affine map back to the root parent.

        Returns:
            A single element handle for integer indices, or a further
            ``VectorView`` for slice indices.

        Raises:
            NotImplementedError: For nested slices with non-positive step
                or negative bounds.
        """
        if isinstance(index, slice):
            return self._nested_slice(index)
        return super().__getitem__(index)

    @overload
    def __setitem__(self, index: int | UInt, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: "Vector[T]") -> None: ...
    def __setitem__(self, index: "int | UInt | slice", value: "T | Vector[T]") -> None:
        """Return a borrowed element / sub-view to its parent slot(s).

        Element assignment (``view[i] = q``) delegates to the inherited
        ``Vector.__setitem__`` path.  Slice assignment
        (``view[a:b] = qmc.h(view[a:b])``) is the same explicit
        borrow-return form as on root vectors: the right-hand side
        must be a ``VectorView`` covering the same root-space slot set
        as ``view[a:b]`` would build (composed back to the shared root
        parent), and the view is consumed and recorded for release in
        IR via :class:`ReleaseSliceViewOperation`.  The static type
        hint accepts ``Vector[T]`` for the slice case; the runtime
        rejects a non-``VectorView`` right-hand side with ``TypeError``
        (see :class:`Vector.__setitem__` for the rationale).

        Args:
            index: View-local index, or a ``slice`` describing the
                sub-view to release.
            value: Handle being returned — element handle for integer
                indices, ``VectorView`` for slice indices.

        Raises:
            TypeError: For a slice index when ``value`` is not a
                ``VectorView``.
            AffineTypeError: For a slice index when ``value``'s root
                parent does not match this view's root parent, or when
                coverage does not match.
            ValueError: For a slice index with symbolic-bound
                metadata on either side.
            NotImplementedError: For a slice index with non-positive
                step or negative bounds.
        """
        if isinstance(index, slice):
            return self._return_slice_view(index, value)
        # See ``Vector.__setitem__`` for the rationale; mirror its
        # explicit ``TypeError`` so the surface is the same on a view.
        if isinstance(value, Vector):
            display = self.value.name or "view"
            # ``VectorView.__setitem__``'s int → UInt conversion lives
            # in the inherited ``Vector.__setitem__``; convert here so
            # the error message renders ``view[0]`` rather than the
            # full ``UInt`` dataclass repr.
            uint_index = (
                self._make_uint_index(index) if isinstance(index, int) else index
            )
            index_str = self._format_index((uint_index,))
            raise TypeError(
                f"Element assignment on view '{display}[{index_str}]' "
                f"expected a single element handle, got "
                f"{type(value).__name__}.  Use ``{display}[a:b] = ...`` "
                "for slice-level assignment."
            )
        super().__setitem__(index, value)

    def _normalize_slice_to_covered(self, s: slice) -> tuple[int, ...]:
        """Root-space coverage of ``self[s]`` for a nested slice, no side effects.

        Composes the outer (this view) and inner (``s``) affine maps
        in pure ``int`` space, mirroring :meth:`_nested_slice`
        including its out-of-range stop / start clamp against
        ``view_length``.

        Args:
            s: The Python ``slice`` applied to this view.

        Returns:
            A tuple of root-space slot indices covered by ``self[s]``,
            in iteration order.

        Raises:
            ValueError: If this view's start / step / length, or any
                of ``s.start`` / ``s.stop`` / ``s.step``, is not a
                compile-time integer constant.
            NotImplementedError: For non-positive ``step`` or negative
                ``start`` / ``stop`` values.
        """
        outer_start = _as_int_const(self._slice_start)
        outer_step = _as_int_const(self._slice_step)
        view_length = _as_int_const(self._shape[0])
        if outer_start is None or outer_step is None or view_length is None:
            raise ValueError(
                "slice assignment on a VectorView requires the outer "
                "view's start, step and length to all be compile-time "
                "integer constants; got a symbolic component."
            )

        raw_start = 0 if s.start is None else s.start
        raw_stop = view_length if s.stop is None else s.stop
        raw_step = 1 if s.step is None else s.step

        inner_start = _as_int_const(raw_start)
        inner_stop = _as_int_const(raw_stop)
        inner_step = _as_int_const(raw_step)
        if inner_start is None or inner_stop is None or inner_step is None:
            raise ValueError(
                "slice assignment with symbolic bounds is not supported "
                "in this revision."
            )

        for bound_name, bound_value in (
            ("start", inner_start),
            ("stop", inner_stop),
            ("step", inner_step),
        ):
            if bound_value < 0:
                raise NotImplementedError(
                    f"Negative {bound_name} ({bound_value}) is not "
                    f"supported for slice assignment on a VectorView."
                )
        if inner_step <= 0:
            raise NotImplementedError(
                f"slice assignment on a VectorView requires a positive "
                f"step (got {inner_step})."
            )

        # Same view-length clamp as ``_nested_slice``, pure int.
        inner_stop = min(inner_stop, view_length)
        inner_start = min(inner_start, inner_stop)

        inner_length = max(0, (inner_stop - inner_start + inner_step - 1) // inner_step)
        composed_start = outer_start + outer_step * inner_start
        composed_step = outer_step * inner_step
        return tuple(composed_start + composed_step * j for j in range(inner_length))

    def consume(self, operation_name: str = "unknown") -> typing.Self:
        """Consume the view and release its parent slice-borrows.

        Validates that every view-local borrow has been returned,
        then dispatches on ``operation_name`` to keep the parent's
        slice-borrow record consistent with the new strict-return
        semantics:

        * **Destructive** (``measure`` / ``cast``): leave ``self``
          parked in the parent's borrow table as a destroyed-slot
          breadcrumb.  ``super().consume()`` flips
          ``self._consumed = True`` and ``self._consumed_by =
          operation_name``, which is what
          :func:`_is_destroyed_slot_owner` reads to reject subsequent
          access at the same slot.
        * **Releasing** (``slice assignment``): drop every parent
          entry that ``self`` currently owns.  The caller (the slice-
          assignment frontend path) also emits a
          ``ReleaseSliceViewOperation`` so the IR-level checker sees
          the release.  This branch is reserved for explicit borrow-
          return paths.
        * **Transfer** (every other op — broadcast gates, rotation,
          phase, ControlledU, sub-kernel call argument consumption,
          etc.): rebind the parent's borrow entry from ``self`` to
          the new view handle returned here.  The new view inherits
          ``self._slice_covered_indices`` so it can be slice-assigned
          back to the parent later — strict-return requires that
          eventual ``parent[a:b:c] = new_view``.

        Operations that produce a fresh sliced ``ArrayValue`` (e.g.
        :func:`qamomile.circuit.frontend.operation.pauli_evolve.pauli_evolve`,
        :class:`QKernel.__call__` for callees that return a sliced
        array) cannot simply use the auto-returned ``new_view``
        because the new view they build wraps a different ``Value``
        than this consume's return.  Those op implementations call
        :meth:`_transfer_borrow_to` after building their result so
        the parent's borrow table tracks the right handle.

        Args:
            operation_name: Name of the operation consuming this view
                (used in error messages and for dispatch).

        Returns:
            A fresh view handle with the same backing state; under
            transfer the parent's borrow table now points at this
            handle, under release / destruction the parent's record
            for the covered slots is finalised.

        Raises:
            QubitConsumedError: If any covered slot was already
                destroyed by a prior destructive view consume on an
                overlapping view that has since gone out of scope.
        """
        self.validate_all_returned()
        mode = _classify_consume(operation_name)

        # A prior destructive view consume can leave a destroyed-slot
        # breadcrumb in the parent's borrow table; reject the consume
        # up-front when any of our covered slots are in that state.
        # Reachable only when the destructively consumed predecessor
        # view has gone out of scope by the time the new view
        # constructs over the same slots — strict no-multi-view
        # rejects the easy live-overlap variant earlier in ``_wrap``.
        if self._slice_covered_indices is not None:
            already_destroyed = sorted(
                idx
                for idx in self._slice_covered_indices
                if _is_destroyed_slot_owner(
                    self._slice_parent._borrowed_indices.get((f"const:{idx}",))
                )
            )
            if already_destroyed:
                raise QubitConsumedError(
                    f"Cannot consume view of '{self._slice_parent.value.name}' "
                    f"via '{operation_name}': slot(s) {already_destroyed} were "
                    f"already destroyed by a prior destructive view operation "
                    f"on overlapping slots.",
                    handle_name=self._slice_parent.value.name or "array",
                    operation_name=operation_name,
                )

        new_view = super().consume(operation_name)
        new_view._slice_parent = self._slice_parent
        new_view._slice_start = self._slice_start
        new_view._slice_step = self._slice_step
        # Preserve the nested-slice outer link so strict-return is still
        # enforced on the post-consume handle (e.g. ``view = qmc.h(b)``
        # where ``b`` was a nested slice off ``a`` keeps requiring
        # ``a[range] = view`` rather than ``root[range] = view``).
        new_view._slice_outer_view = self._slice_outer_view

        if self._slice_covered_indices is not None:
            for idx in self._slice_covered_indices:
                key = (f"const:{idx}",)
                owner = self._slice_parent._borrowed_indices.get(key)
                if mode is ConsumeMode.DESTRUCTIVE:
                    # Destructive consume: keep ``self`` parked under
                    # this slot so ``_is_destroyed_slot_owner`` can
                    # surface the destruction to later accesses.
                    continue
                if owner is self:
                    if mode is ConsumeMode.RELEASING:
                        del self._slice_parent._borrowed_indices[key]
                    else:  # ConsumeMode.TRANSFER
                        # Transfer ownership to the new view handle.
                        self._slice_parent._borrowed_indices[key] = new_view

        # ``_slice_covered_indices`` on the returned handle:
        #   DESTRUCTIVE / RELEASING — None (parent state finalised; the
        #     handle has nothing left to claim)
        #   TRANSFER — same as ``self``'s coverage; the new view is the
        #     live owner of those slots.
        if mode is ConsumeMode.TRANSFER:
            new_view._slice_covered_indices = self._slice_covered_indices
        else:
            new_view._slice_covered_indices = None
        return new_view

    def _transfer_borrow_to(
        self,
        new_owner: "VectorView[T]",
        operation_name: str,
    ) -> None:
        """Hand ownership of this view's parent slots to ``new_owner``.

        Used by op implementations that build a fresh ``VectorView``
        wrapper for the result of a transferring op (e.g.
        ``pauli_evolve``'s next-versioned ``ArrayValue`` or
        ``QKernel.__call__``'s slice-shaped return value).
        :meth:`consume` cannot do this transfer on its own because the
        auto-returned new view handle wraps ``self.value`` rather than
        the op's freshly minted result value.

        The method:

        1. Validates and runs the destroyed-slot precondition check
           (mirroring :meth:`consume`).
        2. Marks ``self`` as consumed via :meth:`Handle.consume`
           bookkeeping (sets ``_consumed`` / ``_consumed_by``).
        3. Rebinds every parent borrow entry currently owned by
           ``self`` to ``new_owner``.
        4. Hands the covered-indices set to ``new_owner`` so it can
           be slice-assigned back to the parent later.

        The caller must have already populated ``new_owner``'s
        ``_slice_parent``, ``_slice_start``, ``_slice_step`` and
        ``value`` fields before invoking this method.

        Args:
            new_owner: The freshly-built ``VectorView`` that should
                take over ``self``'s bulk-borrow on the parent.
            operation_name: Name of the transferring operation
                (recorded on ``self._consumed_by`` for error messages).

        Raises:
            QubitConsumedError: If any covered slot was already
                destroyed by a prior destructive view consume on an
                overlapping view that has since gone out of scope.
        """
        self.validate_all_returned()
        if self._slice_covered_indices is not None:
            already_destroyed = sorted(
                idx
                for idx in self._slice_covered_indices
                if _is_destroyed_slot_owner(
                    self._slice_parent._borrowed_indices.get((f"const:{idx}",))
                )
            )
            if already_destroyed:
                raise QubitConsumedError(
                    f"Cannot consume view of '{self._slice_parent.value.name}' "
                    f"via '{operation_name}': slot(s) {already_destroyed} were "
                    f"already destroyed by a prior destructive view operation "
                    f"on overlapping slots.",
                    handle_name=self._slice_parent.value.name or "array",
                    operation_name=operation_name,
                )
        self._consumed = True
        self._consumed_by = operation_name
        if self._slice_covered_indices is not None:
            for idx in self._slice_covered_indices:
                key = (f"const:{idx}",)
                if self._slice_parent._borrowed_indices.get(key) is self:
                    self._slice_parent._borrowed_indices[key] = new_owner
        new_owner._slice_covered_indices = self._slice_covered_indices
        # Preserve the nested-slice outer link across transfer so strict-
        # return still demands the same outer→root return chain on the
        # post-transfer handle.
        new_owner._slice_outer_view = self._slice_outer_view

    def _nested_slice(self, s: slice) -> "VectorView[T]":
        """Compose a nested slice into a single view over the same parent.

        Normalizes ``None`` start/stop/step against this view's length
        and collapses the nested affine map back to the root parent so
        the resulting view is a one-hop strided slice of the root.  The
        new sliced ``ArrayValue`` carries ``slice_of`` pointing at the
        root parent (not at this intermediate view), matching the
        emit-time resolver expectation.

        Args:
            s: The Python ``slice`` object applied to this view.

        Returns:
            A new ``VectorView`` whose parent is this view's parent
            ``Vector``.  The sliced ``ArrayValue`` behind the new view
            still derives from the root parent via a composed affine
            map; the emit resolver walks ``slice_of`` to the root.

        Raises:
            NotImplementedError: For non-positive step or negative bounds.
        """
        view_length = self._shape[0]

        raw_start = 0 if s.start is None else s.start
        raw_stop = view_length if s.stop is None else s.stop
        raw_step = 1 if s.step is None else s.step

        # Validate bounds via the const-unwrapping helper so a
        # ``UInt`` with const 0 / negative also trips the explicit
        # check here rather than falling through to ``_compute_slice_length``
        # where zero-step causes a ``ZeroDivisionError``.
        for name, value in (("start", s.start), ("stop", s.stop), ("step", s.step)):
            const_val = _as_int_const(value) if value is not None else None
            if const_val is not None and const_val < 0:
                raise NotImplementedError(
                    f"Negative {name} is not supported for VectorView slicing "
                    f"(got {name}={const_val})."
                )
        step_const_validate = _as_int_const(raw_step)
        if step_const_validate is not None and step_const_validate <= 0:
            raise NotImplementedError(
                f"VectorView slicing requires a positive step "
                f"(got step={step_const_validate})."
            )

        # Clamp ``stop`` (and ``start``) to the view length, matching
        # Python slice semantics.  Without this, a nested slice such as
        # ``q[0::2][0:99]`` produces an out-of-bounds coverage set that
        # later emit paths only partially resolve — the same silent-wrong
        # failure mode as the root-level slice clamp bug.  ``_uint_min``
        # routes both literal and symbolic bounds through the same IR
        # ``BinOp(MIN)`` so the constant folder unifies them.  We gate
        # on ``view_length`` being a concrete constant for the same
        # reason as the root-level clamp: if the view's own length is
        # symbolic, the clamped length must remain free for downstream
        # binding resolution.
        if _as_int_const(view_length) is not None:
            raw_stop = _uint_min(raw_stop, view_length)
            raw_start = _uint_min(raw_start, raw_stop)

        inner_start = self._slice_parent._coerce_index(raw_start)
        inner_step = self._slice_parent._coerce_index(raw_step)

        # Compose the affine maps back to the root parent so the new
        # sliced ArrayValue's slice_of points directly at the root;
        # emit never has to walk multi-hop slice chains.
        new_start: UInt = self._slice_start + self._slice_step * inner_start
        new_step: UInt = self._slice_step * inner_step
        new_length = _compute_slice_length(raw_start, raw_stop, raw_step)
        new_length_value = self._slice_parent._to_length_value(new_length)

        # For compile-time-known nested views, enumerate the covered
        # root-space indices so the bulk-borrow tracker can reject
        # aliased direct parent access (e.g. ``inner = q[0::2][1:3];
        # q[2] = x(q[2])``) at trace time.  ``None`` is preserved only
        # when any bound in the composed chain is symbolic.
        outer_start_const = _as_int_const(self._slice_start)
        outer_step_const = _as_int_const(self._slice_step)
        inner_start_const = _as_int_const(raw_start)
        inner_step_const = _as_int_const(raw_step)
        nested_covered: tuple[int, ...] | None = None
        if (
            outer_start_const is not None
            and outer_step_const is not None
            and inner_start_const is not None
            and inner_step_const is not None
            and isinstance(new_length, int)
        ):
            composed_start = outer_start_const + outer_step_const * inner_start_const
            composed_step = outer_step_const * inner_step_const
            nested_covered = tuple(
                composed_start + j * composed_step for j in range(new_length)
            )

        new_sliced_av = ArrayValue(
            type=self._slice_parent.value.type,
            name=f"{self._slice_parent.value.name}[slice]",
            shape=(new_length_value,),
            slice_of=self._slice_parent.value,
            slice_start=new_start.value,
            slice_step=new_step.value,
        )
        slice_op = SliceArrayOperation(
            operands=[self._slice_parent.value, new_start.value, new_step.value],
            results=[new_sliced_av],
        )
        get_current_tracer().add_operation(slice_op)

        # Nested slicing hands the *covered* slots over from ``self``
        # to the new inner view.  Slots ``self`` covers but the new
        # view does NOT touch stay registered against ``self`` in the
        # root parent's borrow table — the outer view remains a valid
        # owner for those slots and can be slice-assigned back to its
        # parent once the inner view is returned.  This implements the
        # "return inner to outer, then outer to root" semantics.
        if nested_covered is not None:
            for idx in nested_covered:
                key = (f"const:{idx}",)
                if self._slice_parent._borrowed_indices.get(key) is self:
                    del self._slice_parent._borrowed_indices[key]

        new_view = VectorView._wrap(
            parent=self._slice_parent,
            sliced_av=new_sliced_av,
            length=new_length,
            start_uint=new_start,
            step_uint=new_step,
            covered_indices=nested_covered,
        )
        # Record the immediate outer view so slice-assignment can
        # enforce that the user returns the nested slice via
        # ``outer[range] = inner`` (and then ``root[range] = outer``)
        # rather than skipping straight back to the root.
        new_view._slice_outer_view = self
        return new_view
