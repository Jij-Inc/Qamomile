from __future__ import annotations

import dataclasses
import typing
import uuid
from typing import Generic, Iterator, TypeVar, overload

from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.arithmetic_operations import BinOpKind
from qamomile.circuit.ir.operation.operation import CInitOperation, QInitOperation
from qamomile.circuit.ir.operation.slice_array import SliceArrayOperation
from qamomile.circuit.ir.types import ValueType
from qamomile.circuit.ir.types.primitives import BitType, FloatType, QubitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.errors import (
    AffineTypeError,
    QubitConsumedError,
    UnreturnedBorrowError,
)

from .handle import Handle, _emit_binop
from .primitives import Bit, Float, Qubit, UInt

T = TypeVar("T", bound=Handle)


class _ConsumedSlotMarker:
    """Sentinel marker indicating a physical qubit slot is destroyed.

    Appears as the owner of a ``_borrowed_indices`` entry when a
    destructive operation (``measure``, ``cast``) consumed a subset of
    a root array's slots via a view.  Any subsequent attempt to
    access, re-measure, or consume the same slot raises
    ``QubitConsumedError`` — mirroring how ``ArrayBase._consumed``
    handles whole-array consumption.

    Only one instance ever exists; equality is by identity.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return "<ConsumedSlot>"


_CONSUMED_SLOT: "_ConsumedSlotMarker" = _ConsumedSlotMarker()


# Operation names whose ``consume`` call physically destroys the
# quantum state of the target qubits (qubit handles cannot be reused,
# and the underlying physical register is gone).  Other consumes
# (gates, sub-kernel calls, ControlledU) only retire the SSA-style
# handle but the physical qubits carry forward in the result handle.
_DESTRUCTIVE_CONSUME_OPS: frozenset[str] = frozenset({"measure", "cast"})


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
    # differs — a tuple of ``UInt`` handles for a direct borrow (used
    # for error formatting and kept for backward compatibility), or the
    # owning ``VectorView`` itself for a slice borrow (used by
    # ``validate_all_returned`` to detect drained views).  For slice
    # borrows, only constant indices are registered; symbolic slices
    # skip registration (best-effort linearity).
    _borrowed_indices: dict[tuple[str, ...], "tuple[UInt, ...] | ArrayBase[T]"] = (
        dataclasses.field(default_factory=dict)
    )

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

        Destructive view operations (``measure(q[1::2])``, etc.) install
        ``_ConsumedSlotMarker`` sentinels in ``_borrowed_indices`` for
        every slot they consume.  Operations that subsequently use the
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
            QubitConsumedError: If any ``_ConsumedSlotMarker`` is present
                in ``_borrowed_indices`` (own or parent's, for a view).
        """
        if not self.value.type.is_quantum():
            return
        consumed_keys = [
            k
            for k, owner in self._borrowed_indices.items()
            if isinstance(owner, _ConsumedSlotMarker)
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
        # SliceLinearityCheckPass picks up the violation post-fold.
        slice_parent = getattr(self, "_slice_parent", None)
        covered = getattr(self, "_slice_covered_indices", None)
        if slice_parent is not None and covered is not None:
            parent_consumed = sorted(
                idx
                for idx in covered
                if isinstance(
                    slice_parent._borrowed_indices.get((f"const:{idx}",)),
                    _ConsumedSlotMarker,
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
                if isinstance(owner, _ConsumedSlotMarker)
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
        """Create a UInt from an integer index."""
        return UInt(
            value=Value(type=UIntType(), name=f"idx_{idx}").with_const(idx),
            init_value=idx,
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
            QubitConsumedError: If the array was consumed, the element is
                already borrowed (for quantum arrays), or the slot is
                currently owned by a ``VectorView`` slice.
        """
        indices_key = self._make_indices_key(indices)
        index_str = self._format_index(indices)

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
        # borrows, slice-held slots, AND the "physically consumed"
        # sentinel that destructive view operations (measure / cast
        # on a sub-view) install.  The owner object tells us which
        # one so we can surface a tailored message.
        if indices_key in self._borrowed_indices and self.value.type.is_quantum():
            owner = self._borrowed_indices[indices_key]
            if isinstance(owner, _ConsumedSlotMarker):
                raise QubitConsumedError(
                    f"Physical qubit '{self.value.name}[{index_str}]' was already "
                    f"consumed by a destructive operation (e.g. measure / cast) "
                    f"on a view covering this slot.",
                    handle_name=f"{self.value.name}[{index_str}]",
                    operation_name="array element access",
                )
            if isinstance(owner, ArrayBase):
                raise QubitConsumedError(
                    f"Parent slot '{self.value.name}[{index_str}]' is currently "
                    f"held by a VectorView slice.\n"
                    f"Access it through the view, or let the view finish "
                    f"before touching the parent directly.",
                    handle_name=f"{self.value.name}[{index_str}]",
                    operation_name="array element access",
                )
            raise QubitConsumedError(
                f"Array element '{self.value.name}[{index_str}]' is already borrowed.\n"
                f"Return it before borrowing again.\n\n"
                f"Fix:\n"
                f"  q = {self.value.name}[{index_str}]\n"
                f"  q = qm.h(q)\n"
                f"  {self.value.name}[{index_str}] = q  # Return the element first",
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

        # Release the borrow.  ``_ConsumedSlotMarker`` entries are not
        # outstanding borrows — they record physically-destroyed slots
        # — so they must never be deleted by a non-destructive return
        # path.  In practice ``_get_element`` already rejects access to
        # consumed slots, so a borrow couldn't have been issued for one;
        # the guard here is defense-in-depth against future paths that
        # might bypass element access (e.g. computed-index returns).
        if release_key in self._borrowed_indices:
            current = self._borrowed_indices[release_key]
            if not isinstance(current, _ConsumedSlotMarker):
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

        First opportunistically releases any slice-owner entries whose
        view is itself drained (no outstanding element borrows on the
        view): this corresponds to "the view is finished using its
        elements" and lets patterns like ``q[0::2] → loop → measure(q)``
        complete without the user explicitly releasing the view.  Only
        after that does the method raise if any borrows remain, formatting
        direct-borrow and view-owned entries with distinct labels.

        Raises:
            UnreturnedBorrowError: If any elements are still borrowed,
                either directly or by a view whose own borrows have not
                all been returned.
        """
        if not self.value.type.is_quantum():
            return

        # Opportunistically release slice-borrows whose owning view has
        # no outstanding element borrows (view is drained).  ``Vector``
        # is a dataclass with ``__hash__ = None``, so deduplicate views
        # by ``id`` rather than by set membership.
        slice_entries = {
            k: v for k, v in self._borrowed_indices.items() if isinstance(v, ArrayBase)
        }
        if slice_entries:
            unique_views: dict[int, ArrayBase[T]] = {}
            for v in slice_entries.values():
                unique_views.setdefault(id(v), v)
            drained_view_ids = {
                vid for vid, v in unique_views.items() if not v._borrowed_indices
            }
            if drained_view_ids:
                to_remove = [
                    k for k, v in slice_entries.items() if id(v) in drained_view_ids
                ]
                for k in to_remove:
                    del self._borrowed_indices[k]

        # ``_ConsumedSlotMarker`` entries are not outstanding borrows
        # — they record physically-destroyed slots — so they must be
        # excluded from the "unreturned borrows" report and from the
        # "any entries left?" check.
        outstanding_entries = {
            k: v
            for k, v in self._borrowed_indices.items()
            if not isinstance(v, _ConsumedSlotMarker)
        }
        if not outstanding_entries:
            return

        borrowed_strs: list[str] = []
        for key, owner in outstanding_entries.items():
            if isinstance(owner, ArrayBase):
                # Slice-held slot: key is a single-element tuple like
                # ``("const:<idx>",)``; surface the slot number directly.
                idx_str = key[0].split(":", 1)[1]
                borrowed_strs.append(
                    f"{self.value.name}[{idx_str}] (held by slice view)"
                )
            else:
                index_str = self._format_index(owner)
                borrowed_strs.append(f"{self.value.name}[{index_str}]")

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
    _borrowed_indices: dict[tuple[str, ...], tuple[UInt, ...]] = dataclasses.field(
        default_factory=dict
    )

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
                negative start/stop values, which are not yet supported.
        """
        if isinstance(index, slice):
            return self._make_slice_view(index)
        if isinstance(index, int):
            index = self._make_uint_index(index)
        return self._get_element((index,))

    def __setitem__(self, index: int | UInt, value: T) -> None:
        """Set element at the given index.

        Raises:
            TypeError: If ``index`` is a ``slice`` — slice assignment is
                not supported; iterate over the view and assign elements
                one at a time instead.
        """
        if isinstance(index, slice):
            raise TypeError(
                "Slice assignment is not supported on Vector. "
                "Iterate over the view and assign elements individually:\n"
                "  view = q[0::2]\n"
                "  for i in qmc.range(view.shape[0]):\n"
                "      view[i] = qmc.h(view[i])"
            )
        if isinstance(index, int):
            index = self._make_uint_index(index)
        self._return_element((index,), value)

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
    _borrowed_indices: dict[tuple[str, ...], tuple[UInt, ...]] = dataclasses.field(
        default_factory=dict
    )

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
    _borrowed_indices: dict[tuple[str, ...], tuple[UInt, ...]] = dataclasses.field(
        default_factory=dict
    )

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
        value: Either a raw Python ``int`` or a ``UInt`` handle whose
            backing ``Value`` may or may not carry a constant.

    Returns:
        The int when ``value`` is an ``int`` directly, or a ``UInt``
        whose ``.value`` is constant and resolvable to an ``int``.
        ``None`` otherwise (i.e. when ``value`` is a symbolic ``UInt``).
    """
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

    a_uint = a if isinstance(a, UInt) else UInt(
        value=Value(type=UIntType(), name="uint_const").with_const(int(a)),
        init_value=int(a),
    )
    b_uint = b if isinstance(b, UInt) else UInt(
        value=Value(type=UIntType(), name="uint_const").with_const(int(b)),
        init_value=int(b),
    )
    result = UInt(value=Value(type=UIntType(), name="uint_min"), init_value=0)
    _emit_binop(a_uint.value, b_uint.value, result.value, BinOpKind.MIN)
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
        ``QubitConsumedError``.  The parent's ``validate_all_returned``
        opportunistically releases the bulk-borrow once the view has no
        outstanding element borrows, so normal usage — slice, loop,
        ``return q`` / ``measure(q)`` — does not require manual release.

        Symbolic slices (``q[lo:hi]`` with ``lo``/``hi`` ``UInt``) cannot
        enumerate their covered slots at trace time and therefore skip
        the bulk-borrow here; ``SliceLinearityCheckPass`` picks them up
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
            return q
        ```
    """

    _slice_parent: "Vector[T]"
    _slice_start: UInt
    _slice_step: UInt
    _slice_covered_indices: tuple[int, ...] | None

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
            QubitConsumedError: If any covered parent slot is already
                held by another slice view or currently borrowed.
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

        if covered_indices is not None and parent.value.type.is_quantum():
            # Opportunistically drain views that already own the
            # target slots but have no outstanding element borrows.
            # This lets sequential same-range slicing (``a = q[0::2];
            # loop(a); b = q[0::2]``) succeed — the first view is
            # effectively done by the time the second is created,
            # even though no explicit ``consume`` happened.
            views_to_drain: list[ArrayBase] = []
            for idx in covered_indices:
                key = (f"const:{idx}",)
                existing = parent._borrowed_indices.get(key)
                if (
                    isinstance(existing, ArrayBase)
                    and not existing._borrowed_indices
                    and existing not in views_to_drain
                ):
                    views_to_drain.append(existing)
            for drained in views_to_drain:
                drained_covered = getattr(drained, "_slice_covered_indices", None)
                if drained_covered is None:
                    continue
                for drained_idx in drained_covered:
                    drained_key = (f"const:{drained_idx}",)
                    if parent._borrowed_indices.get(drained_key) is drained:
                        del parent._borrowed_indices[drained_key]

            for idx in covered_indices:
                key = (f"const:{idx}",)
                existing = parent._borrowed_indices.get(key)
                if existing is not None:
                    if isinstance(existing, ArrayBase):
                        raise QubitConsumedError(
                            f"Parent slot '{parent.value.name}[{idx}]' is already "
                            f"owned by another slice view; overlapping views are "
                            f"not supported.",
                            handle_name=f"{parent.value.name}[{idx}]",
                            operation_name="array slicing",
                        )
                    raise QubitConsumedError(
                        f"Cannot slice across '{parent.value.name}[{idx}]' — "
                        f"it is currently borrowed.  Return the borrowed "
                        f"element before slicing.",
                        handle_name=f"{parent.value.name}[{idx}]",
                        operation_name="array slicing",
                    )
                parent._borrowed_indices[key] = instance

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

    def __setitem__(self, index: int | UInt, value: T) -> None:
        """Return a borrowed element to its view slot.

        Args:
            index: View-local index of the slot being returned.
            value: Handle previously borrowed from the view.

        Raises:
            TypeError: If ``index`` is a ``slice``.
        """
        if isinstance(index, slice):
            raise TypeError(
                "Slice assignment is not supported on VectorView. "
                "Iterate and assign elements individually instead."
            )
        super().__setitem__(index, value)

    def consume(self, operation_name: str = "unknown") -> typing.Self:
        """Consume the view and release its parent slice-borrows.

        Validates that every view-local borrow has been returned, then
        releases the parent's record of the bulk-borrow so the parent
        can be consumed/returned normally once the view is done (e.g.
        passed to another ``@qkernel`` which consumes its argument).

        For a destructive consume (``measure``, ``cast``) the slot
        marker is installed on the parent's borrow table
        *unconditionally* — even if the parent currently records
        ownership against a *different* view of the same slots.  That
        situation arises from the opportunistic drain when a second
        same-range view is sliced before the first has been used; the
        first view's slots are silently transferred to the second view
        in the parent's table.  Without the unconditional mark a later
        ``measure`` on the original view would walk the slot loop, see
        ``owner is self`` is False, and silently emit nothing — leaving
        the second view to reach the backend over destroyed qubits.

        Args:
            operation_name: Name of the operation consuming this view
                (used in error messages).

        Returns:
            A fresh view handle with the same backing state; the
            returned view no longer holds the parent's slice-borrow.

        Raises:
            QubitConsumedError: If any covered slot was already
                destroyed by a prior destructive view consume on an
                overlapping view.
        """
        self.validate_all_returned()
        is_destructive = operation_name in _DESTRUCTIVE_CONSUME_OPS

        # Reject the consume up-front when an overlapping view has
        # already destroyed any of our covered slots.  Without this the
        # second destructive consume would either silently no-op (if
        # ``owner is self`` is False due to drain transfer) or quietly
        # reinstall the marker — neither path tells the user about the
        # double-consume.
        if self._slice_covered_indices is not None:
            already_destroyed = sorted(
                idx
                for idx in self._slice_covered_indices
                if isinstance(
                    self._slice_parent._borrowed_indices.get((f"const:{idx}",)),
                    _ConsumedSlotMarker,
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

        if self._slice_covered_indices is not None:
            for idx in self._slice_covered_indices:
                key = (f"const:{idx}",)
                owner = self._slice_parent._borrowed_indices.get(key)
                if is_destructive:
                    # Always install the consumed sentinel — overriding
                    # any other view that the parent currently records
                    # as owner — so a subsequent operation on either
                    # view sees the destroyed slot via the parent's
                    # table.
                    self._slice_parent._borrowed_indices[key] = _CONSUMED_SLOT
                elif owner is self:
                    del self._slice_parent._borrowed_indices[key]
        new_view = super().consume(operation_name)
        new_view._slice_parent = self._slice_parent
        new_view._slice_start = self._slice_start
        new_view._slice_step = self._slice_step
        new_view._slice_covered_indices = None  # parent already released
        return new_view

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

        return VectorView._wrap(
            parent=self._slice_parent,
            sliced_av=new_sliced_av,
            length=new_length,
            start_uint=new_start,
            step_uint=new_step,
            covered_indices=nested_covered,
        )
