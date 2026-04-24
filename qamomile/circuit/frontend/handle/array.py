from __future__ import annotations

import dataclasses
import typing
import uuid
from typing import Generic, Iterator, TypeVar, overload

from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.operation import CInitOperation, QInitOperation
from qamomile.circuit.ir.types import ValueType
from qamomile.circuit.ir.types.primitives import BitType, FloatType, QubitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.errors import (
    AffineTypeError,
    QubitConsumedError,
    UnreturnedBorrowError,
)

from .handle import Handle
from .primitives import Bit, Float, Qubit, UInt

T = TypeVar("T", bound=Handle)


@dataclasses.dataclass
class ArrayBase(Handle, Generic[T]):
    """Base class for array types (Vector, Matrix, Tensor).

    Provides common functionality for array indexing and element access.
    """

    value: ArrayValue
    element_type: typing.Type[T] = dataclasses.field(init=False)
    _shape: tuple[int | UInt, ...] = dataclasses.field(default_factory=tuple)
    _borrowed_indices: dict[tuple[str, ...], tuple[UInt, ...]] = dataclasses.field(
        default_factory=dict
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

    def consume(self, operation_name: str = "unknown") -> typing.Self:
        """Consume the array, enforcing borrow-return contract for quantum arrays.

        For quantum arrays, all borrowed elements must be returned before the
        array can be consumed. This ensures that no unreturned borrows are
        silently discarded by operations like qkernel calls or controlled gates.
        """
        self.validate_all_returned()
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
            QubitConsumedError: If the array was consumed or the element is
                already borrowed (for quantum arrays).
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

        # Check if already borrowed (only enforce for quantum types)
        if indices_key in self._borrowed_indices and self.value.type.is_quantum():
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

        # Release the borrow
        if release_key in self._borrowed_indices:
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

        This method is useful for ensuring that all borrowed elements
        have been properly written back before using the array in
        operations that require the entire array.

        Raises:
            UnreturnedBorrowError: If any elements are still borrowed.
        """
        if not self._borrowed_indices or not self.value.type.is_quantum():
            return

        # Format borrowed indices for error message
        borrowed_strs = []
        for indices in self._borrowed_indices.values():
            index_str = self._format_index(indices)
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

        Normalizes ``None`` start/stop/step to ``0``, the parent length,
        and ``1`` respectively, converts integer bounds to ``UInt``
        handles, and pre-computes the view length (as a Python ``int``
        when all bounds are compile-time constants, otherwise as a
        symbolic ``UInt`` expression).

        Args:
            s: The Python ``slice`` object from ``vec[slice]``.

        Returns:
            A ``VectorView`` that delegates element access to this
            vector via the affine map ``view[i] -> parent[start + step*i]``.

        Raises:
            NotImplementedError: For non-positive ``step`` or any negative
                ``start``/``stop`` value.  Negative indices and reverse
                strides are not yet supported.
        """
        parent_length = self._shape[0]

        raw_start = 0 if s.start is None else s.start
        raw_stop = parent_length if s.stop is None else s.stop
        raw_step = 1 if s.step is None else s.step

        for name, value in (("start", s.start), ("stop", s.stop), ("step", s.step)):
            if isinstance(value, int) and value < 0:
                raise NotImplementedError(
                    f"Negative {name} is not supported for Vector slicing "
                    f"(got {name}={value}).  Use a non-negative value or "
                    f"compute the index explicitly."
                )
        if isinstance(raw_step, int) and raw_step <= 0:
            raise NotImplementedError(
                f"Vector slicing requires a positive step (got step={raw_step}). "
                f"Reverse/zero strides are not yet supported."
            )

        start_uint = self._coerce_index(raw_start)
        stop_uint = self._coerce_index(raw_stop)
        step_uint = self._coerce_index(raw_step)
        del stop_uint

        length = _compute_slice_length(raw_start, raw_stop, raw_step)

        return VectorView(
            parent=self,
            start=start_uint,
            step=step_uint,
            length=length,
            element_type=self.element_type,
        )

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

    When all three bounds are Python ``int`` constants, returns a concrete
    ``int``.  Otherwise falls back to symbolic ``UInt`` arithmetic, which
    emits ``BinOp`` nodes into the current tracer and relies on constant
    folding in later passes to collapse the expression where possible.

    Args:
        start: Slice start (inclusive).
        stop: Slice stop (exclusive).
        step: Slice step; must be a positive value.

    Returns:
        ``int`` when all inputs are concrete integers, otherwise a ``UInt``
        handle representing the slice length.
    """
    if isinstance(start, int) and isinstance(stop, int) and isinstance(step, int):
        if stop <= start:
            return 0
        return (stop - start + step - 1) // step

    return ((stop - start) + (step - 1)) // step


@dataclasses.dataclass
class VectorView(Generic[T]):
    """Lightweight view over a strided slice of a ``Vector``.

    Produced by slicing a ``Vector`` (e.g. ``q[1::2]``), a ``VectorView``
    behaves like a ``Vector`` for indexed element access but owns no IR
    state of its own: every element access is translated by the affine
    map ``view[i] -> parent[start + step * i]`` and dispatched through
    the parent's ``_get_element`` / ``_return_element``.  Borrow tracking
    is therefore shared with the parent, so a qubit borrowed through the
    view is tracked against the same parent slot as if it had been
    borrowed directly.

    Because the view has no standalone IR value, it cannot be passed as a
    single argument to another ``@qkernel``; iterate over ``view[i]`` and
    pass individual element handles instead.

    Attributes:
        parent: The backing ``Vector`` from which elements are borrowed.
        start: Parent-space index of the first element covered.
        step: Stride in parent-space; always a positive ``UInt`` handle.
        length: Number of elements in the view, as a Python ``int`` when
            it is known at trace time or as a ``UInt`` handle otherwise.
        element_type: Element type forwarded from the parent.

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

    parent: "Vector[T]"
    start: UInt
    step: UInt
    length: int | UInt
    element_type: typing.Type[T]

    @property
    def shape(self) -> tuple[int | UInt]:
        """Return the view's shape as a one-element tuple.

        Returns:
            ``(length,)`` — matches the ``Vector.shape`` convention so the
            view is a drop-in replacement inside ``.shape[0]`` expressions
            and ``qmc.range(view.shape[0])`` loops.
        """
        return (self.length,)

    def _translate(self, index: int | UInt) -> UInt:
        """Map a view-local index to the corresponding parent index.

        Args:
            index: View-local index (``int`` or ``UInt``).

        Returns:
            ``UInt`` handle holding ``start + step * index`` suitable for
            ``parent._get_element`` / ``parent._return_element``.
        """
        if isinstance(index, int):
            idx_uint = self.parent._make_uint_index(index)
        else:
            idx_uint = index

        offset: UInt = self.step * idx_uint
        return self.start + offset

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: UInt) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> "VectorView[T]": ...

    def __getitem__(self, index: int | UInt | slice) -> "T | VectorView[T]":
        """Get element at the given view-local index, or nest a slice.

        Args:
            index: View-local index or ``slice``.  A nested slice is
                composed into the parent via ``start + step * sub_start``
                (and likewise for ``step``), so borrow tracking continues
                to resolve against the root parent.

        Returns:
            A single element handle for integer indices, or a further
            ``VectorView`` for slice indices.

        Raises:
            NotImplementedError: For nested slices with non-positive step
                or negative bounds.
        """
        if isinstance(index, slice):
            return self._nested_slice(index)
        parent_index = self._translate(index)
        return self.parent._get_element((parent_index,))

    def __setitem__(self, index: int | UInt, value: T) -> None:
        """Return a borrowed element to its parent slot via the view.

        Args:
            index: View-local index of the slot being returned.
            value: Handle previously borrowed from the view or a fresh
                element of the correct type.

        Raises:
            TypeError: If ``index`` is a ``slice`` — bulk slice assignment
                is not supported.
        """
        if isinstance(index, slice):
            raise TypeError(
                "Slice assignment is not supported on VectorView. "
                "Iterate and assign elements individually instead."
            )
        parent_index = self._translate(index)
        self.parent._return_element((parent_index,), value)

    def _nested_slice(self, s: slice) -> "VectorView[T]":
        """Compose a nested slice into a single view over the same parent.

        Normalizes ``None`` start/stop/step against this view's length and
        collapses the nested affine map ``(start + step * (inner_start +
        inner_step * j))`` back to the root parent so the result is still
        a one-hop view.

        Args:
            s: The Python ``slice`` object applied to this view.

        Returns:
            A new ``VectorView`` whose parent is this view's parent.

        Raises:
            NotImplementedError: For non-positive step or negative bounds.
        """
        view_length = self.length

        raw_start = 0 if s.start is None else s.start
        raw_stop = view_length if s.stop is None else s.stop
        raw_step = 1 if s.step is None else s.step

        for name, value in (("start", s.start), ("stop", s.stop), ("step", s.step)):
            if isinstance(value, int) and value < 0:
                raise NotImplementedError(
                    f"Negative {name} is not supported for VectorView slicing "
                    f"(got {name}={value})."
                )
        if isinstance(raw_step, int) and raw_step <= 0:
            raise NotImplementedError(
                f"VectorView slicing requires a positive step (got step={raw_step})."
            )

        inner_start = self.parent._coerce_index(raw_start)
        inner_step = self.parent._coerce_index(raw_step)

        new_start: UInt = self.start + self.step * inner_start
        new_step: UInt = self.step * inner_step
        new_length = _compute_slice_length(raw_start, raw_stop, raw_step)

        return VectorView(
            parent=self.parent,
            start=new_start,
            step=new_step,
            length=new_length,
            element_type=self.element_type,
        )

    def __iter__(self) -> Iterator[T]:
        """Direct iteration over VectorView is prohibited.

        The rationale matches ``Vector.__iter__``: iterating the view
        cannot propagate in-place writes back to the parent and would
        silently drop borrows.  Use explicit index-based loops.

        Raises:
            TypeError: Always raised.
        """
        raise TypeError(
            "Direct iteration over VectorView is not supported in @qkernel "
            "functions.  Use explicit index-based iteration:\n"
            "  for i in qmc.range(view.shape[0]):\n"
            "      view[i] = qmc.operation(view[i])"
        )
