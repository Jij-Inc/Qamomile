from __future__ import annotations

import dataclasses
import typing
import uuid
from typing import Generic, Iterator, TypeVar, overload

from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.operation import QInitOperation, CInitOperation
from qamomile.circuit.ir.types import ValueType
from qamomile.circuit.ir.types.primitives import BitType, FloatType, QubitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.errors import QubitConsumedError, UnreturnedBorrowError

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
            Value(type=UIntType(), name=f"dim_{i}", params={"const": dim})
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
    ) -> "ArrayBase[T]":
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
        instance.element_type = type_map[value.type]
        return instance

    @property
    def shape(self) -> tuple[int | UInt, ...]:
        """Return the shape of the array."""
        return self._shape

    def _make_uint_index(self, idx: int) -> UInt:
        """Create a UInt from an integer index."""
        return UInt(
            value=Value(type=UIntType(), name=f"idx_{idx}", params={"const": idx}),
            init_value=idx,
        )

    def _format_index(self, indices: tuple[UInt, ...]) -> str:
        """Format indices for element naming and parameter tracking."""
        parts = []
        for idx in indices:
            if idx.value.is_constant():
                parts.append(str(int(idx.value.params["const"])))
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
                key_parts.append(f"const:{idx.value.params['const']}")
            else:
                # Use the value's uuid for symbolic indices
                key_parts.append(f"sym:{idx.value.uuid}")
        return tuple(key_parts)

    def _get_element(self, indices: tuple[UInt, ...]) -> T:
        """Get an element at the given indices.

        Raises:
            QubitConsumedError: If this element is already borrowed (for quantum arrays).
        """
        indices_key = self._make_indices_key(indices)
        index_str = self._format_index(indices)

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

        params: dict[str, int | float] = {}
        if self.value.is_parameter():
            param_name = self.value.parameter_name()
            element_param_name = f"{param_name}[{index_str}]"
            params = {"parameter": element_param_name}  # type: ignore

        element_value = Value(
            type=self.value.type,
            name=f"{self.value.name}[{index_str}]",
            parent_array=self.value,
            element_indices=tuple(idx.value for idx in indices),
            params=params,
        )
        return self.element_type(value=element_value, parent=self, indices=indices)

    def _set_element(self, indices: tuple[UInt, ...]) -> None:
        """Mark an element as returned (no longer borrowed)."""
        indices_key = self._make_indices_key(indices)
        if indices_key in self._borrowed_indices:
            self._borrowed_indices.pop(indices_key)

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
        import qamomile as qm

        # Create a vector of 3 qubits
        qubits: qm.Vector[qm.Qubit] = qm.Vector(size=3)

        # Access elements
        q0 = qubits[0]
        q0 = qm.h(q0)
        qubits[0] = q0

        # Iterate over elements
        for q in qubits:
            q = qm.h(q)
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

    def __getitem__(self, index: int | UInt) -> T:
        """Get element at the given index."""
        if isinstance(index, int):
            index = self._make_uint_index(index)
        return self._get_element((index,))

    def __setitem__(self, index: int | UInt, value: T) -> None:
        """Set element at the given index."""
        if isinstance(index, int):
            index = self._make_uint_index(index)
        self._set_element((index,))

    def __iter__(self) -> Iterator[T]:
        """Iterate over elements of the vector."""
        size = self._shape[0]
        if isinstance(size, UInt):
            raise TypeError("Cannot iterate over vector with symbolic size")
        for i in range(size):
            yield self[i]


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
        self._set_element((i, j))


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
        self._set_element(converted)
