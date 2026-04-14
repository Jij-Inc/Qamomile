"""Value types and typed metadata for the Qamomile IR."""

from __future__ import annotations

import dataclasses
import typing
import uuid
from collections.abc import Mapping, Sequence

from .types import DictType, ValueType

if typing.TYPE_CHECKING:
    from .types.primitives import TupleType

T = typing.TypeVar("T", bound=ValueType)
_UNSET = object()


def _freeze_data(value: typing.Any) -> typing.Any:
    """Recursively convert runtime data to immutable containers.

    Dicts become ``tuple[tuple[K, V], ...]``, lists/tuples become
    ``tuple[V, ...]``.  Note: ``thaw_data`` cannot distinguish a
    frozen dict from a frozen list of 2-element tuples; callers that
    need round-trip fidelity should track the original type externally.
    """
    if isinstance(value, Mapping):
        return tuple((_freeze_data(k), _freeze_data(v)) for k, v in value.items())
    if isinstance(value, tuple):
        return tuple(_freeze_data(v) for v in value)
    if isinstance(value, list):
        return tuple(_freeze_data(v) for v in value)
    return value


@dataclasses.dataclass(frozen=True)
class ScalarMetadata:
    """Metadata for scalar constants and symbolic parameters."""

    const_value: int | float | bool | None = None
    parameter_name: str | None = None


@dataclasses.dataclass(frozen=True)
class CastMetadata:
    """Metadata describing a cast carrier and its underlying qubits."""

    source_uuid: str
    qubit_uuids: tuple[str, ...]
    source_logical_id: str | None = None
    qubit_logical_ids: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class QFixedMetadata:
    """Metadata for QFixed carriers."""

    qubit_uuids: tuple[str, ...]
    num_bits: int
    int_bits: int


@dataclasses.dataclass(frozen=True)
class ArrayRuntimeMetadata:
    """Metadata for array literals and explicit element identity tracking."""

    const_array: typing.Any = None
    element_uuids: tuple[str, ...] = ()
    element_logical_ids: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class DictRuntimeMetadata:
    """Metadata for transpile-time bound dict values."""

    bound_data: tuple[tuple[typing.Any, typing.Any], ...] = ()


@dataclasses.dataclass(frozen=True)
class ValueMetadata:
    """Typed metadata owned by the compiler/runtime."""

    scalar: ScalarMetadata | None = None
    cast: CastMetadata | None = None
    qfixed: QFixedMetadata | None = None
    array_runtime: ArrayRuntimeMetadata | None = None
    dict_runtime: DictRuntimeMetadata | None = None


@typing.runtime_checkable
class ValueBase(typing.Protocol):
    """Protocol for IR values with typed metadata.

    Attributes are declared as read-only properties to match frozen
    dataclass fields in concrete implementations (Value, ArrayValue, etc.).
    """

    @property
    def uuid(self) -> str: ...
    @property
    def logical_id(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def metadata(self) -> ValueMetadata: ...

    def next_version(self) -> ValueBase: ...
    def is_parameter(self) -> bool: ...
    def parameter_name(self) -> str | None: ...
    def is_constant(self) -> bool: ...
    def get_const(self) -> int | float | bool | None: ...


ValueLike: typing.TypeAlias = "Value | ArrayValue | TupleValue | DictValue"


class _MetadataValueMixin:
    """Shared metadata accessors for all IR values."""

    metadata: ValueMetadata

    def _replace_metadata(self, metadata: ValueMetadata) -> typing.Self:
        return typing.cast(
            typing.Self,
            dataclasses.replace(typing.cast(typing.Any, self), metadata=metadata),
        )

    def is_parameter(self) -> bool:
        return (
            self.metadata.scalar is not None
            and self.metadata.scalar.parameter_name is not None
        )

    def parameter_name(self) -> str | None:
        if self.metadata.scalar is None:
            return None
        return self.metadata.scalar.parameter_name

    def is_constant(self) -> bool:
        return (
            self.metadata.scalar is not None
            and self.metadata.scalar.const_value is not None
        )

    def get_const(self) -> int | float | bool | None:
        if self.metadata.scalar is None:
            return None
        return self.metadata.scalar.const_value

    def with_const(self, const_value: int | float | bool) -> typing.Self:
        return self._replace_metadata(
            dataclasses.replace(
                self.metadata, scalar=ScalarMetadata(const_value=const_value)
            )
        )

    def with_parameter(self, parameter_name: str) -> typing.Self:
        return self._replace_metadata(
            dataclasses.replace(
                self.metadata,
                scalar=ScalarMetadata(parameter_name=parameter_name),
            )
        )

    def is_cast_result(self) -> bool:
        return self.metadata.cast is not None

    def get_cast_source_uuid(self) -> str | None:
        return self.metadata.cast.source_uuid if self.metadata.cast else None

    def get_cast_source_logical_id(self) -> str | None:
        return self.metadata.cast.source_logical_id if self.metadata.cast else None

    def get_cast_qubit_uuids(self) -> tuple[str, ...] | None:
        if self.metadata.cast is None:
            return None
        return self.metadata.cast.qubit_uuids

    def get_cast_qubit_logical_ids(self) -> tuple[str, ...] | None:
        if self.metadata.cast is None or not self.metadata.cast.qubit_logical_ids:
            return None
        return self.metadata.cast.qubit_logical_ids

    def with_cast_metadata(
        self,
        source_uuid: str,
        qubit_uuids: Sequence[str],
        source_logical_id: str | None = None,
        qubit_logical_ids: Sequence[str] | None = None,
    ) -> typing.Self:
        return self._replace_metadata(
            dataclasses.replace(
                self.metadata,
                cast=CastMetadata(
                    source_uuid=source_uuid,
                    source_logical_id=source_logical_id,
                    qubit_uuids=tuple(qubit_uuids),
                    qubit_logical_ids=tuple(qubit_logical_ids or ()),
                ),
            )
        )

    def get_qfixed_qubit_uuids(self) -> tuple[str, ...]:
        if self.metadata.qfixed is None:
            return ()
        return self.metadata.qfixed.qubit_uuids

    def get_qfixed_num_bits(self) -> int | None:
        return self.metadata.qfixed.num_bits if self.metadata.qfixed else None

    def get_qfixed_int_bits(self) -> int | None:
        return self.metadata.qfixed.int_bits if self.metadata.qfixed else None

    def with_qfixed_metadata(
        self,
        qubit_uuids: Sequence[str],
        num_bits: int,
        int_bits: int,
    ) -> typing.Self:
        return self._replace_metadata(
            dataclasses.replace(
                self.metadata,
                qfixed=QFixedMetadata(
                    qubit_uuids=tuple(qubit_uuids),
                    num_bits=num_bits,
                    int_bits=int_bits,
                ),
            )
        )

    def get_const_array(self) -> typing.Any:
        if self.metadata.array_runtime is None:
            return None
        return self.metadata.array_runtime.const_array

    def get_element_uuids(self) -> tuple[str, ...]:
        if self.metadata.array_runtime is None:
            return ()
        return self.metadata.array_runtime.element_uuids

    def with_array_runtime_metadata(
        self,
        *,
        const_array: typing.Any = _UNSET,
        element_uuids: Sequence[str] | object = _UNSET,
        element_logical_ids: Sequence[str] | object = _UNSET,
    ) -> typing.Self:
        current = self.metadata.array_runtime or ArrayRuntimeMetadata()
        new_const_array = (
            current.const_array if const_array is _UNSET else _freeze_data(const_array)
        )
        new_element_uuids = (
            current.element_uuids
            if element_uuids is _UNSET
            else tuple(typing.cast(Sequence[str], element_uuids))
        )
        new_element_logical_ids = (
            current.element_logical_ids
            if element_logical_ids is _UNSET
            else tuple(typing.cast(Sequence[str], element_logical_ids))
        )
        return self._replace_metadata(
            dataclasses.replace(
                self.metadata,
                array_runtime=ArrayRuntimeMetadata(
                    const_array=new_const_array,
                    element_uuids=new_element_uuids,
                    element_logical_ids=new_element_logical_ids,
                ),
            )
        )

    def get_bound_data_items(self) -> tuple[tuple[typing.Any, typing.Any], ...]:
        if self.metadata.dict_runtime is None:
            return ()
        return self.metadata.dict_runtime.bound_data

    def get_bound_data(self) -> dict[typing.Any, typing.Any]:
        return dict(self.get_bound_data_items())

    def with_dict_runtime_metadata(
        self,
        bound_data: Mapping[typing.Any, typing.Any],
    ) -> typing.Self:
        frozen_items = typing.cast(
            tuple[tuple[typing.Any, typing.Any], ...],
            _freeze_data(bound_data),
        )
        return self._replace_metadata(
            dataclasses.replace(
                self.metadata,
                dict_runtime=DictRuntimeMetadata(bound_data=frozen_items),
            )
        )


@dataclasses.dataclass(frozen=True)
class Value(_MetadataValueMixin, typing.Generic[T]):
    """A typed SSA value in the IR."""

    type: T
    name: str
    version: int = 0
    metadata: ValueMetadata = dataclasses.field(default_factory=ValueMetadata)
    uuid: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    logical_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    parent_array: ArrayValue | None = None
    element_indices: tuple[Value, ...] = ()

    def next_version(self) -> Value[T]:
        """Create a new Value with incremented version and fresh UUID.

        Metadata is intentionally preserved across versions so that
        parameter bindings and constant annotations remain accessible
        after quantum gate applications.  The ``logical_id`` stays the
        same to track physical qubit identity across SSA versions.
        """
        return Value(
            type=self.type,
            name=self.name,
            version=self.version + 1,
            metadata=self.metadata,
            uuid=str(uuid.uuid4()),
            logical_id=self.logical_id,
            parent_array=self.parent_array,
            element_indices=self.element_indices,
        )

    def is_array_element(self) -> bool:
        return self.parent_array is not None


@dataclasses.dataclass(frozen=True)
class ArrayValue(Value[T]):
    """An array of typed IR values."""

    type: T
    name: str
    uuid: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    logical_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    shape: tuple[Value, ...] = dataclasses.field(default_factory=tuple)
    metadata: ValueMetadata = dataclasses.field(default_factory=ValueMetadata)

    def next_version(self) -> ArrayValue[T]:
        return ArrayValue(
            type=self.type,
            name=self.name,
            version=self.version + 1,
            metadata=self.metadata,
            uuid=str(uuid.uuid4()),
            logical_id=self.logical_id,
            shape=self.shape,
        )


@dataclasses.dataclass(frozen=True)
class TupleValue(_MetadataValueMixin):
    """A tuple of IR values for structured data."""

    name: str
    elements: tuple[Value, ...] = dataclasses.field(default_factory=tuple)
    metadata: ValueMetadata = dataclasses.field(default_factory=ValueMetadata)
    uuid: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    logical_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def type(self) -> "TupleType":
        from qamomile.circuit.ir.types.primitives import TupleType

        return TupleType(element_types=tuple(e.type for e in self.elements))

    def next_version(self) -> TupleValue:
        return TupleValue(
            name=self.name,
            elements=self.elements,
            metadata=self.metadata,
            uuid=str(uuid.uuid4()),
            logical_id=self.logical_id,
        )

    def is_constant(self) -> bool:
        return all(isinstance(e, Value) and e.is_constant() for e in self.elements)


@dataclasses.dataclass(frozen=True)
class DictValue(_MetadataValueMixin):
    """A dictionary value stored as stable ordered entries."""

    name: str
    entries: tuple[tuple[TupleValue | Value, Value], ...] = dataclasses.field(
        default_factory=tuple
    )
    metadata: ValueMetadata = dataclasses.field(default_factory=ValueMetadata)
    uuid: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    logical_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))

    def next_version(self) -> DictValue:
        return DictValue(
            name=self.name,
            entries=self.entries,
            metadata=self.metadata,
            uuid=str(uuid.uuid4()),
            logical_id=self.logical_id,
        )

    @property
    def type(self) -> DictType:
        if self.entries:
            key, val = self.entries[0]
            return DictType(key_type=key.type, value_type=val.type)
        return DictType()

    def is_constant(self) -> bool:
        return all(v.is_constant() for _, v in self.entries)

    def __len__(self) -> int:
        return len(self.entries)
