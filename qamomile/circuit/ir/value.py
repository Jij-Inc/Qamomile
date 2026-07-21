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
    """Metadata for array literals and explicit element identity tracking.

    ``element_parent_uuids`` / ``element_parent_indices`` are parallel to
    ``element_uuids``: for each tracked element they record the root array's
    UUID and the element's index within that root (as resolved by
    :func:`resolve_root_qubit_address` at trace time). They let an emit pass map
    a packed element back to the physical qubit registered under the root
    array's ``QubitAddress(root_uuid, index)`` key even when the element's own
    UUID was never registered. The sentinel ``("", -1)`` marks an element with
    no array parent (a standalone qubit), for which a flat UUID lookup is used.
    """

    const_array: typing.Any = None
    element_uuids: tuple[str, ...] = ()
    element_logical_ids: tuple[str, ...] = ()
    element_parent_uuids: tuple[str, ...] = ()
    element_parent_indices: tuple[int, ...] = ()


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


def split_indexed_identifier(identifier: str) -> tuple[str, str] | None:
    """Split a legacy indexed identifier into base and index suffix.

    Args:
        identifier (str): Identifier to inspect. Legacy carrier keys use the
            ``"<base>_<index>"`` spelling, where ``index`` is decimal.

    Returns:
        tuple[str, str] | None: ``(base, index)`` when ``identifier`` carries a
            numeric suffix, otherwise ``None``.
    """
    separator = identifier.rfind("_")
    if separator <= 0:
        return None
    suffix = identifier[separator + 1 :]
    if not suffix.isdigit():
        return None
    return identifier[:separator], suffix


def remap_indexed_identifier(
    identifier: str,
    remap_identifier: typing.Callable[[str], str],
) -> str:
    """Remap an identifier while preserving a legacy index suffix.

    Args:
        identifier (str): Scalar identifier or legacy ``"<base>_<index>"``
            carrier key.
        remap_identifier (typing.Callable[[str], str]): Function that remaps
            scalar identifiers and carrier-key bases.

    Returns:
        str: Remapped identifier. Numeric index suffixes are preserved after
            remapping the base identifier.
    """
    parts = split_indexed_identifier(identifier)
    if parts is None:
        return remap_identifier(identifier)
    base, suffix = parts
    return f"{remap_identifier(base)}_{suffix}"


def remap_value_metadata_references(
    metadata: ValueMetadata,
    remap_uuid: typing.Callable[[str], str],
    remap_logical_id: typing.Callable[[str], str],
) -> ValueMetadata:
    """Rewrite UUID and logical-id references inside value metadata.

    Args:
        metadata (ValueMetadata): Metadata bundle whose embedded references
            should be rewritten.
        remap_uuid (typing.Callable[[str], str]): Function that maps scalar
            UUID references (and carrier-key bases) to replacement UUIDs.
        remap_logical_id (typing.Callable[[str], str]): Function that maps
            scalar logical-id references (and carrier-key bases) to
            replacement logical IDs.

    Returns:
        ValueMetadata: Metadata with every embedded UUID / logical-id reference
            rewritten. Legacy ``"<uuid>_<index>"`` carrier keys keep their
            index suffix while remapping the base UUID. The original bundle is
            returned unchanged when no reference is rewritten.
    """
    new_cast = metadata.cast
    if new_cast is not None:
        new_cast = CastMetadata(
            source_uuid=remap_uuid(new_cast.source_uuid),
            qubit_uuids=tuple(
                remap_indexed_identifier(uuid_ref, remap_uuid)
                for uuid_ref in new_cast.qubit_uuids
            ),
            source_logical_id=(
                remap_logical_id(new_cast.source_logical_id)
                if new_cast.source_logical_id is not None
                else None
            ),
            qubit_logical_ids=tuple(
                remap_indexed_identifier(logical_id, remap_logical_id)
                for logical_id in new_cast.qubit_logical_ids
            ),
        )

    new_qfixed = metadata.qfixed
    if new_qfixed is not None:
        new_qfixed = QFixedMetadata(
            qubit_uuids=tuple(
                remap_indexed_identifier(uuid_ref, remap_uuid)
                for uuid_ref in new_qfixed.qubit_uuids
            ),
            num_bits=new_qfixed.num_bits,
            int_bits=new_qfixed.int_bits,
        )

    new_array_rt = metadata.array_runtime
    if new_array_rt is not None:
        new_array_rt = ArrayRuntimeMetadata(
            const_array=new_array_rt.const_array,
            element_uuids=tuple(
                remap_indexed_identifier(uuid_ref, remap_uuid)
                for uuid_ref in new_array_rt.element_uuids
            ),
            element_logical_ids=tuple(
                remap_indexed_identifier(logical_id, remap_logical_id)
                for logical_id in new_array_rt.element_logical_ids
            ),
            # Empty parent UUID is a sentinel for standalone or unresolved
            # elements, not a Value UUID, so keep it unchanged.
            element_parent_uuids=tuple(
                remap_uuid(uuid_ref) if uuid_ref else uuid_ref
                for uuid_ref in new_array_rt.element_parent_uuids
            ),
            element_parent_indices=new_array_rt.element_parent_indices,
        )

    if (
        new_cast == metadata.cast
        and new_qfixed == metadata.qfixed
        and new_array_rt == metadata.array_runtime
    ):
        return metadata

    return dataclasses.replace(
        metadata,
        cast=new_cast,
        qfixed=new_qfixed,
        array_runtime=new_array_rt,
    )


class ValueBase:
    """Nominal base for every typed IR value.

    Runtime compiler passes inspect values in their innermost loops. A nominal
    base keeps those checks constant-time; a runtime-checkable protocol would
    repeatedly scan the protocol members on Python versions that do not cache
    structural checks.
    """

    uuid: str
    logical_id: str
    name: str
    metadata: ValueMetadata

    if typing.TYPE_CHECKING:

        @property
        def type(self) -> ValueType:
            """Return the static IR type carried by this value.

            Returns:
                ValueType: Concrete scalar or container type.
            """
            raise NotImplementedError

    def next_version(self) -> ValueBase:
        """Create the next SSA version of this value.

        Returns:
            ValueBase: A value with a fresh version UUID and preserved logical
                identity.
        """
        raise NotImplementedError

    def is_parameter(self) -> bool:
        """Return whether this value represents a runtime parameter.

        Returns:
            bool: Whether parameter metadata is present.
        """
        raise NotImplementedError

    def parameter_name(self) -> str | None:
        """Return the public parameter name carried by this value.

        Returns:
            str | None: Parameter name, or ``None`` for a non-parameter value.
        """
        raise NotImplementedError

    def is_constant(self) -> bool:
        """Return whether this value carries a scalar constant.

        Returns:
            bool: Whether scalar constant metadata is present.
        """
        raise NotImplementedError

    def get_const(self) -> int | float | bool | None:
        """Return the scalar constant carried by this value.

        Returns:
            int | float | bool | None: Constant value, or ``None`` when the
                value is not constant.
        """
        raise NotImplementedError


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

    def get_element_parent_addresses(self) -> tuple[tuple[str, int] | None, ...]:
        """Return per-element root ``(array_uuid, index)`` addresses.

        Returns:
            tuple[tuple[str, int] | None, ...]: Exactly one entry per element
                in ``get_element_uuids()`` (same length, same order), so
                callers can index by element position without a length check.
                Each entry is the element's root ``(array_uuid, index)``
                address, or ``None`` for a standalone qubit (recorded with the
                ``("", -1)`` sentinel), for an element whose root could not be
                resolved at trace time, or for any element whose parent address
                was never recorded (e.g. metadata that only set
                ``element_uuids``).
        """
        if self.metadata.array_runtime is None:
            return ()
        rt = self.metadata.array_runtime
        # ``element_uuids`` / ``element_parent_uuids`` / ``element_parent_indices``
        # are PARALLEL tuples: position i in each describes the same element i.
        # The only code that populates the two parent tuples (``expval()``'s
        # tuple lowering) always sets all three together, so at the sole call
        # site they are equal length. We deliberately iterate by
        # ``len(element_uuids)`` instead of ``zip``-ing the parent tuples so the
        # result is ALWAYS one entry per element and stays index-aligned with
        # ``get_element_uuids()`` -- the caller (``_build_qubit_map``) indexes it
        # by element position. ``zip`` would instead stop at the shortest tuple
        # and return fewer (or zero) entries for any value whose parent tuples
        # were never set, silently misaligning the result with element_uuids.
        # Defensive only: an element is "past the recorded parent info" once its
        # index reaches the SHORTER of the two parent tuples. In practice the
        # parent tuples are equal length to element_uuids (expval sets all three
        # together), so this never triggers at the real call site -- it only
        # keeps the result aligned with element_uuids for any value whose parent
        # info was never recorded (those positions become None).
        n_parent = min(len(rt.element_parent_uuids), len(rt.element_parent_indices))
        result: list[tuple[str, int] | None] = []
        for i in range(len(rt.element_uuids)):
            if i >= n_parent:
                result.append(None)
                continue
            parent_uuid = rt.element_parent_uuids[i]
            parent_idx = rt.element_parent_indices[i]
            # ``("", -1)`` is the sentinel written by ``expval()`` for a
            # standalone qubit (no array parent) or an element whose root could
            # not be resolved at trace time; decode it back to ``None`` so the
            # caller skips the root-address fallback for that position.
            if parent_uuid == "" or parent_idx < 0:
                result.append(None)
            else:
                result.append((parent_uuid, parent_idx))
        return tuple(result)

    def with_array_runtime_metadata(
        self,
        *,
        const_array: typing.Any = _UNSET,
        element_uuids: Sequence[str] | object = _UNSET,
        element_logical_ids: Sequence[str] | object = _UNSET,
        element_parent_uuids: Sequence[str] | object = _UNSET,
        element_parent_indices: Sequence[int] | object = _UNSET,
    ) -> typing.Self:
        """Return a copy with updated array-runtime metadata.

        Only the fields passed explicitly are changed; any field left as the
        ``_UNSET`` sentinel keeps its current value.

        Args:
            const_array (typing.Any): Frozen literal array payload. Defaults to
                ``_UNSET`` (keep current).
            element_uuids (Sequence[str] | object): Per-element UUIDs. Defaults
                to ``_UNSET`` (keep current).
            element_logical_ids (Sequence[str] | object): Per-element
                logical IDs. Defaults to ``_UNSET`` (keep current).
            element_parent_uuids (Sequence[str] | object): Per-element root
                array UUIDs (parallel to ``element_uuids``; ``""`` for a
                standalone qubit). Defaults to ``_UNSET`` (keep current).
            element_parent_indices (Sequence[int] | object): Per-element index
                within the root array (parallel to ``element_uuids``; ``-1``
                for a standalone qubit). Defaults to ``_UNSET`` (keep current).

        Returns:
            typing.Self: A new value carrying the merged array-runtime metadata.
        """
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
        new_element_parent_uuids = (
            current.element_parent_uuids
            if element_parent_uuids is _UNSET
            else tuple(typing.cast(Sequence[str], element_parent_uuids))
        )
        new_element_parent_indices = (
            current.element_parent_indices
            if element_parent_indices is _UNSET
            else tuple(typing.cast(Sequence[int], element_parent_indices))
        )
        return self._replace_metadata(
            dataclasses.replace(
                self.metadata,
                array_runtime=ArrayRuntimeMetadata(
                    const_array=new_const_array,
                    element_uuids=new_element_uuids,
                    element_logical_ids=new_element_logical_ids,
                    element_parent_uuids=new_element_parent_uuids,
                    element_parent_indices=new_element_parent_indices,
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
class Value(_MetadataValueMixin, ValueBase, typing.Generic[T]):
    """A typed SSA value in the IR.

    The ``name`` field is **display-only**: it labels the value for
    visualization and error messages and has no role in identity. Identity
    is carried by ``uuid`` (per-version) and ``logical_id`` (across
    versions).

    An empty string (``name=""``) is the **anonymous marker** used by
    auto-generated tmp values (arithmetic results, comparison results,
    coerced constants). Compiler-internal identity and writes use UUIDs or
    explicit parameter metadata. Compatibility readers may consult a non-empty
    label only after those identity channels, so anonymous temporaries cannot
    collide through a shared display key.
    """

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
        after the value is updated (e.g. by a gate application or a
        classical operation).  The ``logical_id`` also stays the same:
        it identifies the same logical variable across SSA versions,
        independently of backend resource allocation.  This applies to
        every ``Value`` regardless of its type (``Qubit``, ``Float``,
        ``Bit``, ...) -- it is not specific to qubits.
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
    """An array of typed IR values.

    When ``slice_of`` is set, this array is a strided view over another
    array.  Element accesses on a sliced ``ArrayValue`` resolve to
    physical slots on the root parent via the affine map
    ``parent_index = slice_start + slice_step * view_local_index``,
    applied recursively along ``slice_of`` chains.  The emit-time
    resolver walks this chain to produce the final qubit index; passes
    that substitute or clone values must treat ``slice_of`` /
    ``slice_start`` / ``slice_step`` as Value references that need to
    track through the same mapping as ``parent_array``.

    Attributes:
        type: Element type of the array.
        name: Human-readable name for debug/error messages.
        uuid: Stable identifier across clones.
        logical_id: Preserved across SSA versions (``next_version``).
        shape: Dimension sizes as Values (symbolic or constant).
        metadata: Typed metadata (parameter binding, runtime array data).
        slice_of: Parent array this is a strided view of, or ``None`` for
            a non-sliced array.  For non-``None`` values,
            ``slice_start`` and ``slice_step`` must also be non-``None``.
        slice_start: Parent-space index of the first covered element.
        slice_step: Parent-space stride; always a positive ``UInt`` value
            in supported use cases.
    """

    type: T
    name: str
    uuid: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    logical_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    shape: tuple[Value, ...] = dataclasses.field(default_factory=tuple)
    metadata: ValueMetadata = dataclasses.field(default_factory=ValueMetadata)
    slice_of: "ArrayValue | None" = None
    slice_start: "Value | None" = None
    slice_step: "Value | None" = None

    def next_version(self) -> ArrayValue[T]:
        return ArrayValue(
            type=self.type,
            name=self.name,
            version=self.version + 1,
            metadata=self.metadata,
            uuid=str(uuid.uuid4()),
            logical_id=self.logical_id,
            shape=self.shape,
            slice_of=self.slice_of,
            slice_start=self.slice_start,
            slice_step=self.slice_step,
        )

    def is_slice(self) -> bool:
        """Return True if this array is a strided view of another array.

        Returns:
            ``True`` iff ``slice_of`` is non-``None``.
        """
        return self.slice_of is not None


def resolve_root_array_index(
    array: "ArrayValue",
    index: int,
) -> tuple["ArrayValue", int] | None:
    """Fold a view-local element index into the root array's index space.

    Walks the ``slice_of`` chain root-ward, composing each strided view's
    affine map ``parent_index = start + step * local_index``. This is the
    array-level counterpart of :func:`resolve_root_qubit_address` (which
    starts from an array-element ``Value``); both must stay consistent with
    the composite carrier keys ``"<root_uuid>_<root_index>"`` registered by
    ``QInitOperation`` at emit time.

    Args:
        array (ArrayValue): Array the index is local to. May be a root array
            (``slice_of`` unset) or an arbitrarily nested strided view.
        index (int): Element index in ``array``'s own index space.

    Returns:
        tuple[ArrayValue, int] | None: ``(root_array, composed_index)`` when
            every slice bound on the chain is compile-time constant and
            satisfies the frontend contract (non-negative ``slice_start``,
            positive ``slice_step``). ``None`` when any ``slice_start`` /
            ``slice_step`` on the chain is missing, symbolic, or violates
            that contract; callers must then defer resolution rather than
            guess. Out-of-contract bounds would compose ``index`` onto a
            wrong root slot, so they are refused here too (the frontend
            rejects them at trace time; this guard covers programmatically
            constructed IR).
    """
    current = array
    idx = index
    while current.slice_of is not None:
        # Symbolic view bounds cannot be folded to a fixed root index at
        # this stage; defer to a resolver that has bindings available.
        if (
            current.slice_start is None
            or current.slice_step is None
            or not current.slice_start.is_constant()
            or not current.slice_step.is_constant()
        ):
            return None
        start = int(typing.cast(int, current.slice_start.get_const()))
        step = int(typing.cast(int, current.slice_step.get_const()))
        # Mirror the frontend slice contract (non-negative start, positive
        # step); composing a negative start or non-positive step would remap
        # ``idx`` onto a wrong root slot.
        if start < 0 or step <= 0:
            return None
        idx = start + step * idx
        current = current.slice_of
    return current, idx


def array_static_length(array: "ArrayValue") -> int | None:
    """Resolve a one-dimensional array's compile-time length.

    Args:
        array (ArrayValue): Array whose sole shape dimension is inspected.

    Returns:
        int | None: Non-negative static length, or None when the array is not
            one-dimensional, its length is symbolic/non-integral, or it is
            malformed with a negative length. Boolean constants are rejected
            even though ``bool`` is an ``int`` subclass.
    """
    if len(array.shape) != 1:
        return None
    length_value = array.shape[0]
    if not length_value.is_constant():
        return None
    length = length_value.get_const()
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        return None
    return int(length)


def array_physical_region(
    array: "ArrayValue",
) -> tuple[str, tuple[int, ...]] | None:
    """Resolve a one-dimensional array to its ordered physical region.

    The region is expressed independently of SSA version UUIDs: the first
    element is the root array's ``logical_id`` and the second is the ordered
    tuple of root-space indices addressed by the array. This makes a root
    register and its full view compare equal while keeping partial, strided,
    reordered, and different-root registers distinct.

    Args:
        array (ArrayValue): Root array or nested sliced view to resolve. The
            array must be one-dimensional and have a compile-time integer
            length; every slice bound in its ancestry must also be constant.

    Returns:
        tuple[str, tuple[int, ...]] | None: ``(root_logical_id, indices)``
            when the complete ordered coverage is statically known, otherwise
            ``None``. Symbolic shapes or slice bounds remain unresolved so
            callers can defer to emit-time physical mappings.
    """
    length = array_static_length(array)
    if length is None:
        return None

    root: ArrayValue | None = None
    root_indices: list[int] = []
    for local_index in range(length):
        resolved = resolve_root_array_index(array, local_index)
        if resolved is None:
            return None
        element_root, root_index = resolved
        if root is None:
            root = element_root
        elif element_root.logical_id != root.logical_id:
            return None
        root_indices.append(root_index)

    # Empty views do not visit an element above, but their slice ancestry still
    # identifies the physical root unambiguously when all bounds are valid.
    if root is None:
        resolved = resolve_root_array_index(array, 0)
        if resolved is None:
            return None
        root = resolved[0]
    return root.logical_id, tuple(root_indices)


def arrays_share_physical_region(left: "ArrayValue", right: "ArrayValue") -> bool:
    """Return whether two arrays denote the same ordered physical region.

    Args:
        left (ArrayValue): First root array or sliced view.
        right (ArrayValue): Second root array or sliced view.

    Returns:
        bool: ``True`` for the same SSA value/version lineage, when both arrays
            resolve to the same root logical identity and ordered root indices,
            or when both resolve to empty regions (whose root is unobservable).
            Returns ``False`` for non-empty divergent regions and unresolved
            symbolic coverage.
    """
    if left.uuid == right.uuid:
        return True
    if array_static_length(left) == 0 and array_static_length(right) == 0:
        return True
    left_region = array_physical_region(left)
    right_region = array_physical_region(right)
    if left_region is None or right_region is None:
        return False
    return left_region == right_region


def resolve_root_qubit_address(value: "Value") -> tuple[str, int] | None:
    """Resolve an array-element value to its root ``(array_uuid, index)``.

    Walks the ``parent_array`` / ``slice_of`` chain and composes the nested
    affine slice maps, so ``view[i]`` resolves to
    ``(root_uuid, start + step * i)`` for the composed ``(start, step)``. The
    returned pair is the build-stable identity of the physical qubit slot: the
    root array's ``QInitOperation`` always registers it as
    ``QubitAddress(root_uuid, index)``, so this address resolves even when the
    element's own (per-version) UUID was never registered.

    The transpiler's resource allocator uses the same walk to resolve gate and
    measurement operands; this shared helper keeps both call sites consistent.

    Args:
        value (Value): The value to resolve. Expected to be an array element
            (``parent_array`` set with a single constant ``element_indices``
            entry).

    Returns:
        tuple[str, int] | None: ``(root_array_uuid, composed_index)`` when
            ``value`` is an array element with a constant index whose entire
            ``slice_of`` chain has constant ``slice_start`` / ``slice_step``.
            ``None`` when ``value`` is not an array element, when its index is
            non-constant, or when any slice bound in the chain is non-constant
            (those cases are deferred to the emit-time resolver, which has
            bindings available). Also ``None`` for a negative constant index
            or a chain frame with negative ``slice_start`` / non-positive
            ``slice_step`` — composing those would silently address a wrong
            root slot, so they are refused rather than guessed (the frontend
            rejects them at trace time; this guard covers programmatically
            constructed IR).
    """
    # Not an array element (e.g. a standalone qubit / scalar): there is no
    # root array to address against.
    #
    # ``parent_array`` and ``element_indices`` are always set together -- an
    # array-element access sets both, and ``next_version`` copies both -- so a
    # fully-formed element has both and a non-element has neither; the
    # "only one set" case does not normally arise. ``or`` (not ``and``) is still
    # the right guard because BOTH are needed below (``element_indices[0]`` for
    # the index, ``parent_array`` to walk to the root), so we bail to None if
    # either is missing rather than risk an IndexError / None-walk.
    if value.parent_array is None or not value.element_indices:
        return None
    idx_value = value.element_indices[0]
    # A non-constant (symbolic / loop-variable) index cannot be pinned to a
    # fixed physical slot at this stage. Return None to DEFER it to the
    # emit-time resolver, which has the bindings to resolve it -- guessing here
    # would silently bind to the wrong qubit.
    if not idx_value.is_constant():
        return None
    idx = int(typing.cast(int, idx_value.get_const()))
    # A negative local index would compose through the affine map below into
    # a valid-but-wrong non-negative root index (``view[-1]`` for
    # ``view = q[1:3]`` would address ``q[0]`` instead of ``q[2]``). The
    # frontend rejects constant negative indices at trace time; refuse them
    # here as well so programmatically constructed IR cannot slip through.
    if idx < 0:
        return None
    # Walk the ``slice_of`` chain root-ward. Each strided view contributes the
    # affine map ``parent_index = start + step * local_index``; composing them
    # rewrites a (possibly nested) view element into the underlying root array's
    # own index space, so the result matches the composite key QInit registered.
    # ``resolve_root_array_index`` also refuses out-of-contract slice bounds
    # (negative start / non-positive step) so the negative-index defense lives
    # in one shared place.
    resolved = resolve_root_array_index(value.parent_array, idx)
    if resolved is None:
        return None
    root, root_idx = resolved
    return (root.uuid, root_idx)


@dataclasses.dataclass(frozen=True)
class TupleValue(_MetadataValueMixin, ValueBase):
    """A tuple of IR values for structured data."""

    name: str
    elements: tuple[ValueLike, ...] = dataclasses.field(default_factory=tuple)
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
        return all(element.is_constant() for element in self.elements)


@dataclasses.dataclass(frozen=True)
class DictValue(_MetadataValueMixin, ValueBase):
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


def collect_value_like_uuids(value: "ValueLike") -> set[str]:
    """Collect UUIDs contained in a value-like IR object.

    Args:
        value (ValueLike): Value-like object to inspect.

    Returns:
        set[str]: UUIDs for ``value`` itself, recursively contained tuple/dict
            elements, and array view/element dependencies.
    """
    uuids: set[str] = set()

    def collect(current: ValueLike) -> None:
        """Collect one value and recursively embedded dependencies.

        Args:
            current (ValueLike): Value-like object to visit.
        """
        if current.uuid in uuids:
            return
        uuids.add(current.uuid)
        if isinstance(current, TupleValue):
            for element in current.elements:
                collect(element)
        elif isinstance(current, DictValue):
            for key, entry_value in current.entries:
                collect(key)
                collect(entry_value)
        elif isinstance(current, ArrayValue):
            for dimension in current.shape:
                collect(dimension)
            if current.slice_of is not None:
                collect(current.slice_of)
            if current.slice_start is not None:
                collect(current.slice_start)
            if current.slice_step is not None:
                collect(current.slice_step)
        elif isinstance(current, Value):
            if current.parent_array is not None:
                collect(current.parent_array)
            for index in current.element_indices:
                collect(index)

    collect(value)
    return uuids
