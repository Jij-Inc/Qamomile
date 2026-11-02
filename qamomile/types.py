"""Logical data, modality, and staging types for Qamomile."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, get_args, get_origin

from ._native import compiler as native
from .errors import SerializationError, TypeCheckError

Dimension = int | str
_AVAILABILITY_STAGES = frozenset({"Const", "Param", "Shot"})
_MAPPING_ENTRY_NAME = "__qamomile_mapping_entry__"


class TypeSpec:
    """Base class for immutable semantic types."""

    @property
    def is_quantum(self) -> bool:
        """Whether this type contains a coherent quantum value."""

        return not _inspect_native_type_admissibility(self)[2]

    @property
    def is_linear(self) -> bool:
        """Whether values of this type must be consumed exactly once."""

        return self.is_quantum


@dataclass(frozen=True, slots=True)
class ScalarType(TypeSpec):
    """A scalar logical type."""

    name: str

    def __post_init__(self) -> None:
        """Keep the scalar universe closed and serializable."""

        if self.name not in {
            "Unit",
            "Bit",
            "Bool",
            "Int",
            "Real",
            "Complex",
            "Probability",
            "Observable",
        }:
            raise TypeCheckError(f"Unknown scalar type {self.name!r}.")

    def __str__(self) -> str:
        """Return the source-style type name."""

        return self.name


@dataclass(frozen=True, slots=True)
class ArrayType(TypeSpec):
    """A statically shaped homogeneous logical collection."""

    element: TypeSpec
    shape: tuple[Dimension, ...]

    def __post_init__(self) -> None:
        """Canonicalize nested arrays into one non-empty structural shape."""

        element = self.element
        shape = self.shape
        while isinstance(element, ArrayType):
            shape = (*shape, *element.shape)
            element = element.element
        if not shape:
            raise TypeCheckError("An array must have at least one dimension.")
        if not _inspect_native_type_admissibility(element)[2]:
            raise TypeCheckError("Classical arrays cannot contain owned quantum data.")
        for dimension in shape:
            _dimension(dimension)
        object.__setattr__(self, "element", element)
        object.__setattr__(self, "shape", shape)

    def __str__(self) -> str:
        """Return a compact source-style representation."""

        shape = ", ".join(str(dimension) for dimension in self.shape)
        return f"{_array_constructor(len(self.shape))}[{self.element}, {shape}]"


@dataclass(frozen=True, slots=True)
class MapType(TypeSpec):
    """A fixed immutable typed mapping available during specialization."""

    key: TypeSpec
    value: TypeSpec
    size: Dimension

    def __post_init__(self) -> None:
        """Require classical key/value types and a positive entry count."""

        _dimension(self.size)
        if self.key.is_quantum or self.value.is_quantum:
            raise TypeCheckError("Map keys and values must be classical.")

    def __str__(self) -> str:
        """Return the source-style mapping representation."""

        return f"Map[{self.key}, {self.value}, {self.size}]"


@dataclass(frozen=True, slots=True)
class WordType(TypeSpec):
    """A fixed-width classical bit word with an explicit interpretation."""

    kind: str
    width: Dimension

    def __post_init__(self) -> None:
        """Reject unknown interpretations and invalid word widths."""

        if self.kind not in {"Bits", "UInt"}:
            raise TypeCheckError(f"Unknown word interpretation {self.kind!r}.")
        _dimension(self.width)

    def __str__(self) -> str:
        """Return the source-style word representation."""

        return f"{self.kind}[{self.width}]"


@dataclass(frozen=True, slots=True)
class FixedType(TypeSpec):
    """An unsigned binary fixed-point value with a canonical digit order.

    Coherent and sampled components are ordered from the most-significant
    integer digit through the least-significant fractional digit.  This order
    is semantic and independent of an SDK's classical-register convention.

    Attributes:
        integer_bits (Dimension): Concrete or dependent integer-part width.
        fractional_bits (Dimension): Concrete or dependent fractional width.
    """

    integer_bits: Dimension
    fractional_bits: Dimension

    def __post_init__(self) -> None:
        """Require a nonempty, statically partitioned binary representation.

        Raises:
            TypeCheckError: If either partition is invalid or both concrete
                partitions are empty.
        """

        for _name, width in (
            ("integer_bits", self.integer_bits),
            ("fractional_bits", self.fractional_bits),
        ):
            _fixed_dimension(width)
        if self.integer_bits == 0 and self.fractional_bits == 0:
            raise TypeCheckError("UFixed must contain at least one bit.")

    @property
    def width(self) -> int | None:
        """Return the concrete digit count when both partitions are fixed.

        Returns:
            int | None: Total digits, or ``None`` for a dependent partition.
        """

        if isinstance(self.integer_bits, int) and isinstance(self.fractional_bits, int):
            return self.integer_bits + self.fractional_bits
        return None

    def __str__(self) -> str:
        """Return the source-style fixed-point representation."""

        return f"UFixed[{self.integer_bits}, {self.fractional_bits}]"


@dataclass(frozen=True, slots=True)
class ProductType(TypeSpec):
    """A positional product of logical types."""

    elements: tuple[TypeSpec, ...]

    def __str__(self) -> str:
        """Return a compact product representation."""

        return "(" + ", ".join(str(element) for element in self.elements) + ")"


@dataclass(frozen=True, slots=True)
class RecordType(TypeSpec):
    """A named logical product with stable field names."""

    name: str
    fields: tuple[tuple[str, TypeSpec], ...]

    def __post_init__(self) -> None:
        """Require a portable name and a unique, nonempty field row."""

        names = tuple(name for name, _ in self.fields)
        if (
            not self.name
            or not self.fields
            or any(not name for name in names)
            or len(names) != len(set(names))
        ):
            raise TypeCheckError(
                "Record types require unique nonempty names and fields."
            )

    def __str__(self) -> str:
        """Return the named record representation."""

        return self.name


@dataclass(frozen=True, slots=True)
class FunctionType(TypeSpec):
    """An immutable function from zero or more inputs to one output."""

    inputs: tuple[TypeSpec, ...]
    output: TypeSpec
    parameter_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate an optional complete source-parameter label row."""

        if self.parameter_names and (
            len(self.parameter_names) != len(self.inputs)
            or any(not name for name in self.parameter_names)
            or len(set(self.parameter_names)) != len(self.parameter_names)
        ):
            raise TypeCheckError(
                "QKernel parameter names must be a complete unique nonempty row."
            )

    def __str__(self) -> str:
        """Return the source-style function signature."""

        items = ", ".join(str(item) for item in (*self.inputs, self.output))
        return f"QKernel[{items}]"


def aggregate_element_types(type_spec: TypeSpec) -> tuple[TypeSpec, ...] | None:
    """Return a product row with outer availability distributed to its fields."""

    return _inspect_native_type_shape(type_spec)[3]


@dataclass(frozen=True, slots=True)
class QuantumType(TypeSpec):
    """The coherent modality over one logical data shape."""

    inner: TypeSpec

    def __post_init__(self) -> None:
        """Reject nested modalities and non-bit coherent payloads."""

        if not _inspect_native_type_admissibility(self.inner)[0]:
            raise TypeCheckError(
                "Q[T] requires a bit, word, or nonempty product of those types."
            )

    def __str__(self) -> str:
        """Return the source-style quantum type."""

        return f"Q[{self.inner}]"


@dataclass(frozen=True, slots=True)
class StagedType(TypeSpec):
    """A logical value annotated with its availability stage."""

    stage: str
    inner: TypeSpec

    def __post_init__(self) -> None:
        """Keep availability modalities flat, classical, and closed."""

        if self.stage not in _AVAILABILITY_STAGES:
            raise TypeCheckError(f"Unknown availability stage {self.stage!r}.")
        if not _inspect_native_type_admissibility(self.inner)[1]:
            raise TypeCheckError(
                "Availability stages contain only unstaged classical data."
            )

    def __str__(self) -> str:
        """Return the source-style staged type."""

        return f"{self.stage}[{self.inner}]"


Unit = ScalarType("Unit")
Bit = ScalarType("Bit")
Bool = ScalarType("Bool")
Int = ScalarType("Int")
Real = ScalarType("Real")
Complex = ScalarType("Complex")
Probability = ScalarType("Probability")
Observable = ScalarType("Observable")


def _array_constructor(rank: int) -> str:
    """Return the canonical source constructor for one ordinary array rank."""

    if rank == 1:
        return "Vector"
    if rank == 2:
        return "Matrix"
    if rank >= 3:
        return "Tensor"
    raise TypeCheckError("An array must have at least one dimension.")


def array_type(element: TypeSpec, shape: tuple[Dimension, ...]) -> ArrayType:
    """Construct one canonical ordinary structural array type.

    Args:
        element: Logical leaf element type. An ordinary array element is
            flattened into the returned shape.
        shape: One or more leading dimensions.

    Returns:
        A rank-n array whose display constructor is derived from its rank.

    Raises:
        TypeCheckError: If ``shape`` is empty.
    """

    return ArrayType(element, shape)


def mapping_type(
    key: TypeSpec,
    value: TypeSpec,
    size: Dimension,
) -> MapType:
    """Construct an immutable typed mapping's structural representation.

    The source mapping lowers to a fixed array of nominal key/value records so
    the persistent type and backend ABI reuse QCore's ordinary aggregate
    algebra. Only the frontend and native specialization recognize the nominal
    record as mapping syntax.

    Args:
        key: Classical key type.
        value: Classical value type.
        size: Positive fixed or dependent entry count.

    Returns:
        MapType: Immutable source mapping type.

    Raises:
        TypeCheckError: If a key or value contains owned quantum data.
    """

    return MapType(as_type(key), as_type(value), _dimension(size))


def mapping_components(
    type_spec: TypeSpec,
) -> tuple[TypeSpec, TypeSpec, Dimension] | None:
    """Return key, value, and extent for a source mapping representation.

    Args:
        type_spec: Staged or unstaged semantic type.

    Returns:
        The mapping components, or ``None`` for an ordinary type.
    """

    base = strip_stage(type_spec)
    if isinstance(base, MapType):
        return base.key, base.value, base.size
    if not isinstance(base, ArrayType) or len(base.shape) != 1:
        return None
    entry = base.element
    if not isinstance(entry, RecordType) or entry.name != _MAPPING_ENTRY_NAME:
        return None
    if tuple(name for name, _ in entry.fields) != ("key", "value"):
        return None
    return entry.fields[0][1], entry.fields[1][1], base.shape[0]


def sequence_element_type(type_spec: TypeSpec) -> TypeSpec:
    """Return the type produced by indexing an array or word once.

    Args:
        type_spec: Staged or unstaged sequence type.

    Returns:
        The scalar element or the remaining rank-n array at the same stage.

    Raises:
        TypeCheckError: If ``type_spec`` is not an indexable sequence.
    """

    child = _inspect_native_type_shape(type_spec)[2]
    if child is None:
        raise TypeCheckError(f"Cannot index non-sequence type {type_spec}.")
    return child


class _UIntConstructor:
    """Type constructor for a fixed-width unsigned integer."""

    def __getitem__(self, width: Dimension) -> WordType:
        """Construct a fixed-width unsigned integer type."""

        return WordType("UInt", _dimension(width))


class _BitsConstructor:
    """Type constructor for a fixed-length bit string."""

    def __getitem__(self, size: Dimension) -> WordType:
        """Construct a fixed-length bit-string type."""

        return WordType("Bits", _dimension(size))


class _UFixedConstructor:
    """Type constructor for an unsigned binary fixed-point value."""

    def __getitem__(self, widths: tuple[Dimension, Dimension]) -> FixedType:
        """Construct ``UFixed[integer_bits, fractional_bits]``."""

        if not isinstance(widths, tuple) or len(widths) != 2:
            raise TypeCheckError(
                "UFixed expects UFixed[integer_bits, fractional_bits]."
            )
        integer_bits, fractional_bits = widths
        return FixedType(integer_bits, fractional_bits)


class _VectorConstructor:
    """Type constructor for a statically shaped vector."""

    def __getitem__(
        self,
        item: Dimension | tuple[TypeSpec | type[Any], Dimension],
    ) -> ArrayType:
        """Construct ``Vector[Real, N]`` or the shorthand ``Vector[N]``."""

        if isinstance(item, tuple):
            if len(item) != 2:
                raise TypeCheckError("Vector expects Vector[element, size].")
            element, size = item
            return array_type(as_type(element), (_dimension(size),))
        return array_type(Real, (_dimension(item),))


class _MatrixConstructor:
    """Type constructor for a statically shaped matrix."""

    def __getitem__(
        self,
        item: tuple[TypeSpec | type[Any], Dimension, Dimension]
        | tuple[Dimension, Dimension],
    ) -> ArrayType:
        """Construct ``Matrix[element, rows, cols]`` or a real matrix."""

        if not isinstance(item, tuple):
            raise TypeCheckError("Matrix expects Matrix[rows, cols].")
        if len(item) == 2:
            rows, columns = item
            element = Real
        elif len(item) == 3:
            element, rows, columns = item
            element = as_type(element)
        else:
            raise TypeCheckError("Matrix expects Matrix[element, rows, cols].")
        return array_type(element, (_dimension(rows), _dimension(columns)))


class _TensorConstructor:
    """Type constructor for a statically shaped rank-n tensor."""

    def __getitem__(self, item: tuple[object, ...]) -> ArrayType:
        """Construct ``Tensor[element, depth, rows, columns, ...]``."""

        if not isinstance(item, tuple) or len(item) < 4:
            raise TypeCheckError(
                "Tensor expects Tensor[element, depth, rows, columns, ...]."
            )
        element = as_type(item[0])
        dimensions = item[1:]
        return array_type(element, tuple(_dimension(value) for value in dimensions))


class _MapConstructor:
    """Type constructor for a fixed immutable typed mapping."""

    def __getitem__(
        self,
        item: tuple[TypeSpec | type[Any], TypeSpec | type[Any], Dimension],
    ) -> MapType:
        """Construct ``Map[key, value, size]``.

        Args:
            item: Key type, value type, and fixed or dependent entry count.

        Returns:
            MapType: Immutable typed mapping.

        Raises:
            TypeCheckError: If the constructor does not receive three items.
        """

        if not isinstance(item, tuple) or len(item) != 3:
            raise TypeCheckError("Map expects Map[key, value, size].")
        key, value, size = item
        return mapping_type(as_type(key), as_type(value), size)


class _QuantumConstructor:
    """Type constructor for coherent quantum data."""

    def __getitem__(self, item: object) -> QuantumType:
        """Construct the quantum modality over typed coherent data.

        Args:
            item (object): Logical bit-shaped payload type.

        Returns:
            QuantumType: Coherent form of the supplied logical type. In
                particular, ``Q[Bits[N]]`` is the canonical spelling for
                uninterpreted coherent bits.

        Raises:
            TypeCheckError: If the payload is not admissible coherent data.
        """

        type_spec = as_type(item)
        return QuantumType(type_spec)


class _ConstConstructor:
    """Type constructor for specialization-time classical values."""

    def __getitem__(self, item: object) -> StagedType:
        """Construct a specialization-time staged type."""

        return _staged("Const", item)


class _ParamConstructor:
    """Type constructor for execution-parameter values."""

    def __getitem__(self, item: object) -> StagedType:
        """Construct an execution-parameter staged type."""

        return _staged("Param", item)


class _ShotConstructor:
    """Type constructor for values produced during one execution shot."""

    def __getitem__(self, item: object) -> StagedType:
        """Construct a shot-time staged type."""

        return _staged("Shot", item)


UInt = _UIntConstructor()
Bits = _BitsConstructor()
UFixed = _UFixedConstructor()
Vector = _VectorConstructor()
Matrix = _MatrixConstructor()
Tensor = _TensorConstructor()
Map = _MapConstructor()
Q = _QuantumConstructor()
Const = _ConstConstructor()
Param = _ParamConstructor()
Shot = _ShotConstructor()


def as_type(annotation: object) -> TypeSpec:
    """Normalize a Python or Qamomile annotation into a semantic type.

    Args:
        annotation: Annotation or semantic type to normalize.

    Returns:
        The corresponding immutable semantic type.

    Raises:
        TypeCheckError: If the annotation has no Qamomile type interpretation.
    """

    if isinstance(annotation, TypeSpec):
        return annotation
    if annotation is None or annotation is type(None):
        return Unit
    if annotation is bool:
        return Bool
    if annotation is int:
        return Int
    if annotation is float:
        return Real
    if annotation is complex:
        return Complex
    origin = get_origin(annotation)
    if origin in (tuple,):
        return ProductType(tuple(as_type(item) for item in get_args(annotation)))
    raise TypeCheckError(
        f"{annotation!r} is not a Qamomile logical type. "
        "Use types such as Real, Bits[N], UFixed[I, F], Q[T], Param[T], "
        "or tuple[...]."
    )


def infer_literal_type(value: object) -> TypeSpec:
    """Infer a semantic type for a trace-time Python literal.

    Args:
        value: Literal value embedded in a traced function.

    Returns:
        The inferred logical type.

    Raises:
        TypeCheckError: If no safe literal representation is available.
    """

    infer = getattr(native, "infer_literal_type", None)
    if not callable(infer):
        raise TypeError("Q4511: native literal type inference is unavailable")

    try:
        encoded = _encode_native_literal(value)
    except (SerializationError, TypeError, ValueError) as error:
        raise TypeCheckError(
            f"Cannot infer a semantic type for literal {value!r}."
        ) from error
    result = infer(encoded)
    if result is None:
        raise TypeCheckError(f"Cannot infer a semantic type for literal {value!r}.")
    return _decode_native_type(result)


def strip_stage(type_spec: TypeSpec) -> TypeSpec:
    """Remove one availability-stage wrapper from a type."""

    if isinstance(type_spec, StagedType):
        return type_spec.inner
    return type_spec


def local_peer_types(left: TypeSpec, right: TypeSpec) -> bool:
    """Compare two peer types in one local classical equation.

    Local equations join the outer availability of their operands but do not
    instantiate dependent dimensions or erase stages nested in aggregates.
    Flexible dimension substitution remains exclusive to call-site checking
    through :func:`unify_dimensions`.

    Args:
        left (TypeSpec): First local operand or alternative type.
        right (TypeSpec): Second local operand or alternative type.

    Returns:
        bool: Whether the types differ at most in their outer availability.
    """

    compare = getattr(native, "are_local_peer_types", None)
    if not callable(compare):
        raise TypeError("Q4511: native local type relation is unavailable")
    result = compare(_encode_native_type(left), _encode_native_type(right))
    if type(result) is not bool:
        raise RuntimeError("Q4512: native local type relation is malformed")
    return result


def common_stage(*types: TypeSpec) -> str | None:
    """Return the latest availability stage among several types.

    Args:
        *types (TypeSpec): Logical types whose outer availability is joined.

    Returns:
        str | None: Latest stage, or ``None`` when every type is unstaged.

    Raises:
        TypeError: If the native stage relation is unavailable.
        RuntimeError: If the native relation returns malformed data.
    """

    project = getattr(native, "common_type_stage", None)
    if not callable(project):
        raise TypeError("Q4511: native availability-stage relation is unavailable")
    result = project([_encode_native_type(as_type(value)) for value in types])
    if result is not None and (
        type(result) is not str or result not in _AVAILABILITY_STAGES
    ):
        raise RuntimeError(
            "Q4512: native availability-stage relation returned a malformed result"
        )
    return result


def at_stage(type_spec: TypeSpec, stage: str | None) -> TypeSpec:
    """Wrap a base type at an availability stage when one is present."""

    base = strip_stage(type_spec)
    if stage is None:
        return base
    return StagedType(stage, base)


def stage_of(type_spec: TypeSpec) -> str | None:
    """Return the availability stage of a classical type.

    Args:
        type_spec: Semantic type to inspect.

    Returns:
        ``Const``, ``Param``, ``Shot``, or ``None``.
    """

    if isinstance(type_spec, StagedType):
        return type_spec.stage
    return None


def is_type_assignable(expected: TypeSpec, actual: TypeSpec) -> bool:
    """Return whether a value may cross one declared type boundary.

    Unstaged declarations are availability-polymorphic. An explicit stage is
    an upper bound over the complete aggregate, including stages distributed
    into product fields. Logical shapes are checked by the same dependent
    dimension unifier used for calls.

    Args:
        expected (TypeSpec): Declared destination type.
        actual (TypeSpec): Inferred source value type.

    Returns:
        bool: Whether logical structure and availability constraints both hold.

    Raises:
        TypeError: If either type is invalid or the native relation is
            unavailable.
        RuntimeError: If the native relation returns malformed data.
    """

    expected = as_type(expected)
    actual = as_type(actual)
    assignable = getattr(native, "is_type_assignable", None)
    if not callable(assignable):
        raise TypeError("Q4511: native type relation is unavailable")
    result = assignable(_encode_native_type(expected), _encode_native_type(actual))
    if type(result) is not bool:
        raise RuntimeError("Q4512: native type relation returned a malformed result")
    return result


def unify_dimensions(
    pattern: TypeSpec,
    actual: TypeSpec,
    bindings: Mapping[str, Dimension] | None = None,
) -> dict[str, Dimension] | None:
    """Unify symbolic dimensions in a callable type with one argument type.

    Staging is intentionally orthogonal to the logical shape: an unstaged
    helper parameter is stage-polymorphic, while an explicitly staged
    parameter still constrains availability at the frontend boundary.

    Args:
        pattern: Declared callable parameter or result type.
        actual: Type supplied at a particular call site.
        bindings: Dimension bindings already inferred from earlier arguments.

    Returns:
        A merged dimension mapping, or ``None`` when the logical types differ.
    """

    return _unify_type_rows((pattern,), (actual,), bindings)


def _unify_type_rows(
    patterns: Sequence[TypeSpec],
    actuals: Sequence[TypeSpec],
    bindings: Mapping[str, Dimension] | None = None,
) -> dict[str, Dimension] | None:
    """Unify an ordered callable type row in one native compiler query.

    Args:
        patterns: Declared callable parameter types.
        actuals: Supplied argument types in matching order.
        bindings: Dimension bindings established outside the type row.

    Returns:
        A merged dimension mapping, or ``None`` when the logical rows differ.
    """

    project = getattr(native, "unify_type_dimensions", None)
    if not callable(project):
        raise TypeError("Q4511: native type relation is unavailable")
    initial = tuple(
        _encode_native_dimension(name, value)
        for name, value in (bindings or {}).items()
    )
    result = project(
        tuple(_encode_native_type(pattern) for pattern in patterns),
        tuple(_encode_native_type(actual) for actual in actuals),
        initial,
    )
    if result is None:
        return None
    if type(result) is not list:
        raise RuntimeError("Q4512: native type relation returned a malformed result")
    inferred: dict[str, Dimension] = {}
    for row in result:
        if type(row) is not tuple or len(row) != 3:
            raise RuntimeError(
                "Q4512: native type relation returned a malformed result"
            )
        name, fixed, symbol = row
        if not isinstance(name, str) or not name or name in inferred:
            raise RuntimeError(
                "Q4512: native type relation returned a malformed result"
            )
        if type(fixed) is int and fixed > 0 and symbol is None:
            inferred[name] = fixed
        elif fixed is None and isinstance(symbol, str) and symbol:
            inferred[name] = symbol
        else:
            raise RuntimeError(
                "Q4512: native type relation returned a malformed result"
            )
    return inferred


def _encode_native_type(type_spec: TypeSpec) -> bytes:
    """Encode one semantic type for a native compiler query."""

    from .proto import qcore_pb2 as wire
    from .serialization import encode_type

    encoded = wire.TypeProto()
    encode_type(type_spec, encoded)
    return encoded.SerializeToString(deterministic=True)


def infer_function_application(
    function_type: TypeSpec,
    argument_types: tuple[TypeSpec, ...],
    *,
    positional_count: int,
    keyword_names: tuple[str, ...],
) -> TypeSpec:
    """Infer one labeled function-value call through the native type authority.

    Args:
        function_type: Staged or unstaged closed function type.
        argument_types: Argument types in Python evaluation order.
        positional_count: Leading number of positional arguments.
        keyword_names: Keyword labels in source evaluation order.

    Returns:
        The instantiated and availability-lifted result type.

    Raises:
        TypeError: If the required native relation is unavailable.
        ValueError: If the call cannot be bound or typed.
        RuntimeError: If the native result is malformed.
    """

    infer = getattr(native, "infer_function_application", None)
    if not callable(infer):
        raise TypeError("Q4511: native function application inference is unavailable")
    result = infer(
        _encode_native_type(function_type),
        tuple(_encode_native_type(argument) for argument in argument_types),
        positional_count,
        keyword_names,
    )
    return _decode_native_type(result)


def infer_declared_function_application(
    function_type: FunctionType,
    argument_types: tuple[TypeSpec, ...],
    static_arguments: tuple[Dimension | None, ...],
) -> TypeSpec:
    """Infer one declaration-ordered call through the native type authority.

    Static argument candidates are aligned with the declaration row. Rust
    decides which integer parameters actually bind dependent dimensions, so
    ordinary integer values cannot accidentally acquire shape semantics.

    Args:
        function_type: Complete labeled declaration signature.
        argument_types: Argument types in declaration order.
        static_arguments: Optional static integer or symbolic values aligned
            with ``argument_types``.

    Returns:
        The instantiated and availability-lifted result type.

    Raises:
        TypeError: If the required native relation is unavailable.
        ValueError: If the call cannot be typed or a dimension is invalid.
        RuntimeError: If the native result is malformed.
    """

    infer = getattr(native, "infer_declared_function_application", None)
    if not callable(infer):
        raise TypeError("Q4511: native declared-call inference is unavailable")
    static_projection = tuple(
        None
        if value is None
        else ((str(value), None) if type(value) is int else (None, value))
        for value in static_arguments
    )
    result = infer(
        _encode_native_type(function_type),
        tuple(_encode_native_type(argument) for argument in argument_types),
        static_projection,
    )
    return _decode_native_type(result)


def bind_declared_function_arguments(
    function_type: FunctionType,
    *,
    argument_count: int,
    positional_count: int,
    keyword_names: tuple[str, ...],
    has_defaults: tuple[bool, ...],
) -> tuple[int | None, ...]:
    """Bind one source call into declaration order through Rust.

    Args:
        function_type: Complete labeled declaration signature.
        argument_count: Number of already-evaluated source arguments.
        positional_count: Leading number of positional arguments.
        keyword_names: Keyword labels in source evaluation order.
        has_defaults: Default availability aligned with declaration order.

    Returns:
        Raw argument indices in declaration order, with ``None`` for defaults.

    Raises:
        TypeError: If the required native binder is unavailable.
        ValueError: If source arguments cannot be bound to the declaration.
        RuntimeError: If the native result violates the binding protocol.
    """

    bind = getattr(native, "bind_declared_function_arguments", None)
    if not callable(bind):
        raise TypeError("Q4511: native function argument binding is unavailable")
    result = bind(
        _encode_native_type(function_type),
        argument_count,
        positional_count,
        keyword_names,
        has_defaults,
    )
    malformed = (
        type(result) is not list
        or len(result) != len(function_type.inputs)
        or len(has_defaults) != len(function_type.inputs)
    )
    if not malformed:
        indices: list[int] = []
        for index, item in enumerate(result):
            if item is None:
                if not has_defaults[index]:
                    malformed = True
                    break
                continue
            if type(item) is not int or item < 0 or item >= argument_count:
                malformed = True
                break
            indices.append(item)
        malformed = malformed or sorted(indices) != list(range(argument_count))
    if malformed:
        raise RuntimeError("Q4512: native function argument binding is malformed")
    return tuple(result)


def _encode_native_literal(value: object) -> bytes:
    """Encode one host value into the compiler's portable literal form."""

    from .ir import freeze_value
    from .proto import qcore_pb2 as wire
    from .serialization import encode_literal

    encoded = wire.LiteralProto()
    encode_literal(freeze_value(value), encoded)
    return encoded.SerializeToString(deterministic=True)


def _encode_native_dimension(
    name: object,
    value: object,
) -> tuple[str, int | None, str | None]:
    """Project one initial dependent-dimension binding fail-closed."""

    if not isinstance(name, str) or not name:
        raise ValueError("Q4510: dimension binding names must be nonempty strings")
    if type(value) is int and value > 0:
        return name, value, None
    if isinstance(value, str) and value:
        return name, None, value
    raise ValueError("Q4510: dimensions must be positive integers or nonempty strings")


def substitute_dimensions(
    type_spec: TypeSpec,
    bindings: Mapping[str, Dimension],
) -> TypeSpec:
    """Instantiate symbolic dimensions while preserving availability stages.

    Args:
        type_spec: Type scheme to instantiate.
        bindings: Call-site dimension mapping.

    Returns:
        The recursively instantiated semantic type.
    """

    substitute = getattr(native, "substitute_type_dimensions", None)
    if not callable(substitute):
        raise TypeError("Q4511: native type transformation is unavailable")
    projected = tuple(
        _encode_native_dimension(name, value) for name, value in bindings.items()
    )
    result = substitute(_encode_native_type(type_spec), projected)
    return _decode_native_type(result)


def symbolic_dimensions(type_spec: TypeSpec) -> frozenset[str]:
    """Return the dependent-shape binders occurring in one semantic type."""

    collect = getattr(native, "type_symbolic_dimensions", None)
    if not callable(collect):
        raise TypeError("Q4511: native type transformation is unavailable")
    result = collect(_encode_native_type(type_spec))
    if (
        type(result) is not list
        or any(not isinstance(name, str) or not name for name in result)
        or result != sorted(set(result))
    ):
        raise RuntimeError(
            "Q4512: native type transformation returned malformed symbols"
        )
    return frozenset(result)


def _decode_native_type(result: object) -> TypeSpec:
    """Decode one canonical type returned by the required native compiler."""

    from google.protobuf.message import DecodeError

    from .errors import SerializationError
    from .proto import qcore_pb2 as wire
    from .serialization import decode_type

    encoded = wire.TypeProto()
    try:
        if not isinstance(result, (bytes, bytearray, memoryview)):
            raise TypeError("native type payload is not bytes-like")
        payload = bytes(result)
        encoded.ParseFromString(payload)
        decoded = decode_type(encoded)
        if _encode_native_type(decoded) != payload:
            raise DecodeError("noncanonical native type")
        return decoded
    except (DecodeError, SerializationError, TypeError) as error:
        raise RuntimeError(
            "Q4512: native type transformation returned a malformed type"
        ) from error


def _inspect_native_type_shape(
    type_spec: TypeSpec,
) -> tuple[
    int | None,
    tuple[int, ...] | None,
    TypeSpec | None,
    tuple[TypeSpec, ...] | None,
]:
    """Validate the compiler's complete structural projection for one type."""

    inspect = getattr(native, "inspect_type_shape", None)
    if not callable(inspect):
        raise TypeError("Q4511: native type-shape projection is unavailable")
    result = inspect(_encode_native_type(type_spec))
    if type(result) is not tuple or len(result) != 4:
        raise RuntimeError("Q4512: native type-shape projection is malformed")
    width, leaf_widths, sequence, aggregate = result
    if width is not None and (type(width) is not int or width <= 0):
        raise RuntimeError("Q4512: native type-shape projection has an invalid width")
    if leaf_widths is not None and (
        type(leaf_widths) is not list
        or any(type(value) is not int or value <= 0 for value in leaf_widths)
    ):
        raise RuntimeError(
            "Q4512: native type-shape projection has invalid coherent widths"
        )
    if sequence is not None and not isinstance(
        sequence, (bytes, bytearray, memoryview)
    ):
        raise RuntimeError(
            "Q4512: native type-shape projection has an invalid sequence element"
        )
    if aggregate is not None and (
        type(aggregate) is not list
        or any(
            not isinstance(element, (bytes, bytearray, memoryview))
            for element in aggregate
        )
    ):
        raise RuntimeError(
            "Q4512: native type-shape projection has invalid aggregate elements"
        )
    return (
        width,
        None if leaf_widths is None else tuple(leaf_widths),
        None if sequence is None else _decode_native_type(sequence),
        None
        if aggregate is None
        else tuple(_decode_native_type(element) for element in aggregate),
    )


def _inspect_native_type_admissibility(
    type_spec: TypeSpec,
) -> tuple[bool, bool, bool]:
    """Validate coherent and staged constructor facts from the compiler."""

    inspect = getattr(native, "inspect_type_admissibility", None)
    if not callable(inspect):
        raise TypeError("Q4511: native type admissibility is unavailable")
    result = inspect(_encode_native_type(type_spec))
    if (
        type(result) is not tuple
        or len(result) != 3
        or any(type(value) is not bool for value in result)
    ):
        raise RuntimeError("Q4512: native type admissibility is malformed")
    return result


def logical_width(type_spec: TypeSpec) -> int | None:
    """Return a fixed logical bit width when one is known.

    Args:
        type_spec: Semantic type to inspect.

    Returns:
        The fixed bit width, or ``None`` for another or symbolic shape.
    """

    return _inspect_native_type_shape(type_spec)[0]


def coherent_leaf_widths(type_spec: TypeSpec) -> tuple[int, ...] | None:
    """Return the ordered widths of coherent bit-state leaves.

    Args:
        type_spec: A quantum bit state or product of quantum bit states.

    Returns:
        Fixed leaf widths, or ``None`` for a symbolic or non-coherent shape.
    """

    return _inspect_native_type_shape(type_spec)[1]


def value_conforms(type_spec: TypeSpec, value: object) -> bool:
    """Return whether a host value inhabits a classical semantic type.

    This predicate is shared by specialization and launch ABIs. Python only
    canonicalizes familiar host containers into the portable literal domain;
    Rust owns scalar, shape, record, observable, and staging semantics.

    Args:
        type_spec: Expected semantic type, optionally staged.
        value: Host value to validate.

    Returns:
        ``True`` when the value has the required scalar kind and shape.
    """

    conforms = getattr(native, "literal_conforms_type", None)
    if not callable(conforms):
        raise TypeError("Q4511: native literal/type relation is unavailable")

    try:
        encoded = _encode_native_literal(value)
    except (SerializationError, TypeError, ValueError):
        return False
    result = conforms(
        _encode_native_type(type_spec),
        encoded,
    )
    if type(result) is not bool:
        raise RuntimeError("Q4512: native literal/type relation is malformed")
    return result


def _staged(stage: str, item: object) -> StagedType:
    """Construct and validate a classical staged type."""

    return StagedType(stage, as_type(item))


def _dimension(value: object) -> Dimension:
    """Validate a static or symbolic shape dimension."""

    if isinstance(value, bool):
        raise TypeCheckError("A shape dimension cannot be boolean.")
    if isinstance(value, int):
        if value <= 0:
            raise TypeCheckError("A shape dimension must be positive.")
        return value
    if isinstance(value, str) and value:
        return value
    raise TypeCheckError("A shape dimension must be a positive integer or a name.")


def _fixed_dimension(value: object) -> Dimension:
    """Validate one fixed-point partition width.

    Args:
        value (object): Concrete nonnegative width or dependent width name.

    Returns:
        Dimension: The validated partition width.

    Raises:
        TypeCheckError: If ``value`` is boolean, negative, or not a dimension.
    """

    if isinstance(value, bool):
        raise TypeCheckError("A fixed-point partition cannot be boolean.")
    if isinstance(value, int):
        if value < 0:
            raise TypeCheckError("A fixed-point partition cannot be negative.")
        return value
    if isinstance(value, str) and value:
        return value
    raise TypeCheckError(
        "A fixed-point partition must be a nonnegative integer or a name."
    )
