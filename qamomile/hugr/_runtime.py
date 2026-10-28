"""Adapt a compiled HUGR function to tagged, argument-free service execution.

Selene and Nexus execute a zero-argument entrypoint. Each submission receives
a thin typed caller around the existing compiled function; runtime values never
pass through Qamomile's compile-time binding or structural specialization path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any

import numpy as np

from qamomile.circuit.ir.types import BitType, FloatType, UIntType
from qamomile.circuit.ir.value import ArrayValue, DictValue, TupleValue, ValueLike
from qamomile.circuit.transpiler import (
    CompiledProgram,
    dict_param_key,
    flatten_user_bindings,
)


class _Kind(StrEnum):
    """Enumerate the closed set of runtime ABI scalar and container kinds."""

    BIT = "bit"
    FLOAT = "float"
    UINT = "uint"
    ARRAY = "array"
    TUPLE = "tuple"
    DICT = "dict"


@dataclass(frozen=True)
class _Schema:
    """Describe a native carrier and its corresponding public result shape.

    Args:
        kind (_Kind): Scalar, array, tuple, or dictionary discriminator.
        native (Any): Native HUGR carrier type.
        children (tuple[_Schema, ...]): Ordered child carriers.
        shape (tuple[int, ...]): Public array dimensions.
        element_kind (_Kind | None): Array element type, including empty arrays.
        keys (tuple[Any, ...] | None): Fixed dictionary keys in ABI order, or
            ``None`` when keys themselves remain runtime values.
    """

    kind: _Kind
    native: Any
    children: tuple[_Schema, ...] = ()
    shape: tuple[int, ...] = ()
    element_kind: _Kind | None = None
    keys: tuple[Any, ...] | None = None

    @property
    def leaves(self) -> tuple[_Schema, ...]:
        """Return scalar schemas in stable carrier order.

        Returns:
            tuple[_Schema, ...]: Recursively ordered scalar schemas.
        """
        if self.kind in (_Kind.BIT, _Kind.FLOAT, _Kind.UINT):
            return (self,)
        return tuple(leaf for child in self.children for leaf in child.leaves)


def _tuple_types(native: Any, count: int) -> list[Any]:
    """Check a fixed tuple carrier and return its element types.

    Args:
        native (Any): HUGR type to inspect.
        count (int): Required number of tuple elements.

    Returns:
        list[Any]: Native tuple element types.

    Raises:
        ValueError: If the native carrier disagrees with the public ABI.
    """
    rows = getattr(native, "variant_rows", ())
    if len(rows) != 1 or len(rows[0]) != count:
        raise ValueError("HUGR tuple carrier does not match the public ABI")
    return list(rows[0])


def _schema(value: ValueLike, native: Any) -> _Schema:
    """Match a public ABI value to a native HUGR carrier.

    Args:
        value (ValueLike): Public input or output value.
        native (Any): Corresponding native function port type.

    Returns:
        _Schema: Validated recursive carrier schema.

    Raises:
        ValueError: If shapes, dictionary keys, or native carrier types disagree
            with the ABI.
        TypeError: If a public scalar type is unsupported.
    """
    from hugr import tys

    if isinstance(value, ArrayValue):
        from qamomile.hugr.lowerer import _lower_type

        if not value.shape or any(not size.is_constant() for size in value.shape):
            raise ValueError("HUGR runtime arrays require fixed, resolved dimensions")
        shape = tuple(int(size.get_const()) for size in value.shape)
        if any(size < 0 for size in shape):
            raise ValueError("HUGR runtime array dimensions must be non-negative")
        native_children = _tuple_types(native, math.prod(shape))
        children = tuple(_scalar_schema(value, ty) for ty in native_children)
        element_kind = _scalar_schema(value, _lower_type(value.type)).kind
        return _Schema(_Kind.ARRAY, native, children, shape, element_kind)
    if isinstance(value, TupleValue):
        native_children = _tuple_types(native, len(value.elements))
        return _Schema(
            _Kind.TUPLE,
            native,
            tuple(
                _schema(child, ty)
                for child, ty in zip(value.elements, native_children, strict=True)
            ),
        )
    if isinstance(value, DictValue):
        from qamomile.hugr.lowerer import _dict_entries

        entries = _dict_entries(value)
        native_children = _tuple_types(native, len(entries))
        children = []
        for (key, item), pair_type in zip(entries, native_children, strict=True):
            key_type, item_type = _tuple_types(pair_type, 2)
            children.append(
                _Schema(
                    _Kind.TUPLE,
                    tys.Tuple(key_type, item_type),
                    (_schema(key, key_type), _schema(item, item_type)),
                )
            )
        keys = tuple(_fixed_dictionary_key(key) for key, _ in entries)
        if None not in keys and len(set(keys)) != len(keys):
            raise ValueError(
                f"HUGR dictionary {value.name!r} has duplicate fixed keys in its public ABI"
            )
        return _Schema(
            _Kind.DICT,
            native,
            tuple(children),
            keys=None if any(key is None for key in keys) else keys,
        )
    return _scalar_schema(value, native)


def _fixed_dictionary_key(value: ValueLike) -> int | tuple[int, ...] | None:
    """Read a fixed integer dictionary key without inferring runtime keys.

    Args:
        value (ValueLike): Dictionary key described by the public ABI.

    Returns:
        int | tuple[int, ...] | None: Known key supported by shared binding
        names, or ``None`` for a key requiring a whole dictionary binding.
    """
    if isinstance(value, TupleValue):
        components = []
        for child in value.elements:
            component = _fixed_dictionary_key(child)
            if not isinstance(component, int):
                return None
            components.append(component)
        return tuple(components)
    if isinstance(value.type, UIntType) and value.is_constant():
        constant = value.get_const()
        if isinstance(constant, int) and not isinstance(constant, bool):
            return constant
    return None


def _scalar_schema(value: ValueLike, native: Any) -> _Schema:
    """Validate one native scalar carrier against its semantic type.

    Args:
        value (ValueLike): Scalar value or array carrying its element type.
        native (Any): Native scalar HUGR type.

    Returns:
        _Schema: Scalar carrier schema.

    Raises:
        TypeError: If the semantic scalar type has no execution ABI.
        ValueError: If the native type disagrees with the public ABI.
    """
    from hugr import tys

    type_name = getattr(getattr(native, "type_def", None), "name", None)
    if isinstance(value.type, BitType) and native == tys.Bool:
        return _Schema(_Kind.BIT, native)
    if isinstance(value.type, FloatType) and type_name == "float64":
        return _Schema(_Kind.FLOAT, native)
    if isinstance(value.type, UIntType) and type_name == "int":
        return _Schema(_Kind.UINT, native)
    if not isinstance(value.type, (BitType, FloatType, UIntType)):
        raise TypeError(f"Unsupported HUGR execution value type: {value.type.label()}")
    raise ValueError(f"HUGR carrier for {value.name!r} disagrees with its public type")


def _normalize_scalar(schema: _Schema, value: Any, name: str) -> Any:
    """Validate a scalar without silently changing its semantic domain.

    Args:
        schema (_Schema): Required scalar carrier.
        value (Any): Candidate public value or tagged result.
        name (str): User-facing binding or output name for errors.

    Returns:
        Any: Built-in boolean, finite floating-point number, or unsigned integer.

    Raises:
        TypeError: If the scalar has the wrong Python type.
        ValueError: If a scalar is outside its carrier range or is not finite.
    """
    is_bool = isinstance(value, (bool, np.bool_))
    if schema.kind == _Kind.BIT:
        if not is_bool and not isinstance(value, numbers.Integral):
            raise TypeError(f"Bit {name!r} must be bool, 0, or 1")
        if int(value) not in (0, 1):
            raise ValueError(f"Bit {name!r} must be 0 or 1")
        return bool(value)
    if schema.kind == _Kind.FLOAT:
        if is_bool or not isinstance(value, (numbers.Real, Decimal)):
            raise TypeError(f"Float {name!r} must be a real number")
        try:
            normalized = float(value)
        except (OverflowError, ValueError) as error:
            raise ValueError(f"Float {name!r} must be finite") from error
        if not math.isfinite(normalized):
            raise ValueError(f"Float {name!r} must be finite")
        return normalized
    if is_bool or not isinstance(value, numbers.Integral):
        raise TypeError(f"UInt {name!r} must be an integer")
    width = 1 << schema.native.args[0].n
    if not 0 <= int(value) < (1 << width):
        raise ValueError(f"UInt {name!r} must be in [0, 2**{width})")
    return int(value)


def _normalize_value(schema: _Schema, value: Any, name: str) -> Any:
    """Validate and canonicalize a public runtime value without native constants.

    Args:
        schema (_Schema): Required native input schema.
        value (Any): Public runtime argument.
        name (str): Public argument path used in diagnostics.

    Returns:
        Any: Independently owned value containing canonical scalar types and
        preserving the public container shape and dictionary ABI order.

    Raises:
        TypeError: If a runtime value has an incompatible Python type.
        ValueError: If its shape, keys, or scalar domain is incompatible.
    """
    if schema.kind in (_Kind.BIT, _Kind.FLOAT, _Kind.UINT):
        return _normalize_scalar(schema, value, name)
    if schema.kind == _Kind.ARRAY:
        array = np.asarray(value, dtype=object)
        if array.shape != schema.shape:
            raise ValueError(
                f"HUGR runtime array {name!r} has shape {array.shape}; expected {schema.shape}"
            )
        values = list(array.flat)
    elif schema.kind == _Kind.DICT:
        if not isinstance(value, Mapping):
            raise TypeError(f"HUGR runtime dictionary {name!r} requires a mapping")
        values = list(value.items())
        if schema.keys is not None:
            missing = set(schema.keys) - value.keys()
            unexpected = value.keys() - set(schema.keys)
            if missing or unexpected:
                raise ValueError(
                    f"HUGR runtime dictionary {name!r} has incompatible keys; "
                    f"missing {sorted(missing, key=repr)}, "
                    f"unexpected {sorted(unexpected, key=repr)}"
                )
            # Keep the supplied key objects so scalar validation still rejects
            # float/bool keys that compare equal to the required UInt keys.
            supplied = {key: (key, item) for key, item in values}
            values = [supplied[key] for key in schema.keys]
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"HUGR runtime tuple {name!r} requires a sequence")
        values = list(value)
    if len(values) != len(schema.children):
        raise ValueError(
            f"HUGR runtime value {name!r} has the wrong number of elements"
        )
    return _restore_container(
        schema,
        [
            _normalize_value(child, item, f"{name}[{index}]")
            for index, (child, item) in enumerate(
                zip(schema.children, values, strict=True)
            )
        ],
    )


def _constant(schema: _Schema, value: Any) -> Any:
    """Build a native constant from an already validated canonical runtime value.

    Args:
        schema (_Schema): Required native input schema.
        value (Any): Canonical value returned by ``resolve_runtime_bindings``.

    Returns:
        Any: Native HUGR constant retaining the compiled input type.

    Raises:
        ImportError: If optional HUGR dependencies are unavailable.
    """
    from hugr import val
    from hugr.std.float import FloatVal
    from hugr.std.int import IntVal

    if schema.kind == _Kind.BIT:
        return val.TRUE if value else val.FALSE
    if schema.kind == _Kind.FLOAT:
        return FloatVal(value)
    if schema.kind == _Kind.UINT:
        log_width = schema.native.args[0].n
        width = 1 << log_width
        signed = value if value < (1 << (width - 1)) else value - (1 << width)
        return IntVal(signed, width=log_width)
    if schema.kind == _Kind.ARRAY:
        values = list(np.asarray(value, dtype=object).flat)
    elif schema.kind == _Kind.DICT:
        values = list(value.items())
    else:
        values = list(value)
    return val.Tuple(
        *(
            _constant(child, item)
            for child, item in zip(schema.children, values, strict=True)
        )
    )


def _record_outputs(builder: Any, schema: _Schema, wire: Any, tags: Any) -> None:
    """Record every public leaf exactly once through the result extension.

    Args:
        builder (Any): HUGR entrypoint builder.
        schema (_Schema): Native output schema.
        wire (Any): Output carrier wire.
        tags (Any): Iterator over preallocated stable leaf tags.
    """
    import tket_exts

    from hugr import ops, tys

    if schema.kind in (_Kind.BIT, _Kind.FLOAT, _Kind.UINT):
        tag = next(tags)
        if schema.kind == _Kind.BIT:
            operation = tket_exts.result.result_bool(tag)
        elif schema.kind == _Kind.FLOAT:
            operation = tket_exts.result.result_f64(tag)
        else:
            operation = tket_exts.result.result_uint_def.instantiate(
                [tys.StringArg(tag), schema.native.args[0]],
                tys.FunctionType([schema.native], []),
            )
        recorded = builder.add_op(operation, wire)
        # Result operations have no data outputs, so state-order edges keep
        # their externally visible effects alive through compiler DCE.
        builder.add_state_order(builder.input_node, recorded)
        builder.add_state_order(recorded, builder.output_node)
        return
    wires = builder.add_op(
        ops.UnpackTuple([child.native for child in schema.children]), wire
    )
    for child, child_wire in zip(schema.children, wires, strict=True):
        _record_outputs(builder, child, child_wire, tags)


def _reshape(values: list[Any], shape: tuple[int, ...]) -> tuple[Any, ...]:
    """Restore row-major array leaves as recursively nested tuples.

    Args:
        values (list[Any]): Flattened array leaves.
        shape (tuple[int, ...]): Concrete public dimensions.

    Returns:
        tuple[Any, ...]: Array result retaining all dimensions.
    """
    if len(shape) == 1:
        return tuple(values)
    stride = math.prod(shape[1:])
    return tuple(
        _reshape(values[index * stride : (index + 1) * stride], shape[1:])
        for index in range(shape[0])
    )


def _restore_container(schema: _Schema, values: list[Any]) -> Any:
    """Restore a public container from canonical values in native carrier order.

    Args:
        schema (_Schema): Public container schema.
        values (list[Any]): Canonical scalar leaves or reconstructed children.

    Returns:
        Any: Public tuple, dictionary, or shape-preserving empty array.

    Raises:
        TypeError: If a dictionary key has no hashable public representation.
        ValueError: If canonical dictionary keys collide.
    """
    if schema.kind == _Kind.ARRAY:
        if 0 in schema.shape[:-1]:
            # Nested tuples cannot encode dimensions below an empty axis.
            # Preserve those otherwise invisible dimensions in an ndarray.
            match schema.element_kind:
                case _Kind.BIT:
                    dtype = np.bool_
                case _Kind.UINT:
                    dtype = np.uint64
                case _:
                    dtype = np.float64
            return np.empty(schema.shape, dtype=dtype)
        return _reshape(values, schema.shape)
    if schema.kind == _Kind.DICT:
        mapping = dict(values)
        if len(mapping) != len(values):
            raise ValueError("HUGR dictionary contains duplicate canonical keys")
        return mapping
    return tuple(values)


def _decode(schema: _Schema, entries: Mapping[str, Any], tags: Any) -> Any:
    """Restore one public value from validated tagged scalar leaves.

    Args:
        schema (_Schema): Output reconstruction schema.
        entries (Mapping[str, Any]): Unique tag-to-scalar entries.
        tags (Any): Iterator over output tags in carrier order.

    Returns:
        Any: Public scalar, tuple, dictionary, or shape-preserving empty array.

    Raises:
        TypeError: If a tagged value violates its scalar type.
        ValueError: If a tagged scalar is outside its domain or is not finite.
    """
    if schema.kind in (_Kind.BIT, _Kind.FLOAT, _Kind.UINT):
        tag = next(tags)
        return _normalize_scalar(schema, entries[tag], tag)
    return _restore_container(
        schema, [_decode(child, entries, tags) for child in schema.children]
    )


def _schema_identity(schema: _Schema) -> dict[str, Any]:
    """Describe carrier kinds and public dimensions for typed restoration identity.

    Args:
        schema (_Schema): Recursive native and public ABI schema.

    Returns:
        dict[str, Any]: Deterministic JSON-compatible identity components.
    """
    return {
        "kind": schema.kind.value,
        "native": str(schema.native),
        "shape": schema.shape,
        "element_kind": schema.element_kind,
        "keys": schema.keys,
        "children": [_schema_identity(child) for child in schema.children],
    }


@dataclass(frozen=True)
class PreparedExecution:
    """Hold a submission package and its independent output reconstruction.

    Args:
        package (Any): HUGR package with a zero-argument reporting entrypoint.
        output_tags (tuple[str, ...]): Stable required scalar output tags.
        _outputs (tuple[_Schema, ...]): Public output schemas in port order.
        _inputs (tuple[tuple[str, _Schema], ...]): Runtime names and input schemas.
    """

    package: Any
    output_tags: tuple[str, ...]
    _outputs: tuple[_Schema, ...]
    _inputs: tuple[tuple[str, _Schema], ...] = ()

    @property
    def abi_sha256(self) -> str:
        """Identify public shapes and types independently of native graph equality.

        Returns:
            str: SHA-256 digest of versioned public input and output schemas.
        """
        identity = {
            "version": 1,
            "inputs": [
                (name, _schema_identity(schema)) for name, schema in self._inputs
            ],
            "outputs": [_schema_identity(schema) for schema in self._outputs],
            "tags": self.output_tags,
        }
        encoded = json.dumps(identity, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def decode_shot(self, entries: Mapping[str, Any]) -> Any:
        """Decode one shot and reject incomplete or unexpected result records.

        Args:
            entries (Mapping[str, Any]): One scalar entry per output tag.

        Returns:
            Any: Scalar, nested tuple, dictionary, shape-preserving empty array,
            or ``None`` matching the ABI.

        Raises:
            TypeError: If a tagged carrier has the wrong Python type.
            ValueError: If tags are missing or unexpected or a scalar is invalid.
        """
        expected = set(self.output_tags)
        missing, unexpected = expected - entries.keys(), entries.keys() - expected
        if missing:
            raise ValueError(f"Missing HUGR output tags: {sorted(missing)}")
        if unexpected:
            raise ValueError(f"Unexpected HUGR output tags: {sorted(unexpected)}")
        tags = iter(self.output_tags)
        outputs = tuple(_decode(schema, entries, tags) for schema in self._outputs)
        return outputs[0] if len(outputs) == 1 else outputs or None


def _runtime_input_names(compiled: CompiledProgram[Any]) -> tuple[str, ...]:
    """Validate public binding names against explicit runtime input metadata.

    Args:
        compiled (CompiledProgram[Any]): Artifact and public input ABI.

    Returns:
        tuple[str, ...]: Runtime input names in native function port order.

    Raises:
        ValueError: If runtime metadata disagrees with the public ABI.
    """
    names = tuple(
        compiled.metadata.properties.get("runtime_inputs", compiled.abi.public_inputs)
    )
    if len(set(names)) != len(names) or any(
        name not in compiled.abi.public_inputs for name in names
    ):
        raise ValueError("HUGR runtime input metadata disagrees with the public ABI")
    return names


def _resolve_binding(
    schema: _Schema,
    name: str,
    raw: Mapping[str, Any],
    indexed: Mapping[str, Any],
    consumed: set[str],
) -> Any:
    """Reconstruct one typed argument from whole or shared indexed bindings.

    Args:
        schema (_Schema): Required native and public carrier schema.
        name (str): Public argument path to resolve.
        raw (Mapping[str, Any]): Original bindings, including whole containers.
        indexed (Mapping[str, Any]): Shared flattened binding names and values.
        consumed (set[str]): Raw and indexed names consumed by reconstruction.

    Returns:
        Any: Scalar or reconstructed whole argument retaining ABI order.

    Raises:
        ValueError: If required values are missing, partial arrays have the wrong
            shape, or dictionary keys are not statically known for indexing.
    """
    if name in raw:
        consumed.add(name)
        consumed.update(flatten_user_bindings({name: raw[name]}))
        return raw[name]
    if name in indexed:
        consumed.add(name)
        return indexed[name]
    if schema.kind == _Kind.ARRAY and schema.children:
        # Check partial rows before flattening can erase their declared shape.
        for depth in range(1, len(schema.shape)):
            prefixes = flatten_user_bindings(
                {name: np.empty(schema.shape[:depth], dtype=object)}
            )
            for prefix in prefixes.keys() & raw.keys():
                shape = np.asarray(raw[prefix], dtype=object).shape
                if shape != schema.shape[depth:]:
                    raise ValueError(
                        f"HUGR runtime array {prefix!r} has shape {shape}; "
                        f"expected {schema.shape[depth:]}"
                    )
        template = np.empty(schema.shape, dtype=object)
        names = flatten_user_bindings({name: template})
        values = [
            _resolve_binding(child, key, raw, indexed, consumed)
            for child, key in zip(schema.children, names, strict=True)
        ]
        return _reshape(values, schema.shape)
    if schema.kind == _Kind.TUPLE and schema.children:
        return tuple(
            _resolve_binding(child, f"{name}[{index}]", raw, indexed, consumed)
            for index, child in enumerate(schema.children)
        )
    if schema.kind == _Kind.DICT and schema.children:
        if schema.keys is None:
            raise ValueError(
                f"HUGR runtime dictionary {name!r} requires a whole mapping; "
                "indexed bindings require fixed integer keys in the public ABI"
            )
        return {
            key: _resolve_binding(
                pair.children[1], dict_param_key(name, key), raw, indexed, consumed
            )
            for key, pair in zip(schema.keys, schema.children, strict=True)
        }
    raise ValueError(f"Missing HUGR runtime bindings: {[name]}")


def resolve_runtime_bindings(
    compiled: CompiledProgram[Any], bindings: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Restore public argument values from whole or indexed runtime bindings.

    Whole and indexed forms may be used for different arguments. Specifying a
    container alongside any of its indexed descendants is ambiguous and rejected.
    Fixed dictionary entries follow the public ABI order; dictionaries whose
    keys remain runtime values require an explicit whole mapping.

    Args:
        compiled (CompiledProgram[Any]): Artifact and public runtime input ABI.
        bindings (Mapping[str, Any] | None): Public names or shared scalar keys,
            such as ``angles[1][2]`` and ``coeffs[(0, 1)]``.

    Returns:
        dict[str, Any]: Independently owned whole arguments in native port order,
        containing canonical ``bool``, ``int``, and finite ``float`` values.
        Input mappings and their containers are never mutated.

    Raises:
        TypeError: If binding names or values have incompatible types.
        ValueError: If names conflict, are missing or unexpected, or values
            violate the fixed shape or scalar domain of the ABI.
        EmitError: If a public input has no supported native HUGR carrier.
        ImportError: If optional HUGR dependencies are unavailable.
    """
    from qamomile.hugr.lowerer import _lower_value_type

    raw = dict(bindings or {})
    if any(not isinstance(name, str) for name in raw):
        raise TypeError("HUGR runtime binding names must be strings")
    for name in raw:
        for index, character in enumerate(name):
            if character == "[" and name[:index] in raw:
                raise ValueError(
                    f"Conflicting HUGR runtime bindings: {name[:index]!r} and {name!r}"
                )
    indexed = flatten_user_bindings(raw)
    consumed: set[str] = set()
    resolved = {}
    for name in _runtime_input_names(compiled):
        value = compiled.abi.public_inputs[name]
        schema = _schema(value, _lower_value_type(value))
        restored = _resolve_binding(schema, name, raw, indexed, consumed)
        resolved[name] = _normalize_value(schema, restored, name)
    unexpected = set(indexed) - consumed
    for name, value in raw.items():
        if name not in consumed and not set(flatten_user_bindings({name: value})):
            unexpected.add(name)
    if unexpected:
        raise ValueError(f"Unexpected HUGR runtime bindings: {sorted(unexpected)}")
    return resolved


def validate_runtime_bindings(
    compiled: CompiledProgram[Any], bindings: Mapping[str, Any] | None = None
) -> None:
    """Validate runtime values even when an identity observable needs no shots.

    Args:
        compiled (CompiledProgram[Any]): Artifact or identity-only expectation
            descriptor retaining its public ABI and runtime input metadata.
        bindings (Mapping[str, Any] | None): Runtime public argument values.

    Raises:
        TypeError: If a binding has an incompatible type.
        ValueError: If names, scalar domains, or array dimensions are invalid.
        EmitError: If a public input has no supported native HUGR carrier.
        ImportError: If optional HUGR dependencies are unavailable.
    """
    resolve_runtime_bindings(compiled, bindings)


def prepare_execution(
    compiled: CompiledProgram[Any],
    bindings: Mapping[str, Any] | None = None,
) -> PreparedExecution:
    """Wrap an existing compiled function with typed arguments and output tags.

    Args:
        compiled (CompiledProgram[Any]): HUGR artifact and public ABI.
        bindings (Mapping[str, Any] | None): Runtime values keyed by public
            argument name or shared indexed key. Compile-time bindings cannot
            be overridden here.

    Returns:
        PreparedExecution: Independent, validated package ready for submission.

    Raises:
        ImportError: If optional HUGR dependencies are unavailable.
        TypeError: If a parameter has an incompatible scalar or container type.
        ValueError: If bindings, the entrypoint, or native/public ABI disagree.
        HugrCliError: If native validation rejects the generated caller.
    """
    import tket_exts

    from hugr import ops
    from hugr.build import Module
    from hugr.cli import validate
    from hugr.package import Package

    runtime = resolve_runtime_bindings(compiled, bindings)
    input_names = tuple(runtime)
    if len(compiled.artifact.modules) != 1:
        raise ValueError("HUGR execution requires one compiled module")
    graph = copy.deepcopy(compiled.artifact.modules[0])
    functions = [
        (node, data.op)
        for node, data in graph.nodes()
        if isinstance(data.op, ops.FuncDefn)
    ]
    candidates = [(node, op) for node, op in functions if op.f_name == "main"]
    if len(candidates) != 1:
        raise ValueError("HUGR execution requires exactly one main function")
    original, signature = candidates[0]
    if signature.params:
        raise ValueError("HUGR execution requires a monomorphic main function")
    if len(signature.inputs) != len(input_names):
        raise ValueError("HUGR native input ports disagree with the public ABI")
    if len(signature.outputs) != len(compiled.abi.output_values):
        raise ValueError("HUGR native output ports disagree with the public ABI")
    names = {op.f_name for _, op in functions}
    implementation_name = "_qamomile_runtime_program"
    while implementation_name in names:
        implementation_name += "_"
    signature.f_name = implementation_name
    signature.visibility = "Private"
    module = Module(graph)
    entry = module.define_function("main", [], [], visibility="Public")
    arguments = []
    input_schemas = []
    for name, native in zip(input_names, signature.inputs, strict=True):
        value = compiled.abi.public_inputs[name]
        schema = _schema(value, native)
        input_schemas.append((name, schema))
        [wire] = entry.load(_constant(schema, runtime[name]))
        arguments.append(wire)
    call = entry.call(original, *arguments)
    entry.add_state_order(entry.input_node, call)
    entry.add_state_order(call, entry.output_node)
    output_schemas = tuple(
        _schema(value, native)
        for value, native in zip(
            compiled.abi.output_values, signature.outputs, strict=True
        )
    )
    tags = tuple(
        f"qamomile.output.{port}.{leaf}"
        for port, schema in enumerate(output_schemas)
        for leaf in range(len(schema.leaves))
    )
    tag_iter = iter(tags)
    for schema, wire in zip(output_schemas, call, strict=True):
        _record_outputs(entry, schema, wire, tag_iter)
    entry.set_outputs()
    module.hugr.entrypoint = entry.parent_node
    extensions = list(compiled.artifact.extensions)
    if not any(extension.name == "tket.result" for extension in extensions):
        extensions.append(tket_exts.result())
    package = Package([module.hugr], extensions)
    validate(package.to_bytes())
    return PreparedExecution(package, tags, output_schemas, tuple(input_schemas))
