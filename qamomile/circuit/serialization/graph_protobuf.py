"""Convert semantic IR graph records to the qkernel protobuf schema."""

from __future__ import annotations

import struct
from typing import Any

from qamomile.circuit.serialization.proto import qamomile_ir_pb2 as pb
from qamomile.circuit.serialization.schema import QAMOMILE_VERSION

_PARAMETER_KIND_TO_PROTO: dict[str, pb.ParameterKind] = {
    "POSITIONAL_ONLY": pb.POSITIONAL_ONLY,
    "POSITIONAL_OR_KEYWORD": pb.POSITIONAL_OR_KEYWORD,
    "VAR_POSITIONAL": pb.VAR_POSITIONAL,
    "KEYWORD_ONLY": pb.KEYWORD_ONLY,
    "VAR_KEYWORD": pb.VAR_KEYWORD,
}
_PARAMETER_KIND_FROM_PROTO: dict[int, str] = {
    int(value): name for name, value in _PARAMETER_KIND_TO_PROTO.items()
}
_FRONTEND_ANNOTATION_KIND_TO_PROTO: dict[str, pb.FrontendAnnotationKind] = {
    "QAMOMILE_UINT": pb.QAMOMILE_UINT,
    "PYTHON_INT": pb.PYTHON_INT,
    "QAMOMILE_FLOAT": pb.QAMOMILE_FLOAT,
    "PYTHON_FLOAT": pb.PYTHON_FLOAT,
    "QAMOMILE_BIT": pb.QAMOMILE_BIT,
    "PYTHON_BOOL": pb.PYTHON_BOOL,
    "QAMOMILE_QUBIT": pb.QAMOMILE_QUBIT,
    "QAMOMILE_QFIXED": pb.QAMOMILE_QFIXED,
    "QAMOMILE_OBSERVABLE": pb.QAMOMILE_OBSERVABLE,
    "QAMOMILE_VECTOR": pb.QAMOMILE_VECTOR,
    "QAMOMILE_MATRIX": pb.QAMOMILE_MATRIX,
    "QAMOMILE_TENSOR": pb.QAMOMILE_TENSOR,
    "QAMOMILE_TUPLE": pb.QAMOMILE_TUPLE,
    "QAMOMILE_DICT": pb.QAMOMILE_DICT,
    "PYTHON_TUPLE": pb.PYTHON_TUPLE,
}
_FRONTEND_ANNOTATION_KIND_FROM_PROTO: dict[int, str] = {
    int(value): name for name, value in _FRONTEND_ANNOTATION_KIND_TO_PROTO.items()
}


def qkernel_from_graph_dict(envelope: dict[str, Any]) -> pb.QKernel:
    """Convert the internal static-qkernel graph record into protobuf.

    This bridge keeps graph construction separate from the canonical wire
    schema. The resulting bytes contain only protobuf fields, never JSON or a
    pickled Python dictionary.

    Args:
        envelope (dict[str, Any]): Validated graph record from the qkernel
            encoder.

    Returns:
        pb.QKernel: Canonical qkernel protobuf graph.

    Raises:
        ValueError: If the graph envelope or qkernel interface is malformed.
        TypeError: If a nested payload is outside the closed protobuf union.
    """
    version = envelope.get("qamomile_version")
    if not isinstance(version, str):
        raise ValueError("graph envelope requires a string qamomile_version")
    artifact = envelope.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError("graph envelope requires an artifact record")
    if artifact.get("$type") != "QKernel":
        raise ValueError("graph envelope must contain one QKernel artifact")
    name = artifact.get("name")
    parameters = artifact.get("parameters")
    results = artifact.get("results")
    return_annotation = artifact.get("return_annotation")
    body = artifact.get("body")
    callable_definition = artifact.get("callable_definition")
    if (
        not isinstance(name, str)
        or not isinstance(parameters, list)
        or not isinstance(results, list)
        or not isinstance(return_annotation, dict)
        or not isinstance(body, dict)
        or not isinstance(callable_definition, dict)
    ):
        raise ValueError("QKernel graph artifact has a malformed interface")
    message = pb.QKernel(qamomile_version=version, name=name)
    for parameter in parameters:
        if not isinstance(parameter, dict):
            raise ValueError("QKernel parameter entries must be dictionaries")
        parameter_type = parameter.get("type")
        if not isinstance(parameter_type, dict):
            raise ValueError("QKernel parameter is missing its type")
        item = message.parameters.add(
            name=parameter.get("name", ""),
            has_default=parameter.get("has_default") is True,
            differentiable=parameter.get("differentiable") is True,
        )
        kind = parameter.get("kind")
        if kind not in _PARAMETER_KIND_TO_PROTO:
            raise ValueError(f"unknown qkernel parameter kind {kind!r}")
        item.kind = _PARAMETER_KIND_TO_PROTO[kind]
        item.type.CopyFrom(_kernel_type_to_proto(parameter_type))
        if item.has_default:
            item.default.CopyFrom(_payload_to_proto(parameter.get("default")))
    message.results.extend(_kernel_type_to_proto(item) for item in results)
    message.return_annotation.CopyFrom(_frontend_annotation_to_proto(return_annotation))
    message.body.CopyFrom(_block_to_proto(body))
    message.callable_definition.CopyFrom(
        _callable_definition_to_proto(callable_definition)
    )

    value_table = envelope.get("value_table")
    callable_table = envelope.get("callable_table")
    if not isinstance(value_table, list) or not isinstance(callable_table, list):
        raise ValueError("graph envelope requires value and callable tables")
    message.value_table.extend(_value_to_proto(item) for item in value_table)
    message.callable_table.extend(
        _callable_entry_to_proto(item) for item in callable_table
    )
    return message


def graph_dict_from_qkernel(message: pb.QKernel) -> dict[str, Any]:
    """Convert a qkernel protobuf into the internal semantic graph record.

    Args:
        message (pb.QKernel): Parsed qkernel graph.

    Returns:
        dict[str, Any]: Graph record accepted by the semantic decoders.

    Raises:
        ValueError: If the version, interface, or a typed protobuf field is
            missing or inconsistent.
    """
    if message.qamomile_version != QAMOMILE_VERSION:
        raise ValueError(
            "qamomile_version mismatch: payload reports "
            f"{message.qamomile_version!r}, this loader supports "
            f"{QAMOMILE_VERSION!r}. Cross-version migration is not provided."
        )
    if not message.HasField("body"):
        raise ValueError("protobuf qkernel is missing its body")
    if not message.HasField("callable_definition"):
        raise ValueError("protobuf qkernel is missing its callable definition")
    if not message.HasField("return_annotation"):
        raise ValueError("protobuf qkernel is missing its return annotation")
    seen_names: set[str] = set()
    parameters = []
    for parameter in message.parameters:
        if parameter.name in seen_names:
            raise ValueError(f"duplicate qkernel parameter {parameter.name!r}")
        seen_names.add(parameter.name)
        if not parameter.HasField("type"):
            raise ValueError(f"qkernel parameter {parameter.name!r} has no type")
        parameters.append(
            {
                "name": parameter.name,
                "type": _kernel_type_from_proto(parameter.type),
                "kind": _decode_parameter_kind(parameter.kind),
                "has_default": parameter.has_default,
                "default": (
                    _payload_from_proto(parameter.default)
                    if parameter.has_default
                    else None
                ),
                "differentiable": parameter.differentiable,
            }
        )
    return {
        "qamomile_version": message.qamomile_version,
        "artifact": {
            "$type": "QKernel",
            "name": message.name,
            "parameters": parameters,
            "results": [_kernel_type_from_proto(item) for item in message.results],
            "return_annotation": _frontend_annotation_from_proto(
                message.return_annotation
            ),
            "body": _block_from_proto(message.body),
            "callable_definition": _callable_definition_from_proto(
                message.callable_definition
            ),
        },
        "value_table": [_value_from_proto(item) for item in message.value_table],
        "callable_table": [
            _callable_entry_from_proto(item) for item in message.callable_table
        ],
    }


def _decode_parameter_kind(value: int) -> str:
    """Decode one protobuf qkernel parameter kind.

    Args:
        value (int): Protobuf enum value.

    Returns:
        str: Matching ``inspect.Parameter`` kind name.

    Raises:
        ValueError: If the enum value is unspecified or unknown.
    """
    kind = _PARAMETER_KIND_FROM_PROTO.get(value)
    if kind is None:
        raise ValueError(f"unknown qkernel parameter kind {value!r}")
    return kind


def _kernel_type_to_proto(value: dict[str, Any]) -> pb.KernelType:
    """Encode one qkernel interface type descriptor.

    Args:
        value (dict[str, Any]): IR value type plus array rank.

    Returns:
        pb.KernelType: Typed protobuf descriptor.

    Raises:
        ValueError: If the descriptor fields are malformed.
    """
    raw_value_type = value.get("value_type")
    ndim = value.get("ndim")
    annotation = value.get("annotation")
    if (
        not isinstance(raw_value_type, dict)
        or not isinstance(ndim, int)
        or not isinstance(annotation, dict)
    ):
        raise ValueError(
            "KernelType requires a ValueType, integer ndim, and annotation"
        )
    message = pb.KernelType(ndim=ndim)
    message.value_type.CopyFrom(_value_type_to_proto(raw_value_type))
    message.annotation.CopyFrom(_frontend_annotation_to_proto(annotation))
    return message


def _kernel_type_from_proto(message: pb.KernelType) -> dict[str, Any]:
    """Decode one protobuf qkernel interface type descriptor.

    Args:
        message (pb.KernelType): Typed protobuf descriptor.

    Returns:
        dict[str, Any]: IR value type plus array rank.

    Raises:
        ValueError: If the value type is absent.
    """
    if not message.HasField("value_type") or not message.HasField("annotation"):
        raise ValueError("KernelType is missing its ValueType or annotation")
    return {
        "value_type": _value_type_from_proto(message.value_type),
        "ndim": message.ndim,
        "annotation": _frontend_annotation_from_proto(message.annotation),
    }


def _frontend_annotation_to_proto(
    value: dict[str, Any],
) -> pb.FrontendAnnotation:
    """Encode one recursive frontend annotation descriptor.

    Args:
        value (dict[str, Any]): Annotation kind and nested arguments.

    Returns:
        pb.FrontendAnnotation: Typed protobuf annotation.

    Raises:
        ValueError: If the kind or arguments are malformed.
    """
    kind = value.get("kind")
    arguments = value.get("arguments")
    if kind not in _FRONTEND_ANNOTATION_KIND_TO_PROTO:
        raise ValueError(f"unknown frontend annotation kind {kind!r}")
    if not isinstance(arguments, list):
        raise ValueError("frontend annotation arguments must be a list")
    message = pb.FrontendAnnotation(kind=_FRONTEND_ANNOTATION_KIND_TO_PROTO[kind])
    message.arguments.extend(_frontend_annotation_to_proto(item) for item in arguments)
    return message


def _frontend_annotation_from_proto(
    message: pb.FrontendAnnotation,
) -> dict[str, Any]:
    """Decode one recursive protobuf frontend annotation.

    Args:
        message (pb.FrontendAnnotation): Typed protobuf annotation.

    Returns:
        dict[str, Any]: Annotation kind and nested arguments.

    Raises:
        ValueError: If the enum value is unspecified or unknown.
    """
    kind = _FRONTEND_ANNOTATION_KIND_FROM_PROTO.get(message.kind)
    if kind is None:
        raise ValueError(f"unknown frontend annotation kind {message.kind!r}")
    return {
        "kind": kind,
        "arguments": [
            _frontend_annotation_from_proto(item) for item in message.arguments
        ],
    }


def _integer_to_proto(value: int) -> pb.BigInteger:
    """Encode an arbitrary-precision Python integer.

    Args:
        value (int): Integer to encode. Booleans are not accepted.

    Returns:
        pb.BigInteger: Sign plus unsigned big-endian magnitude.

    Raises:
        TypeError: If ``value`` is not a plain integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected int, got {type(value).__name__}")
    magnitude = abs(value)
    size = (magnitude.bit_length() + 7) // 8
    return pb.BigInteger(
        negative=value < 0,
        magnitude=magnitude.to_bytes(size, "big"),
    )


def _integer_from_proto(message: pb.BigInteger) -> int:
    """Decode an arbitrary-precision protobuf integer.

    Args:
        message (pb.BigInteger): Sign and canonical unsigned magnitude.

    Returns:
        int: Reconstructed Python integer.

    Raises:
        ValueError: If the magnitude has a redundant leading zero or encodes
            negative zero.
    """
    magnitude_bytes = bytes(message.magnitude)
    if magnitude_bytes.startswith(b"\x00"):
        raise ValueError("BigInteger magnitude is not canonical")
    magnitude = int.from_bytes(magnitude_bytes, "big")
    if message.negative and magnitude == 0:
        raise ValueError("BigInteger cannot encode negative zero")
    return -magnitude if message.negative else magnitude


def _float_bits(value: float) -> int:
    """Return the exact IEEE-754 bits of a Python float.

    Args:
        value (float): Python binary64 value.

    Returns:
        int: Unsigned 64-bit bit pattern.
    """
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def _float_from_bits(bits: int) -> float:
    """Restore a Python float from an IEEE-754 bit pattern.

    Args:
        bits (int): Unsigned 64-bit bit pattern.

    Returns:
        float: Exact reconstructed binary64 value.
    """
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _payload_to_proto(value: Any) -> pb.Payload:
    """Encode one closed semantic payload into protobuf.

    Args:
        value (Any): Internal payload record produced by the IR encoder.

    Returns:
        pb.Payload: Typed protobuf union preserving numeric and container
            identity.

    Raises:
        TypeError: If the payload has no protobuf representation.
        ValueError: If a tagged payload record is malformed.
    """
    message = pb.Payload()
    if value is None:
        message.null_value.SetInParent()
    elif isinstance(value, bool):
        message.bool_value = value
    elif isinstance(value, int):
        message.integer_value.CopyFrom(_integer_to_proto(value))
    elif isinstance(value, float):
        message.float_value.bits = _float_bits(value)
    elif isinstance(value, str):
        message.string_value = value
    elif isinstance(value, (bytes, bytearray)):
        message.bytes_value = bytes(value)
    elif isinstance(value, list):
        message.list_value.SetInParent()
        message.list_value.items.extend(_payload_to_proto(item) for item in value)
    elif isinstance(value, dict):
        _tagged_payload_to_proto(value, message)
    else:
        raise TypeError(f"protobuf payload cannot encode {type(value).__name__!r}")
    return message


def _payload_from_proto(message: pb.Payload) -> Any:
    """Decode one protobuf payload union into its semantic record.

    Args:
        message (pb.Payload): Typed protobuf payload.

    Returns:
        Any: Internal payload record accepted by the IR decoder.

    Raises:
        ValueError: If the payload union is unset or malformed.
    """
    kind = message.WhichOneof("value")
    if kind == "null_value":
        return None
    if kind == "bool_value":
        return message.bool_value
    if kind == "integer_value":
        return _integer_from_proto(message.integer_value)
    if kind == "float_value":
        return _float_from_bits(message.float_value.bits)
    if kind == "string_value":
        return message.string_value
    if kind == "bytes_value":
        return bytes(message.bytes_value)
    if kind == "list_value":
        return [_payload_from_proto(item) for item in message.list_value.items]
    if kind in {"tuple_value", "set_value", "frozenset_value"}:
        tag = {
            "tuple_value": "$tuple",
            "set_value": "$set",
            "frozenset_value": "$frozenset",
        }[kind]
        values = getattr(message, kind).items
        return {tag: [_payload_from_proto(item) for item in values]}
    if kind == "map_value":
        return {
            "$map": [
                [_payload_from_proto(entry.key), _payload_from_proto(entry.value)]
                for entry in message.map_value.entries
            ]
        }
    if kind == "complex_value":
        return {
            "$complex_number": [
                _float_from_bits(message.complex_value.real_bits),
                _float_from_bits(message.complex_value.imag_bits),
            ]
        }
    if kind in {"numpy_array", "numpy_scalar"}:
        numpy_value = getattr(message, kind)
        result: dict[str, Any] = {
            "$np_array" if kind == "numpy_array" else "$np_scalar": True,
            "dtype": numpy_value.dtype,
            "data": bytes(numpy_value.data),
        }
        if kind == "numpy_array":
            result["shape"] = list(numpy_value.shape)
        return result
    if kind == "hamiltonian":
        return _hamiltonian_from_proto(message.hamiltonian)
    raise ValueError("protobuf Payload is missing its value union")


def _tagged_payload_to_proto(value: dict[str, Any], message: pb.Payload) -> None:
    """Encode one tagged internal payload record into a protobuf union.

    Args:
        value (dict[str, Any]): Tagged payload record.
        message (pb.Payload): Destination message mutated in place.

    Raises:
        TypeError: If a nested payload is unsupported.
        ValueError: If the wrapper tag or shape is invalid.
    """
    if "$tuple" in value:
        message.tuple_value.SetInParent()
        message.tuple_value.items.extend(
            _payload_to_proto(item) for item in _require_list(value["$tuple"], "$tuple")
        )
        return
    if "$set" in value:
        message.set_value.SetInParent()
        message.set_value.items.extend(
            _canonical_unordered_payloads(_require_list(value["$set"], "$set"))
        )
        return
    if "$frozenset" in value:
        message.frozenset_value.SetInParent()
        message.frozenset_value.items.extend(
            _canonical_unordered_payloads(
                _require_list(value["$frozenset"], "$frozenset")
            )
        )
        return
    if "$map" in value:
        message.map_value.SetInParent()
        entries = _require_list(value["$map"], "$map")
        for raw_entry in entries:
            if not isinstance(raw_entry, list) or len(raw_entry) != 2:
                raise ValueError("$map entries must be two-element lists")
            entry = message.map_value.entries.add()
            entry.key.CopyFrom(_payload_to_proto(raw_entry[0]))
            entry.value.CopyFrom(_payload_to_proto(raw_entry[1]))
        return
    if "$complex_number" in value:
        parts = _require_list(value["$complex_number"], "$complex_number")
        if len(parts) != 2 or not all(isinstance(item, float) for item in parts):
            raise ValueError("$complex_number must contain two floats")
        message.complex_value.real_bits = _float_bits(parts[0])
        message.complex_value.imag_bits = _float_bits(parts[1])
        return
    if value.get("$np_array") is True or value.get("$np_scalar") is True:
        field = "numpy_array" if value.get("$np_array") is True else "numpy_scalar"
        target = getattr(message, field)
        dtype = value.get("dtype")
        data = value.get("data")
        if not isinstance(dtype, str) or not isinstance(data, (bytes, bytearray)):
            raise ValueError("NumPy payload requires string dtype and bytes data")
        target.dtype = dtype
        target.data = bytes(data)
        if field == "numpy_array":
            shape = value.get("shape")
            if not isinstance(shape, list) or not all(
                isinstance(dim, int) and not isinstance(dim, bool) for dim in shape
            ):
                raise ValueError("NumPy array shape must be a list of ints")
            target.shape.extend(shape)
        return
    if value.get("$hamiltonian") is True:
        message.hamiltonian.CopyFrom(_hamiltonian_to_proto(value))
        return
    raise ValueError(f"unknown semantic payload wrapper {sorted(value)}")


def _canonical_unordered_payloads(values: list[Any]) -> list[pb.Payload]:
    """Encode and canonically order an unordered container's elements.

    Args:
        values (list[Any]): Intermediate payloads originating from a set or
            frozenset.

    Returns:
        list[pb.Payload]: Encoded elements sorted by deterministic protobuf
            bytes.
    """
    messages = [_payload_to_proto(value) for value in values]
    return sorted(
        messages,
        key=lambda message: message.SerializeToString(deterministic=True),
    )


def _require_list(value: Any, label: str) -> list[Any]:
    """Require a list-valued tagged payload field.

    Args:
        value (Any): Candidate value.
        label (str): Field label for diagnostics.

    Returns:
        list[Any]: Narrowed list.

    Raises:
        ValueError: If the value is not a list.
    """
    if not isinstance(value, list):
        raise ValueError(f"{label} payload must be a list")
    return value


def _number_to_proto(value: Any) -> pb.Number:
    """Encode a Hamiltonian coefficient into its exact numeric union.

    Args:
        value (Any): Integer, float, or tagged complex coefficient.

    Returns:
        pb.Number: Typed protobuf number.

    Raises:
        TypeError: If the coefficient has an unsupported type.
        ValueError: If a complex wrapper is malformed.
    """
    message = pb.Number()
    if isinstance(value, bool):
        raise TypeError("Hamiltonian coefficient cannot be bool")
    if isinstance(value, int):
        message.integer.CopyFrom(_integer_to_proto(value))
    elif isinstance(value, float):
        message.floating.bits = _float_bits(value)
    elif isinstance(value, dict) and value.get("$complex") is True:
        real = value.get("real")
        imag = value.get("imag")
        if not isinstance(real, float) or not isinstance(imag, float):
            raise ValueError("Hamiltonian complex coefficient requires float parts")
        message.complex.real_bits = _float_bits(real)
        message.complex.imag_bits = _float_bits(imag)
    else:
        raise TypeError(f"unsupported Hamiltonian coefficient {type(value).__name__!r}")
    return message


def _number_from_proto(message: pb.Number) -> Any:
    """Decode a protobuf Hamiltonian coefficient.

    Args:
        message (pb.Number): Typed numeric union.

    Returns:
        Any: Internal integer, float, or tagged complex record.

    Raises:
        ValueError: If the numeric union is unset or malformed.
    """
    kind = message.WhichOneof("value")
    if kind == "integer":
        return _integer_from_proto(message.integer)
    if kind == "floating":
        return _float_from_bits(message.floating.bits)
    if kind == "complex":
        return {
            "$complex": True,
            "real": _float_from_bits(message.complex.real_bits),
            "imag": _float_from_bits(message.complex.imag_bits),
        }
    raise ValueError("protobuf Number is missing its numeric union")


def _hamiltonian_to_proto(value: dict[str, Any]) -> pb.Hamiltonian:
    """Encode an internal Hamiltonian wrapper into protobuf.

    Args:
        value (dict[str, Any]): Validated Hamiltonian wrapper record.

    Returns:
        pb.Hamiltonian: Typed Hamiltonian message preserving term order.

    Raises:
        ValueError: If terms, operators, or declared width are malformed.
        TypeError: If a coefficient is unsupported.
    """
    message = pb.Hamiltonian()
    message.constant.CopyFrom(_number_to_proto(value.get("constant")))
    num_qubits = value.get("num_qubits")
    if num_qubits is not None:
        if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
            raise ValueError("Hamiltonian num_qubits must be an int or None")
        message.num_qubits = num_qubits
    for raw_term in _require_list(value.get("terms"), "Hamiltonian terms"):
        if not isinstance(raw_term, list) or len(raw_term) != 2:
            raise ValueError("Hamiltonian terms must be [operators, coefficient]")
        raw_operators, coefficient = raw_term
        term = message.terms.add()
        for raw_operator in _require_list(raw_operators, "Hamiltonian operators"):
            if not isinstance(raw_operator, list) or len(raw_operator) != 2:
                raise ValueError("Pauli operators must be [name, index]")
            name, index = raw_operator
            if (
                not isinstance(name, str)
                or isinstance(index, bool)
                or not isinstance(index, int)
            ):
                raise ValueError("Pauli operator name/index are malformed")
            operator = term.operators.add()
            operator.pauli = name
            operator.index = index
        term.coefficient.CopyFrom(_number_to_proto(coefficient))
    return message


def _hamiltonian_from_proto(message: pb.Hamiltonian) -> dict[str, Any]:
    """Decode a typed protobuf Hamiltonian into its internal wrapper.

    Args:
        message (pb.Hamiltonian): Hamiltonian protobuf message.

    Returns:
        dict[str, Any]: Wrapper accepted by the semantic payload decoder.

    Raises:
        ValueError: If a coefficient union is unset.
    """
    return {
        "$hamiltonian": True,
        "terms": [
            [
                [[operator.pauli, operator.index] for operator in term.operators],
                _number_from_proto(term.coefficient),
            ]
            for term in message.terms
        ],
        "constant": _number_from_proto(message.constant),
        "num_qubits": message.num_qubits if message.HasField("num_qubits") else None,
    }


_VALUE_TYPE_TO_PROTO: dict[str, pb.ValueTypeKind] = {
    "UIntType": pb.UINT_TYPE,
    "FloatType": pb.FLOAT_TYPE,
    "BitType": pb.BIT_TYPE,
    "QubitType": pb.QUBIT_TYPE,
    "BlockType": pb.BLOCK_TYPE,
    "ObservableType": pb.OBSERVABLE_TYPE,
    "TupleType": pb.TUPLE_TYPE,
    "DictType": pb.DICT_TYPE,
    "QFixedType": pb.QFIXED_TYPE,
    "QUIntType": pb.QUINT_TYPE,
}
_VALUE_TYPE_FROM_PROTO: dict[pb.ValueTypeKind, str] = {
    value: key for key, value in _VALUE_TYPE_TO_PROTO.items()
}


def _value_type_to_proto(value: dict[str, Any]) -> pb.ValueType:
    """Encode one semantic ValueType record.

    Args:
        value (dict[str, Any]): Tagged ValueType record.

    Returns:
        pb.ValueType: Typed protobuf representation.

    Raises:
        ValueError: If the type tag or parametric fields are malformed.
    """
    tag = value.get("$type")
    kind = _VALUE_TYPE_TO_PROTO.get(tag) if isinstance(tag, str) else None
    if kind is None:
        raise ValueError(f"unknown ValueType tag {tag!r}")
    message = pb.ValueType(kind=kind)
    if tag == "TupleType":
        message.element_types.extend(
            _value_type_to_proto(item)
            for item in _require_list(value.get("element_types"), "element_types")
        )
    elif tag == "DictType":
        if value.get("key_type") is not None:
            message.key_type.CopyFrom(_value_type_to_proto(value["key_type"]))
        if value.get("value_type") is not None:
            message.value_type.CopyFrom(_value_type_to_proto(value["value_type"]))
    elif tag == "QFixedType":
        message.integer_bits.CopyFrom(_width_to_proto(value.get("integer_bits")))
        message.fractional_bits.CopyFrom(_width_to_proto(value.get("fractional_bits")))
    elif tag == "QUIntType":
        message.width.CopyFrom(_width_to_proto(value.get("width")))
    return message


def _value_type_from_proto(message: pb.ValueType) -> dict[str, Any]:
    """Decode one protobuf ValueType record.

    Args:
        message (pb.ValueType): Typed ValueType message.

    Returns:
        dict[str, Any]: Tagged semantic ValueType record.

    Raises:
        ValueError: If the kind is unknown or required parameters are absent.
    """
    tag = _VALUE_TYPE_FROM_PROTO.get(message.kind)
    if tag is None:
        raise ValueError(f"unknown protobuf ValueType kind {message.kind}")
    result: dict[str, Any] = {"$type": tag}
    if tag == "TupleType":
        result["element_types"] = [
            _value_type_from_proto(item) for item in message.element_types
        ]
    elif tag == "DictType":
        result["key_type"] = (
            _value_type_from_proto(message.key_type)
            if message.HasField("key_type")
            else None
        )
        result["value_type"] = (
            _value_type_from_proto(message.value_type)
            if message.HasField("value_type")
            else None
        )
    elif tag == "QFixedType":
        if not message.HasField("integer_bits") or not message.HasField(
            "fractional_bits"
        ):
            raise ValueError("QFixedType requires integer and fractional widths")
        result["integer_bits"] = _width_from_proto(message.integer_bits)
        result["fractional_bits"] = _width_from_proto(message.fractional_bits)
    elif tag == "QUIntType":
        if not message.HasField("width"):
            raise ValueError("QUIntType requires width")
        result["width"] = _width_from_proto(message.width)
    return result


def _width_to_proto(value: Any) -> pb.RegisterWidth:
    """Encode a concrete or symbolic register width.

    Args:
        value (Any): Plain integer or ``$value_ref`` wrapper.

    Returns:
        pb.RegisterWidth: Typed width union.

    Raises:
        ValueError: If the width shape is unsupported.
    """
    message = pb.RegisterWidth()
    if isinstance(value, int) and not isinstance(value, bool):
        message.concrete = value
    elif isinstance(value, dict) and isinstance(value.get("$value_ref"), str):
        message.value_ref = value["$value_ref"]
    else:
        raise ValueError(f"invalid register width {value!r}")
    return message


def _width_from_proto(message: pb.RegisterWidth) -> Any:
    """Decode a concrete or symbolic protobuf register width.

    Args:
        message (pb.RegisterWidth): Typed width union.

    Returns:
        Any: Plain integer or semantic ``$value_ref`` wrapper.

    Raises:
        ValueError: If the width union is unset.
    """
    kind = message.WhichOneof("value")
    if kind == "concrete":
        return message.concrete
    if kind == "value_ref":
        return {"$value_ref": message.value_ref}
    raise ValueError("RegisterWidth is missing its value union")


def _metadata_to_proto(value: dict[str, Any]) -> pb.ValueMetadata:
    """Encode the complete semantic ValueMetadata bundle.

    Args:
        value (dict[str, Any]): Internal metadata record.

    Returns:
        pb.ValueMetadata: Typed metadata message.

    Raises:
        ValueError: If a metadata sub-record is malformed.
        TypeError: If a nested payload is unsupported.
    """
    message = pb.ValueMetadata()
    scalar = value.get("scalar")
    if scalar is not None:
        if not isinstance(scalar, dict):
            raise ValueError("scalar metadata must be a record")
        message.scalar.const_value.CopyFrom(
            _payload_to_proto(scalar.get("const_value"))
        )
        if scalar.get("parameter_name") is not None:
            message.scalar.parameter_name = scalar["parameter_name"]
    cast = value.get("cast")
    if cast is not None:
        if not isinstance(cast, dict):
            raise ValueError("cast metadata must be a record")
        message.cast.source_uuid = cast["source_uuid"]
        message.cast.qubit_uuids.extend(cast["qubit_uuids"])
        if cast.get("source_logical_id") is not None:
            message.cast.source_logical_id = cast["source_logical_id"]
        message.cast.qubit_logical_ids.extend(cast["qubit_logical_ids"])
    qfixed = value.get("qfixed")
    if qfixed is not None:
        if not isinstance(qfixed, dict):
            raise ValueError("qfixed metadata must be a record")
        message.qfixed.qubit_uuids.extend(qfixed["qubit_uuids"])
        message.qfixed.num_bits = qfixed["num_bits"]
        message.qfixed.int_bits = qfixed["int_bits"]
    array_runtime = value.get("array_runtime")
    if array_runtime is not None:
        if not isinstance(array_runtime, dict):
            raise ValueError("array runtime metadata must be a record")
        target = message.array_runtime
        target.const_array.CopyFrom(_payload_to_proto(array_runtime.get("const_array")))
        target.element_uuids.extend(array_runtime["element_uuids"])
        target.element_logical_ids.extend(array_runtime["element_logical_ids"])
        for item in array_runtime["element_parent_uuids"]:
            optional = target.element_parent_uuids.add()
            if item is not None:
                optional.value = item
        for item in array_runtime["element_parent_indices"]:
            optional = target.element_parent_indices.add()
            if item is not None:
                optional.value = item
    dict_runtime = value.get("dict_runtime")
    if dict_runtime is not None:
        if not isinstance(dict_runtime, dict):
            raise ValueError("dict runtime metadata must be a record")
        for raw_entry in dict_runtime["bound_data"]:
            if not isinstance(raw_entry, list) or len(raw_entry) != 2:
                raise ValueError("bound_data entries must be key/value pairs")
            entry = message.dict_runtime.bound_data.add()
            entry.key.CopyFrom(_payload_to_proto(raw_entry[0]))
            entry.value.CopyFrom(_payload_to_proto(raw_entry[1]))
    return message


def _metadata_from_proto(message: pb.ValueMetadata) -> dict[str, Any]:
    """Decode a protobuf ValueMetadata bundle.

    Args:
        message (pb.ValueMetadata): Typed metadata message.

    Returns:
        dict[str, Any]: Internal metadata record.

    Raises:
        ValueError: If a nested payload union is malformed.
    """
    scalar = None
    if message.HasField("scalar"):
        scalar = {
            "const_value": _payload_from_proto(message.scalar.const_value),
            "parameter_name": (
                message.scalar.parameter_name
                if message.scalar.HasField("parameter_name")
                else None
            ),
        }
    cast = None
    if message.HasField("cast"):
        cast = {
            "source_uuid": message.cast.source_uuid,
            "qubit_uuids": list(message.cast.qubit_uuids),
            "source_logical_id": (
                message.cast.source_logical_id
                if message.cast.HasField("source_logical_id")
                else None
            ),
            "qubit_logical_ids": list(message.cast.qubit_logical_ids),
        }
    qfixed = None
    if message.HasField("qfixed"):
        qfixed = {
            "qubit_uuids": list(message.qfixed.qubit_uuids),
            "num_bits": message.qfixed.num_bits,
            "int_bits": message.qfixed.int_bits,
        }
    array_runtime = None
    if message.HasField("array_runtime"):
        target = message.array_runtime
        array_runtime = {
            "const_array": _payload_from_proto(target.const_array),
            "element_uuids": list(target.element_uuids),
            "element_logical_ids": list(target.element_logical_ids),
            "element_parent_uuids": [
                item.value if item.HasField("value") else None
                for item in target.element_parent_uuids
            ],
            "element_parent_indices": [
                item.value if item.HasField("value") else None
                for item in target.element_parent_indices
            ],
        }
    dict_runtime = None
    if message.HasField("dict_runtime"):
        dict_runtime = {
            "bound_data": [
                [_payload_from_proto(item.key), _payload_from_proto(item.value)]
                for item in message.dict_runtime.bound_data
            ]
        }
    return {
        "scalar": scalar,
        "cast": cast,
        "qfixed": qfixed,
        "array_runtime": array_runtime,
        "dict_runtime": dict_runtime,
    }


_VALUE_KIND_TO_PROTO: dict[str, pb.ValueKind] = {
    "Value": pb.VALUE,
    "ArrayValue": pb.ARRAY_VALUE,
    "TupleValue": pb.TUPLE_VALUE,
    "DictValue": pb.DICT_VALUE,
}
_VALUE_KIND_FROM_PROTO: dict[pb.ValueKind, str] = {
    value: key for key, value in _VALUE_KIND_TO_PROTO.items()
}


def _value_to_proto(value: Any) -> pb.ValueNode:
    """Encode one module-wide semantic Value node.

    Args:
        value (Any): Tagged Value record.

    Returns:
        pb.ValueNode: Typed protobuf Value node.

    Raises:
        ValueError: If the node kind or fields are malformed.
    """
    if not isinstance(value, dict):
        raise ValueError("value_table entries must be records")
    tag = value.get("$type")
    kind = _VALUE_KIND_TO_PROTO.get(tag) if isinstance(tag, str) else None
    if kind is None:
        raise ValueError(f"unknown Value node tag {tag!r}")
    message = pb.ValueNode(
        value_kind=kind,
        uuid=value["uuid"],
        logical_id=value["logical_id"],
        name=value["name"],
    )
    message.metadata.CopyFrom(_metadata_to_proto(value["metadata"]))
    if tag in {"Value", "ArrayValue"}:
        message.version = value["version"]
        message.value_type.CopyFrom(_value_type_to_proto(value["value_type"]))
    if tag == "Value":
        if value.get("parent_array_ref") is not None:
            message.parent_array_ref = value["parent_array_ref"]
        message.element_index_refs.extend(value["element_index_refs"])
    elif tag == "ArrayValue":
        message.shape_refs.extend(value["shape_refs"])
        for field in ("slice_of_ref", "slice_start_ref", "slice_step_ref"):
            if value.get(field) is not None:
                setattr(message, field, value[field])
    elif tag == "TupleValue":
        message.element_refs.extend(value["element_refs"])
    else:
        for key_ref, value_ref in value["entry_refs"]:
            entry = message.entry_refs.add()
            entry.key_ref = key_ref
            entry.value_ref = value_ref
    return message


def _value_from_proto(message: pb.ValueNode) -> dict[str, Any]:
    """Decode one protobuf Value node.

    Args:
        message (pb.ValueNode): Typed Value node.

    Returns:
        dict[str, Any]: Tagged semantic Value record.

    Raises:
        ValueError: If the node kind is unknown or required type data is absent.
    """
    tag = _VALUE_KIND_FROM_PROTO.get(message.value_kind)
    if tag is None:
        raise ValueError(f"unknown protobuf Value kind {message.value_kind}")
    result: dict[str, Any] = {
        "$type": tag,
        "uuid": message.uuid,
        "logical_id": message.logical_id,
        "name": message.name,
        "metadata": _metadata_from_proto(message.metadata),
    }
    if tag in {"Value", "ArrayValue"}:
        result["version"] = message.version
        result["value_type"] = _value_type_from_proto(message.value_type)
    if tag == "Value":
        result["parent_array_ref"] = (
            message.parent_array_ref if message.HasField("parent_array_ref") else None
        )
        result["element_index_refs"] = list(message.element_index_refs)
    elif tag == "ArrayValue":
        result["shape_refs"] = list(message.shape_refs)
        for field in ("slice_of_ref", "slice_start_ref", "slice_step_ref"):
            result[field] = getattr(message, field) if message.HasField(field) else None
    elif tag == "TupleValue":
        result["element_refs"] = list(message.element_refs)
    else:
        result["entry_refs"] = [
            [entry.key_ref, entry.value_ref] for entry in message.entry_refs
        ]
    return result


_OPERATION_TO_PROTO: dict[str, pb.OperationType] = {
    "GateOperation": pb.GATE_OPERATION,
    "MeasureOperation": pb.MEASURE_OPERATION,
    "ProjectOperation": pb.PROJECT_OPERATION,
    "ResetOperation": pb.RESET_OPERATION,
    "MeasureVectorOperation": pb.MEASURE_VECTOR_OPERATION,
    "MeasureQFixedOperation": pb.MEASURE_QFIXED_OPERATION,
    "DecodeQFixedOperation": pb.DECODE_QFIXED_OPERATION,
    "StoreArrayElementOperation": pb.STORE_ARRAY_ELEMENT_OPERATION,
    "DictGetItemOperation": pb.DICT_GET_ITEM_OPERATION,
    "CastOperation": pb.CAST_OPERATION,
    "QInitOperation": pb.QINIT_OPERATION,
    "CInitOperation": pb.CINIT_OPERATION,
    "SliceArrayOperation": pb.SLICE_ARRAY_OPERATION,
    "ReleaseSliceViewOperation": pb.RELEASE_SLICE_VIEW_OPERATION,
    "ReturnOperation": pb.RETURN_OPERATION,
    "ExpvalOp": pb.EXPVAL_OPERATION,
    "PauliEvolveOp": pb.PAULI_EVOLVE_OPERATION,
    "BinOp": pb.BIN_OPERATION,
    "CompOp": pb.COMP_OPERATION,
    "CondOp": pb.COND_OPERATION,
    "NotOp": pb.NOT_OPERATION,
    "RuntimeClassicalExpr": pb.RUNTIME_CLASSICAL_OPERATION,
    "ForOperation": pb.FOR_OPERATION,
    "ForItemsOperation": pb.FOR_ITEMS_OPERATION,
    "WhileOperation": pb.WHILE_OPERATION,
    "IfOperation": pb.IF_OPERATION,
    "ConcreteControlledU": pb.CONCRETE_CONTROLLED_OPERATION,
    "SymbolicControlledU": pb.SYMBOLIC_CONTROLLED_OPERATION,
    "InvokeOperation": pb.INVOKE_OPERATION,
    "InverseBlockOperation": pb.INVERSE_BLOCK_OPERATION,
    "GlobalPhaseOperation": pb.GLOBAL_PHASE_OPERATION,
    "SelectOperation": pb.SELECT_OPERATION,
}
_OPERATION_FROM_PROTO: dict[pb.OperationType, str] = {
    value: key for key, value in _OPERATION_TO_PROTO.items()
}

_OPERATION_ALLOWED_FIELDS: dict[pb.OperationType, frozenset[str]] = {
    pb.GATE_OPERATION: frozenset({"gate_type"}),
    pb.MEASURE_OPERATION: frozenset(),
    pb.PROJECT_OPERATION: frozenset({"axis"}),
    pb.RESET_OPERATION: frozenset(),
    pb.MEASURE_VECTOR_OPERATION: frozenset(),
    pb.MEASURE_QFIXED_OPERATION: frozenset({"num_bits", "int_bits"}),
    pb.DECODE_QFIXED_OPERATION: frozenset({"num_bits", "int_bits"}),
    pb.STORE_ARRAY_ELEMENT_OPERATION: frozenset(),
    pb.DICT_GET_ITEM_OPERATION: frozenset({"key_arity"}),
    pb.CAST_OPERATION: frozenset({"source_type", "target_type", "qubit_mapping"}),
    pb.QINIT_OPERATION: frozenset(),
    pb.CINIT_OPERATION: frozenset(),
    pb.SLICE_ARRAY_OPERATION: frozenset(),
    pb.RELEASE_SLICE_VIEW_OPERATION: frozenset(),
    pb.RETURN_OPERATION: frozenset(),
    pb.EXPVAL_OPERATION: frozenset(),
    pb.PAULI_EVOLVE_OPERATION: frozenset(),
    pb.BIN_OPERATION: frozenset({"expression_kind"}),
    pb.COMP_OPERATION: frozenset({"expression_kind"}),
    pb.COND_OPERATION: frozenset({"expression_kind"}),
    pb.NOT_OPERATION: frozenset(),
    pb.RUNTIME_CLASSICAL_OPERATION: frozenset({"expression_kind"}),
    pb.FOR_OPERATION: frozenset(
        {
            "loop_var",
            "loop_var_value_ref",
            "loop_carried_rebinds",
            "region_args",
            "body",
        }
    ),
    pb.FOR_ITEMS_OPERATION: frozenset(
        {
            "key_vars",
            "value_var",
            "key_is_vector",
            "key_var_value_refs",
            "has_key_var_value_refs",
            "value_var_value_ref",
            "loop_carried_rebinds",
            "region_args",
            "body",
        }
    ),
    pb.WHILE_OPERATION: frozenset(
        {"max_iterations", "loop_carried_rebinds", "region_args", "body"}
    ),
    pb.IF_OPERATION: frozenset(
        {
            "true_body",
            "false_body",
            "true_yield_refs",
            "false_yield_refs",
            "branch_rebinds",
        }
    ),
    pb.CONCRETE_CONTROLLED_OPERATION: frozenset(
        {
            "num_controls",
            "power",
            "unitary_block",
            "callable_ref",
            "callable_attrs",
            "control_value",
        }
    ),
    pb.SYMBOLIC_CONTROLLED_OPERATION: frozenset(
        {
            "num_controls_ref",
            "power",
            "control_index_refs",
            "has_control_index_refs",
            "num_control_args",
            "unitary_block",
            "callable_ref",
            "callable_attrs",
        }
    ),
    pb.INVOKE_OPERATION: frozenset({"target", "transform", "attrs", "definition_ref"}),
    pb.INVERSE_BLOCK_OPERATION: frozenset(
        {
            "num_control_qubits",
            "num_target_qubits",
            "custom_name",
            "source_block",
            "implementation_block",
            "callable_ref",
            "callable_attrs",
            "control_value",
        }
    ),
    pb.GLOBAL_PHASE_OPERATION: frozenset(),
    pb.SELECT_OPERATION: frozenset({"num_index_qubits", "case_blocks"}),
}

_OPERATION_REQUIRED_FIELDS: dict[pb.OperationType, frozenset[str]] = {
    pb.GATE_OPERATION: frozenset({"gate_type"}),
    pb.PROJECT_OPERATION: frozenset({"axis"}),
    pb.MEASURE_QFIXED_OPERATION: frozenset({"num_bits", "int_bits"}),
    pb.DECODE_QFIXED_OPERATION: frozenset({"num_bits", "int_bits"}),
    pb.DICT_GET_ITEM_OPERATION: frozenset({"key_arity"}),
    pb.BIN_OPERATION: frozenset({"expression_kind"}),
    pb.COMP_OPERATION: frozenset({"expression_kind"}),
    pb.COND_OPERATION: frozenset({"expression_kind"}),
    pb.RUNTIME_CLASSICAL_OPERATION: frozenset({"expression_kind"}),
    pb.FOR_OPERATION: frozenset({"loop_var"}),
    pb.FOR_ITEMS_OPERATION: frozenset(
        {"value_var", "key_is_vector", "has_key_var_value_refs"}
    ),
    pb.CONCRETE_CONTROLLED_OPERATION: frozenset({"num_controls", "power"}),
    pb.SYMBOLIC_CONTROLLED_OPERATION: frozenset(
        {"num_controls_ref", "power", "has_control_index_refs", "num_control_args"}
    ),
    pb.INVOKE_OPERATION: frozenset({"target", "transform", "attrs", "definition_ref"}),
    pb.INVERSE_BLOCK_OPERATION: frozenset({"num_control_qubits", "num_target_qubits"}),
    pb.GLOBAL_PHASE_OPERATION: frozenset(),
    pb.SELECT_OPERATION: frozenset({"num_index_qubits"}),
}


def _validate_operation_fields(message: pb.Operation) -> None:
    """Reject fields that contradict an operation's selected type.

    Args:
        message (pb.Operation): Operation message to validate.

    Raises:
        ValueError: If a required field is absent, an unrelated field is
            present, or an optional-list presence marker is inconsistent.
    """
    base_fields = {"operation_type", "operand_refs", "result_refs"}
    present = {field.name for field, _value in message.ListFields()} - base_fields
    allowed = _OPERATION_ALLOWED_FIELDS[message.operation_type]
    unexpected = sorted(present - allowed)
    if unexpected:
        raise ValueError(
            f"protobuf Operation {message.operation_type} has unrelated fields: "
            f"{unexpected}"
        )
    missing = sorted(
        _OPERATION_REQUIRED_FIELDS.get(message.operation_type, frozenset()) - present
    )
    if missing:
        raise ValueError(
            f"protobuf Operation {message.operation_type} is missing required fields: "
            f"{missing}"
        )
    if message.operation_type == pb.FOR_ITEMS_OPERATION:
        if message.key_var_value_refs and not message.has_key_var_value_refs:
            raise ValueError("key_var_value_refs contradict their presence marker")
    if message.operation_type == pb.SYMBOLIC_CONTROLLED_OPERATION:
        if message.control_index_refs and not message.has_control_index_refs:
            raise ValueError("control_index_refs contradict their presence marker")


def _operation_to_proto(value: dict[str, Any]) -> pb.Operation:
    """Encode one typed semantic operation.

    Args:
        value (dict[str, Any]): Tagged operation record.

    Returns:
        pb.Operation: Typed protobuf operation.

    Raises:
        ValueError: If the operation kind or one of its structured fields is
            malformed.
        TypeError: If an attrs payload is unsupported.
    """
    tag = value.get("$type")
    operation_type = _OPERATION_TO_PROTO.get(tag) if isinstance(tag, str) else None
    if operation_type is None:
        raise ValueError(f"unknown operation tag {tag!r}")
    message = pb.Operation(operation_type=operation_type)
    message.operand_refs.extend(value.get("operand_refs", []))
    message.result_refs.extend(value.get("result_refs", []))

    for field in ("gate_type", "axis", "loop_var", "value_var", "transform"):
        if field in value and value[field] is not None:
            setattr(message, field, value[field])
    for field in (
        "num_bits",
        "int_bits",
        "key_arity",
        "max_iterations",
        "num_controls",
        "num_control_args",
        "num_control_qubits",
        "num_target_qubits",
        "num_index_qubits",
    ):
        if field in value and value[field] is not None:
            setattr(message, field, value[field])
    if "kind" in value:
        message.expression_kind = value["kind"]
    if "key_is_vector" in value:
        message.key_is_vector = value["key_is_vector"]
    for field in (
        "loop_var_value_ref",
        "value_var_value_ref",
        "num_controls_ref",
        "definition_ref",
        "custom_name",
    ):
        if value.get(field) is not None:
            setattr(message, field, value[field])
    for field in (
        "qubit_mapping",
        "key_vars",
        "true_yield_refs",
        "false_yield_refs",
    ):
        if field in value:
            getattr(message, field).extend(value[field])

    if value.get("source_type") is not None:
        message.source_type.CopyFrom(_value_type_to_proto(value["source_type"]))
    if value.get("target_type") is not None:
        message.target_type.CopyFrom(_value_type_to_proto(value["target_type"]))

    if "key_var_value_refs" in value:
        message.has_key_var_value_refs = value["key_var_value_refs"] is not None
        if value["key_var_value_refs"] is not None:
            message.key_var_value_refs.extend(value["key_var_value_refs"])
    if "control_index_refs" in value:
        message.has_control_index_refs = value["control_index_refs"] is not None
        if value["control_index_refs"] is not None:
            message.control_index_refs.extend(value["control_index_refs"])

    for raw in value.get("loop_carried_rebinds", []):
        target = message.loop_carried_rebinds.add()
        target.var_name = raw["var_name"]
        target.before_ref = raw["before_ref"]
        target.after_ref = raw["after_ref"]
        target.before_synthesized = raw["before_synthesized"]
    for raw in value.get("region_args", []):
        target = message.region_args.add()
        target.var_name = raw["var_name"]
        target.init_ref = raw["init_ref"]
        target.block_arg_ref = raw["block_arg_ref"]
        target.yielded_ref = raw["yielded_ref"]
        target.result_ref = raw["result_ref"]
    for raw in value.get("branch_rebinds", []):
        target = message.branch_rebinds.add()
        target.var_name = raw["var_name"]
        target.before_ref = raw["before_ref"]
        target.rebound_in_true = raw["rebound_in_true"]
        target.rebound_in_false = raw["rebound_in_false"]
    for field in ("body", "true_body", "false_body"):
        if field in value:
            getattr(message, field).extend(
                _operation_to_proto(item) for item in value[field]
            )

    if "power" in value:
        message.power.CopyFrom(_integer_or_reference_to_proto(value["power"]))
    if "control_value" in value:
        message.control_value.CopyFrom(_integer_to_proto(value["control_value"]))
    for field in ("unitary_block", "source_block", "implementation_block"):
        if value.get(field) is not None:
            getattr(message, field).CopyFrom(_block_to_proto(value[field]))
    if "case_blocks" in value:
        message.case_blocks.extend(
            _block_to_proto(item) for item in value["case_blocks"]
        )
    for field in ("callable_ref", "target"):
        if value.get(field) is not None:
            getattr(message, field).CopyFrom(_callable_ref_to_proto(value[field]))
    for field in ("callable_attrs", "attrs"):
        if field in value:
            getattr(message, field).CopyFrom(_payload_to_proto(value[field]))
    return message


def _operation_from_proto(message: pb.Operation) -> dict[str, Any]:
    """Decode one typed protobuf operation.

    Args:
        message (pb.Operation): Typed operation message.

    Returns:
        dict[str, Any]: Tagged operation record accepted by the semantic
            decoder.

    Raises:
        ValueError: If the operation type is unknown or a required union is
            absent.
    """
    tag = _OPERATION_FROM_PROTO.get(message.operation_type)
    if tag is None:
        raise ValueError(f"unknown protobuf Operation type {message.operation_type}")
    _validate_operation_fields(message)
    result: dict[str, Any] = {
        "$type": tag,
        "operand_refs": list(message.operand_refs),
        "result_refs": list(message.result_refs),
    }
    _decode_operation_scalars(message, result)
    _decode_operation_structures(message, result)
    if message.operation_type == pb.IF_OPERATION:
        result.setdefault("true_body", [])
        result.setdefault("false_body", [])
        result.setdefault("true_yield_refs", [])
        result.setdefault("false_yield_refs", [])
    return result


def _decode_operation_scalars(
    message: pb.Operation,
    result: dict[str, Any],
) -> None:
    """Copy present scalar protobuf operation fields into a graph record.

    Args:
        message (pb.Operation): Source protobuf operation.
        result (dict[str, Any]): Destination graph record mutated in place.
    """
    for field in (
        "gate_type",
        "axis",
        "num_bits",
        "int_bits",
        "key_arity",
        "loop_var",
        "loop_var_value_ref",
        "value_var",
        "key_is_vector",
        "value_var_value_ref",
        "max_iterations",
        "num_controls",
        "num_controls_ref",
        "num_control_args",
        "transform",
        "definition_ref",
        "num_control_qubits",
        "num_target_qubits",
        "custom_name",
        "num_index_qubits",
    ):
        if message.HasField(field):
            result[field] = getattr(message, field)
    if message.HasField("expression_kind"):
        result["kind"] = message.expression_kind
    for field in (
        "qubit_mapping",
        "key_vars",
        "true_yield_refs",
        "false_yield_refs",
    ):
        if getattr(message, field):
            result[field] = list(getattr(message, field))
    if message.HasField("source_type"):
        result["source_type"] = _value_type_from_proto(message.source_type)
    elif message.operation_type == pb.CAST_OPERATION:
        result["source_type"] = None
    if message.HasField("target_type"):
        result["target_type"] = _value_type_from_proto(message.target_type)
    elif message.operation_type == pb.CAST_OPERATION:
        result["target_type"] = None


def _decode_operation_structures(
    message: pb.Operation,
    result: dict[str, Any],
) -> None:
    """Copy nested protobuf operation fields into a graph record.

    Args:
        message (pb.Operation): Source protobuf operation.
        result (dict[str, Any]): Destination graph record mutated in place.

    Raises:
        ValueError: If a nested payload or integer/reference union is malformed.
    """
    if message.HasField("has_key_var_value_refs"):
        result["key_var_value_refs"] = (
            list(message.key_var_value_refs) if message.has_key_var_value_refs else None
        )
    if message.HasField("has_control_index_refs"):
        result["control_index_refs"] = (
            list(message.control_index_refs) if message.has_control_index_refs else None
        )
    if message.loop_carried_rebinds:
        result["loop_carried_rebinds"] = [
            {
                "var_name": item.var_name,
                "before_ref": item.before_ref,
                "after_ref": item.after_ref,
                "before_synthesized": item.before_synthesized,
            }
            for item in message.loop_carried_rebinds
        ]
    if message.region_args:
        result["region_args"] = [
            {
                "var_name": item.var_name,
                "init_ref": item.init_ref,
                "block_arg_ref": item.block_arg_ref,
                "yielded_ref": item.yielded_ref,
                "result_ref": item.result_ref,
            }
            for item in message.region_args
        ]
    if message.branch_rebinds:
        result["branch_rebinds"] = [
            {
                "var_name": item.var_name,
                "before_ref": item.before_ref,
                "rebound_in_true": item.rebound_in_true,
                "rebound_in_false": item.rebound_in_false,
            }
            for item in message.branch_rebinds
        ]
    for field in ("body", "true_body", "false_body"):
        if getattr(message, field):
            result[field] = [
                _operation_from_proto(item) for item in getattr(message, field)
            ]
    if message.HasField("power"):
        result["power"] = _integer_or_reference_from_proto(message.power)
    if message.HasField("control_value"):
        result["control_value"] = _integer_from_proto(message.control_value)
    for field in ("unitary_block", "source_block", "implementation_block"):
        if message.HasField(field):
            result[field] = _block_from_proto(getattr(message, field))
        elif message.operation_type in {
            pb.CONCRETE_CONTROLLED_OPERATION,
            pb.SYMBOLIC_CONTROLLED_OPERATION,
            pb.INVERSE_BLOCK_OPERATION,
        }:
            result[field] = None
    if message.operation_type == pb.SELECT_OPERATION:
        result["case_blocks"] = [
            _block_from_proto(item) for item in message.case_blocks
        ]
    for field in ("callable_ref", "target"):
        if message.HasField(field):
            result[field] = _callable_ref_from_proto(getattr(message, field))
    for field in ("callable_attrs", "attrs"):
        if message.HasField(field):
            result[field] = _payload_from_proto(getattr(message, field))


def _integer_or_reference_to_proto(value: Any) -> pb.IntegerOrReference:
    """Encode a controlled-power integer or Value reference.

    Args:
        value (Any): Plain integer or ``$value_ref`` wrapper.

    Returns:
        pb.IntegerOrReference: Typed protobuf union.

    Raises:
        ValueError: If the value shape is invalid.
    """
    message = pb.IntegerOrReference()
    if isinstance(value, int) and not isinstance(value, bool):
        message.integer.CopyFrom(_integer_to_proto(value))
    elif isinstance(value, dict) and isinstance(value.get("$value_ref"), str):
        message.value_ref = value["$value_ref"]
    else:
        raise ValueError(f"invalid integer/reference value {value!r}")
    return message


def _integer_or_reference_from_proto(message: pb.IntegerOrReference) -> Any:
    """Decode a controlled-power protobuf union.

    Args:
        message (pb.IntegerOrReference): Typed protobuf union.

    Returns:
        Any: Plain integer or ``$value_ref`` wrapper.

    Raises:
        ValueError: If the union is unset or the integer is malformed.
    """
    kind = message.WhichOneof("value")
    if kind == "integer":
        return _integer_from_proto(message.integer)
    if kind == "value_ref":
        return {"$value_ref": message.value_ref}
    raise ValueError("IntegerOrReference is missing its value union")


def _block_to_proto(value: dict[str, Any]) -> pb.Block:
    """Encode one semantic Block record.

    Args:
        value (dict[str, Any]): Tagged block record.

    Returns:
        pb.Block: Typed protobuf block.

    Raises:
        ValueError: If nested records are malformed.
        TypeError: If a label payload is unsupported.
    """
    if value.get("$type") != "Block":
        raise ValueError("expected a Block graph record")
    message = pb.Block(kind=value["kind"], name=value["name"])
    message.label_args.extend(_payload_to_proto(item) for item in value["label_args"])
    message.input_value_refs.extend(value["input_value_refs"])
    message.output_value_refs.extend(value["output_value_refs"])
    message.output_names.extend(value["output_names"])
    for name, value_ref in value["parameters"].items():
        item = message.parameters.add()
        item.name = name
        item.value_ref = value_ref
    message.operations.extend(_operation_to_proto(item) for item in value["operations"])
    return message


def _block_from_proto(message: pb.Block) -> dict[str, Any]:
    """Decode one protobuf Block message.

    Args:
        message (pb.Block): Typed protobuf block.

    Returns:
        dict[str, Any]: Tagged block record.

    Raises:
        ValueError: If nested messages contain malformed unions.
    """
    parameters: dict[str, str] = {}
    for item in message.parameters:
        if item.name in parameters:
            raise ValueError(f"duplicate Block parameter {item.name!r}")
        parameters[item.name] = item.value_ref
    return {
        "$type": "Block",
        "kind": message.kind,
        "name": message.name,
        "label_args": [_payload_from_proto(item) for item in message.label_args],
        "input_value_refs": list(message.input_value_refs),
        "output_value_refs": list(message.output_value_refs),
        "output_names": list(message.output_names),
        "parameters": parameters,
        "operations": [_operation_from_proto(item) for item in message.operations],
    }


def _callable_ref_to_proto(value: dict[str, Any]) -> pb.CallableRef:
    """Encode a stable callable reference.

    Args:
        value (dict[str, Any]): Callable namespace, name, and version record.

    Returns:
        pb.CallableRef: Typed callable reference.

    Raises:
        ValueError: If any required field is not a string.
    """
    fields = (value.get("namespace"), value.get("name"), value.get("version"))
    if not all(isinstance(field, str) for field in fields):
        raise ValueError("CallableRef fields must be strings")
    return pb.CallableRef(namespace=fields[0], name=fields[1], version=fields[2])


def _callable_ref_from_proto(message: pb.CallableRef) -> dict[str, str]:
    """Decode a protobuf callable reference.

    Args:
        message (pb.CallableRef): Typed callable reference.

    Returns:
        dict[str, str]: Internal callable reference record.
    """
    return {
        "namespace": message.namespace,
        "name": message.name,
        "version": message.version,
    }


def _body_ref_to_proto(value: dict[str, Any]) -> pb.CallableBodyRef:
    """Encode a deferred callable body reference.

    Args:
        value (dict[str, Any]): Body-reference record.

    Returns:
        pb.CallableBodyRef: Typed body reference.

    Raises:
        ValueError: If the reference is malformed.
        TypeError: If attrs contain an unsupported payload.
    """
    message = pb.CallableBodyRef(kind=value["kind"])
    message.ref.CopyFrom(_callable_ref_to_proto(value["ref"]))
    message.attrs.CopyFrom(_payload_to_proto(value["attrs"]))
    return message


def _body_ref_from_proto(message: pb.CallableBodyRef) -> dict[str, Any]:
    """Decode a protobuf deferred body reference.

    Args:
        message (pb.CallableBodyRef): Typed body reference.

    Returns:
        dict[str, Any]: Internal body-reference record.

    Raises:
        ValueError: If attrs contain a malformed payload union.
    """
    return {
        "ref": _callable_ref_from_proto(message.ref),
        "kind": message.kind,
        "attrs": _payload_from_proto(message.attrs),
    }


def _signature_to_proto(value: dict[str, Any]) -> pb.Signature:
    """Encode a callable signature.

    Args:
        value (dict[str, Any]): Signature record with operand/result hints.

    Returns:
        pb.Signature: Typed protobuf signature.

    Raises:
        ValueError: If a hint or ValueType is malformed.
    """
    message = pb.Signature()
    for raw_hint in value.get("operands", []):
        hint = message.operands.add()
        hint.present = raw_hint is not None
        if raw_hint is not None:
            hint.name = raw_hint["name"]
            hint.type.CopyFrom(_value_type_to_proto(raw_hint["type"]))
    for raw_hint in value.get("results", []):
        hint = message.results.add()
        hint.present = True
        hint.name = raw_hint["name"]
        hint.type.CopyFrom(_value_type_to_proto(raw_hint["type"]))
    return message


def _signature_from_proto(message: pb.Signature) -> dict[str, Any]:
    """Decode a protobuf callable signature.

    Args:
        message (pb.Signature): Typed protobuf signature.

    Returns:
        dict[str, Any]: Internal signature record.

    Raises:
        ValueError: If a result hint is marked absent or a type is malformed.
    """
    operands = [
        (
            {
                "name": hint.name,
                "type": _value_type_from_proto(hint.type),
            }
            if hint.present
            else None
        )
        for hint in message.operands
    ]
    results = []
    for hint in message.results:
        if not hint.present:
            raise ValueError("callable result hints cannot be absent")
        results.append({"name": hint.name, "type": _value_type_from_proto(hint.type)})
    return {"operands": operands, "results": results}


def _implementation_to_proto(value: dict[str, Any]) -> pb.CallableImplementation:
    """Encode a backend-neutral callable implementation candidate.

    Args:
        value (dict[str, Any]): Implementation record without an emitter object.

    Returns:
        pb.CallableImplementation: Typed implementation candidate.

    Raises:
        ValueError: If a nested callable or block record is malformed.
        TypeError: If attrs contain an unsupported payload.
    """
    message = pb.CallableImplementation(transform=value["transform"])
    if value.get("backend") is not None:
        message.backend = value["backend"]
    if value.get("strategy") is not None:
        message.strategy = value["strategy"]
    if value.get("body") is not None:
        message.body.CopyFrom(_block_to_proto(value["body"]))
    if value.get("body_ref") is not None:
        message.body_ref.CopyFrom(_body_ref_to_proto(value["body_ref"]))
    message.attrs.CopyFrom(_payload_to_proto(value["attrs"]))
    return message


def _implementation_from_proto(
    message: pb.CallableImplementation,
) -> dict[str, Any]:
    """Decode a protobuf callable implementation candidate.

    Args:
        message (pb.CallableImplementation): Typed implementation candidate.

    Returns:
        dict[str, Any]: Internal implementation record.

    Raises:
        ValueError: If nested payloads are malformed.
    """
    return {
        "transform": message.transform,
        "backend": message.backend if message.HasField("backend") else None,
        "strategy": message.strategy if message.HasField("strategy") else None,
        "body": _block_from_proto(message.body) if message.HasField("body") else None,
        "body_ref": (
            _body_ref_from_proto(message.body_ref)
            if message.HasField("body_ref")
            else None
        ),
        "attrs": _payload_from_proto(message.attrs),
    }


def _callable_definition_to_proto(value: dict[str, Any]) -> pb.CallableDefinition:
    """Encode a callable definition without resource-estimation annotations.

    Args:
        value (dict[str, Any]): Callable-definition graph record.

    Returns:
        pb.CallableDefinition: Typed algorithm definition.

    Raises:
        ValueError: If nested graph records are malformed.
        TypeError: If attrs contain an unsupported payload.
    """
    message = pb.CallableDefinition(default_policy=value["default_policy"])
    message.ref.CopyFrom(_callable_ref_to_proto(value["ref"]))
    if value.get("signature") is not None:
        message.signature.CopyFrom(_signature_to_proto(value["signature"]))
    if value.get("body") is not None:
        message.body.CopyFrom(_block_to_proto(value["body"]))
    if value.get("body_ref") is not None:
        message.body_ref.CopyFrom(_body_ref_to_proto(value["body_ref"]))
    message.implementations.extend(
        _implementation_to_proto(item) for item in value["implementations"]
    )
    message.attrs.CopyFrom(_payload_to_proto(value["attrs"]))
    return message


def _callable_definition_from_proto(
    message: pb.CallableDefinition,
) -> dict[str, Any]:
    """Decode a protobuf callable definition.

    Args:
        message (pb.CallableDefinition): Typed algorithm definition.

    Returns:
        dict[str, Any]: Internal definition record with no ``opaque_cost``.

    Raises:
        ValueError: If nested graph records are malformed.
    """
    return {
        "ref": _callable_ref_from_proto(message.ref),
        "signature": (
            _signature_from_proto(message.signature)
            if message.HasField("signature")
            else None
        ),
        "body": _block_from_proto(message.body) if message.HasField("body") else None,
        "body_ref": (
            _body_ref_from_proto(message.body_ref)
            if message.HasField("body_ref")
            else None
        ),
        "implementations": [
            _implementation_from_proto(item) for item in message.implementations
        ],
        "default_policy": message.default_policy,
        "attrs": _payload_from_proto(message.attrs),
    }


def _callable_entry_to_proto(value: Any) -> pb.CallableEntry:
    """Encode one module-local callable-table entry.

    Args:
        value (Any): Callable-table record.

    Returns:
        pb.CallableEntry: Typed callable-table entry.

    Raises:
        ValueError: If the entry or definition is malformed.
        TypeError: If definition attrs are unsupported.
    """
    if not isinstance(value, dict) or not isinstance(value.get("id"), str):
        raise ValueError("callable_table entries require a string id")
    message = pb.CallableEntry(id=value["id"])
    message.definition.CopyFrom(_callable_definition_to_proto(value["definition"]))
    return message


def _callable_entry_from_proto(message: pb.CallableEntry) -> dict[str, Any]:
    """Decode one protobuf callable-table entry.

    Args:
        message (pb.CallableEntry): Typed callable-table entry.

    Returns:
        dict[str, Any]: Internal callable-table record.

    Raises:
        ValueError: If the nested definition is malformed.
    """
    return {
        "id": message.id,
        "definition": _callable_definition_from_proto(message.definition),
    }
