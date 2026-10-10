"""Restore tagged HUGR outputs from Nexus H2 bit-register results.

QIR boolean records use one bit. Labels identify public output leaves; provider
column order never defines the public return order. Decoding costs
O(shots * recorded bits).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np

from qamomile.circuit.transpiler.errors import ExecutionError


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject repeated JSON object keys instead of silently overwriting them.

    Args:
        pairs (list[tuple[str, Any]]): Object entries in serialized order.

    Returns:
        dict[str, Any]: An object with unique keys.

    Raises:
        ValueError: If an object contains the same key more than once.
    """
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate QIR output tag: {key!r}")
        result[key] = value
    return result


def _validate_outputs(outputs: Mapping[str, Any]) -> dict[str, tuple[str, int]]:
    """Own and validate the supported scalar QIR output descriptors.

    Args:
        outputs (Mapping[str, Any]): Tags mapped to a kind and register width.

    Returns:
        dict[str, tuple[str, int]]: Boolean width-one output tags.

    Raises:
        ValueError: If tags, kinds, or widths are outside the H2 output contract.
    """
    if not isinstance(outputs, Mapping):
        raise ValueError("QIR output manifest must be an object")
    validated = {}
    for tag, descriptor in outputs.items():
        if not isinstance(tag, str) or not tag:
            raise ValueError("QIR output tags must be non-empty strings")
        if (
            not isinstance(descriptor, (list, tuple))
            or len(descriptor) != 2
            or not isinstance(descriptor[0], str)
            or type(descriptor[1]) is not int
        ):
            raise ValueError(f"Invalid QIR output descriptor for {tag!r}")
        kind, width = descriptor
        if (kind, width) != ("bool", 1):
            raise ValueError(f"Unsupported QIR output descriptor for {tag!r}")
        validated[tag] = (kind, width)
    return validated


def parse_qir_output_manifest(encoded: str) -> dict[str, tuple[str, int]]:
    """Read a strict output manifest from a saved execution reference.

    Args:
        encoded (str): JSON object mapping tags to ``[kind, width]`` lists.
            The supported descriptor is ``["bool", 1]``.

    Returns:
        dict[str, tuple[str, int]]: Independently owned scalar output metadata.

    Raises:
        ValueError: If JSON, object keys, scalar kinds, or widths are invalid.
    """
    if not isinstance(encoded, str):
        raise ValueError("QIR output manifest must be a JSON string")
    try:
        outputs = json.loads(encoded, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Invalid JSON QIR output manifest") from error
    return _validate_outputs(outputs)


def _register_columns(
    bits: Sequence[Any], outputs: Mapping[str, tuple[str, int]]
) -> dict[str, dict[int, int]]:
    """Locate each declared register bit in the requested SDK column order.

    Args:
        bits (Sequence[Any]): Native bit identifiers in requested column order.
        outputs (Mapping[str, tuple[str, int]]): Validated tags and widths.

    Returns:
        dict[str, dict[int, int]]: Tag and bit-index to readout-column mappings.

    Raises:
        ExecutionError: If registers are missing, repeated, sparse, or unexpected.
    """
    columns: dict[str, dict[int, int]] = {}
    for column, bit in enumerate(bits):
        tag = getattr(bit, "reg_name", None)
        index = getattr(bit, "index", None)
        if not isinstance(tag, str) or tag not in outputs:
            raise ExecutionError(f"Unexpected QIR result register: {tag!r}")
        if (
            not isinstance(index, (list, tuple))
            or len(index) != 1
            or isinstance(index[0], bool)
            or not isinstance(index[0], Integral)
        ):
            raise ExecutionError(f"Invalid QIR result bit index for {tag!r}")
        position = int(index[0])
        width = outputs[tag][1]
        if not 0 <= position < width:
            raise ExecutionError(f"QIR result bit index exceeds width for {tag!r}")
        register = columns.setdefault(tag, {})
        if position in register:
            raise ExecutionError(f"Duplicate QIR result bit for {tag!r}")
        register[position] = column
    for tag, (_, width) in outputs.items():
        if len(columns.get(tag, {})) != width:
            raise ExecutionError(f"Missing QIR result bits for {tag!r}")
    return columns


def decode_qir_results(
    raw: Any, shots: int, outputs: Mapping[str, tuple[str, int]]
) -> list[dict[str, Any]]:
    """Decode H2 SDK bit registers into the existing tagged-result contract.

    Args:
        raw (Any): Native ``pytket.backends.BackendResult`` with shot readouts.
        shots (int): Required positive number of shots.
        outputs (Mapping[str, tuple[str, int]]): Tags mapped to ``("bool", 1)``.

    Returns:
        list[dict[str, Any]]: One tag-to-Python-bool mapping per shot.

    Raises:
        ValueError: If shots or the output manifest is invalid.
        ExecutionError: If the provider result has incompatible registers,
            readouts, bit values, or shot count.
    """
    count: Any = shots
    if isinstance(count, bool) or not isinstance(count, Integral) or count <= 0:
        raise ValueError("shots must be a positive integer")
    outputs = _validate_outputs(outputs)
    get_bits = getattr(raw, "get_bitlist", None)
    get_shots = getattr(raw, "get_shots", None)
    if not callable(get_bits) or not callable(get_shots):
        raise ExecutionError("Nexus did not return H2 QIR bit-register results")
    try:
        bits = get_bits()
    except Exception as error:
        raise ExecutionError("Cannot read QIR result registers") from error
    if not isinstance(bits, (list, tuple)):
        raise ExecutionError("Nexus returned invalid QIR result registers")
    columns = _register_columns(bits, outputs)
    try:
        readouts = np.asarray(get_shots(cbits=bits))
    except Exception as error:
        raise ExecutionError("Cannot read QIR shot results") from error
    if readouts.shape != (shots, len(bits)):
        raise ExecutionError(
            f"Nexus returned QIR readout shape {readouts.shape}; "
            f"expected {(shots, len(bits))}"
        )
    if readouts.size and (
        readouts.dtype.kind not in "biu" or np.any((readouts != 0) & (readouts != 1))
    ):
        raise ExecutionError("QIR shot readouts must contain only integer bits")
    records = []
    for row in readouts:
        record = {}
        for tag in outputs:
            record[tag] = bool(row[columns[tag][0]])
        records.append(record)
    return records
