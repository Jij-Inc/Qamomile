"""numpy ndarray wrapper used by both JSON and msgpack pipelines.

The wrapper is a tagged dict that both wire formats produce and
consume in identical shape; only the encoding of the ``"data"``
field differs at the wire boundary (base64 string in JSON, native
``bin`` in msgpack).

Allowed dtypes are restricted to an explicit allow-list. The decoder
rejects any dtype string outside the list, so a malicious payload
cannot coax ``numpy`` into instantiating an unexpected dtype object.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Allow-list of dtype names the decoder accepts. Anything outside this
# set raises ValueError. Keep limited to dtypes qamomile actually uses
# in IR payloads (binding values + metadata).
_ALLOWED_DTYPES: frozenset[str] = frozenset(
    {
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "bool",
    }
)

_NP_TAG = "$np_array"


def is_array_wrapper(d: Any) -> bool:
    """Return True if ``d`` is a numpy-array wrapper dict.

    Args:
        d (Any): A value to check. Typically the result of a recursive
            dict walk from ``decode``.

    Returns:
        bool: ``True`` when ``d`` is a dict carrying the
            ``$np_array`` tag with a ``True``-ish value.
    """
    return isinstance(d, dict) and d.get(_NP_TAG) is True


def array_to_dict(arr: np.ndarray) -> dict[str, Any]:
    """Encode a numpy ndarray into the wrapper dict.

    Args:
        arr (np.ndarray): Source array. ``arr.dtype.name`` must be in
            the allow-list (:data:`_ALLOWED_DTYPES`).

    Returns:
        dict[str, Any]: A wrapper dict with ``$np_array``, ``dtype``,
            ``shape`` (list[int]), and ``data`` (raw ``bytes`` from
            ``ndarray.tobytes()``). The wire encoders are responsible
            for any further bytes ⇄ text conversion at format
            boundaries.

    Raises:
        TypeError: If ``arr`` is not a ``numpy.ndarray``.
        ValueError: If the array's dtype is not in the allow-list.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"array_to_dict() expected numpy.ndarray, got {type(arr).__name__}"
        )
    dtype_name = arr.dtype.name
    if dtype_name not in _ALLOWED_DTYPES:
        raise ValueError(
            f"numpy dtype {dtype_name!r} is not in the serialization "
            f"allow-list {sorted(_ALLOWED_DTYPES)}"
        )
    # Ensure contiguous bytes so np.frombuffer + reshape round-trips.
    contiguous = np.ascontiguousarray(arr)
    return {
        _NP_TAG: True,
        "dtype": dtype_name,
        "shape": list(contiguous.shape),
        "data": contiguous.tobytes(),
    }


def dict_to_array(d: dict[str, Any]) -> np.ndarray:
    """Decode a wrapper dict back into a numpy ndarray.

    Args:
        d (dict[str, Any]): A wrapper dict previously produced by
            :func:`array_to_dict` (possibly after a JSON round-trip
            that converted the ``data`` field between bytes and
            base64).

    Returns:
        np.ndarray: The reconstructed array with the original dtype
            and shape.

    Raises:
        ValueError: If ``d`` is not a valid wrapper dict, if the
            dtype is not in the allow-list, or if the byte length is
            inconsistent with shape × dtype.itemsize.
    """
    if not is_array_wrapper(d):
        raise ValueError("dict_to_array() called with a non-wrapper dict")
    dtype_name = d.get("dtype")
    if not isinstance(dtype_name, str) or dtype_name not in _ALLOWED_DTYPES:
        raise ValueError(
            f"numpy dtype {dtype_name!r} is not in the serialization "
            f"allow-list {sorted(_ALLOWED_DTYPES)}"
        )
    raw_shape = d.get("shape")
    if not isinstance(raw_shape, list) or not all(
        isinstance(x, int) for x in raw_shape
    ):
        raise ValueError("numpy wrapper 'shape' must be a list of ints")
    shape = tuple(raw_shape)
    data = d.get("data")
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError(
            "numpy wrapper 'data' must be bytes after wire-format decoding"
        )

    expected_size = int(np.dtype(dtype_name).itemsize)
    for dim in shape:
        expected_size *= dim
    if expected_size != len(data):
        raise ValueError(
            f"numpy wrapper data length {len(data)} does not match "
            f"expected {expected_size} bytes for dtype {dtype_name!r} "
            f"and shape {shape}"
        )

    return np.frombuffer(bytes(data), dtype=np.dtype(dtype_name)).reshape(shape).copy()
