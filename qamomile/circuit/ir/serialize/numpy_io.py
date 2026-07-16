"""Build validated NumPy records for the protobuf payload union.

The semantic encoder uses tagged intermediate records before the protobuf
bridge materializes ``NumpyValue`` messages. Arrays carry shape plus raw bytes;
scalars carry their dtype plus one exact item's bytes.

Allowed dtypes are restricted to an explicit allow-list. The decoder
rejects any dtype string outside the list, so a malicious payload
cannot coax ``numpy`` into instantiating an unexpected dtype object.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qamomile._utils import is_plain_int

# Allow-list of primitive kind/itemsize pairs. The serialized spelling is the
# canonical ``dtype.str``, which also preserves byte order.
_ALLOWED_DTYPE_LAYOUTS: frozenset[tuple[str, int]] = frozenset(
    {
        ("i", 1),
        ("i", 2),
        ("i", 4),
        ("i", 8),
        ("u", 1),
        ("u", 2),
        ("u", 4),
        ("u", 8),
        ("f", 2),
        ("f", 4),
        ("f", 8),
        ("c", 8),
        ("c", 16),
        ("b", 1),
    }
)

_NP_ARRAY_TAG = "$np_array"
_NP_SCALAR_TAG = "$np_scalar"


def _validated_dtype(dtype_spec: Any, label: str) -> np.dtype:
    """Parse one canonical, allow-listed primitive NumPy dtype.

    Args:
        dtype_spec (Any): Candidate canonical ``dtype.str`` value.
        label (str): Diagnostic label such as ``"array"`` or ``"scalar"``.

    Returns:
        np.dtype: Validated dtype preserving its explicit byte order.

    Raises:
        ValueError: If the spelling is non-canonical or the primitive layout
            is outside the allow-list.
    """
    if not isinstance(dtype_spec, str):
        raise ValueError(f"numpy {label} dtype {dtype_spec!r} is not a string")
    try:
        dtype = np.dtype(dtype_spec)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"numpy {label} dtype {dtype_spec!r} is invalid") from exc
    if (
        dtype.str != dtype_spec
        or dtype.fields is not None
        or dtype.subdtype is not None
        or (dtype.kind, dtype.itemsize) not in _ALLOWED_DTYPE_LAYOUTS
    ):
        raise ValueError(
            f"numpy {label} dtype {dtype_spec!r} is not in the serialization "
            "allow-list of primitive bool, integer, float, and complex layouts"
        )
    return dtype


def is_array_wrapper(d: Any) -> bool:
    """Return True if ``d`` is a numpy-array wrapper dict.

    Args:
        d (Any): A value to check. Typically the result of a recursive
            dict walk from ``decode``.

    Returns:
        bool: ``True`` when ``d`` is a dict carrying the
            ``$np_array`` tag with a ``True``-ish value.
    """
    return isinstance(d, dict) and d.get(_NP_ARRAY_TAG) is True


def is_scalar_wrapper(d: Any) -> bool:
    """Return whether ``d`` is a NumPy-scalar wrapper dict.

    Args:
        d (Any): Candidate wire payload.

    Returns:
        bool: Whether ``d`` carries the exact scalar wrapper tag.
    """
    return isinstance(d, dict) and d.get(_NP_SCALAR_TAG) is True


def array_to_dict(arr: np.ndarray) -> dict[str, Any]:
    """Encode a numpy ndarray into the wrapper dict.

    Args:
        arr (np.ndarray): Source array with an allow-listed primitive dtype.

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
    dtype_spec = arr.dtype.str
    _validated_dtype(dtype_spec, "array")
    # Ensure contiguous bytes so np.frombuffer + reshape round-trips.
    # ``np.ascontiguousarray`` promotes a zero-dimensional array to shape
    # ``(1,)``.  ``copy(order="C")`` preserves rank while still producing
    # canonical C-order bytes for non-contiguous inputs.
    contiguous = arr.copy(order="C")
    return {
        _NP_ARRAY_TAG: True,
        "dtype": dtype_spec,
        "shape": list(contiguous.shape),
        "data": contiguous.tobytes(),
    }


def dict_to_array(d: dict[str, Any]) -> np.ndarray:
    """Decode a wrapper dict back into a numpy ndarray.

    Args:
        d (dict[str, Any]): A wrapper dict previously produced by
            :func:`array_to_dict` after protobuf decoding.

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
    dtype_spec = d.get("dtype")
    dtype = _validated_dtype(dtype_spec, "array")
    raw_shape = d.get("shape")
    if not isinstance(raw_shape, list) or not all(is_plain_int(x) for x in raw_shape):
        raise ValueError("numpy wrapper 'shape' must be a list of ints")
    shape = tuple(raw_shape)
    data = d.get("data")
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError(
            "numpy wrapper 'data' must be bytes after wire-format decoding"
        )

    expected_size = int(dtype.itemsize)
    for dim in shape:
        expected_size *= dim
    if expected_size != len(data):
        raise ValueError(
            f"numpy wrapper data length {len(data)} does not match "
            f"expected {expected_size} bytes for dtype {dtype_spec!r} "
            f"and shape {shape}"
        )

    return np.frombuffer(bytes(data), dtype=dtype).reshape(shape).copy()


def scalar_to_dict(value: np.generic) -> dict[str, Any]:
    """Encode an allow-listed NumPy scalar without widening its dtype.

    Args:
        value (np.generic): NumPy scalar whose dtype and exact bytes must be
            preserved.

    Returns:
        dict[str, Any]: Tagged scalar wrapper containing dtype and raw bytes.

    Raises:
        TypeError: If ``value`` is not a NumPy scalar.
        ValueError: If its dtype is outside the portable allow-list.
    """
    if not isinstance(value, np.generic):
        raise TypeError(
            f"scalar_to_dict() expected numpy scalar, got {type(value).__name__}"
        )
    dtype_spec = value.dtype.str
    _validated_dtype(dtype_spec, "scalar")
    return {
        _NP_SCALAR_TAG: True,
        "dtype": dtype_spec,
        "data": np.asarray(value).tobytes(),
    }


def dict_to_scalar(d: dict[str, Any]) -> np.generic:
    """Decode an exact NumPy scalar wrapper.

    Args:
        d (dict[str, Any]): Wrapper produced by :func:`scalar_to_dict`.

    Returns:
        np.generic: Scalar with the original dtype and bit representation.

    Raises:
        ValueError: If the wrapper, dtype, or byte length is malformed.
    """
    if not is_scalar_wrapper(d):
        raise ValueError("dict_to_scalar() called with a non-wrapper dict")
    dtype_spec = d.get("dtype")
    dtype = _validated_dtype(dtype_spec, "scalar")
    data = d.get("data")
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError(
            "numpy scalar wrapper 'data' must be bytes after wire-format decoding"
        )
    if len(data) != dtype.itemsize:
        raise ValueError(
            f"numpy scalar wrapper data length {len(data)} does not match "
            f"dtype {dtype_spec!r} itemsize {dtype.itemsize}"
        )
    return np.frombuffer(bytes(data), dtype=dtype, count=1)[0]
