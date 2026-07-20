"""Standalone helpers for quantum optimization post-processing.

This module hosts small, quantum-agnostic conversion utilities that
tutorials and downstream algorithms (QeMCMC, QRAO rounding, QAOA
post-processing, ...) would otherwise re-implement inline. Keeping them in
one canonical place avoids the subtle sign/offset bugs that come from
hand-writing spin/binary conversions per call site.

Spin/binary convention
-----------------------

Qamomile uses the Z-eigenvalue correspondence throughout its optimization
stack (``BinaryModel``, ``post_process.local_search``, the QRAO
converters): a computational-basis bit maps to a Pauli-Z eigenvalue via

    binary | spin
      0    |  +1
      1    |  -1

which is ``s = 1 - 2 * x`` with inverse ``x = (1 - s) / 2``. These helpers
follow that same convention so their output is directly interchangeable
with the rest of the package. (Note this is the Z-eigenvalue sign, i.e.
``1 - 2 * x``, not the alternative ``2 * x - 1`` mapping.)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, overload

import numpy as np

__all__ = ["binary_to_spin", "spin_to_binary"]


def _spin_from_binary_scalar(value: int) -> int:
    """Convert a single binary bit to its spin value.

    Args:
        value (int): Binary bit; must be ``0`` or ``1`` (``bool`` accepted).

    Returns:
        int: ``+1`` for ``0`` and ``-1`` for ``1`` (``s = 1 - 2 * x``).

    Raises:
        ValueError: If ``value`` is not ``0`` or ``1``.
    """
    if value not in (0, 1):
        raise ValueError(f"binary values must be 0 or 1, got {value!r}")
    return 1 - 2 * int(value)


def _binary_from_spin_scalar(value: int) -> int:
    """Convert a single spin value to its binary bit.

    Args:
        value (int): Spin value; must be ``+1`` or ``-1``.

    Returns:
        int: ``0`` for ``+1`` and ``1`` for ``-1`` (``x = (1 - s) / 2``).

    Raises:
        ValueError: If ``value`` is not ``+1`` or ``-1``.
    """
    if value not in (1, -1):
        raise ValueError(f"spin values must be +1 or -1, got {value!r}")
    return (1 - int(value)) // 2


def _require_integral_dtype(values: np.ndarray) -> None:
    """Reject NumPy arrays whose dtype is not integer or boolean.

    A float or object array would slip through the value-domain check (for
    example ``0.0 == 0``) and silently accept non-bit data, so the dtype is
    guarded up front to keep the contract identical to the scalar path,
    where a ``float`` raises ``TypeError``.

    Args:
        values (np.ndarray): Array to validate.

    Returns:
        None

    Raises:
        TypeError: If ``values.dtype`` is neither integer nor boolean.
    """
    if not (
        np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_)
    ):
        raise TypeError(
            "bit arrays must have an integer or boolean dtype, "
            f"got dtype {values.dtype}"
        )


def _spin_from_binary_array(values: np.ndarray) -> np.ndarray:
    """Vectorized binary-to-spin conversion over a NumPy array.

    Args:
        values (np.ndarray): Integer/boolean array whose entries are all
            ``0`` or ``1``.

    Returns:
        np.ndarray: Integer array of ``+1`` / ``-1`` spins with the same
            shape as ``values``.

    Raises:
        TypeError: If ``values.dtype`` is neither integer nor boolean.
        ValueError: If any entry is not ``0`` or ``1``.
    """
    _require_integral_dtype(values)
    if not np.isin(values, (0, 1)).all():
        raise ValueError("binary arrays must contain only 0 or 1 entries")
    return 1 - 2 * values.astype(int)


def _binary_from_spin_array(values: np.ndarray) -> np.ndarray:
    """Vectorized spin-to-binary conversion over a NumPy array.

    Args:
        values (np.ndarray): Integer/boolean array whose entries are all
            ``+1`` or ``-1``.

    Returns:
        np.ndarray: Integer array of ``0`` / ``1`` bits with the same shape
            as ``values``.

    Raises:
        TypeError: If ``values.dtype`` is neither integer nor boolean.
        ValueError: If any entry is not ``+1`` or ``-1``.
    """
    _require_integral_dtype(values)
    if not np.isin(values, (1, -1)).all():
        raise ValueError("spin arrays must contain only +1 or -1 entries")
    return (1 - values.astype(int)) // 2


@overload
def binary_to_spin(values: int) -> int: ...
@overload
def binary_to_spin(values: np.ndarray) -> np.ndarray: ...
@overload
def binary_to_spin(values: Mapping[Any, Any]) -> dict[Any, Any]: ...
@overload
def binary_to_spin(values: Sequence[Any]) -> list[Any]: ...
def binary_to_spin(values: Any) -> Any:
    """Convert binary (0/1) values to spin (+1/-1) values.

    Applies the Z-eigenvalue mapping ``s = 1 - 2 * x`` (``0 -> +1``,
    ``1 -> -1``), matching the convention used across
    ``qamomile.optimization``. The input shape is preserved: scalars return
    scalars, NumPy arrays return NumPy arrays, sequences return lists, and
    mappings return dicts (converting the values, leaving keys untouched).
    Sequences and mapping values are converted recursively, so
    decoded-sample containers such as ``list[list[int]]`` are supported.

    Args:
        values (int | np.ndarray | Sequence | Mapping): Binary value(s) to
            convert. Every leaf entry must be ``0`` or ``1`` (``bool`` is
            accepted as ``0`` / ``1``).

    Returns:
        int | np.ndarray | list | dict: Spin value(s) in the same container
            shape as ``values``.

    Raises:
        ValueError: If any leaf entry is not ``0`` or ``1``.
        TypeError: If ``values`` is not an ``int``, NumPy array, sequence,
            or mapping, or if a NumPy array's dtype is neither integer nor
            boolean.

    Example:
        >>> binary_to_spin(0)
        1
        >>> binary_to_spin([0, 1, 1])
        [1, -1, -1]
    """
    return _convert(values, _spin_from_binary_scalar, _spin_from_binary_array)


@overload
def spin_to_binary(values: int) -> int: ...
@overload
def spin_to_binary(values: np.ndarray) -> np.ndarray: ...
@overload
def spin_to_binary(values: Mapping[Any, Any]) -> dict[Any, Any]: ...
@overload
def spin_to_binary(values: Sequence[Any]) -> list[Any]: ...
def spin_to_binary(values: Any) -> Any:
    """Convert spin (+1/-1) values to binary (0/1) values.

    Applies the inverse Z-eigenvalue mapping ``x = (1 - s) / 2``
    (``+1 -> 0``, ``-1 -> 1``), matching the convention used across
    ``qamomile.optimization``. The input shape is preserved exactly as in
    :func:`binary_to_spin`, and sequences / mapping values are converted
    recursively so decoded-sample containers are supported.

    Args:
        values (int | np.ndarray | Sequence | Mapping): Spin value(s) to
            convert. Every leaf entry must be ``+1`` or ``-1``.

    Returns:
        int | np.ndarray | list | dict: Binary value(s) in the same
            container shape as ``values``.

    Raises:
        ValueError: If any leaf entry is not ``+1`` or ``-1``.
        TypeError: If ``values`` is not an ``int``, NumPy array, sequence,
            or mapping, or if a NumPy array's dtype is neither integer nor
            boolean.

    Example:
        >>> spin_to_binary(1)
        0
        >>> spin_to_binary([1, -1, -1])
        [0, 1, 1]
    """
    return _convert(values, _binary_from_spin_scalar, _binary_from_spin_array)


def _convert(
    values: Any,
    scalar_fn: Any,
    array_fn: Any,
) -> Any:
    """Dispatch a scalar/array conversion over the supported input shapes.

    Args:
        values (int | np.ndarray | Sequence | Mapping): Value(s) to convert.
        scalar_fn (Callable[[int], int]): Conversion applied to a single
            integer leaf.
        array_fn (Callable[[np.ndarray], np.ndarray]): Vectorized
            conversion applied to a NumPy array.

    Returns:
        int | np.ndarray | list | dict: Converted value(s) in the same
            container shape as ``values``.

    Raises:
        ValueError: Propagated from ``scalar_fn`` / ``array_fn`` on an
            out-of-domain leaf entry.
        TypeError: If ``values`` is not an ``int``, NumPy array, sequence,
            or mapping, or if a NumPy array's dtype is neither integer nor
            boolean.
    """
    if isinstance(values, np.ndarray):
        return array_fn(values)
    if isinstance(values, Mapping):
        return {
            key: _convert(item, scalar_fn, array_fn) for key, item in values.items()
        }
    # bool is a subclass of int; np.integer/np.bool_ cover NumPy scalars.
    if isinstance(values, (int, np.integer, np.bool_)):
        return scalar_fn(int(values))
    # str/bytes/bytearray are Sequences but never valid bit containers.
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        return [_convert(item, scalar_fn, array_fn) for item in values]
    raise TypeError(
        "expected an int, NumPy array, sequence, or mapping of bits, "
        f"got {type(values).__name__}"
    )
