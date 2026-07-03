"""Shared naming for per-key backend parameters of runtime-parameter Dicts.

A ``Dict[K, Float]`` kernel argument kept as a runtime parameter
(``transpile(..., parameters=["coeffs"])``) is decomposed into one
backend parameter per looked-up key. The emit pass creates each backend
parameter from the key it resolves (``coeffs[3]``, ``coeffs[(0, 1)]``),
and the program orchestrator decomposes the execution-time binding
``{"coeffs": {...}}`` into the same names. Both sides MUST agree on the
string format, so the formatting lives here and nowhere else.
"""

from __future__ import annotations

import typing


def dict_param_key(dict_name: str, key: typing.Any) -> str:
    """Format the backend-parameter name for one entry of a Dict parameter.

    Args:
        dict_name (str): The kernel argument name of the Dict parameter.
        key (Any): The looked-up key, already normalized (a plain ``int``
            or a tuple of plain ``int``s — see
            :func:`normalize_dict_binding_key`).

    Returns:
        str: The backend parameter name, e.g. ``"coeffs[3]"`` for an int
            key or ``"coeffs[(0, 1)]"`` for a tuple key.
    """
    return f"{dict_name}[{key}]"


def normalize_dict_binding_key(key: typing.Any) -> typing.Any:
    """Normalize a user-supplied dict key for parameter-name formatting.

    Integer-valued keys are canonicalized to plain ``int`` (``numpy.int64``,
    ``float`` ``1.0``, ...) so that the execution-time decomposition of
    ``{"coeffs": {np.int64(3): 0.5}}`` produces the same parameter name the
    emit pass created from the IR-resolved ``int`` key. Tuples/lists are
    normalized component-wise into a tuple. Non-integer-valued keys (``str``,
    ``1.5``, ...) are returned unchanged — they can never match an emitted
    parameter name (emit only produces int-component keys), so they simply
    stay unused instead of colliding via lossy coercion.

    Args:
        key (Any): A key of the user-supplied binding dict.

    Returns:
        Any: ``int``, ``tuple`` of normalized components, or the original
            key when it has no exact integer representation.
    """
    if isinstance(key, (tuple, list)):
        return tuple(normalize_dict_binding_key(component) for component in key)
    try:
        as_int = int(key)
    except (TypeError, ValueError):
        return key
    # Only adopt the int form when it is exactly equal to the original
    # (guards against silently mapping 1.5 -> 1). Plain bool is int-equal
    # and harmless. String keys like "3" are not int-equal and stay str.
    if as_int == key:
        return as_int
    return key
