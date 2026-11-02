"""Static owned-register selection composed from ordinary Qamomile functions."""

from __future__ import annotations

import qamomile as qm
from qamomile.effects import CallableKind
from qamomile.types import (
    FunctionType,
    infer_declared_function_application,
    is_type_assignable,
)

from ._structure import lift_indices

_MAX_SELECTION_WIDTH = 128


def on_qubits(
    process: qm.QKernel, *, width: int, indices: tuple[int, ...]
) -> qm.QKernel:
    """Apply a closed unary process to statically selected register coordinates.

    Helper coordinate zero is ``indices[0]``, preserving the supplied order.
    The returned kernel consumes and returns the entire ambient register in its
    original coordinate order. Regrouping emits no SWAP gates and creates no
    borrowed views. Capture fixed classical arguments in a unary wrapper before
    lifting a process that otherwise needs additional arguments.

    Args:
        process (qm.QKernel): Closed unitary ``Q[Bits[k]] -> Q[Bits[k]]`` process.
        width (int): Ambient register width, from one through 128.
        indices (tuple[int, ...]): Nonempty unique positions within the register.

    Returns:
        qm.QKernel: Unary transform over ``Q[Bits[width]]``.

    Raises:
        TypeError: If configuration types or the process ABI are incompatible.
        ValueError: If width, selected positions, or construction budget is invalid.
        qm.QamomileError: If native compilation rejects the process.
    """

    if type(width) is not int or not isinstance(indices, tuple):
        raise TypeError("width must be an integer and indices must be a tuple")
    if not 1 <= width <= _MAX_SELECTION_WIDTH:
        raise ValueError("selection width must be between 1 and 128")
    if not indices or len(indices) > width:
        raise ValueError("selection must be nonempty and fit the register")
    if any(type(index) is not int for index in indices):
        raise TypeError("selected positions must be integers")
    if len(set(indices)) != len(indices) or any(
        not 0 <= index < width for index in indices
    ):
        raise ValueError("selected positions must be unique and in range")
    if not isinstance(process, qm.QKernel):
        raise TypeError("process must be a Qamomile QKernel")
    function = process.module().function(process.function_name)
    parameters = tuple(parameter.value.type for parameter in function.parameters)
    if len(parameters) != 1 or function.kind is not CallableKind.UNITARY:
        raise TypeError("selection requires one closed unary unitary process")
    selected_type = qm.Q[qm.Bits[len(indices)]]
    try:
        result = infer_declared_function_application(
            FunctionType(
                parameters,
                function.result_type,
                tuple(p.name for p in function.parameters),
            ),
            (selected_type,),
            (None,),
        )
        if not is_type_assignable(selected_type, result):
            raise ValueError("process must return the selected register")
    except ValueError as error:
        raise TypeError("process must preserve the selected Bits register") from error
    return lift_indices(total_width=width, indices=indices, transform=process)


__all__ = ["on_qubits"]
