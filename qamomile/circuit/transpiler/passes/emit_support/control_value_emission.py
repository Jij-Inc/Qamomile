"""Shared emission support for computational-basis control values."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from qamomile.circuit.ir.operation.control_value import control_pattern_for_value

if TYPE_CHECKING:
    from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass


@contextmanager
def bracket_control_value(
    emit_pass: "StandardEmitPass",
    circuit: Any,
    control_indices: Sequence[int],
    control_value: int | None,
) -> Iterator[None]:
    """Bracket zero-valued controls with target-neutral Pauli-X gates.

    The controlled operation inside the context remains an ordinary all-ones
    control. Controls are interpreted LSB-first in their existing physical
    order, so bit zero of ``control_value`` describes ``control_indices[0]``.
    One bracket surrounds the complete operation, including integral powers
    and vector-target broadcast.

    Args:
        emit_pass (StandardEmitPass): Active emit pass providing the gate
            emitter.
        circuit (Any): Circuit receiving the bracket gates.
        control_indices (Sequence[int]): Ordered physical control qubits.
        control_value (int | None): Required basis value, or ``None`` for the
            ordinary all-ones state.

    Yields:
        None: Control returns while the zero-valued controls are inverted.

    Raises:
        TypeError: If ``control_value`` is not a Python ``int`` or ``None``.
        ValueError: If the activation value does not fit the control width.
    """
    if not control_indices:
        if control_value is not None:
            raise ValueError(
                "control_value requires at least one physical control qubit."
            )
        yield
        return

    pattern = control_pattern_for_value(control_value, len(control_indices))
    zero_controls = tuple(
        physical
        for physical, required in zip(control_indices, pattern, strict=True)
        if required == 0
    )
    for physical in zero_controls:
        emit_pass._emitter.emit_x(circuit, physical)
    try:
        yield
    finally:
        for physical in reversed(zero_controls):
            emit_pass._emitter.emit_x(circuit, physical)


__all__ = ["bracket_control_value"]
