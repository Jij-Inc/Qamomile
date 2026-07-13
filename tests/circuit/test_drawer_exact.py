"""Regression tests for correctness-first qkernel drawing."""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")

import pytest
from matplotlib.figure import Figure

import qamomile.circuit as qmc


@qmc.qkernel
def _scalar_quantum_input(q: qmc.Qubit) -> qmc.Qubit:
    """Apply a gate to a scalar external quantum input.

    Args:
        q (qmc.Qubit): External qubit to update.

    Returns:
        qmc.Qubit: Updated external qubit.
    """
    return qmc.h(q)


@qmc.qkernel
def _vector_quantum_input(
    q: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Apply a gate to every external quantum-vector element.

    Args:
        q (qmc.Vector[qmc.Qubit]): External qubit register to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated external qubit register.
    """
    return qmc.h(q)


@qmc.qkernel
def _symbolic_quantum_index(
    q: qmc.Vector[qmc.Qubit],
    index: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Apply X at a draw-time-selected quantum index.

    Args:
        q (qmc.Vector[qmc.Qubit]): External qubit register to update.
        index (qmc.UInt): Index selecting the target qubit.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated external qubit register.
    """
    q[index] = qmc.x(q[index])
    return q


def _q_wire_labels(figure: Figure) -> list[str]:
    """Return rendered labels belonging to the external ``q`` input.

    Args:
        figure (Figure): Rendered circuit figure.

    Returns:
        list[str]: Labels whose text begins with ``q``.
    """
    axis = figure._qm_ax  # type: ignore[attr-defined]
    return [
        text.get_text()
        for text in axis.texts
        if re.fullmatch(r"q(?:\[\d+\])?", text.get_text())
    ]


def test_draw_accepts_scalar_quantum_input() -> None:
    """A scalar quantum input remains a single named display wire."""
    figure = _scalar_quantum_input.draw()

    assert isinstance(figure, Figure)
    assert _q_wire_labels(figure) == ["q"]


def test_draw_accepts_vector_quantum_input() -> None:
    """A sized quantum-vector input keeps exactly its external wires."""
    figure = _vector_quantum_input.draw(q=3)

    assert isinstance(figure, Figure)
    assert _q_wire_labels(figure) == ["q[0]", "q[1]", "q[2]"]


def test_draw_rejects_unresolved_quantum_index() -> None:
    """Drawing never guesses which wire an unresolved index denotes."""
    with pytest.raises(ValueError, match="index"):
        _symbolic_quantum_index.draw(q=3)


def test_draw_accepts_bound_quantum_index() -> None:
    """A concrete quantum index can be lowered to an exact target wire."""
    figure = _symbolic_quantum_index.draw(q=3, index=2)

    assert isinstance(figure, Figure)
    assert _q_wire_labels(figure) == ["q[0]", "q[1]", "q[2]"]
