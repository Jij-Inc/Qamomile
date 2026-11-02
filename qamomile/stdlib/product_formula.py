# ruff: noqa: F821 -- dimensions are Qamomile dependent shape binders.

"""Product-formula evolution compiled to ordinary Qamomile source."""

from __future__ import annotations

import math

import qamomile as qm
from qamomile._native import compiler as native

_Schedule = tuple[tuple[int, float], ...]


def suzuki_trotter(
    *,
    term_count: int,
    order: int = 2,
    steps: int = 1,
) -> qm.QKernel:
    """Construct fixed-topology Suzuki-Trotter time evolution.

    The factory computes the immutable Suzuki coefficient schedule once. The
    returned Qamomile function contains only a fixed aggregate loop and the
    existing abstract Pauli-evolution operation. Consequently product formulas
    add no QCore opcode, recursion policy, resource-estimator case, or backend
    branch.

    Args:
        term_count: Number of ordered sub-Hamiltonians in the sum.
        order: Formula order: one, or a positive even integer.
        steps: Positive number of equal time slices.

    Returns:
        A transform accepting ``state``, a Const vector of sub-Hamiltonians,
        and the total runtime-parametric evolution time.

    Raises:
        TypeError: If an option is not an integer.
        ValueError: If the term count, order, or step count is outside its
            supported domain.
    """

    _require_integer("term_count", term_count)
    _require_integer("order", order)
    _require_integer("steps", steps)
    if term_count < 2:
        raise ValueError("term_count must be at least two")
    if order != 1 and (order < 2 or order % 2):
        raise ValueError("order must be one or a positive even integer")
    if steps <= 0:
        raise ValueError("steps must be positive")
    schedule = _native_suzuki_schedule(term_count, order)

    @qm.qkernel
    def evolve(
        state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
        hamiltonians: qm.Const[  # type: ignore[valid-type]
            qm.Vector[qm.Observable, term_count]
        ],
        time: qm.Param[qm.Real],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Apply the configured product-formula evolution.

        Args:
            state (qm.Q[qm.Bits[width]]): Owned state to evolve.
            hamiltonians (qm.Const[qm.Vector[qm.Observable, term_count]]):
                Compile-time sequence of Hamiltonian terms.
            time (qm.Param[qm.Real]): Total runtime evolution time.

        Returns:
            qm.Q[qm.Bits[width]]: Evolved state.
        """

        dt = time / steps
        for _ in range(steps):
            for operation in schedule:
                state = qm.pauli_evolve(
                    state,
                    hamiltonians[operation[0]],
                    dt * operation[1],
                )
        return state

    return evolve


def _require_integer(name: str, value: object) -> None:
    """Require one non-Boolean host integer option."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")


def _native_suzuki_schedule(term_count: int, order: int) -> _Schedule:
    """Cross the required native product-formula schedule boundary."""

    derive = getattr(native, "suzuki_product_formula_schedule", None)
    if derive is None:
        raise TypeError("Q7520: native Suzuki schedule synthesis is unavailable")
    try:
        projection = derive(term_count, order)
    except OverflowError as error:
        raise ValueError("Q7523: Suzuki schedule exceeds the entry budget") from error
    if not isinstance(projection, list) or not projection:
        raise TypeError("Q7525: malformed native Suzuki schedule")
    schedule: list[tuple[int, float]] = []
    for row in projection:
        if (
            not isinstance(row, tuple)
            or len(row) != 2
            or isinstance(row[0], bool)
            or not isinstance(row[0], int)
            or not 0 <= row[0] < term_count
            or not isinstance(row[1], float)
            or not math.isfinite(row[1])
        ):
            raise TypeError("Q7525: malformed native Suzuki schedule")
        schedule.append(row)
    return tuple(schedule)


__all__ = ["suzuki_trotter"]
