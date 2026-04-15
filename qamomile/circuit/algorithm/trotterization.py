"""Trotterized time evolution for Hamiltonian simulation.

This module provides :func:`trotterized_time_evolution`, a function that
applies Suzuki-Trotter decomposed time evolution to a qubit register
inside a ``@qkernel`` context.

The caller is responsible for splitting the Hamiltonian into a list of
sub-Hamiltonians; this module handles only the product-formula ordering
and coefficient computation.

Supported orders:

* ``order=1`` — first-order Lie-Trotter decomposition
* ``order=2,4,6,...`` — even-order Suzuki decomposition (recursive)

Example::

    import qamomile.circuit as qmc
    from qamomile.circuit.algorithm.trotterization import (
        trotterized_time_evolution,
    )

    @qmc.qkernel
    def my_circuit(
        n: qmc.UInt,
        t: qmc.Float,
        H_0: qmc.Observable,
        H_1: qmc.Observable,
        H_2: qmc.Observable,
    ) -> qmc.Vector[qmc.Qubit]:
        q = qmc.qubit_array(n, name="q")
        q = trotterized_time_evolution(q, [H_0, H_1, H_2], t, step=2, order=2)
        return q
"""

from __future__ import annotations

from collections.abc import Iterator

import qamomile.circuit as qmc
from qamomile.circuit.frontend.operation.pauli_evolve import pauli_evolve


def trotterized_time_evolution(
    q: qmc.Vector[qmc.Qubit],
    observables: list[qmc.Observable],
    time: qmc.Float,
    step: int,
    order: int,
) -> qmc.Vector[qmc.Qubit]:
    """Apply Suzuki-Trotter decomposed time evolution to a qubit register.

    This is a regular function (not a ``@qkernel``) that operates on
    Handle types.  Call it from within a ``@qkernel`` function, in
    the same way as :func:`qamomile.circuit.pauli_evolve`.

    The caller must supply pre-decomposed sub-Hamiltonians as
    ``observables``.  This function does **not** split a Hamiltonian
    internally.

    Args:
        q: Qubit register handle.
        observables: Observable handles, one per sub-Hamiltonian term.
            Must contain at least 2 elements.
        time: Evolution time as a ``Float`` handle.
        step: Number of Trotter steps (positive integer).
        order: Approximation order — ``1`` (Lie-Trotter) or a positive
            even integer (Suzuki).

    Returns:
        The evolved qubit register.

    Raises:
        ValueError: If ``step``, ``order``, or ``observables`` violate
            the constraints.
    """
    _validate_inputs(len(observables), step, order)
    seq = _full_sequence(len(observables), step, order)
    for term_idx, frac in seq:
        q = pauli_evolve(q, observables[term_idx], frac * time)
    return q


# ======================================================================
# Subroutines (pure functions, no circuit side-effects)
# ======================================================================


def _validate_order(order: int) -> None:
    """Validate the approximation order."""
    if isinstance(order, bool) or not isinstance(order, int):
        raise ValueError(f"order must be 1 or a positive even integer, got {order}")
    if order != 1 and (order <= 0 or order % 2 != 0):
        raise ValueError(f"order must be 1 or a positive even integer, got {order}")


def _validate_n_terms(n_terms: int) -> None:
    """Validate the number of Hamiltonian terms."""
    if n_terms < 2:
        raise ValueError(
            f"n_terms must be at least 2 for Trotter "
            f"decomposition, got {n_terms}. "
            f"For a single-term Hamiltonian, use pauli_evolve directly."
        )


def _validate_inputs(n_terms: int, step: int, order: int) -> None:
    """Validate arguments for trotterized time evolution."""
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError(f"step must be a positive integer, got {step}")
    _validate_order(order)
    _validate_n_terms(n_terms)


def product_formula(
    n_terms: int,
    dt_frac: float,
    order: int,
) -> list[tuple[int, float]]:
    """Compute decomposition sequence for one Trotter step.

    All time coefficients are expressed as fractions of the total
    evolution time.

    Args:
        n_terms: Number of Hamiltonian terms (m).  Must be >= 2.
        dt_frac: Time fraction for this step (``1/step`` at top level).
        order: Approximation order — ``1`` (Lie-Trotter) or a positive
            even integer (Suzuki).

    Returns:
        List of ``(term_index, time_fraction)`` pairs.

    Raises:
        ValueError: If ``n_terms`` < 2 or ``order`` is not 1 or a positive
            even integer.
    """
    _validate_n_terms(n_terms)
    _validate_order(order)
    if order == 1:
        # Lie-Trotter: prod_{i=1}^{m} exp(-i dt H_i)
        return [(i, dt_frac) for i in range(n_terms)]

    if order == 2:
        # Suzuki 2nd order:
        #   prod_{i=1}^{m} exp(-i dt/2 H_i)
        #   * prod_{i=m}^{1} exp(-i dt/2 H_i)
        half = dt_frac / 2
        forward = [(i, half) for i in range(n_terms)]
        backward = [(i, half) for i in range(n_terms - 1, -1, -1)]
        return forward + backward

    # Recursive even-order: order = 2k, k >= 2
    k = order // 2
    p_k = 1.0 / (4.0 - 4.0 ** (1.0 / (2 * k - 1)))

    inner = product_formula(n_terms, p_k * dt_frac, order - 2)
    center = product_formula(n_terms, (1.0 - 4.0 * p_k) * dt_frac, order - 2)

    return inner + inner + center + inner + inner


def _full_sequence(
    n_terms: int,
    step: int,
    order: int,
) -> Iterator[tuple[int, float]]:
    """Iterate over the full Trotter sequence for all steps.

    Yields ``(term_index, time_fraction)`` pairs where each
    ``time_fraction`` is relative to the total evolution time.
    """
    dt_frac = 1.0 / step
    one_step = product_formula(n_terms, dt_frac, order)
    for _ in range(step):
        yield from one_step
