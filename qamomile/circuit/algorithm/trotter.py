"""Trotter circuit implementation.

This module provides :func:`trotter_evolve` and :func:`trotter_state`
``@qkernel`` functions, along with the :class:`TrotterCircuit` helper
class that pre-computes the decomposition sequence.

Supported orders:

* ``order=1`` — first-order Lie-Trotter decomposition
* ``order=2,4,6,...`` — even-order Suzuki decomposition (recursive)

Example::

    import qamomile.circuit as qmc
    import qamomile.observable as qm_o
    from qamomile.circuit.algorithm.trotter import TrotterCircuit

    H = qm_o.Z(0) * qm_o.Z(1) + 0.5 * qm_o.X(0) + 0.5 * qm_o.X(1)
    trotter = TrotterCircuit(hamiltonian=H, time=1.0, step=2, order=2)

    @qmc.qkernel
    def my_circuit(
        n: qmc.UInt,
        time: qmc.Float,
        H_0: qmc.Observable,
        H_1: qmc.Observable,
        H_2: qmc.Observable,
    ) -> qmc.Vector[qmc.Qubit]:
        q = qmc.qubit_array(n, name="q")
        q = trotter.evolve(q, [H_0, H_1, H_2], time)
        return q
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qamomile.circuit.frontend.operation.pauli_evolve import pauli_evolve
from qamomile.observable.hamiltonian import Hamiltonian

if TYPE_CHECKING:
    from qamomile.circuit.frontend.handle import Float, Observable, Qubit, Vector


class TrotterCircuit:
    """Suzuki-Trotter decomposition circuit for Hamiltonian time evolution.

    Pre-computes the product-formula decomposition and provides
    :meth:`evolve` to apply it inside a ``@qkernel`` context.

    Args:
        hamiltonian: Hamiltonian ``H = sum_i H_i`` with at least 2 terms.
        time: Total evolution time *t* (non-negative).
        step: Number of Trotter steps *n* (positive integer).
        order: Approximation order — ``1`` (Lie-Trotter) or a positive
            even integer (Suzuki).

    Raises:
        ValueError: If any input violates the constraints above.
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        time: float,
        step: int,
        order: int,
    ) -> None:
        _validate_inputs(hamiltonian, time, step, order)
        self._hamiltonian = hamiltonian
        self._time = time
        self._step = step
        self._order = order
        self._sub_hamiltonians = _split_hamiltonian(hamiltonian)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def time(self) -> float:
        """Total evolution time."""
        return self._time

    @property
    def step(self) -> int:
        """Number of Trotter steps."""
        return self._step

    @property
    def order(self) -> int:
        """Approximation order."""
        return self._order

    @property
    def num_qubits(self) -> int:
        """Number of qubits required by the Hamiltonian."""
        return self._hamiltonian.num_qubits

    @property
    def sub_hamiltonians(self) -> list[Hamiltonian]:
        """Individual terms of the Hamiltonian as single-term Hamiltonians."""
        return list(self._sub_hamiltonians)

    @property
    def num_terms(self) -> int:
        """Number of terms in the Hamiltonian."""
        return len(self._sub_hamiltonians)

    @property
    def sequence(self) -> list[tuple[int, float]]:
        """Full decomposition sequence with absolute time coefficients.

        Returns a list of ``(term_index, time_coefficient)`` pairs.
        Applying ``exp(-i * time_coefficient * H_{term_index})`` in order
        approximates ``exp(-i H t)``.
        """
        return [
            (idx, frac * self._time)
            for idx, frac in _full_sequence(
                len(self._sub_hamiltonians), self._order, self._step
            )
        ]

    # ------------------------------------------------------------------
    # Circuit building — call inside @qkernel
    # ------------------------------------------------------------------

    def evolve(
        self,
        q: Vector[Qubit],
        observables: list[Observable],
        time: Float,
    ) -> Vector[Qubit]:
        """Apply the Trotter decomposition to a qubit register.

        This is a regular function (not a ``@qkernel``) that operates on
        Handle types.  Call it from within a ``@qkernel`` function, in
        the same way as :func:`qamomile.circuit.pauli_evolve`.

        Args:
            q: Qubit register handle.
            observables: Observable handles, one per sub-Hamiltonian term
                (length must equal :attr:`num_terms`).
            time: Evolution time as a ``Float`` handle.

        Returns:
            The evolved qubit register.

        Example::

            trotter = TrotterCircuit(H, time=1.0, step=2, order=2)

            @qmc.qkernel
            def circuit(
                n: qmc.UInt,
                t: qmc.Float,
                H_0: qmc.Observable,
                H_1: qmc.Observable,
            ) -> qmc.Vector[qmc.Qubit]:
                q = qmc.qubit_array(n, name="q")
                q = trotter.evolve(q, [H_0, H_1], t)
                return q

            bindings = {
                "n": H.num_qubits,
                "t": 1.0,
                "H_0": trotter.sub_hamiltonians[0],
                "H_1": trotter.sub_hamiltonians[1],
            }
        """
        seq = _full_sequence(
            len(self._sub_hamiltonians), self._order, self._step
        )
        for term_idx, frac in seq:
            q = pauli_evolve(q, observables[term_idx], frac * time)
        return q

    def bindings(self, time_key: str = "time") -> dict:
        """Return bindings dict for the sub-Hamiltonians and time.

        The keys are ``"H_0"``, ``"H_1"``, …, ``"H_{m-1}"`` for each
        sub-Hamiltonian term, plus *time_key* for the evolution time.

        Args:
            time_key: Name of the time binding (default ``"time"``).

        Returns:
            A dict suitable for passing to ``transpiler.transpile(..., bindings=...)``.
        """
        b: dict = {time_key: self._time}
        for i, h in enumerate(self._sub_hamiltonians):
            b[f"H_{i}"] = h
        return b


# ======================================================================
# Module-level helpers (pure functions, no circuit side-effects)
# ======================================================================


def _validate_inputs(
    hamiltonian: Hamiltonian,
    time: float,
    step: int,
    order: int,
) -> None:
    """Validate constructor arguments."""
    if not isinstance(step, int) or step <= 0:
        raise ValueError(f"step must be a positive integer, got {step}")
    if time < 0:
        raise ValueError(f"time must be non-negative, got {time}")
    if order != 1 and (order <= 0 or order % 2 != 0):
        raise ValueError(
            f"order must be 1 or a positive even integer, got {order}"
        )
    if len(hamiltonian) < 2:
        raise ValueError(
            f"Hamiltonian must have at least 2 terms for Trotter "
            f"decomposition, got {len(hamiltonian)}. "
            f"For a single-term Hamiltonian, use pauli_evolve directly."
        )


def _split_hamiltonian(hamiltonian: Hamiltonian) -> list[Hamiltonian]:
    """Split a Hamiltonian into individual single-term Hamiltonians."""
    terms: list[Hamiltonian] = []
    for ops, coeff in hamiltonian:
        h = Hamiltonian(num_qubits=hamiltonian.num_qubits)
        h.add_term(ops, coeff)
        terms.append(h)
    return terms


def _product_formula(
    n_terms: int,
    order: int,
    dt_frac: float,
) -> list[tuple[int, float]]:
    """Compute decomposition sequence for one Trotter step.

    All time coefficients are expressed as fractions of the total
    evolution time.

    Args:
        n_terms: Number of Hamiltonian terms (m).
        order: Approximation order.
        dt_frac: Time fraction for this step (``1/step`` at top level).

    Returns:
        List of ``(term_index, time_fraction)`` pairs.
    """
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

    inner = _product_formula(n_terms, order - 2, p_k * dt_frac)
    center = _product_formula(n_terms, order - 2, (1.0 - 4.0 * p_k) * dt_frac)

    return inner + inner + center + inner + inner


def _full_sequence(
    n_terms: int,
    order: int,
    step: int,
) -> list[tuple[int, float]]:
    """Compute the full Trotter sequence for all steps.

    Returns ``(term_index, time_fraction)`` pairs where each
    ``time_fraction`` is relative to the total evolution time.
    """
    dt_frac = 1.0 / step
    one_step = _product_formula(n_terms, order, dt_frac)
    return one_step * step
