"""Convert Qamomile Hamiltonians to Amazon Braket observables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import qamomile.observable as qm_o
from qamomile.observable.hamiltonian import (
    HERMITIAN_IMAG_ATOL,
    PAULI_TERM_ZERO_ATOL,
)

if TYPE_CHECKING:
    from braket.circuits import Observable  # type: ignore[import-not-found]


def hamiltonian_to_braket_observable(
    hamiltonian: qm_o.Hamiltonian,
) -> "Observable":
    """Convert a Qamomile Hamiltonian to a Braket observable.

    A scalar-only Hamiltonian is represented on qubit zero because Braket
    observables always address at least one qubit. The executor evaluates such
    Hamiltonians locally and does not add that synthetic qubit to a task.

    Args:
        hamiltonian (qm_o.Hamiltonian): Hamiltonian to convert.

    Returns:
        Observable: Native Braket observable, including all coefficients.

    Raises:
        ValueError: If a coefficient has a non-negligible imaginary part.
    """
    from braket.circuits import Observable  # type: ignore[import-not-found]

    observable_api = cast(Any, Observable)

    summands: list[Any] = []
    if abs(complex(hamiltonian.constant).imag) > HERMITIAN_IMAG_ATOL:
        raise ValueError("Braket expectation observables require real coefficients")
    if abs(hamiltonian.constant) > PAULI_TERM_ZERO_ATOL:
        summands.append(float(complex(hamiltonian.constant).real) * observable_api.I(0))

    factories = {
        qm_o.Pauli.I: observable_api.I,
        qm_o.Pauli.X: observable_api.X,
        qm_o.Pauli.Y: observable_api.Y,
        qm_o.Pauli.Z: observable_api.Z,
    }
    for operators, coefficient in hamiltonian.terms.items():
        value = complex(coefficient)
        if abs(value.imag) > HERMITIAN_IMAG_ATOL:
            raise ValueError("Braket expectation observables require real coefficients")
        factors = [factories[item.pauli](item.index) for item in operators]
        if not factors:
            continue
        term = factors[0]
        for factor in factors[1:]:
            term = term @ factor
        summands.append(float(value.real) * term)

    if not summands:
        return 0.0 * observable_api.I(0)
    if len(summands) == 1:
        return summands[0]
    return observable_api.Sum(summands)
