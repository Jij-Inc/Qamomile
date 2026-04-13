"""CUDA-Q observable support.

This module provides conversion from qamomile.observable.Hamiltonian
to CUDA-Q SpinOperator for use with ``cudaq.observe``.
"""

from __future__ import annotations

import math
from typing import Any

import qamomile.observable as qm_o


def hamiltonian_to_cudaq_spin_op(hamiltonian: qm_o.Hamiltonian) -> Any:
    """Convert qamomile.observable.Hamiltonian to cudaq.SpinOperator.

    Args:
        hamiltonian: The qamomile Hamiltonian to convert.

    Returns:
        A CUDA-Q SpinOperator built from ``cudaq.spin`` primitives.

    Example:
        ```python
        import qamomile.observable as qm_o
        from qamomile.cudaq.observable import hamiltonian_to_cudaq_spin_op

        H = qm_o.Z(0) * qm_o.Z(1) + 0.5 * (qm_o.X(0) + qm_o.X(1))
        spin_op = hamiltonian_to_cudaq_spin_op(H)
        ```
    """
    from cudaq import spin

    pauli_fn_map = {
        qm_o.Pauli.X: spin.x,  # type: ignore[attr-defined]
        qm_o.Pauli.Y: spin.y,  # type: ignore[attr-defined]
        qm_o.Pauli.Z: spin.z,  # type: ignore[attr-defined]
    }

    result = None

    # Add constant term
    if not math.isclose(hamiltonian.constant, 0.0, abs_tol=1e-15):  # type: ignore[arg-type]
        result = hamiltonian.constant * spin.i(0)  # type: ignore[attr-defined]

    # Convert each term (skip near-zero coefficients to avoid noise)
    for operators, coeff in hamiltonian.terms.items():
        if math.isclose(coeff, 0.0, abs_tol=1e-15):  # type: ignore[arg-type]
            continue
        term = None
        for op in operators:
            if op.pauli == qm_o.Pauli.I:
                continue
            pauli_op = pauli_fn_map[op.pauli](op.index)
            if term is None:
                term = pauli_op
            else:
                term = term * pauli_op

        if term is None:
            # All-identity term
            scaled = coeff * spin.i(0)  # type: ignore[attr-defined]
        else:
            scaled = coeff * term

        if result is None:
            result = scaled
        else:
            result = result + scaled

    if result is None:
        return 0.0 * spin.i(0)  # type: ignore[attr-defined]

    return result
