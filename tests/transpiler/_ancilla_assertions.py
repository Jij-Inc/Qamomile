"""Shared assertion for multi-control clean-ancilla uncomputation in tests.

Backends without a native multi-controlled primitive lower an ``n``-controlled
gate through a Toffoli cascade that borrows clean ancilla qubits appended after
the data qubits and uncomputes them before the circuit ends. Several
statevector tests need to assert those ancillas returned to ``|0>`` and then
compare only the data-qubit amplitudes against a reference. Centralizing the
assertion here keeps the numerical tolerance and the projection boundary from
drifting between test modules that each express the projection in their own
natural unit (a raw dimension versus a data-qubit count).
"""

from __future__ import annotations

import numpy as np


def assert_ancillas_uncomputed(statevector: np.ndarray, dim: int) -> np.ndarray:
    """Assert amplitudes past ``dim`` are ~0 and return the leading ``dim``.

    Args:
        statevector (np.ndarray): Full statevector including any trailing
            clean-ancilla qubits appended after the data qubits.
        dim (int): Number of leading amplitudes to keep — the data-qubit
            dimension (``2**num_data_qubits`` in the little-endian
            convention). Every amplitude at or beyond this index carries an
            ancilla bit set and is asserted to be zero. Must not exceed the
            statevector length; ``dim == len(statevector)`` is the valid
            ancilla-free case (no trailing amplitudes to check).

    Returns:
        np.ndarray: The statevector restricted to its leading ``dim``
            amplitudes (the data qubits).

    Raises:
        AssertionError: If ``dim`` exceeds the statevector length — a
            caller passing a wrong projection would otherwise make the
            zero-ancilla check pass vacuously, since ``statevector[dim:]``
            is empty and ``np.allclose`` returns True for it — or if any
            amplitude at or beyond ``dim`` is not numerically zero, i.e. an
            ancilla did not uncompute back to ``|0>``.
    """
    assert dim <= statevector.shape[0], (
        f"projection dim={dim} exceeds statevector length "
        f"{statevector.shape[0]}; the zero-ancilla check would pass "
        "vacuously (the caller passed a wrong dimension)"
    )
    assert np.allclose(statevector[dim:], 0.0), (
        "multi-control ancilla qubits did not uncompute back to |0>"
    )
    return statevector[:dim]
