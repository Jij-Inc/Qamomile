"""Shared test utilities and fixtures for circuit tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


@pytest.fixture
def qiskit_transpiler():
    """Get Qiskit transpiler."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    return QiskitTranspiler()


@pytest.fixture
def seeded_executor(qiskit_transpiler):
    """Executor with fixed seed for reproducible sampling."""
    from qiskit_aer import AerSimulator

    return qiskit_transpiler.executor(backend=AerSimulator(seed_simulator=901))


def run_statevector(qc: "QuantumCircuit") -> np.ndarray:
    """Run circuit and return statevector (measurements removed).

    Args:
        qc: Qiskit QuantumCircuit to simulate.

    Returns:
        numpy array of complex amplitudes.
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    qc.remove_final_measurements()
    simulator = AerSimulator(method="statevector")
    qc = transpile(qc, simulator)
    qc.save_statevector()
    result = simulator.run(qc).result()
    return np.array(result.get_statevector())
