"""Shared test utilities and fixtures for circuit tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


_SIMULATOR_SEED = 901


@pytest.fixture
def qiskit_transpiler():
    """Get Qiskit transpiler."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    return QiskitTranspiler()


@pytest.fixture
def seeded_executor(qiskit_transpiler):
    """Executor with fixed seed for reproducible sampling."""
    from qiskit.providers.basic_provider import BasicSimulator

    backend = BasicSimulator()
    backend.set_options(seed_simulator=_SIMULATOR_SEED)
    return qiskit_transpiler.executor(backend=backend)


def run_statevector(qc: "QuantumCircuit") -> np.ndarray:
    """Run circuit and return statevector (measurements removed).

    Args:
        qc: Qiskit QuantumCircuit to simulate.

    Returns:
        numpy array of complex amplitudes.
    """
    from qiskit.quantum_info import Statevector

    qc = qc.remove_final_measurements(inplace=False)
    return np.array(Statevector.from_instruction(qc).data)
