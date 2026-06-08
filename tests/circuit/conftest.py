"""Shared test utilities and fixtures for circuit tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


_SIMULATOR_SEED = 901


@dataclass(frozen=True)
class SdkTranspilerCase:
    """Bundle a backend label with its transpiler instance."""

    backend_name: str
    transpiler: Any


@pytest.fixture
def qiskit_transpiler():
    """Get Qiskit transpiler."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    return QiskitTranspiler()


@pytest.fixture(
    params=[
        pytest.param("qiskit", id="qiskit"),
        pytest.param("quri_parts", marks=pytest.mark.quri_parts, id="quri_parts"),
        pytest.param("cudaq", marks=pytest.mark.cudaq, id="cudaq"),
    ]
)
def sdk_transpiler(request):
    """Return a supported SDK transpiler or skip when unavailable."""
    backend = request.param
    if backend == "qiskit":
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return SdkTranspilerCase(backend, QiskitTranspiler())
    if backend == "quri_parts":
        pytest.importorskip("quri_parts.qulacs")
        from qamomile.quri_parts import QuriPartsTranspiler

        return SdkTranspilerCase(backend, QuriPartsTranspiler())
    if backend == "cudaq":
        pytest.importorskip("cudaq")
        from qamomile.cudaq import CudaqTranspiler

        return SdkTranspilerCase(backend, CudaqTranspiler())
    raise AssertionError(f"Unsupported SDK backend fixture value: {backend}")


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
