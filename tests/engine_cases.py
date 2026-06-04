"""Shared engine matrix for cross-SDK semantic tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest


@dataclass(frozen=True)
class EngineCase:
    """Represent one SDK engine used by engine-agnostic tests."""

    name: str
    make_transpiler: Callable[[], Any]
    run_statevector: Callable[[Any], np.ndarray]
    expval_atol: float = 1e-8
    statevector_atol: float = 1e-8

    def transpiler(self) -> Any:
        """Return a fresh transpiler instance for this engine."""
        return self.make_transpiler()

    def statevector(
        self,
        kernel: Any,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> np.ndarray:
        """Transpile a fully-bound measurement-free kernel and return its statevector."""
        if parameters:
            raise AssertionError(
                "EngineCase.statevector() expects compile-time bindings only; "
                f"got runtime parameters for {self.name}: {parameters}"
            )
        exe = self.transpiler().transpile(
            kernel,
            bindings=bindings,
            parameters=parameters,
        )
        circuit = exe.compiled_quantum[0].circuit
        return self.run_statevector(circuit)


def _qiskit_transpiler() -> Any:
    """Return a Qiskit transpiler or skip when Qiskit is unavailable."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    return QiskitTranspiler()


def _quri_parts_transpiler() -> Any:
    """Return a QURI Parts transpiler or skip when QURI Parts is unavailable."""
    pytest.importorskip("quri_parts.qulacs")
    from qamomile.quri_parts import QuriPartsTranspiler

    return QuriPartsTranspiler()


def _cudaq_transpiler() -> Any:
    """Return a CUDA-Q transpiler or skip when CUDA-Q is unavailable."""
    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqTranspiler

    return CudaqTranspiler()


def _qiskit_statevector(circuit: Any) -> np.ndarray:
    """Run a measurement-free Qiskit circuit and return its statevector."""
    pytest.importorskip("qiskit_aer")
    from qiskit_aer import AerSimulator

    if circuit.parameters:
        raise AssertionError(
            "Qiskit statevector helper expects a fully-bound circuit; "
            f"unbound parameters remain: {sorted(str(p) for p in circuit.parameters)}"
        )

    statevector_circuit = circuit.copy()
    statevector_circuit.save_statevector()
    result = AerSimulator(method="statevector").run(statevector_circuit).result()
    return np.array(result.get_statevector())


def _quri_parts_statevector(circuit: Any) -> np.ndarray:
    """Run a fully-bound QURI Parts circuit and return its statevector."""
    pytest.importorskip("quri_parts.qulacs")
    from quri_parts.core.state import GeneralCircuitQuantumState
    from quri_parts.qulacs.simulator import evaluate_state_to_vector

    if hasattr(circuit, "parameter_count") and circuit.parameter_count > 0:
        raise AssertionError(
            "QURI Parts statevector helper expects a fully-bound circuit; "
            f"{circuit.parameter_count} unbound parameters remain"
        )
    elif hasattr(circuit, "bind_parameters"):
        bound_circuit = circuit.bind_parameters([])
    else:
        bound_circuit = circuit

    circuit_state = GeneralCircuitQuantumState(
        bound_circuit.qubit_count,
        bound_circuit,
    )
    statevector = evaluate_state_to_vector(circuit_state)
    return np.array(statevector.vector)


def _cudaq_statevector(circuit: Any) -> np.ndarray:
    """Run a fully-bound CUDA-Q kernel artifact and return its statevector."""
    cudaq = pytest.importorskip("cudaq")
    if getattr(circuit, "param_count", 0):
        raise AssertionError(
            "CUDA-Q statevector helper expects a fully-bound kernel artifact; "
            f"{circuit.param_count} unbound parameters remain"
        )
    return np.array(cudaq.get_state(circuit.kernel_func))


QISKIT_ENGINE = EngineCase("qiskit", _qiskit_transpiler, _qiskit_statevector)
QURI_PARTS_ENGINE = EngineCase(
    "quri_parts",
    _quri_parts_transpiler,
    _quri_parts_statevector,
)
CUDAQ_ENGINE = EngineCase(
    "cudaq",
    _cudaq_transpiler,
    _cudaq_statevector,
    expval_atol=1e-6,
    statevector_atol=1e-7,
)

SDK_ENGINES = (
    pytest.param(QISKIT_ENGINE, id=QISKIT_ENGINE.name),
    pytest.param(
        QURI_PARTS_ENGINE,
        id=QURI_PARTS_ENGINE.name,
        marks=pytest.mark.quri_parts,
    ),
    pytest.param(CUDAQ_ENGINE, id=CUDAQ_ENGINE.name, marks=pytest.mark.cudaq),
)
