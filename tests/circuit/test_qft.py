"""Tests for the QKernel-backed QFT and inverse-QFT composites."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.frontend.qkernel import QKernel
from qamomile.circuit.ir.operation.callable import CompositeGateType, InvokeOperation


@qmc.qkernel
def qft_round_trip(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Apply QFT and IQFT to a computational-basis register."""
    qubits = qmc.qubit_array(n, "q")
    qubits[0] = qmc.x(qubits[0])
    qubits = qmc.qft(qubits)
    qubits = qmc.iqft(qubits)
    return qmc.measure(qubits)


def test_qft_and_iqft_are_composite_qkernels() -> None:
    """The stdlib exposes no parallel QFT class hierarchy."""
    assert isinstance(qmc.qft, QKernel)
    assert isinstance(qmc.iqft, QKernel)
    assert qmc.qft._callable_gate_type is CompositeGateType.QFT
    assert qmc.iqft._callable_gate_type is CompositeGateType.IQFT


@pytest.mark.parametrize(
    ("kernel", "gate_type"),
    [
        (qmc.qft, CompositeGateType.QFT),
        (qmc.iqft, CompositeGateType.IQFT),
    ],
)
def test_qft_call_stays_named(kernel: QKernel, gate_type: CompositeGateType) -> None:
    """QFT calls carry a body and their native backend classification."""

    @qmc.qkernel
    def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
        """Call one Fourier-transform composite."""
        return kernel(qmc.qubit_array(n, "q"))

    invokes = [op for op in circuit.block.operations if isinstance(op, InvokeOperation)]
    assert len(invokes) == 1
    assert invokes[0].gate_type is gate_type
    assert invokes[0].effective_body() is kernel.block


@pytest.mark.parametrize("n_value", [1, 2, 3, 5])
def test_qft_resource_formula(n_value: int) -> None:
    """QFT resource estimation is available through its normal body-backed call."""

    @qmc.qkernel
    def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
        """Apply QFT to an n-qubit register."""
        return qmc.qft(qmc.qubit_array(n, "q"))

    symbolic = circuit.estimate_resources()
    n = sp.Symbol("n", integer=True, positive=True)
    expected = n * (n + 1) / 2 + sp.floor(n / 2)
    assert sp.simplify(symbolic.gates.total - expected) == 0

    concrete = circuit.estimate_resources(substitutions={"n": n_value})
    assert concrete.gates.total == n_value * (n_value + 1) // 2 + n_value // 2
    assert concrete.qubits == n_value


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_qft_round_trip_qiskit(n: int) -> None:
    """QFT followed by IQFT preserves a basis state on a real backend."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(qft_round_trip, bindings={"n": n})
    result = executable.sample(transpiler.executor(), shots=128).result()

    assert result.results == [((1,) + (0,) * (n - 1), 128)]


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_qft_inverse_statevector(seed: int) -> None:
    """Random product-state expvals survive a QFT/inverse-QFT round trip."""
    pytest.importorskip("qiskit")
    import qamomile.observable as qm_o
    from qamomile.qiskit import QiskitTranspiler

    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2 * np.pi, size=3)

    @qmc.qkernel
    def circuit(observable: qmc.Observable) -> qmc.Float:
        """Prepare a random product state and Fourier round trip it."""
        qubits = qmc.qubit_array(3, "q")
        qubits[0] = qmc.ry(qubits[0], float(angles[0]))
        qubits[1] = qmc.ry(qubits[1], float(angles[1]))
        qubits[2] = qmc.ry(qubits[2], float(angles[2]))
        qubits = qmc.qft(qubits)
        qubits = qmc.iqft(qubits)
        return qmc.expval(qubits, observable)

    transpiler = QiskitTranspiler()
    observable = qm_o.Z(0)
    actual_executable = transpiler.transpile(
        circuit,
        bindings={"observable": observable},
    )
    actual = actual_executable.run(transpiler.executor()).result()

    @qmc.qkernel
    def reference(observable: qmc.Observable) -> qmc.Float:
        """Prepare the same random product state without Fourier transforms."""
        qubits = qmc.qubit_array(3, "q")
        qubits[0] = qmc.ry(qubits[0], float(angles[0]))
        qubits[1] = qmc.ry(qubits[1], float(angles[1]))
        qubits[2] = qmc.ry(qubits[2], float(angles[2]))
        return qmc.expval(qubits, observable)

    reference_executable = transpiler.transpile(
        reference,
        bindings={"observable": observable},
    )
    expected = reference_executable.run(transpiler.executor()).result()
    assert np.isclose(actual, expected, atol=1e-8)
