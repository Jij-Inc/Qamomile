"""Bounded cross-target execution of exact modular realization optimization."""

from importlib import import_module

import pytest

import qamomile as qm
from qamomile.circuit import lower_to_circuit
from qamomile.persistence import deserialize, serialize


@qm.qkernel
def modular_interference_state() -> qm.Q[qm.Bits[4]]:
    """Undo modular power with an independent controlled swap on complex input."""
    state = qm.h(qm.qubits(4))
    state = qm.s(state, target=0)
    state = qm.t(state, target=1)
    state = qm.s(state, target=3)
    state = qm.cx(state, control=0, target=3)
    target, exponent = qm.split_at(state, 2)
    target, exponent = qm.mod_pow(
        qm.as_uint(target), qm.as_uint(exponent), base=2, modulus=3
    )
    state = qm.concat(qm.as_bits(target), qm.as_bits(exponent))
    state = qm.cx(state, control=1, target=0)
    state = qm.ccx(state, control1=2, control2=0, target=1)
    state = qm.cx(state, control=1, target=0)
    state = qm.cx(state, control=0, target=3)
    state = qm.sdg(state, target=3)
    state = qm.tdg(state, target=1)
    state = qm.sdg(state, target=0)
    return qm.h(state)


@qm.qkernel
def modular_interference_sample() -> qm.Bits[4]:
    """Measure phase-sensitive agreement on valid and invalid target residues."""
    return qm.measure(modular_interference_state())


def test_small_modular_power_retains_no_arithmetic_scratch() -> None:
    """The second identity factor disappears and the emitted plan stays small."""
    circuit = lower_to_circuit(modular_interference_sample.specialize())
    assert circuit.width == 4
    assert circuit.operation_count <= 40
    assert set(circuit.realizations) == {
        "arithmetic.portable",
        "mod_pow.square_and_multiply",
    }


@pytest.mark.parametrize(
    "backend,dependency,endpoint_name",
    [
        ("qiskit", "qiskit_aer", "Aer"),
        pytest.param("cirq", "cirq", "CirqSimulator", marks=pytest.mark.cirq),
        pytest.param("quri_parts", "qulacs", "Qulacs", marks=pytest.mark.quri_parts),
        pytest.param(
            "braket", "braket.circuits", "BraketLocal", marks=pytest.mark.braket
        ),
        pytest.param("cudaq", "cudaq", "CudaqLocal", marks=pytest.mark.cudaq),
        pytest.param("hugr", "selene_sim", "Selene", marks=pytest.mark.hugr),
        pytest.param("quration", "pyqret", "QurationLocal", marks=pytest.mark.quration),
    ],
)
def test_modular_phase_and_result_abi_across_targets(
    backend: str, dependency: str, endpoint_name: str
) -> None:
    """Compile, round-trip and execute phase-sensitive sampling and expectation."""
    pytest.importorskip(dependency)
    endpoint = getattr(import_module(f"qamomile.{backend}"), endpoint_name)()
    artifact = qm.compile(modular_interference_sample, to=endpoint.target)
    assert "realization:arithmetic.portable" in artifact.report.stages
    restored = deserialize(serialize(artifact))
    assert qm.execute(restored, on=endpoint, shots=4, seed=0).counts == {
        (0, 0, 0, 0): 4
    }
    assert qm.expectation(
        modular_interference_state,
        qm.Z(0) + qm.Z(1) + qm.Z(2) + qm.Z(3),
        on=endpoint,
        shots=4,
        seed=0,
    ).value == pytest.approx(4.0)
