"""Controlled QURI emission preserves phases without dense control matrices."""

from __future__ import annotations

import math

import numpy as np
import pytest

import qamomile as qm
from qamomile.circuit import CircuitInstruction, ParameterValue
from qamomile.persistence import deserialize, serialize

pytestmark = pytest.mark.quri_parts
pytest.importorskip("quri_parts.qulacs")


def _base_unitary(opcode: str, angle: float) -> np.ndarray:
    """Build a small independent mathematical oracle in little-endian order."""
    x = np.array([[0, 1], [1, 0]], complex)
    y = np.array([[0, -1j], [1j, 0]], complex)
    z = np.diag([1, -1]).astype(complex)
    fixed = {"x": x, "y": y, "z": z, "h": (x + z) / np.sqrt(2)}
    if opcode in fixed:
        return fixed[opcode]
    if opcode in {"s", "sdg", "t", "tdg", "p"}:
        phase = {
            "s": math.pi / 2,
            "sdg": -math.pi / 2,
            "t": math.pi / 4,
            "tdg": -math.pi / 4,
            "p": angle,
        }[opcode]
        return np.diag([1, np.exp(1j * phase)])
    if opcode in {"rx", "ry", "rz"}:
        return (
            np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * fixed[opcode[1:]]
        )
    if opcode == "global_phase":
        return np.array([[np.exp(1j * angle)]])
    if opcode == "rzz":
        return np.diag(np.exp(-0.5j * angle * np.array([1, -1, -1, 1])))
    if opcode == "cp":
        return np.diag([1, 1, 1, np.exp(1j * angle)])
    if opcode == "cz":
        return np.diag([1, 1, 1, -1])
    permutation = {
        "cx": [0, 3, 2, 1],
        "ccx": [0, 1, 2, 7, 4, 5, 6, 3],
        "swap": [0, 2, 1, 3],
    }[opcode]
    return np.eye(len(permutation))[:, permutation]


@pytest.mark.parametrize("control_count", [1, 2, 3])
@pytest.mark.parametrize(
    "opcode",
    [
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "p",
        "rx",
        "ry",
        "rz",
        "cx",
        "cz",
        "ccx",
        "cp",
        "swap",
        "rzz",
        "global_phase",
    ],
)
def test_controlled_emission_preserves_complex_amplitudes(opcode, control_count):
    """Both SDK projections agree with independent controlled matrix action."""
    import qulacs
    from quri_parts.circuit import LinearMappedParametricQuantumCircuit, QuantumCircuit
    from quri_parts.qulacs.circuit import convert_circuit

    from qamomile.quri_parts._materialization import (
        _add_controlled_instruction,
        _materialize_parametric_instructions,
    )

    angle = -0.37
    base = _base_unitary(opcode, angle)
    target_count = base.shape[0].bit_length() - 1
    width = control_count + target_count
    # Interleave controls and targets to expose reversed SDK operand order.
    wires = list(range(0, width, 2)) + list(range(1, width, 2))
    controls, targets = wires[:control_count], wires[control_count:]
    rng = np.random.default_rng(72)
    initial = rng.normal(size=1 << width) + 1j * rng.normal(size=1 << width)
    initial /= np.linalg.norm(initial)
    expected = np.zeros_like(initial)
    for column, amplitude in enumerate(initial):
        if not all(column & (1 << control) for control in controls):
            expected[column] += amplitude
            continue
        local_column = sum(
            ((column >> wire) & 1) << i for i, wire in enumerate(targets)
        )
        cleared = column & ~sum(1 << wire for wire in targets)
        for local_row in range(1 << target_count):
            row = cleared | sum(
                ((local_row >> i) & 1) << wire for i, wire in enumerate(targets)
            )
            expected[row] += base[local_row, local_column] * amplitude
    angles = (
        [angle]
        if opcode in {"p", "rx", "ry", "rz", "cp", "rzz", "global_phase"}
        else []
    )
    raw = (
        CircuitInstruction(
            opcode, tuple(targets), tuple(angles), controls=tuple(controls)
        ),
    )
    bound = QuantumCircuit(width)
    _add_controlled_instruction(bound, opcode, controls, targets, tuple(angles))
    parametric = LinearMappedParametricQuantumCircuit(width)
    _materialize_parametric_instructions(parametric, raw, {})
    for circuit in (bound, parametric.bind_parameters([])):
        assert all(gate.name != "UnitaryMatrix" for gate in circuit.gates)
        state = qulacs.QuantumState(width)
        state.load(initial)
        convert_circuit(circuit).update_quantum_state(state)
        np.testing.assert_allclose(state.get_vector(), expected, atol=1e-12)


def test_public_projection_keeps_controlled_global_phase():
    """CompiledProgram round trips, public estimators and execution retain interference."""
    from quri_parts.core.operator import pauli_label
    from quri_parts.core.state import GeneralCircuitQuantumState
    from quri_parts.qulacs.estimator import create_qulacs_vector_estimator

    from qamomile.quri_parts import Qulacs, QuriParts, load_circuit

    source = """
import qamomile as qm
@qm.qkernel
def phase(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.global_phase(state, 3.141592653589793)
@qm.qkernel
def sample() -> qm.Bits[2]:
    control, target = qm.split_at(qm.qubits(2), 1)
    control = qm.h(control)
    control, target = qm.control(phase, control, target)
    control = qm.h(control)
    return qm.measure(qm.concat(control, target))
"""
    artifact = qm.compile(qm.build_source(source).specialize("sample"), to=QuriParts())
    artifact = deserialize(serialize(artifact))
    assert qm.execute(artifact, on=Qulacs(), shots=4, seed=2).counts == {(1, 0): 4}
    circuit = load_circuit(artifact).bind_parameters([])
    estimate = create_qulacs_vector_estimator()(
        pauli_label("Z0"), GeneralCircuitQuantumState(2, circuit)
    )
    assert estimate.value == pytest.approx(-1)


def test_eighteen_qubit_control_stays_native(monkeypatch):
    """Both materializers build one MCX without allocating a matrix or state."""
    from quri_parts.qulacs.circuit import convert_circuit

    from qamomile.compilation import Invocation
    from qamomile.quri_parts import QuriParts, load_circuit
    from qamomile.quri_parts._materialization import _materialize_circuit

    source = """
import qamomile as qm
@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)
@qm.qkernel
def sample() -> qm.Bits[18]:
    controls, target = qm.split_at(qm.qubits(18), 17)
    controls, target = qm.control(flip, controls, target)
    return qm.measure(qm.concat(controls, target))
"""
    artifact = qm.compile(qm.build_source(source).specialize("sample"), to=QuriParts())

    def forbid_matrix(*args, **kwargs):
        raise AssertionError("Dense allocation is not allowed")

    monkeypatch.setattr(np, "zeros", forbid_matrix)
    public = load_circuit(artifact).bind_parameters([])
    bound, _ = _materialize_circuit(artifact, Invocation.create(shots=1, seed=0))
    for circuit in (public, bound):
        assert len(circuit.gates) == 1
        assert circuit.gates[0].name == "MCX"
        assert tuple(circuit.gates[0].control_indices) == tuple(range(17))
        native = convert_circuit(circuit).get_gate(0)
        assert len(native.get_control_index_list()) == 17
        assert native.get_matrix().shape == (2, 2)


def test_controlled_runtime_phase_is_rejected_by_public_projection():
    """Unsupported parametric control fails explicitly instead of losing phase."""
    from quri_parts.circuit import LinearMappedParametricQuantumCircuit

    from qamomile.quri_parts._materialization import (
        _materialize_parametric_instructions,
    )

    raw = (
        CircuitInstruction(
            "global_phase", (), (ParameterValue("theta"),), controls=(0,)
        ),
    )
    with pytest.raises(qm.ExecutionError, match="Q8721"):
        _materialize_parametric_instructions(
            LinearMappedParametricQuantumCircuit(1), raw, {}
        )


def test_missing_sdk_control_gate_fails_without_dense_fallback(monkeypatch):
    """An older SDK cannot silently reintroduce exponential matrix expansion."""
    from quri_parts.circuit import QuantumCircuit, gates

    from qamomile.quri_parts._materialization import _add_controlled_instruction

    monkeypatch.delattr(gates, "MCX")
    with pytest.raises(qm.ExecutionError, match="Q8722"):
        _add_controlled_instruction(QuantumCircuit(18), "x", list(range(17)), [17], ())
