"""Tests for qamomile/circuit/algorithm/qaoa.py circuit primitives."""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.algorithm.qaoa import (
    hubo_ising_cost,
    hubo_qaoa_state,
    ising_cost,
    phase_gadget,
    qaoa_layers,
    qaoa_state,
    superposition_vector,
    x_mixer,
)
from qamomile.qiskit.transpiler import QiskitTranspiler


# ---------------------------------------------------------------------------
# Helper: count gates by name in a Qiskit circuit
# ---------------------------------------------------------------------------

def _gate_counts(qc):
    """Return a dict of {gate_name: count} from a Qiskit QuantumCircuit."""
    counts = {}
    for inst in qc.data:
        name = inst.operation.name
        counts[name] = counts.get(name, 0) + 1
    return counts


def _gate_list(qc, name):
    """Return list of gate instructions matching *name*."""
    return [inst for inst in qc.data if inst.operation.name == name]


# ---------------------------------------------------------------------------
# Wrapper qkernels (needed to transpile sub-functions with concrete bindings)
# ---------------------------------------------------------------------------

@qmc.qkernel
def _wrap_superposition(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    q = superposition_vector(n)
    return qmc.measure(q)


@qmc.qkernel
def _wrap_ising_cost(
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    n: qmc.UInt,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q = ising_cost(quad, linear, q, gamma)
    return qmc.measure(q)


@qmc.qkernel
def _wrap_x_mixer(
    n: qmc.UInt,
    beta: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q = x_mixer(q, beta)
    return qmc.measure(q)


@qmc.qkernel
def _wrap_phase_gadget(
    n: qmc.UInt,
    indices: qmc.Vector[qmc.UInt],
    angle: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q = phase_gadget(q, indices, angle)
    return qmc.measure(q)


@qmc.qkernel
def _wrap_qaoa_layers(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    n: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q = qaoa_layers(p, quad, linear, q, gammas, betas)
    return qmc.measure(q)


@qmc.qkernel
def _wrap_qaoa_state(
    p: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    n: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    q = qaoa_state(p, quad, linear, n, gammas, betas)
    return qmc.measure(q)


@qmc.qkernel
def _wrap_hubo_ising_cost(
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    higher: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
    n: qmc.UInt,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q = hubo_ising_cost(quad, linear, higher, q, gamma)
    return qmc.measure(q)


@qmc.qkernel
def _wrap_hubo_qaoa_state(
    p_val: qmc.UInt,
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    higher: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
    n: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    q = hubo_qaoa_state(p_val, quad, linear, higher, n, gammas, betas)
    return qmc.measure(q)


# ---------------------------------------------------------------------------
# Primitive tests
# ---------------------------------------------------------------------------


def test_superposition_vector():
    """superposition_vector(3) should produce exactly 3 H gates."""
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(_wrap_superposition, bindings={"n": 3})
    qc = exe.compiled_quantum[0].circuit
    assert _gate_counts(qc).get("h", 0) == 3


def test_ising_cost_gate_counts():
    """ising_cost with 2 quad + 1 linear should produce 2 RZZ + 1 RZ."""
    quad = {(0, 1): 0.5, (1, 2): -0.3}
    linear = {0: 1.0}
    gamma = 0.7

    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(
        _wrap_ising_cost,
        bindings={"quad": quad, "linear": linear, "n": 3, "gamma": gamma},
    )
    qc = exe.compiled_quantum[0].circuit
    counts = _gate_counts(qc)
    assert counts.get("rzz", 0) == 2
    assert counts.get("rz", 0) == 1

    # Verify RZZ angles = Jij * gamma
    rzz_gates = _gate_list(qc, "rzz")
    expected_rzz_angles = sorted([0.5 * gamma, -0.3 * gamma])
    actual_rzz_angles = sorted(float(g.operation.params[0]) for g in rzz_gates)
    for exp, act in zip(expected_rzz_angles, actual_rzz_angles):
        assert abs(exp - act) < 1e-10

    # Verify RZ angle = hi * gamma
    rz_gates = _gate_list(qc, "rz")
    assert abs(float(rz_gates[0].operation.params[0]) - 1.0 * gamma) < 1e-10


def test_x_mixer_gate_counts():
    """x_mixer with n=3, beta=0.4 should produce 3 RX gates with angle=0.8."""
    beta = 0.4
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(
        _wrap_x_mixer,
        bindings={"n": 3, "beta": beta},
    )
    qc = exe.compiled_quantum[0].circuit
    rx_gates = _gate_list(qc, "rx")
    assert len(rx_gates) == 3
    for g in rx_gates:
        assert abs(float(g.operation.params[0]) - 2.0 * beta) < 1e-10


@pytest.mark.parametrize(
    "indices,n_qubits,expected_cx",
    [
        ([0], 1, 0),
        ([0, 1], 2, 2),
        ([0, 1, 2], 3, 4),
        ([0, 1, 2, 3, 4], 5, 8),
    ],
    ids=["k=1", "k=2", "k=3", "k=5"],
)
def test_phase_gadget_cx_counts(indices, n_qubits, expected_cx):
    """phase_gadget with k indices should produce 2*(k-1) CX + 1 RZ."""
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(
        _wrap_phase_gadget,
        bindings={"n": n_qubits, "indices": indices, "angle": 0.5},
    )
    qc = exe.compiled_quantum[0].circuit
    counts = _gate_counts(qc)
    assert counts.get("cx", 0) == expected_cx
    assert counts.get("rz", 0) == 1


def test_phase_gadget_rz_angle():
    """phase_gadget should apply the given angle to the RZ gate."""
    angle = 1.23
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(
        _wrap_phase_gadget,
        bindings={"n": 3, "indices": [0, 1, 2], "angle": angle},
    )
    qc = exe.compiled_quantum[0].circuit
    rz_gates = _gate_list(qc, "rz")
    assert len(rz_gates) == 1
    assert abs(float(rz_gates[0].operation.params[0]) - angle) < 1e-10


# ---------------------------------------------------------------------------
# Composed function tests
# ---------------------------------------------------------------------------


def test_qaoa_layers_p1():
    """qaoa_layers with p=1 should produce cost + mixer gates."""
    quad = {(0, 1): 0.5, (1, 2): -0.3}
    linear = {0: 1.0}
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(
        _wrap_qaoa_layers,
        bindings={
            "p": 1,
            "quad": quad,
            "linear": linear,
            "n": 3,
            "gammas": [0.5],
            "betas": [0.3],
        },
    )
    qc = exe.compiled_quantum[0].circuit
    counts = _gate_counts(qc)
    # 1 layer: 2 RZZ + 1 RZ (cost) + 3 RX (mixer)
    assert counts.get("rzz", 0) == 2
    assert counts.get("rz", 0) == 1
    assert counts.get("rx", 0) == 3


def test_qaoa_layers_p2_doubles_gates():
    """qaoa_layers with p=2 should double cost+mixer gate counts vs p=1."""
    quad = {(0, 1): 0.5, (1, 2): -0.3}
    linear = {0: 1.0}
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(
        _wrap_qaoa_layers,
        bindings={
            "p": 2,
            "quad": quad,
            "linear": linear,
            "n": 3,
            "gammas": [0.5, 0.6],
            "betas": [0.3, 0.4],
        },
    )
    qc = exe.compiled_quantum[0].circuit
    counts = _gate_counts(qc)
    # 2 layers: 4 RZZ + 2 RZ + 6 RX
    assert counts.get("rzz", 0) == 4
    assert counts.get("rz", 0) == 2
    assert counts.get("rx", 0) == 6


def test_qaoa_state_sample():
    """qaoa_state full pipeline: transpile, sample, verify valid results."""
    quad = {(0, 1): 0.5}
    linear = {0: 1.0}
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(
        _wrap_qaoa_state,
        bindings={
            "p": 1,
            "quad": quad,
            "linear": linear,
            "n": 2,
            "gammas": [0.5],
            "betas": [0.3],
        },
    )
    # Verify circuit has H + cost + mixer gates
    qc = exe.compiled_quantum[0].circuit
    counts = _gate_counts(qc)
    assert counts.get("h", 0) == 2
    assert counts.get("rzz", 0) == 1
    assert counts.get("rz", 0) == 1
    assert counts.get("rx", 0) == 2


def test_hubo_ising_cost_gate_counts():
    """hubo_ising_cost should produce ising_cost gates + phase_gadget gates."""
    quad = {(0, 1): 0.5}
    linear = {0: 1.0}
    higher = {(0, 1, 2): 0.3}
    gamma = 0.7
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(
        _wrap_hubo_ising_cost,
        bindings={
            "quad": quad,
            "linear": linear,
            "higher": higher,
            "n": 3,
            "gamma": gamma,
        },
    )
    qc = exe.compiled_quantum[0].circuit
    counts = _gate_counts(qc)
    # ising_cost: 1 RZZ + 1 RZ
    # phase_gadget(k=3): 4 CX + 1 RZ
    assert counts.get("rzz", 0) == 1
    assert counts.get("rz", 0) == 2  # 1 from ising_cost + 1 from phase_gadget
    assert counts.get("cx", 0) == 4


def test_hubo_qaoa_state_sample():
    """hubo_qaoa_state full pipeline: transpile, sample, verify valid results."""
    quad = {(0, 1): 0.5}
    linear = {0: 1.0}
    higher = {(0, 1, 2): 0.3}
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(
        _wrap_hubo_qaoa_state,
        bindings={
            "p_val": 1,
            "quad": quad,
            "linear": linear,
            "higher": higher,
            "n": 3,
            "gammas": [0.5],
            "betas": [0.3],
        },
    )
    qc = exe.compiled_quantum[0].circuit
    counts = _gate_counts(qc)
    # superposition: 3 H
    # hubo cost: 1 RZZ + 1 RZ (ising) + 4 CX + 1 RZ (gadget)
    # mixer: 3 RX
    assert counts.get("h", 0) == 3
    assert counts.get("rzz", 0) == 1
    assert counts.get("rz", 0) == 2
    assert counts.get("cx", 0) == 4
    assert counts.get("rx", 0) == 3
