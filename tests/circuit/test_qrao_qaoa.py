"""Tests for pauli_evolve with QRAO QAOA as the motivating use case."""

import pytest

pytest.importorskip("qiskit")

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.basic import superposition_vector
from qamomile.circuit.algorithm.qaoa import x_mixer
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
# Wrapper qkernels (test-local)
# ---------------------------------------------------------------------------


@qmc.qkernel
def _wrap_pauli_evolve(
    n: qmc.UInt,
    hamiltonian: qmc.Observable,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q = qmc.pauli_evolve(q, hamiltonian, gamma)
    return qmc.measure(q)


@qmc.qkernel
def _qrao_qaoa_state(
    p: qmc.UInt,
    hamiltonian: qmc.Observable,
    n: qmc.UInt,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """QRAO QAOA: superposition → (pauli_evolve + X-mixer) × p → measure."""
    q = superposition_vector(n)
    for layer in qmc.range(p):
        q = qmc.pauli_evolve(q, hamiltonian, gammas[layer])
        q = x_mixer(q, betas[layer])
    return qmc.measure(q)


# ---------------------------------------------------------------------------
# Tests for pauli_evolve
# ---------------------------------------------------------------------------


def test_pauli_evolve_single_z():
    """Single Z term: should produce a PauliEvolutionGate (native Qiskit)."""
    H = qm_o.Hamiltonian()
    H.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0),), 1.0)

    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(
        _wrap_pauli_evolve,
        bindings={"n": 1, "hamiltonian": H, "gamma": 0.7},
    )
    qc = exe.compiled_quantum[0].circuit
    assert qc.size() > 0


def test_pauli_evolve_single_x():
    """Single X term: should produce a PauliEvolutionGate or H+RZ+H."""
    H = qm_o.Hamiltonian()
    H.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 0.5)

    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(
        _wrap_pauli_evolve,
        bindings={"n": 1, "hamiltonian": H, "gamma": 0.7},
    )
    qc = exe.compiled_quantum[0].circuit
    assert qc.size() > 0


def test_pauli_evolve_single_y():
    """Single Y term: should produce a PauliEvolutionGate or Sdg+H+RZ+H+S."""
    H = qm_o.Hamiltonian()
    H.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 0),), 0.3)

    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(
        _wrap_pauli_evolve,
        bindings={"n": 1, "hamiltonian": H, "gamma": 0.7},
    )
    qc = exe.compiled_quantum[0].circuit
    assert qc.size() > 0


def test_pauli_evolve_mixed_xz_two_qubit():
    """X_0 * Z_1 term: verify circuit is produced."""
    H = qm_o.Hamiltonian()
    H.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.X, 0), qm_o.PauliOperator(qm_o.Pauli.Z, 1)),
        0.2,
    )

    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(
        _wrap_pauli_evolve,
        bindings={"n": 2, "hamiltonian": H, "gamma": 0.5},
    )
    qc = exe.compiled_quantum[0].circuit
    assert qc.size() > 0
    assert qc.num_qubits == 2


def test_pauli_evolve_multi_term_hamiltonian():
    """Multi-term Hamiltonian with mixed Pauli types."""
    H = qm_o.Hamiltonian()
    H.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0),), 1.0)
    H.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 1),), 0.5)
    H.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.X, 0), qm_o.PauliOperator(qm_o.Pauli.Z, 1)),
        0.2,
    )

    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(
        _wrap_pauli_evolve,
        bindings={"n": 2, "hamiltonian": H, "gamma": 0.7},
    )
    qc = exe.compiled_quantum[0].circuit
    assert qc.size() > 0
    assert qc.num_qubits == 2


# ---------------------------------------------------------------------------
# Fallback decomposition tests (non-native, verifies Pauli gadget pattern)
# ---------------------------------------------------------------------------


def test_fallback_x_term():
    """Fallback decomposition for X term: H + RZ + H."""
    H = qm_o.Hamiltonian()
    H.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 0.5)

    gamma = 0.7
    transpiler = QiskitTranspiler(use_native_composite=False)
    exe = transpiler.transpile(
        _wrap_pauli_evolve,
        bindings={"n": 1, "hamiltonian": H, "gamma": gamma},
    )
    qc = exe.compiled_quantum[0].circuit
    counts = _gate_counts(qc)
    assert counts.get("h", 0) == 2
    assert counts.get("rz", 0) == 1

    # RZ(θ) = exp(-iθZ/2), so θ = 2 * coeff * gamma
    rz_gates = _gate_list(qc, "rz")
    assert abs(float(rz_gates[0].operation.params[0]) - 2.0 * 0.5 * gamma) < 1e-10


def test_fallback_y_term():
    """Fallback decomposition for Y term: Sdg + H + RZ + H + S."""
    H = qm_o.Hamiltonian()
    H.add_term((qm_o.PauliOperator(qm_o.Pauli.Y, 0),), 0.3)

    gamma = 0.7
    transpiler = QiskitTranspiler(use_native_composite=False)
    exe = transpiler.transpile(
        _wrap_pauli_evolve,
        bindings={"n": 1, "hamiltonian": H, "gamma": gamma},
    )
    qc = exe.compiled_quantum[0].circuit
    counts = _gate_counts(qc)
    assert counts.get("sdg", 0) == 1
    assert counts.get("h", 0) == 2
    assert counts.get("rz", 0) == 1
    assert counts.get("s", 0) == 1


def test_fallback_xz_two_qubit():
    """Fallback: X_0 * Z_1 -> H(0) + CX + RZ + CX + H(0)."""
    H = qm_o.Hamiltonian()
    H.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.X, 0), qm_o.PauliOperator(qm_o.Pauli.Z, 1)),
        0.2,
    )

    gamma = 0.5
    transpiler = QiskitTranspiler(use_native_composite=False)
    exe = transpiler.transpile(
        _wrap_pauli_evolve,
        bindings={"n": 2, "hamiltonian": H, "gamma": gamma},
    )
    qc = exe.compiled_quantum[0].circuit
    counts = _gate_counts(qc)
    assert counts.get("h", 0) == 2
    assert counts.get("cx", 0) == 2
    assert counts.get("rz", 0) == 1


# ---------------------------------------------------------------------------
# QRAO QAOA integration (pauli_evolve + mixer in a QAOA circuit)
# ---------------------------------------------------------------------------


def test_qrao_qaoa_p1():
    """Full QRAO QAOA with p=1 using a mixed Pauli Hamiltonian."""
    H = qm_o.Hamiltonian()
    H.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0),), 1.0)
    H.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 1),), 0.5)
    H.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.Z, 0), qm_o.PauliOperator(qm_o.Pauli.Z, 1)),
        -0.3,
    )

    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(
        _qrao_qaoa_state,
        bindings={
            "p": 1,
            "hamiltonian": H,
            "n": 2,
            "gammas": [0.5],
            "betas": [0.3],
        },
    )
    qc = exe.compiled_quantum[0].circuit
    counts = _gate_counts(qc)
    # H from superposition
    assert counts.get("h", 0) >= 2
    # PauliEvolution (native) or decomposed cost gates + RX from mixer
    assert counts.get("PauliEvolution", 0) + counts.get("rx", 0) > 0
    assert qc.num_qubits == 2


def test_qrao_qaoa_p2_more_gates():
    """QRAO QAOA with p=2 should produce more gates than p=1."""
    H = qm_o.Hamiltonian()
    H.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0),), 1.0)
    H.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 1),), 0.5)

    transpiler = QiskitTranspiler()
    exe_p1 = transpiler.transpile(
        _qrao_qaoa_state,
        bindings={
            "p": 1,
            "hamiltonian": H,
            "n": 2,
            "gammas": [0.5],
            "betas": [0.3],
        },
    )
    exe_p2 = transpiler.transpile(
        _qrao_qaoa_state,
        bindings={
            "p": 2,
            "hamiltonian": H,
            "n": 2,
            "gammas": [0.5, 0.6],
            "betas": [0.3, 0.4],
        },
    )
    qc_p1 = exe_p1.compiled_quantum[0].circuit
    qc_p2 = exe_p2.compiled_quantum[0].circuit
    assert qc_p2.size() > qc_p1.size()


# ---------------------------------------------------------------------------
# Native vs fallback consistency
# ---------------------------------------------------------------------------


def test_native_fallback_statevector_match():
    """Native PauliEvolutionGate and fallback decomposition must produce
    the same statevector (up to global phase)."""
    from qiskit.quantum_info import Statevector

    H = qm_o.Hamiltonian()
    H.add_term((qm_o.PauliOperator(qm_o.Pauli.X, 0),), 0.5)
    H.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 1),), -0.3)
    H.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.X, 0), qm_o.PauliOperator(qm_o.Pauli.Z, 1)),
        0.2,
    )

    bindings = {"n": 2, "hamiltonian": H, "gamma": 0.7}

    native_exe = QiskitTranspiler().transpile(_wrap_pauli_evolve, bindings=bindings)
    fallback_exe = QiskitTranspiler(use_native_composite=False).transpile(
        _wrap_pauli_evolve, bindings=bindings
    )

    sv_native = Statevector(
        native_exe.compiled_quantum[0].circuit.remove_final_measurements(inplace=False)
    )
    sv_fallback = Statevector(
        fallback_exe.compiled_quantum[0].circuit.remove_final_measurements(
            inplace=False
        )
    )

    assert sv_native.equiv(sv_fallback), (
        f"Native and fallback statevectors differ.\n"
        f"Native:   {sv_native}\n"
        f"Fallback: {sv_fallback}"
    )


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_qubit_count_mismatch_raises():
    """Passing a 1-qubit Hamiltonian to a 2-qubit register should error."""
    from qamomile.circuit.transpiler.errors import EmitError

    H = qm_o.Hamiltonian()
    H.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0),), 1.0)  # 1-qubit

    transpiler = QiskitTranspiler(use_native_composite=False)
    with pytest.raises(EmitError, match="qubit count mismatch"):
        transpiler.transpile(
            _wrap_pauli_evolve,
            bindings={"n": 2, "hamiltonian": H, "gamma": 0.5},
        )


def test_complex_coefficient_raises():
    """Hamiltonian with complex (non-Hermitian) coefficients should error."""
    from qamomile.circuit.transpiler.errors import EmitError

    H = qm_o.Hamiltonian()
    H.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0),), 1.0 + 0.5j)  # complex

    transpiler = QiskitTranspiler(use_native_composite=False)
    with pytest.raises(EmitError, match="Hermitian"):
        transpiler.transpile(
            _wrap_pauli_evolve,
            bindings={"n": 1, "hamiltonian": H, "gamma": 0.5},
        )
