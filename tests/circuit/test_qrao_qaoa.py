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


def test_smaller_hamiltonian_is_identity_padded():
    """A Hamiltonian narrower than the register is embedded, not rejected.

    Regression for #467: ``Z(0)`` (1 qubit) on a 2-qubit register must
    transpile by acting only on q[0] (identity on q[1]) instead of
    raising a qubit-count mismatch.
    """
    import numpy as np
    from qiskit.quantum_info import Statevector

    H = qm_o.Hamiltonian()
    H.add_term((qm_o.PauliOperator(qm_o.Pauli.Z, 0),), 1.0)  # 1-qubit

    for use_native in (True, False):
        exe = QiskitTranspiler(use_native_composite=use_native).transpile(
            _wrap_pauli_evolve,
            bindings={"n": 2, "hamiltonian": H, "gamma": 0.5},
        )
        circuit = exe.compiled_quantum[0].circuit.remove_final_measurements(
            inplace=False
        )
        # exp(-i * Z_0 * 0.5) on |00> is a global phase; the populations of
        # the statevector must remain concentrated on |00>.
        probs = np.abs(Statevector(circuit).data) ** 2
        expected = np.zeros(4)
        expected[0] = 1.0
        np.testing.assert_allclose(
            probs,
            expected,
            atol=1e-12,
            rtol=0.0,
            err_msg=(
                f"use_native={use_native}: "
                f"padded evolution changed populations: {probs}"
            ),
        )


@pytest.mark.parametrize("use_native", [True, False])
@pytest.mark.parametrize("constant", [0.0, 2.0])
def test_zero_qubit_hamiltonian_evolves_as_global_phase(use_native, constant):
    """An empty / constant-only Hamiltonian emits no gates.

    ``Hamiltonian.num_qubits == 0`` means the evolution is the global
    phase ``exp(-i*gamma*constant)`` at most. Both Qiskit paths must
    transpile (the native path previously crashed with a raw Qiskit
    ``CircuitError``: the 0-qubit Hamiltonian widens to a 1-qubit
    ``SparsePauliOp(["I"])`` whose gate cannot be appended onto an empty
    qubit list) and leave the state on |00>. The native path additionally
    records the phase on ``circuit.global_phase`` so that it survives as
    an observable relative phase when the circuit is placed under
    ``qmc.control``; the shared gadget fallback drops it (a documented
    limitation of the gadget decomposition).
    """
    import numpy as np
    from qiskit.quantum_info import Statevector

    gamma = 0.5
    H = qm_o.Hamiltonian()
    H += constant

    exe = QiskitTranspiler(use_native_composite=use_native).transpile(
        _wrap_pauli_evolve,
        bindings={"n": 2, "hamiltonian": H, "gamma": gamma},
    )
    circuit = exe.compiled_quantum[0].circuit.remove_final_measurements(inplace=False)
    state = np.asarray(Statevector(circuit).data)
    expected = np.zeros(4, dtype=complex)
    # exp(-i*gamma*H) on |00> with H = constant*I is exp(-i*gamma*constant)|00>.
    # The fallback gadget cannot express a global phase, so it yields |00>.
    expected[0] = np.exp(-1j * gamma * constant) if use_native else 1.0
    np.testing.assert_allclose(
        state,
        expected,
        atol=1e-12,
        rtol=0.0,
        err_msg=(
            f"use_native={use_native}, constant={constant}: "
            f"0-qubit Hamiltonian evolution produced wrong state: {state}"
        ),
    )


@qmc.qkernel
def _evolve_register(
    q: qmc.Vector[qmc.Qubit],
    hamiltonian: qmc.Observable,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Evolve a target register under ``hamiltonian`` (controlled-U body)."""
    q = qmc.pauli_evolve(q, hamiltonian, gamma)
    return q


@qmc.qkernel
def _inverse_evolve_register(
    q: qmc.Vector[qmc.Qubit],
    hamiltonian: qmc.Observable,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply the inverse evolution (controlled-U body with qmc.inverse)."""
    inverse_evolve = qmc.inverse(_evolve_register)
    q = inverse_evolve(q, hamiltonian=hamiltonian, gamma=gamma)
    return q


@pytest.mark.parametrize("invert", [False, True])
def test_controlled_constant_only_hamiltonian_relative_phase(invert):
    """A controlled constant-only evolution keeps the relative phase.

    Uncontrolled, ``exp(-i*gamma*c*I)`` is an unobservable global phase,
    but under ``qmc.control`` it becomes the observable relative phase
    ``exp(-i*gamma*c)`` on the control's |1> branch. With the control in
    |+> the statevector amplitude ratio between the control-1 and
    control-0 subspaces must equal ``exp(-i*gamma*c)`` (and the conjugate
    ``exp(+i*gamma*c)`` when the body is wrapped in ``qmc.inverse``,
    which negates gamma). This pins the Qiskit-native global-phase
    emission for 0-qubit Hamiltonians through gate conversion and
    ``Gate.control``, matching the QuriParts / CUDA-Q controlled paths
    which re-apply the constant explicitly.
    """
    import numpy as np
    from qiskit.quantum_info import Statevector

    gamma = 0.5
    constant = 2.0
    H = qm_o.Hamiltonian()
    H += constant

    body = _inverse_evolve_register if invert else _evolve_register

    @qmc.qkernel
    def controlled_const(
        hamiltonian: qmc.Observable, gamma: qmc.Float
    ) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(2, "q")
        q[0] = qmc.h(q[0])
        controlled_evolve = qmc.control(body)
        q[0], targets = controlled_evolve(
            q[0], q[1:2], hamiltonian=hamiltonian, gamma=gamma
        )
        q[1:2] = targets
        return qmc.measure(q)

    exe = QiskitTranspiler().transpile(
        controlled_const, bindings={"hamiltonian": H, "gamma": gamma}
    )
    circuit = exe.compiled_quantum[0].circuit.remove_final_measurements(inplace=False)
    state = np.asarray(Statevector(circuit).data)
    sign = +1.0 if invert else -1.0
    ratio = state[1] / state[0]
    np.testing.assert_allclose(
        ratio,
        np.exp(sign * 1j * gamma * constant),
        atol=1e-12,
        rtol=0.0,
        err_msg=f"invert={invert}: controlled constant phase ratio wrong: {ratio}",
    )


@pytest.mark.parametrize("use_native", [True, False])
def test_larger_hamiltonian_raises(use_native):
    """A Hamiltonian wider than the register is still a genuine error.

    Covers both the native ``PauliEvolutionGate`` path and the shared
    gadget fallback, which validate independently.
    """
    from qamomile.circuit.transpiler.errors import EmitError

    H = qm_o.Hamiltonian()
    H.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.Z, 0), qm_o.PauliOperator(qm_o.Pauli.Z, 2)),
        1.0,
    )  # acts on 3 qubits

    transpiler = QiskitTranspiler(use_native_composite=use_native)
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
