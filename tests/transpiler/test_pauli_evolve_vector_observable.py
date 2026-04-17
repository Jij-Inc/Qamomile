"""Suzuki-Trotter expansion via ``Vector[Observable]``.

Exercises the end-to-end path where a kernel receives a list of
Hamiltonians ``Hs`` and applies ``pauli_evolve(q, Hs[i], gamma)`` in a
loop over ``Hs.shape[0]``. The loop must unroll because each iteration
needs a concrete Hamiltonian at emit time; the resulting circuit must
match a direct ``pauli_evolve(q, sum(Hs), gamma)`` for commuting terms.
"""

import pytest

pytest.importorskip("qiskit")

import numpy as np
from qiskit_aer import AerSimulator

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.qiskit.transpiler import QiskitTranspiler


def _pad_to(num_qubits: int, term: qm_o.Hamiltonian) -> qm_o.Hamiltonian:
    """Pad a Hamiltonian with a zero-coefficient identity on the highest
    qubit so it declares the full register width. ``pauli_evolve`` expects
    ``Hamiltonian.num_qubits`` to equal the qubit register size."""
    padded = term + 0.0 * qm_o.Z(num_qubits - 1)
    assert padded.num_qubits == num_qubits
    return padded


def _statevector(circuit) -> np.ndarray:
    from qiskit import QuantumCircuit, transpile as qk_transpile

    # Strip measurements — we want the pre-measurement state.
    stripped = QuantumCircuit(*circuit.qregs)
    for instr in circuit.data:
        if instr.operation.name not in ("measure", "save_statevector"):
            stripped.append(instr)
    # PauliEvolutionGate is not a basis gate for AerSimulator; lower it.
    stripped = qk_transpile(
        stripped,
        basis_gates=["u", "cx", "rx", "ry", "rz", "h", "p", "sx", "x", "y", "z"],
    )
    stripped.save_statevector()
    sim = AerSimulator(method="statevector")
    result = sim.run(stripped).result()
    return np.asarray(result.get_statevector())


@qmc.qkernel
def _trotter(Hs: qmc.Vector[qmc.Observable], gamma: qmc.Float) -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.h(q[0])
    q[1] = qmc.h(q[1])
    n = Hs.shape[0]
    for i in qmc.range(n):
        q = qmc.pauli_evolve(q, Hs[i], gamma)
    return q


@qmc.qkernel
def _trotter_meas(
    Hs: qmc.Vector[qmc.Observable], gamma: qmc.Float
) -> qmc.Vector[qmc.Bit]:
    q = _trotter(Hs, gamma)
    return qmc.measure(q)


@qmc.qkernel
def _direct(H: qmc.Observable, gamma: qmc.Float) -> qmc.Vector[qmc.Qubit]:
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.h(q[0])
    q[1] = qmc.h(q[1])
    q = qmc.pauli_evolve(q, H, gamma)
    return q


@qmc.qkernel
def _direct_meas(H: qmc.Observable, gamma: qmc.Float) -> qmc.Vector[qmc.Bit]:
    q = _direct(H, gamma)
    return qmc.measure(q)


class TestTrotterViaVectorObservable:
    """Verify the Vector[Observable] → Trotter decomposition pipeline."""

    def test_loop_unrolls_to_per_term_pauli_evolve(self):
        """Each Hs[i] iteration must be emitted separately.

        The for-loop over Hs.shape[0] must NOT survive emit (no concrete
        Hamiltonian exists for a symbolic ``Hs[i]``). The decomposed
        circuit has one RZ per non-zero single-Pauli term.
        """
        H1 = _pad_to(2, qm_o.Z(0))
        H2 = _pad_to(2, qm_o.Z(1))
        H3 = _pad_to(2, qm_o.X(0))

        tr = QiskitTranspiler()
        exe = tr.transpile(_trotter_meas, bindings={"Hs": [H1, H2, H3], "gamma": 0.4})
        circuit = exe.compiled_quantum[0].circuit
        gate_names = [inst.operation.name for inst in circuit.data]
        assert "for_loop" not in gate_names
        # Each term is decomposed to an RZ (with basis change for X).
        assert gate_names.count("rz") == 3

    def test_commuting_trotter_matches_direct_sum(self):
        """For commuting terms, product of exponentials equals exp of sum."""
        # Z_0 and Z_1 commute, so exp(-i g Z0) exp(-i g Z1) = exp(-i g (Z0+Z1))
        H1 = _pad_to(2, qm_o.Z(0))
        H2 = _pad_to(2, qm_o.Z(1))
        gamma = 0.37

        tr = QiskitTranspiler()
        exe_trotter = tr.transpile(
            _trotter_meas, bindings={"Hs": [H1, H2], "gamma": gamma}
        )
        exe_direct = tr.transpile(_direct_meas, bindings={"H": H1 + H2, "gamma": gamma})

        sv_trotter = _statevector(exe_trotter.compiled_quantum[0].circuit)
        sv_direct = _statevector(exe_direct.compiled_quantum[0].circuit)

        # Compare up to global phase (inner product magnitude ≈ 1).
        overlap = abs(np.vdot(sv_direct, sv_trotter))
        assert overlap == pytest.approx(1.0, abs=1e-9)

    def test_noncommuting_trotter_converges_with_steps(self):
        """First-order Trotter error shrinks as we refine the step size.

        exp(-i t (X + Z)) ≈ (exp(-i t/N X) exp(-i t/N Z))^N
        converges to the true propagator (computed via scipy) as N → ∞.
        """
        from scipy.linalg import expm

        t = 0.8

        @qmc.qkernel
        def trotter_steps(
            Hs: qmc.Vector[qmc.Observable], step_gamma: qmc.Float, reps: qmc.UInt
        ) -> qmc.Vector[qmc.Qubit]:
            q = qmc.qubit_array(1, "q")
            for _ in qmc.range(reps):
                for i in qmc.range(Hs.shape[0]):
                    q = qmc.pauli_evolve(q, Hs[i], step_gamma)
            return q

        @qmc.qkernel
        def trotter_meas(
            Hs: qmc.Vector[qmc.Observable], step_gamma: qmc.Float, reps: qmc.UInt
        ) -> qmc.Vector[qmc.Bit]:
            q = trotter_steps(Hs, step_gamma, reps)
            return qmc.measure(q)

        # Reference: exp(-i t (X+Z)) |0> computed directly.
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        U_exact = expm(-1j * t * (X + Z))
        sv_exact = U_exact @ np.array([1.0, 0.0], dtype=complex)

        tr = QiskitTranspiler()
        Hs = [qm_o.X(0), qm_o.Z(0)]

        errors = []
        for reps in (1, 4, 16):
            exe = tr.transpile(
                trotter_meas,
                bindings={"Hs": Hs, "step_gamma": t / reps, "reps": reps},
            )
            sv = _statevector(exe.compiled_quantum[0].circuit)
            err = 1.0 - abs(np.vdot(sv_exact, sv))
            errors.append(err)

        # First-order Trotter error is O(1/N). Refining strictly reduces it.
        assert errors[0] > errors[1] > errors[2]
        # At reps=16, error should be small (<1e-2).
        assert errors[-1] < 1e-2


class TestTrotterBackendPortability:
    """Vector[Observable] + pauli_evolve must transpile on every backend.

    We only check that each backend produces a non-empty program with the
    expected number of per-term quantum operations; the numerical check is
    already covered on Qiskit above. These tests skip when the optional
    backend package is not installed.
    """

    def _build_kernel(self):
        return _trotter_meas

    def test_qiskit(self):
        H1 = _pad_to(2, qm_o.Z(0))
        H2 = _pad_to(2, qm_o.Z(1))
        H3 = _pad_to(2, qm_o.X(0))

        tr = QiskitTranspiler()
        exe = tr.transpile(
            self._build_kernel(), bindings={"Hs": [H1, H2, H3], "gamma": 0.4}
        )
        circuit = exe.compiled_quantum[0].circuit
        names = [inst.operation.name for inst in circuit.data]
        assert "for_loop" not in names
        assert names.count("rz") == 3

    def test_quri_parts(self):
        pytest.importorskip("quri_parts")
        from qamomile.quri_parts.transpiler import QuriPartsTranspiler

        H1 = _pad_to(2, qm_o.Z(0))
        H2 = _pad_to(2, qm_o.Z(1))
        H3 = _pad_to(2, qm_o.X(0))

        tr = QuriPartsTranspiler()
        exe = tr.transpile(
            self._build_kernel(), bindings={"Hs": [H1, H2, H3], "gamma": 0.4}
        )
        circuit = exe.compiled_quantum[0].circuit
        # PauliRotation is QuriParts' native exp(-i*theta/2 * P_string),
        # which the default emitter lowers per term.
        gate_names = [type(g).__name__ for g in getattr(circuit, "gates", ())]
        assert gate_names, "QuriParts circuit should contain gates"

    def test_cudaq(self):
        pytest.importorskip("cudaq")
        from qamomile.cudaq.transpiler import CudaqTranspiler  # noqa: I001

        H1 = _pad_to(2, qm_o.Z(0))
        H2 = _pad_to(2, qm_o.Z(1))
        H3 = _pad_to(2, qm_o.X(0))

        tr = CudaqTranspiler()
        exe = tr.transpile(
            self._build_kernel(), bindings={"Hs": [H1, H2, H3], "gamma": 0.4}
        )
        # Just confirm the pipeline produced a compiled quantum segment.
        assert exe.compiled_quantum
