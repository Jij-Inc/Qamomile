"""Tests for qamomile/circuit/algorithm/trotter.py.

``trotterized_time_evolution`` is a thin Python wrapper that validates
its arguments and delegates to an internal ``@qkernel``.  Correctness
is verified via fidelity against the exact propagator; validation is
verified via ``pytest.raises`` around ``transpile`` (where the outer
wrapper kernel is re-traced with concrete bindings).
"""

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from qiskit import QuantumCircuit, transpile as qk_transpile  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
from scipy.linalg import expm  # noqa: E402

import qamomile.circuit as qmc  # noqa: E402
import qamomile.observable as qm_o  # noqa: E402
from qamomile.circuit.algorithm.trotter import (  # noqa: E402
    trotterized_time_evolution,
)
from qamomile.qiskit import QiskitTranspiler  # noqa: E402

# -----------------------------------------------------------------------
# Test fixtures and helpers
# -----------------------------------------------------------------------

OMEGA = 1.2
OMEGA_DRIVE = 0.8
T_EVOLVE = 1.5

HZ = 0.5 * OMEGA * qm_o.Z(0)
HX = 0.5 * OMEGA_DRIVE * qm_o.X(0)
HS_2TERM = [HZ, HX]


def _exact_state(t: float = T_EVOLVE) -> np.ndarray:
    """Exact ``exp(-i t H) |0>`` for the 1-qubit Rabi Hamiltonian."""
    X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
    Z_mat = np.array([[1, 0], [0, -1]], dtype=complex)
    H_mat = 0.5 * OMEGA * Z_mat + 0.5 * OMEGA_DRIVE * X_mat
    return expm(-1j * t * H_mat) @ np.array([1.0, 0.0], dtype=complex)


def _statevector(circuit) -> np.ndarray:
    """Strip measurements, lower to basis gates, and simulate."""
    stripped = QuantumCircuit(*circuit.qregs)
    for instr in circuit.data:
        if instr.operation.name not in ("measure", "save_statevector"):
            stripped.append(instr)
    stripped = qk_transpile(
        stripped,
        basis_gates=["u", "cx", "rx", "ry", "rz", "h", "p", "sx", "x", "y", "z"],
    )
    stripped.save_statevector()
    sim = AerSimulator(method="statevector")
    return np.asarray(sim.run(stripped).result().get_statevector())


@qmc.qkernel
def _rabi_trotter(
    order: qmc.UInt,
    Hs: qmc.Vector[qmc.Observable],
    gamma: qmc.Float,
    step: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Top-level wrapper kernel that allocates qubits and measures."""
    q = qmc.qubit_array(1, name="q")
    q = trotterized_time_evolution(q, Hs, order, gamma, step)
    return qmc.measure(q)


# -----------------------------------------------------------------------
# Compilation smoke tests
# -----------------------------------------------------------------------


class TestCompilation:
    """Kernel transpiles for every supported order and any step count."""

    @pytest.mark.parametrize("order", [1, 2, 4, 6])
    @pytest.mark.parametrize("step", [1, 2, 4])
    def test_transpile_succeeds(self, order: int, step: int) -> None:
        tr = QiskitTranspiler()
        exe = tr.transpile(
            _rabi_trotter,
            bindings={
                "order": order,
                "Hs": HS_2TERM,
                "gamma": T_EVOLVE,
                "step": step,
            },
        )
        assert exe.compiled_quantum[0].circuit.num_qubits == 1


# -----------------------------------------------------------------------
# Numerical correctness
# -----------------------------------------------------------------------


class TestOrder1Correctness:
    """Lie-Trotter (order=1): global fidelity error scales as O(dt)."""

    def test_converges_to_exact(self) -> None:
        """Fidelity error shrinks as step count grows."""
        sv_exact = _exact_state()
        tr = QiskitTranspiler()

        errors: list[float] = []
        for step in (4, 16, 64):
            exe = tr.transpile(
                _rabi_trotter,
                bindings={
                    "order": 1,
                    "Hs": HS_2TERM,
                    "gamma": T_EVOLVE,
                    "step": step,
                },
            )
            sv = _statevector(exe.compiled_quantum[0].circuit)
            overlap = np.clip(abs(np.vdot(sv_exact, sv)), 0.0, 1.0)
            errors.append(max(1.0 - overlap, np.finfo(float).tiny))

        # Strictly monotone improvement (first-order Trotter error is O(dt)).
        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] < 1e-2


class TestOrder2Correctness:
    """Strang splitting (order=2): global fidelity error scales as O(dt^2)."""

    def test_converges_faster_than_order1(self) -> None:
        """Same step count, order=2 beats order=1 for non-trivial evolutions."""
        sv_exact = _exact_state()
        tr = QiskitTranspiler()

        def _err(order: int, step: int) -> float:
            exe = tr.transpile(
                _rabi_trotter,
                bindings={
                    "order": order,
                    "Hs": HS_2TERM,
                    "gamma": T_EVOLVE,
                    "step": step,
                },
            )
            sv = _statevector(exe.compiled_quantum[0].circuit)
            overlap = np.clip(abs(np.vdot(sv_exact, sv)), 0.0, 1.0)
            return max(1.0 - overlap, np.finfo(float).tiny)

        step = 8
        assert _err(2, step) < _err(1, step)

    def test_strang_slope(self) -> None:
        """log-log slope of order=2 error vs dt is close to 2."""
        sv_exact = _exact_state()
        tr = QiskitTranspiler()

        dts: list[float] = []
        errs: list[float] = []
        for step in (4, 8, 16):
            exe = tr.transpile(
                _rabi_trotter,
                bindings={
                    "order": 2,
                    "Hs": HS_2TERM,
                    "gamma": T_EVOLVE,
                    "step": step,
                },
            )
            sv = _statevector(exe.compiled_quantum[0].circuit)
            overlap = np.clip(abs(np.vdot(sv_exact, sv)), 0.0, 1.0)
            dts.append(T_EVOLVE / step)
            errs.append(max(1.0 - overlap, np.finfo(float).tiny))

        slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
        # Strang ``|<ψ_exact|ψ̃>|`` fidelity error ~ O(dt^4) (state-norm
        # error is O(dt^2), fidelity = 1 - |<ψ|ψ̃>| ≈ ½|ψ-ψ̃|² squares it).
        assert abs(slope - 4.0) < 1.0, f"measured slope={slope:.3f}"


class TestHigherOrderRecursion:
    """Suzuki S_{2k} recursion: fidelity-error slope scales as ``2k``.

    The transpiler folds the self-recursive ``_suzuki_trotter_step``
    under a concrete ``order`` binding.  We verify that the emitted
    circuit realises the textbook Suzuki-Trotter convergence rate.
    """

    def test_order4_convergence_slope(self) -> None:
        """S_4 fidelity-error slope is ~8 across step = 4, 8, 16."""
        sv_exact = _exact_state()
        tr = QiskitTranspiler()

        dts: list[float] = []
        errs: list[float] = []
        for step in (4, 8, 16):
            exe = tr.transpile(
                _rabi_trotter,
                bindings={
                    "order": 4,
                    "Hs": HS_2TERM,
                    "gamma": T_EVOLVE,
                    "step": step,
                },
            )
            sv = _statevector(exe.compiled_quantum[0].circuit)
            overlap = np.clip(abs(np.vdot(sv_exact, sv)), 0.0, 1.0)
            dts.append(T_EVOLVE / step)
            errs.append(max(1.0 - overlap, np.finfo(float).tiny))

        slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
        # Textbook Suzuki S_4 fidelity-error slope is 8; tolerance
        # absorbs BLAS-level float drift across platforms.
        assert abs(slope - 8.0) < 1.5, f"S_4 slope={slope:.3f}, expected 8.0"

    def test_order6_reaches_machine_precision(self) -> None:
        """S_6 on this 2-term Rabi Hamiltonian saturates near float precision.

        For the ``omega=1.2, Omega=0.8, T=1.5`` test Hamiltonian, S_6's
        theoretical fidelity error of ``~(T/N)^{12}`` drops below 1e-14
        by ``step=2`` — so we just verify that the recursion compiles
        and produces a near-exact result, which already implies the
        formula coefficients are correct through two recursion levels
        (``6 → 4 → 2``).
        """
        sv_exact = _exact_state()
        tr = QiskitTranspiler()
        exe = tr.transpile(
            _rabi_trotter,
            bindings={
                "order": 6,
                "Hs": HS_2TERM,
                "gamma": T_EVOLVE,
                "step": 4,
            },
        )
        sv = _statevector(exe.compiled_quantum[0].circuit)
        overlap = abs(np.vdot(sv_exact, sv))
        assert overlap > 1.0 - 1e-12, f"S_6 overlap={overlap:.15f}"

    def test_higher_order_beats_lower_order(self) -> None:
        """For the same step count, S_4 is more accurate than S_2."""
        sv_exact = _exact_state()
        tr = QiskitTranspiler()

        def _err(order: int) -> float:
            exe = tr.transpile(
                _rabi_trotter,
                bindings={
                    "order": order,
                    "Hs": HS_2TERM,
                    "gamma": T_EVOLVE,
                    "step": 4,
                },
            )
            sv = _statevector(exe.compiled_quantum[0].circuit)
            overlap = np.clip(abs(np.vdot(sv_exact, sv)), 0.0, 1.0)
            return max(1.0 - overlap, np.finfo(float).tiny)

        assert _err(4) < _err(2)


class TestOrder2Multiterm:
    """Order=2 Strang splitting for a 3-term Hamiltonian (merged palindrome)."""

    def test_three_term_converges(self) -> None:
        # H = 0.5*Z + 0.4*X + 0.3*Y on a single qubit.
        Hs3 = [0.5 * qm_o.Z(0), 0.4 * qm_o.X(0), 0.3 * qm_o.Y(0)]

        X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
        Z_mat = np.array([[1, 0], [0, -1]], dtype=complex)
        Y_mat = np.array([[0, -1j], [1j, 0]], dtype=complex)
        H_mat = 0.5 * Z_mat + 0.4 * X_mat + 0.3 * Y_mat
        sv_exact = expm(-1j * T_EVOLVE * H_mat) @ np.array([1.0, 0.0], dtype=complex)

        tr = QiskitTranspiler()
        errs: list[float] = []
        for step in (4, 16, 64):
            exe = tr.transpile(
                _rabi_trotter,
                bindings={
                    "order": 2,
                    "Hs": Hs3,
                    "gamma": T_EVOLVE,
                    "step": step,
                },
            )
            sv = _statevector(exe.compiled_quantum[0].circuit)
            overlap = np.clip(abs(np.vdot(sv_exact, sv)), 0.0, 1.0)
            errs.append(max(1.0 - overlap, np.finfo(float).tiny))

        assert errs[0] > errs[1] > errs[2]
        assert errs[-1] < 1e-3


# -----------------------------------------------------------------------
# Gate-structure check
#
# Single-step, order=2 on 2 terms: the merged palindrome emits exactly
# three ``pauli_evolve`` calls → three ``rz`` rotations in the lowered
# Qiskit circuit (one per term, with basis-change for X).
# -----------------------------------------------------------------------


class TestGateCountsSingleStep:
    @pytest.mark.parametrize(
        "order, expected_rz",
        [
            (1, 2),  # H_0, H_1 — one rz each
            (2, 3),  # H_0(1/2), H_1(1), H_0(1/2) — three rz
        ],
    )
    def test_inner_rz_count(self, order: int, expected_rz: int) -> None:
        """Count rz rotations inside the body of the step for_loop.

        ``step`` is a UInt parameter, so the outer Trotter-step loop
        stays as a native Qiskit ``for_loop``; we descend into its body
        to inspect the per-step gate structure.
        """
        tr = QiskitTranspiler()
        exe = tr.transpile(
            _rabi_trotter,
            bindings={
                "order": order,
                "Hs": HS_2TERM,
                "gamma": 1.0,
                "step": 1,
            },
        )
        qc = exe.compiled_quantum[0].circuit

        # The outer step loop is a Qiskit for_loop instruction.  Peek at
        # its body to count per-step gates.
        bodies = [
            inst.operation.blocks[0]
            for inst in qc.data
            if inst.operation.name == "for_loop"
        ]
        assert len(bodies) == 1, f"expected 1 for_loop, got {len(bodies)}"
        body = bodies[0]
        rz_count = sum(1 for inst in body.data if inst.operation.name == "rz")
        assert rz_count == expected_rz


# -----------------------------------------------------------------------
# Argument validation
#
# The validation lives on the Python wrapper ``trotterized_time_evolution``
# and fires when the outer ``_rabi_trotter`` kernel is re-traced with
# concrete bindings — i.e. inside ``tr.transpile(...)``.  The initial
# lazy cache-trace (symbolic dummies, no bindings) must NOT raise so
# that merely importing the module works for any valid/invalid setup.
# -----------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("bad_order", [0, 3, 5, -2])
    def test_reject_non_even_or_nonpositive_order(self, bad_order: int) -> None:
        tr = QiskitTranspiler()
        with pytest.raises(ValueError, match="order must be 1 or"):
            tr.transpile(
                _rabi_trotter,
                bindings={
                    "order": bad_order,
                    "Hs": HS_2TERM,
                    "gamma": T_EVOLVE,
                    "step": 1,
                },
            )

    def test_reject_bool_literal_order(self) -> None:
        """Bool literal inside a kernel body is caught at the call site.

        Note: ``order=True/False`` passed via ``bindings`` is converted
        to ``UInt(1)/UInt(0)`` by ``_create_bound_input`` before the
        wrapper sees it, so the ``isinstance(order, bool)`` guard only
        fires when the bool is a Python literal at the call site (a
        defensive check for direct-literal typos).
        """

        @qmc.qkernel
        def _wrap_with_bool_literal(
            Hs: qmc.Vector[qmc.Observable],
            gamma: qmc.Float,
            step: qmc.UInt,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(1, name="q")
            q = trotterized_time_evolution(q, Hs, True, gamma, step)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        with pytest.raises(ValueError, match="got bool"):
            tr.transpile(
                _wrap_with_bool_literal,
                bindings={"Hs": HS_2TERM, "gamma": T_EVOLVE, "step": 1},
            )

    def test_reject_single_term_hamiltonian(self) -> None:
        tr = QiskitTranspiler()
        with pytest.raises(ValueError, match="at least 2 terms"):
            tr.transpile(
                _rabi_trotter,
                bindings={
                    "order": 1,
                    "Hs": [HZ],
                    "gamma": T_EVOLVE,
                    "step": 1,
                },
            )

    @pytest.mark.parametrize("good_order", [1, 2, 4, 6])
    def test_accept_valid_order(self, good_order: int) -> None:
        """Sanity check: valid orders pass through without raising."""
        tr = QiskitTranspiler()
        tr.transpile(
            _rabi_trotter,
            bindings={
                "order": good_order,
                "Hs": HS_2TERM,
                "gamma": T_EVOLVE,
                "step": 1,
            },
        )
