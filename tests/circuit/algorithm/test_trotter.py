"""Tests for qamomile/circuit/algorithm/trotter.py.

``trotterized_time_evolution`` is a thin Python wrapper that validates
its arguments and delegates to an internal ``@qkernel``.  Correctness
is verified via fidelity against the exact propagator; validation is
verified via ``pytest.raises`` around ``transpile`` (where the outer
wrapper kernel is re-traced with concrete bindings).
"""

from typing import Any

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


def _fidelity_err(sv_a: np.ndarray, sv_b: np.ndarray) -> float:
    """Fidelity error ``1 - |<a|b>|``, clipped and floored at float64 tiny.

    The clip/floor guard log-log slope fits and monotonic-improvement
    checks against float drift that can push ``|<a|b>|`` slightly above 1.
    """
    overlap = np.clip(abs(np.vdot(sv_a, sv_b)), 0.0, 1.0)
    return max(1.0 - overlap, np.finfo(float).tiny)


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
            errors.append(_fidelity_err(sv_exact, sv))

        # Strictly monotone improvement (first-order Trotter error is O(dt)).
        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] < 1e-2


class TestOrder2Correctness:
    """Strang splitting (order=2): global fidelity error scales as O(dt^2)."""

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
            dts.append(T_EVOLVE / step)
            errs.append(_fidelity_err(sv_exact, sv))

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
            dts.append(T_EVOLVE / step)
            errs.append(_fidelity_err(sv_exact, sv))

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
            errs.append(_fidelity_err(sv_exact, sv))

        assert errs[0] > errs[1] > errs[2]
        assert errs[-1] < 1e-3


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


# -----------------------------------------------------------------------
# Cross-SDK coverage
#
# The Suzuki-Trotter sequencing lives in the @qkernel and is lowered by
# the standard ``pauli_evolve`` emitter (CNOT ladder + RZ phase gadget)
# that every backend inherits.  The tests below pin down that the same
# kernel produces physically identical circuits across supported SDKs:
#
#   * Qiskit       — native ``pauli_evolve`` backend.
#   * QURI Parts   — consumes the default decomposition.
#   * CUDA-Q       — emits a Python ``@cudaq.kernel`` source artifact.
#
# CUDA-Q and QURI Parts blocks use ``importorskip`` so the file still
# runs against a qiskit-only environment.
# -----------------------------------------------------------------------


def _qulacs_statevector_from_quri_parts(qp_circuit) -> np.ndarray:
    """Bind parameters (empty) and simulate on Qulacs."""
    from quri_parts.core.state import GeneralCircuitQuantumState
    from quri_parts.qulacs.simulator import evaluate_state_to_vector

    if hasattr(qp_circuit, "parameter_count") and qp_circuit.parameter_count > 0:
        bound = qp_circuit.bind_parameters([0.0] * qp_circuit.parameter_count)
    elif hasattr(qp_circuit, "bind_parameters"):
        bound = qp_circuit.bind_parameters([])
    else:
        bound = qp_circuit
    state = GeneralCircuitQuantumState(bound.qubit_count, bound)
    return np.array(evaluate_state_to_vector(state).vector)


def _cudaq_statevector(cudaq_circuit) -> np.ndarray:
    """Simulate a fully-bound CUDA-Q STATIC artifact via ``cudaq.get_state``."""
    import cudaq

    return np.array(cudaq.get_state(cudaq_circuit.kernel_func))


def _backend_statevector(sdk_transpiler: Any, circuit: Any) -> np.ndarray:
    """Simulate one emitted circuit with its owning backend.

    Args:
        sdk_transpiler (Any): Supported backend fixture case.
        circuit (Any): Backend circuit artifact.

    Returns:
        np.ndarray: Backend statevector.

    Raises:
        AssertionError: If the fixture identifies an unsupported backend.
    """
    if sdk_transpiler.backend_name == "qiskit":
        return _statevector(circuit)
    if sdk_transpiler.backend_name == "quri_parts":
        return _qulacs_statevector_from_quri_parts(circuit)
    if sdk_transpiler.backend_name == "cudaq":
        return _cudaq_statevector(circuit)
    raise AssertionError(f"Unsupported backend {sdk_transpiler.backend_name!r}")


def _rz_count_qiskit(qc) -> int:
    """Count native Pauli rotations inside loops and at top level."""
    count = 0
    for inst in qc.data:
        if inst.operation.name == "for_loop":
            count += _rz_count_qiskit(inst.operation.blocks[0])
        elif inst.operation.name in {"rz", "PauliEvolution"}:
            count += 1
    return count


def _rz_count_quri_parts(qp_circuit) -> int:
    """Count RZ gates in a (bound) QURI Parts circuit."""
    if hasattr(qp_circuit, "parameter_count") and qp_circuit.parameter_count > 0:
        bound = qp_circuit.bind_parameters([0.0] * qp_circuit.parameter_count)
    elif hasattr(qp_circuit, "bind_parameters"):
        bound = qp_circuit.bind_parameters([])
    else:
        bound = qp_circuit
    return sum(1 for g in bound.gates if g.name == "RZ")


def _exp_pauli_count_cudaq(cudaq_circuit) -> int:
    """Count ``exp_pauli(`` call sites in the CUDA-Q kernel source.

    CUDA-Q lowers each ``pauli_evolve`` term to a native ``exp_pauli``
    rather than the ``h`` / ``rz`` gadget the other backends use, so the
    cross-backend per-step invariant is "one ``exp_pauli`` per Pauli
    rotation", which equals the Qiskit / QURI Parts per-step RZ count for
    the single-qubit terms in ``HS_2TERM``.
    """
    return cudaq_circuit.source.count("exp_pauli(")


class TestCrossBackendCompilation:
    """Trotter kernel transpiles across every available SDK.

    Per-step RZ count is formula-dependent and must agree between
    backends: order=1 emits one RZ per term, order=2 uses the merged
    palindrome (three RZs for two terms), and higher orders unfold
    recursively via the standard ``pauli_evolve`` decomposition.
    """

    # Expected RZ counts for a single Trotter step on HS_2TERM.
    _EXPECTED_RZ_PER_STEP = {1: 2, 2: 3, 4: 15}

    @pytest.mark.parametrize("order", [1, 2, 4])
    def test_native_rotation_count(self, sdk_transpiler: Any, order: int) -> None:
        """Every backend emits one native phase rotation per formula term."""
        exe = sdk_transpiler.transpiler.transpile(
            _rabi_trotter,
            bindings={
                "order": order,
                "Hs": HS_2TERM,
                "gamma": T_EVOLVE,
                "step": 1,
            },
        )
        circuit = exe.compiled_quantum[0].circuit
        expected = self._EXPECTED_RZ_PER_STEP[order]
        if sdk_transpiler.backend_name == "qiskit":
            assert circuit.num_qubits == 1
            assert _rz_count_qiskit(circuit) == expected
        elif sdk_transpiler.backend_name == "quri_parts":
            assert circuit.qubit_count == 1
            assert _rz_count_quri_parts(circuit) == expected
        else:
            assert sdk_transpiler.backend_name == "cudaq"
            assert circuit.num_qubits == 1
            assert _exp_pauli_count_cudaq(circuit) == expected


class TestCrossBackendStatevector:
    """Every SDK must reproduce the exact state to within Trotter error.

    For a symmetric S_4 step with 8 Trotter slices on the Rabi
    Hamiltonian, the fidelity error drops below 1e-10 in double
    precision — deep enough that any SDK-specific rotation-sign or
    gate-ordering bug surfaces as a cross-backend mismatch.
    """

    _STEP = 8
    _TOLERANCE = 1e-10

    def test_matches_exact_propagator(self, sdk_transpiler: Any) -> None:
        """Each selected backend reproduces the exact one-qubit propagator."""
        exe = sdk_transpiler.transpiler.transpile(
            _rabi_trotter,
            bindings={
                "order": 4,
                "Hs": HS_2TERM,
                "gamma": T_EVOLVE,
                "step": self._STEP,
            },
        )
        sv = _backend_statevector(
            sdk_transpiler,
            exe.compiled_quantum[0].circuit,
        )
        overlap = abs(np.vdot(_exact_state(), sv))
        assert overlap > 1.0 - self._TOLERANCE


# -----------------------------------------------------------------------
# Sampling distribution check
#
# ``sample()`` counts are the end-to-end observable: a mistake anywhere
# in the pipeline (parameter binding, emitter sign, classical readout)
# shifts the observed |0> / |1> ratio.  We use a scalar-Bit wrapper so
# that the sampling backend reports concrete 0/1 values (measuring a
# one-element ``Vector[Qubit]`` currently yields ``value=None`` in the
# sample result, independent of this module).
# -----------------------------------------------------------------------


@qmc.qkernel
def _rabi_trotter_scalar_bit(
    order: qmc.UInt,
    Hs: qmc.Vector[qmc.Observable],
    gamma: qmc.Float,
    step: qmc.UInt,
) -> qmc.Bit:
    """Scalar-Bit variant of ``_rabi_trotter`` for sampling tests."""
    q = qmc.qubit_array(1, name="q")
    q = trotterized_time_evolution(q, Hs, order, gamma, step)
    return qmc.measure(q[0])


class TestCrossBackendDistribution:
    """Observed sample distribution matches the exact Born probabilities.

    A well-converged Trotter circuit on a single qubit must reproduce
    ``|<1|psi>|^2`` within a few standard deviations of the binomial
    error.  ``STD_TOLERANCE`` is deliberately wide (5 sigma) so the
    test does not flake in CI; a genuine backend bug moves the mean by
    orders of magnitude, not by fractions of a sigma.
    """

    _SHOTS = 4_096
    _STD_TOLERANCE = 5.0
    _ORDER = 4
    _STEP = 8

    def _exact_p_one(self) -> float:
        return float(abs(_exact_state()[1]) ** 2)

    def _sample(self, transpiler) -> tuple[int, int]:
        exe = transpiler.transpile(
            _rabi_trotter_scalar_bit,
            bindings={
                "order": self._ORDER,
                "Hs": HS_2TERM,
                "gamma": T_EVOLVE,
                "step": self._STEP,
            },
        )
        job = exe.sample(transpiler.executor(), bindings={}, shots=self._SHOTS)
        counts = {0: 0, 1: 0}
        for value, count in job.result().results:
            counts[int(value)] += count
        return counts[0], counts[1]

    def _assert_matches_exact(self, n_zero: int, n_one: int) -> None:
        total = n_zero + n_one
        assert total == self._SHOTS
        p_one_exact = self._exact_p_one()
        p_one_obs = n_one / total
        std = np.sqrt(p_one_exact * (1.0 - p_one_exact) / total)
        assert abs(p_one_obs - p_one_exact) < self._STD_TOLERANCE * std, (
            f"observed p(|1>) = {p_one_obs:.4f}, exact = {p_one_exact:.4f}, "
            f"std = {std:.4f}"
        )

    def test_selected_backend(self, sdk_transpiler: Any) -> None:
        """The selected backend's samples match the exact Born probabilities."""
        n0, n1 = self._sample(sdk_transpiler.transpiler)
        self._assert_matches_exact(n0, n1)


# -----------------------------------------------------------------------
# Random Hamiltonian convergence
#
# The handcrafted 1-qubit Rabi problem has enough symmetry that some
# emit bugs cancel.  Drawing a random 2-qubit Hamiltonian (fixed seed)
# with mixed Pauli weights exercises non-diagonal basis changes, CNOT
# ladders, and multi-term commutators that the Suzuki recursion must
# interleave correctly.
# -----------------------------------------------------------------------

_RANDOM_SEED = 42
_RANDOM_N_QUBITS = 2
_RANDOM_T = 0.3


def _random_two_part_hamiltonian(
    seed: int = _RANDOM_SEED,
) -> tuple[list, np.ndarray]:
    """Draw a 2-qubit Hamiltonian split into two non-commuting halves.

    Returns the ``[H_a, H_b]`` list (for use as a ``Vector[Observable]``
    binding) and the exact 4x4 matrix of the full Hamiltonian.
    """
    rng = np.random.default_rng(seed)
    pauli_factories = {
        "X": qm_o.X,
        "Y": qm_o.Y,
        "Z": qm_o.Z,
    }
    pauli_mats = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

    # H_a: single-qubit terms per qubit (deliberately diagonal-free).
    # H_b: two-qubit XX / YY / ZZ couplings between qubits 0 and 1.
    single_letters = ["X", "Y", "Z"]
    pair_letters = ["X", "Y", "Z"]

    h_a = 0 * qm_o.Z(0)  # start from zero Hamiltonian
    h_b = 0 * qm_o.Z(0)
    mat = np.zeros((4, 4), dtype=complex)

    for q in range(_RANDOM_N_QUBITS):
        letter = single_letters[rng.integers(3)]
        coeff = float(rng.uniform(-1.0, 1.0))
        h_a = h_a + coeff * pauli_factories[letter](q)
        # build matrix term: tensor product over both qubits
        term = np.array([[1.0]], dtype=complex)
        for qq in range(_RANDOM_N_QUBITS):
            term = np.kron(term, pauli_mats[letter if qq == q else "I"])
        mat += coeff * term

    letter = pair_letters[rng.integers(3)]
    coeff = float(rng.uniform(-1.0, 1.0))
    h_b = h_b + coeff * pauli_factories[letter](0) * pauli_factories[letter](1)
    mat += coeff * np.kron(pauli_mats[letter], pauli_mats[letter])

    return [h_a, h_b], mat


@qmc.qkernel
def _rabi_trotter_2q(
    order: qmc.UInt,
    Hs: qmc.Vector[qmc.Observable],
    gamma: qmc.Float,
    step: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Two-qubit variant of the trotter wrapper kernel."""
    q = qmc.qubit_array(_RANDOM_N_QUBITS, name="q")
    q = trotterized_time_evolution(q, Hs, order, gamma, step)
    return qmc.measure(q)


class TestRandomHamiltonianConvergence:
    """Fidelity against the exact propagator on a random 2-qubit Hamiltonian."""

    def test_selected_backend_converges(self, sdk_transpiler: Any) -> None:
        """Each backend matches a representative exact two-qubit evolution."""
        hs, mat = _random_two_part_hamiltonian()
        exact = expm(-1j * _RANDOM_T * mat) @ np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=complex
        )
        exe = sdk_transpiler.transpiler.transpile(
            _rabi_trotter_2q,
            bindings={
                "order": 4,
                "Hs": hs,
                "gamma": _RANDOM_T,
                "step": 16,
            },
        )
        state = _backend_statevector(
            sdk_transpiler,
            exe.compiled_quantum[0].circuit,
        )
        state = state.reshape((2,) * _RANDOM_N_QUBITS).transpose().reshape(-1)
        overlap = abs(np.vdot(exact, state))
        assert 1.0 - overlap < 1e-7, (
            f"{sdk_transpiler.backend_name} overlap {overlap:.15f}"
        )
