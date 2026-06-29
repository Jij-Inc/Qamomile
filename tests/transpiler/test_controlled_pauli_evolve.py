"""Cross-backend execution tests for a controlled ``PauliEvolveOp``.

These tests pin the controlled-fallback lowering of ``exp(-i * gamma * H)``
wrapped by ``qmc.control``: the shared walker emits the basis change and CX
ladder uncontrolled and routes only the central ``RZ`` through the
multi-control machinery (``emit_crz`` for one control, the backend's dense
``UnitaryMatrix`` hook for two or more).

QURI Parts is the backend that exercises the shared
``emit_controlled_pauli_evolve`` helper through its recursive fallback
walker (Qiskit uses a native controlled custom gate, CUDA-Q a dedicated
path).  Correctness is checked against Qiskit's native lowering as the
reference: the two statevectors must agree up to global phase, and the two
expectation values must agree to ``atol=1e-8``.  Hamiltonians are kept
single-term or pairwise-commuting so ``exp(-i * gamma * H)`` is realised
exactly by both backends and the comparison is independent of Trotter
ordering.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from qiskit import (
    QuantumCircuit,  # noqa: E402
    transpile as qk_transpile,  # noqa: E402
)
from qiskit_aer import AerSimulator  # noqa: E402

import qamomile.circuit as qmc  # noqa: E402
import qamomile.observable as qm_o  # noqa: E402
from qamomile.circuit.transpiler.errors import EmitError  # noqa: E402
from qamomile.qiskit import QiskitTranspiler  # noqa: E402

pytestmark = pytest.mark.quri_parts


def _quri_parts_transpiler() -> Any:
    """Return a ``QuriPartsTranspiler`` or skip when QURI Parts is absent.

    Returns:
        Any: A fresh ``QuriPartsTranspiler`` instance.
    """
    pytest.importorskip("quri_parts")
    pytest.importorskip("quri_parts.qulacs")
    from qamomile.quri_parts import QuriPartsTranspiler

    return QuriPartsTranspiler()


# ---------------------------------------------------------------------------
# Statevector helpers (mirroring tests/circuit/algorithm/test_trotter.py)
# ---------------------------------------------------------------------------


def _qiskit_statevector(circuit: Any) -> np.ndarray:
    """Strip measurements, lower to basis gates, and simulate on Aer.

    Args:
        circuit (Any): Compiled Qiskit ``QuantumCircuit`` from
            ``executable.compiled_quantum[0].circuit``.

    Returns:
        np.ndarray: Complex amplitudes of the prepared state.
    """
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


def _quri_statevector(qp_circuit: Any) -> np.ndarray:
    """Bind any (empty) parameters and simulate a QURI Parts circuit on Qulacs.

    Args:
        qp_circuit (Any): Compiled QURI Parts circuit from
            ``executable.compiled_quantum[0].circuit``.

    Returns:
        np.ndarray: Complex amplitudes of the prepared state.
    """
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


def _fidelity_err(sv_a: np.ndarray, sv_b: np.ndarray) -> float:
    """Return ``1 - |<a|b>|``, the global-phase-insensitive state distance.

    Args:
        sv_a (np.ndarray): First statevector.
        sv_b (np.ndarray): Second statevector.

    Returns:
        float: Fidelity error in ``[0, 1]`` (``0`` iff equal up to phase).
    """
    overlap = np.clip(abs(np.vdot(sv_a, sv_b)), 0.0, 1.0)
    return max(1.0 - overlap, 0.0)


# ---------------------------------------------------------------------------
# Kernels under test
# ---------------------------------------------------------------------------


@qmc.qkernel
def _evolve_layer(
    q: qmc.Vector[qmc.Qubit], ham: qmc.Observable, gamma: qmc.Float
) -> qmc.Vector[qmc.Qubit]:
    """Leaf sub-kernel applying ``exp(-i * gamma * H)`` to a register."""
    return qmc.pauli_evolve(q, ham, gamma)


# Each kernel prepares the control(s) in superposition (``H``) so both the
# control=0 (identity) and control=1 (evolution) branches are exercised, and
# flips the first target qubit so the evolution acts on a non-trivial state.


@qmc.qkernel
def _cpe_run_c1t1(
    ham: qmc.Observable, gamma: qmc.Float, obs: qmc.Observable
) -> qmc.Float:
    """Expval after a 1-control evolution over a 1-qubit target."""
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.h(q[0])
    q[1] = qmc.x(q[1])
    ce = qmc.control(_evolve_layer)
    q[0], target = ce(q[0], q[1:2], ham=ham, gamma=gamma)
    q[1:2] = target
    return qmc.expval(q, obs)


@qmc.qkernel
def _cpe_run_c1t2(
    ham: qmc.Observable, gamma: qmc.Float, obs: qmc.Observable
) -> qmc.Float:
    """Expval after a 1-control evolution over a 2-qubit target."""
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.h(q[0])
    q[1] = qmc.x(q[1])
    ce = qmc.control(_evolve_layer)
    q[0], target = ce(q[0], q[1:3], ham=ham, gamma=gamma)
    q[1:3] = target
    return qmc.expval(q, obs)


@qmc.qkernel
def _cpe_run_c2t2(
    ham: qmc.Observable, gamma: qmc.Float, obs: qmc.Observable
) -> qmc.Float:
    """Expval after a 2-control evolution over a 2-qubit target."""
    q = qmc.qubit_array(4, "q")
    q[0] = qmc.h(q[0])
    q[1] = qmc.h(q[1])
    q[2] = qmc.x(q[2])
    ce = qmc.control(_evolve_layer, num_controls=2)
    q[0], q[1], target = ce(q[0], q[1], q[2:4], ham=ham, gamma=gamma)
    q[2:4] = target
    return qmc.expval(q, obs)


@qmc.qkernel
def _cpe_sample_c1t1(ham: qmc.Observable, gamma: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Sample a 1-control evolution over a 1-qubit target."""
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.h(q[0])
    q[1] = qmc.x(q[1])
    ce = qmc.control(_evolve_layer)
    q[0], target = ce(q[0], q[1:2], ham=ham, gamma=gamma)
    q[1:2] = target
    return qmc.measure(q)


@qmc.qkernel
def _cpe_sample_c1t2(ham: qmc.Observable, gamma: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Sample a 1-control evolution over a 2-qubit target."""
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.h(q[0])
    q[1] = qmc.x(q[1])
    ce = qmc.control(_evolve_layer)
    q[0], target = ce(q[0], q[1:3], ham=ham, gamma=gamma)
    q[1:3] = target
    return qmc.measure(q)


@qmc.qkernel
def _cpe_sample_c1t3(ham: qmc.Observable, gamma: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Sample a 1-control evolution over a 3-qubit target."""
    q = qmc.qubit_array(4, "q")
    q[0] = qmc.h(q[0])
    q[1] = qmc.x(q[1])
    ce = qmc.control(_evolve_layer)
    q[0], target = ce(q[0], q[1:4], ham=ham, gamma=gamma)
    q[1:4] = target
    return qmc.measure(q)


@qmc.qkernel
def _cpe_sample_c2t1(ham: qmc.Observable, gamma: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Sample a 2-control evolution over a 1-qubit target."""
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.h(q[0])
    q[1] = qmc.h(q[1])
    q[2] = qmc.x(q[2])
    ce = qmc.control(_evolve_layer, num_controls=2)
    q[0], q[1], target = ce(q[0], q[1], q[2:3], ham=ham, gamma=gamma)
    q[2:3] = target
    return qmc.measure(q)


@qmc.qkernel
def _cpe_sample_c2t2(ham: qmc.Observable, gamma: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Sample a 2-control evolution over a 2-qubit target."""
    q = qmc.qubit_array(4, "q")
    q[0] = qmc.h(q[0])
    q[1] = qmc.h(q[1])
    q[2] = qmc.x(q[2])
    ce = qmc.control(_evolve_layer, num_controls=2)
    q[0], q[1], target = ce(q[0], q[1], q[2:4], ham=ham, gamma=gamma)
    q[2:4] = target
    return qmc.measure(q)


@qmc.qkernel
def _cpe_two_hamiltonians(
    ham_a: qmc.Observable, ham_b: qmc.Observable, gamma: qmc.Float
) -> qmc.Vector[qmc.Bit]:
    """Two controlled evolutions in one kernel, each with its own Hamiltonian.

    Exercises that distinct ``Observable`` bindings resolve independently:
    each ``PauliEvolveOp`` carries its own ``observable`` operand, so the two
    layers must use ``ham_a`` and ``ham_b`` respectively rather than collapsing
    to one.
    """
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.h(q[0])
    ce = qmc.control(_evolve_layer)
    q[0], target = ce(q[0], q[1:2], ham=ham_a, gamma=gamma)
    q[1:2] = target
    q[0], target = ce(q[0], q[1:2], ham=ham_b, gamma=gamma)
    q[1:2] = target
    return qmc.measure(q)


@qmc.qkernel
def _cpe_vector_hamiltonian(
    hams: qmc.Vector[qmc.Observable], gamma: qmc.Float
) -> qmc.Vector[qmc.Bit]:
    """Controlled evolution whose Hamiltonian is an element of a Vector[Observable].

    Exercises that a vector-of-observables element (``hams[1]``, a non-zero
    index) resolves correctly through ``resolve_bound_value`` (``index_array``)
    in the controlled path.
    """
    q = qmc.qubit_array(2, "q")
    q[0] = qmc.h(q[0])
    ce = qmc.control(_evolve_layer)
    q[0], target = ce(q[0], q[1:2], ham=hams[1], gamma=gamma)
    q[1:2] = target
    return qmc.measure(q)


# ---------------------------------------------------------------------------
# Hamiltonian builders (seeded; single-term or pairwise-commuting only)
# ---------------------------------------------------------------------------


def _single_x(rng: np.random.Generator) -> qm_o.Hamiltonian:
    """Build single-qubit ``c * X(0)`` with a random real coefficient.

    Args:
        rng (np.random.Generator): Seeded RNG for the coefficient.

    Returns:
        qm_o.Hamiltonian: The one-term Hamiltonian.
    """
    return rng.uniform(0.3, 1.5) * qm_o.X(0)


def _single_y(rng: np.random.Generator) -> qm_o.Hamiltonian:
    """Build single-qubit ``c * Y(0)`` with a random real coefficient.

    Args:
        rng (np.random.Generator): Seeded RNG for the coefficient.

    Returns:
        qm_o.Hamiltonian: The one-term Hamiltonian.
    """
    return rng.uniform(0.3, 1.5) * qm_o.Y(0)


def _single_z(rng: np.random.Generator) -> qm_o.Hamiltonian:
    """Build single-qubit ``c * Z(0)`` with a random real coefficient.

    Args:
        rng (np.random.Generator): Seeded RNG for the coefficient.

    Returns:
        qm_o.Hamiltonian: The one-term Hamiltonian.
    """
    return rng.uniform(0.3, 1.5) * qm_o.Z(0)


def _two_xx(rng: np.random.Generator) -> qm_o.Hamiltonian:
    """Build the single two-qubit term ``c * X(0) X(1)`` (CX-ladder gadget).

    Args:
        rng (np.random.Generator): Seeded RNG for the coefficient.

    Returns:
        qm_o.Hamiltonian: The one-term Hamiltonian.
    """
    return rng.uniform(0.3, 1.5) * qm_o.X(0) * qm_o.X(1)


def _two_xy(rng: np.random.Generator) -> qm_o.Hamiltonian:
    """Build the single mixed two-qubit term ``c * X(0) Y(1)``.

    Args:
        rng (np.random.Generator): Seeded RNG for the coefficient.

    Returns:
        qm_o.Hamiltonian: The one-term Hamiltonian.
    """
    return rng.uniform(0.3, 1.5) * qm_o.X(0) * qm_o.Y(1)


def _two_heisenberg(rng: np.random.Generator) -> qm_o.Hamiltonian:
    """Build pairwise-commuting ``a XX + b YY + c ZZ`` (exact, order-independent).

    Args:
        rng (np.random.Generator): Seeded RNG for the three coefficients.

    Returns:
        qm_o.Hamiltonian: The three-term commuting Hamiltonian.
    """
    a, b, c = rng.uniform(0.3, 1.5, size=3)
    return (
        a * qm_o.X(0) * qm_o.X(1)
        + b * qm_o.Y(0) * qm_o.Y(1)
        + c * qm_o.Z(0) * qm_o.Z(1)
    )


def _two_commuting_z(rng: np.random.Generator) -> qm_o.Hamiltonian:
    """Build pairwise-commuting ``a Z0 + b Z1 + c Z0 Z1`` (diagonal, exact).

    Args:
        rng (np.random.Generator): Seeded RNG for the three coefficients.

    Returns:
        qm_o.Hamiltonian: The three-term diagonal Hamiltonian.
    """
    a, b, c = rng.uniform(0.3, 1.5, size=3)
    return a * qm_o.Z(0) + b * qm_o.Z(1) + c * qm_o.Z(0) * qm_o.Z(1)


def _three_xyz(rng: np.random.Generator) -> qm_o.Hamiltonian:
    """Build the single mixed three-qubit term ``c * X(0) Y(1) Z(2)``.

    Args:
        rng (np.random.Generator): Seeded RNG for the coefficient.

    Returns:
        qm_o.Hamiltonian: The one-term Hamiltonian.
    """
    return rng.uniform(0.3, 1.5) * qm_o.X(0) * qm_o.Y(1) * qm_o.Z(2)


def _single_x_plus_const(rng: np.random.Generator) -> qm_o.Hamiltonian:
    """Build ``a * X(0) + d`` with a nonzero identity (constant) offset.

    The constant ``d`` is a global phase for uncontrolled evolution but an
    observable relative phase once controlled, so it exercises the
    controlled constant-term path.

    Args:
        rng (np.random.Generator): Seeded RNG for the coefficient and offset.

    Returns:
        qm_o.Hamiltonian: A single-Pauli Hamiltonian with a constant offset.
    """
    return rng.uniform(0.3, 1.5) * qm_o.X(0) + rng.uniform(0.3, 1.5)


def _two_heisenberg_plus_const(rng: np.random.Generator) -> qm_o.Hamiltonian:
    """Build the Heisenberg Hamiltonian plus a nonzero constant offset.

    Args:
        rng (np.random.Generator): Seeded RNG for the coefficients and offset.

    Returns:
        qm_o.Hamiltonian: A commuting multi-term Hamiltonian with a constant.
    """
    return _two_heisenberg(rng) + rng.uniform(0.3, 1.5)


def _pure_const(rng: np.random.Generator) -> qm_o.Hamiltonian:
    """Build a single-qubit Hamiltonian whose only nonzero part is a constant.

    The ``X(0) * 0.0`` keeps the register width at one qubit while leaving no
    Pauli term, so the controlled evolution emits only the relative phase.

    Args:
        rng (np.random.Generator): Seeded RNG for the constant offset.

    Returns:
        qm_o.Hamiltonian: A one-qubit Hamiltonian with only a constant term.
    """
    return qm_o.X(0) * 0.0 + rng.uniform(0.3, 1.5)


# (id, run_kernel | None, sample_kernel, n_controls, n_target, ham_builder)
_CASES: list[tuple[str, Any, Any, int, int, Callable[[np.random.Generator], Any]]] = [
    ("c1t1_X", _cpe_run_c1t1, _cpe_sample_c1t1, 1, 1, _single_x),
    ("c1t1_Y", _cpe_run_c1t1, _cpe_sample_c1t1, 1, 1, _single_y),
    ("c1t1_Z", _cpe_run_c1t1, _cpe_sample_c1t1, 1, 1, _single_z),
    ("c1t2_XX", _cpe_run_c1t2, _cpe_sample_c1t2, 1, 2, _two_xx),
    ("c1t2_XY", _cpe_run_c1t2, _cpe_sample_c1t2, 1, 2, _two_xy),
    ("c1t2_heisenberg", _cpe_run_c1t2, _cpe_sample_c1t2, 1, 2, _two_heisenberg),
    ("c1t2_commuting_z", _cpe_run_c1t2, _cpe_sample_c1t2, 1, 2, _two_commuting_z),
    ("c1t3_XYZ", None, _cpe_sample_c1t3, 1, 3, _three_xyz),
    ("c2t1_X", None, _cpe_sample_c2t1, 2, 1, _single_x),
    ("c2t1_Z", None, _cpe_sample_c2t1, 2, 1, _single_z),
    ("c2t2_XX", _cpe_run_c2t2, _cpe_sample_c2t2, 2, 2, _two_xx),
    ("c2t2_heisenberg", _cpe_run_c2t2, _cpe_sample_c2t2, 2, 2, _two_heisenberg),
    # Constant (identity) Hamiltonian terms: dropped uncontrolled (global
    # phase) but an observable relative phase once controlled.
    ("c1t1_X_const", _cpe_run_c1t1, _cpe_sample_c1t1, 1, 1, _single_x_plus_const),
    ("c2t1_X_const", None, _cpe_sample_c2t1, 2, 1, _single_x_plus_const),
    (
        "c1t2_heisenberg_const",
        _cpe_run_c1t2,
        _cpe_sample_c1t2,
        1,
        2,
        _two_heisenberg_plus_const,
    ),
    (
        "c2t2_heisenberg_const",
        _cpe_run_c2t2,
        _cpe_sample_c2t2,
        2,
        2,
        _two_heisenberg_plus_const,
    ),
    # Pure constant (no Pauli term): the controlled evolution is purely the
    # relative phase on the control(s).
    ("c1t1_pure_const", _cpe_run_c1t1, _cpe_sample_c1t1, 1, 1, _pure_const),
    ("c2t1_pure_const", None, _cpe_sample_c2t1, 2, 1, _pure_const),
]

_RUN_CASES = [c for c in _CASES if c[1] is not None]


def _random_observable(rng: np.random.Generator, n: int) -> qm_o.Hamiltonian:
    """Build a random Hermitian ``sum_i (a_i Z_i + b_i X_i)`` over ``n`` qubits.

    Args:
        rng (np.random.Generator): Seeded RNG for the coefficients.
        n (int): Number of qubits the observable spans (``>= 1``).

    Returns:
        qm_o.Hamiltonian: A random Hermitian observable with one ``Z`` and
            one ``X`` term per qubit.
    """
    obs = rng.uniform(-1.0, 1.0) * qm_o.Z(0) + rng.uniform(-1.0, 1.0) * qm_o.X(0)
    for i in range(1, n):
        obs = obs + rng.uniform(-1.0, 1.0) * qm_o.Z(i)
        obs = obs + rng.uniform(-1.0, 1.0) * qm_o.X(i)
    return obs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
def test_controlled_pauli_evolve_statevector_matches_qiskit(
    case: tuple[str, Any, Any, int, int, Any], seed: int
) -> None:
    """QURI Parts controlled Pauli evolution matches Qiskit's statevector.

    Validates the full controlled unitary (not just one observable) up to
    global phase across single-Pauli, multi-term, and X/Y/Z-mixed
    Hamiltonians, 1 to 3 target qubits, and 1 or 2 controls.
    """
    _id, _run_kernel, sample_kernel, _n_controls, _n_target, ham_builder = case
    rng = np.random.default_rng(seed)
    ham = ham_builder(rng)
    gamma = float(rng.uniform(0.2, 2.6))
    bindings = {"ham": ham, "gamma": gamma}

    qk_exe = QiskitTranspiler().transpile(sample_kernel, bindings=bindings)
    qp_exe = _quri_parts_transpiler().transpile(sample_kernel, bindings=bindings)

    sv_qk = _qiskit_statevector(qk_exe.compiled_quantum[0].circuit)
    sv_qp = _quri_statevector(qp_exe.compiled_quantum[0].circuit)

    assert _fidelity_err(sv_qk, sv_qp) < 1e-9


@pytest.mark.parametrize("gamma", [0.0, math.pi, 2 * math.pi])
@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
def test_controlled_pauli_evolve_boundary_gamma_matches_qiskit(
    case: tuple[str, Any, Any, int, int, Any], gamma: float
) -> None:
    """Boundary evolution times (0, pi, 2*pi) match Qiskit's statevector."""
    _id, _run_kernel, sample_kernel, _n_controls, _n_target, ham_builder = case
    ham = ham_builder(np.random.default_rng(7))
    bindings = {"ham": ham, "gamma": gamma}

    qk_exe = QiskitTranspiler().transpile(sample_kernel, bindings=bindings)
    qp_exe = _quri_parts_transpiler().transpile(sample_kernel, bindings=bindings)

    sv_qk = _qiskit_statevector(qk_exe.compiled_quantum[0].circuit)
    sv_qp = _quri_statevector(qp_exe.compiled_quantum[0].circuit)

    assert _fidelity_err(sv_qk, sv_qp) < 1e-9


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
@pytest.mark.parametrize("case", _RUN_CASES, ids=[c[0] for c in _RUN_CASES])
def test_controlled_pauli_evolve_expval_matches_qiskit(
    case: tuple[str, Any, Any, int, int, Any], seed: int
) -> None:
    """QURI Parts and Qiskit agree on the controlled-evolution expectation value.

    Exercises the estimator primitive on both backends (distinct from the
    statevector path, which reads the compiled circuit directly).
    """
    _id, run_kernel, _sample_kernel, n_controls, n_target, ham_builder = case
    rng = np.random.default_rng(seed)
    ham = ham_builder(rng)
    gamma = float(rng.uniform(0.2, 2.6))
    obs = _random_observable(rng, n_controls + n_target)
    bindings = {"ham": ham, "gamma": gamma, "obs": obs}

    qk_tr = QiskitTranspiler()
    qp_tr = _quri_parts_transpiler()
    qk_val = (
        qk_tr.transpile(run_kernel, bindings=bindings).run(qk_tr.executor()).result()
    )
    qp_val = (
        qp_tr.transpile(run_kernel, bindings=bindings).run(qp_tr.executor()).result()
    )

    assert np.isclose(qk_val, qp_val, atol=1e-8), f"qiskit={qk_val}, quri={qp_val}"


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize(
    "case",
    [
        ("c1t1_X", _cpe_sample_c1t1, 1, 1, _single_x),
        ("c1t2_heisenberg", _cpe_sample_c1t2, 1, 2, _two_heisenberg),
        ("c2t2_XX", _cpe_sample_c2t2, 2, 2, _two_xx),
    ],
    ids=lambda c: c[0],
)
def test_controlled_pauli_evolve_sample_matches_qiskit(
    case: tuple[str, Any, int, int, Any], seed: int
) -> None:
    """QURI Parts sampling matches Qiskit sampling within shot noise.

    Exercises the sampler primitive on both backends. Both report counts in
    identical kernel-order bit tuples, so the two empirical distributions are
    compared directly (no endianness conversion).
    """
    _id, sample_kernel, _n_controls, _n_target, ham_builder = case
    rng = np.random.default_rng(seed)
    ham = ham_builder(rng)
    gamma = float(rng.uniform(0.2, 2.6))
    bindings = {"ham": ham, "gamma": gamma}
    shots = 16384

    def _probs(transpiler: Any) -> dict[tuple[int, ...], float]:
        """Sample ``sample_kernel`` and return kernel-order probabilities.

        Args:
            transpiler (Any): Backend transpiler exposing ``transpile`` /
                ``executor``.

        Returns:
            dict[tuple[int, ...], float]: Empirical probability per bit tuple.
        """
        result = (
            transpiler.transpile(sample_kernel, bindings=bindings)
            .sample(transpiler.executor(), shots=shots)
            .result()
        )
        counts: dict[tuple[int, ...], int] = {}
        for bits, count in result.results:
            key = tuple(int(b) for b in bits)
            counts[key] = counts.get(key, 0) + int(count)
        return {key: c / shots for key, c in counts.items()}

    qk_probs = _probs(QiskitTranspiler())
    qp_probs = _probs(_quri_parts_transpiler())

    for bits in set(qk_probs) | set(qp_probs):
        assert abs(qk_probs.get(bits, 0.0) - qp_probs.get(bits, 0.0)) < 0.05, (
            f"{bits}: qiskit={qk_probs.get(bits, 0.0)}, quri={qp_probs.get(bits, 0.0)}"
        )


@pytest.mark.parametrize(
    ("kernel", "ham"),
    [
        (_cpe_sample_c1t1, qm_o.X(0)),
        (_cpe_sample_c2t2, qm_o.X(0) * qm_o.X(1)),
        # Pure constant: the constant-term phase scaling (gamma * constant),
        # not the per-term RZ, is what cannot be expressed parametrically.
        (_cpe_sample_c1t1, qm_o.X(0) * 0.0 + 0.8),
    ],
    ids=["one-control", "two-controls", "constant-term"],
)
def test_controlled_pauli_evolve_parametric_gamma_raises_on_quri_parts(
    kernel: Any, ham: qm_o.Hamiltonian
) -> None:
    """Runtime-parametric gamma raises a clean ``EmitError`` on QURI Parts.

    The central RZ angle is ``2 * coeff * gamma`` and the constant-term phase
    is ``-gamma * constant``; QURI Parts' Rust-backed runtime ``Parameter``
    exposes no Python arithmetic, so neither scaling can be expressed (the
    same pre-existing limitation as uncontrolled ``pauli_evolve``). For two or
    more controls the dense ``UnitaryMatrix`` path independently requires a
    compile-time-numeric angle. Either way the controlled lowering surfaces a
    clear ``EmitError`` at transpile time rather than a raw ``TypeError``.
    """
    qp_tr = _quri_parts_transpiler()
    with pytest.raises(EmitError):
        qp_tr.transpile(kernel, bindings={"ham": ham}, parameters=["gamma"])


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_controlled_pauli_evolve_distinct_hamiltonians_resolve_independently(
    seed: int,
) -> None:
    """Two controlled evolutions with different Hamiltonians match Qiskit.

    Guards that each ``PauliEvolveOp`` resolves its own ``observable`` operand:
    if the two layers collapsed to a single binding the QURI Parts statevector
    would diverge from Qiskit's, which composes the two distinct controlled
    evolutions.
    """
    rng = np.random.default_rng(seed)
    ham_a = _single_x(rng) + rng.uniform(0.3, 1.5)  # X(0) + constant
    ham_b = rng.uniform(0.3, 1.5) * qm_o.Z(0)
    gamma = float(rng.uniform(0.2, 2.6))
    bindings = {"ham_a": ham_a, "ham_b": ham_b, "gamma": gamma}

    qk_exe = QiskitTranspiler().transpile(_cpe_two_hamiltonians, bindings=bindings)
    qp_exe = _quri_parts_transpiler().transpile(
        _cpe_two_hamiltonians, bindings=bindings
    )
    sv_qk = _qiskit_statevector(qk_exe.compiled_quantum[0].circuit)
    sv_qp = _quri_statevector(qp_exe.compiled_quantum[0].circuit)

    assert _fidelity_err(sv_qk, sv_qp) < 1e-9


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_controlled_pauli_evolve_vector_observable_element(seed: int) -> None:
    """A Vector[Observable] element resolves correctly in the controlled path.

    Binds a ``Vector[Observable]`` and evolves under ``hams[1]`` (a non-zero
    index) inside ``qmc.control``; the QURI Parts statevector must match
    Qiskit, confirming ``resolve_bound_value`` indexes the bound vector
    correctly under control just as the uncontrolled path does.
    """
    rng = np.random.default_rng(seed)
    hams = [
        rng.uniform(0.3, 1.5) * qm_o.X(0),
        rng.uniform(0.3, 1.5) * qm_o.Z(0) + rng.uniform(0.3, 1.5),  # the bound one
    ]
    gamma = float(rng.uniform(0.2, 2.6))
    bindings = {"hams": hams, "gamma": gamma}

    qk_exe = QiskitTranspiler().transpile(_cpe_vector_hamiltonian, bindings=bindings)
    qp_exe = _quri_parts_transpiler().transpile(
        _cpe_vector_hamiltonian, bindings=bindings
    )
    sv_qk = _qiskit_statevector(qk_exe.compiled_quantum[0].circuit)
    sv_qp = _quri_statevector(qp_exe.compiled_quantum[0].circuit)

    assert _fidelity_err(sv_qk, sv_qp) < 1e-9
