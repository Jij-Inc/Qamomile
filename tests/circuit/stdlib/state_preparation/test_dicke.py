"""Tests for qamomile/circuit/stdlib/state_preparation/dicke.py primitives."""

import importlib.util
import re

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.stdlib.state_preparation import dicke_state_composition_schedule
from qamomile.circuit.stdlib.state_preparation.dicke import (
    prepare_dicke,
    scs_gate_2q,
    scs_gate_3q,
)

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

# Optional backends carry their pytest marker so the dedicated
# ``-m quri_parts`` / ``-m cudaq`` runs select them and default runs skip them.
BACKENDS: list = []
try:
    import qiskit  # noqa: F401

    from qamomile.qiskit.transpiler import QiskitTranspiler

    BACKENDS.append(pytest.param("qiskit", QiskitTranspiler, id="qiskit"))
except ImportError:
    pass
try:
    import quri_parts  # noqa: F401

    from qamomile.quri_parts.transpiler import QuriPartsTranspiler

    BACKENDS.append(
        pytest.param(
            "quri_parts",
            QuriPartsTranspiler,
            marks=pytest.mark.quri_parts,
            id="quri_parts",
        )
    )
except ImportError:
    pass
# cudaq imports ``torch`` at import time, whose OpenMP runtime segfaults
# alongside qiskit-aer, so the collection-isolation guard
# (tests/_cudaq_isolation.py) forbids a module-level ``import cudaq``.
# ``importlib.util.find_spec`` probes availability without loading the runtime
# (importing ``CudaqTranspiler`` alone does not pull in cudaq).
if importlib.util.find_spec("cudaq") is not None:
    from qamomile.cudaq.transpiler import CudaqTranspiler

    BACKENDS.append(
        pytest.param("cudaq", CudaqTranspiler, marks=pytest.mark.cudaq, id="cudaq")
    )

if not BACKENDS:
    pytest.skip("No quantum backend available", allow_module_level=True)

# ---------------------------------------------------------------------------
# Per-backend gate-count helpers
# ---------------------------------------------------------------------------


def _qiskit_gate_counts(exe) -> dict[str, int]:
    """Counts gates in the transpiled Qiskit circuit by name."""
    qc = exe.compiled_quantum[0].circuit
    counts: dict[str, int] = {}
    for inst in qc.data:
        name = inst.operation.name
        counts[name] = counts.get(name, 0) + 1
    return counts


_QURI_PARTS_CANONICAL: dict[str, str] = {
    "H": "h",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "S": "s",
    "Sdag": "sdg",
    "T": "t",
    "Tdag": "tdg",
    "CNOT": "cx",
    "CZ": "cz",
    "SWAP": "swap",
    "RX": "rx",
    "ParametricRX": "rx",
    "RY": "ry",
    "ParametricRY": "ry",
    "RZ": "rz",
    "ParametricRZ": "rz",
    "PauliRotation": "rzz",
    "ParametricPauliRotation": "rzz",
}


def _quri_parts_gate_counts(exe) -> dict[str, int]:
    """Counts gates in the transpiled Quri Parts circuit by canonical name."""
    circuit = exe.compiled_quantum[0].circuit
    counts: dict[str, int] = {}
    for gate in circuit.gates:
        canon = _QURI_PARTS_CANONICAL.get(gate.name, gate.name.lower())
        counts[canon] = counts.get(canon, 0) + 1
    return counts


_CUDAQ_PATTERNS: dict[str, re.Pattern] = {
    "cx": re.compile(r"x\.ctrl\("),
    "rx": re.compile(r"\brx\("),
    "ry": re.compile(r"\bry\("),
    "rz": re.compile(r"\brz\("),
    "h": re.compile(r"\bh\("),
    "x": re.compile(r"\bx\("),
}


def _cudaq_gate_counts(exe) -> dict[str, int]:
    """Counts gates in the transpiled Cudaq circuit by name."""
    source = exe.compiled_quantum[0].circuit.source
    return {name: len(pat.findall(source)) for name, pat in _CUDAQ_PATTERNS.items()}


# ---------------------------------------------------------------------------
# Wrapper qkernels (needed to transpile sub-functions with concrete bindings)
# ---------------------------------------------------------------------------


@qmc.qkernel
def _wrap_scs_gate_2q(
    n: qmc.UInt,
    t: qmc.UInt,
    c: qmc.UInt,
    theta: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    """Exercises ``scs_gate_2q`` on an ``n``-qubit ``|0...0>`` register through the sampling path.

    Args:
        n (qmc.UInt): Number of qubits in the register.
        t (qmc.UInt): Target qubit index.
        c (qmc.UInt): Control qubit index.
        theta (qmc.Float): SCS rotation angle.

    Returns:
        qmc.Vector[qmc.Bit]: Measurement outcomes of the full register.
    """
    q = qmc.qubit_array(n, name="q")
    q = scs_gate_2q(q, t, c, theta)
    return qmc.measure(q)


@qmc.qkernel
def _wrap_scs_gate_3q(
    n: qmc.UInt,
    t: qmc.UInt,
    c1: qmc.UInt,
    c2: qmc.UInt,
    theta: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    """Exercises ``scs_gate_3q`` on an ``n``-qubit ``|0...0>`` register through the sampling path.

    Args:
        n (qmc.UInt): Number of qubits in the register.
        t (qmc.UInt): Target qubit index.
        c1 (qmc.UInt): First control qubit index.
        c2 (qmc.UInt): Second control qubit index.
        theta (qmc.Float): SCS rotation angle.

    Returns:
        qmc.Vector[qmc.Bit]: Measurement outcomes of the full register.
    """
    q = qmc.qubit_array(n, name="q")
    q = scs_gate_3q(q, t, c1, c2, theta)
    return qmc.measure(q)


@qmc.qkernel
def _wrap_prepare_dicke(
    n: qmc.UInt,
    initial_ones: qmc.Vector[qmc.UInt],
    schedule: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """Exercises ``prepare_dicke`` through the sampling path.

    Args:
        n (qmc.UInt): Number of qubits in the register.
        initial_ones (qmc.Vector[qmc.UInt]): Indices of the qubits initialized to ``|1>``.
        schedule (qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float]): Ordered SCS gate schedule for the Dicke preparation.

    Returns:
        qmc.Vector[qmc.Bit]: Measurement outcomes of the full register.
    """
    q = prepare_dicke(n, initial_ones, schedule)
    return qmc.measure(q)


@qmc.qkernel
def _wrap_prepare_dicke_expval(
    n: qmc.UInt,
    initial_ones: qmc.Vector[qmc.UInt],
    schedule: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
    hamiltonian: qmc.Observable,
) -> qmc.Float:
    """Exercises ``prepare_dicke`` through the expectation-value path.

    Args:
        n (qmc.UInt): Number of qubits in the register.
        initial_ones (qmc.Vector[qmc.UInt]): Indices of the qubits initialized to ``|1>``.
        schedule (qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float]): Ordered SCS gate schedule for the Dicke preparation.
        hamiltonian (qmc.Observable): Observable whose expectation value is returned.

    Returns:
        qmc.Float: Expectation value of ``hamiltonian`` on the prepared state.
    """
    q = prepare_dicke(n, initial_ones, schedule)
    return qmc.expval(q, hamiltonian)


# ---------------------------------------------------------------------------
# Primitive tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
def test_scs_gate_2q_has_entangling_gates(name, TranspilerCls):
    """Tests that the 2-qubit SCS gate emits at least one entangling (CX) gate.

    Exact gate counts are deliberately not asserted here because backend
    transpilers may re-decompose the circuit differently across versions.
    Functional correctness is covered by test_prepare_dicke_z_sum_matches_analytic.
    """
    transpiler = TranspilerCls()
    exe = transpiler.transpile(
        _wrap_scs_gate_2q,
        bindings={"n": 2, "t": 0, "c": 1, "theta": 0.3},
    )

    match name:
        case "qiskit":
            counts = _qiskit_gate_counts(exe)
            assert counts.get("cx", 0) >= 1
        case "quri_parts":
            counts = _quri_parts_gate_counts(exe)
            assert counts.get("cx", 0) >= 1
        case "cudaq":
            counts = _cudaq_gate_counts(exe)
            assert counts.get("cx", 0) >= 1


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
def test_scs_gate_3q_has_entangling_gates(name, TranspilerCls):
    """Tests that the 3-qubit SCS gate emits at least one entangling (CX) gate.

    Exact gate counts are deliberately not asserted here because backend
    transpilers may re-decompose the circuit differently across versions.
    Functional correctness is covered by test_prepare_dicke_z_sum_matches_analytic.
    """
    transpiler = TranspilerCls()
    exe = transpiler.transpile(
        _wrap_scs_gate_3q,
        bindings={"n": 3, "t": 0, "c1": 1, "c2": 2, "theta": 0.3},
    )

    match name:
        case "qiskit":
            counts = _qiskit_gate_counts(exe)
            assert counts.get("cx", 0) >= 1
        case "quri_parts":
            counts = _quri_parts_gate_counts(exe)
            assert counts.get("cx", 0) >= 1
        case "cudaq":
            counts = _cudaq_gate_counts(exe)
            assert counts.get("cx", 0) >= 1


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
def test_prepare_dicke_applies_basis_initialization_and_entangling_gates(
    name, TranspilerCls
):
    """Tests that prepare_dicke applies X gates for initial Hamming weight and at least one entangling gate.

    The number of X gates is exact (one per initial |1> qubit), but CX counts are
    asserted only as a lower bound because backend transpilers may re-decompose the
    SCS blocks differently across versions. Functional correctness is covered by
    test_prepare_dicke_z_sum_matches_analytic and test_prepare_dicke_sample_preserves_hamming_weight.
    """
    initial_ones, schedule = dicke_state_composition_schedule(
        n_qubits=3, block_size=3, hamming_weight=2
    )

    transpiler = TranspilerCls()
    exe = transpiler.transpile(
        _wrap_prepare_dicke,
        bindings={
            "n": 3,
            "initial_ones": initial_ones,
            "schedule": schedule,
        },
    )

    expected_x = len(initial_ones)

    match name:
        case "qiskit":
            counts = _qiskit_gate_counts(exe)
            assert counts.get("x", 0) == expected_x
            assert counts.get("cx", 0) >= 1
        case "quri_parts":
            counts = _quri_parts_gate_counts(exe)
            assert counts.get("x", 0) == expected_x
            assert counts.get("cx", 0) >= 1
        case "cudaq":
            counts = _cudaq_gate_counts(exe)
            assert counts.get("x", 0) == expected_x
            assert counts.get("cx", 0) >= 1


def _dicke_expval(TranspilerCls, n, k, hamiltonian) -> float:
    """Transpiles prepare_dicke for |D^n_k> and returns <hamiltonian>.

    Args:
        TranspilerCls (type): Backend transpiler class to instantiate.
        n (int): Number of qubits.
        k (int): Hamming weight of the Dicke state.
        hamiltonian (qm_o.Hamiltonian): Observable to estimate.

    Returns:
        float: Expectation value from the backend estimator.
    """
    initial_ones, schedule = dicke_state_composition_schedule(
        n_qubits=n, block_size=n, hamming_weight=k
    )
    transpiler = TranspilerCls()
    exe = transpiler.transpile(
        _wrap_prepare_dicke_expval,
        bindings={
            "n": n,
            "initial_ones": initial_ones,
            "schedule": schedule,
            "hamiltonian": hamiltonian,
        },
    )
    return exe.run(transpiler.executor()).result()


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
def test_prepare_dicke_expval_xx_shows_coherence(name, TranspilerCls):
    """Tests that <D^2_1|X_0 X_1|D^2_1> = 1 via the estimator (run) path.

    |D^2_1> = (|01> + |10>) / sqrt(2) is the +1 eigenstate of X_0 X_1, which
    swaps |01> and |10>. The weight-1 basis states |01> and |10>, and their
    incoherent mixture, all give 0, so this separates the Dicke superposition
    from the X-gate initial state that prepare_dicke starts from.
    """
    result = _dicke_expval(TranspilerCls, 2, 1, qm_o.X(0) * qm_o.X(1))

    np.testing.assert_allclose(result, 1.0, atol=1e-6)


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
@pytest.mark.parametrize(
    "n,k",
    [
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 2),
        (3, 1),
        (3, 2),
        (4, 1),
        (4, 2),
        (5, 2),
    ],
)
def test_prepare_dicke_matches_dicke_state_expvals(name, TranspilerCls, n, k):
    """Tests that prepare_dicke produces |D^n_k> via random Z and XX observables.

    On |D^n_k> every weight-k bitstring has amplitude 1/sqrt(C(n, k)), so:

    - Qubit i is 1 in C(n-1, k-1) / C(n, k) = k/n of the bitstrings, giving
      <Z_i> = 1 - 2k/n for every i. A single weight-k basis state gives
      <Z_i> = +/-1 instead, so random weights w_i detect it.
    - X_i X_j maps a bitstring to one of weight k only when its bits i and j
      differ, which holds for 2 C(n-2, k-1) bitstrings. With equal real
      amplitudes, <X_i X_j> = 2 C(n-2, k-1) / C(n, k) = 2k(n-k) / (n(n-1)).
      This checks the relative phases, which Z observables cannot see.

    The weights are drawn from a seeded RNG per (n, k).
    """
    rng = np.random.default_rng(100 * n + k)

    w = rng.uniform(-1.0, 1.0, size=n)
    H_z = qm_o.Hamiltonian()
    for i in range(n):
        H_z = H_z + float(w[i]) * qm_o.Z(i)
    result_z = _dicke_expval(TranspilerCls, n, k, H_z)
    np.testing.assert_allclose(result_z, (1 - 2 * k / n) * w.sum(), atol=1e-5)

    if n < 2:
        return
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    v = rng.uniform(-1.0, 1.0, size=len(pairs))
    H_xx = qm_o.Hamiltonian()
    for (i, j), v_ij in zip(pairs, v):
        H_xx = H_xx + float(v_ij) * (qm_o.X(i) * qm_o.X(j))
    result_xx = _dicke_expval(TranspilerCls, n, k, H_xx)
    expected_xx = 2 * k * (n - k) / (n * (n - 1)) * v.sum()
    np.testing.assert_allclose(result_xx, expected_xx, atol=1e-5)


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
@pytest.mark.parametrize(
    "n,k",
    [
        (2, 1),
        (3, 1),
        (4, 2),
    ],
)
def test_prepare_dicke_sample_preserves_hamming_weight(name, TranspilerCls, n, k):
    """Tests that prepare_dicke samples have Hamming weight k via the sampler path.

    For |D^n_k>, every bitstring in the equal superposition has exactly k set bits,
    so all measurement outcomes must have Hamming weight k.
    """
    initial_ones, schedule = dicke_state_composition_schedule(
        n_qubits=n, block_size=n, hamming_weight=k
    )

    transpiler = TranspilerCls()
    exe = transpiler.transpile(
        _wrap_prepare_dicke,
        bindings={
            "n": n,
            "initial_ones": initial_ones,
            "schedule": schedule,
        },
    )

    job = exe.sample(transpiler.executor(), shots=32)
    result = job.result()

    assert len(result.results) > 0
    for sample, _count in result.results:
        assert sum(sample) == k
