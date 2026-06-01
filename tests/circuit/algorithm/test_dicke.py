"""Tests for qamomile/circuit/algorithm/state_preparation/dicke.py primitives."""

import re

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.state_preparation.dicke import (
    prepare_dicke,
    scs_gate_2q,
    scs_gate_3q,
)
from qamomile.optimization.schedules.dicke import dicke_state_composition_schedule

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

BACKENDS: list[tuple[str, type]] = []
try:
    import qiskit  # noqa: F401

    from qamomile.qiskit.transpiler import QiskitTranspiler

    BACKENDS.append(("qiskit", QiskitTranspiler))
except ImportError:
    pass
try:
    import quri_parts  # noqa: F401

    from qamomile.quri_parts.transpiler import QuriPartsTranspiler

    BACKENDS.append(("quri_parts", QuriPartsTranspiler))
except ImportError:
    pass
try:
    import cudaq  # noqa: F401

    from qamomile.cudaq.transpiler import CudaqTranspiler

    BACKENDS.append(("cudaq", CudaqTranspiler))
except ImportError:
    pass

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
    source = exe.compiled_quantum[0].source
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
    q = qmc.qubit_array(n, name="q")
    q = scs_gate_3q(q, t, c1, c2, theta)
    return qmc.measure(q)


@qmc.qkernel
def _wrap_prepare_dicke(
    n: qmc.UInt,
    initial_ones: qmc.Vector[qmc.UInt],
    schedule: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    q = prepare_dicke(n, initial_ones, schedule)
    return qmc.measure(q)


@qmc.qkernel
def _wrap_prepare_dicke_expval(
    n: qmc.UInt,
    initial_ones: qmc.Vector[qmc.UInt],
    schedule: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
    hamiltonian: qmc.Observable,
) -> qmc.Float:
    q = prepare_dicke(n, initial_ones, schedule)
    return qmc.expval(q, hamiltonian)


# ---------------------------------------------------------------------------
# Primitive tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
def test_scs_gate_2q_uses_expected_cnot_and_ry_counts(name, TranspilerCls):
    """Tests that the 2-qubit SCS gate emits exactly 4 CX and 2 RY gates."""
    transpiler = TranspilerCls()
    exe = transpiler.transpile(
        _wrap_scs_gate_2q,
        bindings={"n": 2, "t": 0, "c": 1, "theta": 0.3},
    )

    match name:
        case "qiskit":
            counts = _qiskit_gate_counts(exe)
            assert counts.get("cx", 0) == 4
            assert counts.get("ry", 0) == 2
        case "quri_parts":
            counts = _quri_parts_gate_counts(exe)
            assert counts.get("cx", 0) == 4
            assert counts.get("ry", 0) == 2
        case "cudaq":
            counts = _cudaq_gate_counts(exe)
            assert counts.get("cx", 0) == 4
            assert counts.get("ry", 0) == 2


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
def test_scs_gate_3q_uses_expected_cnot_and_ry_counts(name, TranspilerCls):
    """Tests that the 3-qubit SCS gate emits exactly 6 CX and 4 RY gates."""
    transpiler = TranspilerCls()
    exe = transpiler.transpile(
        _wrap_scs_gate_3q,
        bindings={"n": 3, "t": 0, "c1": 1, "c2": 2, "theta": 0.3},
    )

    match name:
        case "qiskit":
            counts = _qiskit_gate_counts(exe)
            assert counts.get("cx", 0) == 6
            assert counts.get("ry", 0) == 4
        case "quri_parts":
            counts = _quri_parts_gate_counts(exe)
            assert counts.get("cx", 0) == 6
            assert counts.get("ry", 0) == 4
        case "cudaq":
            counts = _cudaq_gate_counts(exe)
            assert counts.get("cx", 0) == 6
            assert counts.get("ry", 0) == 4


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
def test_prepare_dicke_applies_basis_initialization_and_scs_rotations(
    name, TranspilerCls
):
    """Tests that prepare_dicke applies X gates for initial Hamming weight and the correct number of SCS rotation gates."""
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

    pairs = {k: v for k, v in schedule.items() if k[1] == k[2]}
    triplets = {k: v for k, v in schedule.items() if k[1] != k[2]}
    expected_x = len(initial_ones)
    expected_ry = 2 * len(pairs) + 4 * len(triplets)
    expected_cx = 4 * len(pairs) + 6 * len(triplets)

    match name:
        case "qiskit":
            counts = _qiskit_gate_counts(exe)
            assert counts.get("x", 0) == expected_x
            assert counts.get("ry", 0) == expected_ry
            assert counts.get("cx", 0) == expected_cx
        case "quri_parts":
            counts = _quri_parts_gate_counts(exe)
            assert counts.get("x", 0) == expected_x
            assert counts.get("ry", 0) == expected_ry
            assert counts.get("cx", 0) == expected_cx
        case "cudaq":
            counts = _cudaq_gate_counts(exe)
            assert counts.get("x", 0) == expected_x
            assert counts.get("ry", 0) == expected_ry
            assert counts.get("cx", 0) == expected_cx


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
def test_prepare_dicke_expval_z_sum_is_zero(name, TranspilerCls):
    """Tests that <D^2_1|Z_0 + Z_1|D^2_1> = 0 via the estimator (run) path.

    |D^2_1> = (|01> + |10>) / sqrt(2) is the 2-qubit Dicke state with Hamming
    weight 1. By symmetry, <Z_0> = <Z_1> = 0, so <Z_0 + Z_1> = 0 exactly.
    This test exercises the expval / estimator code path that is not covered
    by the sampling tests.
    """
    initial_ones, schedule = dicke_state_composition_schedule(
        n_qubits=2, block_size=2, hamming_weight=1
    )

    H = qm_o.Z(0) + qm_o.Z(1)

    transpiler = TranspilerCls()
    exe = transpiler.transpile(
        _wrap_prepare_dicke_expval,
        bindings={
            "n": 2,
            "initial_ones": initial_ones,
            "schedule": schedule,
            "hamiltonian": H,
        },
    )

    job = exe.run(transpiler.executor())
    result = job.result()

    np.testing.assert_allclose(result, 0.0, atol=1e-6)


@pytest.mark.parametrize("name,TranspilerCls", BACKENDS)
@pytest.mark.parametrize(
    "n,k",
    [
        (2, 0),
        (2, 1),
        (3, 1),
        (3, 2),
        (4, 1),
        (4, 2),
    ],
)
def test_prepare_dicke_z_sum_matches_analytic(name, TranspilerCls, n, k):
    """Tests that prepare_dicke produces the correct Dicke state |D^n_k>.

    For |D^n_k>, the expected value of sum_i Z_i equals n - 2k. This follows
    from the equal superposition over all weight-k bitstrings: each state
    contributes (n-k) qubits in |0> (+1 eigenvalue) and k qubits in |1>
    (-1 eigenvalue), giving <sum Z_i> = (n-k) - k = n - 2k.
    """
    initial_ones, schedule = dicke_state_composition_schedule(
        n_qubits=n, block_size=n, hamming_weight=k
    )

    H = qm_o.Z(0)
    for i in range(1, n):
        H = H + qm_o.Z(i)

    transpiler = TranspilerCls()
    exe = transpiler.transpile(
        _wrap_prepare_dicke_expval,
        bindings={
            "n": n,
            "initial_ones": initial_ones,
            "schedule": schedule,
            "hamiltonian": H,
        },
    )

    job = exe.run(transpiler.executor())
    result = job.result()

    np.testing.assert_allclose(result, float(n - 2 * k), atol=1e-5)


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
