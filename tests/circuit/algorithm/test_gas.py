"""Tests for qamomile/circuit/algorithm/gas.py circuit primitives."""

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.gas import (
    apply_function_preparation_qubo,
    apply_function_preparation_qubo_dagger,
    first_degree_qft_encoding,
    function_preparation_qubo,
    grover_algorithm,
    qft_encoding,
    second_degree_qft_encoding,
    zero_degree_qft_encoding,
)

# ---------------------------------------------------------------------------
# Backend factories
# ---------------------------------------------------------------------------


def _qiskit_transpiler():
    """Return a QiskitTranspiler, skipping if qiskit is unavailable.

    Returns:
        QiskitTranspiler: A fresh Qiskit backend transpiler.
    """
    pytest.importorskip("qiskit")
    from qamomile.qiskit.transpiler import QiskitTranspiler

    return QiskitTranspiler()


def _quri_parts_transpiler():
    """Return a QuriPartsTranspiler, skipping if quri_parts is unavailable.

    Returns:
        QuriPartsTranspiler: A fresh QuriParts backend transpiler.
    """
    pytest.importorskip("quri_parts")
    from qamomile.quri_parts import QuriPartsTranspiler

    return QuriPartsTranspiler()


def _cudaq_transpiler():
    """Return a CudaqTranspiler, skipping if cudaq is unavailable.

    Returns:
        CudaqTranspiler: A fresh CUDA-Q backend transpiler.
    """
    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqTranspiler

    return CudaqTranspiler()


_BACKENDS = [
    pytest.param(_qiskit_transpiler, id="qiskit"),
    pytest.param(_quri_parts_transpiler, id="quri_parts"),
    pytest.param(_cudaq_transpiler, id="cudaq"),
]


# ---------------------------------------------------------------------------
# Module-level qkernels (reused across tests)
# ---------------------------------------------------------------------------


@qmc.qkernel
def _wrap_qft_encoding(n: qmc.UInt, coef: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Apply qft_encoding on a fresh register and measure it."""
    q = qmc.qubit_array(n, name="q")
    q = qft_encoding(q, coef)
    return qmc.measure(q)


@qmc.qkernel
def _wrap_zero_degree(n: qmc.UInt, m: qmc.UInt, coef: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Apply zero-degree encoding and measure the input register."""
    q_output = qmc.qubit_array(m, name="q_output")
    q_input = qmc.qubit_array(n, name="q_input")
    q_output, q_input = zero_degree_qft_encoding(q_output, q_input, coef)
    return qmc.measure(q_input)


@qmc.qkernel
def _wrap_first_degree(
    n: qmc.UInt,
    m: qmc.UInt,
    control_idx: qmc.UInt,
    coef: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    """Apply first-degree encoding and measure the input register."""
    q_output = qmc.qubit_array(m, name="q_output")
    q_input = qmc.qubit_array(n, name="q_input")
    q_input[control_idx] = qmc.x(q_input[control_idx])
    q_output, q_input = first_degree_qft_encoding(q_output, q_input, control_idx, coef)
    return qmc.measure(q_input)


@qmc.qkernel
def _wrap_second_degree(
    n: qmc.UInt,
    m: qmc.UInt,
    control_idx0: qmc.UInt,
    control_idx1: qmc.UInt,
    coef: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    """Apply second-degree encoding and measure the input register."""
    q_output = qmc.qubit_array(m, name="q_output")
    q_input = qmc.qubit_array(n, name="q_input")
    q_input[control_idx0] = qmc.x(q_input[control_idx0])
    q_input[control_idx1] = qmc.x(q_input[control_idx1])
    q_output, q_input = second_degree_qft_encoding(
        q_output, q_input, control_idx0, control_idx1, coef
    )
    return qmc.measure(q_input)


@qmc.qkernel
def _wrap_apply_then_dagger(
    n: qmc.UInt,
    m: qmc.UInt,
    y: qmc.Float,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """Apply preparation then its dagger and measure the input register."""
    q_output = qmc.qubit_array(m, name="q_output")
    q_input = qmc.qubit_array(n, name="q_input")
    q_output, q_input = apply_function_preparation_qubo(
        q_output, q_input, y, linear, quad
    )
    q_output, q_input = apply_function_preparation_qubo_dagger(
        q_output, q_input, y, linear, quad
    )
    return qmc.measure(q_input)


@qmc.qkernel
def _wrap_function_preparation(
    n: qmc.UInt,
    m: qmc.UInt,
    y: qmc.Float,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """Run full function preparation and measure the input register."""
    q_output, q_input = function_preparation_qubo(n, m, y, linear, quad)
    _ = q_output
    return qmc.measure(q_input)


@qmc.qkernel
def _wrap_grover_algorithm(
    n: qmc.UInt,
    m: qmc.UInt,
    y: qmc.Float,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    iters: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Run grover_algorithm and measure the input register."""
    q_output, q_input = grover_algorithm(n, m, y, linear, quad, iters)
    _ = q_output
    return qmc.measure(q_input)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _sample_results(exe, transpiler, bindings: dict, shots: int = 16):
    """Sample an executable with bindings and return result rows.

    Args:
        exe (ExecutableProgram): The compiled circuit to sample.
        transpiler (Transpiler): Backend transpiler owning the executor.
        bindings (dict): Runtime parameter bindings for the sample call.
        shots (int): Number of shots.

    Returns:
        list: List of (bitstring, count) result rows.
    """
    job = exe.sample(transpiler.executor(), bindings=bindings, shots=shots)
    return job.result().results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_transpiler", _BACKENDS)
def test_qft_encoding_transpile_smoke(make_transpiler):
    """qft_encoding wrapper transpiles and samples valid bitstrings on each backend."""
    tr = make_transpiler()
    exe = tr.transpile(_wrap_qft_encoding, bindings={"n": 3, "coef": 1.2})
    results = _sample_results(exe, tr, bindings={})

    assert len(results) > 0
    for bits, count in results:
        assert len(bits) == 3
        assert count > 0


@pytest.mark.parametrize("make_transpiler", _BACKENDS)
@pytest.mark.parametrize(
    "kernel,bindings,expected_len",
    [
        (_wrap_zero_degree, {"n": 2, "m": 3, "coef": 0.5}, 2),
        (_wrap_first_degree, {"n": 3, "m": 3, "control_idx": 1, "coef": -0.7}, 3),
        (
            _wrap_second_degree,
            {"n": 4, "m": 3, "control_idx0": 1, "control_idx1": 3, "coef": 0.9},
            4,
        ),
    ],
    ids=["zero-degree", "first-degree", "second-degree"],
)
def test_degree_encodings_transpile_and_sample_smoke(
    make_transpiler, kernel, bindings, expected_len
):
    """Degree-specific encoders transpile and produce valid samples on each backend."""
    tr = make_transpiler()
    exe = tr.transpile(kernel, bindings=bindings)
    results = _sample_results(exe, tr, bindings={})

    assert len(results) > 0
    for bits, count in results:
        assert len(bits) == expected_len
        assert count > 0


@pytest.mark.parametrize("make_transpiler", _BACKENDS)
def test_apply_then_dagger_restores_input_register(make_transpiler):
    """Applying preparation then dagger restores the input register to |0...0>."""
    tr = make_transpiler()
    bindings = {
        "n": 3,
        "m": 4,
        "y": 1,
        "linear": {0: 0.5, 2: -1.0},
        "quad": {(0, 1): 0.75, (1, 2): -0.25},
    }
    exe = tr.transpile(_wrap_apply_then_dagger, bindings=bindings)
    results = _sample_results(exe, tr, bindings={}, shots=32)

    assert len(results) > 0
    for bits, count in results:
        assert tuple(int(b) for b in bits) == (0, 0, 0)
        assert count > 0


@pytest.mark.parametrize("make_transpiler", _BACKENDS)
def test_function_preparation_transpile_and_sample_smoke(make_transpiler):
    """function_preparation_qubo transpiles and samples valid input bitstrings on each backend."""
    tr = make_transpiler()
    bindings = {
        "n": 3,
        "m": 4,
        "y": 1,
        "linear": {0: 0.5, 2: -1.0},
        "quad": {(0, 1): 0.75, (1, 2): -0.25},
    }
    exe = tr.transpile(_wrap_function_preparation, bindings=bindings)
    results = _sample_results(exe, tr, bindings={})

    assert len(results) > 0
    for bits, count in results:
        assert len(bits) == 3
        assert count > 0


@pytest.mark.parametrize("make_transpiler", _BACKENDS)
def test_grover_algorithm_transpile_and_sample_smoke(make_transpiler):
    """grover_algorithm transpiles and samples valid input bitstrings on each backend."""
    tr = make_transpiler()
    bindings = {
        "n": 3,
        "m": 4,
        "y": 1,
        "linear": {0: 0.5, 2: -1.0},
        "quad": {(0, 1): 0.75, (1, 2): -0.25},
        "iters": 1,
    }
    exe = tr.transpile(_wrap_grover_algorithm, bindings=bindings)
    results = _sample_results(exe, tr, bindings={})

    assert len(results) > 0
    for bits, count in results:
        assert len(bits) == 3
        assert count > 0


# ---------------------------------------------------------------------------
# Estimator (expval) path
# ---------------------------------------------------------------------------


@qmc.qkernel
def _wrap_qft_encoding_expval(
    coef: qmc.Float,
    H: qmc.Observable,
) -> qmc.Float:
    """Apply qft_encoding on a fresh 1-qubit |0> register and return <H>."""
    q = qmc.qubit_array(1, name="q")
    q = qft_encoding(q, coef)
    return qmc.expval(q, H)


@pytest.mark.parametrize("make_transpiler", _BACKENDS)
def test_qft_encoding_expval_zero_coef_ground_state(make_transpiler):
    """qft_encoding with coef=0 leaves |0> unchanged; Z-observable expval is +1.

    With coef=0, the phase gate applies zero rotation and the state remains |0>.
    The expectation of Z on |0> is exactly +1. This analytic reference is trivial
    (<Z>_{|0>} = 1) and exercises the estimator path independently from the sampler
    path on every backend.
    """
    tr = make_transpiler()
    H = qm_o.Z(0)
    exe = tr.transpile(_wrap_qft_encoding_expval, bindings={"coef": 0.0, "H": H})
    result = exe.run(tr.executor()).result()

    np.testing.assert_allclose(result, 1.0, atol=1e-6)
