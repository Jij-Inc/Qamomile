"""Tests for qamomile/circuit/algorithm/gas.py circuit primitives."""

import pytest

pytest.importorskip("qiskit")

import qamomile.circuit as qmc
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
from qamomile.qiskit.transpiler import QiskitTranspiler


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
    y: qmc.UInt,
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
    y: qmc.UInt,
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
    y: qmc.UInt,
    linear: qmc.Dict[qmc.UInt, qmc.Float],
    quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
    iters: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Run grover_algorithm and measure the input register."""
    q_output, q_input = grover_algorithm(n, m, y, linear, quad, iters)
    _ = q_output
    return qmc.measure(q_input)


def _sample_results(exe, transpiler, bindings: dict, shots: int = 16):
    """Sample an executable with bindings and return result rows."""
    job = exe.sample(transpiler.executor(), bindings=bindings, shots=shots)
    return job.result().results


def test_qft_encoding_transpile_smoke():
    """qft_encoding wrapper transpiles and samples valid bitstrings."""
    tr = QiskitTranspiler()
    exe = tr.transpile(_wrap_qft_encoding, bindings={"n": 3, "coef": 1.2})
    results = _sample_results(exe, tr, bindings={})

    assert len(results) > 0
    for bits, count in results:
        assert len(bits) == 3
        assert count > 0


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
def test_degree_encodings_transpile_and_sample_smoke(kernel, bindings, expected_len):
    """Degree-specific encoders transpile and produce valid samples."""
    tr = QiskitTranspiler()
    exe = tr.transpile(kernel, bindings=bindings)
    results = _sample_results(exe, tr, bindings={})

    assert len(results) > 0
    for bits, count in results:
        assert len(bits) == expected_len
        assert count > 0


def test_apply_then_dagger_restores_input_register():
    """Applying preparation then dagger restores the input register to |0...0>."""
    tr = QiskitTranspiler()
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


def test_function_preparation_transpile_and_sample_smoke():
    """function_preparation_qubo transpiles and samples valid input bitstrings."""
    tr = QiskitTranspiler()
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


def test_grover_algorithm_transpile_and_sample_smoke():
    """grover_algorithm transpiles and samples valid input bitstrings."""
    tr = QiskitTranspiler()
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
