"""Regression tests for nested same-name loop variables.

When an outer ``@qkernel`` and an inlined inner ``@qkernel`` both used
a loop variable with the same display name (e.g. both ``for i in
qm.range(...)``), the emit-time parameter resolver previously consulted
``bindings[name]`` before ``bindings[uuid]``. Because the inner loop
overwrote ``bindings["i"]`` with its own iteration value, expressions
captured from the outer scope (such as ``arr[i]`` in the call site)
were resolved against the *inner* loop's index — producing extra
parameters keyed by inner-loop iterations and failing at qiskit
``assign_parameters``.

The fix removes the legacy name-keyed loop variable binding entirely:
loop variables are bound by ``loop_var_value.uuid`` only, so different
``ForOperation`` instances with identical display names never collide.
"""

import pytest

pytest.importorskip("qiskit")

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.qiskit.transpiler import QiskitExecutor, QiskitTranspiler


def _zz_hamiltonian(num_qubits: int) -> qm_o.Hamiltonian:
    """Build a chain of nearest-neighbour ZZ terms on ``num_qubits``."""
    H = qm_o.Hamiltonian()
    for i in range(num_qubits - 1):
        H.add_term(
            (
                qm_o.PauliOperator(qm_o.Pauli.Z, i),
                qm_o.PauliOperator(qm_o.Pauli.Z, i + 1),
            ),
            1.0,
        )
    return H


@qmc.qkernel
def _inner_uses_same_name(
    q: qmc.Vector[qmc.Qubit], beta: qmc.Float
) -> qmc.Vector[qmc.Qubit]:
    """Apply rx(beta) to every qubit. Inner loop variable is named ``i``."""
    n = q.shape[0]
    for i in qmc.range(n):
        q[i] = qmc.rx(q[i], beta)
    return q


@qmc.qkernel
def _outer_uses_same_name(
    n: qmc.UInt,
    p: qmc.UInt,
    beta: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """Outer loop also named ``i`` — must shadow the inner one cleanly."""
    q = qmc.qubit_array(n, "q")
    for i in qmc.range(p):
        q = _inner_uses_same_name(q, beta[i])
    return qmc.measure(q)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_nested_same_name_loops_param_count(p: int):
    """Number of emitted parameters tracks the outer loop only."""
    n = 3
    tr = QiskitTranspiler()
    exe = tr.transpile(
        _outer_uses_same_name,
        bindings={"n": n, "p": p},
        parameters=["beta"],
    )

    circuit = exe.compiled_quantum[0].circuit
    assert circuit.num_parameters == p, (
        f"Inner loop variable leaked into outer scope: expected {p} "
        f"parameters but got {circuit.num_parameters}"
    )

    expected = {f"beta[{k}]" for k in range(p)}
    assert {str(param) for param in circuit.parameters} == expected


def test_nested_same_name_loops_sample_runs():
    """End-to-end sample with bindings matching the outer loop length."""
    n = 3
    p = 2
    tr = QiskitTranspiler()
    exe = tr.transpile(
        _outer_uses_same_name,
        bindings={"n": n, "p": p},
        parameters=["beta"],
    )

    executor = QiskitExecutor()
    job = exe.sample(executor, bindings={"beta": [0.3, 0.7]}, shots=128)
    sample = job.result()
    assert sample.shots == 128
    assert sum(c for _, c in sample.results) == 128


@qmc.qkernel
def _inner_pauli_evolve(
    q: qmc.Vector[qmc.Qubit], H: qmc.Observable, gamma: qmc.Float
) -> qmc.Vector[qmc.Qubit]:
    return qmc.pauli_evolve(q, H, gamma)


@qmc.qkernel
def _qaoa_with_colliding_names(
    H: qmc.Observable,
    n: qmc.UInt,
    p: qmc.UInt,
    gamma: qmc.Vector[qmc.Float],
    beta: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """QAOA-style kernel with outer ``i`` shadowed by an inner ``i``."""
    q = qmc.qubit_array(n, "q")
    for i in qmc.range(p):
        q = _inner_pauli_evolve(q, H, gamma[i])
        q = _inner_uses_same_name(q, beta[i])
    return qmc.measure(q)


@pytest.mark.parametrize("p", [1, 2])
def test_qaoa_pattern_with_colliding_names(p: int):
    """The original bug repro: QAOA with mixer using a colliding ``i``."""
    H = _zz_hamiltonian(num_qubits=3)
    tr = QiskitTranspiler()
    exe = tr.transpile(
        _qaoa_with_colliding_names,
        bindings={"H": H, "n": H.num_qubits, "p": p},
        parameters=["gamma", "beta"],
    )

    circuit = exe.compiled_quantum[0].circuit
    expected = {f"gamma[{k}]" for k in range(p)} | {f"beta[{k}]" for k in range(p)}
    actual = {str(param) for param in circuit.parameters}
    assert actual == expected, f"Unexpected parameter set: {actual}"

    executor = QiskitExecutor()
    gamma_values = [0.1 * (k + 1) for k in range(p)]
    beta_values = [0.2 * (k + 1) for k in range(p)]
    job = exe.sample(
        executor,
        bindings={"gamma": gamma_values, "beta": beta_values},
        shots=64,
    )
    sample = job.result()
    assert sample.shots == 64
    assert sum(c for _, c in sample.results) == 64
