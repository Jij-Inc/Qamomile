from __future__ import annotations

import jijmodeling as jm
import jijmodeling_transpiler as jmt
import numpy as np
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label
from quri_parts.core.state import ParametricCircuitQuantumState
from quri_parts.qulacs.estimator import (
    create_qulacs_vector_parametric_estimator,
)
from quri_parts.qulacs.sampler import create_qulacs_vector_sampler
from scipy.optimize import OptimizeResult, minimize

import jijmodeling_transpiler_quantum.quri_parts as jmt_qp


def run_qaoa(
    ansatz: LinearMappedUnboundParametricQuantumCircuit,
    hamiltonian: Operator,
    method: str | callable,
) -> tuple[OptimizeResult, list[float]]:
    def _cost_func(parameters, ansatz, hamiltonian, estimator):
        parametric_state = ParametricCircuitQuantumState(
            ansatz.qubit_count, ansatz
        )
        estimate = estimator(hamiltonian, parametric_state, parameters)
        return estimate.value.real

    initial_params = np.array([1.0] * ansatz.parameter_count)

    estimator = create_qulacs_vector_parametric_estimator()
    optimization_history: list[float] = []

    result = minimize(
        _cost_func,
        initial_params,
        args=(ansatz, hamiltonian, estimator),
        method=method,
        callback=lambda x: optimization_history.append(
            _cost_func(x, ansatz, hamiltonian, estimator)
        ),
    )

    return result, optimization_history


def sample_result(
    result: OptimizeResult,
    ansatz: LinearMappedUnboundParametricQuantumCircuit,
    qaoa_builder: jmt_qp.QAOAAnsatzBuilder,
    num_qubits: int,
    num_shots: int = 1000,
) -> jm.SampleSet:
    bind_ansatz_opt = ansatz.bind_parameters(result.x)
    sampler = create_qulacs_vector_sampler()
    sampling_result = sampler(bind_ansatz_opt, num_shots)
    result_bits = {
        bin(key)[2:].zfill(num_qubits)[::-1]: val
        for key, val in sampling_result.items()
    }
    sampleset = qaoa_builder.decode_from_counts(result_bits)
    return sampleset


def test_qaoa_onehot():
    n = jm.Placeholder("n")
    x = jm.BinaryVar("x", shape=(n,))
    i = jm.Element("i", belong_to=n)
    problem = jm.Problem("sample")
    problem += jm.sum(i, x[i])
    problem += jm.Constraint("onehot", jm.sum(i, x[i]) == 1)
    compiled_instance = jmt.core.compile_model(problem, {"n": 3})
    qaoa_builder = jmt_qp.transpile_to_qaoa_ansatz(compiled_instance)
    qaoa_ansatz, cost_func, constant = qaoa_builder.get_qaoa_ansatz(p=1)

    # qubo:  (x0 + x1 + x2 - 1)^2 = - x0 - x1 - x2 + 2x0x1 + 2x1x2 + 2x0x2+ 1
    # ising: = - 1/2(1-z0) - 1/2(1-z1) - 1/2(1-z3) + 1/2(1-z0)(1-z1) + 1/2(1-z1)(1-z2) + 1/2(1-z0)(1-z2)+ 1
    #        = 1/2*z0*z1 - 0.5 + 1

    ANS_op = Operator(
        {
            pauli_label("Z0"): -1,
            pauli_label("Z1"): -1,
            pauli_label("Z2"): -1,
            pauli_label("Z0 Z1"): 0.5,
            pauli_label("Z0 Z2"): 0.5,
            pauli_label("Z1 Z2"): 0.5,
            # PAULI_IDENTITY: 1.5,
            # coeff of identity is included in constant.
        }
    )

    assert qaoa_ansatz.qubit_count == 3
    assert qaoa_ansatz.parameter_count == 2
    assert cost_func == ANS_op
    assert constant == 2.5


def test_qaoa_H_eigenvalue():
    n = jm.Placeholder("n")
    x = jm.BinaryVar("x", shape=(n,))
    i = jm.Element("i", belong_to=n)
    problem = jm.Problem("sample")
    problem += jm.Constraint("onehot", jm.sum(i, x[i]) == 1)
    problem += -10 * x[0]

    compiled_instance = jmt.core.compile_model(problem, {"n": 5})

    qaoa_builder = jmt_qp.transpile_to_qaoa_ansatz(
        compiled_instance, relax_method=jmt.core.pubo.SquaredPenalty
    )

    # qubo:  (x0 + x1 - 1)^2 = - x0 - x1 + 2x0x1 + 1
    # ising: = - 1/2(1-z0) - 1/2(1-z1) + 1/2(1-z0)(1-z1) + 1
    #        = 1/2*z0*z1 - 0.5 + 1

    qaoa_ansatz, hamiltonian, constant = qaoa_builder.get_qaoa_ansatz(
        p=3, multipliers={"onehot": 3.0}
    )
    result, _ = run_qaoa(qaoa_ansatz, hamiltonian, "COBYLA")
    num_shots = 10
    sampleset = sample_result(
        result,
        qaoa_ansatz,
        qaoa_builder,
        qaoa_ansatz.qubit_count,
        num_shots=num_shots,
    )

    assert (
        sampleset.lowest().to_dense().record.solution["x"][0]
        == [
            1,
            0,
            0,
            0,
            0,
        ]
    ).all()
    assert len(sampleset.feasible().record.solution) != 0


def test_qaoa_hamiltonian_3qubit_antiferro():
    x = jm.BinaryVar("x", shape=(3,))
    problem = jm.Problem("sample")

    spin = lambda i: 2*x[i] - 1
    problem += spin(0)*spin(1) + spin(1)*spin(2) + spin(0)*spin(2)

    true_ising = {
        (0, 1): 1,
        (1, 2): 1,
        (0, 2): 1,
    }

    compiled_instance = jmt.core.compile_model(problem, {})
    qaoa_builder = jmt_qp.transpile_to_qaoa_ansatz(compiled_instance)
    qaoa_ansatz, hamiltonian, constant = qaoa_builder.get_qaoa_ansatz(p=1)

    assert constant == 0.0

    true_H = Operator(
        {
            pauli_label("Z0 Z1"): 1/4,
            pauli_label("Z1 Z2"): 1/4,
            pauli_label("Z0 Z2"): 1/4,
        }
    )
    assert hamiltonian == true_H

    num_qubits = 3
    true_ansatz = LinearMappedUnboundParametricQuantumCircuit(num_qubits)
    for i in range(num_qubits):
        true_ansatz.add_H_gate(i)

    for p_level in range(1):
        gamma = true_ansatz.add_parameter(f"gamma{p_level}")
        beta = true_ansatz.add_parameter(f"beta{p_level}")
        for (i, j), coeff in true_ising.items():
            true_ansatz.add_ParametricPauliRotation_gate(
                [i, j],
                pauli_ids=(3, 3),
                angle={gamma: 2 * coeff},
            )
        for i in range(num_qubits):
            true_ansatz.add_ParametricRX_gate(
                i, {beta: 2}
            )

    assert qaoa_ansatz.qubit_count == true_ansatz.qubit_count
    assert qaoa_ansatz.parameter_count == true_ansatz.parameter_count
    assert qaoa_ansatz.gates == true_ansatz.gates
