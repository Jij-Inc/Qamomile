from __future__ import annotations
from collections.abc import Callable
import itertools
import typing as typ
import numpy as np
import qiskit as qk
import qiskit.quantum_info as qk_info
from qiskit.primitives import Estimator


def define_pauli_op(
    num_register_bits: int, ancilla: bool = False
) -> list[qk_info.SparsePauliOp]:
    """Function to define pauli operators

    Args:
        num_register_bits (int): number of register bits
        ancilla (bool, optional): whether to add ancilla qubit |1> or not. Defaults to False.

    Raises:
        ValueError: if num_register_bits < 1

    Returns:
        list[qk_info.SparsePauliOp]: list of pauli operators
    """

    if num_register_bits < 1:
        raise ValueError("num_register_bits must be greater than 0")

    z = [[False], [True]]
    x = [[False], [False]]
    zero_op = qk_info.SparsePauliOp(
        qk_info.PauliList.from_symplectic(z, x), coeffs=[1 / 2, 1 / 2]
    )

    one_op = qk_info.SparsePauliOp(
        qk_info.PauliList.from_symplectic(z, x), coeffs=[1 / 2, -1 / 2]
    )
    identity_op = qk_info.SparsePauliOp(qk_info.Pauli([0], [0]))

    ancilla_operator = one_op if ancilla else identity_op

    pauli_ops = []

    if num_register_bits == 1:
        for _op in [one_op, zero_op]:
            pauli_ops.append(_op.tensor(ancilla_operator))
    else:
        for val in itertools.product(
            [one_op, zero_op], repeat=num_register_bits
        ):
            pauli_op = val[0]
            for _op in val[1:]:
                pauli_op = pauli_op.tensor(_op)

            pauli_ops.append(pauli_op.tensor(ancilla_operator))

    return pauli_ops


def initialize_cost_function(
    qubo: dict[tuple[int, int], float], num_cbits: int
) -> Callable[[np.array], float]:
    """Function to initialize cost function for minimal encoding

    Args:
        qubo (dict[tuple[int, int], float]): QUBO Matrix
        num_cbit (int): number of classical bits

    Returns:
        Callable[[np.array], float]: cost function for minimal encoding
    """

    # define cost function
    def cost_function(coeff_one: np.array) -> float:
        """Function to compute value of cost function for minimal encoding

        Args:
            coeff_one (np.array): the ratio of the expectation value of each pauli operator for when ancilla qubit is |1> and the expectation value of each pauli operator ignoring ancilla qubit

        Returns:
            float: value of cost function
        """

        cost = 0
        for i in range(num_cbits):
            for j in range(i, num_cbits):
                if i != j:
                    cost += 2 * qubo[(i, j)] * coeff_one[i] * coeff_one[j]
                else:
                    cost += qubo[(i, j)] * coeff_one[i]

        return cost

    return cost_function


# def initialize_vqe_process(
#     ansatz: qk.circuit.quantumcircuit.QuantumCircuit,
#     qubo: dict[tuple[int, int], float],
#     num_shots: int = None,
# ) -> Callable[[dict], float]:
#     """Function to initialize VQE process

#     Args:
#         ansatz (qk.circuit.quantumcircuit.QuantumCircuit): variational ansatz (parameterized quantum circuit)
#         qubo (dict[tuple[int, int], float]): QUBO Matrix
#         progress_history (list): list to store the progress of the minimization
#         num_shots (int, optional): The number of shots. If None, it calculates the exact expectation values. Otherwise, it samples from normal distributions with standard errors as standard deviations using normal distribution approximation. Defaults to None.

#     Returns:
#         Callable[[dict], float]: function to be minimized by classical optimizer
#     """
#     num_cbits = max(max(i, j) for i, j in qubo.keys()) + 1
#     num_registar_bits = np.ceil(np.log2(num_cbits)).astype(int)

#     # get expectation values from a circuit
#     # get a list of H (observables), which is a list of SparsePauliOp
#     H = define_pauli_op(num_registar_bits)
#     Ha = define_pauli_op(num_registar_bits, ancilla=True)

#     cost_function = initialize_cost_function(qubo, num_cbits)

#     estimator = Estimator()
#     if num_shots is not None:
#         estimator.set_options(shots=num_shots)

#     def func(theta: dict) -> float:
#         """Function to be minimized by classical optimizer

#         Args:
#             theta (dict): parameters of the circuit

#         Returns:
#             float: value of the cost function
#         """

#         job1 = estimator.run([ansatz] * len(H), H, [theta] * len(H))
#         P = job1.result()

#         job2 = estimator.run([ansatz] * len(Ha), Ha, [theta] * len(Ha))
#         P1 = job2.result()

#         one_coeffs = P1.values / P.values
        
#         cost = cost_function(one_coeffs)

#         return cost

#     return func


# def get_ancilla_prob(
#     circuit: qk.circuit.quantumcircuit.QuantumCircuit,
#     theta: dict,
#     num_register_bits: int,
#     num_classical_bits: int,
# ) -> np.array:
#     """Function to get final binary list from the circuit and optimised parameters.

#     Args:
#         circuit (qk.circuit.quantumcircuit.QuantumCircuit): parameterised quantum circuit
#         theta (dict): optimised parameters
#         num_register_bits (int): number of register qubits
#         num_classical_bits (int): number of classical bits

#     Returns:
#         np.array: final binary list
#     """
#     estimator = Estimator()
#     # define observable to calculate expectation value
#     H = define_pauli_op(num_register_bits, ancilla=False)
#     Ha = define_pauli_op(num_register_bits, ancilla=True)
#     # get expectation values from
#     job1 = estimator.run([circuit] * len(H), H, [theta] * len(H))
#     P = job1.result()

#     job2 = estimator.run([circuit] * len(Ha), Ha, [theta] * len(Ha))
#     P1 = job2.result()
#     prob_one = (P1.values / P.values) ** 2
#     final_binary = np.array(prob_one >= 0.5, dtype=int)

#     return final_binary[:num_classical_bits]
