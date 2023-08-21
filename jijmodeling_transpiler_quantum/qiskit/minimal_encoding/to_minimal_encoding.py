from __future__ import annotations
import typing as typ
from collections.abc import Callable
import numpy as np
import qiskit as qk
import qiskit.quantum_info as qk_info
from qiskit.primitives import Estimator
import jijmodeling as jm
import jijmodeling_transpiler as jmt
from .cost_function import (
    initialize_cost_function,
    define_pauli_op,
)


class MinimalEncodingBuilder:
    def __init__(
        self,
        pubo_builder: jmt.core.pubo.PuboBuilder,
        compiled_instance: jmt.core.CompiledInstance,
    ):
        self.pubo_builder = pubo_builder
        self.compiled_instance = compiled_instance

    def get_encoded_instance(
        self,
        multipliers: typ.Optional[dict[str, float]] = None,
        detail_parameters: typ.Optional[
            dict[str, dict[tuple[int, ...], tuple[float, float]]]
        ] = None,
    ) -> tuple[Callable[[np.array], float], float]:
        qubo, constant = self.pubo_builder.get_qubo_dict(
            multipliers=multipliers, detail_parameters=detail_parameters
        )
        return MinimalEncodedInstance(qubo, constant, self)


def transpiler_to_minimal_encoding(
    compiled_instance: jmt.core.CompiledInstance, normalize: bool = True
) -> MinimalEncodingBuilder:
    pubo_builder = jmt.core.pubo.transpile_to_pubo(
        compiled_instance, normalize=normalize
    )
    """Function to transpile a compiled instance to a minimal encoding builder
    
        Generating Minimal Encoding Builder from a compiled instance.
        Minimal encoding is a method to reduce the number of qubits and classical bits.
        The method is first proposed by Tan, Benjamin, et al. "Qubit-efficient encoding schemes for binary optimisation problems." Quantum 5 (2021): 454.
        More information is on https://quantum-journal.org/papers/q-2021-05-04-454/

    Returns:
        MinimalEncodingBuilder: minimal encoding builder
    """

    return MinimalEncodingBuilder(pubo_builder, compiled_instance)


class MinimalEncodedInstance:
    def __init__(
        self,
        qubo: dict[tuple[int, int], float],
        constant: float,
        minimal_encoding_builder: MinimalEncodingBuilder,
    ):
        self.qubo = qubo
        self.constant = constant
        self.minimal_encoding_builder = minimal_encoding_builder
        self._calculate_register_qubits(self.qubo)

    def _calculate_register_qubits(self, qubo: dict[tuple[int, int], float]):
        """Function to calculate the number of register qubits

        Args:
            qubo (dict[tuple[int, int], float]): QUBO Matrix

        Returns:
            int: number of register qubits
        """
        self.num_cbits = max(max(i, j) for i, j in qubo.keys()) + 1
        self.num_register_bits = np.ceil(np.log2(self.num_cbits)).astype(int)

    def get_cost_function(self) -> Callable[[np.array], float]:
        return initialize_cost_function(self.qubo, self.num_cbits)

    def get_minimized_function(
        self,
        ansatz: qk.circuit.quantumcircuit.QuantumCircuit,
        num_shots: int = None,
    ) -> Callable[[dict], float]:
        """Function to initialize VQE process

        Args:
            ansatz (qk.circuit.quantumcircuit.QuantumCircuit): variational ansatz (parameterized quantum circuit)
            progress_history (list): list to store the progress of the minimization
            num_shots (int, optional): The number of shots. If None, it calculates the exact expectation values. Otherwise, it samples from normal distributions with standard errors as standard deviations using normal distribution approximation. Defaults to None.

        Returns:
            Callable[[dict], float]: function to be minimized by classical optimizer
        """
        # get expectation values from a circuit
        # get a list of H (observables), which is a list of SparsePauliOp
        H = define_pauli_op(self.num_register_bits)
        Ha = define_pauli_op(self.num_register_bits, ancilla=True)

        cost_function = self.get_cost_function()

        estimator = Estimator()
        if num_shots is not None:
            estimator.set_options(shots=num_shots)

        def minimized_func(theta: dict) -> float:
            """Function to be minimized by classical optimizer

            Args:
                theta (dict): parameters of the circuit

            Returns:
                float: value of the cost function
            """

            job1 = estimator.run([ansatz] * len(H), H, [theta] * len(H))
            P = job1.result()

            job2 = estimator.run([ansatz] * len(Ha), Ha, [theta] * len(Ha))
            P1 = job2.result()

            one_coeffs = P1.values / P.values

            cost = cost_function(one_coeffs)

            return cost

        return minimized_func

    def get_optimized_state(
        self, circuit: qk.circuit.quantumcircuit.QuantumCircuit, theta: dict
    ) -> jm.SampleSet:
        """Function to get final binary list from the circuit and optimised parameters.

        Args:
            circuit (qk.circuit.quantumcircuit.QuantumCircuit): parameterised quantum circuit
            theta (dict): optimised parameters
            num_register_bits (int): number of register qubits
            num_classical_bits (int): number of classical bits

        Returns:
            jm.Sampleset: samplset of the results
        """
        estimator = Estimator()
        # define observable to calculate expectation value
        H = define_pauli_op(self.num_register_bits, ancilla=False)
        Ha = define_pauli_op(self.num_register_bits, ancilla=True)
        # get expectation values from
        job1 = estimator.run([circuit] * len(H), H, [theta] * len(H))
        P = job1.result()

        job2 = estimator.run([circuit] * len(Ha), Ha, [theta] * len(Ha))
        P1 = job2.result()
        prob_one = (P1.values / P.values) ** 2
        final_binary = np.array(prob_one >= 0.5, dtype=int)

        sample_set = self._decode_from_binary_values(
            [final_binary[: self.num_cbits]]
        )

        return sample_set

    def _decode_from_binary_values(
        self, binary_list: typ.Iterable[list[int]]
    ) -> jm.SampleSet:
        binary_results = [
            {i: value for i, value in enumerate(binary)}
            for binary in binary_list
        ]
        binary_encoder = (
            self.minimal_encoding_builder.pubo_builder.binary_encoder
        )
        decoded: jm.SampleSet = (
            jmt.core.pubo.binary_decode.decode_from_dict_binary_result(
                binary_results,
                binary_encoder,
                self.minimal_encoding_builder.compiled_instance,
            )
        )
        # decoded.record.num_occurrences = [[1]] * len(binary_results)
        return decoded
