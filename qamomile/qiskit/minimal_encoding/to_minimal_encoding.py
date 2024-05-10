from __future__ import annotations

import typing as typ
from collections.abc import Callable, Sequence

import jijmodeling as jm
import jijmodeling_transpiler as jmt
import numpy as np
import qiskit as qk
import qiskit.quantum_info as qk_info
from qiskit.primitives import Estimator

from .cost_function import define_pauli_op, initialize_cost_function


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
        """method to get encoded instance

        Args:
            multipliers (typ.Optional[dict[str, float]], optional): a multiplier for each penalty. Defaults to None.
            detail_parameters (typ.Optional[ dict[str, dict[tuple[int, ...], tuple[float, float]]] ], optional): detail parameters for each penalty. Defaults to None.

        Returns:
            tuple[Callable[[np.array], float], float]: _description_
        """
        qubo, constant = self.pubo_builder.get_qubo_dict(
            multipliers=multipliers, detail_parameters=detail_parameters
        )
        return MinimalEncodedInstance(qubo, constant, self)


def transpile_to_minimal_encoding(
    compiled_instance: jmt.core.CompiledInstance, normalize: bool = True
) -> MinimalEncodingBuilder:
    """Function to transpile a compiled instance to a minimal encoding builder

        Generating Minimal Encoding Builder from a compiled instance.
        Minimal encoding is a method to reduce the number of qubits and classical bits.
        The method is first proposed by Tan, Benjamin, et al. "Qubit-efficient encoding schemes for binary optimisation problems." Quantum 5 (2021): 454.
        More information is on https://quantum-journal.org/papers/q-2021-05-04-454/

    Args:
        compiled_instance (jmt.core.CompiledInstance): compiled instance
        normalize (bool, optional): whether to normalize the coefficients or not. Defaults to True.

    Returns:
        MinimalEncodingBuilder: minimal encoding builder
    """

    pubo_builder = jmt.core.pubo.transpile_to_pubo(
        compiled_instance, normalize=normalize
    )
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

    def _get_cost_function(self) -> Callable[[np.array], float]:
        return initialize_cost_function(self.qubo, self.num_cbits)

    def get_minimized_function(
        self,
        ansatz: qk.circuit.quantumcircuit.QuantumCircuit,
        num_shots: int = None,
    ) -> Callable[[dict], float]:
        """Method to generate cost function which should be minimized by classical optimizer

            In minimal encoding, rather than directly minimizing the value of the Hamiltonian,
            we minimize the cost function calculated using the expected values obtained from the quantum device.
            Therefore, this function returns a function that carries out a process not just for the Hamiltonian,
            but also for the calculation of expected values and the subsequent calculation of the cost function.

        Args:
            ansatz (qk.circuit.quantumcircuit.QuantumCircuit): variational ansatz (parameterized quantum circuit)
            num_shots (int, optional): The number of shots. If None, it calculates the exact expectation values. Otherwise, it samples from normal distributions with standard errors as standard deviations using normal distribution approximation. Defaults to None.

        Returns:
            Callable[[dict], float]: function to be minimized by classical optimizer
        """
        # get expectation values from a circuit
        # get a list of H (observables), which is a list of SparsePauliOp
        H = define_pauli_op(self.num_register_bits)
        Ha = define_pauli_op(self.num_register_bits, ancilla=True)

        cost_function = self._get_cost_function()

        estimator = Estimator()
        if num_shots is not None:
            estimator.set_options(shots=num_shots)

        def minimized_func(params: Sequence[float]) -> float:
            """Function to be minimized by classical optimizer

            Args:
                params (Sequence[float]): parameters of the circuit

            Returns:
                float: value of the cost function
            """

            job1 = estimator.run([ansatz] * len(H), H, [params] * len(H))
            P = job1.result()

            job2 = estimator.run([ansatz] * len(Ha), Ha, [params] * len(Ha))
            P1 = job2.result()

            one_coeffs = P1.values / P.values

            cost = cost_function(one_coeffs)

            return cost

        return minimized_func

    def get_optimized_state(
        self,
        circuit: qk.circuit.quantumcircuit.QuantumCircuit,
        params: Sequence[float],
    ) -> jm.SampleSet:
        """Function to get final binary list from the circuit and optimised parameters.

            In minimal encoding, we estimate the state not through simple quantum state sampling,
            but as a specific amount of expected values.
            Therefore, in this function, we calculate the expected values for state estimation based on the parameters obtained through optimization.

        Args:
            circuit (qk.circuit.quantumcircuit.QuantumCircuit): parameterised quantum circuit
            params (Sequence[float]): optimised parameters

        Returns:
            jm.Sampleset: samplset of the results
        """
        estimator = Estimator()
        # define observable to calculate expectation value
        H = define_pauli_op(self.num_register_bits, ancilla=False)
        Ha = define_pauli_op(self.num_register_bits, ancilla=True)
        # get expectation values from
        job1 = estimator.run([circuit] * len(H), H, [params] * len(H))
        P = job1.result()

        job2 = estimator.run([circuit] * len(Ha), Ha, [params] * len(Ha))
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

        num_occurrences = [1] * len(binary_results)
        decoded = jm.SampleSet(
            record=jm.Record(
                num_occurrences=num_occurrences,
                solution=decoded.record.solution,
            ),
            evaluation=decoded.evaluation,
            measuring_time=decoded.measuring_time,
            metadata=decoded.metadata,
        )
        return decoded
