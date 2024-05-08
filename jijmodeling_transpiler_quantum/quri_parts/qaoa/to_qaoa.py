from __future__ import annotations

from math import pi

import jijmodeling as jm
import jijmodeling_transpiler as jmt
import numpy as np
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit
from quri_parts.core.operator import Operator, pauli_label
from jijmodeling_transpiler_quantum.core import qubo_to_ising, IsingModel


def ising_to_hamiltonian(
    ising: IsingModel,
) -> tuple[Operator, float]:
    quri_operator = Operator()
    for idx, coeff in ising.linear.items():
        if coeff != 0.0:
            quri_operator += Operator({pauli_label(f"Z{idx}"): coeff})

    for (i, j), coeff in ising.quad.items():
        if coeff != 0.0:
            quri_operator += Operator({pauli_label(f"Z{i} Z{j}"): coeff})

    return quri_operator, ising.constant


class QAOAAnsatzBuilder:
    def __init__(
        self,
        pubo_builder: jmt.core.pubo.PuboBuilder,
        num_vars: int,
        compiled_instance: jmt.core.CompiledInstance,
    ):
        """Initialize the QAOAAnsatzBuilder.

        Args:
            pubo_builder (jmt.core.pubo.PuboBuilder): The PUBO builder to be used.
            num_vars (int): The number of variables.
            compiled_instance (jmt.core.CompiledInstance): The compiled instance to be used.
        """
        self.pubo_builder = pubo_builder
        self.num_vars = num_vars
        self.compiled_instance = compiled_instance

    @property
    def var_map(self) -> dict[str, tuple[int, ...]]:
        return self.compiled_instance.var_map.var_map

    def get_ising_dict(
        self,
        multipliers: dict = None,
        detail_parameters: dict = None,
    ) -> IsingModel:
        """Get the Ising dictionary.

        Args:
            multipliers (dict, optional): Multipliers for the Ising Hamiltonian. Defaults to None.
            detail_parameters (dict, optional): Detailed parameters for the Ising Hamiltonian. Defaults to None.

        Returns:
            tuple[dict[tuple[int, int], float], dict[int, float], float]: The Ising dictionary, the constant offset, and the constant offset.
        """
        qubo, constant = self.pubo_builder.get_qubo_dict(
            multipliers=multipliers, detail_parameters=detail_parameters
        )
        ising = qubo_to_ising(qubo)
        ising.constant += constant
        return ising

    def get_hamiltonian(
        self,
        multipliers: dict = None,
        detail_parameters: dict = None,
    ) -> tuple[Operator, float]:
        """Get the Ising Hamiltonian.

        Args:
            multipliers (dict, optional): Multipliers for the Ising Hamiltonian. Defaults to None.
            detail_parameters (dict, optional): Detailed parameters for the Ising Hamiltonian. Defaults to None.

        Returns:
            tuple[Operator, float]: The Ising operator and the constant offset.
        """
        ising = self.get_ising_dict(multipliers, detail_parameters)

        return ising_to_hamiltonian(ising)

    def get_qaoa_ansatz(
        self,
        p: int,
        multipliers: dict = None,
        detail_parameters: dict = None,
    ) -> tuple[LinearMappedUnboundParametricQuantumCircuit, Operator, float]:
        """Get the QAOA ansatz.

        Args:
            p (int): The number of layers in the QAOA circuit.
            multipliers (dict, optional): Multipliers for the Ising Hamiltonian. Defaults to None.
            detail_parameters (dict, optional): Detailed parameters for the Ising Hamiltonian. Defaults to None.

        Returns:
            tuple[LinearMappedUnboundParametricQuantumCircuit, Operator, float]: The QAOA ansatz, the Ising operator, and the constant offset.
        """
        ising = self.get_ising_dict(multipliers, detail_parameters)
        QAOAAnsatz = LinearMappedUnboundParametricQuantumCircuit(self.num_vars)
        # Create the QAOA ansatz
        for i in range(self.num_vars):
            QAOAAnsatz.add_H_gate(i)

        for p_level in range(p):
            gamma = QAOAAnsatz.add_parameter(f"gamma{p_level}")
            beta = QAOAAnsatz.add_parameter(f"beta{p_level}")

            for idx, coeff in ising.linear.items():
                if coeff != 0.0:
                    QAOAAnsatz.add_ParametricRZ_gate(
                        idx,
                        angle={gamma: 2 * coeff}
                    )
            for (i, j), coeff in ising.quad.items():
                if coeff != 0.0:
                    QAOAAnsatz.add_ParametricPauliRotation_gate(
                        [i, j],
                        pauli_ids=(3, 3),
                        angle={gamma: 2 * coeff},
                    )
            for i in range(self.num_vars):
                QAOAAnsatz.add_ParametricRX_gate(
                    i, {beta: 2}
                )

        ising_operator, constant = ising_to_hamiltonian(ising)

        return QAOAAnsatz, ising_operator, constant

    def decode_from_counts(self, counts: dict[str, int]) -> jm.SampleSet:
        """Decode the result from the counts.

        Args:
            counts (dict[str, int]): The counts to be decoded.

        Returns:
            jm.SampleSet: The decoded sample set.
        """
        samples = []
        num_occurrences = []
        for binary_str, count_num in counts.items():
            binary_values = {idx: int(b) for idx, b in enumerate(binary_str)}
            samples.append(binary_values)
            num_occurrences.append(count_num)

        binary_encoder = self.pubo_builder.binary_encoder
        decoded: jm.SampleSet = (
            jmt.core.pubo.binary_decode.decode_from_dict_binary_result(
                samples, binary_encoder, self.compiled_instance
            )
        )
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

    def decode_from_probs(self, probs: np.array) -> jm.SampleSet:
        """Decode the result from the probabilities.

        Args:
            probs (np.array): The probabilities to be decoded.

        Returns:
            jm.SampleSet: The decoded sample set.
        """
        shots = 10000
        z_basis = [
            format(i, "b").zfill(self.num_vars) for i in range(len(probs))
        ]
        binary_counts = {
            i: int(value * shots) for i, value in zip(z_basis, probs)
        }

        return self.decode_from_counts(binary_counts)


def transpile_to_qaoa_ansatz(
    compiled_instance: jmt.core.CompiledInstance,
    normalize: bool = True,
    relax_method=jmt.core.pubo.RelaxationMethod.AugmentedLagrangian,
) -> QAOAAnsatzBuilder:
    """Transpile to a QAOA ansatz builder.

    Args:
        compiled_instance (jmt.core.CompiledInstance): The compiled instance to be used.
        normalize (bool, optional): Whether to normalize the objective function. Defaults to True.
        relax_method (jmt.core.pubo.RelaxationMethod, optional): The relaxation method to be used. Defaults to AugmentedLagrangian.

    Returns:
        QAOAAnsatzBuilder: The QAOA ansatz builder.
    """
    pubo_builder = jmt.core.pubo.transpile_to_pubo(
        compiled_instance, normalize=normalize, relax_method=relax_method
    )
    var_num = compiled_instance.var_map.var_num
    return QAOAAnsatzBuilder(pubo_builder, var_num, compiled_instance)
