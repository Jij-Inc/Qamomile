from __future__ import annotations
import jijmodeling as jm
import jijmodeling.transpiler as jmt
from jijmodeling_transpiler_quantum.core import qubo_to_ising
from .ising_hamiltonian import to_ising_operator_from_qubo
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit
from quri_parts.core.operator import Operator
from math import pi


class QAOAAnsatzBuilder:
    def __init__(
        self,
        pubo_builder: jmt.core.pubo.PuboBuilder,
        num_vars: int,
        compiled_instance: jmt.core.CompiledInstance,
    ):
        self.pubo_builder = pubo_builder
        self.num_vars = num_vars
        self.compiled_instance = compiled_instance

    @property
    def var_map(self) -> dict[str, tuple[int, ...]]:
        return self.compiled_instance.var_map.var_map

    def get_hamiltonian(
        self,
        multipliers: dict = None,
        detail_parameters: dict = None,
    ) -> tuple[Operator, float]:
        qubo, constant = self.pubo_builder.get_qubo_dict(
            multipliers=multipliers, detail_parameters=detail_parameters
        )
        ising_operator, ising_const = to_ising_operator_from_qubo(qubo, self.num_vars)
        return ising_operator, ising_const + constant

    def get_qaoa_ansatz(
        self,
        p: int,
        multipliers: dict = None,
        detail_parameters: dict = None,
    ) -> tuple[UnboundParametricQuantumCircuit, operator, float]:
        ising_operator, constant = self.get_hamiltonian(
            multipliers=multipliers, detail_parameters=detail_parameters
        )
        pauli_terms = []
        coeff_terms = []
        pauli_z = []
        pauli_zz = []
        pauli_z_coeff = []
        pauli_zz_coeff = []

        for i in range(ising_operator.n_terms - 1):
            keys = list(ising_operator.keys())[i]
            pauli_terms.append(keys.index_and_pauli_id_list[0])
            items = list(ising_operator.items())[i]
            coeff_terms.append(items[1])

        QAOAAnsatz = LinearMappedUnboundParametricQuantumCircuit(self.num_vars)

        pauli_z = pauli_terms[: self.num_vars]
        pauli_zz = pauli_terms[self.num_vars :]
        pauli_zz = [sorted(sublist) for sublist in pauli_zz]
        pauli_z_coeff = coeff_terms[: self.num_vars]
        pauli_zz_coeff = coeff_terms[self.num_vars :]

        for i in pauli_z:
            QAOAAnsatz.add_H_gate(i[0])

        for p_level in range(p):
            Gamma = QAOAAnsatz.add_parameters(f"gamma{p_level}")
            Beta = QAOAAnsatz.add_parameters(f"beta{p_level}")

            for pauli_z_info in zip(pauli_z, pauli_z_coeff):
                QAOAAnsatz.add_ParametricRZ_gate(
                    pauli_z_info[0][0], {Gamma[0]: 2 * pauli_z_info[1]}
                )

            for pauli_zz_info in zip(pauli_zz, pauli_zz_coeff):
                QAOAAnsatz.add_CNOT_gate(pauli_zz_info[0][0], pauli_zz_info[0][1])
                QAOAAnsatz.add_ParametricRZ_gate(
                    pauli_zz_info[0][1], {Gamma[0]: 2 * pauli_zz_info[1]}
                )
                QAOAAnsatz.add_CNOT_gate(pauli_zz_info[0][0], pauli_zz_info[0][1])

            for pauli_z_info2 in pauli_z:
                QAOAAnsatz.add_ParametricRX_gate(pauli_z_info2[0], {Beta[0]: 2})

        return QAOAAnsatz, ising_operator, constant

    def decode_from_counts(self, counts: dict[str, int]) -> jm.SampleSet:
        samples = []
        num_occurances = []
        for binary_str, count_num in counts.items():
            binary_values = {idx: int(b) for idx, b in enumerate(binary_str)}
            samples.append(binary_values)
            num_occurances.append(count_num)

        binary_encoder = self.pubo_builder.binary_encoder
        decoded: jm.SampleSet = (
            jmt.core.pubo.binary_decode.decode_from_dict_binary_result(
                samples, binary_encoder, self.compiled_instance
            )
        )
        decoded.record.num_occurrences = num_occurances
        return decoded

    def decode_from_probs(self, probs: np.array) -> jm.SampleSet:
        shots = 10000
        z_basis = [format(i, "b").zfill(self.num_vars) for i in range(len(probs))]
        binary_counts = {i: int(value * shots) for i, value in zip(z_basis, probs)}

        return self.decode_from_counts(binary_counts)


def transpile_to_qaoa_ansatz(
    compiled_instance: jmt.core.CompiledInstance,
    normalize: bool = True,
    relax_method=jmt.core.pubo.RelaxationMethod.AugmentedLagrangian,
) -> quriQAOAAnsatzBuilder:
    pubo_builder = jmt.core.pubo.transpile_to_pubo(
        compiled_instance, normalize=True, relax_method=relax_method
    )
    var_num = compiled_instance.var_map.var_num
    return QAOAAnsatzBuilder(pubo_builder, var_num, compiled_instance)
