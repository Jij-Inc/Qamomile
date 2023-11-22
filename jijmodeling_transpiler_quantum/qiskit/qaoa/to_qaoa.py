from __future__ import annotations

import typing as typ

import jijmodeling as jm
import jijmodeling_transpiler as jmt
import qiskit as qk
import qiskit.quantum_info as qk_info

from jijmodeling_transpiler_quantum.core import qubo_to_ising

from .ising_hamiltonian import to_ising_operator_from_qubo


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
        multipliers: typ.Optional[dict[str, float]] = None,
        detail_parameters: typ.Optional[
            dict[str, dict[tuple[int, ...], tuple[float, float]]]
        ] = None,
    ) -> tuple[qk_info.SparsePauliOp, float]:
        qubo, constant = self.pubo_builder.get_qubo_dict(
            multipliers=multipliers, detail_parameters=detail_parameters
        )
        ising_operator, ising_const = to_ising_operator_from_qubo(
            qubo, self.num_vars
        )
        return ising_operator, ising_const + constant

    def get_qaoa_ansatz(
        self,
        p: int,
        multipliers: typ.Optional[dict[str, float]] = None,
        detail_parameters: typ.Optional[
            dict[str, dict[tuple[int, ...], tuple[float, float]]]
        ] = None,
    ):
        ising_operator, constant = self.get_hamiltonian(
            multipliers=multipliers, detail_parameters=detail_parameters
        )
        qaoa_ansatz = qk.circuit.library.QAOAAnsatz(ising_operator, reps=p)
        return qaoa_ansatz, ising_operator, constant

    def decode_from_counts(self, counts: dict[str, int]) -> jm.SampleSet:
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

    def decode_from_quasi_dist(
        self, quasi_dist: qk.result.QuasiDistribution
    ) -> jm.SampleSet:
        binary_prob: dict[str, float] = quasi_dist.binary_probabilities()

        shots: int
        if quasi_dist.shots:
            shots = quasi_dist.shots
        else:
            if self.num_vars < 15:
                shots = max(2**self.num_vars, 100)
            else:
                shots = 30000

        binary_counts = {
            key: int(prob * shots) for key, prob in binary_prob.items()
        }

        return self.decode_from_counts(binary_counts)


def transpile_to_qaoa_ansatz(
    compiled_instance: jmt.core.CompiledInstance,
    normalize: bool = True,
    relax_method: jmt.core.pubo.RelaxationMethod = jmt.core.pubo.RelaxationMethod.AugmentedLagrangian,
) -> QAOAAnsatzBuilder:
    pubo_builder = jmt.core.pubo.transpile_to_pubo(
        compiled_instance, normalize=normalize, relax_method=relax_method
    )
    var_num = compiled_instance.var_map.var_num
    return QAOAAnsatzBuilder(pubo_builder, var_num, compiled_instance)
