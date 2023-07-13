from __future__ import annotations

import dataclasses
import typing as typ
from abc import ABC, abstractmethod

import qiskit.quantum_info as qk_info
import jijmodeling as jm
import jijmodeling_transpiler as jmt
from jijmodeling_transpiler_quantum.core.ising_qubo import qubo_to_ising
from jijmodeling_transpiler_quantum.core.qrac.graph_coloring import greedy_graph_coloring
from .qrao31 import qrac31_encode_ising, Pauli
from .qrao21 import qrac21_encode_ising


class QRACBuilder(ABC):
    def __init__(self, pubo_builder, compiled_instance) -> None:
        self.pubo_builder = pubo_builder
        self.compiled_instance = compiled_instance

    @abstractmethod
    def get_hamiltonian(
        self, multipliers=None, detail_parameter=None
    ) -> tuple[qk_info.SparsePauliOp, float, QRACEncodingCache]:
        pass

    def decode_from_binary_values(
        self, binary_list: typ.Iterable[list[int]]
    ) -> jm.SampleSet:
        binary_results = [
            {i: value for i, value in enumerate(binary)} for binary in binary_list
        ]
        binary_encoder = self.pubo_builder.binary_encoder
        decoded: jm.SampleSet = (
            jmt.core.pubo.binary_decode.decode_from_dict_binary_result(
                binary_results, binary_encoder, self.compiled_instance
            )
        )
        decoded.record.num_occurrences = [[1]] * len(binary_results)
        return decoded


@dataclasses.dataclass
class QRACEncodingCache:
    color_group: dict[int, list[int]]
    encoding: dict[int, tuple[int, Pauli]]


class QRAC31Builder(QRACBuilder):
    def get_hamiltonian(
        self, multipliers=None, detail_parameter=None
    ) -> tuple[qk_info.SparsePauliOp, float, QRACEncodingCache]:
        qubo, constant = self.pubo_builder.get_qubo_dict(
            multipliers=multipliers, detail_parameters=detail_parameter
        )
        ising = qubo_to_ising(qubo)
        _, color_group = greedy_graph_coloring(
            ising.quad.keys(), max_color_group_size=3
        )
        qrac_hamiltonian, offset, encoding = qrac31_encode_ising(ising, color_group)
        return (
            qrac_hamiltonian,
            offset + constant,
            QRACEncodingCache(color_group, encoding),
        )


def transpile_to_qrac31_hamiltonian(compiled_instance, normalize=True) -> QRAC31Builder:
    pubo_builder = jmt.core.pubo.transpile_to_pubo(
        compiled_instance, normalize=normalize
    )
    return QRAC31Builder(pubo_builder, compiled_instance)


class QRAC21Builder(QRACBuilder):
    def get_hamiltonian(
        self, multipliers=None, detail_parameter=None
    ) -> tuple[qk_info.SparsePauliOp, float, QRACEncodingCache]:
        qubo, constant = self.pubo_builder.get_qubo_dict(
            multipliers=multipliers, detail_parameters=detail_parameter
        )
        ising = qubo_to_ising(qubo)
        _, color_group = greedy_graph_coloring(
            ising.quad.keys(), max_color_group_size=2
        )
        qrac_hamiltonian, offset, encoding = qrac21_encode_ising(ising, color_group)
        return (
            qrac_hamiltonian,
            offset + constant,
            QRACEncodingCache(color_group, encoding),
        )


def transpile_to_qrac21_hamiltonian(compiled_instance, normalize=True) -> QRAC21Builder:
    pubo_builder = jmt.core.pubo.transpile_to_pubo(
        compiled_instance, normalize=normalize
    )
    return QRAC21Builder(pubo_builder, compiled_instance)
