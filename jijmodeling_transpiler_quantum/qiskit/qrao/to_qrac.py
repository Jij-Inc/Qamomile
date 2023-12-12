from __future__ import annotations

import dataclasses
import typing as typ
from abc import ABC, abstractmethod

import qiskit.quantum_info as qk_info
import jijmodeling as jm
import jijmodeling_transpiler as jmt
import jijmodeling_transpiler_quantum.core as jmt_qc

from .qrao31 import qrac31_encode_ising, Pauli
from .qrao21 import qrac21_encode_ising
from .qrao32 import qrac32_encode_ising
from .qrao_space_efficient import qrac_space_efficient_encode_ising


class QRACBuilder(ABC):
    def __init__(
        self,
        pubo_builder: jmt.core.pubo.PuboBuilder,
        compiled_instance: jmt.core.CompiledInstance,
    ) -> None:
        self.pubo_builder = pubo_builder
        self.compiled_instance = compiled_instance

    @abstractmethod
    def get_hamiltonian(
        self,
        multipliers: typ.Optional[dict[str, float]] = None,
        detail_parameters: typ.Optional[
            dict[str, dict[tuple[int, ...], tuple[float, float]]]
        ] = None,
    ) -> tuple[qk_info.SparsePauliOp, float, QRACEncodingCache]:
        pass

    def decode_from_binary_values(
        self, binary_list: typ.Iterable[list[int]]
    ) -> jm.SampleSet:
        binary_results = [
            {i: value for i, value in enumerate(binary)}
            for binary in binary_list
        ]
        binary_encoder = self.pubo_builder.binary_encoder
        decoded: jm.SampleSet = (
            jmt.core.pubo.binary_decode.decode_from_dict_binary_result(
                binary_results, binary_encoder, self.compiled_instance
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


@dataclasses.dataclass
class QRACEncodingCache:
    color_group: dict[int, list[int]]
    encoding: dict[int, tuple[int, Pauli]]


class QRAC31Builder(QRACBuilder):
    def get_hamiltonian(
        self,
        multipliers: typ.Optional[dict[str, float]] = None,
        detail_parameters: typ.Optional[
            dict[str, dict[tuple[int, ...], tuple[float, float]]]
        ] = None,
    ) -> tuple[qk_info.SparsePauliOp, float, QRACEncodingCache]:
        """Get Quantum Relaxation Hamiltonian based on (3,1,p)-QRAC.

        Args:
            multipliers (typ.Optional[dict[str, float]], optional): a multiplier for each penalty. Defaults to None.
            detail_parameters (typ.Optional[ dict[str, dict[tuple[int, ...], tuple[float, float]]] ], optional): detail parameters for each penalty. Defaults to None.

        Returns:
            tuple[qk_info.SparsePauliOp, float, QRACEncodingCache]: (3,1,p)-QRAC Hamiltonian, constant term, and encoding cache for decoding
        """
        qubo, constant = self.pubo_builder.get_qubo_dict(
            multipliers=multipliers, detail_parameters=detail_parameters
        )
        ising = jmt_qc.qubo_to_ising(qubo)
        max_color_group_size = 3
        _, color_group = jmt_qc.greedy_graph_coloring(
            ising.quad.keys(), max_color_group_size=max_color_group_size
        )
        color_group = jmt_qc.qrac.check_linear_term(
            color_group, ising.linear.keys(), max_color_group_size
        )
        qrac_hamiltonian, offset, encoding = qrac31_encode_ising(
            ising, color_group
        )
        return (
            qrac_hamiltonian,
            offset + constant,
            QRACEncodingCache(color_group, encoding),
        )


def transpile_to_qrac31_hamiltonian(
    compiled_instance: jmt.core.CompiledInstance, normalize: bool = True
) -> QRAC31Builder:
    """Generate Quantum Relaxation Hamiltonian based on (3,1,p)-QRAC builder.

        The generation method is based on the [B. Fuller et al., arXiv (2021)](https://arxiv.org/abs/2111.03167).

    Args:
        compiled_instance (jmt.core.CompiledInstance): Compiled model
        normalize (bool, optional): Normalize objective function. Defaults to True.

    Returns:
        QRAC31Builder: (3,1,p)-QRAC Hamiltonian builder
    """
    pubo_builder = jmt.core.pubo.transpile_to_pubo(
        compiled_instance, normalize=normalize
    )
    return QRAC31Builder(pubo_builder, compiled_instance)


class QRAC21Builder(QRACBuilder):
    def get_hamiltonian(
        self,
        multipliers: typ.Optional[dict[str, float]] = None,
        detail_parameters: typ.Optional[
            dict[str, dict[tuple[int, ...], tuple[float, float]]]
        ] = None,
    ) -> tuple[qk_info.SparsePauliOp, float, QRACEncodingCache]:
        """Get Quantum Relaxation Hamiltonian based on (2,1,p)-QRAC.

        Args:
            multipliers (typ.Optional[dict[str, float]], optional): a multiplier for each penalty. Defaults to None.
            detail_parameters (typ.Optional[ dict[str, dict[tuple[int, ...], tuple[float, float]]] ], optional): detail parameters for each penalty. Defaults to None.

        Returns:
            tuple[qk_info.SparsePauliOp, float, QRACEncodingCache]: (2,1,p)-QRAC Hamiltonian, constant term, and encoding cache for decoding
        """
        qubo, constant = self.pubo_builder.get_qubo_dict(
            multipliers=multipliers, detail_parameters=detail_parameters
        )
        ising = jmt_qc.qubo_to_ising(qubo)
        max_color_group_size = 2

        _, color_group = jmt_qc.greedy_graph_coloring(
            ising.quad.keys(), max_color_group_size=max_color_group_size
        )
        color_group = jmt_qc.qrac.check_linear_term(
            color_group, ising.linear.keys(), max_color_group_size
        )
        qrac_hamiltonian, offset, encoding = qrac21_encode_ising(
            ising, color_group
        )
        return (
            qrac_hamiltonian,
            offset + constant,
            QRACEncodingCache(color_group, encoding),
        )


def transpile_to_qrac21_hamiltonian(
    compiled_instance: jmt.core.CompiledInstance, normalize: bool = True
) -> QRAC21Builder:
    """Generate Quantum Relaxation Hamiltonian based on (2,1,p)-QRAC builder.

        The generation method is based on the [B. Fuller et al., arXiv (2021)](https://arxiv.org/abs/2111.03167).

    Args:
        compiled_instance (jmt.core.CompiledInstance): Compiled model
        normalize (bool, optional): Normalize objective function. Defaults to True.

    Returns:
        QRAC21Builder: (2,1,p)-QRAC Hamiltonian builder
    """
    pubo_builder = jmt.core.pubo.transpile_to_pubo(
        compiled_instance, normalize=normalize
    )
    return QRAC21Builder(pubo_builder, compiled_instance)


class QRAC32Builder(QRACBuilder):
    def get_hamiltonian(
        self,
        multipliers: typ.Optional[dict[str, float]] = None,
        detail_parameters: typ.Optional[
            dict[str, dict[tuple[int, ...], tuple[float, float]]]
        ] = None,
    ) -> tuple[qk_info.SparsePauliOp, float, QRACEncodingCache]:
        """Get Quantum Relaxation Hamiltonian based on (3,2,p)-QRAC.

        Args:
            multipliers (typ.Optional[dict[str, float]], optional): a multiplier for each penalty. Defaults to None.
            detail_parameters (typ.Optional[ dict[str, dict[tuple[int, ...], tuple[float, float]]] ], optional): detail parameters for each penalty. Defaults to None.

        Returns:
            tuple[qk_info.SparsePauliOp, float, QRACEncodingCache]: (3,2,p)-QRAC Hamiltonian, constant term, and encoding cache for decoding
        """
        qubo, constant = self.pubo_builder.get_qubo_dict(
            multipliers=multipliers, detail_parameters=detail_parameters
        )
        ising = jmt_qc.qubo_to_ising(qubo)
        max_color_group_size = 3
        _, color_group = jmt_qc.greedy_graph_coloring(
            ising.quad.keys(), max_color_group_size=max_color_group_size
        )
        color_group = jmt_qc.qrac.check_linear_term(
            color_group, ising.linear.keys(), max_color_group_size
        )
        qrac_hamiltonian, offset, encoding = qrac32_encode_ising(
            ising, color_group
        )
        return (
            qrac_hamiltonian,
            offset + constant,
            QRACEncodingCache(color_group, encoding),
        )


def transpile_to_qrac32_hamiltonian(
    compiled_instance: jmt.core.CompiledInstance, normalize: bool = True
) -> QRAC32Builder:
    """Generate Quantum Relaxation Hamiltonian based on (3,2,p)-QRAC builder.

        The generation method is based on the [K. Teramoto et al., arXiv (2023)](https://arxiv.org/abs/2302.09481).

    Args:
        compiled_instance (jmt.core.CompiledInstance): Compiled model
        normalize (bool, optional): Normalize objective function. Defaults to True.

    Returns:
        QRAC32Builder: (3,2,p)-QRAC Hamiltonian builder
    """
    pubo_builder = jmt.core.pubo.transpile_to_pubo(
        compiled_instance, normalize=normalize
    )
    return QRAC32Builder(pubo_builder, compiled_instance)


class QRACSpaceEfficientBuilder(QRACBuilder):
    def get_hamiltonian(
        self,
        multipliers: typ.Optional[dict[str, float]] = None,
        detail_parameters: typ.Optional[
            dict[str, dict[tuple[int, ...], tuple[float, float]]]
        ] = None,
    ) -> tuple[qk_info.SparsePauliOp, float, QRACEncodingCache]:
        """Get Quantum Relaxation Hamiltonian based on space-efficient QRAC.

        Args:
            multipliers (typ.Optional[dict[str, float]], optional): a multiplier for each penalty. Defaults to None.
            detail_parameters (typ.Optional[ dict[str, dict[tuple[int, ...], tuple[float, float]]] ], optional): detail parameters for each penalty. Defaults to None.

        Returns:
            tuple[qk_info.SparsePauliOp, float, QRACEncodingCache]: Space-efficient QRAC Hamiltonian, constant term, and encoding cache for decoding
        """
        qubo, constant = self.pubo_builder.get_qubo_dict(
            multipliers=multipliers, detail_parameters=detail_parameters
        )
        ising = jmt_qc.qubo_to_ising(qubo)
        qrac_hamiltonian, offset, encoding = qrac_space_efficient_encode_ising(
            ising
        )
        return (
            qrac_hamiltonian,
            offset + constant,
            QRACEncodingCache(color_group={}, encoding=encoding),
        )


def transpile_to_qrac_space_efficient_hamiltonian(
    compiled_instance: jmt.core.CompiledInstance, normalize=True
) -> QRACSpaceEfficientBuilder:
    """Generate Quantum Relaxation Hamiltonian based on space-efficient QRAC builder.

        This relaxation method is based on the [K. Teramoto et al., arXiv (2023)](https://arxiv.org/abs/2302.09481).
        In this paper, this method is called as Space Compression Ratio Preserving Quantum Relaxation.

    Args:
        compiled_instance (jmt.core.CompiledInstance): Compiled model
        normalize (bool, optional): Normalize objective function. Defaults to True.

    Returns:
        QRACSpaceEfficientBuilder: Space-efficient QRAC Hamiltonian builder
    """
    pubo_builder = jmt.core.pubo.transpile_to_pubo(
        compiled_instance, normalize=normalize
    )
    return QRACSpaceEfficientBuilder(pubo_builder, compiled_instance)
