"""Base class for QRAC-based converters.

This module provides the QRACConverterBase class that implements
shared functionality across all QRAC converter variants.
"""

from __future__ import annotations

import abc
from typing import Generic, TypeVar

import ommx.v1

import qamomile.observable as qm_o
from qamomile.circuit.transpiler.job import SampleResult
from qamomile.optimization.binary_model import BinarySampleSet
from qamomile.optimization.converter import MathematicalProblemConverter

EncoderT = TypeVar("EncoderT")


class QRACConverterBase(MathematicalProblemConverter, abc.ABC, Generic[EncoderT]):
    """Abstract base for all QRAC-based converters.

    Subclasses must implement:

    - ``num_qubits`` (property)
    - ``get_encoded_pauli_list()``

    Ansatz Construction:
        This converter does not provide a built-in ansatz, giving users
        full control over their variational circuit design. Use building
        blocks from :mod:`qamomile.circuit.algorithm.basic` to compose
        your own ansatz:

        .. code-block:: python

            import qamomile.circuit as qmc
            from qamomile.circuit.algorithm.basic import (
                ry_layer, rz_layer, cz_entangling_layer,
            )

            @qmc.qkernel
            def my_ansatz(
                n: qmc.UInt,
                depth: qmc.UInt,
                thetas: qmc.Vector[qmc.Float],
            ) -> qmc.Vector[qmc.Qubit]:
                q = qmc.allocate(n)
                for i in qmc.range(n):
                    q[i] = qmc.h(q[i])
                for d in qmc.range(depth):
                    offset = d * 2 * n
                    q = ry_layer(q, thetas, offset)
                    q = rz_layer(q, thetas, offset + n)
                    q = cz_entangling_layer(q)
                return q

        Available layers in ``qamomile.circuit.algorithm.basic``:

        - ``rx_layer``, ``ry_layer``, ``rz_layer``: Single-qubit rotation layers
        - ``cz_entangling_layer``: CZ entangling with linear connectivity

        The total number of variational parameters depends on the ansatz
        design. For the example above it is ``2 * num_qubits * depth``.
    """

    _encoder: EncoderT

    def __post_init__(self) -> None:
        if self.spin_model.higher:
            raise ValueError(
                "QRAC converters do not support higher-order (HUBO) terms. "
                "All interaction terms must be at most quadratic."
            )

    @property
    @abc.abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits after QRAC encoding."""
        ...

    @abc.abstractmethod
    def get_encoded_pauli_list(self) -> list[qm_o.Hamiltonian]: ...

    @property
    def encoder(self) -> EncoderT:
        """The QRAC encoder used by this converter."""
        return self._encoder

    def decode(  # type: ignore[override]
        self,
        rounded_spins_list: list[list[int]],
    ) -> BinarySampleSet | ommx.v1.SampleSet:
        """Decode rounded QRAC spin assignments.

        Unlike the base ``MathematicalProblemConverter.decode``, which
        consumes raw quantum measurement results from sampling, QRAO
        produces problem-variable spins via Pauli expectation rounding
        (see :class:`qamomile.optimization.qrao.SignRounder`). This
        override accepts those spin lists, packages them into a
        synthetic :class:`SampleResult`, and delegates to the base
        class's polymorphic decode — so QRAO converters share the same
        OMMX round-trip behavior as QAOA / FQAOA: an OMMX-backed
        converter returns an :class:`ommx.v1.SampleSet`, a
        :class:`BinaryModel`-backed converter returns a
        :class:`BinarySampleSet`.

        Args:
            rounded_spins_list: List of rounded spin assignments. Each
                inner list contains ±1 values, one per spin-model
                variable, in the order matching ``self.spin_model``'s
                variable indexing. Length per inner list must equal
                ``self.spin_model.num_bits``.

        Returns:
            BinarySampleSet | ommx.v1.SampleSet: see method description.

        Raises:
            ValueError: If any inner list has the wrong length or
                contains values other than +1 / -1.
        """
        n = self.spin_model.num_bits
        results: list[tuple[list[int], int]] = []
        for spins in rounded_spins_list:
            if len(spins) != n:
                raise ValueError(
                    f"each spin list must have length {n} "
                    f"(spin_model.num_bits); got {len(spins)}"
                )
            if not all(s in (1, -1) for s in spins):
                raise ValueError(f"all spin values must be +1 or -1; got {spins}")
            # Map ±1 spin → 0/1 measurement bit via bit = (1 - spin) // 2.
            # Matches BinaryModel.decode_from_sampleresult's SPIN convention
            # (measurement 0 → +1, measurement 1 → -1).
            bits = [(1 - s) // 2 for s in spins]
            results.append((bits, 1))

        sample_result: SampleResult[list[int]] = SampleResult(
            results=results,
            shots=len(rounded_spins_list),
        )
        return super().decode(sample_result)
