"""Base class for QRAC-based converters.

This module provides the QRACConverterBase class that implements
shared functionality across all QRAC converter variants.
"""

from __future__ import annotations
import abc

import qamomile.observable as qm_o
from qamomile.optimization.binary_model import BinarySampleSet, VarType
from qamomile.optimization.converter import MathematicalProblemConverter


class QRACConverterBase(MathematicalProblemConverter):
    """Abstract base for all QRAC-based converters.

    Provides shared methods for result decoding and energy calculation.
    Subclasses must implement:

    - ``num_qubits`` (property)
    - ``get_cost_hamiltonian()``
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

    @property
    @abc.abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits after QRAC encoding."""
        ...

    @abc.abstractmethod
    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian: ...

    @abc.abstractmethod
    def get_encoded_pauli_list(self) -> list[qm_o.Hamiltonian]: ...

    def decode_from_rounded(self, spins: list[int]) -> BinarySampleSet:
        """Decode rounded spin values into a BinarySampleSet.

        Args:
            spins: List of spin values (+1 or -1) for each variable.

        Returns:
            BinarySampleSet in SPIN vartype.
        """
        sample = {i: s for i, s in enumerate(spins)}
        energy = self._calculate_energy(spins)
        return BinarySampleSet(
            samples=[sample],
            num_occurrences=[1],
            energy=[energy],
            vartype=VarType.SPIN,
        )

    def decode_to_binary(self, spins: list[int]) -> BinarySampleSet:
        """Decode rounded spins directly to binary values.

        Conversion: binary = (1 - spin) // 2
            spin +1 -> binary 0
            spin -1 -> binary 1

        Args:
            spins: List of spin values (+1 or -1).

        Returns:
            BinarySampleSet in BINARY vartype.
        """
        sample = {i: (1 - s) // 2 for i, s in enumerate(spins)}
        energy = self._calculate_energy(spins)
        return BinarySampleSet(
            samples=[sample],
            num_occurrences=[1],
            energy=[energy],
            vartype=VarType.BINARY,
        )

    def _calculate_energy(self, spins: list[int]) -> float:
        """Calculate energy for a given spin assignment.

        Args:
            spins: List of spin values (+1 or -1).

        Returns:
            Energy value for the spin configuration.
        """
        energy = self.spin_model.constant
        for idx, coeff in self.spin_model.linear.items():
            energy += coeff * spins[idx]
        for (i, j), coeff in self.spin_model.quad.items():
            energy += coeff * spins[i] * spins[j]
        for inds, coeff in self.spin_model.higher.items():
            prod = 1
            for i in inds:
                prod *= spins[i]
            energy += coeff * prod
        return energy
