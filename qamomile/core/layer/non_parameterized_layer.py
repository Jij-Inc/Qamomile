from abc import abstractmethod
from typing import Optional

import qamomile.core.circuit as qm_c

from .layer import Layer


class EntanglementLayer(Layer):
    """A layer that applies an entanglement operation to a quantum circuit."""

    SUPPORTED_ENTANGLE_TYPES = ["linear", "full", "circular", "reverse_linear"]

    def __init__(self, num_qubits, entangle_type="linear"):
        if entangle_type not in self.SUPPORTED_ENTANGLE_TYPES:
            raise ValueError(
                f"Unsupported entanglement type: {entangle_type}. Supported types: {self.SUPPORTED_ENTANGLE_TYPES}"
            )

        self.num_qubits = num_qubits
        self.entangle_type = entangle_type

    def get_circuit(self) -> qm_c.QuantumCircuit:
        """Apply the entanglement layer to the given quantum circuit.

        Args:
            circuit (QuantumCircuit): The quantum circuit to which the layer will be applied.
        """
        circuit = qm_c.QuantumCircuit(self.num_qubits, 0, name="EntanglementLayer")

        if self.entangle_type == "linear":
            for i in range(self.num_qubits - 1):
                circuit.cnot(i, i + 1)
        elif self.entangle_type == "full":
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    circuit.cnot(i, j)
        elif self.entangle_type == "circular":
            for i in range(self.num_qubits):
                circuit.cnot(i, (i + 1) % self.num_qubits)
        elif self.entangle_type == "reverse_linear":
            for i in range(self.num_qubits - 1, 0, -1):
                circuit.cnot(i, i - 1)

        return circuit


class SuperpositionLayer(Layer):
    """A layer that applies a superposition operation to a quantum circuit."""

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def get_circuit(self) -> qm_c.QuantumCircuit:
        """Apply the superposition layer to the given quantum circuit.

        Args:
            circuit (QuantumCircuit): The quantum circuit to which the layer will be applied.
        """
        circuit = qm_c.QuantumCircuit(self.num_qubits, 0, name="SuperpositionLayer")

        for i in range(self.num_qubits):
            circuit.h(i)

        return circuit
