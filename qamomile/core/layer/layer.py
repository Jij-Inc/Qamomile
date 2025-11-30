from abc import ABC, abstractmethod

import qamomile.core.circuit as qm_c


class Layer(ABC):
    """
    Abstract base class representing a layer in a quantum circuit.

    Methods:
        get_circuit():
            Abstract method to return the quantum circuit representing the layer.
    """

    @abstractmethod
    def get_circuit(self) -> qm_c.QuantumCircuit:
        """
        Returns the quantum circuit representing the layer.
        """
        pass
