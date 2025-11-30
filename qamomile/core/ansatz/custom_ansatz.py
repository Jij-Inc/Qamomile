from typing import Optional

import qamomile.core.circuit as qm_c
from qamomile.core.layer.layer import Layer
from qamomile.core.layer.parameterized_layer import ParameterizedLayer

from .ansatz import Ansatz


class CustomAnsatz(Ansatz):
    def __init__(
        self,
        num_qubits: int,
        layers: list[Layer],
        reps: int = 1,
        use_common_parameter_context: bool = True,
    ):
        self.layers = layers
        self.use_common_parameter_context = use_common_parameter_context
        super().__init__(num_qubits, reps)

    def build(self) -> qm_c.QuantumCircuit:
        circuit = qm_c.QuantumCircuit(self.num_qubits, 0, name="CustomAnsatz")

        for _ in range(self.reps):
            for layer in self.layers:
                if self.use_common_parameter_context and isinstance(
                    layer, ParameterizedLayer
                ):
                    layer.set_parameter_context(self.parameter_context, regenerate=True)
                layer_circuit = layer.get_circuit()
                circuit.append(layer_circuit)

        return circuit
