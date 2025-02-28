from abc import ABC, abstractmethod
from typing import Optional

import qamomile.core.circuit as qm_c
from qamomile.core.layer.parameter_context import ParameterContext


class Ansatz(ABC):
    """基底Ansatzクラス。全てのAnsatzはこれを継承する"""

    def __init__(
        self,
        num_qubits,
        reps=1,
    ):
        self.num_qubits = num_qubits
        self.reps = reps

        self.parameter_context = ParameterContext()

        self.circuit = self.build()

    @abstractmethod
    def build(self) -> qm_c.QuantumCircuit:
        pass

    def get_circuit(self) -> qm_c.QuantumCircuit:
        return self.circuit
