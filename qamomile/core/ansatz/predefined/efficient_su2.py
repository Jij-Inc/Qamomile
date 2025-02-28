from typing import Optional

import qamomile.core.circuit as qm_c
from qamomile.core.layer.non_parameterized_layer import EntanglementLayer
from qamomile.core.layer.parameterized_layer import RotationLayer

from ..ansatz import Ansatz


class EfficientSU2Ansatz(Ansatz):

    DEFAULT_ROTATION_TYPES = ["ry", "rz"]

    def __init__(
        self,
        num_qubits: int,
        rotation_types: Optional[list[str]]  = None,
        entanglement: str = "linear",
        skip_final_rotation_layer: bool = False,
        reps: int = 1,
    ):
        self.rotation_types = (
            rotation_types
            if rotation_types is not None
            else self.DEFAULT_ROTATION_TYPES
        )
        self.entanglement = entanglement
        self.skip_final_rotation_layer = skip_final_rotation_layer
        self.reps = reps

        for rot_type in self.rotation_types:
            if rot_type not in RotationLayer.SUPPORTED_ROTATIONS:
                raise ValueError(
                    f"Unsupported rotation type: {rot_type}. Supported types: {RotationLayer.SUPPORTED_ROTATIONS}"
                )

        super().__init__(num_qubits, reps=reps)

    def build(self) -> qm_c.QuantumCircuit:
        circuit = qm_c.QuantumCircuit(self.num_qubits, 0, name="EfficientSU2")

        entanglement_layer = EntanglementLayer(
            self.num_qubits, self.entanglement
        ).get_circuit()

        for _ in range(self.reps):
            for rotation_type in self.rotation_types:
                rotation_layer = RotationLayer(
                    self.num_qubits, rotation_type, parameter_context=self.parameter_context
                ).get_circuit()
                circuit.append(rotation_layer)

            circuit.append(entanglement_layer)

        if not self.skip_final_rotation_layer:
            for rotation_type in self.rotation_types:
                rotation_layer = RotationLayer(
                    self.num_qubits, rotation_type, parameter_context=self.parameter_context
                ).get_circuit()
                circuit.append(rotation_layer)

        return circuit

def create_efficient_su2(
    num_qubits: int,
    rotation_types: Optional[list[str]] = None,
    entanglement: str = "linear",
    reps: int = 1,
    skip_final_rotation_layer: bool = False,
) -> EfficientSU2Ansatz:
    """
    EfficientSU2Ansatzを簡単に作成するためのファクトリー関数。

    Args:
        num_qubits: 量子ビット数
        rotation_types: 回転演算子のリスト
        entanglement: エンタングルメントの種類
        reps: 繰り返し回数

    Returns:
        EfficientSU2Ansatz: 作成されたAnsatzインスタンス
    """
    return EfficientSU2Ansatz(
        num_qubits=num_qubits,
        rotation_types=rotation_types,
        entanglement=entanglement,
        reps=reps,
        skip_final_rotation_layer=skip_final_rotation_layer
    )

# def create_efficient_su2_circuit(
#     num_qubits: int,
#     rotation_blocks: list[str] | None = None,
#     entanglement: str = "linear",
#     skip_final_rotation_layer: bool = False,
#     reps: int = 1,
# ) -> qm_c.QuantumCircuit:
#     """Creates an Efficient SU2 variational quantum circuit.

#     This function generates a parameterized quantum circuit based on the Efficient SU2 ansatz.
#     The circuit consists of alternating layers of rotation gates and entanglement operations.

#     Args:
#         num_qubits (int): The number of qubits in the circuit.
#         rotation_blocks (list[str] | None): A list of rotation gates to apply in each rotation layer.
#             If None, defaults to ["ry", "rz"].
#         entanglement (str): The type of entanglement to apply. Options are 'linear', 'full',
#             'circular', or 'reverse_linear'.
#         skip_final_rotation_layer (bool): If True, skips the final rotation layer.
#         reps (int): The number of repetitions of the rotation-entanglement block.

#     Returns:
#         qm_c.QuantumCircuit: The constructed Efficient SU2 variational quantum circuit.

#     Raises:
#         NotImplementedError: If an unsupported rotation gate or entanglement type is specified.

#     Example:
#         >>> circuit = create_efficient_su2_circuit(3, rotation_blocks=["rx", "ry"], entanglement="full", reps=2)
#     """
#     if rotation_blocks is None:
#         rotation_blocks = ["ry", "rz"]

#     circuit = qm_c.QuantumCircuit(num_qubits, 0, name="TwoLocal")

#     def add_rotation_blocks(
#         circuit: qm_c.QuantumCircuit, num_params: int
#     ) -> tuple[qm_c.QuantumCircuit, int]:
#         for gate in rotation_blocks:
#             for i in range(num_qubits):
#                 param = qm_c.Parameter(r"\theta_{" + f"{num_params}" + r"}")
#                 num_params += 1
#                 if gate == "ry":
#                     circuit.ry(param, i)
#                 elif gate == "rz":
#                     circuit.rz(param, i)
#                 elif gate == "rx":
#                     circuit.rx(param, i)
#                 else:
#                     raise NotImplementedError(f"Gate {gate} not implemented")
#         return circuit, num_params

#     def add_entanglement_blocks(circuit: qm_c.QuantumCircuit) -> qm_c.QuantumCircuit:
#         if entanglement == "linear":
#             for i in range(num_qubits - 1):
#                 circuit.cx(i, i + 1)

#         elif entanglement == "full":
#             for i in range(num_qubits):
#                 for j in range(i + 1, num_qubits):
#                     circuit.cx(i, j)

#         elif entanglement == "circular":
#             for i in range(num_qubits):
#                 circuit.cx(i, (i + 1) % num_qubits)

#         elif entanglement == "reverse_linear":
#             for i in range(num_qubits - 1, 0, -1):
#                 circuit.cx(i, i - 1)
#         else:
#             raise NotImplementedError(
#                 f"Entanglement type {entanglement} not implemented"
#             )
#         return circuit

#     num_params = 0
#     for _ in range(reps):
#         circuit, num_params = add_rotation_blocks(circuit, num_params)
#         circuit = add_entanglement_blocks(circuit)

#     if not skip_final_rotation_layer:
#         circuit, num_params = add_rotation_blocks(circuit, num_params)

#     return circuit
