"""
Qamomile to Pennylane Transpiler Module

This module provides functionality to convert Qamomile Hamiltonians to their Pennylane equivalents. It includes a `PennylaneTranspiler` class that implements the `QuantumSDKTranspiler` interface for Pennylane compatibility.

Key Features:
- Convert Qamomile Hamiltonians to Pennylane Hamiltonians.

Usage Example:
    ```python
    from qamomile.Pennylane.transpiler import PennylaneTranspiler

    transpiler = PennylaneTranspiler()
    pennylane_operator = transpiler.transpile_hamiltonian(qamomile_hamiltonian)
    ```

Requirements:
- Qamomile
- Pennylane
"""

import qamomile.core.operator
import qamomile.core.circuit
import qamomile.core.bitssample as qm_bs
from qamomile.core.transpiler import QuantumSDKTranspiler
import pennylane as qml
import collections
import numpy as np
from typing import Dict, List, Any
from qamomile.core.transpiler import QuantumSDKTranspiler


class PennylaneTranspiler(QuantumSDKTranspiler[tuple[collections.Counter[int], int]]):
    """
    A transpiler class for converting Qamomile Hamiltonians to Pennylane-compatible Hamiltonians.
    """

    def transpile_hamiltonian(
        self, operator: qamomile.core.operator.Hamiltonian
    ) -> qml.Hamiltonian:
        """
        Converts a Qamomile Hamiltonian to a Pennylane Hamiltonian.

        Args:
            operator (qamomile.core.operator.Hamiltonian): The Qamomile Hamiltonian to be converted.

        Returns:
            qml.Hamiltonian: The corresponding Pennylane Hamiltonian.
        """
        n = operator.num_qubits
        coeffs = []
        ops = []

        for term, coeff in operator.terms.items():
            op_list = [qml.Identity(i) for i in range(n)]
            for pauli in term:
                if pauli.pauli == qamomile.core.operator.Pauli.X:
                    op_list[pauli.index] = qml.PauliX(pauli.index)
                elif pauli.pauli == qamomile.core.operator.Pauli.Y:
                    op_list[pauli.index] = qml.PauliY(pauli.index)
                elif pauli.pauli == qamomile.core.operator.Pauli.Z:
                    op_list[pauli.index] = qml.PauliZ(pauli.index)
                else:
                    raise NotImplementedError(
                        f"Unsupported Pauli operator: {pauli.pauli}"
                    )
            ops.append(qml.operation.Tensor(*op_list))
            coeffs.append(coeff)

        return qml.Hamiltonian(np.array(coeffs), ops)

    def transpile_circuit(
        self, qamomile_circuit: qamomile.core.circuit.QuantumCircuit
    ) -> Any:
        """
        Convert a Qamomile quantum circuit to a PennyLane QNode.

        Args:
            qamomile_circuit (qamomile.core.circuit.QuantumCircuit): The Qamomile quantum circuit to convert.

        Returns:
            Callable: A PennyLane QNode.
        """
        try:
            dev = qml.device("default.qubit", wires=qamomile_circuit.num_qubits)

            @qml.qnode(dev)
            def circuit_fn(*args, **kwargs):
                # Apply gates using kwargs for parameter passing
                self._apply_gates(
                    qamomile_circuit,
                    param_mapping=self._create_param_mapping(qamomile_circuit),
                    params=kwargs,
                )
                return [
                    qml.expval(qml.PauliZ(i))
                    for i in range(qamomile_circuit.num_qubits)
                ]

            return qml.QNode(circuit_fn, dev)

        except Exception as e:
            raise ValueError(f"Error converting circuit: {str(e)}")

    def _create_param_mapping(
        self, qamomile_circuit: qamomile.core.circuit.QuantumCircuit
    ) -> Dict[str, qamomile.core.circuit.Parameter]:
        parameters = qamomile_circuit.get_parameters()

        # Convert list of parameters to a dictionary with parameter names as keys
        param_mapping = {}
        for param in parameters:
            try:
                # Using the name or a unique identifier of the parameter as the key
                param_name = getattr(param, "name", None)
                if param_name is None:
                    raise AttributeError("Parameter has no 'name' attribute")
                param_mapping[param_name] = param
            except AttributeError:
                raise ValueError(f"Cannot determine name of parameter: {param}")
        return param_mapping

    def _apply_gates(
        self,
        qamomile_circuit: qamomile.core.circuit.QuantumCircuit,
        param_mapping: Dict,
        params: Dict[str, Any],
    ):
        for gate in qamomile_circuit.gates:
            if isinstance(gate, qamomile.core.circuit.SingleQubitGate):
                self._apply_single_qubit_gate(gate)
            elif isinstance(gate, qamomile.core.circuit.TwoQubitGate):
                self._apply_two_qubit_gate(gate)

            elif isinstance(gate, qamomile.core.circuit.ParametricSingleQubitGate):
                self._apply_parametric_single_qubit_gate(gate, params)
            elif isinstance(gate, qamomile.core.circuit.ParametricTwoQubitGate):
                self._apply_parametric_two_qubit_gate(gate, params)

            elif isinstance(gate, qamomile.core.circuit.MeasurementGate):
                pass

    def _apply_single_qubit_gate(self, gate: qamomile.core.circuit.SingleQubitGate):
        gate_map = {
            qamomile.core.circuit.SingleQubitGateType.H: qml.Hadamard,
            qamomile.core.circuit.SingleQubitGateType.X: qml.PauliX,
            qamomile.core.circuit.SingleQubitGateType.Y: qml.PauliY,
            qamomile.core.circuit.SingleQubitGateType.Z: qml.PauliZ,
            qamomile.core.circuit.SingleQubitGateType.S: qml.S,
            qamomile.core.circuit.SingleQubitGateType.T: qml.T,
        }
        gate_map[gate.gate](wires=gate.qubit)

    def _apply_two_qubit_gate(self, gate: qamomile.core.circuit.TwoQubitGate):
        gate_map = {
            qamomile.core.circuit.TwoQubitGateType.CNOT: qml.CNOT,
            qamomile.core.circuit.TwoQubitGateType.CZ: qml.CZ,
        }
        gate_map[gate.gate](wires=[gate.control, gate.target])

    def _apply_parametric_single_qubit_gate(
        self,
        gate: qamomile.core.circuit.ParametricSingleQubitGate,
        params: Dict[str, Any],
    ):
        param_name = gate.parameter.name
        if param_name in params:
            angle = params[param_name]
        else:
            raise ValueError(
                f"Parameter '{param_name}' not found in the provided params."
            )

        gate_map = {
            qamomile.core.circuit.ParametricSingleQubitGateType.RX: qml.RX,
            qamomile.core.circuit.ParametricSingleQubitGateType.RY: qml.RY,
            qamomile.core.circuit.ParametricSingleQubitGateType.RZ: qml.RZ,
        }
        gate_map[gate.gate](angle, wires=gate.qubit)

    def _apply_parametric_two_qubit_gate(
        self,
        gate: qamomile.core.circuit.ParametricTwoQubitGate,
        params: Dict[str, Any],
    ):
        param_name = gate.parameter.name
        if param_name in params:
            angle = params[param_name]
        else:
            raise ValueError(
                f"Parameter '{param_name}' not found in the provided params."
            )

        # Map the gate to PennyLane gate
        gate_map = {
            qamomile.core.circuit.ParametricTwoQubitGateType.CRX: qml.CRX,
            qamomile.core.circuit.ParametricTwoQubitGateType.CRY: qml.CRY,
            qamomile.core.circuit.ParametricTwoQubitGateType.CRZ: qml.CRZ,
        }

        # Apply the gate with the provided parameter value
        gate_map[gate.gate](angle, wires=[gate.control, gate.target])

    def convert_result(self, result: Any) -> Any:
        """
        Dummy implementation to satisfy the abstract method requirement.
        """
        raise NotImplementedError("convert_result is not implemented yet.")
