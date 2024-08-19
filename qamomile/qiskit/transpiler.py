"""
Qamomile to Qiskit Transpiler Module

This module provides functionality to convert Qamomile quantum circuits, operators,
and measurement results to their Qiskit equivalents. It includes a QiskitTranspiler
class that implements the QuantumSDKTranspiler interface for Qiskit compatibility.

Key features:
- Convert Qamomile quantum circuits to Qiskit quantum circuits
- Convert Qamomile Hamiltonians to Qiskit SparsePauliOp
- Convert Qiskit measurement results to Qamomile BitsSampleSet

Usage:
    from qamomile.qiskit.transpiler import QiskitTranspiler

    transpiler = QiskitTranspiler()
    qiskit_circuit = transpiler.transpile_circuit(qamomile_circuit)
    qiskit_hamiltonian = transpiler.transpile_hamiltonian(qamomile_hamiltonian)
    qamomile_results = transpiler.convert_result(qiskit_results)

Note: This module requires both Qamomile and Qiskit to be installed.
"""

import numpy as np
from typing import Dict

import qiskit
import qiskit.circuit
import qiskit.primitives as qk_primitives
import qiskit.quantum_info as qk_ope

import qamomile.core.bitssample as qm_bs
import qamomile.core.circuit as qm_c
import qamomile.core.operator as qm_o
from qamomile.core.transpiler import QuantumSDKTranspiler

from .parameter_converter import convert_parameter
from .exceptions import QamomileQiskitTranspileError


class QiskitTranspiler(QuantumSDKTranspiler[qk_primitives.BitArray]):
    """
    Transpiler class for converting between Qamomile and Qiskit quantum objects.

    This class implements the QuantumSDKTranspiler interface for Qiskit compatibility,
    providing methods to convert circuits, Hamiltonians, and measurement results.
    """

    def transpile_circuit(
        self, qamomile_circuit: qm_c.QuantumCircuit
    ) -> qiskit.QuantumCircuit:
        """
        Convert a Qamomile quantum circuit to a Qiskit quantum circuit.

        Args:
            qamomile_circuit (qm_c.QuantumCircuit): The Qamomile quantum circuit to convert.

        Returns:
            qiskit.QuantumCircuit: The converted Qiskit quantum circuit.

        Raises:
            QamomileQiskitConverterError: If there's an error during conversion.
        """
        try:
            parameters = qamomile_circuit.get_parameters()
            param_mapping = {
                param: qiskit.circuit.Parameter(param.name) for param in parameters
            }
            return self._circuit_convert(qamomile_circuit, param_mapping)
        except Exception as e:
            raise QamomileQiskitTranspileError(f"Error converting circuit: {str(e)}")

    def _circuit_convert(
        self,
        qamomile_circuit: qm_c.QuantumCircuit,
        param_mapping: Dict[qm_c.Parameter, qiskit.circuit.Parameter],
    ) -> qiskit.QuantumCircuit:
        """
        Internal method to recursively convert Qamomile circuits to Qiskit circuits.

        Args:
            qamomile_circuit (qm_c.QuantumCircuit): The Qamomile circuit to convert.
            param_mapping (Dict[qm_c.Parameter, qiskit.circuit.Parameter]): Mapping of parameters.

        Returns:
            qiskit.QuantumCircuit: The converted Qiskit circuit.

        Raises:
            QamomileQiskitConverterError: If an unsupported gate type is encountered.
        """
        qiskit_circuit = qiskit.QuantumCircuit(
            qamomile_circuit.num_qubits, qamomile_circuit.num_clbits
        )
        for gate in qamomile_circuit.gates:
            if isinstance(gate, qm_c.SingleQubitGate):
                qiskit_circuit = _single_qubit_gate(gate, qiskit_circuit)
            elif isinstance(gate, qm_c.ParametricSingleQubitGate):
                qiskit_circuit = _parametric_single_qubit_gate(
                    gate, qiskit_circuit, parameters=param_mapping
                )
            elif isinstance(gate, qm_c.TwoQubitGate):
                qiskit_circuit = _two_qubit_gate(gate, qiskit_circuit)
            elif isinstance(gate, qm_c.ParametricTwoQubitGate):
                qiskit_circuit = _parametric_two_qubit_gate(
                    gate, qiskit_circuit, parameters=param_mapping
                )
            elif isinstance(gate, qm_c.ThreeQubitGate):
                qiskit_circuit = _three_qubit_gate(gate, qiskit_circuit)
            elif isinstance(gate, qm_c.Operator):
                qiskit_subcircuit = self._circuit_convert(gate.circuit, param_mapping)
                qubit_indices = list(range(gate.circuit.num_qubits))
                qiskit_circuit.append(
                    qiskit_subcircuit.to_gate(label=gate.label), qubit_indices
                )
            elif isinstance(gate, qm_c.MeasurementGate):
                qiskit_circuit.measure(gate.qubit, gate.cbit)
            else:
                raise QamomileQiskitTranspileError(
                    f"Unsupported gate type: {type(gate)}"
                )
        return qiskit_circuit

    def convert_result(self, result: qk_primitives.BitArray) -> qm_bs.BitsSampleSet:
        """
        Convert Qiskit measurement results to Qamomile BitsSampleSet.

        Args:
            result (qk_primitives.BitArray): Qiskit measurement results.

        Returns:
            qm.BitsSampleSet: Converted Qamomile BitsSampleSet.
        """
        int_counts = result.get_int_counts()
        return qm_bs.BitsSampleSet.from_int_counts(int_counts, result.num_bits)

    def transpile_hamiltonian(self, operator: qm_o.Hamiltonian) -> qk_ope.SparsePauliOp:
        """
        Convert a Qamomile Hamiltonian to a Qiskit SparsePauliOp.

        Args:
            operator (qm_o.Hamiltonian): The Qamomile Hamiltonian to convert.

        Returns:
            qk_ope.SparsePauliOp: The converted Qiskit SparsePauliOp.

        Raises:
            NotImplementedError: If an unsupported Pauli operator is encountered.
        """
        num_qubits = operator.num_qubits
        qk_pauli_list = []
        coeff_list = []
        for term, coeff in operator.terms.items():
            qk_pauli_z = np.zeros(num_qubits, dtype=bool)
            qk_pauli_x = np.zeros(num_qubits, dtype=bool)
            for pauli in term:
                if pauli.pauli == qm_o.Pauli.X:
                    qk_pauli_x[pauli.index] = not qk_pauli_x[pauli.index]
                elif pauli.pauli == qm_o.Pauli.Y:
                    qk_pauli_x[pauli.index] = not qk_pauli_x[pauli.index]
                    qk_pauli_z[pauli.index] = not qk_pauli_z[pauli.index]
                elif pauli.pauli == qm_o.Pauli.Z:
                    qk_pauli_z[pauli.index] = not qk_pauli_z[pauli.index]
                else:
                    raise NotImplementedError("Only Pauli X, Y, and Z are supported")
            qk_pauli = qk_ope.Pauli((qk_pauli_z, qk_pauli_x))
            qk_pauli_list.append(qk_pauli)
            coeff_list.append(coeff)
        return qk_ope.SparsePauliOp(qk_pauli_list, coeffs=np.array(coeff_list))


def _single_qubit_gate(
    gate: qm_c.SingleQubitGate, qk_circuit: qiskit.QuantumCircuit
) -> qiskit.QuantumCircuit:
    """Apply a single qubit gate to the Qiskit circuit."""
    gate_map = {
        qm_c.SingleQubitGateType.H: qk_circuit.h,
        qm_c.SingleQubitGateType.X: qk_circuit.x,
        qm_c.SingleQubitGateType.Y: qk_circuit.y,
        qm_c.SingleQubitGateType.Z: qk_circuit.z,
        qm_c.SingleQubitGateType.S: qk_circuit.s,
        qm_c.SingleQubitGateType.T: qk_circuit.t,
    }
    gate_map[gate.gate](gate.qubit)
    return qk_circuit


def _parametric_single_qubit_gate(
    gate: qm_c.ParametricSingleQubitGate,
    qk_circuit: qiskit.QuantumCircuit,
    parameters: Dict[qm_c.Parameter, qiskit.circuit.Parameter],
) -> qiskit.QuantumCircuit:
    """Apply a parametric single qubit gate to the Qiskit circuit."""
    angle = convert_parameter(gate.parameter, parameters=parameters)
    gate_map = {
        qm_c.ParametricSingleQubitGateType.RX: qk_circuit.rx,
        qm_c.ParametricSingleQubitGateType.RY: qk_circuit.ry,
        qm_c.ParametricSingleQubitGateType.RZ: qk_circuit.rz,
    }
    gate_map[gate.gate](angle, gate.qubit)
    return qk_circuit


def _two_qubit_gate(
    gate: qm_c.TwoQubitGate, qk_circuit: qiskit.QuantumCircuit
) -> qiskit.QuantumCircuit:
    """Apply a two qubit gate to the Qiskit circuit."""
    gate_map = {
        qm_c.TwoQubitGateType.CNOT: qk_circuit.cx,
        qm_c.TwoQubitGateType.CZ: qk_circuit.cz,
    }
    gate_map[gate.gate](gate.control, gate.target)
    return qk_circuit


def _parametric_two_qubit_gate(
    gate: qm_c.ParametricTwoQubitGate,
    qk_circuit: qiskit.QuantumCircuit,
    parameters: Dict[qm_c.Parameter, qiskit.circuit.Parameter],
) -> qiskit.QuantumCircuit:
    """Apply a parametric two qubit gate to the Qiskit circuit."""
    angle = convert_parameter(gate.parameter, parameters=parameters)
    match gate.gate:
        case qm_c.ParametricTwoQubitGateType.CRX:
            qk_circuit.crx(angle, gate.control, gate.target)
        case qm_c.ParametricTwoQubitGateType.CRY:
            qk_circuit.cry(angle, gate.control, gate.target)
        case qm_c.ParametricTwoQubitGateType.CRZ:
            qk_circuit.crz(angle, gate.control, gate.target)
        case qm_c.ParametricTwoQubitGateType.RXX:
            qk_circuit.rxx(angle, gate.control, gate.target)
        case qm_c.ParametricTwoQubitGateType.RYY:
            qk_circuit.ryy(angle, gate.control, gate.target)
        case qm_c.ParametricTwoQubitGateType.RZZ:
            qk_circuit.rzz(angle, gate.control, gate.target)
        case _:
            raise QamomileQiskitTranspileError(f"Unsupported parametric two qubit gate: {gate.gate}")
    return qk_circuit


def _three_qubit_gate(
    gate: qm_c.ThreeQubitGate, qk_circuit: qiskit.QuantumCircuit
) -> qiskit.QuantumCircuit:
    """Apply a three qubit gate to the Qiskit circuit."""
    if gate.gate == qm_c.ThreeQubitGateType.CCX:
        qk_circuit.ccx(gate.control1, gate.control2, gate.target)
    return qk_circuit
