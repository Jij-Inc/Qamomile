"""
Qamomile to QuriParts Transpiler Module

This module provides functionality to convert Qamomile quantum circuits, operators,
and measurement results to their QuriParts equivalents. It includes a QuriPartsTranspiler
class that implements the QuantumSDKTranspiler interface for QuriParts compatibility.

Key features:
- Convert Qamomile quantum circuits to QuriParts quantum circuits
- Convert Qamomile Hamiltonians to QuriParts Operators
- Convert QuriParts measurement results to Qamomile BitsSampleSet

Usage:
    from qamomile.quriparts.transpiler import QuriPartsTranspiler

    transpiler = QuriPartsTranspiler()
    qp_circuit = transpiler.transpile_circuit(qamomile_circuit)
    qp_operator = transpiler.transpile_hamiltonian(qamomile_hamiltonian)
    qamomile_results = transpiler.convert_result(quriparts_results)

Note: This module requires both Qamomile and QuriParts to be installed.
"""

import quri_parts.core.operator as qp_o
import quri_parts.circuit as qp_c
import collections
import qamomile.core.bitssample as qm_bs
import qamomile.core.circuit as qm_c
import qamomile.core.operator as qm_o
from qamomile.core.transpiler import QuantumSDKTranspiler

from .parameter_converter import convert_parameter
from .exceptions import QamomileQuriPartsTranspileError


class QuriPartsTranspiler(
    QuantumSDKTranspiler[tuple[collections.Counter[int], int]]
):
    """
    Transpiler class for converting between Qamomile and QuriParts quantum objects.

    This class implements the QuantumSDKTranspiler interface for QuriParts compatibility,
    providing methods to convert circuits, Hamiltonians, and measurement results.
    """

    def transpile_circuit(
        self, qamomile_circuit: qm_c.QuantumCircuit
    ) -> qp_c.LinearMappedUnboundParametricQuantumCircuit:
        """
        Convert a Qamomile quantum circuit to a QuriParts quantum circuit.

        Args:
            qamomile_circuit (qm_c.QuantumCircuit): The Qamomile quantum circuit to convert.

        Returns:
            qp_c.LinearMappedUnboundParametricQuantumCircuit: The converted QuriParts quantum circuit.

        Raises:
            QamomileQuriPartsTranspileError: If there's an error during conversion.
        """
        try:
            parameters = qamomile_circuit.get_parameters()
            qp_circuit = qp_c.LinearMappedUnboundParametricQuantumCircuit(
                qamomile_circuit.num_qubits, qamomile_circuit.num_clbits
            )
            param_mapping = {
                param: qp_circuit.add_parameter(param.name)
                for param in parameters
            }
            return self._circuit_convert(qamomile_circuit, qp_circuit, param_mapping)
        except Exception as e:
            raise QamomileQuriPartsTranspileError(f"Error converting circuit: {str(e)}")

    def _circuit_convert(
        self,
        qamomile_circuit: qm_c.QuantumCircuit,
        qp_circuit: qp_c.LinearMappedUnboundParametricQuantumCircuit,
        param_mapping: dict[qm_c.Parameter, qp_c.Parameter],
    ) -> qp_c.LinearMappedUnboundParametricQuantumCircuit:
        """
        Internal method to recursively convert Qamomile circuits to QuriParts circuits.

        Args:
            qamomile_circuit (qm_c.QuantumCircuit): The Qamomile circuit to convert.
            qp_circuit (qp_c.LinearMappedUnboundParametricQuantumCircuit): The QuriParts circuit.
            param_mapping (Dict[qm_c.Parameter, qp_c.Parameter]): Mapping of parameters.

        Returns:
            qp_c.LinearMappedUnboundParametricQuantumCircuit: The converted QuriParts circuit.

        Raises:
            QamomileQuriPartsTranspileError: If an unsupported gate type is encountered.
        """

        for gate in qamomile_circuit.gates:
            if isinstance(gate, qm_c.SingleQubitGate):
                qp_circuit = _single_qubit_gate(gate, qp_circuit)
            elif isinstance(gate, qm_c.ParametricSingleQubitGate):
                qp_circuit = _parametric_single_qubit_gate(
                    gate, qp_circuit, parameters=param_mapping
                )
            elif isinstance(gate, qm_c.TwoQubitGate):
                qp_circuit = _two_qubit_gate(gate, qp_circuit)
            elif isinstance(gate, qm_c.ParametricTwoQubitGate):
                qp_circuit = _parametric_two_qubit_gate(
                    gate, qp_circuit, parameters=param_mapping
                )
            elif isinstance(gate, qm_c.ThreeQubitGate):
                qp_circuit = _three_qubit_gate(gate, qp_circuit)
            elif isinstance(gate, qm_c.Operator):
                qp_circuit = self._circuit_convert(
                                    gate.circuit,
                                    qp_circuit,
                                    param_mapping
                                )
            elif isinstance(gate, qm_c.MeasurementGate):
                qp_circuit.measure(gate.qubit, gate.cbit)
            else:
                raise QamomileQuriPartsTranspileError(
                    f"Unsupported gate type: {type(gate)}"
                )
        return qp_circuit

    def convert_result(
        self,
        result: tuple[collections.Counter[int], int]
    ) -> qm_bs.BitsSampleSet:
        """
        Convert QuriParts measurement results to Qamomile BitsSampleSet.

        Args:
            result (tuple[collections.Counter[int], int]): QuriParts measurement results.

        Returns:
            qm_bs.BitsSampleSet: Converted Qamomile BitsSampleSet.
        """

        counter, num_bits = result
        int_counts = dict(counter)
        return qm_bs.BitsSampleSet.from_int_counts(int_counts, num_bits)

    def transpile_hamiltonian(
        self,
        operator: qm_o.Hamiltonian
    ) -> qp_o.Operator:
        """
        Convert a Qamomile Hamiltonian to a QuriParts Operator.

        Args:
            operator (qm_o.Hamiltonian): The Qamomile Hamiltonian to convert.

        Returns:
            qp_o.Operator: The converted QuriParts Operator.

        Raises:
            NotImplementedError: If an unsupported Pauli operator is encountered.
        """
        qp_pauli_terms = {}
        for term, coeff in operator.terms.items():
            pauli_list = []
            for pauli in term:
                match pauli.pauli:
                    case qm_o.Pauli.X:
                        pauli_list.append((pauli.index, qp_o.SinglePauli.X))
                    case qm_o.Pauli.Y:
                        pauli_list.append((pauli.index, qp_o.SinglePauli.Y))
                    case qm_o.Pauli.Z:
                        pauli_list.append((pauli.index, qp_o.SinglePauli.Z))
                    case _:
                        raise NotImplementedError("Only Pauli X, Y, and Z are supported")

            qp_pauli_terms[qp_o.pauli_label(pauli_list)] = coeff
        return qp_o.Operator(qp_pauli_terms)


def _single_qubit_gate(
    gate: qm_c.SingleQubitGate,
    qp_circuit: qp_c.LinearMappedUnboundParametricQuantumCircuit
) -> qp_c.LinearMappedUnboundParametricQuantumCircuit:
    """Apply a single qubit gate to the quri-parts circuit."""
    gate_map = {
        qm_c.SingleQubitGateType.H: qp_circuit.add_H_gate,
        qm_c.SingleQubitGateType.X: qp_circuit.add_X_gate,
        qm_c.SingleQubitGateType.Y: qp_circuit.add_Y_gate,
        qm_c.SingleQubitGateType.Z: qp_circuit.add_Z_gate,
        qm_c.SingleQubitGateType.S: qp_circuit.add_S_gate,
        qm_c.SingleQubitGateType.T: qp_circuit.add_T_gate,
    }
    gate_map[gate.gate](gate.qubit)
    return qp_circuit


def _parametric_single_qubit_gate(
    gate: qm_c.ParametricSingleQubitGate,
    qp_circuit: qp_c.LinearMappedUnboundParametricQuantumCircuit,
    parameters: dict[qm_c.Parameter, qp_c.Parameter],
) -> qp_c.LinearMappedUnboundParametricQuantumCircuit:
    """Apply a parametric single qubit gate to the quri-parts circuit."""
    angle = convert_parameter(gate.parameter, parameters=parameters)
    gate_map = {
        qm_c.ParametricSingleQubitGateType.RX: qp_circuit.add_ParametricRX_gate,
        qm_c.ParametricSingleQubitGateType.RY: qp_circuit.add_ParametricRY_gate,
        qm_c.ParametricSingleQubitGateType.RZ: qp_circuit.add_ParametricRZ_gate,
    }
    gate_map[gate.gate](gate.qubit, angle)
    return qp_circuit


def _two_qubit_gate(
    gate: qm_c.TwoQubitGate,
    qp_circuit: qp_c.LinearMappedUnboundParametricQuantumCircuit
) -> qp_c.LinearMappedUnboundParametricQuantumCircuit:
    """Apply a two qubit gate to the quri-parts circuit."""
    gate_map = {
        qm_c.TwoQubitGateType.CNOT: qp_circuit.add_CNOT_gate,
        qm_c.TwoQubitGateType.CZ: qp_circuit.add_CZ_gate,
    }
    gate_map[gate.gate](gate.control, gate.target)
    return qp_circuit


def _parametric_two_qubit_gate(
    gate: qm_c.ParametricTwoQubitGate,
    qp_circuit: qp_c.LinearMappedUnboundParametricQuantumCircuit,
    parameters: dict[qm_c.Parameter, qp_c.Parameter],
) -> qp_c.LinearMappedUnboundParametricQuantumCircuit:
    """Apply a parametric two qubit gate to the Qiskit circuit."""
    angle = convert_parameter(gate.parameter, parameters=parameters)
    match gate.gate:
        # case qm_c.ParametricTwoQubitGateType.CRX:
        #     qk_circuit.crx(angle, gate.control, gate.target)
        # case qm_c.ParametricTwoQubitGateType.CRY:
        #     qk_circuit.cry(angle, gate.control, gate.target)
        # case qm_c.ParametricTwoQubitGateType.CRZ:
        #     qk_circuit.crz(angle, gate.control, gate.target)
        case qm_c.ParametricTwoQubitGateType.RXX:
            qp_circuit.add_ParametricPauliRotation_gate(
                [gate.control, gate.target],
                pauli_ids=[1, 1],
                angle=angle
            )
        case qm_c.ParametricTwoQubitGateType.RYY:
            qp_circuit.add_ParametricPauliRotation_gate(
                [gate.control, gate.target],
                pauli_ids=[2, 2],
                angle=angle
            )
        case qm_c.ParametricTwoQubitGateType.RZZ:
            qp_circuit.add_ParametricPauliRotation_gate(
                [gate.control, gate.target],
                pauli_ids=[3, 3],
                angle=angle
            )
        case _:
            raise QamomileQuriPartsTranspileError(
                f"Unsupported parametric two qubit gate: {gate.gate}"
            )
    return qp_circuit


def _three_qubit_gate(
    gate: qm_c.ThreeQubitGate,
    qp_circuit: qp_c.LinearMappedUnboundParametricQuantumCircuit
) -> qp_c.LinearMappedUnboundParametricQuantumCircuit:
    """Apply a three qubit gate to the quri-parts circuit."""
    if gate.gate == qm_c.ThreeQubitGateType.CCX:
        qp_circuit.add_TOFFOLI_gate(gate.control1, gate.control2, gate.target)
    return qp_circuit
