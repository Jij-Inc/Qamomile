"""
Qamomile to CUDA-Q Transpiler Module

This module provides functionality to convert Qamomile Hamiltonians to their CUDA-Q equivalents. It includes a `CudaqTranspiler` class that implements the `QuantumSDKTranspiler` interface for CUDA-Q compatibility.

Key Features:
- Convert Qamomile Hamiltonians to CUDA-Q Hamiltonians.

Usage Example:
    ```python
    from qamomile.cudaq.transpiler import CudaqTranspiler

    transpiler = CudaqTranspiler()
    cudaq_operator = transpiler.transpile_hamiltonian(qamomile_hamiltonian)
    ```
Requirements:
- Qamomile
- cudaq
"""

import collections

import cudaq
import numpy as np

import qamomile
import qamomile.core
import qamomile.core.bitssample as qm_bs
from qamomile.core.transpiler import QuantumSDKTranspiler
from .exceptions import QamomileCudaqTranspileError
from .parameter_converter import convert_parameter


class CudaqTranspiler(QuantumSDKTranspiler[tuple[collections.Counter[int], int]]):
    """
    A transpiler class for converting Qamomile Hamiltonians to CUDA-Q-compatible Hamiltonians.
    """

    def transpile_hamiltonian(
        self, operator: qamomile.core.operator.Hamiltonian
    ) -> cudaq.SpinOperator:
        """
        Converts a Qamomile Hamiltonian to a CUDA-Q Hamiltonian.

        Args:
            operator (qamomile.core.operator.Hamiltonian): The Qamomile Hamiltonian to be converted.

        Returns:
            cudaq.SpinOperator: The corresponding CUDA-Q Hamiltonian.
        """
        # Iterate over the terms in the given Qamomile Hamiltonian.
        hamltonian = 0
        for term, coeff in operator.terms.items():

            # Create each term for the CUDA-Q hamiltonian.
            hamltonian_term = coeff
            for pauli in term:
                match pauli.pauli:
                    case qamomile.core.operator.Pauli.X:
                        hamltonian_term *= cudaq.spin.x(pauli.index)
                    case qamomile.core.operator.Pauli.Y:
                        hamltonian_term *= cudaq.spin.y(pauli.index)
                    case qamomile.core.operator.Pauli.Z:
                        hamltonian_term *= cudaq.spin.z(pauli.index)
                    case _:
                        raise NotImplementedError(
                            "Only Pauli X, Y, and Z are supported"
                        )

            # Add the term to the hamiltonian.
            hamltonian += hamltonian_term

        return hamltonian

    def transpile_circuit(
        self, qamomile_circuit: qamomile.core.circuit.QuantumCircuit
    ) -> cudaq.Kernel:
        """Transpile a Qamomile quantum circuit to a CUDA-Q kernel, which corresponds to the Qamomile quantum circuit.

        Args:
            qamomile_circuit (qamomile.core.circuit.QuantumCircuit): the Qamomile quantum circuit to be transpiled

        Raises:
            QamomileCudaqTranspileError: If the transpiler encountered an error while converting the circuit.

        Returns:
            cudaq.Kernel: the CUDA-Q kernel that corresponds to the Qamomile quantum circuit
        """
        try:
            # Get the parameters from the Qamomile circuit.
            qamomile_parameters = qamomile_circuit.get_parameters()
            # Creat the CUDA-Q kernel and its parameters and qubits.
            kernel, cudaq_parameters = cudaq.make_kernel(list[float])
            qvector = kernel.qalloc(qamomile_circuit.num_qubits)
            # Create a parameter map from Qamomile parameters to CUDA-Q parameters.
            self.param_mapping = {
                param: cudaq_parameters[i]
                for i, param in enumerate(qamomile_parameters)
            }
            # Run and return the convetered circuit.
            return self._convert_circuit(
                qamomile_circuit, kernel, self.param_mapping, qvector
            )
        except Exception as e:
            raise QamomileCudaqTranspileError(f"Error converting circuit: {str(e)}")

    def _convert_circuit(
        self,
        qamomile_circuit: qamomile.core.circuit.QuantumCircuit,
        kernel: cudaq.Kernel,
        param_mapping: dict[
            qamomile.core.circuit.parameter.Parameter, cudaq.QuakeValue
        ],
        qvector: cudaq.qvector,
    ) -> cudaq.Kernel:
        """Convert a Qamomile quantum circuit to a CUDA-Q kernel.

        Args:
            qamomile_circuit (qamomile.core.circuit.QuantumCircuit): the Qamomile quantum circuit to be converted
            kernel (cudaq.Kernel): the CUDA-Q kernel to be applied the gates to
            param_mapping (dict[qamomile.core.circuit.parameter.Parameter, cudaq.QuakeValue]): the map of Qamomile parameters to CUDA-Q parameters
            qvector (cudaq.qvector): the qubits in the kernel

        Returns:
            cudaq.Kernel: the CUDA-Q kernel that corresponds to the Qamomile quantum circuit
        """
        num_qubits = qamomile_circuit.num_qubits

        # Sequentially iterate over the gates of the given Qamomile circuit to apply each gate to the CUDA-Q kernel.
        for gate in qamomile_circuit.gates:
            # Apply a gate corresponding to a qamomil.core.circuit.SingleQubitGate.
            if isinstance(gate, qamomile.core.circuit.SingleQubitGate):
                self._apply_single_qubit_gate(kernel, gate, qvector[gate.qubit])
            # Apply a ParametricSingleQubitGate.
            elif isinstance(gate, qamomile.core.circuit.ParametricSingleQubitGate):
                parameter = convert_parameter(gate.parameter, param_mapping)
                self._apply_parametric_single_qubit_gate(
                    kernel, gate, parameter, qvector[gate.qubit]
                )
            # Apply a gate corresponding to a qamomil.core.circuit.TwoQubitGate.
            elif isinstance(gate, qamomile.core.circuit.TwoQubitGate):
                self._apply_two_qubit_gate(
                    kernel, gate, qvector[gate.control], qvector[gate.target]
                )
            # Apply a gate corresponding to a qamomil.core.circuit.ParametricTwoQubitGate.
            elif isinstance(gate, qamomile.core.circuit.ParametricTwoQubitGate):
                parameter = convert_parameter(gate.parameter, param_mapping)
                self._apply_parametric_two_qubit_gate(
                    kernel,
                    gate,
                    parameter,
                    qvector[gate.control],
                    qvector[gate.target],
                )
            # Apply a gate corresponding to a qamomil.core.circuit.ThreeQubitGate.
            elif isinstance(gate, qamomile.core.circuit.ThreeQubitGate):
                self._apply_three_qubit_gate(
                    kernel,
                    gate,
                    qvector[gate.control1],
                    qvector[gate.control2],
                    qvector[gate.target],
                )
            # Apply a gate corresponding to a qamomil.core.circuit.ParametricExpGate.
            elif isinstance(gate, qamomile.core.circuit.ParametricExpGate):
                for term, coefficient in gate.hamiltonian.terms.items():
                    # Create a Pauli word for the term.
                    word_list = ["I" for _ in range(num_qubits)]
                    for index, op in enumerate(term):
                        match op.pauli:
                            case qamomile.core.operator.Pauli.X:
                                word_list[index] = "X"
                            case qamomile.core.operator.Pauli.Y:
                                word_list[index] = "Y"
                            case qamomile.core.operator.Pauli.Z:
                                word_list[index] = "Z"
                            case _:
                                raise NotImplementedError(
                                    f"Unsupported Pauli operator: {op.pauli}"
                                )
                    word = "".join(word_list)
                    # Qamomile's ParametricExpGate corresponds to e^{-i t H}.
                    #    https://github.com/Jij-Inc/Qamomile/blob/0ba3cfb6b6d1ffc8e64e482c66f3f092d0359dc8/qamomile/core/circuit/circuit.py#L140-L147
                    # Meanwhile, CUDA-Q's exp_pauli gate corresponds to e^{i *H}.
                    #    https://nvidia.github.io/cuda-quantum/latest/examples/python/operators.html#Pauli-Words-and-Exponentiating-Pauli-Words
                    # Thus, we need to negate the coefficient: (-1).
                    parameter = (
                        (-1)
                        * coefficient  # a coefficient of each Pauli term in Qamomile
                        * convert_parameter(
                            gate.parameter, param_mapping
                        )  # time t in Qamomile's exp_evolution
                    )  # Convert the parameter to CUDA-Q format.
                    self._apply_parametric_exp_gate(kernel, parameter, qvector, word)
            # Apply a gate corresponding to a qamomil.core.circuit.MeasurementGate.
            elif isinstance(gate, qamomile.core.circuit.MeasurementGate):
                self._apply_measurement(kernel, qvector[gate.qubit])
            elif isinstance(gate, qamomile.core.circuit.Operator):
                self._convert_circuit(gate.circuit, kernel, param_mapping, qvector)
            else:
                raise NotImplementedError(
                    f"Unsupported gate type: {type(gate).__name__}"
                )

        return kernel

    def _apply_single_qubit_gate(
        self,
        kernel: cudaq.Kernel,
        gate: qamomile.core.circuit.Gate,
        qubit: cudaq.qubit,
    ) -> None:
        """Apply a SingleQubitGate to the given qubit in the given kernel.

        Args:
            kernel (cudaq.Kernel): the kernel to be applied the gate to
            gate (qamomile.core.circuit.Gate): the gate to be applied
            qubit (cudaq.qubit): the qubit to apply the gate to

        Raises:
            NotImplementedError: If the gate type is not supported.
        """
        match gate.gate:
            case qamomile.core.circuit.SingleQubitGateType.H:
                kernel.h(qubit)
            case qamomile.core.circuit.SingleQubitGateType.X:
                kernel.x(qubit)
            case qamomile.core.circuit.SingleQubitGateType.Y:
                kernel.y(qubit)
            case qamomile.core.circuit.SingleQubitGateType.Z:
                kernel.z(qubit)
            case qamomile.core.circuit.SingleQubitGateType.S:
                kernel.s(qubit)
            case qamomile.core.circuit.SingleQubitGateType.T:
                kernel.t(qubit)
            case _:
                raise NotImplementedError(f"Unsupported single-qubit gate: {gate.gate}")

    def _apply_parametric_single_qubit_gate(
        self,
        kernel: cudaq.Kernel,
        gate: qamomile.core.circuit.Gate,
        parameter: cudaq.QuakeValue,
        qubit: cudaq.qubit,
    ) -> None:
        """Apply a ParametricSingleQubitGate to the given qubit in the given kernel.

        Args:
            kernel (cudaq.Kernel): the kernel to be applied the gate to
            gate (qamomile.core.circuit.Gate): the gate to be applied
            parameter (cudaq.QuakeValue): the parameter for the gate
            qubit (cudaq.qubit): the qubit to apply the gate to

        Raises:
            NotImplementedError: If the gate type is not supported.
        """
        match gate.gate:
            case qamomile.core.circuit.ParametricSingleQubitGateType.RX:
                kernel.rx(parameter, qubit)
            case qamomile.core.circuit.ParametricSingleQubitGateType.RY:
                kernel.ry(parameter, qubit)
            case qamomile.core.circuit.ParametricSingleQubitGateType.RZ:
                kernel.rz(parameter, qubit)
            case _:
                raise NotImplementedError(
                    f"Unsupported parametric single-qubit gate: {gate.gate}"
                )

    def _apply_two_qubit_gate(
        self,
        kernel: cudaq.Kernel,
        gate: qamomile.core.circuit.Gate,
        qubit_1: cudaq.qubit,
        qubit_2: cudaq.qubit,
    ) -> None:
        """Apply a TwoQubitGate to the given qubits in the given kernel.

        Args:
            kernel (cudaq.Kernel): the kernel to be applied the gate to
            gate (qamomile.core.circuit.Gate): the gate to be applied
            qubit_1 (cudaq.qubit): the first qubit to apply the gate to (for now all of them are control)
            qubit_2 (cudaq.qubit): the second qubit to apply the gate to (for now all of them are target)

        Raises:
            NotImplementedError: If the gate type is not supported.
        """
        match gate.gate:
            case qamomile.core.circuit.TwoQubitGateType.CNOT:
                kernel.cx(qubit_1, qubit_2)  # qubit_1: ctrl, qubit_2: target
            case qamomile.core.circuit.TwoQubitGateType.CZ:
                kernel.cz(qubit_1, qubit_2)  # qubit_1: ctrl, qubit_2: target
            case _:
                raise NotImplementedError(f"Unsupported two-qubit gate: {gate.gate}")

    def _apply_parametric_two_qubit_gate(
        self,
        kernel: cudaq.Kernel,
        gate: qamomile.core.circuit.Gate,
        parameter: cudaq.QuakeValue,
        qubit_1: cudaq.qubit,
        qubit_2: cudaq.qubit,
    ) -> None:
        """Apply a ParametricTwoQubitGate to the given qubits in the given kernel.

        Args:
            kernel (cudaq.Kernel): the kernel to be applied the gate to
            gate (qamomile.core.circuit.Gate): the gate to be applied
            parameter (cudaq.QuakeValue): the parameter for the gate
            qubit_1 (cudaq.qubit): the first qubit to apply the gate to
            qubit_2 (cudaq.qubit): the second qubit to apply the gate to

        Raises:
            NotImplementedError: If the gate type is not supported.
        """
        match gate.gate:
            case qamomile.core.circuit.ParametricTwoQubitGateType.CRX:
                kernel.crx(
                    parameter, qubit_1, qubit_2
                )  # qubit_1: ctrl, qubit_2: target
            case qamomile.core.circuit.ParametricTwoQubitGateType.CRY:
                kernel.cry(
                    parameter, qubit_1, qubit_2
                )  # qubit_1: ctrl, qubit_2: target
            case qamomile.core.circuit.ParametricTwoQubitGateType.CRZ:
                kernel.crz(
                    parameter, qubit_1, qubit_2
                )  # qubit_1: ctrl, qubit_2: target
            case qamomile.core.circuit.ParametricTwoQubitGateType.RXX:
                # Ref: https://github.com/Qiskit/qiskit/blob/stable/2.0/qiskit/circuit/library/standard_gates/rxx.py
                kernel.h(qubit_1)
                kernel.h(qubit_2)
                kernel.cx(qubit_1, qubit_2)
                kernel.rz(parameter, qubit_2)
                kernel.cx(qubit_1, qubit_2)
                kernel.h(qubit_1)
                kernel.h(qubit_2)
            case qamomile.core.circuit.ParametricTwoQubitGateType.RYY:
                # Ref: https://github.com/Qiskit/qiskit/blob/stable/2.0/qiskit/circuit/library/standard_gates/ryy.py
                kernel.rx(np.pi / 2, qubit_1)
                kernel.rx(np.pi / 2, qubit_2)
                kernel.cx(qubit_1, qubit_2)
                kernel.rz(parameter, qubit_2)
                kernel.cx(qubit_1, qubit_2)
                kernel.rx(-np.pi / 2, qubit_1)
                kernel.rx(-np.pi / 2, qubit_2)
            case qamomile.core.circuit.ParametricTwoQubitGateType.RZZ:
                # Ref: https://github.com/Qiskit/qiskit/blob/stable/2.0/qiskit/circuit/library/standard_gates/rzz.py
                kernel.cx(qubit_1, qubit_2)
                kernel.rz(parameter, qubit_2)
                kernel.cx(qubit_1, qubit_2)
            case _:
                raise NotImplementedError(
                    f"Unsupported parametric two-qubit gate: {gate.gate}"
                )

    def _apply_three_qubit_gate(
        self,
        kernel: cudaq.Kernel,
        gate: qamomile.core.circuit.Gate,
        qubit_1: cudaq.qubit,
        qubit_2: cudaq.qubit,
        qubit_3: cudaq.qubit,
    ) -> None:
        """Apply a ThreeQubitGate to the given qubits in the given kernel.

        Args:
            kernel (cudaq.Kernel): the kernel to be applied the gate to
            gate (qamomile.core.circuit.Gate): the gate to be applied
            qubit_1 (cudaq.qubit): the first qubit to apply the gate to (for now all of them are control)
            qubit_2 (cudaq.qubit): the second qubit to apply the gate to (for now all of them are control)
            qubit_3 (cudaq.qubit): the third qubit to apply the gate to (for now all of them are target)

        Raises:
            NotImplementedError: If the gate type is not supported.
        """
        match gate.gate:
            case qamomile.core.circuit.ThreeQubitGateType.CCX:
                kernel.cx(
                    [qubit_1, qubit_2], qubit_3
                )  # qubit_1, qubit_2: ctrl, qubit_3: target
            case _:
                raise NotImplementedError(f"Unsupported three-qubit gate: {gate.gate}")

    def _apply_parametric_exp_gate(
        self,
        kernel: cudaq.Kernel,
        coefficient: cudaq.QuakeValue | float,
        qubits: cudaq.qvector,
        pauli_word: cudaq.pauli_word,
    ) -> None:
        """Apply a ParametricExpGate to the given qubits in the given kernel.

        Args:
            kernel (cudaq.Kernel): the kernel to be applied the gate to
            coefficient (cudaq.QuakeValue | float): the coefficient for the Pauli term
            qubits cudaq.qvector: the qubits to apply the gate to
            pauli_word (cudaq.pauli_word): the Pauli word to be applied
        """
        kernel.exp_pauli(coefficient, qubits, pauli_word)

    def _apply_measurement(
        self,
        kernel: cudaq.Kernel,
        qubit: cudaq.qubit,
    ) -> None:
        """
        Apply measurement operations to the given qubits in the CUDA-Q kernel.

        Args:
            kernel (cudaq.Kernel): The kernel to apply measurements to.
            qubit cudaq.qubit: The qubit to be measured.
        """
        kernel.mz(qubit)

    def convert_result(self, sample: cudaq.SampleResult) -> qm_bs.BitsSampleSet:
        """Convert a CUDA-Q sample result to a Qamomile BitsSampleSet.

        Args:
            sample (cudaq.SampleResult): the sample result from a CUDA-Q kernel execution

        Returns:
            qm_bs.BitsSampleSet: the converted BitsSampleSet containing the samples as bit strings
        """
        # Create a dictionary whose keys and values are (decimal) integer representations of the bit strings and their counts.
        decimal_result = {}
        for key, value in sample.items():
            decimal_result[int(key, 2)] = value
        # Get the number of bits from the last key.
        num_bits = len(key)

        return qm_bs.BitsSampleSet.from_int_counts(decimal_result, num_bits)
