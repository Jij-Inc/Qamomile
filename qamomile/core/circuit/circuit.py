"""
Quantum Circuit and Gates Module

This module provides a comprehensive framework for defining and manipulating quantum circuits
and gates. It includes classes for various types of quantum gates (single-qubit, two-qubit,
three-qubit, and parametric gates) as well as a QuantumCircuit class for constructing
quantum circuits.

Key components:
- Gate: Abstract base class for all quantum gates
- SingleQubitGate: Represents unparameterized single-qubit gates (H, X, Y, Z, S, T)
- ParametricSingleQubitGate: Represents parameterized single-qubit gates (RX, RY, RZ)
- TwoQubitGate: Represents two-qubit gates (CNOT, CZ)
- ParametricTwoQubitGate: Represents parametric two-qubit gates (CRX, CRY, CRZ, RXX, RYY, RZZ)
- ThreeQubitGate: Represents three-qubit gates (Toffoli/CCX)
- Operator: Represents a sub-circuit that can be added as a gate
- QuantumCircuit: Main class for constructing quantum circuits

This module is essential for building quantum algorithms and simulations. It provides
a flexible and extensible structure for defining quantum operations and circuits.


Example:
    ```python
    # Bell state circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cnot(0, 1)
    qc.measure_all()
    ```

"""

import typing as typ
import dataclasses
import abc
import enum
from .parameter import ParameterExpression, Parameter


class Gate(abc.ABC):
    """Abstract base class for all quantum gates."""

    pass


class SingleQubitGateType(enum.Enum):
    """Enum class for single qubit gates."""

    H = 0  # Hadamard gate
    X = 1  # Pauli-X gate
    Y = 2  # Pauli-Y gate
    Z = 3  # Pauli-Z gate
    S = 4  # S gate (phase gate)
    T = 5  # T gate (π/8 gate)


@dataclasses.dataclass
class SingleQubitGate(Gate):
    """Unparameterized single qubit gate class."""

    gate: SingleQubitGateType
    qubit: int


class ParametricSingleQubitGateType(enum.Enum):
    """Enum class for parametric single qubit gates."""

    RX = 0  # Rotation around X-axis
    RY = 1  # Rotation around Y-axis
    RZ = 2  # Rotation around Z-axis


@dataclasses.dataclass
class ParametricSingleQubitGate(Gate):
    """Parameterized single qubit gate class."""

    gate: ParametricSingleQubitGateType
    qubit: int
    parameter: ParameterExpression


class TwoQubitGateType(enum.Enum):
    """Enum class for two qubit gates."""

    CNOT = 0  # Controlled-NOT gate
    CZ = 1  # Controlled-Z gate


@dataclasses.dataclass
class TwoQubitGate(Gate):
    """Two qubit gate class."""

    gate: TwoQubitGateType
    control: int
    target: int


class ParametricTwoQubitGateType(enum.Enum):
    """Enum class for parametric two qubit gates."""

    CRX = 0  # Controlled-RX gate
    CRY = 1  # Controlled-RY gate
    CRZ = 2  # Controlled-RZ gate
    RXX = 3  # XX rotation gate
    RYY = 4  # YY rotation gate
    RZZ = 5  # ZZ rotation gate


@dataclasses.dataclass
class ParametricTwoQubitGate(Gate):
    """Parameterized two qubit gate class."""

    gate: ParametricTwoQubitGateType
    control: int
    target: int
    parameter: ParameterExpression


class ThreeQubitGateType(enum.Enum):
    """Enum class for three qubit gates."""

    CCX = 0  # Toffoli Gate (Controlled-Controlled-X)


@dataclasses.dataclass
class ThreeQubitGate(Gate):
    """Three qubit gate class."""

    gate: ThreeQubitGateType
    control1: int
    control2: int
    target: int


@dataclasses.dataclass
class MeasurementGate(Gate):
    """Measurement gate class."""

    qubit: int
    cbit: int


class Operator(Gate):
    """Represents a sub-circuit that can be added as a gate."""

    def __init__(
        self, circuit: "QuantumCircuit", label: typ.Optional[str] = None
    ) -> None:
        self.circuit = circuit
        self.label = label

    def operated_qubits(self) -> list[int]:
        operated_qubits = []
        for gate in self.circuit.gates:
            if isinstance(gate, (SingleQubitGate, ParametricSingleQubitGate)):
                operated_qubits.append(gate.qubit)
            elif isinstance(gate, (TwoQubitGate, ParametricTwoQubitGate)):
                operated_qubits.append(gate.control)
                operated_qubits.append(gate.target)
            elif isinstance(gate, ThreeQubitGate):
                operated_qubits.append(gate.control1)
                operated_qubits.append(gate.control2)
                operated_qubits.append(gate.target)
            elif isinstance(gate, Operator):
                operated_qubits.extend(gate.operated_qubits())
            else:
                raise ValueError(f"Invalid gate type: {type(gate)}")
        return list(set(operated_qubits))


class QuantumCircuit:
    """
    Quantum circuit class.

    This class represents a quantum circuit and provides methods to add various
    quantum gates and operators to the circuit.
    """

    def __init__(
        self,
        num_qubits: int,
        num_clbits: int = 0,
        name: typ.Optional[str] = None,
    ) -> None:
        """
        Initialize a quantum circuit with a specified number of qubits.

        Args:
            num_qubits (int): The number of qubits in the circuit.
        """
        self.gates: list[Gate] = []
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits

        self.name = name

        self._qubits_label: list[str] = ["q_{" + str(i) + "}" for i in range(num_qubits)]

    def update_qubits_label(self, qubits_label: dict[int, str]):
        """
        Update the qubits label.

        Args:
            qubits_label (dict[int, str]): A dictionary of qubit index and its label.
        """
        for i in range(self.num_qubits):
            if i in qubits_label:
                self._qubits_label[i] = qubits_label[i]

    @property
    def qubits_label(self) -> list[str]:
        return self._qubits_label

    def add_gate(self, gate: Gate):
        """
        Add a gate to the quantum circuit.

        This method checks if the gate's qubit indices are valid before adding it to the circuit.

        Args:
            gate (Gate): A gate to be added.

        Raises:
            ValueError: If the gate's qubit indices are invalid.
        """
        # Check the number of qubits
        if isinstance(gate, (SingleQubitGate, ParametricSingleQubitGate)):
            if not gate.qubit < self.num_qubits:
                raise ValueError(f"Invalid qubit index: {gate.qubit}")
        elif isinstance(gate, TwoQubitGate):
            control_q_check = gate.control < self.num_qubits
            target_q_check = gate.target < self.num_qubits
            if not (control_q_check and target_q_check):
                raise ValueError(
                    f"Invalid qubit index. controled_qubit: {gate.control}, target_qubit: {gate.target}"
                )
        elif isinstance(gate, ThreeQubitGate):
            control_q_check = gate.control1 < self.num_qubits
            target_q_check = gate.control2 < self.num_qubits
            target_q2_check = gate.target < self.num_qubits
            if not (control_q_check and target_q_check and target_q2_check):
                raise ValueError(
                    f"Invalid qubit index. controled_qubit1: {gate.control1}, controled_qubit2: {gate.control2}, target_qubit: {gate.target}"
                )
        elif isinstance(gate, Operator):
            if gate.circuit.num_qubits < self.num_qubits:
                raise ValueError(
                    f"Invalid number of qubits. Expected: {self.num_qubits}, Actual: {gate.circuit.num_qubits}"
                )
        elif isinstance(gate, MeasurementGate):
            if gate.qubit >= self.num_qubits or gate.cbit >= self.num_clbits:
                raise ValueError(
                    f"Invalid index. qubit: {gate.qubit}, classical bit: {gate.cbit}"
                )

        self.gates.append(gate)

    def __repr__(self) -> str:
        """String representation of the QuantumCircuit."""
        return f"{self.gates}"

    # Methods for adding single-qubit gates
    def x(self, index: int):
        """Add a Pauli X gate to the quantum circuit."""
        self.add_gate(SingleQubitGate(SingleQubitGateType.X, index))

    def y(self, index: int):
        """Add a Pauli Y gate to the quantum circuit."""
        self.add_gate(SingleQubitGate(SingleQubitGateType.Y, index))

    def z(self, index: int):
        """Add a Pauli Z gate to the quantum circuit."""
        self.add_gate(SingleQubitGate(SingleQubitGateType.Z, index))

    def h(self, index: int):
        """Add a Hadamard gate to the quantum circuit."""
        self.add_gate(SingleQubitGate(SingleQubitGateType.H, index))

    def s(self, index: int):
        """Add an S gate to the quantum circuit."""
        self.add_gate(SingleQubitGate(SingleQubitGateType.S, index))

    def t(self, index: int):
        """Add a T gate to the quantum circuit."""
        self.add_gate(SingleQubitGate(SingleQubitGateType.T, index))

    # Methods for adding parametric single-qubit gates
    def rx(self, angle: ParameterExpression, index: int):
        r"""Add a parametric RX gate to the quantum circuit.

        .. math::
            RX(\theta) = \exp\left(-i\theta X/2\right)
            = \begin{bmatrix}
            \cos(\theta/2) & -i\sin(\theta/2) \\
            -i\sin(\theta/2) & \cos(\theta/2)
            \end{bmatrix}

        Args:
            angle (ParameterExpression): The angle parameter for the gate.
            index (int): The index of the qubit to apply the gate. 
        """
        self.add_gate(
            ParametricSingleQubitGate(ParametricSingleQubitGateType.RX, index, angle)
        )

    def ry(self, angle: ParameterExpression, index: int):
        r"""Add a parametric RY gate to the quantum circuit.

        .. math::
            RY(\theta) = \exp\left(-i\theta Y/2\right)
            = \begin{bmatrix}
            \cos(\theta/2) & -\sin(\theta/2) \\
            \sin(\theta/2) & \cos(\theta/2)
            \end{bmatrix}

        Args:
            angle (ParameterExpression): The angle parameter for the gate.
            index (int): The index of the qubit to apply the gate.
        """
        self.add_gate(
            ParametricSingleQubitGate(ParametricSingleQubitGateType.RY, index, angle)
        )

    def rz(self, angle: ParameterExpression, index: int):
        r"""Add a parametric RZ gate to the quantum circuit.

        .. math::
            RZ(\theta) = \exp\left(-i\theta Z/2\right)
            = \begin{bmatrix}
            e^{-i\theta/2} & 0 \\
            0 & e^{i\theta/2}
            \end{bmatrix}

        Args:
            angle (ParameterExpression): The angle parameter for the gate.
            index (int): The index of the qubit to apply the gate. 
        """
        self.add_gate(
            ParametricSingleQubitGate(ParametricSingleQubitGateType.RZ, index, angle)
        )

    # Methods for adding two-qubit gates
    def cnot(self, controled_qubit: int, target_qubit: int):
        """Add a CNOT gate to the quantum circuit."""
        self.cx(controled_qubit, target_qubit)

    # Methods for adding two-qubit gates
    def cx(self, controled_qubit: int, target_qubit: int):
        """Add a CNOT gate to the quantum circuit."""
        self.add_gate(
            TwoQubitGate(TwoQubitGateType.CNOT, controled_qubit, target_qubit)
        )

    def cz(self, controled_qubit: int, target_qubit: int):
        """Add a CZ gate to the quantum circuit."""
        self.add_gate(TwoQubitGate(TwoQubitGateType.CZ, controled_qubit, target_qubit))

    def crx(self, angle: ParameterExpression, controled_qubit: int, target_qubit: int):
        """Add a CRX gate to the quantum circuit."""
        self.add_gate(
            ParametricTwoQubitGate(
                ParametricTwoQubitGateType.CRX, controled_qubit, target_qubit, angle
            )
        )
    
    def cry(self, angle: ParameterExpression, controled_qubit: int, target_qubit: int):
        """Add a CRY gate to the quantum circuit."""
        self.add_gate(
            ParametricTwoQubitGate(
                ParametricTwoQubitGateType.CRY, controled_qubit, target_qubit, angle
            )
        )
    
    def crz(self, angle: ParameterExpression, controled_qubit: int, target_qubit: int):
        """Add a CRZ gate to the quantum circuit."""
        self.add_gate(
            ParametricTwoQubitGate(
                ParametricTwoQubitGateType.CRZ, controled_qubit, target_qubit, angle
            )
        )

    def rxx(self, angle: ParameterExpression, qubit1: int, qubit2: int):
        r"""Add a RXX gate to the quantum circuit.

        .. math::
            R_{XX}(\theta) = \exp\left(-i\theta X\otimes X/2\right)
        """
        self.add_gate(
            ParametricTwoQubitGate(
                ParametricTwoQubitGateType.RXX, qubit1, qubit2, angle
            )
        )

    def ryy(self, angle: ParameterExpression, qubit1: int, qubit2: int):
        r"""Add a RYY gate to the quantum circuit.
        
        .. math::
            R_{YY}(\theta) = \exp\left(-i\theta Y\otimes Y/2\right)
        """
        self.add_gate(
            ParametricTwoQubitGate(
                ParametricTwoQubitGateType.RYY, qubit1, qubit2, angle
            )
        )

    def rzz(self, angle: ParameterExpression, qubit1: int, qubit2: int):
        r"""Add a RZZ gate to the quantum circuit.

        .. math::
            R_{ZZ}(\theta) = \exp\left(-i\theta Z\otimes Z/2\right)
        """
        self.add_gate(
            ParametricTwoQubitGate(
                ParametricTwoQubitGateType.RZZ, qubit1, qubit2, angle
            )
        )

    # Method for adding three-qubit gate
    def ccx(self, control1: int, control2: int, target: int):
        """Add a Toffoli gate to the quantum circuit."""
        self.add_gate(
            ThreeQubitGate(ThreeQubitGateType.CCX, control1, control2, target)
        )

    def measure(self, qubit: int, cbit: int):
        """
        Add a measurement gate to the quantum circuit.

        Args:
            qubit (int): The index of the qubit to be measured.
            cbit (int): The index of the classical bit to store the measurement result.
        """
        if qubit >= self.num_qubits:
            raise ValueError(f"Invalid qubit index: {qubit}")
        if cbit >= self.num_clbits:
            raise ValueError(f"Invalid classical bit index: {cbit}")
        self.add_gate(MeasurementGate(qubit, cbit))

    def measure_all(self):
        """
        Add measurement gates for all qubits.
        """
        if self.num_clbits < self.num_qubits:
            # Add classical bits if not enough
            self.num_clbits = self.num_qubits
        for i in range(self.num_qubits):
            self.measure(i, i)

    def append(self, gate: typ.Union[Gate, "QuantumCircuit"]):
        """
        Append another quantum circuit to this quantum circuit.

        Args:
            qc (QuantumCircuit): The quantum circuit to be appended.
        """
        if isinstance(gate, QuantumCircuit):
            self.add_gate(gate.to_gate())
        else:
            self.add_gate(gate)

    def to_gate(self, label: typ.Optional[str] = None) -> Operator:
        """
        Convert the quantum circuit to an operator (sub-circuit).

        Args:
            label (str): The label for the operator.

        Returns:
            Operator: The operator representing the quantum circuit.
        """
        if label is None:
            label = self.name
        self.num_clbits = 0
        return Operator(self, label=label)

    def get_parameters(self) -> set[Parameter]:
        """
        Get the parameters in the quantum circuit.

        Returns:
            set[Parameter]: The unique set of parameters in the quantum circuit.
        """
        parameters: list[Parameter] = []
        for gate in self.gates:
            if isinstance(gate, ParametricSingleQubitGate):
                parameters.extend(gate.parameter.get_parameters())
            elif isinstance(gate, ParametricTwoQubitGate):
                parameters.extend(gate.parameter.get_parameters())
            elif isinstance(gate, Operator):
                parameters.extend(gate.circuit.get_parameters())
        return set(parameters)
