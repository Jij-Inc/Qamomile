core.circuit
============

.. py:module:: core.circuit

.. autoapi-nested-parse::

   Quantum Circuit Package

   This package provides tools for creating and manipulating quantum circuits,
   including parameter expressions and various quantum gates.

   It includes:
   - Parameter expressions for defining parameterized quantum operations
   - Quantum gate definitions (single-qubit, two-qubit, three-qubit, and parametric gates)
   - Quantum circuit class for building and manipulating quantum circuits

   Usage:
       from qamomile.core.circuit import QuantumCircuit, Parameter
       qc = QuantumCircuit(2)
       theta = Parameter('theta')
       qc.rx(theta, 0)
       qc.cnot(0, 1)



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/core/circuit/circuit/index
   /autoapi/core/circuit/parameter/index


Classes
-------

.. autoapisummary::

   core.circuit.Parameter
   core.circuit.ParameterExpression
   core.circuit.BinaryOperator
   core.circuit.Value
   core.circuit.BinaryOpeKind
   core.circuit.QuantumCircuit
   core.circuit.Gate
   core.circuit.SingleQubitGate
   core.circuit.ParametricSingleQubitGate
   core.circuit.TwoQubitGate
   core.circuit.ThreeQubitGate
   core.circuit.Operator
   core.circuit.SingleQubitGateType
   core.circuit.ParametricSingleQubitGateType
   core.circuit.TwoQubitGateType
   core.circuit.ThreeQubitGateType
   core.circuit.ParametricTwoQubitGate
   core.circuit.ParametricTwoQubitGateType
   core.circuit.MeasurementGate


Package Contents
----------------

.. py:class:: Parameter(name)

   Bases: :py:obj:`ParameterExpression`


   Represents a named parameter in a quantum circuit.


   .. py:attribute:: name


   .. py:method:: get_parameters() -> list[Parameter]

      Return this parameter in a list.



.. py:class:: ParameterExpression

   Bases: :py:obj:`abc.ABC`


   Abstract base class for parameter expressions in quantum circuits.

   This class defines the basic operations (addition, multiplication, division)
   that can be performed on parameter expressions.


   .. py:method:: get_parameters() -> list[Parameter]

      Get the parameters in the expression.

      :returns: The parameters in the expression.
      :rtype: list[Parameter]



.. py:class:: BinaryOperator(left, right, kind)

   Bases: :py:obj:`ParameterExpression`


   Represents a binary operation between two ParameterExpressions.


   .. py:attribute:: left
      :type:  ParameterExpression


   .. py:attribute:: right
      :type:  ParameterExpression


   .. py:attribute:: kind
      :type:  BinaryOpeKind


   .. py:method:: get_parameters() -> list[Parameter]

      Get all parameters involved in this binary operation.

      :returns: A list of all parameters in the expression.
      :rtype: list[Parameter]



.. py:class:: Value(value)

   Bases: :py:obj:`ParameterExpression`


   Represents a constant numeric value in an expression.


   .. py:attribute:: value


.. py:class:: BinaryOpeKind(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   Enumeration of binary operation types.


   .. py:attribute:: ADD
      :value: '+'



   .. py:attribute:: MUL
      :value: '*'



   .. py:attribute:: DIV
      :value: '/'



.. py:class:: QuantumCircuit(num_qubits: int, num_clbits: int = 0, name: Optional[str] = None)

   Quantum circuit class.

   This class represents a quantum circuit and provides methods to add various
   quantum gates and operators to the circuit.


   .. py:attribute:: gates
      :value: []



   .. py:attribute:: num_qubits


   .. py:attribute:: num_clbits


   .. py:attribute:: name


   .. py:method:: add_gate(gate: Gate)

      Add a gate to the quantum circuit.

      This method checks if the gate's qubit indices are valid before adding it to the circuit.

      :param gate: A gate to be added.
      :type gate: Gate

      :raises ValueError: If the gate's qubit indices are invalid.



   .. py:method:: x(index: int)

      Add a Pauli X gate to the quantum circuit.



   .. py:method:: y(index: int)

      Add a Pauli Y gate to the quantum circuit.



   .. py:method:: z(index: int)

      Add a Pauli Z gate to the quantum circuit.



   .. py:method:: h(index: int)

      Add a Hadamard gate to the quantum circuit.



   .. py:method:: s(index: int)

      Add an S gate to the quantum circuit.



   .. py:method:: t(index: int)

      Add a T gate to the quantum circuit.



   .. py:method:: rx(angle: core.circuit.parameter.ParameterExpression, index: int)

      Add a parametric RX gate to the quantum circuit.

      .. math::
          RX(\theta) = \exp\left(-i\theta X/2\right)
          = \begin{bmatrix}
          \cos(\theta/2) & -i\sin(\theta/2) \\
          -i\sin(\theta/2) & \cos(\theta/2)
          \end{bmatrix}

      :param angle: The angle parameter for the gate.
      :type angle: ParameterExpression
      :param index: The index of the qubit to apply the gate.
      :type index: int



   .. py:method:: ry(angle: core.circuit.parameter.ParameterExpression, index: int)

      Add a parametric RY gate to the quantum circuit.

      .. math::
          RY(\theta) = \exp\left(-i\theta Y/2\right)
          = \begin{bmatrix}
          \cos(\theta/2) & -\sin(\theta/2) \\
          \sin(\theta/2) & \cos(\theta/2)
          \end{bmatrix}

      :param angle: The angle parameter for the gate.
      :type angle: ParameterExpression
      :param index: The index of the qubit to apply the gate.
      :type index: int



   .. py:method:: rz(angle: core.circuit.parameter.ParameterExpression, index: int)

      Add a parametric RZ gate to the quantum circuit.

      .. math::
          RZ(\theta) = \exp\left(-i\theta Z/2\right)
          = \begin{bmatrix}
          e^{-i\theta/2} & 0 \\
          0 & e^{i\theta/2}
          \end{bmatrix}

      :param angle: The angle parameter for the gate.
      :type angle: ParameterExpression
      :param index: The index of the qubit to apply the gate.
      :type index: int



   .. py:method:: cnot(controled_qubit: int, target_qubit: int)

      Add a CNOT gate to the quantum circuit.



   .. py:method:: cx(controled_qubit: int, target_qubit: int)

      Add a CNOT gate to the quantum circuit.



   .. py:method:: cz(controled_qubit: int, target_qubit: int)

      Add a CZ gate to the quantum circuit.



   .. py:method:: crx(angle: core.circuit.parameter.ParameterExpression, controled_qubit: int, target_qubit: int)

      Add a CRX gate to the quantum circuit.



   .. py:method:: cry(angle: core.circuit.parameter.ParameterExpression, controled_qubit: int, target_qubit: int)

      Add a CRY gate to the quantum circuit.



   .. py:method:: crz(angle: core.circuit.parameter.ParameterExpression, controled_qubit: int, target_qubit: int)

      Add a CRZ gate to the quantum circuit.



   .. py:method:: rxx(angle: core.circuit.parameter.ParameterExpression, qubit1: int, qubit2: int)

      Add a RXX gate to the quantum circuit.

      .. math::
          R_{XX}(\theta) = \exp\left(-i\theta X\otimes X/2\right)



   .. py:method:: ryy(angle: core.circuit.parameter.ParameterExpression, qubit1: int, qubit2: int)

      Add a RYY gate to the quantum circuit.

      .. math::
          R_{YY}(\theta) = \exp\left(-i\theta Y\otimes Y/2\right)



   .. py:method:: rzz(angle: core.circuit.parameter.ParameterExpression, qubit1: int, qubit2: int)

      Add a RZZ gate to the quantum circuit.

      .. math::
          R_{ZZ}(\theta) = \exp\left(-i\theta Z\otimes Z/2\right)



   .. py:method:: ccx(control1: int, control2: int, target: int)

      Add a Toffoli gate to the quantum circuit.



   .. py:method:: measure(qubit: int, cbit: int)

      Add a measurement gate to the quantum circuit.

      :param qubit: The index of the qubit to be measured.
      :type qubit: int
      :param cbit: The index of the classical bit to store the measurement result.
      :type cbit: int



   .. py:method:: measure_all()

      Add measurement gates for all qubits.



   .. py:method:: append(gate: Union[Gate, QuantumCircuit])

      Append another quantum circuit to this quantum circuit.

      :param qc: The quantum circuit to be appended.
      :type qc: QuantumCircuit



   .. py:method:: to_gate(label: Optional[str] = None) -> Operator

      Convert the quantum circuit to an operator (sub-circuit).

      :param label: The label for the operator.
      :type label: str

      :returns: The operator representing the quantum circuit.
      :rtype: Operator



   .. py:method:: get_parameters() -> set[core.circuit.parameter.Parameter]

      Get the parameters in the quantum circuit.

      :returns: The unique set of parameters in the quantum circuit.
      :rtype: set[Parameter]



.. py:class:: Gate

   Bases: :py:obj:`abc.ABC`


   Abstract base class for all quantum gates.


.. py:class:: SingleQubitGate

   Bases: :py:obj:`Gate`


   Unparameterized single qubit gate class.


   .. py:attribute:: gate
      :type:  SingleQubitGateType


   .. py:attribute:: qubit
      :type:  int


.. py:class:: ParametricSingleQubitGate

   Bases: :py:obj:`Gate`


   Parameterized single qubit gate class.


   .. py:attribute:: gate
      :type:  ParametricSingleQubitGateType


   .. py:attribute:: qubit
      :type:  int


   .. py:attribute:: parameter
      :type:  core.circuit.parameter.ParameterExpression


.. py:class:: TwoQubitGate

   Bases: :py:obj:`Gate`


   Two qubit gate class.


   .. py:attribute:: gate
      :type:  TwoQubitGateType


   .. py:attribute:: control
      :type:  int


   .. py:attribute:: target
      :type:  int


.. py:class:: ThreeQubitGate

   Bases: :py:obj:`Gate`


   Three qubit gate class.


   .. py:attribute:: gate
      :type:  ThreeQubitGateType


   .. py:attribute:: control1
      :type:  int


   .. py:attribute:: control2
      :type:  int


   .. py:attribute:: target
      :type:  int


.. py:class:: Operator(circuit: QuantumCircuit, label: Optional[str] = None)

   Bases: :py:obj:`Gate`


   Represents a sub-circuit that can be added as a gate.


   .. py:attribute:: circuit


   .. py:attribute:: label


.. py:class:: SingleQubitGateType(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   Enum class for single qubit gates.


   .. py:attribute:: H
      :value: 0



   .. py:attribute:: X
      :value: 1



   .. py:attribute:: Y
      :value: 2



   .. py:attribute:: Z
      :value: 3



   .. py:attribute:: S
      :value: 4



   .. py:attribute:: T
      :value: 5



.. py:class:: ParametricSingleQubitGateType(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   Enum class for parametric single qubit gates.


   .. py:attribute:: RX
      :value: 0



   .. py:attribute:: RY
      :value: 1



   .. py:attribute:: RZ
      :value: 2



.. py:class:: TwoQubitGateType(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   Enum class for two qubit gates.


   .. py:attribute:: CNOT
      :value: 0



   .. py:attribute:: CZ
      :value: 1



.. py:class:: ThreeQubitGateType(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   Enum class for three qubit gates.


   .. py:attribute:: CCX
      :value: 0



.. py:class:: ParametricTwoQubitGate

   Bases: :py:obj:`Gate`


   Parameterized two qubit gate class.


   .. py:attribute:: gate
      :type:  ParametricTwoQubitGateType


   .. py:attribute:: control
      :type:  int


   .. py:attribute:: target
      :type:  int


   .. py:attribute:: parameter
      :type:  core.circuit.parameter.ParameterExpression


.. py:class:: ParametricTwoQubitGateType(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   Enum class for parametric two qubit gates.


   .. py:attribute:: CRX
      :value: 0



   .. py:attribute:: CRY
      :value: 1



   .. py:attribute:: CRZ
      :value: 2



   .. py:attribute:: RXX
      :value: 3



   .. py:attribute:: RYY
      :value: 4



   .. py:attribute:: RZZ
      :value: 5



.. py:class:: MeasurementGate

   Bases: :py:obj:`Gate`


   Measurement gate class.


   .. py:attribute:: qubit
      :type:  int


   .. py:attribute:: cbit
      :type:  int


