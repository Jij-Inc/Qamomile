core.circuit.circuit
====================

.. py:module:: core.circuit.circuit

.. autoapi-nested-parse::

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
   - ThreeQubitGate: Represents three-qubit gates (Toffoli/CCX)
   - Operator: Represents a sub-circuit that can be added as a gate
   - QuantumCircuit: Main class for constructing quantum circuits

   This module is essential for building quantum algorithms and simulations. It provides
   a flexible and extensible structure for defining quantum operations and circuits.


   .. rubric:: Example

   ```python
   # Bell state circuit
   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cnot(0, 1)
   qc.measure_all()
   ```



Classes
-------

.. autoapisummary::

   core.circuit.circuit.Gate
   core.circuit.circuit.SingleQubitGateType
   core.circuit.circuit.SingleQubitGate
   core.circuit.circuit.ParametricSingleQubitGateType
   core.circuit.circuit.ParametricSingleQubitGate
   core.circuit.circuit.TwoQubitGateType
   core.circuit.circuit.TwoQubitGate
   core.circuit.circuit.ThreeQubitGateType
   core.circuit.circuit.ThreeQubitGate
   core.circuit.circuit.MeasurementGate
   core.circuit.circuit.Operator
   core.circuit.circuit.QuantumCircuit


Module Contents
---------------

.. py:class:: Gate

   Bases: :py:obj:`abc.ABC`


   Abstract base class for all quantum gates.


.. py:class:: SingleQubitGateType

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



.. py:class:: SingleQubitGate

   Bases: :py:obj:`Gate`


   Unparameterized single qubit gate class.


   .. py:attribute:: gate
      :type:  SingleQubitGateType


   .. py:attribute:: qubit
      :type:  int


.. py:class:: ParametricSingleQubitGateType

   Bases: :py:obj:`enum.Enum`


   Enum class for parametric single qubit gates.


   .. py:attribute:: RX
      :value: 0



   .. py:attribute:: RY
      :value: 1



   .. py:attribute:: RZ
      :value: 2



.. py:class:: ParametricSingleQubitGate

   Bases: :py:obj:`Gate`


   Parameterized single qubit gate class.


   .. py:attribute:: gate
      :type:  ParametricSingleQubitGateType


   .. py:attribute:: qubit
      :type:  int


   .. py:attribute:: parameter
      :type:  core.circuit.parameter.ParameterExpression


.. py:class:: TwoQubitGateType

   Bases: :py:obj:`enum.Enum`


   Enum class for two qubit gates.


   .. py:attribute:: CNOT
      :value: 0



   .. py:attribute:: CZ
      :value: 1



.. py:class:: TwoQubitGate

   Bases: :py:obj:`Gate`


   Two qubit gate class.


   .. py:attribute:: gate
      :type:  TwoQubitGateType


   .. py:attribute:: control
      :type:  int


   .. py:attribute:: target
      :type:  int


.. py:class:: ThreeQubitGateType

   Bases: :py:obj:`enum.Enum`


   Enum class for three qubit gates.


   .. py:attribute:: CCX
      :value: 0



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


.. py:class:: MeasurementGate

   Bases: :py:obj:`Gate`


   Measurement gate class.


   .. py:attribute:: qubit
      :type:  int


   .. py:attribute:: cbit
      :type:  int


.. py:class:: Operator(circuit: QuantumCircuit, label: Optional[str] = None)

   Bases: :py:obj:`Gate`


   Represents a sub-circuit that can be added as a gate.


   .. py:attribute:: circuit


   .. py:attribute:: label


.. py:class:: QuantumCircuit(num_qubits: int, num_clbits: Optional[int] = None, name: Optional[str] = None)

   Quantum circuit class.

   This class represents a quantum circuit and provides methods to add various
   quantum gates and operators to the circuit.


   .. py:attribute:: gates
      :value: []



   .. py:attribute:: num_qubits


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



   .. py:method:: ry(angle: core.circuit.parameter.ParameterExpression, index: int)

      Add a parametric RY gate to the quantum circuit.



   .. py:method:: rz(angle: core.circuit.parameter.ParameterExpression, index: int)

      Add a parametric RZ gate to the quantum circuit.



   .. py:method:: cnot(controled_qubit: int, target_qubit: int)

      Add a CNOT gate to the quantum circuit.



   .. py:method:: cx(controled_qubit: int, target_qubit: int)

      Add a CNOT gate to the quantum circuit.



   .. py:method:: cz(controled_qubit: int, target_qubit: int)

      Add a CZ gate to the quantum circuit.



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



