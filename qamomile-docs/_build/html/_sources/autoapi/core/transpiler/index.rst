core.transpiler
===============

.. py:module:: core.transpiler

.. autoapi-nested-parse::

   This module defines the abstract base class for quantum SDK transpilers in Qamomile.

   The QuantumSDKTranspiler class serves as a template for creating transpilers
   that convert between Qamomile's internal representations and other quantum SDKs.
   It ensures a consistent interface for circuit transpilation, Hamiltonian conversion,
   and result interpretation across different quantum computing platforms.

   Key Features:
   - Abstract methods for converting quantum circuits, Hamiltonians, and measurement results.
   - Generic typing to specify the expected result type from the target SDK.
   - Designed to be subclassed for each specific quantum SDK integration.

   Usage:
   Developers implementing transpilers for specific quantum SDKs should subclass
   QuantumSDKTranspiler and implement all abstract methods. The ResultType generic
   parameter should be set to the appropriate result type of the target SDK.

   .. rubric:: Example

   class QiskitTranspiler(QuantumSDKTranspiler[qiskit.result.Result]):
       def convert_result(self, result: qiskit.result.Result) -> qm.BitsSampleSet:
           # Implementation for converting Qiskit results to Qamomile's BitsSampleSet
           ...

       def transpile_circuit(self, circuit: qm_c.QuantumCircuit) -> qiskit.QuantumCircuit:
           # Implementation for converting Qamomile's QuantumCircuit to Qiskit's QuantumCircuit
           ...

       def transpile_hamiltonian(self, operator: qm_o.Hamiltonian) -> qiskit.quantum_info.Operator:
           # Implementation for converting Qamomile's Hamiltonian to Qiskit's Operator
           ...



Attributes
----------

.. autoapisummary::

   core.transpiler.ResultType


Classes
-------

.. autoapisummary::

   core.transpiler.QuantumSDKTranspiler


Module Contents
---------------

.. py:data:: ResultType

.. py:class:: QuantumSDKTranspiler

   Bases: :py:obj:`Generic`\ [\ :py:obj:`ResultType`\ ], :py:obj:`abc.ABC`


   Abstract base class for quantum SDK transpilers in Qamomile.

   This class defines the interface for transpilers that convert between
   Qamomile's internal representations and other quantum SDKs. It uses
   generic typing to specify the expected result type from the target SDK.

   .. attribute:: ResultType

      A type variable representing the result type of the target SDK.

      :type: TypeVar

   .. method:: convert_result

      Convert SDK-specific result to Qamomile's BitsSampleSet.

   .. method:: transpile_circuit

      Convert Qamomile's QuantumCircuit to SDK-specific circuit.

   .. method:: transpile_hamiltonian

      Convert Qamomile's Hamiltonian to SDK-specific operator.
      

   .. note::

      When subclassing, specify the concrete type for ResultType, e.g.,
      class QiskitTranspiler(QuantumSDKTranspiler[qiskit.result.Result]):


   .. py:method:: convert_result(result: ResultType) -> qamomile.core.BitsSampleSet
      :abstractmethod:


      Convert the result from the target SDK to Qamomile's BitsSampleSet.

      This method should be implemented to interpret the measurement results
      from the target SDK and convert them into Qamomile's BitsSampleSet format.

      :param result: The measurement result from the target SDK.
      :type result: ResultType

      :returns: The converted result in Qamomile's format.
      :rtype: qm.BitsSampleSet

      :raises NotImplementedError: If the method is not implemented in the subclass.



   .. py:method:: transpile_circuit(circuit: qamomile.core.circuit.QuantumCircuit)
      :abstractmethod:


      Transpile a Qamomile QuantumCircuit to the target SDK's circuit representation.

      This method should be implemented to convert Qamomile's internal
      QuantumCircuit representation to the equivalent circuit in the target SDK.

      :param circuit: The Qamomile QuantumCircuit to be transpiled.
      :type circuit: qm_c.QuantumCircuit

      :returns: The equivalent circuit in the target SDK's format.

      :raises NotImplementedError: If the method is not implemented in the subclass.



   .. py:method:: transpile_operators(operators: list[qamomile.core.operator.Hamiltonian])

      Transpile a list of Qamomile Hamiltonians to the target SDK's operator representation.

      This method should be implemented to convert a list of Qamomile Hamiltonians
      to the equivalent operators in the target SDK.

      :param operators: The list of Qamomile Hamiltonians to be transpiled.
      :type operators: list[qm_o.Hamiltonian]

      :returns: The equivalent operators in the target SDK's format.
      :rtype: list

      :raises NotImplementedError: If the method is not implemented in the subclass.



   .. py:method:: transpile_hamiltonian(operator: qamomile.core.operator.Hamiltonian)
      :abstractmethod:


      Transpile a Qamomile Hamiltonian to the target SDK's operator representation.

      This method should be implemented to convert Qamomile's internal
      Hamiltonian representation to the equivalent operator in the target SDK.

      :param operator: The Qamomile Hamiltonian to be transpiled.
      :type operator: qm_o.Hamiltonian

      :returns: The equivalent operator in the target SDK's format.

      :raises NotImplementedError: If the method is not implemented in the subclass.



