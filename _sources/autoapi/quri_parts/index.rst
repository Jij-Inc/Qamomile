quri_parts
==========

.. py:module:: quri_parts


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/quri_parts/exceptions/index
   /autoapi/quri_parts/parameter_converter/index
   /autoapi/quri_parts/transpiler/index


Exceptions
----------

.. autoapisummary::

   quri_parts.QamomileQuriPartsTranspileError


Classes
-------

.. autoapisummary::

   quri_parts.QuriPartsTranspiler


Package Contents
----------------

.. py:class:: QuriPartsTranspiler

   Bases: :py:obj:`qamomile.core.transpiler.QuantumSDKTranspiler`\ [\ :py:obj:`tuple`\ [\ :py:obj:`collections.Counter`\ [\ :py:obj:`int`\ ]\ , :py:obj:`int`\ ]\ ]


   Transpiler class for converting between Qamomile and QuriParts quantum objects.

   This class implements the QuantumSDKTranspiler interface for QuriParts compatibility,
   providing methods to convert circuits, Hamiltonians, and measurement results.


   .. py:method:: transpile_circuit(qamomile_circuit: qamomile.core.circuit.QuantumCircuit) -> quri_parts.circuit.LinearMappedUnboundParametricQuantumCircuit

      Convert a Qamomile quantum circuit to a QuriParts quantum circuit.

      :param qamomile_circuit: The Qamomile quantum circuit to convert.
      :type qamomile_circuit: qm_c.QuantumCircuit

      :returns: The converted QuriParts quantum circuit.
      :rtype: qp_c.LinearMappedUnboundParametricQuantumCircuit

      :raises QamomileQuriPartsTranspileError: If there's an error during conversion.



   .. py:method:: convert_result(result: tuple[collections.Counter[int], int]) -> qamomile.core.bitssample.BitsSampleSet

      Convert QuriParts measurement results to Qamomile BitsSampleSet.

      :param result: QuriParts measurement results.
      :type result: tuple[collections.Counter[int], int]

      :returns: Converted Qamomile BitsSampleSet.
      :rtype: qm_bs.BitsSampleSet



   .. py:method:: transpile_hamiltonian(operator: qamomile.core.operator.Hamiltonian) -> quri_parts.core.operator.Operator

      Convert a Qamomile Hamiltonian to a QuriParts Operator.

      :param operator: The Qamomile Hamiltonian to convert.
      :type operator: qm_o.Hamiltonian

      :returns: The converted QuriParts Operator.
      :rtype: qp_o.Operator

      :raises NotImplementedError: If an unsupported Pauli operator is encountered.



.. py:exception:: QamomileQuriPartsTranspileError

   Bases: :py:obj:`Exception`


   Exception raised for errors in the Qamomile to Qiskit conversion process.


