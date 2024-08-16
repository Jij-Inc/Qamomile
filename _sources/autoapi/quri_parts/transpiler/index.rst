quri_parts.transpiler
=====================

.. py:module:: quri_parts.transpiler

.. autoapi-nested-parse::

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



Classes
-------

.. autoapisummary::

   quri_parts.transpiler.QuriPartsTranspiler


Module Contents
---------------

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



