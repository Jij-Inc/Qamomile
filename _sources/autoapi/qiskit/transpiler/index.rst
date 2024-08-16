qiskit.transpiler
=================

.. py:module:: qiskit.transpiler

.. autoapi-nested-parse::

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



Classes
-------

.. autoapisummary::

   qiskit.transpiler.QiskitTranspiler


Module Contents
---------------

.. py:class:: QiskitTranspiler

   Bases: :py:obj:`qamomile.core.transpiler.QuantumSDKTranspiler`\ [\ :py:obj:`qiskit.primitives.BitArray`\ ]


   Transpiler class for converting between Qamomile and Qiskit quantum objects.

   This class implements the QuantumSDKTranspiler interface for Qiskit compatibility,
   providing methods to convert circuits, Hamiltonians, and measurement results.


   .. py:method:: transpile_circuit(qamomile_circuit: qamomile.core.circuit.QuantumCircuit) -> qiskit.QuantumCircuit

      Convert a Qamomile quantum circuit to a Qiskit quantum circuit.

      :param qamomile_circuit: The Qamomile quantum circuit to convert.
      :type qamomile_circuit: qm_c.QuantumCircuit

      :returns: The converted Qiskit quantum circuit.
      :rtype: qiskit.QuantumCircuit

      :raises QamomileQiskitConverterError: If there's an error during conversion.



   .. py:method:: convert_result(result: qiskit.primitives.BitArray) -> qamomile.core.bitssample.BitsSampleSet

      Convert Qiskit measurement results to Qamomile BitsSampleSet.

      :param result: Qiskit measurement results.
      :type result: qk_primitives.BitArray

      :returns: Converted Qamomile BitsSampleSet.
      :rtype: qm.BitsSampleSet



   .. py:method:: transpile_hamiltonian(operator: qamomile.core.operator.Hamiltonian) -> qiskit.quantum_info.SparsePauliOp

      Convert a Qamomile Hamiltonian to a Qiskit SparsePauliOp.

      :param operator: The Qamomile Hamiltonian to convert.
      :type operator: qm_o.Hamiltonian

      :returns: The converted Qiskit SparsePauliOp.
      :rtype: qk_ope.SparsePauliOp

      :raises NotImplementedError: If an unsupported Pauli operator is encountered.



