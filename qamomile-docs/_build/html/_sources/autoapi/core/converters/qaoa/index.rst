core.converters.qaoa
====================

.. py:module:: core.converters.qaoa

.. autoapi-nested-parse::

   This module implements the Quantum Approximate Optimization Algorithm (QAOA) converter
   for the Qamomile framework. It provides functionality to convert optimization problems
   into QAOA circuits, construct cost Hamiltonians, and decode quantum computation results.

   The QAOAConverter class extends the QuantumConverter base class, specializing in
   QAOA-specific operations such as ansatz circuit generation and result decoding.

   Key Features:
   - Generation of QAOA ansatz circuits
   - Construction of cost Hamiltonians for QAOA
   - Decoding of quantum computation results into classical optimization solutions

   Usage:
       from qamomile.core.qaoa.qaoa import QAOAConverter

       # Initialize with a compiled optimization problem instance
       qaoa_converter = QAOAConverter(compiled_instance)

       # Generate QAOA circuit
       p = 2  # Number of QAOA layers
       qaoa_circuit = qaoa_converter.get_ansatz_circuit(p)

       # Get cost Hamiltonian
       cost_hamiltonian = qaoa_converter.get_cost_hamiltonian()


   Note: This module requires jijmodeling and jijmodeling_transpiler for problem representation
   and decoding functionalities.



Classes
-------

.. autoapisummary::

   core.converters.qaoa.QAOAConverter


Module Contents
---------------

.. py:class:: QAOAConverter(compiled_instance, relax_method: jijmodeling_transpiler.core.pubo.RelaxationMethod = jmt.pubo.RelaxationMethod.AugmentedLagrangian)

   Bases: :py:obj:`qamomile.core.converters.converter.QuantumConverter`


   QAOA (Quantum Approximate Optimization Algorithm) converter class.

   This class provides methods to convert optimization problems into QAOA circuits,
   construct cost Hamiltonians, and decode quantum computation results.


   .. py:method:: get_cost_ansatz(beta: qamomile.core.circuit.Parameter, name: str = 'Cost') -> qamomile.core.circuit.QuantumCircuit

      Generate the cost ansatz circuit for QAOA.

      :param beta: The beta parameter for the cost ansatz.
      :type beta: qm_c.Parameter
      :param name: Name of the circuit. Defaults to "Cost".
      :type name: str, optional

      :returns: The cost ansatz circuit.
      :rtype: qm_c.QuantumCircuit



   .. py:method:: get_qaoa_ansatz(p: int, initial_hadamard: bool = True) -> qamomile.core.circuit.QuantumCircuit

      Generate the complete QAOA ansatz circuit.

      :param p: Number of QAOA layers.
      :type p: int
      :param initial_hadamard: Whether to apply initial Hadamard gates. Defaults to True.
      :type initial_hadamard: bool, optional

      :returns: The complete QAOA ansatz circuit.
      :rtype: qm_c.QuantumCircuit



   .. py:method:: get_cost_hamiltonian() -> qamomile.core.operator.Hamiltonian

      Construct the cost Hamiltonian for QAOA.

      :returns: The cost Hamiltonian.
      :rtype: qm_o.Hamiltonian



