core.converters.qrao.qrao31
===========================

.. py:module:: core.converters.qrao.qrao31


Classes
-------

.. autoapisummary::

   core.converters.qrao.qrao31.QRAC31Converter


Functions
---------

.. autoapisummary::

   core.converters.qrao.qrao31.color_group_to_qrac_encode
   core.converters.qrao.qrao31.qrac31_encode_ising


Module Contents
---------------

.. py:function:: color_group_to_qrac_encode(color_group: dict[int, list[int]]) -> dict[int, qamomile.core.operator.PauliOperator]

   qrac encode

   :param color_group: key is color (qubit's index). value is list of bit's index.
   :type color_group: dict[int, list[int]]

   :returns: key is bit's index. value is tuple of qubit's index and Pauli operator kind.
   :rtype: dict[int, tuple[int, Pauli]]

   .. rubric:: Examples

   >>> color_group = {0: [0, 1, 2], 1: [3, 4], 2: [6,]}
   >>> color_group_for_qrac_encode(color_group)
   {0: (0, <Pauli.Z: 3>), 1: (0, <Pauli.X: 1>), 2: (0, <Pauli.Y: 2>), 3: (1, <Pauli.Z: 3>), 4: (1, <Pauli.X: 1>), 6: (2, <Pauli.Z: 3>)}


.. py:function:: qrac31_encode_ising(ising: qamomile.core.ising_qubo.IsingModel, color_group: dict[int, list[int]]) -> tuple[qamomile.core.operator.Hamiltonian, dict[int, qamomile.core.operator.PauliOperator]]

.. py:class:: QRAC31Converter(compiled_instance, relax_method: jijmodeling_transpiler.core.pubo.RelaxationMethod = jmt.pubo.RelaxationMethod.AugmentedLagrangian)

   Bases: :py:obj:`qamomile.core.converters.converter.QuantumConverter`


   Abstract base class for quantum problem converters in Qamomile.

   This class provides methods for encoding classical optimization problems
   into quantum representations (e.g., Ising models) and decoding quantum
   computation results back into classical problem solutions.

   .. attribute:: compiled_instance

      The compiled instance of the optimization problem.

   .. attribute:: pubo_builder

      The PUBO (Polynomial Unconstrained Binary Optimization) builder.

   .. attribute:: _ising

      Cached Ising model representation.

      :type: Optional[IsingModel]

   .. method:: get_ising

      Retrieve or compute the Ising model representation.

   .. method:: ising_encode

      Encode the problem into an Ising model.

   .. method:: get_cost_hamiltonian

      Abstract method to get the cost Hamiltonian.

   .. method:: decode

      Decode quantum computation results into a SampleSet.

   .. method:: decode_bits_to_sampleset

      Abstract method to convert BitsSampleSet to SampleSet.
      


   .. py:attribute:: max_color_group_size
      :value: 3



   .. py:method:: ising_encode(multipliers: Optional[dict[str, float]] = None, detail_parameters: Optional[dict[str, dict[tuple[int, Ellipsis], tuple[float, float]]]] = None) -> qamomile.core.ising_qubo.IsingModel

      Encode the problem to an Ising model.

      This method converts the problem from QUBO (Quadratic Unconstrained Binary Optimization)
      to Ising model representation.

      :param multipliers: Multipliers for constraint terms.
      :type multipliers: Optional[dict[str, float]]
      :param detail_parameters: Detailed parameters for the encoding process.
      :type detail_parameters: Optional[dict[str, dict[tuple[int, ...], tuple[float, float]]]]

      :returns: The encoded Ising model.
      :rtype: IsingModel



   .. py:method:: get_cost_hamiltonian() -> qamomile.core.operator.Hamiltonian

      Construct the cost Hamiltonian for QRAC31.

      :returns: The cost Hamiltonian.
      :rtype: qm_o.Hamiltonian



   .. py:method:: get_encoded_pauli_list() -> list[qamomile.core.operator.Hamiltonian]


