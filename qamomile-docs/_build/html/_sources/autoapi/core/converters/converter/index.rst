core.converters.converter
=========================

.. py:module:: core.converters.converter

.. autoapi-nested-parse::

   qamomile/core/converter.py

   This module defines the QuantumConverter abstract base class for converting between
   different problem representations and quantum models in Qamomile.

   The QuantumConverter class provides a framework for encoding classical optimization
   problems into quantum representations (e.g., Ising models) and decoding quantum
   computation results back into classical problem solutions.

   Key Features:
   - Conversion between classical optimization problems and Ising models.
   - Abstract methods for generating cost Hamiltonians and decoding results.
   - Integration with QuantumSDKTranspiler for SDK-specific result handling.

   Usage:
   Developers implementing specific quantum conversion strategies should subclass
   QuantumConverter and implement the abstract methods. The class is designed to work
   with jijmodeling for problem representation and various quantum SDKs through
   the QuantumSDKTranspiler interface.

   .. rubric:: Example

   class QAOAConverter(QuantumConverter):
       def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
           # Implementation for generating QAOA cost Hamiltonian
           ...



Attributes
----------

.. autoapisummary::

   core.converters.converter.ResultType


Classes
-------

.. autoapisummary::

   core.converters.converter.QuantumConverter


Functions
---------

.. autoapisummary::

   core.converters.converter.decode_from_dict_binary_result


Module Contents
---------------

.. py:data:: ResultType

.. py:class:: QuantumConverter(compiled_instance, relax_method: jijmodeling_transpiler.core.pubo.RelaxationMethod = jmt.pubo.RelaxationMethod.AugmentedLagrangian)

   Bases: :py:obj:`abc.ABC`


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
      


   .. py:attribute:: pubo_builder


   .. py:attribute:: compiled_instance


   .. py:method:: get_ising() -> qamomile.core.ising_qubo.IsingModel

      Get the Ising model representation of the problem.

      :returns: The Ising model representation.
      :rtype: IsingModel



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
      :abstractmethod:


      Abstract method to get the cost Hamiltonian for the quantum problem.

      This method should be implemented in subclasses to define how the
      cost Hamiltonian is constructed for specific quantum algorithms.

      :returns: The cost Hamiltonian for the quantum problem.
      :rtype: qm_o.Hamiltonian

      :raises NotImplementedError: If the method is not implemented in the subclass.



   .. py:method:: decode(transpiler: qamomile.core.transpiler.QuantumSDKTranspiler[ResultType], result: ResultType) -> jijmodeling.experimental.SampleSet

      Decode quantum computation results into a SampleSet.

      This method uses the provided transpiler to convert SDK-specific results
      into a BitsSampleSet, then calls decode_bits_to_sampleset to produce
      the final SampleSet.

      :param transpiler: The transpiler for the specific quantum SDK.
      :type transpiler: QuantumSDKTranspiler[ResultType]
      :param result: The raw result from the quantum computation.
      :type result: ResultType

      :returns: The decoded results as a SampleSet.
      :rtype: jm.experimental.SampleSet



   .. py:method:: decode_bits_to_sampleset(bitssampleset: qamomile.core.bitssample.BitsSampleSet) -> jijmodeling.experimental.SampleSet

      Decode a BitArraySet to a SampleSet.

      This method converts the quantum computation results (bitstrings)
      into a format that represents solutions to the original optimization problem.

      :param bitarray_set: The set of bitstring results from quantum computation.
      :type bitarray_set: qm_c.BitArraySet

      :returns: The decoded results as a SampleSet.
      :rtype: jm.experimental.SampleSet



.. py:function:: decode_from_dict_binary_result(samples: Iterable[dict[int, int | float]], binary_encoder, compiled_model: jijmodeling_transpiler.core.CompiledInstance) -> jijmodeling.SampleSet

   Decode binary results into a SampleSet.

   :param samples: Iterable of sample dictionaries.
   :param binary_encoder: Binary encoder from jijmodeling_transpiler.
   :param compiled_model: Compiled instance of the optimization problem.

   :returns: Decoded sample set.
   :rtype: jm.SampleSet


