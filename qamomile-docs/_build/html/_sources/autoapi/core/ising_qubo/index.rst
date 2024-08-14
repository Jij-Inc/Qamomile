core.ising_qubo
===============

.. py:module:: core.ising_qubo


Classes
-------

.. autoapisummary::

   core.ising_qubo.IsingModel


Functions
---------

.. autoapisummary::

   core.ising_qubo.calc_qubo_energy
   core.ising_qubo.qubo_to_ising


Module Contents
---------------

.. py:class:: IsingModel

   .. py:attribute:: quad
      :type:  dict[tuple[int, int], float]


   .. py:attribute:: linear
      :type:  dict[int, float]


   .. py:attribute:: constant
      :type:  float


   .. py:attribute:: index_map
      :type:  Optional[dict[int, int]]
      :value: None



   .. py:method:: num_bits() -> int


   .. py:method:: calc_energy(state: list[int]) -> float

      Calculates the energy of the state.

      .. rubric:: Examples

      >>> ising = IsingModel({(0, 1): 2.0}, {0: 4.0, 1: 5.0}, 6.0)
      >>> ising.calc_energy([1, -1])
      3.0



   .. py:method:: ising2qubo_index(index: int) -> int


.. py:function:: calc_qubo_energy(qubo: dict[tuple[int, int], float], state: list[int]) -> float

   Calculates the energy of the state.

   .. rubric:: Examples

   >>> calc_qubo_energy({(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0}, [1, 1])
   6.0


.. py:function:: qubo_to_ising(qubo: dict[tuple[int, int], float], constant: float = 0.0, simplify=True) -> IsingModel

   Converts a Quadratic Unconstrained Binary Optimization (QUBO) problem to an equivalent Ising model.

   QUBO:
       $$\sum_{ij} Q_{ij} x_i x_j,~x_i \in \{0, 1\}$$

   Ising:
       $$\sum_{ij} J_{ij} z_i z_j + \sum_i h_i z_i, ~z_i \in \{-1, 1\}$$

   Correspondence:
       $$x_i = \frac{1 - z_i}{2}$$
       where \(x_i \in \{0, 1\}\) and \(z_i \in \{-1, 1\}\).

   This transformation is derived from the conventions used to describe the eigenstates and eigenvalues of the Pauli Z operator in quantum computing. Specifically, the eigenstates |0⟩ and |1⟩ of the Pauli Z operator correspond to the eigenvalues +1 and -1, respectively:

       $$Z|0⟩ = |0⟩, \quad Z|1⟩ = -|1⟩$$

   This relationship is leveraged to map the binary variables \(x_i\) in QUBO to the spin variables \(z_i\) in the Ising model.

   .. rubric:: Examples

   >>> qubo = {(0, 0): 1.0, (0, 1): 2.0, (1, 1): 3.0}
   >>> ising = qubo_to_ising(qubo)
   >>> binary = [1, 0]
   >>> spin = [-1, 1]
   >>> qubo_energy = calc_qubo_energy(qubo, binary)
   >>> assert qubo_energy == ising.calc_energy(spin)

   >>> qubo = {(0, 1): 2, (0, 0): -1, (1, 1): -1}
   >>> ising = qubo_to_ising(qubo)
   >>> assert ising.constant == -0.5
   >>> assert ising.linear == {}
   >>> assert ising.quad == {(0, 1): 0.5}


