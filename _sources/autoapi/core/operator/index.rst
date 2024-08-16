core.operator
=============

.. py:module:: core.operator

.. autoapi-nested-parse::

   This module provides the intermediate representation of Hamiltonian for quantum systems.

   It defines classes and functions to create and manipulate Pauli operators and Hamiltonians,
   which are fundamental in quantum mechanics and quantum computing.

   Key Components:
   - Pauli: An enumeration of Pauli operators (X, Y, Z).
   - PauliOperator: A class representing a single Pauli operator acting on a specific qubit.
   - Hamiltonian: A class representing a quantum Hamiltonian as a sum of Pauli operator products.

   Usage:
       from qamomile.operator.hamiltonian import X, Y, Z, Hamiltonian

       # Create Pauli operators
       X0 = X(0)  # Pauli X operator on qubit 0
       Y1 = Y(1)  # Pauli Y operator on qubit 1

       # Create a Hamiltonian
       H = Hamiltonian()
       H.add_term((X0, Y1), 0.5)  # Add term 0.5 * X0 * Y1 to the Hamiltonian
       H.add_term((Z(2),), 1.0)   # Add term 1.0 * Z2 to the Hamiltonian

       # Access Hamiltonian properties
       print(H.terms)
       print(H.num_qubits)



Classes
-------

.. autoapisummary::

   core.operator.Pauli
   core.operator.PauliOperator
   core.operator.Hamiltonian


Functions
---------

.. autoapisummary::

   core.operator.X
   core.operator.Y
   core.operator.Z


Module Contents
---------------

.. py:class:: Pauli(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   Enum class for Pauli operators.

   .. attribute:: X

      Pauli X operator, represented by 0.

      :type: int

   .. attribute:: Y

      Pauli Y operator, represented by 1.

      :type: int

   .. attribute:: Z

      Pauli Z operator, represented by 2.

      :type: int


   .. py:attribute:: X
      :value: 0



   .. py:attribute:: Y
      :value: 1



   .. py:attribute:: Z
      :value: 2



.. py:class:: PauliOperator

   Represents a single Pauli operator acting on a specific qubit.

   .. attribute:: pauli

      The type of Pauli operator (X, Y, or Z).

      :type: Pauli

   .. attribute:: index

      The index of the qubit on which this operator acts.

      :type: int

   .. rubric:: Example

   >>> X0 = PauliOperator(Pauli.X, 0)
   >>> print(X0)
   X0


   .. py:attribute:: pauli
      :type:  Pauli


   .. py:attribute:: index
      :type:  int


.. py:function:: X(index: int) -> PauliOperator

   Creates a Pauli X operator for a specified qubit.

   :param index: The index of the qubit.
   :type index: int

   :returns: A Pauli X operator acting on the specified qubit.
   :rtype: PauliOperator

   .. rubric:: Example

   >>> X0 = X(0)
   >>> print(X0)
   X0


.. py:function:: Y(index: int) -> PauliOperator

   Creates a Pauli Y operator for a specified qubit.

   :param index: The index of the qubit.
   :type index: int

   :returns: A Pauli Y operator acting on the specified qubit.
   :rtype: PauliOperator

   .. rubric:: Example

   >>> Y1 = Y(1)
   >>> print(Y1)
   Y1


.. py:function:: Z(index: int) -> PauliOperator

   Creates a Pauli Z operator for a specified qubit.

   :param index: The index of the qubit.
   :type index: int

   :returns: A Pauli Z operator acting on the specified qubit.
   :rtype: PauliOperator

   .. rubric:: Example

   >>> Z2 = Z(2)
   >>> print(Z2)
   Z2


.. py:class:: Hamiltonian(num_qubits: Optional[int] = None)

   Represents a quantum Hamiltonian as a sum of Pauli operator products.

   The Hamiltonian is stored as a dictionary where keys are tuples of PauliOperators
   and values are their corresponding coefficients.

   .. attribute:: _terms

      The terms of the Hamiltonian.

      :type: Dict[Tuple[PauliOperator, ...], complex]

   .. attribute:: constant

      A constant term added to the Hamiltonian.

      :type: float

   .. rubric:: Example

   >>> H = Hamiltonian()
   >>> H.add_term((X(0), Y(1)), 0.5)
   >>> H.add_term((Z(2),), 1.0)
   >>> print(H.terms)
   {(X0, Y1): 0.5, (Z2,): 1.0}


   .. py:attribute:: constant
      :type:  float
      :value: 0.0



   .. py:property:: terms
      :type: Dict[Tuple[PauliOperator, Ellipsis], complex]

      Getter for the terms of the Hamiltonian.

      :returns: A dictionary representing the Hamiltonian terms.
      :rtype: Dict[Tuple[PauliOperator, ...], complex]

      .. rubric:: Example

      >>> H = Hamiltonian()
      >>> H.add_term((X(0), Y(1)), 0.5)
      >>> print(H.terms)
      {(X0, Y1): 0.5}


   .. py:method:: add_term(operators: Tuple[PauliOperator, Ellipsis], coeff: Union[float, complex])

      Adds a term to the Hamiltonian.

      This method adds a product of Pauli operators with a given coefficient to the Hamiltonian.
      If the term already exists, the coefficients are summed.

      :param operators: A tuple of PauliOperators representing the term.
      :type operators: Tuple[PauliOperator, ...]
      :param coeff: The coefficient of the term.
      :type coeff: Union[float, complex]

      .. rubric:: Example

      >>> H = Hamiltonian()
      >>> H.add_term((X(0), Y(1)), 0.5)
      >>> H.add_term((X(0), Y(1)), 0.5j)
      >>> print(H.terms)
      {(X0, Y1): (0.5+0.5j)}



   .. py:property:: num_qubits
      :type: int

      Calculates the number of qubits in the Hamiltonian.

      :returns: The number of qubits, which is the highest qubit index plus one.
      :rtype: int

      .. rubric:: Example

      >>> H = Hamiltonian()
      >>> H.add_term((X(0), Y(3)), 1.0)
      >>> print(H.num_qubits)
      4


