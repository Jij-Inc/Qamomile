core.circuit.parameter
======================

.. py:module:: core.circuit.parameter

.. autoapi-nested-parse::

   Quantum Circuit Parameter Expressions Module

   This module provides a framework for defining and manipulating parameter expressions
   in quantum circuits. It allows for the creation of complex mathematical expressions
   involving named parameters, constant values, and binary operations.

   Key components:
   - ParameterExpression: Abstract base class for all expressions
   - Parameter: Represents a named parameter in a quantum circuit
   - Value: Represents a constant numeric value
   - BinaryOperator: Represents operations between two expressions
   - BinaryOpeKind: Enumeration of supported binary operations

   This module is essential for building parameterized quantum circuits, enabling
   the definition of circuits with variable parameters that can be optimized or
   swept over during execution or simulation.



Classes
-------

.. autoapisummary::

   core.circuit.parameter.ParameterExpression
   core.circuit.parameter.Parameter
   core.circuit.parameter.Value
   core.circuit.parameter.BinaryOpeKind
   core.circuit.parameter.BinaryOperator


Module Contents
---------------

.. py:class:: ParameterExpression

   Bases: :py:obj:`abc.ABC`


   Abstract base class for parameter expressions in quantum circuits.

   This class defines the basic operations (addition, multiplication, division)
   that can be performed on parameter expressions.


   .. py:method:: get_parameters() -> list[Parameter]

      Get the parameters in the expression.

      :returns: The parameters in the expression.
      :rtype: list[Parameter]



.. py:class:: Parameter(name)

   Bases: :py:obj:`ParameterExpression`


   Represents a named parameter in a quantum circuit.


   .. py:attribute:: name


   .. py:method:: get_parameters() -> list[Parameter]

      Return this parameter in a list.



.. py:class:: Value(value)

   Bases: :py:obj:`ParameterExpression`


   Represents a constant numeric value in an expression.


   .. py:attribute:: value


.. py:class:: BinaryOpeKind(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   Enumeration of binary operation types.


   .. py:attribute:: ADD
      :value: '+'



   .. py:attribute:: MUL
      :value: '*'



   .. py:attribute:: DIV
      :value: '/'



.. py:class:: BinaryOperator(left, right, kind)

   Bases: :py:obj:`ParameterExpression`


   Represents a binary operation between two ParameterExpressions.


   .. py:attribute:: left
      :type:  ParameterExpression


   .. py:attribute:: right
      :type:  ParameterExpression


   .. py:attribute:: kind
      :type:  BinaryOpeKind


   .. py:method:: get_parameters() -> list[Parameter]

      Get all parameters involved in this binary operation.

      :returns: A list of all parameters in the expression.
      :rtype: list[Parameter]



