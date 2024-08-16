qiskit.parameter_converter
==========================

.. py:module:: qiskit.parameter_converter

.. autoapi-nested-parse::

   Qamomile to Qiskit Parameter Converter

   This module provides functionality to convert Qamomile parameter expressions
   to Qiskit parameter expressions. It is essential for transpiling Qamomile
   circuits to Qiskit circuits, ensuring that parameterized gates are correctly
   translated between the two frameworks.

   The main function, convert_parameter, recursively traverses Qamomile parameter
   expressions and converts them to equivalent Qiskit expressions.

   Usage:
       from qamomile.qiskit.parameter_converter import convert_parameter
       qiskit_param = convert_parameter(qamomile_param, parameter_map)

   Note: This module requires both Qamomile and Qiskit to be installed.



Functions
---------

.. autoapisummary::

   qiskit.parameter_converter.convert_parameter


Module Contents
---------------

.. py:function:: convert_parameter(param: qamomile.core.circuit.ParameterExpression, parameters: dict[qamomile.core.circuit.Parameter, qiskit.circuit.Parameter]) -> qiskit.circuit.ParameterExpression

   Convert a Qamomile parameter expression to a Qiskit parameter expression.

   This function recursively traverses the Qamomile parameter expression and
   converts it to an equivalent Qiskit parameter expression. It handles
   Parameters, Values, and BinaryOperators.

   :param param: The Qamomile parameter expression to convert.
   :type param: ParameterExpression
   :param parameters: A mapping of Qamomile
                      Parameters to their corresponding Qiskit Parameters.
   :type parameters: dict[Parameter, qiskit.circuit.Parameter]

   :returns: The equivalent Qiskit parameter expression.
   :rtype: qiskit.circuit.ParameterExpression

   :raises QamomileQiskitTranspileError: If an unsupported parameter type or binary
       operation is encountered.

   .. rubric:: Examples

   >>> qamomile_param = Parameter('theta')
   >>> qiskit_param = qiskit.circuit.Parameter('theta')
   >>> param_map = {qamomile_param: qiskit_param}
   >>> result = convert_parameter(qamomile_param, param_map)
   >>> isinstance(result, qiskit.circuit.Parameter)
   True


