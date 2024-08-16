quri_parts.parameter_converter
==============================

.. py:module:: quri_parts.parameter_converter


Functions
---------

.. autoapisummary::

   quri_parts.parameter_converter.convert_parameter


Module Contents
---------------

.. py:function:: convert_parameter(param: qamomile.core.circuit.ParameterExpression, parameters: dict[qamomile.core.circuit.Parameter, quri_parts.circuit.Parameter]) -> dict[quri_parts.circuit.Parameter, float]

   Convert a Qamomile parameter expression to a QuriParts parameter expression.

   This function recursively traverses the Qamomile parameter expression and
   converts it to an equivalent QuriParts parameter expression. It handles
   Parameters, Values, and BinaryOperators.

   :param param: The Qamomile parameter expression to convert.
   :type param: qm_c.ParameterExpression
   :param parameters: A mapping of Qamomile
                      Parameters to their corresponding QuriParts Parameters.
   :type parameters: dict[qm_c.Parameter, qp_c.Parameter]

   :returns: The equivalent QuriParts parameter expression.
   :rtype: dict[qp_c.Parameter, float]

   .. rubric:: Examples

   >>> qamomile_param = qm_c.Parameter('theta')
   >>> quri_param = qp_c.Parameter('theta')
   >>> param_map = {qamomile_param: quri_param}
   >>> result = convert_parameter(qamomile_param, param_map)
   >>> isinstance(result, dict)
   True


