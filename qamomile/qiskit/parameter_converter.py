"""
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
"""

import qiskit
import qiskit.circuit
from qamomile.core.circuit import (
    Parameter,
    BinaryOperator,
    Value,
    BinaryOpeKind,
    ParameterExpression,
)
from .exceptions import QamomileQiskitTranspileError


def convert_parameter(
    param: ParameterExpression, parameters: dict[Parameter, qiskit.circuit.Parameter]
) -> qiskit.circuit.ParameterExpression:
    """
    Convert a Qamomile parameter expression to a Qiskit parameter expression.

    This function recursively traverses the Qamomile parameter expression and
    converts it to an equivalent Qiskit parameter expression. It handles
    Parameters, Values, and BinaryOperators.

    Args:
        param (ParameterExpression): The Qamomile parameter expression to convert.
        parameters (dict[Parameter, qiskit.circuit.Parameter]): A mapping of Qamomile
            Parameters to their corresponding Qiskit Parameters.

    Returns:
        qiskit.circuit.ParameterExpression: The equivalent Qiskit parameter expression.

    Raises:
        QamomileQiskitTranspileError: If an unsupported parameter type or binary
            operation is encountered.

    Examples:
        >>> qamomile_param = Parameter('theta')
        >>> qiskit_param = qiskit.circuit.Parameter('theta')
        >>> param_map = {qamomile_param: qiskit_param}
        >>> result = convert_parameter(qamomile_param, param_map)
        >>> isinstance(result, qiskit.circuit.Parameter)
        True
    """
    if isinstance(param, Parameter):
        # Direct parameter conversion
        return parameters[param]
    elif isinstance(param, Value):
        # Convert constant values
        return param.value
    elif isinstance(param, BinaryOperator):
        # Recursively convert binary operations
        left = convert_parameter(param.left, parameters)
        right = convert_parameter(param.right, parameters)
        if param.kind == BinaryOpeKind.ADD:
            return left + right
        elif param.kind == BinaryOpeKind.MUL:
            return left * right
        elif param.kind == BinaryOpeKind.DIV:
            return left / right
        else:
            raise QamomileQiskitTranspileError(
                f"Unsupported binary operation: {param.kind}"
            )
    else:
        raise QamomileQiskitTranspileError(f"Unsupported parameter type: {type(param)}")
