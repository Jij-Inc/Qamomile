"""
Qamomile to CUDA-Q Parameter Converter

This module provides functionality to convert Qamomile parameter expressions
to CUDA-Q parameter expressions. It is essential for transpiling Qamomile
circuits to CUDA-Q kernels, ensuring that parameterized gates are correctly
translated between the two frameworks.

The main function, convert_parameter, recursively traverses Qamomile parameter
expressions and converts them to equivalent CUDA-Q expressions.

Usage:
    from qamomile.cudaq.parameter_converter import convert_parameter
    cudaq_param = convert_parameter(qamomile_param, parameter_map)

Note: This module requires both Qamomile and CUDA-Q to be installed.
"""

import cudaq

from qamomile.core.circuit import (
    Parameter,
    BinaryOperator,
    Value,
    BinaryOpeKind,
    ParameterExpression,
)
from .exceptions import QamomileCudaqTranspileError


def convert_parameter(
    param: ParameterExpression, parameters: dict[Parameter, cudaq.QuakeValue]
) -> cudaq.QuakeValue:
    """
    Convert a Qamomile parameter expression to a CUDA-Q parameter expression.

    This function recursively traverses the Qamomile parameter expression and
    converts it to an equivalent CUDA-Q parameter expression. It handles
    Parameters, Values, and BinaryOperators.

    Args:
        param (ParameterExpression): The Qamomile parameter expression to convert.
        parameters (dict[Parameter, cudaq.QuakeValue]): A mapping of Qamomile
            Parameters to their corresponding CUDA-Q Parameters.

    Returns:
        cudaq.QuakeValue: The equivalent CUDA-Q parameter expression.

    Raises:
        QamomileCudaqTranspileError: If an unsupported parameter type or binary
            operation is encountered.

    Examples:
        >>> qamomile_param = Parameter('theta')
        >>> cudaq_param = cudaq.QuakeValue("theta", cudaq.make_kernel())
        >>> param_map = {qamomile_param: cudaq_param}
        >>> result = convert_parameter(qamomile_param, param_map)
        >>> isinstance(result, cudaq.QuakeValue)
        True
    """
    if isinstance(param, Parameter):
        # Direct parameter conversion
        return parameters[param]
    elif isinstance(param, Value):
        # Convert constant values
        return float(param.value)
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
            raise QamomileCudaqTranspileError(
                f"Unsupported binary operation: {param.kind}"
            )
    else:
        raise QamomileCudaqTranspileError(f"Unsupported parameter type: {type(param)}")
