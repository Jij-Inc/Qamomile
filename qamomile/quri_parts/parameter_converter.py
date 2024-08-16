import quri_parts.circuit as qp_c
import qamomile.core.circuit as qm_c


def convert_parameter(
    param: qm_c.ParameterExpression,
    parameters: dict[qm_c.Parameter, qp_c.Parameter]
) -> dict[qp_c.Parameter, float]:
    """
    Convert a Qamomile parameter expression to a QuriParts parameter expression.

    This function recursively traverses the Qamomile parameter expression and
    converts it to an equivalent QuriParts parameter expression. It handles
    Parameters, Values, and BinaryOperators.

    Args:
        param (qm_c.ParameterExpression): The Qamomile parameter expression to convert.
        parameters (dict[qm_c.Parameter, qp_c.Parameter]): A mapping of Qamomile
            Parameters to their corresponding QuriParts Parameters.

    Returns:
        dict[qp_c.Parameter, float]: The equivalent QuriParts parameter expression.

    Examples:
        >>> qamomile_param = qm_c.Parameter('theta')
        >>> quri_param = qp_c.Parameter('theta')
        >>> param_map = {qamomile_param: quri_param}
        >>> result = convert_parameter(qamomile_param, param_map)
        >>> isinstance(result, dict)
        True
    """
    if isinstance(param, qm_c.Parameter):
        # Direct parameter conversion
        return {parameters[param]: 1.0}
    elif isinstance(param, qm_c.Value):
        # Convert constant values
        return {qp_c.CONST: param.value}
    elif isinstance(param, qm_c.BinaryOperator):
        # Convert binary operations
        left = convert_parameter(param.left, parameters)
        left_const = left.pop(qp_c.CONST, 0.0)
        right = convert_parameter(param.right, parameters)
        right_const = right.pop(qp_c.CONST, 0.0)

        match param.kind:
            case qm_c.BinaryOpeKind.ADD:
                for r_param, r_value in right.items():
                    left[r_param] = left.get(r_param, 0.0) + r_value
                left[qp_c.CONST] = left_const + right_const
                if left[qp_c.CONST] == 0:
                    left.pop(qp_c.CONST)
                return left
            case qm_c.BinaryOpeKind.MUL:
                if len(left) > 0 and len(right) > 0:
                    raise ValueError("QuriParts does not support non-linear parameter expression.")
                if len(left) > 0:
                    param_expr = {param: value * right_const for param, value in left.items()}
                else:
                    param_expr = {param: value * left_const for param, value in right.items()}
                param_expr[qp_c.CONST] = left_const * right_const
                if param_expr[qp_c.CONST] == 0:
                    param_expr.pop(qp_c.CONST)
                return param_expr
            case qm_c.BinaryOpeKind.DIV:
                if len(right) > 0:
                    raise ValueError("QuriParts does not support non-linear parameter expression.")
                param_expr = {param: value / right_const for param, value in left.items()}
                param_expr[qp_c.CONST] = left_const / right_const
                if param_expr[qp_c.CONST] == 0:
                    param_expr.pop(qp_c.CONST)
                return param_expr
            case _:
                raise ValueError(f"Unsupported binary operation: {param.kind}")
    else:
        raise ValueError(f"Unsupported parameter type: {type(param)}")
