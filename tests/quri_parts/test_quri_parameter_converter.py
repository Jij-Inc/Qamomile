import pytest
import quri_parts.circuit as qp_c
import qamomile.core.circuit as qm_c
from qamomile.quri_parts.parameter_converter import convert_parameter


def test_convert_simple_parameter():
    qamomile_param = qm_c.Parameter("theta")
    quri_param = qp_c.Parameter("theta")
    param_map = {qamomile_param: quri_param}

    result = convert_parameter(qamomile_param, param_map)

    assert isinstance(result, dict)
    assert len(result) == 1
    assert quri_param in result
    assert result[quri_param] == 1.0


def test_convert_value():
    value = qm_c.Value(2.5)
    param_map = {}

    result = convert_parameter(value, param_map)

    assert isinstance(result, dict)
    assert len(result) == 1
    assert qp_c.CONST in result
    assert result[qp_c.CONST] == 2.5


def test_convert_binary_operator_add():
    param1 = qm_c.Parameter("theta")
    param2 = qm_c.Parameter("phi")
    quri_param1 = qp_c.Parameter("theta")
    quri_param2 = qp_c.Parameter("phi")
    param_map = {param1: quri_param1, param2: quri_param2}

    qamomile_expr = qm_c.BinaryOperator(param1, param2, qm_c.BinaryOpeKind.ADD)
    result = convert_parameter(qamomile_expr, param_map)

    assert isinstance(result, dict)
    assert len(result) == 2
    assert result[quri_param1] == 1.0
    assert result[quri_param2] == 1.0


def test_convert_binary_operator_mul():
    param = qm_c.Parameter("theta")
    value = qm_c.Value(2)
    quri_param = qp_c.Parameter("theta")
    param_map = {param: quri_param}

    qamomile_expr = qm_c.BinaryOperator(param, value, qm_c.BinaryOpeKind.MUL)
    result = convert_parameter(qamomile_expr, param_map)

    assert isinstance(result, dict)
    assert len(result) == 1
    assert result[quri_param] == 2.0


def test_convert_binary_operator_div():
    param = qm_c.Parameter("theta")
    value = qm_c.Value(2)
    quri_param = qp_c.Parameter("theta")
    param_map = {param: quri_param}

    qamomile_expr = param/value
    result = convert_parameter(qamomile_expr, param_map)

    assert isinstance(result, dict)
    assert len(result) == 1
  
    assert result[quri_param] == 0.5


def test_convert_complex_expression():
    theta = qm_c.Parameter("theta")
    phi = qm_c.Parameter("phi")
    quri_theta = qp_c.Parameter("theta")
    quri_phi = qp_c.Parameter("phi")
    param_map = {theta: quri_theta, phi: quri_phi}

    qamomile_expr = qm_c.BinaryOperator(
        qm_c.BinaryOperator(theta, qm_c.Value(2), qm_c.BinaryOpeKind.MUL),
        qm_c.BinaryOperator(phi, qm_c.Value(3), qm_c.BinaryOpeKind.ADD),
        qm_c.BinaryOpeKind.ADD,
    )

    result = convert_parameter(qamomile_expr, param_map)

    assert isinstance(result, dict)
    assert len(result) == 3
    assert result[quri_theta] == 2.0
    assert result[quri_phi] == 1.0
    assert result[qp_c.CONST] == 3.0


def test_unsupported_binary_operation():
    param = qm_c.Parameter("theta")
    quri_param = qp_c.Parameter("theta")
    param_map = {param: quri_param}

    class UnsupportedBinaryOpeKind:
        pass

    unsupported_expr = qm_c.BinaryOperator(param, param, UnsupportedBinaryOpeKind())

    with pytest.raises(ValueError, match="Unsupported binary operation"):
        convert_parameter(unsupported_expr, param_map)


def test_unsupported_parameter_type():
    class UnsupportedParam:
        pass

    unsupported_param = UnsupportedParam()
    param_map = {}

    with pytest.raises(ValueError, match="Unsupported parameter type"):
        convert_parameter(unsupported_param, param_map)


def test_non_linear_parameter_expression():
    param1 = qm_c.Parameter("theta")
    param2 = qm_c.Parameter("phi")
    quri_param1 = qp_c.Parameter("theta")
    quri_param2 = qp_c.Parameter("phi")
    param_map = {param1: quri_param1, param2: quri_param2}

    qamomile_expr = qm_c.BinaryOperator(param1, param2, qm_c.BinaryOpeKind.MUL)

    with pytest.raises(
        ValueError, match="QuriParts does not support non-linear parameter expression"
    ):
        convert_parameter(qamomile_expr, param_map)
