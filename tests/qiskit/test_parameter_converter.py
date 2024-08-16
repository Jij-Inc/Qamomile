# File: tests/qiskit/test_parameter_converter.py

import pytest
import qiskit
from qamomile.core.circuit import Parameter, BinaryOperator, Value, BinaryOpeKind
from qamomile.qiskit.parameter_converter import convert_parameter
from qamomile.qiskit.exceptions import QamomileQiskitTranspileError


def test_convert_simple_parameter():
    qamomile_param = Parameter("theta")
    qiskit_param = qiskit.circuit.Parameter("theta")
    param_map = {qamomile_param: qiskit_param}

    result = convert_parameter(qamomile_param, param_map)

    assert isinstance(result, qiskit.circuit.Parameter)
    assert result == qiskit_param


def test_convert_value():
    value = Value(2.5)
    param_map = {}

    result = convert_parameter(value, param_map)

    assert isinstance(result, float)
    assert result == 2.5


def test_convert_binary_operator_add():
    param1 = Parameter("theta")
    param2 = Parameter("phi")
    qiskit_param1 = qiskit.circuit.Parameter("theta")
    qiskit_param2 = qiskit.circuit.Parameter("phi")
    param_map = {param1: qiskit_param1, param2: qiskit_param2}

    qamomile_expr = BinaryOperator(param1, param2, BinaryOpeKind.ADD)
    result = convert_parameter(qamomile_expr, param_map)

    assert isinstance(result, qiskit.circuit.ParameterExpression)
    assert len(result.parameters) == 2  # Check binary operator


def test_convert_binary_operator_mul():
    param = Parameter("theta")
    value = Value(2)
    qiskit_param = qiskit.circuit.Parameter("theta")
    param_map = {param: qiskit_param}

    qamomile_expr = BinaryOperator(param, value, BinaryOpeKind.MUL)
    result = convert_parameter(qamomile_expr, param_map)

    assert isinstance(result, qiskit.circuit.ParameterExpression)
    assert str(result) == "2*theta"


def test_convert_binary_operator_div():
    param = Parameter("theta")
    value = Value(2)
    qiskit_param = qiskit.circuit.Parameter("theta")
    param_map = {param: qiskit_param}

    qamomile_expr = BinaryOperator(param, value, BinaryOpeKind.DIV)
    result = convert_parameter(qamomile_expr, param_map)

    assert isinstance(result, qiskit.circuit.ParameterExpression)
    assert str(result) == "theta/2"


def test_convert_complex_expression():
    theta = Parameter("theta")
    phi = Parameter("phi")
    qiskit_theta = qiskit.circuit.Parameter("theta")
    qiskit_phi = qiskit.circuit.Parameter("phi")
    param_map = {theta: qiskit_theta, phi: qiskit_phi}

    qamomile_expr = BinaryOperator(
        BinaryOperator(theta, Value(2), BinaryOpeKind.MUL),
        BinaryOperator(phi, Value(3), BinaryOpeKind.ADD),
        BinaryOpeKind.DIV,
    )

    result = convert_parameter(qamomile_expr, param_map)

    assert isinstance(result, qiskit.circuit.ParameterExpression)
    assert len(result.parameters) == 2  # Check binary operator


def test_unsupported_binary_operation():
    param = Parameter("theta")
    qiskit_param = qiskit.circuit.Parameter("theta")
    param_map = {param: qiskit_param}

    class UnsupportedBinaryOpeKind:
        pass

    unsupported_expr = BinaryOperator(param, param, UnsupportedBinaryOpeKind())

    with pytest.raises(
        QamomileQiskitTranspileError, match="Unsupported binary operation"
    ):
        convert_parameter(unsupported_expr, param_map)


def test_unsupported_parameter_type():
    class UnsupportedParam:
        pass

    unsupported_param = UnsupportedParam()
    param_map = {}

    with pytest.raises(
        QamomileQiskitTranspileError, match="Unsupported parameter type"
    ):
        convert_parameter(unsupported_param, param_map)
