from qamomile.core.circuit import Parameter, BinaryOperator, Value, BinaryOpeKind


def test_parameter_creation():
    param = Parameter("theta")
    assert param.name == "theta"


def test_parameter_equality():
    param1 = Parameter("theta")
    param2 = Parameter("theta")
    param3 = Parameter("phi")
    assert param1 == param2
    assert param1 != param3


def test_parameter_addition():
    param1 = Parameter("theta")
    param2 = Parameter("phi")
    result = param1 + param2
    assert isinstance(result, BinaryOperator)
    assert result.kind == BinaryOpeKind.ADD


def test_parameter_multiplication():
    param = Parameter("theta")
    value = 2
    result = param * value
    assert isinstance(result, BinaryOperator)
    assert result.kind == BinaryOpeKind.MUL


def test_parameter_division():
    param = Parameter("theta")
    value = 2
    result = param / value
    assert isinstance(result, BinaryOperator)
    assert result.kind == BinaryOpeKind.DIV


def test_complex_expression():
    theta = Parameter("theta")
    phi = Parameter("phi")
    expression = (2 * theta + phi) / 3
    assert isinstance(expression, BinaryOperator)
    assert expression.kind == BinaryOpeKind.DIV
    assert isinstance(expression.left, BinaryOperator)
    assert isinstance(expression.right, Value)


def test_get_parameters():
    theta = Parameter("theta")
    phi = Parameter("phi")
    expression = theta + 2 * phi
    params = expression.get_parameters()
    assert len(params) == 2
    assert theta in params and phi in params
