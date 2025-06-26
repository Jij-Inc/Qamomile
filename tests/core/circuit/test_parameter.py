import numpy as np
import pytest

from qamomile.core.circuit.parameter import (
    BinaryOpeKind,
    BinaryOperator,
    Value,
    Parameter,
)
from tests.mock import ParameterExpressionChildMock


# >>> ParameterExpression >>>


def test_parameter_expression_creation():
    """Create a mock ParameterExpressionChildMock instance.

    Check if
    1. No error is raised.
    """
    # 1. No error is raised.
    ParameterExpressionChildMock()


def test_add():
    """Add two of ParameterExpressionChildMock instances.

    Check if
    1. the returned value is BinaryOperator.
    2. the returned value's left is the same as expr1.
    3. the returned value's right is the same as the expr2.
    4. the returned value's kind is BinaryOpeKind.ADD.
    """
    expr1 = ParameterExpressionChildMock()
    expr2 = ParameterExpressionChildMock()

    result = expr1 + expr2

    # 1. the returned value is BinaryOperator.
    assert isinstance(result, BinaryOperator)
    # 2. the returned value's left is the same as expr1.
    assert result.left == expr1
    # 3. the returned value's right is the same as the expr2.
    assert result.right == expr2
    # 4. the returned value's kind is BinaryOpeKind.ADD.
    assert result.kind == BinaryOpeKind.ADD


@pytest.mark.parametrize("other", [1, 1.0, np.int64(1), np.float64(1.0)])
def test_add_with_real(other):
    """Add a ParameterExpressionChildMock instance with a real number.

    Check if
    1. the returned value is BinaryOperator.
    2. the returned value's left is the same as expr1.
    3. the returned value's right is Value.
    4. the returned value's right's value is the same as the given other.
    5. the returned value's kind is BinaryOpeKind.ADD.
    """
    expr = ParameterExpressionChildMock()

    result = expr + other

    # 1. the returned value is BinaryOperator.
    assert isinstance(result, BinaryOperator)
    # 2. the returned value's left is the same as expr1.
    assert result.left == expr
    # 3. the returned value's right is Value.
    assert isinstance(result.right, Value)
    # 4. the returned value's right's value is the same as the given other.
    assert result.right.value == other
    # 5. the returned value's kind is BinaryOpeKind.ADD.
    assert result.kind == BinaryOpeKind.ADD


@pytest.mark.parametrize("other", ["1", [1, 2], 1j])
def test_add_with_unsupported_type(other):
    """Add a ParameterExpressionChildMock instance with an unsupported type.

    Check if
    1. the returned value is NotImplemented.
    """
    expr = ParameterExpressionChildMock()

    assert expr + other == NotImplemented


@pytest.mark.parametrize("other", [1, 1.0, np.int64(1), np.float64(1.0)])
def test_radd_with_real(other):
    """Right add a ParameterExpressionChildMock instance with a real number.

    Check if
    1. the returned value is BinaryOperator.
    2. the returned value's left is expr.
    3. the returned value's right is Value.
    4. the returned value's right's value is the same as the given other.
    5. the returned value's kind is BinaryOpeKind.ADD.
    """
    expr = ParameterExpressionChildMock()

    result = other + expr

    # 1. the returned value is BinaryOperator.
    assert isinstance(result, BinaryOperator)
    # 2. the returned value's left is expr.
    assert result.left == expr
    # 3. the returned value's right is Value.
    assert isinstance(result.right, Value)
    # 4. the returned value's right's value is the same as the given other.
    assert result.right.value == other
    # 5. the returned value's kind is BinaryOpeKind.ADD.
    assert result.kind == BinaryOpeKind.ADD


@pytest.mark.parametrize("other", ["1", [1, 2], 1j])
def test_radd_with_unsupported_type(other):
    """Right add a ParameterExpressionChildMock instance with an unsupported type.

    Check if
    1. the returned value is NotImplemented.
    """
    expr = ParameterExpressionChildMock()

    assert other + expr == NotImplemented


def test_mul():
    """Mul two of ParameterExpressionChildMock instances.

    Check if
    1. the returned value is BinaryOperator.
    2. the returned value's left is the same as expr1.
    3. the returned value's right is the same as the expr2.
    4. the returned value's kind is BinaryOpeKind.MUL.
    """
    expr1 = ParameterExpressionChildMock()
    expr2 = ParameterExpressionChildMock()

    result = expr1 * expr2

    # 1. the returned value is BinaryOperator.
    assert isinstance(result, BinaryOperator)
    # 2. the returned value's left is the same as expr1.
    assert result.left == expr1
    # 3. the returned value's right is the same as the expr2.
    assert result.right == expr2
    # 4. the returned value's kind is BinaryOpeKind.MUL.
    assert result.kind == BinaryOpeKind.MUL


@pytest.mark.parametrize("other", ["1", [1, 2], 1j])
def test_mul_with_unsupported_type(other):
    """Mul a ParameterExpressionChildMock instance with an unsupported type.

    Check if
    1. the returned value is NotImplemented.
    """
    expr = ParameterExpressionChildMock()

    assert expr * other == NotImplemented


@pytest.mark.parametrize("other", [1, 1.0, np.int64(1), np.float64(1.0)])
def test_mul_with_real(other):
    """Mul a ParameterExpressionChildMock instance with a real number.

    Check if
    1. the returned value is BinaryOperator.
    2. the returned value's right is the same as expr1.
    3. the returned value's left is Value.
    4. the returned value's left's value is the same as the given other.
    5. the returned value's kind is BinaryOpeKind.MUL.
    """
    expr = ParameterExpressionChildMock()

    result = expr * other

    # 1. the returned value is BinaryOperator.
    assert isinstance(result, BinaryOperator)
    # 2. the returned value's right is the same as expr1.
    assert result.right == expr
    # 3. the returned value's left is Value.
    assert isinstance(result.left, Value)
    # 4. the returned value's left's value is the same as the given other.
    assert result.left.value == other
    # 5. the returned value's kind is BinaryOpeKind.MUL.
    assert result.kind == BinaryOpeKind.MUL


@pytest.mark.parametrize("other", [1, 1.0, np.int64(1), np.float64(1.0)])
def test_rmul_with_real(other):
    """Right mul a ParameterExpressionChildMock instance with a real number.

    Check if
    1. the returned value is BinaryOperator.
    2. the returned value's right is the same as expr1.
    3. the returned value's left is Value.
    4. the returned value's left's value is the same as the given other.
    5. the returned value's kind is BinaryOpeKind.MUL.
    """
    expr = ParameterExpressionChildMock()

    result = other + expr

    # 1. the returned value is BinaryOperator.
    assert isinstance(result, BinaryOperator)
    # 2. the returned value's right is the same as expr1.
    assert result.right == expr
    # 3. the returned value's left is Value.
    assert isinstance(result.left, Value)
    # 4. the returned value's left's value is the same as the given other.
    assert result.left.value == other
    # 5. the returned value's kind is BinaryOpeKind.MUL.
    assert result.kind == BinaryOpeKind.MUL


@pytest.mark.parametrize("other", ["1", [1, 2], 1j])
def test_rmul_with_unsupported_type(other):
    """Right mul a ParameterExpressionChildMock instance with an unsupported type.

    Check if
    1. the returned value is NotImplemented.
    """
    expr = ParameterExpressionChildMock()

    assert other * expr == NotImplemented


def test_neg():
    """Negate a ParameterExpressionChildMock instance.

    Check if
    1. the returned value is BinaryOperator.
    2. the returned value's left is Value.
    3. the returned value's left's value is -1.0.
    4. the returned value's right is the same as the expr.
    5. the returned value's kind is BinaryOpeKind.MUL.
    """
    expr = ParameterExpressionChildMock()

    result = -expr

    # 1. the returned value is BinaryOperator.
    assert isinstance(result, BinaryOperator)
    # 2. the returned value's left is Value.
    assert isinstance(result.left, Value)
    # 3. the returned value's left's value is -1.0.
    assert result.left.value == -1.0
    # 4. the returned value's right is the same as the expr.
    assert result.right == expr
    # 5. the returned value's kind is BinaryOpeKind.MUL.
    assert result.kind == BinaryOpeKind.MUL


def test_sub():
    """Sub two of ParameterExpressionChildMock instances.

    Check if
    1. the returned value is BinaryOperator.
    2. the returned value's left is the same as expr1.
    3. the returned value's right is BinaryOperator.
    4. the returned value's right's left is Value.
    5. the returned value's right's left's value is -1.0.
    6. the returned value's right's right is the same as the expr2.
    7. the returned value's right's kind is BinaryOpeKind.MUL.
    8. the returned value's kind is BinaryOpeKind.ADD.
    """
    expr1 = ParameterExpressionChildMock()
    expr2 = ParameterExpressionChildMock()

    result = expr1 - expr2

    # 1. the returned value is BinaryOperator.
    assert isinstance(result, BinaryOperator)
    # 2. the returned value's left is the same as expr1.
    assert result.left == expr1
    # 3. the returned value's right is BinaryOperator.
    assert isinstance(result.right, BinaryOperator)
    # 4. the returned value's right's left is Value.
    assert isinstance(result.right.left, Value)
    # 5. the returned value's right's left's value is -1.0.
    assert result.right.left.value == -1.0
    # 6. the returned value's right's right is the same as the expr2.
    assert result.right.right == expr2
    # 7. the returned value's right's kind is BinaryOpeKind.MUL.
    assert result.right.kind == BinaryOpeKind.MUL
    # 8. the returned value's kind is BinaryOpeKind.ADD.
    assert result.kind == BinaryOpeKind.ADD


@pytest.mark.parametrize("other", [1, 1.0, np.int64(1), np.float64(1.0)])
def test_sub_with_real(other):
    """Sub a ParameterExpressionChildMock instance with a real number.

    Check if
    1. the returned value is BinaryOperator.
    2. the resulted value's left is the same as expr.
    3. the returned value's right is Value.
    4. the returned value's right's value is -other.
    5. the returned value's kind is BinaryOpeKind.ADD.
    """
    expr = ParameterExpressionChildMock()

    result = expr - other

    # 1. the returned value is BinaryOperator.
    assert isinstance(result, BinaryOperator)
    # 2. the resulted value's left is the same as expr.
    assert result.left == expr
    # 3. the returned value's right is Value.
    assert isinstance(result.right, Value)
    # 4. the returned value's right's value is -other.
    assert result.right.value == -other
    # 5. the returned value's kind is BinaryOpeKind.ADD.
    assert result.kind == BinaryOpeKind.ADD


@pytest.mark.parametrize("other", ["1", [1, 2], 1j])
def test_sub_with_unsupported_type(other):
    """Sub a ParameterExpressionChildMock instance with an unsupported type.

    Check if
    1. the returned value is NotImplemented.
    """
    expr = ParameterExpressionChildMock()

    assert expr - other == NotImplemented


@pytest.mark.parametrize("other", [1, 1.0, np.int64(1), np.float64(1.0)])
def test_rsub_with_real(other):
    """Right sub a ParameterExpressionChildMock instance with a real number.

    Check if
    1. the returned value is BinaryOperator.
    2. the returned value's left is Value.
    3. the returned value's left's value is other.
    4. the returned value's right is BinaryOperator.
    5. the returned value's right's left is Value.
    6. the returned value's right's left's value is -1.0.
    7. the returned value's right's right is the same as the expr.
    8. the returned value's right's kind is BinaryOpeKind.MUL.
    9. the returned value's kind is BinaryOpeKind.ADD.
    """
    expr = ParameterExpressionChildMock()

    result = other - expr

    # 1. the returned value is BinaryOperator.
    assert isinstance(result, BinaryOperator)
    # 2. the returned value's left is Value.
    assert isinstance(result.left, Value)
    # 3. the returned value's left's value is other.
    assert result.left.value == other
    # 4. the returned value's right is BinaryOperator.
    assert isinstance(result.right, BinaryOperator)
    # 5. the returned value's right's left is Value.
    assert isinstance(result.right.left, Value)
    # 6. the returned value's right's left's value is -1.0.
    assert result.right.left.value == -1.0
    # 7. the returned value's right's right is the same as the expr.
    assert result.right.right == expr
    # 8. the returned value's right's kind is BinaryOpeKind.MUL.
    assert result.right.kind == BinaryOpeKind.MUL
    # 9. the returned value's kind is BinaryOpeKind.ADD.
    assert result.kind == BinaryOpeKind.ADD


@pytest.mark.parametrize("other", ["1", [1, 2], 1j])
def test_rsub_with_unsupported_type(other):
    """Right sub a ParameterExpressionChildMock instance with an unsupported type.

    Check if
    1. the returned value is NotImplemented.
    """
    expr = ParameterExpressionChildMock()

    assert other - expr == NotImplemented


def test_truediv():
    """Div two of ParameterExpressionChildMock instances.

    Check if
    1. the returned value is BinaryOperator.
    2. the returned value's left is the same as expr1.
    3. the returned value's right is the same as the expr2.
    4. the returned value's kind is BinaryOpeKind.DIV.
    """
    expr1 = ParameterExpressionChildMock()
    expr2 = ParameterExpressionChildMock()

    result = expr1 / expr2

    # 1. the returned value is BinaryOperator.
    assert isinstance(result, BinaryOperator)
    # 2. the returned value's left is the same as expr1.
    assert result.left == expr1
    # 3. the returned value's right is the same as the expr2.
    assert result.right == expr2
    # 4. the returned value's kind is BinaryOpeKind.DIV.
    assert result.kind == BinaryOpeKind.DIV


@pytest.mark.parametrize("other", [int(1), float(1.0), np.int64(1), np.float64(1.0)])
def test_truediv_with_nonzero_real(other):
    """Div two of ParameterExpressionChildMock instances.

    Check if
    1. the returned value is BinaryOperator.
    2. the returned value's left is the same as expr.
    3. the returned value's right is Value.
    4. the returned value's right's value is the same as the given other.
    5. the returned value's kind is BinaryOpeKind.DIV.
    """
    expr = ParameterExpressionChildMock()

    result = expr / other

    # 1. the returned value is BinaryOperator.
    assert isinstance(result, BinaryOperator)
    # 2. the returned value's left is the same as expr.
    assert result.left == expr
    # 3. the returned value's right is Value.
    assert isinstance(result.right, Value)
    # 4. the returned value's right's value is the same as the given other.
    assert result.right.value == other
    # 5. the returned value's kind is BinaryOpeKind.DIV.
    assert result.kind == BinaryOpeKind.DIV


def test_truediv_with_zero():
    """Div a ParameterExpressionChildMock instance with zero.

    Check if
    1. ZeroDivisionError is raised.
    """
    expr = ParameterExpressionChildMock()

    with pytest.raises(ZeroDivisionError):
        expr / 0


@pytest.mark.parametrize("other", ["1", [1, 2], 1j])
def test_truedix_with_unsupported_type(other):
    """Div a ParameterExpressionChildMock instance with an unsupported type.

    Check if
    1. the returned value is NotImplemented.
    """
    expr = ParameterExpressionChildMock()

    assert other / expr == NotImplemented


@pytest.mark.parametrize("other", [int(1), float(1.0), np.int64(1), np.float64(1.0)])
def test_rtruediv_with_nonzero_real(other):
    """Right div a ParameterExpressionChildMock instance with a real number.

    Check if
    1. the returned value is BinaryOperator.
    2. the returned value's left is Value.
    3. the returned value's left's value is the same as the given other.
    4. the returned value's right is expr.
    5. the returned value's kind is BinaryOpeKind.DIV.
    """
    expr = ParameterExpressionChildMock()

    result = other / expr

    # 1. the returned value is BinaryOperator.
    assert isinstance(result, BinaryOperator)
    # 2. the returned value's left is Value.
    assert isinstance(result.left, Value)
    # 3. the returned value's left's value is the same as the given other.
    assert result.left.value == other
    # 4. the returned value's right is expr.
    assert result.right == expr
    # 5. the returned value's kind is BinaryOpeKind.DIV.
    assert result.kind == BinaryOpeKind.DIV


def test_get_parameters():
    """Get parameters from a ParameterExpressionChildMock instance.

    Check if
    1. the returned value is an empty list.
    """
    expr = ParameterExpressionChildMock()

    result = expr.get_parameters()

    # 1. the returned value is an empty list.
    assert result == []


# <<< ParameterExpression <<<
# >>> Parameter >>>


@pytest.mark.parametrize("name", ["theta", "PhI"])
def test_parameter_creation(name):
    """Create a Parameter instance.

    1. Check if the name is the same as the given name.
    """
    param = Parameter(name)
    # 1. Check if the name is the same as the given name.
    assert param.name == name


@pytest.mark.parametrize("name", ["theta", "PhI"])
def test_repr(name):
    """Run __repr__ on a Parameter instance.

    Check if
    1. the returned value is the same as given name.
    """
    parameter = Parameter(name)
    # 1. the returned value is the same as given name.
    assert repr(parameter) == name


@pytest.mark.parametrize("name", ["theta", "PhI"])
def test_parameter_get_parameters(name):
    """Call get_parameters on a Parameter instance.

    Check if
    1. the returned value is list.
    2. the length of the returned list is 1.
    3. the first element of the returned list is the same as the parameter instance.
    """
    param = Parameter(name)

    gotten_parameters = param.get_parameters()

    # 1. the returned value is list.
    assert isinstance(gotten_parameters, list)
    # 2. the length of the returned list is 1.
    assert len(gotten_parameters) == 1
    # 3. the first element of the returned list is the same as the parameter instance.
    assert gotten_parameters[0] == param


@pytest.mark.parametrize("name", ["theta", "PhI"])
def test_hash(name):
    """Parameter instance can be a key of a dictionary.

    check if
    1. no error is raised.
    """
    param = Parameter(name)

    # 1. no error is raised.
    {param: "THIS IS DICT!"}


@pytest.mark.parametrize("name", ["theta", "PhI"])
def test_eq(name):
    """Check equality of two Parameter instances.

    Check if
    1. the returned value is True if the names of different two Parameters are the same.
    2. the returned value is False if the names of different two Parameters are different.
    3. the returned value is False if the other object is just a string with the same name.
    """
    param1 = Parameter(name)
    param2 = Parameter(name)

    # 1. the returned value is True if the names of different two Parameters are the same.
    assert param1 == param2

    param3 = Parameter(f"{name}_")
    # 2. the returned value is False if the names of different two Parameters are different.
    assert param1 != param3
    # 3. the returned value is False if the other object is just a string with the same name.
    assert param1 != name


# <<< Parameter <<<
