import numpy as np
import pytest

from qamomile.core.circuit.parameter import BinaryOpeKind, BinaryOperator, Value
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
    1. TypeError is raised.
    """
    expr = ParameterExpressionChildMock()

    with pytest.raises(ValueError):
        expr + other


def test_radd():
    """Right add two of ParameterExpressionChildMock instances.

    Check if
    1. expr1 is BinaryOperator.
    2. expr1's left is ParameterExpressionChildMock.
    3. expr1's right is the same as the expr2.
    4. expr1's kind is BinaryOpeKind.ADD.
    """
    expr1 = ParameterExpressionChildMock()
    expr2 = ParameterExpressionChildMock()

    expr1 += expr2

    print(type(expr1))
    print(expr1)

    # 1. expr1 is BinaryOperator.
    assert isinstance(expr1, BinaryOperator)
    # 2. expr1's left is ParameterExpressionChildMock.
    assert isinstance(expr1.left, ParameterExpressionChildMock)
    # 3. expr1's right is the same as the expr2.
    assert expr1.right == expr2
    # 4. expr1's kind is BinaryOpeKind.ADD.
    assert expr1.kind == BinaryOpeKind.ADD


@pytest.mark.parametrize("other", [1, 1.0, np.int64(1), np.float64(1.0)])
def test_radd_with_real(other):
    """Right add a ParameterExpressionChildMock instance with a real number.

    Check if
    1. expr is BinaryOperator.
    2. expr's left is ParameterExpressionChildMock.
    3. expr's right is Value.
    4. expr's right's value is the same as the given other.
    5. expr's kind is BinaryOpeKind.ADD.
    """
    expr = ParameterExpressionChildMock()

    expr += other

    # 1. expr is BinaryOperator.
    assert isinstance(expr, BinaryOperator)
    # 2. expr's left is ParameterExpressionChildMock.
    assert isinstance(expr.left, ParameterExpressionChildMock)
    # 3. expr's right is Value.
    assert isinstance(expr.right, Value)
    # 4. expr's right's value is the same as the given other.
    assert expr.right.value == other
    # 5. expr's kind is BinaryOpeKind.ADD.
    assert expr.kind == BinaryOpeKind.ADD


@pytest.mark.parametrize("other", ["1", [1, 2], 1j])
def test_aadd_with_unsupported_type(other):
    """Right add a ParameterExpressionChildMock instance with an unsupported type.

    Check if
    1. TypeError is raised.
    """
    expr = ParameterExpressionChildMock()

    with pytest.raises(ValueError):
        expr += other


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


def test_rmul():
    """Rigth mul two of ParameterExpressionChildMock instances.

    Check if
    1. expr1 is BinaryOperator.
    2. expr1's left is ParameterExpressionChildMock.
    3. expr1's right is the same as the expr2.
    4. expr1's kind is BinaryOpeKind.MUL.
    """
    expr1 = ParameterExpressionChildMock()
    expr2 = ParameterExpressionChildMock()

    expr1 *= expr2

    print(type(expr1))
    print(expr1)

    # 1. expr1 is BinaryOperator.
    assert isinstance(expr1, BinaryOperator)
    # 2. expr1's left is ParameterExpressionChildMock.
    assert isinstance(expr1.left, ParameterExpressionChildMock)
    # 3. expr1's right is the same as the expr2.
    assert expr1.right == expr2
    # 4. expr1's kind is BinaryOpeKind.MUL.
    assert expr1.kind == BinaryOpeKind.MUL


@pytest.mark.parametrize("other", [1, 1.0, np.int64(1), np.float64(1.0)])
def test_rmul_with_real(other):
    """Right mul a ParameterExpressionChildMock instance with a real number.

    Check if
    1. expr is BinaryOperator.
    2. expr's right is ParameterExpressionChildMock.
    3. expr's left is Value.
    4. expr's left's value is the same as the given other.
    5. expr's kind is BinaryOpeKind.MUL.
    """
    expr = ParameterExpressionChildMock()

    expr *= other

    # 1. expr is BinaryOperator.
    assert isinstance(expr, BinaryOperator)
    # 2. expr's right is ParameterExpressionChildMock.
    assert isinstance(expr.right, ParameterExpressionChildMock)
    # 3. expr's left is Value.
    assert isinstance(expr.left, Value)
    # 4. expr's left's value is the same as the given other.
    assert expr.left.value == other
    # 5. expr's kind is BinaryOpeKind.MUL.
    assert expr.kind == BinaryOpeKind.MUL


# <<< ParameterExpression <<<
