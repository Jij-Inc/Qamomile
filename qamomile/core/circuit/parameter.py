"""
Quantum Circuit Parameter Expressions Module

This module provides a framework for defining and manipulating parameter expressions
in quantum circuits. It allows for the creation of complex mathematical expressions
involving named parameters, constant values, and binary operations.

Key components:
- ParameterExpression: Abstract base class for all expressions
- Parameter: Represents a named parameter in a quantum circuit
- Value: Represents a constant numeric value
- BinaryOperator: Represents operations between two expressions
- BinaryOpeKind: Enumeration of supported binary operations

This module is essential for building parameterized quantum circuits, enabling
the definition of circuits with variable parameters that can be optimized or
swept over during execution or simulation.
"""

import enum
import abc


class ParameterExpression(abc.ABC):
    """
    Abstract base class for parameter expressions in quantum circuits.

    This class defines the basic operations (addition, multiplication, division)
    that can be performed on parameter expressions.
    """

    def __add__(self, other):
        """
        Add this expression to another expression or value.

        Args:
            other: Another ParameterExpression or a numeric value.

        Returns:
            BinaryOperator: A new expression representing the addition.
        """
        if isinstance(other, ParameterExpression):
            return BinaryOperator(self, other, BinaryOpeKind.ADD)
        else:
            return BinaryOperator(self, Value(other), BinaryOpeKind.ADD)

    def __radd__(self, other):
        """Enable reverse addition for non-ParameterExpression objects."""
        return self.__add__(other)

    def __mul__(self, other):
        """
        Multiply this expression by another expression or value.

        Args:
            other: Another ParameterExpression or a numeric value.

        Returns:
            BinaryOperator: A new expression representing the multiplication.
        """
        if isinstance(other, ParameterExpression):
            return BinaryOperator(self, other, BinaryOpeKind.MUL)
        else:
            return BinaryOperator(Value(other), self, BinaryOpeKind.MUL)

    def __rmul__(self, other):
        """Enable reverse multiplication for non-ParameterExpression objects."""
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, ParameterExpression):
            return BinaryOperator(self, other, BinaryOpeKind.DIV)
        elif isinstance(other, (int, float)):
            return BinaryOperator(self, Value(other), BinaryOpeKind.DIV)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return BinaryOperator(Value(other), self, BinaryOpeKind.DIV)
        return NotImplemented

    def get_parameters(self) -> list["Parameter"]:
        """
        Get the parameters in the expression.

        Returns:
            list[Parameter]: The parameters in the expression.
        """
        return []


class Parameter(ParameterExpression):
    """
    Represents a named parameter in a quantum circuit.
    """

    def __init__(self, name):
        """
        Initialize a Parameter with a name.

        Args:
            name (str): The name of the parameter.
        """
        self.name = name

    def __repr__(self):
        """String representation of the Parameter."""
        return self.name

    def get_parameters(self) -> list["Parameter"]:
        """Return this parameter in a list."""
        return [self]

    def __hash__(self) -> int:
        """Hash function for the Parameter, based on its name."""
        return hash(self.name)

    def __eq__(self, other):
        """Equality comparison for Parameters."""
        return isinstance(other, Parameter) and self.name == other.name


class Value(ParameterExpression):
    """
    Represents a constant numeric value in an expression.
    """

    def __init__(self, value):
        """
        Initialize a Value with a numeric constant.

        Args:
            value (number): The constant value.
        """
        self.value = value

    def __repr__(self):
        """String representation of the Value."""
        return str(self.value)


class BinaryOpeKind(enum.Enum):
    """
    Enumeration of binary operation types.
    """

    ADD = "+"
    MUL = "*"
    DIV = "/"


class BinaryOperator(ParameterExpression):
    """
    Represents a binary operation between two ParameterExpressions.
    """

    def __init__(self, left, right, kind):
        """
        Initialize a BinaryOperator.

        Args:
            left (ParameterExpression): The left operand.
            right (ParameterExpression): The right operand.
            kind (BinaryOpeKind): The type of binary operation.
        """
        self.left: ParameterExpression = left
        self.right: ParameterExpression = right
        self.kind: BinaryOpeKind = kind

    def get_parameters(self) -> list["Parameter"]:
        """
        Get all parameters involved in this binary operation.

        Returns:
            list[Parameter]: A list of all parameters in the expression.
        """
        return self.left.get_parameters() + self.right.get_parameters()

    def __repr__(self):
        """String representation of the BinaryOperator."""
        return f"{self.left} {self.kind.value} {self.right}"
