"""Unit tests for shared compile-time array-element value resolution."""

from typing import Any

import numpy as np
import pytest

from qamomile.circuit.ir.types import ObservableType
from qamomile.circuit.ir.types.primitives import BitType, FloatType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.value_resolver import ValueResolver


def _array_element(array_type: Any, name: str = "values") -> Value:
    """Create a one-dimensional array-element Value.

    Args:
        array_type (Any): Element type assigned to the parent
            ``ArrayValue`` and returned element ``Value``.
        name (str): Parent array name used for binding lookup. Defaults
            to ``"values"``.

    Returns:
        Value: A Value representing ``name[0]`` with constant index 0.
    """
    parent = ArrayValue(
        type=array_type,
        name=name,
        shape=(Value(type=UIntType(), name="dim").with_const(1),),
    )
    index = Value(type=UIntType(), name="idx").with_const(0)
    return Value(
        type=array_type,
        name=f"{name}[0]",
        parent_array=parent,
        element_indices=(index,),
    )


@pytest.mark.parametrize(
    ("array_type", "value", "expected", "expected_type"),
    [
        (FloatType(), np.float32(0.25), 0.25, float),
        (UIntType(), np.int64(7), 7, int),
        (BitType(), np.bool_(True), True, bool),
    ],
)
def test_bound_array_element_normalizes_numpy_scalar(
    array_type, value, expected, expected_type
):
    """A bound ndarray element resolves to Python primitives, not NumPy scalars."""
    resolved = ValueResolver(bindings={"values": np.array([value])}).resolve(
        _array_element(array_type)
    )

    assert resolved == expected
    assert type(resolved) is expected_type


def test_bound_array_element_preserves_non_numeric_object():
    """A bound object-array element resolves without numeric coercion."""
    marker = object()
    resolved = ValueResolver(
        bindings={"values": np.array([marker], dtype=object)}
    ).resolve(_array_element(ObservableType()))

    assert resolved is marker
