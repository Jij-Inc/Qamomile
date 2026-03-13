"""Tests for Value and ArrayValue IR classes."""

import pytest

from qamomile.circuit.ir.value import Value, ArrayValue
from qamomile.circuit.ir.types.primitives import QubitType, UIntType, FloatType


class TestValueNextVersion:
    """Test Value.next_version() method."""

    def test_next_version_increments_version(self):
        """next_version() should increment the version number."""
        v = Value(type=QubitType(), name="q", version=0)
        v2 = v.next_version()
        assert v2.version == 1
        assert v.version == 0  # Original unchanged

    def test_next_version_preserves_type(self):
        """next_version() should preserve the type."""
        v = Value(type=QubitType(), name="q")
        v2 = v.next_version()
        assert v2.type == QubitType()

    def test_next_version_preserves_name(self):
        """next_version() should preserve the name."""
        v = Value(type=QubitType(), name="my_qubit")
        v2 = v.next_version()
        assert v2.name == "my_qubit"

    def test_next_version_preserves_logical_id(self):
        """next_version() should preserve logical_id for physical identity tracking."""
        v = Value(type=QubitType(), name="q")
        v2 = v.next_version()
        assert v2.logical_id == v.logical_id

    def test_next_version_creates_new_uuid(self):
        """next_version() should create a new uuid."""
        v = Value(type=QubitType(), name="q")
        v2 = v.next_version()
        assert v2.uuid != v.uuid

    def test_next_version_copies_params(self):
        """next_version() should copy params (not share reference)."""
        v = Value(type=QubitType(), name="q", params={"foo": "bar"})
        v2 = v.next_version()
        assert v2.params == {"foo": "bar"}
        # Modifying v2.params should not affect v.params
        v2.params["new_key"] = "new_value"
        assert "new_key" not in v.params


class TestArrayValueNextVersion:
    """Test ArrayValue.next_version() method."""

    def test_next_version_increments_version(self):
        """next_version() should increment the version number."""
        shape = (Value(type=UIntType(), name="dim0", params={"const": 3}),)
        av = ArrayValue(type=QubitType(), name="q_array", shape=shape, version=0)
        av2 = av.next_version()
        assert av2.version == 1
        assert av.version == 0  # Original unchanged

    def test_next_version_preserves_shape(self):
        """next_version() should preserve the shape tuple."""
        dim0 = Value(type=UIntType(), name="dim0", params={"const": 3})
        dim1 = Value(type=UIntType(), name="dim1", params={"const": 5})
        shape = (dim0, dim1)

        av = ArrayValue(type=QubitType(), name="q_array", shape=shape)
        av2 = av.next_version()

        assert av2.shape == shape
        assert len(av2.shape) == 2
        assert av2.shape[0].params.get("const") == 3
        assert av2.shape[1].params.get("const") == 5

    def test_next_version_preserves_symbolic_shape(self):
        """next_version() should preserve symbolic (non-constant) shape dimensions."""
        # Symbolic dimension (no 'const' param)
        dim0 = Value(type=UIntType(), name="n_dim0", params={})
        shape = (dim0,)

        av = ArrayValue(type=QubitType(), name="q_array", shape=shape)
        av2 = av.next_version()

        assert av2.shape == shape
        assert len(av2.shape) == 1
        assert not av2.shape[0].is_constant()

    def test_next_version_preserves_type(self):
        """next_version() should preserve the element type."""
        shape = (Value(type=UIntType(), name="dim0", params={"const": 3}),)
        av = ArrayValue(type=QubitType(), name="q_array", shape=shape)
        av2 = av.next_version()
        assert av2.type == QubitType()

    def test_next_version_preserves_name(self):
        """next_version() should preserve the name."""
        shape = (Value(type=UIntType(), name="dim0", params={"const": 3}),)
        av = ArrayValue(type=QubitType(), name="my_qubits", shape=shape)
        av2 = av.next_version()
        assert av2.name == "my_qubits"

    def test_next_version_preserves_logical_id(self):
        """next_version() should preserve logical_id for physical identity tracking."""
        shape = (Value(type=UIntType(), name="dim0", params={"const": 3}),)
        av = ArrayValue(type=QubitType(), name="q_array", shape=shape)
        av2 = av.next_version()
        assert av2.logical_id == av.logical_id

    def test_next_version_creates_new_uuid(self):
        """next_version() should create a new uuid."""
        shape = (Value(type=UIntType(), name="dim0", params={"const": 3}),)
        av = ArrayValue(type=QubitType(), name="q_array", shape=shape)
        av2 = av.next_version()
        assert av2.uuid != av.uuid

    def test_next_version_with_empty_shape(self):
        """next_version() should handle empty shape gracefully."""
        av = ArrayValue(type=QubitType(), name="q_array", shape=())
        av2 = av.next_version()
        assert av2.shape == ()

    def test_multiple_next_versions_preserve_shape(self):
        """Calling next_version() multiple times should always preserve shape."""
        dim0 = Value(type=UIntType(), name="dim0", params={"const": 5})
        shape = (dim0,)

        av = ArrayValue(type=QubitType(), name="q_array", shape=shape)
        av2 = av.next_version()
        av3 = av2.next_version()
        av4 = av3.next_version()

        assert av4.version == 3
        assert av4.shape == shape
        assert av4.shape[0].params.get("const") == 5
