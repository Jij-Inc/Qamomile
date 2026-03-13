"""Tests for Tuple and Dict types in qkernel."""

import pytest
import qamomile.circuit as qmc
from qamomile.circuit.ir.types.primitives import TupleType, DictType, UIntType, FloatType
from qamomile.circuit.ir.value import TupleValue, DictValue, Value
from qamomile.circuit.frontend.handle.containers import Tuple, Dict
from qamomile.circuit.frontend.func_to_block import (
    handle_type_map,
    is_tuple_type,
    is_dict_type,
    create_dummy_input,
)


class TestTupleType:
    """Tests for TupleType IR type."""

    def test_tuple_type_creation(self):
        """Test creating a TupleType."""
        tt = TupleType(element_types=(UIntType(), UIntType()))
        assert len(tt.element_types) == 2
        assert all(isinstance(t, UIntType) for t in tt.element_types)

    def test_tuple_type_equality(self):
        """Test TupleType equality based on element types."""
        tt1 = TupleType(element_types=(UIntType(), UIntType()))
        tt2 = TupleType(element_types=(UIntType(), UIntType()))
        tt3 = TupleType(element_types=(UIntType(), FloatType()))

        assert tt1 == tt2
        assert tt1 != tt3

    def test_tuple_type_hash(self):
        """Test TupleType hashing."""
        tt1 = TupleType(element_types=(UIntType(), UIntType()))
        tt2 = TupleType(element_types=(UIntType(), UIntType()))

        # Same element types should hash the same
        assert hash(tt1) == hash(tt2)

    def test_tuple_type_label(self):
        """Test TupleType label method."""
        tt = TupleType(element_types=(UIntType(), FloatType()))
        assert "Tuple" in tt.label()
        assert "UIntType" in tt.label()
        assert "FloatType" in tt.label()


class TestDictType:
    """Tests for DictType IR type."""

    def test_dict_type_creation(self):
        """Test creating a DictType."""
        key_type = TupleType(element_types=(UIntType(), UIntType()))
        value_type = FloatType()
        dt = DictType(key_type=key_type, value_type=value_type)

        assert dt.key_type == key_type
        assert dt.value_type == value_type

    def test_dict_type_equality(self):
        """Test DictType equality."""
        key_type = TupleType(element_types=(UIntType(), UIntType()))
        dt1 = DictType(key_type=key_type, value_type=FloatType())
        dt2 = DictType(key_type=key_type, value_type=FloatType())
        dt3 = DictType(key_type=key_type, value_type=UIntType())

        assert dt1 == dt2
        assert dt1 != dt3

    def test_dict_type_hash(self):
        """Test DictType hashing."""
        key_type = TupleType(element_types=(UIntType(), UIntType()))
        dt1 = DictType(key_type=key_type, value_type=FloatType())
        dt2 = DictType(key_type=key_type, value_type=FloatType())

        assert hash(dt1) == hash(dt2)

    def test_dict_type_label(self):
        """Test DictType label method."""
        key_type = TupleType(element_types=(UIntType(), UIntType()))
        dt = DictType(key_type=key_type, value_type=FloatType())

        label = dt.label()
        assert "Dict" in label


class TestTupleValue:
    """Tests for TupleValue IR value."""

    def test_tuple_value_creation(self):
        """Test creating a TupleValue."""
        elem1 = Value(type=UIntType(), name="i")
        elem2 = Value(type=UIntType(), name="j")

        tv = TupleValue(name="idx", elements=(elem1, elem2))

        assert tv.name == "idx"
        assert len(tv.elements) == 2

    def test_tuple_value_is_parameter(self):
        """Test TupleValue parameter detection."""
        elem = Value(type=UIntType(), name="i")
        tv1 = TupleValue(name="idx", elements=(elem,), params={"parameter": "idx"})
        tv2 = TupleValue(name="idx", elements=(elem,))

        assert tv1.is_parameter()
        assert not tv2.is_parameter()


class TestDictValue:
    """Tests for DictValue IR value."""

    def test_dict_value_creation(self):
        """Test creating a DictValue."""
        dv = DictValue(name="ising", entries=[])

        assert dv.name == "ising"
        assert len(dv.entries) == 0

    def test_dict_value_with_entries(self):
        """Test DictValue with entries."""
        key = Value(type=UIntType(), name="key")
        value = Value(type=FloatType(), name="value")
        dv = DictValue(name="ising", entries=[(key, value)])

        assert len(dv) == 1

    def test_dict_value_is_parameter(self):
        """Test DictValue parameter detection."""
        dv1 = DictValue(name="ising", entries=[], params={"parameter": "ising"})
        dv2 = DictValue(name="ising", entries=[])

        assert dv1.is_parameter()
        assert not dv2.is_parameter()


class TestTypeDetection:
    """Tests for type detection functions."""

    def test_is_tuple_type(self):
        """Test is_tuple_type detection."""
        assert is_tuple_type(qmc.Tuple[qmc.UInt, qmc.UInt])
        assert not is_tuple_type(qmc.Vector[qmc.Qubit])
        assert not is_tuple_type(qmc.UInt)

    def test_is_dict_type(self):
        """Test is_dict_type detection."""
        assert is_dict_type(qmc.Dict[qmc.UInt, qmc.Float])
        assert is_dict_type(qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float])
        assert not is_dict_type(qmc.Vector[qmc.Qubit])
        assert not is_dict_type(qmc.UInt)


class TestHandleTypeMap:
    """Tests for handle_type_map function."""

    def test_tuple_type_mapping(self):
        """Test mapping Tuple type to TupleType."""
        result = handle_type_map(qmc.Tuple[qmc.UInt, qmc.UInt])

        assert isinstance(result, TupleType)
        assert len(result.element_types) == 2
        assert all(isinstance(t, UIntType) for t in result.element_types)

    def test_nested_tuple_type_mapping(self):
        """Test mapping Tuple with different element types."""
        result = handle_type_map(qmc.Tuple[qmc.UInt, qmc.Float])

        assert isinstance(result, TupleType)
        assert isinstance(result.element_types[0], UIntType)
        assert isinstance(result.element_types[1], FloatType)

    def test_dict_type_mapping(self):
        """Test mapping Dict type to DictType."""
        result = handle_type_map(qmc.Dict[qmc.UInt, qmc.Float])

        assert isinstance(result, DictType)
        assert isinstance(result.key_type, UIntType)
        assert isinstance(result.value_type, FloatType)

    def test_dict_with_tuple_key_mapping(self):
        """Test mapping Dict[Tuple[UInt, UInt], Float] to DictType."""
        result = handle_type_map(qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float])

        assert isinstance(result, DictType)
        assert isinstance(result.key_type, TupleType)
        assert isinstance(result.value_type, FloatType)


class TestCreateDummyInput:
    """Tests for create_dummy_input with Tuple and Dict types."""

    def test_create_tuple_input(self):
        """Test creating dummy Tuple input."""
        result = create_dummy_input(qmc.Tuple[qmc.UInt, qmc.UInt], "idx")

        assert isinstance(result, Tuple)
        assert result.name == "idx"
        assert len(result._elements) == 2

    def test_create_dict_input(self):
        """Test creating dummy Dict input."""
        result = create_dummy_input(
            qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float], "ising"
        )

        assert isinstance(result, Dict)
        assert result.name == "ising"
        assert result.value.is_parameter()


class TestTupleHandle:
    """Tests for Tuple handle operations."""

    def test_tuple_getitem(self):
        """Test Tuple element access."""
        elem1 = Value(type=UIntType(), name="i")
        elem2 = Value(type=UIntType(), name="j")
        tv = TupleValue(name="idx", elements=(elem1, elem2))

        from qamomile.circuit.frontend.handle.primitives import UInt

        handle1 = UInt(value=elem1)
        handle2 = UInt(value=elem2)

        tuple_handle = Tuple(value=tv, _elements=(handle1, handle2))

        assert tuple_handle[0] is handle1
        assert tuple_handle[1] is handle2

    def test_tuple_len(self):
        """Test Tuple length."""
        elem1 = Value(type=UIntType(), name="i")
        elem2 = Value(type=UIntType(), name="j")
        tv = TupleValue(name="idx", elements=(elem1, elem2))

        from qamomile.circuit.frontend.handle.primitives import UInt

        handle1 = UInt(value=elem1)
        handle2 = UInt(value=elem2)

        tuple_handle = Tuple(value=tv, _elements=(handle1, handle2))

        assert len(tuple_handle) == 2


class TestDictHandle:
    """Tests for Dict handle operations."""

    def test_dict_items(self):
        """Test Dict.items() iterator."""
        dv = DictValue(name="ising", entries=[])
        dict_handle = Dict(value=dv, _entries=[])

        items_iter = dict_handle.items()
        assert list(items_iter) == []

    def test_dict_len(self):
        """Test Dict length."""
        dv = DictValue(name="ising", entries=[])
        dict_handle = Dict(value=dv, _entries=[])

        assert len(dict_handle) == 0
