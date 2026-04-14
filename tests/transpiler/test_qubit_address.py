"""Unit tests for QubitAddress type."""

from __future__ import annotations

from qamomile.circuit.transpiler.passes.emit_support.qubit_address import (
    ClbitMap,
    QubitAddress,
    QubitMap,
)


class TestQubitAddressBasics:
    def test_scalar_address(self) -> None:
        addr = QubitAddress(uuid="abc123")
        assert addr.uuid == "abc123"
        assert addr.element_index is None
        assert not addr.is_array_element
        assert str(addr) == "abc123"

    def test_array_element_address(self) -> None:
        addr = QubitAddress(uuid="abc123", element_index=5)
        assert addr.uuid == "abc123"
        assert addr.element_index == 5
        assert addr.is_array_element
        assert str(addr) == "abc123_5"

    def test_with_element(self) -> None:
        base = QubitAddress(uuid="arr_uuid")
        elem = base.with_element(3)
        assert elem.uuid == "arr_uuid"
        assert elem.element_index == 3
        assert elem.is_array_element

    def test_matches_array(self) -> None:
        addr = QubitAddress(uuid="arr1", element_index=0)
        assert addr.matches_array("arr1")
        assert not addr.matches_array("arr2")

        # Scalar address does not match any array
        scalar = QubitAddress(uuid="arr1")
        assert not scalar.matches_array("arr1")


class TestQubitAddressHashability:
    def test_hashable_as_dict_key(self) -> None:
        qmap: QubitMap = {}
        addr1 = QubitAddress(uuid="a", element_index=0)
        addr2 = QubitAddress(uuid="a", element_index=1)
        addr3 = QubitAddress(uuid="b")

        qmap[addr1] = 0
        qmap[addr2] = 1
        qmap[addr3] = 2

        assert qmap[addr1] == 0
        assert qmap[addr2] == 1
        assert qmap[addr3] == 2
        assert len(qmap) == 3

    def test_equality(self) -> None:
        a1 = QubitAddress(uuid="x", element_index=3)
        a2 = QubitAddress(uuid="x", element_index=3)
        assert a1 == a2
        assert hash(a1) == hash(a2)

    def test_inequality(self) -> None:
        a = QubitAddress(uuid="x", element_index=0)
        b = QubitAddress(uuid="x", element_index=1)
        c = QubitAddress(uuid="y", element_index=0)
        d = QubitAddress(uuid="x")
        assert a != b
        assert a != c
        assert a != d

    def test_frozen_immutable(self) -> None:
        addr = QubitAddress(uuid="test", element_index=1)
        import dataclasses

        assert dataclasses.is_dataclass(addr)
        # Frozen dataclass raises on attribute assignment
        try:
            addr.uuid = "other"  # type: ignore[misc]
            assert False, "Should have raised"
        except dataclasses.FrozenInstanceError:
            pass


class TestQubitAddressInMaps:
    def test_array_iteration_with_matches(self) -> None:
        """Replace the legacy key.startswith(prefix) pattern."""
        qmap: QubitMap = {
            QubitAddress("arr1", 0): 0,
            QubitAddress("arr1", 1): 1,
            QubitAddress("arr1", 2): 2,
            QubitAddress("arr2", 0): 3,
            QubitAddress("scalar"): 4,
        }

        # Find all elements of arr1
        arr1_addrs = [addr for addr in qmap if addr.matches_array("arr1")]
        assert len(arr1_addrs) == 3
        assert all(addr.uuid == "arr1" for addr in arr1_addrs)

    def test_clbit_map_type_alias(self) -> None:
        cmap: ClbitMap = {QubitAddress("c1", 0): 0}
        assert cmap[QubitAddress("c1", 0)] == 0
