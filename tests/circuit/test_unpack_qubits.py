"""Tests for unpack_qubits API.

Validates the unpack_qubits frontend function: input validation,
num_elements / indices modes, linear-type enforcement, and E2E integration.
"""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.tracer import trace
from qamomile.circuit.transpiler.errors import QubitConsumedError


class TestUnpackQubitsExport:
    """Verify the API is properly exported."""

    def test_exists_and_exported(self):
        assert hasattr(qmc, "unpack_qubits")
        assert "unpack_qubits" in qmc.__all__


class TestUnpackQubitsRejects:
    """Tests for input validation."""

    def test_rejects_both_num_elements_and_indices(self):
        """Cannot specify both num_elements and indices."""

        @qmc.qkernel
        def k() -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(4, name="qs")
            packed = qmc.pack_qubits(qs)
            a, b = qmc.unpack_qubits(
                packed,
                num_unpacked=2,
                num_elements=[2, 2],
                indices=[[0, 1], [2, 3]],
            )
            return a

        with pytest.raises(ValueError, match=r"Cannot specify both"):
            k.build()

    def test_rejects_neither_specified(self):
        """Must specify at least one of num_elements or indices."""

        @qmc.qkernel
        def k() -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(4, name="qs")
            packed = qmc.pack_qubits(qs)
            (a,) = qmc.unpack_qubits(packed, num_unpacked=1)
            return a

        with pytest.raises(ValueError, match=r"Must specify either"):
            k.build()

    def test_rejects_num_unpacked_zero(self):
        from qamomile.circuit.ir.value import Value, ArrayValue
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.frontend.handle.array import Vector

        with trace():
            size_val = Value(type=UIntType(), name="sz", params={"const": 2})
            arr = ArrayValue(
                type=QubitType(),
                name="p",
                shape=(size_val,),
                params={"element_uuids": ["a", "b"], "element_logical_ids": ["a", "b"]},
            )
            packed = Vector._create_from_value(value=arr, shape=(2,), name="p")
            with pytest.raises(ValueError, match=r"num_unpacked must be >= 1"):
                qmc.unpack_qubits(packed, num_unpacked=0, num_elements=[])

    def test_rejects_num_elements_length_mismatch(self):
        from qamomile.circuit.ir.value import Value, ArrayValue
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.frontend.handle.array import Vector

        with trace():
            size_val = Value(type=UIntType(), name="sz", params={"const": 4})
            arr = ArrayValue(
                type=QubitType(),
                name="p",
                shape=(size_val,),
                params={
                    "element_uuids": ["a", "b", "c", "d"],
                    "element_logical_ids": ["a", "b", "c", "d"],
                },
            )
            packed = Vector._create_from_value(value=arr, shape=(4,), name="p")
            with pytest.raises(ValueError, match=r"len\(num_elements\).*num_unpacked"):
                qmc.unpack_qubits(packed, num_unpacked=3, num_elements=[2, 2])

    def test_rejects_indices_duplicate(self):
        from qamomile.circuit.ir.value import Value, ArrayValue
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.frontend.handle.array import Vector

        with trace():
            size_val = Value(type=UIntType(), name="sz", params={"const": 3})
            arr = ArrayValue(
                type=QubitType(),
                name="p",
                shape=(size_val,),
                params={
                    "element_uuids": ["a", "b", "c"],
                    "element_logical_ids": ["a", "b", "c"],
                },
            )
            packed = Vector._create_from_value(value=arr, shape=(3,), name="p")
            with pytest.raises(ValueError, match=r"Duplicate index"):
                qmc.unpack_qubits(packed, num_unpacked=2, indices=[[0, 1], [1, 2]])

    def test_rejects_indices_out_of_range(self):
        from qamomile.circuit.ir.value import Value, ArrayValue
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.frontend.handle.array import Vector

        with trace():
            size_val = Value(type=UIntType(), name="sz", params={"const": 3})
            arr = ArrayValue(
                type=QubitType(),
                name="p",
                shape=(size_val,),
                params={
                    "element_uuids": ["a", "b", "c"],
                    "element_logical_ids": ["a", "b", "c"],
                },
            )
            packed = Vector._create_from_value(value=arr, shape=(3,), name="p")
            with pytest.raises(ValueError, match=r"out of range"):
                qmc.unpack_qubits(packed, num_unpacked=2, indices=[[0, 1], [5]])

    def test_rejects_indices_missing_elements(self):
        from qamomile.circuit.ir.value import Value, ArrayValue
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.frontend.handle.array import Vector

        with trace():
            size_val = Value(type=UIntType(), name="sz", params={"const": 4})
            arr = ArrayValue(
                type=QubitType(),
                name="p",
                shape=(size_val,),
                params={
                    "element_uuids": ["a", "b", "c", "d"],
                    "element_logical_ids": ["a", "b", "c", "d"],
                },
            )
            packed = Vector._create_from_value(value=arr, shape=(4,), name="p")
            with pytest.raises(ValueError, match=r"cover all.*qubits"):
                qmc.unpack_qubits(packed, num_unpacked=2, indices=[[0], [2]])


class TestUnpackQubitsNumElements:
    """Tests for num_elements (consecutive split) mode."""

    def test_splits_in_order(self):
        from qamomile.circuit.ir.value import Value, ArrayValue
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.frontend.handle.array import Vector

        with trace():
            size_val = Value(type=UIntType(), name="sz", params={"const": 7})
            uuids = [f"q{i}" for i in range(7)]
            arr = ArrayValue(
                type=QubitType(),
                name="p",
                shape=(size_val,),
                params={
                    "element_uuids": uuids,
                    "element_logical_ids": uuids,
                },
            )
            packed = Vector._create_from_value(value=arr, shape=(7,), name="p")
            a, b, c = qmc.unpack_qubits(packed, num_unpacked=3, num_elements=[3, 1, 3])

            assert a.value.params["element_uuids"] == ["q0", "q1", "q2"]
            assert b.value.params["element_uuids"] == ["q3"]
            assert c.value.params["element_uuids"] == ["q4", "q5", "q6"]

    def test_singleton_segment_returns_vector(self):
        """Even size-1 segments return Vector[Qubit], not Qubit."""
        from qamomile.circuit.ir.value import Value, ArrayValue
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.frontend.handle.array import Vector

        with trace():
            size_val = Value(type=UIntType(), name="sz", params={"const": 2})
            arr = ArrayValue(
                type=QubitType(),
                name="p",
                shape=(size_val,),
                params={
                    "element_uuids": ["a", "b"],
                    "element_logical_ids": ["a", "b"],
                },
            )
            packed = Vector._create_from_value(value=arr, shape=(2,), name="p")
            a, b = qmc.unpack_qubits(packed, num_unpacked=2, num_elements=[1, 1])

            assert isinstance(a, Vector)
            assert isinstance(b, Vector)
            assert a.shape == (1,)
            assert b.shape == (1,)


class TestUnpackQubitsIndices:
    """Tests for indices (arbitrary reordering) mode."""

    def test_custom_order(self):
        from qamomile.circuit.ir.value import Value, ArrayValue
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.frontend.handle.array import Vector

        with trace():
            size_val = Value(type=UIntType(), name="sz", params={"const": 4})
            uuids = ["q0", "q1", "q2", "q3"]
            arr = ArrayValue(
                type=QubitType(),
                name="p",
                shape=(size_val,),
                params={
                    "element_uuids": uuids,
                    "element_logical_ids": uuids,
                },
            )
            packed = Vector._create_from_value(value=arr, shape=(4,), name="p")
            a, b = qmc.unpack_qubits(packed, num_unpacked=2, indices=[[3, 1], [0, 2]])

            assert a.value.params["element_uuids"] == ["q3", "q1"]
            assert b.value.params["element_uuids"] == ["q0", "q2"]


class TestUnpackQubitsLinearType:
    """Tests for consume behaviour."""

    def test_consumes_input_vector(self):
        from qamomile.circuit.ir.value import Value, ArrayValue
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.frontend.handle.array import Vector

        with trace():
            size_val = Value(type=UIntType(), name="sz", params={"const": 2})
            arr = ArrayValue(
                type=QubitType(),
                name="p",
                shape=(size_val,),
                params={
                    "element_uuids": ["a", "b"],
                    "element_logical_ids": ["a", "b"],
                },
            )
            packed = Vector._create_from_value(value=arr, shape=(2,), name="p")
            a, b = qmc.unpack_qubits(packed, num_unpacked=2, num_elements=[1, 1])

            # packed is consumed; element access should fail
            with pytest.raises(QubitConsumedError, match=r"consumed"):
                _ = packed[0]


class TestPackUnpackE2E:
    """End-to-end tests combining pack and unpack."""

    def test_pack_unpack_gate_measure(self):
        """pack -> unpack -> gate -> measure should build successfully."""

        @qmc.qkernel
        def k(q0: qmc.Qubit, q1: qmc.Qubit, q2: qmc.Qubit) -> qmc.Vector[qmc.Qubit]:
            packed = qmc.pack_qubits(q0, q1, q2)
            a, b = qmc.unpack_qubits(packed, num_unpacked=2, num_elements=[2, 1])
            # Apply gate to unpacked sub-vector element
            q = b[0]
            q = qmc.h(q)
            b[0] = q
            # Repack for return
            result = qmc.pack_qubits(a, b)
            return result

        graph = k.build()
        assert graph is not None
