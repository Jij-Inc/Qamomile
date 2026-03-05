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
            assert isinstance(b, qmc.Qubit)
            assert b.value.parent_array is not None
            assert b.value.parent_array.params["element_uuids"] == ["q3"]
            assert c.value.params["element_uuids"] == ["q4", "q5", "q6"]

    def test_singleton_segment_returns_qubit(self):
        """Size-1 num_elements segments should return Qubit."""
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

            assert isinstance(a, qmc.Qubit)
            assert isinstance(b, qmc.Qubit)
            assert a.value.parent_array is not None
            assert b.value.parent_array is not None
            assert a.value.parent_array.params["element_uuids"] == ["a"]
            assert b.value.parent_array.params["element_uuids"] == ["b"]


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


class TestUnpackQubitsNoElementUuids:
    """Phase 1: unpack_qubits on vectors without element_uuids."""

    def test_derives_uuids_from_source(self):
        """Vectors without element_uuids should derive metadata from source UUID."""
        from qamomile.circuit.ir.value import Value, ArrayValue
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.frontend.handle.array import Vector

        with trace():
            size_val = Value(type=UIntType(), name="sz", params={"const": 4})
            arr = ArrayValue(
                type=QubitType(),
                name="qs",
                shape=(size_val,),
                params={},  # NO element_uuids
            )
            vec = Vector._create_from_value(value=arr, shape=(4,), name="qs")
            a, b = qmc.unpack_qubits(vec, num_unpacked=2, num_elements=[3, 1])

            # Derived element_uuids should use {source_uuid}_{i}
            assert len(a.value.params["element_uuids"]) == 3
            assert a.value.params["element_uuids"][0] == f"{arr.uuid}_0"
            assert a.value.params["element_uuids"][2] == f"{arr.uuid}_2"
            assert isinstance(b, qmc.Qubit)
            assert b.value.parent_array is not None
            assert len(b.value.parent_array.params["element_uuids"]) == 1
            assert b.value.parent_array.params["element_uuids"][0] == f"{arr.uuid}_3"

    def test_derives_logical_ids_from_source(self):
        """Derived element_logical_ids should use source logical_id."""
        from qamomile.circuit.ir.value import Value, ArrayValue
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.frontend.handle.array import Vector

        with trace():
            size_val = Value(type=UIntType(), name="sz", params={"const": 3})
            arr = ArrayValue(
                type=QubitType(),
                name="qs",
                shape=(size_val,),
                params={},
            )
            vec = Vector._create_from_value(value=arr, shape=(3,), name="qs")
            a, b = qmc.unpack_qubits(vec, num_unpacked=2, num_elements=[2, 1])

            assert len(a.value.params["element_logical_ids"]) == 2
            assert a.value.params["element_logical_ids"][0] == f"{arr.logical_id}_0"


class TestUnpackQubitsSymbolic:
    """Phase 2: unpack_qubits with symbolic num_elements."""

    def test_symbolic_num_elements_produces_split_spec(self):
        """Symbolic num_elements should store split_spec instead of element_uuids."""
        from qamomile.circuit.ir.value import Value, ArrayValue
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.frontend.handle.array import Vector
        from qamomile.circuit.frontend.handle.primitives import UInt

        with trace():
            sym_size = Value(type=UIntType(), name="n_sym")
            arr = ArrayValue(type=QubitType(), name="qs_sym", shape=(sym_size,))
            vec = Vector._create_from_value(
                value=arr,
                shape=(UInt(value=sym_size),),
                name="qs_sym",
            )
            n = vec.shape[0]
            a, b = qmc.unpack_qubits(vec, num_unpacked=2, num_elements=[n - 1, 1])

            # Symbolic path produces split_spec, not element_uuids
            assert "split_spec" in a.value.params
            assert a.value.params["split_spec"]["mode"] == "num_elements"
            assert a.value.params["split_spec"]["group_index"] == 0
            assert isinstance(b, qmc.Qubit)
            assert b.value.parent_array is not None
            assert "split_spec" in b.value.parent_array.params
            assert b.value.parent_array.params["split_spec"]["group_index"] == 1

    def test_mixed_symbolic_concrete_num_elements(self):
        """Mix of symbolic and concrete entries in num_elements."""
        from qamomile.circuit.ir.value import Value, ArrayValue
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.frontend.handle.array import Vector
        from qamomile.circuit.frontend.handle.primitives import UInt

        with trace():
            sym_size = Value(type=UIntType(), name="n")
            arr = ArrayValue(type=QubitType(), name="qs", shape=(sym_size,))
            vec = Vector._create_from_value(
                value=arr,
                shape=(UInt(value=sym_size),),
                name="qs",
            )
            n = vec.shape[0]
            # [n-1, 1] — first is symbolic, second is concrete int
            a, b = qmc.unpack_qubits(vec, num_unpacked=2, num_elements=[n - 1, 1])

            # Concrete entry (1) still goes through symbolic path
            assert isinstance(b, qmc.Qubit)
            assert b.value.parent_array is not None
            assert "split_spec" in b.value.parent_array.params

    def test_symbolic_does_not_call_range(self):
        """Symbolic path should not produce element_uuids (no range() call)."""
        from qamomile.circuit.ir.value import Value, ArrayValue
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.frontend.handle.array import Vector
        from qamomile.circuit.frontend.handle.primitives import UInt

        with trace():
            sym_size = Value(type=UIntType(), name="n")
            arr = ArrayValue(type=QubitType(), name="qs", shape=(sym_size,))
            vec = Vector._create_from_value(
                value=arr,
                shape=(UInt(value=sym_size),),
                name="qs",
            )
            n = vec.shape[0]
            a, b = qmc.unpack_qubits(vec, num_unpacked=2, num_elements=[n - 1, 1])

            # Should NOT have element_uuids in symbolic path
            assert "element_uuids" not in a.value.params
            assert isinstance(b, qmc.Qubit)
            assert b.value.parent_array is not None
            assert "element_uuids" not in b.value.parent_array.params

    def test_concrete_int_negative_rejected_in_symbolic_mode(self):
        """Concrete int < 1 in num_elements should be rejected even in symbolic mode."""
        from qamomile.circuit.ir.value import Value, ArrayValue
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.frontend.handle.array import Vector
        from qamomile.circuit.frontend.handle.primitives import UInt

        with trace():
            sym_size = Value(type=UIntType(), name="n")
            arr = ArrayValue(type=QubitType(), name="qs", shape=(sym_size,))
            vec = Vector._create_from_value(
                value=arr,
                shape=(UInt(value=sym_size),),
                name="qs",
            )
            n = vec.shape[0]
            with pytest.raises(ValueError, match=r"must be >= 1"):
                qmc.unpack_qubits(vec, num_unpacked=2, num_elements=[n, 0])

    def test_indices_with_unresolved_total_rejected(self):
        """indices= with symbolic vector should raise ValueError."""
        from qamomile.circuit.ir.value import Value, ArrayValue
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.frontend.handle.array import Vector
        from qamomile.circuit.frontend.handle.primitives import UInt

        with trace():
            sym_size = Value(type=UIntType(), name="n")
            arr = ArrayValue(type=QubitType(), name="qs", shape=(sym_size,))
            vec = Vector._create_from_value(
                value=arr,
                shape=(UInt(value=sym_size),),
                name="qs",
            )
            with pytest.raises(ValueError, match=r"fixed-size"):
                qmc.unpack_qubits(vec, num_unpacked=2, indices=[[0, 1], [2]])

    def test_symbolic_indices_values_rejected(self):
        """Non-int values in indices= should raise TypeError."""
        from qamomile.circuit.ir.value import Value, ArrayValue
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.frontend.handle.array import Vector
        from qamomile.circuit.frontend.handle.primitives import UInt

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
            sym_idx = UInt(value=Value(type=UIntType(), name="idx_sym"))
            with pytest.raises(TypeError, match=r"indices values must be int"):
                qmc.unpack_qubits(packed, num_unpacked=2, indices=[[sym_idx], [1, 2]])


class TestPackUnpackE2E:
    """End-to-end tests combining pack and unpack."""

    def test_pack_unpack_gate_measure(self):
        """pack -> unpack -> gate -> measure should build successfully."""

        @qmc.qkernel
        def k(q0: qmc.Qubit, q1: qmc.Qubit, q2: qmc.Qubit) -> qmc.Vector[qmc.Qubit]:
            packed = qmc.pack_qubits(q0, q1, q2)
            a, b = qmc.unpack_qubits(packed, num_unpacked=2, num_elements=[2, 1])
            # Apply gate to singleton output
            b = qmc.h(b)
            # Repack for return
            result = qmc.pack_qubits(a, b)
            return result

        graph = k.build()
        assert graph is not None

    def test_symbolic_unpack_pack_round_trip(self):
        """Symbolic unpack -> pack should build successfully."""

        @qmc.qkernel
        def k(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            n = qs.shape[0]
            a, b = qmc.unpack_qubits(qs, num_unpacked=2, num_elements=[n - 1, 1])
            result = qmc.pack_qubits(a, b)
            return result

        graph = k.build()
        assert graph is not None

    def test_symbolic_unpack_pack_measure_transpiles(self):
        """Symbolic unpack -> pack -> measure should transpile with bindings."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def k(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(n, name="qs")
            a, b = qmc.unpack_qubits(qs, num_unpacked=2, num_elements=[n - 1, 1])
            qs2 = qmc.pack_qubits(a, b)
            return qmc.measure(qs2)

        transpiler = QiskitTranspiler()
        executable = transpiler.transpile(k, bindings={"n": 3})
        assert executable is not None
        assert len(executable.compiled_quantum) > 0

    def test_symbolic_unpack_pack_measure_no_bindings_fails(self):
        """Symbolic unpack -> pack -> measure without bindings should fail."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        @qmc.qkernel
        def k(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(n, name="qs")
            a, b = qmc.unpack_qubits(qs, num_unpacked=2, num_elements=[n - 1, 1])
            qs2 = qmc.pack_qubits(a, b)
            return qmc.measure(qs2)

        transpiler = QiskitTranspiler()
        with pytest.raises(Exception):
            transpiler.transpile(k)

    def test_constant_fold_canonicalizes_measure_shape(self):
        """After constant_fold with bindings, MeasureVectorOperation shape should be const."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler
        from qamomile.circuit.ir.value import ArrayValue
        from qamomile.circuit.ir.operation.gate import MeasureVectorOperation

        @qmc.qkernel
        def k(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            qs = qmc.qubit_array(n, name="qs")
            a, b = qmc.unpack_qubits(qs, num_unpacked=2, num_elements=[n - 1, 1])
            qs2 = qmc.pack_qubits(a, b)
            return qmc.measure(qs2)

        transpiler = QiskitTranspiler()
        block = transpiler.to_block(k, bindings={"n": 3})
        inlined = transpiler.inline(block)
        folded = transpiler.constant_fold(inlined, bindings={"n": 3})

        # Find MeasureVectorOperation and verify shape is concrete
        found = False
        for op in folded.operations:
            if isinstance(op, MeasureVectorOperation):
                result = op.results[0]
                if isinstance(result, ArrayValue) and result.shape:
                    dim = result.shape[0]
                    assert dim.is_constant(), (
                        f"shape[0] should be constant after folding, "
                        f"got name={dim.name}, is_constant={dim.is_constant()}"
                    )
                    assert dim.get_const() == 3
                    found = True
        assert found, "MeasureVectorOperation not found in folded block"
