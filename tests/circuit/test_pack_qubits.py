"""Tests for pack_qubits API (Phase 2b).

Validates the pack_qubits frontend function: input validation, linear-type
enforcement, canonical element_uuids key, and downstream integration.
"""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.tracer import trace
from qamomile.circuit.transpiler.errors import QubitConsumedError


class TestPackQubitsExport:
    """Verify the API is properly exported."""

    def test_exists_and_exported(self):
        assert hasattr(qmc, "pack_qubits")
        assert "pack_qubits" in qmc.__all__


class TestPackQubitsRejects:
    """Tests for input validation."""

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match=r"requires at least one qubit"):
            qmc.pack_qubits()

    def test_rejects_non_qubit_input(self):
        from qamomile.circuit.ir.value import Value
        from qamomile.circuit.ir.types.primitives import FloatType

        with trace():
            f = qmc.Float(value=Value(type=FloatType(), name="f"))
            with pytest.raises(
                TypeError, match=r"accepts only Qubit or Vector\[Qubit\]"
            ):
                qmc.pack_qubits(f)

    def test_rejects_dynamic_vector(self):
        """Dynamic-size Vector[Qubit] should be rejected."""
        from qamomile.circuit.ir.value import ArrayValue, Value
        from qamomile.circuit.ir.types.primitives import QubitType, UIntType
        from qamomile.circuit.frontend.handle.array import Vector
        from qamomile.circuit.frontend.handle.primitives import UInt

        with trace():
            # Create a symbolic (non-constant) size
            sym_size = Value(type=UIntType(), name="n_sym")
            arr_val = ArrayValue(type=QubitType(), name="qs_dyn", shape=(sym_size,))
            dyn_vec = Vector._create_from_value(
                value=arr_val,
                shape=(UInt(value=sym_size),),
                name="qs_dyn",
            )
            with pytest.raises(
                ValueError, match=r"requires fixed-size Vector\[Qubit\]"
            ):
                qmc.pack_qubits(dyn_vec)


class TestPackQubitsAccepts:
    """Tests for valid inputs."""

    def test_accepts_single_qubit(self):
        @qmc.qkernel
        def k(q: qmc.Qubit) -> qmc.Vector[qmc.Qubit]:
            packed = qmc.pack_qubits(q)
            return packed

        graph = k.build()
        assert graph is not None

    def test_accepts_multiple_qubits(self):
        @qmc.qkernel
        def k(q0: qmc.Qubit, q1: qmc.Qubit) -> qmc.Vector[qmc.Qubit]:
            packed = qmc.pack_qubits(q0, q1)
            return packed

        graph = k.build()
        assert graph is not None

    def test_accepts_fixed_vector(self):
        """Vector created with qubit_array (concrete size) should be accepted."""

        @qmc.qkernel
        def k() -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(3, name="qs")
            packed = qmc.pack_qubits(qs)
            return packed

        graph = k.build()
        assert graph is not None

    def test_accepts_qubit_and_fixed_vector(self):
        @qmc.qkernel
        def k(q: qmc.Qubit) -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(2, name="qs")
            packed = qmc.pack_qubits(q, qs)
            return packed

        graph = k.build()
        assert graph is not None


class TestPackQubitsLinearType:
    """Tests for consume/alias behaviour."""

    def test_consumes_input_qubits(self):
        @qmc.qkernel
        def k(q: qmc.Qubit) -> qmc.Vector[qmc.Qubit]:
            packed = qmc.pack_qubits(q)
            # q is consumed; using it again should fail
            q = qmc.h(q)
            return packed

        with pytest.raises(QubitConsumedError):
            k.build()

    def test_disallows_alias_reuse_of_original_handles(self):
        @qmc.qkernel
        def k(q0: qmc.Qubit, q1: qmc.Qubit) -> qmc.Vector[qmc.Qubit]:
            packed = qmc.pack_qubits(q0, q1)
            # Reusing consumed q0 should fail
            q0 = qmc.x(q0)
            return packed

        with pytest.raises(QubitConsumedError):
            k.build()


class TestPackQubitsCanonicalKey:
    """Verify element_uuids canonical key is present."""

    def test_element_uuids_populated(self):
        from qamomile.circuit.ir.value import Value
        from qamomile.circuit.ir.types.primitives import QubitType
        from qamomile.circuit.frontend.handle.primitives import Qubit

        with trace():
            v0 = Value(type=QubitType(), name="q0")
            v1 = Value(type=QubitType(), name="q1")
            h0 = Qubit(value=v0)
            h1 = Qubit(value=v1)
            packed = qmc.pack_qubits(h0, h1)

            uuids = packed.value.params.get("element_uuids")
            assert uuids is not None
            assert len(uuids) == 2
            assert uuids[0] == v0.uuid
            assert uuids[1] == v1.uuid
