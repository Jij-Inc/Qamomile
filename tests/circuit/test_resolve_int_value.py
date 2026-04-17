"""Test that resolve_int_value uses UUID-based lookup for BinOp results."""

import numpy as np

from qamomile.circuit.ir.types.primitives import FloatType, UIntType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.passes.emit_support import ValueResolver


class TestResolveIntValueUuidLookup:
    """Test that resolve_int_value prefers UUID over name for binding lookup."""

    def test_uuid_lookup_prevents_name_collision(self):
        """UUID lookup returns the correct value when multiple BinOp results share the same name."""
        resolver = ValueResolver()

        # BinOp result 1: n // 2 = 4 (uuid_a, name="uint_tmp")
        val_a = Value(type=UIntType(), name="uint_tmp")
        # BinOp result 2: n * 2 = 16 (uuid_b, name="uint_tmp")
        val_b = Value(type=UIntType(), name="uint_tmp")

        # Reproduce the storage pattern of _evaluate_binop
        bindings: dict[str, int] = {}
        bindings[val_a.uuid] = 4
        bindings[val_a.name] = 4  # "uint_tmp" = 4
        bindings[val_b.uuid] = 16
        bindings[val_b.name] = 16  # "uint_tmp" = 16 (overwritten!)

        # Lookup via val_a: UUID should return 4, name would return 16
        result = resolver.resolve_int_value(val_a, bindings)
        assert result == 4, (
            f"Expected 4 (n//2), got {result}. "
            f"resolve_int_value used name-based lookup and got n*2=16 instead"
        )

    def test_uuid_lookup_falls_back_to_name(self):
        """Falls back to name-based lookup when UUID is not in bindings."""
        resolver = ValueResolver()

        val = Value(type=UIntType(), name="some_var")
        bindings: dict[str, int] = {"some_var": 42}
        # UUID is not in bindings

        result = resolver.resolve_int_value(val, bindings)
        assert result == 42

    def test_constant_value_unaffected(self):
        """Constant Values resolve correctly regardless of UUID/name lookup."""
        resolver = ValueResolver()

        val = Value(type=UIntType(), name="const").with_const(7)
        bindings: dict[str, int] = {}

        result = resolver.resolve_int_value(val, bindings)
        assert result == 7

    def test_numpy_integer_binding_is_accepted(self):
        """NumPy integer scalars should resolve like built-in ints."""
        resolver = ValueResolver()

        val = Value(type=UIntType(), name="some_var")
        bindings = {"some_var": np.uint64(42)}

        result = resolver.resolve_int_value(val, bindings)
        assert result == 42


class TestResolveArrayElementValue:
    """Tests for resolving bound array elements."""

    def test_numpy_integer_array_element_is_accepted(self):
        """Nested array lookups should normalize NumPy integer results."""
        resolver = ValueResolver()

        edges = ArrayValue(type=UIntType(), name="edges")
        loop_idx = Value(type=UIntType(), name="e")
        col_idx = Value(type=UIntType(), name="idx_0").with_const(0)
        nested = Value(
            type=UIntType(),
            name="edges[e,0]",
            parent_array=edges,
            element_indices=(loop_idx, col_idx),
        )
        bindings = {
            "edges": np.array([[0, 1], [1, 2]], dtype=np.uint64),
            "e": 1,
        }

        result = resolver.resolve_classical_value(nested, bindings)

        assert result == 1
        assert isinstance(result, int)


class TestResolveClassicalScalarNormalization:
    """Scalar bindings should be normalized the same as array elements."""

    def test_numpy_float_scalar_binding_normalized(self):
        """np.float64 binding resolves to native Python float."""
        resolver = ValueResolver()
        theta = Value(type=FloatType(), name="theta")
        bindings = {"theta": np.float64(np.pi / 4)}

        result = resolver.resolve_classical_value(theta, bindings)

        assert type(result) is float
        assert result == np.pi / 4

    def test_numpy_int_scalar_binding_normalized(self):
        """np.int64 binding resolves to native Python int."""
        resolver = ValueResolver()
        n = Value(type=UIntType(), name="n")
        bindings = {"n": np.int64(7)}

        result = resolver.resolve_classical_value(n, bindings)

        assert type(result) is int
        assert result == 7

    def test_bool_binding_preserved(self):
        """bool bindings must not be coerced to int."""
        resolver = ValueResolver()
        flag = Value(type=UIntType(), name="flag")
        bindings = {"flag": True}

        result = resolver.resolve_classical_value(flag, bindings)

        assert result is True

    def test_non_numeric_binding_passes_through(self):
        """Non-numeric bindings (e.g. Hamiltonian-like) are returned as-is."""
        resolver = ValueResolver()
        obs = Value(type=UIntType(), name="obs")
        marker = object()
        bindings = {"obs": marker}

        assert resolver.resolve_classical_value(obs, bindings) is marker
