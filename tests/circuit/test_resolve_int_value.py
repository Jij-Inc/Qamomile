"""Test that resolve_int_value uses UUID-based lookup for BinOp results."""

import numpy as np

from qamomile.circuit.ir.types.primitives import FloatType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.ir.value import ArrayValue
from qamomile.circuit.ir.types.primitives import UIntType
from qamomile.circuit.transpiler.passes.emit_base import ValueResolver


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

        val = Value(type=UIntType(), name="const", params={"const": 7})
        bindings: dict[str, int] = {}

        result = resolver.resolve_int_value(val, bindings)
        assert result == 7

    def test_dimension_name_resolves_from_bound_vector(self):
        """Synthetic values like hi_dim0 resolve from the bound array length."""
        resolver = ValueResolver()

        val = Value(type=UIntType(), name="hi_dim0")
        bindings = {"hi": np.array([0.1, 0.2, 0.3])}

        result = resolver.resolve_int_value(val, bindings)
        assert result == 3

    def test_dimension_name_resolves_nonzero_matrix_axis(self):
        """Synthetic dimension names also work for higher-rank arrays."""
        resolver = ValueResolver()

        val = Value(type=UIntType(), name="edges_dim1")
        bindings = {"edges": np.zeros((3, 2), dtype=np.uint64)}

        result = resolver.resolve_int_value(val, bindings)
        assert result == 2

    def test_parent_array_shape_value_resolves_first_dimension(self):
        """Pre-inline shape values still resolve through their parent array."""
        resolver = ValueResolver()

        array = ArrayValue(type=FloatType(), name="weights")
        val = Value(type=UIntType(), name="shape_dim", parent_array=array)
        bindings = {"weights": np.array([1.0, 2.0, 3.0, 4.0])}

        result = resolver.resolve_int_value(val, bindings)
        assert result == 4
