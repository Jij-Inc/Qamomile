"""Test that resolve_int_value uses UUID-based lookup for BinOp results."""

from qamomile.circuit.ir.types.primitives import UIntType
from qamomile.circuit.ir.value import Value
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
