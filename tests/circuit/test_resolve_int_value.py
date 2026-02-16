"""resolve_int_value が BinOp 結果の UUID で正しく lookup することを検証"""

import pytest
from qamomile.circuit.ir.value import Value
from qamomile.circuit.ir.types.primitives import UIntType
from qamomile.circuit.transpiler.passes.emit_base import ValueResolver


class TestResolveIntValueUuidLookup:
    """resolve_int_value が UUID を name より優先して lookup することを検証"""

    def test_uuid_lookup_prevents_name_collision(self):
        """同じ name を持つ複数の BinOp 結果で、UUID を使って正しい値を取得できることを検証"""
        resolver = ValueResolver()

        # BinOp 結果1: n // 2 = 4 (uuid_a, name="uint_tmp")
        val_a = Value(type=UIntType(), name="uint_tmp")
        # BinOp 結果2: n * 2 = 16 (uuid_b, name="uint_tmp")
        val_b = Value(type=UIntType(), name="uint_tmp")

        # _evaluate_binop が行う格納を再現
        bindings: dict[str, int] = {}
        bindings[val_a.uuid] = 4
        bindings[val_a.name] = 4  # "uint_tmp" = 4
        bindings[val_b.uuid] = 16
        bindings[val_b.name] = 16  # "uint_tmp" = 16 (上書き!)

        # val_a で lookup → UUID なら 4、name なら 16
        result = resolver.resolve_int_value(val_a, bindings)
        assert result == 4, (
            f"Expected 4 (n//2), got {result}. "
            f"resolve_int_value used name-based lookup and got n*2=16 instead"
        )

    def test_uuid_lookup_falls_back_to_name(self):
        """UUID が bindings にない場合は name にフォールバックすることを検証"""
        resolver = ValueResolver()

        val = Value(type=UIntType(), name="some_var")
        bindings: dict[str, int] = {"some_var": 42}
        # UUID は bindings にない

        result = resolver.resolve_int_value(val, bindings)
        assert result == 42

    def test_constant_value_unaffected(self):
        """定数 Value は UUID/name lookup に関係なく正しく解決されることを検証"""
        resolver = ValueResolver()

        val = Value(type=UIntType(), name="const", params={"const": 7})
        bindings: dict[str, int] = {}

        result = resolver.resolve_int_value(val, bindings)
        assert result == 7
