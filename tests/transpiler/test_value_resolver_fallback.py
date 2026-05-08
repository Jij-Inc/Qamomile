"""Unit tests for Layer 4: silent 0 fallback removal in ``ValueResolver``.

Before Layer 4, ``ValueResolver.resolve_int_value`` returned ``0`` for any
Value it could not resolve — constant, parameter, or otherwise. That
fallback silently masked unresolved symbolic shape dims (``gamma_dim0``)
reaching emit time, producing empty loops and compiled circuits that
looked correct but ignored their parameters.

Layer 4 is a defensive safety net: in normal pipeline operation the
upstream ``SymbolicShapeValidationPass`` (Layer 3) should catch
unresolved shape dims before they reach emit. These tests exercise
``resolve_int_value`` directly so the contract is exercised regardless
of upstream behaviour.
"""

from qamomile.circuit.ir.types.primitives import UIntType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
    ValueResolver,
)


class TestResolveIntValueFallback:
    def test_unbound_symbolic_value_returns_none(self):
        """Unresolvable symbolic Values must propagate as ``None``."""
        resolver = ValueResolver()
        symbolic = Value(type=UIntType(), name="gamma_dim0")
        assert resolver.resolve_int_value(symbolic, bindings={}) is None

    def test_bound_parameter_still_resolves(self):
        """Sanity check: bound parameter values still resolve correctly."""
        resolver = ValueResolver(parameters={"p"})
        param = Value(type=UIntType(), name="p").with_parameter("p")
        assert resolver.resolve_int_value(param, bindings={"p": 3}) == 3

    def test_constant_value_still_resolves(self):
        """Sanity check: constant Values still resolve correctly."""
        resolver = ValueResolver()
        const = Value(type=UIntType(), name="dim_0").with_const(5)
        assert resolver.resolve_int_value(const, bindings={}) == 5

    def test_plain_python_int_still_resolves(self):
        """Sanity check: plain Python ``int`` passes through."""
        resolver = ValueResolver()
        assert resolver.resolve_int_value(7, bindings={}) == 7

    def test_unbound_value_not_in_bindings_returns_none(self):
        """Non-constant non-parameter Values without bindings → ``None``."""
        resolver = ValueResolver()
        symbolic = Value(type=UIntType(), name="uint_tmp")
        assert resolver.resolve_int_value(symbolic, bindings={}) is None
