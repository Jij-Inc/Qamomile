"""Test UUID-safe emit-time threading of loop RegionArgs."""

from qamomile.circuit.ir.operation.control_flow import ForOperation, RegionArg
from qamomile.circuit.ir.types.primitives import UIntType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.emit_context import EmitContext
from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
    _advance_region_args,
    _publish_region_results,
    _seed_region_args,
)
from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
    ValueResolver,
)


class _StubEmitPass:
    """Provide the resolver surface used by RegionArg emission helpers."""

    def __init__(self) -> None:
        """Create a stub with no backend runtime parameters."""
        self._resolver = ValueResolver()

    def _get_or_create_parameter(self, key: str, value_uuid: str) -> object:
        """Fail if a test unexpectedly takes the symbolic parameter path.

        Args:
            key (str): Requested backend parameter name.
            value_uuid (str): Requested IR value identity.

        Returns:
            object: Never returns.

        Raises:
            AssertionError: Always; these tests require concrete carries.
        """
        raise AssertionError(
            f"unexpected backend parameter creation for {key!r} ({value_uuid})"
        )


def _value(name: str = "shared") -> Value:
    """Create a symbolic UInt value with a deliberately reusable label.

    Args:
        name (str): Display label. Defaults to ``"shared"``.

    Returns:
        Value: Fresh symbolic UInt value.
    """
    return Value(type=UIntType(), name=name)


def _loop(*args: RegionArg) -> ForOperation:
    """Create a minimal loop carrying the supplied region arguments.

    Args:
        *args (RegionArg): Region arguments attached to the loop.

    Returns:
        ForOperation: Minimal loop whose results match the carried results.
    """
    return ForOperation(region_args=args, results=[arg.result for arg in args])


def test_region_arg_seed_and_publish_use_every_ssa_identity() -> None:
    """Init, block-arg, yielded, and result identities stay UUID-scoped."""
    init_a, init_b = _value(), _value()
    block_a, block_b = _value(), _value()
    result_a, result_b = _value(), _value()
    arg_a = RegionArg("a", init_a, block_a, block_b, result_a)
    arg_b = RegionArg("b", init_b, block_b, block_a, result_b)
    loop = _loop(arg_a, arg_b)

    bindings = EmitContext()
    bindings.set_value(init_a.uuid, 1)
    bindings.set_value(init_b.uuid, 2)
    bindings["shared"] = 999

    carried = _seed_region_args(_StubEmitPass(), loop, bindings)
    assert carried == {block_a.uuid: 1, block_b.uuid: 2}

    # Both yields are resolved before either block argument is updated, so a
    # simultaneous swap cannot accidentally observe a partially-updated map.
    loop_bindings = bindings.copy()
    loop_bindings.set_value(block_a.uuid, carried[block_a.uuid])
    loop_bindings.set_value(block_b.uuid, carried[block_b.uuid])
    _advance_region_args(_StubEmitPass(), loop, carried, loop_bindings)
    assert carried == {block_a.uuid: 2, block_b.uuid: 1}

    _publish_region_results(loop, carried, bindings)
    assert bindings[result_a.uuid] == 2
    assert bindings[result_b.uuid] == 1
    assert bindings._values[result_a.uuid] == 2
    assert bindings._values[result_b.uuid] == 1
    assert bindings["shared"] == 999


def test_zero_trip_region_args_publish_initial_values() -> None:
    """Publishing immediately after seeding implements zero-trip passthrough."""
    init = _value("carry").with_const(7)
    block_arg = _value("carry")
    yielded = _value("carry")
    result = _value("carry")
    loop = _loop(RegionArg("carry", init, block_arg, yielded, result))
    bindings = EmitContext()

    carried = _seed_region_args(_StubEmitPass(), loop, bindings)
    _publish_region_results(loop, carried, bindings)

    assert bindings[result.uuid] == 7
    assert "carry" not in bindings
