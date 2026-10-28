"""Tests for static explicit-region interface analysis."""

import ast
import textwrap

import qamomile.circuit as qmc
from qamomile.circuit.frontend.region_analysis import (
    RegionLocation,
    RegionSignature,
    analyze_region_signatures,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    IfOperation,
    WhileOperation,
)


def _analyze(source: str) -> dict[RegionLocation, RegionSignature]:
    """Parse one test function and return its region signatures.

    Args:
        source (str): Source containing one function definition.

    Returns:
        dict[RegionLocation, RegionSignature]: Analyzed structured-region
            interfaces keyed by source location.
    """
    module = ast.parse(textwrap.dedent(source))
    definition = module.body[0]
    assert isinstance(definition, ast.FunctionDef)
    return analyze_region_signatures(definition)


def test_array_subscript_update_becomes_loop_carried_state() -> None:
    """Persistent array updates create explicit iter-args and results."""
    signatures = _analyze(
        """
        def kernel(bits, precision):
            for index in range(precision):
                bits[index] = measure(index)
            return bits
        """
    )

    loop = signatures[RegionLocation("for", 3, 4)]
    assert loop.carried == ("bits",)
    assert loop.results == ("bits",)
    assert loop.captures == ()


def test_nested_loop_signatures_are_independent() -> None:
    """Nested loops expose their own carries and read-only captures."""
    signatures = _analyze(
        """
        def kernel(bits, state, factors, precision):
            for round_index in range(precision):
                state = modmul(state, factors[round_index])
                for previous_index in range(round_index):
                    if bits[previous_index]:
                        state = feedback(state, round_index, previous_index)
                bits[round_index] = measure(state)
            return bits
        """
    )

    outer = signatures[RegionLocation("for", 3, 4)]
    assert set(outer.carried) == {"bits", "state"}
    assert outer.results == ("bits",)
    assert outer.captures == ("factors",)

    inner = signatures[RegionLocation("for", 5, 8)]
    assert inner.carried == ("state",)
    assert set(inner.captures) == {"bits", "round_index"}


def test_if_signature_routes_updated_values_and_captures() -> None:
    """Branch signatures distinguish merged values from read-only inputs."""
    signatures = _analyze(
        """
        def kernel(condition, state, angle):
            if condition:
                state = rotate(state, angle)
            return state
        """
    )

    branch = signatures[RegionLocation("if", 3, 4)]
    assert branch.carried == ("state",)
    assert branch.results == ("state",)
    assert branch.captures == ("angle",)
    assert set(branch.inputs) == {"state", "angle"}


def test_while_signature_carries_condition_and_body_updates() -> None:
    """While analysis shares explicit state between condition and body."""
    signatures = _analyze(
        """
        def kernel(bit, state):
            while bit:
                state = update(state)
                bit = measure(state)
            return state
        """
    )

    loop = signatures[RegionLocation("while", 3, 4)]
    assert set(loop.carried) == {"bit", "state"}
    assert loop.results == ("state",)
    assert loop.captures == ()


def test_range_lowering_uses_explicit_bindings_without_frame_locals() -> None:
    """Generated range code routes carries without ``locals()`` probes."""

    @qmc.qkernel
    def accumulate() -> qmc.UInt:
        """Accumulate two symbolic loop indices."""
        total = qmc.uint(0)
        for index in qmc.range(2):
            total = total + index
        return total

    generated_names = set(accumulate.func.__code__.co_names)
    loop = next(
        operation
        for operation in accumulate.build().operations
        if isinstance(operation, ForOperation)
    )

    assert "locals" not in generated_names
    assert "explicit_loop_bindings" in generated_names
    assert [argument.var_name for argument in loop.region_args] == ["total"]


def test_while_lowering_builds_multiple_explicit_state_slots() -> None:
    """Non-condition while state is represented by multiple region args."""

    @qmc.qkernel
    def accumulate_while() -> qmc.UInt:
        """Build a measurement loop with two carried counters."""
        bit = qmc.measure(qmc.qubit("condition"))
        count = qmc.uint(0)
        doubled = qmc.uint(0)
        while bit:
            count = count + 1
            doubled = doubled + 2
            bit = qmc.measure(qmc.qubit("update"))
        return count + doubled

    loop = next(
        operation
        for operation in accumulate_while.build().operations
        if isinstance(operation, WhileOperation)
    )

    assert [argument.var_name for argument in loop.region_args] == [
        "count",
        "doubled",
    ]
    assert [argument.result for argument in loop.region_args] == loop.results
    assert not loop.loop_carried_rebinds


def test_classical_array_store_builds_explicit_state_slot() -> None:
    """Persistent element stores carry the current classical array version."""

    @qmc.qkernel
    def fill(values: qmc.Vector[qmc.UInt]) -> qmc.Vector[qmc.UInt]:
        """Fill two classical elements with their loop indices."""
        for index in qmc.range(2):
            values[index] = index
        return values

    loop = next(
        operation
        for operation in fill.build(values=[0, 0]).operations
        if isinstance(operation, ForOperation)
    )

    assert [argument.var_name for argument in loop.region_args] == ["values"]
    assert [argument.result for argument in loop.region_args] == loop.results
    assert not loop.loop_carried_rebinds


def test_frontend_materializes_captures_before_normalization() -> None:
    """Decoration-time signatures populate loop and branch captures directly."""

    @qmc.qkernel
    def captured_regions(limit: qmc.UInt) -> qmc.UInt:
        """Read outer values from a loop and a runtime branch."""
        enabled = qmc.measure(qmc.qubit("condition"))
        total = qmc.uint(0)
        for _ in qmc.range(2):
            total = total + limit
        if enabled:
            total = total + limit
        return total

    operations = captured_regions.build().operations
    loop = next(
        operation for operation in operations if isinstance(operation, ForOperation)
    )
    branch = next(
        operation for operation in operations if isinstance(operation, IfOperation)
    )

    assert [value.name for value in loop.captures] == ["limit"]
    assert [value.name for value in branch.true_captures] == ["total", "limit"]
    assert [value.name for value in branch.false_captures] == ["total", "limit"]
