"""Tests for user-facing resource-estimate serialization."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import sympy as sp

import qamomile.circuit as qm
from qamomile.circuit.estimator._metrics import (
    ResourceTraceNode,
    _ConditionIndicator,
    _ConstraintRange,
    _ResourceConstraint,
)
from qamomile.circuit.estimator._serialization import (
    SymbolRegistry,
    normalize_expression,
    stringify_expression,
)
from qamomile.circuit.estimator._wire import (
    _WireExpressionDecoder,
    resource_estimate_from_wire,
    resource_estimate_to_wire,
)
from qamomile.circuit.estimator.resource_estimator import _CappedRangeSum


def test_call_map_merge_order_is_hash_seed_independent() -> None:
    """Merged call maps and same-name aliases are deterministic across runs."""
    script = """
import json
import sympy as sp
import qamomile.circuit as qm

left_symbol = sp.Symbol("k", integer=True, positive=True)
right_symbol = sp.Symbol("k", integer=True, nonnegative=True)
left = qm.ResourceEstimate(
    calls=qm.CallResources(
        calls_by_name={"charlie": 1, "alpha": left_symbol},
    ),
)
right = qm.ResourceEstimate(
    calls=qm.CallResources(
        calls_by_name={"delta": 1, "bravo": right_symbol},
    ),
)
flag = sp.Symbol("flag", integer=True, nonnegative=True)

def payload(estimate):
    serialized_calls = estimate.to_dict()["calls"]["calls_by_name"]
    return {
        "keys": list(serialized_calls),
        "values": serialized_calls,
        "parameters": list(estimate.parameters),
    }

print(json.dumps({
    "conditional": payload(left.conditional(right, flag > 0)),
    "choice": payload(left.choice(right)),
}))
"""
    repository = Path(__file__).resolve().parents[3]
    payloads: list[dict[str, object]] = []
    for seed in ("1", "2", "3", "4"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repository,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        payloads.append(json.loads(completed.stdout))

    assert all(payload == payloads[0] for payload in payloads)
    for composition in payloads[0].values():
        assert isinstance(composition, dict)
        keys = composition["keys"]
        assert isinstance(keys, list)
        assert keys == sorted(keys)


def test_large_quantified_finite_requirement_does_not_ignore_a_pole() -> None:
    """Analytic range validation never proves finiteness across a singularity."""
    index = sp.Dummy("index", integer=True, nonnegative=True)
    requirement = _ResourceConstraint(
        expression=1 / (index - 5000) ** 2,
        minimum=0,
        label="log2 input",
        integer=False,
        minimum_inclusive=False,
        finite=True,
        ranges=(
            _ConstraintRange(
                symbol=index,
                start=sp.Integer(0),
                step=sp.Integer(1),
                iterations=sp.Integer(10000),
            ),
        ),
    )

    with pytest.raises(ValueError, match="validate log2 input exhaustively"):
        requirement.validate()


def test_dummy_normalization_preserves_assumptions_and_expression_structure() -> None:
    """Dummy normalization retains domains across nested symbolic forms."""
    index = sp.Dummy("index", integer=True, nonnegative=True)
    limit = sp.Symbol("limit", integer=True, nonnegative=True)
    nested = sp.Piecewise(
        (
            sp.Max(
                limit,
                sp.Piecewise(
                    (index + 1, sp.Lt(index, limit)),
                    (index + 2, True),
                ),
            ),
            sp.Le(index, limit + 1),
        ),
        (sp.Integer(0), True),
    )

    normalized = normalize_expression(nested)
    (external_index,) = [
        symbol for symbol in normalized.free_symbols if symbol.name == "index"
    ]
    serialized = stringify_expression(nested)
    restored = sp.sympify(
        serialized,
        locals={"index": external_index, "limit": limit},
    )

    assert not normalized.atoms(sp.Dummy)
    assert external_index.is_integer is True
    assert external_index.is_nonnegative is True
    assert "_index" not in serialized
    assert restored == normalized


def test_same_name_dummy_normalization_preserves_distinct_identities() -> None:
    """Normalization assigns aliases instead of collapsing same-name dummies."""
    left = sp.Dummy("total_after_loop", integer=True, nonnegative=True)
    right = sp.Dummy("total_after_loop", integer=True, nonnegative=True)
    expression = sp.Piecewise(
        (sp.Integer(3), sp.Ne(left, right, evaluate=False)),
        (sp.Integer(1), True),
        evaluate=False,
    )

    normalized = normalize_expression(expression)
    serialized = stringify_expression(expression)
    public_symbols = sorted(normalized.free_symbols, key=lambda symbol: symbol.name)

    assert [symbol.name for symbol in public_symbols] == [
        "total_after_loop",
        "total_after_loop__2",
    ]
    assert all(symbol.is_integer is True for symbol in public_symbols)
    assert all(symbol.is_nonnegative is True for symbol in public_symbols)
    assert normalized != 1
    assert "Ne(total_after_loop, total_after_loop__2)" in serialized


def test_symbol_registry_skips_reserved_suffix_names_deterministically() -> None:
    """Generated aliases never steal a spelling declared by another symbol."""
    first = sp.Dummy("item")
    reserved = sp.Symbol("item__2")
    second = sp.Dummy("item")
    registry = SymbolRegistry.from_expressions((first, reserved, second))

    assert registry.name(first) == "item"
    assert registry.name(reserved) == "item__2"
    assert registry.name(second) == "item__3"
    assert SymbolRegistry.from_expressions((first, reserved, second)).stringify(
        first + reserved + second
    ) == registry.stringify(first + reserved + second)


def test_resource_payload_exposes_every_same_name_parameter_identity() -> None:
    """One payload registry keeps colliding parameters distinct and substitutable."""
    left = sp.Dummy("total_after_loop", integer=True, nonnegative=True)
    right = sp.Dummy("total_after_loop", integer=True, nonnegative=True)
    expression = sp.Piecewise(
        (sp.Integer(3), sp.Ne(left, right, evaluate=False)),
        (sp.Integer(1), True),
        evaluate=False,
    )
    estimate = qm.ResourceEstimate(gates=qm.GateResources(total=expression))

    payload = json.loads(json.dumps(estimate.to_dict()))

    assert payload == estimate.to_dict()
    assert list(estimate.parameters) == [
        "total_after_loop",
        "total_after_loop__2",
    ]
    assert payload["parameters"] == {
        "total_after_loop": "total_after_loop",
        "total_after_loop__2": "total_after_loop__2",
    }
    assert payload["gates"]["total"] == (
        "Piecewise((3, Ne(total_after_loop, total_after_loop__2)), (1, True))"
    )
    public_bindings = {
        name: value for name, value in zip(payload["parameters"], (0, 1), strict=True)
    }
    assert estimate.substitute(**public_bindings).gates.total == 3

    partially_bound = estimate.substitute(total_after_loop=0)
    assert partially_bound.to_dict()["parameters"] == {
        "total_after_loop__2": "total_after_loop__2"
    }
    assert partially_bound.substitute(total_after_loop__2=0).gates.total == 1


def test_payload_registry_is_shared_with_quantified_range_symbols() -> None:
    """Requirement expressions and range binders share metric symbol aliases."""
    parameter = sp.Dummy("index", integer=True, nonnegative=True)
    loop_index = sp.Dummy("index", integer=True, nonnegative=True)
    requirement = _ResourceConstraint(
        expression=parameter + loop_index,
        minimum=0,
        label="Indexed width",
        ranges=(
            _ConstraintRange(
                symbol=loop_index,
                start=sp.Integer(0),
                step=sp.Integer(1),
                iterations=parameter,
            ),
        ),
    )
    estimate = qm.ResourceEstimate(
        gates=qm.GateResources(total=parameter),
        _constraints=(requirement,),
    )

    payload = estimate.to_dict()
    serialized_requirement = payload["requirements"][0]

    assert payload["parameters"] == {"index": "index"}
    assert payload["gates"]["total"] == "index"
    assert serialized_requirement["expression"] == "index + index__2"
    assert serialized_requirement["ranges"] == [
        {
            "symbol": "index__2",
            "start": "0",
            "step": "1",
            "iterations": "index",
        }
    ]


def test_resource_dict_uses_one_name_for_quantified_dummy_expressions() -> None:
    """JSON payloads align requirement expressions with their range names."""
    target_index = sp.Dummy(
        "target_index",
        integer=True,
        nonnegative=True,
    )
    iterations = sp.Symbol("iterations", integer=True, nonnegative=True)
    constrained = sp.Piecewise(
        (sp.Max(target_index + 1, iterations), sp.Lt(target_index, iterations)),
        (target_index, True),
    )
    requirement = _ResourceConstraint(
        expression=constrained,
        minimum=0,
        label="Target index",
        expected=sp.Max(target_index, iterations),
        ranges=(
            _ConstraintRange(
                symbol=target_index,
                start=sp.Integer(0),
                step=sp.Integer(1),
                iterations=iterations,
            ),
        ),
    )
    estimate = qm.ResourceEstimate(
        gates=qm.GateResources(total=constrained),
        _constraints=(requirement,),
    )

    payload = json.loads(json.dumps(estimate.to_dict()))
    serialized_gate = payload["gates"]["total"]
    serialized_requirement = payload["requirements"][0]
    external_target = sp.Symbol(
        "target_index",
        integer=True,
        nonnegative=True,
    )
    restored = sp.sympify(
        serialized_requirement["expression"],
        locals={
            "target_index": external_target,
            "iterations": iterations,
        },
    )

    assert "_target_index" not in serialized_gate
    assert "_target_index" not in serialized_requirement["expression"]
    assert "_target_index" not in serialized_requirement["expected"]
    assert serialized_requirement["ranges"][0]["symbol"] == "target_index"
    assert restored == normalize_expression(constrained)


def test_resource_wire_rejects_executable_expression_syntax() -> None:
    """Opaque cost decoding accepts constructors but never Python execution."""
    wire = resource_estimate_to_wire(
        qm.ResourceEstimate(gates=qm.GateResources(total=1))
    )
    wire["gates"]["total"] = "__import__('os').system('false')"

    with pytest.raises(
        ValueError,
        match="outside safe constructors|non-constructor",
    ):
        resource_estimate_from_wire(wire)


def test_resource_wire_rejects_unsupported_expression_before_encoding() -> None:
    """A fixed cost is never persisted when its expression cannot decode."""
    repetitions = sp.Symbol("repetitions", integer=True, nonnegative=True)
    estimate = qm.ResourceEstimate(
        gates=qm.GateResources(total=sp.factorial(repetitions))
    )

    with pytest.raises(ValueError, match="unsupported symbolic constructor"):
        resource_estimate_to_wire(estimate)


def test_resource_wire_round_trips_large_substituted_capped_range_sum() -> None:
    """A specialized large loop remains serializable without eager replay."""
    index = sp.Dummy("index", integer=True, nonnegative=True)
    iterations = sp.Symbol("iterations", integer=True, nonnegative=True)
    work_per_iteration = sp.Symbol(
        "work_per_iteration",
        integer=True,
        nonnegative=True,
    )
    work = _CappedRangeSum(
        sp.Lambda(
            index,
            work_per_iteration * _ConditionIndicator(sp.Gt(index, 0)),
        ),
        sp.Integer(0),
        sp.Integer(1),
        iterations,
        evaluate=False,
    )
    estimate = qm.ResourceEstimate(gates=qm.GateResources(total=work))
    specialized = estimate.substitute(iterations=2000)

    restored = resource_estimate_from_wire(resource_estimate_to_wire(specialized))

    assert restored.gates.total.has(_CappedRangeSum)
    assert set(restored.parameters) == {"work_per_iteration"}
    assert restored.substitute(work_per_iteration=0).gates.total == 0
    assert restored.substitute(work_per_iteration=1).gates.total == 2


def test_resource_wire_round_trips_supported_symbolic_constructors() -> None:
    """Sum, E, and loop-carry functions survive the closed wire format."""
    index = sp.Dummy("index", integer=True, nonnegative=True)
    iterations = sp.Symbol("iterations", integer=True, nonnegative=True)
    loop_carry = sp.Function("loop_carry")
    expression = sp.Sum(loop_carry(index), (index, 0, iterations)) + sp.E
    estimate = qm.ResourceEstimate(gates=qm.GateResources(total=expression))

    restored = resource_estimate_from_wire(resource_estimate_to_wire(estimate))

    assert restored.gates.total.has(sp.Sum)
    assert restored.gates.total.has(sp.E)
    assert "loop_carry" in str(restored.gates.total)
    assert set(restored.parameters) == {"iterations"}


def test_resource_wire_round_trips_extreme_finite_float() -> None:
    """A compact decimal exponent is not charged as mantissa digits."""
    tiny_cost = sp.Float("1e-1300")
    estimate = qm.ResourceEstimate(gates=qm.GateResources(total=tiny_cost))

    restored = resource_estimate_from_wire(resource_estimate_to_wire(estimate))

    assert restored.gates.total == tiny_cost


def test_resource_wire_rejects_caller_scoped_liveness_before_encoding() -> None:
    """Opaque definition costs cannot persist private caller owner mappings."""
    hidden_size = sp.Symbol("hidden_size", integer=True, nonnegative=True)
    estimate = qm.ResourceEstimate(
        _output_sizes={"caller-owner": hidden_size},
        _has_output_summary=True,
    )

    with pytest.raises(ValueError, match="caller-scoped input/output liveness"):
        resource_estimate_to_wire(estimate)


def test_resource_wire_preserves_trace_only_symbol_identity() -> None:
    """Trace guards round-trip even when no metric exposes their symbol."""
    trace_flag = sp.Dummy("trace_flag", integer=True, nonnegative=True)
    estimate = qm.ResourceEstimate(
        trace=ResourceTraceNode(
            name="conditional trace",
            source_kind="test",
            active_when=sp.Gt(trace_flag, 0),
        )
    )

    wire = resource_estimate_to_wire(estimate)
    restored = resource_estimate_from_wire(wire)

    assert "dummy_index=0" in wire["trace"]["nodes"][0]["active_when"]
    assert restored.parameters == {}
    assert restored.explain() == (
        "Resource estimate\n  conditional trace [test] when=trace_flag > 0"
    )


def test_resource_wire_shares_dummy_identity_across_one_payload() -> None:
    """One payload restores the same Dummy identity in every expression."""
    shared = sp.Dummy("shared", integer=True, nonnegative=True)
    estimate = qm.ResourceEstimate(
        gates=qm.GateResources(total=shared),
        trace=ResourceTraceNode(
            name="conditional trace",
            source_kind="test",
            active_when=sp.Gt(shared, 0),
        ),
    )

    restored = resource_estimate_from_wire(resource_estimate_to_wire(estimate))
    (metric_symbol,) = restored.gates.total.free_symbols
    (trace_symbol,) = restored.trace.active_when.free_symbols

    assert isinstance(metric_symbol, sp.Dummy)
    assert metric_symbol is trace_symbol


def test_resource_wire_decodes_legacy_nondeterministic_dummy_indices() -> None:
    """Older payloads with process-assigned Dummy indices remain readable."""
    trace_flag = sp.Dummy("trace_flag", integer=True, nonnegative=True)
    estimate = qm.ResourceEstimate(
        trace=ResourceTraceNode(
            name="conditional trace",
            source_kind="test",
            active_when=sp.Gt(trace_flag, 0),
        )
    )
    wire = resource_estimate_to_wire(estimate)
    wire["trace"]["nodes"][0]["active_when"] = wire["trace"]["nodes"][0][
        "active_when"
    ].replace("dummy_index=0", "dummy_index=987654")
    wire["symbol_aliases"]["trace_flag"] = wire["symbol_aliases"]["trace_flag"].replace(
        "dummy_index=0", "dummy_index=987654"
    )

    restored = resource_estimate_from_wire(wire)

    assert restored.explain() == (
        "Resource estimate\n  conditional trace [test] when=trace_flag > 0"
    )


def test_resource_wire_round_trips_capped_symbolic_loop_work() -> None:
    """A bounded batching sum remains specializable after wire serialization."""
    index = sp.Dummy("index", integer=True, nonnegative=True)
    repetitions = sp.Symbol("repetitions", integer=True, nonnegative=True)
    work = _CappedRangeSum(
        sp.Lambda(index, _ConditionIndicator(sp.Gt(index, 0))),
        sp.Integer(0),
        sp.Integer(1),
        repetitions,
    )
    estimate = qm.ResourceEstimate(gates=qm.GateResources(total=work))

    restored = resource_estimate_from_wire(resource_estimate_to_wire(estimate))

    assert set(restored.parameters) == {"repetitions"}
    assert restored.substitute(repetitions=0).gates.total == 0
    assert restored.substitute(repetitions=1).gates.total == 0
    assert restored.substitute(repetitions=2).gates.total == 1
    assert restored.substitute(repetitions=3).gates.total == 2


def test_separately_decoded_costs_keep_independent_symbol_identities() -> None:
    """Independent fixed-cost payloads do not merge same-named parameters."""
    left_symbol = sp.Dummy("n", integer=True, nonnegative=True)
    right_symbol = sp.Dummy("n", integer=True, nonnegative=True)
    left = qm.ResourceEstimate(
        gates=qm.GateResources(total=left_symbol),
    )
    right = qm.ResourceEstimate(
        gates=qm.GateResources(total=right_symbol),
    )

    restored_left = resource_estimate_from_wire(resource_estimate_to_wire(left))
    restored_right = resource_estimate_from_wire(resource_estimate_to_wire(right))
    combined = restored_left.seq(restored_right)

    assert set(combined.parameters) == {"n", "n__2"}
    assert combined.substitute(n=2, n__2=5).gates.total == 7


def test_separately_decoded_costs_preserve_shared_symbol_identity() -> None:
    """Ordinary Symbols retain one name across independent cost payloads."""
    shared_symbol = sp.Symbol("n", integer=True, nonnegative=True)
    left = qm.ResourceEstimate(
        gates=qm.GateResources(total=shared_symbol),
    )
    right = qm.ResourceEstimate(
        gates=qm.GateResources(total=shared_symbol),
    )

    restored_left = resource_estimate_from_wire(resource_estimate_to_wire(left))
    restored_right = resource_estimate_from_wire(resource_estimate_to_wire(right))
    combined = restored_left.seq(restored_right)

    assert set(combined.parameters) == {"n"}
    assert combined.substitute(n=3).gates.total == 6


def test_shared_wire_stream_preserves_shared_dummy_identity() -> None:
    """One encoder/decoder stream retains a shared identity-only parameter."""
    shared_symbol = sp.Dummy("n", integer=True, nonnegative=True)
    left = qm.ResourceEstimate(gates=qm.GateResources(total=shared_symbol))
    right = qm.ResourceEstimate(gates=qm.GateResources(total=shared_symbol))
    dummy_slots: dict[sp.Dummy, int] = {}
    decoder = _WireExpressionDecoder()

    restored_left = resource_estimate_from_wire(
        resource_estimate_to_wire(left, dummy_slots=dummy_slots),
        decoder=decoder,
    )
    restored_right = resource_estimate_from_wire(
        resource_estimate_to_wire(right, dummy_slots=dummy_slots),
        decoder=decoder,
    )
    (left_symbol,) = restored_left.gates.total.free_symbols
    (right_symbol,) = restored_right.gates.total.free_symbols
    combined = restored_left.seq(restored_right)

    assert isinstance(left_symbol, sp.Dummy)
    assert left_symbol is right_symbol
    assert set(combined.parameters) == {"n"}
    assert combined.substitute(n=3).gates.total == 6
