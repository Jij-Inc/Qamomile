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
from qamomile.circuit.estimator._serialization import (
    SymbolRegistry,
    normalize_expression,
    stringify_expression,
)
from qamomile.circuit.estimator.resource_estimator import (
    _ConstraintRange,
    _ResourceConstraint,
)


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
