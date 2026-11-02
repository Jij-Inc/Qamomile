"""Typed inverse definitions preserve stages, call sharing, and recursion."""

import pytest

import qamomile as qm
from qamomile.compilation import OptimizationPipeline
from qamomile.persistence import deserialize, serialize

from .conftest import MULTI_INPUT_CLEANUP_SOURCE, lower_adjoint_calls
from .test_library_compatibility import approximate_roundtrip
from .test_recursion import SOURCE as RECURSIVE_SOURCE

RECURSIVE_INVERSES = (
    RECURSIVE_SOURCE
    + """
@qm.qkernel
def mutual_inverse(n: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.adjoint(mutual_a, qm.qubits(1), n))
@qm.qkernel
def tree_inverse(n: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.adjoint(tree, qm.qubits(1), n))
@qm.qkernel
def shared_inverse() -> qm.Shot[qm.Bits[1]]:
    state = qm.adjoint(recurse, qm.qubits(1), 2)
    state = qm.adjoint(recurse, state, 3)
    return qm.measure(state)
"""
)

STAGED_CLEANUP = (
    MULTI_INPUT_CLEANUP_SOURCE.replace(
        "def combine(\n", "def combine(\n    i: qm.Const[qm.Int],\n"
    )
    .replace(
        "state = qm.h(state, target=0)",
        "state = qm.h(state, target=i // 2)",
    )
    .replace("combine(left, right)", "combine(0, left, right)")
)


NESTED_RETURNS = RECURSIVE_SOURCE.replace(
    "    else:\n        return recurse(qm.x(state), rounds - 1)",
    "    elif rounds == 1:\n        return qm.x(state)\n"
    "    return recurse(qm.x(state), rounds - 1)",
    1,
)


def operations(block):
    """Walk retained regions without traversing call edges."""
    for operation in block.operations:
        yield operation
        for region in operation.regions:
            for child in region.blocks:
                yield from operations(child)


def test_inverse_helpers_preserve_formal_stages_and_multiple_owned_inputs():
    """A nonleading, multiple-owner call keeps a declaration-order inverse ABI."""
    program = qm.build_source(STAGED_CLEANUP).specialize("multi_input_cleanup")
    lowered = lower_adjoint_calls(program)
    forward = next(f for f in lowered.functions if f.symbol == "module.combine")
    inverse = next(
        f for f in lowered.functions if f.symbol == "module.combine.__qamomile_adjoint"
    )
    assert inverse.parameters[0].value.type == forward.result_type
    assert inverse.parameters[1].value.type == forward.parameters[0].value.type
    assert inverse.result_type.elements == tuple(
        p.value.type for p in forward.parameters[1:]
    )
    assert any(o.opcode == "classical.floordiv" for o in operations(inverse.body))
    assert "qcore.adjoint_call" not in lowered.inspect()


@pytest.mark.parametrize("entry", ["mutual_inverse", "tree_inverse", "shared_inverse"])
def test_inverse_call_graph_is_shared_and_recursive(entry):
    """Recursive edges survive once per helper, independently of bound arguments."""
    module = qm.build_source(RECURSIVE_INVERSES)
    bindings = {} if entry == "shared_inverse" else {"n": 10_000}
    program = module.specialize(entry, **bindings)
    lowered = lower_adjoint_calls(program)
    helpers = {
        f.symbol: f for f in lowered.functions if ".__qamomile_adjoint" in f.symbol
    }
    expected = {
        "mutual_inverse": {"mutual_a", "mutual_b"},
        "tree_inverse": {"tree"},
        "shared_inverse": {"recurse"},
    }[entry]
    assert set(helpers) == {f"module.{name}.__qamomile_adjoint" for name in expected}
    edges = {
        (name, operation.attribute("callee"))
        for name, helper in helpers.items()
        for operation in operations(helper.body)
        if operation.opcode == "qcore.call"
    }
    assert edges
    assert all(target in helpers for _, target in edges)
    assert sum(len(tuple(operations(f.body))) for f in helpers.values()) < 50
    assert "qcore.adjoint_call" not in lowered.inspect()
    # Repeated lowering is canonical, including worklist order and fresh symbols.
    assert lowered.inspect() == lower_adjoint_calls(program).inspect()
    original = {f.symbol: f for f in program.functions}
    for f in lowered.functions:
        if f.symbol in original:
            # Source branches are not normalized merely because another callable
            # needs an inverse. The forward recursion still has terminal returns.
            if f.symbol.removeprefix("module.") in expected:
                assert f.body == original[f.symbol].body


def test_inverse_normalizes_nested_and_fallthrough_terminal_returns():
    """A nested exit joins owned successors without moving branch evaluation."""
    program = qm.build_source(NESTED_RETURNS).specialize("inverse", rounds=3)
    lowered = lower_adjoint_calls(program)
    helper = next(
        f for f in lowered.functions if f.symbol == "module.recurse.__qamomile_adjoint"
    )
    conditionals = [o for o in operations(helper.body) if o.opcode == "qcore.if"]
    assert len(conditionals) == 2
    assert all(
        region.entry.terminator.kind == "yield"
        for operation in conditionals
        for region in operation.regions
    )


@pytest.mark.parametrize(
    "optimizer", [None, OptimizationPipeline.multilevel(family=False)]
)
def test_recursive_inverse_pair_does_not_require_termination_proof(optimizer):
    """Inverse derivation is finite, and algebraic cancellation needs no termination proof."""
    source = """import qamomile as qm
@qm.qkernel
def forever(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return forever(state)
@qm.qkernel
def sample() -> qm.Shot[qm.Bits[1]]:
    state = forever(qm.qubits(1))
    return qm.measure(qm.adjoint(forever, state))
"""
    program = qm.build_source(source).specialize("sample")
    lowered = lower_adjoint_calls(program)
    assert len(lowered.functions) == 3
    assert qm.run(
        program, on=qm.Reference(), optimizer=optimizer, shots=1
    ).result.counts == {(0,): 1}


@pytest.mark.hugr
@pytest.mark.parametrize(
    "optimizer",
    [
        None,
        OptimizationPipeline(),
        OptimizationPipeline.multilevel(family=False, realization=None, circuit=None),
    ],
    ids=["default", "empty", "multilevel"],
)
def test_hugr_executes_retained_inverse_functions(optimizer):
    """Every preset executes retained QFT, recursive, and multi-input inverses."""
    pytest.importorskip("hugr")
    from qamomile.hugr import Selene

    module = qm.build_source(RECURSIVE_INVERSES)
    witnesses = [
        (approximate_roundtrip.specialize(), (1, 0, 1, 1, 0)),
        (module.specialize("inverse", rounds=3), (0,)),
        (module.specialize("mutual_inverse", n=3), (0,)),
        (module.specialize("tree_inverse", n=3), (1,)),
        (module.specialize("shared_inverse"), (1,)),
        (qm.build_source(NESTED_RETURNS).specialize("inverse", rounds=3), (0,)),
        (qm.build_source(STAGED_CLEANUP).specialize("multi_input_cleanup"), (1,)),
    ]
    endpoint = Selene(timeout=30.0)
    for program, expected in witnesses:
        restored = qm.Program.from_bytes(program.to_bytes())
        artifact = qm.compile(restored, to=endpoint.target, optimizer=optimizer)
        artifact = deserialize(serialize(artifact))
        assert qm.execute(artifact, on=endpoint, shots=4, seed=7).counts == {
            expected: 4
        }
