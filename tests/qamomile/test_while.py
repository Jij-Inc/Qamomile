"""Structured while loops retain ownership and symbolic iteration assumptions."""

import pytest

import qamomile as qm
from qamomile import resources as qr

SOURCE = """
import qamomile as qm
@qm.qkernel
def classical(limit: qm.Param[qm.Int]) -> qm.Int:
    value = 0
    total = 0
    while value < limit:
        value += 1
        if value == 2:
            continue
        if value == 4:
            break
        total += value
    else:
        total += 10
    return total
@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]], rounds: qm.Const[qm.Int]) -> qm.Q[qm.Bits[1]]:
    index = 0
    while index < rounds:
        state = qm.x(state)
        index += 1
    return state
@qm.qkernel
def sample(rounds: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(flip(qm.qubits(1), rounds))
"""


@pytest.mark.parametrize("limit, expected", [(0, 10), (1, 11), (3, 14), (5, 4)])
def test_while_classical_exits_and_persistence(limit, expected):
    """Normal exit runs else; break skips it; continue rechecks the condition."""
    program = qm.build_source(SOURCE).specialize("classical")
    assert (
        qm.evaluate(qm.Program.from_bytes(program.to_bytes()), limit=limit) == expected
    )


@pytest.mark.parametrize("rounds", [0, 3, 4])
def test_while_quantum_reference(rounds):
    """Static conditions expand only at emission and carry one quantum owner."""
    program = qm.build_source(SOURCE).specialize("sample", rounds=rounds)
    assert qm.run(program, on=qm.Reference(), shots=8, seed=7).result.counts == {
        (rounds % 2,): 8
    }


def test_while_symbolic_resources():
    """The unbounded trip count remains substitutable symbolic data."""
    study = qr.logical(qm.build_source(SOURCE)["flip"])
    gates = study.metric(qr.LOGICAL_GATES)
    assert gates.expression is not None
    assert gates.expression.evaluate({"iterations": 3}).numeric() == 3


def test_while_iteration_assumptions_preserve_program_and_relations():
    """Scenarios and asserted upper bounds remain conditional persisted facts."""
    study = qr.logical(qm.build_source(SOURCE)["flip"])
    (symbol,) = study.iteration_symbols
    scenario = study.assume_iterations({symbol: 3})
    bounded = study.bound_iterations({symbol: 5})
    assert scenario.metric(qr.LOGICAL_GATES).numeric_enclosure() == (3, 3)
    assert bounded.metric(qr.LOGICAL_GATES).relation is qr.Relation.UPPER_BOUND
    assert bounded.metric(qr.LOGICAL_GATES).numeric_enclosure() == (None, 5)
    assert scenario.semantic_digest == study.semantic_digest == bounded.semantic_digest
    assert scenario.metric(qr.LOGICAL_GATES).assumptions
    assert qr.ResourceStudy.from_bytes(bounded.to_bytes()) == bounded
    for invalid in (-1, 1.5, True):
        with pytest.raises((ValueError, TypeError)):
            study.assume_iterations({symbol: invalid})
    with pytest.raises(ValueError, match="absent"):
        study.assume_iterations({"missing::iterations": 3})


def test_while_iteration_limits_do_not_promote_lower_bounds():
    """An iteration limit cannot turn a resource lower bound into an upper bound."""
    from dataclasses import replace

    study = qr.logical(qm.build_source(SOURCE)["flip"])
    (symbol,) = study.iteration_symbols
    gates = study.metric(qr.LOGICAL_GATES)
    assert gates.expression is not None
    lower = qr.ResourceValue.lower_bound(qr.LOGICAL_GATES, gates.expression)
    lower_study = replace(
        study,
        metrics=tuple(
            (name, lower if name == qr.LOGICAL_GATES.identifier else value)
            for name, value in study.metrics
        ),
    )
    bounded = lower_study.bound_iterations({symbol: 5}).metric(qr.LOGICAL_GATES)
    assert bounded.relation is qr.Relation.UNKNOWN
    assert bounded.numeric_enclosure()[1] is None
    assert bounded.unknown_contributors
    assert bounded.assumptions


def test_nested_while_and_for_exits():
    """Each structured exit belongs to its nearest enclosing loop."""
    source = """
import qamomile as qm
@qm.qkernel
def nested() -> qm.Int:
    total = 0
    outer = 0
    while outer < 3:
        outer += 1
        inner = 0
        while inner < 4:
            inner += 1
            if inner == 2:
                continue
            if inner == 3:
                break
            total += 1
        else:
            total += 100
    else:
        for i in range(2):
            while True:
                total += 1
                break
    return total
"""
    assert qm.evaluate(qm.build_source(source).specialize("nested")) == 5


def test_while_early_return():
    """A return from a loop body exits the enclosing callable."""
    source = """
import qamomile as qm
@qm.qkernel
def find(limit: qm.Param[qm.Int]) -> qm.Int:
    index = 0
    while index < limit:
        if index == 2:
            return index
        index += 1
    return -1
"""
    program = qm.build_source(source).specialize("find")
    assert qm.evaluate(program, limit=0) == -1
    assert qm.evaluate(program, limit=5) == 2


def test_while_nontermination_is_work_bounded():
    """An accepted loop does not promise termination or bypass execution budgets."""
    source = """
import qamomile as qm
@qm.qkernel
def forever() -> qm.Int:
    while True:
        pass
    return 0
"""
    with pytest.raises(qm.QamomileError, match="budget"):
        qm.evaluate(qm.build_source(source).specialize("forever"))


def test_while_measured_stage_and_dynamic_capability():
    """Measured conditions retain Shot staging and require runtime loop support."""
    source = """
import qamomile as qm
@qm.qkernel
def dynamic() -> qm.Shot[qm.Bool]:
    bit, state = qm.measure_reset(qm.prepare(True))
    while bit:
        bit, state = qm.measure_reset(qm.x(state))
    return qm.measure(state)
"""
    program = qm.build_source(source).specialize("dynamic")
    assert qr.logical(program).iteration_symbols
    with pytest.raises(qm.QamomileError, match="while.*runtime loop"):
        qm.run(program, on=qm.Reference(), shots=1)


def test_while_quantum_launch_condition():
    """Param loops materialize through the native launch boundary."""
    source = SOURCE.replace("qm.Const[qm.Int]", "qm.Param[qm.Int]")
    program = qm.build_source(source).specialize("sample")
    for rounds in (0, 3, 4):
        assert qm.run(
            program, on=qm.Reference(), shots=8, seed=7, parameters={"rounds": rounds}
        ).result.counts == {(rounds % 2,): 8}


def test_while_condition_abi_is_verified():
    """A forged condition cannot omit or reorder its owned carries."""
    from dataclasses import replace

    module = qm.build_source(SOURCE)
    entry = module.function("flip")
    loop = next(op for op in entry.body.operations if op.opcode == "qcore.while")
    condition = loop.regions[0].blocks[0]
    bad_condition = replace(
        condition,
        terminator=replace(
            condition.terminator, operands=condition.terminator.operands[:1]
        ),
    )
    forged = replace(
        loop,
        regions=(replace(loop.regions[0], blocks=(bad_condition,)), *loop.regions[1:]),
    )
    with pytest.raises(qm.VerificationError):
        replace(
            module,
            functions=tuple(
                replace(
                    function,
                    body=replace(
                        function.body,
                        operations=tuple(
                            forged if op is loop else op
                            for op in function.body.operations
                        ),
                    ),
                )
                if function is entry
                else function
                for function in module.functions
            ),
        )


def test_while_call_contexts_do_not_share_actual_counts():
    """Identical formal inputs at separate call sites have distinct loop counts."""
    source = (
        SOURCE
        + """
@qm.qkernel
def twice(rounds: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    state = flip(qm.qubits(1), rounds)
    state = flip(state, rounds)
    return qm.measure(state)
"""
    )
    study = qr.logical(qm.build_source(source)["twice"])
    assert len(study.iteration_symbols) == 2
    scenario = study.assume_iterations(
        dict(zip(study.iteration_symbols, (2, 3), strict=True))
    )
    assert scenario.metric(qr.LOGICAL_GATES).numeric_enclosure() == (5, 5)


def test_nested_while_counts_remain_unknown():
    """A varying inner trip count cannot be priced as one repeated scalar."""
    source = """
import qamomile as qm
@qm.qkernel
def triangular(state: qm.Q[qm.Bits[1]], n: qm.Const[qm.Int]) -> qm.Q[qm.Bits[1]]:
    outer = 0
    while outer < n:
        inner = 0
        while inner < outer:
            state = qm.x(state)
            inner += 1
        outer += 1
    return state
"""
    study = qr.logical(qm.build_source(source)["triangular"])
    assert study.metric(qr.LOGICAL_GATES).unknown_contributors


def test_while_returning_else_closes_all_paths():
    """A returning else arm closes the zero-iteration and continued paths."""
    source = """
import qamomile as qm
@qm.qkernel
def finish(limit: qm.Param[qm.Int]) -> qm.Int:
    index = 0
    while index < limit:
        index += 1
    else:
        return index
"""
    program = qm.build_source(source).specialize("finish")
    assert qm.evaluate(program, limit=0) == 0
    assert qm.evaluate(program, limit=3) == 3
    assert qr.logical(program)


def test_while_preserves_unchanged_const_carries():
    """Param control does not lift an unchanged unrelated Const capture."""
    source = """
import qamomile as qm
@qm.qkernel
def untouched(limit: qm.Param[qm.Int]) -> qm.Int:
    marker = 17
    index = 0
    while index < limit:
        index += 1
        if index == 2:
            break
    return marker
"""
    assert qm.evaluate(qm.build_source(source).specialize("untouched"), limit=5) == 17


def test_while_resource_context_expansion_is_bounded():
    """Fresh invocation symbols cannot turn a compact call DAG into unlimited work."""
    source = """
import qamomile as qm
@qm.qkernel
def leaf() -> qm.Unit:
    while False:
        pass
    return
"""
    previous = "leaf"
    for level in range(13):
        name = f"level_{level}"
        source += f"\n@qm.qkernel\ndef {name}() -> qm.Unit:\n    {previous}()\n    {previous}()\n    return\n"
        previous = name
    with pytest.raises(qm.QamomileError, match="while resource context budget"):
        qr.logical(qm.build_source(source)[previous])


def test_while_missing_native_assumption_capability_fails_closed(monkeypatch):
    """A stale extension cannot silently evaluate assumptions in Python."""
    from qamomile._native import compiler

    study = qr.logical(qm.build_source(SOURCE)["flip"])
    monkeypatch.delattr(compiler, "assume_resource_iterations")
    with pytest.raises(
        qm.CapabilityError, match="iteration assumptions are unavailable"
    ):
        study.assume_iterations({study.iteration_symbols[0]: 3})


def test_while_qiskit_static_execution():
    """The existing SDK adapter receives the same resolved loop circuit."""
    from qamomile.qiskit import Aer

    program = qm.build_source(SOURCE).specialize("sample", rounds=3)
    assert qm.run(program, on=Aer(), shots=8, seed=7).result.counts == {(1,): 8}


@pytest.mark.hugr
def test_while_hugr_static_execution():
    """Direct HUGR emission expands the retained static condition regions."""
    from qamomile.hugr import Selene

    program = qm.build_source(SOURCE).specialize("sample", rounds=3)
    assert qm.run(program, on=Selene(timeout=30.0), shots=8, seed=7).result.counts == {
        (1,): 8
    }


@pytest.mark.parametrize("entry, expected", [("inverse", (0,)), ("controlled", (1, 1))])
def test_while_static_callable_transforms(entry, expected):
    """Resolved while circuits preserve existing adjoint and control semantics."""
    source = (
        SOURCE
        + """
@qm.qkernel
def inverse(rounds: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.adjoint(flip, flip(qm.qubits(1), rounds), rounds))
@qm.qkernel
def controlled(rounds: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[2]]:
    control, target = qm.control(flip, qm.prepare((1,)), qm.qubits(1), rounds)
    return qm.measure(qm.concat(control, target))
"""
    )
    program = qm.build_source(source).specialize(entry, rounds=3)
    assert qm.run(program, on=qm.Reference(), shots=8, seed=7).result.counts == {
        expected: 8
    }


def test_while_changing_inner_range_cost_is_unknown_but_executable():
    """Changing scalar carries preserve partial estimates and correct execution."""
    source = """
import qamomile as qm
@qm.qkernel
def triangle(state: qm.Q[qm.Bits[1]], n: qm.Const[qm.Int]) -> qm.Q[qm.Bits[1]]:
    index = 0
    while index < n:
        for lane in range(index):
            state = qm.x(state)
        index += 1
    return state
@qm.qkernel
def sample() -> qm.Shot[qm.Bits[1]]:
    return qm.measure(triangle(qm.qubits(1), 3))
"""
    module = qm.build_source(source)
    gates = qr.logical(module["triangle"]).metric(qr.LOGICAL_GATES)
    assert gates.unknown_contributors
    assert gates.numeric_enclosure()[1] is None
    assert qm.run(
        module["sample"], on=qm.Reference(), shots=8, seed=7
    ).result.counts == {(1,): 8}
