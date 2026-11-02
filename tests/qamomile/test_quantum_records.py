"""Quantum NamedTuple records transfer ownership by complete destructuring."""

from __future__ import annotations

import pytest

import qamomile as qm

SOURCE = """
from typing import NamedTuple
import qamomile as qm

class Pair(NamedTuple):
    first: qm.Q[qm.Bits[1]]
    second: qm.Q[qm.Bits[2]]

@qm.qkernel
def flip_first(pair: Pair) -> Pair:
    first, second = pair
    first = qm.x(first)
    return Pair(second=second, first=first)

@qm.qkernel
def state() -> qm.Q[qm.Bits[3]]:
    pair = Pair(qm.prepare((0,)), qm.prepare((1, 0)))
    first, second = flip_first(pair)
    return qm.concat(first, second)

@qm.qkernel
def sample() -> qm.Shot[qm.Bits[3]]:
    return qm.measure(state())

@qm.qkernel
def inverse() -> qm.Shot[qm.Bits[3]]:
    pair = Pair(qm.prepare((0,)), qm.prepare((1, 0)))
    first, second = qm.adjoint(flip_first, flip_first(pair))
    return qm.measure(qm.concat(first, second))

@qm.qkernel
def controlled() -> qm.Shot[qm.Bits[4]]:
    control = qm.prepare((1,))
    pair = Pair(qm.prepare((0,)), qm.prepare((1, 0)))
    control, pair = qm.control(flip_first, control, pair)
    first, second = pair
    return qm.measure(qm.concat(control, first, second))

class Single(NamedTuple):
    state: qm.Q[qm.Bool]

@qm.qkernel
def unary() -> qm.Shot[qm.Bool]:
    record = Single(qm.prepare(False))
    state, = record
    product = (qm.x(state),)
    state, = product
    return qm.measure(state)
"""


def exercise_quantum_record_backend(endpoint, *, seed=7):
    """Compile and execute the same asymmetric record source on every SDK."""
    module = qm.build_source(SOURCE)
    program = qm.Program.from_bytes(module.specialize("sample").to_bytes())
    artifact = qm.compile(program, to=endpoint.target)
    assert qm.execute(artifact, on=endpoint, shots=8, seed=seed).counts == {
        (1, 1, 0): 8
    }
    assert (
        qm.expectation(module["state"], qm.Z(0), on=endpoint, shots=8, seed=seed).value
        == -1.0
    )
    assert qm.run(module["inverse"], on=endpoint, shots=8, seed=seed).result.counts == {
        (0, 1, 0): 8
    }
    assert qm.run(
        module["controlled"], on=endpoint, shots=8, seed=seed
    ).result.counts == {(1, 1, 1, 0): 8}
    assert qm.run(module["unary"], on=endpoint, shots=8, seed=seed).result.counts == {
        (1,): 8
    }


def test_quantum_record_reference_execution_and_persistence():
    """Whole-record calls, transforms, and persisted programs share one meaning."""
    exercise_quantum_record_backend(qm.Reference())


@pytest.mark.parametrize("access", ["pair.first", "pair[0]"])
def test_quantum_record_requires_complete_destructuring(access):
    """Named and indexed projections cannot introduce implicit partial moves."""
    source = SOURCE.replace(
        "    first, second = pair\n    first = qm.x(first)",
        f"    first = qm.x({access})\n    second = pair.second",
    )
    with pytest.raises(qm.FrontendError, match="Q5099.*destructured as a whole"):
        qm.build_source(source)


@pytest.mark.parametrize(
    "body,result,diagnostic",
    [
        (
            "first, second = pair\n    a, b = pair\n    return Pair(qm.x(first), second)",
            "Pair",
            "Q5020",
        ),
        ("first, second = pair\n    return Pair(first, first)", "Pair", "Q2001|Q2002"),
        (
            "alias = pair\n    first, second = pair\n    a, b = alias\n    return Pair(first, second)",
            "Pair",
            "Q5020",
        ),
    ],
)
def test_quantum_record_ownership_rejections(body, result, diagnostic):
    """Consumed source names and duplicate native owners are rejected."""
    source = (
        """
from typing import NamedTuple
import qamomile as qm
class Pair(NamedTuple):
    first: qm.Q[qm.Bool]
    second: qm.Q[qm.Bool]
@qm.qkernel
def invalid(pair: Pair) -> RESULT:
    """.replace("RESULT", result)
        + body
        + "\n"
    )
    with pytest.raises(qm.QamomileError, match=diagnostic):
        qm.build_source(source).specialize("invalid")


def test_nested_quantum_records_with_classical_metadata_and_control_flow():
    """Record owners follow nested structure through branches and loop carries."""
    source = (
        SOURCE
        + """
class Tagged(NamedTuple):
    pair: Pair
    flips: qm.Int = 2

@qm.qkernel
def nested(enabled: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[3]]:
    record = Tagged(Pair(qm.prepare((0,)), qm.prepare((1, 0))))
    pair, flips = record
    for _ in range(flips):
        pair = flip_first(pair)
    if enabled:
        pair = flip_first(pair)
    first, second = pair
    return qm.measure(qm.concat(first, second))
"""
    )
    program = qm.build_source(source).specialize("nested")
    artifact = qm.compile(
        qm.Program.from_bytes(program.to_bytes()), to=qm.Reference().target
    )
    for enabled, bits in [(False, (0, 1, 0)), (True, (1, 1, 0))]:
        assert qm.execute(
            artifact, on=qm.Reference(), parameters={"enabled": enabled}, shots=8
        ).counts == {bits: 8}


def test_single_element_classical_records_and_tuples_keep_element_types():
    """The shared unary unpacking boundary also preserves classical behavior."""
    source = """
from typing import NamedTuple
import qamomile as qm
class Single(NamedTuple):
    number: qm.Int
@qm.qkernel
def sample(number: qm.Param[qm.Int]) -> qm.Shot[qm.Bool]:
    record = Single(number)
    value, = record
    product = (value + 1,)
    value, = product
    return qm.measure(qm.prepare(value == 4))
"""
    assert qm.run(
        qm.build_source(source)["sample"],
        on=qm.Reference(),
        parameters={"number": 3},
        shots=8,
    ).result.counts == {(1,): 8}


def test_quantum_record_defaults_require_an_explicit_owner():
    """A default cannot silently create or reuse a quantum field owner."""
    source = """
from typing import NamedTuple
import qamomile as qm
class Single(NamedTuple):
    state: qm.Q[qm.Bool] = False
"""
    with pytest.raises(qm.FrontendError, match="Q5099.*cannot have defaults"):
        qm.build_source(source)
