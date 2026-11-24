"""A while diagram exposes its test, loop-carried values and normal exit."""

import re
from xml.etree import ElementTree as ET

import pytest

import qamomile as qm
from qamomile.visualization import draw

_COUNTED = """
import qamomile as qm
@qm.qkernel
def flip(rounds: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    i = 0
    while i < rounds:
        state = qm.x(state)
        i += 1
    else:
        state = qm.h(state)
    return qm.measure(state)
"""
_MEASURED = """
import qamomile as qm
@qm.qkernel
def feedback() -> qm.Shot[qm.Bit]:
    bit, state = qm.measure_reset(qm.prepare(True))
    while bit:
        bit, state = qm.measure_reset(qm.x(state))
    return qm.measure(state)
"""


def test_while_body_shows_predicate_next_carries_and_separate_final_values():
    """One symbolic body exposes its quantum output without claiming it always ran."""
    program = qm.build_source(_COUNTED).specialize("flip", rounds=3)
    body = draw(program)
    loop = next(n for n in body.nodes if n.opcode == "qcore.while")
    guard, iteration, exit_state, completion = loop.children
    assert guard.outputs[1:] == iteration.inputs
    assert set(iteration.outputs) <= set(exit_state.inputs)
    assert set(guard.outputs[1:]) <= set(exit_state.inputs)
    assert exit_state.outputs == completion.inputs
    assert completion.inputs != loop.inputs
    assert completion.inputs != iteration.outputs
    root = ET.fromstring(body.to_svg())
    ns = "{http://www.w3.org/2000/svg}"
    flip = root.find(f'.//{ns}g[@data-opcode="quantum.x"]')
    assert flip is not None
    assert any(
        pin.get("data-value") == str(iteration.outputs[-1])
        for pin in flip.findall(f"{ns}g")
    )
    assert (
        root.find(f'.//{ns}g[@data-value="{guard.outputs[0]}"][@data-edge="dataflow"]')
        is not None
    )
    assert root.find(f'.//{ns}g[@data-edge="control-summary"]') is not None


@pytest.mark.parametrize("rounds", [0, 3])
def test_counted_while_draws_each_taken_body_then_else(rounds):
    """The final failed check still reaches else, including for a zero-trip loop."""
    program = qm.build_source(_COUNTED).specialize("flip", rounds=rounds)
    assert not next(
        n for n in draw(program, loops="folded").nodes if n.kind == "loop"
    ).children
    expanded = draw(program, loops="expanded")
    loop = next(n for n in expanded.nodes if n.opcode == "qcore.while")
    assert [n.opcode for n in loop.children] == [
        *["while.condition", "loop.iteration"] * rounds,
        "while.condition",
        "loop.completion",
    ]
    for index in range(rounds):
        guard, iteration, next_check = loop.children[2 * index : 2 * index + 3]
        assert guard.outputs[1:] == iteration.inputs
        assert iteration.outputs == next_check.inputs
    assert loop.children[-2].outputs[1:] == loop.children[-1].inputs
    svg = expanded.to_svg()
    assert svg.count('data-opcode="quantum.x"') == rounds
    assert svg.count('data-opcode="quantum.h"') == 1


def test_measurement_dependent_while_rejects_expansion_and_exposes_its_exit():
    """Measured decisions have visible symbolic control and an unavailable expansion."""
    program = qm.build_source(_MEASURED).specialize("feedback")
    diagram = draw(program)
    loop = next(n for n in diagram.nodes if n.opcode == "qcore.while")
    assert [n.opcode for n in loop.children] == [
        "while.condition",
        "loop.body",
        "while.exit_state",
        "loop.completion",
    ]
    assert "first check: bit" in diagram.to_svg()
    with pytest.raises(qm.CapabilityError, match="Q7402.*while"):
        draw(program, loops="expanded")
    with pytest.raises(qm.CapabilityError, match="Q7402.*while"):
        draw(program, loops="expanded", wires="qubits")
    assert re.search(
        r'<button[^>]+data-loops="expanded"[^>]+disabled', diagram.to_html()
    )


def test_while_expansion_stops_at_budget_or_control_exit():
    """Unbounded and early-exiting bodies cannot yield a misleading finite diagram."""
    large = qm.build_source(_COUNTED).specialize("flip", rounds=10**12)
    with pytest.raises(qm.CapabilityError, match="Q7401"):
        draw(large, loops="expanded", max_nodes=100)
    source = _COUNTED.replace(
        "        state = qm.x(state)",
        "        if i == 1:\n            break\n        state = qm.x(state)",
    )
    early = qm.build_source(source).specialize("flip", rounds=3)
    assert 'data-opcode="break"' in draw(early).to_svg()
    with pytest.raises(qm.CapabilityError, match="Q7402"):
        draw(early, loops="expanded")


def test_operation_lines_name_the_compiled_source_for_text_and_linked_kernels():
    """Labels identify the exact input text used for the operation's line number."""
    direct = qm.build_source(_COUNTED).specialize("flip", rounds=3)
    line = _COUNTED.splitlines().index("        state = qm.x(state)") + 1
    xml = ET.fromstring(draw(direct).to_svg())
    gate = xml.find('.//{*}g[@data-opcode="quantum.x"]')
    assert gate is not None
    assert f"compiled source line {line}" in tuple(gate.itertext())

    from tests.qamomile.test_stdlib import qaoa_sample

    boxed = draw(qaoa_sample.specialize(hamiltonian=qm.Z(0)), inline_depth=0)
    call = next(node for node in boxed.nodes if node.kind == "call")
    assert call.line is not None
    assert "qaoa_state(" in qaoa_sample.source_text.splitlines()[call.line - 1]
    assert f"compiled source line {call.line}" in boxed.to_svg()


@pytest.mark.parametrize("inline_depth", [0, 1])
def test_straight_line_helper_proves_while_count_even_if_boxed(inline_depth):
    """Call expansion depth changes the presentation, not the known trip count."""
    source = _COUNTED.replace(
        "@qm.qkernel\ndef flip",
        "@qm.qkernel\ndef limit(n: qm.Int) -> qm.Int:\n    return n + 1\n@qm.qkernel\ndef flip",
    ).replace("while i < rounds:", "while i < limit(rounds):")
    program = qm.build_source(source).specialize("flip", rounds=2)
    svg = draw(program, loops="expanded", inline_depth=inline_depth).to_svg()
    assert svg.count('data-opcode="quantum.x"') == 3


def test_concrete_view_uses_the_loop_count_resolved_by_circuit_lowering():
    """A statically chosen helper branch can determine a concrete gate sequence."""
    source = _COUNTED.replace(
        "@qm.qkernel\ndef flip",
        """@qm.qkernel
def limit(n: qm.Int) -> qm.Int:
    if n > 1:
        stop = n + 1
    else:
        stop = 1
    return stop
@qm.qkernel
def flip""",
    ).replace("while i < rounds:", "while i < limit(rounds):")
    program = qm.build_source(source).specialize("flip", rounds=2)
    svg = draw(program, loops="expanded", wires="qubits").to_svg()
    assert svg.count('data-gate="x"') == 3
    assert svg.count('data-gate="h"') == 1


@pytest.mark.parametrize(
    "update, count",
    [
        ("if i == 0:\n            i += 2\n        else:\n            i += 1", 3),
        (
            "bit, state = qm.measure_z(state, target=0)\n        if bit:\n            state = qm.h(state)\n        i += 1",
            4,
        ),
    ],
)
def test_while_count_survives_branch_carried_values(update, count):
    """Known choices and values identical on both paths keep the trip count exact."""
    source = _COUNTED.replace("i += 1", update)
    program = qm.build_source(source).specialize("flip", rounds=4)
    svg = draw(program, loops="expanded").to_svg()
    assert svg.count('data-opcode="quantum.x"') == count


def test_measured_branch_can_make_the_next_while_count_unknown():
    """A conditional counter update must not unroll as if one outcome were chosen."""
    source = _COUNTED.replace(
        "i += 1",
        "bit, state = qm.measure_z(state, target=0)\n"
        "        if bit:\n"
        "            i += 2\n"
        "        else:\n"
        "            i += 1",
    )
    program = qm.build_source(source).specialize("flip", rounds=4)
    with pytest.raises(qm.CapabilityError, match="Q7402.*while"):
        draw(program, loops="expanded")
