"""Check visible geometry and annotations on native qubit lanes."""

import re
from xml.etree import ElementTree as ET

import pytest

import qamomile as qm
from qamomile.visualization import draw
from tests.qamomile.conftest import CONTROLLED_PHASE_SOURCE


def concrete_svg(source, name="sample"):
    """Render a source circuit through the public concrete view."""
    program = qm.build_source(source).specialize(name)
    return ET.fromstring(draw(program, loops="expanded", wires="qubits").to_svg())


def visible_labels(element):
    """Read the visible SVG text independently of hover titles."""
    return [label[0].tail or "" for label in element.iterfind(".//{*}text")]


NESTED = """
import qamomile as qm
@qm.qkernel
def sample() -> qm.Shot[qm.Bit]:
    a = qm.measure(qm.h(qm.prepare(False)))
    b = qm.measure(qm.h(qm.prepare(False)))
    target = qm.prepare(False)
    if a:
        if b:
            target = qm.x(target)
    else:
        if b:
            target = qm.z(target)
    return qm.measure(target)
"""


def assert_measurement_reaches(root, bit, condition_count):
    """Follow rendered path endpoints from a measurement box to its predicates."""
    connections = {}
    ends = []

    def collect(element, x=0, y=0):
        if transform := element.get("transform"):
            dx, dy = map(float, re.findall(r"[-0-9.]+", transform))
            x, y = x + dx, y + dy
        if element.get("data-gate") == "measure" and bit in visible_labels(element):
            box = element.find("{*}rect")
            starts.append(
                (
                    x + float(box.get("x")) + float(box.get("width")) / 2,
                    y + float(box.get("y")) + float(box.get("height")),
                )
            )
        if bit in (element.get("data-routed-bit"), element.get("data-condition-bit")):
            path = element.find("{*}path").get("d")
            sx, sy, *rest = re.findall(r"[-0-9.]+", path)
            start = (x + float(sx), y + float(sy))
            end = (x + float(rest[-1]), y + float(rest[-2]))
            connections.setdefault(start, []).append(end)
            if marker := element.findall("{*}circle"):
                ends.append(
                    (x + float(marker[0].get("cx")), y + float(marker[0].get("cy")))
                )
        for child in element:
            collect(child, x, y)

    starts = []
    collect(root)
    assert len(starts) == 1 and len(ends) == condition_count
    reached = set(starts)
    pending = list(starts)
    while pending:
        new = set(connections.get(pending.pop(), ())) - reached
        reached.update(new)
        pending.extend(new)
    assert set(ends) <= reached


def test_measurement_reaches_nested_conditions_in_both_alternatives():
    """Each exclusive arm can use a bit measured before the outer branch."""
    root = concrete_svg(NESTED)
    assert visible_labels(root).count("if m1") == 2
    assert_measurement_reaches(root, "m1", 2)


def test_branch_measurements_reach_a_subsequent_condition():
    """Values measured in exclusive alternatives emerge at the next predicate."""
    root = concrete_svg("""
import qamomile as qm
@qm.qkernel
def sample() -> qm.Shot[qm.Bit]:
    choice = qm.measure(qm.h(qm.prepare(False)))
    target = qm.prepare(False)
    if choice:
        use_x = qm.measure(qm.h(qm.prepare(False)))
    else:
        use_x = qm.measure(qm.prepare(False))
    if use_x:
        target = qm.x(target)
    return qm.measure(target)
""")
    assert_measurement_reaches(root, "m0", 2)
    assert_measurement_reaches(root, "m1", 1)
    assert_measurement_reaches(root, "m2", 1)


def test_branch_entry_meets_the_child_qubit_lane():
    """The slanted branch entry terminates on the first child horizontal wire."""
    root = concrete_svg(NESTED)
    outer = root.find('.//{*}g[@data-gate="if"]')
    child = outer.find("{*}g[@transform]")
    tx, ty = map(float, re.findall(r"[-0-9.]+", child.attrib["transform"]))
    first_wire = child.find("{*}path")
    x, y = map(float, re.findall(r"[-0-9.]+", first_wire.attrib["d"])[:2])
    branch_ends = {
        tuple(map(float, re.findall(r"[-0-9.]+", line.attrib["d"])[-2:]))
        for line in outer.findall("{*}path")
    }
    assert (tx + x, ty + y) in branch_ends


def test_unequal_branches_rejoin_beyond_the_longest_arm():
    """The short arm's exit does not cut through the larger sibling diagram."""
    source = NESTED[: NESTED.index("    else:")] + "    return qm.measure(target)\n"
    outer = concrete_svg(source).find('.//{*}g[@data-gate="if"]')
    bend_positions = set()
    for path in outer.findall("{*}path"):
        x1, y1, _, y2 = map(float, re.findall(r"[-0-9.]+", path.get("d")))
        if y1 > y2:
            bend_positions.add(x1)
    assert len(bend_positions) == 1


@pytest.mark.parametrize(
    ("source_condition", "label"),
    [("a == b", "m0 == m1"), ("a != b", "m0 != m1"), ("a == b != a", "m0 == m1 != m0")],
)
def test_two_bit_comparison_preserves_operator(source_condition, label):
    """Concrete bit predicates preserve each operator in a comparison chain."""
    source = NESTED[: NESTED.index("    if a:")] + (
        f"    if {source_condition}:\n"
        "        target = qm.x(target)\n"
        "    return qm.measure(target)\n"
    )
    predicate = next(
        label
        for label in visible_labels(concrete_svg(source))
        if label.startswith("if ")
    )
    assert label in predicate


def test_angle_labels_keep_vector_indices_visible():
    """Two rotations in successive layers display their distinct array elements."""
    root = concrete_svg("""
import qamomile as qm
@qm.qkernel
def sample(gammas: qm.Param[qm.Vector[qm.Real, 2]]) -> qm.Shot[qm.Bits[1]]:
    q = qm.qubits(1)
    for i in range(2):
        q = qm.rz(q, 2.0 * gammas[i])
    return qm.measure(q)
""")
    angles = [visible_labels(g)[0] for g in root.findall('.//{*}g[@data-gate="rz"]')]
    assert len(angles) == 2
    assert "gammas[0]" in angles[0] and "gammas[1]" in angles[1]


def test_controlled_phase_displays_its_operation():
    """A zero-target operation with a quantum control still has a visible name."""
    root = concrete_svg(CONTROLLED_PHASE_SOURCE, "controlled_phase")
    gate = root.find('.//{*}g[@data-gate="global_phase"]')
    assert "global_phase" in visible_labels(gate)
    assert any(label.startswith("3.14") for label in visible_labels(gate))
