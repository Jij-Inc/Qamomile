"""Visible diagram labels retain the choices in a quantum program."""

import xml.etree.ElementTree as ET

import pytest

import qamomile as qm
from qamomile.visualization import draw


def _labels(source, entry="sample", **options):
    """Read visible SVG text without its tooltip."""
    program = qm.build_source(source).specialize(entry)
    xml = ET.fromstring(draw(program, **options).to_svg())
    return [text[0].tail or "" for text in xml.findall(".//{*}text")]


@pytest.mark.parametrize("comparison", ["n < 1", "n > 1", "0 <= n != 2"])
def test_branch_shows_ordered_comparison(comparison):
    """The visible predicate distinguishes chained and reversed comparisons."""
    source = f"""
import qamomile as qm
@qm.qkernel
def sample(n: qm.Param[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    q = qm.qubits(1)
    if {comparison}:
        q = qm.x(q)
    return qm.measure(q)
"""
    assert f"if {comparison}" in _labels(source)


@pytest.mark.parametrize("mask", [1, 2])
def test_visible_control_mask(mask):
    """The activation state is legible in the ordinary register view."""
    source = f"""
import qamomile as qm
@qm.qkernel
def flip(q: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(q)
@qm.qkernel
def sample() -> qm.Shot[qm.Bits[3]]:
    c, q = qm.split_at(qm.qubits(3), 2)
    c, q = qm.control(flip, c, q, control_state={mask})
    return qm.measure(qm.concat(c, q))
"""
    assert any(
        f"control_state={mask}" in label and "flip" in label
        for label in _labels(source)
    )


@pytest.mark.parametrize("order", ["stay, flip", "flip, stay"])
def test_multiplex_shows_choice_order(order):
    """Switching which operation is selected by zero changes the visible label."""
    source = f"""
import qamomile as qm
@qm.qkernel
def stay(q: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return q
@qm.qkernel
def flip(q: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(q)
@qm.qkernel
def sample() -> qm.Shot[qm.Bits[2]]:
    idx, q = qm.split_at(qm.prepare((1, 0)), 1)
    idx, q = qm.multiplex(idx, ({order}), q)
    return qm.measure(qm.concat(idx, q))
"""
    assert any(order in label for label in _labels(source))


def test_large_control_mask_preserves_bits_and_target():
    """Both the complete mask and the callee survive compact notation."""
    template = """
import qamomile as qm
@qm.qkernel
def flip(q: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(q)
@qm.qkernel
def keep(q: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return q
@qm.qkernel
def sample() -> qm.Q[qm.Bits[803]]:
    c, q = qm.split_at(qm.qubits(803), 802)
    c, q = qm.control(TARGET, c, q, control_state=MASK)
    return qm.concat(c, q)
"""
    for mask, target in (
        (1 << 801, "flip"),
        (1 << 801 | 1, "flip"),
        (1 << 801, "keep"),
    ):
        source = template.replace("MASK", str(mask)).replace("TARGET", target)
        label = next(label for label in _labels(source) if "control_state=" in label)
        digits, name = label.removeprefix("control_state=").split(" · ")
        assert (int(digits, 0), name) == (mask, target)


def test_boxed_call_preserves_its_specialization_identity():
    """A boxed higher-order function identifies which specialization is called."""
    template = """
import qamomile as qm
@qm.qkernel
def flip(q: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(q)
@qm.qkernel
def keep(q: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return q
@qm.qkernel
def apply(unitary: qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]], q: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return unitary(q)
@qm.qkernel
def sample() -> qm.Bits[1]:
    return qm.measure(apply(TARGET, qm.qubits(1)))
"""
    captions = []
    for target in ("flip", "keep"):
        labels = _labels(template.replace("TARGET", target), inline_depth=0)
        captions.append(next(label for label in labels if label.startswith("apply ")))
    assert captions[0] != captions[1]


def test_register_inputs_and_crossing_wires_have_distinct_labels():
    """Each crossed input label is anchored on its own destination lane."""
    source = """
import qamomile as qm
@qm.qkernel
def sample(angles: qm.Param[qm.Vector[qm.Real, 3]]) -> qm.Bits[3]:
    q = qm.qubits(3)
    for i in range(3):
        q = qm.ry(q, angles[i], target=2-i)
    return qm.measure(q)
"""
    xml = ET.fromstring(draw(qm.build_source(source).specialize("sample")).to_svg())
    for parent in xml.iter():
        texts = [
            (text.get("x"), text.get("y"))
            for edge in parent.findall('{*}g[@data-edge="dataflow"]')
            for text in edge.findall("{*}text")
            if (text[0].tail or "").startswith("v")
        ]
        assert len(texts) == len(set(texts))


@pytest.mark.parametrize("depth", [0, 1])
def test_long_call_name_fits_its_box_with_and_without_the_function_body(depth):
    """Even an identity function can have a title much wider than its contents."""
    name = "W" * 180
    source = f"""
import qamomile as qm
@qm.qkernel
def {name}(q: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return q
@qm.qkernel
def sample() -> qm.Bits[1]:
    return qm.measure({name}(qm.qubits(1)))
"""
    diagram = draw(qm.build_source(source).specialize("sample"), inline_depth=depth)
    call = ET.fromstring(diagram.to_svg()).find('.//{*}g[@data-opcode="qcore.call"]')
    assert call is not None
    title = call.find("{*}text")
    box = call.find("{*}rect")
    assert title is not None and title[0].tail == name
    assert box is not None and float(box.attrib["width"]) > 11 * len(name)
