"""Structured native diagrams preserve control flow, types, and ownership."""

import re
import sys
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest

import qamomile as qm
from qamomile.circuit import lower_to_circuit
from qamomile.visualization import DisplayNode, draw
from qamomile.visualization._model import decode_projection

LOOP_SOURCE = """
import qamomile as qm
@qm.qkernel
def rotation(state: qm.Q[qm.Bits[3]], angle: qm.Param[qm.Real], i: qm.Const[qm.Int]) -> qm.Q[qm.Bits[3]]:
    return qm.rx(state, 2.0 * angle, target=i)
@qm.qkernel
def example(n: qm.Const[qm.Int], angle: qm.Param[qm.Real]) -> qm.Shot[qm.Bits[3]]:
    state = qm.qubits(3)
    for i in range(n):
        state = rotation(state, angle, i)
    return qm.measure(state)
"""


def walk(nodes: tuple[DisplayNode, ...]):
    """Walk nested presentation nodes in source order."""
    for node in nodes:
        yield node
        yield from walk(node.children)


@pytest.fixture
def loop_program():
    """A loop whose helper contains a launch-dependent classical expression."""
    return qm.build_source(LOOP_SOURCE).specialize("example", n=3)


def test_loop_modes_and_call_expansion_preserve_carried_identity(loop_program):
    """Unrolling resolves indices and connects each yielded state to its successor."""
    folded = draw(loop_program, loops="folded")
    assert not next(node for node in folded.nodes if node.kind == "loop").children
    body = draw(loop_program)
    assert len([n for n in walk(body.nodes) if n.opcode == "quantum.rx"]) == 1
    assert not any(n.kind == "iteration" for n in walk(body.nodes))
    expanded = draw(loop_program, loops="expanded")
    iterations = [n for n in walk(expanded.nodes) if n.kind == "iteration"]
    assert [n.label for n in iterations] == ["i = 0", "i = 1", "i = 2"]
    assert iterations[1].inputs == iterations[0].outputs
    assert iterations[2].inputs == iterations[1].outputs
    rotations = [n for n in walk(expanded.nodes) if n.opcode == "quantum.rx"]
    assert [expanded.values[n.inputs[-1]].label for n in rotations] == ["0", "1", "2"]
    boxed = draw(loop_program, inline_depth=0)
    assert not any(n.opcode == "quantum.rx" for n in walk(boxed.nodes))


def test_classical_dependencies_types_and_native_qubit_view(loop_program):
    """Presentation changes neither compiler facts nor source bytes."""
    before = loop_program.to_bytes()
    compact = draw(loop_program)
    expanded = draw(loop_program, classical="expanded")
    svg = expanded.to_svg()
    assert 'data-kind="classical"' in svg
    assert 'data-kind="classical"' not in compact.to_svg()
    assert 'data-stage="Param"' in svg
    assert 'stroke-dasharray="4 3"' in svg
    angle = next(v for v in expanded.values if v.label == "(2 * angle)")
    assert angle.stage == "Param" and not angle.quantum
    state = next(v for v in expanded.values if v.quantum)
    assert state.type_label == "Q[Bits[3]]" and state.width == 3
    concrete = draw(loop_program, loops="expanded", wires="qubits")
    assert concrete._circuit == lower_to_circuit(loop_program)
    assert all(f">q{i}</text>" in concrete.to_svg() for i in range(3))
    ET.fromstring(svg)
    assert loop_program.to_bytes() == before
    with pytest.raises(FrozenInstanceError):
        compact.values[0].label = "changed"


def test_shot_branch_interfaces_are_exclusive_and_share_owned_inputs():
    """Measurement feeds a Shot predicate without merging both alternatives."""
    program = qm.build_source("""
import qamomile as qm
@qm.qkernel
def correction() -> qm.Q[qm.Bits[1]]:
    alice, bob = qm.split_at(qm.h(qm.qubits(2)), 1)
    result = qm.measure(alice)
    if bool(result[0]):
        bob = qm.x(bob)
    else:
        bob = qm.z(bob)
    return bob
""").specialize("correction")
    diagram = draw(program, classical="expanded")
    branch = next(n for n in diagram.nodes if n.kind == "branch")
    assert diagram.values[branch.inputs[0]].stage == "Shot"
    assert [n.label for n in branch.children] == ["true", "false"]
    assert branch.children[0].inputs == branch.children[1].inputs
    assert branch.children[0].outputs != branch.children[1].outputs
    assert "Exclusive alternatives" in diagram.to_svg()


def test_quantum_arithmetic_keeps_both_registers_and_interpretations():
    """Arithmetic is one coherent operation; reinterpretation retains bit width."""
    program = qm.build_source("""
import qamomile as qm
@qm.qkernel
def arithmetic() -> tuple[qm.Q[qm.UInt[3]], qm.Q[qm.UInt[3]]]:
    a = qm.as_uint(qm.qubits(3))
    b = qm.as_uint(qm.h(qm.qubits(3)))
    a, b = a + b
    return a, b
""").specialize("arithmetic")
    diagram = draw(program)
    coherent = [v for v in diagram.values if v.type_label == "Q[UInt[3]]"]
    assert coherent and all(v.width == 3 and v.quantum for v in coherent)
    arithmetic = next(n for n in walk(diagram.nodes) if n.opcode == "quantum.add_into")
    assert len(arithmetic.inputs) == 1
    assert diagram.values[arithmetic.inputs[0]].width == 6
    assert any(n.kind == "structure" for n in walk(diagram.nodes))
    assert len(diagram.nodes[-1].inputs) == 1  # Aggregate ownership, not copies.


def test_concrete_cnot_uses_ordered_control_and_target_glyphs():
    """The native circuit's first CNOT operand is drawn as its control."""
    program = qm.build_source("""
import qamomile as qm
@qm.qkernel
def bell() -> qm.Shot[qm.Bits[2]]:
    state = qm.h(qm.qubits(2), target=0)
    state = qm.cx(state, control=0, target=1)
    return qm.measure(state)
""").specialize("bell")
    svg = draw(program, loops="expanded", wires="qubits").to_svg()
    assert ">m0</text>" in svg and ">m1</text>" in svg
    root = ET.fromstring(svg)
    gate = root.find('.//{http://www.w3.org/2000/svg}g[@data-gate="cx"]')
    assert gate is not None
    assert (
        gate.find('{http://www.w3.org/2000/svg}circle[@data-control="q0"]') is not None
    )
    assert (
        gate.find('{http://www.w3.org/2000/svg}circle[@data-target="q1"]') is not None
    )


@pytest.mark.parametrize(
    "start,stop,stride,expected", [(3, -1, -2, [3, 1]), (0, 0, 1, [])]
)
def test_range_edges_and_loop_else(start, stop, stride, expected):
    """Empty and descending loops carry exact values into normal completion."""
    source = f"""
import qamomile as qm
@qm.qkernel
def ranges() -> qm.Int:
    total = 0
    for i in range({start}, {stop}, {stride}):
        total += i
    else:
        total += 2
    for j in range(total):
        total += 0
    return total
"""
    diagram = draw(qm.build_source(source).specialize("ranges"), loops="expanded")
    loops = [n for n in diagram.nodes if n.kind == "loop"]
    assert [n.label for n in loops[0].children if n.kind == "iteration"] == [
        f"i = {i}" for i in expected
    ]
    assert (
        len([n for n in loops[1].children if n.kind == "iteration"])
        == sum(expected) + 2
    )
    completion = loops[0].children[-1]
    if expected:
        assert completion.inputs == loops[0].children[-2].outputs


def test_symbolic_and_large_loops_remain_abstract_and_disable_expansion():
    """Unknown and enormous ranges never cause uncontrolled iteration allocation."""
    family = qm.ProgramFamily(qm.build_source(LOOP_SOURCE), "example")
    assert any(v.quantum and v.width == 3 for v in draw(family).values)
    html = draw(family).to_html()
    assert re.search(r'<button[^>]+data-loops="expanded"[^>]+disabled', html)
    with pytest.raises(qm.CapabilityError, match="Q7402"):
        draw(family, loops="expanded")
    large = family.specialize(n=10**12)
    assert len(draw(large, loops="folded").values) < 20
    with pytest.raises(qm.CapabilityError, match="Q7401"):
        draw(large, loops="expanded", max_nodes=100)


def test_early_exit_is_visible_and_expansion_fails_closed():
    """Break stays explicit rather than being drawn as unconditional repetition."""
    program = qm.build_source("""
import qamomile as qm
@qm.qkernel
def early(flag: qm.Param[qm.Bool]) -> qm.Int:
    total = 0
    for i in range(4):
        if flag:
            break
        total += 1
    else:
        total += 10
    return total
""").specialize("early")
    body = draw(program)
    assert any(n.opcode == "break" for n in walk(body.nodes))
    completion = next(n for n in walk(body.nodes) if n.opcode == "loop.completion")
    assert all(body.values[i].label == "final carried value" for i in completion.inputs)
    with pytest.raises(qm.CapabilityError, match="Q7402"):
        draw(program, loops="expanded")


def test_qkernel_and_symbolic_fixed_point_widths():
    """Source declarations and unknown widths remain abstract and typed."""

    @qm.qkernel
    def fixed(state: qm.Q[qm.Bits[4]]) -> qm.Q[qm.UFixed[2, 2]]:
        return qm.as_fixed(state, integer_bits=2)

    assert any(
        v.type_label == "Q[UFixed[2, 2]]" and v.width == 4 for v in draw(fixed).values
    )
    module = qm.build_source("""
import qamomile as qm
@qm.qkernel
def symbolic(n: qm.Const[qm.Int], state: qm.Q[qm.UInt[n]]) -> qm.Q[qm.UInt[n]]:
    return state
""")
    diagram = draw(qm.ProgramFamily(module, "symbolic"))
    port = next(v for v in diagram.values if v.quantum)
    assert port.type_label == "Q[UInt[n]]" and port.width is None
    assert "symbolic" in diagram.to_svg()


def test_labels_are_escaped_and_xml_safe(loop_program):
    """Presentation labels cannot introduce markup or break XML parsing."""
    diagram = draw(loop_program, classical="expanded")
    values = list(diagram.values)
    values[0] = replace(values[0], label="<script>alert(1)</script>\x00")
    diagram = replace(diagram, display=replace(diagram.display, values=tuple(values)))
    svg = diagram.to_svg()
    ET.fromstring(svg)
    assert "<script>" not in svg and "&lt;script&gt;" in svg
    assert "\\u003c" in diagram.to_html()


def test_exports_and_missing_or_malformed_native_capability(
    loop_program, tmp_path, monkeypatch
):
    """Exports are portable and incompatible native results never trigger a fallback."""
    from qamomile.visualization import _api

    diagram = draw(loop_program)
    ET.fromstring(diagram.save(tmp_path / "algorithm.svg").read_text())
    saved = diagram.save(tmp_path / "algorithm.html").read_text()
    assert "fetch(" not in saved and "https://" not in saved
    assert "body:expressions" in saved and "expanded:expanded" in saved
    with pytest.raises(ValueError):
        diagram.save(tmp_path / "algorithm.jpg")
    with pytest.raises(qm.CapabilityError, match="Q7404"):
        decode_projection(
            SimpleNamespace(values=[], inputs=[0], nodes=[], entrypoint="bad")
        )
    monkeypatch.setattr(_api.native, "project_visualization", None)
    with pytest.raises(qm.CapabilityError, match="Q7404"):
        draw(loop_program)


@pytest.mark.parametrize(
    "options,exception",
    [
        ({"loops": "all"}, ValueError),
        ({"max_nodes": True}, TypeError),
        ({"inline_depth": 33}, ValueError),
        ({"max_nodes": 0}, ValueError),
        ({"wires": "qubits"}, ValueError),
    ],
)
def test_invalid_presentation_options(loop_program, options, exception):
    """Reject unsupported choices before projection or concrete expansion."""
    with pytest.raises(exception):
        draw(loop_program, **options)


def svg_geometry(svg):
    """Collect absolute pin and edge positions through nested SVG translations."""
    pins, edges, operations = set(), [], []

    def visit(element, x=0.0, y=0.0):
        """Accumulate local translations for a connected native display graph."""
        translation = re.fullmatch(
            r"translate\(([-\d.]+),([-\d.]+)\)", element.get("transform", "")
        )
        if translation:
            x += float(translation[1])
            y += float(translation[2])
        if element.get("data-opcode"):
            operations.append((element.get("data-opcode"), x, y))
        identifier = element.get("data-value")
        if identifier is not None:
            circle = element.find("{http://www.w3.org/2000/svg}circle")
            if circle is not None:
                pins.add(
                    (
                        int(identifier),
                        x + float(circle.get("cx")),
                        y + float(circle.get("cy")),
                    )
                )
            if element.get("data-edge") == "dataflow":
                edges.append(
                    (
                        int(identifier),
                        x + float(element.get("data-from-x")),
                        y + float(element.get("data-from-y")),
                        x + float(element.get("data-to-x")),
                        y + float(element.get("data-to-y")),
                    )
                )
        for child in element:
            visit(child, x, y)

    visit(ET.fromstring(svg))
    return pins, edges, operations


@pytest.mark.parametrize("loops", ["folded", "body", "expanded"])
@pytest.mark.parametrize("classical", ["expressions", "expanded"])
def test_horizontal_edges_connect_actual_pins_in_nested_scopes(
    loop_program, loops, classical
):
    """Every dependency joins matching native values and advances to the right."""
    svg = draw(loop_program, loops=loops, classical=classical).to_svg()
    pins, edges, operations = svg_geometry(svg)
    assert edges
    for identifier, x1, y1, x2, y2 in edges:
        assert x1 < x2
        assert (identifier, x1, y1) in pins
        assert (identifier, x2, y2) in pins
    if loops == "expanded":
        rotations = [x for opcode, x, _ in operations if opcode == "quantum.rx"]
        assert len(rotations) == 3
        assert rotations == sorted(rotations)
        # The carried quantum output crosses the iteration boundaries as a bus.
        diagram = draw(loop_program, loops=loops, classical=classical)
        iterations = [n for n in walk(diagram.nodes) if n.kind == "iteration"]
        carried = next(i for i in iterations[0].outputs if diagram.values[i].quantum)
        assert len([edge for edge in edges if edge[0] == carried]) >= 2


def test_measurement_predicate_and_exclusive_result_bindings_are_connected():
    """Shot dependencies reach if, and each alternative binds the shared result."""
    program = qm.build_source("""
import qamomile as qm
@qm.qkernel
def feedback() -> qm.Q[qm.Bits[1]]:
    state = qm.h(qm.qubits(2), target=0)
    data, probe = qm.split_at(state, 1)
    m = qm.measure(probe)
    if bool(m[0]):
        data = qm.x(data)
    return data
""").specialize("feedback")
    diagram = draw(program)
    branch = next(n for n in diagram.nodes if n.kind == "branch")
    predicate = branch.inputs[0]
    svg = diagram.to_svg()
    pins, edges, operations = svg_geometry(svg)
    assert any(i == predicate for i, *_ in edges)
    for identifier, x1, y1, x2, y2 in edges:
        assert x1 < x2
        assert (identifier, x1, y1) in pins
        assert (identifier, x2, y2) in pins
    result = next(i for i in branch.outputs if diagram.values[i].quantum)
    root = ET.fromstring(svg)
    bindings = root.findall(
        f'.//{{http://www.w3.org/2000/svg}}g[@data-result-value="{result}"]'
    )
    assert len(bindings) == 2
    assert {int(g.get("data-value")) for g in bindings} == {
        next(i for i in arm.outputs if diagram.values[i].quantum)
        for arm in branch.children
    }
    assert all(g.get("data-edge") == "result-binding" for g in bindings)
    positions = {opcode: x for opcode, x, _ in operations}
    assert (
        positions["quantum.measure"]
        < positions["classical.index"]
        < positions["qcore.if"]
    )


def test_long_concrete_circuit_does_not_wrap_time_into_rows():
    """More than ten native instructions stay on one continuous qubit lane."""
    source = """
import qamomile as qm
@qm.qkernel
def long_circuit() -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    for i in range(12):
        state = qm.x(state)
    return qm.measure(state)
"""
    program = qm.build_source(source).specialize("long_circuit")
    svg = draw(program, loops="expanded", wires="qubits").to_svg()
    root = ET.fromstring(svg)
    gates = root.findall('.//{http://www.w3.org/2000/svg}g[@data-gate="x"]')
    assert len(gates) == 12
    positions = [float(g.get("data-x")) for g in gates]
    assert positions == sorted(positions) and len(set(positions)) == 12
    target_rows = {
        float(g.find("{http://www.w3.org/2000/svg}rect").get("y")) for g in gates
    }
    assert len(target_rows) == 1
    assert svg.count(">q0</text>") == 1


def test_layout_limits_fail_before_allocating_an_unbounded_canvas():
    """A bounded native projection can still exceed the horizontal canvas limit."""
    from qamomile.visualization._model import Display, DisplayValue
    from qamomile.visualization._render import render_svg

    count = 500
    values = tuple(
        DisplayValue(i, "state", "Q[Bits[1]]", True, "Value", 1)
        for i in range(count + 1)
    )
    nodes = tuple(
        DisplayNode("quantum", "quantum.x", "x", (i,), (i + 1,), (), None)
        for i in range(count)
    )
    with pytest.raises(qm.CapabilityError, match="Q7401.*canvas"):
        render_svg(Display("too_wide", values, (0,), nodes), "expressions")


def test_concrete_measurement_wires_reach_the_controlling_instruction():
    """Native mN identities connect measurement output to its exclusive branch."""
    program = qm.build_source("""
import qamomile as qm
@qm.qkernel
def feedback_sample() -> qm.Shot[qm.Bits[1]]:
    data, probe = qm.split_at(qm.h(qm.qubits(2), target=0), 1)
    m = qm.measure(probe)
    if bool(m[0]):
        data = qm.x(data)
    return qm.measure(data)
""").specialize("feedback_sample")
    svg = draw(program, loops="expanded", wires="qubits").to_svg()
    root = ET.fromstring(svg)
    edge = root.find('.//{http://www.w3.org/2000/svg}g[@data-condition-bit="m0"]')
    assert edge is not None
    assert edge.find("{http://www.w3.org/2000/svg}path").get("stroke-dasharray")
    gates = root.findall('.//{http://www.w3.org/2000/svg}g[@data-gate="if"]')
    assert len(gates) == 1
    assert "alternative 0 (exclusive)" in svg and "alternative 1 (exclusive)" in svg


def test_symbolic_body_keeps_else_carries_separate_from_the_displayed_yield():
    """One shown body cannot assert the SSA identity of the final else input."""
    program = qm.build_source("""
import qamomile as qm
@qm.qkernel
def completed() -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    for i in range(2):
        state = qm.h(state)
    else:
        state = qm.x(state)
    return qm.measure(state)
""").specialize("completed")
    diagram = draw(program)
    loop = next(n for n in diagram.nodes if n.kind == "loop")
    body, completion = loop.children
    assert completion.inputs != body.outputs
    svg = diagram.to_svg()
    pins, edges, _ = svg_geometry(svg)
    for identifier, x1, y1, x2, y2 in edges:
        assert x1 < x2
        assert (identifier, x1, y1) in pins
        assert (identifier, x2, y2) in pins
    assert any(edge[0] == completion.inputs[0] for edge in edges)
    root = ET.fromstring(svg)
    assert (
        root.find('.//{http://www.w3.org/2000/svg}g[@data-edge="control-summary"]')
        is not None
    )
    # Symbolic final results use the collector, never a false direct binding
    # from the body yield to the parent's new SSA value.
    assert (
        root.find(
            f'.//{{http://www.w3.org/2000/svg}}g[@data-result-value="{loop.outputs[0]}"]'
        )
        is None
    )


@pytest.fixture
def wide_program():
    """A 65-lane drawing witness that never executes a simulator."""
    return qm.build_source("""
import qamomile as qm
@qm.qkernel
def wide(width: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[width]]:
    return qm.measure(qm.qubits(width))
""").specialize("wide", width=65)


def test_more_than_64_qubit_lanes_and_combined_budget(wide_program):
    """The default budget renders all 65 lanes and includes instruction work."""
    before = wide_program.to_bytes()
    diagram = draw(wide_program, loops="expanded", wires="qubits")
    svg = diagram.to_svg()
    root = ET.fromstring(svg)
    # Text includes a title child, so extract its visible tail.
    labels = [e[-1].tail for e in root.iter("{http://www.w3.org/2000/svg}text")]
    assert [f"q{i}" for i in range(65)] == [
        s for s in labels if s and re.fullmatch(r"q\d+", s)
    ]
    assert wide_program.to_bytes() == before
    required = diagram._circuit.width + diagram._circuit.operation_count
    draw(wide_program, loops="expanded", wires="qubits", max_nodes=required)
    with pytest.raises(qm.CapabilityError, match="Q7401.*lanes and instructions"):
        draw(wide_program, loops="expanded", wires="qubits", max_nodes=required - 1)


def test_huge_qubit_width_rejected_before_circuit_expansion(monkeypatch):
    """A native register width is checked before allocating concrete lanes."""
    from qamomile.visualization import _api

    program = qm.build_source("""
import qamomile as qm
@qm.qkernel
def huge(width: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[width]]:
    return qm.measure(qm.qubits(width))
""").specialize("huge", width=10**12)

    def unexpected_expansion(*args, **kwargs):
        pytest.fail("oversized width reached circuit expansion")

    monkeypatch.setattr(_api, "lower_to_circuit", unexpected_expansion)
    with pytest.raises(qm.CapabilityError, match="Q7401.*lanes exceed max_nodes"):
        draw(program, loops="expanded", wires="qubits")
    assert draw(program).to_svg()


@pytest.mark.parametrize("concrete", [False, True])
def test_png_and_vector_pdf_exports(loop_program, tmp_path, concrete, static_export):
    """Both views export nonempty PNG pixels and a vector PDF at the SVG size."""
    import io
    import struct

    from PIL import Image

    diagram = draw(
        loop_program, **({"loops": "expanded", "wires": "qubits"} if concrete else {})
    )
    root = ET.fromstring(diagram.to_svg())
    width, height = float(root.get("width")), float(root.get("height"))
    path = tmp_path / "drawing.PNG"
    assert diagram.save(path) == path
    content = path.read_bytes()
    assert content[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", content[16:24]) == (round(width), round(height))
    image = Image.open(io.BytesIO(content)).convert("RGB")
    assert len(image.getcolors(image.width * image.height)) > 2
    scaled = diagram.save(tmp_path / "scaled.png", scale=0.5)
    with Image.open(scaled) as smaller:
        assert smaller.size == (round(width * 0.5), round(height * 0.5))
    pdf = diagram.save(tmp_path / "drawing.pdf").read_bytes()
    assert pdf.startswith(b"%PDF-") and pdf.rstrip().endswith(b"%%EOF")
    assert b"/Subtype /Image" not in pdf


@pytest.mark.parametrize(
    "scale,error",
    [
        (0, ValueError),
        (-1, ValueError),
        (float("inf"), ValueError),
        (float("nan"), ValueError),
        (True, TypeError),
        ("2", TypeError),
    ],
)
def test_invalid_export_scale(loop_program, tmp_path, scale, error):
    """Invalid scales leave the destination unwritten."""
    path = tmp_path / "bad.png"
    with pytest.raises(error):
        draw(loop_program).save(path, scale=scale)
    assert not path.exists()


def test_static_export_budgets_and_missing_dependency(
    loop_program, tmp_path, monkeypatch
):
    """Large raster allocations and unavailable exporters fail before writing."""
    diagram = draw(loop_program)
    path = tmp_path / "drawing.png"
    path.write_bytes(b"existing")
    with pytest.raises(qm.CapabilityError, match="Q7401.*pixel budget"):
        diagram.save(path, scale=10)
    assert path.read_bytes() == b"existing"
    monkeypatch.setitem(sys.modules, "cairosvg", None)
    with pytest.raises(
        qm.CapabilityError, match=r"Q7404.*qamomile\[visualization\].*Cairo"
    ):
        diagram.save(path)
    assert path.read_bytes() == b"existing"
    assert diagram.save(tmp_path / "drawing.svg").exists()


def test_concrete_routing_and_canvas_bounds(loop_program):
    """Concrete lane/slot work and lane height are bounded before SVG assembly."""
    from qamomile.visualization._render import _CircuitLayout

    wide = qm.build_source("""
import qamomile as qm
@qm.qkernel
def routing() -> qm.Shot[qm.Bits[2000]]:
    return qm.measure(qm.qubits(2000))
""").specialize("routing")
    diagram = draw(wide, loops="expanded", wires="qubits", max_nodes=4096)
    with pytest.raises(qm.CapabilityError, match="Q7401.*routing"):
        diagram.to_svg()
    circuit = lower_to_circuit(loop_program)
    with pytest.raises(qm.CapabilityError, match="Q7401.*canvas"):
        _CircuitLayout(replace(circuit, width=2500), 16384).block(
            circuit.body, "bounded"
        )
