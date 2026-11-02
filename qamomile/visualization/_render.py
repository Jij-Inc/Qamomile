"""Pure presentation of native display ports and concrete circuit instructions."""

from __future__ import annotations

import html
import json
import uuid
from dataclasses import dataclass, field

from qamomile.circuit import (
    CircuitBlock,
    CircuitProgram,
    CircuitValue,
    ClassicalExpr,
    MeasurementBit,
    ParameterValue,
    ScalarParameter,
)
from qamomile.errors import CapabilityError

from ._model import Display, DisplayNode

_MAX_SVG_BYTES = 4_000_000
_MAX_HTML_BYTES = 16_000_000
_QUANTUM = "#246b83"
_CLASSICAL = "#92601f"
_BORDER = "#ccd9de"


class _SvgParts(list[str]):
    """Bound allocated SVG fragments before joining the complete document.

    Attributes:
        size (int): Encoded bytes retained by this builder.
    """

    def __init__(self) -> None:
        """Create an empty fragment builder."""
        super().__init__()
        self.size = 0

    def append(self, part: str) -> None:
        """Append a fragment within the output allocation limit.

        Args:
            part (str): One generated SVG fragment.

        Raises:
            CapabilityError: If the SVG output budget would be exceeded.
        """
        self.size += len(part.encode())
        if self.size > _MAX_SVG_BYTES:
            raise CapabilityError("Q7401: rendered SVG exceeds its output limit")
        super().append(part)


def _text(
    x: float,
    y: float,
    label: str,
    *,
    limit: int = 55,
    size: int = 12,
    anchor: str = "start",
) -> str:
    """Place bounded text with an accessible, escaped full label.

    Args:
        x (float): Horizontal position.
        y (float): Baseline position.
        label (str): Full native-derived label.
        limit (int): Maximum visible characters; defaults to 55.
        size (int): Font size in SVG units; defaults to 12.
        anchor (str): Text alignment; defaults to start.

    Returns:
        str: Escaped SVG text and title.
    """
    # XML 1.0 rejects several valid source-string characters. Sanitize display
    # text only, preserving the original immutable model.
    label = "".join(
        character
        if character in "\t\n\r"
        or 0x20 <= ord(character) <= 0xD7FF
        or 0xE000 <= ord(character) <= 0xFFFD
        or 0x10000 <= ord(character) <= 0x10FFFF
        else "�"
        for character in label
    )
    visible = label if len(label) <= limit else label[: limit - 1] + "…"
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">'
        f"<title>{html.escape(label)}</title>{html.escape(visible)}</text>"
    )


def _rect(
    x: float, y: float, width: float, height: float, fill: str = "#ffffff"
) -> str:
    """Draw one operation or structured control boundary.

    Args:
        x (float): Left coordinate.
        y (float): Top coordinate.
        width (float): Horizontal extent.
        height (float): Vertical extent.
        fill (str): Surface color; defaults to white.

    Returns:
        str: SVG rectangle.
    """
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="6" fill="{fill}" stroke="{_BORDER}"/>'


def _line(x1: float, y1: float, x2: float, y2: float, quantum: bool = True) -> str:
    """Draw a quantum edge or explicitly dashed classical dependency.

    Args:
        x1 (float): Start horizontal coordinate.
        y1 (float): Start vertical coordinate.
        x2 (float): End horizontal coordinate.
        y2 (float): End vertical coordinate.
        quantum (bool): Use solid quantum styling when true.

    Returns:
        str: SVG dataflow edge.
    """
    dash = "" if quantum else ' stroke-dasharray="4 3"'
    return f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{_QUANTUM if quantum else _CLASSICAL}" stroke-width="1.5"{dash}/>'


@dataclass
class _Scene:
    """Measured local geometry for a horizontal operation or sequence.

    Attributes:
        width (float): Horizontal extent in SVG units.
        height (float): Vertical extent in SVG units.
        content (str): Escaped SVG fragments in local coordinates.
        inputs (tuple[tuple[int, float, float], ...]): Incoming pins in visual order.
        outputs (tuple[tuple[int, float, float], ...]): Outgoing pins in visual order.
    """

    width: float
    height: float
    content: str
    inputs: tuple[tuple[int, float, float], ...] = ()
    outputs: tuple[tuple[int, float, float], ...] = ()


def _place(scene: _Scene, x: float, y: float) -> str:
    """Translate measured local geometry without changing its dataflow facts.

    Args:
        scene (_Scene): Local geometry.
        x (float): Horizontal offset in SVG units.
        y (float): Vertical offset in SVG units.

    Returns:
        str: Translated SVG group.
    """
    return f'<g transform="translate({x},{y})">{scene.content}</g>'


@dataclass
class _Layout:
    """Lay out native value dependencies with one left-to-right time axis.

    Attributes:
        display (Display): Immutable native facts.
        classical (str): Requested classical presentation.
        literals (set[int]): Native constant outputs shown as inline arguments.
        work (int): Presentation routing work consumed within the bounded request.
    """

    display: Display
    classical: str
    literals: set[int] = field(default_factory=set)
    work: int = 0

    def collect_literals(self, nodes: tuple[DisplayNode, ...]) -> None:
        """Identify constant annotations using native operation categories.

        Args:
            nodes (tuple[DisplayNode, ...]): Native nodes in the current subtree.
        """
        for node in nodes:
            if node.opcode == "qcore.constant":
                self.literals.update(node.outputs)
            self.collect_literals(node.children)

    def ports(
        self, ids: tuple[int, ...], *, keep_literals: bool = False
    ) -> tuple[int, ...]:
        """Order quantum buses above classical dependencies for presentation.

        Args:
            ids (tuple[int, ...]): Native ordered interface IDs.
            keep_literals (bool): Preserve literal exit bindings; defaults to false.

        Returns:
            tuple[int, ...]: Stable quantum-first order, excluding inline literals.
        """
        return tuple(
            sorted(
                (
                    i
                    for i in ids
                    if keep_literals
                    or self.classical == "expanded"
                    or i not in self.literals
                ),
                key=lambda i: not self.display.values[i].quantum,
            )
        )

    def pin(self, identifier: int, x: float, y: float) -> str:
        """Draw an accessible value pin carrying the exact native type and stage.

        Args:
            identifier (int): Display value ID.
            x (float): Horizontal position.
            y (float): Vertical position.

        Returns:
            str: Selectable SVG pin with a full typed tooltip.
        """
        value = self.display.values[identifier]
        color = _QUANTUM if value.quantum else _CLASSICAL
        title = _text(x, y, f"v{identifier} · {value.label} · {value.type_label}")
        # Reuse the XML-safe text sanitizer for accessible labels as well.
        safe_title = title.split("<title>", 1)[1].split("</title>", 1)[0]
        return (
            f'<g data-value="{identifier}" data-stage="{value.stage}" '
            f'data-quantum="{str(value.quantum).lower()}" tabindex="0" role="button" '
            f'aria-label="{safe_title}"><title>{safe_title}</title>'
            f'<circle cx="{x}" cy="{y}" r="4" fill="#fff" stroke="{color}"/></g>'
        )

    def edge(
        self,
        identifier: int,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        track: float | None = None,
        annotation: bool = True,
        interface: bool = False,
        result: int | None = None,
    ) -> str:
        """Connect resolved pins with a monotone horizontal dataflow path.

        Args:
            identifier (int): Native source value ID.
            start (tuple[float, float]): Producer position.
            end (tuple[float, float]): Consumer position.
            track (float | None): Optional bypass lane below intervening operations.
            annotation (bool): Include a bounded typed label; defaults to true.
            interface (bool): Mark a region result binding, rather than SSA identity.
            result (int | None): Bound outer value ID for an interface edge.

        Returns:
            str: Selectable solid bus or dashed classical dependency.
        """
        value = self.display.values[identifier]
        x1, y1 = start
        x2, y2 = end
        bend = min(22, max(0, (x2 - x1) / 3))
        if track is None:
            middle = (x1 + x2) / 2
            path = f"M{x1},{y1} C{middle},{y1} {middle},{y2} {x2},{y2}"
            label_x, label_y = middle, min(y1, y2)
        else:
            path = (
                f"M{x1},{y1} L{x1 + bend},{y1} L{x1 + bend},{track} "
                f"L{x2 - bend},{track} L{x2 - bend},{y2} L{x2},{y2}"
            )
            label_x, label_y = (x1 + x2) / 2, track
        result_attribute = (
            f' data-result-value="{result}"' if result is not None else ""
        )
        dash = "" if value.quantum else ' stroke-dasharray="4 3"'
        color = _QUANTUM if value.quantum else _CLASSICAL
        parts = (
            f'<g data-value="{identifier}" data-stage="{value.stage}" '
            f'data-quantum="{str(value.quantum).lower()}" data-edge="'
            f'{"result-binding" if interface else "dataflow"}" '
            f'data-from-x="{x1}" data-from-y="{y1}" data-to-x="{x2}" data-to-y="{y2}"{result_attribute}>'
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.8"{dash}/>'
        )
        if annotation:
            parts += _text(
                label_x,
                label_y - 8,
                f"v{identifier} · {value.label}",
                limit=22,
                size=10,
                anchor="middle",
            )
            parts += _text(
                label_x,
                label_y + 15,
                value.type_label,
                limit=23,
                size=10,
                anchor="middle",
            )
        if value.quantum and value.width != 1:
            parts += _line(label_x - 3, label_y + 4, label_x + 3, label_y - 4)
            parts += _text(
                label_x + 8,
                label_y + 4,
                str(value.width) if value.width is not None else "symbolic",
                limit=9,
                size=9,
            )
        return parts + "</g>"

    def node(self, node: DisplayNode, wanted: set[int] | None = None) -> _Scene:
        """Measure an operation and place its interiors on the same time axis.

        Args:
            node (DisplayNode): Resolved native operation or region.
            wanted (set[int] | None): Used outgoing values; None keeps all outputs.

        Returns:
            _Scene: Typed interfaces and nested horizontal geometry.

        Raises:
            CapabilityError: If a routing, canvas, or output bound is exceeded.
        """
        outputs = self.ports(
            tuple(i for i in node.outputs if wanted is None or i in wanted)
        )
        native_inputs = node.inputs
        if node.kind == "exit" and node.opcode == "yield" and wanted is not None:
            native_inputs = tuple(i for i in native_inputs if i in wanted)
        inputs = self.ports(native_inputs, keep_literals=node.kind == "exit")
        if node.kind == "exit":
            outputs = inputs  # Exit bindings, not newly produced SSA values.
        width = 132.0
        parts = _SvgParts()
        children: _Scene | None = None
        result_ids: tuple[int, ...] = ()
        if node.children:
            if node.kind == "branch":
                alternatives = []
                bindings = []
                for native_child in node.children:
                    pairs = [
                        (source, target)
                        for source, target in zip(native_child.outputs, node.outputs)
                        if target in outputs
                    ]
                    child = self.node(native_child, {source for source, _ in pairs})
                    alternatives.append(child)
                    bindings.append(pairs)
                inner_width = max(child.width for child in alternatives)
                inner_height = sum(child.height + 28 for child in alternatives)
                inner_parts = _SvgParts()
                in_pins = []
                yy = 0.0
                branch_bindings = []
                for child, pairs in zip(alternatives, bindings):
                    inner_parts.append(_place(child, 0, yy))
                    in_pins.extend((i, x, y + yy) for i, x, y in child.inputs)
                    positions = {i: (x, y) for i, x, y in child.outputs}
                    for source, target in pairs:
                        if source in positions:
                            x, y = positions[source]
                            branch_bindings.append((source, target, x, y + yy))
                    yy += child.height + 28
                children = _Scene(
                    inner_width, inner_height, "".join(inner_parts), tuple(in_pins)
                )
            else:
                if node.kind in {"region", "iteration"}:
                    result_ids = node.outputs
                elif node.kind == "call":
                    result_ids = node.children[-1].inputs
                elif node.kind == "loop" and all(
                    child.opcode != "loop.body" for child in node.children
                ):
                    result_ids = node.children[-1].outputs
                desired = tuple(
                    source
                    for source, target in zip(result_ids, node.outputs)
                    if target in outputs
                )
                retained: tuple[int, ...] = ()
                if node.kind == "loop" and not result_ids:
                    # Body and normal-completion values are separate native
                    # interfaces. Keep relevant body yields without aliasing
                    # them to the unknown final carries entering else.
                    body, final = node.children[0], node.children[-1]
                    retained = tuple(
                        source
                        for source, target in zip(body.outputs, node.outputs)
                        if target in outputs
                    )
                    desired = tuple(
                        source
                        for source, target in zip(final.outputs, node.outputs)
                        if target in outputs
                    )
                children = self.scope(
                    node.children, node.inputs, desired, retained=retained
                )
            used = {i for i, _, _ in children.inputs}
            control_count = (
                3 if node.kind == "loop" else 1 if node.kind == "branch" else 0
            )
            inputs = self.ports(
                tuple(
                    i
                    for j, i in enumerate(node.inputs)
                    if j < control_count or i in used
                )
            )
            width = children.width + 120
        height = max(
            children.height + 124 if children else 0,
            76 + 44 * max(len(inputs), len(outputs), 1),
        )
        if width > 100_000 or height > 100_000:
            raise CapabilityError(
                "Q7401: diagram canvas exceeds its layout limit; fold loops or reduce inline_depth"
            )
        kind = (
            "expression"
            if node.kind == "classical" and self.classical == "expressions"
            else node.kind
        )
        parts.append(
            f'<g data-kind="{kind}" data-opcode="{html.escape(node.opcode, quote=True)}" data-direction="left-to-right">'
        )
        fill = (
            "#eef5f7"
            if node.kind in {"loop", "branch", "call"}
            else "#fff8ee"
            if node.kind == "classical"
            else "#ffffff"
        )
        parts.append(_rect(0, 0, width, height, fill))
        title = (
            node.opcode.split(".")[-1]
            if node.kind == "classical" and self.classical == "expanded"
            else node.label
            if node.kind
            in {"loop", "branch", "region", "iteration", "call", "classical"}
            else node.opcode.split(".")[-1]
        )
        parts.append(
            _text(14, 23, title, limit=max(8, min(90, int((width - 28) / 7))), size=12)
        )
        subtitle = (
            "Exclusive alternatives"
            if node.kind == "branch"
            else f"{node.kind.upper()}"
            + (f" · line {node.line}" if node.line is not None else "")
        )
        parts.append(_text(14, 41, subtitle, limit=35, size=9))
        if not children:
            literals = [
                self.display.values[i].label for i in node.inputs if i not in inputs
            ]
            if literals:
                parts.append(
                    _text(
                        14,
                        height - 13,
                        "args: " + ", ".join(literals),
                        limit=23,
                        size=10,
                    )
                )
        if node.kind == "classical" and self.classical == "expanded":
            parts.append(_text(14, height - 13, node.label, limit=20, size=10))
        in_pins = tuple((i, 0.0, 68.0 + index * 44) for index, i in enumerate(inputs))
        out_pins = tuple(
            (i, width, 68.0 + index * 44) for index, i in enumerate(outputs)
        )
        if children is not None:
            cx, cy = 60.0, 62.0
            # Interior wires precede child surfaces, so routing stays outside gates.
            incoming = {i: (x, y) for i, x, y in in_pins}
            for identifier, x, y in children.inputs:
                if identifier in incoming:
                    parts.append(
                        self.edge(
                            identifier,
                            incoming[identifier],
                            (x + cx, y + cy),
                            annotation=False,
                        )
                    )
            outgoing = {i: (x, y) for i, x, y in out_pins}
            if node.kind == "branch":
                result_bindings = branch_bindings
            else:
                positions = {i: (x, y) for i, x, y in children.outputs}
                result_bindings = [
                    (source, target, *positions[source])
                    for source, target in zip(result_ids, node.outputs)
                    if source in positions and target in outgoing
                ]
            for identifier, target, x, y in result_bindings:
                if target in outgoing:
                    parts.append(
                        self.edge(
                            identifier,
                            (x + cx, y + cy),
                            outgoing[target],
                            annotation=False,
                            interface=True,
                            result=target,
                        )
                    )
            parts.append(_place(children, cx, cy))
            if node.kind == "loop":
                note = (
                    "One symbolic body; loop results include zero iterations and exits."
                    if any(child.opcode == "loop.body" for child in node.children)
                    else "Iterations carry yields left → right; else follows normal completion."
                )
                parts.append(_text(14, height - 14, note, limit=105, size=10))
                if not result_ids:
                    # This is a control summary, not an SSA identity edge: zero
                    # iterations or an early exit can bypass the displayed yield.
                    for identifier, x, y in out_pins:
                        parts.append(
                            f'<circle cx="{x - 24}" cy="{y}" r="10" fill="#fff" stroke="{_BORDER}"/>'
                        )
                        parts.append(
                            self.edge(
                                identifier,
                                (x - 14, y),
                                (x, y),
                                annotation=False,
                                interface=True,
                            )
                        )
                    positions = {i: (x, y) for i, x, y in children.outputs}
                    final = node.children[-1]
                    for source, target in zip(final.outputs, node.outputs):
                        if source in positions and target in outgoing:
                            x, y = positions[source]
                            tx, ty = outgoing[target]
                            parts.append(
                                f'<g data-edge="control-summary"><title>Loop result includes zero iterations and exits; the displayed yield is one possible source.</title><path d="M{x + cx},{y + cy} L{tx - 34},{ty}" fill="none" stroke="#748a94" stroke-dasharray="2 4"/></g>'
                            )
                    parts.append(
                        _text(
                            width - 10,
                            height - 34,
                            "loop result",
                            limit=18,
                            size=10,
                            anchor="end",
                        )
                    )
        elif node.kind == "exit":
            for identifier, _, y in in_pins:
                parts.append(
                    self.edge(
                        identifier, (0, y), (width, y), annotation=False, interface=True
                    )
                )
        elif node.kind == "loop":
            parts.append(_text(14, height - 13, "Body folded", size=10))
        for identifier, x, y in (*in_pins, *out_pins):
            parts.append(self.pin(identifier, x, y))
        parts.append("</g>")
        return _Scene(width, height, "".join(parts), in_pins, out_pins)

    def scope(
        self,
        nodes: tuple[DisplayNode, ...],
        inputs: tuple[int, ...],
        outputs: tuple[int, ...] | None = None,
        *,
        retained: tuple[int, ...] = (),
    ) -> _Scene:
        """Connect a source-ordered sequence, including values bypassing operations.

        Args:
            nodes (tuple[DisplayNode, ...]): Native sequence in source order.
            inputs (tuple[int, ...]): Captured or entrypoint values.
            outputs (tuple[int, ...] | None): Bound exit values; None retains the root exit.
            retained (tuple[int, ...]): Relevant symbolic body yields; defaults to empty.

        Returns:
            _Scene: Connected sequence with local inputs and outputs.

        Raises:
            CapabilityError: If a routing, canvas, or output bound is exceeded.
        """
        wanted = set(outputs or ()) | set(retained)
        reverse_scenes = []
        for node in reversed(nodes):
            if self.classical != "expanded" and node.opcode == "qcore.constant":
                continue
            scene = self.node(
                node, wanted if node.kind != "exit" or outputs is not None else None
            )
            reverse_scenes.append(scene)
            wanted.update(i for i, _, _ in scene.inputs)
        scenes = list(reversed(reverse_scenes))
        used = {i for scene in scenes for i, _, _ in scene.inputs} | set(outputs or ())
        declared = list(self.ports(tuple(i for i in inputs if i in used)))
        known = set(declared)
        for scene in scenes:
            for identifier, _, _ in scene.inputs:
                if identifier not in known:
                    declared.append(identifier)
                    known.add(identifier)
            known.update(i for i, _, _ in scene.outputs)
        in_pins = tuple((i, 0.0, 68.0 + index * 44) for index, i in enumerate(declared))
        origins = {i: (x, y, -1) for i, x, y in in_pins}
        positions = []
        quantum_scope = any(
            self.display.values[i].quantum
            for scene in scenes
            for i, _, _ in (*scene.inputs, *scene.outputs)
        )
        xx = 128.0
        for scene in scenes:
            # Classical computations and measurements sit below the coherent
            # path. Source order remains the horizontal coordinate.
            coherent_output = any(
                self.display.values[i].quantum for i, _, _ in scene.outputs
            )
            coherent_input = any(
                self.display.values[i].quantum for i, _, _ in scene.inputs
            )
            yy = (
                0.0
                if not quantum_scope
                or coherent_output
                or coherent_input
                and not scene.outputs
                else 180.0
            )
            positions.append((xx, yy))
            xx += scene.width + 128
            if xx > 100_000:
                raise CapabilityError(
                    "Q7401: diagram canvas exceeds its layout limit; fold loops or reduce inline_depth"
                )
        width = max(328.0, xx)
        height = max(
            [
                100.0 + 44 * len(declared),
                *(s.height + y for s, (_, y) in zip(scenes, positions)),
            ]
        )
        edges = _SvgParts()
        track_count = 0
        for index, (scene, (xx, yy)) in enumerate(zip(scenes, positions)):
            for identifier, x, y in scene.inputs:
                sx, sy, producer = origins[identifier]
                ey = y + yy
                self.work += index - producer + 1
                if self.work > 1_000_000:
                    raise CapabilityError(
                        "Q7401: diagram routing exceeds its work limit; fold loops or reduce inline_depth"
                    )
                intervening = [
                    (s, pos)
                    for s, pos in zip(
                        scenes[producer + 1 : index], positions[producer + 1 : index]
                    )
                ]
                clear = sy == ey and all(
                    not pos[1] - 20 <= sy <= pos[1] + s.height + 20
                    for s, pos in intervening
                )
                track = None
                if intervening and not clear:
                    track = (
                        max(pos[1] + s.height for s, pos in intervening)
                        + 28
                        + track_count * 38
                    )
                    track_count += 1
                    height = max(height, track + 24)
                edges.append(self.edge(identifier, (sx, sy), (xx + x, ey), track=track))
            for identifier, x, y in scene.outputs:
                origins[identifier] = (xx + x, yy + y, index)
        out_pins = []
        for identifier in self.ports(outputs or (), keep_literals=True):
            if identifier in origins:
                sx, sy, _ = origins[identifier]
                y = 68.0 + len(out_pins) * 44
                edges.append(self.edge(identifier, (sx, sy), (width, y)))
                out_pins.append((identifier, width, y))
        if width > 100_000 or height > 100_000:
            raise CapabilityError(
                "Q7401: diagram canvas exceeds its layout limit; fold loops or reduce inline_depth"
            )
        parts = _SvgParts()
        parts.append("".join(edges))
        for scene, (xx, yy) in zip(scenes, positions):
            parts.append(_place(scene, xx, yy))
        for identifier, x, y in in_pins:
            parts.append(self.pin(identifier, x, y))
            value = self.display.values[identifier]
            parts.append(_text(x + 8, y - 12, value.label, limit=20, size=10))
        for identifier, x, y in out_pins:
            parts.append(self.pin(identifier, x, y))
        return _Scene(width, height, "".join(parts), in_pins, tuple(out_pins))


def _svg_document(width: float, height: float, content: str, label: str) -> str:
    """Wrap the rendered geometry in an accessible standalone SVG.

    Args:
        width (float): Canvas width.
        height (float): Canvas height.
        content (str): Escaped generated SVG elements.
        label (str): Accessible document title.

    Returns:
        str: Complete SVG.

    Raises:
        CapabilityError: If the SVG exceeds its output budget.
    """
    result = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">'
        f"<title>{html.escape(label)}</title><style>text{{font-family:system-ui,sans-serif;fill:#22343d}}</style>"
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>{content}</svg>'
    )
    if len(result.encode()) > _MAX_SVG_BYTES:
        raise CapabilityError("Q7401: rendered SVG exceeds its output limit")
    return result


def render_svg(display: Display, classical: str) -> str:
    """Lay out typed dataflow ports and structured regions from native facts.

    Args:
        display (Display): Verified native projection.
        classical (str): expressions or expanded.

    Returns:
        str: Standalone SVG with connected typed logical buses.

    Raises:
        CapabilityError: If the rendering exceeds its output budget.
    """
    layout = _Layout(display, classical)
    layout.collect_literals(display.nodes)
    scene = layout.scope(display.nodes, display.inputs)
    parts = _SvgParts()
    parts.append(_text(24, 28, display.entrypoint, limit=130, size=16))
    parts.append(
        _text(
            24,
            49,
            "Time → · solid: quantum bus / width · dashed: classical dependency · select a wire to follow its uses",
            limit=140,
            size=11,
        )
    )
    parts.append(_place(scene, 24, 76))
    return _svg_document(
        scene.width + 48, scene.height + 100, "".join(parts), display.entrypoint
    )


def _parameter_label(value: CircuitValue | ScalarParameter, depth: int = 0) -> str:
    """Format an already derived scalar expression without evaluating it.

    Args:
        value (CircuitValue | ScalarParameter): Existing native circuit projection value.
        depth (int): Current label nesting; defaults to zero.

    Returns:
        str: Bounded parameter annotation.
    """
    if depth > 5:
        return "…"
    if isinstance(value, ParameterValue):
        return value.name + "".join(f"[{part}]" for part in value.path)
    if isinstance(value, ScalarParameter):
        return _parameter_label(value.value, depth + 1)
    if isinstance(value, MeasurementBit):
        return f"m{value.index}"
    if isinstance(value, ClassicalExpr):
        operands = ", ".join(
            _parameter_label(item, depth + 1) for item in value.operands[:4]
        )
        return f"{value.opcode.split('.')[-1]}({operands})"[:120]
    if isinstance(value, tuple):
        return (
            "("
            + ", ".join(_parameter_label(item, depth + 1) for item in value[:4])
            + ")"
        )
    if value is None or isinstance(value, (bool, int, float, complex, str)):
        return str(value)[:120]
    return type(value).__name__


def _condition_bits(value: CircuitValue | ScalarParameter, depth: int = 0) -> set[int]:
    """Read measurement identities referenced by a native condition expression.

    Args:
        value (CircuitValue | ScalarParameter): Native condition tree.
        depth (int): Current expression depth; defaults to zero.

    Returns:
        set[int]: Referenced native measurement bit identities.

    Raises:
        CapabilityError: If condition presentation exceeds its work or nesting bound.
    """
    pending = [(value, depth)]
    seen: set[int] = set()
    bits: set[int] = set()
    visits = 0
    while pending:
        current, level = pending.pop()
        visits += 1
        if level > 32 or visits > 1_000_000:
            raise CapabilityError(
                "Q7401: circuit condition exceeds its presentation limit"
            )
        if isinstance(current, MeasurementBit):
            bits.add(current.index)
        elif id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, ScalarParameter):
                pending.append((current.value, level + 1))
            else:
                operands = (
                    current.operands
                    if isinstance(current, ClassicalExpr)
                    else current
                    if isinstance(current, tuple)
                    else ()
                )
                if len(pending) + len(operands) > 1_000_000:
                    raise CapabilityError(
                        "Q7401: circuit condition exceeds its presentation limit"
                    )
                pending.extend((operand, level + 1) for operand in operands)
    return bits


@dataclass
class _CircuitLayout:
    """Place native concrete instructions and exclusive regions on one time axis.

    Attributes:
        circuit (CircuitProgram): Native concrete projection.
        limit (int): Maximum visited instruction count.
        visits (int): Instruction count consumed by the layout.
        routing_work (int): Lane/slot combinations consumed across all regions.
    """

    circuit: CircuitProgram
    limit: int
    visits: int = 0
    routing_work: int = 0

    def block(self, block: CircuitBlock, caption: str, depth: int = 0) -> _Scene:
        """Draw a native block, anchoring alternatives at their parent instruction.

        Args:
            block (CircuitBlock): Already lowered native instructions.
            caption (str): Scope caption, including exclusive alternative labels.
            depth (int): Regional nesting depth; defaults to zero.

        Returns:
            _Scene: Horizontal circuit geometry in local coordinates.

        Raises:
            CapabilityError: If an instruction, canvas, or output bound is exceeded.
        """
        if depth > 32:
            raise CapabilityError("Q7401: concrete view exceeds its nesting limit")
        slots = []
        xx = 70.0
        lane_height = max(1, self.circuit.width) * 40 + 100.0
        # Each block draws every lane through its instruction slots. Bound that
        # product before constructing SVG fragments, including nested regions.
        self.routing_work += self.circuit.width * (len(block.instructions) + 1)
        if self.routing_work > 1_000_000:
            raise CapabilityError("Q7401: concrete routing exceeds its work limit")
        if lane_height > 100_000:
            raise CapabilityError("Q7401: concrete canvas exceeds its layout limit")
        height = lane_height
        for instruction in block.instructions:
            self.visits += 1
            if self.visits > self.limit:
                raise CapabilityError("Q7401: concrete view exceeds its output limit")
            name = instruction.opcode.split(".")[-1]
            regions = [
                self.block(
                    region,
                    f"{name} · alternative {i} (exclusive)"
                    if name == "if"
                    else f"{name} · region {i}",
                    depth + 1,
                )
                for i, region in enumerate(instruction.regions)
            ]
            slot_width = (
                max((region.width for region in regions), default=0) + 140
                if regions
                else 108.0
            )
            slots.append((instruction, xx, slot_width, regions))
            xx += slot_width
            height = max(
                height, lane_height + sum(region.height + 30 for region in regions)
            )
            if xx > 100_000 or height > 100_000:
                raise CapabilityError("Q7401: concrete canvas exceeds its layout limit")
        width = max(250.0, xx + 30)
        parts = _SvgParts()
        parts.append(_text(14, 22, caption, limit=80))
        for qubit in range(self.circuit.width):
            yy = 68 + qubit * 40
            parts.append(_text(14, yy + 4, f"q{qubit}", size=11))
            previous = 50.0
            for _, left, slot_width, regions in slots:
                if regions:
                    parts.append(_line(previous, yy, left, yy))
                    previous = left + slot_width
            parts.append(_line(previous, yy, width - 10, yy))
        measured: dict[int, tuple[float, float]] = {}
        for instruction, left, slot_width, regions in slots:
            name = instruction.opcode.split(".")[-1]
            if instruction.condition is not None:
                for bit in sorted(_condition_bits(instruction.condition)):
                    if bit in measured:
                        sx, sy = measured[bit]
                        track = lane_height - 18
                        parts.append(
                            f'<g data-condition-bit="m{bit}"><title>m{bit} feeds this condition</title><path d="M{sx},{sy} L{sx + 8},{track} L{left - 8},{track} L{left - 8},48 L{left},48" fill="none" stroke="{_CLASSICAL}" stroke-dasharray="4 3"/><circle cx="{left}" cy="48" r="3" fill="#fff" stroke="{_CLASSICAL}"/></g>'
                        )
            parts.append(
                f'<g data-gate="{html.escape(name, quote=True)}" data-direction="left-to-right" data-x="{left}">'
            )
            if regions:
                parts.append(
                    _rect(
                        left,
                        30,
                        slot_width,
                        lane_height
                        + sum(region.height + 30 for region in regions)
                        - 40,
                        "#eef5f7",
                    )
                )
                parts.append(
                    _text(
                        left + 12,
                        48,
                        "if " + _parameter_label(instruction.condition)
                        if instruction.condition is not None
                        else name,
                        limit=55,
                    )
                )
                ry = lane_height
                for region in regions:
                    for qubit in range(self.circuit.width):
                        yy = 68 + qubit * 40
                        parts.append(_line(left, yy, left + 60, ry + yy))
                        parts.append(
                            _line(
                                left + 60 + region.width - 10,
                                ry + yy,
                                left + slot_width,
                                yy,
                            )
                        )
                    parts.append(_place(region, left + 60, ry))
                    ry += region.height + 30
            else:
                xx = left + 54
                qubits = (*instruction.controls, *instruction.qubits)
                if qubits:
                    parts.append(
                        _line(xx, 68 + min(qubits) * 40, xx, 68 + max(qubits) * 40)
                    )
                controls, targets = instruction.controls, instruction.qubits
                glyph = name
                if name in {"cx", "ccx", "cz", "cp"}:
                    controls = (*controls, *targets[:-1])
                    targets = targets[-1:]
                    glyph = {"cx": "x", "ccx": "x", "cz": "z", "cp": "p"}[name]
                parameters = ", ".join(
                    _parameter_label(value) for value in instruction.parameters[:3]
                )
                if name == "measure":
                    for bit, qubit in zip(instruction.parameters, instruction.qubits):
                        if isinstance(bit, int):
                            measured[bit] = (xx, 68 + qubit * 40)
                    parameters = ", ".join(
                        f"m{value}" for value in instruction.parameters
                    )
                parts.append(
                    _text(xx, 46, parameters, limit=15, size=10, anchor="middle")
                )
                for qubit in controls:
                    parts.append(
                        f'<circle data-control="q{qubit}" cx="{xx}" cy="{68 + qubit * 40}" r="4" fill="{_QUANTUM}"/>'
                    )
                for qubit in targets:
                    yy = 68 + qubit * 40
                    if name in {"cx", "ccx"}:
                        parts.append(
                            f'<circle data-target="q{qubit}" cx="{xx}" cy="{yy}" r="10" fill="#fff" stroke="{_QUANTUM}"/>'
                        )
                        parts.append(_line(xx - 10, yy, xx + 10, yy))
                        parts.append(_line(xx, yy - 10, xx, yy + 10))
                    else:
                        parts.append(_rect(xx - 32, yy - 13, 64, 26))
                        parts.append(
                            _text(xx, yy + 4, glyph, limit=8, size=11, anchor="middle")
                        )
                if not qubits:
                    parts.append(
                        _text(xx, lane_height - 28, name, limit=12, anchor="middle")
                    )
                if instruction.condition is not None:
                    parts.append(
                        _text(
                            xx,
                            lane_height - 10,
                            "if " + _parameter_label(instruction.condition),
                            limit=15,
                            size=10,
                            anchor="middle",
                        )
                    )
            parts.append("</g>")
        return _Scene(width, height, "".join(parts))


def render_circuit(circuit: CircuitProgram, limit: int) -> str:
    """Draw a continuous horizontal circuit using authoritative native placement.

    Args:
        circuit (CircuitProgram): Authoritative concrete projection.
        limit (int): Maximum rendered instruction/region count.

    Returns:
        str: Standalone SVG with source-ordered qubit lanes.

    Raises:
        CapabilityError: If an instruction, canvas, or output bound is exceeded.
    """
    scene = _CircuitLayout(circuit, limit).block(circuit.body, "Time →")
    parts = _SvgParts()
    parts.append(
        _text(24, 28, circuit.name + " · concrete qubit view", limit=130, size=16)
    )
    parts.append(
        _text(
            24,
            46,
            "mN identifies a measurement bit; it does not show an observed outcome.",
            limit=100,
            size=11,
        )
    )
    parts.append(_place(scene, 20, 62))
    return _svg_document(
        scene.width + 40, scene.height + 80, "".join(parts), circuit.name
    )


def render_html(
    entrypoint: str,
    variants: dict[str, str],
    unavailable: dict[str, str],
    loops: str,
    classical: str,
    wires: str,
) -> str:
    """Create a portable viewer whose controls only change presentation.

    Args:
        entrypoint (str): Selected callable name.
        variants (dict[str, str]): Available loop/classical SVG combinations.
        unavailable (dict[str, str]): Native reasons an alternative is unavailable.
        loops (str): Initial loop display.
        classical (str): Initial classical display.
        wires (str): Register or concrete-qubit view.

    Returns:
        str: Self-contained HTML document with an initially visible SVG.

    Raises:
        CapabilityError: If combined alternatives exceed the HTML output budget.
    """
    identifier = "qamomile-" + uuid.uuid4().hex
    initial = f"{loops}:{classical}"
    footer = (
        "Time runs left → right. Alternatives share an instruction position; scroll horizontally or use Fit to inspect the whole circuit."
        if wires == "qubits"
        else unavailable.get(
            "expanded",
            "Time runs left → right. Select a wire to follow its uses. Scroll horizontally or use Fit; branch alternatives are exclusive.",
        )
    )
    payload = json.dumps(variants).replace("<", "\\u003c")
    buttons = "".join(
        f'<button type="button" data-loops="{mode}" aria-pressed="{str(mode == loops).lower()}" '
        f"{'disabled' if mode in unavailable or (wires == 'qubits' and mode != loops) else ''} "
        f'title="{html.escape(unavailable.get(mode, ""), quote=True)}">{label}</button>'
        for mode, label in (
            ("folded", "Fold loops"),
            ("body", "Open body"),
            ("expanded", "Expand iterations"),
        )
    )
    result = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(entrypoint)} — Qamomile</title>
<div id="{identifier}">
<style>
#{identifier}{{font:14px/1.5 system-ui,sans-serif;color:light-dark(#22343d,#e6eef1);background:light-dark(#fff,#172126);color-scheme:light dark;border:1px solid light-dark(#ccd9de,#40515b);border-radius:8px;overflow:hidden}}
#{identifier} header{{padding:14px 18px;border-bottom:1px solid light-dark(#ccd9de,#40515b);display:flex;align-items:center;gap:14px;flex-wrap:wrap}}
#{identifier} header strong{{overflow-wrap:anywhere;min-width:0;max-width:100%}}
#{identifier} button,#{identifier} select{{font:inherit;padding:6px 10px;border-radius:5px;border:1px solid light-dark(#ccd9de,#40515b);background:light-dark(#fff,#172126);color:inherit}}
#{identifier} button[aria-pressed=true]{{background:light-dark(#e8f2f5,#294652)}}
#{identifier} button:disabled{{opacity:.5}}
#{identifier} .diagram{{overflow:auto;max-height:none;background:#fff}}
#{identifier} .diagram>svg{{display:block}}
#{identifier} [data-value]{{cursor:pointer}}
#{identifier} .selected-port path,#{identifier} .selected-port circle{{stroke-width:3}}
#{identifier} .selected-port text{{font-weight:700;fill:#246b83}}
#{identifier} footer{{padding:10px 18px;font-size:12px}}
</style>
<header><strong>Qamomile · {html.escape(entrypoint)}</strong>{buttons}
<label>Classical <select aria-label="Classical operations" {"disabled" if wires == "qubits" else ""}><option value="expressions" {"selected" if classical == "expressions" else ""}>Expressions</option><option value="expanded" {"selected" if classical == "expanded" else ""}>Dependencies</option></select></label><button type="button" data-zoom="fit">Fit</button><button type="button" data-zoom="1">100%</button></header>
<div class="diagram" tabindex="0" aria-label="Algorithm diagram; scroll horizontally to follow time">{variants[initial]}</div><footer aria-live="polite">{html.escape(footer)}</footer>
<script>(()=>{{
const root=document.getElementById("{identifier}"),views={payload};
let loops={json.dumps(loops)},zoom=1;
const select=root.querySelector('select'),diagram=root.querySelector('.diagram');
function resize(){{const svg=diagram.querySelector('svg');if(!svg)return;const box=svg.viewBox.baseVal;const scale=zoom==='fit'?Math.min(1,diagram.clientWidth/box.width):zoom;svg.style.width=box.width*scale+'px';svg.style.height=box.height*scale+'px';}}
function update(){{const key=loops+':'+select.value;if(views[key])diagram.innerHTML=views[key];root.querySelectorAll('[data-loops]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.loops===loops)));resize();}}
root.querySelectorAll('[data-loops]').forEach(b=>b.addEventListener('click',()=>{{loops=b.dataset.loops;update();}}));
root.querySelectorAll('[data-zoom]').forEach(b=>b.addEventListener('click',()=>{{zoom=b.dataset.zoom==='fit'?'fit':1;resize();}}));
select.addEventListener('change',update);
function highlight(event){{const port=event.target.closest('[data-value]');if(!port)return;root.querySelectorAll('[data-value]').forEach(item=>item.classList.toggle('selected-port',item.dataset.value===port.dataset.value||item.dataset.resultValue===port.dataset.value));root.querySelector('footer').textContent='v'+port.dataset.value+' · highlighted uses and result bindings within this view.';}}
diagram.addEventListener('click',highlight);
diagram.addEventListener('keydown',event=>{{if(event.key==='Enter'||event.key===' '){{if(event.target.closest('[data-value]')){{event.preventDefault();highlight(event);}}}}}});
new ResizeObserver(resize).observe(diagram);resize();
}})();</script>
</div></html>'''
    if len(result.encode()) > _MAX_HTML_BYTES:
        raise CapabilityError("Q7401: interactive diagram exceeds its output limit")
    return result
