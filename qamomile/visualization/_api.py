"""Portable diagrams from the native compiler's read-only projections."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from qamomile._native import compiler as native
from qamomile.circuit import CircuitProgram, lower_to_circuit
from qamomile.errors import CapabilityError
from qamomile.ir import Program
from qamomile.serialization import encode_analysis_request
from qamomile.source import ProgramFamily, QKernel

from ._export import export_static
from ._model import Display, DisplayNode, DisplayValue, decode_projection
from ._render import render_circuit, render_html, render_svg

LoopMode = Literal["folded", "body", "expanded"]
ClassicalMode = Literal["expressions", "expanded"]
WireMode = Literal["registers", "qubits"]


@dataclass(frozen=True, slots=True)
class Diagram:
    """Read-only diagram supporting notebook display and portable export.

    Attributes:
        display (Display): Native-derived structured view.
        loops (LoopMode): Selected loop presentation.
        classical (ClassicalMode): Connected expression tiles with inline
            literals, or all native classical operations and their dependencies.
        wires (WireMode): Typed logical ports or native concrete qubit lanes.
    """

    display: Display
    loops: LoopMode
    classical: ClassicalMode
    wires: WireMode
    _payload: Any = field(repr=False, compare=False)
    _family: bool = field(repr=False, compare=False)
    _inline_depth: int = field(repr=False, compare=False)
    _max_nodes: int = field(repr=False, compare=False)
    _circuit: CircuitProgram | None = field(default=None, repr=False, compare=False)

    @property
    def values(self) -> tuple[DisplayValue, ...]:
        """Return immutable typed ports from this view.

        Returns:
            tuple[DisplayValue, ...]: Native types, stages, widths, and expressions.
        """
        return self.display.values

    @property
    def nodes(self) -> tuple[DisplayNode, ...]:
        """Return structured nodes with explicit SSA dataflow interfaces.

        Returns:
            tuple[DisplayNode, ...]: Root operations and control regions.
        """
        return self.display.nodes

    def to_svg(self) -> str:
        """Render the selected presentation as standalone SVG.

        Returns:
            str: Escaped, self-contained vector diagram.

        Raises:
            CapabilityError: If the rendered diagram exceeds its output budget.
        """
        if self._circuit is not None:
            return render_circuit(self._circuit, self._max_nodes)
        return render_svg(self.display, self.classical)

    def to_html(self) -> str:
        """Render a self-contained viewer with local presentation controls.

        Returns:
            str: Portable HTML, including bounded available loop presentations.

        Raises:
            CapabilityError: If the selected view exceeds a rendering budget.
        """
        variants: dict[str, str] = {}
        unavailable: dict[str, str] = {}
        if self._circuit is not None:
            variants[f"{self.loops}:{self.classical}"] = self.to_svg()
        else:
            for mode in ("folded", "body", "expanded"):
                try:
                    view = (
                        self.display
                        if mode == self.loops
                        else _project(
                            self._payload,
                            self._family,
                            mode,
                            self._inline_depth,
                            self._max_nodes,
                        )
                    )
                    for classical in ("expressions", "expanded"):
                        variants[f"{mode}:{classical}"] = render_svg(view, classical)
                except CapabilityError as error:
                    if mode == self.loops:
                        raise
                    unavailable[mode] = str(error)
        return render_html(
            self.display.entrypoint,
            variants,
            unavailable,
            self.loops,
            self.classical,
            self.wires,
        )

    def save(self, path: str | Path, *, scale: float = 1.0) -> Path:
        """Save the diagram as SVG, standalone HTML, PNG, or vector PDF.

        PNG and PDF use the selected static SVG presentation and require the
        visualization extra and a system Cairo installation.

        Args:
            path (str | Path): Destination ending in .svg, .html, .png, or .pdf.
            scale (float): Positive finite image/page scale for PNG and PDF;
                defaults to 1.0. SVG and HTML require the default scale.

        Returns:
            Path: Written destination.

        Raises:
            TypeError: If scale is not a real number.
            ValueError: If the requested format or scale is unsupported.
            OSError: If the destination cannot be written.
            CapabilityError: If rendering exceeds a bounded output limit or
                the optional exporter is unavailable.
        """
        destination = Path(path)
        suffix = destination.suffix.lower()
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise TypeError("scale must be a real number")
        if suffix in {".png", ".pdf"}:
            destination.write_bytes(export_static(self.to_svg(), suffix, scale))
            return destination
        if scale != 1.0:
            raise ValueError("scale is supported only for PNG and PDF")
        if suffix == ".svg":
            content = self.to_svg()
        elif suffix == ".html":
            content = self.to_html()
        else:
            raise ValueError("diagram output must end in .svg, .html, .png, or .pdf")
        destination.write_text(content, encoding="utf-8")
        return destination

    def _repr_html_(self) -> str:
        """Display the interactive diagram in a notebook.

        Returns:
            str: Self-contained viewer.
        """
        return self.to_html()


def draw(
    source: Program | ProgramFamily | QKernel[Any, Any],
    *,
    loops: LoopMode = "body",
    inline_depth: int = 1,
    classical: ClassicalMode = "expressions",
    wires: WireMode = "registers",
    max_nodes: int = 2048,
) -> Diagram:
    """Draw a verified algorithm with a consistent left-to-right time axis.

    Args:
        source (Program | ProgramFamily | QKernel): Specialized program or
            symbolic source family. QKernel source is compiled without tracing.
        loops (LoopMode): Fold loops, show one symbolic body, or expand known
            normally yielding ranges. Defaults to body.
        inline_depth (int): Ordinary-call expansion depth, between 0 and 32.
        classical (ClassicalMode): Connected expression tiles with inline
            literals, or all native classical operations and their dependencies.
        wires (WireMode): Connected typed register buses, or concrete native lanes.
            Qubit mode requires a Program and expanded loops.
        max_nodes (int): Native output bound between 1 and 16384; defaults to 2048.
            Concrete views also bound the sum of qubit lanes and instructions.

    Returns:
        Diagram: Notebook-displayable immutable diagram with portable export.

    Raises:
        TypeError: If the source or numeric options have incompatible types.
        ValueError: If presentation options are invalid.
        CapabilityError: If the native API is missing, expansion is unsupported,
            or work/output bounds are exceeded.
    """
    if (
        loops not in {"folded", "body", "expanded"}
        or classical not in {"expressions", "expanded"}
        or wires not in {"registers", "qubits"}
    ):
        raise ValueError("invalid loops, classical, or wires presentation")
    if type(inline_depth) is not int or type(max_nodes) is not int:
        raise TypeError("inline_depth and max_nodes must be integers")
    if not 0 <= inline_depth <= 32 or not 1 <= max_nodes <= 16384:
        raise ValueError("inline_depth must be 0..32 and max_nodes 1..16384")
    if isinstance(source, QKernel):
        source = source.family()
    if isinstance(source, Program):
        payload, family = source._native_program(), False
    elif isinstance(source, ProgramFamily):
        payload, family = (
            encode_analysis_request(source.module, source.entrypoint),
            True,
        )
    else:
        raise TypeError("draw expects a Program, ProgramFamily, or QKernel")
    if wires == "qubits" and (family or loops != "expanded"):
        raise ValueError(
            "qubit wires require a specialized Program and loops='expanded'"
        )
    display = _project(payload, family, loops, inline_depth, max_nodes)
    circuit = None
    if wires == "qubits":
        if any(
            value.quantum and (value.width is not None and value.width > max_nodes)
            for value in display.values
        ):
            raise CapabilityError(
                "Q7401: qubit lanes exceed max_nodes; increase max_nodes or use registers"
            )
        assert isinstance(source, Program)
        circuit = lower_to_circuit(source)
        if circuit.width + circuit.operation_count > max_nodes:
            raise CapabilityError(
                "Q7401: concrete qubit lanes and instructions exceed max_nodes"
            )
    return Diagram(
        display,
        loops,
        classical,
        wires,
        payload,
        family,
        inline_depth,
        max_nodes,
        circuit,
    )


def _project(payload: Any, family: bool, loops: str, depth: int, limit: int) -> Display:
    """Call the authoritative native projector and validate its protocol.

    Args:
        payload (Any): Verified owner or selected analysis-request bytes.
        family (bool): Whether the request selects a symbolic family.
        loops (str): Requested loop presentation.
        depth (int): Ordinary-call expansion bound.
        limit (int): Maximum display nodes and values.

    Returns:
        Display: Immutable native projection.

    Raises:
        CapabilityError: If native functionality is missing or rejects the request.
    """
    projector = getattr(native, "project_visualization", None)
    if not callable(projector):
        raise CapabilityError(
            "Q7404: native algorithm visualization is unavailable; reinstall Qamomile"
        )
    try:
        projection = projector(
            payload, family=family, loops=loops, inline_depth=depth, max_nodes=limit
        )
    except ValueError as error:
        raise CapabilityError(str(error)) from error
    return decode_projection(projection)
