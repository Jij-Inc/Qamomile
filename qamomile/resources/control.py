"""Explicit circuit-resource models for coherent control realization."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from qamomile.ir import Block, Function, Module, Operation, Program, Value
from qamomile.types import logical_width

from .metrics import (
    CIRCUIT_CLEAN_ANCILLA_QUBITS,
    CIRCUIT_GATE_DEPTH,
    CIRCUIT_TOFFOLI_GATES,
    CIRCUIT_VIRTUAL_GATES,
    CIRCUIT_VIRTUAL_QUBITS,
    LOGICAL_GATES,
    LOGICAL_PEAK_QUBITS,
    LOGICAL_TOFFOLI_GATES,
    logical_operation_count,
)
from .projection import project
from .report import ResourceStudy
from .values import Evidence, EvidenceAuthority, ResourceValue

_MODEL_IDENTITY = "qamomile.clean_ancilla_toffoli_control.v1"
_MODEL_ASSUMPTIONS = (
    "each multi-control condition is computed into clean |0> ancillas by a "
    "linear Toffoli ladder and uncomputed after the controlled callable",
    "clean ancillas are reusable between sequential controlled calls",
    "the final singly controlled callable is counted as one circuit gate",
)


@dataclass(frozen=True, slots=True)
class _CleanAncillaControlModel:
    """Project coherent controls through a clean-ancilla Toffoli ladder."""

    @property
    def identity(self) -> str:
        """Return the stable model identity.

        Returns:
            Versioned identity recorded in the projected study.
        """

        return _MODEL_IDENTITY

    def project(self, study: ResourceStudy) -> Mapping[str, ResourceValue]:
        """Return circuit coordinates for one immutable logical study.

        Args:
            study: Logical resource study retaining its verified source.

        Returns:
            Circuit gate, Toffoli, depth, width, and clean-ancilla metrics.

        Raises:
            ValueError: If control widths or call multiplicities are not
                concrete, or another unresolved logical-gate contributor is
                present.
            TypeError: If the retained Module or Program is malformed.
        """

        sites = _control_sites(study)
        evidence = (
            Evidence(
                EvidenceAuthority.TRUSTED_AXIOM,
                _MODEL_IDENTITY,
                details=(("control_sites", str(len(sites))),),
            ),
        )
        control_calls = sum(count for _, count in sites)
        generated_toffoli_gates = sum(
            2 * max(width - 1, 0) * count for width, count in sites
        )
        realized_control_gates = control_calls + generated_toffoli_gates

        logical_gates = study.metric(LOGICAL_GATES)
        logical_lower, _ = logical_gates.numeric_enclosure()
        if not isinstance(logical_lower, int):
            raise ValueError(
                "clean-ancilla control projection requires a concrete logical "
                "gate lower endpoint"
            )
        unresolved = tuple(
            item
            for item in logical_gates.unknown_contributors
            if "controlled transform" not in item
        )
        if unresolved:
            raise ValueError(
                "clean-ancilla control projection cannot resolve: "
                + ", ".join(unresolved)
            )
        base_gates = logical_lower - control_calls
        if base_gates < 0:
            raise TypeError("control-call count exceeds the logical gate lower bound")
        total_gates = base_gates + realized_control_gates

        logical_toffoli_lower, _ = study.metric(
            LOGICAL_TOFFOLI_GATES
        ).numeric_enclosure()
        if not isinstance(logical_toffoli_lower, int):
            raise ValueError(
                "clean-ancilla control projection requires a concrete logical "
                "Toffoli lower endpoint"
            )
        toffoli_gates = logical_toffoli_lower + generated_toffoli_gates

        logical_qubits = study.metric(LOGICAL_PEAK_QUBITS)
        qubit_lower, qubit_upper = logical_qubits.numeric_enclosure()
        if (
            not isinstance(qubit_lower, int)
            or not isinstance(qubit_upper, int)
            or qubit_lower != qubit_upper
        ):
            raise ValueError(
                "clean-ancilla control projection requires a concrete logical "
                "peak width"
            )
        clean_ancillas = max((max(width - 1, 0) for width, _ in sites), default=0)
        circuit_qubits = qubit_lower + clean_ancillas

        return {
            metric.identifier: ResourceValue.equality(
                metric,
                value,
                evidence=evidence,
                assumptions=_MODEL_ASSUMPTIONS,
            )
            for metric, value in (
                (CIRCUIT_VIRTUAL_GATES, total_gates),
                (CIRCUIT_TOFFOLI_GATES, toffoli_gates),
                (CIRCUIT_GATE_DEPTH, total_gates),
                (CIRCUIT_CLEAN_ANCILLA_QUBITS, clean_ancillas),
                (CIRCUIT_VIRTUAL_QUBITS, circuit_qubits),
            )
        }


def clean_ancilla_controls(study: ResourceStudy) -> ResourceStudy:
    """Project controlled calls through the clean-ancilla Toffoli model.

    Logical facts remain untouched. The returned study adds circuit-coordinate
    metrics under explicit model evidence and assumptions. A control set of
    width ``k >= 2`` uses ``k - 1`` clean ancillas and ``2(k - 1)`` Toffoli
    gates to compute and uncompute the conjunction; the resulting singly
    controlled callable contributes one additional circuit gate.

    Args:
        study: Logical study produced by :func:`qamomile.resources.logical`.

    Returns:
        A new study containing clean-ancilla circuit metrics.

    Raises:
        TypeError: If ``study`` is not a resource study or its retained source
            violates the verified facade.
        ValueError: If the requested circuit totals cannot be proved from
            concrete control widths and multiplicities.
    """

    return project(study, _CleanAncillaControlModel())


def _control_sites(study: ResourceStudy) -> tuple[tuple[int, int], ...]:
    """Return concrete ``(control width, multiplicity)`` model inputs.

    Args:
        study: Logical study retaining QCore and local contributions.

    Returns:
        Control width and analyzed invocation count for every source site.

    Raises:
        TypeError: If the retained Module or Program is unavailable or malformed.
        ValueError: If a control width or multiplicity is symbolic.
    """

    source = study._semantic_source()
    if not isinstance(source, (Module, Program)):
        raise TypeError(
            "clean-ancilla control projection requires a verified Module or Program"
        )
    widths = _control_widths(source)
    contributions = study.contributions(logical_operation_count("qcore.control_call"))
    sites: list[tuple[int, int]] = []
    for contribution in contributions:
        key = (contribution.owner, contribution.source)
        try:
            width = widths[key]
        except KeyError as error:
            raise TypeError(
                "control-call resource contribution has no matching QCore site"
            ) from error
        lower, upper = contribution.value.numeric_enclosure()
        if not isinstance(lower, int) or lower != upper:
            raise ValueError(
                "clean-ancilla control projection requires concrete call "
                f"multiplicity at {contribution.source}"
            )
        sites.append((width, lower))
    return tuple(sites)


def _control_widths(
    source: Module | Program,
) -> dict[tuple[str, str], int]:
    """Read fixed control widths from verified QCore operand types.

    Args:
        source: Verified Module or specialized Program.

    Returns:
        Mapping from callable/source identity to positive control width.

    Raises:
        TypeError: If SSA references are malformed.
        ValueError: If a control width is symbolic or empty.
    """

    widths: dict[tuple[str, str], int] = {}
    for function in source.functions:
        values = _function_values(function)
        for operation in _operations(function.body):
            if operation.opcode != "qcore.control_call":
                continue
            try:
                control = values[operation.operands[0]]
            except (IndexError, KeyError) as error:
                raise TypeError(
                    "control call references an absent SSA value"
                ) from error
            width = logical_width(control.type)
            if not isinstance(width, int) or width <= 0:
                raise ValueError("clean-ancilla model requires a fixed control width")
            key = (function.symbol, _source_location(operation))
            if key in widths:
                raise TypeError("multiple control calls share one source identity")
            widths[key] = width
    return widths


def _function_values(function: Function) -> dict[int, Value]:
    """Collect every SSA definition in one structured function.

    Args:
        function: Verified QCore function.

    Returns:
        Mapping from SSA identifier to its typed value.
    """

    values = {parameter.value.id: parameter.value for parameter in function.parameters}
    for block in _blocks(function.body):
        values.update((argument.id, argument) for argument in block.arguments)
        for operation in block.operations:
            values.update((result.id, result) for result in operation.results)
    return values


def _blocks(block: Block) -> Iterator[Block]:
    """Yield one block and all nested structured blocks.

    Args:
        block: Root QCore block.

    Yields:
        The root block followed by nested region blocks in source order.
    """

    yield block
    for operation in block.operations:
        for region in operation.regions:
            for nested in region.blocks:
                yield from _blocks(nested)


def _operations(block: Block) -> Iterator[Operation]:
    """Yield operations from one block and its nested regions.

    Args:
        block: Root QCore block.

    Yields:
        Operations in deterministic pre-order.
    """

    for current in _blocks(block):
        yield from current.operations


def _source_location(operation: Operation) -> str:
    """Return the source identity used by resource contributions.

    Args:
        operation: Source-addressable QCore operation.

    Returns:
        URI, line, and column in contribution format.
    """

    span = operation.span
    return f"{span.uri}:{span.start_line}:{span.start_column}"


__all__ = ["clean_ancilla_controls"]
