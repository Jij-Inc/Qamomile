"""Immutable display records and validation of the native projection protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qamomile.errors import CapabilityError


@dataclass(frozen=True, slots=True)
class DisplayValue:
    """A native-derived typed port, identified within one diagram.

    Attributes:
        id (int): Unique display identity; not a physical qubit number.
        label (str): Source hint or bounded symbolic expression.
        type_label (str): Logical type and numeric interpretation.
        quantum (bool): Native coherent-containment fact.
        stage (str): Latest availability, or Value for unstaged data.
        width (int | None): Native coherent width; None for symbolic/classical values.
    """

    id: int
    label: str
    type_label: str
    quantum: bool
    stage: str
    width: int | None


@dataclass(frozen=True, slots=True)
class DisplayNode:
    """A resolved operation or control region for presentation.

    Attributes:
        kind (str): Native closed display category.
        opcode (str): Original operation or structural marker.
        label (str): Bounded human-readable expression.
        inputs (tuple[int, ...]): Ordered incoming display value IDs.
        outputs (tuple[int, ...]): Ordered outgoing display value IDs.
        children (tuple[DisplayNode, ...]): Nested regions and operations.
        line (int | None): Original source line, if available.
    """

    kind: str
    opcode: str
    label: str
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    children: tuple[DisplayNode, ...]
    line: int | None


@dataclass(frozen=True, slots=True)
class Display:
    """One complete native-derived view.

    Attributes:
        entrypoint (str): Selected callable identity.
        values (tuple[DisplayValue, ...]): Deterministic port table.
        inputs (tuple[int, ...]): Entrypoint input IDs.
        nodes (tuple[DisplayNode, ...]): Ordered root nodes.
    """

    entrypoint: str
    values: tuple[DisplayValue, ...]
    inputs: tuple[int, ...]
    nodes: tuple[DisplayNode, ...]


def decode_projection(projection: Any) -> Display:
    """Validate protocol shape and expose immutable native records.

    Args:
        projection (Any): Result from the required native display entrypoint.

    Returns:
        Display: Immutable projection with checked local reference IDs.

    Raises:
        CapabilityError: If the native protocol is missing or malformed.
    """
    try:
        values = tuple(
            DisplayValue(
                row.id, row.label, row.type_label, row.quantum, row.stage, row.width
            )
            for row in projection.values
        )
        for index, value in enumerate(values):
            if (
                type(value.id) is not int
                or value.id != index
                or type(value.quantum) is not bool
                or not isinstance(value.label, str)
                or not isinstance(value.type_label, str)
                or value.stage not in {"Value", "Const", "Param", "Shot"}
                or (
                    value.width is not None
                    and (type(value.width) is not int or value.width < 0)
                )
            ):
                raise ValueError("invalid display port")
        inputs = _references(projection.inputs, len(values))
        nodes = tuple(_decode_node(row, len(values), 0) for row in projection.nodes)
        if not isinstance(projection.entrypoint, str):
            raise ValueError("invalid entrypoint")
        return Display(projection.entrypoint, values, inputs, nodes)
    except (AttributeError, TypeError, ValueError) as error:
        raise CapabilityError(
            "Q7404: native visualization projection is incompatible"
        ) from error


def _references(values: Any, count: int) -> tuple[int, ...]:
    """Check a row of display-local IDs without deriving semantic facts.

    Args:
        values (Any): Native reference row.
        count (int): Size of the associated port table.

    Returns:
        tuple[int, ...]: Checked ordered IDs.

    Raises:
        ValueError: If a reference lies outside the table.
    """
    result = tuple(values)
    if any(type(value) is not int or not 0 <= value < count for value in result):
        raise ValueError("invalid display reference")
    return result


def _decode_node(row: Any, count: int, depth: int) -> DisplayNode:
    """Decode one bounded native node and its ordered children.

    Args:
        row (Any): Native node.
        count (int): Port table size.
        depth (int): Current presentation nesting.

    Returns:
        DisplayNode: Immutable node.

    Raises:
        ValueError: If a node violates the closed protocol or depth limit.
    """
    if (
        depth > 32
        or row.kind
        not in {
            "quantum",
            "classical",
            "structure",
            "call",
            "loop",
            "iteration",
            "branch",
            "region",
            "exit",
        }
        or not isinstance(row.opcode, str)
        or not isinstance(row.label, str)
        or (row.line is not None and type(row.line) is not int)
    ):
        raise ValueError("invalid display node")
    return DisplayNode(
        row.kind,
        row.opcode,
        row.label,
        _references(row.inputs, count),
        _references(row.outputs, count),
        tuple(_decode_node(child, count, depth + 1) for child in row.children),
        row.line,
    )
