"""Unified block representation for all pipeline stages."""

from __future__ import annotations

import dataclasses
from enum import Enum, auto
from typing import TYPE_CHECKING

from qamomile.circuit.ir.value import Value
from qamomile.circuit.ir.operation import Operation

if TYPE_CHECKING:
    pass


class BlockKind(Enum):
    """Classification of block structure for pipeline stages."""

    HIERARCHICAL = auto()  # May contain CallBlockOperations
    LINEAR = auto()  # No CallBlockOperations, For/If preserved
    ANALYZED = auto()  # Validated and dependency-analyzed


@dataclasses.dataclass
class Block:
    """Unified block representation for all pipeline stages.

    Replaces both BlockValue and Graph with a single structure.
    The `kind` field indicates which pipeline stage this block is at.
    """

    name: str = ""
    label_args: list[str] = dataclasses.field(default_factory=list)
    input_values: list[Value] = dataclasses.field(default_factory=list)
    output_values: list[Value] = dataclasses.field(default_factory=list)
    operations: list[Operation] = dataclasses.field(default_factory=list)

    # Pipeline stage indicator
    kind: BlockKind = BlockKind.HIERARCHICAL

    # Parameters (unbound values for circuit parameters)
    parameters: dict[str, Value] = dataclasses.field(default_factory=dict)

    # Dependency information (populated after analysis pass)
    _dependency_graph: dict[str, set[str]] | None = dataclasses.field(
        default=None, repr=False
    )

    def unbound_parameters(self) -> list[str]:
        """Return list of unbound parameter names."""
        return list(self.parameters.keys())

    def is_linear(self) -> bool:
        """Check if block contains no CallBlockOperations."""
        return self.kind in (BlockKind.LINEAR, BlockKind.ANALYZED)

    @classmethod
    def from_block_value(
        cls,
        block_value: "BlockValue",
        parameters: dict[str, Value] | None = None,
    ) -> "Block":
        """Create a Block from a BlockValue.

        Args:
            block_value: The BlockValue to convert
            parameters: Optional parameter bindings

        Returns:
            A new Block in HIERARCHICAL state
        """

        return cls(
            name=block_value.name,
            label_args=block_value.label_args,
            input_values=block_value.input_values,
            output_values=block_value.return_values,
            operations=block_value.operations,
            kind=BlockKind.HIERARCHICAL,
            parameters=parameters or {},
        )


# Import BlockValue for type checking
if TYPE_CHECKING:
    from qamomile.circuit.ir.block_value import BlockValue
