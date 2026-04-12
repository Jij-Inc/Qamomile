"""Unified block representation for all pipeline stages."""

from __future__ import annotations

import dataclasses
from enum import Enum, auto
from typing import TYPE_CHECKING

from qamomile.circuit.ir.value import Value

if TYPE_CHECKING:
    from qamomile.circuit.ir.operation import Operation
    from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation


class BlockKind(Enum):
    """Classification of block structure for pipeline stages."""

    TRACED = auto()  # Direct output of frontend tracing / build()
    HIERARCHICAL = auto()  # May contain CallBlockOperations
    AFFINE = auto()  # No CallBlockOperations, For/If preserved
    ANALYZED = auto()  # Validated and dependency-analyzed


@dataclasses.dataclass
class Block:
    """Unified block representation for all pipeline stages.

    Replaces the older traced and callable IR wrappers with a single structure.
    The `kind` field indicates which pipeline stage this block is at.
    """

    name: str = ""
    label_args: list[str] = dataclasses.field(default_factory=list)
    input_values: list[Value] = dataclasses.field(default_factory=list)
    output_values: list[Value] = dataclasses.field(default_factory=list)
    output_names: list[str] = dataclasses.field(default_factory=list)
    operations: list["Operation"] = dataclasses.field(default_factory=list)

    # Pipeline stage indicator
    kind: BlockKind = BlockKind.HIERARCHICAL

    # Parameters (unbound values for circuit parameters)
    parameters: dict[str, Value] = dataclasses.field(default_factory=dict)

    # Dependency information (populated after analysis pass)
    _dependency_graph: dict[str, set[str]] | None = dataclasses.field(
        default=None, repr=False
    )

    def __post_init__(self):
        if self.label_args and len(self.label_args) != len(self.input_values):
            raise ValueError(
                f"label_args length ({len(self.label_args)}) must match "
                f"input_values length ({len(self.input_values)})"
            )

    def unbound_parameters(self) -> list[str]:
        """Return list of unbound parameter names."""
        return list(self.parameters.keys())

    def is_affine(self) -> bool:
        """Check if block contains no CallBlockOperations."""
        return self.kind in (BlockKind.AFFINE, BlockKind.ANALYZED)

    def call(self, **kwargs: Value) -> "CallBlockOperation":
        """Create a CallBlockOperation against this block."""
        from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation

        inputs = [kwargs[label] for label in self.label_args]
        dummy_inputs = {v.logical_id: idx for idx, v in enumerate(self.input_values)}

        results = []
        for dummy_return in self.output_values:
            if dummy_return.logical_id in dummy_inputs:
                input_idx = dummy_inputs[dummy_return.logical_id]
                results.append(inputs[input_idx].next_version())
            else:
                results.append(dummy_return)

        return CallBlockOperation(
            block=self,
            operands=inputs,
            results=results,
        )
