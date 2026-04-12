"""Entrypoint validation pass for top-level transpilation."""

from __future__ import annotations

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.transpiler.errors import (
    EntrypointValidationError,
    ValidationError,
)
from qamomile.circuit.transpiler.passes import Pass


class EntrypointValidationPass(Pass[Block, Block]):
    """Validate top-level entrypoint constraints."""

    @property
    def name(self) -> str:
        return "validate_entrypoint"

    def run(self, input: Block) -> Block:
        if input.kind not in (BlockKind.HIERARCHICAL, BlockKind.TRACED):
            raise ValidationError(
                f"EntrypointValidationPass expects HIERARCHICAL or TRACED block, "
                f"got {input.kind}",
            )

        quantum_inputs = [
            v.name for v in input.input_values if hasattr(v, "type") and v.type.is_quantum()
        ]
        quantum_outputs = [
            v.name for v in input.output_values if hasattr(v, "type") and v.type.is_quantum()
        ]

        if not quantum_inputs and not quantum_outputs:
            return input

        details = []
        if quantum_inputs:
            details.append(f"quantum inputs={quantum_inputs}")
        if quantum_outputs:
            details.append(f"quantum outputs={quantum_outputs}")

        raise EntrypointValidationError(
            "Top-level kernels passed to transpile()/to_circuit() must have "
            "classical inputs and outputs only. "
            f"Found {', '.join(details)}. "
            "Use the quantum-I/O kernel as a subroutine inside a classical-I/O "
            "entrypoint, or call build() when you only need tracing/composition.",
        )
