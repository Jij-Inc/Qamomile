"""Entrypoint validation pass for top-level transpilation."""

from __future__ import annotations

from qamomile.circuit.ir.block import Block
from qamomile.circuit.transpiler.errors import EntrypointValidationError
from qamomile.circuit.transpiler.passes import Pass


class EntrypointValidationPass(Pass[Block, Block]):
    """Validate top-level entrypoint constraints."""

    @property
    def name(self) -> str:
        """Return the pass name used in diagnostics.

        Returns:
            str: The literal ``"validate_entrypoint"``.
        """
        return "validate_entrypoint"

    def run(self, input: Block) -> Block:
        """Check that the block qualifies as an executable entrypoint.

        The classical-I/O requirement is kind-agnostic; the pass accepts
        every ``BlockKind`` so it can guard both the QKernel entry of
        ``transpile()`` (HIERARCHICAL / TRACED) and the deserialized-IR
        entry of ``transpile_block()`` (AFFINE / ANALYZED).

        Args:
            input (Block): The candidate entrypoint block.

        Returns:
            Block: ``input`` unchanged when validation passes.

        Raises:
            EntrypointValidationError: If the block has quantum inputs
                or outputs (entrypoints must be classical-I/O only).
        """

        quantum_inputs = [
            v.name
            for v in input.input_values
            if hasattr(v, "type") and v.type.is_quantum()
        ]
        quantum_outputs = [
            v.name
            for v in input.output_values
            if hasattr(v, "type") and v.type.is_quantum()
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
