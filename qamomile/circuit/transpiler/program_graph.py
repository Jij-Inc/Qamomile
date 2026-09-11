"""Public compiler support for direct program-graph targets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import CallPolicy, InvokeOperation
from qamomile.circuit.transpiler.passes.analyze import (
    reject_control_flow_quantum_discard,
)
from qamomile.circuit.transpiler.passes.inline import InlinePass
from qamomile.circuit.transpiler.passes.validate_while import (
    ValidateWhileContractPass,
)
from qamomile.circuit.transpiler.prepared import PreparedModule

__all__ = ["inline_callables", "validate_program_graph_semantics"]


def inline_callables(
    block: Block,
    *,
    body_selector: Callable[[InvokeOperation], Block | None] | None = None,
) -> Block:
    """Expand inline-policy callables once with the compiler's default policy.

    Boxed calls, transformed calls, and recursive calls that remain after one
    pass are preserved. Direct program-graph targets use this narrow helper at
    target legalization boundaries without depending on the configurable
    compiler pass implementation.

    Args:
        block (Block): Hierarchical or traced semantic block to rewrite.
        body_selector (Callable[[InvokeOperation], Block | None] | None):
            Optional selector returning the body to inline, or ``None`` to
            retain a call boundary. Defaults to the ordinary effective body.

    Returns:
        Block: Block after one default callable-inlining pass.

    Raises:
        QubitConsumedError: If an invocation binds the same quantum resource
            to multiple formal operands.
    """
    return InlinePass(body_selector=body_selector).run(block)


def validate_program_graph_semantics(program: PreparedModule) -> None:
    """Validate shared semantics for a direct program-graph target.

    Circuit-family planning runs these checks during partial evaluation,
    analysis, and segmentation. A direct program-graph target preserves the
    prepared structure, so this helper runs only the non-destructive semantic
    checks. Inline-policy callables are expanded in the validation view so
    their formal values retain call-site provenance; the prepared program
    itself remains hierarchical.

    Args:
        program (PreparedModule): Prepared entrypoint and callable bodies.

    Raises:
        ValidationError: If a while condition is not measurement-backed.
        AffineTypeError: If structured control flow discards a quantum value.
        QubitConsumedError: If callable inlining binds one quantum resource to
            multiple formal operands.
    """
    blocks: list[tuple[Block, Mapping[str, Any]]] = [
        (inline_callables(program.entrypoint), program.bindings)
    ]
    blocks.extend(
        (inline_callables(definition.body), {})
        for definition in program.definitions.values()
        if definition.body is not None
        and definition.default_policy is not CallPolicy.INLINE
    )
    visited: set[int] = set()
    for block, bindings in blocks:
        if id(block) in visited:
            continue
        visited.add(id(block))
        ValidateWhileContractPass().run(block)
        reject_control_flow_quantum_discard(block.operations, dict(bindings))
