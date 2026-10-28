"""Validate execution-mode constraints from cached kernel effects."""

from __future__ import annotations

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.effect import callable_bodies, format_kernel_effects
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.passes import Pass


def _contains_expval(block: Block, seen: set[int]) -> bool:
    """Return whether a block or reachable owned body contains expval.

    Args:
        block (Block): Semantic block to inspect.
        seen (set[int]): Block identities already visited.

    Returns:
        bool: ``True`` when expectation-value execution is reachable.
    """
    identity = id(block)
    if identity in seen:
        return False
    seen.add(identity)
    return _operations_contain_expval(block.operations, seen)


def _operations_contain_expval(
    operations: list[Operation],
    seen: set[int],
) -> bool:
    """Return whether operations or their semantic bodies contain expval.

    Args:
        operations (list[Operation]): Operation sequence to inspect.
        seen (set[int]): Block identities already visited.

    Returns:
        bool: ``True`` when an ``ExpvalOp`` is reachable.
    """
    for operation in operations:
        if isinstance(operation, ExpvalOp):
            return True
        if isinstance(operation, HasNestedOps) and any(
            _operations_contain_expval(list(region.operations), seen)
            for region in operation.nested_regions()
        ):
            return True
        if isinstance(operation, InvokeOperation) and operation.definition is not None:
            if any(
                _contains_expval(body, seen)
                for body in callable_bodies(
                    operation.definition,
                    operation.transform,
                )
            ):
                return True
        if isinstance(operation, ControlledUOperation):
            if operation.block is not None and _contains_expval(operation.block, seen):
                return True
        if isinstance(operation, InverseBlockOperation):
            for owned in (
                operation.source_block,
                operation.implementation_block,
            ):
                if owned is not None and _contains_expval(owned, seen):
                    return True
        if isinstance(operation, SelectOperation) and any(
            _contains_expval(case_block, seen) for case_block in operation.case_blocks
        ):
            return True
    return False


class EffectValidationPass(Pass):
    """Reject execution-mode conflicts using cached kernel effects."""

    @property
    def name(self) -> str:
        """Return the stable compiler-pass name.

        Returns:
            str: Human-readable pass name.
        """
        return "effect_validation"

    def run(self, block: Block) -> Block:
        """Validate expectation-value compatibility for one entrypoint.

        Args:
            block (Block): Hierarchical entrypoint block.

        Returns:
            Block: The unchanged validated block.

        Raises:
            ValidationError: If expectation-value execution is combined with
                measurement, reset, or feed-forward effects.
        """
        if not block.effects.is_unitary and _contains_expval(block, set()):
            raise ValidationError(
                f"Kernel {block.name!r} has non-unitary kernel effects "
                f"[{format_kernel_effects(block.effects)}] and is sample-only; "
                "it cannot also contain qmc.expval(). Split stochastic "
                "sampling from expectation-value estimation into separate "
                "qkernels."
            )
        return block
