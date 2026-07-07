"""Substitution pass for replacing callable bodies and strategies.

This pass allows replacing:
1. Inline callable bodies (QKernel subroutines) with alternative implementations
2. InvokeOperation strategies with specified decomposition strategies

Example:
    # Replace a custom oracle with an optimized version
    config = SubstitutionConfig(
        rules=[
            SubstitutionRule(source_name="my_oracle", target=optimized_oracle),
            SubstitutionRule(source_name="qft", strategy="approximate"),
        ]
    )
    pass = SubstitutionPass(config)
    new_block = pass.run(block)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableRef,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.passes import Pass

if TYPE_CHECKING:
    from qamomile.circuit.frontend.qkernel import QKernel


class SignatureCompatibilityError(Exception):
    """Error raised when signatures are incompatible during substitution."""

    pass


def check_signature_compatibility(
    source: Block,
    target: Block,
    strict: bool = True,
) -> tuple[bool, str | None]:
    """Check signature compatibility between two Blocks.

    Args:
        source: The source Block being replaced
        target: The target Block to replace with
        strict: If True, require exact type matches

    Returns:
        Tuple of (is_compatible, error_message).
        If compatible, error_message is None.
    """
    # Check input count
    if len(source.input_values) != len(target.input_values):
        return False, (
            f"Input count mismatch: source has {len(source.input_values)}, "
            f"target has {len(target.input_values)}"
        )

    # Check input types
    for i, (src_val, tgt_val) in enumerate(
        zip(source.input_values, target.input_values)
    ):
        if src_val.type != tgt_val.type:
            return False, (
                f"Input type mismatch at position {i}: "
                f"source has {src_val.type}, target has {tgt_val.type}"
            )

    # Check return count
    if len(source.output_values) != len(target.output_values):
        return False, (
            f"Return count mismatch: source has {len(source.output_values)}, "
            f"target has {len(target.output_values)}"
        )

    # Check return types
    for i, (src_val, tgt_val) in enumerate(
        zip(source.output_values, target.output_values)
    ):
        if src_val.type != tgt_val.type:
            return False, (
                f"Return type mismatch at position {i}: "
                f"source has {src_val.type}, target has {tgt_val.type}"
            )

    return True, None


@dataclass
class SubstitutionRule:
    """A single substitution rule.

    Attributes:
        source_name: Name of the target to replace (block name or gate name)
        target: Replacement Block or QKernel (for inline callables)
        strategy: Strategy name (for callable/composite operations)
        validate_signature: If True, validate signature compatibility when
            replacing blocks. Default is True.
    """

    source_name: str
    target: "Block | QKernel | None" = None
    strategy: str | None = None
    validate_signature: bool = True

    def __post_init__(self) -> None:
        """Validate rule configuration."""
        if self.target is None and self.strategy is None:
            raise ValueError(
                f"SubstitutionRule for '{self.source_name}' must specify "
                "either 'target' or 'strategy'"
            )


@dataclass
class SubstitutionConfig:
    """Configuration for the substitution pass.

    Attributes:
        rules: List of substitution rules to apply
    """

    rules: list[SubstitutionRule] = field(default_factory=list)

    def get_rule_for_name(self, name: str) -> SubstitutionRule | None:
        """Find a rule matching the given name.

        Args:
            name: Name to look up

        Returns:
            Matching SubstitutionRule or None
        """
        for rule in self.rules:
            if rule.source_name == name:
                return rule
        return None


class SubstitutionPass(Pass[Block, Block]):
    """Pass that substitutes call operations and callable strategies.

    This pass traverses the block and applies substitution rules:
    - For inline InvokeOperation: replaces the callable body with the target
    - For boxed InvokeOperation: sets the strategy field

    The pass preserves the block structure and only modifies matching operations.

    Input: Block (any kind)
    Output: Block with substitutions applied (same kind as input)
    """

    def __init__(self, config: SubstitutionConfig) -> None:
        """Initialize the pass with configuration.

        Args:
            config: Substitution configuration with rules
        """
        self._config = config
        self._rules_by_name: dict[str, SubstitutionRule] = {
            rule.source_name: rule for rule in config.rules
        }

    @property
    def name(self) -> str:
        """Return pass name."""
        return "substitution"

    def run(self, input: Block) -> Block:
        """Apply substitutions to the block.

        Args:
            input: Block to transform

        Returns:
            Block with substitutions applied
        """
        if input.kind not in (BlockKind.HIERARCHICAL,):
            raise ValidationError(
                f"SubstitutionPass expects HIERARCHICAL block, got {input.kind}",
            )

        if not self._config.rules:
            # No rules, return input unchanged
            return input

        # Transform operations
        transformed_ops = self._transform_operations(input.operations)

        return Block(
            name=input.name,
            label_args=input.label_args,
            input_values=input.input_values,
            output_values=input.output_values,
            operations=transformed_ops,
            kind=input.kind,
            parameters=input.parameters,
            param_slots=input.param_slots,
        )

    def _transform_operations(
        self,
        operations: list[Operation],
    ) -> list[Operation]:
        """Transform a list of operations, applying substitutions.

        Args:
            operations: Operations to transform

        Returns:
            Transformed operations
        """
        result: list[Operation] = []

        for op in operations:
            transformed = self._transform_operation(op)
            result.append(transformed)

        return result

    def _transform_operation(self, op: Operation) -> Operation:
        """Transform a single operation.

        Args:
            op: Operation to transform

        Returns:
            Transformed operation (may be the same object if no changes)
        """
        if isinstance(op, InvokeOperation):
            return self._transform_invoke(op)

        if isinstance(op, HasNestedOps):
            # Recursively transform all nested operation lists
            new_lists = [
                self._transform_operations(op_list) for op_list in op.nested_op_lists()
            ]
            return op.rebuild_nested(new_lists)

        # Other operations pass through unchanged
        return op

    @staticmethod
    def _replacement_block(rule: SubstitutionRule) -> Block | None:
        """Return the replacement block configured by a rule.

        Args:
            rule (SubstitutionRule): Rule to resolve.

        Returns:
            Block | None: Replacement block, or ``None`` when the rule has no
            block target.
        """
        from qamomile.circuit.frontend.qkernel import QKernel

        if isinstance(rule.target, QKernel):
            return rule.target.block
        if isinstance(rule.target, Block):
            return rule.target
        return None

    def _transform_invoke(
        self,
        op: InvokeOperation,
    ) -> InvokeOperation:
        """Transform an InvokeOperation.

        Args:
            op (InvokeOperation): Operation to transform.

        Returns:
            InvokeOperation: Transformed operation.
        """
        gate_names = [
            str(op.attrs.get("custom_name", "")),
            op.target.name,
            str(op.attrs.get("gate_type", "")).lower(),
        ]
        rule = next(
            (
                self._rules_by_name[name]
                for name in gate_names
                if name in self._rules_by_name
            ),
            None,
        )
        if rule is None:
            return op

        replacement = self._replacement_block(rule)
        if replacement is not None:
            current_body = op.effective_body()
            if isinstance(current_body, Block) and rule.validate_signature:
                is_compatible, error_msg = check_signature_compatibility(
                    current_body,
                    replacement,
                )
                if not is_compatible:
                    raise SignatureCompatibilityError(
                        f"Cannot substitute '{op.target.name}': {error_msg}"
                    )

            new_ref = op.target
            if replacement.name:
                new_ref = CallableRef(
                    namespace=op.target.namespace,
                    name=replacement.name,
                    version=op.target.version,
                )
            definition = op.definition or CallableDef(ref=new_ref)
            new_definition = dataclasses.replace(
                definition,
                ref=new_ref,
                body=replacement,
            )
            return dataclasses.replace(
                op,
                target=new_ref,
                definition=new_definition,
            )

        if rule.strategy is None:
            return op

        attrs = dict(op.attrs)
        attrs["strategy_name"] = rule.strategy
        resource = _resource_for_composite_strategy(
            gate_type_name=str(attrs.get("gate_type", "")),
            num_target_qubits=op.num_target_qubits,
            strategy=rule.strategy,
            fallback=op.effective_resource(strategy=rule.strategy),
        )
        definition = op.definition or CallableDef(ref=op.target)
        new_definition = dataclasses.replace(
            definition,
            resource=resource,
            attrs={**definition.attrs, **attrs},
        )
        return dataclasses.replace(
            op,
            attrs=attrs,
            definition=new_definition,
        )


def _resource_for_composite_strategy(
    *,
    gate_type_name: str,
    num_target_qubits: int,
    strategy: str,
    fallback: object,
) -> object:
    """Return resource metadata for a stdlib composite strategy.

    Args:
        gate_type_name (str): Composite gate type name such as ``"QFT"``.
        num_target_qubits (int): Number of target qubits.
        strategy (str): Strategy name to estimate.
        fallback (object): Existing resource object when no specialized
            resource model is available.

    Returns:
        object: Strategy-specific resource metadata when known, otherwise
        ``fallback``.
    """
    if gate_type_name == "QFT":
        from qamomile.circuit.stdlib.qft import QFT

        return QFT(num_target_qubits).get_resources_for_strategy(strategy) or fallback
    if gate_type_name == "IQFT":
        from qamomile.circuit.stdlib.qft import IQFT

        return IQFT(num_target_qubits).get_resources_for_strategy(strategy) or fallback
    return fallback


def create_substitution_pass(
    *,
    block_replacements: dict[str, "Block | QKernel"] | None = None,
    strategy_overrides: dict[str, str] | None = None,
    validate_signatures: bool = True,
) -> SubstitutionPass:
    """Convenience factory for creating a SubstitutionPass.

    Args:
        block_replacements: Map of block name to replacement
        strategy_overrides: Map of gate name to strategy name
        validate_signatures: If True, validate signature compatibility
            when replacing blocks. Default is True.

    Returns:
        Configured SubstitutionPass

    Example:
        pass = create_substitution_pass(
            block_replacements={"my_oracle": optimized_oracle},
            strategy_overrides={"qft": "approximate", "iqft": "approximate"},
        )
    """
    rules: list[SubstitutionRule] = []

    if block_replacements:
        for name, target in block_replacements.items():
            rules.append(
                SubstitutionRule(
                    source_name=name,
                    target=target,
                    validate_signature=validate_signatures,
                )
            )

    if strategy_overrides:
        for name, strategy in strategy_overrides.items():
            rules.append(SubstitutionRule(source_name=name, strategy=strategy))

    return SubstitutionPass(SubstitutionConfig(rules=rules))
