"""Substitution pass for replacing callable bodies and strategies.

This pass allows replacing:
1. Inline callable bodies (QKernel subroutines) with alternative implementations
2. InvokeOperation strategies with specified decomposition strategies

Example:
    # Replace a custom oracle with an optimized version
    config = SubstitutionConfig(
        rules=[
            SubstitutionRule(source_name="my_oracle", target=optimized_oracle),
            SubstitutionRule(source_name="qft", strategy="approximate_k2"),
        ]
    )
    substitution_pass = SubstitutionPass(config)
    new_block = substitution_pass.run(block)
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableRef,
    CallTransform,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.value import ArrayValue, ValueLike
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.passes import Pass

if TYPE_CHECKING:
    from qamomile.circuit.frontend.qkernel import QKernel


class SignatureCompatibilityError(Exception):
    """Report an incompatible callable substitution signature.

    Example:
        Correct usage leaves compatible blocks unchanged::

            source = Block()
            target = Block()
            compatible, _ = check_signature_compatibility(source, target)
            assert compatible

        An incompatible replacement raises this error::

            raise SignatureCompatibilityError(
                "Cannot substitute 'oracle': input count mismatch"
            )
    """


def _shape_compatibility_error(
    source: ValueLike,
    target: ValueLike,
    *,
    position: int,
    role: str,
) -> str | None:
    """Return an array-shape incompatibility diagnostic when one exists.

    A symbolic target extent can specialize a fixed source extent. A fixed
    target cannot implement a symbolic source extent because it would only be
    valid for one of the source callable's accepted shapes.

    Args:
        source (ValueLike): Value in the source callable contract.
        target (ValueLike): Value in the replacement callable contract.
        position (int): Positional index within inputs or results.
        role (str): Human-readable contract role.

    Returns:
        str | None: Diagnostic text, or ``None`` when shapes are compatible.
    """
    source_is_array = isinstance(source, ArrayValue)
    target_is_array = isinstance(target, ArrayValue)
    if source_is_array != target_is_array:
        return (
            f"{role} shape mismatch at position {position}: "
            f"source is {'an array' if source_is_array else 'a scalar'}, "
            f"target is {'an array' if target_is_array else 'a scalar'}"
        )
    if not source_is_array:
        return None

    assert isinstance(source, ArrayValue)
    assert isinstance(target, ArrayValue)
    if len(source.shape) != len(target.shape):
        return (
            f"{role} rank mismatch at position {position}: source has rank "
            f"{len(source.shape)}, target has rank {len(target.shape)}"
        )
    for dimension, (source_extent, target_extent) in enumerate(
        zip(source.shape, target.shape, strict=True)
    ):
        source_is_fixed = source_extent.is_constant()
        target_is_fixed = target_extent.is_constant()
        if not source_is_fixed and target_is_fixed:
            return (
                f"{role} shape mismatch at position {position}, dimension "
                f"{dimension}: source extent is symbolic but target extent is fixed"
            )
        if source_is_fixed and target_is_fixed:
            source_value = source_extent.get_const()
            target_value = target_extent.get_const()
            if source_value != target_value:
                return (
                    f"{role} extent mismatch at position {position}, dimension "
                    f"{dimension}: source has {source_value}, target has "
                    f"{target_value}"
                )
    return None


def check_signature_compatibility(
    source: Block,
    target: Block,
    strict: bool = True,
) -> tuple[bool, str | None]:
    """Check signature compatibility between two Blocks.

    Args:
        source (Block): Source callable contract.
        target (Block): Replacement callable contract.
        strict (bool): Whether types must match exactly. The current IR only
            supports exact type compatibility. Defaults to ``True``.

    Returns:
        tuple[bool, str | None]: Compatibility flag and optional diagnostic.
    """
    del strict
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

    Args:
        source_name (str): Target block or callable name.
        target (Block | QKernel | None): Replacement body. Defaults to
            ``None``.
        strategy (str | None): Callable lowering strategy. Defaults to
            ``None``.
        validate_signature (bool): Whether to validate replacement signatures.
            Defaults to ``True``.
    """

    source_name: str
    target: "Block | QKernel | None" = None
    strategy: str | None = None
    validate_signature: bool = True

    def __post_init__(self) -> None:
        """Validate the rule configuration.

        Raises:
            ValueError: If neither a target nor a strategy is supplied.
        """
        if self.target is None and self.strategy is None:
            raise ValueError(
                f"SubstitutionRule for '{self.source_name}' must specify "
                "either 'target' or 'strategy'"
            )


@dataclass
class SubstitutionConfig:
    """Configuration for the substitution pass.

    Args:
        rules (list[SubstitutionRule]): Rules to apply. Defaults to an empty
            list.
    """

    rules: list[SubstitutionRule] = field(default_factory=list)

    def get_rule_for_name(self, name: str) -> SubstitutionRule | None:
        """Find a rule matching the given name.

        Args:
            name (str): Name to look up.

        Returns:
            SubstitutionRule | None: Matching rule, or ``None``.
        """
        for rule in self.rules:
            if rule.source_name == name:
                return rule
        return None


class SubstitutionPass(Pass[Block, Block]):
    """Apply Configure rules and exact per-call opaque replacements.

    Configure keeps its display-name-first behavior. ``oracle_bindings`` is a
    stricter, internal extension of the same pass: it matches the callable
    definition name, accepts only resource-only opaque definitions, preserves
    their identity, and takes precedence over a Configure body replacement.

    Args:
        config (SubstitutionConfig): Persistent Configure rules.
        oracle_bindings (Mapping[str, Block] | None): Validated per-call
            opaque replacements. Defaults to ``None``.
    """

    def __init__(
        self,
        config: SubstitutionConfig,
        *,
        oracle_bindings: Mapping[str, Block] | None = None,
    ) -> None:
        """Initialize the pass.

        Args:
            config (SubstitutionConfig): Persistent Configure rules.
            oracle_bindings (Mapping[str, Block] | None): Validated opaque
                definition-name replacements. Defaults to ``None``.
        """
        self._config = config
        self._rules_by_name = {rule.source_name: rule for rule in config.rules}
        self._oracle_bindings = dict(oracle_bindings or {})
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset graph traversal state before one pass invocation."""
        self._matched_bindings: set[str] = set()
        self._block_cache: dict[tuple[int, bool], Block] = {}
        self._definition_cache: dict[int, CallableDef] = {}
        self._active_bindings: list[str] = []
        self._active_replacement_blocks: dict[int, str] = {}
        self._active_block_keys: list[tuple[int, bool]] = []

    @property
    def name(self) -> str:
        """Return the pass name.

        Returns:
            str: Stable pass name.
        """
        return "substitution"

    def run(self, input: Block) -> Block:
        """Apply configured substitutions and opaque bindings.

        Args:
            input (Block): Traced or hierarchical source block.

        Returns:
            Block: Non-destructively transformed block graph.

        Raises:
            ValidationError: If the block is at an unsupported pipeline stage.
            ValueError: If an opaque binding is unused, targets an unsupported
                callable, or forms a cyclic implementation dependency.
            SignatureCompatibilityError: If a configured replacement body is
                incompatible with its source callable.
        """
        allowed_kinds = {BlockKind.HIERARCHICAL}
        if self._oracle_bindings:
            allowed_kinds.add(BlockKind.TRACED)
        if input.kind not in allowed_kinds:
            expected = "HIERARCHICAL"
            if self._oracle_bindings:
                expected += " or TRACED"
            raise ValidationError(
                f"SubstitutionPass expects a {expected} block, got {input.kind}",
            )
        if not self._config.rules and not self._oracle_bindings:
            return input

        self._reset_state()
        transformed = self._transform_block(input, apply_rules=True)
        unused = self._oracle_bindings.keys() - self._matched_bindings
        if unused:
            names = ", ".join(repr(name) for name in sorted(unused))
            raise ValueError(
                "oracle_bindings keys must match direct or controlled "
                "resource-only opaque definition names; no match found for "
                f"{names}"
            )
        return transformed

    def _transform_block(self, block: Block, *, apply_rules: bool) -> Block:
        """Clone and transform one source block.

        Args:
            block (Block): Source block.
            apply_rules (bool): Whether Configure rules apply in this region.

        Returns:
            Block: Cached transformed block.
        """
        key = (id(block), apply_rules)
        cached = self._block_cache.get(key)
        if cached is not None:
            return cached
        transformed = dataclasses.replace(block, operations=[])
        self._block_cache[key] = transformed
        self._active_block_keys.append(key)
        try:
            transformed.operations = self._transform_operations(
                block.operations,
                apply_rules=apply_rules,
            )
        finally:
            self._active_block_keys.pop()
        return transformed

    def _transform_operations(
        self,
        operations: list[Operation],
        *,
        apply_rules: bool,
    ) -> list[Operation]:
        """Transform one operation sequence.

        Args:
            operations (list[Operation]): Operations to transform.
            apply_rules (bool): Whether Configure rules apply in this region.

        Returns:
            list[Operation]: Transformed operations.
        """
        return [
            self._transform_operation(operation, apply_rules=apply_rules)
            for operation in operations
        ]

    def _transform_operation(
        self,
        op: Operation,
        *,
        apply_rules: bool,
    ) -> Operation:
        """Transform one operation and its source-owned regions.

        Args:
            op (Operation): Operation to transform.
            apply_rules (bool): Whether Configure rules apply in this region.

        Returns:
            Operation: Transformed operation.
        """
        if isinstance(op, InvokeOperation):
            return self._transform_invoke(op, apply_rules=apply_rules)
        if self._oracle_bindings and isinstance(op, ControlledUOperation):
            return dataclasses.replace(
                op,
                block=(
                    self._transform_block(op.block, apply_rules=False)
                    if op.block is not None
                    else None
                ),
            )
        if isinstance(op, SelectOperation):
            return dataclasses.replace(
                op,
                case_blocks=[
                    self._transform_block(case_block, apply_rules=apply_rules)
                    for case_block in op.case_blocks
                ],
            )
        if isinstance(op, HasNestedOps):
            return op.rebuild_nested(
                [
                    self._transform_operations(nested, apply_rules=apply_rules)
                    for nested in op.nested_op_lists()
                ]
            )
        return op

    def _transform_definition(self, definition: CallableDef) -> CallableDef:
        """Clone a source callable definition and bind its reachable bodies.

        Configure rules do not enter existing definition bodies, preserving
        the behavior of the pass before per-call bindings were introduced.

        Args:
            definition (CallableDef): Definition to transform.

        Returns:
            CallableDef: Cached transformed definition.
        """
        key = id(definition)
        cached = self._definition_cache.get(key)
        if cached is not None:
            return cached
        transformed = dataclasses.replace(
            definition,
            body=None,
            implementations=[],
            attrs=dict(definition.attrs),
        )
        self._definition_cache[key] = transformed
        transformed.body = (
            self._transform_block(definition.body, apply_rules=False)
            if definition.body is not None
            else None
        )
        transformed.implementations = [
            dataclasses.replace(
                implementation,
                body=(
                    self._transform_block(implementation.body, apply_rules=False)
                    if implementation.body is not None
                    else None
                ),
            )
            for implementation in definition.implementations
        ]
        return transformed

    @staticmethod
    def _replacement_block(rule: SubstitutionRule) -> Block | None:
        """Return the replacement block configured by a rule.

        Args:
            rule (SubstitutionRule): Rule to resolve.

        Returns:
            Block | None: Replacement block, or ``None`` when absent.
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
        *,
        apply_rules: bool,
    ) -> InvokeOperation:
        """Transform one invocation.

        Args:
            op (InvokeOperation): Invocation to transform.
            apply_rules (bool): Whether Configure rules apply here.

        Returns:
            InvokeOperation: Transformed invocation.

        Raises:
            ValueError: If an exact binding targets an unsupported callable or
                forms a cyclic implementation dependency.
            SignatureCompatibilityError: If a replacement signature differs.
        """
        rule = self._rule_for(op) if apply_rules else None
        if op.target.name in self._oracle_bindings:
            replacement = self._oracle_bindings[op.target.name]
            return self._bind_opaque(op, replacement, rule)

        configured_replacement = self._replacement_block(rule) if rule else None
        if configured_replacement is not None:
            assert rule is not None
            replacement = configured_replacement
            if self._oracle_bindings:
                replacement = self._transform_block(replacement, apply_rules=False)
            current_body = op.effective_body()
            if isinstance(current_body, Block) and rule.validate_signature:
                compatible, error = check_signature_compatibility(
                    current_body,
                    replacement,
                )
                if not compatible:
                    raise SignatureCompatibilityError(
                        f"Cannot substitute '{op.target.name}': {error}"
                    )
            new_ref = op.target
            if replacement.name:
                new_ref = CallableRef(
                    namespace=op.target.namespace,
                    name=replacement.name,
                    version=op.target.version,
                )
            definition = op.definition or CallableDef(ref=new_ref)
            return dataclasses.replace(
                op,
                target=new_ref,
                definition=dataclasses.replace(
                    definition,
                    ref=new_ref,
                    body=replacement,
                ),
            )

        definition = op.definition or CallableDef(ref=op.target)
        transformed_definition = (
            self._transform_definition(definition)
            if self._oracle_bindings
            else definition
        )
        transformed = dataclasses.replace(op, definition=transformed_definition)
        if rule is None or rule.strategy is None:
            return transformed

        attrs = {**transformed.attrs, "strategy_name": rule.strategy}
        return dataclasses.replace(
            transformed,
            attrs=attrs,
            definition=dataclasses.replace(
                transformed_definition,
                attrs={**transformed_definition.attrs, **attrs},
            ),
        )

    def _bind_opaque(
        self,
        op: InvokeOperation,
        replacement: Block,
        rule: SubstitutionRule | None,
    ) -> InvokeOperation:
        """Attach a direct body to one resource-only opaque invocation.

        Args:
            op (InvokeOperation): Matching opaque invocation.
            replacement (Block): Direct implementation body.
            rule (SubstitutionRule | None): Selected Configure rule, if any.

        Returns:
            InvokeOperation: Invocation with the supplied body.

        Raises:
            ValueError: If the invocation is not a direct or controlled
                resource-only opaque callable.
            ValidationError: If the body signature is incompatible.
        """
        definition = op.definition
        definition_kind = definition.attrs.get("kind") if definition else None
        kind = op.attrs.get("kind", definition_kind)
        resource_only = (
            definition is not None
            and kind == "oracle"
            and definition.body is None
            and definition.body_ref is None
            and not definition.implementations
        )
        if not resource_only or op.transform not in {
            CallTransform.DIRECT,
            CallTransform.CONTROLLED,
        }:
            raise ValueError(
                f"oracle_bindings[{op.target.name!r}] can only replace a "
                "direct or controlled resource-only opaque definition"
            )

        assert definition is not None
        self._validate_oracle_signature(op, replacement)
        replacement = self._transform_binding_body(op.target.name, replacement)
        self._matched_bindings.add(op.target.name)
        attrs = dict(op.attrs)
        if rule is not None and rule.strategy is not None:
            attrs["strategy_name"] = rule.strategy
        bound_definition = dataclasses.replace(
            definition,
            body=replacement,
            body_ref=None,
            opaque_cost=None,
            attrs={**definition.attrs, **attrs},
        )
        return dataclasses.replace(op, attrs=attrs, definition=bound_definition)

    def _transform_binding_body(self, name: str, replacement: Block) -> Block:
        """Bind opaque calls inside one supplied implementation body.

        Args:
            name (str): Binding key whose implementation is being expanded.
            replacement (Block): Supplied implementation body.

        Returns:
            Block: Non-destructively transformed implementation body.

        Raises:
            ValueError: If implementation bindings form a direct or indirect
                dependency cycle.
        """
        if name in self._active_bindings:
            start = self._active_bindings.index(name)
            cycle = [*self._active_bindings[start:], name]
            raise ValueError("Cyclic oracle bindings detected: " + " -> ".join(cycle))

        active_owner = self._active_replacement_blocks.get(id(replacement))
        if active_owner is not None:
            start = self._active_bindings.index(active_owner)
            cycle = [*self._active_bindings[start:], name, active_owner]
            raise ValueError("Cyclic oracle bindings detected: " + " -> ".join(cycle))
        if (id(replacement), False) in self._active_block_keys:
            raise ValueError(
                f"Cyclic oracle binding detected for {name!r}: its implementation "
                "re-enters an active source or configured replacement block"
            )

        self._active_bindings.append(name)
        self._active_replacement_blocks[id(replacement)] = name
        try:
            return self._transform_block(replacement, apply_rules=False)
        finally:
            self._active_replacement_blocks.pop(id(replacement))
            self._active_bindings.pop()

    def _rule_for(self, op: InvokeOperation) -> SubstitutionRule | None:
        """Return the Configure rule selected for an invocation.

        Args:
            op (InvokeOperation): Invocation to match.

        Returns:
            SubstitutionRule | None: Display-name-first matching rule.
        """
        names = (str(op.attrs.get("custom_name", "")), op.target.name)
        return next(
            (
                self._rules_by_name[name]
                for name in names
                if name in self._rules_by_name
            ),
            None,
        )

    @staticmethod
    def _validate_oracle_signature(op: InvokeOperation, replacement: Block) -> None:
        """Validate a direct implementation against an opaque call site.

        Explicit controls are excluded because the existing controlled-call
        path synthesizes them around the supplied direct implementation.

        Args:
            op (InvokeOperation): Opaque invocation being bound.
            replacement (Block): Direct implementation body.

        Raises:
            ValidationError: If arity, types, or array shapes are incompatible.
        """
        control_count = op.num_control_qubits
        source = Block(
            name=op.target.name,
            input_values=list(op.operands[control_count:]),
            output_values=list(op.results[control_count:]),
        )
        compatible, error = check_signature_compatibility(source, replacement)
        if not compatible:
            raise ValidationError(
                f"Cannot bind opaque oracle {op.target.name!r}: {error}"
            )
        for role, source_values, target_values in (
            ("Input", source.input_values, replacement.input_values),
            ("Return", source.output_values, replacement.output_values),
        ):
            for position, (source_value, target_value) in enumerate(
                zip(source_values, target_values, strict=True)
            ):
                error = _shape_compatibility_error(
                    source_value,
                    target_value,
                    position=position,
                    role=role,
                )
                if error is not None:
                    raise ValidationError(
                        f"Cannot bind opaque oracle {op.target.name!r}: {error}"
                    )


def create_substitution_pass(
    *,
    block_replacements: dict[str, "Block | QKernel"] | None = None,
    strategy_overrides: dict[str, str] | None = None,
    validate_signatures: bool = True,
) -> SubstitutionPass:
    """Create a substitution pass from replacement mappings.

    Args:
        block_replacements (dict[str, Block | QKernel] | None): Block names
            mapped to replacement bodies. Defaults to ``None``.
        strategy_overrides (dict[str, str] | None): Callable names mapped to
            strategy names. Defaults to ``None``.
        validate_signatures (bool): Whether to validate signature compatibility
            when replacing blocks. Default is True.

    Returns:
        SubstitutionPass: Configured substitution pass.

    Example:
        substitution_pass = create_substitution_pass(
            block_replacements={"my_oracle": optimized_oracle},
            strategy_overrides={"qft": "approximate_k2", "iqft": "approximate_k2"},
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
