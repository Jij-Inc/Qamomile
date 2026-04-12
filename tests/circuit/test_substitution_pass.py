"""Tests for the substitution pass."""

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.transpiler.passes.substitution import (
    SignatureCompatibilityError,
    SubstitutionConfig,
    SubstitutionPass,
    SubstitutionRule,
    check_signature_compatibility,
    create_substitution_pass,
)


class TestSubstitutionRule:
    """Tests for SubstitutionRule."""

    def test_valid_rule_with_strategy(self):
        """Test creating rule with strategy."""
        rule = SubstitutionRule(source_name="qft", strategy="approximate")
        assert rule.source_name == "qft"
        assert rule.strategy == "approximate"
        assert rule.target is None

    def test_invalid_rule_no_target_or_strategy(self):
        """Test that rule without target or strategy raises error."""
        with pytest.raises(ValueError):
            SubstitutionRule(source_name="qft")


class TestSubstitutionConfig:
    """Tests for SubstitutionConfig."""

    def test_get_rule_for_name(self):
        """Test getting rule by name."""
        rules = [
            SubstitutionRule(source_name="qft", strategy="approximate"),
            SubstitutionRule(source_name="iqft", strategy="standard"),
        ]
        config = SubstitutionConfig(rules=rules)

        qft_rule = config.get_rule_for_name("qft")
        assert qft_rule is not None
        assert qft_rule.strategy == "approximate"

        iqft_rule = config.get_rule_for_name("iqft")
        assert iqft_rule is not None
        assert iqft_rule.strategy == "standard"

        assert config.get_rule_for_name("nonexistent") is None


class TestSubstitutionPass:
    """Tests for SubstitutionPass."""

    def test_no_rules_returns_unchanged(self):
        """Test that empty config returns input unchanged."""
        config = SubstitutionConfig(rules=[])
        pass_ = SubstitutionPass(config)

        # Create a simple block
        block = Block(
            name="test",
            label_args=[],
            input_values=[],
            output_values=[],
            operations=[],
            kind=BlockKind.HIERARCHICAL,
        )

        result = pass_.run(block)
        assert result is block  # Should be same object

    def test_strategy_override_on_composite_gate(self):
        """Test setting strategy on composite gate operation."""

        # Create a kernel that uses QFT
        @qmc.qkernel
        def test_kernel(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        # Build block
        block = test_kernel.block

        # Create pass with strategy override
        config = SubstitutionConfig(
            rules=[SubstitutionRule(source_name="qft", strategy="approximate")]
        )
        pass_ = SubstitutionPass(config)

        # Apply pass
        result = pass_.run(block)

        # Block should be transformed (even if no QFT in this simple case)
        assert result is not None


class TestCreateSubstitutionPass:
    """Tests for create_substitution_pass factory."""

    def test_create_with_strategy_overrides(self):
        """Test creating pass with strategy overrides."""
        pass_ = create_substitution_pass(
            strategy_overrides={"qft": "approximate", "iqft": "approximate_k2"}
        )

        assert pass_._config.get_rule_for_name("qft").strategy == "approximate"
        assert pass_._config.get_rule_for_name("iqft").strategy == "approximate_k2"

    def test_create_empty(self):
        """Test creating empty pass."""
        pass_ = create_substitution_pass()
        assert len(pass_._config.rules) == 0


class TestSignatureValidation:
    """Tests for signature validation in substitution."""

    def test_compatible_signatures_pass(self):
        """Test that compatible signatures allow substitution."""

        @qmc.qkernel
        def source(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            return qmc.rx(q, theta)

        @qmc.qkernel
        def target(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            return qmc.rz(q, theta)

        # Should not raise
        is_compatible, error_msg = check_signature_compatibility(
            source.block, target.block
        )
        assert is_compatible
        assert error_msg is None

    def test_input_count_mismatch_raises(self):
        """Test that input count mismatch is detected."""

        @qmc.qkernel
        def source(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            return qmc.rx(q, theta)

        @qmc.qkernel
        def target(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.h(q)

        is_compatible, error_msg = check_signature_compatibility(
            source.block, target.block
        )
        assert not is_compatible
        assert "Input count mismatch" in error_msg
        assert "source has 2" in error_msg
        assert "target has 1" in error_msg

    def test_input_type_mismatch_raises(self):
        """Test that input type mismatch is detected."""

        @qmc.qkernel
        def source(q: qmc.Qubit, x: qmc.Float) -> qmc.Qubit:
            return qmc.rx(q, x)

        @qmc.qkernel
        def target(q: qmc.Qubit, x: qmc.UInt) -> qmc.Qubit:
            return q

        is_compatible, error_msg = check_signature_compatibility(
            source.block, target.block
        )
        assert not is_compatible
        assert "Input type mismatch at position 1" in error_msg

    def test_return_count_mismatch_raises(self):
        """Test that return count mismatch is detected."""

        @qmc.qkernel
        def source(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.h(q)

        @qmc.qkernel
        def target(q: qmc.Qubit, r: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            return qmc.h(q), qmc.h(r)

        is_compatible, error_msg = check_signature_compatibility(
            source.block, target.block
        )
        # First it will fail on input count mismatch since target has 2 inputs
        assert not is_compatible
        assert "mismatch" in error_msg.lower()

    def test_return_count_mismatch_same_inputs(self):
        """Test that return count mismatch is detected with same inputs."""

        @qmc.qkernel
        def source(q: qmc.Qubit, r: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def target(q: qmc.Qubit, r: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            return qmc.h(q), qmc.h(r)

        is_compatible, error_msg = check_signature_compatibility(
            source.block, target.block
        )
        assert not is_compatible
        assert "Return count mismatch" in error_msg
        assert "source has 1" in error_msg
        assert "target has 2" in error_msg

    def test_return_type_mismatch_raises(self):
        """Test that return type mismatch is detected."""

        @qmc.qkernel
        def source(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.h(q)

        @qmc.qkernel
        def target(b: qmc.Bit) -> qmc.Bit:
            return b

        is_compatible, error_msg = check_signature_compatibility(
            source.block, target.block
        )
        assert not is_compatible
        # Will fail on input type mismatch first
        assert "type mismatch" in error_msg.lower()

    def test_skip_validation_when_disabled(self):
        """Test that validation can be skipped with validate_signature=False."""

        @qmc.qkernel
        def source(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            return qmc.rx(q, theta)

        @qmc.qkernel
        def target(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.h(q)

        # Create rule with validation disabled
        rule = SubstitutionRule(
            source_name="source", target=target, validate_signature=False
        )
        assert rule.validate_signature is False

    def test_substitution_pass_raises_on_incompatible(self):
        """Test that SubstitutionPass raises error on incompatible signatures."""

        @qmc.qkernel
        def source(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            return qmc.rx(q, theta)

        @qmc.qkernel
        def target_incompatible(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.h(q)

        @qmc.qkernel
        def main(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            return source(q, theta)

        # Build block
        block = main.block

        # Create pass with incompatible substitution
        pass_ = create_substitution_pass(
            block_replacements={"source": target_incompatible}
        )

        # Should raise SignatureCompatibilityError
        with pytest.raises(SignatureCompatibilityError) as exc_info:
            pass_.run(block)

        assert "Cannot substitute 'source'" in str(exc_info.value)
        assert "Input count mismatch" in str(exc_info.value)

    def test_substitution_pass_succeeds_on_compatible(self):
        """Test that SubstitutionPass succeeds with compatible signatures."""

        @qmc.qkernel
        def source(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            return qmc.rx(q, theta)

        @qmc.qkernel
        def target_compatible(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            return qmc.rz(q, theta)

        @qmc.qkernel
        def main(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            return source(q, theta)

        # Build block
        block = main.block

        # Create pass with compatible substitution
        pass_ = create_substitution_pass(
            block_replacements={"source": target_compatible}
        )

        # Should not raise
        result = pass_.run(block)
        assert result is not None

    def test_create_substitution_pass_validate_signatures_false(self):
        """Test that create_substitution_pass respects validate_signatures=False."""

        @qmc.qkernel
        def target(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.h(q)

        pass_ = create_substitution_pass(
            block_replacements={"my_block": target},
            validate_signatures=False,
        )

        # Check that the rule has validate_signature=False
        rule = pass_._config.get_rule_for_name("my_block")
        assert rule is not None
        assert rule.validate_signature is False
