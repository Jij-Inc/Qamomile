"""Normalize per-call opaque implementations for the substitution pass."""

from __future__ import annotations

from collections.abc import Mapping

from qamomile.circuit.frontend.qkernel_like import OracleBindings
from qamomile.circuit.ir.block import Block
from qamomile.circuit.transpiler.passes.substitution import (
    SubstitutionConfig,
    SubstitutionPass,
)


def _normalize_oracle_bindings(
    oracle_bindings: OracleBindings | None,
) -> dict[str, Block]:
    """Validate oracle bindings and extract implementation blocks.

    Args:
        oracle_bindings (OracleBindings | None): Definition names mapped to
            qkernel-like objects or blocks. Defaults to ``None``.

    Returns:
        dict[str, Block]: Validated definition-name to block mapping.

    Raises:
        TypeError: If the mapping, a key, or an implementation is invalid.
        ValueError: If a key is empty.
    """
    if oracle_bindings is None:
        return {}
    if not isinstance(oracle_bindings, Mapping):
        raise TypeError("oracle_bindings must be a mapping")

    normalized: dict[str, Block] = {}
    for name, target in oracle_bindings.items():
        if not isinstance(name, str):
            raise TypeError("oracle_bindings keys must be strings")
        if not name:
            raise ValueError("oracle_bindings keys must not be empty")
        replacement = (
            target if isinstance(target, Block) else getattr(target, "block", None)
        )
        if not isinstance(replacement, Block):
            raise TypeError(
                "oracle_bindings values must be QKernel or Block instances; "
                f"got {type(target).__name__} for {name!r}"
            )
        normalized[name] = replacement
    return normalized


def _apply_oracle_bindings(
    block: Block,
    oracle_bindings: OracleBindings | None,
) -> Block:
    """Attach direct implementations to exact opaque definition names.

    The transformation is per-call and does not mutate the source or supplied
    implementation blocks. The same direct body supplies ordinary controlled
    calls through the existing call transform. Generated inverse callables are
    outside this binding contract. Bindings are recursively applied inside
    supplied implementation bodies, and cyclic dependencies are rejected.

    Args:
        block (Block): Traced or hierarchical semantic block.
        oracle_bindings (OracleBindings | None): Definition-name bindings.
            Defaults to ``None``.

    Returns:
        Block: Transformed block, or ``block`` when no bindings are supplied.

    Raises:
        TypeError: If a key or implementation is invalid.
        ValueError: If a key is empty or unused, targets an unsupported
            callable, or the implementations form a cycle.
        ValidationError: If an implementation is incompatible.
    """
    bindings = _normalize_oracle_bindings(oracle_bindings)
    if not bindings:
        return block
    return SubstitutionPass(
        SubstitutionConfig(),
        oracle_bindings=bindings,
    ).run(block)


def _apply_compiler_substitutions(
    block: Block,
    configured: SubstitutionConfig,
    oracle_bindings: OracleBindings | None,
) -> Block:
    """Apply Configure rules and exact opaque bindings in one graph pass.

    A per-call binding wins over a Configure body replacement for the same
    opaque invocation, while any selected strategy still composes with the
    binding. Configure keeps its legacy display-name matching for all other
    calls.

    Args:
        block (Block): Hierarchical semantic block to transform.
        configured (SubstitutionConfig): Persistent Configure rules.
        oracle_bindings (OracleBindings | None): Per-call opaque bindings.
            Defaults to ``None``.

    Returns:
        Block: Block after configured and per-call substitutions.

    Raises:
        TypeError: If a binding key or implementation is invalid.
        ValueError: If a binding is unused, targets an unsupported callable,
            or the implementations form a cycle.
        ValidationError: If an opaque replacement signature differs.
        SignatureCompatibilityError: If a configured replacement signature
            differs.
    """
    bindings = _normalize_oracle_bindings(oracle_bindings)
    if not configured.rules and not bindings:
        return block
    return SubstitutionPass(
        configured,
        oracle_bindings=bindings,
    ).run(block)


__all__ = ["OracleBindings"]
