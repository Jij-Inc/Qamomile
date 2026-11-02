"""Authoritative native lowering from typed HIR to persistent QCore."""

from __future__ import annotations

from ..ir import Module
from .hir import HIRModule


def lower_hir(module: HIRModule) -> Module:
    """Lower a typed HIR module through the Rust ownership-SSA compiler.

    Args:
        module: Fully typed ownership HIR produced by the source frontend.

    Returns:
        Immutable QCore module independently verified by the Rust compiler.

    Raises:
        CapabilityError: If the required native compiler is unavailable or
            does not implement a valid HIR construct.
        VerificationError: If Rust rejects the HIR or its produced QCore.
    """

    from .hir_wire import lower_native_hir

    return lower_native_hir(module)
