"""Lazy block construction helpers for ``QKernel`` objects."""

from __future__ import annotations

from typing import Any, cast

from qamomile.circuit.frontend.func_to_block import func_to_block
from qamomile.circuit.frontend.qkernel_definition import (
    refresh_qkernel_function_namespace,
)
from qamomile.circuit.frontend.qkernel_self_call import finalize_pending_self_calls
from qamomile.circuit.ir.block import Block
from qamomile.circuit.transpiler.errors import FrontendTransformError


def get_or_build_block(kernel: Any) -> Block:
    """Return a qkernel's cached hierarchical block, building it if needed.

    Args:
        kernel (Any): QKernel-like object with ``func``, ``name``, ``_block``,
            ``_block_building``, and ``_pending_self_calls`` attributes.

    Returns:
        Block: Cached or freshly traced hierarchical block.

    Raises:
        FrontendTransformError: If a self-recursive qkernel accesses ``.block``
            directly while its own block is being built.
    """
    with kernel._block_lock:
        if kernel._block is not None:
            return kernel._block

        if kernel._block_building:
            raise FrontendTransformError(
                f"Self-recursive @qkernel '{kernel.name}' accessed "
                f".block during its own build.  Self-calls in a "
                f"@qkernel body must use the plain call syntax "
                f"(`{kernel.name}(args)`); direct `.block` access "
                f"from inside the body is not supported."
            )

        kernel._block_building = True
        try:
            refresh_qkernel_function_namespace(kernel)
            # Use the AST-transformed function so qmc.range() and control flow are
            # represented through frontend builder operations.
            kernel._block = func_to_block(kernel.func)
            finalize_pending_self_calls(kernel)
        finally:
            kernel._block_building = False
        return cast(Block, kernel._block)
