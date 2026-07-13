"""Pure geometry utilities for circuit visualization.

Standalone functions shared by layout and analysis.
"""

from __future__ import annotations

from .style import CircuitStyle

__all__ = ["compute_border_padding"]


def compute_border_padding(style: CircuitStyle, depth: int) -> float:
    """Compute border padding for a given nesting depth.

    Args:
        style (CircuitStyle): Visual style configuration.
        depth (int): Nesting depth of the block.

    Returns:
        float: Border padding clamped to ``min_block_padding``.
    """
    return max(
        style.min_block_padding,
        style.border_padding_base - depth * style.border_padding_depth_factor,
    )
