"""Pure geometry utilities for circuit visualization.

Standalone functions for computing border padding and block box bounds,
shared by Layout and Renderer without requiring Analyzer.
"""

from __future__ import annotations

from .style import CircuitStyle


def compute_border_padding(style: CircuitStyle, depth: int) -> float:
    """Compute border padding for a given nesting depth.

    Args:
        style: Visual style configuration.
        depth: Nesting depth of the block.

    Returns:
        Border padding value, clamped to min_block_padding.
    """
    return max(
        style.min_block_padding,
        style.border_padding_base - depth * style.border_padding_depth_factor,
    )


def compute_block_box_bounds(
    style: CircuitStyle,
    name: str,
    start_x: float,
    end_x: float,
    depth: int,
    max_gate_width: float,
    power: int = 1,
) -> tuple[float, float]:
    """Compute (box_left, box_right) for an inlined block border.

    Label expansion is right-only: box_left is always gate-based,
    box_right expands rightward if the label text needs more space.

    Args:
        style: Visual style configuration.
        name: Block label text.
        start_x: X position of the first gate in the block.
        end_x: X position of the last gate in the block.
        depth: Nesting depth of the block.
        max_gate_width: Width of the widest gate in the block.
        power: Power annotation value (displayed as "pow=N" when > 1).

    Returns:
        Tuple of (box_left, box_right).
    """
    padding = compute_border_padding(style, depth)
    gtp = style.gate_text_padding
    box_left = start_x - max_gate_width / 2 - padding - gtp
    gate_box_right = end_x + max_gate_width / 2 + padding + gtp
    display_name = f"{name}  pow={power}" if power > 1 else name
    title_text_width = len(display_name) * style.char_width_base
    label_right = (
        box_left
        + style.label_horizontal_padding
        + title_text_width
        + style.label_horizontal_padding
    )
    box_right = max(gate_box_right, label_right)
    return box_left, box_right
