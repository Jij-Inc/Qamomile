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


def compute_nested_block_box_bounds(
    style: CircuitStyle,
    name: str,
    start_x: float,
    end_x: float,
    depth: int,
    max_gate_width: float,
    power: int = 1,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute (inner_bounds, outer_bounds) for an inlined block border.

    When power <= 1, inner_bounds == outer_bounds (no wrapper).
    When power > 1, outer_bounds expand by power_wrapper_margin
    and account for the "pow=N" label width.

    Args:
        style: Visual style configuration.
        name: Block label text.
        start_x: X position of the first gate in the block.
        end_x: X position of the last gate in the block.
        depth: Nesting depth of the block.
        max_gate_width: Width of the widest gate in the block.
        power: Power annotation value (displayed as "pow=N" when > 1).

    Returns:
        Tuple of ((inner_left, inner_right), (outer_left, outer_right)).
    """
    padding = compute_border_padding(style, depth)
    gtp = style.gate_text_padding

    # Inner box: uses only block name
    inner_left = start_x - max_gate_width / 2 - padding - gtp
    gate_box_right = end_x + max_gate_width / 2 + padding + gtp
    title_text_width = len(name) * style.char_width_base
    label_right = (
        inner_left
        + style.label_horizontal_padding
        + title_text_width
        + style.label_horizontal_padding
    )
    inner_right = max(gate_box_right, label_right)

    if power <= 1:
        return (inner_left, inner_right), (inner_left, inner_right)

    # Outer box: expand by wrapper margin
    margin = style.power_wrapper_margin
    outer_left = inner_left - margin
    outer_right_from_inner = inner_right + margin

    # Ensure outer box is wide enough for "pow=N" label
    pow_label = f"pow={power}"
    pow_text_width = len(pow_label) * style.char_width_base
    pow_label_right = (
        outer_left
        + style.label_horizontal_padding
        + pow_text_width
        + style.label_horizontal_padding
    )
    outer_right = max(outer_right_from_inner, pow_label_right)

    return (inner_left, inner_right), (outer_left, outer_right)


def compute_block_box_bounds(
    style: CircuitStyle,
    name: str,
    start_x: float,
    end_x: float,
    depth: int,
    max_gate_width: float,
    power: int = 1,
) -> tuple[float, float]:
    """Compute outermost (box_left, box_right) for an inlined block border.

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
    _, outer = compute_nested_block_box_bounds(
        style, name, start_x, end_x, depth, max_gate_width, power
    )
    return outer
