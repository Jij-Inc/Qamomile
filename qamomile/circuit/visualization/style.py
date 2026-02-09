"""Circuit visualization style configuration.

This module provides style configuration for circuit drawings,
inspired by Qiskit's matplotlib drawer styling approach.
"""

from dataclasses import dataclass


@dataclass
class CircuitStyle:
    """Style configuration for circuit visualization.

    Attributes:
        gate_width: Width of gate boxes (in coordinate units).
        gate_height: Height of gate boxes (in coordinate units).
        gate_corner_radius: Corner radius for rounded gate boxes (in coordinate units).
        background_color: Background color for the figure.
        wire_color: Color for qubit wires.
        gate_face_color: Fill color for gate boxes.
        gate_symbol_color: Color for multi-qubit gate symbols (control dots, target X, SWAP X, connection lines).
        gate_symbol_edge_color: Edge/outline color for multi-qubit gate symbols.
        gate_text_color: Text color for gate labels.
        block_face_color: Fill color for block boxes (block mode).
        block_text_color: Text color for block labels (block mode).
        block_border_color: Border color for inline block borders.
        block_box_edge_color: Edge color for block mode boxes.
        font_size: Font size for gate labels.
        subfont_size: Font size for subscripts and smaller text.
        margin: Margins around the circuit (left, right, top, bottom).
    """

    # Size parameters
    gate_width: float = 0.65
    gate_height: float = 0.65
    gate_corner_radius: float = 0.2

    # Color scheme
    background_color: str = "#FFFFFF"
    wire_color: str = "#000000"
    gate_face_color: str = "#F2C94C"  # Chamomile yellow
    gate_symbol_color: str = "#F2C94C"  # Chamomile yellow for multi-qubit gate symbols
    gate_symbol_edge_color: str = "#000000"  # Black edge for multi-qubit gate symbols
    gate_text_color: str = "#000000"  # Black
    connection_line_color: str = "#000000"  # Color for connection lines in multi-qubit gates
    block_face_color: str = "#5B7F61"  # Sage green for block boxes
    block_text_color: str = "#FFFFFF"  # White text for blocks
    block_border_color: str = "#4A6B50"  # Dark sage for inline block borders
    block_box_edge_color: str = "#4A6B50"  # Dark sage for block mode boxes
    measure_face_color: str = "#D5CCC4"  # Light taupe for measurement boxes
    measure_symbol_color: str = "#6B5F55"  # Dark taupe for measurement symbol

    # For-loop box colors (distinct from gates and blocks)
    for_loop_face_color: str = "#E8D5F0"  # Soft lavender
    for_loop_text_color: str = "#000000"  # Black
    for_loop_edge_color: str = "#9B7EAE"  # Medium purple

    # Font sizes
    font_size: int = 13
    subfont_size: int = 8
    param_font_size: int = 9  # Font size for parametric gates

    # Layout
    margin: tuple[float, float, float, float] = (
        0.5,
        0.1,
        0.1,
        0.3,
    )  # left, right, top, bottom (reduced left margin from 2.0 to 0.5)

    # Layout constants (spacing and sizing)
    gate_gap: float = 0.3  # Horizontal gap between gates
    char_width_base: float = (
        0.12  # Base character width estimate for text sizing (titles, labels)
    )
    char_width_gate: float = 0.14  # Character width for gate text estimation
    char_width_block: float = 0.17  # Character width for block label estimation
    char_width_monospace: float = (
        0.24  # Character width for monospace text (loop operations)
    )
    text_padding: float = 0.25  # Padding around text in boxes
    border_padding_base: float = 0.3  # Base padding for block borders
    border_padding_depth_factor: float = 0.1  # Reduction factor per nesting depth
    min_left_margin: float = 0.3  # Minimum left margin for blocks
    label_height: float = 0.35  # Height reserved for block labels

    # Box padding (for For/If/While/CallBlock boxes)
    box_padding_x: float = 0.3  # Horizontal padding inside boxes
    box_padding_y: float = 0.2  # Vertical padding inside boxes

    # Label positioning
    label_vertical_offset: float = 0.05  # Offset from box top for label y position
    label_horizontal_padding: float = 0.1  # Padding for label left/right positioning

    # Wire layout
    initial_wire_position: float = 0.3  # Initial position for qubit wires after labels
    wire_extension: float = 0.3  # Wire extension beyond first/last gate edges

    # Width calculation constants
    operation_width_padding: float = 0.4  # Added to operation box width
    operation_content_padding: float = 0.6  # Added to content width

    # Line height for multi-line content
    line_height: float = 0.4  # Vertical spacing between lines in boxes

    # Text estimation fallbacks (when matplotlib bbox unavailable)
    fallback_char_width: float = 0.15  # Character width for text size estimation
    fallback_text_height: float = 0.15  # Default text height

    # Font scaling
    font_scaling_adjustment: float = 0.85  # Fine-tuning factor for font size scaling

    # Block border layout constants
    nested_margin: float = 0.15  # Margin between nested block borders
    border_extra_margin_right: float = 0.8  # Extra margin on right side of blocks
    border_extra_margin_left: float = 0.3  # Extra margin on left side of blocks

    # Folded block width constants (Issue 1, 5, 7)
    folded_loop_width: float = 1.5  # Fixed width for folded ForOperation blocks
    folded_call_block_width: float = 1.5  # Fixed width for folded CallBlock boxes

    # Gate text padding (Issue 3)
    gate_text_padding: float = 0.10  # Padding around gate text (left and right)

    # Nested block padding (Issue 8)
    nested_padding_decay: float = 0.85  # Padding decay factor per nesting depth
    min_block_padding: float = 0.1  # Minimum block padding

    # Figure size constants
    figure_scale_factor: float = 0.8  # Scaling factor for figure dimensions
    figure_min_width: float = 4.0  # Minimum figure width in inches
    figure_min_height: float = 2.0  # Minimum figure height in inches
    x_left_min_bound: float = -1.0  # Minimum allowed x_left value


# Default style instance
DEFAULT_STYLE = CircuitStyle()
