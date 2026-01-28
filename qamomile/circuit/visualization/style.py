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
        connection_line_color: Color for multi-qubit gate connection lines.
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
    connection_line_color: str = "#F2C94C"  # Chamomile yellow for connection lines
    gate_text_color: str = "#333333"
    block_face_color: str = "#5B7F61"  # Sage green for block boxes
    block_text_color: str = "#FFFFFF"  # White text for blocks
    block_border_color: str = "#4A6B50"  # Dark sage for inline block borders
    block_box_edge_color: str = "#4A6B50"  # Dark sage for block mode boxes

    # Font sizes
    font_size: int = 13
    subfont_size: int = 8
    param_font_size: int = 9  # Font size for parametric gates

    # Layout
    margin: tuple[float, float, float, float] = (
        2.0,
        0.1,
        0.1,
        0.3,
    )  # left, right, top, bottom


# Default style instance
DEFAULT_STYLE = CircuitStyle()
