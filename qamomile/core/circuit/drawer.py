from typing import Dict, Union, Optional, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, Arc
from contextlib import contextmanager
from .circuit import (QuantumCircuit, Gate, SingleQubitGate, ParametricSingleQubitGate,
                      TwoQubitGate, ParametricTwoQubitGate, ThreeQubitGate, MeasurementGate,
                      ParametricSingleQubitGateType, ParametricTwoQubitGateType, TwoQubitGateType)

def get_text_width(text: str, fontsize: int, ratio: float = 0.4) -> float:
    """Calculate the width of text based on font size and a ratio."""
    return len(text) * fontsize * ratio

def gate_name(gate: Gate) -> str:
    """Get the display name for a quantum gate."""
    if isinstance(gate, SingleQubitGate):
        return "$" + str(gate.gate.name) + "$"
    elif isinstance(gate, ParametricSingleQubitGate):
        return f"$R_{gate.gate.name[-1].lower()}({gate.parameter})$"
    elif isinstance(gate, TwoQubitGate):
        return gate.gate.name
    elif isinstance(gate, ParametricTwoQubitGate):
        if gate.gate in [ParametricTwoQubitGateType.CRX, ParametricTwoQubitGateType.CRY, ParametricTwoQubitGateType.CRZ]:
            return f"$R_{gate.gate.name[-1].lower()}({gate.parameter})$"
        else:
            return r"$R_{" + gate.gate.name[-2:].lower() + r"}" + f"({gate.parameter})$"
    elif isinstance(gate, ThreeQubitGate):
        return gate.gate.name
    elif isinstance(gate, MeasurementGate):
        return "M"
    else:
        raise ValueError(f"Unsupported gate type: {type(gate)}")

def calculate_gate_width(gate: Gate) -> float:
    """Calculate the width of a gate based on its text representation."""
    default_gate_width = 0.6
    if isinstance(gate, (SingleQubitGate, ParametricSingleQubitGate)):
        gate_text = str(gate_name(gate))
        text_width = get_text_width(gate_text, 8)
        return max(default_gate_width, text_width / 50)
    elif isinstance(gate, MeasurementGate):
        return default_gate_width
    elif isinstance(gate, ParametricTwoQubitGate):
        gate_text = str(gate_name(gate))
        text_width = get_text_width(gate_text, 8,ratio = 0.3)
        return max(default_gate_width, text_width / 50)
    else:
        return default_gate_width

def update_multi_qubit_gate_position(gate: Gate, qubit_pos: List[float], gate_width: float, gate_space: float):
    """
    Update the position of multi-qubit gates.

    Args:
        gate (Gate): The multi-qubit gate to update.
        qubit_pos (List[float]): The current positions of qubits.
        gate_width (float): The width of the gate.
        gate_space (float): The space between gates.

    Raises:
        ValueError: If an unsupported gate type is encountered.
    """
    if isinstance(gate, (TwoQubitGate, ParametricTwoQubitGate)):
        qubits = [gate.control, gate.target]
    elif isinstance(gate, ThreeQubitGate):
        qubits = [gate.control1, gate.control2, gate.target]
    else:
        raise ValueError(f"Unsupported gate type: {type(gate)}")


    _range = range(min(qubits), max(qubits) + 1)
    time_step = max((qubit_pos[_qubit_index] for _qubit_index in _range))
    gate.time_step = time_step
    for _qubit_index in _range:
        qubit_pos[_qubit_index] = time_step + gate_width + gate_space

def assign_time_steps(circuit: QuantumCircuit) -> Dict[int, float]:
    """Assign time steps to gates in the quantum circuit."""
    gate_space = 0.2
    gate_widths = {}
    qubit_pos = [0.0] * circuit.num_qubits

    for gate_index, gate in enumerate(circuit.gates):
        gate_width = calculate_gate_width(gate)
        gate_widths[gate_index] = gate_width

        if isinstance(gate, (SingleQubitGate, ParametricSingleQubitGate, MeasurementGate)):
            gate.time_step = qubit_pos[gate.qubit]
            qubit_pos[gate.qubit] += gate_width + gate_space
        elif isinstance(gate, (TwoQubitGate, ParametricTwoQubitGate, ThreeQubitGate)):
            update_multi_qubit_gate_position(gate, qubit_pos, gate_width, gate_space)
        else:
            raise ValueError(f"Unsupported gate type: {type(gate)}")

    return gate_widths

def add_gate_shape(ax: plt.Axes, gate: Gate, time_step: float, gate_width: float):
    """Add the basic shape of a gate to the quantum circuit diagram."""
    base_gate_size = 0.6
    qubit = gate.qubit if hasattr(gate, 'qubit') else gate.target
    rect = Rectangle((time_step - gate_width/2, qubit - base_gate_size/2), gate_width, base_gate_size, 
                     facecolor='white', edgecolor='black', lw=0.5, zorder=2)
    ax.add_patch(rect)
    return qubit

def add_gate_label(ax: plt.Axes, gate: Gate, time_step: float, qubit: int):
    """Add the label for a gate to the quantum circuit diagram."""
    gate_text = gate_name(gate)
    ax.text(time_step, qubit, gate_text, ha='center', va='center', fontweight='bold', zorder=3, fontsize=8)

def add_control_target(ax: plt.Axes, gate: Union[TwoQubitGate, ParametricTwoQubitGate, ThreeQubitGate], time_step: float):
    """Add control and target points for multi-qubit gates."""
    if isinstance(gate, (TwoQubitGate, ParametricTwoQubitGate)):
        add_control(ax, gate.control, time_step)
        if isinstance(gate, TwoQubitGate) and gate.gate == TwoQubitGateType.CNOT:
            add_target(ax, gate.target, time_step)
        elif isinstance(gate, TwoQubitGate) and gate.gate == TwoQubitGateType.CZ:
            add_control(ax, gate.target, time_step)
        add_connection(ax, gate.control, gate.target, time_step)
    elif isinstance(gate, ThreeQubitGate):
        add_control(ax, gate.control1, time_step)
        add_control(ax, gate.control2, time_step)
        add_target(ax, gate.target, time_step)
        add_connection(ax, gate.control1, gate.target, time_step)
        add_connection(ax, gate.control2, gate.target, time_step)

def add_gate(ax: plt.Axes, gate: Gate, time_step: float, gate_widths: Dict[int, float], circuit: QuantumCircuit):
    """Add a gate to the quantum circuit diagram."""
    gate_index = circuit.gates.index(gate)
    gate_width = gate_widths.get(gate_index, 0.6)

    if isinstance(gate, (SingleQubitGate, ParametricSingleQubitGate)):
        qubit = add_gate_shape(ax, gate, time_step, gate_width)
        add_gate_label(ax, gate, time_step, qubit)
    elif isinstance(gate, (TwoQubitGate, ParametricTwoQubitGate, ThreeQubitGate)):
        if isinstance(gate, ParametricTwoQubitGate):
            qubit = add_gate_shape(ax, gate, time_step, gate_width)
            add_gate_label(ax, gate, time_step, qubit)
        add_control_target(ax, gate, time_step)
    elif isinstance(gate, MeasurementGate):
        add_measurement(ax, gate.qubit, time_step)
    else:
        raise ValueError(f"Unsupported gate type: {type(gate)}")

def add_control(ax: plt.Axes, qubit: int, time_step: float):
    """Add a control point to the quantum circuit diagram."""
    ax.plot(time_step, qubit, 'ko', markersize=5, zorder=3)

def add_target(ax: plt.Axes, qubit: int, time_step: float):
    """Add a target point to the quantum circuit diagram."""
    circle = Circle((time_step, qubit), 0.15, facecolor='white', edgecolor='black', lw=0.5, zorder=2)
    ax.add_patch(circle)
    ax.plot([time_step-0.15, time_step+0.15], [qubit, qubit], 'black', linewidth=0.5, zorder=3)
    ax.plot([time_step, time_step], [qubit-0.15, qubit+0.15], 'black', linewidth=0.5, zorder=3)

def add_measurement(ax: plt.Axes, qubit: int, time_step: float):
    """Add a measurement gate to the quantum circuit diagram."""
    gate_size = 0.6
    rect = Rectangle((time_step - gate_size/2, qubit - gate_size/2), gate_size, gate_size, 
                     facecolor='white', edgecolor='black', lw=0.5, zorder=2)
    ax.add_patch(rect)
    
    arc_radius = 0.2
    arc = Arc((time_step, qubit + arc_radius/2), arc_radius, arc_radius, 
              theta1=180, theta2=360, lw=0.5, color='black', zorder=3)
    ax.add_patch(arc)
    
    needle_length = arc_radius * np.sqrt(2) / 4
    ax.plot([time_step, time_step + needle_length], 
            [qubit + arc_radius/2, qubit - arc_radius/4], 'black', linewidth=0.5, zorder=3)

def add_connection(ax: plt.Axes, qubit1: int, qubit2: int, time_step: float):
    """Add a connection line between two qubits in the quantum circuit diagram."""
    y_min, y_max = min(qubit1, qubit2), max(qubit1, qubit2)
    ax.vlines(x=time_step, ymin=y_min, ymax=y_max, linewidth=0.5, color='black', zorder=1)

def check_final_measurement(circuit: QuantumCircuit) -> bool:
    """Check if the quantum circuit ends with measurement gates."""
    return any(isinstance(gate, MeasurementGate) for gate in circuit.gates)

def draw_quantum_circuit(circuit: QuantumCircuit, max_time: float) -> Tuple[plt.Figure, plt.Axes]:
    """Draw the basic structure of a quantum circuit."""
    n_qubits = circuit.num_qubits
    fig, ax = plt.subplots(figsize=(max_time * 0.8 + 1, n_qubits * 0.7))
    
    ax.set_ylim(n_qubits - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    
    for i in range(n_qubits):
        ax.text(-0.8, i, r'$q_{' + str(i) + '}:\ ' + r'|0\rangle$', ha='right', va='center', fontsize=14)
    
    x_max = max_time + (0.5 if check_final_measurement(circuit) else 1.0)
    ax.set_xlim(-0.8, x_max)
    for i in range(n_qubits):
        ax.hlines(y=i, xmin=-0.5, xmax=x_max-0.5, linewidth=1, color='black', zorder=1)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    return fig, ax

@contextmanager
def plot_style():
    """Context manager for setting plot style."""
    with plt.style.context({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'cm'}):
        yield


def plot_quantum_circuit(circuit: QuantumCircuit, title: Union[bool, str] = True) -> None:
    """
    Plot a quantum circuit diagram.

    Args:
        circuit (QuantumCircuit): The quantum circuit to plot.
        title (Union[bool, str], optional): Title of the plot. If True, use default title.
                                            If False, no title. If string, use as custom title.
                                            Defaults to True.
    """
    with plot_style():
        gate_widths = assign_time_steps(circuit)
        max_time = max(gate.time_step for gate in circuit.gates)
        fig, ax = draw_quantum_circuit(circuit, max_time)
        
        for gate in circuit.gates:
            add_gate(ax, gate, gate.time_step, gate_widths, circuit)

        if title:
            if isinstance(title, str):
                plt.title(title)
            else:
                plt.title(f"Quantum Circuit: {circuit.name}" if circuit.name else "Quantum Circuit")
        
        plt.tight_layout()
        plt.show()