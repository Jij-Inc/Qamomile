from typing import Union, List, Tuple
import typing as typ
# Check matplotlib installation
# Because the drawer module is optional, `qamomile` should not depend on `matplotlib`
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(
        "The `matplotlib` package is required to use "
        + "the `qamomile.core.circuit.drawer.plot_quantum_circuit` function."
    )
import numpy as np
from matplotlib.patches import Rectangle, Circle, Arc
from contextlib import contextmanager
from .circuit import (
    QuantumCircuit,
    Gate,
    SingleQubitGate,
    ParametricSingleQubitGate,
    TwoQubitGate,
    ParametricTwoQubitGate,
    ThreeQubitGate,
    MeasurementGate,
    ParametricSingleQubitGateType,
    ParametricTwoQubitGateType,
    TwoQubitGateType,
    Operator
)


# named tuple for gate components
class GateComponent(typ.NamedTuple):
    """Gate component for plotting quantum circuits.

        <-->  = gate width
        ----
     --|    |--
        ----
          ^-- gate (center) position

    """
    gate: Gate
    position: float
    width: float


def plot_quantum_circuit(
    circuit: QuantumCircuit, title: Union[bool, str] = True,
    decompose_level: int = 0
) -> None:
    """
    Plot a quantum circuit diagram.

    Args:
        circuit (QuantumCircuit): The quantum circuit to plot.
        title (Union[bool, str], optional): Title of the plot. If True, use default title.
                                            If False, no title. If string, use as custom title.
                                            Defaults to True.
        decompose_level (int, optional): The level of circuit decomposition.
    """
    with plot_style():

        # Calculate gate widths and positions
        gate_components = _create_components(circuit, decompose_level)

        # Draw only wires of Quantum Circuit ------------
        # Ex. 2 qubits
        # q0 |0> -----------
        # q1 |0> -----------
        fig, ax = _draw_quantum_circuit(gate_components, circuit.num_qubits, circuit)

        for gate in gate_components:
            add_gate(ax, gate.gate, gate.position, gate.width)

        if title:
            if isinstance(title, str):
                plt.title(title)
            else:
                plt.title(
                    f"Quantum Circuit: {circuit.name}"
                    if circuit.name
                    else "Quantum Circuit"
                )

        plt.tight_layout()
        plt.show()


def _create_components(
    circuit: QuantumCircuit,
    decompose_level: int = 0,
    gate_space=0.2
) -> list[GateComponent]:
    gate_components: list[GateComponent] = []
    _qubit_pos = np.zeros(circuit.num_qubits)
    # Gate position = A center of the gate
    #    ----
    # --|    |--
    #    ----
    #      ^-- gate position

    def _decompose(_gate_list, decompose_level):
        if decompose_level == 0:
            return _gate_list
        new_gate_list = []
        for gate in _gate_list:
            if isinstance(gate, Operator):
                new_gate_list.extend(_decompose(gate.circuit.gates, decompose_level - 1))
            else:
                new_gate_list.append(gate)
        return new_gate_list

    gate_list = _decompose(circuit.gates, decompose_level)

    for gate in gate_list:
        gate_width = calculate_gate_width(gate)
        MULTI_QUBIT_GATE_TYPE = (
            TwoQubitGate,
            ParametricTwoQubitGate,
            ThreeQubitGate,
            Operator,
        )
        if isinstance(
            gate, (SingleQubitGate, ParametricSingleQubitGate, MeasurementGate)
        ):
            gate_pos = _qubit_pos[gate.qubit] + gate_width / 2
            _qubit_pos[gate.qubit] += gate_width + gate_space
            gate_components.append(GateComponent(gate, gate_pos, gate_width))
        elif isinstance(gate, MULTI_QUBIT_GATE_TYPE):
            if isinstance(gate, (TwoQubitGate, ParametricTwoQubitGate)):
                operated_qubits = [gate.control, gate.target]
            elif isinstance(gate, ThreeQubitGate):
                operated_qubits = [gate.control1, gate.control2, gate.target]
            elif isinstance(gate, Operator):
                operated_qubits = gate.operated_qubits()
            straddling_qubits = np.arange(min(operated_qubits), max(operated_qubits) + 1)
            max_posision = np.max(_qubit_pos[straddling_qubits])
            gate_pos = max_posision + gate_width / 2
            for qubit in straddling_qubits:
                _qubit_pos[qubit] = max_posision + gate_width + gate_space
            gate_components.append(GateComponent(gate, gate_pos, gate_width))
        else:
            raise ValueError(f"Unsupported gate type: {type(gate)}")
    return gate_components


def _draw_quantum_circuit(
    gate_components: List[GateComponent],
    n_qubits: int,
    circuit: QuantumCircuit,
) -> Tuple[plt.Figure, plt.Axes]:
    """Draw the basic structure of a quantum circuit.

          <-> = wire mergin
    q0 |0>   -----------
             <-> = wire padding
    q1 |0>   ---■-----

    """

    WIRE_MERGIN = 0.4
    WIRE_PADDING = 0.2

    max_time = max(gate.position + gate.width / 2 for gate in gate_components)
    fig, ax = plt.subplots(figsize=(max_time * 0.8 + 1, n_qubits * 0.7))

    ax.set_ylim(n_qubits - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xticks([])

    # add qubit labels and initial states
    for i in range(n_qubits):
        ax.text(
            -WIRE_MERGIN,
            i,
            f"${circuit.qubits_label[i]}:\ " + r"|0\rangle$",
            ha="right",
            va="center",
            fontsize=14,
        )

    # calculate x-axis limits
    # when the circuit ends with measurement gates, add extra space
    x_max = max_time + (0.5 if check_final_measurement(circuit) else 1.0)
    ax.set_xlim(-0.8, x_max)

    # add horizontal lines for qubits (wires)
    for i in range(n_qubits):
        ax.hlines(
            y=i, xmin=-WIRE_PADDING, xmax=x_max - 0.5, linewidth=1, color="black", zorder=1
        )

    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig, ax


def add_gate(
    ax: plt.Axes,
    gate: Gate,
    time_step: float,
    gate_width: float,
):
    """Add a gate to the quantum circuit diagram."""

    def gate_rectangle(qubits: list[int], gate_space=0.3):
        min_qubit = min(qubits)
        max_qubit = max(qubits)
        rectangle_top = min_qubit - gate_space
        rectangle_bottom = max_qubit + gate_space
        return Rectangle(
            (time_step - gate_width / 2, rectangle_top),
            gate_width,
            rectangle_bottom - rectangle_top,
            facecolor="white",
            edgecolor="black",
            lw=0.5,
            zorder=2,
        )

    def add_gate_text(
        qubits: list[int], text: str, gate_space=0.3, fontsize=8
    ):
        min_qubit = min(qubits)
        max_qubit = max(qubits)
        rectangle_top = min_qubit - gate_space
        rectangle_bottom = max_qubit + gate_space
        ax.text(
            time_step,
            rectangle_bottom - (rectangle_bottom - rectangle_top) / 2,
            text,
            ha="center",
            va="center",
            # fontweight="bold",
            zorder=3,
            fontsize=fontsize,
        )
        return (rectangle_top, rectangle_bottom)

    if isinstance(gate, (SingleQubitGate, ParametricSingleQubitGate)):
        rect = gate_rectangle([gate.qubit])
        ax.add_patch(rect)
        gate_text = gate_name(gate)
        add_gate_text([gate.qubit], gate_text)
    elif isinstance(gate, (TwoQubitGate, ParametricTwoQubitGate, ThreeQubitGate)):
        if isinstance(gate, ParametricTwoQubitGate):
            rect = gate_rectangle([gate.target])
            ax.add_patch(rect)
            add_gate_text([gate.target], gate_name(gate))
        add_control_target(ax, gate, time_step)
    elif isinstance(gate, MeasurementGate):
        add_measurement(ax, gate.qubit, time_step)
    elif isinstance(gate, Operator):
        operated_qubits = gate.operated_qubits()
        rect = gate_rectangle(operated_qubits)
        for qubit in operated_qubits:
            ax.text(
                time_step - gate_width / 2 + 0.1,
                qubit,
                f"{qubit}",
                ha="left",
                va="center",
                fontsize=10,
                zorder=3,
            )
        ax.add_patch(rect)
        # Add gate label in the center of the gate
        gate_top, gate_bottom = add_gate_text(operated_qubits, gate_name(gate), fontsize=10)
        parameters_name = [param.name for param in gate.circuit.get_parameters()]
        gate_center = gate_bottom - (gate_top + gate_bottom) / 2
        for i, param in enumerate(parameters_name):
            ax.text(
                time_step,
                gate_center + 0.4 + 0.2 * i,
                param,
                ha="center",
                va="center",
                fontsize=8,
                zorder=3,
            )
    else:
        raise ValueError(f"Unsupported gate type: {type(gate)}")


def gate_name(gate: Gate) -> str:
    """Get the display name for a quantum gate.

    Args:
        gate (Gate): The quantum gate.

    Returns:
        str: The display name of the quantum gate.

    Examples:
        >>> gate_name(SingleQubitGate(0, SingleQubitGateType.H))
        '$H$'
    """
    match gate:
        case SingleQubitGate():
            return "$" + str(gate.gate.name) + "$"
        case ParametricSingleQubitGate():
            return f"$R_{gate.gate.name[-1].lower()}({gate.parameter})$"
        case TwoQubitGate():
            return gate.gate.name
        case ParametricTwoQubitGate():
            if gate.gate in [
                ParametricTwoQubitGateType.CRX,
                ParametricTwoQubitGateType.CRY,
                ParametricTwoQubitGateType.CRZ,
            ]:
                return f"$R_{gate.gate.name[-1].lower()}({gate.parameter})$"
            else:
                return (
                    r"$R_{"
                    + gate.gate.name[-2:].lower()
                    + r"}"
                    + f"({gate.parameter})$"
                )
        case ThreeQubitGate():
            return gate.gate.name
        case MeasurementGate():
            return "M"
        case Operator():
            gate_label = gate.label if gate.label else "U"
            return gate_label
        case _:
            raise ValueError(f"Unsupported gate type: {type(gate)}")


def get_text_width(text: str, fontsize: int, ratio: float = 0.4) -> float:
    """Calculate the width of text based on font size and a ratio."""
    return len(text) * fontsize * ratio


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
        text_width = get_text_width(gate_text, 8, ratio=0.3)
        return max(default_gate_width, text_width / 50)
    elif isinstance(gate, Operator):
        gate_text = "U" if not gate.label else gate.label
        gate_text = f"  {gate_text}  "
        text_width = get_text_width(gate_text, 8)
        return 0.2 * len(gate_text)
    else:
        return default_gate_width


def add_control_target(
    ax: plt.Axes,
    gate: Union[TwoQubitGate, ParametricTwoQubitGate, ThreeQubitGate],
    time_step: float,
):
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


def add_control(ax: plt.Axes, qubit: int, time_step: float):
    """Add a control point to the quantum circuit diagram."""
    ax.plot(time_step, qubit, "ko", markersize=5, zorder=3)


def add_target(ax: plt.Axes, qubit: int, time_step: float):
    """Add a target point to the quantum circuit diagram."""
    circle = Circle(
        (time_step, qubit), 0.15, facecolor="white", edgecolor="black", lw=0.5, zorder=2
    )
    ax.add_patch(circle)
    ax.plot(
        [time_step - 0.15, time_step + 0.15],
        [qubit, qubit],
        "black",
        linewidth=0.5,
        zorder=3,
    )
    ax.plot(
        [time_step, time_step],
        [qubit - 0.15, qubit + 0.15],
        "black",
        linewidth=0.5,
        zorder=3,
    )


def add_measurement(ax: plt.Axes, qubit: int, time_step: float):
    """Add a measurement gate to the quantum circuit diagram."""
    gate_size = 0.6
    rect = Rectangle(
        (time_step - gate_size / 2, qubit - gate_size / 2),
        gate_size,
        gate_size,
        facecolor="white",
        edgecolor="black",
        lw=0.5,
        zorder=2,
    )
    ax.add_patch(rect)

    arc_radius = 0.2
    arc = Arc(
        (time_step, qubit + arc_radius / 2),
        arc_radius,
        arc_radius,
        theta1=180,
        theta2=360,
        lw=0.5,
        color="black",
        zorder=3,
    )
    ax.add_patch(arc)

    needle_length = arc_radius * np.sqrt(2) / 4
    ax.plot(
        [time_step, time_step + needle_length],
        [qubit + arc_radius / 2, qubit - arc_radius / 4],
        "black",
        linewidth=0.5,
        zorder=3,
    )


def add_connection(ax: plt.Axes, qubit1: int, qubit2: int, time_step: float):
    """Add a connection line between two qubits in the quantum circuit diagram."""
    y_min, y_max = min(qubit1, qubit2), max(qubit1, qubit2)
    ax.vlines(
        x=time_step, ymin=y_min, ymax=y_max, linewidth=0.5, color="black", zorder=1
    )


def check_final_measurement(circuit: QuantumCircuit) -> bool:
    """Check if the quantum circuit ends with measurement gates."""
    return any(isinstance(gate, MeasurementGate) for gate in circuit.gates)


@contextmanager
def plot_style():
    """Context manager for setting plot style."""
    with plt.style.context({"font.family": "DejaVu Sans", "mathtext.fontset": "cm"}):
        yield
