"""Controlled gate operations."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Union

from qamomile.circuit.frontend.handle.primitives import Qubit, Float, UInt
from qamomile.circuit.frontend.handle import Handle
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.value import Value
from qamomile.circuit.ir.types.primitives import FloatType

if TYPE_CHECKING:
    from qamomile.circuit.frontend.qkernel import QKernel

# Type alias for parameter values
ParamValue = Union[float, int, Float, UInt]


class ControlledGate:
    """Wrapper for controlled version of a QKernel.

    Created by calling `controlled(qkernel)`. The resulting object
    can be called like a gate function.

    Example:
        @qmc.qkernel
        def phase_gate(q: Qubit, theta: float) -> Qubit:
            return qmc.p(q, theta)

        controlled_phase = qmc.controlled(phase_gate)
        ctrl_out, tgt_out = controlled_phase(ctrl, target, theta=0.5)
    """

    def __init__(self, qkernel: "QKernel") -> None:
        self._qkernel = qkernel

    def __call__(
        self,
        control: Qubit,
        *target_args: Qubit,
        power: int = 1,
        **params: ParamValue,
    ) -> tuple[Qubit, ...]:
        """Apply controlled gate.

        Args:
            control: The control qubit
            *target_args: Target qubit(s) for the underlying gate
            power: Number of times to apply U. Default is 1.
                   For QPE, this is 2^k. The emitter creates Controlled(U^power),
                   NOT Controlled(U)^power.
            **params: Parameters for the underlying gate (e.g., theta=0.5)

        Returns:
            Tuple of (control_out, *target_outs)
        """
        # Get BlockValue from qkernel
        block = self._qkernel.block

        # Build operands: [BlockValue, control, target(s), param(s)]
        operands: list[Any] = [block, control.value]

        for tgt in target_args:
            operands.append(tgt.value)

        # Add parameter values
        for param_name, param_value in params.items():
            if isinstance(param_value, Handle):
                # Handle types (Float, UInt, etc.) have a .value attribute
                operands.append(param_value.value)
            else:
                # Create constant Value for raw float/int
                param_val = Value(
                    type=FloatType(),
                    name=f"ctrl_param_{param_name}",
                    params={"const": float(param_value)},
                )
                operands.append(param_val)

        # Build results: [control_out, target_out(s)]
        results: list[Value] = [control.value.next_version()]
        for tgt in target_args:
            results.append(tgt.value.next_version())

        # Create ControlledUOperation with power
        op = ControlledUOperation(
            operands=operands,
            results=results,
            num_controls=1,
            power=power,
        )

        # Add to tracer
        tracer = get_current_tracer()
        tracer.add_operation(op)

        # Return output handles
        output_qubits: list[Qubit] = []

        # Control qubit output
        output_qubits.append(
            Qubit(
                value=results[0],
                parent=control.parent,
                indices=control.indices,
            )
        )

        # Target qubit outputs
        for i, tgt in enumerate(target_args):
            output_qubits.append(
                Qubit(
                    value=results[1 + i],
                    parent=tgt.parent,
                    indices=tgt.indices,
                )
            )

        return tuple(output_qubits)


def controlled(qkernel: "QKernel") -> ControlledGate:
    """Create a controlled version of a quantum gate (QKernel).

    Args:
        qkernel: A QKernel defining the gate to control.

    Returns:
        ControlledGate that can be called with (control, *targets, **params)

    Example:
        @qmc.qkernel
        def rx_gate(q: Qubit, theta: float) -> Qubit:
            return qmc.rx(q, theta)

        crx = qmc.controlled(rx_gate)
        ctrl_out, tgt_out = crx(ctrl_qubit, target_qubit, theta=0.5)
    """
    return ControlledGate(qkernel)
