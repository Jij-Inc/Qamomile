"""Controlled gate operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

from qamomile.circuit.frontend.handle import Handle
from qamomile.circuit.frontend.handle.primitives import Float, Qubit, UInt
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.types.primitives import FloatType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.errors import QubitAliasError

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

        # Double-controlled
        cc_phase = qmc.controlled(phase_gate, num_controls=2)
        c0, c1, tgt = cc_phase(ctrl0, ctrl1, target, theta=0.5)
    """

    def __init__(self, qkernel: "QKernel", num_controls: int = 1) -> None:
        if num_controls < 1:
            raise ValueError(
                f"num_controls must be >= 1, got {num_controls}."
            )
        self._qkernel = qkernel
        self._num_controls = num_controls

    def __call__(
        self,
        *args: Qubit,
        power: int = 1,
        **params: ParamValue,
    ) -> tuple[Qubit, ...]:
        """Apply controlled gate.

        Args:
            *args: First num_controls qubits are controls, rest are targets.
            power: Number of times to apply U. Default is 1.
                   For QPE, this is 2^k. The emitter creates Controlled(U^power),
                   NOT Controlled(U)^power.
            **params: Parameters for the underlying gate (e.g., theta=0.5)

        Returns:
            Tuple of (control_0, ..., control_N, target_0, ..., target_M)
        """
        num_controls = self._num_controls

        # Split args into controls and targets
        if len(args) <= num_controls:
            raise ValueError(
                f"ControlledU requires at least {num_controls + 1} qubits "
                f"({num_controls} controls + at least 1 target), got {len(args)}."
            )
        controls = args[:num_controls]
        target_args = args[num_controls:]

        # Check for aliasing (same physical qubit used in multiple positions)
        seen_ids: set[str] = set()
        for q in args:
            lid = q.value.logical_id
            if lid in seen_ids:
                q_name = q.name or "unnamed"
                raise QubitAliasError(
                    f"Cannot use the same qubit in multiple positions of ControlledU.\n"
                    f"Qubit '{q_name}' appears more than once.\n\n"
                    f"Fix: Use distinct qubits for each control and target.",
                    handle_name=q_name,
                    operation_name="ControlledU",
                )
            seen_ids.add(lid)

        # Consume all qubit handles (enforces linear type)
        controls = tuple(c.consume(operation_name="ControlledU[control]") for c in controls)
        target_args = tuple(t.consume(operation_name="ControlledU[target]") for t in target_args)

        # Get BlockValue from qkernel
        block = self._qkernel.block

        # Build operands: [BlockValue, control(s), target(s), param(s)]
        operands: list[Any] = [block]

        for ctrl in controls:
            operands.append(ctrl.value)

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

        # Build results: [control_out(s), target_out(s)]
        results: list[Value] = []
        for ctrl in controls:
            results.append(ctrl.value.next_version())
        for tgt in target_args:
            results.append(tgt.value.next_version())

        # Create ControlledUOperation with power
        op = ControlledUOperation(
            operands=operands,
            results=results,
            num_controls=num_controls,
            power=power,
        )

        # Add to tracer
        tracer = get_current_tracer()
        tracer.add_operation(op)

        # Return output handles
        output_qubits: list[Qubit] = []

        # Control qubit outputs
        for i, ctrl in enumerate(controls):
            output_qubits.append(
                Qubit(
                    value=results[i],
                    parent=ctrl.parent,
                    indices=ctrl.indices,
                )
            )

        # Target qubit outputs
        for i, tgt in enumerate(target_args):
            output_qubits.append(
                Qubit(
                    value=results[num_controls + i],
                    parent=tgt.parent,
                    indices=tgt.indices,
                )
            )

        return tuple(output_qubits)


def controlled(qkernel: "QKernel", num_controls: int = 1) -> ControlledGate:
    """Create a controlled version of a quantum gate (QKernel).

    Args:
        qkernel: A QKernel defining the gate to control.
        num_controls: Number of control qubits (default: 1).

    Returns:
        ControlledGate that can be called with (*controls, *targets, **params)

    Example:
        @qmc.qkernel
        def rx_gate(q: Qubit, theta: float) -> Qubit:
            return qmc.rx(q, theta)

        crx = qmc.controlled(rx_gate)
        ctrl_out, tgt_out = crx(ctrl_qubit, target_qubit, theta=0.5)

        # Double-controlled
        ccrx = qmc.controlled(rx_gate, num_controls=2)
        c0, c1, tgt = ccrx(ctrl0, ctrl1, target, theta=0.5)
    """
    return ControlledGate(qkernel, num_controls=num_controls)
