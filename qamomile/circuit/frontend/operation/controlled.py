"""Controlled gate operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

from qamomile.circuit.frontend.handle import Handle
from qamomile.circuit.frontend.handle.primitives import Float, Qubit, UInt
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.types.primitives import FloatType, UIntType
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

    def __init__(self, qkernel: "QKernel", num_controls: int | UInt = 1) -> None:
        if isinstance(num_controls, int) and num_controls < 1:
            raise ValueError(f"num_controls must be >= 1, got {num_controls}.")
        # For UInt (symbolic), validation is deferred to emit time
        self._qkernel = qkernel
        self._num_controls = num_controls

    @staticmethod
    def _normalize_power(power: int | UInt) -> int | Value:
        """Normalize power to an IR-compatible type (``int`` or ``Value``).

        ``UInt`` handles are unwrapped to their underlying ``Value`` so
        that the IR never stores frontend types.  Concrete ``int`` values
        are validated for strict positivity.

        Args:
            power: The power value from the user API.

        Returns:
            ``int`` for concrete values, ``Value`` for symbolic expressions.

        Raises:
            TypeError: If *power* is ``bool``, ``float``, or another
                unsupported type.
            ValueError: If a concrete *power* is ``<= 0``.
        """
        if isinstance(power, bool):
            raise TypeError(
                f"power must be a positive integer, got bool ({power}). "
                f"Use an integer value like power=1 or power=2."
            )
        if isinstance(power, UInt):
            return power.value
        if isinstance(power, int):
            if power <= 0:
                raise ValueError(
                    f"power must be a strictly positive integer, got {power}."
                )
            return power
        raise TypeError(f"power must be int or UInt, got {type(power).__name__}.")

    def __call__(
        self,
        *args: Any,
        power: int | UInt = 1,
        target_indices: list[int | UInt] | None = None,
        controlled_indices: list[int | UInt] | None = None,
        **params: ParamValue,
    ) -> tuple[Any, ...] | Any:
        """Apply controlled gate.

        For concrete num_controls (int):
            *args: First num_controls qubits are controls, rest are targets.
        For symbolic num_controls (UInt):
            args[0]: Vector[Qubit] (controls), args[1:]: individual Qubit targets.
        With target_indices or controlled_indices:
            args[0]: Single Vector[Qubit], indices specify which elements are
            targets (or controls). Returns a single Vector (not a tuple).

        Args:
            *args: Control and target qubits.
            power: Number of times to apply U. Must be a strictly positive
                integer. Accepts UInt handles for symbolic expressions
                (e.g., 2**k in QPE).
            target_indices: Indices within the Vector that are targets.
                The remaining elements become controls.
            controlled_indices: Indices within the Vector that are controls.
                The remaining elements become targets.
            **params: Parameters for the underlying gate (e.g., theta=0.5)

        Returns:
            Tuple of output handles, or single Vector when using index spec.
        """
        power = self._normalize_power(power)

        if target_indices is not None and controlled_indices is not None:
            raise ValueError(
                "Cannot specify both target_indices and controlled_indices. "
                "Use one or the other."
            )

        if target_indices is not None or controlled_indices is not None:
            return self._call_with_index_spec(
                args, power, target_indices, controlled_indices, params
            )

        num_controls = self._num_controls

        if isinstance(num_controls, UInt):
            return self._call_symbolic(args, power, params)

        # --- Concrete path (existing logic, unchanged) ---

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
        controls = tuple(
            c.consume(operation_name="ControlledU[control]") for c in controls
        )
        target_args = tuple(
            t.consume(operation_name="ControlledU[target]") for t in target_args
        )

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

    def _call_with_index_spec(
        self,
        args: tuple[Any, ...],
        power: int | Value,
        target_indices: list[int | UInt] | None,
        controlled_indices: list[int | UInt] | None,
        params: dict[str, ParamValue],
    ) -> Any:
        """Handle index-spec mode: single Vector with explicit target/control indices."""
        from qamomile.circuit.frontend.handle.array import ArrayBase, Vector

        # Validate: exactly one Vector argument
        if len(args) != 1 or not isinstance(args[0], ArrayBase):
            raise ValueError(
                "When target_indices or controlled_indices is specified, "
                "exactly one Vector[Qubit] must be provided."
            )

        indices = target_indices if target_indices is not None else controlled_indices
        assert indices is not None

        # Validate: non-empty
        if len(indices) == 0:
            raise ValueError(
                "At least one index must be specified in "
                "target_indices or controlled_indices."
            )

        # Validate: no duplicate concrete indices
        concrete = [i for i in indices if isinstance(i, int)]
        if len(set(concrete)) != len(concrete):
            raise ValueError(
                "Duplicate indices are not allowed in "
                "target_indices/controlled_indices."
            )

        # Convert indices to Value list
        def _to_value_list(idx_list: list[int | UInt]) -> list[Value]:
            result: list[Value] = []
            for idx in idx_list:
                if isinstance(idx, int):
                    val = Value(
                        type=UIntType(),
                        name=f"idx_{idx}",
                        params={"const": idx},
                    )
                    result.append(val)
                elif isinstance(idx, UInt):
                    result.append(idx.value)
                else:
                    raise TypeError(f"Index must be int or UInt, got {type(idx)}")
            return result

        ti_values = (
            _to_value_list(target_indices) if target_indices is not None else None
        )
        ci_values = (
            _to_value_list(controlled_indices)
            if controlled_indices is not None
            else None
        )

        vector = args[0]
        # Consume the Vector (linear type)
        vector = vector.consume(operation_name="ControlledU[index_spec]")

        block = self._qkernel.block

        # operands: [BlockValue, ArrayValue, params...]
        operands: list[Any] = [block, vector.value]
        for param_name, param_value in params.items():
            if isinstance(param_value, Handle):
                operands.append(param_value.value)
            else:
                param_val = Value(
                    type=FloatType(),
                    name=f"ctrl_param_{param_name}",
                    params={"const": float(param_value)},
                )
                operands.append(param_val)

        # results: [ArrayValue (updated Vector)]
        results = [vector.value.next_version()]

        # Get num_controls as IR Value
        nc = self._num_controls
        if isinstance(nc, UInt):
            nc = nc.value

        op = ControlledUOperation(
            operands=operands,
            results=results,
            num_controls=nc,
            power=power,
            target_indices=ti_values,
            controlled_indices=ci_values,
        )

        tracer = get_current_tracer()
        tracer.add_operation(op)

        # Return a single Vector (not tuple)
        return Vector._create_from_value(results[0], vector.shape, vector.value.name)

    def _call_symbolic(
        self,
        args: tuple[Any, ...],
        power: int | Value,
        params: dict[str, ParamValue],
    ) -> tuple[Any, ...]:
        """Handle symbolic num_controls (UInt).

        Convention: args[0] is Vector[Qubit] (controls), args[1:] are targets.
        """
        from qamomile.circuit.frontend.handle.array import ArrayBase, Vector

        num_controls = self._num_controls
        assert isinstance(num_controls, UInt)

        if not args or not isinstance(args[0], ArrayBase):
            raise ValueError(
                "When num_controls is symbolic (UInt), the first argument "
                "must be a Vector[Qubit] for control qubits."
            )

        control_vector = args[0]
        target_args = args[1:]

        if not target_args:
            raise ValueError("ControlledU requires at least 1 target qubit.")

        # Consume control vector (linear type enforcement)
        control_vector = control_vector.consume(operation_name="ControlledU[controls]")

        # Consume target qubits
        consumed_targets = []
        for tgt in target_args:
            consumed_targets.append(tgt.consume(operation_name="ControlledU[target]"))

        # Get BlockValue from qkernel
        block = self._qkernel.block

        # Build operands: [BlockValue, control_vector, target(s), param(s)]
        operands: list[Any] = [block, control_vector.value]
        for tgt in consumed_targets:
            operands.append(tgt.value)

        # Add parameter values
        for param_name, param_value in params.items():
            if isinstance(param_value, Handle):
                operands.append(param_value.value)
            else:
                param_val = Value(
                    type=FloatType(),
                    name=f"ctrl_param_{param_name}",
                    params={"const": float(param_value)},
                )
                operands.append(param_val)

        # Build results: [control_vector_out, target_out(s)]
        results: list[Value] = []
        results.append(control_vector.value.next_version())
        for tgt in consumed_targets:
            results.append(tgt.value.next_version())

        # Create ControlledUOperation with symbolic num_controls
        op = ControlledUOperation(
            operands=operands,
            results=results,
            num_controls=num_controls.value,  # IR Value, not UInt handle
            power=power,
        )

        # Add to tracer
        tracer = get_current_tracer()
        tracer.add_operation(op)

        # Return output handles
        output_handles: list[Any] = []

        # Control vector output
        control_out = Vector._create_from_value(
            results[0], control_vector.shape, control_vector.value.name
        )
        output_handles.append(control_out)

        # Target outputs
        for i, tgt in enumerate(consumed_targets):
            output_handles.append(
                Qubit(
                    value=results[1 + i],
                    parent=tgt.parent,
                    indices=tgt.indices,
                )
            )

        return tuple(output_handles)


def controlled(qkernel: "QKernel", num_controls: int | UInt = 1) -> ControlledGate:
    """Create a controlled version of a quantum gate (QKernel).

    Args:
        qkernel: A QKernel defining the gate to control.
        num_controls: Number of control qubits (default: 1).
            Can be int (concrete) or UInt (symbolic).

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
