"""CUDA-Q backend transpiler implementation.

This module provides CudaqTranspiler for converting Qamomile QKernels
into CUDA-Q kernels, along with CudaqEmitPass and CudaqExecutor.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence, TYPE_CHECKING

from qamomile.circuit.ir.operation.control_flow import ForOperation, IfOperation
from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.passes.emit_base import resolve_if_condition
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.separate import SeparatePass
from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass
from qamomile.circuit.transpiler.executable import (
    QuantumExecutor,
    ParameterMetadata,
)

from .emitter import CudaqCircuit, CudaqGateEmitter

if TYPE_CHECKING:
    import qamomile.observable as qm_o


def _build_block_qubit_map(
    block_value: Any,
    target_indices: list[int],
) -> dict[str, int]:
    """Build a UUID-to-physical-target map for a controlled block.

    Seeds the map from block quantum input_values (positionally matching
    ``target_indices``), then propagates through all GateOperations so
    that SSA result versions inherit the same physical target as their
    operands.

    This allows resolving which physical target an inner gate's operand
    refers to, even across multiple SSA versions.
    """
    qubit_map: dict[str, int] = {}

    # Seed from block inputs.
    if hasattr(block_value, "input_values"):
        qubit_idx = 0
        for iv in block_value.input_values:
            if hasattr(iv, "type") and iv.type.is_quantum():
                if qubit_idx < len(target_indices):
                    qubit_map[iv.uuid] = target_indices[qubit_idx]
                qubit_idx += 1

    # Propagate through operations: result inherits operand's target.
    if hasattr(block_value, "operations"):
        for op in block_value.operations:
            if isinstance(op, GateOperation):
                for i, result in enumerate(op.results):
                    if hasattr(result, "type") and result.type.is_quantum():
                        # Find the corresponding operand's physical target.
                        if i < len(op.operands):
                            operand = op.operands[i]
                            if hasattr(operand, "uuid") and operand.uuid in qubit_map:
                                qubit_map[result.uuid] = qubit_map[operand.uuid]

    return qubit_map


def _resolve_gate_targets(
    op: GateOperation,
    qubit_map: dict[str, int],
    fallback_targets: list[int],
) -> list[int]:
    """Resolve physical target indices for an inner gate's operands.

    For each quantum operand of the gate, looks up the corresponding
    physical target via ``qubit_map``.  Falls back to ``fallback_targets``
    if no mapping is found.
    """
    resolved: list[int] = []
    for operand in op.operands:
        if hasattr(operand, "type") and operand.type.is_quantum():
            if operand.uuid in qubit_map:
                resolved.append(qubit_map[operand.uuid])
            elif fallback_targets:
                resolved.append(fallback_targets[0])
    return resolved if resolved else fallback_targets


@dataclasses.dataclass
class BoundCudaqCircuit:
    """CUDA-Q kernel with bound parameter values.

    Used as the return type of ``CudaqExecutor.bind_parameters``.
    The executor dispatches to ``cudaq.sample(kernel, param_values, ...)``
    or ``cudaq.observe(kernel, spin_op, param_values)`` when it receives
    this type.

    Args:
        kernel (Any): The CUDA-Q kernel builder instance.
        num_qubits (int): Number of qubits in the circuit.
        param_values (list[float]): Bound parameter values in order.
    """

    kernel: Any
    num_qubits: int
    param_values: list[float]


class CudaqEmitPass(StandardEmitPass[CudaqCircuit]):
    """CUDA-Q-specific emission pass.

    Uses StandardEmitPass with CudaqGateEmitter for gate emission.
    Measurement-dependent conditional branching (``c_if``) is not supported
    under CUDA-Q 0.14.x and raises ``EmitError`` at emit time.
    For-loops are unrolled and while-loops raise ``EmitError``.
    """

    def __init__(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ):
        parametric = bool(parameters)
        emitter = CudaqGateEmitter(parametric=parametric)
        composite_emitters: list[Any] = []
        super().__init__(emitter, bindings, parameters, composite_emitters)

    def _emit_if(
        self,
        circuit: CudaqCircuit,
        op: IfOperation,
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> None:
        """Reject measurement-dependent conditional branching.

        CUDA-Q 0.14.x removed the builder ``c_if`` API.  Any
        ``IfOperation`` whose condition is a runtime measurement result
        (i.e. a Value with a UUID) is therefore unsupported and raises
        ``EmitError``.

        Compile-time constant conditions are forwarded to the base-class
        implementation (``StandardEmitPass._emit_if``) which handles
        them without relying on ``c_if``.

        Args:
            circuit (CudaqCircuit): The CUDA-Q circuit being built.
            op (IfOperation): The if-operation from the IR.
            qubit_map (dict[str, int]): UUID-to-qubit-index mapping.
            clbit_map (dict[str, int]): UUID-to-clbit-index mapping.
            bindings (dict[str, Any]): Parameter bindings.

        Raises:
            EmitError: When the condition is a runtime measurement result.
        """
        condition = op.condition

        # Compile-time constant conditions are handled by the base class.
        if resolve_if_condition(condition, bindings) is not None:
            super()._emit_if(circuit, op, qubit_map, clbit_map, bindings)
            return

        raise EmitError(
            "CUDA-Q 0.14.x does not support measurement-dependent conditional "
            "branching. The `c_if` builder API was removed in 0.14.0. "
            "Refactor to use static circuits without runtime conditional branching."
        )

    def _emit_controlled_fallback(
        self,
        circuit: CudaqCircuit,
        block_value: Any,
        num_controls: int,
        control_indices: list[int],
        target_indices: list[int],
        power: int,
        bindings: dict[str, Any],
    ) -> None:
        """Emit controlled-U using CUDA-Q native multi-control.

        CUDA-Q supports ``kernel.<gate>([controls], target)`` for
        multi-controlled single-qubit gates.  This override handles
        multi-control by iterating over the block body and emitting each
        gate with native multi-control.

        An operand-to-target resolver maps each inner gate's operand to
        the correct physical target index based on block input positions,
        rather than hardcoding ``target_indices[0]``.

        For single-control cases, compile-time resolvable ``ForOperation``
        loops are unrolled and their body gates emitted with correct
        operand-to-target mapping.  Multi-control helpers with non-gate
        operations raise ``EmitError``.

        Args:
            circuit: The CUDA-Q circuit being built.
            block_value: The block value containing operations to control.
            num_controls: Number of control qubits.
            control_indices: Physical indices of control qubits.
            target_indices: Physical indices of target qubits.
            power: Number of times to repeat the controlled operation.
            bindings: Parameter bindings.

        Raises:
            EmitError: When the block body contains unsupported operations
                or operand-to-target resolution fails.
        """
        if not hasattr(block_value, "operations"):
            raise EmitError(
                "Cannot emit controlled operation: block has no operations.",
                operation="ControlledUOperation",
            )

        # Build operand-to-target map from block inputs, propagated
        # through SSA versions.
        block_qubit_map = _build_block_qubit_map(block_value, target_indices)

        emitter: CudaqGateEmitter = self._emitter  # type: ignore[assignment]

        for _ in range(power):
            self._emit_cudaq_controlled_ops(
                circuit,
                block_value.operations,
                num_controls,
                control_indices,
                target_indices,
                block_qubit_map,
                emitter,
                bindings,
            )

    def _emit_cudaq_controlled_ops(
        self,
        circuit: CudaqCircuit,
        operations: list[Any],
        num_controls: int,
        control_indices: list[int],
        target_indices: list[int],
        qubit_map: dict[str, int],
        emitter: CudaqGateEmitter,
        bindings: dict[str, Any],
    ) -> None:
        """Recursively emit controlled operations with operand-to-target mapping.

        Handles ``GateOperation`` via the CUDA-Q multi-control path,
        ``ReturnOperation`` by skipping, and ``ForOperation`` by unrolling
        compile-time resolvable loops (single-control only).

        Args:
            circuit: The CUDA-Q circuit being built.
            operations: List of operations to process.
            num_controls: Number of control qubits.
            control_indices: Physical indices of control qubits.
            target_indices: Physical indices of target qubits.
            qubit_map: Mutable UUID-to-physical-target map for SSA tracking.
            emitter: The CUDA-Q gate emitter.
            bindings: Parameter bindings.

        Raises:
            EmitError: When an unsupported operation is encountered.
        """
        for op in operations:
            if isinstance(op, ReturnOperation):
                continue
            if isinstance(op, GateOperation):
                gate_target_indices = _resolve_gate_targets(
                    op, qubit_map, target_indices
                )
                self._emit_cudaq_multi_controlled_gate(
                    circuit,
                    op,
                    emitter,
                    control_indices,
                    gate_target_indices,
                    bindings,
                )
                # Propagate SSA: results inherit operand's physical target.
                for i, result in enumerate(op.results):
                    if hasattr(result, "type") and result.type.is_quantum():
                        if i < len(op.operands):
                            operand = op.operands[i]
                            if hasattr(operand, "uuid") and operand.uuid in qubit_map:
                                qubit_map[result.uuid] = qubit_map[operand.uuid]
                continue
            if isinstance(op, ForOperation) and num_controls == 1:
                start = (
                    self._resolver.resolve_int_value(op.operands[0], bindings)
                    if len(op.operands) > 0
                    else 0
                )
                stop = (
                    self._resolver.resolve_int_value(op.operands[1], bindings)
                    if len(op.operands) > 1
                    else 1
                )
                step = (
                    self._resolver.resolve_int_value(op.operands[2], bindings)
                    if len(op.operands) > 2
                    else 1
                )
                if start is not None and stop is not None and step is not None:
                    for i in range(start, stop, step):
                        loop_bindings = bindings.copy()
                        loop_bindings[op.loop_var] = i
                        self._emit_cudaq_controlled_ops(
                            circuit,
                            op.operations,
                            num_controls,
                            control_indices,
                            target_indices,
                            qubit_map,
                            emitter,
                            loop_bindings,
                        )
                else:
                    raise EmitError(
                        "Cannot resolve ForOperation bounds in CUDA-Q "
                        "controlled block body.",
                        operation="ControlledUOperation",
                    )
                continue
            raise EmitError(
                f"Unsupported operation {type(op).__name__} in "
                f"CUDA-Q controlled block body. Only GateOperation "
                f"is supported in helper kernels.",
                operation="ControlledUOperation",
            )

    def _emit_cudaq_multi_controlled_gate(
        self,
        circuit: CudaqCircuit,
        op: GateOperation,
        emitter: CudaqGateEmitter,
        control_indices: list[int],
        target_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit a single multi-controlled gate via CUDA-Q native multi-control.

        CUDA-Q natively supports multi-controlled X via
        ``kernel.cx([controls], target)``.  Other gate types are
        decomposed into multi-controlled X plus single-qubit rotations
        using standard conjugation identities, or raise ``EmitError``
        when no decomposition is available.

        Args:
            circuit: The CUDA-Q circuit being built.
            op: The gate operation to emit with controls.
            emitter: The CUDA-Q gate emitter.
            control_indices: Physical indices of control qubits.
            target_indices: Physical indices of target qubits (resolved
                per-gate by the caller).
            bindings: Parameter bindings.

        Raises:
            EmitError: When the gate type is unsupported for multi-control.
        """
        if not target_indices:
            return

        target_idx = target_indices[0]
        gate_type = op.gate_type

        # Multi-controlled X: native CUDA-Q cx with list of controls
        if gate_type == GateOperationType.X:
            emitter.emit_multi_controlled_x(
                circuit,
                control_indices,
                target_idx,
            )
            return

        # Multi-controlled CX normalization: treat inner control as extra
        # control, inner target as the final X target.
        if gate_type == GateOperationType.CX:
            if len(target_indices) < 2:
                raise EmitError(
                    "Controlled-CX requires at least 2 target qubits "
                    "(inner control + inner target).",
                    operation="ControlledGate",
                )
            inner_control = target_indices[0]
            inner_target = target_indices[1]
            emitter.emit_multi_controlled_x(
                circuit,
                control_indices + [inner_control],
                inner_target,
            )
            return

        # Multi-controlled SWAP (Fredkin):
        #   CNOT(b, a) → MC-X(ctrls + [a], b) → CNOT(b, a)
        if gate_type == GateOperationType.SWAP:
            if len(target_indices) < 2:
                raise EmitError(
                    "Controlled-SWAP requires at least 2 target qubits.",
                    operation="ControlledGate",
                )
            tgt_a = target_indices[0]
            tgt_b = target_indices[1]
            emitter.emit_cx(circuit, tgt_b, tgt_a)
            emitter.emit_multi_controlled_x(
                circuit,
                control_indices + [tgt_a],
                tgt_b,
            )
            emitter.emit_cx(circuit, tgt_b, tgt_a)
            return

        # Single-control: fall back to existing controlled-gate emitters
        if len(control_indices) == 1:
            self._emit_controlled_gate(
                circuit,
                op,
                control_indices[0],
                target_indices,
                bindings,
            )
            return

        raise EmitError(
            f"Unsupported gate type {gate_type!r} in CUDA-Q multi-controlled "
            f"block decomposition. Only X and SWAP are natively supported "
            f"with multiple controls.",
            operation="ControlledGate",
        )


class CudaqExecutor(QuantumExecutor[CudaqCircuit]):
    """CUDA-Q quantum executor.

    Supports sampling via ``cudaq.sample`` and expectation value estimation
    via ``cudaq.observe``.

    Args:
        target: CUDA-Q target name (e.g., ``"qpp-cpu"``). If None, uses
            the default CUDA-Q target.
    """

    def __init__(self, target: str | None = None):
        import cudaq

        self._target = target
        if self._target:
            cudaq.set_target(self._target)

    def _ensure_target(self) -> None:
        """Reapply this executor's target before a runtime call.

        CUDA-Q target selection is process-global.  If another executor (or
        any other code) has called ``cudaq.set_target`` since this instance
        was created, the global target may no longer match ``self._target``.
        Calling this method before every ``cudaq.sample`` / ``cudaq.observe``
        guarantees the correct backend is active.
        """
        if self._target:
            import cudaq

            cudaq.set_target(self._target)

    def execute(self, circuit: Any, shots: int) -> dict[str, int]:
        """Execute circuit and return canonical big-endian bitstring counts.

        CUDA-Q ``sample`` returns bitstrings in allocation order (first
        declared qubit = leftmost bit).  The executor contract requires
        big-endian format (highest qubit index = leftmost bit), so this
        method reverses each bitstring before returning.

        For non-parametric circuits (``CudaqCircuit``), calls
        ``cudaq.sample(kernel, shots_count=shots)``.
        For bound circuits (``BoundCudaqCircuit``), passes parameter values.

        CUDA-Q ``sample`` automatically measures all qubits when no
        explicit ``mz`` calls are present in the kernel.
        """
        import cudaq

        self._ensure_target()

        if isinstance(circuit, BoundCudaqCircuit):
            result = cudaq.sample(
                circuit.kernel, circuit.param_values, shots_count=shots
            )
            num_qubits = circuit.num_qubits
        else:
            result = cudaq.sample(circuit.kernel, shots_count=shots)
            num_qubits = circuit.num_qubits

        counts: dict[str, int] = {}
        for bitstring in result:
            count = result.count(bitstring)
            if len(bitstring) > num_qubits:
                raise ValueError(
                    f"Bitstring '{bitstring}' has length {len(bitstring)} > "
                    f"num_qubits={num_qubits}"
                )
            padded = bitstring.zfill(num_qubits)
            canonical = padded[::-1]  # allocation-order → big-endian
            counts[canonical] = counts.get(canonical, 0) + count

        return counts

    def bind_parameters(
        self,
        circuit: CudaqCircuit,
        bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> BoundCudaqCircuit:
        """Bind parameters by wrapping the kernel with parameter values.

        CUDA-Q does not support in-place parameter binding. Instead,
        parameters are passed at execution time. This method creates a
        ``BoundCudaqCircuit`` that stores the kernel and parameter values
        together.
        """
        param_values = []
        for param_info in parameter_metadata.parameters:
            if param_info.name in bindings:
                param_values.append(float(bindings[param_info.name]))
            else:
                raise ValueError(
                    f"Missing binding for parameter '{param_info.name}'. "
                    f"Provided bindings: {list(bindings.keys())}. "
                    f"Required parameters: "
                    f"{[p.name for p in parameter_metadata.parameters]}"
                )

        return BoundCudaqCircuit(
            kernel=circuit.kernel,
            num_qubits=circuit.num_qubits,
            param_values=param_values,
        )

    def estimate(
        self,
        circuit: Any,
        hamiltonian: "qm_o.Hamiltonian",
        params: Sequence[float] | None = None,
    ) -> float:
        """Estimate expectation value using ``cudaq.observe``.

        Dispatches based on circuit type:
        - ``BoundCudaqCircuit``: uses stored param_values
        - ``CudaqCircuit`` with params: passes params to observe
        - ``CudaqCircuit`` without params: no-parameter observe
        """
        import cudaq
        import qamomile.observable as qm_o

        from qamomile.cudaq.observable import hamiltonian_to_cudaq_spin_op

        self._ensure_target()

        if isinstance(hamiltonian, qm_o.Hamiltonian):
            spin_op = hamiltonian_to_cudaq_spin_op(hamiltonian)
        else:
            spin_op = hamiltonian

        if isinstance(circuit, BoundCudaqCircuit):
            result = cudaq.observe(circuit.kernel, spin_op, circuit.param_values)
        elif isinstance(circuit, CudaqCircuit):
            if params is not None:
                result = cudaq.observe(circuit.kernel, spin_op, list(params))
            else:
                result = cudaq.observe(circuit.kernel, spin_op)
        else:
            raise TypeError(f"Unexpected circuit type: {type(circuit)}")

        return result.expectation()


class CudaqTranspiler(Transpiler[CudaqCircuit]):
    """CUDA-Q transpiler for qamomile.circuit module.

    Converts Qamomile QKernels into CUDA-Q kernels.

    Example:
        from qamomile.cudaq import CudaqTranspiler
        import qamomile.circuit as qm

        @qm.qkernel
        def bell_state(q0: qm.Qubit, q1: qm.Qubit) -> tuple[qm.Bit, qm.Bit]:
            q0 = qm.h(q0)
            q0, q1 = qm.cx(q0, q1)
            return qm.measure(q0), qm.measure(q1)

        transpiler = CudaqTranspiler()
        executable = transpiler.transpile(bell_state)
    """

    def _create_separate_pass(self) -> SeparatePass:
        return SeparatePass()

    def _create_emit_pass(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> EmitPass[CudaqCircuit]:
        return CudaqEmitPass(bindings, parameters)

    def executor(
        self,
        target: str | None = None,
    ) -> CudaqExecutor:
        """Create a CUDA-Q executor.

        Args:
            target: CUDA-Q target name (e.g., ``"qpp-cpu"``).
                If None, uses the default CUDA-Q target.
        """
        return CudaqExecutor(target)
