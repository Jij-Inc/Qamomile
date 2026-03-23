"""CUDA-Q backend transpiler implementation.

This module provides CudaqTranspiler for converting Qamomile QKernels
into CUDA-Q decorator-kernel artifacts, along with CudaqEmitPass and
CudaqExecutor.

All circuits (static and runtime) are emitted through a single
``CudaqKernelEmitter`` codegen path.  The emitter produces
``CudaqKernelArtifact`` instances whose ``execution_mode`` determines
whether ``cudaq.sample()`` / ``cudaq.observe()`` / ``cudaq.get_state()``
(STATIC) or ``cudaq.run()`` (RUNNABLE) is used for execution.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence, TYPE_CHECKING

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    IfOperation,
    WhileOperation,
)
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

from .emitter import (
    CudaqKernelArtifact,
    CudaqKernelEmitter,
    ExecutionMode,
)

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
class BoundCudaqKernelArtifact:
    """CUDA-Q kernel artifact with bound parameter values.

    Used as the return type of ``CudaqExecutor.bind_parameters``.
    The executor dispatches to the appropriate CUDA-Q runtime API
    based on ``execution_mode``.

    Args:
        kernel_func: The ``@cudaq.kernel`` decorated function.
        num_qubits: Number of qubits in the circuit.
        num_clbits: Number of classical bits in the circuit.
        param_values: Bound parameter values in order.
        execution_mode: Execution mode inherited from the source artifact.
    """

    kernel_func: Any
    num_qubits: int
    num_clbits: int
    param_values: list[float]
    execution_mode: ExecutionMode = ExecutionMode.STATIC

    @property
    def kernel(self) -> Any:
        """Backward-compatibility alias for ``kernel_func``."""
        return self.kernel_func


# Backward compatibility aliases
BoundCudaqCircuit = BoundCudaqKernelArtifact
"""Alias for backward compatibility. Use ``BoundCudaqKernelArtifact`` instead."""

BoundCudaqRuntimeCircuit = BoundCudaqKernelArtifact
"""Alias for backward compatibility. Use ``BoundCudaqKernelArtifact`` instead."""


def _has_runtime_control_flow(
    operations: list[Operation],
    bindings: dict[str, Any],
) -> bool:
    """Check whether operations contain runtime measurement-dependent control flow.

    Returns True if any ``IfOperation`` has a condition that cannot be
    resolved at compile time, or if any ``WhileOperation`` is present.
    """
    for op in operations:
        if isinstance(op, IfOperation):
            if resolve_if_condition(op.condition, bindings) is None:
                return True
            # Also check inside branches
            if _has_runtime_control_flow(op.true_operations, bindings):
                return True
            if _has_runtime_control_flow(op.false_operations, bindings):
                return True
        elif isinstance(op, WhileOperation):
            return True
        elif isinstance(op, ForOperation):
            if _has_runtime_control_flow(op.operations, bindings):
                return True
    return False


class CudaqEmitPass(StandardEmitPass[CudaqKernelArtifact]):
    """CUDA-Q-specific emission pass.

    Uses a single ``CudaqKernelEmitter`` for all circuits.  The emitter
    generates ``@cudaq.kernel`` decorated Python source for both static
    and runtime control flow circuits.  The execution mode (STATIC or
    RUNNABLE) is determined by a pre-scan of the operations.
    """

    def __init__(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ):
        parametric = bool(parameters)
        emitter = CudaqKernelEmitter(parametric=parametric)
        composite_emitters: list[Any] = []
        super().__init__(emitter, bindings, parameters, composite_emitters)

    def _emit_quantum_segment(
        self,
        operations: list[Operation],
        bindings: dict[str, Any],
    ) -> tuple[CudaqKernelArtifact, dict[str, int], dict[str, int]]:
        """Emit a quantum segment through the unified codegen path.

        Determines the execution mode via ``_has_runtime_control_flow``,
        configures the emitter accordingly, emits all operations, and
        finalizes the artifact.
        """
        emitter: CudaqKernelEmitter = self._emitter  # type: ignore[assignment]
        is_runtime = _has_runtime_control_flow(operations, bindings)
        mode = ExecutionMode.RUNNABLE if is_runtime else ExecutionMode.STATIC

        # Configure emitter for this segment's mode
        emitter.noop_measurement = mode == ExecutionMode.STATIC

        # Allocate resources
        qubit_map, clbit_map = self._allocator.allocate(operations, bindings)
        qubit_count = max(qubit_map.values()) + 1 if qubit_map else 0
        clbit_count = max(clbit_map.values()) + 1 if clbit_map else 0

        # Create circuit and emit operations
        circuit = emitter.create_circuit(qubit_count, clbit_count)
        self._measurement_qubit_map.clear()
        self._emit_operations(circuit, operations, qubit_map, clbit_map, bindings)

        # For STATIC mode, measurement_qubit_map is populated by base class
        # via noop_measurement flag.  For RUNNABLE mode, copy from emitter.
        if mode == ExecutionMode.RUNNABLE:
            for clbit_idx, qubit_idx in emitter._measurement_map.items():
                self._measurement_qubit_map[clbit_idx] = qubit_idx

        # Finalize: compile source into @cudaq.kernel function
        circuit = emitter.finalize(circuit, mode)

        return circuit, qubit_map, clbit_map

    def _emit_if(
        self,
        circuit: Any,
        op: IfOperation,
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> None:
        """Handle if-operations for both static and runtime paths.

        Compile-time constant conditions are always handled by the
        base class.  Runtime measurement-dependent conditions are
        delegated to ``StandardEmitPass._emit_if`` when the emitter
        is in RUNNABLE mode (``supports_if_else() == True``), or raise
        ``EmitError`` when in STATIC mode.
        """
        condition = op.condition

        # Compile-time constant conditions are handled by the base class.
        if resolve_if_condition(condition, bindings) is not None:
            super()._emit_if(circuit, op, qubit_map, clbit_map, bindings)
            return

        # Runtime condition: delegate to StandardEmitPass if emitter supports it
        if self._emitter.supports_if_else():
            StandardEmitPass._emit_if(self, circuit, op, qubit_map, clbit_map, bindings)
            return

        raise EmitError(
            "CUDA-Q 0.14.x does not support measurement-dependent conditional "
            "branching via the builder API. Use a circuit with runtime control "
            "flow support (automatically handled when runtime if/while is detected)."
        )

    def _emit_controlled_fallback(
        self,
        circuit: CudaqKernelArtifact,
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
            circuit: The CUDA-Q kernel artifact being built.
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

        emitter: CudaqKernelEmitter = self._emitter  # type: ignore[assignment]

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
        circuit: CudaqKernelArtifact,
        operations: list[Any],
        num_controls: int,
        control_indices: list[int],
        target_indices: list[int],
        qubit_map: dict[str, int],
        emitter: CudaqKernelEmitter,
        bindings: dict[str, Any],
    ) -> None:
        """Recursively emit controlled operations with operand-to-target mapping.

        Handles ``GateOperation`` via the CUDA-Q multi-control path,
        ``ReturnOperation`` by skipping, and ``ForOperation`` by unrolling
        compile-time resolvable loops (single-control only).

        Args:
            circuit: The CUDA-Q kernel artifact being built.
            operations: List of operations to process.
            num_controls: Number of control qubits.
            control_indices: Physical indices of control qubits.
            target_indices: Physical indices of target qubits.
            qubit_map: Mutable UUID-to-physical-target map for SSA tracking.
            emitter: The CUDA-Q kernel emitter.
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
        circuit: CudaqKernelArtifact,
        op: GateOperation,
        emitter: CudaqKernelEmitter,
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
            circuit: The CUDA-Q kernel artifact being built.
            op: The gate operation to emit with controls.
            emitter: The CUDA-Q kernel emitter.
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
        #   CNOT(b, a) -> MC-X(ctrls + [a], b) -> CNOT(b, a)
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


class CudaqExecutor(QuantumExecutor[CudaqKernelArtifact]):
    """CUDA-Q quantum executor.

    Supports sampling via ``cudaq.sample`` and expectation value estimation
    via ``cudaq.observe``.  Dispatches to the appropriate CUDA-Q runtime
    API based on the artifact's ``execution_mode``.

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

        Dispatches based on ``execution_mode``:

        - ``STATIC``: uses ``cudaq.sample()`` on the decorator kernel.
        - ``RUNNABLE``: uses ``cudaq.run()`` on the runnable kernel.

        Both paths return bitstrings in big-endian format (highest qubit
        index = leftmost bit).
        """
        mode = getattr(circuit, "execution_mode", ExecutionMode.STATIC)
        if mode == ExecutionMode.RUNNABLE:
            return self._execute_runtime(circuit, shots)
        return self._execute_sample(circuit, shots)

    def _execute_sample(self, circuit: Any, shots: int) -> dict[str, int]:
        """Execute via ``cudaq.sample()`` for STATIC-mode kernels."""
        import cudaq

        self._ensure_target()

        if isinstance(circuit, BoundCudaqKernelArtifact):
            result = cudaq.sample(
                circuit.kernel_func, circuit.param_values, shots_count=shots
            )
        else:
            result = cudaq.sample(circuit.kernel_func, shots_count=shots)

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
            canonical = padded[::-1]  # allocation-order -> big-endian
            counts[canonical] = counts.get(canonical, 0) + count

        return counts

    def _execute_runtime(
        self,
        circuit: CudaqKernelArtifact | BoundCudaqKernelArtifact,
        shots: int,
    ) -> dict[str, int]:
        """Execute via ``cudaq.run()`` for RUNNABLE-mode kernels.

        ``cudaq.run()`` returns a list of per-shot return values.  Each
        return value is ``list[bool]`` (from ``return mz(q)``), with one
        bool per qubit in allocation order.

        This method aggregates per-shot results into canonical big-endian
        bitstring counts matching the ``QuantumExecutor.execute()`` contract.
        """
        import cudaq

        self._ensure_target()

        if isinstance(circuit, BoundCudaqKernelArtifact):
            results = cudaq.run(
                circuit.kernel_func, circuit.param_values, shots_count=shots
            )
        else:
            results = cudaq.run(circuit.kernel_func, shots_count=shots)

        num_qubits = circuit.num_qubits
        counts: dict[str, int] = {}
        for shot_result in results:
            # shot_result is list[bool] in allocation order (from mz(q))
            bits = ["1" if b else "0" for b in shot_result]
            # Pad to num_qubits if shorter
            while len(bits) < num_qubits:
                bits.append("0")
            # Reverse: allocation-order -> big-endian
            canonical = "".join(reversed(bits))
            counts[canonical] = counts.get(canonical, 0) + 1

        return counts

    def bind_parameters(
        self,
        circuit: Any,
        bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> BoundCudaqKernelArtifact:
        """Bind parameters to a circuit for execution.

        Returns a ``BoundCudaqKernelArtifact`` with the execution mode
        inherited from the source artifact.
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

        return BoundCudaqKernelArtifact(
            kernel_func=circuit.kernel_func,
            num_qubits=circuit.num_qubits,
            num_clbits=circuit.num_clbits,
            param_values=param_values,
            execution_mode=getattr(circuit, "execution_mode", ExecutionMode.STATIC),
        )

    def estimate(
        self,
        circuit: Any,
        hamiltonian: "qm_o.Hamiltonian",
        params: Sequence[float] | None = None,
    ) -> float:
        """Estimate expectation value using ``cudaq.observe``.

        Only supported for STATIC-mode artifacts.  RUNNABLE-mode artifacts
        are not compatible with ``cudaq.observe()`` and raise ``TypeError``.
        """
        import cudaq
        import qamomile.observable as qm_o

        from qamomile.cudaq.observable import hamiltonian_to_cudaq_spin_op

        mode = getattr(circuit, "execution_mode", ExecutionMode.STATIC)
        if mode == ExecutionMode.RUNNABLE:
            raise TypeError(
                "cudaq.observe() is not supported for runtime control flow "
                "circuits. Use sample() or run() instead."
            )

        self._ensure_target()

        if isinstance(hamiltonian, qm_o.Hamiltonian):
            spin_op = hamiltonian_to_cudaq_spin_op(hamiltonian)
        else:
            spin_op = hamiltonian

        if isinstance(circuit, BoundCudaqKernelArtifact):
            result = cudaq.observe(circuit.kernel_func, spin_op, circuit.param_values)
        else:
            if params is not None:
                result = cudaq.observe(circuit.kernel_func, spin_op, list(params))
            else:
                result = cudaq.observe(circuit.kernel_func, spin_op)

        return result.expectation()


class CudaqTranspiler(Transpiler[CudaqKernelArtifact]):
    """CUDA-Q transpiler for qamomile.circuit module.

    Converts Qamomile QKernels into CUDA-Q decorator-kernel artifacts.

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
    ) -> EmitPass[CudaqKernelArtifact]:
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
