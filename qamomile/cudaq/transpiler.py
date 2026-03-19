"""CUDA-Q backend transpiler implementation.

This module provides CudaqTranspiler for converting Qamomile QKernels
into CUDA-Q kernels, along with CudaqEmitPass and CudaqExecutor.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence, TYPE_CHECKING

from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType
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
        gate with native multi-control, falling back to the base-class
        single-control decomposition when possible.

        Args:
            circuit: The CUDA-Q circuit being built.
            block_value: The block value containing operations to control.
            num_controls: Number of control qubits.
            control_indices: Physical indices of control qubits.
            target_indices: Physical indices of target qubits.
            power: Number of times to repeat the controlled operation.
            bindings: Parameter bindings.

        Raises:
            EmitError: When the block body contains unsupported operations.
        """
        if not hasattr(block_value, "operations"):
            raise EmitError(
                "Cannot emit controlled operation: block has no operations.",
                operation="ControlledUOperation",
            )

        emitter: CudaqGateEmitter = self._emitter  # type: ignore[assignment]

        for _ in range(power):
            for op in block_value.operations:
                if not isinstance(op, GateOperation):
                    continue
                self._emit_cudaq_multi_controlled_gate(
                    circuit,
                    op,
                    emitter,
                    control_indices,
                    target_indices,
                    bindings,
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
            target_indices: Physical indices of target qubits.
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
