"""CUDA-Q GateEmitter implementation.

This module provides CudaqGateEmitter, which implements the GateEmitter
protocol for CUDA-Q backends using the kernel builder API.

CUDA-Q uses a kernel builder pattern where gates are applied to qubit
references obtained from ``kernel.qalloc(n)``. Parametric circuits use
``cudaq.make_kernel(list)`` to obtain a vector parameter.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any


@dataclasses.dataclass
class CudaqCircuit:
    """Wrapper around CUDA-Q kernel with qubit register.

    This serves as the circuit type ``T`` for the CUDA-Q backend.

    Args:
        kernel (Any): The CUDA-Q kernel builder instance.
        qubits (Any): Qubit register allocated via ``kernel.qalloc(n)``.
        num_qubits (int): Number of qubits in the circuit.
        num_clbits (int): Number of classical bits in the circuit.
        param_vector (Any | None): Vector parameter from
            ``cudaq.make_kernel(list)``, or None for non-parametric kernels.
        param_count (int): Number of parameters created so far.
    """

    kernel: Any
    qubits: Any
    num_qubits: int
    num_clbits: int
    param_vector: Any = None
    param_count: int = 0


class CudaqGateEmitter:
    """GateEmitter implementation for CUDA-Q.

    Emits individual quantum gates to CUDA-Q kernels via the builder API.

    Args:
        parametric (bool): If True, creates a parametric kernel using
            ``cudaq.make_kernel(list)`` so that symbolic parameters
            can be indexed from the vector.
    """

    noop_measurement: bool = True

    def __init__(self, parametric: bool = False) -> None:
        self._parametric = parametric
        self._param_map: dict[str, int] = {}
        self._current_circuit: CudaqCircuit | None = None

    def create_circuit(self, num_qubits: int, num_clbits: int) -> CudaqCircuit:
        """Create a new CUDA-Q kernel with allocated qubits."""
        import cudaq

        if self._parametric:
            kernel, thetas = cudaq.make_kernel(list)
            qubits = kernel.qalloc(num_qubits)
            circuit = CudaqCircuit(
                kernel=kernel,
                qubits=qubits,
                num_qubits=num_qubits,
                num_clbits=num_clbits,
                param_vector=thetas,
                param_count=0,
            )
        else:
            kernel = cudaq.make_kernel()
            qubits = kernel.qalloc(num_qubits)
            circuit = CudaqCircuit(
                kernel=kernel,
                qubits=qubits,
                num_qubits=num_qubits,
                num_clbits=num_clbits,
            )

        self._current_circuit = circuit
        self._param_map = {}
        return circuit

    def create_parameter(self, name: str) -> Any:
        """Create a parameter by indexing into the kernel's vector parameter."""
        if self._current_circuit is None:
            raise RuntimeError("create_circuit must be called before create_parameter")

        circuit = self._current_circuit
        if circuit.param_vector is None:
            raise RuntimeError(
                "Cannot create parameters on a non-parametric kernel. "
                "Ensure CudaqGateEmitter is initialized with parametric=True."
            )

        if name not in self._param_map:
            idx = circuit.param_count
            self._param_map[name] = idx
            circuit.param_count += 1

        return circuit.param_vector[self._param_map[name]]

    # ------------------------------------------------------------------
    # Single-qubit gates (no parameters)
    # ------------------------------------------------------------------

    def emit_h(self, circuit: CudaqCircuit, qubit: int) -> None:
        """Emit Hadamard gate on the specified qubit."""
        circuit.kernel.h(circuit.qubits[qubit])

    def emit_x(self, circuit: CudaqCircuit, qubit: int) -> None:
        """Emit Pauli-X gate on the specified qubit."""
        circuit.kernel.x(circuit.qubits[qubit])

    def emit_y(self, circuit: CudaqCircuit, qubit: int) -> None:
        """Emit Pauli-Y gate on the specified qubit."""
        circuit.kernel.y(circuit.qubits[qubit])

    def emit_z(self, circuit: CudaqCircuit, qubit: int) -> None:
        """Emit Pauli-Z gate on the specified qubit."""
        circuit.kernel.z(circuit.qubits[qubit])

    def emit_s(self, circuit: CudaqCircuit, qubit: int) -> None:
        """Emit S (phase) gate on the specified qubit."""
        circuit.kernel.s(circuit.qubits[qubit])

    def emit_sdg(self, circuit: CudaqCircuit, qubit: int) -> None:
        """Emit S-dagger gate on the specified qubit."""
        circuit.kernel.sdg(circuit.qubits[qubit])

    def emit_t(self, circuit: CudaqCircuit, qubit: int) -> None:
        """Emit T gate on the specified qubit."""
        circuit.kernel.t(circuit.qubits[qubit])

    def emit_tdg(self, circuit: CudaqCircuit, qubit: int) -> None:
        """Emit T-dagger gate on the specified qubit."""
        circuit.kernel.tdg(circuit.qubits[qubit])

    # ------------------------------------------------------------------
    # Single-qubit rotation gates
    # ------------------------------------------------------------------

    def emit_rx(self, circuit: CudaqCircuit, qubit: int, angle: float | Any) -> None:
        """Emit RX rotation gate on the specified qubit."""
        if isinstance(angle, (int, float)):
            angle = float(angle)
        circuit.kernel.rx(angle, circuit.qubits[qubit])

    def emit_ry(self, circuit: CudaqCircuit, qubit: int, angle: float | Any) -> None:
        """Emit RY rotation gate on the specified qubit."""
        if isinstance(angle, (int, float)):
            angle = float(angle)
        circuit.kernel.ry(angle, circuit.qubits[qubit])

    def emit_rz(self, circuit: CudaqCircuit, qubit: int, angle: float | Any) -> None:
        """Emit RZ rotation gate on the specified qubit."""
        if isinstance(angle, (int, float)):
            angle = float(angle)
        circuit.kernel.rz(angle, circuit.qubits[qubit])

    def emit_p(self, circuit: CudaqCircuit, qubit: int, angle: float | Any) -> None:
        """Emit phase gate (R1) on the specified qubit."""
        if isinstance(angle, (int, float)):
            angle = float(angle)
        circuit.kernel.r1(angle, circuit.qubits[qubit])

    # ------------------------------------------------------------------
    # Two-qubit gates
    # ------------------------------------------------------------------

    def emit_cx(self, circuit: CudaqCircuit, control: int, target: int) -> None:
        """Emit CNOT (controlled-X) gate."""
        circuit.kernel.cx(circuit.qubits[control], circuit.qubits[target])

    def emit_cz(self, circuit: CudaqCircuit, control: int, target: int) -> None:
        """Emit controlled-Z gate."""
        circuit.kernel.cz(circuit.qubits[control], circuit.qubits[target])

    def emit_swap(self, circuit: CudaqCircuit, qubit1: int, qubit2: int) -> None:
        """Emit SWAP gate between two qubits."""
        circuit.kernel.swap(circuit.qubits[qubit1], circuit.qubits[qubit2])

    # ------------------------------------------------------------------
    # Two-qubit rotation gates
    # ------------------------------------------------------------------

    def emit_cp(
        self, circuit: CudaqCircuit, control: int, target: int, angle: float | Any
    ) -> None:
        """Emit controlled-Phase gate using decomposition.

        CP(ctrl, tgt, theta) =
            RZ(tgt, theta/2)
            CNOT(ctrl, tgt)
            RZ(tgt, -theta/2)
            CNOT(ctrl, tgt)
            RZ(ctrl, theta/2)
        """
        k = circuit.kernel
        q = circuit.qubits
        if isinstance(angle, (int, float)):
            half = angle / 2.0
            k.rz(half, q[target])
            k.cx(q[control], q[target])
            k.rz(-half, q[target])
            k.cx(q[control], q[target])
            k.rz(half, q[control])
        else:
            half = angle * 0.5
            neg_half = angle * (-0.5)
            k.rz(half, q[target])
            k.cx(q[control], q[target])
            k.rz(neg_half, q[target])
            k.cx(q[control], q[target])
            k.rz(half, q[control])

    def emit_rzz(
        self, circuit: CudaqCircuit, qubit1: int, qubit2: int, angle: float | Any
    ) -> None:
        """Emit RZZ gate using CNOT + RZ decomposition.

        RZZ(q1, q2, theta) =
            CNOT(q1, q2)
            RZ(q2, theta)
            CNOT(q1, q2)
        """
        if isinstance(angle, (int, float)):
            angle = float(angle)
        k = circuit.kernel
        q = circuit.qubits
        k.cx(q[qubit1], q[qubit2])
        k.rz(angle, q[qubit2])
        k.cx(q[qubit1], q[qubit2])

    # ------------------------------------------------------------------
    # Three-qubit gates
    # ------------------------------------------------------------------

    def emit_toffoli(
        self, circuit: CudaqCircuit, control1: int, control2: int, target: int
    ) -> None:
        """Emit Toffoli (CCX) gate via multi-control CX."""
        q = circuit.qubits
        circuit.kernel.cx([q[control1], q[control2]], q[target])

    # ------------------------------------------------------------------
    # Controlled single-qubit gates
    # ------------------------------------------------------------------

    def emit_ch(self, circuit: CudaqCircuit, control: int, target: int) -> None:
        """Emit controlled-Hadamard using decomposition.

        CH(ctrl, tgt) =
            RY(tgt, pi/4)
            CNOT(ctrl, tgt)
            RY(tgt, -pi/4)
        """
        k = circuit.kernel
        q = circuit.qubits
        k.ry(math.pi / 4, q[target])
        k.cx(q[control], q[target])
        k.ry(-math.pi / 4, q[target])

    def emit_cy(self, circuit: CudaqCircuit, control: int, target: int) -> None:
        """Emit controlled-Y using decomposition.

        CY(ctrl, tgt) =
            Sdg(tgt)
            CNOT(ctrl, tgt)
            S(tgt)
        """
        k = circuit.kernel
        q = circuit.qubits
        k.sdg(q[target])
        k.cx(q[control], q[target])
        k.s(q[target])

    def emit_crx(
        self, circuit: CudaqCircuit, control: int, target: int, angle: float | Any
    ) -> None:
        """Emit controlled-RX gate (native CUDA-Q)."""
        if isinstance(angle, (int, float)):
            angle = float(angle)
        circuit.kernel.crx(angle, circuit.qubits[control], circuit.qubits[target])

    def emit_cry(
        self, circuit: CudaqCircuit, control: int, target: int, angle: float | Any
    ) -> None:
        """Emit controlled-RY gate (native CUDA-Q)."""
        if isinstance(angle, (int, float)):
            angle = float(angle)
        circuit.kernel.cry(angle, circuit.qubits[control], circuit.qubits[target])

    def emit_crz(
        self, circuit: CudaqCircuit, control: int, target: int, angle: float | Any
    ) -> None:
        """Emit controlled-RZ gate (native CUDA-Q)."""
        if isinstance(angle, (int, float)):
            angle = float(angle)
        circuit.kernel.crz(angle, circuit.qubits[control], circuit.qubits[target])

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def emit_measure(self, circuit: CudaqCircuit, qubit: int, clbit: int) -> None:
        """No-op: ``cudaq.sample()`` auto-measures all qubits.

        Explicit ``mz()`` calls are not emitted during measurement processing.
        ``cudaq.sample()`` automatically measures all qubits when no explicit
        ``mz()`` calls are present, keeping the state pure for
        ``cudaq.get_state()`` in statevector tests.
        """
        pass

    # ------------------------------------------------------------------
    # Barrier
    # ------------------------------------------------------------------

    def emit_barrier(self, circuit: CudaqCircuit, qubits: list[int]) -> None:
        """No-op: CUDA-Q does not support barrier instructions."""
        pass

    # ------------------------------------------------------------------
    # Sub-circuit support
    # ------------------------------------------------------------------

    def circuit_to_gate(self, circuit: CudaqCircuit, name: str = "U") -> Any:
        """No-op: CUDA-Q kernel builder does not support circuit-to-gate conversion."""
        return None

    def append_gate(self, circuit: CudaqCircuit, gate: Any, qubits: list[int]) -> None:
        """No-op: not supported by CUDA-Q kernel builder."""
        pass

    def gate_power(self, gate: Any, power: int) -> Any:
        """No-op: not supported by CUDA-Q kernel builder."""
        return None

    def gate_controlled(self, gate: Any, num_controls: int) -> Any:
        """No-op: not supported by CUDA-Q kernel builder."""
        return None

    # ------------------------------------------------------------------
    # Multi-controlled gate emission
    # ------------------------------------------------------------------

    def emit_multi_controlled_x(
        self,
        circuit: CudaqCircuit,
        control_indices: list[int],
        target_idx: int,
    ) -> None:
        """Emit a multi-controlled X gate using CUDA-Q's native ``cx``.

        CUDA-Q supports ``kernel.cx([controls], target)`` for
        multi-controlled CNOT, which generalises to N-controlled X.

        Args:
            circuit: The CUDA-Q circuit being built.
            control_indices: Physical indices of control qubits.
            target_idx: Physical index of target qubit.
        """
        k = circuit.kernel
        q = circuit.qubits
        controls = [q[i] for i in control_indices]
        k.cx(controls, q[target_idx])

    # ------------------------------------------------------------------
    # Control flow support
    # ------------------------------------------------------------------

    def supports_for_loop(self) -> bool:
        """Return False: CUDA-Q does not support native for-loop emission."""
        return False

    def supports_if_else(self) -> bool:
        """Return False: conditional branching on measurement results is not supported.

        CUDA-Q 0.14.x removed the builder ``c_if`` API.  The generic
        ``emit_if_start`` / ``emit_else_start`` / ``emit_if_end``
        protocol is not implemented and measurement-dependent ``IfOperation``
        is rejected at emit time by ``CudaqEmitPass._emit_if``.
        """
        return False

    def supports_while_loop(self) -> bool:
        """Return False: CUDA-Q does not support native while-loop emission."""
        return False

    def emit_for_loop_start(self, circuit: CudaqCircuit, indexset: range) -> Any:
        """Not supported: for-loops are unrolled by the transpiler."""
        raise NotImplementedError

    def emit_for_loop_end(self, circuit: CudaqCircuit, context: Any) -> None:
        """Not supported: for-loops are unrolled by the transpiler."""
        raise NotImplementedError

    def emit_if_start(self, circuit: CudaqCircuit, clbit: int, value: int = 1) -> Any:
        """Not used: ``CudaqEmitPass._emit_if`` handles c_if directly."""
        raise NotImplementedError("Use CudaqEmitPass._emit_if instead")

    def emit_else_start(self, circuit: CudaqCircuit, context: Any) -> None:
        """Not used: ``CudaqEmitPass._emit_if`` handles c_if directly."""
        raise NotImplementedError("Use CudaqEmitPass._emit_if instead")

    def emit_if_end(self, circuit: CudaqCircuit, context: Any) -> None:
        """Not used: ``CudaqEmitPass._emit_if`` handles c_if directly."""
        raise NotImplementedError("Use CudaqEmitPass._emit_if instead")

    def emit_while_start(
        self, circuit: CudaqCircuit, clbit: int, value: int = 1
    ) -> Any:
        """Not supported: while-loop raises ``EmitError`` in the transpiler."""
        raise NotImplementedError

    def emit_while_end(self, circuit: CudaqCircuit, context: Any) -> None:
        """Not supported: while-loop raises ``EmitError`` in the transpiler."""
        raise NotImplementedError
