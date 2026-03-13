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
    """

    kernel: Any
    qubits: Any
    num_qubits: int
    num_clbits: int
    _param_vector: Any = None
    _param_count: int = 0


class CudaqGateEmitter:
    """GateEmitter implementation for CUDA-Q.

    Emits individual quantum gates to CUDA-Q kernels via the builder API.

    Args:
        parametric: If True, creates a parametric kernel using
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
                _param_vector=thetas,
                _param_count=0,
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
        self._param_map.clear()
        return circuit

    def create_parameter(self, name: str) -> Any:
        """Create a parameter by indexing into the kernel's vector parameter."""
        if self._current_circuit is None:
            raise RuntimeError("create_circuit must be called before create_parameter")

        circuit = self._current_circuit
        if circuit._param_vector is None:
            raise RuntimeError(
                "Cannot create parameters on a non-parametric kernel. "
                "Ensure CudaqGateEmitter is initialized with parametric=True."
            )

        if name not in self._param_map:
            idx = circuit._param_count
            self._param_map[name] = idx
            circuit._param_count += 1

        return circuit._param_vector[self._param_map[name]]

    # ------------------------------------------------------------------
    # Single-qubit gates (no parameters)
    # ------------------------------------------------------------------

    def emit_h(self, circuit: CudaqCircuit, qubit: int) -> None:
        circuit.kernel.h(circuit.qubits[qubit])

    def emit_x(self, circuit: CudaqCircuit, qubit: int) -> None:
        circuit.kernel.x(circuit.qubits[qubit])

    def emit_y(self, circuit: CudaqCircuit, qubit: int) -> None:
        circuit.kernel.y(circuit.qubits[qubit])

    def emit_z(self, circuit: CudaqCircuit, qubit: int) -> None:
        circuit.kernel.z(circuit.qubits[qubit])

    def emit_s(self, circuit: CudaqCircuit, qubit: int) -> None:
        circuit.kernel.s(circuit.qubits[qubit])

    def emit_sdg(self, circuit: CudaqCircuit, qubit: int) -> None:
        circuit.kernel.sdg(circuit.qubits[qubit])

    def emit_t(self, circuit: CudaqCircuit, qubit: int) -> None:
        circuit.kernel.t(circuit.qubits[qubit])

    def emit_tdg(self, circuit: CudaqCircuit, qubit: int) -> None:
        circuit.kernel.tdg(circuit.qubits[qubit])

    # ------------------------------------------------------------------
    # Single-qubit rotation gates
    # ------------------------------------------------------------------

    def emit_rx(self, circuit: CudaqCircuit, qubit: int, angle: float | Any) -> None:
        circuit.kernel.rx(angle, circuit.qubits[qubit])

    def emit_ry(self, circuit: CudaqCircuit, qubit: int, angle: float | Any) -> None:
        circuit.kernel.ry(angle, circuit.qubits[qubit])

    def emit_rz(self, circuit: CudaqCircuit, qubit: int, angle: float | Any) -> None:
        circuit.kernel.rz(angle, circuit.qubits[qubit])

    def emit_p(self, circuit: CudaqCircuit, qubit: int, angle: float | Any) -> None:
        circuit.kernel.r1(angle, circuit.qubits[qubit])

    # ------------------------------------------------------------------
    # Two-qubit gates
    # ------------------------------------------------------------------

    def emit_cx(self, circuit: CudaqCircuit, control: int, target: int) -> None:
        circuit.kernel.cx(circuit.qubits[control], circuit.qubits[target])

    def emit_cz(self, circuit: CudaqCircuit, control: int, target: int) -> None:
        circuit.kernel.cz(circuit.qubits[control], circuit.qubits[target])

    def emit_swap(self, circuit: CudaqCircuit, qubit1: int, qubit2: int) -> None:
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
        circuit.kernel.crx(angle, circuit.qubits[control], circuit.qubits[target])

    def emit_cry(
        self, circuit: CudaqCircuit, control: int, target: int, angle: float | Any
    ) -> None:
        """Emit controlled-RY gate (native CUDA-Q)."""
        circuit.kernel.cry(angle, circuit.qubits[control], circuit.qubits[target])

    def emit_crz(
        self, circuit: CudaqCircuit, control: int, target: int, angle: float | Any
    ) -> None:
        """Emit controlled-RZ gate (native CUDA-Q)."""
        circuit.kernel.crz(angle, circuit.qubits[control], circuit.qubits[target])

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def emit_measure(self, circuit: CudaqCircuit, qubit: int, clbit: int) -> None:
        """No-op: measurements are handled by the executor."""
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
        """CUDA-Q kernel builder does not support circuit-to-gate conversion."""
        return None

    def append_gate(self, circuit: CudaqCircuit, gate: Any, qubits: list[int]) -> None:
        pass

    def gate_power(self, gate: Any, power: int) -> Any:
        return None

    def gate_controlled(self, gate: Any, num_controls: int) -> Any:
        return None

    # ------------------------------------------------------------------
    # Control flow support
    # ------------------------------------------------------------------

    def supports_for_loop(self) -> bool:
        return False

    def supports_if_else(self) -> bool:
        return False

    def supports_while_loop(self) -> bool:
        return False

    def emit_for_loop_start(self, circuit: CudaqCircuit, indexset: range) -> Any:
        raise NotImplementedError

    def emit_for_loop_end(self, circuit: CudaqCircuit, context: Any) -> None:
        raise NotImplementedError

    def emit_if_start(self, circuit: CudaqCircuit, clbit: int, value: int = 1) -> Any:
        raise NotImplementedError

    def emit_else_start(self, circuit: CudaqCircuit, context: Any) -> None:
        raise NotImplementedError

    def emit_if_end(self, circuit: CudaqCircuit, context: Any) -> None:
        raise NotImplementedError

    def emit_while_start(
        self, circuit: CudaqCircuit, clbit: int, value: int = 1
    ) -> Any:
        raise NotImplementedError

    def emit_while_end(self, circuit: CudaqCircuit, context: Any) -> None:
        raise NotImplementedError
