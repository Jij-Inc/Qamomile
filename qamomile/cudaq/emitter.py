"""CUDA-Q emitter implementation.

This module provides a unified emitter for the CUDA-Q backend:

- ``CudaqKernelEmitter``: Generates ``@cudaq.kernel`` decorated Python source
  code for all circuits.  The emitter supports two execution modes:

  - **STATIC**: Measurement-free kernels compatible with ``cudaq.sample()``,
    ``cudaq.get_state()``, and ``cudaq.observe()``.
  - **RUNNABLE**: Kernels with explicit mid-circuit measurements and a final
    ``return mz(q)``, compatible with ``cudaq.run()``.

The execution mode is determined by the emit pass based on whether the
circuit contains runtime measurement-dependent control flow.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any


class ExecutionMode(enum.Enum):
    """Execution mode for a CUDA-Q kernel artifact.

    Determines which CUDA-Q runtime API is used for execution:

    - ``STATIC``: Compatible with ``cudaq.sample()``, ``cudaq.get_state()``,
      and ``cudaq.observe()``.  The kernel has no explicit terminal
      measurements and no return value.
    - ``RUNNABLE``: Compatible with ``cudaq.run()``.  The kernel has explicit
      mid-circuit measurements and returns ``mz(q)`` (``list[bool]``).
    """

    STATIC = "static"
    RUNNABLE = "runnable"


@dataclasses.dataclass
class CudaqKernelArtifact:
    """Compiled CUDA-Q decorator-kernel artifact.

    Wraps a ``@cudaq.kernel`` decorated function produced by
    ``CudaqKernelEmitter``.  The ``execution_mode`` field determines
    which CUDA-Q runtime APIs are valid for this artifact.

    Args:
        kernel_func: The ``@cudaq.kernel`` decorated function, or None
            before ``finalize()`` is called.
        num_qubits: Number of qubits in the circuit.
        num_clbits: Number of classical bits in the circuit.
        source: Generated Python source code (for debugging).
        execution_mode: Whether this is a STATIC or RUNNABLE kernel.
        param_count: Number of variational parameters.
    """

    kernel_func: Any
    num_qubits: int
    num_clbits: int
    source: str = ""
    execution_mode: ExecutionMode = ExecutionMode.STATIC
    param_count: int = 0

    @property
    def kernel(self) -> Any:
        """Backward-compatibility alias for ``kernel_func``."""
        return self.kernel_func


class CudaqKernelEmitter:
    """Unified GateEmitter that generates ``@cudaq.kernel`` Python source code.

    Used for all CUDA-Q circuits.  Gate calls are accumulated as Python
    source lines forming a ``@cudaq.kernel`` decorated function.  The
    ``finalize()`` method compiles the source via ``exec()`` and populates
    the artifact's ``kernel_func``.

    The ``noop_measurement`` flag controls measurement behaviour:

    - ``True`` (STATIC mode): ``emit_measure`` is a no-op; ``cudaq.sample()``
      auto-measures all qubits.
    - ``False`` (RUNNABLE mode): ``emit_measure`` emits explicit ``mz()``
      calls for mid-circuit measurement variables.

    Args:
        parametric: If True, the generated function accepts a
            ``thetas: list[float]`` parameter for variational circuits.
    """

    def __init__(self, parametric: bool = False) -> None:
        self._parametric = parametric
        self._param_map: dict[str, int] = {}
        self._param_count: int = 0
        self._lines: list[str] = []
        self._indent: int = 1  # Start inside function body
        self._measurement_map: dict[int, int] = {}  # clbit -> qubit
        self._num_clbits: int = 0
        self.noop_measurement: bool = True

    def _emit(self, line: str) -> None:
        """Append an indented source line."""
        self._lines.append("    " * self._indent + line)

    # ------------------------------------------------------------------
    # Circuit lifecycle
    # ------------------------------------------------------------------

    def create_circuit(self, num_qubits: int, num_clbits: int) -> CudaqKernelArtifact:
        """Start building a ``@cudaq.kernel`` function.

        Resets internal state and emits the ``q = cudaq.qvector(N)``
        allocation line.

        Args:
            num_qubits: Number of qubits to allocate.
            num_clbits: Number of classical bits.

        Returns:
            A new ``CudaqKernelArtifact`` with ``kernel_func=None``
            (populated by ``finalize()``).
        """
        self._num_clbits = num_clbits
        self._measurement_map.clear()
        self._param_map.clear()
        self._param_count = 0
        self._lines.clear()
        self._indent = 1

        self._emit(f"q = cudaq.qvector({num_qubits})")
        for i in range(num_clbits):
            self._emit(f"__b{i} = False")

        return CudaqKernelArtifact(
            kernel_func=None,
            num_qubits=num_qubits,
            num_clbits=num_clbits,
        )

    def finalize(
        self, circuit: CudaqKernelArtifact, mode: ExecutionMode
    ) -> CudaqKernelArtifact:
        """Compile accumulated source into a ``@cudaq.kernel`` function.

        For RUNNABLE mode, appends ``return [__b0, __b1, ...]`` to
        return per-shot logical clbit values.  For STATIC mode,
        generates a void function with no explicit terminal measurement.

        Args:
            circuit: The artifact to finalize.
            mode: Execution mode determining the function signature.

        Returns:
            The same artifact with ``kernel_func`` and ``source`` populated.
        """
        import cudaq  # noqa: F811

        if mode == ExecutionMode.RUNNABLE:
            clbit_list = ", ".join(f"__b{i}" for i in range(self._num_clbits))
            self._emit(f"return [{clbit_list}]")
            if self._parametric:
                sig = "def _qamomile_kernel(thetas: list[float]) -> list[bool]:"
            else:
                sig = "def _qamomile_kernel() -> list[bool]:"
        else:
            if self._parametric:
                sig = "def _qamomile_kernel(thetas: list[float]):"
            else:
                sig = "def _qamomile_kernel():"

        body = "\n".join(self._lines)
        source = f"@cudaq.kernel\n{sig}\n{body}\n"

        namespace: dict[str, Any] = {"cudaq": cudaq}
        exec(source, namespace)  # noqa: S102

        circuit.kernel_func = namespace["_qamomile_kernel"]
        circuit.source = source
        circuit.execution_mode = mode
        circuit.param_count = self._param_count
        return circuit

    def create_parameter(self, name: str) -> str:
        """Return a source expression referencing ``thetas[i]``.

        Args:
            name: Symbolic parameter name.

        Returns:
            String expression like ``"thetas[0]"``.
        """
        if name not in self._param_map:
            self._param_map[name] = self._param_count
            self._param_count += 1
        return f"thetas[{self._param_map[name]}]"

    # ------------------------------------------------------------------
    # Single-qubit gates (no parameters)
    # ------------------------------------------------------------------

    def emit_h(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        """Emit Hadamard gate."""
        self._emit(f"h(q[{qubit}])")

    def emit_x(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        """Emit Pauli-X gate."""
        self._emit(f"x(q[{qubit}])")

    def emit_y(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        """Emit Pauli-Y gate."""
        self._emit(f"y(q[{qubit}])")

    def emit_z(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        """Emit Pauli-Z gate."""
        self._emit(f"z(q[{qubit}])")

    def emit_s(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        """Emit S (phase) gate."""
        self._emit(f"s(q[{qubit}])")

    def emit_sdg(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        """Emit S-dagger gate."""
        self._emit(f"sdg(q[{qubit}])")

    def emit_t(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        """Emit T gate."""
        self._emit(f"t(q[{qubit}])")

    def emit_tdg(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        """Emit T-dagger gate."""
        self._emit(f"tdg(q[{qubit}])")

    # ------------------------------------------------------------------
    # Single-qubit rotation gates
    # ------------------------------------------------------------------

    def _angle_expr(self, angle: float | str | Any) -> str:
        """Convert an angle to a source expression."""
        if isinstance(angle, str):
            return angle  # Already a source expression (e.g. "thetas[0]")
        if isinstance(angle, (int, float)):
            return repr(float(angle))
        return str(angle)

    def emit_rx(
        self, circuit: CudaqKernelArtifact, qubit: int, angle: float | Any
    ) -> None:
        """Emit RX rotation gate."""
        self._emit(f"rx({self._angle_expr(angle)}, q[{qubit}])")

    def emit_ry(
        self, circuit: CudaqKernelArtifact, qubit: int, angle: float | Any
    ) -> None:
        """Emit RY rotation gate."""
        self._emit(f"ry({self._angle_expr(angle)}, q[{qubit}])")

    def emit_rz(
        self, circuit: CudaqKernelArtifact, qubit: int, angle: float | Any
    ) -> None:
        """Emit RZ rotation gate."""
        self._emit(f"rz({self._angle_expr(angle)}, q[{qubit}])")

    def emit_p(
        self, circuit: CudaqKernelArtifact, qubit: int, angle: float | Any
    ) -> None:
        """Emit phase gate (R1)."""
        self._emit(f"r1({self._angle_expr(angle)}, q[{qubit}])")

    # ------------------------------------------------------------------
    # Two-qubit gates
    # ------------------------------------------------------------------

    def emit_cx(self, circuit: CudaqKernelArtifact, control: int, target: int) -> None:
        """Emit CNOT (controlled-X) gate."""
        self._emit(f"x.ctrl(q[{control}], q[{target}])")

    def emit_cz(self, circuit: CudaqKernelArtifact, control: int, target: int) -> None:
        """Emit controlled-Z gate."""
        self._emit(f"z.ctrl(q[{control}], q[{target}])")

    def emit_swap(self, circuit: CudaqKernelArtifact, qubit1: int, qubit2: int) -> None:
        """Emit SWAP gate."""
        self._emit(f"swap(q[{qubit1}], q[{qubit2}])")

    # ------------------------------------------------------------------
    # Two-qubit rotation gates
    # ------------------------------------------------------------------

    def emit_cp(
        self,
        circuit: CudaqKernelArtifact,
        control: int,
        target: int,
        angle: float | Any,
    ) -> None:
        """Emit controlled-Phase via decomposition.

        CP(ctrl, tgt, theta) =
            RZ(tgt, theta/2)
            CNOT(ctrl, tgt)
            RZ(tgt, -theta/2)
            CNOT(ctrl, tgt)
            RZ(ctrl, theta/2)
        """
        a = self._angle_expr(angle)
        if isinstance(angle, (int, float)):
            half = repr(angle / 2.0)
            neg_half = repr(-angle / 2.0)
        else:
            half = f"({a}) * 0.5"
            neg_half = f"({a}) * (-0.5)"
        self._emit(f"rz({half}, q[{target}])")
        self._emit(f"x.ctrl(q[{control}], q[{target}])")
        self._emit(f"rz({neg_half}, q[{target}])")
        self._emit(f"x.ctrl(q[{control}], q[{target}])")
        self._emit(f"rz({half}, q[{control}])")

    def emit_rzz(
        self,
        circuit: CudaqKernelArtifact,
        qubit1: int,
        qubit2: int,
        angle: float | Any,
    ) -> None:
        """Emit RZZ via CNOT + RZ decomposition.

        RZZ(q1, q2, theta) =
            CNOT(q1, q2)
            RZ(q2, theta)
            CNOT(q1, q2)
        """
        a = self._angle_expr(angle)
        self._emit(f"x.ctrl(q[{qubit1}], q[{qubit2}])")
        self._emit(f"rz({a}, q[{qubit2}])")
        self._emit(f"x.ctrl(q[{qubit1}], q[{qubit2}])")

    # ------------------------------------------------------------------
    # Three-qubit gates
    # ------------------------------------------------------------------

    def emit_toffoli(
        self,
        circuit: CudaqKernelArtifact,
        control1: int,
        control2: int,
        target: int,
    ) -> None:
        """Emit Toffoli (CCX) gate.

        Delegates to :meth:`emit_multi_controlled_x` so that all
        multi-controlled X emission flows through a single canonical helper.
        """
        self.emit_multi_controlled_x(circuit, [control1, control2], target)

    # ------------------------------------------------------------------
    # Controlled single-qubit gates
    # ------------------------------------------------------------------

    def emit_ch(self, circuit: CudaqKernelArtifact, control: int, target: int) -> None:
        """Emit controlled-Hadamard via decomposition."""
        self._emit(f"ry({math.pi / 4}, q[{target}])")
        self._emit(f"x.ctrl(q[{control}], q[{target}])")
        self._emit(f"ry({-math.pi / 4}, q[{target}])")

    def emit_cy(self, circuit: CudaqKernelArtifact, control: int, target: int) -> None:
        """Emit controlled-Y via decomposition."""
        self._emit(f"sdg(q[{target}])")
        self._emit(f"x.ctrl(q[{control}], q[{target}])")
        self._emit(f"s(q[{target}])")

    def emit_crx(
        self,
        circuit: CudaqKernelArtifact,
        control: int,
        target: int,
        angle: float | Any,
    ) -> None:
        """Emit controlled-RX gate."""
        a = self._angle_expr(angle)
        self._emit(f"rx.ctrl({a}, q[{control}], q[{target}])")

    def emit_cry(
        self,
        circuit: CudaqKernelArtifact,
        control: int,
        target: int,
        angle: float | Any,
    ) -> None:
        """Emit controlled-RY gate."""
        a = self._angle_expr(angle)
        self._emit(f"ry.ctrl({a}, q[{control}], q[{target}])")

    def emit_crz(
        self,
        circuit: CudaqKernelArtifact,
        control: int,
        target: int,
        angle: float | Any,
    ) -> None:
        """Emit controlled-RZ gate."""
        a = self._angle_expr(angle)
        self._emit(f"rz.ctrl({a}, q[{control}], q[{target}])")

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def emit_measure(
        self, circuit: CudaqKernelArtifact, qubit: int, clbit: int
    ) -> None:
        """Emit measurement, respecting the current mode.

        In STATIC mode (``noop_measurement=True``), this is a no-op:
        ``cudaq.sample()`` auto-measures all qubits.

        In RUNNABLE mode (``noop_measurement=False``), emits explicit
        ``mz()`` to capture mid-circuit measurement results as local
        variables for ``if`` / ``while`` conditions.
        """
        if self.noop_measurement:
            return
        self._emit(f"__b{clbit} = mz(q[{qubit}])")
        self._measurement_map[clbit] = qubit

    # ------------------------------------------------------------------
    # Barrier
    # ------------------------------------------------------------------

    def emit_barrier(self, circuit: CudaqKernelArtifact, qubits: list[int]) -> None:
        """No-op: CUDA-Q does not support barrier instructions."""
        pass

    # ------------------------------------------------------------------
    # Sub-circuit support (not applicable for codegen)
    # ------------------------------------------------------------------

    def circuit_to_gate(self, circuit: CudaqKernelArtifact, name: str = "U") -> Any:
        """No-op: not supported by CUDA-Q codegen path."""
        return None

    def append_gate(
        self, circuit: CudaqKernelArtifact, gate: Any, qubits: list[int]
    ) -> None:
        """No-op: not supported by CUDA-Q codegen path."""
        pass

    def gate_power(self, gate: Any, power: int) -> Any:
        """No-op: not supported by CUDA-Q codegen path."""
        return None

    def gate_controlled(self, gate: Any, num_controls: int) -> Any:
        """No-op: not supported by CUDA-Q codegen path."""
        return None

    # ------------------------------------------------------------------
    # Multi-controlled gate emission
    # ------------------------------------------------------------------

    def emit_multi_controlled_x(
        self,
        circuit: CudaqKernelArtifact,
        control_indices: list[int],
        target_idx: int,
    ) -> None:
        """Emit multi-controlled X using ``x.ctrl(c0, c1, ..., target)``."""
        args = ", ".join(f"q[{i}]" for i in control_indices)
        self._emit(f"x.ctrl({args}, q[{target_idx}])")

    # ------------------------------------------------------------------
    # Control flow support
    # ------------------------------------------------------------------

    def supports_for_loop(self) -> bool:
        """Return False: for-loops are unrolled by the transpiler."""
        return False

    def supports_if_else(self) -> bool:
        """Return True only in RUNNABLE mode (measurement-dependent branching)."""
        return not self.noop_measurement

    def supports_while_loop(self) -> bool:
        """Return True only in RUNNABLE mode."""
        return not self.noop_measurement

    def emit_for_loop_start(self, circuit: CudaqKernelArtifact, indexset: range) -> Any:
        """Not supported: for-loops are unrolled by the transpiler."""
        raise NotImplementedError

    def emit_for_loop_end(self, circuit: CudaqKernelArtifact, context: Any) -> None:
        """Not supported: for-loops are unrolled by the transpiler."""
        raise NotImplementedError

    def emit_if_start(
        self, circuit: CudaqKernelArtifact, clbit: int, value: int = 1
    ) -> dict[str, Any]:
        """Emit ``if __b{clbit}:`` and increase indentation."""
        self._emit(f"if __b{clbit}:")
        self._indent += 1
        return {"clbit": clbit}

    def emit_else_start(
        self, circuit: CudaqKernelArtifact, context: dict[str, Any]
    ) -> None:
        """Emit ``else:`` block."""
        self._indent -= 1
        self._emit("else:")
        self._indent += 1

    def emit_if_end(
        self, circuit: CudaqKernelArtifact, context: dict[str, Any]
    ) -> None:
        """Close ``if`` block by decreasing indentation."""
        self._indent -= 1

    def emit_while_start(
        self, circuit: CudaqKernelArtifact, clbit: int, value: int = 1
    ) -> dict[str, Any]:
        """Emit ``while __b{clbit}:`` and increase indentation."""
        self._emit(f"while __b{clbit}:")
        self._indent += 1
        return {"clbit": clbit}

    def emit_while_end(
        self, circuit: CudaqKernelArtifact, context: dict[str, Any]
    ) -> None:
        """Close ``while`` block by decreasing indentation."""
        self._indent -= 1


# ------------------------------------------------------------------
# Backward compatibility aliases
# ------------------------------------------------------------------

CudaqCircuit = CudaqKernelArtifact
"""Alias for backward compatibility. Use ``CudaqKernelArtifact`` instead."""

CudaqRuntimeCircuit = CudaqKernelArtifact
"""Alias for backward compatibility. Use ``CudaqKernelArtifact`` instead."""

CudaqGateEmitter = CudaqKernelEmitter
"""Alias for backward compatibility. Use ``CudaqKernelEmitter`` instead."""

CudaqCodegenEmitter = CudaqKernelEmitter
"""Alias for backward compatibility. Use ``CudaqKernelEmitter`` instead."""
