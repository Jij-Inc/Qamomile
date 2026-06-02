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
import itertools
import linecache
import math
from collections.abc import Callable
from typing import Any

from qamomile.circuit.transpiler.gate_emitter import MeasurementMode


def _to_expr(value: Any) -> str:
    """Convert a value to a CUDA-Q source expression string.

    Args:
        value: A ``CudaqExpr``, int, or float.

    Returns:
        Source expression string suitable for embedding in codegen output.
    """
    if isinstance(value, CudaqExpr):
        return value._expr
    return repr(value)


class CudaqExpr:
    """Arithmetic-capable wrapper for CUDA-Q source expressions.

    Enables operator-overload-based expression building in
    ``StandardEmitPass._evaluate_binop()``.  Each arithmetic operation
    returns a new ``CudaqExpr`` with a properly parenthesized string
    representation.  ``str(expr)`` yields the raw source expression
    suitable for embedding directly in generated ``@cudaq.kernel`` code.

    Args:
        expr: Source expression string (e.g. ``"thetas[0]"``).
    """

    def __init__(self, expr: str) -> None:
        self._expr = expr

    def __str__(self) -> str:
        return self._expr

    def __repr__(self) -> str:
        return f"CudaqExpr({self._expr!r})"

    # Forward arithmetic
    def __add__(self, other: Any) -> "CudaqExpr":
        return CudaqExpr(f"({self._expr}) + ({_to_expr(other)})")

    def __sub__(self, other: Any) -> "CudaqExpr":
        return CudaqExpr(f"({self._expr}) - ({_to_expr(other)})")

    def __mul__(self, other: Any) -> "CudaqExpr":
        return CudaqExpr(f"({self._expr}) * ({_to_expr(other)})")

    def __truediv__(self, other: Any) -> "CudaqExpr":
        return CudaqExpr(f"({self._expr}) / ({_to_expr(other)})")

    def __pow__(self, other: Any) -> "CudaqExpr":
        return CudaqExpr(f"({self._expr}) ** ({_to_expr(other)})")

    # Reflected arithmetic (concrete op parameter, e.g. 2.0 * theta)
    def __radd__(self, other: Any) -> "CudaqExpr":
        return CudaqExpr(f"({_to_expr(other)}) + ({self._expr})")

    def __rsub__(self, other: Any) -> "CudaqExpr":
        return CudaqExpr(f"({_to_expr(other)}) - ({self._expr})")

    def __rmul__(self, other: Any) -> "CudaqExpr":
        return CudaqExpr(f"({_to_expr(other)}) * ({self._expr})")

    def __rtruediv__(self, other: Any) -> "CudaqExpr":
        return CudaqExpr(f"({_to_expr(other)}) / ({self._expr})")

    def __rpow__(self, other: Any) -> "CudaqExpr":
        return CudaqExpr(f"({_to_expr(other)}) ** ({self._expr})")

    def __neg__(self) -> "CudaqExpr":
        return CudaqExpr(f"-({self._expr})")


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
        entry_source: Generated Python source for the entry-point kernel
            only.  ``source`` may also include helper kernels emitted
            for ``cudaq.control``.
        execution_mode: Whether this is a STATIC or RUNNABLE kernel.
        param_count: Number of variational parameters.
    """

    kernel_func: Any
    num_qubits: int
    num_clbits: int
    source: str = ""
    entry_source: str = ""
    execution_mode: ExecutionMode = ExecutionMode.STATIC
    param_count: int = 0


class CudaqKernelEmitter:
    """Unified GateEmitter that generates ``@cudaq.kernel`` Python source code.

    Used for all CUDA-Q circuits.  Gate calls are accumulated as Python
    source lines forming a ``@cudaq.kernel`` decorated function.  The
    ``finalize()`` method compiles the source via ``exec()`` and populates
    the artifact's ``kernel_func``.

    The ``measurement_mode`` property controls measurement behaviour:

    - ``MeasurementMode.STATIC``: ``emit_measure`` is a no-op;
      ``cudaq.sample()`` auto-measures all qubits.
    - ``MeasurementMode.RUNNABLE``: ``emit_measure`` emits explicit ``mz()``
      calls for mid-circuit measurement variables.

    Note:
        The ``_parametric`` flag is late-bound by ``CudaqEmitPass``: after
        all operations have been emitted, the flag is updated to
        ``_param_count > 0`` so that the kernel signature reflects the
        actual surviving backend parameters, not the originally requested
        parameter list.

    Args:
        parametric: Initial hint for parametricity.  The emit pass
            overrides this after emission based on ``_param_count``.
    """

    _kernel_counter = itertools.count()

    def __init__(self, parametric: bool = False) -> None:
        self._parametric = parametric
        self._param_map: dict[str, int] = {}
        self._param_count: int = 0
        self._lines: list[str] = []
        self._indent: int = 1  # Start inside function body
        self._measurement_map: dict[int, int] = {}  # clbit -> qubit
        self._num_clbits: int = 0
        self._measurement_mode: MeasurementMode = MeasurementMode.STATIC
        self._boxed_clbits: set[int] = set()
        self._helper_sources: list[str] = []
        self._helper_cache: dict[tuple[int, bool, tuple[str, ...]], str] = {}
        self._helper_counter = itertools.count()
        self._qubit_refs: dict[int, str] = {}
        self._building_helper: bool = False
        self._helper_param_used: bool = False

    @property
    def measurement_mode(self) -> MeasurementMode:
        """Return the current measurement mode.

        Mutable: set via the ``measurement_mode`` setter by the emit
        pass based on whether the circuit requires runtime control flow.
        """
        return self._measurement_mode

    @measurement_mode.setter
    def measurement_mode(self, value: MeasurementMode) -> None:
        self._measurement_mode = value

    def _emit(self, line: str) -> None:
        """Append an indented source line."""
        self._lines.append("    " * self._indent + line)

    def _qref(self, idx: int) -> str:
        """Return the source expression for a qubit slot.

        Args:
            idx (int): Helper-local or entry-kernel qubit slot.

        Returns:
            str: Source expression that references the slot.
        """
        return self._qubit_refs.get(idx, f"q[{idx}]")

    def _control_ref(self, indices: list[int]) -> str:
        """Return the source expression for a CUDA-Q control operand.

        Args:
            indices (list[int]): Control qubit slots.

        Returns:
            str: Single-qubit expression for one control, or a list
            expression for multiple controls.
        """
        if len(indices) == 1:
            return self._qref(indices[0])
        return "[" + ", ".join(self._qref(index) for index in indices) + "]"

    def _clbit_ref(self, idx: int) -> str:
        """Return the source expression to read clbit *idx*.

        Boxed (loop-carried) clbits use ``__b{idx}[0]``; scalar clbits
        use ``__b{idx}``.
        """
        if idx in self._boxed_clbits:
            return f"__b{idx}[0]"
        return f"__b{idx}"

    def _clbit_store(self, idx: int, expr: str) -> str:
        """Return a source statement that stores *expr* into clbit *idx*."""
        return f"{self._clbit_ref(idx)} = {expr}"

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
        self._helper_sources.clear()
        self._helper_cache.clear()
        self._qubit_refs.clear()
        self._building_helper = False
        self._helper_param_used = False
        self._indent = 1

        self._emit(f"q = cudaq.qvector({num_qubits})")
        for i in range(num_clbits):
            if i in self._boxed_clbits:
                self._emit(f"__b{i} = [False]")
            else:
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
            clbit_list = ", ".join(self._clbit_ref(i) for i in range(self._num_clbits))
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
        main_source = f"@cudaq.kernel\n{sig}\n{body}\n"
        source = "\n".join([*self._helper_sources, main_source])

        # Register source in linecache with a unique synthetic filename so
        # that ``inspect.getsource()`` succeeds.  CUDA-Q's decorator calls
        # ``inspect.getsourcelines()`` on the function object, which requires
        # a resolvable filename backed by linecache.
        kernel_id = next(self._kernel_counter)
        filename = f"<qamomile_cudaq_kernel_{kernel_id}>"
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(True),
            filename,
        )
        code = compile(source, filename, "exec")

        namespace: dict[str, Any] = {"cudaq": cudaq}
        exec(code, namespace)  # noqa: S102

        circuit.kernel_func = namespace["_qamomile_kernel"]
        circuit.source = source
        circuit.entry_source = main_source
        circuit.execution_mode = mode
        circuit.param_count = self._param_count
        return circuit

    def build_controlled_helper(
        self,
        num_targets: int,
        emit_body: Callable[[], None],
    ) -> tuple[str, bool]:
        """Build a pure-device helper kernel for ``cudaq.control``.

        Args:
            num_targets (int): Number of target qubit arguments accepted
                by the helper.
            emit_body (Callable[[], None]): Callback that emits the
                helper body into this emitter while qubit slots are mapped
                to helper arguments.

        Returns:
            tuple[str, bool]: Helper function name, and whether the
            caller must pass the entry-point ``thetas`` parameter list.

        Raises:
            EmitError: Propagated from ``emit_body`` when the helper
                body cannot be emitted for CUDA-Q.
        """
        saved_lines = self._lines
        saved_indent = self._indent
        saved_qubit_refs = self._qubit_refs
        saved_mode = self._measurement_mode
        saved_num_clbits = self._num_clbits
        saved_boxed_clbits = self._boxed_clbits
        saved_suppress_trace = getattr(self, "_suppress_trace", False)
        saved_building_helper = self._building_helper
        saved_helper_param_used = self._helper_param_used

        self._lines = []
        self._indent = 1
        self._qubit_refs = {i: f"t{i}" for i in range(num_targets)}
        self._measurement_mode = MeasurementMode.STATIC
        self._num_clbits = 0
        self._boxed_clbits = set()
        self._building_helper = True
        self._helper_param_used = False
        setattr(self, "_suppress_trace", True)

        try:
            emit_body()
            helper_lines = self._lines
            uses_thetas = self._helper_param_used
            cache_key = (num_targets, uses_thetas, tuple(helper_lines))
            cached_name = self._helper_cache.get(cache_key)
            if cached_name is not None:
                return cached_name, uses_thetas

            helper_name = f"_qamomile_controlled_{next(self._helper_counter)}"
            args = [f"t{i}: cudaq.qubit" for i in range(num_targets)]
            if uses_thetas:
                args.append("thetas: list[float]")
            body = "\n".join(helper_lines) if helper_lines else "    pass"
            self._helper_sources.append(
                f"@cudaq.kernel\ndef {helper_name}({', '.join(args)}):\n{body}\n"
            )
            self._helper_cache[cache_key] = helper_name
            return helper_name, uses_thetas
        finally:
            self._lines = saved_lines
            self._indent = saved_indent
            self._qubit_refs = saved_qubit_refs
            self._measurement_mode = saved_mode
            self._num_clbits = saved_num_clbits
            self._boxed_clbits = saved_boxed_clbits
            self._building_helper = saved_building_helper
            self._helper_param_used = saved_helper_param_used
            setattr(self, "_suppress_trace", saved_suppress_trace)

    def emit_controlled_kernel_call(
        self,
        circuit: CudaqKernelArtifact,
        helper_name: str,
        control_indices: list[int],
        target_indices: list[int],
        uses_thetas: bool,
    ) -> None:
        """Emit a ``cudaq.control`` call to a generated helper kernel.

        Args:
            circuit (CudaqKernelArtifact): Artifact currently being
                emitted.  The argument is accepted for GateEmitter
                symmetry and tracing subclasses.
            helper_name (str): Name of the generated helper function.
            control_indices (list[int]): Physical control qubit slots.
            target_indices (list[int]): Physical target qubit slots.
            uses_thetas (bool): Whether to pass the entry-point
                ``thetas`` parameter list to the helper.
        """
        del circuit
        args = [
            helper_name,
            self._control_ref(control_indices),
            *[self._qref(index) for index in target_indices],
        ]
        if uses_thetas:
            args.append("thetas")
            if self._building_helper:
                self._helper_param_used = True
        self._emit(f"cudaq.control({', '.join(args)})")

    def create_parameter(self, name: str) -> CudaqExpr:
        """Return a ``CudaqExpr`` referencing ``thetas[i]``.

        Returns an arithmetic-capable expression object so that
        ``StandardEmitPass._evaluate_binop()`` can compose gate-angle
        expressions (e.g. ``gamma * Jij``) without triggering a
        ``TypeError`` on raw string operands.

        Args:
            name: Symbolic parameter name.

        Returns:
            ``CudaqExpr`` wrapping a source expression like
            ``"thetas[0]"``.
        """
        if name not in self._param_map:
            self._param_map[name] = self._param_count
            self._param_count += 1
        if self._building_helper:
            self._helper_param_used = True
        return CudaqExpr(f"thetas[{self._param_map[name]}]")

    # ------------------------------------------------------------------
    # Single-qubit gates (no parameters)
    # ------------------------------------------------------------------

    def emit_h(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        """Emit Hadamard gate."""
        self._emit(f"h({self._qref(qubit)})")

    def emit_x(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        """Emit Pauli-X gate."""
        self._emit(f"x({self._qref(qubit)})")

    def emit_y(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        """Emit Pauli-Y gate."""
        self._emit(f"y({self._qref(qubit)})")

    def emit_z(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        """Emit Pauli-Z gate."""
        self._emit(f"z({self._qref(qubit)})")

    def emit_s(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        """Emit S (phase) gate."""
        self._emit(f"s({self._qref(qubit)})")

    def emit_sdg(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        """Emit S-dagger gate."""
        self._emit(f"sdg({self._qref(qubit)})")

    def emit_t(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        """Emit T gate."""
        self._emit(f"t({self._qref(qubit)})")

    def emit_tdg(self, circuit: CudaqKernelArtifact, qubit: int) -> None:
        """Emit T-dagger gate."""
        self._emit(f"tdg({self._qref(qubit)})")

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
        self._emit(f"rx({self._angle_expr(angle)}, {self._qref(qubit)})")

    def emit_ry(
        self, circuit: CudaqKernelArtifact, qubit: int, angle: float | Any
    ) -> None:
        """Emit RY rotation gate."""
        self._emit(f"ry({self._angle_expr(angle)}, {self._qref(qubit)})")

    def emit_rz(
        self, circuit: CudaqKernelArtifact, qubit: int, angle: float | Any
    ) -> None:
        """Emit RZ rotation gate."""
        self._emit(f"rz({self._angle_expr(angle)}, {self._qref(qubit)})")

    def emit_p(
        self, circuit: CudaqKernelArtifact, qubit: int, angle: float | Any
    ) -> None:
        """Emit phase gate (R1)."""
        self._emit(f"r1({self._angle_expr(angle)}, {self._qref(qubit)})")

    # ------------------------------------------------------------------
    # Two-qubit gates
    # ------------------------------------------------------------------

    def emit_cx(self, circuit: CudaqKernelArtifact, control: int, target: int) -> None:
        """Emit CNOT (controlled-X) gate."""
        self._emit(f"x.ctrl({self._qref(control)}, {self._qref(target)})")

    def emit_cz(self, circuit: CudaqKernelArtifact, control: int, target: int) -> None:
        """Emit controlled-Z gate."""
        self._emit(f"z.ctrl({self._qref(control)}, {self._qref(target)})")

    def emit_swap(self, circuit: CudaqKernelArtifact, qubit1: int, qubit2: int) -> None:
        """Emit SWAP gate."""
        self._emit(f"swap({self._qref(qubit1)}, {self._qref(qubit2)})")

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

        Follows the shared CP_DECOMPOSITION recipe from
        ``qamomile.circuit.transpiler.decompositions``.
        Inlined here because CudaqExpr angles require string-based codegen.
        """
        a = self._angle_expr(angle)
        if isinstance(angle, (int, float)):
            half = repr(angle / 2.0)
            neg_half = repr(-angle / 2.0)
        else:
            half = f"({a}) * 0.5"
            neg_half = f"({a}) * (-0.5)"
        self._emit(f"rz({half}, {self._qref(target)})")
        self._emit(f"x.ctrl({self._qref(control)}, {self._qref(target)})")
        self._emit(f"rz({neg_half}, {self._qref(target)})")
        self._emit(f"x.ctrl({self._qref(control)}, {self._qref(target)})")
        self._emit(f"rz({half}, {self._qref(control)})")

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
        self._emit(f"x.ctrl({self._qref(qubit1)}, {self._qref(qubit2)})")
        self._emit(f"rz({a}, {self._qref(qubit2)})")
        self._emit(f"x.ctrl({self._qref(qubit1)}, {self._qref(qubit2)})")

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
        """Emit controlled-Hadamard via decomposition.

        Mirrors the ``CH_DECOMPOSITION`` recipe from
        ``qamomile.circuit.transpiler.decompositions``.  Inlined here (rather
        than calling :func:`emit_decomposition`) so that the string-based
        codegen produces the exact source contract expected by the tracing
        test emitter, without double-recording primitive gate calls.
        """
        self._emit(f"ry({math.pi / 4}, {self._qref(target)})")
        self._emit(f"x.ctrl({self._qref(control)}, {self._qref(target)})")
        self._emit(f"ry({-math.pi / 4}, {self._qref(target)})")

    def emit_cy(self, circuit: CudaqKernelArtifact, control: int, target: int) -> None:
        """Emit controlled-Y via decomposition.

        Mirrors the ``CY_DECOMPOSITION`` recipe from
        ``qamomile.circuit.transpiler.decompositions``.  See
        :meth:`emit_ch` for why this is inlined rather than delegating to
        :func:`emit_decomposition`.
        """
        self._emit(f"sdg({self._qref(target)})")
        self._emit(f"x.ctrl({self._qref(control)}, {self._qref(target)})")
        self._emit(f"s({self._qref(target)})")

    def emit_crx(
        self,
        circuit: CudaqKernelArtifact,
        control: int,
        target: int,
        angle: float | Any,
    ) -> None:
        """Emit controlled-RX gate."""
        a = self._angle_expr(angle)
        self._emit(f"rx.ctrl({a}, {self._qref(control)}, {self._qref(target)})")

    def emit_cry(
        self,
        circuit: CudaqKernelArtifact,
        control: int,
        target: int,
        angle: float | Any,
    ) -> None:
        """Emit controlled-RY gate."""
        a = self._angle_expr(angle)
        self._emit(f"ry.ctrl({a}, {self._qref(control)}, {self._qref(target)})")

    def emit_crz(
        self,
        circuit: CudaqKernelArtifact,
        control: int,
        target: int,
        angle: float | Any,
    ) -> None:
        """Emit controlled-RZ gate."""
        a = self._angle_expr(angle)
        self._emit(f"rz.ctrl({a}, {self._qref(control)}, {self._qref(target)})")

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def emit_measure(
        self, circuit: CudaqKernelArtifact, qubit: int, clbit: int
    ) -> None:
        """Emit measurement, respecting the current mode.

        In STATIC mode (``measurement_mode == STATIC``), this is a
        no-op: ``cudaq.sample()`` auto-measures all qubits.

        In RUNNABLE mode (``measurement_mode == RUNNABLE``), emits
        explicit ``mz()`` to capture mid-circuit measurement results as
        local variables for ``if`` / ``while`` conditions.
        """
        if self.measurement_mode == MeasurementMode.STATIC:
            return
        self._emit(self._clbit_store(clbit, f"mz({self._qref(qubit)})"))
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
        args = ", ".join(self._qref(i) for i in control_indices)
        self._emit(f"x.ctrl({args}, {self._qref(target_idx)})")

    # ------------------------------------------------------------------
    # Control flow support
    # ------------------------------------------------------------------

    def supports_for_loop(self) -> bool:
        """Return False: for-loops are unrolled by the transpiler."""
        return False

    def supports_if_else(self) -> bool:
        """Return True only in RUNNABLE mode (measurement-dependent branching)."""
        return self.measurement_mode == MeasurementMode.RUNNABLE

    def supports_while_loop(self) -> bool:
        """Return True only in RUNNABLE mode."""
        return self.measurement_mode == MeasurementMode.RUNNABLE

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
        self._emit(f"if {self._clbit_ref(clbit)}:")
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
        self._emit(f"while {self._clbit_ref(clbit)}:")
        self._indent += 1
        return {"clbit": clbit}

    def emit_while_end(
        self, circuit: CudaqKernelArtifact, context: dict[str, Any]
    ) -> None:
        """Close ``while`` block by decreasing indentation."""
        self._indent -= 1
