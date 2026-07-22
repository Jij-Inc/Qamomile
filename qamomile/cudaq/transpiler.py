"""CUDA-Q backend transpiler implementation.

This module provides CudaqTranspiler for converting Qamomile QKernels
into CUDA-Q decorator-kernel artifacts, along with CudaqExecutor.

All circuits (static and runtime) are emitted through a single
``CudaqKernelEmitter`` codegen path.  The emitter produces
``CudaqKernelArtifact`` instances whose ``execution_mode`` determines
whether ``cudaq.sample()`` / ``cudaq.observe()`` / ``cudaq.get_state()``
(STATIC) or ``cudaq.run()`` (RUNNABLE) is used for execution.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Sequence

from qamomile.circuit.transpiler.circuit_ir import CircuitBackendEmitPass
from qamomile.circuit.transpiler.executable import (
    ParameterMetadata,
    QuantumExecutor,
)
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.separate import SegmentationPass
from qamomile.circuit.transpiler.transpiler import Transpiler

from .emitter import (
    CudaqKernelArtifact,
    ExecutionMode,
)
from .materializer import CudaqMaterializer

if TYPE_CHECKING:
    import qamomile.observable as qm_o


@dataclasses.dataclass
class BoundCudaqKernelArtifact:
    """Hold a CUDA-Q kernel with runtime parameters bound.

    Args:
        kernel_func: Decorated CUDA-Q kernel.
        num_qubits: Number of physical qubits.
        num_clbits: Number of logical measurement bits.
        param_values: Bound parameter values in backend order.
        execution_mode: Runtime API required by the kernel.
    """

    kernel_func: Any
    num_qubits: int
    num_clbits: int
    param_values: list[float]
    execution_mode: ExecutionMode = ExecutionMode.STATIC


class CudaqExecutor(QuantumExecutor[CudaqKernelArtifact]):
    """CUDA-Q quantum executor.

    Supports sampling via ``cudaq.sample`` / ``cudaq.run`` and expectation
    value estimation via ``cudaq.observe``.  Dispatches to the appropriate
    CUDA-Q runtime API based on the artifact's ``execution_mode``:

    - ``STATIC`` artifacts (no mid-circuit measurement or runtime control
      flow) support both sampling (``cudaq.sample``) and expectation-value
      estimation (``cudaq.observe``).
    - ``RUNNABLE`` artifacts (mid-circuit measurement or measurement-dependent
      control flow such as ``if bit:`` / ``while bit:``) support sampling only,
      via ``cudaq.run``.

    Expectation-value estimation is therefore **static-only**: calling
    :meth:`estimate` on a ``RUNNABLE`` artifact raises ``TypeError`` because
    ``cudaq.observe`` cannot consume a kernel that requires ``cudaq.run``.
    See :meth:`estimate` for the full rationale.

    Args:
        target (str | None): CUDA-Q target name (e.g., ``"qpp-cpu"``). If
            None, uses the default CUDA-Q target.
    """

    def __init__(self, target: str | None = None):
        import cudaq

        self._target = target
        if self._target:
            cudaq.set_target(self._target)  # type: ignore[operator]

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

            cudaq.set_target(self._target)  # type: ignore[operator]

    def execute(self, circuit: Any, shots: int) -> dict[str, int]:
        """Execute circuit and return canonical big-endian bitstring counts.

        Dispatches based on ``execution_mode``:

        - ``STATIC``: uses ``cudaq.sample()`` on the decorator kernel.
        - ``RUNNABLE``: uses ``cudaq.run()`` on the runnable kernel.

        Both paths return bitstrings in big-endian format (highest qubit
        index = leftmost bit). An artifact without quantum or classical bits
        returns ``{"": shots}`` without calling the CUDA-Q runtime.
        """
        if circuit.num_qubits == 0 and circuit.num_clbits == 0:
            return {"": shots}

        mode = getattr(circuit, "execution_mode", ExecutionMode.STATIC)
        if mode == ExecutionMode.RUNNABLE:
            return self._execute_runtime(circuit, shots)
        return self._execute_sample(circuit, shots)

    def _execute_sample(self, circuit: Any, shots: int) -> dict[str, int]:
        """Execute via ``cudaq.sample()`` for STATIC-mode kernels."""
        import cudaq

        self._ensure_target()

        if isinstance(circuit, BoundCudaqKernelArtifact):
            result = cudaq.sample(  # type: ignore[operator]
                circuit.kernel_func, circuit.param_values, shots_count=shots
            )
        else:
            result = cudaq.sample(circuit.kernel_func, shots_count=shots)  # type: ignore[operator]

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
            canonical = padded[::-1]  # little-endian (allocation-order) -> big-endian
            counts[canonical] = counts.get(canonical, 0) + count

        return counts

    def _execute_runtime(
        self,
        circuit: CudaqKernelArtifact | BoundCudaqKernelArtifact,
        shots: int,
    ) -> dict[str, int]:
        """Execute via ``cudaq.run()`` for RUNNABLE-mode kernels.

        ``cudaq.run()`` returns a list of per-shot return values.  Each
        return value is ``list[bool]`` (from ``return [__b0, __b1, ...]``),
        with one bool per logical clbit.

        This method aggregates per-shot results into canonical big-endian
        bitstring counts matching the ``QuantumExecutor.execute()`` contract.
        """
        import cudaq

        self._ensure_target()

        if isinstance(circuit, BoundCudaqKernelArtifact):
            results = cudaq.run(  # type: ignore[operator]
                circuit.kernel_func, circuit.param_values, shots_count=shots
            )
        else:
            results = cudaq.run(circuit.kernel_func, shots_count=shots)  # type: ignore[operator]

        num_clbits = circuit.num_clbits
        counts: dict[str, int] = {}
        for shot_result in results:
            # shot_result is list[bool] in clbit order (from [__b0, __b1, ...])
            bits = ["1" if b else "0" for b in shot_result]
            # Pad to num_clbits if shorter
            while len(bits) < num_clbits:
                bits.append("0")
            # Reverse: clbit-order -> big-endian
            canonical = "".join(reversed(bits))
            counts[canonical] = counts.get(canonical, 0) + 1

        return counts

    def bind_parameters(  # type: ignore[override]
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
        """Estimate an expectation value using ``cudaq.observe``.

        Expectation-value estimation on the CUDA-Q backend is **static-only**.
        It is supported exclusively for ``STATIC``-mode artifacts, which are
        evaluated with ``cudaq.observe()``.  ``RUNNABLE``-mode artifacts (a
        kernel containing mid-circuit measurement or measurement-dependent
        control flow such as ``if bit:`` / ``while bit:``) are emitted for
        ``cudaq.run()``, and ``cudaq.observe()`` cannot consume them.  Such
        artifacts therefore raise ``TypeError`` rather than silently returning
        a meaningless value.

        This is a CUDA-Q API limitation, not merely a Qamomile gap:
        ``cudaq.observe()`` evaluates the analytic expectation value of a
        deterministic state-preparation kernel and has no execution path for a
        kernel that requires the per-shot ``cudaq.run()`` runtime.  There is no
        safe alternative that preserves the exact-expectation contract of this
        method, so the ``TypeError`` path is intentional.  Under the current
        affine model a single Qamomile kernel also cannot be both ``RUNNABLE``
        and carry a terminal ``qmc.expval`` (reusing a measured qubit is
        rejected), so the ``ExecutableProgram.run`` expval path does not reach
        this guard; it is reached only when ``estimate`` is called directly on
        a ``RUNNABLE`` artifact.

        Args:
            circuit (Any): A CUDA-Q artifact (``CudaqKernelArtifact`` or
                ``BoundCudaqKernelArtifact``).  Must be ``STATIC``-mode;
                ``RUNNABLE``-mode artifacts are rejected.
            hamiltonian (qm_o.Hamiltonian): The observable to measure.  A
                ``qamomile.observable.Hamiltonian`` is converted to a CUDA-Q
                ``SpinOperator``; an already-converted operator is used as-is.
            params (Sequence[float] | None): Values for the kernel's runtime
                ``parameters`` slots, in declared parameter order.  Compile-
                time-bound parameters are already baked into the artifact and
                are not included here.  Defaults to None for a non-parametric
                or already-bound (``BoundCudaqKernelArtifact``) kernel.

        Returns:
            float: The estimated expectation value ``<psi|H|psi>``.

        Raises:
            TypeError: If ``circuit`` is a ``RUNNABLE``-mode artifact, since
                ``cudaq.observe()`` only accepts static state-preparation
                kernels (see the static-only note above).

        Example:
            >>> transpiler = CudaqTranspiler()
            >>> exe = transpiler.transpile(static_ansatz, bindings={"H": H})
            >>> executor = transpiler.executor()
            >>> value = executor.estimate(exe.get_first_circuit(), H)
        """
        import cudaq
        import qamomile.observable as qm_o
        from qamomile.cudaq.observable import hamiltonian_to_cudaq_spin_op

        mode = getattr(circuit, "execution_mode", ExecutionMode.STATIC)
        if mode == ExecutionMode.RUNNABLE:
            raise TypeError(
                "Expectation-value estimation is not available for this CUDA-Q "
                "circuit. The kernel uses mid-circuit measurement or "
                "measurement-dependent control flow (e.g. `if bit:` / "
                "`while bit:`), so it is emitted in RUNNABLE mode and executed "
                "with `cudaq.run()`. CUDA-Q's expectation-value primitive "
                "`cudaq.observe()` only accepts static state-preparation "
                "kernels and cannot consume a RUNNABLE kernel. To estimate an "
                "expectation value, express the state preparation as a static "
                "kernel without measurement-dependent control flow; "
                "expectation values cannot be computed for measurement-"
                "conditioned circuits on the CUDA-Q backend."
            )

        self._ensure_target()  # type: ignore[unreachable]

        if isinstance(hamiltonian, qm_o.Hamiltonian):  # type: ignore[unreachable]
            spin_op = hamiltonian_to_cudaq_spin_op(hamiltonian)
        else:
            spin_op = hamiltonian  # type: ignore[unreachable]

        if isinstance(circuit, BoundCudaqKernelArtifact):
            if params is not None:
                raise ValueError(
                    "params must be omitted for a BoundCudaqKernelArtifact; "
                    "its runtime parameter values are already fixed."
                )
            result: Any = cudaq.observe(
                circuit.kernel_func, spin_op, circuit.param_values
            )  # type: ignore[operator]
        else:
            if params is not None:
                result = cudaq.observe(circuit.kernel_func, spin_op, list(params))  # type: ignore[operator]
            else:
                result = cudaq.observe(circuit.kernel_func, spin_op)  # type: ignore[operator]

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

    def _create_segmentation_pass(self) -> SegmentationPass:
        """Create the host-orchestrated circuit segmentation pass.

        Returns:
            SegmentationPass: Standard single-quantum-segment planner.
        """
        return SegmentationPass()

    def _create_emit_pass(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> EmitPass[CudaqKernelArtifact]:
        """Create the capability-driven CUDA-Q materialization pipeline.

        Args:
            bindings (dict[str, Any] | None): Compile-time bindings. Defaults
                to ``None``.
            parameters (list[str] | None): Runtime parameter names. Defaults
                to ``None``.

        Returns:
            EmitPass[CudaqKernelArtifact]: Circuit lowering, legalization,
                and CUDA-Q materialization pass.
        """
        return CircuitBackendEmitPass(
            CudaqMaterializer(),
            bindings,
            parameters,
        )

    def executor(  # type: ignore[override]
        self,
        target: str | None = None,
    ) -> CudaqExecutor:
        """Create a CUDA-Q executor.

        Args:
            target: CUDA-Q target name (e.g., ``"qpp-cpu"``).
                If None, uses the default CUDA-Q target.
        """
        return CudaqExecutor(target)
