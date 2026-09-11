"""Executable program structure for compiled quantum-classical programs."""

from __future__ import annotations

import dataclasses
from typing import Any, Generic, TypeVar

from qamomile.circuit.ir.value import ValueLike

# Re-export for backward compatibility (used by engines and passes)
from qamomile.circuit.transpiler.classical_executor import (
    ClassicalExecutor as ClassicalExecutor,  # noqa: F401
)
from qamomile.circuit.transpiler.compiled_segments import (
    CompiledClassicalSegment,
    CompiledExpvalSegment,
    CompiledQuantumSegment,
)
from qamomile.circuit.transpiler.errors import ExecutionError
from qamomile.circuit.transpiler.execution_context import ExecutionContext
from qamomile.circuit.transpiler.execution_request import EstimationAccuracy
from qamomile.circuit.transpiler.job import (
    ExpvalJob,
    JobSnapshot,
    RunJob,
    SampleJob,
)
from qamomile.circuit.transpiler.parameter_binding import (
    ParameterArrayInfo,
    ParameterContainerKind,
    ParameterInfo,
    ParameterMetadata,
)
from qamomile.circuit.transpiler.quantum_executor import QuantumExecutor
from qamomile.circuit.transpiler.segments import ProgramPlan

# Re-export for backward compatibility
__all__ = [
    "ClassicalExecutor",
    "CompiledClassicalSegment",
    "CompiledExpvalSegment",
    "CompiledQuantumSegment",
    "ExecutableProgram",
    "ExecutionContext",
    "ParameterArrayInfo",
    "ParameterContainerKind",
    "ParameterInfo",
    "ParameterMetadata",
    "QuantumExecutor",
]

T = TypeVar("T")  # Engine circuit type


@dataclasses.dataclass
class ExecutableProgram(Generic[T]):
    """A fully compiled program ready for execution.

    Contains compiled quantum, classical, and expectation-value segments.
    Use ``sample()`` for multi-shot execution or ``run()`` for single
    execution.

    Example:
        executable = transpiler.compile(kernel)

        # Sample: multiple shots, returns counts
        job = executable.sample(executor, shots=1000)
        result = job.result()  # SampleResult with counts

        # Run: single shot, returns typed result
        job = executable.run(executor)
        result = job.result()  # Returns kernel's return type
    """

    plan: ProgramPlan | None = None
    compiled_quantum: list[CompiledQuantumSegment[T]] = dataclasses.field(
        default_factory=list
    )
    compiled_classical: list[CompiledClassicalSegment] = dataclasses.field(
        default_factory=list
    )
    compiled_expval: list[CompiledExpvalSegment] = dataclasses.field(
        default_factory=list
    )

    # Final output values
    output_values: list[ValueLike] = dataclasses.field(default_factory=list)

    # ------------------------------------------------------------------
    # Data access properties
    # ------------------------------------------------------------------

    @property
    def parameter_names(self) -> list[str]:
        """Get list of parameter names that need binding."""
        if not self.compiled_quantum:
            return []
        return [p.name for p in self.compiled_quantum[0].parameter_metadata.parameters]

    @property
    def has_parameters(self) -> bool:
        """Check if this program has unbound parameters."""
        return len(self.parameter_names) > 0

    @property
    def quantum_circuit(self) -> T:
        """Get the single quantum circuit.

        Returns the quantum circuit from the single quantum segment.
        This property enforces Qamomile's C->Q->C execution pattern.

        Returns:
            The engine-specific quantum circuit

        Raises:
            ExecutionError: If no quantum circuit exists
        """
        if not self.compiled_quantum:
            raise ExecutionError("No quantum circuit")
        return self.compiled_quantum[0].circuit

    def get_circuits(self) -> list[T]:
        """Get all quantum circuits in execution order."""
        return [seg.circuit for seg in self.compiled_quantum]

    def get_first_circuit(self) -> T | None:
        """Get the first quantum circuit, or None if no quantum segments."""
        if self.compiled_quantum:
            return self.compiled_quantum[0].circuit
        return None

    # ------------------------------------------------------------------
    # Execution facade (delegates to ProgramOrchestrator)
    # ------------------------------------------------------------------

    def sample(
        self,
        executor: QuantumExecutor[T],
        shots: int = 1024,
        bindings: dict[str, Any] | None = None,
    ) -> SampleJob[Any]:
        """Submit a multi-shot execution and return its lazy job.

        Args:
            executor (QuantumExecutor[T]): Engine-specific quantum executor.
            shots (int): Number of shots to run.
            bindings (dict[str, Any] | None): Parameter bindings. Supports
                three formats:
                - Vector: {"gammas": [0.1, 0.2], "betas": [0.3, 0.4]}
                - Dict parameter: {"coeffs": {0: 0.1, (0, 1): 0.2}},
                  decomposed per key onto the emitted parameters
                - Indexed: {"gammas[0]": 0.1, "coeffs[(0, 1)]": 0.2}

        Returns:
            SampleJob[Any]: A job that resolves to a SampleResult with the
                per-bitstring counts.

        Raises:
            ExecutionError: If no quantum circuit to execute
            ValueError: If required parameters are missing

        Example:
            job = executable.sample(executor, shots=1000, bindings={"gamma": [0.5]})
            result = job.result()
            print(result.results)  # [(0.25, 500), (0.75, 500)]
        """
        from qamomile.circuit.transpiler.program_orchestrator import (
            ProgramOrchestrator,
        )

        return ProgramOrchestrator(self).sample(executor, shots, bindings)

    def run(
        self,
        executor: QuantumExecutor[T],
        bindings: dict[str, Any] | None = None,
        *,
        estimation: EstimationAccuracy | None = None,
    ) -> RunJob[Any] | ExpvalJob:
        """Submit one execution and return its lazy result job.

        Args:
            executor (QuantumExecutor[T]): Engine-specific quantum executor.
            bindings (dict[str, Any] | None): Parameter bindings. Supports
                three formats:
                - Vector: {"gammas": [0.1, 0.2], "betas": [0.3, 0.4]}
                - Dict parameter: {"coeffs": {0: 0.1, (0, 1): 0.2}},
                  decomposed per key onto the emitted parameters
                - Indexed: {"gammas[0]": 0.1, "coeffs[(0, 1)]": 0.2}
            estimation (EstimationAccuracy | None): Optional per-execution
                expectation accuracy policy. Defaults to the executor's
                configured behavior.

        Returns:
            RunJob[Any] | ExpvalJob: A RunJob that resolves to the kernel's
                return type, or an ExpvalJob when the program contains an
                expectation-value computation.

        Raises:
            ExecutionError: If no quantum circuit to execute
            ValueError: If required parameters are missing

        Example:
            job = executable.run(executor, bindings={"gamma": [0.5]})
            result = job.result()
            print(result)  # 0.25 (for QFixed) or (0, 1) (for bits)
        """
        from qamomile.circuit.transpiler.program_orchestrator import (
            ProgramOrchestrator,
        )

        return ProgramOrchestrator(self).run(executor, bindings, estimation)

    def restore(
        self,
        executor: QuantumExecutor[T],
        snapshot: JobSnapshot,
        bindings: dict[str, Any] | None = None,
    ) -> SampleJob[Any] | RunJob[Any] | ExpvalJob:
        """Restore saved executions with this program's typed result ABI.

        Snapshots retain provider identifiers, completed local raw values, and
        ordered execution groups. Legacy flat provider snapshots remain
        supported. Reuse the same compiled program and pass the original runtime
        bindings explicitly to reproduce classical pre- and post-processing.
        Credentials, arbitrary bindings, and Python callables are not saved.
        Restoration reconnects to remote jobs without resubmitting or waiting
        for results; local values need no provider restoration support.

        Args:
            executor (QuantumExecutor[T]): Engine adapter configured with the
                provider credentials and target used by the original job.
            snapshot (JobSnapshot): Snapshot returned by the original public
                job's ``snapshot()`` method.
            bindings (dict[str, Any] | None): Original runtime parameter
                bindings. Defaults to ``None`` for parameter-free programs.

        Returns:
            SampleJob[Any] | RunJob[Any] | ExpvalJob: Restored lazy job with
                the same typed public result conversion as a new execution.

        Raises:
            ExecutionError: If the snapshot operation or execution shape does
                not match this executable program.
            NotImplementedError: If the executor cannot restore the referenced
                provider execution.
            ValueError: If required bindings are missing or invalid.

        Example:
            >>> original = executable.sample(executor, shots=1000)
            >>> snapshot = original.snapshot()
            >>> restored = executable.restore(executor, snapshot)
            >>> restored.result()
        """
        from qamomile.circuit.transpiler.program_orchestrator import (
            ProgramOrchestrator,
        )

        return ProgramOrchestrator(self).restore(executor, snapshot, bindings)

    def _run_expval(
        self,
        executor: QuantumExecutor[T],
        bindings: dict[str, Any] | None = None,
        *,
        estimation: EstimationAccuracy | None = None,
    ) -> ExpvalJob:
        """Submit a pure expectation execution through the compatibility API.

        Args:
            executor (QuantumExecutor[T]): Engine execution adapter.
            bindings (dict[str, Any] | None): Runtime public bindings.
            estimation (EstimationAccuracy | None): Optional accuracy policy.

        Returns:
            ExpvalJob: Deferred expectation result.

        Raises:
            ExecutionError: If the program does not contain one pure
                expectation computation.
        """
        from qamomile.circuit.transpiler.program_orchestrator import (
            ProgramOrchestrator,
        )

        return ProgramOrchestrator(self).run_expval(
            executor,
            bindings,
            estimation,
        )
