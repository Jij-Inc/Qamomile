"""Executable program structure for compiled quantum-classical programs."""

from __future__ import annotations

import dataclasses
from typing import Any, Generic, TypeVar

# Re-export for backward compatibility (used by backends and passes)
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
from qamomile.circuit.transpiler.job import ExpvalJob, RunJob, SampleJob
from qamomile.circuit.transpiler.parameter_binding import (
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
    "ParameterInfo",
    "ParameterMetadata",
    "QuantumExecutor",
]

T = TypeVar("T")  # Backend circuit type


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

    # Final output references
    output_refs: list[str] = dataclasses.field(default_factory=list)

    # Number of output bits (for result conversion)
    num_output_bits: int = 0

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
            The backend-specific quantum circuit

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
        """Execute with multiple shots and return counts.

        Args:
            executor: Backend-specific quantum executor.
            shots: Number of shots to run.
            bindings: Parameter bindings. Supports two formats:
                - Vector: {"gammas": [0.1, 0.2], "betas": [0.3, 0.4]}
                - Indexed: {"gammas[0]": 0.1, "gammas[1]": 0.2}

        Returns:
            SampleJob that resolves to SampleResult with results.

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
    ) -> RunJob[Any] | ExpvalJob:
        """Execute once and return single result.

        Args:
            executor: Backend-specific quantum executor.
            bindings: Parameter bindings. Supports two formats:
                - Vector: {"gammas": [0.1, 0.2], "betas": [0.3, 0.4]}
                - Indexed: {"gammas[0]": 0.1, "gammas[1]": 0.2}

        Returns:
            RunJob that resolves to the kernel's return type, or
            ExpvalJob if the program contains expectation value computation.

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

        return ProgramOrchestrator(self).run(executor, bindings)

    def _run_expval(
        self,
        executor: QuantumExecutor[T],
        bindings: dict[str, Any] | None = None,
    ) -> ExpvalJob:
        """Backward-compatible helper for pure expval execution."""
        from qamomile.circuit.transpiler.program_orchestrator import (
            ProgramOrchestrator,
        )

        return ProgramOrchestrator(self).run_expval(executor, bindings)
