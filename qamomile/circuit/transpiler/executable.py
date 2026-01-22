"""Executable program structure for compiled quantum-classical programs."""

from __future__ import annotations

import dataclasses
from typing import Any, Generic, TypeVar

from qamomile.circuit.transpiler.classical_executor import ClassicalExecutor
from qamomile.circuit.transpiler.compiled_segments import (
    CompiledClassicalSegment,
    CompiledExpvalSegment,
    CompiledQuantumSegment,
)
from qamomile.circuit.transpiler.errors import ExecutionError
from qamomile.circuit.transpiler.execution_context import ExecutionContext
from qamomile.circuit.transpiler.job import RunJob, SampleJob, ExpvalJob
from qamomile.circuit.transpiler.parameter_binding import (
    ParameterInfo,
    ParameterMetadata,
)
from qamomile.circuit.transpiler.quantum_executor import QuantumExecutor

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

    This is the Orchestrator - manages execution of mixed
    classical/quantum programs.

    Example:
        executable = transpiler.compile(kernel)

        # Sample: multiple shots, returns counts
        job = executable.sample(executor, shots=1000)
        result = job.result()  # SampleResult with counts

        # Run: single shot, returns typed result
        job = executable.run(executor)
        result = job.result()  # Returns kernel's return type
    """

    compiled_quantum: list[CompiledQuantumSegment[T]] = dataclasses.field(
        default_factory=list
    )
    compiled_classical: list[CompiledClassicalSegment] = dataclasses.field(
        default_factory=list
    )
    compiled_expval: list[CompiledExpvalSegment] = dataclasses.field(
        default_factory=list
    )

    # Execution order: indices into compiled_quantum, compiled_classical, or compiled_expval
    # Tuple of (segment_type: str, index: int) where segment_type is "quantum", "classical", or "expval"
    execution_order: list[tuple[str, int]] = dataclasses.field(default_factory=list)

    # Final output references
    output_refs: list[str] = dataclasses.field(default_factory=list)

    # Number of output bits (for result conversion)
    num_output_bits: int = 0

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

    def _convert_user_bindings(
        self,
        bindings: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Convert user-friendly bindings to indexed format.

        Args:
            bindings: User bindings like {"gammas": [0.1, 0.2], "theta": 1.5}

        Returns:
            Indexed bindings like {"gammas[0]": 0.1, "gammas[1]": 0.2, "theta": 1.5}
        """
        if bindings is None:
            return {}

        import numpy as np

        result: dict[str, Any] = {}
        for key, value in bindings.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                # Vector parameter: expand to indexed keys
                for i, v in enumerate(value):
                    result[f"{key}[{i}]"] = v
            else:
                # Scalar parameter: use as-is
                result[key] = value
        return result

    def _validate_bindings(
        self,
        indexed_bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> None:
        """Validate that all required parameters are bound.

        Raises:
            ValueError: If required parameters are missing
        """
        required = {p.name for p in parameter_metadata.parameters}
        provided = set(indexed_bindings.keys())
        missing = required - provided

        if missing:
            array_names = {
                p.array_name for p in parameter_metadata.parameters if p.name in missing
            }
            raise ValueError(
                f"Missing parameter bindings: {sorted(missing)}. "
                f"Provide bindings for: {sorted(array_names)} "
                f"(e.g., bindings={{'{list(array_names)[0] if array_names else 'param'}': [...]}})"
            )

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
        circuit = self.get_first_circuit()
        if circuit is None:
            raise ExecutionError("No quantum circuit to execute")

        # Get parameter metadata
        param_metadata = (
            self.compiled_quantum[0].parameter_metadata
            if self.compiled_quantum
            else ParameterMetadata()
        )

        # Convert and validate bindings if there are parameters
        if param_metadata.parameters:
            indexed_bindings = self._convert_user_bindings(bindings)
            self._validate_bindings(indexed_bindings, param_metadata)
            circuit = executor.bind_parameters(
                circuit, indexed_bindings, param_metadata
            )

        # Execute circuit and get counts
        raw_counts = executor.execute(circuit, shots)

        # Capture references for closure
        compiled_quantum = self.compiled_quantum
        compiled_classical = self.compiled_classical
        execution_order = self.execution_order
        output_refs = self.output_refs

        def convert_counts(raw_counts: dict[str, int]) -> list[tuple[Any, int]]:
            """Convert bitstring counts by executing classical segments."""
            results: list[tuple[Any, int]] = []
            classical_executor = ClassicalExecutor()

            for bitstring, count in raw_counts.items():
                # 1. Create context and load bits
                context = ExecutionContext()
                bits = tuple(int(b) for b in reversed(bitstring))

                # Load measurement results using clbit_map
                if compiled_quantum:
                    clbit_map = compiled_quantum[0].clbit_map
                    for uuid, clbit_idx in clbit_map.items():
                        if clbit_idx < len(bits):
                            context.set(uuid, bits[clbit_idx])

                # 2. Execute ClassicalSegments
                for seg_type, index in execution_order:
                    if seg_type == "classical":
                        segment = compiled_classical[index].segment
                        segment_results = classical_executor.execute(segment, context)
                        context.update(segment_results)

                # 3. Get output values
                if output_refs:
                    output_values = []
                    for ref in output_refs:
                        val = context.get(ref) if context.has(ref) else None
                        if val is None:
                            # Check if this is an array result - collect individual bits
                            array_bits = []
                            i = 0
                            while context.has(f"{ref}_{i}"):
                                array_bits.append(context.get(f"{ref}_{i}"))
                                i += 1
                            if array_bits:
                                val = tuple(array_bits)
                        output_values.append(val)
                    output_values = tuple(output_values)
                    # Return single value if only one output, otherwise tuple
                    if len(output_values) == 1:
                        results.append((output_values[0], count))
                    else:
                        results.append((output_values, count))
                else:
                    # No classical processing, return bits
                    results.append((bits, count))

            return results

        return SampleJob(raw_counts, convert_counts, shots)

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
        # Check if this is an expval-only execution
        if self.compiled_expval and not any(
            seg_type == "classical" for seg_type, _ in self.execution_order
        ):
            return self._run_expval(executor, bindings)

        circuit = self.get_first_circuit()
        if circuit is None:
            raise ExecutionError("No quantum circuit to execute")

        # Get parameter metadata
        param_metadata = (
            self.compiled_quantum[0].parameter_metadata
            if self.compiled_quantum
            else ParameterMetadata()
        )

        # Convert and validate bindings if there are parameters
        if param_metadata.parameters:
            indexed_bindings = self._convert_user_bindings(bindings)
            self._validate_bindings(indexed_bindings, param_metadata)
            circuit = executor.bind_parameters(
                circuit, indexed_bindings, param_metadata
            )

        # Execute circuit with single shot
        raw_counts = executor.execute(circuit, shots=1)

        # Capture references for closure
        compiled_quantum = self.compiled_quantum
        compiled_classical = self.compiled_classical
        execution_order = self.execution_order
        output_refs = self.output_refs

        def convert_result(bitstring: str) -> Any:
            """Convert bitstring by executing classical segments."""
            # 1. Create context and load bits
            context = ExecutionContext()
            bits = tuple(int(b) for b in reversed(bitstring))

            # Load measurement results using clbit_map
            if compiled_quantum:
                clbit_map = compiled_quantum[0].clbit_map
                for uuid, clbit_idx in clbit_map.items():
                    if clbit_idx < len(bits):
                        context.set(uuid, bits[clbit_idx])

            # 2. Execute ClassicalSegments
            classical_executor = ClassicalExecutor()
            for seg_type, index in execution_order:
                if seg_type == "classical":
                    segment = compiled_classical[index].segment
                    segment_results = classical_executor.execute(segment, context)
                    context.update(segment_results)

            # 3. Get output values
            if output_refs:
                output_values = tuple(context.get(ref) for ref in output_refs)
                # Return single value if only one output, otherwise tuple
                if len(output_values) == 1:
                    return output_values[0]
                return output_values
            else:
                # No classical processing, return bits
                return bits

        return RunJob(raw_counts, convert_result)

    def _run_expval(
        self,
        executor: QuantumExecutor[T],
        bindings: dict[str, Any] | None = None,
    ) -> ExpvalJob:
        """Execute expectation value computation.

        This is used when the program has expval segments and returns
        a Float expectation value.

        Args:
            executor: Backend-specific quantum executor with estimate() support.
            bindings: Parameter bindings.

        Returns:
            ExpvalJob that resolves to the expectation value.
        """
        circuit = self.get_first_circuit()
        if circuit is None:
            raise ExecutionError("No quantum circuit to execute for expval")

        # Get parameter metadata
        param_metadata = (
            self.compiled_quantum[0].parameter_metadata
            if self.compiled_quantum
            else ParameterMetadata()
        )

        # Convert and validate bindings if there are parameters
        indexed_bindings = {}
        if param_metadata.parameters:
            indexed_bindings = self._convert_user_bindings(bindings)
            self._validate_bindings(indexed_bindings, param_metadata)
            circuit = executor.bind_parameters(
                circuit, indexed_bindings, param_metadata
            )

        # Execute expval segments in order
        context = ExecutionContext()
        result_value = None

        for seg_type, index in self.execution_order:
            if seg_type == "expval":
                expval_seg = self.compiled_expval[index]

                # Apply qubit mapping to remap Pauli indices to physical qubits
                observable = expval_seg.observable
                if expval_seg.qubit_map:
                    observable = observable.remap_qubits(expval_seg.qubit_map)

                # Use executor's estimate method
                exp_val = executor.estimate(circuit, observable)
                context.set(expval_seg.result_ref, exp_val)
                result_value = exp_val

        # Return the final expval result
        if result_value is None:
            raise ExecutionError("No expectation value computed")

        return ExpvalJob(result_value)

    def get_circuits(self) -> list[T]:
        """Get all quantum circuits in execution order."""
        return [seg.circuit for seg in self.compiled_quantum]

    def get_first_circuit(self) -> T | None:
        """Get the first quantum circuit, or None if no quantum segments."""
        if self.compiled_quantum:
            return self.compiled_quantum[0].circuit
        return None
