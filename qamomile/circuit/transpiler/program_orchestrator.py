"""Orchestration logic for compiled quantum-classical programs.

This module is internal. Users interact with ExecutableProgram.sample()/run().
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from qamomile.circuit.transpiler.classical_executor import ClassicalExecutor
from qamomile.circuit.transpiler.compiled_segments import (
    CompiledClassicalSegment,
    CompiledExpvalSegment,
    CompiledQuantumSegment,
)
from qamomile.circuit.transpiler.errors import ExecutionError
from qamomile.circuit.transpiler.execution_context import ExecutionContext
from qamomile.circuit.transpiler.job import ExpvalJob, RunJob, SampleJob
from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata
from qamomile.circuit.transpiler.quantum_executor import QuantumExecutor
from qamomile.circuit.transpiler.segments import (
    ClassicalStep,
    ExpvalStep,
    QuantumStep,
)

if __builtins__:  # always True; avoids circular import at module level
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from qamomile.circuit.transpiler.executable import ExecutableProgram

T = TypeVar("T")  # Backend circuit type


class ProgramOrchestrator(Generic[T]):
    """Orchestrates execution of compiled quantum-classical programs.

    This is an internal class that handles the runtime execution logic.
    Users should use ``ExecutableProgram.sample()`` and
    ``ExecutableProgram.run()`` instead.
    """

    def __init__(self, program: ExecutableProgram[T]) -> None:
        self._program = program

    # ------------------------------------------------------------------
    # Public entry points (called by ExecutableProgram facade)
    # ------------------------------------------------------------------

    def sample(
        self,
        executor: QuantumExecutor[T],
        shots: int,
        bindings: dict[str, Any] | None,
    ) -> SampleJob[Any]:
        """Execute with multiple shots and return counts."""
        program = self._program

        if program.plan and any(
            isinstance(step, ExpvalStep) for step in program.plan.steps
        ):
            raise ExecutionError(
                "sample() does not support programs with expectation value segments. "
                "Use run() with an executor that supports estimate()."
            )

        indexed_bindings = self._convert_user_bindings(bindings)
        context = self._create_execution_context(bindings, indexed_bindings)
        circuit = self._prepare_quantum_execution(context, executor)

        raw_counts = executor.execute(circuit, shots)

        def convert_counts(raw_counts: dict[str, int]) -> list[tuple[Any, int]]:
            results: list[tuple[Any, int]] = []
            for bitstring, count in raw_counts.items():
                shot_context = context.copy()
                bits = self._bitstring_to_tuple(bitstring)
                self._load_measurements(shot_context, bits)
                self._execute_post_quantum_steps(shot_context, executor, circuit)

                if program.output_refs:
                    results.append((self._resolve_outputs(shot_context), count))
                else:
                    results.append((bits, count))
            return results

        return SampleJob(raw_counts, convert_counts, shots)

    def run(
        self,
        executor: QuantumExecutor[T],
        bindings: dict[str, Any] | None,
    ) -> RunJob[Any] | ExpvalJob:
        """Execute once and return single result."""
        program = self._program

        indexed_bindings = self._convert_user_bindings(bindings)
        context = self._create_execution_context(bindings, indexed_bindings)
        circuit = self._prepare_quantum_execution(context, executor)

        if program.plan and any(
            isinstance(step, ExpvalStep) for step in program.plan.steps
        ):
            result = self._execute_post_quantum_steps(context, executor, circuit)
            if not any(isinstance(step, ClassicalStep) for step in program.plan.steps):
                return ExpvalJob(float(result))
            return RunJob({"": 1}, lambda _: result)

        raw_counts = executor.execute(circuit, shots=1)

        def convert_result(bitstring: str) -> Any:
            run_context = context.copy()
            bits = self._bitstring_to_tuple(bitstring)
            self._load_measurements(run_context, bits)
            self._execute_post_quantum_steps(run_context, executor, circuit)

            if program.output_refs:
                return self._resolve_outputs(run_context)
            return bits

        return RunJob(raw_counts, convert_result)

    def run_expval(
        self,
        executor: QuantumExecutor[T],
        bindings: dict[str, Any] | None,
    ) -> ExpvalJob:
        """Backward-compatible helper for pure expval execution."""
        indexed_bindings = self._convert_user_bindings(bindings)
        context = self._create_execution_context(bindings, indexed_bindings)
        circuit = self._prepare_quantum_execution(context, executor)
        result_value = self._execute_post_quantum_steps(context, executor, circuit)
        if result_value is None:
            raise ExecutionError("No expectation value computed")
        return ExpvalJob(float(result_value))

    # ------------------------------------------------------------------
    # Binding conversion and validation
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_user_bindings(
        bindings: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Convert user-friendly bindings to indexed format."""
        if bindings is None:
            return {}

        import numpy as np

        result: dict[str, Any] = {}
        for key, value in bindings.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                for i, v in enumerate(value):
                    result[f"{key}[{i}]"] = v
            else:
                result[key] = value
        return result

    @staticmethod
    def _validate_bindings(
        indexed_bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> None:
        """Validate that all required parameters are bound."""
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

    # ------------------------------------------------------------------
    # Execution context and measurement handling
    # ------------------------------------------------------------------

    def _create_execution_context(
        self,
        bindings: dict[str, Any] | None,
        indexed_bindings: dict[str, Any],
    ) -> ExecutionContext:
        """Create execution context seeded with user-provided bindings."""
        context = ExecutionContext()
        for key, value in (bindings or {}).items():
            context.set(key, value)
        for key, value in indexed_bindings.items():
            context.set(key, value)
        plan = self._program.plan
        if plan is not None:
            for name, value in plan.abi.public_inputs.items():
                if bindings and name in bindings:
                    context.set(value.uuid, bindings[name])
                elif name in indexed_bindings:
                    context.set(value.uuid, indexed_bindings[name])
        return context

    @staticmethod
    def _bitstring_to_tuple(bitstring: str) -> tuple[int, ...]:
        """Convert a backend bitstring to little-endian tuple order."""
        return tuple(int(b) for b in reversed(bitstring))

    def _load_measurements(
        self,
        context: ExecutionContext,
        bits: tuple[int, ...],
    ) -> None:
        """Populate measured bit values into the execution context."""
        compiled_quantum = self._program.compiled_quantum
        if not compiled_quantum:
            return

        clbit_map = compiled_quantum[0].clbit_map
        meas_map = compiled_quantum[0].measurement_qubit_map
        for addr, clbit_idx in clbit_map.items():
            bit_idx = meas_map.get(clbit_idx, clbit_idx)
            if bit_idx < len(bits):
                context.set(str(addr), bits[bit_idx])

    # ------------------------------------------------------------------
    # Quantum execution preparation
    # ------------------------------------------------------------------

    def _prepare_quantum_execution(
        self,
        context: ExecutionContext,
        executor: QuantumExecutor[T],
    ) -> T:
        """Execute pre-quantum classical steps and bind the quantum circuit."""
        program = self._program

        if program.plan is None:
            circuit = program.get_first_circuit()
            if circuit is None:
                raise ExecutionError("No quantum circuit to execute")
            return circuit

        classical_executor = ClassicalExecutor()
        for step in program.plan.steps:
            if isinstance(step, ClassicalStep):
                segment = self._get_compiled_classical(step.segment).segment
                segment_results = classical_executor.execute(segment, context)
                context.update(segment_results)
                continue

            if isinstance(step, QuantumStep):
                compiled = self._get_compiled_quantum(step.segment)
                bindings = self._resolve_quantum_bindings(
                    context,
                    compiled.parameter_metadata,
                )
                if compiled.parameter_metadata.parameters:
                    return executor.bind_parameters(
                        compiled.circuit,
                        bindings,
                        compiled.parameter_metadata,
                    )
                return compiled.circuit

        raise ExecutionError("No quantum circuit to execute")

    @staticmethod
    def _resolve_quantum_bindings(
        context: ExecutionContext,
        parameter_metadata: ParameterMetadata,
    ) -> dict[str, Any]:
        """Resolve backend parameter bindings from the current execution context."""
        bindings: dict[str, Any] = {}
        missing: list[str] = []

        for param in parameter_metadata.parameters:
            candidate_keys = []
            if param.source_ref is not None:
                candidate_keys.append(param.source_ref)
            candidate_keys.append(param.name)
            if param.array_name != param.name:
                candidate_keys.append(param.array_name)

            value_found = False
            for key in candidate_keys:
                if context.has(key):
                    bindings[param.name] = context.get(key)
                    value_found = True
                    break
            if not value_found:
                missing.append(param.name)

        if missing:
            raise ValueError(
                f"Missing parameter bindings: {sorted(missing)}. "
                "These values must be provided by user bindings or classical prep."
            )

        return bindings

    # ------------------------------------------------------------------
    # Post-quantum execution
    # ------------------------------------------------------------------

    def _execute_post_quantum_steps(
        self,
        context: ExecutionContext,
        executor: QuantumExecutor[T],
        circuit: T,
    ) -> Any:
        """Execute all program steps after the single quantum step."""
        program = self._program

        if program.plan is None:
            return self._resolve_outputs(context) if program.output_refs else None

        classical_executor = ClassicalExecutor()
        result_value = None
        seen_quantum = False

        for step in program.plan.steps:
            if isinstance(step, QuantumStep):
                seen_quantum = True
                continue
            if not seen_quantum:
                continue

            if isinstance(step, ClassicalStep):
                segment = self._get_compiled_classical(step.segment).segment
                segment_results = classical_executor.execute(segment, context)
                context.update(segment_results)
            elif isinstance(step, ExpvalStep):
                expval_seg = self._get_compiled_expval(step.segment)

                hamiltonian = expval_seg.hamiltonian
                if expval_seg.qubit_map:
                    hamiltonian = hamiltonian.remap_qubits(expval_seg.qubit_map)

                exp_val = executor.estimate(circuit, hamiltonian)
                context.set(expval_seg.result_ref, exp_val)
                result_value = exp_val

        if program.output_refs:
            return self._resolve_outputs(context)
        if result_value is not None:
            return result_value
        return None

    # ------------------------------------------------------------------
    # Segment lookup helpers
    # ------------------------------------------------------------------

    def _get_compiled_quantum(
        self,
        segment: object,
    ) -> CompiledQuantumSegment[T]:
        for compiled in self._program.compiled_quantum:
            if compiled.segment is segment:
                return compiled
        raise ExecutionError("Compiled quantum segment not found")

    def _get_compiled_classical(
        self,
        segment: object,
    ) -> CompiledClassicalSegment:
        for compiled in self._program.compiled_classical:
            if compiled.segment is segment:
                return compiled
        raise ExecutionError("Compiled classical segment not found")

    def _get_compiled_expval(
        self,
        segment: object,
    ) -> CompiledExpvalSegment:
        for compiled in self._program.compiled_expval:
            if compiled.segment is segment:
                return compiled
        raise ExecutionError("Compiled expectation-value segment not found")

    # ------------------------------------------------------------------
    # Output resolution
    # ------------------------------------------------------------------

    def _resolve_outputs(self, context: ExecutionContext) -> Any:
        """Read final output values from execution context."""
        output_values = []
        for ref in self._program.output_refs:
            val = context.get(ref) if context.has(ref) else None
            if val is None:
                array_bits = []
                i = 0
                while context.has(f"{ref}_{i}"):
                    array_bits.append(context.get(f"{ref}_{i}"))
                    i += 1
                if array_bits:
                    val = tuple(array_bits)
            output_values.append(val)

        output_tuple = tuple(output_values)
        if len(output_tuple) == 1:
            return output_tuple[0]
        return output_tuple
