"""Orchestration logic for compiled quantum-classical programs.

This module is internal. Users interact with ExecutableProgram.sample()/run().
"""

from __future__ import annotations

import numbers
from typing import Any, Generic, TypeVar

from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueBase,
)
from qamomile.circuit.transpiler.classical_executor import ClassicalExecutor
from qamomile.circuit.transpiler.compiled_segments import (
    CompiledClassicalSegment,
    CompiledExpvalSegment,
    CompiledQuantumSegment,
)
from qamomile.circuit.transpiler.errors import ExecutionError
from qamomile.circuit.transpiler.execution_context import ExecutionContext
from qamomile.circuit.transpiler.job import ExpvalJob, RunJob, SampleJob
from qamomile.circuit.transpiler.param_keys import (
    dict_param_key,
    is_decomposable_dict_binding_key,
    normalize_dict_binding_key,
)
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
_MISSING = object()


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
        """Convert user-friendly bindings to indexed format.

        Sequence values (array parameters) expand positionally to
        ``name[0]``, ``name[1]``, ...; dict values (Dict runtime
        parameters) expand per key to ``name[<key>]`` using the same
        naming the emit pass used when creating the backend parameters.

        Args:
            bindings (dict[str, Any] | None): User-supplied bindings
                keyed by kernel argument name. ``None`` means no
                bindings.

        Returns:
            dict[str, Any]: Flat mapping from backend parameter names to
                scalar values, with non-container bindings passed
                through unchanged.
        """
        if bindings is None:
            return {}

        import numpy as np

        result: dict[str, Any] = {}
        for key, value in bindings.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                for i, v in enumerate(value):
                    result[f"{key}[{i}]"] = v
            elif isinstance(value, dict):
                for k, v in value.items():
                    normalized = normalize_dict_binding_key(k)
                    # Keys that can never match an emitted parameter name
                    # (str, non-integer floats, ...) must not be string-
                    # formatted: "1" would collide with the int key 1
                    # (both format as name[1]) and bind the wrong
                    # parameter. Leaving them out means a genuinely
                    # missing integer key still errors loudly downstream.
                    if not is_decomposable_dict_binding_key(normalized):
                        continue
                    result[dict_param_key(key, normalized)] = v
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
        """Create an execution context seeded with every binding source.

        Args:
            bindings (dict[str, Any] | None): User-facing scalar or container
                bindings keyed by public parameter name.
            indexed_bindings (dict[str, Any]): Flattened array-element
                bindings keyed by backend parameter names.

        Returns:
            ExecutionContext: Context containing user bindings, compile-time
                output constants, and UUID aliases from the program ABI.
        """
        context = ExecutionContext()
        for key, value in (bindings or {}).items():
            context.set(key, value)
        for key, value in indexed_bindings.items():
            context.set(key, value)
        plan = self._program.plan
        if plan is not None:
            for ref, value in plan.abi.constant_outputs.items():
                context.set(ref, value)
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
        """Resolve backend parameter bindings from the current execution context.

        Args:
            context (ExecutionContext): Execution context seeded with
                user bindings (both raw and indexed forms).
            parameter_metadata (ParameterMetadata): The emitted
                circuit's parameter manifest.

        Returns:
            dict[str, Any]: Mapping from backend parameter name to its
                scalar value for this execution.

        Raises:
            ValueError: If any backend parameter has no value in the
                context.
        """
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
                    candidate = context.get(key)
                    # A raw dict is the whole Dict-parameter binding (the
                    # context holds it under the bare dict name); a scalar
                    # backend parameter must never bind to it. Skip so a
                    # genuinely missing per-key entry surfaces as a
                    # missing-binding error instead of a dict-typed angle.
                    if isinstance(candidate, dict):
                        continue
                    bindings[param.name] = candidate
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

                # Pad ``_num_qubits`` to the circuit's width so the
                # backend's observable-to-SparsePauliOp conversion emits
                # a Pauli string of the same length as the circuit's
                # qubit count. Without this, expval over a subset of
                # qubits (``expval(q[1::2], Z(0))``) produces a 1- or
                # 2-qubit observable and the backend estimator rejects
                # it with a "circuit (N) vs observable (k)" mismatch.
                #
                # Critically, we must NOT mutate the user's binding.
                # ``remap_qubits`` returns ``self`` when the qubit_map
                # is empty (identity expval on the full register) — a
                # direct ``hamiltonian._num_qubits = ...`` would then
                # poison the user's binding and break reuse of the
                # same observable on a differently-sized circuit
                # (P1-1 regression).  Clone when the remap was a
                # no-op, then pad the copy.
                circuit_num_qubits = getattr(circuit, "num_qubits", None)
                if (
                    circuit_num_qubits is not None
                    and hamiltonian.num_qubits < circuit_num_qubits
                ):
                    if hamiltonian is expval_seg.hamiltonian:
                        hamiltonian = hamiltonian.copy()
                    hamiltonian._num_qubits = circuit_num_qubits

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
        """Read final output values from execution context.

        Args:
            context (ExecutionContext): Execution state after quantum and
                classical post-processing steps.

        Returns:
            Any: The sole public output or a tuple in public return order.
        """
        output_descriptors = (
            self._program.plan.abi.output_values
            if self._program.plan is not None
            else {}
        )
        output_values = []
        for ref in self._program.output_refs:
            val = context.get(ref) if context.has(ref) else _MISSING
            descriptor = output_descriptors.get(ref)
            if val is _MISSING and descriptor is not None:
                val = self._resolve_structural_output(descriptor, context)
            if val is _MISSING:
                array_bits = []
                i = 0
                while context.has(f"{ref}_{i}"):
                    array_bits.append(context.get(f"{ref}_{i}"))
                    i += 1
                if array_bits:
                    val = tuple(array_bits)
            output_values.append(None if val is _MISSING else val)

        output_tuple = tuple(output_values)
        if len(output_tuple) == 1:
            return output_tuple[0]
        return output_tuple

    def _resolve_structural_output(
        self,
        value: ValueBase,
        context: ExecutionContext,
    ) -> Any:
        """Resolve a public structural output from runtime state.

        Tuple and dictionary descriptors are reconstructed recursively. The
        reconstruction is atomic: if any nested key, value, or tuple element
        is unavailable, the complete container remains unresolved instead of
        exposing a partially populated result.

        Args:
            value (ValueBase): Public output descriptor retained by the ABI.
            context (ExecutionContext): Runtime state containing root arrays or
                per-element measurement keys.

        Returns:
            Any: Resolved scalar, array, tuple, or dictionary contents, or the
                private ``_MISSING`` sentinel when the complete structure is
                not materialized.
        """
        if context.has(value.uuid):
            return context.get(value.uuid)

        if isinstance(value, TupleValue):
            elements: list[Any] = []
            for element in value.elements:
                resolved = self._resolve_structural_output(element, context)
                if resolved is _MISSING:
                    return _MISSING
                elements.append(resolved)
            return tuple(elements)

        if isinstance(value, DictValue):
            entries: dict[Any, Any] = {}
            for key, item in value.entries:
                resolved_key = self._resolve_structural_output(key, context)
                resolved_item = self._resolve_structural_output(item, context)
                if resolved_key is _MISSING or resolved_item is _MISSING:
                    return _MISSING
                try:
                    entries[resolved_key] = resolved_item
                except TypeError:
                    return _MISSING
            return entries

        if isinstance(value, Value) and value.is_array_element():
            indices: list[int] = []
            for index_value in value.element_indices:
                index = self._resolve_output_index(index_value, context)
                if index is None or index < 0:
                    return _MISSING
                indices.append(index)
            if not indices or value.parent_array is None:
                return _MISSING
            location = self._resolve_output_array_location(
                value.parent_array,
                indices[0],
                context,
            )
            if location is None:
                return _MISSING
            root, root_index = location
            return self._read_output_array_element(
                root,
                (root_index, *indices[1:]),
                context,
            )

        if isinstance(value, ArrayValue):
            if not value.shape:
                return _MISSING
            length = self._resolve_output_index(value.shape[0], context)
            if length is None or length < 0:
                return _MISSING
            # A zero-length array has one canonical materialization regardless
            # of whether it is a root, a view, or a branch merge. There are no
            # per-element context keys to discover, so resolve it explicitly.
            if length == 0:
                return ()
            if value.slice_of is None:
                return _MISSING
            elements: list[Any] = []
            for local_index in range(length):
                location = self._resolve_output_array_location(
                    value,
                    local_index,
                    context,
                )
                if location is None:
                    return _MISSING
                root, root_index = location
                element = self._read_output_array_element(
                    root,
                    (root_index,),
                    context,
                )
                if element is _MISSING:
                    return _MISSING
                elements.append(element)
            return tuple(elements)

        if isinstance(value, Value) and value.is_constant():
            return value.get_const()

        return _MISSING

    @staticmethod
    def _resolve_output_index(
        value: Value,
        context: ExecutionContext,
    ) -> int | None:
        """Resolve an output index without display-name fallbacks.

        Args:
            value (Value): Index or shape value to resolve.
            context (ExecutionContext): Runtime state keyed by UUID and public
                parameter provenance.

        Returns:
            int | None: Concrete integer, or None when unresolved or not an
                integral scalar. In particular, booleans, floats, and numeric
                strings are rejected instead of being silently coerced.
        """
        raw: Any = _MISSING
        if context.has(value.uuid):
            raw = context.get(value.uuid)
        elif value.is_constant():
            raw = value.get_const()
        elif value.is_parameter():
            parameter_name = value.parameter_name()
            if parameter_name and context.has(parameter_name):
                raw = context.get(parameter_name)
        if isinstance(raw, bool) or not isinstance(raw, numbers.Integral):
            return None
        return int(raw)

    @classmethod
    def _resolve_output_array_location(
        cls,
        array: ArrayValue,
        index: int,
        context: ExecutionContext,
    ) -> tuple[ArrayValue, int] | None:
        """Compose a view-local output index into root-array coordinates.

        Args:
            array (ArrayValue): Immediate parent array or sliced view.
            index (int): Non-negative index local to ``array``.
            context (ExecutionContext): Runtime state for symbolic slice bounds.

        Returns:
            tuple[ArrayValue, int] | None: Root array and composed index, or
                None when a slice frame is unresolved or invalid.
        """
        current = array
        root_index = index
        while current.slice_of is not None:
            if current.slice_start is None or current.slice_step is None:
                return None
            start = cls._resolve_output_index(current.slice_start, context)
            step = cls._resolve_output_index(current.slice_step, context)
            if start is None or step is None or start < 0 or step <= 0:
                return None
            root_index = start + step * root_index
            current = current.slice_of
        return current, root_index

    @staticmethod
    def _read_output_array_element(
        root: ArrayValue,
        indices: tuple[int, ...],
        context: ExecutionContext,
    ) -> Any:
        """Read one structural output from a root container or clbit key.

        Args:
            root (ArrayValue): Root array owning the requested element.
            indices (tuple[int, ...]): Root-local element indices.
            context (ExecutionContext): Runtime arrays and measurement values.

        Returns:
            Any: Element contents, or the private ``_MISSING`` sentinel.
        """
        if context.has(root.uuid):
            container = context.get(root.uuid)
            try:
                for index in indices:
                    container = container[index]
            except (IndexError, KeyError, TypeError):
                return _MISSING
            return container
        if len(indices) == 1:
            composite_key = f"{root.uuid}_{indices[0]}"
            if context.has(composite_key):
                return context.get(composite_key)
        return _MISSING
