"""Orchestration logic for compiled quantum-classical programs.

This module is internal. Users interact with ExecutableProgram.sample()/run().
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Generic, TypeVar

from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueLike,
    resolve_root_array_index,
)
from qamomile.circuit.transpiler.classical_executor import (
    ClassicalExecutor,
    resolve_runtime_array_location,
)
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
_MISSING = object()
_NUMPY_MODULE: Any | None = None
_NUMPY_IMPORT_ATTEMPTED = False


def _get_numpy_module() -> Any | None:
    """Return the cached NumPy module when it is available.

    Returns:
        Any | None: Imported NumPy module, or ``None`` when NumPy is not
            installed.
    """
    global _NUMPY_IMPORT_ATTEMPTED, _NUMPY_MODULE

    if not _NUMPY_IMPORT_ATTEMPTED:
        np_module: Any | None
        try:
            import numpy as np_module
        except ImportError:
            np_module = None
        _NUMPY_MODULE = np_module
        _NUMPY_IMPORT_ATTEMPTED = True
    return _NUMPY_MODULE


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
            aggregated: dict[Hashable, tuple[Any, int]] = {}
            for bitstring, count in raw_counts.items():
                shot_context = context.copy()
                bits = self._bitstring_to_tuple(bitstring)
                self._load_measurements(shot_context, bits)
                self._execute_post_quantum_steps(shot_context, executor, circuit)

                if program.output_values:
                    value = self._resolve_outputs(shot_context)
                else:
                    value = bits
                key = self._sample_result_key(value)
                if key in aggregated:
                    original_value, existing_count = aggregated[key]
                    aggregated[key] = (original_value, existing_count + count)
                else:
                    aggregated[key] = (value, count)
            return list(aggregated.values())

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

            if program.output_values:
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
            self._seed_tuple_input_aliases(context, plan.abi.public_inputs)
        return context

    def _seed_tuple_input_aliases(
        self,
        context: ExecutionContext,
        public_inputs: dict[str, ValueLike],
    ) -> None:
        """Seed runtime aliases for elements of tuple-typed public inputs.

        Tuple dummy inputs use element Values such as ``pair_0`` while user
        bindings are supplied as either ``{"pair": (2, 3)}`` or indexed
        entries such as ``{"pair[0]": 2}``. This helper bridges those names
        through ABI metadata without creating aliases that collide with other
        top-level public inputs.

        Args:
            context (ExecutionContext): Execution context seeded with user
                bindings.
            public_inputs (dict[str, ValueLike]): Runtime-visible public
                inputs from the program ABI.

        Returns:
            None: This method mutates ``context`` in place.
        """
        top_level_names = set(public_inputs)
        for input_name, value in public_inputs.items():
            if isinstance(value, TupleValue):
                self._seed_tuple_elements(
                    context=context,
                    tuple_name=input_name,
                    tuple_value=value,
                    top_level_names=top_level_names,
                )

    def _seed_tuple_elements(
        self,
        context: ExecutionContext,
        tuple_name: str,
        tuple_value: TupleValue,
        top_level_names: set[str],
    ) -> None:
        """Seed aliases for one tuple input's direct elements.

        Args:
            context (ExecutionContext): Execution context seeded with user
                bindings.
            tuple_name (str): Public input name for the tuple.
            tuple_value (TupleValue): Tuple IR value whose elements should be
                aliased.
            top_level_names (set[str]): Names of all public inputs, used to
                avoid alias collisions with separate top-level arguments.

        Returns:
            None: This method mutates ``context`` in place.
        """
        tuple_data = self._resolve_tuple_input_data(context, tuple_name, tuple_value)
        for index, element in enumerate(tuple_value.elements):
            element_data = self._resolve_tuple_element_data(
                context,
                tuple_name,
                index,
                tuple_data,
            )
            if element_data is _MISSING:
                continue
            self._set_context_if_absent(context, element.uuid, element_data)
            for alias in self._tuple_element_aliases(element, top_level_names):
                self._set_context_if_absent(context, alias, element_data)

    def _resolve_tuple_input_data(
        self,
        context: ExecutionContext,
        tuple_name: str,
        tuple_value: TupleValue,
    ) -> Any:
        """Resolve a concrete tuple binding from context when available.

        Args:
            context (ExecutionContext): Execution context seeded with user
                bindings.
            tuple_name (str): Public input name for the tuple.
            tuple_value (TupleValue): Tuple IR value.

        Returns:
            Any: The bound tuple-like object, or a private sentinel when only
                indexed element bindings are available.
        """
        if context.has(tuple_value.uuid):
            return context.get(tuple_value.uuid)
        if context.has(tuple_name):
            return context.get(tuple_name)
        if tuple_value.name and context.has(tuple_value.name):
            return context.get(tuple_value.name)
        return _MISSING

    def _resolve_tuple_element_data(
        self,
        context: ExecutionContext,
        tuple_name: str,
        index: int,
        tuple_data: Any,
    ) -> Any:
        """Resolve one tuple element from whole-tuple or indexed bindings.

        Args:
            context (ExecutionContext): Execution context seeded with user
                bindings.
            tuple_name (str): Public input name for the tuple.
            index (int): Element index to resolve.
            tuple_data (Any): Whole tuple binding or the private missing
                sentinel.

        Returns:
            Any: The concrete element value, or the private missing sentinel.
        """
        if tuple_data is not _MISSING:
            try:
                return tuple_data[index]
            except (IndexError, KeyError, TypeError):
                pass
        indexed_key = f"{tuple_name}[{index}]"
        if context.has(indexed_key):
            return context.get(indexed_key)
        return _MISSING

    def _tuple_element_aliases(
        self,
        element: Value,
        top_level_names: set[str],
    ) -> tuple[str, ...]:
        """Return non-conflicting context aliases for a tuple element.

        Args:
            element (Value): Tuple element IR value.
            top_level_names (set[str]): Names of all public inputs.

        Returns:
            tuple[str, ...]: Alias keys that do not collide with top-level
                public input names.
        """
        aliases: list[str] = []
        for alias in (element.name, element.parameter_name()):
            if alias and alias not in top_level_names and alias not in aliases:
                aliases.append(alias)
        return tuple(aliases)

    def _set_context_if_absent(
        self,
        context: ExecutionContext,
        key: str,
        value: Any,
    ) -> None:
        """Set a context value without overwriting explicit bindings.

        Args:
            context (ExecutionContext): Execution context to update.
            key (str): Context key to seed.
            value (Any): Concrete runtime value.

        Returns:
            None: This method mutates ``context`` in place.
        """
        if not context.has(key):
            context.set(key, value)

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
            return self._resolve_outputs(context) if program.output_values else None

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

        if program.output_values:
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
        output_values = [
            self._resolve_output_value_like(value, context)
            for value in self._program.output_values
        ]
        output_tuple = tuple(output_values)
        if len(output_tuple) == 1:
            return output_tuple[0]
        return output_tuple

    def _resolve_output_value_like(
        self,
        value: ValueLike,
        context: ExecutionContext,
    ) -> Any:
        """Resolve an output IR value from the execution context.

        Args:
            value (ValueLike): Output IR value to resolve.
            context (ExecutionContext): Execution context populated by
                measurement loading and post-quantum classical execution.

        Returns:
            Any: Concrete Python value, or ``None`` when the output cannot be
                found.
        """
        if isinstance(value, TupleValue):
            resolved = self._resolve_direct_output_value(value, context)
            if resolved is not None:
                return resolved
            return tuple(
                self._resolve_output_value_like(element, context)
                for element in value.elements
            )

        if isinstance(value, DictValue):
            resolved = self._resolve_direct_output_value(value, context)
            if resolved is not None:
                return resolved
            return {
                self._resolve_output_value_like(key, context): (
                    self._resolve_output_value_like(entry_value, context)
                )
                for key, entry_value in value.entries
            }

        resolved = self._resolve_context_value_uuid(value.uuid, context)
        if resolved is not None:
            return resolved
        if isinstance(value, ArrayValue):
            array_resolved = self._resolve_array_output(value, context)
            if array_resolved is not None:
                return array_resolved
        if value.is_array_element():
            element_resolved = self._resolve_array_element_output(value, context)
            if element_resolved is not None:
                return element_resolved
        resolved = self._resolve_direct_output_value(value, context)
        if resolved is not None:
            return resolved
        return None

    def _resolve_direct_output_value(
        self,
        value: ValueLike,
        context: ExecutionContext,
    ) -> Any:
        """Resolve a value directly from runtime state or static metadata.

        Args:
            value (ValueLike): IR value-like object to resolve.
            context (ExecutionContext): Execution context populated with
                bindings and results.

        Returns:
            Any: Concrete value, or ``None`` when no direct binding/constant
                exists.
        """
        if context.has(value.uuid):
            return context.get(value.uuid)
        if value.name and context.has(value.name):
            return context.get(value.name)
        if isinstance(value, (Value, ArrayValue)) and value.is_constant():
            return value.get_const()
        if isinstance(value, ArrayValue):
            const_array = value.get_const_array()
            if const_array is not None:
                return const_array
        param_name = value.parameter_name()
        if param_name and context.has(param_name):
            return context.get(param_name)
        return None

    def _resolve_array_output(
        self,
        value: ArrayValue,
        context: ExecutionContext,
    ) -> Any | None:
        """Resolve a whole array output, including runtime-bound views.

        Args:
            value (ArrayValue): Array output value.
            context (ExecutionContext): Execution context populated with
                measured bits and runtime bindings.

        Returns:
            Any | None: Tuple of resolved elements, or ``None`` when the array
                cannot be reconstructed.
        """
        direct = self._resolve_direct_output_value(value, context)
        if direct is not None:
            return direct
        if not value.shape:
            return None
        length = self._resolve_context_int_value(value.shape[0], context)
        if length is None or length < 0:
            return None

        elements: list[Any] = []
        for local_index in range(length):
            resolved_location = resolve_runtime_array_location(
                value,
                (local_index,),
                lambda v: self._resolve_context_int_value(v, context),
            )
            if resolved_location is None:
                return None
            root, root_indices = resolved_location
            element = self._resolve_array_location_output(root, root_indices, context)
            if element is None:
                return None
            elements.append(element)
        return tuple(elements)

    def _resolve_context_value_uuid(
        self,
        uuid: str,
        context: ExecutionContext,
    ) -> Any | None:
        """Resolve a value UUID from context, including indexed carriers.

        Args:
            uuid (str): IR value UUID.
            context (ExecutionContext): Execution context populated by
                execution.

        Returns:
            Any | None: Concrete Python value, tuple reconstructed from indexed
                entries, or ``None`` when no value is available.
        """
        val = context.get(uuid) if context.has(uuid) else None
        if val is None:
            array_bits = []
            i = 0
            while context.has(f"{uuid}_{i}"):
                array_bits.append(context.get(f"{uuid}_{i}"))
                i += 1
            if array_bits:
                val = tuple(array_bits)
        return val

    def _resolve_array_element_output(
        self,
        value: Value,
        context: ExecutionContext,
    ) -> Any | None:
        """Resolve an array-element output through its parent array carrier.

        Args:
            value (Value): Output value that carries ``parent_array``
                metadata.
            context (ExecutionContext): Execution context populated by
                measurement loading.

        Returns:
            Any | None: Concrete element value, or ``None`` when the element
                cannot be resolved.
        """
        parent = value.parent_array
        if parent is None:
            return None
        indices = self._resolve_output_indices(value, context)
        if indices is None:
            return None

        container = self._resolve_array_container_output(parent, context)
        if container is not None:
            if len(indices) == 1:
                return container[indices[0]]
            return container[indices]

        if len(indices) != 1:
            return None

        resolved = resolve_root_array_index(parent, indices[0])
        if resolved is not None:
            root, root_idx = resolved
            root_key = f"{root.uuid}_{root_idx}"
            if context.has(root_key):
                return context.get(root_key)

        parent_key = f"{parent.uuid}_{indices[0]}"
        if context.has(parent_key):
            return context.get(parent_key)

        if parent.name:
            indexed_key = f"{parent.name}[{indices[0]}]"
            if context.has(indexed_key):
                return context.get(indexed_key)

        resolved_location = resolve_runtime_array_location(
            parent,
            indices,
            lambda v: self._resolve_context_int_value(v, context),
        )
        if resolved_location is not None:
            root, root_indices = resolved_location
            return self._resolve_array_location_output(root, root_indices, context)
        return None

    def _resolve_array_container_output(
        self,
        value: ArrayValue,
        context: ExecutionContext,
    ) -> Any:
        """Resolve an array container from output-visible state.

        Args:
            value (ArrayValue): Array value to resolve.
            context (ExecutionContext): Execution context populated with
                bindings and results.

        Returns:
            Any: Concrete array-like container, or ``None`` when not available.
        """
        resolved = self._resolve_context_value_uuid(value.uuid, context)
        if resolved is not None:
            return resolved
        return self._resolve_direct_output_value(value, context)

    def _resolve_array_location_output(
        self,
        array: ArrayValue,
        indices: tuple[int, ...],
        context: ExecutionContext,
    ) -> Any:
        """Resolve an array element from root-coordinate indices.

        Args:
            array (ArrayValue): Root array value.
            indices (tuple[int, ...]): Concrete indices in ``array``
                coordinates.
            context (ExecutionContext): Execution context populated with
                measurements/bindings.

        Returns:
            Any: Concrete element value, or ``None`` when no carrier is
                available.
        """
        container = self._resolve_array_container_output(array, context)
        if container is not None:
            if len(indices) == 1:
                return container[indices[0]]
            return container[indices]
        if len(indices) != 1:
            return None
        root_key = f"{array.uuid}_{indices[0]}"
        if context.has(root_key):
            return context.get(root_key)
        if array.name:
            indexed_key = f"{array.name}[{indices[0]}]"
            if context.has(indexed_key):
                return context.get(indexed_key)
        return None

    def _resolve_output_indices(
        self,
        value: Value,
        context: ExecutionContext,
    ) -> tuple[int, ...] | None:
        """Resolve output array indices to concrete integers.

        Args:
            value (Value): Array-element output value.
            context (ExecutionContext): Execution context that may contain
                runtime index values.

        Returns:
            tuple[int, ...] | None: Tuple of integer indices, or ``None`` if
                any index is unresolved.
        """
        indices: list[int] = []
        for index in value.element_indices:
            if index.is_constant():
                indices.append(int(index.get_const()))
                continue
            if context.has(index.uuid):
                indices.append(int(context.get(index.uuid)))
                continue
            param_name = index.parameter_name()
            if param_name and context.has(param_name):
                indices.append(int(context.get(param_name)))
                continue
            return None
        return tuple(indices)

    def _resolve_context_int_value(
        self,
        value: Value,
        context: ExecutionContext,
    ) -> int | None:
        """Resolve a scalar integer value from execution context.

        Args:
            value (Value): Scalar value to resolve.
            context (ExecutionContext): Execution context containing
                bindings/results.

        Returns:
            int | None: Integer value, or ``None`` when unresolved.
        """
        if value.is_constant():
            return int(value.get_const())
        if context.has(value.uuid):
            return int(context.get(value.uuid))
        if value.name and context.has(value.name):
            return int(context.get(value.name))
        param_name = value.parameter_name()
        if param_name and context.has(param_name):
            return int(context.get(param_name))
        return None

    @classmethod
    def _sample_result_key(cls, value: Any) -> Hashable:
        """Build an exact, hashable aggregation key for a sample result.

        Args:
            value (Any): Typed sample result value.

        Returns:
            Hashable: Key that preserves container structure. Floating-point
                values are not rounded; NaN values are not intentionally
                coalesced because normal exact equality does not make two NaNs
                equal.
        """
        if isinstance(value, tuple):
            return ("tuple", tuple(cls._sample_result_key(v) for v in value))
        if isinstance(value, list):
            return ("list", tuple(cls._sample_result_key(v) for v in value))
        if isinstance(value, dict):
            return (
                "dict",
                frozenset(
                    (cls._sample_result_key(k), cls._sample_result_key(v))
                    for k, v in value.items()
                ),
            )

        np_module = _get_numpy_module()

        if np_module is not None:
            if isinstance(value, np_module.ndarray):
                return (
                    "ndarray",
                    value.dtype.str,
                    tuple(value.shape),
                    tuple(
                        cls._sample_result_key(v) for v in value.reshape(-1).tolist()
                    ),
                )
            if isinstance(value, np_module.generic):
                return (
                    "np_scalar",
                    value.dtype.str,
                    cls._sample_result_key(value.item()),
                )

        if isinstance(value, Hashable):
            return ("scalar", type(value), value)
        return ("identity", id(value))
