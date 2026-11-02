"""Direct structured QCore-to-HUGR emitter."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, cast

from qamomile._native import compiler as native
from qamomile.compilation import ResultABI
from qamomile.errors import CapabilityError
from qamomile.ir import (
    Block,
    FrozenMapping,
    Function,
    Operation,
    PortableValue,
    Program,
    Value,
    attributes,
    coherent_call_argument_offset,
    thaw_value,
)
from qamomile.layout import reshape_coherent
from qamomile.lowering import RealizationOptions
from qamomile.observable import Hamiltonian
from qamomile.operations import (
    GATES_BY_OPCODE,
    evaluate_classical,
    normalize_aggregate_slice,
)
from qamomile.serialization import _TargetProgramView
from qamomile.types import (
    Bit,
    Bool,
    FixedType,
    QuantumType,
    RecordType,
    ScalarType,
    TypeSpec,
    WordType,
    aggregate_element_types,
    coherent_leaf_widths,
    logical_width,
    strip_stage,
)

MAX_HUGR_LOGICAL_OPERATIONS = 1_000_000
MAX_HUGR_PORTS = 100_000
MAX_HUGR_CONTROLLED_PHASE_TERMS = 100_000
MAX_HUGR_BOOLEAN_LIFT_WIDTH = 4_096
MAX_HUGR_GENERAL_PERMUTATION_BITS = cast(
    int,
    getattr(native, "MAX_GENERAL_BOOLEAN_PERMUTATION_BITS"),
)
MAX_HUGR_STATIC_LOOP_ITERATIONS = 100_000
_HUGR_REAL_RUNTIME_ERROR_CODE = 2_001
_EmissionProgram = Program | _TargetProgramView
_BooleanMonomial = tuple[int, ...]
_BooleanOutputPlan = tuple[_BooleanMonomial, ...]
_BooleanLiftPlan = tuple[
    int,
    tuple[_BooleanOutputPlan, ...],
    tuple[int, ...] | None,
    tuple[tuple[int, int], ...] | None,
]
_NormalExitKind = Literal["yield", "break", "continue"]
_RealRuntimeGuard = Literal["none", "nonzero_divisor", "power_domain"]


@dataclass(frozen=True, slots=True)
class _MeasuredWire:
    """A HUGR wire carrying stable source-measurement provenance."""

    wire: Any
    index: int


@dataclass(frozen=True, slots=True)
class _RealWire:
    """A HUGR float64 wire retaining its verified QCore Real type."""

    wire: Any
    guaranteed_finite: bool = False


@dataclass(frozen=True, slots=True)
class _BooleanCondition:
    """A symbolic measured Boolean retained until HUGR control lowering."""

    opcode: str
    operands: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _DynamicReturn:
    """A HUGR sum selecting structured fallthrough or a callable return."""

    wire: Any
    normal_kind: _NormalExitKind
    normal_types: tuple[TypeSpec, ...]
    normal_provenance: tuple[int | None, ...]
    return_provenance: tuple[int | None, ...]


@dataclass(frozen=True, slots=True)
class _DynamicIfBranch:
    """One emitted conditional arm before its common output row is selected."""

    builder: Any
    lowerer: _DirectHugrLowerer
    kind: str
    flattened: tuple[Any, ...]
    provenance: tuple[int | None, ...]
    flow: _DynamicReturn | None


@dataclass(slots=True)
class _EmissionBudget:
    """Track bounded work and realization provenance for one HUGR package.

    Args:
        remaining_operations (int): Shared limit on emitted logical operations.
        options (RealizationOptions): Exact operation-realization overrides.
        realizations (set[str]): Compiler-selected identities encountered.
    """

    remaining_operations: int = MAX_HUGR_LOGICAL_OPERATIONS
    options: RealizationOptions = RealizationOptions()
    realizations: set[str] = field(default_factory=set)

    def consume_operation(self) -> None:
        """Charge one emitted logical operation before constructing nodes."""

        if self.remaining_operations <= 0:
            raise CapabilityError(
                "Q8682: HUGR emission exceeds the logical-operation budget"
            )
        self.remaining_operations -= 1


class _DirectHugrLowerer:
    """Direct structured QCore emitter for the local TKET HUGR profile."""

    def __init__(
        self,
        program: _EmissionProgram,
        builder: Any,
        *,
        definitions: dict[str, Any],
        adjoint_helpers: dict[str, str],
        boolean_plans: dict[str, _BooleanLiftPlan],
        entry_bindings: Mapping[str, object] | None = None,
        emission_budget: _EmissionBudget | None = None,
        record_results: bool = False,
    ) -> None:
        """Initialize QCore definitions and HUGR emission state."""

        self.program = program
        self.builder = builder
        self.functions = {function.symbol: function for function in program.functions}
        self.definitions = definitions
        self.adjoint_helpers = adjoint_helpers
        self.boolean_plans = boolean_plans
        self.entry_bindings = dict(entry_bindings or {})
        self.emission_budget = emission_budget or _EmissionBudget()
        self.record_results = record_results
        self.previous_measurement: Any | None = None
        self.live_qubits = 0
        self.allocated_qubits = 0
        self.logical_operations = 0
        self.native_operations = 0
        self.next_measurement = 0
        self.result_order: tuple[int, ...] = ()
        self.uses_tail_result_measurements = False
        self.has_dynamic_measurements = False
        self.coherent_controls: list[object] = []
        self.coherent_partitions: dict[int, tuple[int, ...]] = {}
        self.function_result_types: list[TypeSpec] = []
        self.function_value_types: list[dict[int, TypeSpec]] = []

    def _nested_lowerer(self, builder: Any) -> _DirectHugrLowerer:
        """Create a region lowerer with the current emission state."""

        nested = _DirectHugrLowerer(
            self.program,
            builder,
            definitions=self.definitions,
            adjoint_helpers=self.adjoint_helpers,
            boolean_plans=self.boolean_plans,
            emission_budget=self.emission_budget,
            record_results=self.record_results,
        )
        nested.next_measurement = self.next_measurement
        nested.live_qubits = self.live_qubits
        nested.allocated_qubits = self.allocated_qubits
        nested.has_dynamic_measurements = self.has_dynamic_measurements
        nested.coherent_partitions = dict(self.coherent_partitions)
        nested.function_result_types = list(self.function_result_types)
        nested.function_value_types = list(self.function_value_types)
        return nested

    def _reserve_qubits(self, width: int) -> None:
        """Reject an allocation that exceeds the HUGR live-qubit budget."""

        if width < 0 or self.live_qubits + width > MAX_HUGR_PORTS:
            raise CapabilityError("Q8681: HUGR emission exceeds the qubit budget")

    def lower(self, declared_result: ResultABI) -> None:
        """Lower the closed QCore entrypoint into the HUGR main graph.

        Args:
            declared_result: ABI derived from the verified semantic program
                before target-local callable expansion.
        """

        environment: dict[int, object] = {}
        bindings = dict(self.program.const_bindings)
        for parameter in self.program.entry.parameters:
            if parameter.name in self.entry_bindings:
                environment[parameter.value.id] = self.entry_bindings[parameter.name]
            elif parameter.name in bindings:
                environment[parameter.value.id] = thaw_value(bindings[parameter.name])
            elif parameter.has_default:
                environment[parameter.value.id] = thaw_value(parameter.default)
            else:
                raise CapabilityError(
                    f"Q8606: HUGR parameter {parameter.name!r} is not specialized"
                )
        kind, returned = self._emit_function_body(self.program.entry, environment)
        if kind != "return":
            raise CapabilityError("Q8607: HUGR entrypoint did not return normally")
        if declared_result.width:
            if len(returned) != 1:
                raise CapabilityError("Q8651: HUGR sampled entrypoint result arity")
            wires = _flatten_logical_value(returned[0], self.program.entry.result_type)
            derived_result = any(not isinstance(wire, _MeasuredWire) for wire in wires)
            if self.has_dynamic_measurements or derived_result:
                self._emit_tail_result_measurements(wires)
                self.uses_tail_result_measurements = True
                self.result_order = tuple(range(len(wires)))
            else:
                try:
                    self.result_order = tuple(
                        _measurement_index(wire) for wire in wires
                    )
                except TypeError as error:
                    raise CapabilityError(
                        "Q8652: HUGR result bit is not produced by measurement"
                    ) from error
                if self.record_results:
                    self._record_result_bits(wires)
            if len(self.result_order) != declared_result.width:
                raise CapabilityError("Q8653: HUGR result measurement width mismatch")
        elif self.has_dynamic_measurements:
            self.uses_tail_result_measurements = True

    def _emit_tail_result_measurements(self, values: tuple[Any, ...]) -> None:
        """Report dynamic-event results through a fixed trailing measurement row."""

        from tket_exts import quantum

        for value in values:
            self._reserve_qubits(1)
            condition = self._materialize_boolean_condition(value)
            ancilla = self.builder.add_op(quantum.qAlloc)[0]
            self.live_qubits += 1
            self.allocated_qubits += 1
            branch = self.builder.add_if(condition, ancilla)
            true_ancilla = branch.add_op(quantum.X, branch.inputs()[0])[0]
            branch.set_outputs(true_ancilla)
            otherwise = branch.add_else()
            otherwise.set_outputs(*otherwise.inputs())
            measurement = self.builder.add_op(
                quantum.measure,
                branch.conditional_node[0],
            )
            if self.previous_measurement is not None:
                self.builder.hugr.add_order_link(
                    self.previous_measurement,
                    measurement,
                )
            self.previous_measurement = measurement
            post, bit = measurement
            if self.record_results:
                self._record_result_bits((bit,))
            self.builder.add_op(quantum.qFree, post)
            self.live_qubits -= 1
            self.native_operations += 4

    def _record_result_bits(self, values: tuple[Any, ...]) -> None:
        """Emit ordered QSystem records for logical result bits.

        Args:
            values (tuple[Any, ...]): Boolean HUGR wires in source result order.
        """

        from tket_exts import result

        for value in values:
            wire = value.wire if isinstance(value, _MeasuredWire) else value
            record = self.builder.add_op(result.result_bool("qamomile"), wire)
            if self.previous_measurement is not None:
                self.builder.hugr.add_order_link(self.previous_measurement, record)
            self.previous_measurement = record
            self.native_operations += 1

    def lower_function(
        self,
        function: Function,
        inputs: tuple[object, ...],
    ) -> tuple[str, tuple[object, ...]]:
        """Lower one retained QCore callable into its HUGR definition."""

        if len(inputs) != len(function.parameters):
            raise CapabilityError("Q8637: retained HUGR callable input mismatch")
        self.live_qubits = _parameter_qubit_width(function.parameters)
        self.allocated_qubits = self.live_qubits
        environment = {
            parameter.value.id: value
            for parameter, value in zip(function.parameters, inputs, strict=True)
        }
        return self._emit_function_body(function, environment)

    def _emit_function_body(
        self,
        function: Function,
        environment: dict[int, object],
    ) -> tuple[str, tuple[object, ...]]:
        """Emit one function while retaining its result type for nested exits."""

        self.function_result_types.append(function.result_type)
        self.function_value_types.append(_function_value_type_index(function))
        try:
            return self._emit_block(function.body, environment, ())
        finally:
            self.function_value_types.pop()
            self.function_result_types.pop()

    def _emit_block(
        self,
        block: Block,
        outer: dict[int, object],
        arguments: tuple[object, ...],
    ) -> tuple[str, tuple[object, ...]]:
        """Emit one QCore block directly into the current HUGR graph."""

        environment = dict(outer)
        if len(arguments) != len(block.arguments):
            raise CapabilityError("Q8608: HUGR region argument mismatch")
        for definition, value in zip(block.arguments, arguments, strict=True):
            environment[definition.id] = value
        return self._emit_block_from(block, environment, 0)

    def _emit_block_from(
        self,
        block: Block,
        environment: dict[int, object],
        start: int,
    ) -> tuple[str, tuple[object, ...]]:
        """Emit a block continuation from one operation index."""

        for index, operation in enumerate(block.operations[start:], start=start):
            self.emission_budget.consume_operation()
            self.logical_operations += 1
            operands = tuple(environment[value] for value in operation.operands)
            if operation.opcode == "qcore.if":
                kind, values = self._emit_if(operation, operands)
            elif operation.opcode == "qcore.for":
                kind, values = self._emit_for(operation, operands)
            elif operation.opcode == "qcore.while":
                kind, values = self._emit_while(operation, operands)
            else:
                values = self._emit_operation(operation, operands)
                kind = "yield"
            if kind == "dynamic_return":
                [flow] = values
                if not isinstance(flow, _DynamicReturn):
                    raise CapabilityError("Q8666: malformed dynamic HUGR return flow")
                return self._emit_dynamic_return_continuation(
                    block,
                    index + 1,
                    environment,
                    operation,
                    flow,
                )
            if kind != "yield":
                return kind, values
            if len(values) != len(operation.results):
                raise CapabilityError(
                    f"Q8609: {operation.opcode} HUGR result arity mismatch"
                )
            for definition, value in zip(operation.results, values, strict=True):
                environment[definition.id] = value
        return (
            block.terminator.kind,
            tuple(environment[value] for value in block.terminator.operands),
        )

    def _emit_operation(
        self,
        operation: Operation,
        operands: tuple[object, ...],
    ) -> tuple[object, ...]:
        """Emit one QCore operation into TKET extension operations.

        Args:
            operation (Operation): Verified QCore operation to emit.
            operands (tuple[object, ...]): Bound host values and emitted SDK wires.

        Returns:
            tuple[object, ...]: Values corresponding to the operation's results.

        Raises:
            CapabilityError: If a selector or SDK representation is invalid.
        """

        opcode = operation.opcode
        if opcode == "qcore.constant":
            value = cast(PortableValue, operation.attribute("value"))
            return (thaw_value(value),)
        if opcode == "qcore.stage_lift":
            return operands
        if opcode == "qcore.reshape":
            partitions = self.coherent_partitions.get(operation.operands[0])
            if partitions is None:
                partitions = _coherent_value_leaf_widths(operands[0])
            try:
                reshaped = reshape_coherent(
                    operands[0],
                    operation.results[0].type,
                    is_leaf=_is_quantum_wire_tuple,
                    unpack_leaf=lambda value: cast(tuple[Any, ...], value),
                    pack_leaf=lambda wires: wires,
                )
            except ValueError as error:
                if partitions is None:
                    raise CapabilityError(
                        f"Q8678: invalid coherent reshape: {error}"
                    ) from error
                try:
                    reshaped = reshape_coherent(
                        operands[0],
                        operation.results[0].type,
                        is_leaf=_is_quantum_wire_tuple,
                        unpack_leaf=lambda value: cast(tuple[Any, ...], value),
                        pack_leaf=lambda wires: wires,
                        target_leaf_widths=partitions,
                    )
                except ValueError as partition_error:
                    raise CapabilityError(
                        f"Q8678: invalid coherent reshape: {partition_error}"
                    ) from partition_error
            if len(operation.results) == 1 and partitions is not None:
                self.coherent_partitions[operation.results[0].id] = partitions
            return (reshaped,)
        if opcode in {"tuple", "array", "qcore.tuple"}:
            return (tuple(operands),)
        if opcode == "qcore.unpack":
            if len(operands) != 1:
                raise CapabilityError("Q8610: HUGR cannot unpack a non-product")
            aggregate = operands[0]
            if isinstance(aggregate, tuple):
                return tuple(aggregate)
            operand_type = (
                self.function_value_types[-1].get(operation.operands[0])
                if self.function_value_types
                else None
            )
            record_type = (
                strip_stage(operand_type) if operand_type is not None else None
            )
            if isinstance(aggregate, Mapping) and isinstance(record_type, RecordType):
                try:
                    return tuple(aggregate[name] for name, _ in record_type.fields)
                except KeyError as error:
                    raise CapabilityError(
                        "Q8610: HUGR record value disagrees with its declared fields"
                    ) from error
            raise CapabilityError("Q8610: HUGR cannot unpack a non-product")
        if opcode.startswith("classical."):
            if opcode in {
                "classical.index",
                "classical.project",
                "classical.slice",
                "classical.update",
            } and isinstance(operands[0], tuple):
                return (self._emit_dynamic_classical_structure(operation, operands),)
            if any(_contains_hugr_wire(value) for value in operands):
                structured = self._emit_dynamic_classical_structure(
                    operation,
                    operands,
                )
                if structured is not None:
                    return (structured,)
                boolean = self._emit_dynamic_boolean(operation, operands)
                if boolean is not None:
                    return (boolean,)
                dynamic_real = self._emit_dynamic_real(operation, operands)
                if dynamic_real is not None:
                    return (dynamic_real,)
                name = opcode.removeprefix("classical.")
                raise CapabilityError(
                    f"Q8611: dynamic classical HUGR operation {name!r} is unsupported"
                )
            return (
                evaluate_classical(
                    opcode,
                    operands,
                    attributes=operation.attributes,
                ),
            )

        if opcode == "qcore.call":
            callee = operation.attribute("callee")
            if not isinstance(callee, str) or callee not in self.functions:
                raise CapabilityError(f"Q8612: unresolved HUGR callable {callee!r}")
            function = self.functions[callee]
            definition = self.definitions.get(callee)
            if definition is not None and not self.coherent_controls:
                flattened = tuple(
                    wire
                    for parameter, value in zip(
                        function.parameters, operands, strict=True
                    )
                    for wire in self._flatten_value(value, parameter.value.type)
                )
                call = self.builder.call(definition, *flattened)
                self.native_operations += 1
                return (
                    _reconstruct_hugr_value(
                        function.result_type,
                        tuple(call),
                    ),
                )
            kind, values = self._emit_function_body(
                function,
                {
                    parameter.value.id: value
                    for parameter, value in zip(
                        function.parameters, operands, strict=True
                    )
                },
            )
            if kind != "return":
                raise CapabilityError("Q8613: inlined HUGR callable did not return")
            return values
        if opcode == "qcore.phase_by":
            return self._emit_phase_by(operation, operands)
        if opcode == "qcore.xor_by":
            return self._emit_xor_by(operation, operands)
        if opcode == "qcore.permute_by":
            return self._emit_permute_by(operation, operands, adjoint=False)
        if opcode == "qcore.control_call":
            return self._emit_control_call(operation, operands)
        if opcode == "qcore.select_call":
            return self._emit_select_call(operation, operands)
        if opcode == "qcore.adjoint":
            return self._emit_adjoint(operation, operands)
        if opcode in {"qcore.adjoint_call"}:
            raise CapabilityError(
                "Q8617: modular HUGR adjoint lowering is not implemented"
            )
        if opcode.startswith("quantum."):
            return self._emit_quantum(operation, operands)
        raise CapabilityError(f"Q8618: HUGR cannot lower {opcode!r}")

    def _emit_phase_by(
        self,
        operation: Operation,
        operands: tuple[object, ...],
    ) -> tuple[object, ...]:
        """Emit a native-proved Boolean phase polynomial without reinterpreting it."""

        if len(operands) != 1:
            raise CapabilityError("Q8685: malformed qcore.phase_by operand ABI")
        predicate = operation.attribute("predicate")
        plan = self.boolean_plans.get(predicate) if isinstance(predicate, str) else None
        if plan is None:
            raise CapabilityError(f"Q8685: missing native phase plan for {predicate!r}")
        width, outputs, _permutation_order, _basis_transpositions = plan
        if len(outputs) != 1:
            raise CapabilityError("Q8685: phase lift requires one Boolean output")
        [monomials] = outputs
        wires = list(_require_wires(operands[0], "qcore.phase_by"))
        if len(wires) != width:
            raise CapabilityError("Q8685: phase predicate width mismatch")
        for monomial in monomials:
            if any(index < 0 or index >= width for index in monomial):
                raise CapabilityError("Q8685: phase monomial index is out of bounds")
            selected = [wires[index] for index in monomial]
            control_count = len(self.coherent_controls)
            combined = [*self.coherent_controls, *selected]
            if not combined:
                _emit_global_phase(self.builder, math.pi)
                self.native_operations += 1
                continue
            updated = self._emit_controlled_phase(math.pi, combined)
            self.coherent_controls[:] = updated[:control_count]
            for index, wire in zip(monomial, updated[control_count:], strict=True):
                wires[index] = wire
        return (tuple(wires),)

    def _emit_xor_by(
        self,
        operation: Operation,
        operands: tuple[object, ...],
    ) -> tuple[object, ...]:
        """Emit one native-proved Boolean XOR embedding."""

        if (
            len(operands) != 1
            or not isinstance(operands[0], tuple)
            or len(operands[0]) != 2
        ):
            raise CapabilityError("Q8685: malformed qcore.xor_by operand ABI")
        source = list(_require_wires(operands[0][0], "qcore.xor_by source"))
        target = list(_require_wires(operands[0][1], "qcore.xor_by target"))
        predicate = operation.attribute("predicate")
        plan = self.boolean_plans.get(predicate) if isinstance(predicate, str) else None
        if plan is None:
            raise CapabilityError(
                f"Q8685: missing native Boolean plan for {predicate!r}"
            )
        width, outputs, _permutation_order, _basis_transpositions = plan
        if len(source) != width or len(target) != len(outputs):
            raise CapabilityError("Q8685: Boolean XOR width mismatch")
        for target_index, monomials in enumerate(outputs):
            for monomial in monomials:
                if any(index < 0 or index >= width for index in monomial):
                    raise CapabilityError(
                        "Q8685: Boolean monomial index is out of bounds"
                    )
                outer_count = len(self.coherent_controls)
                self.coherent_controls.extend(source[index] for index in monomial)
                if self.coherent_controls:
                    [emitted_target] = self._emit_controlled_gate(
                        "x",
                        target,
                        (),
                        {"target": target_index},
                        adjoint=False,
                    )
                    target = list(_require_wires(emitted_target, "qcore.xor_by target"))
                    for index, wire in zip(
                        monomial,
                        self.coherent_controls[outer_count:],
                        strict=True,
                    ):
                        source[index] = wire
                    self.coherent_controls[:] = self.coherent_controls[:outer_count]
                else:
                    from tket_exts import quantum

                    target[target_index] = self.builder.add_op(
                        quantum.X,
                        target[target_index],
                    )[0]
                    self.native_operations += 1
        return ((tuple(source), tuple(target)),)

    def _emit_permute_by(
        self,
        operation: Operation,
        operands: tuple[object, ...],
        *,
        adjoint: bool,
    ) -> tuple[object, ...]:
        """Emit one native-proved triangular or finite Boolean permutation.

        Args:
            operation (Operation): Verified abstract permutation operation.
            operands (tuple[object, ...]): One owned coherent state.
            adjoint (bool): Whether to reverse the proved realization.

        Returns:
            tuple[object, ...]: The updated owned coherent state.

        Raises:
            CapabilityError: If the checked native plan or operand ABI is invalid.
        """

        if len(operands) != 1:
            raise CapabilityError("Q8685: malformed qcore.permute_by operand ABI")
        wires = list(_require_wires(operands[0], "qcore.permute_by"))
        predicate = operation.attribute("predicate")
        plan = self.boolean_plans.get(predicate) if isinstance(predicate, str) else None
        if plan is None:
            raise CapabilityError(
                f"Q8685: missing native Boolean plan for {predicate!r}"
            )
        width, outputs, permutation_order, basis_transpositions = plan
        if len(wires) != width or len(outputs) != width:
            raise CapabilityError("Q8685: Boolean permutation width mismatch")
        if permutation_order is not None and basis_transpositions is None:
            if len(permutation_order) != width:
                raise CapabilityError("Q8685: Boolean permutation order mismatch")
            targets = (
                reversed(permutation_order) if adjoint else iter(permutation_order)
            )
            for target_index in targets:
                monomials = outputs[target_index]
                terms = reversed(monomials) if adjoint else iter(monomials)
                for monomial in terms:
                    if monomial == (target_index,):
                        continue
                    if target_index in monomial or any(
                        index < 0 or index >= width for index in monomial
                    ):
                        raise CapabilityError("Q8685: invalid Boolean permutation term")
                    outer_count = len(self.coherent_controls)
                    self.coherent_controls.extend(wires[index] for index in monomial)
                    wires = self._emit_permutation_x(wires, target_index)
                    for index, wire in zip(
                        monomial,
                        self.coherent_controls[outer_count:],
                        strict=True,
                    ):
                        wires[index] = wire
                    self.coherent_controls[:] = self.coherent_controls[:outer_count]
        elif permutation_order is None and basis_transpositions is not None:
            transpositions = (
                reversed(basis_transpositions)
                if adjoint
                else iter(basis_transpositions)
            )
            for left, right in transpositions:
                wires = self._emit_basis_transposition(wires, left, right)
        else:
            raise CapabilityError("Q8685: ambiguous Boolean permutation proof")
        return (tuple(wires),)

    def _emit_basis_transposition(
        self,
        wires: list[object],
        left: int,
        right: int,
    ) -> list[object]:
        """Emit one canonical Gray-path transposition of two basis states.

        Args:
            wires (list[object]): Current state wires in logical bit order.
            left (int): First basis-state index.
            right (int): Second basis-state index.

        Returns:
            list[object]: Updated state wires after the transposition.

        Raises:
            CapabilityError: If the checked native projection is out of range.
        """

        width = len(wires)
        state_limit = 1 << width
        if not 0 <= left < state_limit or not 0 <= right < state_limit or left == right:
            raise CapabilityError("Q8685: invalid Boolean basis transposition")
        differing = left ^ right
        current = left
        gates: list[tuple[int, int]] = []
        for target in range(width):
            if differing & (1 << target):
                gates.append((target, current))
                current ^= 1 << target
        if current != right:
            raise CapabilityError("Q8685: invalid Boolean transposition path")
        for target, basis_state in (*gates, *reversed(gates[:-1])):
            wires = self._emit_basis_state_flip(wires, target, basis_state)
        return wires

    def _emit_basis_state_flip(
        self,
        wires: list[object],
        target: int,
        basis_state: int,
    ) -> list[object]:
        """Flip one target when every other lane matches one basis state.

        Args:
            wires (list[object]): Current state wires in logical bit order.
            target (int): Target lane of the multi-controlled X.
            basis_state (int): Required computational-basis control pattern.

        Returns:
            list[object]: Updated state wires after the controlled flip.
        """

        zero_controls = [
            index
            for index in range(len(wires))
            if index != target and not basis_state & (1 << index)
        ]
        for index in zero_controls:
            wires = self._emit_permutation_x(wires, index)
        outer_count = len(self.coherent_controls)
        control_indices = [index for index in range(len(wires)) if index != target]
        self.coherent_controls.extend(wires[index] for index in control_indices)
        wires = self._emit_permutation_x(wires, target)
        for index, wire in zip(
            control_indices,
            self.coherent_controls[outer_count:],
            strict=True,
        ):
            wires[index] = wire
        self.coherent_controls[:] = self.coherent_controls[:outer_count]
        for index in reversed(zero_controls):
            wires = self._emit_permutation_x(wires, index)
        return wires

    def _emit_permutation_x(
        self,
        wires: list[object],
        target: int,
    ) -> list[object]:
        """Emit X under the currently active outer/internal controls.

        Args:
            wires (list[object]): Current state wires in logical bit order.
            target (int): Target lane for the X operation.

        Returns:
            list[object]: Updated state wires after the X operation.
        """

        if self.coherent_controls:
            [emitted_state] = self._emit_controlled_gate(
                "x",
                wires,
                (),
                {"target": target},
                adjoint=False,
            )
            return list(_require_wires(emitted_state, "qcore.permute_by state"))
        from tket_exts import quantum

        wires[target] = self.builder.add_op(quantum.X, wires[target])[0]
        self.native_operations += 1
        return wires

    def _emit_dynamic_classical_structure(
        self,
        operation: Operation,
        operands: tuple[object, ...],
    ) -> object | None:
        """Project fixed classical aggregate structure without interpreting it.

        Args:
            operation (Operation): Verified QCore structural operation.
            operands (tuple[object, ...]): Reconstructed aggregate and scalar values.

        Returns:
            object | None: Selected component, exact length, sliced or updated SDK
            wire tuple, or ``None`` when the opcode is not structural.

        Raises:
            CapabilityError: If the verified operation reaches emission with
                an unsupported dynamic index or malformed representation.
        """

        name = operation.opcode.removeprefix("classical.")
        if name == "len":
            if len(operands) != 1 or not isinstance(operands[0], tuple):
                raise CapabilityError("Q8611: malformed dynamic aggregate length")
            return len(operands[0])
        if name in {"slice", "update"}:
            if not operands or not isinstance(operands[0], tuple):
                raise CapabilityError("Q8611: malformed dynamic aggregate operation")
            aggregate = operands[0]
            if name == "slice":
                supplied = iter(operands[1:])
                boundaries: list[int | None] = []
                for bound in ("start", "stop", "step"):
                    if operation.attribute(bound) is None:
                        boundaries.append(None)
                        continue
                    value = next(supplied, None)
                    if type(value) is not int:
                        raise CapabilityError(
                            "Q8611: HUGR slice requires resolved Const boundaries"
                        )
                    boundaries.append(value)
                try:
                    positions = normalize_aggregate_slice(len(aggregate), *boundaries)
                except ValueError as error:
                    raise CapabilityError(f"Q8611: {error}") from error
                return tuple(aggregate[position] for position in positions)
            if len(operands) != 3 or type(operands[1]) is not int:
                raise CapabilityError(
                    "Q8611: HUGR cannot dynamically address an aggregate update"
                )
            try:
                # Route SDK wires through the native immutable update equation;
                # the replacement sentinel is outside the source position row.
                positions = normalize_aggregate_slice(len(aggregate), None, None, None)
                route = evaluate_classical(
                    "classical.update", (positions, operands[1], -1)
                )
            except (ValueError, TypeError) as error:
                raise CapabilityError(f"Q8611: {error}") from error
            if (
                not isinstance(route, tuple)
                or len(route) != len(aggregate)
                or any(
                    type(position) is not int or not -1 <= position < len(aggregate)
                    for position in route
                )
            ):
                raise CapabilityError("Q8611: malformed native aggregate update route")
            return tuple(
                operands[2] if position == -1 else aggregate[position]
                for position in route
            )
        if name not in {"index", "project"}:
            return None
        if not operands or not isinstance(operands[0], tuple):
            raise CapabilityError("Q8611: malformed dynamic aggregate projection")
        aggregate = operands[0]
        if name == "index":
            if (
                len(operands) != 2
                or isinstance(operands[1], bool)
                or not isinstance(operands[1], int)
            ):
                raise CapabilityError(
                    "Q8611: HUGR cannot dynamically address a classical aggregate"
                )
            index = operands[1]
        else:
            if len(operands) != 1:
                raise CapabilityError("Q8611: malformed dynamic aggregate projection")
            index = operation.attribute("position")
            if isinstance(index, bool) or not isinstance(index, int):
                raise CapabilityError("Q8611: malformed aggregate projection position")
        position = _native_aggregate_position(len(aggregate), index)
        if position is None:
            raise CapabilityError(
                f"Q4303: aggregate index {index} is outside extent {len(aggregate)}"
            )
        return aggregate[position]

    def _emit_dynamic_boolean(
        self,
        operation: Operation,
        operands: tuple[object, ...],
    ) -> object | None:
        """Lower the verified dynamic Bool subset to HUGR conditions.

        Args:
            operation: Verified QCore classical operation.
            operands: Reconstructed static values and HUGR wires.

        Returns:
            A deferred HUGR Boolean condition, raw Bool wire, or ``None`` when
            the operation does not have a Bool lowering.

        Raises:
            CapabilityError: If supported Bool metadata is malformed.
        """

        if len(operation.results) != 1 or strip_stage(
            operation.results[0].type
        ) not in {Bit, Bool}:
            return None
        name = operation.opcode.removeprefix("classical.")
        if name == "compare_chain":
            operators = operation.attribute("operators")
            if (
                not isinstance(operators, tuple)
                or len(operators) + 1 != len(operands)
                or any(
                    operator not in {"eq", "ne", "in", "not_in"}
                    for operator in operators
                )
            ):
                return None
            comparisons: list[object] = []
            for operator, left, right in zip(
                operators,
                operands[:-1],
                operands[1:],
                strict=True,
            ):
                if operator in {"in", "not_in"}:
                    if not isinstance(right, tuple):
                        return None
                    members: list[object] = []
                    for candidate in right:
                        equality = self._emit_dynamic_value_equality(left, candidate)
                        if equality is None:
                            return None
                        members.append(equality)
                    comparison: object = (
                        _fold_boolean_condition("or", tuple(members))
                        if members
                        else False
                    )
                    if operator == "not_in":
                        comparison = _BooleanCondition("not", (comparison,))
                else:
                    equality = self._emit_dynamic_value_equality(left, right)
                    if equality is None:
                        return None
                    comparison = (
                        equality
                        if operator == "eq"
                        else _BooleanCondition("not", (equality,))
                    )
                comparisons.append(comparison)
            return _fold_boolean_condition("and", tuple(comparisons))
        if name == "cast":
            if len(operands) != 1 or operation.attribute("target") != "Bool":
                raise CapabilityError("Q8611: malformed dynamic Bool cast")
            [operand] = operands
            if isinstance(operand, _RealWire):
                from hugr.std.float import FloatVal

                zero = self.builder.load(FloatVal(0.0))[0]
                self.native_operations += 1
                equal = self.builder.add_op(
                    _float_operation("feq", 2, boolean=True), operand.wire, zero
                )[0]
                return _BooleanCondition("not", (equal,))
            if _is_boolean_operand(operand):
                return operand
            return None
        if name in {"not", "invert"}:
            if len(operands) != 1 or not _is_boolean_operand(operands[0]):
                return None
            return _BooleanCondition("not", operands)
        if name in {"and", "or"}:
            if not operands or any(not _is_boolean_operand(item) for item in operands):
                return None
            return _fold_boolean_condition(cast(Literal["and", "or"], name), operands)
        if name in {"and_bits", "or_bits", "xor"}:
            if len(operands) != 2 or any(
                not _is_boolean_operand(item) for item in operands
            ):
                return None
            if name == "and_bits":
                return _BooleanCondition("and", operands)
            if name == "or_bits":
                return _BooleanCondition("or", operands)
            return _BooleanCondition("not", (_boolean_equality(*operands),))
        return None

    def _emit_dynamic_value_equality(
        self,
        left: object,
        right: object,
    ) -> object | None:
        """Project equality for one verified HUGR-representable value pair.

        Args:
            left: Reconstructed left QCore value.
            right: Reconstructed right QCore value.

        Returns:
            A deferred Boolean condition, raw Bool wire, or ``None`` when a
            leaf has no representation-preserving HUGR equality.
        """

        if isinstance(left, tuple) or isinstance(right, tuple):
            if (
                not isinstance(left, tuple)
                or not isinstance(right, tuple)
                or len(left) != len(right)
            ):
                return None
            comparisons: list[object] = []
            for left_item, right_item in zip(left, right, strict=True):
                comparison = self._emit_dynamic_value_equality(left_item, right_item)
                if comparison is None:
                    return None
                comparisons.append(comparison)
            return (
                _fold_boolean_condition("and", tuple(comparisons))
                if comparisons
                else True
            )
        if isinstance(left, _RealWire) or isinstance(right, _RealWire):
            left_wire = _real_operand_wire(self.builder, left)
            right_wire = _real_operand_wire(self.builder, right)
            self.native_operations += 1
            return self.builder.add_op(
                _float_operation("feq", 2, boolean=True),
                left_wire,
                right_wire,
            )[0]
        if _is_boolean_operand(left) and _is_boolean_operand(right):
            return _boolean_equality(left, right)
        if not _contains_hugr_wire(left) and not _contains_hugr_wire(right):
            result = evaluate_classical(
                "classical.compare_chain",
                (left, right),
                attributes=attributes(operators=("eq",)),
            )
            return result if isinstance(result, bool) else None
        return None

    def _emit_dynamic_real(
        self,
        operation: Operation,
        operands: tuple[object, ...],
    ) -> object | None:
        """Project the sound QCore binary64 subset to HUGR float operations.

        Args:
            operation: Verified QCore classical operation.
            operands: Reconstructed static values and typed HUGR wires.

        Returns:
            The typed HUGR result, or ``None`` when the opcode is outside the
            deliberately supported binary64 subset.

        Raises:
            CapabilityError: If a selected operation receives a non-Real
                dynamic operand or malformed comparison metadata.
        """

        name = operation.opcode.removeprefix("classical.")
        if name in {"add", "sub", "mul", "truediv", "pow"}:
            if len(operands) != 2:
                raise CapabilityError("Q8611: malformed dynamic Real arithmetic")
            runtime_guard, guaranteed_finite = _native_real_operation_facts(
                operation.opcode,
                operands,
            )
            left, right = (
                _real_operand_wire(self.builder, operand) for operand in operands
            )
            if runtime_guard == "nonzero_divisor":
                from hugr.std.float import FloatVal

                zero = self.builder.load(FloatVal(0.0))[0]
                equal_zero = self.builder.add_op(
                    _float_operation("feq", 2, boolean=True),
                    right,
                    zero,
                )[0]
                nonzero = self._materialize_boolean_condition(
                    _BooleanCondition("not", (equal_zero,))
                )
                right = _guard_real_wire(
                    self.builder,
                    right,
                    nonzero,
                    "Q8611: Real division by zero",
                )
                self.native_operations += 4
            elif runtime_guard == "power_domain":
                right = self._guard_real_power_domain(left, right)
            result = self.builder.add_op(
                _float_operation(
                    {"truediv": "fdiv", "pow": "fpow"}.get(name, f"f{name}"),
                    2,
                ),
                left,
                right,
            )[0]
            self.native_operations += 1
            return _RealWire(result, guaranteed_finite)
        if name in {"neg", "pos"}:
            if len(operands) != 1:
                raise CapabilityError("Q8611: malformed dynamic Real unary operation")
            [operand] = operands
            _, guaranteed_finite = _native_real_operation_facts(
                operation.opcode,
                operands,
            )
            if not isinstance(operand, _RealWire):
                wire = _real_operand_wire(self.builder, operand)
                operand = _RealWire(wire, guaranteed_finite)
            if name == "pos":
                return _RealWire(operand.wire, guaranteed_finite)
            result = self.builder.add_op(
                _float_operation("fneg", 1),
                operand.wire,
            )[0]
            self.native_operations += 1
            return _RealWire(result, guaranteed_finite)
        if name != "compare_chain":
            return None

        operators = operation.attribute("operators")
        comparison_names = {
            "eq": "feq",
            "ne": "feq",
            "lt": "flt",
            "le": "fle",
            "gt": "fgt",
            "ge": "fge",
        }
        if (
            not isinstance(operators, tuple)
            or len(operators) + 1 != len(operands)
            or any(operator not in comparison_names for operator in operators)
        ):
            raise CapabilityError("Q8611: malformed dynamic Real comparison chain")
        wires = tuple(_real_operand_wire(self.builder, operand) for operand in operands)
        comparisons: list[object] = []
        for operator, left, right in zip(
            operators,
            wires[:-1],
            wires[1:],
            strict=True,
        ):
            comparison = self.builder.add_op(
                _float_operation(comparison_names[operator], 2, boolean=True),
                left,
                right,
            )[0]
            comparisons.append(
                _BooleanCondition("not", (comparison,))
                if operator == "ne"
                else comparison
            )
            self.native_operations += 1
        if not comparisons:
            raise CapabilityError("Q8611: empty dynamic Real comparison chain")
        result = comparisons[0]
        for comparison in comparisons[1:]:
            result = _BooleanCondition("and", (result, comparison))
        return result

    def _guard_real_power_domain(self, base: Any, exponent: Any) -> Any:
        """Guard the QCore binary64 power domain before HUGR ``fpow``.

        Args:
            base: Runtime binary64 base wire.
            exponent: Runtime binary64 exponent wire.

        Returns:
            The exponent wire carried through the successful guard path.

        Notes:
            Exact integrality is derived from the IEEE 754 binary64 exponent
            and fraction fields. The dynamic shift is masked to ``0..63`` so
            every emitted integer operation has defined behavior.
        """

        from hugr.std.float import FloatVal
        from hugr.std.int import IntVal

        def integer(value: int) -> Any:
            return self.builder.load(IntVal(value, 6))[0]

        largest_value = float.fromhex("0x1.fffffffffffffp+1023")
        smallest = self.builder.load(FloatVal(-largest_value))[0]
        largest = self.builder.load(FloatVal(largest_value))[0]
        zero_real = self.builder.load(FloatVal(0.0))[0]
        exponent_finite = _BooleanCondition(
            "and",
            (
                self.builder.add_op(
                    _float_operation("fge", 2, boolean=True),
                    exponent,
                    smallest,
                )[0],
                self.builder.add_op(
                    _float_operation("fle", 2, boolean=True),
                    exponent,
                    largest,
                )[0],
            ),
        )

        bits = self.builder.add_op(_float64_to_int64_bits_operation(), exponent)[0]
        raw_exponent = self.builder.add_op(
            _int64_operation("ishr", 2),
            bits,
            integer(52),
        )[0]
        exponent_bits = self.builder.add_op(
            _int64_operation("iand", 2),
            raw_exponent,
            integer(0x7FF),
        )[0]
        large_integral = self.builder.add_op(
            _int64_operation("ige_u", 2, boolean=True),
            exponent_bits,
            integer(1075),
        )[0]
        at_least_one = self.builder.add_op(
            _int64_operation("ige_u", 2, boolean=True),
            exponent_bits,
            integer(1023),
        )[0]
        below_large = self.builder.add_op(
            _int64_operation("ilt_u", 2, boolean=True),
            exponent_bits,
            integer(1075),
        )[0]
        shift_delta = self.builder.add_op(
            _int64_operation("isub", 2),
            exponent_bits,
            integer(1023),
        )[0]
        safe_shift = self.builder.add_op(
            _int64_operation("iand", 2),
            shift_delta,
            integer(63),
        )[0]
        shifted_fraction = self.builder.add_op(
            _int64_operation("ishl", 2),
            bits,
            safe_shift,
        )[0]
        fractional_bits = self.builder.add_op(
            _int64_operation("iand", 2),
            shifted_fraction,
            integer((1 << 52) - 1),
        )[0]
        middle_integral = self.builder.add_op(
            _int64_operation("ieq", 2, boolean=True),
            fractional_bits,
            integer(0),
        )[0]
        signless = self.builder.add_op(
            _int64_operation("ishl", 2),
            bits,
            integer(1),
        )[0]
        is_zero = self.builder.add_op(
            _int64_operation("ieq", 2, boolean=True),
            signless,
            integer(0),
        )[0]
        is_integral = _BooleanCondition(
            "or",
            (
                large_integral,
                _BooleanCondition(
                    "or",
                    (
                        is_zero,
                        _BooleanCondition(
                            "and",
                            (
                                _BooleanCondition("and", (at_least_one, below_large)),
                                middle_integral,
                            ),
                        ),
                    ),
                ),
            ),
        )

        zero_base = self.builder.add_op(
            _float_operation("feq", 2, boolean=True),
            base,
            zero_real,
        )[0]
        negative_exponent = self.builder.add_op(
            _float_operation("flt", 2, boolean=True),
            exponent,
            zero_real,
        )[0]
        negative_base = self.builder.add_op(
            _float_operation("flt", 2, boolean=True),
            base,
            zero_real,
        )[0]
        invalid = _BooleanCondition(
            "and",
            (
                exponent_finite,
                _BooleanCondition(
                    "or",
                    (
                        _BooleanCondition("and", (zero_base, negative_exponent)),
                        _BooleanCondition(
                            "and",
                            (
                                negative_base,
                                _BooleanCondition("not", (is_integral,)),
                            ),
                        ),
                    ),
                ),
            ),
        )
        valid = self._materialize_boolean_condition(
            _BooleanCondition("not", (invalid,))
        )
        guarded = _guard_real_wire(
            self.builder,
            exponent,
            valid,
            "Q8611: Real power is outside its domain",
        )
        self.native_operations += 21
        return guarded

    def _require_angle(self, value: object, label: str) -> float | _RealWire:
        """Require a finite static angle or guard a runtime Real angle.

        Args:
            value: Candidate QCore angle value.
            label: Source-facing operation label used in diagnostics.

        Returns:
            A finite static angle or a runtime Real wire guarded before use.

        Raises:
            CapabilityError: If the value is not a Real scalar or a static
                value is nonfinite.
        """

        if isinstance(value, _RealWire):
            if value.guaranteed_finite:
                return value

            from hugr.std.float import FloatVal

            largest_value = float.fromhex("0x1.fffffffffffffp+1023")
            smallest = self.builder.load(FloatVal(-largest_value))[0]
            largest = self.builder.load(FloatVal(largest_value))[0]
            at_least_smallest = self.builder.add_op(
                _float_operation("fge", 2, boolean=True),
                value.wire,
                smallest,
            )[0]
            at_most_largest = self.builder.add_op(
                _float_operation("fle", 2, boolean=True),
                value.wire,
                largest,
            )[0]
            finite = self._materialize_boolean_condition(
                _BooleanCondition("and", (at_least_smallest, at_most_largest))
            )
            guarded = _guard_real_wire(
                self.builder,
                value.wire,
                finite,
                f"Q8628: {label} must remain finite",
            )
            self.native_operations += 5
            return _RealWire(guarded, True)

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CapabilityError(f"Q8628: {label} must be a real scalar")
        angle = float(value)
        if not math.isfinite(angle):
            raise CapabilityError(f"Q8628: {label} must be finite")
        return angle

    def _emit_control_call(
        self,
        operation: Operation,
        operands: tuple[object, ...],
    ) -> tuple[object, ...]:
        """Inline a unitary body while threading its coherent control wires.

        Args:
            operation (Operation): Verified coherent-call operation.
            operands (tuple[object, ...]): Control value and callee arguments.

        Returns:
            tuple[object, ...]: Updated controls and packed callee result.

        Raises:
            CapabilityError: If the checked call ABI or basis state is invalid.
        """

        from tket_exts import quantum

        callee_name = operation.attribute("callee")
        callee = (
            self.functions.get(callee_name) if isinstance(callee_name, str) else None
        )
        if callee is None or len(operands) < 2:
            raise CapabilityError("Q8616: malformed HUGR controlled callable")
        controls = list(_require_wires(operands[0], "qcore.control_call"))
        argument_offset = coherent_call_argument_offset(operation, callee)
        if argument_offset is None:
            raise CapabilityError("Q8616: malformed HUGR controlled callable")
        component: int | None = None
        selected_controls = controls
        if argument_offset == 2:
            component = _require_int(operands[1], "controlled-call component")
            _validate_selectors("qcore.control_call", len(controls), (component,))
            selected_controls = [controls[component]]
        state = operation.attribute("control_state")
        if state is None:
            state = (1 << len(selected_controls)) - 1
        if (
            isinstance(state, bool)
            or not isinstance(state, int)
            or state < 0
            or state >= 1 << len(selected_controls)
        ):
            raise CapabilityError("Q8616: invalid HUGR controlled-call basis state")
        zero_indices = tuple(
            index for index in range(len(selected_controls)) if not (state >> index) & 1
        )
        for index in zero_indices:
            selected_controls[index] = self.builder.add_op(
                quantum.X, selected_controls[index]
            )[0]
            self.native_operations += 1
        outer = self.coherent_controls
        self.coherent_controls = [*outer, *selected_controls]
        try:
            kind, values = self._emit_function_body(
                callee,
                {
                    parameter.value.id: value
                    for parameter, value in zip(
                        callee.parameters, operands[argument_offset:], strict=True
                    )
                },
            )
            if kind != "return" or len(values) != 1:
                raise CapabilityError("Q8616: controlled HUGR callable did not return")
            updated_outer = self.coherent_controls[: len(outer)]
            selected_controls = self.coherent_controls[len(outer) :]
        finally:
            self.coherent_controls = outer
        outer[:] = updated_outer
        for index in reversed(zero_indices):
            selected_controls[index] = self.builder.add_op(
                quantum.X, selected_controls[index]
            )[0]
            self.native_operations += 1
        if component is not None:
            controls[component] = selected_controls[0]
        else:
            controls = selected_controls
        return ((tuple(controls), values[0]),)

    def _emit_select_call(
        self,
        operation: Operation,
        operands: tuple[object, ...],
    ) -> tuple[object, ...]:
        """Lower total selection by composing the controlled-call primitive."""

        if len(operands) < 2:
            raise CapabilityError("Q8616: malformed HUGR selected callable")
        index = tuple(_require_wires(operands[0], "qcore.select_call"))
        raw_callees = operation.attribute("callees")
        if (
            not isinstance(raw_callees, tuple)
            or len(raw_callees) != 1 << len(index)
            or any(
                not isinstance(name, str) or name not in self.functions
                for name in raw_callees
            )
        ):
            raise CapabilityError("Q8616: malformed HUGR selected-call case table")
        current = operands[1]
        for control_state, callee_name in enumerate(raw_callees):
            synthetic = Operation(
                "qcore.control_call",
                operation.operands,
                (),
                attributes(callee=callee_name, control_state=control_state),
                span=operation.span,
            )
            packed = self._emit_control_call(
                synthetic,
                (index, current, *operands[2:]),
            )
            index, current = cast(tuple[tuple[object, ...], object], packed[0])
        return ((index, current),)

    def _emit_if(
        self,
        operation: Operation,
        operands: tuple[object, ...],
    ) -> tuple[str, tuple[object, ...]]:
        """Emit one conditional and preserve a common structured exit kind."""

        condition = operands[0]
        if _is_hugr_wire(condition):
            return self._emit_dynamic_if(operation, condition, operands[1:])
        region = operation.regions[0 if bool(condition) else 1]
        return self._emit_block(region.entry, {}, operands[1:])

    def _emit_dynamic_if(
        self,
        operation: Operation,
        condition: object,
        captures: tuple[object, ...],
    ) -> tuple[str, tuple[object, ...]]:
        """Preserve a measurement-backed conditional and its callable exits."""

        from hugr import ops, tys

        if not _is_hugr_wire(condition) or len(operation.regions) != 2:
            raise CapabilityError("Q8614: malformed dynamic HUGR if")
        argument_types = tuple(
            argument.type for argument in operation.regions[0].entry.arguments
        )
        if len(argument_types) != len(captures):
            raise CapabilityError("Q8646: HUGR if capture arity mismatch")
        flat_captures = tuple(
            wire
            for value, type_spec in zip(captures, argument_types, strict=True)
            for wire in self._flatten_value(value, type_spec)
        )
        condition_wire = self._materialize_boolean_condition(condition)
        true_builder = self.builder.add_if(condition_wire, *flat_captures)
        branch_builder = true_builder
        branches: list[_DynamicIfBranch] = []
        parent_allocations = self.allocated_qubits
        branch_allocation_bounds: list[int] = []
        normal_types = tuple(result.type for result in operation.results)
        for index, region in enumerate(operation.regions):
            branch = self._nested_lowerer(branch_builder)
            branch.allocated_qubits = parent_allocations
            arguments = _copy_provenance(
                captures,
                _reconstruct_block_arguments(
                    region.entry,
                    tuple(branch_builder.inputs()),
                ),
            )
            kind, values = branch._emit_block(region.entry, {}, arguments)
            flow: _DynamicReturn | None = None
            if kind == "yield":
                value_types = normal_types
            elif kind == "return" and self.function_result_types:
                value_types = (self.function_result_types[-1],)
            elif kind == "dynamic_return" and self.function_result_types:
                [candidate] = values
                if not isinstance(candidate, _DynamicReturn):
                    raise CapabilityError(
                        "Q8647: malformed dynamic HUGR if return flow"
                    )
                if (
                    candidate.normal_kind != "yield"
                    or candidate.normal_types != normal_types
                ):
                    raise CapabilityError(
                        "Q8647: dynamic HUGR if return flow has the wrong normal row"
                    )
                flow = candidate
                values = ()
                value_types = ()
            else:
                raise CapabilityError(
                    "Q8647: early exit in a dynamic HUGR if is unsupported"
                )
            if len(values) != len(value_types):
                raise CapabilityError("Q8647: dynamic HUGR if exit arity mismatch")
            flattened_values = tuple(
                wire
                for value, type_spec in zip(values, value_types, strict=True)
                for wire in _flatten_logical_value(value, type_spec)
            )
            provenance = tuple(
                item.index if isinstance(item, _MeasuredWire) else None
                for item in flattened_values
            )
            branches.append(
                _DynamicIfBranch(
                    branch_builder,
                    branch,
                    kind,
                    flattened_values,
                    provenance,
                    flow,
                )
            )
            self.logical_operations += branch.logical_operations
            self.native_operations += branch.native_operations
            branch_allocation_bounds.append(branch.allocated_qubits)
            if branch.live_qubits != self.live_qubits:
                raise CapabilityError(
                    "Q8657: HUGR branches have different live-qubit counts"
                )
            if index == 0:
                branch_builder = true_builder.add_else()
        self.allocated_qubits = max(
            branch_allocation_bounds, default=parent_allocations
        )
        if any(
            branch.lowerer.next_measurement != self.next_measurement
            for branch in branches
        ):
            self.has_dynamic_measurements = True
            self.previous_measurement = true_builder.conditional_node
        self.next_measurement = max(
            self.next_measurement,
            *(branch.lowerer.next_measurement for branch in branches),
        )
        branch_kinds = {branch.kind for branch in branches}
        if len(branch_kinds) == 1 and "yield" in branch_kinds:
            for branch in branches:
                branch.builder.set_outputs(
                    *(
                        branch.lowerer._materialize_boolean_condition(item)
                        for item in branch.flattened
                    )
                )
            conditional = true_builder.conditional_node
            reconstructed = _reconstruct_typed_values(
                operation.results, tuple(conditional)
            )
            provenance = _common_provenance(branch.provenance for branch in branches)
            if provenance:
                reconstructed = _apply_flat_provenance(
                    reconstructed,
                    operation.results,
                    provenance,
                )
            return "yield", reconstructed
        result_type = self.function_result_types[-1]
        if len(branch_kinds) == 1 and "return" in branch_kinds:
            for branch in branches:
                branch.builder.set_outputs(
                    *(
                        branch.lowerer._materialize_boolean_condition(item)
                        for item in branch.flattened
                    )
                )
            conditional = true_builder.conditional_node
            returned = _reconstruct_hugr_value(result_type, tuple(conditional))
            provenance = _common_provenance(branch.provenance for branch in branches)
            if provenance:
                returned = _apply_value_provenance(
                    returned,
                    result_type,
                    provenance,
                )
            return "return", (returned,)
        normal_row = [
            item for type_spec in normal_types for item in _hugr_type_row(type_spec)
        ]
        return_row = _hugr_type_row(result_type)
        flow_type = tys.Either(normal_row, return_row)
        normal_provenance: list[tuple[int | None, ...]] = []
        return_provenance: list[tuple[int | None, ...]] = []
        for branch in branches:
            if branch.kind == "yield":
                tagged = branch.builder.add_op(
                    ops.Tag(0, flow_type),
                    *(
                        branch.lowerer._materialize_boolean_condition(item)
                        for item in branch.flattened
                    ),
                )[0]
                normal_provenance.append(branch.provenance)
            elif branch.kind == "return":
                tagged = branch.builder.add_op(
                    ops.Tag(1, flow_type),
                    *(
                        branch.lowerer._materialize_boolean_condition(item)
                        for item in branch.flattened
                    ),
                )[0]
                return_provenance.append(branch.provenance)
            else:
                if branch.flow is None:
                    raise CapabilityError("Q8647: missing dynamic HUGR if return flow")
                tagged = branch.flow.wire
                normal_provenance.append(branch.flow.normal_provenance)
                return_provenance.append(branch.flow.return_provenance)
            branch.builder.set_outputs(tagged)
        conditional = true_builder.conditional_node
        return "dynamic_return", (
            _DynamicReturn(
                conditional[0],
                "yield",
                normal_types,
                _common_provenance(normal_provenance),
                _common_provenance(return_provenance),
            ),
        )

    def _materialize_boolean_condition(self, condition: object) -> Any:
        """Lower a measured Boolean expression to one HUGR sum wire."""

        from hugr import val

        if isinstance(condition, bool) or (
            isinstance(condition, int)
            and not isinstance(condition, bool)
            and condition in {0, 1}
        ):
            return self.builder.load(val.TRUE if bool(condition) else val.FALSE)[0]
        if not isinstance(condition, _BooleanCondition):
            return _raw_wire(condition)

        expected_arity = 1 if condition.opcode == "not" else 2
        if (
            condition.opcode not in {"not", "and", "or"}
            or len(condition.operands) != expected_arity
        ):
            raise CapabilityError("Q8660: malformed measured Boolean expression")
        raw_operands = tuple(
            self._materialize_boolean_condition(operand)
            for operand in condition.operands
        )
        carried = raw_operands[1:]
        true_builder = self.builder.add_if(raw_operands[0], *carried)
        if condition.opcode == "not":
            true_builder.set_outputs(true_builder.load(val.FALSE)[0])
        elif condition.opcode == "and":
            true_builder.set_outputs(true_builder.inputs()[0])
        else:
            true_builder.set_outputs(true_builder.load(val.TRUE)[0])
        else_builder = true_builder.add_else()
        if condition.opcode == "not":
            else_builder.set_outputs(else_builder.load(val.TRUE)[0])
        elif condition.opcode == "and":
            else_builder.set_outputs(else_builder.load(val.FALSE)[0])
        else:
            else_builder.set_outputs(else_builder.inputs()[0])
        return true_builder.conditional_node[0]

    def _flatten_value(
        self,
        value: object,
        type_spec: TypeSpec,
    ) -> tuple[Any, ...]:
        """Flatten one value and materialize deferred Boolean expressions."""

        return tuple(
            self._materialize_boolean_condition(item)
            for item in _flatten_logical_value(value, type_spec)
        )

    def _emit_while(
        self,
        operation: Operation,
        operands: tuple[object, ...],
    ) -> tuple[str, tuple[object, ...]]:
        """Expand a resolvable while loop at the HUGR emission boundary.

        Args:
            operation (Operation): Verified condition, body, and else regions.
            operands (tuple[object, ...]): Initial loop carries.

        Returns:
            tuple[str, tuple[object, ...]]: Structured exit and outgoing values.

        Raises:
            CapabilityError: If the condition needs runtime loop support, the
                iteration budget is exceeded, or an exit is unsupported.
        """
        carries = operands
        iterations = 0
        while True:
            kind, values = self._emit_block(operation.regions[0].entry, {}, carries)
            if kind != "yield":
                raise CapabilityError("Q8687: malformed HUGR while condition exit")
            condition, *successors = values
            carries = tuple(successors)
            if _is_hugr_wire(condition):
                raise CapabilityError(
                    "Q8687: dynamic HUGR while requires runtime loop support"
                )
            if not bool(condition):
                return self._emit_block(operation.regions[2].entry, {}, carries)
            if iterations >= MAX_HUGR_STATIC_LOOP_ITERATIONS:
                raise CapabilityError(
                    "Q8682: HUGR while-loop exceeds the static iteration budget"
                )
            iterations += 1
            kind, values = self._emit_block(operation.regions[1].entry, {}, carries)
            if kind in {"yield", "continue"}:
                carries = values
            elif kind == "break":
                return "yield", values
            elif kind == "return":
                return kind, values
            else:
                raise CapabilityError("Q8688: unsupported HUGR while body exit")

    def _emit_for(
        self,
        operation: Operation,
        operands: tuple[object, ...],
    ) -> tuple[str, tuple[object, ...]]:
        """Late-unroll a statically bounded QCore for region."""

        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in operands[:3]
        ):
            raise CapabilityError("Q8619: HUGR for bounds must be specialized")
        start, stop, step = (_require_int(value, "for bound") for value in operands[:3])
        if step == 0:
            raise CapabilityError("Q8655: HUGR for-loop step cannot be zero")
        carries = tuple(operands[3:])
        broke = False
        iterations = range(start, stop, step)
        try:
            iteration_count = len(iterations)
        except OverflowError as error:
            raise CapabilityError(
                "Q8682: HUGR for-loop exceeds the static iteration budget"
            ) from error
        if iteration_count > MAX_HUGR_STATIC_LOOP_ITERATIONS:
            raise CapabilityError(
                "Q8682: HUGR for-loop exceeds the static iteration budget"
            )
        if operation.attribute("reverse", False):
            iterations = reversed(iterations)
        for index in iterations:
            kind, values = self._emit_block(
                operation.regions[0].entry, {}, (index, *carries)
            )
            if kind in {"yield", "continue"}:
                carries = values
            elif kind == "break":
                carries = values
                broke = True
                break
            elif kind == "return":
                return kind, values
            else:
                raise CapabilityError("Q8620: malformed HUGR for body exit")
        if not broke and len(operation.regions) == 2:
            kind, values = self._emit_block(operation.regions[1].entry, {}, carries)
            if kind == "return":
                return kind, values
            if kind != "yield":
                raise CapabilityError("Q8621: malformed HUGR for else")
            carries = values
        return "yield", carries

    def _emit_dynamic_return_continuation(
        self,
        block: Block,
        start: int,
        environment: dict[int, object],
        operation: Operation,
        flow: _DynamicReturn,
    ) -> tuple[str, tuple[object, ...]]:
        """Compose an early structured return with the remaining continuation."""

        if not self.function_result_types:
            raise CapabilityError("Q8666: dynamic return has no callable result type")
        expected_normal_types = tuple(result.type for result in operation.results)
        if flow.normal_kind != "yield" or flow.normal_types != expected_normal_types:
            raise CapabilityError("Q8666: malformed dynamic HUGR return flow")
        result_type = self.function_result_types[-1]
        conditional = self.builder.add_conditional(flow.wire)
        normal_builder = conditional.add_case(0)
        normal = self._nested_lowerer(normal_builder)
        normal_values = _reconstruct_typed_values(
            operation.results,
            tuple(normal_builder.inputs()),
        )
        if flow.normal_provenance:
            normal_values = _apply_flat_provenance(
                normal_values,
                operation.results,
                flow.normal_provenance,
            )
        normal_environment = dict(environment)
        for definition, value in zip(
            operation.results,
            normal_values,
            strict=True,
        ):
            normal_environment[definition.id] = value
        kind, returned = normal._emit_block_from(
            block,
            normal_environment,
            start,
        )
        if kind not in {"return", "yield", "break", "continue"}:
            raise CapabilityError(
                "Q8667: dynamic return continuation has an unsupported exit"
            )
        if kind == "return":
            if len(returned) != 1:
                raise CapabilityError("Q8667: malformed dynamic return continuation")
            normal_types = (result_type,)
        else:
            definitions = _block_terminator_definitions(block)
            if len(returned) != len(definitions):
                raise CapabilityError("Q8667: malformed yielding return continuation")
            normal_types = tuple(definition.type for definition in definitions)
        flat_normal = tuple(
            wire
            for value, type_spec in zip(returned, normal_types, strict=True)
            for wire in normal._flatten_value(value, type_spec)
        )
        normal_provenance = tuple(
            item.index if isinstance(item, _MeasuredWire) else None
            for value, type_spec in zip(returned, normal_types, strict=True)
            for item in _flatten_logical_value(value, type_spec)
        )
        returned_builder = conditional.add_case(1)
        if kind == "return":
            normal_builder.set_outputs(*flat_normal)
            returned_builder.set_outputs(*returned_builder.inputs())
        else:
            from hugr import ops, tys

            normal_row = [
                item for type_spec in normal_types for item in _hugr_type_row(type_spec)
            ]
            return_row = _hugr_type_row(result_type)
            propagated_type = tys.Either(normal_row, return_row)
            normal_flow = normal_builder.add_op(
                ops.Tag(0, propagated_type),
                *flat_normal,
            )[0]
            returned_flow = returned_builder.add_op(
                ops.Tag(1, propagated_type),
                *returned_builder.inputs(),
            )[0]
            normal_builder.set_outputs(normal_flow)
            returned_builder.set_outputs(returned_flow)
        if normal.next_measurement != self.next_measurement:
            self.has_dynamic_measurements = True
            self.previous_measurement = conditional.parent_node
        self.next_measurement = max(
            self.next_measurement,
            normal.next_measurement,
        )
        self.logical_operations += normal.logical_operations
        self.native_operations += normal.native_operations
        self.allocated_qubits = max(self.allocated_qubits, normal.allocated_qubits)
        if kind != "return":
            return "dynamic_return", (
                _DynamicReturn(
                    conditional.parent_node[0],
                    cast(_NormalExitKind, kind),
                    normal_types,
                    normal_provenance,
                    flow.return_provenance,
                ),
            )
        selected = _reconstruct_hugr_value(
            result_type,
            tuple(conditional.parent_node),
        )
        if normal_provenance == flow.return_provenance:
            selected = _apply_value_provenance(
                selected,
                result_type,
                normal_provenance,
            )
        return "return", (selected,)

    def _emit_quantum(
        self,
        operation: Operation,
        operands: tuple[object, ...],
    ) -> tuple[object, ...]:
        """Emit one abstract quantum operation into TKET HUGR operations.

        Args:
            operation (Operation): Verified quantum allocation, gate, or measurement.
            operands (tuple[object, ...]): Bound values and owned SDK wires.

        Returns:
            tuple[object, ...]: Updated quantum or measurement results.

        Raises:
            CapabilityError: If selectors or the requested operation are unsupported.
        """

        from tket_exts import quantum

        opcode = operation.opcode
        if opcode == "quantum.alloc":
            width = logical_width(operation.results[0].type)
            if width is None:
                width = _require_int(operands[0], "allocation width")
            if width <= 0:
                raise CapabilityError("Q8656: HUGR allocation width must be positive")
            self._reserve_qubits(width)
            wires = tuple(self.builder.add_op(quantum.qAlloc)[0] for _ in range(width))
            self.live_qubits += width
            self.allocated_qubits += width
            self.native_operations += width
            return (wires,)
        if opcode == "quantum.prepare":
            prepared_bits = _literal_bits(
                operands[0],
                width=logical_width(operation.results[0].type),
            )
            self._reserve_qubits(len(prepared_bits))
            wires = [self.builder.add_op(quantum.qAlloc)[0] for _ in prepared_bits]
            self.live_qubits += len(prepared_bits)
            self.allocated_qubits += len(prepared_bits)
            self.native_operations += len(prepared_bits)
            for index, bit in enumerate(prepared_bits):
                if bit:
                    wires[index] = self.builder.add_op(quantum.X, wires[index])[0]
                    self.native_operations += 1
            return (tuple(wires),)
        if opcode == "quantum.measure":
            wires = _require_wires(operands[0], opcode)
            measured_bits: list[object] = []
            for wire in wires:
                measurement = self.builder.add_op(quantum.measure, wire)
                if self.previous_measurement is not None:
                    self.builder.hugr.add_order_link(
                        self.previous_measurement, measurement
                    )
                self.previous_measurement = measurement
                post, bit = measurement
                bit = _MeasuredWire(bit, self.next_measurement)
                self.next_measurement += 1
                self.builder.add_op(quantum.qFree, post)
                self.live_qubits -= 1
                measured_bits.append(bit)
                self.native_operations += 2
            return (
                _reconstruct_hugr_value(
                    operation.results[0].type,
                    tuple(measured_bits),
                ),
            )
        if opcode in {"quantum.measure_x", "quantum.measure_y", "quantum.measure_z"}:
            wires = list(_require_wires(operands[0], opcode))
            positional, keywords = _gate_arguments(operation, operands[1:])
            target = _require_int(
                keywords.get("target", positional[0] if positional else 0),
                "measurement target",
            )
            _validate_selectors(opcode, len(wires), (target,))
            wire = wires[target]
            before = (
                (quantum.H,)
                if opcode == "quantum.measure_x"
                else (quantum.Sdg, quantum.H)
                if opcode == "quantum.measure_y"
                else ()
            )
            after = (
                (quantum.H,)
                if opcode == "quantum.measure_x"
                else (quantum.H, quantum.S)
                if opcode == "quantum.measure_y"
                else ()
            )
            for gate in before:
                wire = self.builder.add_op(gate, wire)[0]
                self.native_operations += 1
            measurement = self.builder.add_op(quantum.measure, wire)
            if self.previous_measurement is not None:
                self.builder.hugr.add_order_link(self.previous_measurement, measurement)
            self.previous_measurement = measurement
            wire, bit = measurement
            bit = _MeasuredWire(bit, self.next_measurement)
            self.next_measurement += 1
            self.native_operations += 1
            for gate in after:
                wire = self.builder.add_op(gate, wire)[0]
                self.native_operations += 1
            wires[target] = wire
            return ((bit, tuple(wires)),)
        if opcode == "quantum.measure_reset":
            measured_bits: list[object] = []
            reset_wires: list[object] = []
            for wire in _require_wires(operands[0], opcode):
                measurement = self.builder.add_op(quantum.measure, wire)
                if self.previous_measurement is not None:
                    self.builder.hugr.add_order_link(
                        self.previous_measurement, measurement
                    )
                self.previous_measurement = measurement
                post, bit = measurement
                measured_bits.append(_MeasuredWire(bit, self.next_measurement))
                self.next_measurement += 1
                reset_wires.append(self.builder.add_op(quantum.reset, post)[0])
                self.native_operations += 2
            return (
                _reconstruct_hugr_value(
                    operation.results[0].type,
                    (*measured_bits, *reset_wires),
                ),
            )
        if opcode == "quantum.discard":
            for wire in _require_wires(operands[0], opcode):
                measurement = self.builder.add_op(quantum.measure, wire)
                if self.previous_measurement is not None:
                    self.builder.hugr.add_order_link(
                        self.previous_measurement,
                        measurement,
                    )
                self.previous_measurement = measurement
                post, _ = measurement
                self.next_measurement += 1
                self.builder.add_op(quantum.qFree, post)
                self.live_qubits -= 1
                self.native_operations += 2
            return (None,)
        if opcode == "quantum.release":
            wires = _require_wires(operands[0], opcode)
            for wire in wires:
                self.builder.add_op(quantum.qFree, wire)
                self.native_operations += 1
            self.live_qubits -= len(wires)
            return ()
        if opcode == "quantum.reset":
            wires = tuple(
                self.builder.add_op(quantum.reset, wire)[0]
                for wire in _require_wires(operands[0], opcode)
            )
            self.native_operations += len(wires)
            return (wires,)
        if opcode == "quantum.unprepare":
            wires = list(_require_wires(operands[0], opcode))
            bits = _literal_bits(operands[1])
            for index, bit in enumerate(bits):
                if bit:
                    wires[index] = self.builder.add_op(quantum.X, wires[index])[0]
                    self.native_operations += 1
            for wire in wires:
                self.builder.add_op(quantum.qFree, wire)
                self.native_operations += 1
            self.live_qubits -= len(wires)
            return ()
        return self._emit_gate(opcode, operands, operation, adjoint=False)

    def _emit_adjoint(
        self,
        operation: Operation,
        operands: tuple[object, ...],
    ) -> tuple[object, ...]:
        """Emit a verified primitive or coherent-transform inverse."""

        original = operation.attribute("original")
        original_attributes = operation.attribute("original_attributes")
        if not isinstance(original, str) or not isinstance(
            original_attributes, FrozenMapping
        ):
            raise CapabilityError("Q8627: malformed HUGR primitive adjoint")
        synthetic = Operation(
            original,
            (),
            operation.results,
            original_attributes.entries,
            span=operation.span,
        )
        if original.startswith("quantum."):
            return self._emit_gate(original, operands, synthetic, adjoint=True)
        if original == "qcore.phase_by":
            return self._emit_phase_by(synthetic, operands)
        if original == "qcore.xor_by":
            return self._emit_xor_by(synthetic, operands)
        if original == "qcore.permute_by":
            return self._emit_permute_by(synthetic, operands, adjoint=True)
        if original not in {"qcore.control_call", "qcore.select_call"}:
            raise CapabilityError(
                f"Q8627: unsupported HUGR adjoint origin {original!r}"
            )
        if not operands or not isinstance(operands[0], tuple) or len(operands[0]) != 2:
            raise CapabilityError("Q8627: malformed coherent HUGR adjoint state")
        selector, target = operands[0]
        original_values = dict(original_attributes.entries)
        if original == "qcore.control_call":
            callee_name = original_values.get("callee")
            helper_name = (
                self.adjoint_helpers.get(callee_name)
                if isinstance(callee_name, str)
                else None
            )
            callee = (
                self.functions.get(callee_name)
                if isinstance(callee_name, str)
                else None
            )
            if helper_name is None or callee is None:
                raise CapabilityError("Q8627: unresolved controlled HUGR adjoint")
            has_component = len(operands) == len(callee.parameters) + 1
            if len(operands) not in {
                len(callee.parameters),
                len(callee.parameters) + 1,
            }:
                raise CapabilityError("Q8627: malformed controlled HUGR adjoint ABI")
            if has_component:
                if len(operands) < 2:
                    raise CapabilityError(
                        "Q8627: missing controlled HUGR adjoint component"
                    )
                arguments = (selector, operands[1], target, *operands[2:])
            else:
                arguments = (selector, target, *operands[1:])
            inverse = Operation(
                original,
                (),
                operation.results,
                attributes(
                    callee=helper_name,
                    control_state=original_values.get("control_state"),
                ),
                span=operation.span,
            )
            return self._emit_control_call(inverse, arguments)
        callee_names = original_values.get("callees")
        if not isinstance(callee_names, tuple) or any(
            not isinstance(name, str) or name not in self.adjoint_helpers
            for name in callee_names
        ):
            raise CapabilityError("Q8627: unresolved selected HUGR adjoint")
        typed_callees = cast(tuple[str, ...], callee_names)
        inverse = Operation(
            original,
            (),
            operation.results,
            attributes(
                callees=tuple(self.adjoint_helpers[name] for name in typed_callees)
            ),
            span=operation.span,
        )
        return self._emit_select_call(inverse, (selector, target, *operands[1:]))

    def _emit_addition(
        self,
        wires: list[object],
        operation: Operation,
        *,
        adjoint: bool,
    ) -> tuple[object, ...]:
        """Project the Rust-owned exact arithmetic schedule into HUGR gates.

        Args:
            wires (list[object]): State in semantic little-endian lane order.
            operation (Operation): Original source operation for diagnostics.
            adjoint (bool): Whether to invert the selected unitary schedule.

        Returns:
            tuple[object, ...]: One aggregate containing the updated wires.

        Raises:
            CapabilityError: If native planning is unavailable or rejects the
                shape, choices, work budget, or target gate capabilities.
        """

        planner = getattr(native, "plan_addition", None)
        if not callable(planner):
            raise CapabilityError("Q8049: native arithmetic realization is unavailable")
        try:
            identity, schedule = planner(
                len(wires), (), self.emission_budget.options.overrides
            )
        except ValueError as error:
            raise CapabilityError(str(error)) from error
        self.emission_budget.realizations.add(identity)
        return self._emit_arithmetic_steps(wires, operation, schedule, adjoint=adjoint)

    def _emit_mod_pow(
        self,
        wires: list[object],
        operation: Operation,
        positional: tuple[object, ...],
        *,
        inverse: bool,
    ) -> tuple[object, ...]:
        """Project the shared exact modular-exponentiation plan into HUGR.

        Args:
            wires (list[object]): Target and exponent wires in semantic leaf order.
            operation (Operation): Verified modular exponentiation operation.
            positional (tuple[object, ...]): Specialized base and modulus values.
            inverse (bool): Whether to emit the inverse semantic operation.

        Returns:
            tuple[object, ...]: One aggregate containing the updated state wires.

        Raises:
            CapabilityError: If the native planner is unavailable, the semantic
                shape is invalid, or the selected plan exceeds an emission budget.
        """

        planner = getattr(native, "plan_mod_pow", None)
        if not callable(planner):
            raise CapabilityError("Q8049: native modular realization is unavailable")
        if len(operation.results) != 1:
            raise CapabilityError(
                "Q8665: modular exponentiation result shape is invalid"
            )
        widths = self.coherent_partitions.get(operation.operands[0])
        if widths is None:
            widths = coherent_leaf_widths(operation.results[0].type)
        if widths is None or len(widths) != 2 or sum(widths) != len(wires):
            raise CapabilityError(
                "Q8665: modular exponentiation state shape is invalid"
            )
        if len(positional) != 2:
            raise CapabilityError("Q8665: modular exponentiation constants are missing")
        base = _require_int(positional[0], "modular exponentiation base")
        modulus = _require_int(positional[1], "modular exponentiation modulus")
        try:
            disposition, identity, scratch_width, schedule, children = planner(
                widths[0],
                widths[1],
                base,
                modulus,
                inverse,
                len(self.coherent_controls),
                (),
                self.emission_budget.options.overrides,
            )
        except ValueError as error:
            raise CapabilityError(str(error)) from error
        self.coherent_partitions[operation.results[0].id] = widths
        return self._emit_modular_schedule(
            wires, disposition, identity, scratch_width, schedule, children
        )

    def _emit_modular_schedule(
        self,
        wires: list[object],
        disposition: str,
        identity: str,
        scratch_width: int,
        schedule: list[tuple[str, list[int], list[int]]],
        children: list[str],
    ) -> tuple[object, ...]:
        """Emit the checked modular schedule with clean scratch and retained controls.

        Args:
            wires (list[object]): Semantic state wires in compiler order.
            disposition (str): Compiler-selected gate-plan disposition.
            identity (str): Selected parent realization label.
            scratch_width (int): Checked clean scratch requirement.
            schedule (list[tuple[str, list[int], list[int]]]): Checked gate,
                intrinsic wire, and additional control projections.
            children (list[str]): Selected child realization labels.

        Returns:
            tuple[object, ...]: Updated semantic state in one aggregate.

        Raises:
            CapabilityError: If the bridge returns a plan outside HUGR capabilities.
        """

        from tket_exts import quantum

        if disposition != "gates":
            raise CapabilityError("Q8665: direct HUGR requires a modular gate schedule")
        self.emission_budget.realizations.add(identity)
        self.emission_budget.realizations.update(children)
        self._reserve_qubits(scratch_width)
        scratch = [self.builder.add_op(quantum.qAlloc)[0] for _ in range(scratch_width)]
        self.live_qubits += scratch_width
        self.allocated_qubits += scratch_width
        self.native_operations += scratch_width
        state_width = len(wires)
        control_width = len(self.coherent_controls)
        mapped = [*wires, *self.coherent_controls, *scratch]
        gates = {
            "h": quantum.H,
            "x": quantum.X,
            "cx": quantum.CX,
            "ccx": quantum.toffoli,
            "reset": quantum.reset,
        }
        for gate, indices, controls in schedule:
            if controls:
                raise CapabilityError(
                    "Q8665: direct HUGR requires decomposed modular controls"
                )
            try:
                operation_type = gates[gate]
                selected = [mapped[index] for index in indices]
            except (IndexError, KeyError) as error:
                raise CapabilityError(
                    "Q8665: native modular realization projection is malformed"
                ) from error
            self.emission_budget.consume_operation()
            node = self.builder.add_op(operation_type, *selected)
            updated = tuple(node[index] for index in range(len(indices)))
            for index, wire in zip(indices, updated, strict=True):
                mapped[index] = wire
            self.native_operations += 1
        wires[:] = mapped[:state_width]
        self.coherent_controls[:] = mapped[state_width : state_width + control_width]
        for wire in mapped[state_width + control_width :]:
            self.builder.add_op(quantum.qFree, wire)
        self.live_qubits -= scratch_width
        self.native_operations += scratch_width
        return (tuple(wires),)

    def _emit_arithmetic_permutation(
        self,
        wires: list[object],
        operation: Operation,
        constants: tuple[object, ...],
        *,
        adjoint: bool,
    ) -> tuple[object, ...]:
        """Project a compiler-owned exact arithmetic permutation into HUGR gates.

        Args:
            wires (list[object]): Coherent state in logical little-endian order.
            operation (Operation): Verified arithmetic operation.
            constants (tuple[object, ...]): Specialized classical operands.
            adjoint (bool): Whether the inverse unitary is required.

        Returns:
            tuple[object, ...]: One updated coherent state.

        Raises:
            CapabilityError: If native planning or static operands are invalid.
        """

        planner = getattr(native, "plan_arithmetic_permutation", None)
        if not callable(planner):
            raise CapabilityError("Q8049: native arithmetic realization is unavailable")
        values = [_require_int(value, "arithmetic constant") for value in constants]
        if operation.opcode in {"quantum.mod_mul", "quantum.mod_div"}:
            modular_planner = getattr(native, "plan_mod_mul", None)
            if not callable(modular_planner):
                raise CapabilityError(
                    "Q8049: native modular multiplication is unavailable"
                )
            if len(values) != 2:
                raise CapabilityError(
                    "Q8665: modular multiplication constants are missing"
                )
            try:
                modular = modular_planner(
                    len(wires),
                    values[0],
                    values[1],
                    adjoint ^ (operation.opcode == "quantum.mod_div"),
                    len(self.coherent_controls),
                    self.emission_budget.options.overrides,
                )
            except ValueError as error:
                raise CapabilityError(str(error)) from error
            if modular is not None:
                return self._emit_modular_schedule(wires, *modular)
        try:
            identity, schedule = planner(operation.opcode, len(wires), values, adjoint)
        except ValueError as error:
            raise CapabilityError(str(error)) from error
        self.emission_budget.realizations.add(identity)
        return self._emit_arithmetic_steps(wires, operation, schedule, adjoint=False)

    def _emit_arithmetic_steps(
        self,
        wires: list[object],
        operation: Operation,
        schedule: list[tuple[str, list[int], list[int], float | None, bool]],
        *,
        adjoint: bool,
    ) -> tuple[object, ...]:
        """Emit one checked Rust schedule through the target's gate projection.

        Args:
            wires (list[object]): Current coherent state.
            operation (Operation): Original operation for source diagnostics.
            schedule (list[tuple[str, list[int], list[int], float | None, bool]]):
                Checked primitive steps and their internal controls.
            adjoint (bool): Whether to reverse and invert the supplied schedule.

        Returns:
            tuple[object, ...]: Updated coherent state.

        Raises:
            CapabilityError: If the target cannot emit a required gate.
        """

        for gate, indices, internal_controls, angle, inherit_controls in (
            reversed(schedule) if adjoint else schedule
        ):
            self.emission_budget.consume_operation()
            saved = self.coherent_controls
            self.coherent_controls = list(saved) if inherit_controls else []
            outer_count = len(self.coherent_controls)
            self.coherent_controls.extend(wires[index] for index in internal_controls)
            operands = (
                tuple(wires),
                *((angle,) if angle is not None else ()),
                *indices,
            )
            primitive = Operation(f"quantum.{gate}", (), (), span=operation.span)
            [updated] = self._emit_gate(
                primitive.opcode, operands, primitive, adjoint=adjoint
            )
            wires = list(_require_wires(updated, primitive.opcode))
            for index, wire in zip(
                internal_controls, self.coherent_controls[outer_count:], strict=True
            ):
                wires[index] = wire
            if inherit_controls:
                saved[:] = self.coherent_controls[:outer_count]
            self.coherent_controls = saved
        return (tuple(wires),)

    def _emit_gate(
        self,
        opcode: str,
        operands: tuple[object, ...],
        operation: Operation,
        *,
        adjoint: bool,
    ) -> tuple[object, ...]:
        """Emit one retained primitive gate on an aggregate HUGR state.

        Args:
            opcode (str): Canonical QCore quantum opcode.
            operands (tuple[object, ...]): Reconstructed semantic operands.
            operation (Operation): Static arguments and source context.
            adjoint (bool): Whether to emit the inverse primitive.

        Returns:
            tuple[object, ...]: Updated aggregate quantum state.

        Raises:
            CapabilityError: If direct HUGR has no exact projection for the
                retained primitive or its transform context.
        """

        from tket_exts import quantum

        wires = list(_require_wires(operands[0], opcode))
        positional, keywords = _gate_arguments(operation, operands[1:])
        spec = GATES_BY_OPCODE.get(opcode)
        if spec is not None:
            selector_args = positional[int(spec.has_angle) :]
            if not (
                spec.broadcasts_when_unselected
                and not selector_args
                and not any(key in keywords for key in spec.selectors)
            ):
                selected = tuple(
                    _require_int(
                        keywords.get(
                            label,
                            selector_args[index]
                            if index < len(selector_args)
                            else spec.selector_defaults[index],
                        ),
                        label,
                    )
                    for index, label in enumerate(spec.selectors)
                )
                _validate_selectors(opcode, len(wires), selected)
        name = opcode.removeprefix("quantum.")
        if name in {"mod_pow", "mod_pow_inverse"}:
            return self._emit_mod_pow(
                wires,
                operation,
                positional,
                inverse=adjoint ^ (name == "mod_pow_inverse"),
            )
        if name in {"add_into", "subtract_into"}:
            return self._emit_addition(
                wires,
                operation,
                adjoint=adjoint ^ (name == "subtract_into"),
            )
        if name in {
            "add_constant",
            "subtract_constant",
            "mod_add",
            "mod_sub",
            "mod_add_constant",
            "mod_sub_constant",
            "mul_add",
            "mul_sub",
            "mod_mul",
            "mod_div",
            "less_than_xor",
        }:
            return self._emit_arithmetic_permutation(
                wires, operation, positional, adjoint=adjoint
            )
        if name == "pauli_evolution":
            if self.coherent_controls:
                raise CapabilityError(
                    "Q8660: direct HUGR cannot coherently control Pauli evolution"
                )
            if len(positional) < 2:
                raise CapabilityError("Q8628: Pauli evolution is missing operands")
            hamiltonian = _require_hamiltonian(positional[0])
            time = _require_static_angle(positional[1], "Pauli evolution time")
            if adjoint:
                time = -time
            planner = getattr(native, "plan_pauli_realization", None)
            if not callable(planner):
                raise CapabilityError(
                    "Q8661: native Pauli evolution planner is unavailable"
                )
            try:
                identity, schedule = planner(
                    hamiltonian._native_literal,
                    len(wires),
                    self.emission_budget.options.overrides,
                )
            except ValueError as error:
                raise CapabilityError(f"Q8661: {error}") from error
            self.emission_budget.realizations.add(identity)
            for gate, indices, scale in schedule:
                self.emission_budget.consume_operation()
                if gate == "global_phase":
                    _emit_global_phase(self.builder, scale * time)
                elif gate == "rz":
                    [target] = indices
                    rotation = _rotation_wire(self.builder, scale * time)
                    wires[target] = self.builder.add_op(
                        quantum.Rz,
                        wires[target],
                        rotation,
                    )[0]
                elif gate == "cx":
                    control, target = indices
                    wires[control], wires[target] = self.builder.add_op(
                        quantum.CX,
                        wires[control],
                        wires[target],
                    )
                else:
                    [target] = indices
                    operation_type = {
                        "h": quantum.H,
                        "s": quantum.S,
                        "sdg": quantum.Sdg,
                    }[gate]
                    wires[target] = self.builder.add_op(
                        operation_type,
                        wires[target],
                    )[0]
                self.native_operations += 1
            return (tuple(wires),)
        if name == "global_phase":
            if not positional:
                raise CapabilityError(
                    "Q8628: direct HUGR global phase is missing an angle"
                )
            angle = self._require_angle(positional[0], "global phase")
            if adjoint:
                angle = _negate_angle(self.builder, angle)
                self.native_operations += isinstance(angle, _RealWire)
            if self.coherent_controls:
                self.coherent_controls[:] = self._emit_controlled_phase(
                    angle,
                    list(self.coherent_controls),
                )
            else:
                _emit_global_phase(self.builder, angle)
                self.native_operations += 1
            return (tuple(wires),)
        if self.coherent_controls:
            return self._emit_controlled_gate(
                name,
                wires,
                positional,
                keywords,
                adjoint=adjoint,
            )
        single = {
            "h": quantum.H,
            "x": quantum.X,
            "y": quantum.Y,
            "z": quantum.Z,
            "s": quantum.Sdg if adjoint else quantum.S,
            "sdg": quantum.S if adjoint else quantum.Sdg,
            "t": quantum.Tdg if adjoint else quantum.T,
            "tdg": quantum.T if adjoint else quantum.Tdg,
        }
        if name in single:
            targets = (
                range(len(wires))
                if "target" not in keywords and not positional
                else (
                    _require_int(
                        keywords["target"] if "target" in keywords else positional[0],
                        "gate target",
                    ),
                )
            )
            for target in targets:
                wires[target] = self.builder.add_op(single[name], wires[target])[0]
                self.native_operations += 1
            return (tuple(wires),)
        if name in {"p", "rx", "ry", "rz"}:
            if not positional:
                raise CapabilityError(f"Q8628: direct HUGR {name} is missing an angle")
            angle = self._require_angle(positional[0], f"{name} angle")
            if adjoint:
                angle = _negate_angle(self.builder, angle)
                self.native_operations += isinstance(angle, _RealWire)
            if "target" in keywords:
                targets = (_require_int(keywords["target"], "rotation target"),)
            elif len(positional) == 1:
                targets = range(len(wires))
            else:
                targets = (_require_int(positional[-1], "rotation target"),)
            for target in targets:
                if name == "p":
                    [wires[target]] = self._emit_controlled_phase(
                        angle, [wires[target]]
                    )
                    continue
                rotation = _rotation_wire(self.builder, angle)
                gate = {"rx": quantum.Rx, "ry": quantum.Ry, "rz": quantum.Rz}[name]
                wires[target] = self.builder.add_op(gate, wires[target], rotation)[0]
                self.native_operations += 1
            return (tuple(wires),)
        if name == "ccx":
            control1 = _require_int(
                keywords.get("control1", positional[0] if positional else 0),
                "first gate control",
            )
            control2 = _require_int(
                keywords.get("control2", positional[1] if len(positional) > 1 else 1),
                "second gate control",
            )
            target = _require_int(
                keywords.get("target", positional[2] if len(positional) > 2 else 2),
                "gate target",
            )
            wires[control1], wires[control2], wires[target] = self.builder.add_op(
                quantum.toffoli,
                wires[control1],
                wires[control2],
                wires[target],
            )
            self.native_operations += 1
            return (tuple(wires),)
        if name in {"cx", "cz"}:
            control = _require_int(
                keywords.get("control", positional[0] if positional else 0),
                "gate control",
            )
            target = _require_int(
                keywords.get("target", positional[1] if len(positional) > 1 else 1),
                "gate target",
            )
            gate = quantum.CX if name == "cx" else quantum.CZ
            wires[control], wires[target] = self.builder.add_op(
                gate, wires[control], wires[target]
            )
            self.native_operations += 1
            return (tuple(wires),)
        if name in {"cp", "rzz"}:
            if not positional:
                raise CapabilityError(f"Q8628: direct HUGR {name} is missing an angle")
            angle = self._require_angle(positional[0], f"{name} angle")
            if adjoint:
                angle = _negate_angle(self.builder, angle)
                self.native_operations += isinstance(angle, _RealWire)
            left = _require_int(
                keywords.get(
                    "control" if name == "cp" else "left",
                    positional[1] if len(positional) > 1 else 0,
                ),
                f"{name} left",
            )
            right = _require_int(
                keywords.get(
                    "target" if name == "cp" else "right",
                    positional[2] if len(positional) > 2 else 1,
                ),
                f"{name} right",
            )
            if name == "cp":
                wires[left], wires[right] = self._emit_controlled_phase(
                    angle,
                    [wires[left], wires[right]],
                )
            else:
                wires[left], wires[right] = self.builder.add_op(
                    quantum.CX, wires[left], wires[right]
                )
                rotation = _rotation_wire(self.builder, angle)
                wires[right] = self.builder.add_op(quantum.Rz, wires[right], rotation)[
                    0
                ]
                wires[left], wires[right] = self.builder.add_op(
                    quantum.CX, wires[left], wires[right]
                )
                self.native_operations += 3
            return (tuple(wires),)
        if name == "swap":
            left = _require_int(
                keywords.get("left", positional[0] if positional else 0), "swap left"
            )
            right = _require_int(
                keywords.get("right", positional[1] if len(positional) > 1 else 1),
                "swap right",
            )
            for control, target in ((left, right), (right, left), (left, right)):
                wires[control], wires[target] = self.builder.add_op(
                    quantum.CX, wires[control], wires[target]
                )
            self.native_operations += 3
            return (tuple(wires),)
        raise CapabilityError(f"Q8629: unsupported direct HUGR gate {name!r}")

    def _emit_controlled_gate(
        self,
        name: str,
        wires: list[object],
        positional: tuple[object, ...],
        keywords: dict[str, object],
        *,
        adjoint: bool,
    ) -> tuple[object, ...]:
        """Emit the exact discrete controlled subset exposed by TKET HUGR."""

        from tket_exts import quantum

        controls = self.coherent_controls
        if name in {"p", "cp"}:
            if not positional:
                raise CapabilityError(
                    f"Q8660: controlled direct HUGR {name} is missing an angle"
                )
            angle = self._require_angle(positional[0], f"controlled {name} angle")
            if adjoint:
                angle = _negate_angle(self.builder, angle)
                self.native_operations += isinstance(angle, _RealWire)
            if name == "p":
                if "target" in keywords:
                    selected = [
                        _require_int(keywords["target"], "controlled phase target")
                    ]
                elif len(positional) == 1:
                    selected = list(range(len(wires)))
                else:
                    selected = [_require_int(positional[-1], "controlled phase target")]
            else:
                selected = [
                    _require_int(
                        keywords.get(
                            "control", positional[1] if len(positional) > 1 else 0
                        ),
                        "controlled CP control",
                    ),
                    _require_int(
                        keywords.get(
                            "target", positional[2] if len(positional) > 2 else 1
                        ),
                        "controlled CP target",
                    ),
                ]
            if name == "p":
                for index in selected:
                    combined = self._emit_controlled_phase(
                        angle,
                        [*controls, wires[index]],
                    )
                    controls[:] = combined[:-1]
                    wires[index] = combined[-1]
            else:
                combined = self._emit_controlled_phase(
                    angle,
                    [*controls, *(wires[index] for index in selected)],
                )
                controls[:] = combined[: len(controls)]
                for index, wire in zip(
                    selected, combined[len(controls) :], strict=True
                ):
                    wires[index] = wire
            return (tuple(wires),)
        if len(controls) == 1 and name in {"rx", "ry", "rz"}:
            if not positional:
                raise CapabilityError(
                    f"Q8660: controlled direct HUGR {name} is missing an angle"
                )
            angle = self._require_angle(positional[0], f"controlled {name} angle")
            if adjoint:
                angle = _negate_angle(self.builder, angle)
                self.native_operations += isinstance(angle, _RealWire)
            if "target" in keywords:
                targets = (
                    _require_int(keywords["target"], "controlled rotation target"),
                )
            elif len(positional) == 1:
                targets = range(len(wires))
            else:
                targets = (_require_int(positional[-1], "controlled rotation target"),)
            before = (
                (quantum.H,)
                if name == "rx"
                else (quantum.Sdg, quantum.H)
                if name == "ry"
                else ()
            )
            after = (
                (quantum.H,)
                if name == "rx"
                else (quantum.H, quantum.S)
                if name == "ry"
                else ()
            )
            for target in targets:
                for gate in before:
                    wires[target] = self.builder.add_op(gate, wires[target])[0]
                rotation = _rotation_wire(self.builder, angle)
                controls[0], wires[target] = self.builder.add_op(
                    quantum.CRz,
                    controls[0],
                    wires[target],
                    rotation,
                )
                for gate in after:
                    wires[target] = self.builder.add_op(gate, wires[target])[0]
                self.native_operations += len(before) + 1 + len(after)
            return (tuple(wires),)
        if name in {"x", "y", "z"}:
            targets = (
                range(len(wires))
                if "target" not in keywords and not positional
                else (
                    _require_int(
                        keywords["target"] if "target" in keywords else positional[0],
                        "controlled gate target",
                    ),
                )
            )
            for target in targets:
                self._emit_controlled_discrete_gate(
                    name,
                    wires,
                    target,
                )
            return (tuple(wires),)
        if name in {"cx", "cz"}:
            inner = _require_int(
                keywords.get("control", positional[0] if positional else 0),
                "controlled gate inner control",
            )
            target = _require_int(
                keywords.get("target", positional[1] if len(positional) > 1 else 1),
                "controlled gate target",
            )
            if len(controls) == 1:
                if name == "cz":
                    wires[target] = self.builder.add_op(quantum.H, wires[target])[0]
                controls[0], wires[inner], wires[target] = self.builder.add_op(
                    quantum.toffoli,
                    controls[0],
                    wires[inner],
                    wires[target],
                )
                if name == "cz":
                    wires[target] = self.builder.add_op(quantum.H, wires[target])[0]
                    self.native_operations += 2
                self.native_operations += 1
                return (tuple(wires),)
            if name == "cx":
                wires[target] = self.builder.add_op(quantum.H, wires[target])[0]
            combined = self._emit_controlled_phase(
                math.pi,
                [*controls, wires[inner], wires[target]],
            )
            controls[:] = combined[: len(controls)]
            wires[inner] = combined[-2]
            wires[target] = combined[-1]
            if name == "cx":
                wires[target] = self.builder.add_op(quantum.H, wires[target])[0]
                self.native_operations += 2
            return (tuple(wires),)
        if len(controls) == 1 and name == "swap":
            left = _require_int(
                keywords.get("left", positional[0] if positional else 0),
                "controlled swap left",
            )
            right = _require_int(
                keywords.get("right", positional[1] if len(positional) > 1 else 1),
                "controlled swap right",
            )
            wires[right], wires[left] = self.builder.add_op(
                quantum.CX,
                wires[right],
                wires[left],
            )
            controls[0], wires[left], wires[right] = self.builder.add_op(
                quantum.toffoli,
                controls[0],
                wires[left],
                wires[right],
            )
            wires[right], wires[left] = self.builder.add_op(
                quantum.CX,
                wires[right],
                wires[left],
            )
            self.native_operations += 3
            return (tuple(wires),)

        raise CapabilityError(f"Q8660: direct HUGR cannot coherently control {name!r}")

    def _emit_controlled_discrete_gate(
        self,
        name: str,
        wires: list[object],
        target: int,
    ) -> None:
        """Emit one discrete gate with the active coherent controls.

        Args:
            name (str): Primitive gate name in ``x``, ``y``, or ``z``.
            wires (list[object]): Mutable target wire row.
            target (int): Selected target position in ``wires``.

        Raises:
            CapabilityError: If the active control shape is unsupported.
        """

        from tket_exts import quantum

        controls = self.coherent_controls
        if name in {"x", "y", "z"}:
            if len(controls) == 1:
                gate = {"x": quantum.CX, "y": quantum.CY, "z": quantum.CZ}[name]
                controls[0], wires[target] = self.builder.add_op(
                    gate,
                    controls[0],
                    wires[target],
                )
                self.native_operations += 1
                return
            if len(controls) == 2 and name == "x":
                controls[0], controls[1], wires[target] = self.builder.add_op(
                    quantum.toffoli,
                    controls[0],
                    controls[1],
                    wires[target],
                )
                self.native_operations += 1
                return

            before = (
                (quantum.H,)
                if name == "x"
                else (quantum.Sdg, quantum.H)
                if name == "y"
                else ()
            )
            after = (
                (quantum.H,)
                if name == "x"
                else (quantum.H, quantum.S)
                if name == "y"
                else ()
            )
            for gate in before:
                wires[target] = self.builder.add_op(gate, wires[target])[0]
            combined = self._emit_controlled_phase(
                math.pi,
                [*controls, wires[target]],
            )
            controls[:] = combined[:-1]
            wires[target] = combined[-1]
            for gate in after:
                wires[target] = self.builder.add_op(gate, wires[target])[0]
            self.native_operations += len(before) + len(after)
            return
        raise CapabilityError(
            f"Q8660: unsupported controlled direct HUGR gate {name!r}"
        )

    def _emit_controlled_phase(
        self,
        angle: float | _RealWire,
        controls: list[Any],
        *,
        scale: float = 1.0,
    ) -> list[Any]:
        """Emit an exact phase on the all-one state of coherent controls.

        For three or more components, the Boolean AND is expanded into its
        XOR-parity polynomial. Each term is therefore only a CNOT parity
        sandwich around the existing one-component phase. This uniform
        construction needs neither target-specific ancillas nor a second
        lifetime model in the emitter.
        """

        from tket_exts import quantum

        if not controls:
            raise CapabilityError("Q8660: controlled phase requires at least one wire")
        if len(controls) == 1:
            _emit_global_phase(self.builder, angle, scale=0.5 * scale)
            rotation = _rotation_wire(self.builder, angle, scale=scale)
            controls[0] = self.builder.add_op(quantum.Rz, controls[0], rotation)[0]
            self.native_operations += 2
            return controls
        if len(controls) == 2:
            _emit_global_phase(self.builder, angle, scale=0.25 * scale)
            half = _rotation_wire(self.builder, angle, scale=0.5 * scale)
            controls[0] = self.builder.add_op(quantum.Rz, controls[0], half)[0]
            full = _rotation_wire(self.builder, angle, scale=scale)
            controls[0], controls[1] = self.builder.add_op(
                quantum.CRz,
                controls[0],
                controls[1],
                full,
            )
            self.native_operations += 3
            return controls

        if len(controls) >= MAX_HUGR_CONTROLLED_PHASE_TERMS.bit_length():
            raise CapabilityError(
                "Q8682: HUGR controlled-phase expansion exceeds the work budget"
            )
        denominator = 1 << (len(controls) - 1)
        for mask in range(1, 1 << len(controls)):
            selected = tuple(
                index for index in range(len(controls)) if mask & (1 << index)
            )
            pivot = selected[-1]
            for index in selected[:-1]:
                controls[index], controls[pivot] = self.builder.add_op(
                    quantum.CX,
                    controls[index],
                    controls[pivot],
                )
            coefficient = scale / denominator
            if len(selected) % 2 == 0:
                coefficient = -coefficient
            [controls[pivot]] = self._emit_controlled_phase(
                angle,
                [controls[pivot]],
                scale=coefficient,
            )
            for index in reversed(selected[:-1]):
                controls[index], controls[pivot] = self.builder.add_op(
                    quantum.CX,
                    controls[index],
                    controls[pivot],
                )
            self.native_operations += 2 * (len(selected) - 1)
        return controls


def _reachable_functions(program: _EmissionProgram) -> tuple[Function, ...]:
    """Return the deterministic static call closure of the entrypoint."""

    functions = {function.symbol: function for function in program.functions}
    pending = [program.entrypoint]
    reachable: set[str] = set()
    while pending:
        symbol = pending.pop()
        if symbol in reachable:
            continue
        function = functions.get(symbol)
        if function is None:
            raise CapabilityError(f"Q8638: missing HUGR callable {symbol!r}")
        reachable.add(symbol)
        for operation in _walk_qcore_operations(function.body):
            if operation.opcode in {"qcore.call", "qcore.adjoint_call"}:
                callee = operation.attribute("callee")
                if isinstance(callee, str):
                    pending.append(callee)
    return tuple(functions[symbol] for symbol in sorted(reachable))


def _walk_qcore_operations(block: Block) -> Iterator[Operation]:
    """Yield QCore operations recursively from one structured block."""

    for operation in block.operations:
        yield operation
        for region in operation.regions:
            for nested in region.blocks:
                yield from _walk_qcore_operations(nested)


def _hugr_type_row_for_parameters(function: Function) -> list[Any]:
    """Flatten a QCore callable parameter row into HUGR port types."""

    return _hugr_type_row_for_parameter_row(function.parameters)


def _hugr_type_row_for_parameter_row(parameters: tuple[Any, ...]) -> list[Any]:
    """Flatten a selected parameter row into bounded HUGR port types."""

    row: list[Any] = []
    for parameter in parameters:
        row.extend(_hugr_type_row(parameter.value.type))
        if len(row) > MAX_HUGR_PORTS:
            raise CapabilityError("Q8681: HUGR callable exceeds the port budget")
    return row


def _hugr_type_row(type_spec: TypeSpec) -> list[Any]:
    """Flatten one supported logical type into HUGR dataflow types."""

    from hugr import tys
    from hugr.std.float import FLOAT_T

    base = strip_stage(type_spec)
    if isinstance(base, QuantumType):
        width = logical_width(base)
        if width is None:
            raise CapabilityError("Q8639: HUGR quantum width must be static")
        if width > MAX_HUGR_PORTS:
            raise CapabilityError("Q8681: HUGR quantum type exceeds the port budget")
        return [tys.Qubit for _ in range(width)]
    if isinstance(base, (FixedType, WordType)):
        width = logical_width(base)
        if width is None:
            raise CapabilityError("Q8640: HUGR bit width must be static")
        if width > MAX_HUGR_PORTS:
            raise CapabilityError("Q8681: HUGR bit type exceeds the port budget")
        return [tys.Bool for _ in range(width)]
    elements = aggregate_element_types(base)
    if elements is not None:
        row: list[Any] = []
        for element in elements:
            row.extend(_hugr_type_row(element))
            if len(row) > MAX_HUGR_PORTS:
                raise CapabilityError("Q8681: HUGR value exceeds the port budget")
        return row
    if base in {Bit, Bool}:
        return [tys.Bool]
    if isinstance(base, ScalarType) and base.name == "Real":
        return [FLOAT_T]
    if isinstance(base, ScalarType) and base.name == "Unit":
        return []
    raise CapabilityError(f"Q8641: {type_spec} has no Selene HUGR ABI type")


def _reconstruct_parameters(
    function: Function,
    wires: tuple[Any, ...],
) -> tuple[object, ...]:
    """Rebuild aggregate QCore parameters from flat HUGR input wires."""

    return _reconstruct_parameter_row(function.parameters, wires)


def _reconstruct_parameter_row(
    parameters: tuple[Any, ...],
    wires: tuple[Any, ...],
) -> tuple[object, ...]:
    """Rebuild a selected parameter row from flat HUGR input wires."""

    offset = 0
    values: list[object] = []
    for parameter in parameters:
        width = len(_hugr_type_row(parameter.value.type))
        values.append(
            _reconstruct_hugr_value(
                parameter.value.type,
                wires[offset : offset + width],
            )
        )
        offset += width
    if offset != len(wires):
        raise CapabilityError("Q8642: extra HUGR callable input wires")
    return tuple(values)


def _parameter_qubit_width(parameters: tuple[Any, ...]) -> int:
    """Return the bounded coherent width of a selected parameter row."""

    width = sum(
        logical_width(parameter.value.type) or 0
        for parameter in parameters
        if parameter.value.type.is_quantum
    )
    if width > MAX_HUGR_PORTS:
        raise CapabilityError("Q8681: HUGR callable exceeds the qubit budget")
    return width


def _reconstruct_block_arguments(
    block: Block,
    wires: tuple[Any, ...],
) -> tuple[object, ...]:
    """Rebuild structured-region arguments from flat HUGR input wires."""

    offset = 0
    values: list[object] = []
    for argument in block.arguments:
        width = len(_hugr_type_row(argument.type))
        values.append(
            _reconstruct_hugr_value(
                argument.type,
                wires[offset : offset + width],
            )
        )
        offset += width
    if offset != len(wires):
        raise CapabilityError("Q8648: extra HUGR region input wires")
    return tuple(values)


def _block_terminator_definitions(block: Block) -> tuple[Value, ...]:
    """Resolve one structured block's exit operands to their SSA definitions."""

    definitions = {argument.id: argument for argument in block.arguments}
    for operation in block.operations:
        definitions.update((result.id, result) for result in operation.results)
    try:
        return tuple(definitions[value] for value in block.terminator.operands)
    except KeyError as error:
        raise CapabilityError(
            "Q8667: dynamic return continuation has an unresolved exit value"
        ) from error


def _common_provenance(
    rows: Iterable[tuple[int | None, ...]],
) -> tuple[int | None, ...]:
    """Retain only measurement provenance shared by every possible path."""

    candidates = tuple(rows)
    if not candidates:
        return ()
    width = len(candidates[0])
    if any(len(candidate) != width for candidate in candidates[1:]):
        raise CapabilityError("Q8647: dynamic HUGR flow provenance arity mismatch")
    return tuple(
        first
        if all(candidate[index] == first for candidate in candidates[1:])
        else None
        for index, first in enumerate(candidates[0])
    )


def _reconstruct_typed_values(
    definitions: tuple[Value, ...],
    wires: tuple[Any, ...],
) -> tuple[object, ...]:
    """Reconstruct parallel QCore SSA results from flat HUGR outputs."""

    offset = 0
    values: list[object] = []
    for definition in definitions:
        width = len(_hugr_type_row(definition.type))
        values.append(
            _reconstruct_hugr_value(
                definition.type,
                wires[offset : offset + width],
            )
        )
        offset += width
    if offset != len(wires):
        raise CapabilityError("Q8650: extra HUGR result wires")
    return tuple(values)


def _function_value_type_index(function: Function) -> dict[int, TypeSpec]:
    """Index every SSA value type in one retained QCore function.

    Args:
        function: Function whose parameters and nested blocks are indexed.

    Returns:
        Mapping from function-local SSA identifiers to verified logical types.
    """

    values = {
        parameter.value.id: parameter.value.type for parameter in function.parameters
    }
    pending = [function.body]
    while pending:
        block = pending.pop()
        values.update((argument.id, argument.type) for argument in block.arguments)
        for operation in block.operations:
            values.update((result.id, result.type) for result in operation.results)
            for region in operation.regions:
                pending.extend(region.blocks)
    return values


def _reconstruct_hugr_value(
    type_spec: TypeSpec,
    wires: tuple[Any, ...],
) -> object:
    """Rebuild one logical aggregate from a flat HUGR wire row."""

    base = strip_stage(type_spec)
    expected = len(_hugr_type_row(type_spec))
    if len(wires) != expected:
        raise CapabilityError("Q8643: HUGR logical value width mismatch")
    if isinstance(base, QuantumType):
        return wires
    if isinstance(base, (FixedType, WordType)):
        return wires
    elements = aggregate_element_types(base)
    if elements is not None:
        offset = 0
        values: list[object] = []
        for element in elements:
            width = len(_hugr_type_row(element))
            values.append(
                _reconstruct_hugr_value(element, wires[offset : offset + width])
            )
            offset += width
        return tuple(values)
    if isinstance(base, ScalarType) and base.name == "Unit":
        return None
    if isinstance(base, ScalarType) and base.name == "Real":
        return wires[0] if isinstance(wires[0], _RealWire) else _RealWire(wires[0])
    return wires[0]


def _flatten_logical_value(value: object, type_spec: TypeSpec) -> tuple[Any, ...]:
    """Flatten a logical HUGR value while retaining provenance wrappers."""

    base = strip_stage(type_spec)
    if isinstance(base, ScalarType) and base.name == "Unit":
        return ()
    if base in {Bit, Bool} and (
        isinstance(value, bool)
        or isinstance(value, int)
        and not isinstance(value, bool)
        and value in {0, 1}
    ):
        return (value,)
    elements = aggregate_element_types(base)
    if elements is not None:
        if not isinstance(value, tuple) or len(value) != len(elements):
            raise CapabilityError("Q8644: HUGR product value shape mismatch")
        return tuple(
            wire
            for item, element in zip(value, elements, strict=True)
            for wire in _flatten_logical_value(item, element)
        )
    width = len(_hugr_type_row(type_spec))
    if width == 1 and _is_hugr_wire(value):
        return (value,)
    if (
        isinstance(value, tuple)
        and len(value) == width
        and all(_is_hugr_wire(item) for item in value)
    ):
        return value
    raise CapabilityError(
        "Q8645: HUGR logical value is not a compatible wire aggregate"
    )


def _copy_provenance(
    originals: tuple[object, ...],
    reconstructed: tuple[object, ...],
) -> tuple[object, ...]:
    """Copy measurement provenance through a region's block arguments."""

    if len(originals) != len(reconstructed):
        raise CapabilityError("Q8658: HUGR provenance argument arity mismatch")
    copied: list[object] = []
    for original, value in zip(originals, reconstructed, strict=True):
        if isinstance(original, _MeasuredWire):
            narrowed = (
                value[0] if isinstance(value, tuple) and len(value) == 1 else value
            )
            copied.append(_MeasuredWire(_raw_wire(narrowed), original.index))
        elif isinstance(original, _RealWire):
            copied.append(_RealWire(_raw_wire(value), original.guaranteed_finite))
        elif (
            isinstance(original, tuple)
            and len(original) == 1
            and isinstance(original[0], _MeasuredWire)
            and _is_hugr_wire(value)
        ):
            copied.append(_MeasuredWire(_raw_wire(value), original[0].index))
        elif isinstance(original, tuple) and isinstance(value, tuple):
            copied.append(_copy_provenance(original, value))
        else:
            copied.append(value)
    return tuple(copied)


def _apply_flat_provenance(
    values: tuple[object, ...],
    definitions: tuple[Value, ...],
    provenance: tuple[int | None, ...],
) -> tuple[object, ...]:
    """Reapply common branch provenance to conditional output wires."""

    flat = [
        item
        for value, definition in zip(values, definitions, strict=True)
        for item in _flatten_logical_value(value, definition.type)
    ]
    if len(flat) != len(provenance):
        raise CapabilityError("Q8659: HUGR branch provenance width mismatch")
    tagged = tuple(
        _MeasuredWire(_raw_wire(item), index) if index is not None else item
        for item, index in zip(flat, provenance, strict=True)
    )
    return _reconstruct_typed_values(definitions, tagged)


def _apply_value_provenance(
    value: object,
    type_spec: TypeSpec,
    provenance: tuple[int | None, ...],
) -> object:
    """Reapply common branch provenance to one returned logical value."""

    flat = _flatten_logical_value(value, type_spec)
    if len(flat) != len(provenance):
        raise CapabilityError("Q8659: HUGR return provenance width mismatch")
    tagged = tuple(
        _MeasuredWire(_raw_wire(item), index) if index is not None else item
        for item, index in zip(flat, provenance, strict=True)
    )
    return _reconstruct_hugr_value(type_spec, tagged)


def _validate_selectors(opcode: str, width: int, selected: tuple[int, ...]) -> None:
    """Apply Rust's bounds and alias rules immediately before SDK emission.

    Args:
        opcode (str): Operation selecting the quantum wires.
        width (int): Number of wires in the owned quantum state.
        selected (tuple[int, ...]): Concrete lane indices from the native gate catalog.

    Raises:
        CapabilityError: If a selector is negative, out of range, or repeated.
    """
    try:
        native.validate_circuit_selectors(opcode, width, list(selected))
    except (ValueError, OverflowError) as error:
        raise CapabilityError(str(error)) from error


def _gate_arguments(
    operation: Operation,
    operands: tuple[object, ...],
) -> tuple[tuple[object, ...], dict[str, object]]:
    """Split positional and named QCore gate arguments."""

    names = cast(tuple[str, ...], operation.attribute("keywords", ()))
    if not names:
        return operands, {}
    positional_count = len(operands) - len(names)
    return (
        operands[:positional_count],
        dict(zip(names, operands[positional_count:], strict=True)),
    )


def _require_wires(value: object, opcode: str) -> tuple[Any, ...]:
    """Require one aggregate HUGR quantum value."""

    if not isinstance(value, tuple) or not all(_is_hugr_wire(item) for item in value):
        raise CapabilityError(f"Q8633: {opcode} expected HUGR quantum wires")
    return tuple(_raw_wire(item) for item in value)


def _is_hugr_wire(value: object) -> bool:
    """Return whether a value is a HUGR wire without importing its class eagerly."""

    if isinstance(value, (_MeasuredWire, _RealWire, _BooleanCondition)):
        return True
    return value.__class__.__module__.startswith("hugr.") and hasattr(value, "out_port")


def _contains_hugr_wire(value: object) -> bool:
    """Return whether a reconstructed classical value contains a HUGR wire."""

    if _is_hugr_wire(value):
        return True
    return isinstance(value, tuple) and any(_contains_hugr_wire(item) for item in value)


def _is_boolean_operand(value: object) -> bool:
    """Return whether a verified runtime value has HUGR Bool representation."""

    if isinstance(value, _RealWire):
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return value in {0, 1}
    return _is_hugr_wire(value)


def _fold_boolean_condition(
    opcode: Literal["and", "or"], values: tuple[object, ...]
) -> object:
    """Fold one nonempty Boolean row without introducing an eager evaluator."""

    if not values:
        raise ValueError("Boolean condition fold requires at least one value")
    result = values[0]
    for value in values[1:]:
        result = _BooleanCondition(opcode, (result, value))
    return result


def _boolean_equality(left: object, right: object) -> _BooleanCondition:
    """Represent exact HUGR Bool equality through the closed condition algebra."""

    return _BooleanCondition(
        "or",
        (
            _BooleanCondition("and", (left, right)),
            _BooleanCondition(
                "and",
                (
                    _BooleanCondition("not", (left,)),
                    _BooleanCondition("not", (right,)),
                ),
            ),
        ),
    )


def _is_quantum_wire_tuple(value: object) -> bool:
    """Return whether a value is one nonempty coherent HUGR leaf."""

    return (
        isinstance(value, tuple)
        and bool(value)
        and all(_is_hugr_wire(item) for item in value)
    )


def _coherent_value_leaf_widths(value: object) -> tuple[int, ...] | None:
    """Return concrete coherent leaf widths from one reconstructed value.

    Args:
        value (object): Potentially nested coherent value before a reshape.

    Returns:
        tuple[int, ...] | None: Ordered concrete leaf widths, or ``None`` when
            the runtime value is not a coherent aggregate.
    """

    if _is_quantum_wire_tuple(value):
        return (len(cast(tuple[object, ...], value)),)
    if not isinstance(value, tuple) or not value:
        return None
    rows = tuple(_coherent_value_leaf_widths(item) for item in value)
    if any(row is None for row in rows):
        return None
    return tuple(width for row in rows if row is not None for width in row)


def _raw_wire(value: object) -> Any:
    """Return the underlying HUGR wire from a provenance wrapper."""

    if isinstance(value, (_MeasuredWire, _RealWire)):
        return value.wire
    if isinstance(value, _BooleanCondition):
        raise TypeError("compound Boolean conditions have no single HUGR wire")
    return value


def _measurement_index(value: object) -> int:
    """Return stable measurement provenance for a sampled result wire."""

    if not isinstance(value, _MeasuredWire):
        raise TypeError("wire has no measurement provenance")
    return value.index


def _require_int(value: object, label: str) -> int:
    """Require a statically known integer selector."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise CapabilityError(f"Q8634: {label} must be a static integer")
    return value


def _real_operand_wire(builder: Any, value: object) -> Any:
    """Materialize one verified Real operand as a HUGR float64 wire.

    Args:
        builder: Active HUGR dataflow builder.
        value: Typed runtime Real wire or static binary64 value.

    Returns:
        A raw HUGR float64 wire.

    Raises:
        CapabilityError: If the operand is not a Real wire or finite static
            float.
    """

    from hugr.std.float import FloatVal

    if isinstance(value, _RealWire):
        return value.wire
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CapabilityError("Q8611: HUGR cannot encode a nonfinite Real constant")
        return builder.load(FloatVal(value))[0]
    raise CapabilityError("Q8611: dynamic Real operation has a non-Real operand")


def _native_real_operation_facts(
    opcode: str,
    operands: tuple[object, ...],
) -> tuple[_RealRuntimeGuard, bool]:
    """Project the compiler-owned Real runtime guard and finiteness fact.

    Args:
        opcode: Verified QCore classical opcode.
        operands: Typed runtime Real facts and static binary64 operands.

    Returns:
        The required runtime guard and whether every result is finite.

    Raises:
        CapabilityError: If native proof support is absent, rejects the fact
            row, or returns a malformed projection.
    """

    derive = getattr(native, "real_operation_facts", None)
    if not callable(derive):
        raise CapabilityError("Q8685: native Real operation facts are unavailable")
    facts: list[tuple[bool, float | None]] = []
    for operand in operands:
        if isinstance(operand, _RealWire):
            facts.append((operand.guaranteed_finite, None))
        elif isinstance(operand, float):
            facts.append((math.isfinite(operand), operand))
        else:
            raise CapabilityError("Q8685: malformed Real finiteness operand")
    try:
        result = derive(opcode, tuple(facts))
    except (TypeError, ValueError) as error:
        raise CapabilityError(
            f"Q8685: native Real operation facts failed: {error}"
        ) from error
    if (
        not isinstance(result, tuple)
        or len(result) != 2
        or not isinstance(result[0], str)
        or type(result[1]) is not bool
    ):
        raise CapabilityError("Q8685: native Real operation facts are malformed")
    runtime_guard, guaranteed_finite = result
    allowed_guards: dict[str, frozenset[str]] = {
        "classical.add": frozenset({"none"}),
        "classical.sub": frozenset({"none"}),
        "classical.mul": frozenset({"none"}),
        "classical.truediv": frozenset({"none", "nonzero_divisor"}),
        "classical.pow": frozenset({"none", "power_domain"}),
        "classical.neg": frozenset({"none"}),
        "classical.pos": frozenset({"none"}),
    }
    if runtime_guard not in allowed_guards.get(opcode, frozenset()):
        raise CapabilityError("Q8685: native Real operation facts are inconsistent")
    return cast(_RealRuntimeGuard, runtime_guard), guaranteed_finite


def _native_aggregate_position(extent: int, index: int) -> int | None:
    """Normalize one aggregate index through the compiler-owned rule.

    Args:
        extent: Runtime representation's verified aggregate extent.
        index: Exact static QCore integer index.

    Returns:
        The normalized zero-based position, or ``None`` when out of bounds.

    Raises:
        CapabilityError: If the native projection is missing, fails, or returns
            a malformed position.
    """

    normalize = getattr(native, "normalize_aggregate_index", None)
    if not callable(normalize):
        raise CapabilityError("Q8686: native aggregate index projection is unavailable")
    try:
        result = normalize(extent, str(index))
    except (OverflowError, TypeError, ValueError) as error:
        raise CapabilityError(
            f"Q8686: native aggregate index projection failed: {error}"
        ) from error
    if result is None:
        return None
    if type(result) is not int or not 0 <= result < extent:
        raise CapabilityError("Q8686: native aggregate index projection is malformed")
    return result


def _float_operation(name: str, arity: int, *, boolean: bool = False) -> Any:
    """Instantiate one HUGR float operation with its concrete port types.

    Args:
        name: Operation name in the standard float extension.
        arity: Number of binary64 input ports.
        boolean: Whether the operation returns HUGR Bool instead of float64.

    Returns:
        A concretely typed HUGR extension operation.
    """

    from hugr import tys
    from hugr.std.float import FLOAT_OPS_EXTENSION, FLOAT_T

    return FLOAT_OPS_EXTENSION.get_op(name).instantiate(
        concrete_signature=tys.FunctionType(
            [FLOAT_T for _ in range(arity)],
            [tys.Bool if boolean else FLOAT_T],
        )
    )


def _float64_to_int64_bits_operation() -> Any:
    """Instantiate the exact binary64-to-int64 bitcast operation.

    Returns:
        A concretely typed HUGR conversion operation.
    """

    from hugr import tys
    from hugr.std.float import FLOAT_T
    from hugr.std.int import CONVERSIONS_EXTENSION, int_t

    return CONVERSIONS_EXTENSION.get_op("bytecast_float64_to_int64").instantiate(
        concrete_signature=tys.FunctionType([FLOAT_T], [int_t(6)])
    )


def _int64_operation(name: str, arity: int, *, boolean: bool = False) -> Any:
    """Instantiate one HUGR fixed-width 64-bit integer operation.

    Args:
        name: Operation name in the standard integer extension.
        arity: Number of int64 input ports.
        boolean: Whether the operation returns HUGR Bool instead of int64.

    Returns:
        A concretely typed HUGR extension operation.
    """

    from hugr import tys
    from hugr.std.int import INT_OPS_EXTENSION, int_t

    int64 = int_t(6)
    return INT_OPS_EXTENSION.get_op(name).instantiate(
        args=[tys.BoundedNatArg(6)],
        concrete_signature=tys.FunctionType(
            [int64 for _ in range(arity)],
            [tys.Bool if boolean else int64],
        ),
    )


def _guard_real_wire(
    builder: Any,
    value: Any,
    condition: Any,
    message: str,
) -> Any:
    """Return a Real wire on success and terminate its HUGR path on failure.

    Args:
        builder: Active HUGR dataflow builder.
        value: Runtime float64 wire carried through the guard.
        condition: HUGR Bool selecting the successful path.
        message: Stable Qamomile runtime diagnostic.

    Returns:
        The guarded HUGR float64 wire.
    """

    from hugr import tys
    from hugr.std.float import FLOAT_T
    from hugr.std.prelude import PRELUDE_EXTENSION, STRING_T, StringVal

    successful = builder.add_if(condition, value)
    successful.set_outputs(*successful.inputs())
    failed = successful.add_else()
    [failed_value] = failed.inputs()
    code = failed.add_op(
        PRELUDE_EXTENSION.get_op("load_nat").instantiate(
            args=[tys.BoundedNatArg(_HUGR_REAL_RUNTIME_ERROR_CODE)],
            concrete_signature=tys.FunctionType([], [tys.USize()]),
        )
    )[0]
    text = failed.load(StringVal(message))[0]
    error_type = PRELUDE_EXTENSION.types["error"].instantiate([])
    error = failed.add_op(
        PRELUDE_EXTENSION.get_op("MakeError").instantiate(
            concrete_signature=tys.FunctionType(
                [tys.USize(), STRING_T],
                [error_type],
            )
        ),
        code,
        text,
    )[0]
    panic_args = [
        tys.ListArg([tys.TypeTypeArg(FLOAT_T)]),
        tys.ListArg([tys.TypeTypeArg(FLOAT_T)]),
    ]
    terminated = failed.add_op(
        PRELUDE_EXTENSION.get_op("panic").instantiate(
            args=panic_args,
            concrete_signature=tys.FunctionType(
                [error_type, FLOAT_T],
                [FLOAT_T],
            ),
        ),
        error,
        failed_value,
    )[0]
    failed.set_outputs(terminated)
    return successful.conditional_node[0]


def _require_static_angle(value: object, label: str) -> float:
    """Require one finite statically specialized radian angle."""

    if isinstance(value, _RealWire):
        raise CapabilityError(f"Q8628: {label} must be statically specialized")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CapabilityError(f"Q8628: {label} must be a real scalar")
    angle = float(value)
    if not math.isfinite(angle):
        raise CapabilityError(f"Q8628: {label} must be finite")
    return angle


def _negate_angle(builder: Any, value: float | _RealWire) -> float | _RealWire:
    """Negate a static angle or its typed HUGR float64 wire."""

    if isinstance(value, _RealWire):
        result = builder.add_op(
            _float_operation("fneg", 1),
            value.wire,
        )[0]
        return _RealWire(result, value.guaranteed_finite)
    return -value


def _require_hamiltonian(value: object) -> Hamiltonian:
    """Decode one static portable Hamiltonian for direct HUGR lowering."""

    try:
        return Hamiltonian.from_portable(value)
    except (TypeError, ValueError) as error:
        raise CapabilityError("Q8661: malformed Pauli Hamiltonian") from error


def _rotation_wire(
    builder: Any,
    angle: float | _RealWire,
    *,
    scale: float = 1.0,
) -> Any:
    """Convert a static or runtime radian angle to TKET half turns."""

    from hugr.std.float import FloatVal
    from tket_exts import rotation

    if isinstance(angle, _RealWire):
        if not angle.guaranteed_finite:
            raise CapabilityError(
                "Q8628: unguarded runtime angle reached HUGR rotation emission"
            )
        factor = builder.load(FloatVal(scale / math.pi))[0]
        halfturns = builder.add_op(
            _float_operation("fmul", 2),
            angle.wire,
            factor,
        )[0]
    else:
        halfturns = builder.load(FloatVal(angle * scale / math.pi))[0]
    return builder.add_op(rotation.from_halfturns_unchecked, halfturns)[0]


def _emit_global_phase(
    builder: Any,
    angle: float | _RealWire,
    *,
    scale: float = 1.0,
) -> None:
    """Emit TKET's exact zero-qubit global-phase operation."""

    import tket_exts

    rotation = _rotation_wire(builder, angle, scale=scale)
    operation = tket_exts.global_phase().get_op("global_phase").instantiate()
    builder.add_op(operation, rotation)


def _literal_bits(value: object, *, width: int | None = None) -> tuple[int, ...]:
    """Normalize static logical data as source-ordered bits.

    Args:
        value: Boolean, bit sequence, or nonnegative unsigned integer.
        width: Declared logical width for unsigned-integer expansion.

    Returns:
        Source-ordered bits with the most-significant UInt bit first.

    Raises:
        CapabilityError: If the value cannot inhabit the declared bit width.
    """

    if isinstance(value, bool):
        return (int(value),)
    if isinstance(value, int) and value in {0, 1}:
        if width is None or width == 1:
            return (value,)
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        and width is not None
        and value.bit_length() <= width
    ):
        return tuple((value >> shift) & 1 for shift in reversed(range(width)))
    if isinstance(value, tuple) and all(item in {0, 1, False, True} for item in value):
        return tuple(int(item) for item in value)
    raise CapabilityError(f"Q8635: {value!r} is not static bit data")
