"""Analyze pass: Validate and analyze dependencies in a linear block."""

from __future__ import annotations

import dataclasses

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.operation import OperationKind
from qamomile.circuit.ir.operation.gate import MeasureOperation
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.passes.control_flow_visitor import ControlFlowVisitor
from qamomile.circuit.transpiler.errors import DependencyError, ValidationError


class AnalyzePass(Pass[Block, Block]):
    """Analyze and validate a linear block.

    This pass:
    1. Builds a dependency graph between values
    2. Validates that quantum ops don't depend on non-parameter classical results
    3. Checks that block inputs/outputs are classical

    Input: Block with BlockKind.LINEAR
    Output: Block with BlockKind.ANALYZED (with _dependency_graph populated)
    """

    @property
    def name(self) -> str:
        return "analyze"

    def run(self, input: Block) -> Block:
        """Analyze the block and validate dependencies."""
        if input.kind != BlockKind.LINEAR:
            raise ValidationError(f"AnalyzePass expects LINEAR block, got {input.kind}")

        # Check inputs/outputs are classical
        self._validate_io_classical(input)

        # Build dependency graph
        dependency_graph = self._build_dependency_graph(input.operations)

        # Validate quantum-classical dependencies
        self._validate_quantum_dependencies(
            input.operations,
            dependency_graph,
            input.parameters,
        )

        return dataclasses.replace(
            input,
            kind=BlockKind.ANALYZED,
            _dependency_graph=dependency_graph,
        )

    def _validate_io_classical(self, block: Block) -> None:
        """Ensure all block inputs and outputs are classical types."""
        for value in block.input_values:
            if value.type.is_quantum():
                raise ValidationError(
                    f"Block input '{value.name}' must be classical type, "
                    f"got {value.type.label()}",
                    value_name=value.name,
                )

        for value in block.output_values:
            if value.type.is_quantum():
                raise ValidationError(
                    f"Block output '{value.name}' must be classical type, "
                    f"got {value.type.label()}",
                    value_name=value.name,
                )

    def _build_dependency_graph(
        self,
        operations: list[Operation],
    ) -> dict[str, set[str]]:
        """Build a map from each value UUID to the UUIDs it depends on."""

        class DependencyGraphBuilder(ControlFlowVisitor):
            def __init__(self):
                self.graph: dict[str, set[str]] = {}

            def visit_operation(self, op: Operation) -> None:
                # Each result depends on all operands
                operand_uuids = {
                    v.uuid for v in op.operands if isinstance(v, Value)
                }
                for result in op.results:
                    if result.uuid not in self.graph:
                        self.graph[result.uuid] = set()
                    self.graph[result.uuid].update(operand_uuids)

        builder = DependencyGraphBuilder()
        builder.visit_operations(operations)
        return builder.graph

    def _validate_quantum_dependencies(
        self,
        operations: list[Operation],
        dependency_graph: dict[str, set[str]],
        parameters: dict[str, Value],
    ) -> None:
        """Ensure quantum ops don't depend on non-parameter classical results.

        A quantum operation can depend on:
        - Other quantum values
        - Parameters (will be bound at runtime)
        - Constant classical values

        It cannot depend on:
        - Classical values computed from measurements (runtime-only)
        """
        # Collect UUIDs of parameter values
        parameter_uuids = {v.uuid for v in parameters.values()}

        # Collect UUIDs of measurement results (runtime classical)
        measurement_uuids = self._find_measurement_results(operations)

        # Collect UUIDs of values derived from measurements
        derived_from_measurement: set[str] = set()
        self._find_measurement_derived_values(
            dependency_graph, measurement_uuids, derived_from_measurement
        )

        outer_self = self

        class QuantumDependencyValidator(ControlFlowVisitor):
            def visit_operation(self, op: Operation) -> None:
                if op.operation_kind != OperationKind.QUANTUM:
                    return

                for operand in op.operands:
                    if not isinstance(operand, Value):
                        continue

                    if outer_self._depends_on_measurement(
                        operand.uuid,
                        dependency_graph,
                        measurement_uuids,
                        parameter_uuids,
                        derived_from_measurement,
                    ):
                        raise DependencyError(
                            f"Quantum operation '{type(op).__name__}' depends on "
                            f"measurement result via value '{operand.name}'. "
                            f"JIT compilation not supported - classical values "
                            f"used in quantum ops must be parameters or constants.",
                            quantum_op=type(op).__name__,
                            classical_value=operand.name,
                        )

        validator = QuantumDependencyValidator()
        validator.visit_operations(operations)

    def _find_measurement_results(
        self,
        operations: list[Operation],
    ) -> set[str]:
        """Find all value UUIDs that are measurement results."""

        class MeasurementResultCollector(ControlFlowVisitor):
            def __init__(self):
                self.result_uuids: set[str] = set()

            def visit_operation(self, op: Operation) -> None:
                if isinstance(op, MeasureOperation):
                    for result in op.results:
                        self.result_uuids.add(result.uuid)

        collector = MeasurementResultCollector()
        collector.visit_operations(operations)
        return collector.result_uuids

    def _find_measurement_derived_values(
        self,
        dependency_graph: dict[str, set[str]],
        measurement_uuids: set[str],
        derived: set[str],
    ) -> None:
        """Find all values that are derived from measurements."""
        # Use a worklist algorithm to find all derived values
        worklist = list(measurement_uuids)

        while worklist:
            current = worklist.pop()
            if current in derived:
                continue
            derived.add(current)

            # Find values that depend on current
            for uuid, deps in dependency_graph.items():
                if current in deps and uuid not in derived:
                    worklist.append(uuid)

    def _depends_on_measurement(
        self,
        value_uuid: str,
        dependency_graph: dict[str, set[str]],
        measurement_uuids: set[str],
        parameter_uuids: set[str],
        derived_from_measurement: set[str],
    ) -> bool:
        """Check if a value transitively depends on a measurement result.

        Returns True if there's a dependency path to a measurement
        that doesn't go through a parameter.
        """
        # Direct measurement dependency
        if value_uuid in measurement_uuids:
            return True

        # Derived from measurement
        if value_uuid in derived_from_measurement:
            return True

        # Parameters are OK
        if value_uuid in parameter_uuids:
            return False

        # Check transitive dependencies
        visited: set[str] = set()

        def dfs(uuid: str) -> bool:
            if uuid in visited:
                return False
            visited.add(uuid)

            # Direct measurement
            if uuid in measurement_uuids:
                return True

            # Parameters are OK
            if uuid in parameter_uuids:
                return False

            # Check dependencies
            deps = dependency_graph.get(uuid, set())
            return any(dfs(dep) for dep in deps)

        return dfs(value_uuid)
