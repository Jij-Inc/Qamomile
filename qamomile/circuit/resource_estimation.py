"""Resource estimation for quantum circuits.

This module provides tools for estimating the resources required by
quantum circuits without full compilation, enabling optimization
decisions before execution.

Example:
    from qamomile.circuit.resource_estimation import ResourceEstimator

    estimator = ResourceEstimator()

    # Estimate resources for a kernel
    estimate = estimator.estimate(my_kernel, bindings={"n": 5})
    print(f"Total gates: {estimate.total_gates}")
    print(f"CNOT count: {estimate.cnot_count}")

    # Compare strategies
    comparison = estimator.compare_strategies(
        my_kernel,
        strategies=["standard", "approximate"],
        bindings={"n": 5},
    )
    for name, est in comparison.items():
        print(f"{name}: {est.total_gates} gates")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from qamomile.circuit.frontend.decomposition import DecompositionConfig
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.operation import CInitOperation, QInitOperation
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    GateOperation,
    GateOperationType,
    MeasureOperation,
    MeasureVectorOperation,
    MeasureQFixedOperation,
)
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.classical_ops import DecodeQFixedOperation
from qamomile.circuit.ir.operation.arithmetic_operations import BinaryOperationBase
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateOperation,
    CompositeGateType,
    ResourceMetadata,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    ForItemsOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.transpiler.passes.inline import InlinePass
from qamomile.circuit.transpiler.passes.linear_validate import LinearValidationPass

if TYPE_CHECKING:
    from qamomile.circuit.frontend.qkernel import QKernel


def linear_preflight(block: Block, bindings: dict[str, Any] | None = None) -> Block:
    """Run inline + linear validation as a preflight check.

    This ensures that the block is free of linear type violations
    before resource estimation proceeds.

    Args:
        block: The block to validate.
        bindings: Optional parameter bindings.

    Returns:
        The inlined, validated block.

    Raises:
        LinearTypeError: If linear type violations are detected.
    """
    inlined = InlinePass().run(block)
    LinearValidationPass().run(inlined)
    return inlined


@dataclass
class ResourceEstimate:
    """Resource estimate for a quantum circuit.

    Attributes:
        total_gates: Total number of gates
        t_gate_count: Number of T gates
        cnot_count: Number of CNOT/CX gates
        qubit_count: Number of qubits used
        depth: Estimated circuit depth (approximation)
        gate_counts: Breakdown by gate type
        composite_resources: Resources from composite gates
        measurement_count: Number of measurements
    """

    total_gates: int = 0
    t_gate_count: int = 0
    cnot_count: int = 0
    qubit_count: int = 0
    depth: int = 0
    gate_counts: dict[str, int] = field(default_factory=dict)
    composite_resources: dict[str, ResourceMetadata] = field(default_factory=dict)
    measurement_count: int = 0

    def __add__(self, other: "ResourceEstimate") -> "ResourceEstimate":
        """Combine two resource estimates."""
        combined_gate_counts = dict(self.gate_counts)
        for gate, count in other.gate_counts.items():
            combined_gate_counts[gate] = combined_gate_counts.get(gate, 0) + count

        combined_composite = dict(self.composite_resources)
        combined_composite.update(other.composite_resources)

        return ResourceEstimate(
            total_gates=self.total_gates + other.total_gates,
            t_gate_count=self.t_gate_count + other.t_gate_count,
            cnot_count=self.cnot_count + other.cnot_count,
            qubit_count=max(self.qubit_count, other.qubit_count),
            depth=self.depth + other.depth,
            gate_counts=combined_gate_counts,
            composite_resources=combined_composite,
            measurement_count=self.measurement_count + other.measurement_count,
        )

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Resource Estimate:",
            f"  Total gates: {self.total_gates}",
            f"  Qubits: {self.qubit_count}",
            f"  CNOT gates: {self.cnot_count}",
            f"  T gates: {self.t_gate_count}",
            f"  Measurements: {self.measurement_count}",
            f"  Estimated depth: {self.depth}",
        ]

        if self.gate_counts:
            lines.append("  Gate breakdown:")
            for gate, count in sorted(self.gate_counts.items()):
                lines.append(f"    {gate}: {count}")

        if self.composite_resources:
            lines.append("  Composite gate resources:")
            for name, meta in self.composite_resources.items():
                lines.append(f"    {name}:")
                if meta.t_gate_count is not None:
                    lines.append(f"      T gates: {meta.t_gate_count}")
                if meta.custom_metadata:
                    for k, v in meta.custom_metadata.items():
                        lines.append(f"      {k}: {v}")

        return "\n".join(lines)


class ResourceEstimator:
    """Estimates resources required by quantum circuits.

    This class analyzes QKernel or Block to estimate gate counts,
    qubit usage, and other resources without full compilation.

    Example:
        estimator = ResourceEstimator()
        estimate = estimator.estimate(my_kernel)
        print(estimate.summary())
    """

    def __init__(
        self,
        decomposition_config: DecompositionConfig | None = None,
    ) -> None:
        """Initialize the estimator.

        Args:
            decomposition_config: Configuration for decomposition strategies.
                Used to determine which strategy to use for composite gates.
        """
        self._config = decomposition_config or DecompositionConfig()

    def estimate(
        self,
        kernel: "QKernel",
        bindings: dict[str, Any] | None = None,
        validate_linear: bool = False,
    ) -> ResourceEstimate:
        """Estimate resources for a QKernel.

        Args:
            kernel: The kernel to analyze.
            bindings: Optional parameter bindings for accurate loop counts.
            validate_linear: If True, run linear type validation before
                estimating. Default is False for backward compatibility.
                Will become True in a future version.

        Returns:
            ResourceEstimate with gate counts and other metrics.

        Raises:
            LinearTypeError: If ``validate_linear`` is True and violations
                are detected.
        """
        # Get the block from the kernel
        if bindings:
            graph = kernel.build(**(bindings or {}))
            block = Block(
                name=graph.name,
                label_args=[],
                input_values=graph.input_values,
                output_values=graph.output_values,
                operations=graph.operations,
                kind=BlockKind.HIERARCHICAL,
                parameters=graph.parameters,
            )
        else:
            from qamomile.circuit.ir.block_value import BlockValue

            block_value = kernel.block
            block = Block.from_block_value(block_value, {})

        return self.estimate_block(
            block, bindings or {}, validate_linear=validate_linear
        )

    def estimate_block(
        self,
        block: Block,
        bindings: dict[str, Any] | None = None,
        validate_linear: bool = False,
    ) -> ResourceEstimate:
        """Estimate resources for a Block.

        Args:
            block: The block to analyze.
            bindings: Optional parameter bindings.
            validate_linear: If True, run linear type validation before
                estimating. Default is False for backward compatibility.
                Will become True in a future version.

        Returns:
            ResourceEstimate with gate counts and other metrics.

        Raises:
            LinearTypeError: If ``validate_linear`` is True and violations
                are detected.
        """
        bindings = bindings or {}

        if validate_linear:
            linear_preflight(block, bindings)

        # Count qubits from QInitOperations
        qubit_count = 0
        for op in block.operations:
            if isinstance(op, QInitOperation):
                qubit_count += 1

        # Also count from input values
        for inp in block.input_values:
            if hasattr(inp, "type") and inp.type.is_quantum():
                qubit_count += 1

        estimate = self._estimate_operations(block.operations, bindings)
        estimate.qubit_count = max(estimate.qubit_count, qubit_count)

        return estimate

    def compare_strategies(
        self,
        kernel: "QKernel",
        strategies: list[str],
        bindings: dict[str, Any] | None = None,
    ) -> dict[str, ResourceEstimate]:
        """Compare resource estimates across different strategies.

        Args:
            kernel: The kernel to analyze
            strategies: List of strategy names to compare
            bindings: Optional parameter bindings

        Returns:
            Map of strategy name to ResourceEstimate
        """
        results: dict[str, ResourceEstimate] = {}

        for strategy in strategies:
            config = DecompositionConfig(
                strategy_overrides={
                    "qft": strategy,
                    "iqft": strategy,
                },
            )
            estimator = ResourceEstimator(decomposition_config=config)
            results[strategy] = estimator.estimate(kernel, bindings)

        return results

    def _estimate_operations(
        self,
        operations: list[Operation],
        bindings: dict[str, Any],
    ) -> ResourceEstimate:
        """Estimate resources for a list of operations.

        Args:
            operations: Operations to analyze
            bindings: Parameter bindings

        Returns:
            ResourceEstimate
        """
        estimate = ResourceEstimate()

        for op in operations:
            if isinstance(op, QInitOperation):
                estimate.qubit_count += 1

            elif isinstance(op, GateOperation):
                self._count_gate(op, estimate)

            elif isinstance(
                op, (MeasureOperation, MeasureVectorOperation, MeasureQFixedOperation)
            ):
                estimate.measurement_count += 1

            elif isinstance(op, CompositeGateOperation):
                self._count_composite_gate(op, estimate)

            elif isinstance(op, ForOperation):
                self._count_for_loop(op, estimate, bindings)

            elif isinstance(op, ForItemsOperation):
                self._count_for_items_loop(op, estimate, bindings)

            elif isinstance(op, IfOperation):
                self._count_if_operation(op, estimate, bindings)

            elif isinstance(op, WhileOperation):
                # While loops are hard to estimate; count body once
                body_estimate = self._estimate_operations(op.operations, bindings)
                estimate = estimate + body_estimate

            elif isinstance(op, CallBlockOperation):
                from qamomile.circuit.ir.block_value import BlockValue

                called_block = op.operands[0]
                if isinstance(called_block, BlockValue):
                    body_estimate = self._estimate_operations(
                        called_block.operations, bindings
                    )
                    estimate = estimate + body_estimate

            elif isinstance(op, ControlledUOperation):
                from qamomile.circuit.ir.block_value import BlockValue

                controlled_block = op.block
                if isinstance(controlled_block, BlockValue):
                    body_estimate = self._estimate_operations(
                        controlled_block.operations, bindings
                    )
                    power = self._resolve_int(op.power, bindings, 1)
                    if power is not None and power > 1:
                        body_estimate = ResourceEstimate(
                            total_gates=body_estimate.total_gates * power,
                            t_gate_count=body_estimate.t_gate_count * power,
                            cnot_count=body_estimate.cnot_count * power,
                            qubit_count=body_estimate.qubit_count,
                            depth=body_estimate.depth * power,
                            gate_counts={
                                k: v * power
                                for k, v in body_estimate.gate_counts.items()
                            },
                            composite_resources=body_estimate.composite_resources,
                            measurement_count=(body_estimate.measurement_count * power),
                        )
                    estimate = estimate + body_estimate

            elif isinstance(
                op,
                (
                    CInitOperation,
                    ReturnOperation,
                    CastOperation,
                    DecodeQFixedOperation,
                ),
            ):
                pass  # No gate-count contribution

            elif isinstance(op, BinaryOperationBase):
                pass  # Classical arithmetic — no gates

            else:
                raise ValueError(
                    f"ResourceEstimator encountered unhandled operation type "
                    f"'{type(op).__name__}'. This may result in inaccurate "
                    f"estimates. Please report this as a bug."
                )

        return estimate

    def _count_gate(self, op: GateOperation, estimate: ResourceEstimate) -> None:
        """Count a single gate operation."""
        gate_name = op.gate_type.name.lower()
        estimate.gate_counts[gate_name] = estimate.gate_counts.get(gate_name, 0) + 1
        estimate.total_gates += 1
        estimate.depth += 1  # Simplified depth estimation

        # Count specific gate types
        if op.gate_type == GateOperationType.T:
            estimate.t_gate_count += 1
        elif op.gate_type == GateOperationType.CX:
            estimate.cnot_count += 1

    def _count_composite_gate(
        self,
        op: CompositeGateOperation,
        estimate: ResourceEstimate,
    ) -> None:
        """Count resources for a composite gate."""
        # Get strategy name from operation or config
        strategy_name = op.strategy_name
        if strategy_name is None:
            strategy_name = self._config.get_strategy_for_gate(op.name)

        # Try to get resources from the composite gate instance
        if op.composite_gate_instance is not None:
            resources = op.composite_gate_instance.get_resources_for_strategy(
                strategy_name
            )
            if resources is not None:
                self._add_composite_resources(op.name, resources, estimate)
                return

        # Fall back to resource_metadata on the operation
        if op.resource_metadata is not None:
            self._add_composite_resources(op.name, op.resource_metadata, estimate)
            return

        # Estimate based on gate type
        self._estimate_composite_fallback(op, estimate)

    def _add_composite_resources(
        self,
        name: str,
        resources: ResourceMetadata,
        estimate: ResourceEstimate,
    ) -> None:
        """Add composite gate resources to the estimate."""
        estimate.composite_resources[name] = resources

        if resources.t_gate_count is not None:
            estimate.t_gate_count += resources.t_gate_count

        if resources.custom_metadata:
            meta = resources.custom_metadata
            if "total_gates" in meta:
                estimate.total_gates += meta["total_gates"]
                estimate.depth += meta.get("depth", meta["total_gates"])
            if "num_cp_gates" in meta:
                # CP gates contribute to depth
                pass

    def _estimate_composite_fallback(
        self,
        op: CompositeGateOperation,
        estimate: ResourceEstimate,
    ) -> None:
        """Estimate composite gate resources as fallback."""
        n = op.num_target_qubits

        if op.gate_type == CompositeGateType.QFT:
            # Standard QFT: n H gates + n(n-1)/2 CP gates + n/2 SWAPs
            h_gates = n
            cp_gates = n * (n - 1) // 2
            swap_gates = n // 2
            total = h_gates + cp_gates + swap_gates
            estimate.total_gates += total
            estimate.gate_counts["h"] = estimate.gate_counts.get("h", 0) + h_gates
            estimate.gate_counts["cp"] = estimate.gate_counts.get("cp", 0) + cp_gates
            estimate.gate_counts["swap"] = (
                estimate.gate_counts.get("swap", 0) + swap_gates
            )
            estimate.depth += n

        elif op.gate_type == CompositeGateType.IQFT:
            # Same as QFT
            h_gates = n
            cp_gates = n * (n - 1) // 2
            swap_gates = n // 2
            total = h_gates + cp_gates + swap_gates
            estimate.total_gates += total
            estimate.depth += n

        elif op.gate_type == CompositeGateType.QPE:
            # QPE: H on counting qubits + controlled-U powers + IQFT
            num_counting = op.num_control_qubits
            estimate.total_gates += num_counting  # H gates
            # Controlled-U powers (rough estimate)
            estimate.total_gates += num_counting * n
            # IQFT
            estimate.total_gates += (
                num_counting + num_counting * (num_counting - 1) // 2
            )

        else:
            # Unknown composite gate - just count as 1 operation
            estimate.total_gates += 1

    def _count_for_loop(
        self,
        op: ForOperation,
        estimate: ResourceEstimate,
        bindings: dict[str, Any],
    ) -> None:
        """Count resources for a for loop."""
        # Try to resolve loop bounds
        start = self._resolve_int(
            op.operands[0] if len(op.operands) > 0 else None, bindings, 0
        )
        stop = self._resolve_int(
            op.operands[1] if len(op.operands) > 1 else None, bindings, 1
        )
        step = self._resolve_int(
            op.operands[2] if len(op.operands) > 2 else None, bindings, 1
        )

        if start is not None and stop is not None and step is not None and step != 0:
            iterations = max(0, (stop - start + step - 1) // step) if step > 0 else 0
        else:
            # Unknown iteration count - assume 1
            iterations = 1

        # Estimate body resources
        body_estimate = self._estimate_operations(op.operations, bindings)

        # Multiply by iterations
        for gate, count in body_estimate.gate_counts.items():
            estimate.gate_counts[gate] = (
                estimate.gate_counts.get(gate, 0) + count * iterations
            )

        estimate.total_gates += body_estimate.total_gates * iterations
        estimate.t_gate_count += body_estimate.t_gate_count * iterations
        estimate.cnot_count += body_estimate.cnot_count * iterations
        estimate.depth += body_estimate.depth * iterations
        estimate.measurement_count += body_estimate.measurement_count * iterations

    def _count_for_items_loop(
        self,
        op: ForItemsOperation,
        estimate: ResourceEstimate,
        bindings: dict[str, Any],
    ) -> None:
        """Count resources for a for-items loop."""
        # Try to resolve dict size from bindings
        iterations = 1
        if op.operands:
            dict_val = op.operands[0]
            if hasattr(dict_val, "name") and dict_val.name in bindings:
                bound = bindings[dict_val.name]
                if isinstance(bound, dict):
                    iterations = len(bound)

        body_estimate = self._estimate_operations(op.operations, bindings)

        for gate, count in body_estimate.gate_counts.items():
            estimate.gate_counts[gate] = (
                estimate.gate_counts.get(gate, 0) + count * iterations
            )

        estimate.total_gates += body_estimate.total_gates * iterations
        estimate.t_gate_count += body_estimate.t_gate_count * iterations
        estimate.cnot_count += body_estimate.cnot_count * iterations
        estimate.depth += body_estimate.depth * iterations

    def _count_if_operation(
        self,
        op: IfOperation,
        estimate: ResourceEstimate,
        bindings: dict[str, Any],
    ) -> None:
        """Count resources for an if operation (conservative: count both branches)."""
        true_estimate = self._estimate_operations(op.true_operations, bindings)
        false_estimate = self._estimate_operations(op.false_operations, bindings)

        # Take maximum of both branches (conservative estimate)
        for gate in set(true_estimate.gate_counts) | set(false_estimate.gate_counts):
            true_count = true_estimate.gate_counts.get(gate, 0)
            false_count = false_estimate.gate_counts.get(gate, 0)
            estimate.gate_counts[gate] = estimate.gate_counts.get(gate, 0) + max(
                true_count, false_count
            )

        estimate.total_gates += max(
            true_estimate.total_gates, false_estimate.total_gates
        )
        estimate.t_gate_count += max(
            true_estimate.t_gate_count, false_estimate.t_gate_count
        )
        estimate.cnot_count += max(true_estimate.cnot_count, false_estimate.cnot_count)
        estimate.depth += max(true_estimate.depth, false_estimate.depth)

    def _resolve_int(
        self,
        value: Any,
        bindings: dict[str, Any],
        default: int,
    ) -> int | None:
        """Resolve a value to an integer."""
        if value is None:
            return default

        if isinstance(value, int):
            return value

        if hasattr(value, "get_const"):
            const = value.get_const()
            if const is not None:
                return int(const)

        if hasattr(value, "is_parameter") and value.is_parameter():
            name = value.parameter_name()
            if name and name in bindings:
                return int(bindings[name])

        if hasattr(value, "name") and value.name in bindings:
            return int(bindings[value.name])

        return None
