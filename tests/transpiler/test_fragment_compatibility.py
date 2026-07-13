"""Compatibility regressions for circuit-fragment compiler extensions."""

from __future__ import annotations

from typing import Any

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.transpiler import EntrypointMode, QamomileCompiler
from qamomile.circuit.transpiler.circuit_ir import CircuitBuilder
from qamomile.circuit.transpiler.circuit_ir.lowering import CircuitLoweringPass
from qamomile.circuit.transpiler.circuit_planner import CircuitPlanningPipeline
from qamomile.circuit.transpiler.executable import QuantumExecutor
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.emit_support import ClbitMap, QubitMap
from qamomile.circuit.transpiler.passes.separate import SegmentationPass
from qamomile.circuit.transpiler.prepared import prepare_module
from qamomile.circuit.transpiler.segments import (
    ProgramPlan,
    QuantumSegment,
    QuantumStep,
)
from qamomile.circuit.transpiler.transpiler import Transpiler


@qmc.qkernel
def _ordinary_program() -> qmc.Bit:
    """Build a minimal executable program.

    Returns:
        qmc.Bit: Measurement result of one internally allocated qubit.
    """
    qubit = qmc.qubit("qubit")
    return qmc.measure(qubit)


@qmc.qkernel
def _quantum_fragment(qubit: qmc.Qubit) -> qmc.Qubit:
    """Build a minimal external-wire circuit fragment.

    Args:
        qubit (qmc.Qubit): External input qubit.

    Returns:
        qmc.Qubit: External qubit after one Hadamard gate.
    """
    return qmc.h(qubit)


class _LegacyEmitPass(EmitPass[list[Operation]]):
    """Record the value delivered through the pre-fragment emit hook."""

    def __init__(self) -> None:
        """Initialize the hook recorder."""
        super().__init__()
        self.received: object | None = None

    def _emit_quantum_segment(
        self,
        operations: list[Operation],
        bindings: dict[str, Any],
    ) -> tuple[list[Operation], QubitMap, ClbitMap]:
        """Record the legacy operation-list hook argument.

        Args:
            operations (list[Operation]): Semantic quantum operations.
            bindings (dict[str, Any]): Compile-time emit bindings.

        Returns:
            tuple[list[Operation], QubitMap, ClbitMap]: Recorded operations and
                empty resource maps.
        """
        del bindings
        self.received = operations
        circuit = operations if isinstance(operations, list) else []
        return circuit, {}, {}


class _HookTrackingTranspiler(Transpiler[object]):
    """Track calls through the existing overrideable planning stages."""

    def __init__(self) -> None:
        """Initialize stage counters."""
        self.unroll_recursion_calls = 0
        self.partial_eval_calls = 0
        self.plan_calls = 0

    def _create_segmentation_pass(self) -> SegmentationPass:
        """Create the default segmentation pass.

        Returns:
            SegmentationPass: Default circuit-family segmentation pass.
        """
        return SegmentationPass()

    def _create_emit_pass(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> EmitPass[object]:
        """Reject emission, which is outside this planning-only fixture.

        Args:
            bindings (dict[str, Any] | None): Unused compile-time bindings.
            parameters (list[str] | None): Unused runtime parameter names.

        Returns:
            EmitPass[object]: This method never returns.

        Raises:
            NotImplementedError: Always, because the fixture tests planning.
        """
        del bindings, parameters
        raise NotImplementedError

    def executor(self, **kwargs: Any) -> QuantumExecutor[object]:
        """Reject executor creation for the planning-only fixture.

        Args:
            **kwargs (Any): Unused executor options.

        Returns:
            QuantumExecutor[object]: This method never returns.

        Raises:
            NotImplementedError: Always, because the fixture tests planning.
        """
        del kwargs
        raise NotImplementedError

    def partial_eval(
        self,
        block: Block,
        compile_values: dict[str, Any] | None = None,
    ) -> Block:
        """Record and delegate one partial-evaluation stage.

        Args:
            block (Block): Affine block to partially evaluate.
            compile_values (dict[str, Any] | None): Compile-time bindings.

        Returns:
            Block: Partially evaluated block.
        """
        self.partial_eval_calls += 1
        return super().partial_eval(block, compile_values)

    def unroll_recursion(
        self,
        block: Block,
        compile_values: dict[str, Any] | None = None,
    ) -> Block:
        """Record recursion expansion with a renamed binding parameter.

        Args:
            block (Block): Hierarchical or affine block to expand.
            compile_values (dict[str, Any] | None): Compile-time bindings.

        Returns:
            Block: Expanded affine block.
        """
        self.unroll_recursion_calls += 1
        return super().unroll_recursion(block, compile_values)

    def plan(self, block: Block) -> ProgramPlan:
        """Record and delegate the segmentation stage.

        Args:
            block (Block): Analyzed block to segment.

        Returns:
            ProgramPlan: Circuit-family execution plan.
        """
        self.plan_calls += 1
        return super().plan(block)


def test_plan_circuit_preserves_transpiler_stage_overrides() -> None:
    """The shared planner must not bypass backend planning extensions."""
    prepared = QamomileCompiler().prepare(_ordinary_program)
    transpiler = _HookTrackingTranspiler()

    transpiler.plan_circuit(prepared)

    assert transpiler.partial_eval_calls == 1
    assert transpiler.unroll_recursion_calls == 1
    assert transpiler.plan_calls == 1


class _LegacyCircuitLoweringPass(CircuitLoweringPass):
    """Record the legacy StandardEmitPass operation-list hook."""

    def __init__(self) -> None:
        """Initialize the hook recorder."""
        super().__init__()
        self.legacy_override_called = False

    def _emit_quantum_segment(
        self,
        operations: list[Operation],
        bindings: dict[str, Any],
    ) -> tuple[CircuitBuilder, QubitMap, ClbitMap]:
        """Record and delegate the established StandardEmitPass hook.

        Args:
            operations (list[Operation]): Semantic quantum operations.
            bindings (dict[str, Any]): Compile-time emit bindings.

        Returns:
            tuple[CircuitBuilder, QubitMap, ClbitMap]: Lowered builder and
                resource maps.
        """
        self.legacy_override_called = True
        return super()._emit_quantum_segment(operations, bindings)


def test_standard_emit_subclass_keeps_legacy_quantum_hook() -> None:
    """Ordinary segments still invoke StandardEmitPass subclass hooks."""
    prepared = QamomileCompiler().prepare(_ordinary_program)
    plan = CircuitPlanningPipeline().run(prepared)
    emit_pass = _LegacyCircuitLoweringPass()

    emit_pass.run(plan)

    assert emit_pass.legacy_override_called


def test_emit_pass_legacy_hook_receives_operation_list() -> None:
    """Existing EmitPass subclasses keep their operation-list contract."""
    segment = QuantumSegment()
    emit_pass = _LegacyEmitPass()

    emit_pass.run(ProgramPlan(steps=[QuantumStep(segment)]))

    assert emit_pass.received is segment.operations


def test_prepare_block_accepts_string_fragment_mode() -> None:
    """The public preparation boundary normalizes a valid mode string."""
    prepared = QamomileCompiler().prepare_block(
        _quantum_fragment.block,
        mode="circuit_fragment",
    )

    assert prepared.mode is EntrypointMode.CIRCUIT_FRAGMENT
    assert (
        prepare_module(_quantum_fragment.block, mode="circuit_fragment").mode
        is EntrypointMode.CIRCUIT_FRAGMENT
    )


def test_prepare_block_rejects_unknown_mode() -> None:
    """An invalid entrypoint mode fails immediately with valid choices."""
    with pytest.raises(ValueError, match=r"program.*circuit_fragment"):
        QamomileCompiler().prepare_block(
            _quantum_fragment.block,
            mode="approximate",
        )
    with pytest.raises(ValueError, match=r"program.*circuit_fragment"):
        prepare_module(_quantum_fragment.block, mode="approximate")
