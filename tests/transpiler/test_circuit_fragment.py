"""Tests for quantum-I/O circuit-fragment planning and lowering."""

from __future__ import annotations

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.qkernel_build import create_traced_block
from qamomile.circuit.ir.block import Block
from qamomile.circuit.transpiler import (
    CircuitPlanningPipeline,
    EntrypointMode,
    QamomileCompiler,
)
from qamomile.circuit.transpiler.circuit_ir import (
    CircuitProgram,
    GateInstruction,
    MeasureInstruction,
    WireId,
    lower_circuit_plan,
    verify_circuit,
)
from qamomile.circuit.transpiler.errors import EntrypointValidationError
from qamomile.circuit.transpiler.gate_emitter import GateKind
from qamomile.circuit.transpiler.passes.analyze import AnalyzePass
from qamomile.circuit.transpiler.passes.emit_support import QubitAddress
from qamomile.circuit.transpiler.passes.inline import InlinePass


@qmc.qkernel
def _scalar_fragment(qubit: qmc.Qubit) -> qmc.Qubit:
    """Apply one gate to an externally supplied scalar qubit.

    Args:
        qubit (qmc.Qubit): External input qubit.

    Returns:
        qmc.Qubit: Updated external qubit.
    """
    return qmc.h(qubit)


@qmc.qkernel
def _measure_fragment(qubit: qmc.Qubit) -> qmc.Bit:
    """Measure an external qubit without a preceding allocating gate.

    Args:
        qubit (qmc.Qubit): External input qubit.

    Returns:
        qmc.Bit: Measurement result.
    """
    return qmc.measure(qubit)


@qmc.qkernel
def _vector_fragment(
    qubits: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Apply one gate to the middle wire of an external register.

    Args:
        qubits (qmc.Vector[qmc.Qubit]): External three-qubit register.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated external register.
    """
    qubits[1] = qmc.h(qubits[1])
    return qubits


@qmc.qkernel
def _identity_fragment(qubit: qmc.Qubit) -> qmc.Qubit:
    """Return one external wire without adding an operation.

    Args:
        qubit (qmc.Qubit): External input qubit.

    Returns:
        qmc.Qubit: Unchanged external qubit.
    """
    return qubit


@qmc.qkernel
def _ordinary_program() -> qmc.Bit:
    """Build a classical-I/O program for compatibility coverage.

    Returns:
        qmc.Bit: Measurement result from the internally allocated qubit.
    """
    qubit = qmc.qubit("qubit")
    qubit = qmc.h(qubit)
    return qmc.measure(qubit)


def _lower_fragment(block: Block) -> tuple[CircuitProgram, dict[QubitAddress, int]]:
    """Prepare, plan, and lower a quantum-I/O semantic block.

    Args:
        block (Block): Traced or hierarchical quantum-I/O block.

    Returns:
        tuple[CircuitProgram, dict[QubitAddress, int]]: Verified circuit and
            its semantic-address-to-slot map.

    Raises:
        AssertionError: If lowering does not produce exactly one quantum
            segment.
    """
    compiler = QamomileCompiler()
    prepared = compiler.prepare_block(
        block,
        mode=EntrypointMode.CIRCUIT_FRAGMENT,
    )
    plan = CircuitPlanningPipeline().run(prepared)
    executable = lower_circuit_plan(plan)
    assert len(executable.compiled_quantum) == 1
    compiled = executable.compiled_quantum[0]
    verify_circuit(compiled.circuit)
    return compiled.circuit, compiled.qubit_map


def test_scalar_quantum_input_lowers_as_external_wire() -> None:
    """A scalar fragment starts at an input wire without QInit/reset."""
    block = _scalar_fragment.block
    program, qubit_map = _lower_fragment(block)

    assert program.num_qubits == 1
    assert program.input_wires == (WireId(0),)
    assert len(program.operations) == 1
    [gate] = program.operations
    assert isinstance(gate, GateInstruction)
    assert gate.kind is GateKind.H
    assert gate.inputs == (WireId(0),)
    assert qubit_map[QubitAddress(block.input_values[0].uuid)] == 0


def test_measure_only_fragment_seeds_external_wire() -> None:
    """Measurement can consume an external qubit with no earlier gate."""
    program, _ = _lower_fragment(_measure_fragment.block)

    assert program.num_qubits == 1
    assert len(program.operations) == 1
    assert isinstance(program.operations[0], MeasureInstruction)


def test_vector_quantum_input_preserves_exact_external_width() -> None:
    """A concrete Vector input maps to its declared wires without a phantom."""
    block = create_traced_block(
        _vector_fragment,
        parameters=[],
        kwargs={},
        qubit_sizes={"qubits": 3},
        emit_qubit_init=False,
        emit_return_op=True,
    )
    program, qubit_map = _lower_fragment(block)
    input_array = block.input_values[0]

    assert program.num_qubits == 3
    assert program.input_wires == (WireId(0), WireId(1), WireId(2))
    assert [qubit_map[QubitAddress(input_array.uuid, index)] for index in range(3)] == [
        0,
        1,
        2,
    ]
    [gate] = program.operations
    assert isinstance(gate, GateInstruction)
    assert gate.kind is GateKind.H
    assert gate.inputs == (WireId(1),)


def test_identity_fragment_produces_empty_identity_program() -> None:
    """A gate-free quantum interface remains an identity CircuitProgram."""
    program, _ = _lower_fragment(_identity_fragment.block)

    assert program.num_qubits == 1
    assert program.operations == ()
    assert program.input_wires == program.output_wires == (WireId(0),)


def test_program_mode_still_rejects_quantum_io() -> None:
    """Executable program preparation keeps the classical-only ABI rule."""
    with pytest.raises(
        EntrypointValidationError,
        match="classical inputs and outputs only",
    ):
        QamomileCompiler().prepare_block(
            _scalar_fragment.block,
            mode=EntrypointMode.PROGRAM,
        )


@pytest.mark.parametrize("stage", ["affine", "analyzed"])
def test_program_mode_rejects_later_stage_quantum_io(stage: str) -> None:
    """The executable ABI contract is independent of compiler stage.

    Args:
        stage (str): Later semantic stage supplied to preparation.
    """
    affine = InlinePass().run(_scalar_fragment.block)
    block = (
        affine
        if stage == "affine"
        else AnalyzePass(validate_classical_io=False).run(affine)
    )

    with pytest.raises(
        EntrypointValidationError,
        match="classical inputs and outputs only",
    ):
        QamomileCompiler().prepare_block(block, mode=EntrypointMode.PROGRAM)


def test_ordinary_program_uses_shared_circuit_planner() -> None:
    """The extracted planner preserves ordinary classical-I/O lowering."""
    compiler = QamomileCompiler()
    prepared = compiler.prepare(_ordinary_program)
    plan = CircuitPlanningPipeline().run(prepared)
    executable = lower_circuit_plan(plan)
    [compiled] = executable.compiled_quantum

    verify_circuit(compiled.circuit)
    assert compiled.circuit.num_qubits == 1
    assert [type(operation) for operation in compiled.circuit.operations] == [
        GateInstruction,
        MeasureInstruction,
    ]
