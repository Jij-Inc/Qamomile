"""Tests for backend-neutral circuit code-generation IR."""

from __future__ import annotations

import dataclasses

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.circuit_ir import (
    CallInstruction,
    CircuitBuilder,
    ClassicalBitExpr,
    GateInstruction,
    IfInstruction,
    LiteralExpr,
    MaterializedCircuit,
    MeasureVectorInstruction,
    ParameterExpr,
    ReusableCircuit,
    WireId,
    lower_circuit_plan,
    materialize_executable,
    verify_circuit,
)
from qamomile.circuit.transpiler.gate_emitter import GateKind
from qamomile.qiskit import QiskitTranspiler


@qmc.qkernel
def _lowered_bell(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
    """Build a parameterized Bell-like circuit for lowering tests."""
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    left = qmc.ry(left, theta)
    left, right = qmc.cx(left, right)
    return qmc.measure(left), qmc.measure(right)


@qmc.qkernel
def _two_parameter_rotation(alpha: qmc.Float, beta: qmc.Float) -> qmc.Bit:
    """Apply two parameters in a stable semantic order."""
    qubit = qmc.qubit("qubit")
    qubit = qmc.rx(qubit, alpha)
    qubit = qmc.rz(qubit, beta)
    return qmc.measure(qubit)


@qmc.qkernel
def _vector_measurement() -> qmc.Vector[qmc.Bit]:
    """Measure a complete register as one semantic vector operation."""
    qubits = qmc.qubit_array(3, "qubits")
    qubits[1] = qmc.x(qubits[1])
    return qmc.measure(qubits)


@qmc.composite_gate(name="semantic_helper")
def _semantic_helper(qubit: qmc.Qubit) -> qmc.Qubit:
    """Provide a user-defined callable identity for lowering tests."""
    return qmc.h(qubit)


@qmc.qkernel
def _semantic_helper_caller() -> qmc.Bit:
    """Invoke a user-defined composite and measure its result."""
    qubit = qmc.qubit("qubit")
    qubit = _semantic_helper(qubit)
    return qmc.measure(qubit)


def test_builder_versions_wires_and_preserves_symbolic_parameters() -> None:
    """Primitive gates consume old wires and produce target-neutral versions."""
    builder = CircuitBuilder(2, 1)
    theta = ParameterExpr("theta")
    builder.append_gate(GateKind.H, (0,))
    builder.append_gate(GateKind.CX, (0, 1))
    builder.append_gate(GateKind.RZ, (1,), (theta * 2.0,))
    builder.append_measure(1, 0)
    program = builder.freeze()

    verify_circuit(program)
    gates = [op for op in program.operations if isinstance(op, GateInstruction)]
    assert [gate.kind for gate in gates] == [GateKind.H, GateKind.CX, GateKind.RZ]
    assert gates[0].inputs == (WireId(0),)
    assert gates[1].inputs[0] == gates[0].outputs[0]
    assert program.output_wires[1].value > gates[-1].outputs[0].value


def test_builder_preserves_vector_measurement_as_one_instruction() -> None:
    """Vector measurement remains grouped at the circuit-codegen boundary."""
    builder = CircuitBuilder(3, 3)
    builder.append_measure_vector((2, 0, 1), (0, 1, 2))
    program = builder.freeze()

    verify_circuit(program)
    [measurement] = program.operations
    assert isinstance(measurement, MeasureVectorInstruction)
    assert measurement.clbits == (0, 1, 2)
    assert len(measurement.inputs) == 3


def test_builder_encodes_structured_if_with_explicit_wire_merges() -> None:
    """Conditional regions share inputs and yield fresh merged outputs."""
    builder = CircuitBuilder(1, 1)
    context = builder.begin_if(ClassicalBitExpr(0))
    builder.append_gate(GateKind.X, (0,))
    builder.begin_else(context)
    builder.append_gate(GateKind.Z, (0,))
    builder.end_if(context)
    builder.append_measure(0, 0)
    program = builder.freeze()

    verify_circuit(program)
    branch = program.operations[0]
    assert isinstance(branch, IfInstruction)
    assert branch.true_outputs != branch.false_outputs
    assert branch.outputs[0] not in branch.true_outputs
    assert branch.outputs[0] not in branch.false_outputs


def test_builder_encodes_structured_loops() -> None:
    """For and while regions expose loop-carried virtual wires."""
    builder = CircuitBuilder(1, 1)
    variable = builder.begin_for(range(3))
    builder.append_gate(GateKind.RX, (0,), (variable,))
    builder.end_for()
    context = builder.begin_while(ClassicalBitExpr(0))
    builder.append_gate(GateKind.H, (0,))
    builder.end_while(context)
    program = builder.freeze()

    verify_circuit(program)
    assert len(program.operations) == 2


def test_verifier_rejects_reused_output_wire_definition() -> None:
    """A malformed program cannot define one virtual wire twice."""
    builder = CircuitBuilder(1, 0)
    builder.append_gate(GateKind.H, (0,))
    program = builder.freeze()
    [gate] = program.operations
    assert isinstance(gate, GateInstruction)
    malformed_gate = dataclasses.replace(gate, outputs=program.input_wires)
    malformed = dataclasses.replace(program, operations=(malformed_gate,))

    with pytest.raises(ValueError, match="not unique"):
        verify_circuit(malformed)


def test_verifier_rejects_gate_arity_mismatch() -> None:
    """Primitive gate arity is verified before target materialization."""
    builder = CircuitBuilder(2, 0)
    builder.append_gate(GateKind.H, (0, 1))

    with pytest.raises(ValueError, match="H gate expects 1 qubits"):
        verify_circuit(builder.freeze())


def test_verifier_rejects_measurement_encoded_as_gate() -> None:
    """Measurement must use its dedicated linear instruction form."""
    builder = CircuitBuilder(1, 0)
    builder.append_gate(GateKind.MEASURE, (0,))

    with pytest.raises(ValueError, match="MeasureInstruction"):
        verify_circuit(builder.freeze())


def test_verifier_rejects_out_of_range_classical_reference() -> None:
    """Structured predicates cannot reference an unallocated classical bit."""
    builder = CircuitBuilder(1, 1)
    context = builder.begin_if(ClassicalBitExpr(2))
    builder.append_gate(GateKind.X, (0,))
    builder.end_if(context)
    program = builder.freeze()

    with pytest.raises(ValueError, match="out of range"):
        verify_circuit(program)


def test_verifier_rejects_permuted_structured_region_yields() -> None:
    """Loop carry order is part of the circuit IR ABI, not merely a wire set."""
    builder = CircuitBuilder(2, 0)
    builder.begin_for(range(2))
    builder.append_gate(GateKind.H, (0,))
    builder.append_gate(GateKind.X, (1,))
    builder.end_for()
    program = builder.freeze()
    [loop] = program.operations
    malformed_loop = dataclasses.replace(
        loop,
        body_outputs=tuple(reversed(loop.body_outputs)),  # type: ignore[attr-defined]
    )

    with pytest.raises(ValueError, match="body outputs"):
        verify_circuit(dataclasses.replace(program, operations=(malformed_loop,)))


def test_verifier_recurses_into_reusable_circuit_bodies() -> None:
    """Malformed nested reusable bodies cannot bypass top-level verification."""
    body_builder = CircuitBuilder(1, 0, name="body")
    body_builder.append_gate(GateKind.H, (0,))
    body = body_builder.freeze()
    [gate] = body.operations
    assert isinstance(gate, GateInstruction)
    malformed_body = dataclasses.replace(
        body,
        operations=(dataclasses.replace(gate, outputs=body.input_wires),),
    )
    caller = CircuitBuilder(1, 0)
    caller.append_call(ReusableCircuit(malformed_body, "malformed"), (0,))

    with pytest.raises(ValueError, match="not unique"):
        verify_circuit(caller.freeze())


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"power": 0}, "power must be positive"),
        ({"controls": -1}, "control count must be non-negative"),
    ],
)
def test_verifier_rejects_invalid_reusable_call_transforms(
    changes: dict[str, int],
    message: str,
) -> None:
    """Reusable-call transforms must satisfy the circuit boundary invariants.

    Args:
        changes (dict[str, int]): Invalid callee fields to inject.
        message (str): Expected verifier diagnosis.
    """
    body = CircuitBuilder(1, 0, name="body").freeze()
    caller = CircuitBuilder(1, 0)
    caller.append_call(ReusableCircuit(body, "body"), (0,))
    program = caller.freeze()
    [call] = program.operations
    assert isinstance(call, CallInstruction)
    malformed_call = dataclasses.replace(
        call,
        callee=dataclasses.replace(call.callee, **changes),
    )

    with pytest.raises(ValueError, match=message):
        verify_circuit(dataclasses.replace(program, operations=(malformed_call,)))


@pytest.mark.parametrize(
    ("call_arguments", "message"),
    [
        ((("", LiteralExpr(0.5)),), "names must be non-empty"),
        (
            (("theta", LiteralExpr(0.5)), ("theta", LiteralExpr(1.0))),
            "names must be unique",
        ),
        ((("theta", object()),), "Unsupported scalar expression"),
    ],
)
def test_verifier_rejects_malformed_reusable_call_arguments(
    call_arguments: tuple[tuple[str, object], ...],
    message: str,
) -> None:
    """Call-site metadata must contain valid, uniquely named scalar values.

    Args:
        call_arguments (tuple[tuple[str, object], ...]): Malformed metadata.
        message (str): Expected verifier diagnosis.
    """
    body = CircuitBuilder(1, 0, name="body").freeze()
    caller = CircuitBuilder(1, 0)
    caller.append_call(
        ReusableCircuit(
            body,
            "body",
            call_arguments=call_arguments,  # type: ignore[arg-type]
        ),
        (0,),
    )

    with pytest.raises(ValueError, match=message):
        verify_circuit(caller.freeze())


def test_verifier_checks_call_argument_classical_bit_bounds() -> None:
    """Call arguments cannot reference an unallocated classical bit."""
    body = CircuitBuilder(1, 0, name="body").freeze()
    caller = CircuitBuilder(1, 1)
    caller.append_call(
        ReusableCircuit(
            body,
            "body",
            call_arguments=(("flag", ClassicalBitExpr(1)),),
        ),
        (0,),
    )

    with pytest.raises(ValueError, match="out of range"):
        verify_circuit(caller.freeze())


def test_materialization_rejects_positional_parameter_order_drift() -> None:
    """A positional backend cannot silently reorder runtime parameters."""
    transpiler = QiskitTranspiler()
    prepared = transpiler.prepare(
        _two_parameter_rotation,
        parameters=["alpha", "beta"],
    )
    lowered = lower_circuit_plan(
        transpiler.plan_circuit(prepared),
        parameters=["alpha", "beta"],
    )

    class _ReorderedMaterializer:
        """Return a deliberately reversed positional parameter ABI."""

        def materialize(self, program: object) -> MaterializedCircuit[object]:
            """Materialize a fake artifact with the wrong positional order."""
            del program
            return MaterializedCircuit(
                artifact=object(),
                parameters={"beta": object(), "alpha": object()},
                parameter_order=("beta", "alpha"),
            )

    with pytest.raises(ValueError, match="positional parameter order"):
        materialize_executable(lowered, _ReorderedMaterializer())


def test_qamomile_plan_lowers_to_backend_neutral_circuit_ir() -> None:
    """The full semantic circuit path produces verified circuit IR."""
    transpiler = QiskitTranspiler()
    prepared = transpiler.prepare(_lowered_bell, parameters=["theta"])
    plan = transpiler.plan_circuit(prepared)
    lowered = lower_circuit_plan(plan, parameters=["theta"])
    program = lowered.quantum_circuit

    verify_circuit(program)
    gates = [op for op in program.operations if isinstance(op, GateInstruction)]
    assert [gate.kind for gate in gates] == [GateKind.RY, GateKind.CX]
    assert isinstance(gates[0].parameters[0], ParameterExpr)
    assert program.num_qubits == 2
    assert program.num_clbits == 2


def test_lowering_keeps_semantic_vector_measurement_grouped() -> None:
    """Semantic vector measurement reaches CircuitProgram without expansion."""
    transpiler = QiskitTranspiler()
    prepared = transpiler.prepare(_vector_measurement)
    lowered = lower_circuit_plan(transpiler.plan_circuit(prepared))
    program = lowered.quantum_circuit

    measurements = [
        operation
        for operation in program.operations
        if isinstance(operation, MeasureVectorInstruction)
    ]
    assert len(measurements) == 1
    assert len(measurements[0].inputs) == 3
    verify_circuit(program)


def test_lowering_keeps_open_user_composite_identity() -> None:
    """User composites retain an open semantic key and fallback body."""
    transpiler = QiskitTranspiler()
    prepared = transpiler.prepare(_semantic_helper_caller)
    lowered = lower_circuit_plan(transpiler.plan_circuit(prepared))
    program = lowered.quantum_circuit

    calls = [
        operation
        for operation in program.operations
        if isinstance(operation, CallInstruction)
    ]
    assert len(calls) == 1
    identity = calls[0].callee.identity
    assert identity is not None
    assert identity.key.namespace.startswith("user.composite.")
    assert identity.key.name == "semantic_helper"
    assert calls[0].callee.operand_widths == (1,)
    assert calls[0].callee.body.operations
