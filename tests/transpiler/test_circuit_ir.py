"""Tests for backend-neutral circuit code-generation IR."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.ir.operation.callable import (
    CallableImplementation,
    CallTransform,
    InvokeOperation,
)
from qamomile.circuit.transpiler.circuit_ir import (
    SELECT_SEMANTIC_KEY,
    CallInstruction,
    CircuitBuilder,
    CircuitLoweringPass,
    ClassicalBitExpr,
    ForInstruction,
    GateInstruction,
    IfInstruction,
    LiteralExpr,
    LoopVariableExpr,
    MaterializedCircuit,
    MeasureVectorInstruction,
    ParameterExpr,
    PauliEvolutionInstruction,
    ReusableCircuit,
    WhileInstruction,
    WireId,
    lower_circuit_plan,
    materialize_executable,
    verify_circuit,
)
from qamomile.circuit.transpiler.gate_emitter import GateKind
from qamomile.qiskit import QiskitTranspiler


class _PartialDecliningEmitter:
    """Append a partial gate before declining controlled-call emission."""

    def emit(
        self,
        circuit: CircuitBuilder,
        op: InvokeOperation,
        qubit_indices: list[int],
        bindings: dict[str, Any],
    ) -> bool:
        """Append a partial Hadamard gate and decline the operation.

        Args:
            circuit (CircuitBuilder): Builder receiving the partial attempt.
            op (InvokeOperation): Normalized controlled invocation.
            qubit_indices (list[int]): Physical control and target slots.
            bindings (dict[str, Any]): Active emit bindings.

        Returns:
            bool: Always ``False`` to select the generic fallback.
        """
        del op, bindings
        circuit.append_gate(GateKind.H, (qubit_indices[-1],))
        return False


@qmc.composite_gate(
    name="declining_controlled_composite",
    implementations=(
        CallableImplementation(
            transform=CallTransform.CONTROLLED,
            emitter=_PartialDecliningEmitter(),
        ),
    ),
)
def _declining_controlled_composite(qubit: qmc.Qubit) -> qmc.Qubit:
    """Apply the fallback X used after the native emitter declines."""
    return qmc.x(qubit)


@qmc.qkernel
def _value_controlled_declining_composite() -> qmc.Bit:
    """Invoke the declining composite on a zero-valued control."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(
        _declining_controlled_composite,
        control_value=0,
    )(control, target)
    return qmc.measure(target)


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


@qmc.qkernel
def _value_activated_control() -> qmc.Bit:
    """Control X when two controls hold the LSB-first value two."""
    controls = qmc.qubit_array(2, "controls")
    target = qmc.qubit("target")
    controls, target = qmc.control(
        qmc.x,
        num_controls=2,
        control_value=2,
    )(controls, target)
    return qmc.measure(target)


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


@qmc.qkernel
def _select_identity(qubit: qmc.Qubit) -> qmc.Qubit:
    """Return one SELECT target unchanged."""
    return qubit


@qmc.qkernel
def _select_x(qubit: qmc.Qubit) -> qmc.Qubit:
    """Apply X to one SELECT target."""
    return qmc.x(qubit)


@qmc.qkernel
def _select_phased_identity(qubit: qmc.Qubit) -> qmc.Qubit:
    """Apply a nonzero phase to an identity SELECT case."""
    return qmc.global_phase(_select_identity, 0.25)(qubit)


@qmc.qkernel
def _select_phased_x(qubit: qmc.Qubit) -> qmc.Qubit:
    """Apply a nonzero phase to an X SELECT case."""
    return qmc.global_phase(qmc.x, 0.25)(qubit)


@qmc.qkernel
def _four_case_select() -> qmc.Bit:
    """Build a four-case SELECT for circuit-IR structure tests."""
    index = qmc.qubit_array(2, "index")
    target = qmc.qubit("target")
    index, target = qmc.select(
        [_select_identity, _select_x, _select_identity, _select_x]
    )(index, target)
    return qmc.measure(target)


@qmc.qkernel
def _phase_only_select() -> qmc.Bit:
    """Build a SELECT whose selected case contains only global phase."""
    index = qmc.qubit("index")
    target = qmc.qubit("target")
    index, target = qmc.select([_select_identity, _select_phased_identity])(
        index,
        target,
    )
    return qmc.measure(target)


@qmc.qkernel
def _broadcast_phase_select() -> qmc.Vector[qmc.Bit]:
    """Broadcast a phased scalar SELECT case over three target qubits."""
    index = qmc.qubit("index")
    targets = qmc.qubit_array(3, "targets")
    index, targets = qmc.select([_select_identity, _select_phased_x])(
        index,
        targets,
    )
    return qmc.measure(targets)


@qmc.qkernel
def _controlled_broadcast_phase() -> qmc.Vector[qmc.Bit]:
    """Broadcast a phased scalar controlled body over three targets."""
    control = qmc.qubit("control")
    targets = qmc.qubit_array(3, "targets")
    control, targets = qmc.control(_select_phased_x)(control, targets)
    return qmc.measure(targets)


@qmc.qkernel
def _controlled_pauli_body(
    qubits: qmc.Vector[qmc.Qubit],
    hamiltonian: qmc.Observable,
    time: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply a Pauli evolution inside a controlled reusable body."""
    return qmc.pauli_evolve(qubits, hamiltonian, time)


@qmc.qkernel
def _controlled_pauli_entrypoint(
    hamiltonian: qmc.Observable,
    time: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    """Expose a controlled Pauli identity term to circuit lowering."""
    qubits = qmc.qubit_array(2, "qubits")
    qubits[0], target = qmc.control(_controlled_pauli_body)(
        qubits[0],
        qubits[1:2],
        hamiltonian=hamiltonian,
        time=time,
    )
    qubits[1:2] = target
    return qmc.measure(qubits)


@qmc.qkernel
def _branch_local_while_overwrite() -> qmc.Bit:
    """Read one snapshot only on the branch that does not overwrite it."""
    selector = qmc.measure(qmc.qubit("selector"))
    source_qubit = qmc.x(qmc.qubit("source"))
    source = qmc.measure(source_qubit)
    target = qmc.qubit("target")
    if selector:
        condition = source
        while condition:
            condition = qmc.measure(qmc.qubit("stop"))
    else:
        if source:
            target = qmc.x(target)
    return qmc.measure(target)


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


def test_builder_accumulates_phase_across_nested_static_for_loops() -> None:
    """Static loop phases scale by each concrete trip count at the root."""
    builder = CircuitBuilder(1, 0)
    theta = ParameterExpr("theta")
    builder.add_global_phase(0.125)
    builder.begin_for(range(2))
    builder.add_global_phase(0.25)
    builder.begin_for(range(1, 7, 2))
    builder.add_global_phase(theta)
    builder.end_for()
    builder.end_for()
    program = builder.freeze()

    assert program.global_phase == LiteralExpr(0.125) + 2 * (
        LiteralExpr(0.25) + 3 * theta
    )
    [outer_loop] = program.operations
    assert isinstance(outer_loop, ForInstruction)
    assert isinstance(outer_loop.body[0], ForInstruction)
    verify_circuit(program)


@pytest.mark.parametrize(
    ("indexset", "expected"),
    [
        (range(0), LiteralExpr(0.0)),
        (range(5, 1, -2), 2 * ParameterExpr("theta")),
    ],
)
def test_builder_scales_phase_for_empty_and_descending_static_loops(
    indexset: range,
    expected: object,
) -> None:
    """Static phase accumulation follows the concrete Python range length."""
    builder = CircuitBuilder(1, 0)
    builder.begin_for(indexset)
    builder.add_global_phase(ParameterExpr("theta"))
    builder.end_for()
    program = builder.freeze()

    assert program.global_phase == expected
    verify_circuit(program)


def test_builder_aggregates_pauli_identity_term_into_program_phase() -> None:
    """Hamiltonian identity terms use the canonical program-phase channel."""
    builder = CircuitBuilder(1, 0)
    theta = ParameterExpr("theta")
    builder.append_pauli_evolution((0,), qm_o.Z(0) + 1.5, theta)
    program = builder.freeze()

    [evolution] = program.operations
    assert isinstance(evolution, PauliEvolutionInstruction)
    assert evolution.hamiltonian.constant == 0
    assert program.global_phase == -1.5 * theta
    verify_circuit(program)


def test_builder_preserves_tiny_pauli_identity_phase_exactly() -> None:
    """Canonicalization never treats a nonzero identity phase as tolerance."""
    builder = CircuitBuilder(1, 0)
    theta = ParameterExpr("theta")
    builder.append_pauli_evolution((0,), qm_o.Z(0) + 1e-16, theta)
    program = builder.freeze()

    [evolution] = program.operations
    assert isinstance(evolution, PauliEvolutionInstruction)
    assert evolution.hamiltonian.constant == 0
    assert program.global_phase == -1e-16 * theta
    verify_circuit(program)


def test_builder_reduces_constant_only_pauli_evolution_to_phase() -> None:
    """A constant-only Hamiltonian leaves no empty evolution instruction."""
    builder = CircuitBuilder(1, 0)
    theta = ParameterExpr("theta")
    builder.append_pauli_evolution((0,), qm_o.Hamiltonian.identity(2.0), theta)
    program = builder.freeze()

    assert program.operations == ()
    assert program.output_wires == program.input_wires
    assert program.global_phase == -2.0 * theta
    verify_circuit(program)


def test_builder_hoists_uncontrolled_call_phase_with_inverse_and_power() -> None:
    """An unconditional call contributes one canonical enclosing phase."""
    theta = ParameterExpr("theta")
    body = CircuitBuilder(1, 0, name="phased-body")
    body.add_global_phase(theta)
    caller = CircuitBuilder(1, 0)
    caller.append_call(
        ReusableCircuit(
            body.freeze(),
            "phased-body",
            inverse=True,
            power=3,
        ),
        (0,),
    )
    program = caller.freeze()

    [call] = program.operations
    assert isinstance(call, CallInstruction)
    assert program.global_phase == -3 * theta
    assert call.callee.body.global_phase == LiteralExpr(0.0)
    verify_circuit(program)


def test_builder_preserves_nested_if_and_while_phases() -> None:
    """Dynamic branch and loop phases remain attached to their regions."""
    builder = CircuitBuilder(1, 1)
    theta = ParameterExpr("theta")
    builder.add_global_phase(0.125)
    conditional = builder.begin_if(ClassicalBitExpr(0))
    builder.add_global_phase(theta)
    loop = builder.begin_while(ClassicalBitExpr(0))
    builder.add_global_phase(0.25)
    builder.end_while(loop)
    builder.begin_else(conditional)
    builder.add_global_phase(0.5)
    builder.end_if(conditional)
    program = builder.freeze()

    assert program.global_phase == LiteralExpr(0.125)
    [branch] = program.operations
    assert isinstance(branch, IfInstruction)
    assert branch.true_global_phase == theta
    assert branch.false_global_phase == LiteralExpr(0.5)
    [loop] = branch.true_body
    assert isinstance(loop, WhileInstruction)
    assert loop.body_global_phase == LiteralExpr(0.25)
    verify_circuit(program)


def test_builder_preserves_measurement_selected_root_phase_contribution() -> None:
    """A measurement-dependent phase is retained for target validation."""
    builder = CircuitBuilder(1, 1)
    theta = ParameterExpr("theta")
    builder.add_global_phase(theta)
    builder.add_global_phase(theta * ClassicalBitExpr(0))
    program = builder.freeze()

    assert program.global_phase == theta + theta * ClassicalBitExpr(0)
    verify_circuit(program)


def test_verifier_checks_structured_region_phase_expressions() -> None:
    """Nested phases cannot reference classical bits outside the program."""
    builder = CircuitBuilder(1, 1)
    context = builder.begin_if(ClassicalBitExpr(0))
    builder.add_global_phase(0.25)
    builder.end_if(context)
    program = builder.freeze()
    [branch] = program.operations
    assert isinstance(branch, IfInstruction)
    malformed = dataclasses.replace(
        program,
        operations=(
            dataclasses.replace(branch, true_global_phase=ClassicalBitExpr(1)),
        ),
    )

    with pytest.raises(ValueError, match="out of range"):
        verify_circuit(malformed)


def test_builder_rejects_loop_dependent_static_for_phase() -> None:
    """An induction-dependent phase must be unrolled before circuit IR."""
    builder = CircuitBuilder(1, 0)
    variable = builder.begin_for(range(3))
    builder.add_global_phase(variable)

    with pytest.raises(RuntimeError, match="cannot depend on its induction variable"):
        builder.end_for()


def test_builder_rejects_dangling_loop_variable_in_root_phase() -> None:
    """A loop variable cannot escape its owning structured region."""
    builder = CircuitBuilder(1, 0)
    variable = builder.begin_for(range(1))
    builder.end_for()
    builder.add_global_phase(variable)

    with pytest.raises(RuntimeError, match="outside its owning structured region"):
        builder.freeze()


def test_verifier_rejects_unbound_loop_variable_in_root_phase() -> None:
    """Structural verification rejects a root phase with no owning loop."""
    program = dataclasses.replace(
        CircuitBuilder(0, 0).freeze(),
        global_phase=LoopVariableExpr("missing"),
    )

    with pytest.raises(ValueError, match="outside its owning structured for-loop"):
        verify_circuit(program)


@pytest.mark.parametrize("branch", ["true", "false"])
def test_verifier_rejects_unbound_loop_variable_in_if_phase(branch: str) -> None:
    """Both conditional phase fields reject a loop variable with no owner."""
    builder = CircuitBuilder(1, 1)
    context = builder.begin_if(ClassicalBitExpr(0))
    if branch == "false":
        builder.begin_else(context)
    builder.add_global_phase(LoopVariableExpr("missing"))
    builder.end_if(context)

    with pytest.raises(ValueError, match="outside its owning structured for-loop"):
        verify_circuit(builder.freeze())


def test_verifier_rejects_unbound_loop_variable_in_while_phase() -> None:
    """A while-body phase cannot reference a loop variable outside a for."""
    builder = CircuitBuilder(1, 1)
    context = builder.begin_while(ClassicalBitExpr(0))
    builder.add_global_phase(LoopVariableExpr("missing"))
    builder.end_while(context)

    with pytest.raises(ValueError, match="outside its owning structured for-loop"):
        verify_circuit(builder.freeze())


def test_verifier_rejects_phase_using_completed_inner_loop_variable() -> None:
    """A phase outside an inner for cannot reuse its completed induction value."""
    builder = CircuitBuilder(1, 1)
    builder.begin_for(range(1))
    inner_variable = builder.begin_for(range(1))
    builder.end_for()
    context = builder.begin_if(ClassicalBitExpr(0))
    builder.add_global_phase(inner_variable)
    builder.end_if(context)
    builder.end_for()

    with pytest.raises(ValueError, match="outside its owning structured for-loop"):
        verify_circuit(builder.freeze())


def test_verifier_accepts_nested_for_variables_in_scoped_phases() -> None:
    """Nested conditional and while phases may use both active for variables."""
    builder = CircuitBuilder(1, 1)
    outer_variable = builder.begin_for(range(2))
    inner_variable = builder.begin_for(range(3))
    branch = builder.begin_if(ClassicalBitExpr(0))
    builder.add_global_phase(outer_variable + inner_variable)
    builder.end_if(branch)
    loop = builder.begin_while(ClassicalBitExpr(0))
    builder.add_global_phase(outer_variable - inner_variable)
    builder.end_while(loop)
    builder.end_for()
    builder.end_for()

    verify_circuit(builder.freeze())


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


def test_verifier_rejects_reusable_body_capturing_enclosing_loop_variable() -> None:
    """Reusable bodies cannot capture an enclosing loop's induction value."""
    caller = CircuitBuilder(1, 0)
    induction = caller.begin_for(range(2))
    body = CircuitBuilder(1, 0, name="capturing-body")
    body.append_gate(GateKind.RZ, (0,), (induction,))
    caller.append_call(ReusableCircuit(body.freeze(), "capturing-body"), (0,))
    caller.end_for()

    with pytest.raises(ValueError, match="outside its owning structured for-loop"):
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

        def materialize(
            self,
            program: object,
            parameter_names: tuple[str, ...] = (),
        ) -> MaterializedCircuit[object]:
            """Materialize a fake artifact with the wrong positional order."""
            del program, parameter_names
            return MaterializedCircuit(
                artifact=object(),
                parameters={"beta": object(), "alpha": object()},
                parameter_order=("beta", "alpha"),
            )

    with pytest.raises(ValueError, match="positional parameter order"):
        materialize_executable(lowered, _ReorderedMaterializer())


def test_materialization_accepts_legacy_one_argument_materializer() -> None:
    """A public materializer with the former signature remains callable."""
    transpiler = QiskitTranspiler()
    parameter_names = ["alpha", "beta"]
    prepared = transpiler.prepare(
        _two_parameter_rotation,
        parameters=parameter_names,
    )
    lowered = lower_circuit_plan(
        transpiler.plan_circuit(prepared),
        parameters=parameter_names,
    )
    artifact = object()

    class _LegacyMaterializer:
        """Implement the public pre-parameter-order materializer protocol."""

        def materialize(self, program: object) -> MaterializedCircuit[object]:
            """Materialize without accepting the new ABI-order keyword.

            Args:
                program (object): Circuit program accepted by the legacy API.

            Returns:
                MaterializedCircuit[object]: Fake materialized artifact.
            """
            del program
            return MaterializedCircuit(
                artifact=artifact,
                parameters={name: object() for name in parameter_names},
            )

    executable = materialize_executable(lowered, _LegacyMaterializer())

    assert executable.compiled_quantum[0].circuit is artifact


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


def test_lowering_preserves_tiny_controlled_pauli_identity_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exact nonzero identity coefficient reaches CircuitProgram as phase."""
    monkeypatch.setattr(
        CircuitLoweringPass,
        "_blockvalue_to_gate",
        lambda self, *args, **kwargs: None,
    )
    transpiler = QiskitTranspiler()
    bindings = {
        "hamiltonian": qm_o.Hamiltonian.identity(1e-16),
        "time": 1.0,
    }
    prepared = transpiler.prepare(
        _controlled_pauli_entrypoint,
        bindings=bindings,
    )
    lowered = lower_circuit_plan(
        transpiler.plan_circuit(prepared),
        bindings=bindings,
    )
    program = lowered.quantum_circuit

    phase_gates = [
        operation
        for operation in program.operations
        if isinstance(operation, GateInstruction) and operation.kind is GateKind.P
    ]
    assert len(phase_gates) == 1
    assert phase_gates[0].parameters == (LiteralExpr(-1e-16),)
    verify_circuit(program)


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


def test_lowering_keeps_select_as_one_semantic_call_with_lsb_fallback() -> None:
    """SELECT stays abstract while its fallback uses LSB-first X brackets."""
    transpiler = QiskitTranspiler()
    prepared = transpiler.prepare(_four_case_select)
    lowered = lower_circuit_plan(transpiler.plan_circuit(prepared))
    program = lowered.quantum_circuit

    calls = [
        operation
        for operation in program.operations
        if isinstance(operation, CallInstruction)
    ]
    assert len(calls) == 1
    select_call = calls[0]
    identity = select_call.callee.identity
    assert identity is not None
    assert identity.key == SELECT_SEMANTIC_KEY
    case_fingerprints = identity.arguments.get("case_fingerprints")
    assert isinstance(case_fingerprints, tuple)
    assert len(case_fingerprints) == 4
    assert case_fingerprints[0] == case_fingerprints[2]
    assert case_fingerprints[1] == case_fingerprints[3]
    assert case_fingerprints[0] != case_fingerprints[1]
    assert identity.arguments.get("index_order") == "lsb0"
    assert identity.arguments.get("num_cases") == 4
    assert identity.arguments.get("num_index_qubits") == 2
    assert select_call.callee.operand_widths == (2, 1)

    fallback = select_call.callee.body
    assert [type(operation) for operation in fallback.operations] == [
        GateInstruction,
        CallInstruction,
        GateInstruction,
        CallInstruction,
    ]
    first_bracket, first_case, second_bracket, second_case = fallback.operations
    assert isinstance(first_bracket, GateInstruction)
    assert first_bracket.kind is GateKind.X
    assert first_bracket.inputs == (WireId(1),)
    assert isinstance(second_bracket, GateInstruction)
    assert second_bracket.kind is GateKind.X
    assert isinstance(first_case, CallInstruction)
    assert isinstance(second_case, CallInstruction)
    assert first_case.callee.controls == 2
    assert second_case.callee.controls == 2
    verify_circuit(program)


def test_lowering_brackets_control_value_at_the_circuit_boundary() -> None:
    """Mixed-polarity control brackets one abstract controlled call."""
    transpiler = QiskitTranspiler()
    prepared = transpiler.prepare(_value_activated_control)
    lowered = lower_circuit_plan(transpiler.plan_circuit(prepared))
    program = lowered.quantum_circuit

    opening, controlled_call, closing, _measurement = program.operations
    assert isinstance(opening, GateInstruction)
    assert opening.kind is GateKind.X
    assert opening.inputs == (WireId(0),)
    assert isinstance(controlled_call, CallInstruction)
    assert controlled_call.callee.controls == 2
    assert controlled_call.inputs[:2] == (opening.outputs[0], WireId(1))
    assert isinstance(closing, GateInstruction)
    assert closing.kind is GateKind.X
    assert closing.inputs == (controlled_call.outputs[0],)
    verify_circuit(program)


def test_declining_emitter_rolls_back_bracket_and_partial_gate() -> None:
    """A declining emitter leaves only the complete controlled fallback."""
    transpiler = QiskitTranspiler()
    prepared = transpiler.prepare(_value_controlled_declining_composite)
    lowered = lower_circuit_plan(transpiler.plan_circuit(prepared))
    program = lowered.quantum_circuit

    opening, controlled_call, closing, _measurement = program.operations
    assert isinstance(opening, GateInstruction)
    assert opening.kind is GateKind.X
    assert isinstance(controlled_call, CallInstruction)
    assert controlled_call.callee.controls == 1
    assert isinstance(closing, GateInstruction)
    assert closing.kind is GateKind.X
    assert all(
        not isinstance(operation, GateInstruction) or operation.kind is not GateKind.H
        for operation in program.operations
    )
    verify_circuit(program)


def test_lowering_keeps_phase_only_identity_case_under_index_control() -> None:
    """A case program's global phase remains inside its controlled call."""
    transpiler = QiskitTranspiler()
    prepared = transpiler.prepare(_phase_only_select)
    lowered = lower_circuit_plan(transpiler.plan_circuit(prepared))
    program = lowered.quantum_circuit

    select_call = next(
        operation
        for operation in program.operations
        if isinstance(operation, CallInstruction)
    )
    [phase_case] = select_call.callee.body.operations
    assert isinstance(phase_case, CallInstruction)
    assert phase_case.callee.controls == 1
    assert all(
        isinstance(operation, CallInstruction)
        for operation in phase_case.callee.body.operations
    )
    assert phase_case.callee.body.global_phase == LiteralExpr(0.25)
    assert select_call.callee.body.global_phase == LiteralExpr(0.0)
    verify_circuit(program)


@pytest.mark.parametrize(
    "kernel",
    [_broadcast_phase_select, _controlled_broadcast_phase],
)
def test_scalar_broadcast_repeats_complete_case_on_each_lane(kernel) -> None:
    """SELECT and control preserve each scalar case's phase per target lane."""
    transpiler = QiskitTranspiler()
    prepared = transpiler.prepare(kernel)
    lowered = lower_circuit_plan(transpiler.plan_circuit(prepared))
    program = lowered.quantum_circuit

    if kernel is _broadcast_phase_select:
        outer_call = next(
            operation
            for operation in program.operations
            if isinstance(operation, CallInstruction)
        )
        operations = outer_call.callee.body.operations
    else:
        operations = program.operations

    calls = [
        operation for operation in operations if isinstance(operation, CallInstruction)
    ]
    assert len(calls) == 3
    assert all(call.callee.body.operations for call in calls)
    assert all(call.callee.body.global_phase == LiteralExpr(0.25) for call in calls)
    verify_circuit(program)


def test_lowering_rejects_unknown_quantum_operation() -> None:
    """Circuit lowering fails closed instead of silently changing semantics."""
    from qamomile.circuit.ir.operation.operation import (
        Operation,
        OperationKind,
        Signature,
    )
    from qamomile.circuit.transpiler.errors import EmitError

    class UnknownQuantumOperation(Operation):
        """Represent a quantum operation with no registered lowering."""

        @property
        def signature(self) -> Signature:
            """Return an empty test signature."""
            return Signature(operands=[], results=[])

        @property
        def operation_kind(self) -> OperationKind:
            """Classify this test operation as quantum."""
            return OperationKind.QUANTUM

    with pytest.raises(EmitError, match="dropping it would change program semantics"):
        CircuitLoweringPass()._emit_operations(
            CircuitBuilder(0, 0),
            [UnknownQuantumOperation()],
            {},
            {},
            {},
        )


def test_lowering_isolates_while_snapshot_overwrites_between_if_branches() -> None:
    """A while overwrite in one runtime branch does not poison its sibling."""
    transpiler = QiskitTranspiler()
    prepared = transpiler.prepare(_branch_local_while_overwrite)
    lowered = lower_circuit_plan(transpiler.plan_circuit(prepared))
    program = lowered.quantum_circuit

    branches = [op for op in program.operations if isinstance(op, IfInstruction)]
    assert len(branches) == 1
    [outer] = branches
    assert any(isinstance(op, WhileInstruction) for op in outer.true_body)
    assert any(isinstance(op, IfInstruction) for op in outer.false_body)
    verify_circuit(program)
