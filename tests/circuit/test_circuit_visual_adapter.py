"""Structural tests for the target-neutral circuit-to-visual adapter."""

from __future__ import annotations

import dataclasses

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableRef,
    CallPolicy,
    CallTransform,
    CompositeGateType,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.gate import GateOperationType
from qamomile.circuit.transpiler.circuit_ir.emitter import CircuitGateEmitter
from qamomile.circuit.transpiler.circuit_ir.legalize import verify_target_legal
from qamomile.circuit.transpiler.circuit_ir.lowering import (
    lower_circuit_plan,
    lower_circuit_plan_with_trace,
)
from qamomile.circuit.transpiler.circuit_ir.model import (
    BinaryExpr,
    BinaryOperator,
    CallableIdentity,
    CallInstruction,
    CircuitBuilder,
    CircuitProgram,
    ClassicalBitExpr,
    GateInstruction,
    LiteralExpr,
    ParameterExpr,
    ReusableCircuit,
    SemanticOpKey,
)
from qamomile.circuit.transpiler.circuit_ir.trace import (
    CircuitProgramTrace,
    RangeLoopOrigin,
    SpecializedLoopKind,
    TraceRegion,
    TracingCircuitBuilder,
)
from qamomile.circuit.transpiler.circuit_ir.verify import verify_circuit
from qamomile.circuit.transpiler.circuit_planner import CircuitPlanningPipeline
from qamomile.circuit.transpiler.compiler import QamomileCompiler
from qamomile.circuit.transpiler.errors import EmitError, TargetCapabilityError
from qamomile.circuit.transpiler.gate_emitter import GATE_SPECS, GateKind
from qamomile.circuit.transpiler.passes.analyze import AnalyzePass
from qamomile.circuit.transpiler.passes.inline import InlinePass
from qamomile.circuit.transpiler.prepared import EntrypointMode
from qamomile.circuit.visualization.circuit_adapter import (
    circuit_program_to_visual_ir,
)
from qamomile.circuit.visualization.layout import CircuitLayoutEngine
from qamomile.circuit.visualization.style import DEFAULT_STYLE
from qamomile.circuit.visualization.visual_ir import (
    VFoldedBlock,
    VGate,
    VGateKind,
    VInlineBlock,
    VSkip,
    VUnfoldedKind,
    VUnfoldedSequence,
)
from qamomile.qiskit.materializer import QiskitMaterializer


@qmc.qkernel
def _indexed_specialized_loop() -> qmc.Vector[qmc.Qubit]:
    """Apply different source iterations to distinct indexed qubits.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated two-qubit register.
    """
    q = qmc.qubit_array(2, "q")
    for i in qmc.range(2):
        q[i] = qmc.x(q[i])
    return q


@qmc.qkernel
def _parameterized_inline_helper(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Apply one source-call rotation.

    Args:
        q (qmc.Qubit): Target qubit.
        theta (qmc.Float): Rotation angle.

    Returns:
        qmc.Qubit: Updated qubit.
    """
    return qmc.rz(q, theta)


@qmc.qkernel
def _use_parameterized_inline_helper(theta: qmc.Float) -> qmc.Qubit:
    """Invoke a parameterized inline-by-default qkernel.

    Args:
        theta (qmc.Float): Rotation angle.

    Returns:
        qmc.Qubit: Updated allocated qubit.
    """
    q = qmc.qubit("q")
    return _parameterized_inline_helper(q, theta)


@qmc.qkernel
def _dynamic_inline_helper(q: qmc.Qubit) -> qmc.Bit:
    """Nest measurement-driven if and while regions in an inline call.

    Args:
        q (qmc.Qubit): Qubit measured to seed runtime control flow.

    Returns:
        qmc.Bit: Final loop condition.
    """
    q = qmc.h(q)
    bit = qmc.measure(q)
    if bit:
        branch = qmc.qubit("branch")
        branch = qmc.x(branch)
        qmc.measure(branch)
    while bit:
        loop = qmc.qubit("loop")
        loop = qmc.x(loop)
        bit = qmc.measure(loop)
    return bit


@qmc.qkernel
def _use_dynamic_inline_helper() -> qmc.Bit:
    """Invoke an ordinary qkernel containing runtime structured regions.

    Returns:
        qmc.Bit: Final helper condition.
    """
    q = qmc.qubit("q")
    return _dynamic_inline_helper(q)


@qmc.qkernel
def _post_measure_inline_helper(q: qmc.Qubit) -> qmc.Bit:
    """Apply classical postprocessing after a source-call measurement.

    Args:
        q (qmc.Qubit): Qubit to measure.

    Returns:
        qmc.Bit: Negated measurement result.
    """
    return ~qmc.measure(q)


@qmc.qkernel
def _use_post_measure_inline_helper() -> qmc.Bit:
    """Invoke a helper with post-measurement classical dataflow.

    Returns:
        qmc.Bit: Negated helper measurement.
    """
    q = qmc.qubit("q")
    return _post_measure_inline_helper(q)


@qmc.qkernel
def _unused_bit_inline_helper(q: qmc.Qubit, flag: qmc.Bit) -> qmc.Qubit:
    """Accept an unused classical actual without creating quantum dataflow.

    Args:
        q (qmc.Qubit): Target qubit.
        flag (qmc.Bit): Source-call label argument intentionally unused.

    Returns:
        qmc.Qubit: Updated target qubit.
    """
    return qmc.h(q)


@qmc.qkernel
def _use_measurement_arg_inline_helper() -> qmc.Qubit:
    """Pass a measurement-derived bit through drawing-only call metadata.

    Returns:
        qmc.Qubit: Updated target qubit.
    """
    measured = qmc.qubit("measured")
    target = qmc.qubit("target")
    flag = qmc.measure(measured)
    return _unused_bit_inline_helper(target, flag)


@qmc.qkernel
def _identity_inline_helper(
    q: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Return a quantum interface without emitting an instruction.

    Args:
        q (qmc.Vector[qmc.Qubit]): Quantum interface to preserve.

    Returns:
        qmc.Vector[qmc.Qubit]: Unchanged interface.
    """
    return q


@qmc.qkernel
def _use_identity_inline_helper(
    q: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Invoke a completely identity quantum helper.

    Args:
        q (qmc.Vector[qmc.Qubit]): External two-qubit interface.

    Returns:
        qmc.Vector[qmc.Qubit]: Unchanged interface.
    """
    return _identity_inline_helper(q)


@qmc.qkernel
def _partial_inline_helper(
    q: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Touch only the first wire of a wider quantum interface.

    Args:
        q (qmc.Vector[qmc.Qubit]): Quantum interface to update.

    Returns:
        qmc.Vector[qmc.Qubit]: Interface with only the first wire modified.
    """
    q[0] = qmc.h(q[0])
    return q


@qmc.qkernel
def _use_partial_inline_helper(
    q: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Invoke a helper whose body touches a subset of its interface.

    Args:
        q (qmc.Vector[qmc.Qubit]): External two-qubit interface.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated interface.
    """
    return _partial_inline_helper(q)


@qmc.qkernel
def _nested_identity_then_rotation(
    q: qmc.Vector[qmc.Qubit],
    theta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Run an identity helper after a classical prelude, then one gate.

    Args:
        q (qmc.Vector[qmc.Qubit]): Quantum interface.
        theta (qmc.Float): Rotation angle.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated interface.
    """
    angle = theta + 0.0
    q = _identity_inline_helper(q)
    q[0] = qmc.rz(q[0], angle)
    return q


@qmc.qkernel
def _use_nested_identity_then_rotation(
    q: qmc.Vector[qmc.Qubit],
    theta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Invoke a nested identity region followed by real quantum work.

    Args:
        q (qmc.Vector[qmc.Qubit]): Quantum interface.
        theta (qmc.Float): Rotation angle.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated interface.
    """
    return _nested_identity_then_rotation(q, theta)


_INTRINSIC_CONTROL_OPAQUE = qmc.Oracle(
    "intrinsic_control_opaque",
    num_qubits=1,
    num_control_qubits=1,
)
_WRAPPED_CONTROL_OPAQUE = qmc.opaque("wrapped_control_opaque", num_qubits=1)


def _emit_inverse_opaque(q: qmc.Qubit) -> qmc.Qubit:
    """Emit the bodyless inverse InvokeOperation accepted by the compiler IR.

    Args:
        q (qmc.Qubit): Target qubit to consume.

    Returns:
        qmc.Qubit: Next-version target returned by the opaque invocation.
    """
    consumed = q.consume(operation_name="inverse opaque test marker")
    result = consumed.value.next_version()
    ref = CallableRef(namespace="user.oracle", name="wrapped_control_opaque")
    attrs = {
        "kind": "oracle",
        "gate_type": CompositeGateType.CUSTOM.name,
        "num_control_qubits": 0,
        "num_target_qubits": 1,
        "custom_name": "wrapped_control_opaque",
        "default_policy": CallPolicy.PRESERVE_BOX.name,
    }
    get_current_tracer().add_operation(
        InvokeOperation(
            operands=[consumed.value],
            results=[result],
            target=ref,
            transform=CallTransform.INVERSE,
            attrs=attrs,
            definition=CallableDef(
                ref=ref,
                default_policy=CallPolicy.PRESERVE_BOX,
                attrs=attrs,
            ),
        )
    )
    return qmc.Qubit(result)


@qmc.qkernel
def _use_direct_opaque() -> qmc.Qubit:
    """Invoke a bodyless opaque callable directly.

    Returns:
        qmc.Qubit: Updated target qubit.
    """
    q = qmc.qubit("q")
    (q,) = _WRAPPED_CONTROL_OPAQUE(q)
    return q


@qmc.qkernel
def _use_inverse_opaque() -> qmc.Qubit:
    """Invoke an inverse transform of a bodyless opaque callable.

    Returns:
        qmc.Qubit: Updated target qubit.
    """
    q = qmc.qubit("q")
    return _emit_inverse_opaque(q)


@qmc.qkernel
def _use_intrinsic_control_opaque() -> qmc.Vector[qmc.Qubit]:
    """Invoke a bodyless oracle declaring one intrinsic control.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated control and target register.
    """
    q = qmc.qubit_array(2, "q")
    q[0], q[1] = _INTRINSIC_CONTROL_OPAQUE(q[1], controls=(q[0],))
    return q


@qmc.qkernel
def _use_wrapped_control_opaque() -> qmc.Vector[qmc.Qubit]:
    """Apply ``qmc.control`` to a bodyless opaque callable.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated control and target register.
    """
    q = qmc.qubit_array(2, "q")
    controlled = qmc.control(_WRAPPED_CONTROL_OPAQUE)
    q[0], q[1] = controlled(q[0], q[1])
    return q


def _lower_qkernel_with_trace(
    kernel: object,
    bindings: dict[str, object] | None = None,
) -> tuple[CircuitProgram, CircuitProgramTrace]:
    """Run the real planning/lowering walk with drawing provenance enabled.

    Args:
        kernel (object): Frontend qkernel exposing visualization graph tracing.
        bindings (dict[str, object] | None): Concrete structural bindings.
            Defaults to ``None``.

    Returns:
        tuple[CircuitProgram, CircuitProgramTrace]: Verified circuit program
            and its aligned drawing trace.
    """
    concrete = dict(bindings or {})
    graph = kernel._build_graph_for_visualization(**concrete)  # type: ignore[attr-defined]
    prepared = QamomileCompiler().prepare_block(
        graph,
        concrete,
        mode=EntrypointMode.CIRCUIT_FRAGMENT,
    )
    plan = CircuitPlanningPipeline(inline_pass=InlinePass(preserve_regions=True)).run(
        prepared, concrete
    )
    executable, traces = lower_circuit_plan_with_trace(
        plan,
        bindings=concrete,
        parameters=[name for name in graph.parameters if name not in concrete],
        preserve_semantic_call_names=True,
    )
    assert len(executable.compiled_quantum) == 1
    assert len(traces) == 1
    return executable.compiled_quantum[0].circuit, traces[0]


@pytest.mark.parametrize(
    "kind",
    [kind for kind in GateKind if kind is not GateKind.MEASURE],
)
def test_adapter_supports_every_primitive_gate_kind(kind: GateKind) -> None:
    """Every target-neutral primitive becomes one exact visual gate."""
    specification = GATE_SPECS[kind]
    builder = CircuitBuilder(specification.num_qubits, 0)
    parameters = (LiteralExpr(0.25),) if specification.has_angle else ()
    builder.append_gate(
        kind,
        tuple(range(specification.num_qubits)),
        parameters,
    )

    circuit = circuit_program_to_visual_ir(builder.freeze())

    [gate] = circuit.children
    assert isinstance(gate, VGate)
    assert gate.qubit_indices == list(range(specification.num_qubits))
    if kind in {
        GateKind.CH,
        GateKind.CY,
        GateKind.CRX,
        GateKind.CRY,
        GateKind.CRZ,
    }:
        assert gate.kind is VGateKind.CONTROLLED_U_BOX
        assert gate.control_count == 1
    else:
        assert gate.kind is VGateKind.GATE


def test_stateless_pass_subclasses_keep_legacy_initialization_contract() -> None:
    """Public stateless classes tolerate subclasses that omit ``super``."""

    class LegacyCircuitGateEmitter(CircuitGateEmitter):
        """Model a pre-option emitter subclass."""

        def __init__(self) -> None:
            """Initialize without calling the modern base initializer."""

    class LegacyInlinePass(InlinePass):
        """Model a pre-option inliner subclass."""

        def __init__(self) -> None:
            """Initialize without calling the modern base initializer."""

    class LegacyAnalyzePass(AnalyzePass):
        """Model a pre-option analyzer subclass."""

        def __init__(self) -> None:
            """Initialize without calling the modern base initializer."""

    emitter_builder = LegacyCircuitGateEmitter().create_circuit(1, 0)
    assert type(emitter_builder) is CircuitBuilder

    inlined = LegacyInlinePass().run(_use_parameterized_inline_helper.block)
    assert inlined.kind is BlockKind.AFFINE

    empty = Block(name="empty", kind=BlockKind.AFFINE)
    analyzed = LegacyAnalyzePass().run(empty)
    assert analyzed.kind is BlockKind.ANALYZED


def test_adapter_tracks_wire_versions_through_structured_regions() -> None:
    """Nested operations retain their physical slots across fresh WireIds."""
    builder = CircuitBuilder(3, 1)
    builder.append_measure(1, 0)
    conditional = builder.begin_if(ClassicalBitExpr(0))
    builder.append_gate(GateKind.X, (2,))
    builder.begin_else(conditional)
    builder.append_gate(GateKind.H, (0,))
    builder.end_if(conditional)
    loop_variable = builder.begin_for(range(2))
    builder.append_gate(GateKind.RX, (2,), (loop_variable,))
    builder.end_for()
    while_context = builder.begin_while(ClassicalBitExpr(0))
    builder.append_gate(GateKind.Z, (1,))
    builder.end_while(while_context)

    circuit = circuit_program_to_visual_ir(
        builder.freeze(),
        fold_loops=False,
    )

    measurement, conditional_node, loop, while_node = circuit.children
    assert isinstance(measurement, VGate)
    assert isinstance(conditional_node, VUnfoldedSequence)
    assert conditional_node.affected_qubits == [0, 2]
    assert conditional_node.condition_measure_node_key == measurement.node_key
    assert conditional_node.condition_measure_qubit_indices == [1]
    assert isinstance(loop, VUnfoldedSequence)
    assert loop.kind is VUnfoldedKind.FOR
    loop_gates = [iteration[0] for iteration in loop.iterations]
    assert all(isinstance(gate, VGate) for gate in loop_gates)
    assert [gate.label for gate in loop_gates if isinstance(gate, VGate)] == [
        "$R_x$(0)",
        "$R_x$(1)",
    ]
    assert loop.affected_qubits == [2]
    assert isinstance(while_node, VUnfoldedSequence)
    assert while_node.kind is VUnfoldedKind.WHILE
    assert while_node.affected_qubits == [1]
    assert while_node.condition_measure_node_key == measurement.node_key


def test_adapter_groups_exact_specialized_loop_iterations() -> None:
    """A draw trace restores distinct bodies erased by loop specialization."""
    builder = TracingCircuitBuilder(2, 0)
    loop = builder.begin_specialized_loop(
        SpecializedLoopKind.RANGE,
        RangeLoopOrigin("i", range(2)),
    )
    builder.begin_specialized_iteration(loop, "i=0")
    builder.append_gate(GateKind.H, (0,))
    builder.end_specialized_iteration(loop)
    builder.begin_specialized_iteration(loop, "i=1")
    builder.append_gate(GateKind.X, (1,))
    builder.end_specialized_iteration(loop)
    builder.end_specialized_loop(loop)
    program = builder.freeze()
    trace = builder.freeze_trace(program)

    assert len(program.operations) == 2
    [folded] = circuit_program_to_visual_ir(
        program,
        trace=trace,
        fold_loops=True,
    ).children
    assert isinstance(folded, VFoldedBlock)
    assert folded.affected_qubits == [0, 1]
    assert folded.body_lines[0] == "2 specialized iterations (2 distinct bodies)"
    assert "$H$" in folded.body_lines[1]
    assert "$X$" in folded.body_lines[2]

    [unfolded] = circuit_program_to_visual_ir(
        program,
        trace=trace,
        fold_loops=False,
    ).children
    assert isinstance(unfolded, VUnfoldedSequence)
    assert unfolded.kind is VUnfoldedKind.FOR
    assert len(unfolded.iterations) == 2
    assert [
        gate.qubit_indices
        for iteration in unfolded.iterations
        for gate in iteration
        if isinstance(gate, VGate)
    ] == [[0], [1]]


def test_adapter_preserves_structured_regions_inside_specialized_loop() -> None:
    """Nested runtime control flow remains nested in an unrolled source loop."""
    builder = TracingCircuitBuilder(1, 1)
    loop = builder.begin_specialized_loop(
        SpecializedLoopKind.RANGE,
        RangeLoopOrigin("i", range(1)),
    )
    builder.begin_specialized_iteration(loop, "i=0")
    builder.append_measure(0, 0)
    conditional = builder.begin_if(ClassicalBitExpr(0))
    builder.append_gate(GateKind.X, (0,))
    builder.end_if(conditional)
    builder.end_specialized_iteration(loop)
    builder.end_specialized_loop(loop)
    program = builder.freeze()
    trace = builder.freeze_trace(program)

    [unfolded] = circuit_program_to_visual_ir(
        program,
        trace=trace,
        fold_loops=False,
    ).children

    assert isinstance(unfolded, VUnfoldedSequence)
    measurement, conditional_node = unfolded.iterations[0]
    assert isinstance(measurement, VGate)
    assert isinstance(conditional_node, VUnfoldedSequence)
    assert conditional_node.kind is VUnfoldedKind.IF
    assert conditional_node.condition_measure_node_key == measurement.node_key


def test_adapter_collapses_and_expands_inline_source_trace() -> None:
    """An erased qkernel call remains a labelled optional visual boundary."""
    builder = TracingCircuitBuilder(1, 0)
    builder.begin_inline_region(
        "phase_helper",
        3,
        (("theta", LiteralExpr(0.25)),),
        (0,),
    )
    builder.append_gate(GateKind.RZ, (0,), (LiteralExpr(0.25),))
    builder.end_inline_region("phase_helper", 3, (0,))
    program = builder.freeze()
    trace = builder.freeze_trace(program)

    [collapsed] = circuit_program_to_visual_ir(program, trace=trace).children
    assert isinstance(collapsed, VGate)
    assert collapsed.kind is VGateKind.BLOCK_BOX
    assert collapsed.label == "phase_helper($\\theta$=0.25)"
    assert collapsed.qubit_indices == [0]

    [expanded] = circuit_program_to_visual_ir(
        program,
        trace=trace,
        expand_calls=True,
    ).children
    assert isinstance(expanded, VInlineBlock)
    assert expanded.label == collapsed.label
    assert expanded.affected_qubits == [0]
    [rotation] = expanded.children
    assert isinstance(rotation, VGate)
    assert rotation.gate_type is GateOperationType.RZ


def test_adapter_rejects_trace_that_does_not_cover_program() -> None:
    """A malformed sidecar cannot omit authoritative flat instructions."""
    builder = CircuitBuilder(1, 0)
    builder.append_gate(GateKind.H, (0,))
    program = builder.freeze()
    malformed = CircuitProgramTrace(TraceRegion(()))

    with pytest.raises(ValueError, match="does not match instructions"):
        circuit_program_to_visual_ir(program, trace=malformed)


def test_real_lowering_walk_traces_index_specialized_loop() -> None:
    """PR586 lowering records the exact iterations it must unroll for q[i]."""
    program, trace = _lower_qkernel_with_trace(_indexed_specialized_loop)

    assert len(program.operations) == 2
    assert len(trace.root.nodes) == 1
    [source_loop] = circuit_program_to_visual_ir(
        program,
        trace=trace,
        fold_loops=False,
    ).children
    assert isinstance(source_loop, VUnfoldedSequence)
    assert len(source_loop.iterations) == 2
    assert [
        gate.qubit_indices
        for iteration in source_loop.iterations
        for gate in iteration
        if isinstance(gate, VGate)
    ] == [[0], [1]]


def test_real_inliner_trace_retains_source_call_and_scalar_argument() -> None:
    """Drawing-only inlining keeps an exact optional source qkernel box."""
    program, trace = _lower_qkernel_with_trace(
        _use_parameterized_inline_helper,
        {"theta": 0.5},
    )

    assert len(program.operations) == 1
    assert len(trace.root.nodes) == 1
    [collapsed] = circuit_program_to_visual_ir(program, trace=trace).children
    assert isinstance(collapsed, VGate)
    assert collapsed.kind is VGateKind.BLOCK_BOX
    assert collapsed.label == "_parameterized_inline_helper($\\theta$=0.50)"

    [expanded] = circuit_program_to_visual_ir(
        program,
        trace=trace,
        expand_calls=True,
    ).children
    assert isinstance(expanded, VInlineBlock)
    [rotation] = expanded.children
    assert isinstance(rotation, VGate)
    assert rotation.label == "$R_z$(0.5)"


def test_real_inliner_trace_preserves_measurement_if_and_while() -> None:
    """Source-call provenance does not turn dynamic helpers into fake gates."""
    program, trace = _lower_qkernel_with_trace(_use_dynamic_inline_helper)

    [collapsed] = circuit_program_to_visual_ir(program, trace=trace).children
    assert isinstance(collapsed, VGate)
    assert collapsed.kind is VGateKind.BLOCK_BOX
    assert collapsed.label == "_dynamic_inline_helper"
    assert collapsed.qubit_indices == [0, 1, 2]

    [expanded] = circuit_program_to_visual_ir(
        program,
        trace=trace,
        expand_calls=True,
        fold_ifs=False,
        fold_whiles=False,
    ).children
    assert isinstance(expanded, VInlineBlock)
    region_kinds = {
        node.kind for node in expanded.children if isinstance(node, VUnfoldedSequence)
    }
    assert region_kinds == {VUnfoldedKind.IF, VUnfoldedKind.WHILE}


def test_inline_marker_does_not_split_measurement_postprocessing() -> None:
    """Metadata markers do not create an extra post-measure quantum segment."""
    program, trace = _lower_qkernel_with_trace(_use_post_measure_inline_helper)

    assert len(program.operations) == 1
    [collapsed] = circuit_program_to_visual_ir(program, trace=trace).children
    assert isinstance(collapsed, VGate)
    assert collapsed.kind is VGateKind.BLOCK_BOX
    assert collapsed.qubit_indices == [0]

    [expanded] = circuit_program_to_visual_ir(
        program,
        trace=trace,
        expand_calls=True,
    ).children
    assert isinstance(expanded, VInlineBlock)
    [measurement] = expanded.children
    assert isinstance(measurement, VGate)
    assert measurement.kind is VGateKind.MEASURE
    assert measurement.terminates_wire is True


def test_inline_call_metadata_does_not_create_quantum_dependency() -> None:
    """A measurement-derived label argument remains non-semantic metadata."""
    program, trace = _lower_qkernel_with_trace(_use_measurement_arg_inline_helper)

    measurement, helper = circuit_program_to_visual_ir(
        program,
        trace=trace,
    ).children
    assert isinstance(measurement, VGate)
    assert measurement.kind is VGateKind.MEASURE
    assert isinstance(helper, VGate)
    assert helper.kind is VGateKind.BLOCK_BOX
    assert helper.label.startswith("_unused_bit_inline_helper(flag=")


def test_identity_inline_call_keeps_complete_quantum_interface() -> None:
    """An instruction-free source helper remains visible on every wire."""
    program, trace = _lower_qkernel_with_trace(
        _use_identity_inline_helper,
        {"q": 2},
    )

    assert program.operations == ()
    [collapsed] = circuit_program_to_visual_ir(program, trace=trace).children
    assert isinstance(collapsed, VGate)
    assert collapsed.kind is VGateKind.BLOCK_BOX
    assert collapsed.qubit_indices == [0, 1]

    [expanded] = circuit_program_to_visual_ir(
        program,
        trace=trace,
        expand_calls=True,
    ).children
    assert isinstance(expanded, VInlineBlock)
    assert expanded.children == []
    assert expanded.affected_qubits == [0, 1]


def test_partial_inline_call_box_keeps_untouched_interface_wire() -> None:
    """A source call box covers identity wires omitted by its emitted body."""
    program, trace = _lower_qkernel_with_trace(
        _use_partial_inline_helper,
        {"q": 2},
    )

    [collapsed] = circuit_program_to_visual_ir(program, trace=trace).children
    assert isinstance(collapsed, VGate)
    assert collapsed.qubit_indices == [0, 1]

    [expanded] = circuit_program_to_visual_ir(
        program,
        trace=trace,
        expand_calls=True,
    ).children
    assert isinstance(expanded, VInlineBlock)
    assert expanded.affected_qubits == [0, 1]
    [gate] = expanded.children
    assert isinstance(gate, VGate)
    assert gate.qubit_indices == [0]


def test_nested_identity_marker_waits_for_real_quantum_segment() -> None:
    """Nested marker-only calls do not manufacture a quantum segment."""
    program, trace = _lower_qkernel_with_trace(
        _use_nested_identity_then_rotation,
        {"q": 2, "theta": 0.25},
    )

    assert len(program.operations) == 1
    [outer] = circuit_program_to_visual_ir(
        program,
        trace=trace,
        expand_calls=True,
        inline_depth=1,
    ).children
    assert isinstance(outer, VInlineBlock)
    nested_identity, rotation = outer.children
    assert isinstance(nested_identity, VGate)
    assert nested_identity.kind is VGateKind.BLOCK_BOX
    assert nested_identity.qubit_indices == [0, 1]
    assert isinstance(rotation, VGate)
    assert rotation.qubit_indices == [0]


@pytest.mark.parametrize(
    ("kernel", "label"),
    [
        (_use_intrinsic_control_opaque, "intrinsic_control_opaque"),
        (_use_wrapped_control_opaque, "wrapped_control_opaque"),
    ],
)
def test_controlled_opaque_call_keeps_control_outside_placeholder_body(
    kernel: object,
    label: str,
) -> None:
    """Both opaque control APIs retain an exact controlled box.

    Args:
        kernel (object): Bodyless controlled qkernel fixture.
        label (str): Expected semantic box label.
    """
    program, trace = _lower_qkernel_with_trace(kernel)

    [instruction] = program.operations
    assert isinstance(instruction, CallInstruction)
    assert instruction.callee.opaque is True
    assert instruction.callee.controls == 1
    assert instruction.callee.body.num_qubits == 1
    assert instruction.callee.body.operations == ()
    assert instruction.callee.operand_widths == (1,)

    [call] = circuit_program_to_visual_ir(
        program,
        trace=trace,
        expand_calls=True,
    ).children
    assert isinstance(call, VGate)
    assert call.kind is VGateKind.CONTROLLED_U_BOX
    assert call.label == label
    assert call.qubit_indices == [0, 1]
    assert call.control_count == 1


def test_opaque_placeholder_requires_identity_and_empty_body() -> None:
    """Structural verification distinguishes opaque calls from identities."""
    body = CircuitBuilder(1, 0, name="opaque_body").freeze()
    identity = CallableIdentity(
        key=SemanticOpKey("user.oracle", "opaque_body"),
        symbol="opaque_body",
    )
    builder = CircuitBuilder(1, 0)
    builder.append_call(
        ReusableCircuit(
            body=body,
            name="opaque_body",
            identity=identity,
            operand_widths=(1,),
            opaque=True,
        ),
        (0,),
    )
    valid = builder.freeze()
    verify_circuit(valid)
    [call] = valid.operations
    assert isinstance(call, CallInstruction)

    missing_identity = dataclasses.replace(
        valid,
        operations=(
            dataclasses.replace(
                call,
                callee=dataclasses.replace(call.callee, identity=None),
            ),
        ),
    )
    with pytest.raises(ValueError, match="requires semantic identity"):
        verify_circuit(missing_identity)

    nonempty_builder = CircuitBuilder(1, 0, name="not_a_placeholder")
    nonempty_builder.append_gate(GateKind.X, (0,))
    nonempty = dataclasses.replace(
        valid,
        operations=(
            dataclasses.replace(
                call,
                callee=dataclasses.replace(
                    call.callee,
                    body=nonempty_builder.freeze(),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="empty arity placeholder"):
        verify_circuit(nonempty)


def test_inverse_opaque_call_remains_an_unexpanded_inverse_box() -> None:
    """Drawing preserves an opaque inverse without requiring a body."""
    program, trace = _lower_qkernel_with_trace(_use_inverse_opaque)

    [instruction] = program.operations
    assert isinstance(instruction, CallInstruction)
    assert instruction.callee.opaque is True
    assert instruction.callee.inverse is True
    assert instruction.callee.body.num_qubits == 1

    [call] = circuit_program_to_visual_ir(
        program,
        trace=trace,
        expand_calls=True,
    ).children
    assert isinstance(call, VGate)
    assert call.kind is VGateKind.COMPOSITE_BOX
    assert call.label == "wrapped_control_opaque^-1"
    assert call.qubit_indices == [0]


def test_opaque_call_without_native_realization_is_not_materializable() -> None:
    """A drawing placeholder cannot pass target legality as an identity."""
    program, _ = _lower_qkernel_with_trace(_use_direct_opaque)

    with pytest.raises(TargetCapabilityError, match="no target-native realization"):
        verify_target_legal(program, QiskitMaterializer().capabilities)


@pytest.mark.parametrize(
    "kernel",
    [
        _use_direct_opaque,
        _use_intrinsic_control_opaque,
        _use_wrapped_control_opaque,
        _use_inverse_opaque,
    ],
)
def test_normal_lowering_still_rejects_bodyless_opaque_call(
    kernel: object,
) -> None:
    """The opaque relaxation remains exclusive to drawing lowering.

    Args:
        kernel (object): Direct or transformed bodyless callable fixture.
    """
    graph = kernel._build_graph_for_visualization()  # type: ignore[attr-defined]
    prepared = QamomileCompiler().prepare_block(
        graph,
        {},
        mode=EntrypointMode.CIRCUIT_FRAGMENT,
    )
    plan = CircuitPlanningPipeline().run(prepared, {})

    with pytest.raises(EmitError):
        lower_circuit_plan(plan)


def test_collapsed_inline_measurement_connects_from_call_boundary() -> None:
    """A condition after a collapsed source call connects to its box."""
    builder = TracingCircuitBuilder(1, 1)
    builder.begin_inline_region("measure_helper", 1, (), (0,))
    builder.append_measure(0, 0)
    builder.end_inline_region("measure_helper", 1, (0,))
    conditional = builder.begin_if(ClassicalBitExpr(0))
    builder.append_gate(GateKind.X, (0,))
    builder.end_if(conditional)
    program = builder.freeze()
    trace = builder.freeze_trace(program)

    call, conditional_node = circuit_program_to_visual_ir(
        program,
        trace=trace,
    ).children

    assert isinstance(call, VGate)
    assert isinstance(conditional_node, VUnfoldedSequence)
    assert conditional_node.condition_measure_node_key == call.node_key
    assert conditional_node.condition_measure_qubit_indices == [0]

    expanded_call, expanded_conditional = circuit_program_to_visual_ir(
        program,
        trace=trace,
        expand_calls=True,
    ).children
    assert isinstance(expanded_call, VInlineBlock)
    [measurement] = expanded_call.children
    assert isinstance(measurement, VGate)
    assert isinstance(expanded_conditional, VUnfoldedSequence)
    assert expanded_conditional.condition_measure_node_key == measurement.node_key
    assert expanded_conditional.condition_measure_qubit_indices == [0]


def test_folded_specialized_measurement_connects_from_loop_boundary() -> None:
    """A condition after a folded specialized loop connects to the loop."""
    builder = TracingCircuitBuilder(1, 1)
    loop = builder.begin_specialized_loop(
        SpecializedLoopKind.RANGE,
        RangeLoopOrigin("i", range(1)),
    )
    builder.begin_specialized_iteration(loop, "i=0")
    builder.append_measure(0, 0)
    builder.end_specialized_iteration(loop)
    builder.end_specialized_loop(loop)
    conditional = builder.begin_if(ClassicalBitExpr(0))
    builder.append_gate(GateKind.X, (0,))
    builder.end_if(conditional)
    program = builder.freeze()
    trace = builder.freeze_trace(program)

    folded, conditional_node = circuit_program_to_visual_ir(
        program,
        trace=trace,
        fold_loops=True,
    ).children

    assert isinstance(folded, VFoldedBlock)
    assert isinstance(conditional_node, VUnfoldedSequence)
    assert conditional_node.condition_measure_node_key == folded.node_key
    assert conditional_node.condition_measure_qubit_indices == [0]


def test_folded_if_proxies_all_branch_measurement_sources() -> None:
    """One folded branch container may expose multiple source qubits."""
    builder = CircuitBuilder(3, 2)
    builder.append_measure(0, 0)
    branches = builder.begin_if(ClassicalBitExpr(0))
    builder.append_measure(1, 1)
    builder.begin_else(branches)
    builder.append_measure(2, 1)
    builder.end_if(branches)
    following = builder.begin_if(ClassicalBitExpr(1))
    builder.append_gate(GateKind.X, (0,))
    builder.end_if(following)

    _, folded_branches, conditional = circuit_program_to_visual_ir(
        builder.freeze(),
        fold_ifs=True,
    ).children

    assert isinstance(folded_branches, VFoldedBlock)
    assert isinstance(conditional, VFoldedBlock)
    assert conditional.condition_measure_node_key == folded_branches.node_key
    assert conditional.condition_measure_qubit_indices == [1, 2]


def test_folded_loop_uses_body_footprint_not_all_live_wires() -> None:
    """A structured loop frame includes only slots used by its body."""
    builder = CircuitBuilder(4, 0)
    builder.begin_for(range(3))
    builder.append_gate(GateKind.H, (3,))
    builder.end_for()

    circuit = circuit_program_to_visual_ir(builder.freeze(), fold_loops=True)

    [loop] = circuit.children
    assert isinstance(loop, VFoldedBlock)
    assert loop.affected_qubits == [3]


def test_condition_connector_requires_one_reaching_measurement() -> None:
    """Branch-dependent measurement provenance is never guessed."""
    builder = CircuitBuilder(2, 2)
    builder.append_measure(0, 0)
    first = builder.begin_if(ClassicalBitExpr(0))
    builder.append_measure(0, 1)
    builder.begin_else(first)
    builder.append_measure(1, 1)
    builder.end_if(first)
    second = builder.begin_if(ClassicalBitExpr(1))
    builder.append_gate(GateKind.X, (0,))
    builder.end_if(second)

    circuit = circuit_program_to_visual_ir(builder.freeze())

    trailing = circuit.children[-1]
    assert isinstance(trailing, VUnfoldedSequence)
    assert trailing.condition_measure_node_key is None
    assert trailing.condition_measure_qubit_indices == []


def test_condition_connector_preserves_vector_measurement_sources() -> None:
    """One vector measurement can feed a condition from multiple wires."""
    builder = CircuitBuilder(2, 2)
    builder.append_measure_vector((0, 1), (0, 1))
    conditional = builder.begin_if(
        BinaryExpr(
            BinaryOperator.AND,
            ClassicalBitExpr(0),
            ClassicalBitExpr(1),
        )
    )
    builder.append_gate(GateKind.X, (0,))
    builder.end_if(conditional)

    measurement, conditional_node = circuit_program_to_visual_ir(
        builder.freeze()
    ).children

    assert isinstance(measurement, VGate)
    assert isinstance(conditional_node, VUnfoldedSequence)
    assert conditional_node.condition_measure_node_key == measurement.node_key
    assert conditional_node.condition_measure_qubit_indices == [0, 1]


def test_adapter_preserves_measure_reset_barrier_and_pauli_evolution() -> None:
    """Dedicated circuit instructions remain visible and ordered."""
    builder = CircuitBuilder(2, 2)
    builder.append_reset(0)
    builder.append_barrier((1, 0))
    builder.append_pauli_evolution((1,), "Z1", ParameterExpr("tau"))
    builder.append_measure_vector((1, 0), (0, 1))

    circuit = circuit_program_to_visual_ir(builder.freeze())

    reset, barrier, evolution, measurement = circuit.children
    assert isinstance(reset, VGate) and reset.label == "RESET"
    assert reset.qubit_indices == [0]
    assert isinstance(barrier, VGate) and barrier.label == "BARRIER"
    assert barrier.qubit_indices == [1, 0]
    assert isinstance(evolution, VGate)
    assert evolution.label == "EVOLVE(Z1, t=tau)"
    assert evolution.qubit_indices == [1]
    assert isinstance(measurement, VGate)
    assert measurement.kind is VGateKind.MEASURE_VECTOR
    assert measurement.qubit_indices == [1, 0]


def test_adapter_collapses_transformed_call_and_expands_direct_body() -> None:
    """Call transforms stay boxed while direct body slots translate exactly."""
    body_builder = CircuitBuilder(2, 0, name="pair")
    body_builder.append_gate(GateKind.SWAP, (0, 1))
    body = body_builder.freeze()

    transformed_builder = CircuitBuilder(3, 0)
    transformed_builder.append_call(
        ReusableCircuit(
            body=body,
            name="pair",
            controls=1,
            power=2,
            inverse=True,
        ),
        (1, 2, 0),
    )
    transformed_circuit = circuit_program_to_visual_ir(
        transformed_builder.freeze(),
        expand_calls=True,
    )
    [transformed] = transformed_circuit.children
    transformed_layout = CircuitLayoutEngine(DEFAULT_STYLE).compute_layout(
        transformed_circuit
    )
    assert isinstance(transformed, VGate)
    assert transformed.kind is VGateKind.CONTROLLED_U_BOX
    assert transformed.qubit_indices == [1, 2, 0]
    assert transformed.control_count == 1
    assert transformed.power == 2
    assert transformed.gate_type is None
    assert "^-1" in transformed.label
    assert "^2" not in transformed.label
    assert transformed.node_key in transformed_layout.powered_gate_layouts

    direct_builder = CircuitBuilder(3, 0)
    direct_builder.append_call(ReusableCircuit(body, "pair"), (2, 0))
    [collapsed] = circuit_program_to_visual_ir(direct_builder.freeze()).children
    assert isinstance(collapsed, VGate)
    assert collapsed.kind is VGateKind.COMPOSITE_BOX
    assert collapsed.qubit_indices == [2, 0]

    [direct] = circuit_program_to_visual_ir(
        direct_builder.freeze(),
        expand_calls=True,
    ).children
    assert isinstance(direct, VInlineBlock)
    [swap] = direct.children
    assert isinstance(swap, VGate)
    assert swap.qubit_indices == [2, 0]
    assert direct.affected_qubits == [0, 2]

    controlled_builder = CircuitBuilder(3, 0)
    controlled_builder.append_call(
        ReusableCircuit(body, "controlled_pair", controls=1),
        (1, 2, 0),
    )
    [controlled] = circuit_program_to_visual_ir(
        controlled_builder.freeze(),
        expand_calls=True,
    ).children
    assert isinstance(controlled, VInlineBlock)
    assert controlled.control_qubit_indices == [1]
    assert controlled.affected_qubits == [0, 2]
    [controlled_swap] = controlled.children
    assert isinstance(controlled_swap, VGate)
    assert controlled_swap.qubit_indices == [2, 0]


def test_inline_depth_counts_only_reusable_call_nesting() -> None:
    """Control-flow nesting does not consume reusable-call expansion depth."""
    body_builder = CircuitBuilder(1, 0, name="leaf")
    body_builder.append_gate(GateKind.H, (0,))
    leaf = ReusableCircuit(body_builder.freeze(), "leaf")

    builder = CircuitBuilder(1, 1)
    builder.begin_for(range(1))
    conditional = builder.begin_if(ClassicalBitExpr(0))
    loop = builder.begin_while(ClassicalBitExpr(0))
    builder.append_call(leaf, (0,))
    builder.end_while(loop)
    builder.end_if(conditional)
    builder.end_for()

    [for_node] = circuit_program_to_visual_ir(
        builder.freeze(),
        expand_calls=True,
        inline_depth=1,
        fold_loops=False,
    ).children

    assert isinstance(for_node, VUnfoldedSequence)
    [if_node] = for_node.iterations[0]
    assert isinstance(if_node, VUnfoldedSequence)
    [while_node] = if_node.iterations[0]
    assert isinstance(while_node, VUnfoldedSequence)
    [call] = while_node.iterations[0]
    assert isinstance(call, VInlineBlock)


def test_unfolded_native_loop_specializes_reusable_call_arguments() -> None:
    """Native loop call labels use each concrete visual induction value."""
    body_builder = CircuitBuilder(1, 0, name="phase")
    body_builder.append_gate(GateKind.X, (0,))
    builder = CircuitBuilder(1, 0)
    loop_variable = builder.begin_for(range(3))
    builder.append_call(
        ReusableCircuit(
            body_builder.freeze(),
            "phase",
            call_arguments=(("angle", loop_variable),),
        ),
        (0,),
    )
    builder.end_for()

    [folded] = circuit_program_to_visual_ir(
        builder.freeze(),
        fold_loops=True,
    ).children
    [unfolded] = circuit_program_to_visual_ir(
        builder.freeze(),
        fold_loops=False,
    ).children

    assert isinstance(folded, VFoldedBlock)
    assert folded.body_lines == [f"phase(angle={loop_variable.name})"]
    assert isinstance(unfolded, VUnfoldedSequence)
    assert [
        iteration[0].label
        for iteration in unfolded.iterations
        if isinstance(iteration[0], VGate)
    ] == ["phase(angle=0)", "phase(angle=1)", "phase(angle=2)"]


def test_expanded_multi_control_call_keeps_conditional_global_phase() -> None:
    """Expanded controls stay outside an exact sparse target body and phase."""
    body_builder = CircuitBuilder(2, 0, name="phase_pair")
    body_builder.append_gate(GateKind.X, (1,))
    body_builder.add_global_phase(LiteralExpr(0.125))
    builder = CircuitBuilder(5, 0)
    builder.append_call(
        ReusableCircuit(body_builder.freeze(), "phase_pair", controls=2),
        (3, 0, 4, 1),
    )

    [expanded] = circuit_program_to_visual_ir(
        builder.freeze(),
        expand_calls=True,
    ).children

    assert isinstance(expanded, VInlineBlock)
    assert expanded.control_qubit_indices == [3, 0]
    assert expanded.affected_qubits == [1]
    assert "phase=" in expanded.label
    assert "0.125" in expanded.label
    [target_x] = expanded.children
    assert isinstance(target_x, VGate)
    assert target_x.qubit_indices == [1]


def test_inline_depth_still_limits_nested_reusable_calls() -> None:
    """Only one reusable body expands when inline depth is one."""
    leaf_builder = CircuitBuilder(1, 0, name="leaf")
    leaf_builder.append_gate(GateKind.H, (0,))
    leaf = ReusableCircuit(leaf_builder.freeze(), "leaf")
    parent_builder = CircuitBuilder(1, 0, name="parent")
    parent_builder.append_call(leaf, (0,))
    parent = ReusableCircuit(parent_builder.freeze(), "parent")
    builder = CircuitBuilder(1, 0)
    builder.append_call(parent, (0,))

    [expanded_parent] = circuit_program_to_visual_ir(
        builder.freeze(),
        expand_calls=True,
        inline_depth=1,
    ).children

    assert isinstance(expanded_parent, VInlineBlock)
    [collapsed_leaf] = expanded_parent.children
    assert isinstance(collapsed_leaf, VGate)
    assert collapsed_leaf.label == "leaf"


def test_adapter_uses_semantic_symbol_for_unnamed_call() -> None:
    """An unnamed preserved call displays its semantic operation symbol."""
    body_builder = CircuitBuilder(2, 0)
    body_builder.append_gate(GateKind.SWAP, (0, 1))
    builder = CircuitBuilder(2, 0)
    builder.append_call(
        ReusableCircuit(
            body=body_builder.freeze(),
            name="",
            identity=CallableIdentity(
                key=SemanticOpKey("qamomile.stdlib", "modular_increment"),
                symbol="modular_increment",
            ),
        ),
        (0, 1),
    )

    [call] = circuit_program_to_visual_ir(builder.freeze()).children

    assert isinstance(call, VGate)
    assert call.label == "modular_increment"


def test_adapter_preserves_nonzero_global_phase_metadata() -> None:
    """A symbolic global phase crosses the visual boundary unchanged."""
    builder = CircuitBuilder(1, 0)
    builder.add_global_phase(ParameterExpr("phi") / 2)

    circuit = circuit_program_to_visual_ir(
        builder.freeze(),
        qubit_names={0: "input"},
        output_names=["output"],
    )

    assert circuit.global_phase == "(0.0 + (phi / 2))"
    assert circuit.qubit_names == {0: "input"}
    assert circuit.output_names == ["output"]


def test_adapter_treats_numerically_close_literal_phases_as_zero() -> None:
    """The shared numerical tolerance applies at every visual phase boundary."""
    body_builder = CircuitBuilder(1, 0, name="x_body")
    body_builder.append_gate(GateKind.X, (0,))
    body_builder.add_global_phase(LiteralExpr(5e-16))
    body = body_builder.freeze()

    builder = CircuitBuilder(3, 0)
    builder.add_global_phase(LiteralExpr(5e-16))
    builder.append_call(ReusableCircuit(body, "direct_x"), (0,))
    builder.append_call(
        ReusableCircuit(body, "controlled_x", controls=1),
        (1, 2),
    )

    circuit = circuit_program_to_visual_ir(
        builder.freeze(),
        expand_calls=True,
    )

    direct, controlled = circuit.children
    assert circuit.global_phase is None
    assert isinstance(direct, VInlineBlock)
    assert direct.label == "direct_x"
    assert isinstance(controlled, VInlineBlock)
    assert controlled.control_qubit_indices == [1]
    assert controlled.affected_qubits == [2]
    [controlled_x] = controlled.children
    assert isinstance(controlled_x, VGate)
    assert controlled_x.gate_type is GateOperationType.X


def test_adapter_omits_provably_zero_phase_and_empty_loop() -> None:
    """Identity-only circuit structure does not invent visible operations."""
    builder = CircuitBuilder(1, 0)
    builder.begin_for(range(0))
    builder.append_gate(GateKind.X, (0,))
    builder.end_for()

    circuit = circuit_program_to_visual_ir(builder.freeze())

    [loop] = circuit.children
    assert isinstance(loop, VSkip)
    assert circuit.global_phase is None


def test_adapter_verifies_circuit_before_conversion() -> None:
    """Malformed wire lineage raises instead of producing a guessed drawing."""
    builder = CircuitBuilder(1, 0)
    builder.append_gate(GateKind.H, (0,))
    program = builder.freeze()
    [gate] = program.operations
    assert isinstance(gate, GateInstruction)
    malformed = dataclasses.replace(
        program,
        operations=(dataclasses.replace(gate, outputs=program.input_wires),),
    )

    with pytest.raises(ValueError, match="not unique"):
        circuit_program_to_visual_ir(malformed)
