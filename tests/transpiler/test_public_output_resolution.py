"""Regression tests for structural public outputs and UUID-first execution."""

from __future__ import annotations

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    BinOpKind,
    RuntimeClassicalExpr,
    RuntimeOpKind,
)
from qamomile.circuit.ir.operation.control_flow import BranchRebind, IfOperation
from qamomile.circuit.ir.operation.gate import MeasureVectorOperation
from qamomile.circuit.ir.types.primitives import BitType, QubitType, UIntType
from qamomile.circuit.ir.value import ArrayValue, DictValue, TupleValue, Value
from qamomile.circuit.transpiler.classical_executor import ClassicalExecutor
from qamomile.circuit.transpiler.errors import (
    EmitError,
    ExecutionError,
    TargetCapabilityError,
)
from qamomile.circuit.transpiler.executable import ExecutableProgram
from qamomile.circuit.transpiler.execution_context import ExecutionContext
from qamomile.circuit.transpiler.passes.emit_support.qubit_address import (
    QubitAddress,
)
from qamomile.circuit.transpiler.passes.emit_support.value_resolver import (
    ValueResolver as EmitValueResolver,
)
from qamomile.circuit.transpiler.passes.separate import SegmentationPass
from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass
from qamomile.circuit.transpiler.program_orchestrator import ProgramOrchestrator
from qamomile.circuit.transpiler.segments import (
    ClassicalSegment,
    ProgramABI,
    ProgramPlan,
    QuantumSegment,
)


@pytest.fixture(
    params=[
        "qiskit",
        pytest.param("quri_parts", marks=pytest.mark.quri_parts),
        pytest.param("cudaq", marks=pytest.mark.cudaq),
    ]
)
def backend_transpiler(request):
    """Yield each installed target name and transpiler."""
    if request.param == "qiskit":
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return request.param, QiskitTranspiler()
    if request.param == "quri_parts":
        pytest.importorskip("quri_parts")
        pytest.importorskip("quri_parts.qulacs")
        from qamomile.quri_parts import QuriPartsTranspiler

        return request.param, QuriPartsTranspiler()
    if request.param == "cudaq":
        pytest.importorskip("cudaq")
        from qamomile.cudaq import CudaqTranspiler

        return request.param, CudaqTranspiler()
    raise AssertionError(f"Unknown backend {request.param!r}")


@qmc.qkernel
def _direct_measured_element() -> qmc.Bit:
    """Return element one directly from a measured-vector expression."""
    qubits = qmc.qubit_array(2, "direct")
    qubits[1] = qmc.x(qubits[1])
    return qmc.measure(qubits)[1]


@qmc.qkernel
def _nested_view_element() -> qmc.Bit:
    """Return an element addressed through two sliced views."""
    qubits = qmc.qubit_array(4, "nested")
    qubits[3] = qmc.x(qubits[3])
    bits = qmc.measure(qubits)
    outer = bits[1:]
    inner = outer[1:]
    return inner[1]


@qmc.qkernel
def _measured_view() -> qmc.Vector[qmc.Bit]:
    """Return a sliced view of an already measured vector."""
    qubits = qmc.qubit_array(3, "view")
    qubits[1] = qmc.x(qubits[1])
    qubits[2] = qmc.x(qubits[2])
    bits = qmc.measure(qubits)
    return bits[1:]


@qmc.qkernel
def _empty_measured_view() -> qmc.Vector[qmc.Bit]:
    """Return an empty sliced view of an already measured vector."""
    qubits = qmc.qubit_array(3, "empty_view")
    qubits[1] = qmc.x(qubits[1])
    bits = qmc.measure(qubits)
    return bits[1:1]


@qmc.qkernel
def _direct_empty_measured_vector() -> qmc.Vector[qmc.Bit]:
    """Return a directly measured zero-length root vector."""
    return qmc.measure(qmc.qubit_array(0, "direct_empty"))


@qmc.qkernel
def _branch_local_empty_measured_vectors() -> qmc.Vector[qmc.Bit]:
    """Merge measurements of distinct empty roots created per branch."""
    selector_qubit = qmc.qubit("branch_empty_selector")
    selector_qubit = qmc.x(selector_qubit)
    selector = qmc.measure(selector_qubit)
    if selector:
        values = qmc.measure(qmc.qubit_array(0, "branch_empty_left"))
    else:
        values = qmc.measure(qmc.qubit_array(0, "branch_empty_right"))
    return values


@qmc.qkernel
def _premeasured_empty_vector_merge() -> qmc.Vector[qmc.Bit]:
    """Select between distinct empty measured roots in classical control."""
    selector = qmc.measure(qmc.qubit("premeasured_empty_selector"))
    left = qmc.measure(qmc.qubit_array(0, "premeasured_empty_left"))
    right = qmc.measure(qmc.qubit_array(0, "premeasured_empty_right"))
    if selector:
        values = left
    else:
        values = right
    return values


@qmc.qkernel
def _same_empty_vector_merge() -> qmc.Vector[qmc.Bit]:
    """Select one empty measured root on both runtime branch paths."""
    selector = qmc.measure(qmc.qubit("same_empty_selector"))
    empty = qmc.measure(qmc.qubit_array(0, "same_empty"))
    if selector:
        values = empty
    else:
        values = empty
    return values


@qmc.qkernel
def _empty_quantum_vector_merge() -> qmc.Vector[qmc.Bit]:
    """Measure a runtime merge of distinct zero-length quantum roots."""
    selector = qmc.measure(qmc.qubit("quantum_empty_selector"))
    if selector:
        values = qmc.qubit_array(0, "quantum_empty_left")
    else:
        values = qmc.qubit_array(0, "quantum_empty_right")
    return qmc.measure(values)


_DYNAMIC_IF_OUTPUT_KERNELS = (
    _branch_local_empty_measured_vectors,
    _empty_quantum_vector_merge,
)


@qmc.qkernel
def _multiple_measured_elements() -> tuple[qmc.Bit, qmc.Bit]:
    """Return multiple measured elements in public tuple order."""
    qubits = qmc.qubit_array(3, "multiple")
    qubits[0] = qmc.x(qubits[0])
    bits = qmc.measure(qubits)
    return bits[2], bits[0]


@qmc.qkernel
def _whole_measured_vector() -> qmc.Vector[qmc.Bit]:
    """Return the original measured vector without a structural alias."""
    qubits = qmc.qubit_array(2, "whole")
    qubits[1] = qmc.x(qubits[1])
    return qmc.measure(qubits)


@qmc.qkernel
def _bound_view(
    values: qmc.Vector[qmc.UInt],
) -> tuple[qmc.Vector[qmc.UInt], qmc.Bit]:
    """Return a compile-time-bound classical view and a quantum anchor."""
    anchor = qmc.qubit("bound_view_anchor")
    return values[1:], qmc.measure(anchor)


@qmc.qkernel
def _stored_element() -> qmc.Bit:
    """Return an element from a host-side updated measured array."""
    qubits = qmc.qubit_array(2, "stored")
    qubits[0] = qmc.x(qubits[0])
    bits = qmc.measure(qubits)
    bits[1] = bits[0]
    return bits[1]


@qmc.qkernel
def _loop_final_element() -> qmc.Bit:
    """Return the vector element selected by the final loop iteration."""
    qubits = qmc.qubit_array(3, "loop")
    qubits[2] = qmc.x(qubits[2])
    bits = qmc.measure(qubits)
    selected = bits[0]
    for index in qmc.range(3):
        selected = bits[index]
    return selected


@qmc.qkernel
def _constant_name_collision(uint_const: qmc.UInt) -> qmc.UInt:
    """Combine a parameter with a same-named typed constant."""
    anchor = qmc.qubit("collision_anchor")
    qmc.measure(anchor)
    one = qmc.uint(1)
    return uint_const + one


@qmc.qkernel
def _runtime_expr_loop_carry(set_condition: qmc.UInt) -> qmc.UInt:
    """Carry a measurement-selected arithmetic value across a static loop."""
    condition_qubit = qmc.qubit("runtime_expr_condition")
    if set_condition > 0:
        condition_qubit = qmc.x(condition_qubit)
    measured_bit = qmc.measure(condition_qubit)
    value = qmc.uint(5)
    for _ in qmc.range(2):
        if measured_bit:
            value = qmc.uint(1)
        else:
            value = value + 2
    return value


def _sample(kernel, *, transpiler=None, bindings=None, parameters=None):
    """Transpile and sample one deterministic qkernel.

    Args:
        kernel (QKernel): Kernel to compile.
        transpiler (Transpiler | None): Backend transpiler/executor provider.
            Defaults to a fresh Qiskit transpiler.
        bindings (dict[str, object] | None): Compile-time bindings. Defaults
            to None.
        parameters (list[str] | None): Runtime parameter names. Defaults to
            None.

    Returns:
        object: The sole deterministic public output.
    """
    if transpiler is None:
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
    executable = transpiler.transpile(
        kernel,
        bindings=bindings,
        parameters=parameters,
    )
    runtime_bindings = {"uint_const": 5} if parameters else None
    result = executable.sample(
        transpiler.executor(),
        shots=16,
        bindings=runtime_bindings,
    ).result()
    assert len(result.results) == 1
    output, count = result.results[0]
    assert count == 16
    return output


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        (_direct_measured_element, 1),
        (_nested_view_element, 1),
        (_measured_view, (1, 1)),
        (_empty_measured_view, ()),
        (_direct_empty_measured_vector, ()),
        (_branch_local_empty_measured_vectors, ()),
        (_premeasured_empty_vector_merge, ()),
        (_same_empty_vector_merge, ()),
        (_empty_quantum_vector_merge, ()),
        (_multiple_measured_elements, (0, 1)),
        (_whole_measured_vector, (0, 1)),
        (_stored_element, 1),
        (_loop_final_element, 1),
    ],
)
def test_structural_bit_outputs_resolve(backend_transpiler, kernel, expected):
    """Elements, nested views, stores, and loop-final selections stay live."""
    target, transpiler = backend_transpiler
    if target == "quri_parts" and kernel in _DYNAMIC_IF_OUTPUT_KERNELS:
        with pytest.raises(
            TargetCapabilityError,
            match="cannot represent measurement-conditioned branching",
        ):
            transpiler.transpile(kernel)
        return
    assert _sample(kernel, transpiler=transpiler) == expected


def test_compile_time_bound_view_is_materialized():
    """A bound classical view is returned in public output order."""
    assert _sample(_bound_view, bindings={"values": [4, 5, 6]}) == ((5, 6), 0)


def test_typed_constant_wins_over_same_named_parameter():
    """A ``uint_const`` parameter cannot shadow ``qmc.uint(1)``."""
    assert _sample(_constant_name_collision, parameters=["uint_const"]) == 6


@pytest.mark.parametrize(
    ("set_condition", "expected"),
    [(0, 9), (1, 1)],
)
def test_runtime_classical_expr_executes_inside_loop_carry(
    backend_transpiler,
    set_condition,
    expected,
):
    """Lowered arithmetic executes for both measurement-selected branches."""
    _, transpiler = backend_transpiler
    assert (
        _sample(
            _runtime_expr_loop_carry,
            transpiler=transpiler,
            bindings={"set_condition": set_condition},
        )
        == expected
    )


def test_classical_executor_uses_identity_and_metadata_before_names():
    """UUID, array metadata, and dict metadata beat legacy display bindings."""
    executor = ClassicalExecutor()
    context = ExecutionContext(
        {
            "shared": 99,
            "array": (99, 99),
            "mapping": {99: 99},
        }
    )

    temporary = Value(type=UIntType(), name="shared")
    assert executor._get_value(temporary, context, {temporary.uuid: 7}, {}) == 7

    constant = Value(type=UIntType(), name="shared").with_const(1)
    assert executor._get_value(constant, context, {}, {}) == 1

    array = ArrayValue(type=UIntType(), name="array").with_array_runtime_metadata(
        const_array=(2, 3)
    )
    assert executor._get_array_data(array, context, {}, {}) == (2, 3)

    mapping = DictValue(name="mapping").with_dict_runtime_metadata({1: 2})
    assert executor._get_iterable(mapping, context, {}, {}) == [(1, 2)]


@pytest.mark.parametrize(
    ("kind", "raw_operands", "expected"),
    [
        (RuntimeOpKind.EQ, [2, 2], True),
        (RuntimeOpKind.NEQ, [2, 3], True),
        (RuntimeOpKind.LT, [2, 3], True),
        (RuntimeOpKind.LE, [2, 2], True),
        (RuntimeOpKind.GT, [3, 2], True),
        (RuntimeOpKind.GE, [2, 2], True),
        (RuntimeOpKind.AND, [1, 0], False),
        (RuntimeOpKind.OR, [0, 1], True),
        (RuntimeOpKind.NOT, [0], True),
        (RuntimeOpKind.ADD, [5, 2], 7),
        (RuntimeOpKind.SUB, [5, 2], 3),
        (RuntimeOpKind.MUL, [5, 2], 10),
        (RuntimeOpKind.DIV, [5, 2], 2.5),
        (RuntimeOpKind.FLOORDIV, [5, 2], 2),
        (RuntimeOpKind.MOD, [5, 2], 1),
        (RuntimeOpKind.POW, [5, 2], 25),
    ],
)
def test_classical_executor_dispatches_every_runtime_expression_kind(
    kind,
    raw_operands,
    expected,
):
    """Every lowered runtime-expression kind shares scalar op semantics."""
    operands = [Value(type=UIntType(), name="shared") for _ in range(len(raw_operands))]
    output = Value(type=UIntType(), name="runtime_output")
    operation = RuntimeClassicalExpr(
        kind=kind,
        operands=operands,
        results=[output],
    )
    uuid_values = {
        operand.uuid: raw for operand, raw in zip(operands, raw_operands, strict=True)
    }
    context = ExecutionContext({"shared": 99, **uuid_values})

    results = ClassicalExecutor().execute(
        ClassicalSegment(operations=[operation]),
        context,
    )

    assert results[output.uuid] == expected


def test_emit_resolver_uses_uuid_and_structure_before_names():
    """Emit-time lookup leaves display-name compatibility as the last resort."""
    resolver = EmitValueResolver()

    constant = Value(type=UIntType(), name="shared").with_const(1)
    assert resolver.resolve_classical_value(constant, {"shared": 99}) == 1
    assert (
        resolver.resolve_classical_value(
            constant,
            {"shared": 99, constant.uuid: 7},
        )
        == 7
    )

    array = ArrayValue(type=UIntType(), name="array").with_array_runtime_metadata(
        const_array=(2, 3)
    )
    index = Value(type=UIntType(), name="index").with_const(1)
    element = Value(
        type=UIntType(),
        name="array[index]",
        parent_array=array,
        element_indices=(index,),
    )
    assert resolver.resolve_classical_value(element, {"array": (99, 99)}) == 3


@pytest.mark.parametrize("raw", [True, 1.0, 1.5, "1"])
def test_structural_output_indices_reject_non_integral_runtime_values(raw):
    """Structural output addressing fails closed on non-integral values."""
    index = Value(type=UIntType(), name="index")
    context = ExecutionContext({index.uuid: raw})
    orchestrator = ProgramOrchestrator(ExecutableProgram())

    assert orchestrator._resolve_context_int_value(index, context) is None


def test_nested_container_public_output_is_reconstructed_atomically():
    """Nested tuple/dict outputs combine measured and constant leaves."""
    size = Value(type=UIntType(), name="size").with_const(1)
    measured = ArrayValue(type=BitType(), name="measured", shape=(size,))
    index = Value(type=UIntType(), name="index").with_const(0)
    bit = Value(
        type=BitType(),
        name="measured[index]",
        parent_array=measured,
        element_indices=(index,),
    )
    key = Value(type=UIntType(), name="key").with_const(7)
    constant = Value(type=UIntType(), name="constant").with_const(3)
    payload = TupleValue(name="payload", elements=(bit, constant))
    mapping = DictValue(name="mapping", entries=((key, payload),))
    output = TupleValue(name="output", elements=(constant, mapping))
    plan = ProgramPlan(abi=ProgramABI(output_values=[output]))
    executable = ExecutableProgram(plan=plan, output_values=[output])
    context = ExecutionContext({f"{measured.uuid}_0": 1})

    resolved = ProgramOrchestrator(executable)._resolve_outputs(context)

    assert resolved == (3, {7: (1, 3)})


def test_nested_container_public_output_rejects_partial_materialization():
    """One unresolved nested leaf makes the complete output unavailable."""
    constant = Value(type=UIntType(), name="constant").with_const(3)
    missing = Value(type=UIntType(), name="missing")
    payload = TupleValue(name="payload", elements=(constant, missing))
    mapping = DictValue(name="mapping", entries=((constant, payload),))
    output = TupleValue(name="output", elements=(constant, mapping))
    plan = ProgramPlan(abi=ProgramABI(output_values=[output]))
    executable = ExecutableProgram(plan=plan, output_values=[output])

    with pytest.raises(ExecutionError, match="could not be resolved"):
        ProgramOrchestrator(executable)._resolve_outputs(ExecutionContext())


def test_segment_outputs_contain_only_definitions_live_across_boundaries():
    """Dead definitions are omitted while structural root reads stay live."""
    size = Value(type=UIntType(), name="size").with_const(2)
    live_qubits = ArrayValue(type=QubitType(), name="live_q", shape=(size,))
    dead_qubits = ArrayValue(type=QubitType(), name="dead_q", shape=(size,))
    live_bits = ArrayValue(type=BitType(), name="live_b", shape=(size,))
    dead_bits = ArrayValue(type=BitType(), name="dead_b", shape=(size,))
    live_measure = MeasureVectorOperation(
        operands=[live_qubits],
        results=[live_bits],
    )
    dead_measure = MeasureVectorOperation(
        operands=[dead_qubits],
        results=[dead_bits],
    )

    index = Value(type=UIntType(), name="index").with_const(1)
    element = Value(
        type=BitType(),
        name="live_b[index]",
        parent_array=live_bits,
        element_indices=(index,),
    )
    output = Value(type=UIntType(), name="output")
    post_op = BinOp(
        kind=BinOpKind.ADD,
        operands=[element, element],
        results=[output],
    )
    quantum = QuantumSegment(operations=[live_measure, dead_measure])
    classical = ClassicalSegment(operations=[post_op])
    block = Block(
        input_values=[live_qubits, dead_qubits],
        output_values=[output],
        operations=[live_measure, dead_measure, post_op],
    )

    SegmentationPass()._compute_segment_io([quantum, classical], block)

    assert set(quantum.output_refs) == {live_bits.uuid}
    assert live_bits.uuid in classical.input_refs
    assert dead_bits.uuid not in quantum.output_refs
    assert classical.output_refs == [output.uuid]


def test_structural_reference_walk_descends_into_tuple_and_dict_values():
    """Tuple elements and dictionary entries participate in escape liveness."""
    key_part = Value(type=UIntType(), name="key_part")
    item = Value(type=UIntType(), name="item")
    key = TupleValue(name="key", elements=(key_part,))
    mapping = DictValue(name="mapping", entries=((key, item),))

    references = SegmentationPass()._value_reference_uuids(mapping)

    assert references == {mapping.uuid, key.uuid, key_part.uuid, item.uuid}


def test_nested_container_bit_outputs_register_structural_clbit_aliases():
    """Tuple and dictionary Bit leaves receive public clbit aliases."""
    size = Value(type=UIntType(), name="size").with_const(2)
    measured = ArrayValue(type=BitType(), name="measured", shape=(size,))
    zero = Value(type=UIntType(), name="zero").with_const(0)
    one = Value(type=UIntType(), name="one").with_const(1)
    first = Value(
        type=BitType(),
        name="measured[zero]",
        parent_array=measured,
        element_indices=(zero,),
    )
    second = Value(
        type=BitType(),
        name="measured[one]",
        parent_array=measured,
        element_indices=(one,),
    )
    key = TupleValue(name="key", elements=(first,))
    mapping = DictValue(name="mapping", entries=((key, second),))
    emit_pass = StandardEmitPass(object())
    emit_pass._program_output_values = (mapping,)
    clbit_map = {
        QubitAddress(measured.uuid, 0): 4,
        QubitAddress(measured.uuid, 1): 7,
    }

    emit_pass._register_public_output_clbit_aliases(clbit_map)

    assert clbit_map[QubitAddress(first.uuid)] == 4
    assert clbit_map[QubitAddress(second.uuid)] == 7


def test_nested_dict_bit_output_detects_overwritten_while_snapshot():
    """A stale Bit nested in a dictionary remains a checked live output."""
    size = Value(type=UIntType(), name="size").with_const(2)
    measured = ArrayValue(type=BitType(), name="measured", shape=(size,))
    zero = Value(type=UIntType(), name="zero").with_const(0)
    stale = Value(
        type=BitType(),
        name="measured[zero]",
        parent_array=measured,
        element_indices=(zero,),
    )
    key = Value(type=UIntType(), name="key").with_const(1)
    mapping = DictValue(name="mapping", entries=((key, stale),))
    emit_pass = StandardEmitPass(object())
    emit_pass._program_output_refs = frozenset({mapping.uuid})
    emit_pass._program_output_values = (mapping,)
    emit_pass._overwritten_runtime_condition_sources = {
        (str(QubitAddress(measured.uuid, 0)), 3)
    }

    with pytest.raises(EmitError, match="snapshot remains live"):
        emit_pass._reject_overwritten_live_condition_outputs()


def test_nested_dict_disjoint_bit_output_survives_while_clbit_reuse():
    """Container ancestry does not widen one Bit leaf to its whole vector."""
    size = Value(type=UIntType(), name="size").with_const(2)
    measured = ArrayValue(type=BitType(), name="measured", shape=(size,))
    one = Value(type=UIntType(), name="one").with_const(1)
    sibling = Value(
        type=BitType(),
        name="measured[one]",
        parent_array=measured,
        element_indices=(one,),
    )
    key = Value(type=UIntType(), name="key").with_const(1)
    mapping = DictValue(name="mapping", entries=((key, sibling),))
    emit_pass = StandardEmitPass(object())
    emit_pass._program_output_refs = frozenset({mapping.uuid})
    emit_pass._program_output_values = (mapping,)
    emit_pass._overwritten_runtime_condition_sources = {
        (str(QubitAddress(measured.uuid, 0)), 3)
    }

    emit_pass._reject_overwritten_live_condition_outputs()


def test_structural_public_element_keeps_root_measurement_live():
    """A public element descriptor retains its defining measured root."""
    size = Value(type=UIntType(), name="size").with_const(2)
    qubits = ArrayValue(type=QubitType(), name="q", shape=(size,))
    bits = ArrayValue(type=BitType(), name="b", shape=(size,))
    measure = MeasureVectorOperation(operands=[qubits], results=[bits])
    index = Value(type=UIntType(), name="index").with_const(1)
    output = Value(
        type=BitType(),
        name="b[index]",
        parent_array=bits,
        element_indices=(index,),
    )
    quantum = QuantumSegment(operations=[measure])
    block = Block(
        input_values=[qubits],
        output_values=[output],
        operations=[measure],
    )

    SegmentationPass()._compute_segment_io([quantum], block)

    assert quantum.output_refs == [bits.uuid]


def test_constant_scalar_reference_keeps_only_structural_dependencies():
    """A materialized scalar drops its own read but retains array addressing."""
    size = Value(type=UIntType(), name="size").with_const(2)
    parent = ArrayValue(type=UIntType(), name="values", shape=(size,))
    index = Value(type=UIntType(), name="index")
    element = Value(
        type=UIntType(),
        name="values[index]",
        parent_array=parent,
        element_indices=(index,),
    ).with_const(7)

    references = SegmentationPass()._value_reference_uuids(element)

    assert element.uuid not in references
    assert parent.uuid in references
    assert index.uuid in references
    assert size.uuid not in references


def test_unused_divergent_array_merge_in_rebind_metadata_is_not_rejected():
    """Overwrite bookkeeping does not count as a divergent-merge read."""
    root_size = Value(type=UIntType(), name="root_size").with_const(3)
    view_size = Value(type=UIntType(), name="view_size").with_const(2)
    zero = Value(type=UIntType(), name="zero").with_const(0)
    one = Value(type=UIntType(), name="one").with_const(1)
    root = ArrayValue(type=QubitType(), name="root", shape=(root_size,))
    left = ArrayValue(
        type=QubitType(),
        name="left",
        shape=(view_size,),
        slice_of=root,
        slice_start=zero,
        slice_step=one,
    )
    right = ArrayValue(
        type=QubitType(),
        name="right",
        shape=(view_size,),
        slice_of=root,
        slice_start=one,
        slice_step=one,
    )
    condition = Value(type=BitType(), name="condition")
    divergent = ArrayValue(type=QubitType(), name="view", shape=(view_size,))
    selecting_if = IfOperation(operands=[condition])
    selecting_if.add_merge(left, right, divergent)

    replacement = left.next_version()
    overwritten = ArrayValue(type=QubitType(), name="view", shape=(view_size,))
    overwrite_if = IfOperation(
        operands=[condition],
        branch_rebinds=(
            BranchRebind(
                var_name="view",
                before=divergent,
                rebound_in_true=True,
                rebound_in_false=True,
            ),
        ),
    )
    overwrite_if.add_merge(replacement, replacement, overwritten)

    segmentation = SegmentationPass()
    segmentation._measurement_tainted = {condition.uuid}
    segmentation._dependency_graph = {}
    segmentation._quantum_value_uuids = set()
    segmentation._block_output_values = ()

    assert divergent.uuid not in segmentation._segment_read_uuids([overwrite_if])
    segmentation._reject_quantum_segment_carry_escapes(
        [QuantumSegment(operations=[selecting_if, overwrite_if])]
    )
