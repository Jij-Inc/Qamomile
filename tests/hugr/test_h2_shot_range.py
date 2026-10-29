"""Validate shot-range lowering against real HUGR and the pinned H2 converter."""

from __future__ import annotations

import pytest

pytest.importorskip("hugr")
pytest.importorskip("tket_exts")

import qamomile.circuit as qmc
from qamomile.circuit.transpiler import ExecutionError
from qamomile.hugr import HugrTranspiler
from qamomile.hugr._qir import to_h2_qir
from qamomile.hugr._runtime import prepare_execution
from qamomile.hugr._shot_range import (
    SHOT_SUCCESS_TAG,
    normalize_h2_shots,
    prepare_h2_shot_range,
)

pytestmark = pytest.mark.hugr


@qmc.qkernel
def _uint_table(values: qmc.Vector[qmc.UInt]) -> qmc.Bit:
    return values[qmc.get_current_shot()] == 1


@qmc.qkernel
def _bit_table(values: qmc.Vector[qmc.Bit]) -> qmc.Bit:
    return values[qmc.get_current_shot()]


@qmc.qkernel
def _affine_table(values: qmc.Vector[qmc.UInt]) -> qmc.Bit:
    return values[2 * qmc.get_current_shot() + 1] == 1


@qmc.qkernel
def _affine_bits(values: qmc.Vector[qmc.Bit]) -> qmc.Bit:
    return values[2 * qmc.get_current_shot() + 1]


@qmc.qkernel
def _bound_affine_table(values: qmc.Vector[qmc.UInt], scale: qmc.UInt) -> qmc.Bit:
    return values[qmc.get_current_shot() * scale + 1] == 1


@qmc.qkernel
def _measurement_index(values: qmc.Vector[qmc.UInt]) -> qmc.Bit:
    register = qmc.qubit_array(1, "q")
    index = qmc.measure(qmc.cast(register, qmc.QInt))
    return values[qmc.get_current_shot() + index] == 1


@qmc.qkernel
def _nonlinear(values: qmc.Vector[qmc.UInt]) -> qmc.Bit:
    shot = qmc.get_current_shot()
    return values[shot * shot] == 1


@qmc.qkernel
def _overflow(values: qmc.Vector[qmc.UInt]) -> qmc.Bit:
    return values[qmc.get_current_shot() + 18446744073709551615] == 1


@qmc.qkernel
def _shot_only() -> qmc.Bit:
    return qmc.get_current_shot() == 1


@qmc.qkernel
def _bound_index(values: qmc.Vector[qmc.UInt], index: qmc.UInt) -> qmc.Bit:
    return values[index] == 1


@qmc.qkernel
def _no_shot() -> qmc.Bit:
    return qmc.measure(qmc.qubit("q"))


def _package(kernel=_uint_table, size=4, values=None, **runtime):
    if values is None:
        values = list(range(size))
    if kernel in (_bit_table, _affine_bits):
        return prepare_execution(
            HugrTranspiler().compile(kernel, bindings={"values": values})
        ).package
    program = HugrTranspiler().transpile(
        kernel,
        parameters=["values", *runtime],
        parameter_shapes={"values": (size,)},
    )
    return prepare_execution(program._compiled, {"values": values, **runtime}).package


@pytest.mark.parametrize(
    "kernel,size,values",
    [
        (_uint_table, 4, [0, 1, 2, 3]),
        (_bit_table, 4, [False, True, False, True]),
        (_affine_table, 8, [0, 1] * 4),
        (_affine_bits, 8, [False, True] * 4),
    ],
)
@pytest.mark.parametrize("origin", [0, 1, 7])
def test_actual_qkernel_tables_produce_validated_h2_qir(kernel, size, values, origin):
    pytest.importorskip("hugr_qir")
    pyqir = pytest.importorskip("pyqir")
    package = _package(kernel, size, values)
    before = package.to_bytes()
    guarded, required = prepare_h2_shot_range(package, 4, origin)
    bitcode, outputs = to_h2_qir(guarded)
    llvm = str(pyqir.Module.from_bitcode(pyqir.Context(), bitcode))
    assert required
    assert SHOT_SUCCESS_TAG in outputs
    assert outputs["qamomile.output.0.0"] == ("bool", 1)
    assert "@abort" not in llvm
    assert "ret i64" in llvm
    assert package.to_bytes() == before


def test_runtime_scalar_coefficients_are_resolved_through_wrapper_calls():
    pytest.importorskip("hugr_qir")
    guarded, required = prepare_h2_shot_range(
        _package(_bound_affine_table, 8, scale=2), 4, 1
    )
    assert required
    assert to_h2_qir(guarded)[0].startswith(b"BC\xc0\xde")


def test_normalized_shot_links_remain_traversable():
    package = _package(_affine_table, 8)
    graph = package.modules[0]
    shot_nodes = [
        node
        for node, data in graph.nodes()
        if callable(getattr(data.op, "name", None))
        and data.op.name() == "tket.qsystem.utils.GetCurrentShot"
    ]
    guarded, _ = prepare_h2_shot_range(package, 4, 1)
    graph = guarded.modules[0]
    for node in shot_nodes:
        consumers = list(graph.linked_ports(node.out(0)))
        assert len(consumers) == 1
        assert graph[consumers[0].node].op.name().startswith("arithmetic.int.isub")


@pytest.mark.parametrize(
    "shots,origin,match",
    [
        (5, 1, "exceeds table length"),
        (4, None, "known UInt64 shot_origin"),
        (4, -1, "known UInt64 shot_origin"),
        (4, True, "known UInt64 shot_origin"),
        (0, 1, "positive UInt64"),
        (True, 1, "positive UInt64"),
        (2**64, 0, "positive UInt64"),
        (4, 2**64 - 4, "exclusive bound"),
    ],
)
def test_invalid_shot_contract_and_short_tables_are_rejected(shots, origin, match):
    with pytest.raises(ExecutionError, match=match):
        prepare_h2_shot_range(_package(), shots, origin)


@pytest.mark.parametrize(
    "kernel,match",
    [
        (_nonlinear, "multiplication by constants"),
        (_measurement_index, "Cannot prove"),
        (_overflow, "overflow"),
    ],
)
def test_unproved_indices_are_rejected_before_conversion(kernel, match):
    with pytest.raises(ExecutionError, match=match):
        prepare_h2_shot_range(_package(kernel), 4, 1)


@pytest.mark.parametrize("kernel", [_shot_only, _no_shot])
def test_programs_without_dynamic_table_reads_are_unchanged(kernel):
    package = prepare_execution(HugrTranspiler().compile(kernel)).package
    actual, required = prepare_h2_shot_range(package, 4, None)
    assert actual is package
    assert required is False


def test_selection_metadata_cannot_override_the_actual_bounds_predicate():
    package = _package()
    graph = package.modules[0]
    selections = [
        node
        for node, data in graph.nodes()
        if "qamomile.classical_selection.v1" in data.metadata
    ]
    node = selections[0]
    predicate = next(iter(graph.linked_ports(node.inp(0))))
    index = next(iter(graph.linked_ports(predicate.node.inp(0))))
    size = next(iter(graph.linked_ports(predicate.node.inp(1))))
    graph.delete_link(index, predicate.node.inp(0))
    graph.add_link(size, predicate.node.inp(0))
    with pytest.raises(ExecutionError, match="metadata does not match"):
        prepare_h2_shot_range(package, 4, 1)


def test_selection_metadata_version_is_checked():
    package = _package()
    for _, data in package.modules[0].nodes():
        if "qamomile.classical_selection.v1" in data.metadata:
            data.metadata["qamomile.classical_selection.v1"] = True
    with pytest.raises(ExecutionError, match="metadata version"):
        prepare_h2_shot_range(package, 4, 1)


def test_guard_cannot_be_applied_twice():
    guarded, _ = prepare_h2_shot_range(_package(), 4, 1)
    with pytest.raises(ExecutionError, match="cannot be applied twice"):
        prepare_h2_shot_range(guarded, 4, 1)


def test_another_entrypoint_is_rejected_without_retargeting_the_program():
    from hugr.build import Module
    from hugr.cli import validate

    package = _package()
    graph = package.modules[0]
    alternate = Module(graph).define_function("alternate", [], [], visibility="Public")
    alternate.set_outputs()
    graph.entrypoint = alternate.parent_node
    validate(package.to_bytes())
    original = package.to_bytes()
    with pytest.raises(ExecutionError, match="main as the entrypoint"):
        prepare_h2_shot_range(package, 4, 1)
    assert package.to_bytes() == original


def test_bound_scalar_table_read_without_shot_keeps_existing_execution_path():
    package = _package(_bound_index, index=1)
    assert any(
        "qamomile.classical_selection.v1" in data.metadata
        for _, data in package.modules[0].nodes()
    )
    actual, required = prepare_h2_shot_range(package, 4, None)
    assert actual is package
    assert required is False


def test_indirect_call_cannot_bypass_the_function_argument_range_proof():
    import tket_exts
    from hugr import ops, tys
    from hugr.build.dfg import Dfg
    from hugr.cli import validate
    from hugr.package import Package

    from qamomile.hugr.lowerer import _hugr_uint_type, _hugr_uint_value

    package = _package(_bound_index, values=[1, 0, 0, 0], index=0)
    graph = package.modules[0]
    call, data = next(
        (node, data) for node, data in graph.nodes() if isinstance(data.op, ops.Call)
    )
    signature = data.op.signature.body
    function = next(iter(graph.linked_ports(call.inp(len(signature.input))))).node
    table = next(iter(graph.linked_ports(call.inp(0))))
    builder = Dfg._new_existing(graph, graph[call].parent)
    loaded = builder.load_function(function)
    index = builder.load(_hugr_uint_value(4))[0]
    indirect = builder.add_op(ops.CallIndirect(signature), loaded[0], table, index)
    builder.add_op(tket_exts.result.result_bool("indirect_result"), indirect[0])
    builder.add_op(
        tket_exts.qsystem_utils()
        .get_op("GetCurrentShot")
        .instantiate(concrete_signature=tys.FunctionType([], [_hugr_uint_type()]))
    )
    package = Package([graph], [*package.extensions, tket_exts.qsystem_utils()])
    validate(package.to_bytes())
    with pytest.raises(ExecutionError, match="requires direct function calls"):
        prepare_h2_shot_range(package, 1, 0)


def test_normalization_leaves_a_program_without_shot_reads_identical():
    package = prepare_execution(HugrTranspiler().compile(_no_shot)).package
    actual, guarded = normalize_h2_shots(package, 4, None)
    assert actual is package
    assert guarded is False


def test_normalization_guards_no_array_reads_and_rejects_repeated_preparation():
    package = prepare_execution(HugrTranspiler().compile(_shot_only)).package
    before = package.to_bytes()
    normalized, guarded = normalize_h2_shots(package, 4, 1)
    assert guarded
    assert normalized is not package
    assert package.to_bytes() == before
    with pytest.raises(ExecutionError, match="cannot be applied twice"):
        normalize_h2_shots(normalized, 4, 1)
    pytest.importorskip("hugr_qir.hugr_to_qir")
    pytest.importorskip("pyqir")
    assert SHOT_SUCCESS_TAG in to_h2_qir(normalized)[1]
