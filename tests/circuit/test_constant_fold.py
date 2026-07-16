"""Tests for constant folding of expressions and structural outputs."""

import qamomile.circuit as qmc
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.arithmetic_operations import BinOp, BinOpKind
from qamomile.circuit.ir.operation.classical_ops import StoreArrayElementOperation
from qamomile.circuit.ir.types.primitives import FloatType, UIntType
from qamomile.circuit.ir.value import ArrayValue, DictValue, TupleValue, Value
from qamomile.circuit.transpiler.passes.constant_fold import ConstantFoldingPass
from qamomile.qiskit import QiskitTranspiler


def test_constant_fold_binop_theta():
    """BinOp result used as GateOperation.theta should be correctly folded."""

    @qmc.qkernel
    def circuit(theta: qmc.Float) -> qmc.Bit:
        doubled = theta * 2
        q = qmc.qubit("q")
        q = qmc.rz(q, doubled)
        return qmc.measure(q)

    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(circuit, bindings={"theta": 0.5})
    rz_params = [
        instr.operation.params[0]
        for instr in executable.quantum_circuit.data
        if instr.operation.name == "rz"
    ]
    assert len(rz_params) == 1
    assert abs(float(rz_params[0]) - 1.0) < 1e-10


def test_constant_fold_chained_binop_theta():
    """Chained BinOp results should be correctly folded through to theta."""

    @qmc.qkernel
    def circuit(theta: qmc.Float) -> qmc.Bit:
        a = theta * 2  # 0.5 * 2 = 1.0
        b = a + 0.5  # 1.0 + 0.5 = 1.5
        q = qmc.qubit("q")
        q = qmc.rx(q, b)
        return qmc.measure(q)

    transpiler = QiskitTranspiler()
    executable = transpiler.transpile(circuit, bindings={"theta": 0.5})
    rx_params = [
        instr.operation.params[0]
        for instr in executable.quantum_circuit.data
        if instr.operation.name == "rx"
    ]
    assert len(rx_params) == 1
    assert abs(float(rx_params[0]) - 1.5) < 1e-10


def test_nested_structural_output_keeps_folded_store():
    """A store returned inside TupleValue remains available at runtime."""
    size = Value(type=UIntType(), name="size").with_const(2)
    source = ArrayValue(
        type=FloatType(), name="values", shape=(size,)
    ).with_array_runtime_metadata(const_array=(1.0, 0.0))
    result = ArrayValue(
        type=FloatType(),
        name=source.name,
        version=source.version + 1,
        logical_id=source.logical_id,
        shape=source.shape,
    )
    stored_value = Value(type=FloatType(), name="stored").with_const(2.0)
    index = Value(type=UIntType(), name="index").with_const(1)
    store = StoreArrayElementOperation(
        operands=[source, stored_value, index],
        results=[result],
    )
    block = Block(
        name="nested_store_output",
        operations=[store],
        output_values=[TupleValue(name="output", elements=(result,))],
        kind=BlockKind.AFFINE,
    )

    folded = ConstantFoldingPass().run(block)

    assert len(folded.operations) == 1
    assert isinstance(folded.operations[0], StoreArrayElementOperation)
    [output] = folded.output_values
    assert isinstance(output, TupleValue)
    [folded_result] = output.elements
    assert isinstance(folded_result, ArrayValue)
    assert folded_result.get_const_array() == (1.0, 2.0)


def test_constant_fold_rewrites_value_inside_dict_output():
    """A folded scalar remains reachable through a structural DictValue."""
    left = Value(type=UIntType(), name="left").with_const(1)
    right = Value(type=UIntType(), name="right").with_const(2)
    result = Value(type=UIntType(), name="sum")
    operation = BinOp(
        kind=BinOpKind.ADD,
        operands=[left, right],
        results=[result],
    )
    key = Value(type=UIntType(), name="key").with_const(7)
    output = DictValue(name="output", entries=((key, result),))
    block = Block(
        name="dict_output",
        operations=[operation],
        output_values=[output],
        kind=BlockKind.AFFINE,
    )

    folded = ConstantFoldingPass().run(block)

    assert folded.operations == []
    [folded_output] = folded.output_values
    assert isinstance(folded_output, DictValue)
    [(folded_key, folded_value)] = folded_output.entries
    assert folded_key is key
    assert folded_value.get_const() == 3


def test_store_folding_does_not_resolve_array_by_display_name():
    """An unrelated binding with the same display name cannot seed a store."""
    size = Value(type=UIntType(), name="size").with_const(1)
    source = ArrayValue(type=FloatType(), name="collision", shape=(size,))
    result = ArrayValue(
        type=FloatType(),
        name=source.name,
        version=1,
        logical_id=source.logical_id,
        shape=source.shape,
    )
    store = StoreArrayElementOperation(
        operands=[
            source,
            Value(type=FloatType(), name="stored").with_const(2.0),
            Value(type=UIntType(), name="index").with_const(0),
        ],
        results=[result],
    )
    block = Block(
        name="display_name_collision",
        operations=[store],
        output_values=[],
        kind=BlockKind.AFFINE,
    )

    folded = ConstantFoldingPass(bindings={"collision": [9.0]}).run(block)

    assert folded.operations == [store]


def test_binop_folding_rejects_non_scalar_parameter_bindings():
    """Container bindings cannot be folded through Python list arithmetic."""
    left = Value(type=FloatType(), name="left").with_parameter("left")
    right = Value(type=FloatType(), name="right").with_const(2.0)
    result = Value(type=FloatType(), name="result")
    operation = BinOp(
        kind=BinOpKind.MUL,
        operands=[left, right],
        results=[result],
    )
    block = Block(
        name="non_scalar_binop",
        operations=[operation],
        output_values=[result],
        kind=BlockKind.AFFINE,
    )

    folded = ConstantFoldingPass(bindings={"left": [1.0]}).run(block)

    assert folded.operations == [operation]
    assert folded.output_values == [result]
