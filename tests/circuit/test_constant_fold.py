"""Tests for constant folding of expressions and structural outputs."""

import qamomile.circuit as qmc
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation.classical_ops import StoreArrayElementOperation
from qamomile.circuit.ir.types.primitives import FloatType, UIntType
from qamomile.circuit.ir.value import ArrayValue, TupleValue, Value
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
