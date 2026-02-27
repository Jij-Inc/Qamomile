"""Tests that each frontend gate correctly registers its GateOperation in the IR graph."""

import linecache

import pytest

import qamomile.circuit as qm
from qamomile.circuit.frontend.handle import Qubit
from qamomile.circuit.frontend.qkernel import qkernel
from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.value import Value


class TestSingleQubitGates:
    """Non-parameterized single-qubit gates."""

    ALL_GATES_WITH_IDS = [
        ((qm.h, GateOperationType.H), "H"),
        ((qm.x, GateOperationType.X), "X"),
        ((qm.z, GateOperationType.Z), "Z"),
    ]
    ALL_GATES = [gate_info for gate_info, _ in ALL_GATES_WITH_IDS]
    IDS = [_id for _, _id in ALL_GATES_WITH_IDS]

    @pytest.mark.parametrize("gate_fn, expected_type", ALL_GATES, ids=IDS)
    def test_gate_ir(self, gate_fn, expected_type):
        @qkernel
        def circuit(q: Qubit) -> Qubit:
            q = gate_fn(q)
            return q

        graph = circuit.build()
        assert len(graph.operations) == 2
        assert isinstance(graph.operations[0], QInitOperation)
        assert isinstance(graph.operations[1], GateOperation)
        gate_op = graph.operations[1]
        assert gate_op.gate_type == expected_type
        assert gate_op.theta is None
        assert len(gate_op.operands) == 1
        assert len(gate_op.results) == 1


class TestRotationGates:
    """Parameterized single-qubit gates."""

    ALL_GATES_WITH_IDS = [
        ((qm.rx, GateOperationType.RX), "RX"),
        ((qm.ry, GateOperationType.RY), "RY"),
        ((qm.rz, GateOperationType.RZ), "RZ"),
        ((qm.p, GateOperationType.P), "P"),
    ]
    ALL_GATES = [gate_info for gate_info, _ in ALL_GATES_WITH_IDS]
    IDS = [_id for _, _id in ALL_GATES_WITH_IDS]

    @pytest.mark.parametrize("gate_fn, expected_type", ALL_GATES, ids=IDS)
    def test_gate_ir_float_literal(self, gate_fn, expected_type):
        @qkernel
        def circuit(q: Qubit) -> Qubit:
            q = gate_fn(q, 0.5)
            return q

        graph = circuit.build()
        assert len(graph.operations) == 2
        assert isinstance(graph.operations[0], QInitOperation)
        assert isinstance(graph.operations[1], GateOperation)
        gate_op = graph.operations[1]
        assert gate_op.gate_type == expected_type
        assert gate_op.theta == 0.5
        assert len(gate_op.operands) == 1
        assert len(gate_op.results) == 1

    @pytest.mark.parametrize("gate_fn, expected_type", ALL_GATES, ids=IDS)
    def test_gate_ir_float_handle(self, gate_fn, expected_type):
        @qkernel
        def circuit(q: Qubit, theta: qm.Float) -> Qubit:
            q = gate_fn(q, theta)
            return q

        graph = circuit.build(parameters=["theta"])
        assert len(graph.operations) == 2
        assert isinstance(graph.operations[0], QInitOperation)
        assert isinstance(graph.operations[1], GateOperation)
        gate_op = graph.operations[1]
        assert gate_op.gate_type == expected_type
        assert isinstance(gate_op.theta, Value)
        assert len(gate_op.operands) == 1
        assert len(gate_op.results) == 1


class TestTwoQubitGates:
    """Non-parameterized two-qubit gates."""

    ALL_GATES_WITH_IDS = [
        ((qm.cx, GateOperationType.CX), "CX"),
        ((qm.cz, GateOperationType.CZ), "CZ"),
        ((qm.swap, GateOperationType.SWAP), "SWAP"),
    ]
    ALL_GATES = [gate_info for gate_info, _ in ALL_GATES_WITH_IDS]
    IDS = [_id for _, _id in ALL_GATES_WITH_IDS]

    @pytest.mark.parametrize("gate_fn, expected_type", ALL_GATES, ids=IDS)
    def test_gate_ir(self, gate_fn, expected_type):
        @qkernel
        def circuit(q1: Qubit, q2: Qubit) -> tuple[Qubit, Qubit]:
            q1, q2 = gate_fn(q1, q2)
            return q1, q2

        graph = circuit.build()
        assert len(graph.operations) == 3
        assert isinstance(graph.operations[0], QInitOperation)
        assert isinstance(graph.operations[1], QInitOperation)
        assert isinstance(graph.operations[2], GateOperation)
        gate_op = graph.operations[2]
        assert gate_op.gate_type == expected_type
        assert gate_op.theta is None
        assert len(gate_op.operands) == 2
        assert len(gate_op.results) == 2


class TestTwoQubitParamGates:
    """Parameterized two-qubit gates."""

    ALL_GATES_WITH_IDS = [
        ((qm.cp, GateOperationType.CP), "CP"),
        ((qm.rzz, GateOperationType.RZZ), "RZZ"),
    ]
    ALL_GATES = [gate_info for gate_info, _ in ALL_GATES_WITH_IDS]
    IDS = [_id for _, _id in ALL_GATES_WITH_IDS]

    @pytest.mark.parametrize("gate_fn, expected_type", ALL_GATES, ids=IDS)
    def test_gate_ir_float_literal(self, gate_fn, expected_type):
        @qkernel
        def circuit(q1: Qubit, q2: Qubit) -> tuple[Qubit, Qubit]:
            q1, q2 = gate_fn(q1, q2, 0.5)
            return q1, q2

        graph = circuit.build()
        assert len(graph.operations) == 3
        assert isinstance(graph.operations[0], QInitOperation)
        assert isinstance(graph.operations[1], QInitOperation)
        assert isinstance(graph.operations[2], GateOperation)
        gate_op = graph.operations[2]
        assert gate_op.gate_type == expected_type
        assert gate_op.theta == 0.5
        assert len(gate_op.operands) == 2
        assert len(gate_op.results) == 2

    @pytest.mark.parametrize("gate_fn, expected_type", ALL_GATES, ids=IDS)
    def test_gate_ir_float_handle(self, gate_fn, expected_type):
        @qkernel
        def circuit(q1: Qubit, q2: Qubit, theta: qm.Float) -> tuple[Qubit, Qubit]:
            q1, q2 = gate_fn(q1, q2, theta)
            return q1, q2

        graph = circuit.build(parameters=["theta"])
        assert len(graph.operations) == 3
        assert isinstance(graph.operations[0], QInitOperation)
        assert isinstance(graph.operations[1], QInitOperation)
        assert isinstance(graph.operations[2], GateOperation)
        gate_op = graph.operations[2]
        assert gate_op.gate_type == expected_type
        assert isinstance(gate_op.theta, Value)
        assert len(gate_op.operands) == 2
        assert len(gate_op.results) == 2


class TestThreeQubitGates:
    """ccx — three-qubit gate."""

    ALL_GATES_WITH_IDS = [((qm.ccx, GateOperationType.TOFFOLI), "CCX")]
    ALL_GATES = [gate_info for gate_info, _ in ALL_GATES_WITH_IDS]
    IDS = [_id for _, _id in ALL_GATES_WITH_IDS]

    @pytest.mark.parametrize("gate_fn, expected_type", ALL_GATES, ids=IDS)
    def test_ccx_gate_ir(self, gate_fn, expected_type):
        @qkernel
        def circuit(q1: Qubit, q2: Qubit, q3: Qubit) -> tuple[Qubit, Qubit, Qubit]:
            q1, q2, q3 = gate_fn(q1, q2, q3)
            return q1, q2, q3

        graph = circuit.build()
        assert len(graph.operations) == 4
        assert isinstance(graph.operations[0], QInitOperation)
        assert isinstance(graph.operations[1], QInitOperation)
        assert isinstance(graph.operations[2], QInitOperation)
        assert isinstance(graph.operations[3], GateOperation)
        gate_op = graph.operations[3]
        assert gate_op.gate_type == expected_type
        assert gate_op.theta is None
        assert len(gate_op.operands) == 3
        assert len(gate_op.results) == 3


class TestAllGates:
    """Test that all gates can be used together in a single circuit."""

    CATEGORIZED_GATES = [
        (TestSingleQubitGates.ALL_GATES_WITH_IDS, "q0 = qm.{name}(q0)"),
        (TestRotationGates.ALL_GATES_WITH_IDS, "q0 = qm.{name}(q0, 0.5)"),
        (TestTwoQubitGates.ALL_GATES_WITH_IDS, "q0, q1 = qm.{name}(q0, q1)"),
        (TestTwoQubitParamGates.ALL_GATES_WITH_IDS, "q0, q1 = qm.{name}(q0, q1, 0.5)"),
        (TestThreeQubitGates.ALL_GATES_WITH_IDS, "q0, q1, q2 = qm.{name}(q0, q1, q2)"),
    ]

    ALL_GATES = [gate_info for gates, _ in CATEGORIZED_GATES for gate_info, _ in gates]

    def test_all_gates_in_one_circuit(self):
        """Build a single @qkernel with every gate, verify flat GateOperations."""
        body_lines = []
        for gates, template in self.CATEGORIZED_GATES:
            for (gate_fn, _), _ in gates:
                body_lines.append("    " + template.format(name=gate_fn.__name__))

        func_source = (
            "def _circuit(q0: Qubit, q1: Qubit, q2: Qubit)"
            " -> tuple[Qubit, Qubit, Qubit]:\n"
            + "\n".join(body_lines)
            + "\n    return q0, q1, q2\n"
        )

        filename = "<all_gates_circuit>"
        compiled = compile(func_source, filename, "exec")
        linecache.cache[filename] = (
            len(func_source),
            None,
            func_source.splitlines(True),
            filename,
        )
        local_ns: dict = {}
        exec(compiled, {**globals()}, local_ns)  # noqa: S102
        circuit = qkernel(local_ns["_circuit"])

        graph = circuit.build()

        n_gates = len(self.ALL_GATES)
        n_qinits = 3
        ops = graph.operations
        assert len(ops) == n_qinits + n_gates

        expected_types = [gate_type for _, gate_type in self.ALL_GATES]
        for op in ops[:n_qinits]:
            assert isinstance(op, QInitOperation)
        for op, expected_type in zip(ops[n_qinits:], expected_types, strict=True):
            assert isinstance(op, GateOperation)
            assert op.gate_type == expected_type
