"""Tests that each frontend gate correctly registers its GateOperation in the IR graph."""

import linecache
from collections.abc import Callable

import numpy as np
import pytest

import qamomile.circuit as qm
from qamomile.circuit.frontend.handle import Qubit
from qamomile.circuit.frontend.qkernel import QKernel, qkernel
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.gate import GateOperation, GateOperationType
from qamomile.circuit.ir.operation.operation import QInitOperation
from qamomile.circuit.ir.value import Value


def _build_qkernel(func_source: str, filename: str) -> QKernel:
    """Compile a function source string into a @qkernel."""
    compiled = compile(func_source, filename, "exec")
    linecache.cache[filename] = (
        len(func_source),
        None,
        func_source.splitlines(True),
        filename,
    )
    local_ns: dict = {}
    exec(compiled, {**globals()}, local_ns)  # noqa: S102
    return qkernel(local_ns["_circuit"])


def _build_random_gate_circuit(
    gates: list[tuple[Callable, GateOperationType]],
    template: str,
    n_qubits: int,
    seed: int,
    *,
    theta: str | float | list[float] | None = None,
) -> tuple[Block, list[GateOperationType], int, list[float] | None]:
    """Pick random gates, build a qkernel, return (graph, expected_types, num_gates).

    Args:
        theta: None for non-parameterized gates, "literal" to auto-generate
               random per-gate float values, a scalar (e.g. 0.5) to use the
               same literal for every gate, a list of floats for per-gate
               literal values, or "symbolic" for per-gate Float parameters.

    Returns:
        (graph, expected_types, num_gates, thetas) where thetas is None for
        non-parameterized / symbolic gates, or a list of float values.
    """
    rng = np.random.default_rng(seed)
    num_gates = rng.integers(2, 100)
    random_gates = rng.choice(gates, size=num_gates)

    # Resolve theta values.
    thetas = None
    if theta == "literal":
        thetas = rng.uniform(0, 2 * np.pi, size=num_gates).tolist()
    elif isinstance(theta, list):
        thetas = theta
    elif theta is not None and theta != "symbolic":
        thetas = [theta] * num_gates

    # Build the parameter section.
    params = ", ".join(f"q{i}: Qubit" for i in range(n_qubits))
    build_parameters = None
    if theta == "symbolic":
        params += ", " + ", ".join(f"theta{i}: qm.Float" for i in range(num_gates))
        build_parameters = [f"theta{i}" for i in range(num_gates)]
    # Build the body section.
    body_lines = []
    for idx, (fn, _) in enumerate(random_gates):
        fmt = {"name": fn.__name__, "i": idx}
        if theta == "symbolic":
            fmt["theta"] = f"theta{idx}"
        elif thetas is not None:
            fmt["theta"] = str(thetas[idx])
        body_lines.append(f"    {template.format(**fmt)}")
    # Build the return section.
    returns = ", ".join(f"q{i}" for i in range(n_qubits))
    return_type = (
        "Qubit" if n_qubits == 1 else f"tuple[{', '.join(['Qubit'] * n_qubits)}]"
    )
    # Build the full function source.
    func_source = (
        f"def _circuit({params}) -> {return_type}:\n"
        + "\n".join(body_lines)
        + f"\n    return {returns}\n"
    )
    # Build QKernel form of the function.
    circuit = _build_qkernel(func_source, f"<randomized_gates_{seed}>")

    # Build the graph and obtain expected types list.
    graph = circuit.build(parameters=build_parameters)
    expected_types = [gate_type for _, gate_type in random_gates]

    return graph, expected_types, num_gates, thetas


class TestSingleQubitGates:
    """Non-parameterized single-qubit gates."""

    GATE_TEMPLATE = "q0 = qm.{name}(q0)"
    N_QUBITS = 1

    ALL_GATES_WITH_IDS = [
        ((qm.h, GateOperationType.H), "H"),
        ((qm.x, GateOperationType.X), "X"),
        ((qm.y, GateOperationType.Y), "Y"),
        ((qm.z, GateOperationType.Z), "Z"),
        ((qm.t, GateOperationType.T), "T"),
        ((qm.tdg, GateOperationType.TDG), "TDG"),
        ((qm.s, GateOperationType.S), "S"),
        ((qm.sdg, GateOperationType.SDG), "SDG"),
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
        assert len(gate_op.qubit_operands) == 1
        assert len(gate_op.results) == 1

    @pytest.mark.parametrize("seed", [901 + offset for offset in range(50)])
    def test_randomized_gates(self, seed):
        """Test random sequences of non-parameterized single-qubit gates."""
        graph, expected_types, num_gates, _ = _build_random_gate_circuit(
            gates=self.ALL_GATES,
            template=self.GATE_TEMPLATE,
            n_qubits=self.N_QUBITS,
            seed=seed,
        )
        assert len(graph.operations) == self.N_QUBITS + num_gates
        for op in graph.operations[: self.N_QUBITS]:
            assert isinstance(op, QInitOperation)
        for op, expected_type in zip(
            graph.operations[self.N_QUBITS :], expected_types, strict=True
        ):
            assert isinstance(op, GateOperation)
            assert op.gate_type == expected_type
            assert op.theta is None
            assert len(op.qubit_operands) == self.N_QUBITS
            assert len(op.results) == self.N_QUBITS

    @pytest.mark.parametrize("gate_fn, expected_type", ALL_GATES, ids=IDS)
    @pytest.mark.parametrize("actual_n", [1, 2, 5, 10])
    def test_gate_ir_given_n(self, gate_fn, expected_type, actual_n):
        @qkernel
        def circuit(n: qm.UInt) -> qm.Vector[Qubit]:
            qs = qm.qubit_array(n, "qs")
            qs[0] = gate_fn(qs[0])
            return qs

        graph = circuit.build(n=actual_n)
        assert len(graph.operations) == 2
        assert isinstance(graph.operations[0], QInitOperation)
        assert isinstance(graph.operations[1], GateOperation)
        gate_op = graph.operations[1]
        assert gate_op.gate_type == expected_type
        assert gate_op.theta is None
        assert len(gate_op.qubit_operands) == 1
        assert len(gate_op.results) == 1

    @pytest.mark.parametrize("gate_fn, expected_type", ALL_GATES, ids=IDS)
    def test_gate_ir_qubits(self, gate_fn, expected_type):
        @qkernel
        def circuit(qs: qm.Vector[Qubit]) -> qm.Vector[Qubit]:
            qs[0] = gate_fn(qs[0])
            return qs

        graph = circuit.build()
        assert len(graph.operations) == 2
        assert isinstance(graph.operations[0], QInitOperation)
        assert isinstance(graph.operations[1], GateOperation)
        gate_op = graph.operations[1]
        assert gate_op.gate_type == expected_type
        assert gate_op.theta is None
        assert len(gate_op.qubit_operands) == 1
        assert len(gate_op.results) == 1


class TestRotationGates:
    """Parameterized single-qubit gates."""

    GATE_TEMPLATE = "q0 = qm.{name}(q0, {theta})"
    N_QUBITS = 1

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
        assert gate_op.theta.get_const() == pytest.approx(0.5)
        assert len(gate_op.qubit_operands) == 1
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
        assert len(gate_op.qubit_operands) == 1
        assert len(gate_op.results) == 1

    @pytest.mark.parametrize("seed", [901 + offset for offset in range(50)])
    def test_randomized_gates_literal(self, seed):
        """Test random sequences of rotation gates with literal theta."""
        graph, expected_types, num_gates, thetas = _build_random_gate_circuit(
            self.ALL_GATES,
            self.GATE_TEMPLATE,
            self.N_QUBITS,
            seed,
            theta="literal",
        )
        assert thetas is not None
        assert len(graph.operations) == self.N_QUBITS + num_gates
        for op in graph.operations[: self.N_QUBITS]:
            assert isinstance(op, QInitOperation)
        for op, expected_type, expected_theta in zip(
            graph.operations[self.N_QUBITS :], expected_types, thetas, strict=True
        ):
            assert isinstance(op, GateOperation)
            assert op.gate_type == expected_type
            assert op.theta.get_const() == pytest.approx(expected_theta)
            assert len(op.qubit_operands) == self.N_QUBITS
            assert len(op.results) == self.N_QUBITS

    @pytest.mark.parametrize("seed", [901 + offset for offset in range(50)])
    def test_randomized_gates_symbolic(self, seed):
        """Test random sequences of rotation gates with symbolic theta."""
        graph, expected_types, num_gates, _ = _build_random_gate_circuit(
            self.ALL_GATES,
            self.GATE_TEMPLATE,
            self.N_QUBITS,
            seed,
            theta="symbolic",
        )
        assert len(graph.operations) == self.N_QUBITS + num_gates
        for op in graph.operations[: self.N_QUBITS]:
            assert isinstance(op, QInitOperation)
        for op, expected_type in zip(
            graph.operations[self.N_QUBITS :], expected_types, strict=True
        ):
            assert isinstance(op, GateOperation)
            assert op.gate_type == expected_type
            assert isinstance(op.theta, Value)
            assert len(op.qubit_operands) == self.N_QUBITS
            assert len(op.results) == self.N_QUBITS

    @pytest.mark.parametrize("gate_fn, expected_type", ALL_GATES, ids=IDS)
    @pytest.mark.parametrize("actual_n", [1, 2, 5, 10])
    def test_gate_ir_given_n(self, gate_fn, expected_type, actual_n):
        @qkernel
        def circuit(n: qm.UInt) -> qm.Vector[Qubit]:
            qs = qm.qubit_array(n, "qs")
            qs[0] = gate_fn(qs[0], 0.5)
            return qs

        graph = circuit.build(n=actual_n)
        assert len(graph.operations) == 2
        assert isinstance(graph.operations[0], QInitOperation)
        assert isinstance(graph.operations[1], GateOperation)
        gate_op = graph.operations[1]
        assert gate_op.gate_type == expected_type
        assert gate_op.theta.get_const() == pytest.approx(0.5)
        assert len(gate_op.qubit_operands) == 1
        assert len(gate_op.results) == 1

    @pytest.mark.parametrize("gate_fn, expected_type", ALL_GATES, ids=IDS)
    def test_gate_ir_qubits(self, gate_fn, expected_type):
        @qkernel
        def circuit(qs: qm.Vector[Qubit]) -> qm.Vector[Qubit]:
            qs[0] = gate_fn(qs[0], 0.5)
            return qs

        graph = circuit.build()
        assert len(graph.operations) == 2
        assert isinstance(graph.operations[0], QInitOperation)
        assert isinstance(graph.operations[1], GateOperation)
        gate_op = graph.operations[1]
        assert gate_op.gate_type == expected_type
        assert gate_op.theta.get_const() == pytest.approx(0.5)
        assert len(gate_op.qubit_operands) == 1
        assert len(gate_op.results) == 1


class TestTwoQubitGates:
    """Non-parameterized two-qubit gates."""

    GATE_TEMPLATE = "q0, q1 = qm.{name}(q0, q1)"
    N_QUBITS = 2

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
        assert len(gate_op.qubit_operands) == 2
        assert len(gate_op.results) == 2

    @pytest.mark.parametrize("seed", [901 + offset for offset in range(50)])
    def test_randomized_gates(self, seed):
        """Test random sequences of non-parameterized two-qubit gates."""
        graph, expected_types, num_gates, _ = _build_random_gate_circuit(
            self.ALL_GATES,
            self.GATE_TEMPLATE,
            self.N_QUBITS,
            seed,
        )
        assert len(graph.operations) == self.N_QUBITS + num_gates
        for op in graph.operations[: self.N_QUBITS]:
            assert isinstance(op, QInitOperation)
        for op, expected_type in zip(
            graph.operations[self.N_QUBITS :], expected_types, strict=True
        ):
            assert isinstance(op, GateOperation)
            assert op.gate_type == expected_type
            assert op.theta is None
            assert len(op.qubit_operands) == self.N_QUBITS
            assert len(op.results) == self.N_QUBITS

    @pytest.mark.parametrize("gate_fn, expected_type", ALL_GATES, ids=IDS)
    @pytest.mark.parametrize("actual_n", [2, 5, 10])
    def test_gate_ir_given_n(self, gate_fn, expected_type, actual_n):
        @qkernel
        def circuit(n: qm.UInt) -> qm.Vector[Qubit]:
            qs = qm.qubit_array(n, "qs")
            qs[0], qs[1] = gate_fn(qs[0], qs[1])
            return qs

        graph = circuit.build(n=actual_n)
        assert len(graph.operations) == 2
        assert isinstance(graph.operations[0], QInitOperation)
        assert isinstance(graph.operations[1], GateOperation)
        gate_op = graph.operations[1]
        assert gate_op.gate_type == expected_type
        assert gate_op.theta is None
        assert len(gate_op.qubit_operands) == 2
        assert len(gate_op.results) == 2

    @pytest.mark.parametrize("gate_fn, expected_type", ALL_GATES, ids=IDS)
    def test_gate_ir_qubits(self, gate_fn, expected_type):
        @qkernel
        def circuit(qs: qm.Vector[Qubit]) -> qm.Vector[Qubit]:
            qs[0], qs[1] = gate_fn(qs[0], qs[1])
            return qs

        graph = circuit.build()
        assert len(graph.operations) == 2
        assert isinstance(graph.operations[0], QInitOperation)
        assert isinstance(graph.operations[1], GateOperation)
        gate_op = graph.operations[1]
        assert gate_op.gate_type == expected_type
        assert gate_op.theta is None
        assert len(gate_op.qubit_operands) == 2
        assert len(gate_op.results) == 2


class TestTwoQubitParamGates:
    """Parameterized two-qubit gates."""

    GATE_TEMPLATE = "q0, q1 = qm.{name}(q0, q1, {theta})"
    N_QUBITS = 2

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
        assert gate_op.theta.get_const() == pytest.approx(0.5)
        assert len(gate_op.qubit_operands) == 2
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
        assert len(gate_op.qubit_operands) == 2
        assert len(gate_op.results) == 2

    @pytest.mark.parametrize("seed", [901 + offset for offset in range(50)])
    def test_randomized_gates_literal(self, seed):
        """Test random sequences of parameterized two-qubit gates with literal theta."""
        graph, expected_types, num_gates, thetas = _build_random_gate_circuit(
            self.ALL_GATES,
            self.GATE_TEMPLATE,
            self.N_QUBITS,
            seed,
            theta="literal",
        )
        assert thetas is not None
        assert len(graph.operations) == self.N_QUBITS + num_gates
        for op in graph.operations[: self.N_QUBITS]:
            assert isinstance(op, QInitOperation)
        for op, expected_type, expected_theta in zip(
            graph.operations[self.N_QUBITS :], expected_types, thetas, strict=True
        ):
            assert isinstance(op, GateOperation)
            assert op.gate_type == expected_type
            assert op.theta.get_const() == pytest.approx(expected_theta)
            assert len(op.qubit_operands) == self.N_QUBITS
            assert len(op.results) == self.N_QUBITS

    @pytest.mark.parametrize("seed", [901 + offset for offset in range(50)])
    def test_randomized_gates_symbolic(self, seed):
        """Test random sequences of parameterized two-qubit gates with symbolic theta."""
        graph, expected_types, num_gates, _ = _build_random_gate_circuit(
            self.ALL_GATES,
            self.GATE_TEMPLATE,
            self.N_QUBITS,
            seed,
            theta="symbolic",
        )
        assert len(graph.operations) == self.N_QUBITS + num_gates
        for op in graph.operations[: self.N_QUBITS]:
            assert isinstance(op, QInitOperation)
        for op, expected_type in zip(
            graph.operations[self.N_QUBITS :], expected_types, strict=True
        ):
            assert isinstance(op, GateOperation)
            assert op.gate_type == expected_type
            assert isinstance(op.theta, Value)
            assert len(op.qubit_operands) == self.N_QUBITS
            assert len(op.results) == self.N_QUBITS

    @pytest.mark.parametrize("gate_fn, expected_type", ALL_GATES, ids=IDS)
    @pytest.mark.parametrize("actual_n", [2, 5, 10])
    def test_gate_ir_given_n(self, gate_fn, expected_type, actual_n):
        @qkernel
        def circuit(n: qm.UInt) -> qm.Vector[Qubit]:
            qs = qm.qubit_array(n, "qs")
            qs[0], qs[1] = gate_fn(qs[0], qs[1], 0.5)
            return qs

        graph = circuit.build(n=actual_n)
        assert len(graph.operations) == 2
        assert isinstance(graph.operations[0], QInitOperation)
        assert isinstance(graph.operations[1], GateOperation)
        gate_op = graph.operations[1]
        assert gate_op.gate_type == expected_type
        assert gate_op.theta.get_const() == pytest.approx(0.5)
        assert len(gate_op.qubit_operands) == 2
        assert len(gate_op.results) == 2

    @pytest.mark.parametrize("gate_fn, expected_type", ALL_GATES, ids=IDS)
    def test_gate_ir_qubits(self, gate_fn, expected_type):
        @qkernel
        def circuit(qs: qm.Vector[Qubit]) -> qm.Vector[Qubit]:
            qs[0], qs[1] = gate_fn(qs[0], qs[1], 0.5)
            return qs

        graph = circuit.build()
        assert len(graph.operations) == 2
        assert isinstance(graph.operations[0], QInitOperation)
        assert isinstance(graph.operations[1], GateOperation)
        gate_op = graph.operations[1]
        assert gate_op.gate_type == expected_type
        assert gate_op.theta.get_const() == pytest.approx(0.5)
        assert len(gate_op.qubit_operands) == 2
        assert len(gate_op.results) == 2


class TestThreeQubitGates:
    """ccx — three-qubit gate."""

    GATE_TEMPLATE = "q0, q1, q2 = qm.{name}(q0, q1, q2)"
    N_QUBITS = 3

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
        assert len(gate_op.qubit_operands) == 3
        assert len(gate_op.results) == 3

    @pytest.mark.parametrize("seed", [901 + offset for offset in range(50)])
    def test_randomized_gates(self, seed):
        """Test random sequences of three-qubit gates."""
        graph, expected_types, num_gates, _ = _build_random_gate_circuit(
            self.ALL_GATES,
            self.GATE_TEMPLATE,
            self.N_QUBITS,
            seed,
        )
        assert len(graph.operations) == self.N_QUBITS + num_gates
        for op in graph.operations[: self.N_QUBITS]:
            assert isinstance(op, QInitOperation)
        for op, expected_type in zip(
            graph.operations[self.N_QUBITS :], expected_types, strict=True
        ):
            assert isinstance(op, GateOperation)
            assert op.gate_type == expected_type
            assert op.theta is None
            assert len(op.qubit_operands) == self.N_QUBITS
            assert len(op.results) == self.N_QUBITS

    @pytest.mark.parametrize("gate_fn, expected_type", ALL_GATES, ids=IDS)
    @pytest.mark.parametrize("actual_n", [3, 5, 10])
    def test_gate_ir_given_n(self, gate_fn, expected_type, actual_n):
        @qkernel
        def circuit(n: qm.UInt) -> qm.Vector[Qubit]:
            qs = qm.qubit_array(n, "qs")
            qs[0], qs[1], qs[2] = gate_fn(qs[0], qs[1], qs[2])
            return qs

        graph = circuit.build(n=actual_n)
        assert len(graph.operations) == 2
        assert isinstance(graph.operations[0], QInitOperation)
        assert isinstance(graph.operations[1], GateOperation)
        gate_op = graph.operations[1]
        assert gate_op.gate_type == expected_type
        assert gate_op.theta is None
        assert len(gate_op.qubit_operands) == 3
        assert len(gate_op.results) == 3

    @pytest.mark.parametrize("gate_fn, expected_type", ALL_GATES, ids=IDS)
    def test_gate_ir_qubits(self, gate_fn, expected_type):
        @qkernel
        def circuit(qs: qm.Vector[Qubit]) -> qm.Vector[Qubit]:
            qs[0], qs[1], qs[2] = gate_fn(qs[0], qs[1], qs[2])
            return qs

        graph = circuit.build()
        assert len(graph.operations) == 2
        assert isinstance(graph.operations[0], QInitOperation)
        assert isinstance(graph.operations[1], GateOperation)
        gate_op = graph.operations[1]
        assert gate_op.gate_type == expected_type
        assert gate_op.theta is None
        assert len(gate_op.qubit_operands) == 3
        assert len(gate_op.results) == 3


class TestAllGates:
    """Test that all gates can be used together in a single circuit."""

    CATEGORIZED_GATES = [
        (
            TestSingleQubitGates.ALL_GATES_WITH_IDS,
            TestSingleQubitGates.GATE_TEMPLATE,
            None,
        ),
        (TestRotationGates.ALL_GATES_WITH_IDS, TestRotationGates.GATE_TEMPLATE, 0.5),
        (TestTwoQubitGates.ALL_GATES_WITH_IDS, TestTwoQubitGates.GATE_TEMPLATE, None),
        (
            TestTwoQubitParamGates.ALL_GATES_WITH_IDS,
            TestTwoQubitParamGates.GATE_TEMPLATE,
            0.5,
        ),
        (
            TestThreeQubitGates.ALL_GATES_WITH_IDS,
            TestThreeQubitGates.GATE_TEMPLATE,
            None,
        ),
    ]

    ALL_GATES = [
        gate_info for gates, _, _ in CATEGORIZED_GATES for gate_info, _ in gates
    ]
    ALL_GATES_WITH_TEMPLATES = [
        (gate_info, template, theta)
        for gates, template, theta in CATEGORIZED_GATES
        for gate_info, _ in gates
    ]

    def test_all_gates_in_one_circuit(self):
        """Build a single @qkernel with every gate, verify flat GateOperations."""
        body_lines = []
        for gates, template, theta in self.CATEGORIZED_GATES:
            for (gate_fn, _), _ in gates:
                fmt = {"name": gate_fn.__name__, "i": 0}
                if theta is not None:
                    fmt["theta"] = str(theta)
                body_lines.append("    " + template.format(**fmt))

        func_source = (
            "def _circuit(q0: Qubit, q1: Qubit, q2: Qubit)"
            " -> tuple[Qubit, Qubit, Qubit]:\n"
            + "\n".join(body_lines)
            + "\n    return q0, q1, q2\n"
        )

        circuit = _build_qkernel(func_source, "<all_gates_circuit>")

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

    @pytest.mark.parametrize("seed", [901 + offset for offset in range(50)])
    def test_randomized_all_gates(self, seed):
        """Test random mix of gates from all categories in one circuit."""
        rng = np.random.default_rng(seed)
        num_gates = rng.integers(2, 100)
        indices = rng.integers(0, len(self.ALL_GATES_WITH_TEMPLATES), size=num_gates)
        chosen = [self.ALL_GATES_WITH_TEMPLATES[i] for i in indices]

        body_lines = []
        expected_thetas = []
        for idx, ((fn, _), template, theta) in enumerate(chosen):
            fmt = {"name": fn.__name__, "i": idx}
            if theta is not None:
                literal_theta = rng.uniform(0, 2 * np.pi)
                fmt["theta"] = str(literal_theta)
                expected_thetas.append(literal_theta)
            else:
                expected_thetas.append(None)
            body_lines.append("    " + template.format(**fmt))

        func_source = (
            "def _circuit(q0: Qubit, q1: Qubit, q2: Qubit)"
            " -> tuple[Qubit, Qubit, Qubit]:\n"
            + "\n".join(body_lines)
            + "\n    return q0, q1, q2\n"
        )
        circuit = _build_qkernel(func_source, f"<randomized_all_gates_{seed}>")
        graph = circuit.build()

        n_qinits = 3
        assert len(graph.operations) == n_qinits + num_gates
        expected_types = [gt for (_, gt), _, _ in chosen]
        for op in graph.operations[:n_qinits]:
            assert isinstance(op, QInitOperation)
        for op, expected_type, expected_theta in zip(
            graph.operations[n_qinits:], expected_types, expected_thetas, strict=True
        ):
            assert isinstance(op, GateOperation)
            assert op.gate_type == expected_type
            if expected_theta is None:
                assert op.theta is None
            else:
                assert op.theta.get_const() == pytest.approx(expected_theta)
