"""Bounded witnesses for exact, delayed Qiskit realization."""

from __future__ import annotations

import numpy as np
import pytest

import qamomile as qm
from qamomile.circuit import lower_to_circuit
from qamomile.persistence import deserialize, serialize

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import AnnotatedOperation
from qiskit.circuit.library import (
    BitFlipOracleGate,
    LinearFunction,
    ModularAdderGate,
    PauliProductMeasurement,
    PhaseOracleGate,
)
from qiskit.quantum_info import Operator, Pauli

from qamomile.qiskit import Aer, Qiskit, load_circuit, to_circuit
from qamomile.qiskit._emission import emit_circuit
from qamomile.qiskit._payload import QiskitExecutable


def _inspect(source: str, name: str = "state"):
    """Inspect the same source through retained and portable circuit paths."""
    program = qm.build_source("import qamomile as qm\n" + source).specialize(name)
    circuit = to_circuit(program)
    portable = emit_circuit(lower_to_circuit(program))
    return program, circuit, portable


def _operator(circuit):
    """Compare all input states without the SDK's initial-zero assumption."""
    from qiskit.transpiler.passes import UnrollForLoops

    circuit = UnrollForLoops()(circuit)
    return Operator(
        transpile(
            circuit,
            basis_gates=["u", "cx"],
            optimization_level=0,
            qubits_initially_zero=False,
        )
    ).data


def _structured_names(circuit):
    """Inspect control-flow structure without forcing lazy gate definitions."""
    for row in circuit:
        yield row.operation.name
        for block in getattr(row.operation, "blocks", ()):
            yield from _structured_names(block)


def _qpy_roundtrip(circuit):
    """Round-trip an inspected circuit using SDK storage, not an executable ABI."""
    return QiskitExecutable.from_bytes(QiskitExecutable(circuit).to_bytes()).circuit()


def _assert_exact(program, circuit, portable):
    """Check full complex matrices and explicit QPY round trips."""
    expected = _operator(portable)
    np.testing.assert_allclose(_operator(circuit), expected, atol=2e-12)
    np.testing.assert_allclose(_operator(_qpy_roundtrip(circuit)), expected, atol=2e-12)
    restored_source = qm.Program.from_bytes(program.to_bytes())
    np.testing.assert_allclose(
        _operator(to_circuit(restored_source)), expected, atol=2e-12
    )


@pytest.mark.parametrize(
    "body,width",
    [
        ("qm.add_into(state)", 4),
        ("qm.h(state)", 1),
        ("qm.ry(state, 0.37)", 1),
        ("qm.rzz(state, 0.37)", 2),
        ("qm.swap(state)", 2),
        ("qm.global_phase(state, 0.37)", 1),
    ],
)
def test_control_and_inverse_preserve_every_sector(body, width):
    """Coherent control includes inactive inputs, dirty targets, and phase."""
    source = f"""
@qm.qkernel
def operation(state: qm.Q[qm.Bits[{width}]]) -> qm.Q[qm.Bits[{width}]]:
    return {body}

@qm.qkernel
def state() -> qm.Q[qm.Bits[{width + 2}]]:
    controls, target = qm.split_at(qm.qubits({width + 2}), 2)
    controls, target = qm.control(operation, controls, target)
    return qm.concat(controls, target)
"""
    program, circuit, portable = _inspect(source)
    assert any(isinstance(row.operation, AnnotatedOperation) for row in circuit)
    _assert_exact(program, circuit, portable)


def test_subtraction_does_not_force_adder_definition():
    """An inverse modifier leaves the SDK's modular adder unsynthesized."""
    program, circuit, portable = _inspect("""
@qm.qkernel
def state() -> qm.Q[qm.Bits[4]]:
    left, right = qm.split_at(qm.qubits(4), 2)
    left, right = qm.as_uint(left) - qm.as_uint(right)
    return qm.concat(qm.as_bits(left), qm.as_bits(right))
""")
    operation = circuit.data[0].operation
    assert isinstance(operation, AnnotatedOperation)
    assert isinstance(operation.base_op, ModularAdderGate)
    assert operation.base_op._definition is None
    _assert_exact(program, circuit, portable)


def test_named_box_uses_only_local_operands_and_reuses_definition(monkeypatch):
    """Repeated calls on different wires build one narrow SDK definition."""
    original = QuantumCircuit.to_gate
    labels = []

    def counted(self, *args, **kwargs):
        labels.append(kwargs.get("label"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(QuantumCircuit, "to_gate", counted)
    program, circuit, portable = _inspect("""
@qm.composite_gate(name="pair")
@qm.qkernel
def pair(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    return qm.cx(qm.h(state, target=0), control=0, target=1)

@qm.qkernel
def state() -> qm.Q[qm.Bits[6]]:
    a, rest = qm.split_at(qm.qubits(6), 2)
    b, c = qm.split_at(rest, 2)
    a = pair(a)
    b = pair(b)
    return qm.concat(a, qm.concat(b, c))
""")
    assert [row.operation.num_qubits for row in circuit] == [2, 2]
    assert not labels
    left, right = (row.operation for row in circuit)
    assert left._owner.identity == right._owner.identity
    assert left._definition is None and right._definition is None
    _assert_exact(program, circuit, portable)


@pytest.mark.parametrize("kind", ["phase", "xor", "linear", "nonlinear", "multi"])
def test_boolean_native_gates_match_all_basis_states(kind):
    """SDK Boolean operations retain output ordering and arbitrary targets."""
    definitions = {
        "phase": (
            "qm.Bool",
            "not bool(bits[0]) and bool(bits[1])",
            "qm.phase_by(predicate, state)",
            2,
        ),
        "xor": (
            "qm.Bool",
            "bool(bits[0]) ^ bool(bits[1])",
            "qm.concat(*qm.xor_by(predicate, *qm.split_at(state, 2)))",
            3,
        ),
        "linear": (
            "tuple[qm.Bool, qm.Bool]",
            "(bool(bits[0]) ^ bool(bits[1]), bool(bits[1]))",
            "qm.permute_by(predicate, state)",
            2,
        ),
        "nonlinear": (
            "tuple[qm.Bool, qm.Bool]",
            "(not bool(bits[0]), bool(bits[1]))",
            "qm.permute_by(predicate, state)",
            2,
        ),
        "multi": (
            "tuple[qm.Bool, qm.Bool]",
            "(bool(bits[0]) and bool(bits[1]), bool(bits[0]) ^ bool(bits[1]))",
            "unused",
            4,
        ),
    }
    result, expression, operation, width = definitions[kind]
    if kind in {"xor", "multi"}:
        body = "source, target = qm.split_at(state, 2)\n    source, target = qm.xor_by(predicate, source, target)\n    return qm.concat(source, target)"
    else:
        body = f"return {operation}"
    program, circuit, portable = _inspect(f"""
@qm.qkernel
def predicate(bits: qm.Bits[2]) -> {result}:
    return {expression}

@qm.qkernel
def state() -> qm.Q[qm.Bits[{width}]]:
    state = qm.qubits({width})
    {body}
""")
    expected = {
        "phase": PhaseOracleGate,
        "xor": BitFlipOracleGate,
        "linear": LinearFunction,
    }.get(kind)
    if expected:
        assert isinstance(circuit.data[0].operation, expected)
    _assert_exact(program, circuit, portable)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_axis_measurements_preserve_kraus_operators_and_repeatability(axis):
    """Retained Pauli measurement is non-destructive and keeps its ABI bit."""
    program, circuit, _ = _inspect(f"""
@qm.qkernel
def state() -> tuple[qm.Bool, qm.Bool]:
    state = qm.ry(qm.qubits(1), 0.7)
    first, state = qm.measure_{axis}(state)
    second, state = qm.measure_{axis}(state)
    qm.discard(state)
    return first, second
""")
    operations = [
        row.operation
        for row in circuit
        if isinstance(row.operation, PauliProductMeasurement)
    ]
    assert len(operations) == 2
    definition = operations[0].definition
    before, after = np.eye(2, dtype=complex), np.eye(2, dtype=complex)
    measured = False
    for row in definition:
        if row.operation.name == "measure":
            measured = True
        elif measured:
            after = Operator(row.operation).data @ after
        else:
            before = Operator(row.operation).data @ before
    for bit in (0, 1):
        projector = np.zeros((2, 2))
        projector[bit, bit] = 1
        expected = (np.eye(2) + (-1) ** bit * Pauli(axis.upper()).to_matrix()) / 2
        np.testing.assert_allclose(after @ projector @ before, expected, atol=1e-15)
    artifact = qm.compile(program, to=Qiskit())
    for current in (artifact, deserialize(serialize(artifact))):
        result = qm.run(current, on=Aer(), shots=16, seed=7).result
        assert set(result.counts) <= {(0, 0), (1, 1)}


@pytest.mark.parametrize(
    "body,width",
    [("state = qm.rx(state, 0.1 * index)", 1), ("state = qm.add_into(state)", 4)],
)
def test_loops_retain_index_and_composite_body(body, width):
    """Index angles and composite operations use one structured SDK loop."""
    program, circuit, portable = _inspect(f"""
@qm.qkernel
def state() -> qm.Q[qm.Bits[{width}]]:
    state = qm.qubits({width})
    for index in range(3):
        {body}
    return state
""")
    assert circuit.count_ops() == {"for_loop": 1}
    from qiskit_aer import AerSimulator

    # Cancel the portable unitary after running the loop from a nonzero input.
    for retained in (
        circuit,
        _qpy_roundtrip(circuit),
    ):
        test = QuantumCircuit(width, width)
        for wire in range(width):
            test.ry(0.2 + 0.1 * wire, wire)
        prefix = test.remove_final_measurements(inplace=False)
        test.compose(retained, inplace=True)
        test.compose(portable.inverse(), inplace=True)
        test.compose(prefix.inverse(), inplace=True)
        test.measure(range(width), range(width))
        backend = AerSimulator(max_parallel_threads=1)
        result = backend.run(
            transpile(test, backend, optimization_level=0), shots=16
        ).result()
        assert result.get_counts() == {"0" * width: 16}


def test_measurement_xor_uses_sdk_classical_ast():
    """A measured parity can directly control a device operation."""
    source = """
@qm.qkernel
def state() -> tuple[qm.Bool, qm.Bool, qm.Bits[1]]:
    a = bool(qm.measure(qm.h(qm.qubits(1)))[0])
    b = bool(qm.measure(qm.h(qm.qubits(1)))[0])
    target = qm.qubits(1)
    if a ^ b:
        target = qm.x(target)
    return a, b, qm.measure(target)
"""
    program = qm.build_source("import qamomile as qm\n" + source).specialize("state")
    result = qm.run(program, on=Aer(), shots=32, seed=4).result
    assert all(row[2] == row[0] ^ row[1] for row in result.counts)


def test_aer_preparation_obeys_policy_and_reuses_template(monkeypatch):
    """Endpoint options belong in both synthesis and its bounded cache key."""
    import qiskit
    from qiskit_aer import AerSimulator

    original = qiskit.transpile
    options = []

    def counted(*args, **kwargs):
        options.append((kwargs["optimization_level"], kwargs["seed_transpiler"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(qiskit, "transpile", counted)
    owner = QiskitExecutable(QuantumCircuit(1))
    backend = AerSimulator(max_parallel_threads=1)
    first = owner.aer_circuit(backend, "automatic", 1, 7)
    first.x(0)
    assert not owner.aer_circuit(backend, "automatic", 1, 7).data
    owner.aer_circuit(backend, "automatic", 2, 9)
    assert options == [(1, 7), (2, 9)]


@pytest.mark.parametrize("expression", ["True", "False", "not bool(bits[0])"])
def test_boolean_constants_and_unused_inputs_keep_controlled_phase(expression):
    """Constant terms, unused wires, negative control, and dagger remain exact."""
    program, circuit, portable = _inspect(f"""
@qm.qkernel
def predicate(bits: qm.Bits[2]) -> qm.Bool:
    return {expression}

@qm.qkernel
def oracle(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    return qm.phase_by(predicate, state)

@qm.qkernel
def inverse(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    return qm.adjoint(oracle, state)

@qm.qkernel
def state() -> qm.Q[qm.Bits[4]]:
    controls, target = qm.split_at(qm.qubits(4), 2)
    controls, target = qm.control(inverse, controls, target, control_state=1)
    return qm.concat(controls, target)
""")
    _assert_exact(program, circuit, portable)


@pytest.mark.parametrize(
    "operation",
    [
        "value + 2",
        "value - 2",
        "qm.mod_add(value, 2, 3)",
        "qm.mod_mul(value, factor=2, modulus=3)",
    ],
)
@pytest.mark.parametrize("inverse", [False, True])
def test_arithmetic_composites_preserve_invalid_residues(operation, inverse):
    """Native wrapping addition cannot replace total arbitrary-modulus arithmetic."""
    call = "qm.adjoint(apply, value)" if inverse else "apply(value)"
    program, circuit, portable = _inspect(f"""
@qm.qkernel
def apply(value: qm.Q[qm.UInt[2]]) -> qm.Q[qm.UInt[2]]:
    return {operation}

@qm.qkernel
def state() -> qm.Q[qm.Bits[2]]:
    value = qm.as_uint(qm.qubits(2))
    value = {call}
    return qm.as_bits(value)
""")
    assert len(circuit.data) == 1
    _assert_exact(program, circuit, portable)


def test_large_boolean_uses_bounded_anf_instead_of_sdk_truth_table():
    """Nine source bits already bypass the exponential SDK oracle synthesizer."""
    program, circuit, _ = _inspect("""
@qm.qkernel
def predicate(bits: qm.Bits[9]) -> qm.Bool:
    return bool(bits[8])

@qm.qkernel
def state() -> qm.Q[qm.Bits[9]]:
    return qm.phase_by(predicate, qm.qubits(9))
""")
    assert not isinstance(circuit.data[0].operation, PhaseOracleGate)
    assert circuit.data[0].operation.definition.count_ops() == {"z": 1}
    np.testing.assert_allclose(_operator(_qpy_roundtrip(circuit)), _operator(circuit))


def test_large_loop_is_structural_and_launch_bindings_remain_separate():
    """Loop indices do not become launch arguments and bindings stay immutable."""
    program = qm.build_source("""
import qamomile as qm
@qm.qkernel
def state(theta: qm.Param[qm.Real]) -> qm.Bits[1]:
    state = qm.qubits(1)
    for index in range(3):
        state = qm.rx(state, theta * index)
    return qm.measure(state)
""").specialize("state")
    artifact = qm.compile(program, to=Qiskit())
    circuit = load_circuit(artifact)
    assert circuit.count_ops()["for_loop"] == 1
    assert {
        p.name for p in circuit.parameters if p != circuit.data[0].operation.params[1]
    } == {"p0"}
    for current in (artifact, deserialize(serialize(artifact))):
        for angle, bit in [(0.0, 0), (np.pi / 3, 1), (0.0, 0)]:
            assert qm.run(
                current, on=Aer(), parameters={"theta": angle}, shots=4
            ).result.counts == {(bit,): 4}
    source = """
@qm.qkernel
def state() -> qm.Q[qm.Bits[1]]:
    state = qm.qubits(1)
    for index in range(100000):
        state = qm.rx(state, 0.1 * index)
    return state
"""
    program = qm.build_source("import qamomile as qm\n" + source).specialize("state")
    assert to_circuit(program, target=Qiskit()).count_ops() == {"for_loop": 1}


@pytest.mark.parametrize(
    "body",
    [
        "state = qm.x(state, target=index)",
        "state = qm.rx(state, 0.1 * ((index + 1) - index))",
    ],
)
def test_loop_fallback_preserves_selector_and_integer_meaning(body):
    """An unsupported scoped index falls back without leaked instructions."""
    program, circuit, portable = _inspect(f"""
@qm.qkernel
def state() -> qm.Q[qm.Bits[2]]:
    state = qm.qubits(2)
    for index in range(2):
        {body}
    return state
""")
    assert "for_loop" not in circuit.count_ops()
    _assert_exact(program, circuit, portable)


def test_loop_does_not_reuse_discarded_allocations():
    """A fresh ancilla in each call must not silently become a reused wire."""
    program, circuit, portable = _inspect("""
@qm.qkernel
def helper(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    extra = qm.qubits(1)
    joined = qm.concat(state, extra)
    joined = qm.cx(joined, control=1, target=0)
    joined = qm.x(joined, target=1)
    state, extra = qm.split_at(joined, 1)
    qm.discard(extra)
    return state

@qm.qkernel
def state() -> qm.Q[qm.Bits[1]]:
    state = qm.qubits(1)
    for index in range(2):
        state = helper(state)
    return state
""")
    assert "for_loop" not in circuit.count_ops()
    assert circuit.num_qubits == portable.num_qubits == 3
    _assert_exact(program, circuit, portable)


def test_scoped_loop_index_is_free_if_it_escapes_the_body():
    """Only lexical binding excludes a parameter from launch validation."""
    from qiskit.circuit import Parameter

    from qamomile.qiskit._integration import _free_circuit_parameters

    index = Parameter("_loop_i_0")
    circuit = QuantumCircuit(1)
    with circuit.for_loop(range(2), index):
        circuit.rx(index, 0)
    assert not _free_circuit_parameters(circuit)
    circuit.rz(index, 0)
    assert _free_circuit_parameters(circuit) == {index}


@pytest.mark.parametrize("opcode,width", [("x", 4), ("swap", 5)])
def test_modular_primitives_retain_native_control_and_exact_permutation(opcode, width):
    """An MCX or controlled SWAP is one operation before endpoint synthesis."""
    from qiskit.circuit.library import MCXGate

    from qamomile.circuit import CircuitInstruction
    from qamomile.compilation import RealizationCapability
    from qamomile.qiskit._emission import _emit_instruction

    assert {
        RealizationCapability.MCX,
        RealizationCapability.CONTROLLED_SWAP,
    } <= Qiskit().capabilities.realization_capabilities
    circuit = QuantumCircuit(width)
    _emit_instruction(
        circuit,
        CircuitInstruction(opcode, tuple(range(3, width)), controls=(0, 1, 2)),
        {},
    )
    assert len(circuit.data) == 1
    assert isinstance(
        circuit.data[0].operation, MCXGate if opcode == "x" else AnnotatedOperation
    )
    expected = np.zeros((1 << width, 1 << width), dtype=complex)
    for source in range(1 << width):
        destination = source
        if source & 7 == 7:
            if opcode == "x":
                destination ^= 1 << 3
            elif ((source >> 3) ^ (source >> 4)) & 1:
                destination ^= (1 << 3) | (1 << 4)
        expected[destination, source] = 1
    np.testing.assert_allclose(_operator(circuit), expected, atol=1e-12)


def test_plain_and_controlled_calls_keep_structured_loops():
    """A call boundary must not undo structured-loop retention."""
    source = """
import qamomile as qm
@qm.qkernel
def repeated(state: qm.Q[qm.Bits[1]], count: qm.Const[qm.Int]) -> qm.Q[qm.Bits[1]]:
    for index in range(count):
        state = qm.rx(state, 0.1 * index)
    return state

@qm.qkernel
def state() -> qm.Q[qm.Bits[1]]:
    return repeated(qm.qubits(1), 100000)
"""
    program = qm.build_source(source).specialize("state")
    circuit = to_circuit(program, target=Qiskit())
    assert circuit.count_ops() == {"for_loop": 1}
    small = source.replace("100000", "3")
    program, circuit, portable = _inspect(
        small.removeprefix("\nimport qamomile as qm\n")
    )
    assert circuit.count_ops() == {"for_loop": 1}
    # The same call under coherent control retains its exact full unitary.
    controlled = (
        small[: small.index("@qm.qkernel\ndef state()")]
        + """
@qm.qkernel
def state() -> qm.Q[qm.Bits[2]]:
    control, target = qm.split_at(qm.qubits(2), 1)
    control, target = qm.control(repeated, control, target, 3)
    return qm.concat(control, target)
"""
    )
    program, circuit, portable = _inspect(
        controlled.removeprefix("\nimport qamomile as qm\n")
    )
    _assert_exact(program, circuit, portable)


@pytest.mark.parametrize("transform", ["control", "adjoint", "both"])
@pytest.mark.parametrize("bounds", ["1, 6, 2", "5, -1, -2"])
def test_transformed_loops_keep_order_phase_and_structure(transform, bounds):
    """Structural transforms retain index values and the full controlled phase."""
    if transform == "control":
        invoke = "control, target = qm.control(repeated, control, target)"
    elif transform == "adjoint":
        invoke = "target = qm.adjoint(repeated, target)"
    else:
        invoke = "control, target = qm.control(backward, control, target)"
    program, circuit, portable = _inspect(f"""
@qm.qkernel
def repeated(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    for index in range({bounds}):
        state = qm.rx(state, 0.13 * index)
        state = qm.rz(state, 0.21 * index)
        state = qm.global_phase(state, 0.07 * index)
    return state

@qm.qkernel
def backward(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.adjoint(repeated, state)

@qm.qkernel
def state() -> qm.Q[qm.Bits[2]]:
    control, target = qm.split_at(qm.qubits(2), 1)
    {invoke}
    return qm.concat(control, target)
""")
    assert circuit.count_ops() == {"for_loop": 1}
    _assert_exact(program, circuit, portable)


def test_deferred_parameter_bindings_do_not_mutate_the_template():
    """Binding lazy gates retains native ownership and independent SDK copies."""
    from qamomile.qiskit._deferred import DeferredGate

    program, circuit, portable = _inspect("""
@qm.qkernel
def rotate(state: qm.Q[qm.Bits[1]], angle: qm.Param[qm.Real]) -> qm.Q[qm.Bits[1]]:
    return qm.rz(qm.rx(state, angle), 2.0 * angle)

@qm.qkernel
def state(angle: qm.Param[qm.Real]) -> qm.Q[qm.Bits[1]]:
    return rotate(qm.qubits(1), angle)
""")
    assert isinstance(circuit.data[0].operation, DeferredGate)
    assert circuit.data[0].operation._definition is None
    for value in [0.12, 0.73, -0.21]:
        bound = circuit.assign_parameters({next(iter(circuit.parameters)): value})
        reference = portable.assign_parameters({next(iter(portable.parameters)): value})
        np.testing.assert_allclose(_operator(bound), _operator(reference), atol=2e-12)
    assert circuit.data[0].operation._definition is None
    restored = _qpy_roundtrip(circuit)
    np.testing.assert_allclose(
        _operator(restored.assign_parameters({next(iter(restored.parameters)): 0.37})),
        _operator(portable.assign_parameters({next(iter(portable.parameters)): 0.37})),
        atol=2e-12,
    )


@pytest.mark.parametrize("bits", range(8))
@pytest.mark.parametrize("return_selection", [False, True])
def test_reused_boolean_select_uses_one_top_level_store(bits, return_selection):
    """Every select truth-table row survives private storage, shots, and export."""
    prepared = [
        "qm.x(qm.qubits(1))" if bits & (1 << i) else "qm.qubits(1)" for i in range(3)
    ]
    result_type = (
        "tuple[qm.Bool, qm.Bits[1], qm.Bits[1]]"
        if return_selection
        else "tuple[qm.Bits[1], qm.Bits[1]]"
    )
    result_values = (
        "selected, qm.measure(left), qm.measure(right)"
        if return_selection
        else "qm.measure(left), qm.measure(right)"
    )
    program = qm.build_source(f"""
import qamomile as qm
@qm.qkernel
def state() -> {result_type}:
    c = bool(qm.measure({prepared[0]})[0])
    a = bool(qm.measure({prepared[1]})[0])
    b = bool(qm.measure({prepared[2]})[0])
    selected = a if c else b
    left = qm.qubits(1)
    right = qm.qubits(1)
    if selected:
        left = qm.x(left)
    if selected:
        right = qm.x(right)
    return {result_values}
""").specialize("state")
    endpoint = Aer(use_classical_stores=True)
    artifact = qm.compile(program, to=endpoint.target)
    circuit = to_circuit(program, target=endpoint.target)
    assert circuit.count_ops().get("store") == 1
    assert to_circuit(program, target=Qiskit()).count_ops().get("store", 0) == 0
    expected = bool(bits & 2) if bits & 1 else bool(bits & 4)
    for executable in [artifact, deserialize(serialize(artifact))]:
        for seed in [1, 9]:
            result = qm.run(executable, shots=8, seed=seed, on=endpoint).result
            assert result.counts == {
                (int(expected),) * (3 if return_selection else 2): 8
            }


def test_deferred_modular_power_matches_all_input_states():
    """Modular recipes preserve invalid residues and exponent register ordering."""
    program, circuit, portable = _inspect("""
@qm.qkernel
def state() -> qm.Q[qm.Bits[3]]:
    target, exponent = qm.split_at(qm.qubits(3), 2)
    target, exponent = qm.mod_pow(qm.as_uint(target), qm.as_uint(exponent), base=2, modulus=3)
    return qm.concat(qm.as_bits(target), qm.as_bits(exponent))
""")
    assert len(circuit.data) == 1
    assert circuit.data[0].operation._definition is None
    _assert_exact(program, circuit, portable)


def test_nested_select_does_not_insert_branch_local_stores():
    """The conservative profile never relies on Aer's nested Store behavior."""
    program = qm.build_source("""
import qamomile as qm
@qm.qkernel
def state() -> tuple[qm.Bool, qm.Bool, qm.Bool, qm.Bits[1], qm.Bits[1]]:
    c = bool(qm.measure(qm.h(qm.qubits(1)))[0])
    a = bool(qm.measure(qm.h(qm.qubits(1)))[0])
    b = bool(qm.measure(qm.h(qm.qubits(1)))[0])
    selected = a if c else b
    left = qm.qubits(1)
    right = qm.qubits(1)
    if c:
        if selected:
            left = qm.x(left)
        if selected:
            right = qm.x(right)
    return c, a, b, qm.measure(left), qm.measure(right)
""").specialize("state")
    endpoint = Aer(use_classical_stores=True)
    artifact = qm.compile(program, to=endpoint.target)
    circuit = to_circuit(program, target=endpoint.target)
    assert "store" not in set(_structured_names(circuit))
    for row in qm.run(artifact, on=endpoint, shots=64, seed=7).result.counts:
        assert row[3] == row[4] == (row[0] and row[1])


def test_controlled_loop_with_negative_control_and_nested_iterations():
    """Control polarity and nested inverse iteration keep a structured body."""
    program, circuit, portable = _inspect("""
@qm.qkernel
def repeated(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    for outer in range(2):
        for inner in range(3):
            state = qm.rx(state, 0.12 * inner)
            state = qm.rz(state, 0.13 * outer)
    return state

@qm.qkernel
def inverse(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.adjoint(repeated, state)

@qm.qkernel
def state() -> qm.Q[qm.Bits[3]]:
    control, target = qm.split_at(qm.qubits(3), 2)
    control, target = qm.control(inverse, control, target, control_state=1)
    return qm.concat(control, target)
""")
    assert circuit.count_ops().get("for_loop") == 1
    assert list(_structured_names(circuit)).count("for_loop") == 2
    _assert_exact(program, circuit, portable)


def test_large_controlled_loop_retains_one_body_without_simulation():
    """A hundred thousand iterations do not allocate an expanded gate list."""
    program = qm.build_source("""
import qamomile as qm
@qm.qkernel
def repeated(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    for index in range(100000):
        state = qm.rx(state, 0.01 * index)
    return state

@qm.qkernel
def state() -> qm.Q[qm.Bits[2]]:
    control, target = qm.split_at(qm.qubits(2), 1)
    control, target = qm.control(repeated, control, target)
    return qm.concat(control, target)
""").specialize("state")
    circuit = to_circuit(program, target=Qiskit())
    assert circuit.count_ops() == {"for_loop": 1}
    loop = circuit.data[0].operation
    assert len(loop.params[0]) == 100000
    assert len(loop.blocks[0].data) == 1


def test_call_reuse_keeps_compiler_scratch_private_without_simulation():
    """The same schoolbook callable shares its checked definition and scratch."""
    program = qm.build_source("""
import qamomile as qm
@qm.qkernel
def multiply(state: qm.Q[qm.UInt[2]]) -> qm.Q[qm.UInt[2]]:
    return qm.mod_mul(state, factor=2, modulus=3)

@qm.qkernel
def state() -> qm.Q[qm.Bits[4]]:
    left, right = qm.split_at(qm.qubits(4), 2)
    left = multiply(qm.as_uint(left))
    right = multiply(qm.as_uint(right))
    return qm.concat(qm.as_bits(left), qm.as_bits(right))
""").specialize("state")
    target = Qiskit(
        realizations=qm.RealizationOptions({qm.mod_mul: "mod_mul.schoolbook_ripple"})
    )
    circuit = to_circuit(program, target=target)
    left, right = (row.operation for row in circuit)
    assert left._owner.identity == right._owner.identity
    assert left._definition is None and right._definition is None
    scratch_left = set(circuit.data[0].qubits) - set(circuit.qubits[:4])
    scratch_right = set(circuit.data[1].qubits) - set(circuit.qubits[:4])
    assert scratch_left and scratch_left == scratch_right


def test_store_option_preserves_existing_positional_target_arguments():
    """The optional profile does not shift existing constructor arguments."""
    endpoint = Aer("statevector", 0, 19)
    assert endpoint.optimization_level == 0
    assert endpoint.seed_transpiler == 19
    assert endpoint.target.use_classical_stores is False
    target = Qiskit((), 0, 19, False, qm.RealizationOptions())
    assert target.use_native_pauli_evolution is False
    assert target.use_classical_stores is False
    assert Qiskit().fingerprint == Aer().target.fingerprint


@pytest.mark.parametrize("label", ["x", "h"])
def test_callable_label_cannot_select_sdk_gate_semantics(label):
    """Display names never select the SDK standard equivalence-library entry."""
    from qamomile.compilation import RealizationCapability

    @qm.composite_gate(name=label)
    @qm.qkernel
    def operation(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
        return qm.rz(qm.rx(state, 0.37), 0.21)

    @qm.qkernel
    def state() -> qm.Q[qm.Bits[1]]:
        return operation(qm.qubits(1))

    program = state.specialize()
    circuit = to_circuit(program)
    portable = emit_circuit(lower_to_circuit(program))
    assert circuit.data[0].operation.label == label
    assert circuit.data[0].operation.name != label
    _assert_exact(program, circuit, portable)
    explicit_body = emit_circuit(
        lower_to_circuit(
            program,
            realization_capabilities=frozenset({RealizationCapability.NAMED_CALL}),
        )
    )
    assert explicit_body.data[0].operation.name != label
    np.testing.assert_allclose(
        _operator(explicit_body), _operator(portable), atol=2e-12
    )
