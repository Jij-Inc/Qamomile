"""Quration backend tests, including optional PyQret execution coverage."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.stdlib import amplitude_encoding
from qamomile.circuit.transpiler.circuit_ir import (
    DEFAULT_POLICY,
    SELECT_SEMANTIC_KEY,
    BinaryExpr,
    BinaryOperator,
    CallableIdentity,
    CircuitBuilder,
    CircuitProgram,
    GateInstruction,
    LiteralExpr,
    LoopVariableExpr,
    ParameterExpr,
    ReusableCircuit,
    SemanticArguments,
    legalize_program,
    lower_circuit_plan,
    verify_target_legal,
)
from qamomile.circuit.transpiler.errors import EmitError, TargetCapabilityError
from qamomile.circuit.transpiler.gate_emitter import GateKind
from qamomile.linalg import PauliLCU
from qamomile.quration import QurationTranspiler
from qamomile.quration.materializer import (
    PyQretMaterializer,
    _emit_call,
    _emit_gate,
    _emit_global_phase,
    evaluate_scalar,
)


@qmc.qkernel
def _quration_bell(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
    """Prepare and measure a Bell state with a bound rotation."""
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    left = qmc.ry(left, theta)
    left, right = qmc.cx(left, right)
    return qmc.measure(left), qmc.measure(right)


@qmc.qkernel
def _quration_expval(observable: qmc.Observable) -> qmc.Float:
    """Prepare |+> and evaluate an observable."""
    qubit = qmc.qubit("qubit")
    qubit = qmc.h(qubit)
    return qmc.expval(qubit, observable)


@qmc.qkernel
def _quration_resource_bell() -> tuple[qmc.Bit, qmc.Bit]:
    """Prepare a Clifford Bell state for FTQC resource compilation."""
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    left = qmc.h(left)
    left, right = qmc.cx(left, right)
    return qmc.measure(left), qmc.measure(right)


@qmc.qkernel
def _quration_gate_helper(
    left: qmc.Qubit,
    right: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Exercise Quration's composite two-qubit gate decompositions."""
    left, right = qmc.swap(left, right)
    left, right = qmc.rzz(left, right, 0.25)
    return qmc.cp(left, right, -0.5)


@qmc.composite_gate(name="named_rotation")
def _quration_named_rotation(qubit: qmc.Qubit) -> qmc.Qubit:
    """Provide a named reusable body for PyQret call-graph tests."""
    qubit = qmc.h(qubit)
    return qmc.t(qubit)


@qmc.qkernel
def _quration_named_call() -> qmc.Bit:
    """Invoke one named composite twice before measurement."""
    qubit = qmc.qubit("qubit")
    qubit = _quration_named_rotation(qubit)
    qubit = _quration_named_rotation(qubit)
    return qmc.measure(qubit)


@qmc.qkernel
def _quration_lowering_coverage(
    observable: qmc.Observable,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    """Exercise reusable calls, loop unrolling, and Pauli evolution."""
    qubits = qmc.qubit_array(2, "qubits")
    for _ in qmc.range(2):
        qubits[0], qubits[1] = _quration_gate_helper(qubits[0], qubits[1])
    qubits = qmc.pauli_evolve(qubits, observable, gamma)
    return qmc.measure(qubits)


@qmc.qkernel
def _quration_state_preparation() -> qmc.Vector[qmc.Bit]:
    """Exercise semantic state preparation through the PyQret fallback.

    Returns:
        qmc.Vector[qmc.Bit]: Prepared-state measurements.
    """
    qubits = qmc.qubit_array(2, "qubits")
    qubits = amplitude_encoding(qubits, [1.0, 2.0, 3.0, 4.0])
    return qmc.measure(qubits)


@qmc.qkernel
def _quration_ripple_carry() -> tuple[qmc.Vector[qmc.Bit], qmc.Bit, qmc.Bit]:
    """Exercise semantic full addition through the PyQret fallback.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Bit, qmc.Bit]: Accumulator, carry,
            and overflow measurements.
    """
    left = qmc.qubit_array(2, "left")
    right = qmc.qubit_array(2, "right")
    carry = qmc.qubit("carry")
    overflow = qmc.qubit("overflow")
    _, right, carry, overflow = qmc.ripple_carry_add(left, right, carry, overflow)
    return qmc.measure(right), qmc.measure(carry), qmc.measure(overflow)


@qmc.qkernel
def _quration_multi_controlled_x() -> qmc.Bit:
    """Exercise semantic MCX through PyQret's native bounded realization.

    Returns:
        qmc.Bit: Target measurement.
    """
    controls = qmc.x(qmc.qubit_array(2, "controls"))
    target = qmc.qubit("target")
    _, target = qmc.mcx(controls, target)
    return qmc.measure(target)


@qmc.qkernel
def _quration_identity(qubit: qmc.Qubit) -> qmc.Qubit:
    """Return one unchanged qubit for phase-kickback coverage."""
    return qubit


@qmc.qkernel
def _quration_phased_identity(qubit: qmc.Qubit) -> qmc.Qubit:
    """Apply a pi global phase to an identity body."""
    return qmc.global_phase(_quration_identity, math.pi)(qubit)


_quration_controlled_phase = qmc.control(_quration_phased_identity)
_quration_zero_controlled_phase = qmc.control(
    _quration_phased_identity,
    control_value=0,
)


@qmc.qkernel
def _quration_phase_kickback() -> qmc.Bit:
    """Turn a controlled global phase into a deterministic measured bit."""
    control = qmc.h(qmc.qubit("control"))
    target = qmc.qubit("target")
    control, target = _quration_controlled_phase(control, target)
    control = qmc.h(control)
    return qmc.measure(control)


@qmc.qkernel
def _quration_zero_control_x() -> qmc.Bit:
    """Apply X when one control remains in its zero state."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(qmc.x, control_value=0)(control, target)
    return qmc.measure(target)


@qmc.qkernel
def _quration_zero_control_phase_kickback() -> qmc.Bit:
    """Turn a control-zero phase into a deterministic measured bit."""
    control = qmc.h(qmc.qubit("control"))
    target = qmc.qubit("target")
    control, target = _quration_zero_controlled_phase(control, target)
    control = qmc.h(control)
    return qmc.measure(control)


@qmc.qkernel
def _quration_select_phase_kickback() -> qmc.Bit:
    """Observe an identity-case phase through a two-case SELECT."""
    index = qmc.h(qmc.qubit("index"))
    target = qmc.qubit("target")
    index, target = qmc.select([_quration_identity, _quration_phased_identity])(
        index, target
    )
    index = qmc.h(index)
    return qmc.measure(index)


@qmc.qkernel
def _quration_wide_select() -> qmc.Bit:
    """Build a four-case SELECT beyond Quration's control capability."""
    index = qmc.qubit_array(2, "index")
    target = qmc.qubit("target")
    index, target = qmc.select(
        [
            _quration_identity,
            _quration_phased_identity,
            _quration_identity,
            _quration_phased_identity,
        ]
    )(index, target)
    return qmc.measure(target)


@qmc.qkernel
def _quration_explicit_overwide_select() -> qmc.Bit:
    """Keep an explicit two-bit SELECT width beyond Quration's capability."""
    index = qmc.qubit_array(2, "index")
    target = qmc.qubit("target")
    index, target = qmc.select(
        [_quration_identity, _quration_phased_identity],
        num_index_qubits=2,
    )(index, target)
    return qmc.measure(target)


@qmc.qkernel
def _quration_symbolic_width_select(width: qmc.UInt) -> qmc.Bit:
    """Resolve a symbolic SELECT width before Quration capability checks."""
    index = qmc.qubit_array(2, "index")
    target = qmc.qubit("target")
    index, target = qmc.select(
        [_quration_identity, _quration_phased_identity],
        num_index_qubits=width,
    )(index, target)
    return qmc.measure(target)


@qmc.qkernel
def _quration_select_body(
    index: qmc.Qubit,
    target: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Apply the supported two-case SELECT as a controllable body."""
    return qmc.select([_quration_identity, _quration_phased_identity])(
        index,
        target,
    )


@qmc.qkernel
def _quration_outer_controlled_select() -> qmc.Bit:
    """Add an outer control beyond Quration's SELECT fallback limit."""
    outer = qmc.qubit("outer")
    index = qmc.qubit("index")
    target = qmc.qubit("target")
    outer, index, target = qmc.control(_quration_select_body)(
        outer,
        index,
        target,
    )
    return qmc.measure(target)


_quration_two_term_lcu = PauliLCU.from_matrix(
    np.array([[1j, 0.5], [0.5, 1j]], dtype=np.complex128),
    atol=1e-12,
)
_quration_two_term_encoding = qmc.pauli_lcu_block_encoding(_quration_two_term_lcu)
_quration_three_term_lcu = PauliLCU.from_matrix(
    np.array(
        [[0.25 + 1j, 0.5], [0.5, -0.25 + 1j]],
        dtype=np.complex128,
    ),
    atol=1e-12,
)
_quration_three_term_encoding = qmc.pauli_lcu_block_encoding(_quration_three_term_lcu)


@qmc.qkernel
def _quration_two_term_pauli_lcu() -> qmc.Bit:
    """Apply the supported two-term Pauli LCU block encoding."""
    signal = qmc.qubit_array(1, "signal")
    system = qmc.qubit_array(1, "system")
    signal, _ = _quration_two_term_encoding.unitary(signal, system)
    return qmc.measure(signal[0])


@qmc.qkernel
def _quration_two_term_pauli_lcu_expval(
    observable: qmc.Observable,
) -> qmc.Float:
    """Apply the supported complex LCU and evaluate an observable."""
    signal = qmc.qubit_array(1, "signal")
    system = qmc.qubit_array(1, "system")
    signal, system = _quration_two_term_encoding.unitary(signal, system)
    return qmc.expval((signal[0], system[0]), observable)


@qmc.qkernel
def _quration_three_term_pauli_lcu() -> qmc.Bit:
    """Build an LCU whose two-bit SELECT exceeds Quration's control bound."""
    signal = qmc.qubit_array(2, "signal")
    system = qmc.qubit_array(1, "system")
    signal, _ = _quration_three_term_encoding.unitary(signal, system)
    return qmc.measure(signal[0])


def _lower_quration_program(kernel: qmc.QKernel) -> CircuitProgram:
    """Lower one public qkernel to backend-neutral circuit IR.

    Args:
        kernel (qmc.QKernel): Classical-I/O entrypoint to lower.

    Returns:
        CircuitProgram: First lowered quantum segment.
    """
    transpiler = QurationTranspiler()
    prepared = transpiler.prepare(kernel)
    plan = transpiler.plan_circuit(prepared)
    program = lower_circuit_plan(plan).get_first_circuit()
    assert isinstance(program, CircuitProgram)
    return program


def _quration_select_program(
    index_width: int,
    outer_controls: int = 0,
) -> CircuitProgram:
    """Build a representative semantic SELECT fallback for capability tests.

    The case body deliberately includes both a primitive gate and a global
    phase so Quration must validate its distributed-control and relative-phase
    contracts together.

    Args:
        index_width (int): Number of controls on each SELECT case call.
        outer_controls (int): Controls on the complete SELECT call. Defaults
            to zero.

    Returns:
        CircuitProgram: Caller containing one semantic SELECT call.
    """
    num_cases = 1 << index_width
    case = CircuitBuilder(1, 0, name="select-case")
    case.append_gate(GateKind.X, (0,))
    case.add_global_phase(0.25)
    case_callee = ReusableCircuit(
        case.freeze(),
        "select-case",
        controls=index_width,
        operand_widths=(1,),
    )

    fallback = CircuitBuilder(index_width + 1, 0, name="select")
    index_slots = tuple(range(index_width))
    target_slot = index_width
    for case_index in range(num_cases):
        zero_slots = tuple(
            slot for slot in index_slots if ((case_index >> slot) & 1) == 0
        )
        for slot in zero_slots:
            fallback.append_gate(GateKind.X, (slot,))
        fallback.append_call(case_callee, (*index_slots, target_slot))
        for slot in reversed(zero_slots):
            fallback.append_gate(GateKind.X, (slot,))

    select_callee = ReusableCircuit(
        fallback.freeze(),
        "select",
        controls=outer_controls,
        identity=CallableIdentity(
            SELECT_SEMANTIC_KEY,
            "select",
            SemanticArguments.from_mapping(
                {
                    "index_order": "lsb0",
                    "num_cases": num_cases,
                    "num_index_qubits": index_width,
                }
            ),
        ),
        operand_widths=(index_width, 1),
    )
    caller = CircuitBuilder(
        outer_controls + index_width + 1,
        0,
        name="select-caller",
    )
    caller.append_call(
        select_callee,
        tuple(range(outer_controls + index_width + 1)),
    )
    return caller.freeze()


def test_quration_scalar_evaluator_handles_literals_loops_and_arithmetic() -> None:
    """PyQret materialization evaluates only concrete scalar expressions."""
    expression = BinaryExpr(
        BinaryOperator.MUL,
        LoopVariableExpr("i"),
        LiteralExpr(0.5),
    )

    assert evaluate_scalar(expression, {"i": 3}) == 1.5


def test_quration_scalar_evaluator_rejects_runtime_parameters() -> None:
    """Unbound circuit parameters fail before entering the PyQret builder."""
    with pytest.raises(EmitError, match="must be supplied through bindings"):
        evaluate_scalar(ParameterExpr("theta"))


def test_quration_does_not_drop_tiny_preserved_phase() -> None:
    """The PRESERVE capability treats only exact literal zero as a no-op."""
    calls: list[tuple[object, float, float]] = []

    class _Intrinsic:
        @staticmethod
        def global_phase(builder: object, phase: float, precision: float) -> None:
            calls.append((builder, phase, precision))

    builder_token = object()
    context = SimpleNamespace(
        intrinsic=_Intrinsic(),
        builder=builder_token,
        precision=1e-9,
    )
    builder = CircuitBuilder(1, 0)
    builder.add_global_phase(1e-16)

    _emit_global_phase(builder.freeze(), context)

    assert calls == [(builder_token, 1e-16, 1e-9)]


@pytest.mark.parametrize(
    ("kind", "qubits", "phase"),
    [
        (GateKind.P, ("target",), 0.2),
        (GateKind.CP, ("control", "target"), 0.1),
    ],
)
def test_quration_gate_decompositions_preserve_global_factors(
    kind: GateKind,
    qubits: tuple[str, ...],
    phase: float,
) -> None:
    """P and CP decompositions emit the exact missing global factor."""
    calls: list[tuple[object, float, float]] = []

    class _Intrinsic:
        def __getattr__(self, name: str) -> Any:
            return lambda *args: None

        @staticmethod
        def global_phase(builder: object, angle: float, precision: float) -> None:
            calls.append((builder, angle, precision))

        @staticmethod
        def rz(qubit: object, angle: float, precision: float) -> None:
            pass

        @staticmethod
        def cx(control: object, target: object) -> None:
            pass

    builder_token = object()
    operation = GateInstruction(
        kind=kind,
        inputs=(),
        outputs=(),
        parameters=(LiteralExpr(0.4),),
    )

    _emit_gate(
        operation,
        qubits,
        _Intrinsic(),
        1e-9,
        {},
        builder_token,
    )

    assert calls == [(builder_token, phase, 1e-9)]


@pytest.mark.parametrize(
    ("controls", "inverse"),
    [(1, False), (0, True), (1, True)],
)
def test_quration_accepts_bounded_transformed_phase_calls(
    controls: int,
    inverse: bool,
) -> None:
    """Quration verifies its declared control, inverse, and phase fallback."""
    body = CircuitBuilder(1, 0, name="phase-only")
    body.append_gate(GateKind.RZ, (0,), (LiteralExpr(0.5),))
    body.add_global_phase(0.25)
    caller = CircuitBuilder(controls + 1, 0, name="phase-caller")
    caller.append_call(
        ReusableCircuit(
            body.freeze(),
            "phase-only",
            controls=controls,
            inverse=inverse,
        ),
        tuple(range(controls + 1)),
    )
    capabilities = PyQretMaterializer().capabilities
    legalized = legalize_program(caller.freeze(), capabilities, DEFAULT_POLICY)

    verify_target_legal(legalized, capabilities)


def test_quration_accepts_two_case_select_fallback() -> None:
    """One index control fits Quration's distributed-control profile."""
    capabilities = PyQretMaterializer().capabilities
    legalized = legalize_program(
        _quration_select_program(index_width=1),
        capabilities,
        DEFAULT_POLICY,
    )

    verify_target_legal(legalized, capabilities)


def test_quration_accepts_two_term_pauli_lcu() -> None:
    """A one-bit LCU SELECT fits Quration's distributed-control profile."""
    capabilities = PyQretMaterializer().capabilities
    legalized = legalize_program(
        _lower_quration_program(_quration_two_term_pauli_lcu),
        capabilities,
        DEFAULT_POLICY,
    )

    verify_target_legal(legalized, capabilities)


def test_quration_rejects_three_term_pauli_lcu() -> None:
    """A two-bit LCU SELECT exceeds Quration's distributed-control profile."""
    capabilities = PyQretMaterializer().capabilities
    legalized = legalize_program(
        _lower_quration_program(_quration_three_term_pauli_lcu),
        capabilities,
        DEFAULT_POLICY,
    )

    with pytest.raises(TargetCapabilityError, match=r"quration.*controls=2"):
        verify_target_legal(legalized, capabilities)


@pytest.mark.parametrize("index_width", [2, 3])
def test_quration_rejects_wide_select_fallback(index_width: int) -> None:
    """Four- and eight-case SELECT exceed Quration's control bound."""
    capabilities = PyQretMaterializer().capabilities
    legalized = legalize_program(
        _quration_select_program(index_width=index_width),
        capabilities,
        DEFAULT_POLICY,
    )

    with pytest.raises(TargetCapabilityError, match="reusable call transforms"):
        verify_target_legal(legalized, capabilities)


@pytest.mark.parametrize(
    ("kernel", "bindings"),
    [
        (_quration_explicit_overwide_select, None),
        (_quration_symbolic_width_select, {"width": 2}),
    ],
)
def test_quration_checks_resolved_declared_select_width(
    kernel: qmc.QKernel,
    bindings: dict[str, int] | None,
) -> None:
    """Quration validates the declared width, not the case-count minimum.

    Args:
        kernel (qmc.QKernel): Two-case SELECT with a declared two-bit index.
        bindings (dict[str, int] | None): Compile-time width bindings.
    """
    transpiler = QurationTranspiler()
    prepared = transpiler.prepare(kernel, bindings=bindings)
    plan = transpiler.plan_circuit(prepared, bindings=bindings)
    [segment] = lower_circuit_plan(plan, bindings=bindings).compiled_quantum
    capabilities = PyQretMaterializer().capabilities
    legalized = legalize_program(segment.circuit, capabilities, DEFAULT_POLICY)

    with pytest.raises(TargetCapabilityError, match="reusable call transforms"):
        verify_target_legal(legalized, capabilities)


def test_quration_rejects_outer_controlled_select_fallback() -> None:
    """An outer control plus the SELECT index exceeds Quration's bound."""
    capabilities = PyQretMaterializer().capabilities
    legalized = legalize_program(
        _quration_select_program(index_width=1, outer_controls=1),
        capabilities,
        DEFAULT_POLICY,
    )

    with pytest.raises(TargetCapabilityError, match="reusable call transforms"):
        verify_target_legal(legalized, capabilities)


def test_quration_rejects_multiple_distributed_controls() -> None:
    """More than one added control fails before PyQret materialization."""
    body = CircuitBuilder(1, 0, name="single-target")
    body.append_gate(GateKind.X, (0,))
    caller = CircuitBuilder(3, 0, name="too-many-controls")
    caller.append_call(
        ReusableCircuit(body.freeze(), "single-target", controls=2),
        (0, 1, 2),
    )
    capabilities = PyQretMaterializer().capabilities
    legalized = legalize_program(caller.freeze(), capabilities, DEFAULT_POLICY)

    with pytest.raises(TargetCapabilityError, match="reusable call transforms"):
        verify_target_legal(legalized, capabilities)


def test_quration_rejects_undeclared_controlled_body_gate() -> None:
    """A controlled primitive outside the explicit profile fails early."""
    body = CircuitBuilder(3, 0, name="toffoli-body")
    body.append_gate(GateKind.TOFFOLI, (0, 1, 2))
    caller = CircuitBuilder(4, 0, name="controlled-toffoli")
    caller.append_call(
        ReusableCircuit(body.freeze(), "toffoli-body", controls=1),
        (0, 1, 2, 3),
    )
    capabilities = PyQretMaterializer().capabilities
    legalized = legalize_program(caller.freeze(), capabilities, DEFAULT_POLICY)

    with pytest.raises(TargetCapabilityError, match="onto TOFFOLI"):
        verify_target_legal(legalized, capabilities)


def test_quration_inlines_controlled_inverse_power_and_phase_without_pyqret() -> None:
    """Mock intrinsics expose exact body and phase transform composition."""
    calls: list[tuple[object, ...]] = []

    class _Intrinsic:
        def __getattr__(self, name: str) -> Any:
            return lambda *args: None

        @staticmethod
        def global_phase(builder: object, angle: float, precision: float) -> None:
            calls.append(("global_phase", builder, angle, precision))

        @staticmethod
        def rz(qubit: object, angle: float, precision: float) -> None:
            calls.append(("rz", qubit, angle, precision))

        @staticmethod
        def cx(target: object, control: object) -> None:
            calls.append(("cx", target, control))

    body = CircuitBuilder(1, 0, name="phased-rz")
    body.append_gate(GateKind.RZ, (0,), (LiteralExpr(0.4),))
    body.add_global_phase(0.3)
    caller = CircuitBuilder(2, 0, name="caller")
    caller.append_call(
        ReusableCircuit(
            body.freeze(),
            "phased-rz",
            power=2,
            controls=1,
            inverse=True,
        ),
        (0, 1),
    )
    [operation] = caller.freeze().operations
    environment = {
        operation.inputs[0]: "control",
        operation.inputs[1]: "target",
    }
    context = SimpleNamespace(
        intrinsic=_Intrinsic(),
        builder="builder",
        precision=1e-9,
        reusable_circuits={},
    )

    _emit_call(operation, environment, (), context, {})

    iteration = [
        ("rz", "target", -0.2, 1e-9),
        ("cx", "target", "control"),
        ("rz", "target", 0.2, 1e-9),
        ("cx", "target", "control"),
        ("global_phase", "builder", -0.15, 1e-9),
        ("rz", "control", -0.3, 1e-9),
    ]
    assert calls == [*iteration, *iteration]
    assert environment[operation.outputs[0]] == "control"
    assert environment[operation.outputs[1]] == "target"


def test_quration_inverts_static_for_order_and_angles_without_pyqret() -> None:
    """Inverse inlining reverses static iterations and each loop body."""
    calls: list[tuple[object, ...]] = []

    class _Intrinsic:
        def __getattr__(self, name: str) -> Any:
            return lambda *args: None

        @staticmethod
        def x(qubit: object) -> None:
            calls.append(("x", qubit))

        @staticmethod
        def rz(qubit: object, angle: float, precision: float) -> None:
            calls.append(("rz", qubit, angle, precision))

    body = CircuitBuilder(1, 0, name="loop-body")
    induction = body.begin_for(range(1, 4))
    body.append_gate(GateKind.RZ, (0,), (induction,))
    body.append_gate(GateKind.X, (0,))
    body.end_for()
    caller = CircuitBuilder(1, 0, name="inverse-loop")
    caller.append_call(
        ReusableCircuit(body.freeze(), "loop-body", inverse=True),
        (0,),
    )
    [operation] = caller.freeze().operations
    environment = {operation.inputs[0]: "target"}
    context = SimpleNamespace(
        intrinsic=_Intrinsic(),
        builder="builder",
        precision=1e-9,
        reusable_circuits={},
    )

    _emit_call(operation, environment, (), context, {})

    assert calls == [
        ("x", "target"),
        ("rz", "target", -3.0, 1e-9),
        ("x", "target"),
        ("rz", "target", -2.0, 1e-9),
        ("x", "target"),
        ("rz", "target", -1.0, 1e-9),
    ]
    assert environment[operation.outputs[0]] == "target"


def test_quration_missing_dependency_has_actionable_error() -> None:
    """Quration use fails clearly when source-built PyQret is unavailable."""
    if importlib.util.find_spec("pyqret") is not None:
        pytest.skip("PyQret is installed in this environment")

    with pytest.raises(ImportError, match="requires the optional 'pyqret'"):
        QurationTranspiler().transpile(
            _quration_bell,
            bindings={"theta": math.pi / 2},
        )
    with pytest.raises(ImportError, match="requires the optional 'pyqret'"):
        QurationTranspiler().transpile(_quration_phase_kickback)


@pytest.mark.quration
def test_quration_transpiles_and_samples_bell_state() -> None:
    """Quration transpilation executes a sampled algorithm through PyQret."""
    pytest.importorskip("pyqret")
    transpiler = QurationTranspiler()
    executable = transpiler.transpile(
        _quration_bell,
        bindings={"theta": math.pi / 2},
    )

    result = executable.sample(transpiler.executor(seed=3), shots=128).result()
    counts = dict(result.results)
    assert sum(counts.values()) == 128
    assert set(counts) <= {(0, 0), (1, 1)}


@pytest.mark.quration
def test_quration_transpiles_and_executes_expectation_value() -> None:
    """Quration full-state simulation evaluates the expval execution path."""
    pytest.importorskip("pyqret")
    transpiler = QurationTranspiler()
    executable = transpiler.transpile(
        _quration_expval,
        bindings={"observable": qm_o.X(0)},
    )

    result = executable.run(transpiler.executor(seed=5)).result()
    assert result == pytest.approx(1.0, abs=1e-10)


@pytest.mark.quration
def test_quration_samples_two_term_complex_pauli_lcu() -> None:
    """Quration samples the supported complex two-term block encoding."""
    pytest.importorskip("pyqret")
    transpiler = QurationTranspiler()
    executable = transpiler.transpile(_quration_two_term_pauli_lcu)
    shots = 1024

    result = executable.sample(transpiler.executor(seed=13), shots=shots).result()
    counts = dict(result.results)
    assert sum(counts.values()) == shots
    assert set(counts) <= {0, 1}

    expected_success = (
        abs(1j) ** 2 + abs(0.5) ** 2
    ) / _quration_two_term_encoding.normalization**2
    tolerance = (
        6.0 * math.sqrt(expected_success * (1.0 - expected_success) / shots) + 0.02
    )
    assert counts.get(0, 0) / shots == pytest.approx(
        expected_success,
        abs=tolerance,
    )


@pytest.mark.quration
def test_quration_executes_two_term_complex_pauli_lcu() -> None:
    """Quration executes the supported complex two-term block encoding."""
    pytest.importorskip("pyqret")
    projected_y = qm_o.Hamiltonian(num_qubits=2)
    projected_y.add_term(
        (qm_o.PauliOperator(qm_o.Pauli.Y, 1),),
        0.5,
    )
    projected_y.add_term(
        (
            qm_o.PauliOperator(qm_o.Pauli.Z, 0),
            qm_o.PauliOperator(qm_o.Pauli.Y, 1),
        ),
        0.5,
    )
    transpiler = QurationTranspiler()
    executable = transpiler.transpile(
        _quration_two_term_pauli_lcu_expval,
        bindings={"observable": projected_y},
    )

    result = executable.run(transpiler.executor(seed=12)).result()
    identity_weight = 1j
    x_weight = 0.5
    expected_y = (
        2.0
        * np.imag(np.conj(identity_weight) * x_weight)
        / _quration_two_term_encoding.normalization**2
    )
    assert float(result) == pytest.approx(expected_y, abs=1e-8)


@pytest.mark.quration
def test_quration_executes_calls_loops_decompositions_and_pauli_evolution() -> None:
    """Representative Quration lowering paths produce an executable circuit."""
    pytest.importorskip("pyqret")
    transpiler = QurationTranspiler()
    executable = transpiler.transpile(
        _quration_lowering_coverage,
        bindings={"observable": qm_o.X(0) * qm_o.Y(1), "gamma": 0.125},
    )

    result = executable.sample(transpiler.executor(seed=7), shots=32).result()
    assert sum(count for _, count in result.results) == 32


@pytest.mark.quration
def test_quration_executes_controlled_global_phase_kickback() -> None:
    """Quration preserves a reusable body phase as a relative phase."""
    pytest.importorskip("pyqret")
    transpiler = QurationTranspiler()
    executable = transpiler.transpile(_quration_phase_kickback)

    result = executable.sample(transpiler.executor(seed=9), shots=16).result()
    assert dict(result.results) == {1: 16}


@pytest.mark.quration
@pytest.mark.parametrize(
    "kernel",
    [_quration_zero_control_x, _quration_zero_control_phase_kickback],
)
def test_quration_executes_zero_activated_controls(kernel: qmc.QKernel) -> None:
    """Quration executes X brackets around gate and phase bodies.

    Args:
        kernel (qmc.QKernel): Zero-activated controlled program.
    """
    pytest.importorskip("pyqret")
    transpiler = QurationTranspiler()
    executable = transpiler.transpile(kernel)

    result = executable.sample(transpiler.executor(seed=12), shots=16).result()
    assert dict(result.results) == {1: 16}


@pytest.mark.quration
def test_quration_executes_two_case_select_phase_kickback() -> None:
    """The public Quration path executes its supported SELECT fallback."""
    pytest.importorskip("pyqret")
    transpiler = QurationTranspiler()
    executable = transpiler.transpile(_quration_select_phase_kickback)

    result = executable.sample(transpiler.executor(seed=10), shots=16).result()
    assert dict(result.results) == {1: 16}


@pytest.mark.quration
@pytest.mark.parametrize(
    ("kernel", "bindings"),
    [
        (_quration_wide_select, None),
        (_quration_explicit_overwide_select, None),
        (_quration_symbolic_width_select, {"width": 2}),
        (_quration_outer_controlled_select, None),
    ],
)
def test_quration_public_select_rejects_unsupported_controls(
    kernel: qmc.QKernel,
    bindings: dict[str, int] | None,
) -> None:
    """Public SELECT lowering uses the resolved declared index width.

    Args:
        kernel (qmc.QKernel): SELECT program expected to exceed Quration's
            distributed-control bound.
        bindings (dict[str, int] | None): Compile-time width bindings.
    """
    pytest.importorskip("pyqret")

    with pytest.raises(TargetCapabilityError, match="reusable call transforms"):
        QurationTranspiler().transpile(kernel, bindings=bindings)


@pytest.mark.quration
@pytest.mark.parametrize(
    "kernel",
    [
        _quration_state_preparation,
        _quration_ripple_carry,
        _quration_multi_controlled_x,
    ],
)
def test_quration_transpiles_semantic_composites(kernel: qmc.QKernel) -> None:
    """New semantic composites select executable native or fallback paths."""
    pytest.importorskip("pyqret")
    transpiler = QurationTranspiler()
    executable = transpiler.transpile(kernel)

    result = executable.sample(transpiler.executor(seed=11), shots=4).result()
    assert sum(count for _, count in result.results) == 4


@pytest.mark.quration
def test_quration_preserves_named_composites_as_pyqret_calls() -> None:
    """Named Qamomile composites remain nodes in PyQret's call graph."""
    pytest.importorskip("pyqret")
    executable = QurationTranspiler().transpile(_quration_named_call)
    circuit = executable.compiled_quantum[0].circuit

    call_graph = circuit.get_ir().gen_call_graph(display_num_calls=True)
    assert call_graph.count('label = "named_rotation') == 1
    assert 'label = "2"' in call_graph


@pytest.mark.quration
def test_quration_emits_program_global_phase_natively() -> None:
    """A concrete global phase reaches PyQret's intrinsic phase operation."""
    pytest.importorskip("pyqret")
    builder = CircuitBuilder(1, 0, name="global_phase")
    builder.add_global_phase(0.25)

    artifact = PyQretMaterializer().materialize(builder.freeze()).artifact

    assert "GlobalPhase 0.25" in artifact.get_ir().gen_cfg()


@pytest.mark.quration
def test_quration_compiles_ftqc_resources_and_preserves_native_owners() -> None:
    """Quration compilation returns stable pass and resource information."""
    pytest.importorskip("pyqret")
    from pyqret.backend import CompileOption, OptLevel, ScLsFixedV0Option

    topology = Path(__file__).parent / "data" / "plane.yaml"
    option = CompileOption(
        opt_level=OptLevel.O0,
        sc_ls_fixed_v0_option=ScLsFixedV0Option(topology=str(topology)),
    )

    result = QurationTranspiler().compile_resources(
        _quration_resource_bell,
        option,
    )

    assert result.compiler.option.opt_level is OptLevel.O0
    assert result.compile_result.get_run_order()
    assert result.compile_info.gate_count > 0
    assert result.circuit.has_mf()
