"""Regressions for runtime parameters inspected by controlled-body profiling."""

from __future__ import annotations

from typing import Any

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.gate import ConcreteControlledU
from qamomile.circuit.transpiler.circuit_ir.emitter import CircuitGateEmitter
from qamomile.circuit.transpiler.circuit_ir.model import ParameterExpr
from qamomile.circuit.transpiler.emit_context import EmitContext
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.parameter_binding import ParameterContainerKind
from qamomile.circuit.transpiler.passes.emit_support import controlled_emission
from qamomile.circuit.transpiler.passes.emit_support.multi_control_ancilla import (
    MultiControlAncillaPool,
)
from qamomile.circuit.transpiler.passes.emit_support.qubit_address import (
    QubitAddress,
)
from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass

ENGINES = [
    pytest.param("qiskit", id="qiskit"),
    pytest.param("quri_parts", marks=pytest.mark.quri_parts, id="quri_parts"),
    pytest.param("cudaq", marks=pytest.mark.cudaq, id="cudaq"),
]


def _make_transpiler(engine: str) -> Any:
    """Build one installed engine transpiler or skip its test.

    Args:
        engine (str): One of ``qiskit``, ``quri_parts``, or ``cudaq``.

    Returns:
        Any: Engine transpiler when its optional SDK is installed.
    """
    if engine == "qiskit":
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return QiskitTranspiler()
    if engine == "quri_parts":
        pytest.importorskip("quri_parts")
        pytest.importorskip("quri_parts.qulacs")
        from qamomile.quri_parts import QuriPartsTranspiler

        return QuriPartsTranspiler()
    if engine == "cudaq":
        pytest.importorskip("cudaq")
        from qamomile.cudaq import CudaqTranspiler

        return CudaqTranspiler()
    raise AssertionError(f"Unknown engine {engine!r}")


@qmc.qkernel
def _parameterized_identity(
    qubit: qmc.Qubit,
    theta: qmc.Float,
) -> qmc.Qubit:
    """Use a runtime parameter only in zero-work classical bookkeeping.

    Args:
        qubit (qmc.Qubit): Qubit returned unchanged.
        theta (qmc.Float): Runtime parameter inspected by the profile.

    Returns:
        qmc.Qubit: Unchanged input qubit.
    """
    _ignored = theta + 0.0
    return qubit


@qmc.qkernel
def _parameterized_rotation(
    qubit: qmc.Qubit,
    theta: qmc.Float,
) -> qmc.Qubit:
    """Apply one rotation whose parameter survives real emission.

    Args:
        qubit (qmc.Qubit): Rotation target.
        theta (qmc.Float): Runtime rotation angle.

    Returns:
        qmc.Qubit: Rotated qubit.
    """
    return qmc.rx(qubit, theta)


@qmc.qkernel
def _parameterized_x(
    qubit: qmc.Qubit,
    theta: qmc.Float,
) -> qmc.Qubit:
    """Apply real quantum work without using the runtime parameter.

    Args:
        qubit (qmc.Qubit): Gate target.
        theta (qmc.Float): Runtime parameter used only in dead bookkeeping.

    Returns:
        qmc.Qubit: Target after one X gate.
    """
    _ignored = theta + 0.0
    return qmc.x(qubit)


@qmc.qkernel
def _parameterized_identity_via_invoke(
    qubit: qmc.Qubit,
    theta: qmc.Float,
) -> qmc.Qubit:
    """Delegate a zero-work body through an InvokeOperation boundary.

    Args:
        qubit (qmc.Qubit): Qubit returned unchanged.
        theta (qmc.Float): Runtime parameter forwarded to the nested body.

    Returns:
        qmc.Qubit: Unchanged input qubit.
    """
    return _parameterized_identity(qubit, theta)


@qmc.qkernel
def _plain_identity(qubit: qmc.Qubit) -> qmc.Qubit:
    """Return one qubit unchanged.

    Args:
        qubit (qmc.Qubit): Qubit returned unchanged.

    Returns:
        qmc.Qubit: Unchanged input qubit.
    """
    return qubit


@qmc.qkernel
def _parameterized_identity_via_inverse(
    qubit: qmc.Qubit,
    theta: qmc.Float,
) -> qmc.Qubit:
    """Keep an inverse identity inside zero-work bookkeeping.

    Args:
        qubit (qmc.Qubit): Qubit returned through the inverse identity.
        theta (qmc.Float): Runtime parameter used only in bookkeeping.

    Returns:
        qmc.Qubit: Unchanged input qubit.
    """
    _ignored = theta + 0.0
    return qmc.inverse(_plain_identity)(qubit)


@qmc.qkernel
def _identity_select_case(qubit: qmc.Qubit) -> qmc.Qubit:
    """Return one SELECT target unchanged.

    Args:
        qubit (qmc.Qubit): Target qubit.

    Returns:
        qmc.Qubit: Unchanged target.
    """
    return qubit


@qmc.qkernel
def _parameterized_identity_via_select(
    index: qmc.Qubit,
    target: qmc.Qubit,
    theta: qmc.Float,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Route an identity target through a zero-work SELECT.

    Args:
        index (qmc.Qubit): SELECT index qubit.
        target (qmc.Qubit): SELECT target qubit.
        theta (qmc.Float): Runtime parameter used only in bookkeeping.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit]: Unchanged index and target.
    """
    _ignored = theta + 0.0
    return qmc.select([_identity_select_case, _identity_select_case])(
        index,
        target,
    )


@qmc.qkernel
def _parameterized_identity_via_controlled_u(
    inner_control: qmc.Qubit,
    target: qmc.Qubit,
    theta: qmc.Float,
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Nest a zero-work controlled-U inside another controlled body.

    Args:
        inner_control (qmc.Qubit): Inner coherent control.
        target (qmc.Qubit): Unchanged target qubit.
        theta (qmc.Float): Runtime parameter forwarded to the identity body.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit]: Unchanged control and target.
    """
    return qmc.control(_parameterized_identity)(inner_control, target, theta)


@qmc.qkernel
def _controlled_parameterized_identity(theta: qmc.Float) -> qmc.Bit:
    """Control a parameterized body that has no quantum work.

    Args:
        theta (qmc.Float): Runtime value used only inside bookkeeping.

    Returns:
        qmc.Bit: Measurement of the unchanged target.
    """
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_parameterized_identity)(
        control,
        target,
        theta,
    )
    return qmc.measure(target)


@qmc.qkernel
def _inverse_controlled_parameterized_identity(theta: qmc.Float) -> qmc.Bit:
    """Invert a controlled parameterized body with no quantum work.

    Args:
        theta (qmc.Float): Runtime value used only inside bookkeeping.

    Returns:
        qmc.Bit: Measurement of the unchanged target.
    """
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.inverse(qmc.control(_parameterized_identity))(
        control,
        target,
        theta,
    )
    return qmc.measure(target)


@qmc.qkernel
def _controlled_nested_parameterized_identity(theta: qmc.Float) -> qmc.Bit:
    """Control a zero-work body containing a nested callable invocation.

    Args:
        theta (qmc.Float): Runtime value forwarded through the call boundary.

    Returns:
        qmc.Bit: Measurement of the unchanged target.
    """
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_parameterized_identity_via_invoke)(
        control,
        target,
        theta,
    )
    return qmc.measure(target)


@qmc.qkernel
def _controlled_inverse_nested_parameterized_identity(theta: qmc.Float) -> qmc.Bit:
    """Control a zero-work composite containing an inverse identity.

    Args:
        theta (qmc.Float): Runtime value used only inside bookkeeping.

    Returns:
        qmc.Bit: Measurement of the unchanged target.
    """
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_parameterized_identity_via_inverse)(
        control,
        target,
        theta,
    )
    return qmc.measure(target)


@qmc.qkernel
def _controlled_select_parameterized_identity(
    theta: qmc.Float,
) -> tuple[qmc.Bit, qmc.Bit]:
    """Control a zero-work body containing a SELECT operation.

    Args:
        theta (qmc.Float): Runtime value used only in bookkeeping.

    Returns:
        tuple[qmc.Bit, qmc.Bit]: Measurements of unchanged index and target.
    """
    outer_control = qmc.qubit("outer_control")
    index = qmc.qubit("index")
    target = qmc.qubit("target")
    outer_control, index, target = qmc.control(_parameterized_identity_via_select)(
        outer_control, index, target, theta
    )
    return qmc.measure(index), qmc.measure(target)


@qmc.qkernel
def _controlled_nested_controlled_parameterized_identity(
    theta: qmc.Float,
) -> qmc.Bit:
    """Control a body containing another zero-work controlled-U.

    Args:
        theta (qmc.Float): Runtime value forwarded through both controls.

    Returns:
        qmc.Bit: Measurement of the unchanged target.
    """
    outer_control = qmc.qubit("outer_control")
    inner_control = qmc.qubit("inner_control")
    target = qmc.qubit("target")
    outer_control, inner_control, target = qmc.control(
        _parameterized_identity_via_controlled_u
    )(outer_control, inner_control, target, theta)
    return qmc.measure(target)


@qmc.qkernel
def _controlled_parameterized_rotation(theta: qmc.Float) -> qmc.Bit:
    """Control a body whose runtime parameter is used by a real gate.

    Args:
        theta (qmc.Float): Runtime rotation angle.

    Returns:
        qmc.Bit: Measurement of the rotated target.
    """
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_parameterized_rotation)(
        control,
        target,
        theta,
    )
    return qmc.measure(target)


@qmc.qkernel
def _controlled_parameterized_x(theta: qmc.Float) -> qmc.Bit:
    """Control real work whose runtime parameter remains unused.

    Args:
        theta (qmc.Float): Runtime value used only in dead bookkeeping.

    Returns:
        qmc.Bit: Measurement of the X target.
    """
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_parameterized_x)(control, target, theta)
    return qmc.measure(target)


@qmc.qkernel
def _invalid_zero_work_array_return(
    qubits: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Return a branch-selected qubit to the wrong array slot.

    Args:
        qubits (qmc.Vector[qmc.Qubit]): Two-qubit input register.

    Returns:
        qmc.Vector[qmc.Qubit]: Invalidly rebound register.
    """
    for index in qmc.range(2):
        if index == 0:
            selected = qubits[0]
        else:
            selected = qubits[1]
        qubits[1] = selected
    return qubits


@qmc.qkernel
def _controlled_invalid_zero_work_array_return() -> qmc.Vector[qmc.Bit]:
    """Control an invalid body whose operations all have zero gate weight.

    Returns:
        qmc.Vector[qmc.Bit]: Measurements that must never be emitted.
    """
    control = qmc.qubit("control")
    qubits = qmc.qubit_array(2, "qubits")
    control, qubits = qmc.control(_invalid_zero_work_array_return)(
        control,
        qubits,
    )
    return qmc.measure(qubits)


def test_parameter_probe_does_not_mutate_any_abi_map() -> None:
    """A profile probe returns a neutral symbol without recording an ABI."""
    emit_pass = StandardEmitPass(
        CircuitGateEmitter(),
        parameters=["theta"],
    )

    with emit_pass._parameter_probe():
        parameter = emit_pass._get_or_create_parameter(
            "theta",
            source_ref="source",
        )

    assert parameter == ParameterExpr("theta")
    assert emit_pass._parameter_map == {}
    assert emit_pass._parameter_sources == {}
    assert emit_pass._parameter_container_kinds == {}
    assert emit_pass._parameter_container_names == {}


def test_analysis_emission_transaction_restores_all_mutable_state() -> None:
    """An exceptional dry run leaves pass state, maps, and bindings unchanged."""
    emit_pass = StandardEmitPass(CircuitGateEmitter(), parameters=["theta"])
    original_emitter = emit_pass._emitter
    original_pool = MultiControlAncillaPool(first_index=4, count=1)
    original_composites = [object()]
    original_active_qubits = {QubitAddress("active"): 0}
    original_parameter_map = {"theta": object()}
    original_parameter_sources = {"theta": "source"}
    original_parameter_kinds = {"theta": ParameterContainerKind.SCALAR}
    original_parameter_names = {"theta": "theta"}
    original_measurements = {0: 0}
    original_overwritten = {("condition", 0)}
    original_safe_outputs = frozenset({"merge"})
    emit_pass._mc_ancilla_pool = original_pool
    emit_pass._composite_emitters = original_composites
    emit_pass._active_qubit_map = original_active_qubits
    emit_pass._parameter_map = original_parameter_map
    emit_pass._parameter_sources = original_parameter_sources
    emit_pass._parameter_container_kinds = original_parameter_kinds
    emit_pass._parameter_container_names = original_parameter_names
    emit_pass._measurement_qubit_map = original_measurements
    emit_pass._overwritten_runtime_condition_sources = original_overwritten
    emit_pass._safe_mixed_bit_merge_outputs = original_safe_outputs
    bindings = EmitContext.from_user_bindings({"bound": 1})
    bindings.set_value("existing", 2)
    original_bindings = bindings.snapshot_state()
    qubit_map = {QubitAddress("input"): 1}
    clbit_map = {QubitAddress("bit"): 0}

    with pytest.raises(RuntimeError, match="analysis failure"):
        with emit_pass._analysis_emission_transaction(
            2,
            qubit_map,
            clbit_map,
            bindings,
        ) as (circuit, analysis_qubits, analysis_clbits, _pool):
            assert analysis_qubits == qubit_map
            assert analysis_qubits is not qubit_map
            assert analysis_clbits == clbit_map
            assert analysis_clbits is not clbit_map
            assert emit_pass._active_qubit_map is analysis_qubits
            assert emit_pass._counting_emission
            circuit.append_call(object(), (0, 1))
            analysis_qubits[QubitAddress("temporary")] = 2
            analysis_clbits[QubitAddress("temporary-bit")] = 1
            emit_pass._parameter_map["temporary"] = object()
            emit_pass._parameter_sources["temporary"] = "temporary-source"
            emit_pass._parameter_container_kinds["temporary"] = (
                ParameterContainerKind.SCALAR
            )
            emit_pass._parameter_container_names["temporary"] = "temporary"
            emit_pass._measurement_qubit_map[1] = 1
            emit_pass._overwritten_runtime_condition_sources.add(("temporary", 1))
            emit_pass._safe_mixed_bit_merge_outputs = frozenset({"temporary"})
            emit_pass._composite_emitters.append(object())
            bindings.set_value("temporary", 3)
            raise RuntimeError("analysis failure")

    assert emit_pass._emitter is original_emitter
    assert emit_pass._mc_ancilla_pool is original_pool
    assert not emit_pass._counting_emission
    assert emit_pass._composite_emitters is original_composites
    assert emit_pass._active_qubit_map is original_active_qubits
    assert emit_pass._parameter_map is original_parameter_map
    assert emit_pass._parameter_sources is original_parameter_sources
    assert emit_pass._parameter_container_kinds is original_parameter_kinds
    assert emit_pass._parameter_container_names is original_parameter_names
    assert emit_pass._measurement_qubit_map is original_measurements
    assert emit_pass._overwritten_runtime_condition_sources is original_overwritten
    assert emit_pass._safe_mixed_bit_merge_outputs is original_safe_outputs
    assert bindings.snapshot_state() == original_bindings
    assert qubit_map == {QubitAddress("input"): 1}
    assert clbit_map == {QubitAddress("bit"): 0}


@pytest.mark.parametrize(
    "kernel",
    [
        pytest.param(_controlled_parameterized_identity, id="controlled"),
        pytest.param(
            _inverse_controlled_parameterized_identity,
            id="inverse-controlled",
        ),
        pytest.param(
            _controlled_nested_parameterized_identity,
            id="nested-invoke",
        ),
        pytest.param(
            _controlled_inverse_nested_parameterized_identity,
            id="nested-inverse",
        ),
        pytest.param(
            _controlled_select_parameterized_identity,
            id="nested-select",
        ),
        pytest.param(
            _controlled_nested_controlled_parameterized_identity,
            id="nested-controlled-u",
        ),
    ],
)
def test_zero_work_profile_does_not_create_phantom_parameter(
    kernel: Any,
) -> None:
    """An empty controlled body has no runtime parameter in its emitted ABI."""
    executable = _make_transpiler("qiskit").transpile(kernel, parameters=["theta"])
    segment = executable.compiled_quantum[0]

    assert not segment.circuit.parameters
    assert segment.parameter_metadata.parameters == []
    assert all(
        instruction.operation.name == "measure" for instruction in segment.circuit.data
    )


def test_zero_work_huge_power_validates_bookkeeping_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A huge identity power does not replay zero-work validation."""
    controlled = next(
        operation
        for operation in _controlled_parameterized_identity.block.operations
        if isinstance(operation, ConcreteControlledU)
    )
    monkeypatch.setattr(controlled, "power", 10**100)

    original = controlled_emission.emit_controlled_operations
    validation_calls = 0

    def count_validation_calls(*args: Any, **kwargs: Any) -> None:
        """Count zero-work validation walks and fail before a large replay."""
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls > 1:
            raise AssertionError("zero-work bookkeeping was replayed for power")
        original(*args, **kwargs)

    monkeypatch.setattr(
        controlled_emission,
        "emit_controlled_operations",
        count_validation_calls,
    )

    executable = _make_transpiler("qiskit").transpile(
        _controlled_parameterized_identity,
        parameters=["theta"],
    )
    segment = executable.compiled_quantum[0]

    assert validation_calls == 1
    assert not segment.circuit.parameters
    assert segment.parameter_metadata.parameters == []


def test_profile_then_real_emission_records_one_parameter() -> None:
    """A real controlled rotation records its probed parameter exactly once."""
    executable = _make_transpiler("qiskit").transpile(
        _controlled_parameterized_rotation,
        parameters=["theta"],
    )
    segment = executable.compiled_quantum[0]

    assert {parameter.name for parameter in segment.circuit.parameters} == {"theta"}
    assert [parameter.name for parameter in segment.parameter_metadata.parameters] == [
        "theta"
    ]


@pytest.mark.parametrize(
    "engine",
    ENGINES,
)
def test_parameter_abi_tracks_real_use_on_every_engine(
    engine: str,
) -> None:
    """Every engine omits probed-only parameters and retains gate inputs."""
    transpiler = _make_transpiler(engine)
    parameter_names = []
    for kernel in (
        _controlled_parameterized_identity,
        _controlled_parameterized_x,
        _controlled_parameterized_rotation,
    ):
        executable = transpiler.transpile(kernel, parameters=["theta"])
        parameter_names.append(
            [
                parameter.name
                for parameter in executable.compiled_quantum[
                    0
                ].parameter_metadata.parameters
            ]
        )

    assert parameter_names == [[], [], ["theta"]]


def test_zero_work_controlled_body_still_runs_return_validation() -> None:
    """Zero quantum work cannot hide invalid array-return bookkeeping."""
    with pytest.raises(EmitError, match="target does not match"):
        _make_transpiler("qiskit").transpile(_controlled_invalid_zero_work_array_return)
