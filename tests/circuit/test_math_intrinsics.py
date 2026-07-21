"""Tests for exact unary mathematical qkernel expressions."""

from __future__ import annotations

import math

import pytest
import sympy as sp

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.ir.dataflow import walk_operations
from qamomile.circuit.ir.operation.arithmetic_operations import (
    UnaryMathOp,
    UnaryMathOpKind,
)
from qamomile.circuit.ir.types.primitives import FloatType, QubitType, UIntType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.errors import EmitError


@qmc.qkernel
def _log_width_register(register_size: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Allocate a zero register with ``ceil(log2(register_size))`` qubits."""
    width = qmc.ceil(qmc.log2(register_size))
    return qmc.qubit_array(width, name="log_width")


@qmc.qkernel
def _derived_log_width(register_size: qmc.UInt) -> qmc.UInt:
    """Return a logarithmic structural width from a nested qkernel."""
    return qmc.ceil(qmc.log2(register_size))


@qmc.qkernel
def _float_log2(value: qmc.Float) -> qmc.Float:
    """Return a symbolic base-two logarithm of one floating-point input."""
    return qmc.log2(value)


@qmc.qkernel
def _float_ceil(value: qmc.Float) -> qmc.UInt:
    """Return a symbolic unsigned ceiling of one floating-point input."""
    return qmc.ceil(value)


@qmc.qkernel
def _log_width_sample(register_size: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Sample a structural logarithmic-width zero register."""
    width = _derived_log_width(register_size)
    return qmc.measure(qmc.qubit_array(width, name="log_width"))


@qmc.qkernel
def _log_width_expval(
    register_size: qmc.UInt,
    observable: qmc.Observable,
) -> qmc.Float:
    """Evaluate an observable on a structural logarithmic-width register."""
    width = qmc.ceil(qmc.log2(register_size))
    return qmc.expval(qmc.qubit_array(width, name="log_width"), observable)


def test_log2_and_ceil_remain_exact_in_resource_expressions() -> None:
    """The estimator preserves an exact ceiling of a base-two logarithm."""
    estimate = _log_width_register.estimate_resources()
    n = estimate.parameters["register_size"]

    assert (
        sp.simplify(
            estimate.width.allocated_qubits - sp.ceiling(sp.log(n, 2)),
        )
        == 0
    )
    for register_size in (2, 3, 8, 9, 1024):
        concrete = estimate.substitute(register_size=register_size)
        assert concrete.qubits == math.ceil(math.log2(register_size))


def test_log2_resource_requirement_preserves_the_source_domain() -> None:
    """Resource substitution reports UInt log2 domain errors directly."""
    estimate = _log_width_register.estimate_resources()

    assert estimate.substitute(register_size=1).qubits == 0
    with pytest.raises(ValueError, match="log2 input must be at least 1"):
        estimate.substitute(register_size=0)
    with pytest.raises(ValueError, match="log2 input must be at least 1"):
        _log_width_register.estimate_resources(inputs={"register_size": 0})
    for boolean in (False, True):
        with pytest.raises(TypeError, match="expects UIntType, got bool"):
            _log_width_register.estimate_resources(inputs={"register_size": boolean})


@pytest.mark.parametrize(
    "value", [0.0, -0.5, float("nan"), float("inf"), -float("inf"), 1j]
)
def test_float_log2_resource_requirement_rejects_invalid_domain(
    value: complex | float,
) -> None:
    """Float log2 estimation retains positivity, finiteness, and reality."""
    with pytest.raises(ValueError, match="log2 input must be"):
        _float_log2.estimate_resources(inputs={"value": value})


def test_float_log2_resource_requirement_accepts_fractional_input() -> None:
    """A finite positive Float remains valid even when smaller than one."""
    estimate = _float_log2.estimate_resources(inputs={"value": 0.5})

    assert estimate.gates.total == 0
    assert estimate.parameters == {}


@pytest.mark.parametrize("value", [-1.0, -1.25, float("nan"), float("inf"), 1j])
def test_float_ceil_resource_requirement_rejects_invalid_domain(
    value: complex | float,
) -> None:
    """Float ceil estimation rejects values outside its UInt result domain."""
    with pytest.raises(ValueError, match="ceil input must be"):
        _float_ceil.estimate_resources(inputs={"value": value})


@pytest.mark.parametrize("value", [-0.5, 0.0, 2.25])
def test_float_ceil_resource_requirement_accepts_nonnegative_result(
    value: float,
) -> None:
    """Float ceil accepts exactly the finite inputs whose result is UInt."""
    estimate = _float_ceil.estimate_resources(inputs={"value": value})

    assert estimate.gates.total == 0
    assert estimate.parameters == {}


def test_float_log2_requirement_serializes_exclusive_finite_domain() -> None:
    """Serialized requirements expose the exact Float log2 domain."""
    estimate = _float_log2.estimate_resources()
    requirement = next(
        item
        for item in estimate.to_dict()["requirements"]
        if item["label"] == "log2 input"
    )

    assert requirement["minimum"] == 0
    assert requirement["minimum_inclusive"] is False
    assert requirement["finite"] is True


@pytest.mark.parametrize(
    ("kind", "input_type", "output_type"),
    [
        (UnaryMathOpKind.LOG2, QubitType(), FloatType()),
        (UnaryMathOpKind.LOG2, UIntType(), UIntType()),
        (UnaryMathOpKind.CEIL, FloatType(), FloatType()),
    ],
)
def test_resource_estimation_rejects_malformed_unary_math_types(
    kind: UnaryMathOpKind,
    input_type: object,
    output_type: object,
) -> None:
    """Raw malformed unary math IR fails closed instead of costing zero."""
    operation = UnaryMathOp(
        operands=[Value(type=input_type, name="input")],
        results=[Value(type=output_type, name="output")],
        kind=kind,
    )

    with pytest.raises(ValueError, match=f"malformed {kind.name} operation"):
        qmc.estimate_resources([operation])


def test_log2_and_ceil_are_independent_ir_operations() -> None:
    """Composition preserves two abstract operations in frontend IR."""
    operations = [
        operation
        for operation in walk_operations(_log_width_register.build().operations)
        if isinstance(operation, UnaryMathOp)
    ]

    assert [operation.kind for operation in operations] == [
        UnaryMathOpKind.LOG2,
        UnaryMathOpKind.CEIL,
    ]


def test_concrete_math_inputs_fold_and_validate_domains() -> None:
    """Concrete calls fold eagerly and reject invalid input domains."""
    assert math.isclose(
        qmc.log2(8).value.get_const(),
        3.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert qmc.ceil(3.25).value.get_const() == 4
    with pytest.raises(TypeError, match="Boolean"):
        qmc.log2(True)
    with pytest.raises(ValueError, match="finite and strictly positive"):
        qmc.log2(0)
    with pytest.raises(ValueError, match="finite and strictly positive"):
        qmc.log2(float("nan"))
    with pytest.raises(ValueError, match="non-negative"):
        qmc.ceil(-1.25)
    with pytest.raises(ValueError, match="finite"):
        qmc.ceil(float("inf"))


def test_structural_math_requires_compile_time_binding(sdk_transpiler) -> None:
    """Runtime parameters cannot determine a circuit allocation shape."""
    with pytest.raises(
        EmitError,
        match="Structural UInt parameters must be bound at transpile time",
    ):
        sdk_transpiler.transpiler.transpile(
            _log_width_sample,
            parameters=["register_size"],
        )


@pytest.mark.parametrize("register_size", [2, 3, 9])
def test_log2_and_ceil_sample_across_supported_backends(
    sdk_transpiler,
    register_size: int,
) -> None:
    """Compile-time structural math executes through every sampler."""
    transpiler = sdk_transpiler.transpiler
    executable = transpiler.transpile(
        _log_width_sample,
        bindings={"register_size": register_size},
    )
    result = executable.sample(transpiler.executor(), shots=8).result()
    width = math.ceil(math.log2(register_size))
    assert result.results == [((0,) * width, 8)]


@pytest.mark.parametrize("register_size", [2, 3, 9])
def test_log2_and_ceil_expval_across_supported_backends(
    sdk_transpiler,
    register_size: int,
) -> None:
    """Compile-time structural math executes through every estimator."""
    transpiler = sdk_transpiler.transpiler
    executable = transpiler.transpile(
        _log_width_expval,
        bindings={
            "register_size": register_size,
            "observable": qm_o.Z(0),
        },
    )
    value = executable.run(transpiler.executor()).result()
    assert float(value) == pytest.approx(1.0, abs=1e-8, rel=0.0)
