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
