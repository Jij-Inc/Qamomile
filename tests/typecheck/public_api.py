"""Static type contracts for public APIs with input-dependent results."""

from __future__ import annotations

from typing import Any, assert_type

import numpy as np
import sympy as sp

import qamomile.circuit as qmc


def _check_packed_cast_types(
    integer_register: qmc.Vector[qmc.Qubit],
    default_fixed_register: qmc.Vector[qmc.Qubit],
    zero_fixed_register: qmc.Vector[qmc.Qubit],
    one_fixed_register: qmc.Vector[qmc.Qubit],
) -> None:
    """Check packed cast results with default and explicit layout arguments.

    Args:
        integer_register (qmc.Vector[qmc.Qubit]): Source for the QInt cast.
        default_fixed_register (qmc.Vector[qmc.Qubit]): Source for default QFixed.
        zero_fixed_register (qmc.Vector[qmc.Qubit]): Source for zero integer bits.
        one_fixed_register (qmc.Vector[qmc.Qubit]): Source for one integer bit.
    """
    assert_type(qmc.cast(integer_register, qmc.QInt), qmc.QInt)
    assert_type(qmc.cast(default_fixed_register, qmc.QFixed), qmc.QFixed)
    assert_type(qmc.cast(zero_fixed_register, qmc.QFixed, int_bits=0), qmc.QFixed)
    assert_type(qmc.cast(one_fixed_register, qmc.QFixed, int_bits=1), qmc.QFixed)


def _check_oracle_result_shapes(
    oracle: qmc.Oracle,
    qubit: qmc.Qubit,
    vector: qmc.Vector[qmc.Qubit],
    view: qmc.VectorView[qmc.Qubit],
) -> None:
    """Verify that oracle calls preserve their input form statically."""
    assert_type(oracle(qubit), tuple[qmc.Qubit, ...])
    assert_type(oracle(vector), qmc.Vector[qmc.Qubit])
    assert_type(oracle(view), qmc.VectorView[qmc.Qubit])


def _check_modmul_result_shapes(
    control: qmc.Qubit,
    register: qmc.Vector[qmc.Qubit],
) -> None:
    """Verify that modular multiplication narrows on its control argument."""
    assert_type(
        qmc.modmul_const(register, multiplier=2, modulus=15),
        qmc.Vector[qmc.Qubit],
    )
    assert_type(
        qmc.modmul_const(
            register,
            multiplier=2,
            modulus=15,
            control=control,
        ),
        tuple[qmc.Qubit, qmc.Vector[qmc.Qubit]],
    )


def _check_grover_types(
    oracle: qmc.Oracle,
    register: qmc.Vector[qmc.Qubit],
    numpy_integer: np.integer[Any],
    sympy_integer: sp.Integer,
    symbolic: sp.Expr,
) -> None:
    """Verify concrete, symbolic, and opaque-oracle Grover contracts."""
    assert_type(qmc.grover_iteration_count(3), int)
    assert_type(qmc.grover_iteration_count(numpy_integer), int)
    assert_type(qmc.grover_iteration_count(symbolic), sp.Expr)
    assert_type(qmc.grover_iteration_count(3, symbolic), sp.Expr)
    assert_type(qmc.grover_iteration_count(sympy_integer), sp.Expr)
    assert_type(qmc.grover_iteration_count(3, sympy_integer), sp.Expr)
    assert_type(
        qmc.grover_search(
            register,
            oracle,
            qmc.grover_iteration_count(3),
        ),
        qmc.Vector[qmc.Qubit],
    )
