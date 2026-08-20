"""Static type contracts for public APIs with input-dependent results."""

from __future__ import annotations

from typing import Any, assert_type

import numpy as np
import ommx.v1
import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.job import SampleResult
from qamomile.optimization.binary_model import BinarySampleSet
from qamomile.optimization.qsvt_eigenstate_filter import QSVTEigenstateFilterConverter


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


def _check_qsvt_eigenstate_filter_decode_types(
    converter: QSVTEigenstateFilterConverter,
    probe_result: SampleResult[tuple[list[int], list[int], list[int]]],
) -> None:
    """Verify the probe's three-register payload is accepted statically.

    The base converter decodes a flat ``list[int]`` per shot, but this
    converter's probe returns projector, signal, and system registers, so both
    decode entry points narrow the payload rather than falling back to ``Any``.
    """
    assert_type(
        converter.decode(probe_result),
        BinarySampleSet | ommx.v1.SampleSet,
    )
    assert_type(
        converter.decode_to_binary_sampleset(probe_result),
        BinarySampleSet,
    )
