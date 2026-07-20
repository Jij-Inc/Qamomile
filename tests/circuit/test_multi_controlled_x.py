"""Cross-backend execution tests for the semantic multi-controlled X."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.transpiler.errors import EmitError


@qmc.qkernel
def _prepare_controls(
    controls: qmc.Vector[qmc.Qubit],
    bits: qmc.Vector[qmc.UInt],
    size: qmc.UInt,
) -> qmc.Vector[qmc.Qubit]:
    """Prepare a computational-basis control register.

    Args:
        controls (qmc.Vector[qmc.Qubit]): Zero-initialized control register.
        bits (qmc.Vector[qmc.UInt]): Computational-basis bits.
        size (qmc.UInt): Register width.

    Returns:
        qmc.Vector[qmc.Qubit]: Prepared control register.
    """
    for index in qmc.range(size):
        controls[index] = qmc.rx(
            controls[index],
            cast(qmc.Float, math.pi * bits[index]),
        )
    return controls


@qmc.qkernel
def _mcx_sample(
    size: qmc.UInt,
    bits: qmc.Vector[qmc.UInt],
) -> tuple[qmc.Vector[qmc.Bit], qmc.Bit]:
    """Prepare controls, apply semantic MCX, and measure all qubits.

    Args:
        size (qmc.UInt): Control-register width.
        bits (qmc.Vector[qmc.UInt]): Computational-basis control values.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Bit]: Control and target measurements.
    """
    controls = _prepare_controls(qmc.qubit_array(size, "controls"), bits, size)
    target = qmc.qubit("target")
    controls, target = qmc.mcx(controls, target)
    return qmc.measure(controls), qmc.measure(target)


@qmc.qkernel
def _mcx_expval(
    size: qmc.UInt,
    bits: qmc.Vector[qmc.UInt],
    observable: qmc.Observable,
) -> qmc.Float:
    """Apply semantic MCX and evaluate the target expectation value.

    Args:
        size (qmc.UInt): Control-register width.
        bits (qmc.Vector[qmc.UInt]): Computational-basis control values.
        observable (qmc.Observable): Target-qubit observable.

    Returns:
        qmc.Float: Target expectation value.
    """
    controls = _prepare_controls(qmc.qubit_array(size, "controls"), bits, size)
    target = qmc.qubit("target")
    _, target = qmc.multi_controlled_x(controls, target)
    return qmc.expval(target, observable)


@pytest.mark.parametrize("size,seed", [(1, 0), (2, 1), (3, 2), (5, 42)])
@pytest.mark.parametrize("all_enabled", [False, True])
def test_multi_controlled_x_cross_backend(
    sdk_transpiler: Any,
    size: int,
    seed: int,
    all_enabled: bool,
) -> None:
    """Random controls pass both sampling and expectation-value paths."""
    rng = np.random.default_rng(seed)
    bits = np.ones(size, dtype=int)
    if not all_enabled:
        bits[int(rng.integers(0, size))] = 0
    bindings = {"size": size, "bits": bits.tolist()}
    transpiler = sdk_transpiler.transpiler

    sample = (
        transpiler.transpile(_mcx_sample, bindings=bindings)
        .sample(transpiler.executor(), shots=16)
        .result()
    )
    assert sample.results == [((tuple(bits.tolist()), int(all_enabled)), 16)]

    actual = float(
        transpiler.transpile(
            _mcx_expval,
            bindings={**bindings, "observable": qm_o.Z(0)},
        )
        .run(transpiler.executor())
        .result()
    )
    expected = -1.0 if all_enabled else 1.0
    tolerance = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert np.isclose(actual, expected, atol=tolerance)


def test_multi_controlled_x_alias_is_same_qkernel() -> None:
    """The short and descriptive public APIs refer to one callable."""
    assert qmc.mcx is qmc.multi_controlled_x
    assert "mcx" in qmc.__all__
    assert "multi_controlled_x" in qmc.__all__


def test_multi_controlled_x_rejects_empty_control_register() -> None:
    """An empty control register fails with a stable positive-width diagnostic."""
    from qamomile.qiskit import QiskitTranspiler

    with pytest.raises(EmitError, match="strictly positive integer"):
        QiskitTranspiler().transpile(
            _mcx_sample,
            bindings={"size": 0, "bits": []},
        )
