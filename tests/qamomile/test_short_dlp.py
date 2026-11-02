"""Short-DLP composition across resource and SDK paths."""

# ruff: noqa: F821 -- ell and m are Qamomile dependent shape binders.
from __future__ import annotations

from importlib import import_module

import pytest

import qamomile as qm
from qamomile import resources as qr
from qamomile.persistence import serialize
from qamomile.stdlib import shor_order_finding_state


@qm.qkernel
def short_dlp_sample(
    generator: qm.Const[qm.Int],
    inverse_y: qm.Const[qm.Int],
    modulus: qm.Const[qm.Int],
    n: qm.Const[qm.Int],
    ell: qm.Const[qm.Int],
    m: qm.Const[qm.Int],
) -> tuple[qm.Shot[qm.UInt[ell]], qm.Shot[qm.UInt[m]]]:
    """Sample two Fourier frequencies using a shared modular-power target."""
    long_exponent = qm.qubits(ell)
    short_exponent = qm.qubits(m)
    target = qm.x(qm.qubits(n), target=0)
    long_exponent, target = shor_order_finding_state(
        long_exponent, target, generator, modulus, ell
    )
    short_exponent, target = shor_order_finding_state(
        short_exponent, target, inverse_y, modulus, m
    )
    long_frequency = qm.measure(qm.as_uint(long_exponent))
    short_frequency = qm.measure(qm.as_uint(short_exponent))
    _ignored = qm.measure(target)
    return long_frequency, short_frequency


@pytest.fixture(scope="module")
def short_dlp_module():
    """Build the regression source once for all target checks."""
    return short_dlp_sample.module()


@pytest.mark.parametrize(
    "backend",
    [
        "qiskit",
        pytest.param("cirq", marks=pytest.mark.cirq),
        pytest.param("quri_parts", marks=pytest.mark.quri_parts),
        pytest.param("braket", marks=pytest.mark.braket),
        pytest.param("cudaq", marks=pytest.mark.cudaq),
        pytest.param("hugr", marks=pytest.mark.hugr),
    ],
)
def test_short_dlp_source_across_targets(short_dlp_module, backend):
    """Check two secret signs and the one-bit boundary on the same source.

    This entrypoint only returns samples, so an expectation ABI does not
    apply. Direct HUGR validates the shared modular-power realization without
    requiring a local execution endpoint. Each executed source has five qubits
    and uses four shots.
    """
    dependencies = {
        "qiskit": "qiskit_aer",
        "cirq": "cirq",
        "quri_parts": "qulacs",
        "braket": "braket.circuits",
        "cudaq": "cudaq",
        "hugr": "hugr",
    }
    endpoints = {
        "qiskit": "Aer",
        "cirq": "CirqSimulator",
        "quri_parts": "Qulacs",
        "braket": "BraketLocal",
        "cudaq": "CudaqLocal",
    }
    pytest.importorskip(dependencies[backend])
    integration = import_module(f"qamomile.{backend}")
    if backend == "hugr":
        endpoint = None
        target = integration.Hugr()
    else:
        endpoint = getattr(integration, endpoints[backend])()
        target = endpoint.target

    for generator, secret, order, m, modulus, n in (
        (2, 1, 2, 1, 3, 2),
        (2, 0, 2, 1, 3, 2),
    ):
        program = short_dlp_module.specialize(
            "short_dlp_sample",
            generator=generator,
            inverse_y=pow(pow(generator, secret, modulus), -1, modulus),
            modulus=modulus,
            n=n,
            ell=2 * m,
            m=m,
        )
        if backend == "hugr":
            assert serialize(qm.compile(program, to=target))
            continue
        artifact = qm.compile(program, to=target)
        assert serialize(artifact)
        result = qm.execute(
            artifact, on=endpoint, shots=4, seed=0 if backend == "braket" else 9
        )
        expected = {
            ((1 << (2 * m)) * t // order, ((-secret * t) % order) * (1 << m) // order)
            for t in range(order)
        }
        assert sum(result.values.values()) == 4
        assert set(result.values) <= expected


def test_symbolic_resource_round_trip_and_realized_arithmetic():
    """Retain finite exact-arithmetic enclosures through persistence."""
    study = qr.project(qr.logical(short_dlp_sample), qr.GateCostModel())
    restored = qr.ResourceStudy.from_bytes(study.to_bytes())
    assert restored == study
    point = dict(n=2048, ell=2050, m=1025)
    assert restored.metric(qr.LOGICAL_PEAK_QUBITS).evaluate(
        point
    ).numeric_enclosure() == (
        5123,
        5123,
    )
    gate_bounds = (
        restored.metric(qr.portable_metric("gates")).evaluate(point).numeric_enclosure()
    )
    depth_bounds = (
        restored.metric(qr.portable_metric("gate_depth"))
        .evaluate(point)
        .numeric_enclosure()
    )
    scratch = (
        restored.metric(qr.portable_metric("clean_ancilla_qubits"))
        .evaluate(point)
        .numeric_enclosure()
    )
    assert gate_bounds[1] is not None
    assert depth_bounds[1] is not None
    assert scratch == (0, 4_105)


def test_short_dlp_rejects_nonunitary_modular_multiplier():
    """Non-coprime factors must not become a silently nonunitary oracle."""
    with pytest.raises(qm.ExecutionError, match="Q8050.*reversible permutation"):
        program = qm.specialize(
            short_dlp_sample,
            generator=0,
            inverse_y=2,
            modulus=3,
            n=2,
            ell=2,
            m=1,
        )
        qm.run(program, on=qm.Reference(), shots=1, seed=0)
