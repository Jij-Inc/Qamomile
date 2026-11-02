"""Execute one later-bound block algorithm across supported targets."""

from __future__ import annotations

import importlib

import pytest

import qamomile as qm
from qamomile.persistence import serialize

from .test_stdlib import deferred_block_sample, exercise_deferred_block_backend


@pytest.mark.parametrize(
    ("dependency", "module_name", "endpoint_name"),
    [
        ("qiskit_aer", "qamomile.qiskit", "Aer"),
        pytest.param("cirq", "qamomile.cirq", "CirqSimulator", marks=pytest.mark.cirq),
        pytest.param(
            "qulacs", "qamomile.quri_parts", "Qulacs", marks=pytest.mark.quri_parts
        ),
        pytest.param("cudaq", "qamomile.cudaq", "CudaqLocal", marks=pytest.mark.cudaq),
        pytest.param(
            "braket.circuits",
            "qamomile.braket",
            "BraketLocal",
            marks=pytest.mark.braket,
        ),
    ],
)
def test_deferred_block_executes_on_sdk_targets(
    dependency: str,
    module_name: str,
    endpoint_name: str,
) -> None:
    """Every available SDK executes both specializations of one source."""

    pytest.importorskip(dependency)
    package = importlib.import_module(module_name)
    endpoint = getattr(package, endpoint_name)()
    exercise_deferred_block_backend(endpoint=endpoint)


@pytest.mark.hugr
def test_deferred_block_compiles_for_hugr() -> None:
    """HUGR emits a closed artifact although its local endpoint is suspended."""

    pytest.importorskip("hugr")
    from qamomile.hugr import Hugr

    from .test_stdlib import x_block_encoding

    artifact = qm.compile(deferred_block_sample, to=Hugr(), block=x_block_encoding)
    assert serialize(artifact)
