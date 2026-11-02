"""Guard mixed-backend SDK launches without importing or running an SDK."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.qamomile import conftest as backend_config


@pytest.mark.parametrize(
    "parameters",
    [{"backend": "cudaq"}, {"module_name": "qamomile.cudaq"}],
)
@pytest.mark.parametrize(
    ("qubits", "instructions", "shots", "message"),
    [
        (18, 7803, 4, "limited to 10 qubits"),
        (5, 7803, 4, "bounded shallow example"),
        (5, 33, 129, "limited to 128 shots"),
        (5, 33, 4, None),
    ],
)
def test_mixed_backend_guard_checks_realized_work_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    parameters: dict[str, str],
    qubits: int,
    instructions: int,
    shots: int,
    message: str | None,
) -> None:
    """A five-source-qubit test cannot launch an oversized realized circuit."""

    launched = []

    class FakeEndpoint:
        def launch(self, artifact, invocation):
            launched.append((artifact, invocation))
            return "completed"

    def skip_dependency(name):
        assert name == "cudaq"
        return SimpleNamespace()

    monkeypatch.setitem(
        sys.modules,
        "qamomile.cudaq",
        SimpleNamespace(CudaqLocal=FakeEndpoint),
    )
    monkeypatch.setattr(pytest, "importorskip", skip_dependency)
    request = SimpleNamespace(
        path=Path("test_mixed_backend.py"),
        node=SimpleNamespace(
            callspec=SimpleNamespace(params=parameters),
            get_closest_marker=lambda _: None,
        ),
    )
    backend_config._bound_backend_execution_scale.__wrapped__(request, monkeypatch)
    metrics = {
        "qamomile.circuit.virtual_qubits": qubits,
        "qamomile.emitted.instructions": instructions,
        "qamomile.logical.gate_depth": 8,
    }

    def metric(identifier):
        upper = metrics[identifier]
        return SimpleNamespace(numeric_enclosure=lambda: (upper, upper))

    artifact = SimpleNamespace(report=SimpleNamespace(metric=metric))
    invocation = SimpleNamespace(shots=shots)
    if message is None:
        assert FakeEndpoint().launch(artifact, invocation) == "completed"
        assert len(launched) == 1
    else:
        with pytest.raises(AssertionError, match=message):
            FakeEndpoint().launch(artifact, invocation)
        assert launched == []


def test_unrelated_parameters_do_not_import_optional_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrelated parametrized backend value keeps Reference-only guarding."""

    def unexpected_import(_):
        pytest.fail("an unrelated test must not import an optional SDK")

    monkeypatch.setattr(pytest, "importorskip", unexpected_import)
    assert backend_config._backend_endpoint_types(
        "test_mixed_backend.py", backend="unrelated"
    ) == (backend_config.Reference,)
    assert backend_config._backend_endpoint_types("test_mixed_backend.py") == (
        backend_config.Reference,
    )


def test_short_dlp_optional_sdk_parameters_carry_their_selection_markers() -> None:
    """Default marker exclusion must remove optional short-DLP SDK cases."""

    from tests.qamomile.test_short_dlp import test_short_dlp_source_across_targets

    parameterization = next(
        mark
        for mark in test_short_dlp_source_across_targets.pytestmark
        if mark.name == "parametrize" and mark.args[0] == "backend"
    )
    marked = {
        parameter.values[0]: {mark.name for mark in parameter.marks}
        for parameter in parameterization.args[1]
        if not isinstance(parameter, str)
    }
    for backend in ("cirq", "quri_parts", "braket", "cudaq", "hugr"):
        assert backend in marked[backend]


def test_short_dlp_kernel_fixture_does_not_execute_or_import_aer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Building the short-DLP source must not execute a kernel or import Aer."""

    from tests.qamomile import test_short_dlp as short_dlp_tests

    def unexpected_execution(*args, **kwargs):
        pytest.fail("building the short-DLP source must not execute a kernel")

    monkeypatch.setattr(short_dlp_tests.qm, "run", unexpected_execution)
    for name in ("qamomile.qiskit", "qiskit", "qiskit_aer"):
        monkeypatch.setitem(sys.modules, name, None)
    module = short_dlp_tests.short_dlp_module.__wrapped__()
    program = module.specialize(
        "short_dlp_sample", generator=2, inverse_y=2, modulus=3, n=2, ell=2, m=1
    )
    assert program.to_bytes()
