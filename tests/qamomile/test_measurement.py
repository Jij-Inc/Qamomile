"""Host-side estimator contracts and reusable ordinary measurement programs."""

from __future__ import annotations

import importlib
import math
from dataclasses import FrozenInstanceError

import pytest

import qamomile as qm
from qamomile.measurement import Estimator

SOURCE = """
import qamomile as qm
@qm.qkernel
def state(n: qm.Const[qm.Int], theta: qm.Param[qm.Real]) -> qm.Q[qm.Bits[n]]:
    q = qm.qubits(n)
    q = qm.ry(q, theta, target=0)
    return q
@qm.qkernel
def bell() -> qm.Q[qm.Bits[2]]:
    q = qm.h(qm.qubits(2), target=0)
    return qm.cx(q, control=0, target=1)
@qm.qkernel
def measured() -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.qubits(1))
@qm.qkernel
def quantum_input(q: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return q
"""


def test_reuses_compilation_and_keeps_configuration_immutable(monkeypatch):
    """Parameters and budgets change results without rebuilding target artifacts."""
    module = importlib.import_module("qamomile.measurement")
    compile_calls = []
    original = module.compile

    def compile_counted(program, *, to):
        compile_calls.append(to)
        return original(program, to=to)

    monkeypatch.setattr(module, "compile", compile_counted)
    estimator = Estimator(qm.build_source(SOURCE)["state"], qm.Z(0), n=2)
    assert compile_calls == []
    for theta, expected in [(0.0, 1.0), (math.pi, -1.0)]:
        result = estimator.estimate(
            on=qm.Reference(), parameters={"theta": theta}, shots=8, seed=2
        )
        assert result.value == expected
        assert result.standard_error == 0.0
        assert result.shots == 8
    assert len(compile_calls) == 1
    with pytest.raises(FrozenInstanceError):
        estimator.grouping = "independent"
    assert (
        "quantum.measure"
        in estimator._cache[0][1][0].report.resources._semantic_source().inspect()
    )


@pytest.mark.parametrize("grouping", ["independent", "qwc", "commuting"])
def test_total_budget_and_single_shot_uncertainty(grouping):
    """All strategies use a total budget and never claim known error from N=1."""
    estimator = Estimator(
        qm.build_source(SOURCE)["state"], qm.X(0) + qm.Z(0), n=1, grouping=grouping
    )
    with pytest.raises(qm.ExecutionError, match="Q7318"):
        estimator.estimate(on=qm.Reference(), parameters={"theta": 0.0}, shots=1)
    result = estimator.estimate(
        on=qm.Reference(), parameters={"theta": 0.0}, shots=3, seed=2
    )
    assert result.shots == 3
    assert tuple(group.shots for group in result.groups) == (2, 1)
    assert result.standard_error is None


def test_joint_covariance_is_preserved():
    """Bell Z0-Z1 has zero shot score despite uncertain individual parities."""
    estimator = Estimator(qm.build_source(SOURCE)["bell"], qm.Z(0) - qm.Z(1))
    result = estimator.estimate(on=qm.Reference(), shots=128, seed=42)
    assert len(result.groups) == 1
    assert result.value == result.standard_error == 0.0
    assert all(term.variance is not None and term.variance > 0 for term in result.terms)


@pytest.mark.parametrize("entry", ["measured", "quantum_input"])
def test_source_contract_checked_at_construction(entry):
    """Measurement entry points and quantum-input helpers cannot prepare states."""
    with pytest.raises(qm.TypeCheckError, match="Q7304"):
        Estimator(qm.build_source(SOURCE)[entry], qm.Z(0))


def test_observable_width_and_policies_checked_before_execution():
    """Invalid observables and policies are rejected during construction."""
    state = qm.build_source(SOURCE)["state"]
    with pytest.raises(qm.TypeCheckError, match="Q7302"):
        Estimator(state, qm.Z(2), n=2)
    with pytest.raises(qm.TypeCheckError):
        Estimator(state, 1j * qm.Z(0), n=2)
    with pytest.raises(TypeError, match="allocation"):
        Estimator(state, qm.Z(0), n=2, allocation="adaptive")


def test_identity_uses_statistics_without_execution(monkeypatch):
    """Constant observables need no shots but still validate their source."""
    module = importlib.import_module("qamomile.measurement")

    def reject(*args, **kwargs):
        raise AssertionError("identity must not compile or execute")

    monkeypatch.setattr(module, "compile", reject)
    monkeypatch.setattr(module, "run", reject)
    estimator = Estimator(
        qm.build_source(SOURCE)["bell"], qm.Hamiltonian.identity(0.25, num_qubits=2)
    )
    result = estimator.estimate(on=qm.Reference(), shots=10)
    assert result.value == 0.25
    assert result.standard_error == result.shots == 0


@pytest.mark.parametrize(
    ("module_name", "endpoint_name", "dependency"),
    [
        ("qamomile.compilation", "Reference", "qamomile"),
        ("qamomile.qiskit", "Aer", "qiskit_aer"),
        pytest.param("qamomile.cirq", "CirqSimulator", "cirq", marks=pytest.mark.cirq),
        pytest.param(
            "qamomile.quri_parts", "Qulacs", "qulacs", marks=pytest.mark.quri_parts
        ),
        pytest.param(
            "qamomile.braket", "BraketLocal", "braket", marks=pytest.mark.braket
        ),
        pytest.param("qamomile.cudaq", "CudaqLocal", "cudaq", marks=pytest.mark.cudaq),
        pytest.param("qamomile.hugr", "Selene", "selene_sim", marks=pytest.mark.hugr),
        ("qamomile.quration", "QurationLocal", "pyqret"),
    ],
)
def test_noncommuting_y_measurements_on_backends(
    module_name, endpoint_name, dependency
):
    """Generated X/Y/Z programs execute through the same public lifecycle."""
    pytest.importorskip(dependency)
    endpoint = getattr(importlib.import_module(module_name), endpoint_name)()
    state = qm.build_source("""
import qamomile as qm
@qm.qkernel
def state() -> qm.Q[qm.Bits[1]]:
    return qm.s(qm.h(qm.qubits(1)))
""")["state"]
    estimator = Estimator(state, qm.Y(0) + 0.01 * qm.X(0) + 0.01 * qm.Z(0))
    seed = 2
    if module_name == "qamomile.braket":
        with pytest.raises(qm.ExecutionError, match="Q8414"):
            estimator.estimate(on=endpoint, shots=96, seed=2)
        seed = 0
    result = estimator.estimate(on=endpoint, shots=96, seed=seed)
    assert result.value == pytest.approx(1.0, abs=0.02)
    assert result.shots == 96


def test_measured_program_persistence_and_target_cache(monkeypatch):
    """Saved measured definitions execute normally and distinct targets recompile."""
    module = importlib.import_module("qamomile.measurement")
    calls = []
    original = module.compile

    def counted(program, *, to):
        calls.append(to.fingerprint)
        return original(program, to=to)

    monkeypatch.setattr(module, "compile", counted)
    estimator = Estimator(
        qm.build_source(SOURCE)["bell"], qm.Z(0) * qm.Z(1), grouping="commuting"
    )
    estimator.estimate(on=qm.Reference(), shots=4)
    source = estimator._cache[0][1][0].report.resources._semantic_source()
    restored = qm.Program.from_bytes(source.to_bytes())
    result = qm.run(restored, on=qm.Reference(), shots=4, seed=2).result
    assert isinstance(result, qm.SampleResult)
    assert result.shots == 4
    from qamomile.qiskit import Aer

    for endpoint in [qm.Reference(), qm.Reference(), Aer(), Aer(), qm.Reference()]:
        assert estimator.estimate(on=endpoint, shots=4).value == 1.0
    assert len(calls) == 3
    assert len(estimator._cache) == 1


def test_internal_measurement_and_feedback_prepares_an_unconditional_state():
    """Ordinary measurement control remains within each source invocation."""
    state = qm.build_source("""
import qamomile as qm
@qm.qkernel
def state() -> qm.Q[qm.Bits[1]]:
    coin = qm.measure(qm.h(qm.qubits(1)))
    q = qm.qubits(1)
    if coin[0]:
        q = qm.x(q)
    return q
""")["state"]
    result = Estimator(state, qm.Z(0)).estimate(on=qm.Reference(), shots=128, seed=2)
    assert abs(result.value) < 0.4
    assert result.standard_error is not None and result.standard_error > 0


def test_constructor_does_not_generate_measurement_programs(monkeypatch):
    """Even grouping is deferred until an endpoint and budget are provided."""
    module = importlib.import_module("qamomile.measurement")

    def forbidden(*args, **kwargs):
        raise AssertionError("constructor must only validate")

    monkeypatch.setattr(module, "_group_measurement_program", forbidden)
    monkeypatch.setattr(Estimator, "_plan", forbidden)
    Estimator(qm.build_source(SOURCE)["state"], qm.X(0) + qm.Z(0), n=1)


@pytest.mark.parametrize("shots", [1, 3, 7, 32, 127])
def test_aer_native_estimator_exact_budget_and_uncertainty(shots, monkeypatch):
    """Native inference consumes the requested shots without compiling samples."""
    from qamomile.qiskit import Aer

    module = importlib.import_module("qamomile.measurement")

    def forbidden(*args, **kwargs):
        raise AssertionError("native route must not generate measured programs")

    monkeypatch.setattr(module, "_group_measurement_program", forbidden)
    estimator = Estimator(qm.build_source(SOURCE)["state"], 2 * qm.Z(0), n=1)
    result = estimator.estimate(
        on=Aer(), parameters={"theta": math.pi / 2}, shots=shots, seed=7
    )
    assert result.method == "qiskit.backend_estimator_v2"
    assert result.shots == shots
    if shots == 1:
        assert result.standard_error is None
    else:
        mean = result.terms[0].mean
        expected_variance = (1 - mean**2) * shots / (shots - 1)
        assert result.terms[0].variance == pytest.approx(expected_variance)
        assert result.standard_error == pytest.approx(
            2 * math.sqrt(expected_variance / shots)
        )


def test_aer_covariance_uses_sampling():
    """Shared Bell measurements retain covariance instead of native marginal errors."""
    from qamomile.qiskit import Aer

    estimator = Estimator(qm.build_source(SOURCE)["bell"], qm.Z(0) - qm.Z(1))
    result = estimator.estimate(on=Aer(), shots=64, seed=7)
    assert result.method == "sampling"
    assert result.value == result.standard_error == 0
    assert result.shots == 64


def test_native_execution_failure_is_not_retried(monkeypatch):
    """A selected native route cannot silently consume a second budget."""
    from qiskit.primitives import BackendEstimatorV2

    from qamomile.qiskit import Aer

    module = importlib.import_module("qamomile.measurement")

    def failed(*args, **kwargs):
        raise RuntimeError("provider failure")

    def forbidden(*args, **kwargs):
        raise AssertionError("no sampling retry")

    monkeypatch.setattr(BackendEstimatorV2, "run", failed)
    monkeypatch.setattr(module, "run", forbidden)
    with pytest.raises(qm.ExecutionError, match="provider failure"):
        Estimator(qm.build_source(SOURCE)["bell"], qm.Z(0)).estimate(on=Aer(), shots=4)


def test_aer_independent_budget_remainder_and_y_wire_order():
    """Native Pauli mapping uses source output order, including a permuted return."""
    from qamomile.qiskit import Aer

    state = qm.build_source("""
import qamomile as qm
@qm.qkernel
def state() -> qm.Q[qm.Bits[2]]:
    a = qm.x(qm.qubits(1))
    b = qm.s(qm.h(qm.qubits(1)))
    return qm.concat(b, a)
""")["state"]
    estimator = Estimator(state, qm.Y(0) + 2 * qm.Z(1), grouping="independent")
    result = estimator.estimate(on=Aer(), shots=7, seed=3)
    assert result.method == "qiskit.backend_estimator_v2"
    assert result.value == -1
    assert result.standard_error == 0
    assert result.shots == 7
    assert [group.shots for group in result.groups] == [4, 3]


def test_backend_summary_can_omit_measurement_details():
    """Shot accounting does not depend on a provider exposing individual groups."""
    from qamomile.expectation import ExpectationResult

    class SummaryEndpoint:
        name = "test.summary"
        target = qm.Reference()
        capabilities = qm.Reference().capabilities

        def launch(self, artifact, invocation):
            raise AssertionError("selected estimator does not launch samples")

        def _prepare_estimation(self, source, observable, plan, parameters):
            def execute():
                return ExpectationResult(
                    0.5,
                    None,
                    self.name,
                    sum(g.shots for g in plan),
                    method="test.summary",
                )

            return execute

    result = Estimator(qm.build_source(SOURCE)["bell"], qm.Z(0)).estimate(
        on=SummaryEndpoint(), shots=5
    )
    assert result.shots == 5
    assert result.terms is result.groups is result.standard_error is None


def test_backend_wrong_shot_metadata_does_not_fall_back(monkeypatch):
    """A provider contract violation is an execution error, not a route change."""
    from types import SimpleNamespace

    from qiskit.primitives import BackendEstimatorV2

    from qamomile.qiskit import Aer

    monkeypatch.setattr(
        BackendEstimatorV2,
        "run",
        lambda *a, **k: SimpleNamespace(
            result=lambda: [SimpleNamespace(metadata={"shots": 100})]
        ),
    )
    with pytest.raises(qm.ExecutionError, match="incorrect shot count"):
        Estimator(qm.build_source(SOURCE)["bell"], qm.Z(0)).estimate(on=Aer(), shots=3)


def test_aer_feedback_uses_common_route():
    """Native estimator exclusions preserve valid per-shot feedback programs."""
    from qamomile.qiskit import Aer

    state = qm.build_source("""
import qamomile as qm
@qm.qkernel
def state() -> qm.Q[qm.Bits[1]]:
    coin = qm.measure(qm.h(qm.qubits(1)))
    q = qm.qubits(1)
    if coin[0]:
        q = qm.x(q)
    return q
""")["state"]
    result = Estimator(state, qm.Z(0)).estimate(on=Aer(), shots=128, seed=7)
    assert result.method == "sampling"
    assert result.shots == 128
    assert abs(result.value) < 0.4
