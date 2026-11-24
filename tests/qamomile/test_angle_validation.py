"""Finite quantum angles are checked before backend execution."""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import replace

import pytest

import qamomile as qm
from qamomile.compilation import Invocation
from qamomile.ir import FrozenMapping
from qamomile.measurement import Estimator
from qamomile.persistence import deserialize, serialize

SOURCE = """
import qamomile as qm

@qm.qkernel
def scalar(theta: qm.Param[qm.Real]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.ry(qm.qubits(1), theta))

@qm.qkernel
def defaulted(theta: qm.Param[qm.Real] = 0.0) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.ry(qm.qubits(1), theta))

@qm.qkernel
def const_angle(theta: qm.Const[qm.Real]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.ry(qm.qubits(1), theta))

@qm.qkernel
def state(
    thetas: qm.Param[qm.Vector[qm.Real, 2]],
    scale: qm.Param[qm.Real] = 1.0,
) -> qm.Q[qm.Bits[1]]:
    return qm.ry(qm.qubits(1), thetas[1] * scale)

@qm.qkernel
def sample(
    thetas: qm.Param[qm.Vector[qm.Real, 2]],
    scale: qm.Param[qm.Real] = 1.0,
) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(state(thetas, scale))

@qm.qkernel
def non_angle(value: qm.Param[qm.Real]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.qubits(1))

@qm.qkernel
def reciprocal(theta: qm.Param[qm.Real]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.ry(qm.qubits(1), 1.0 / theta))

@qm.qkernel
def reciprocal_square(theta: qm.Param[qm.Real]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.ry(qm.qubits(1), 1.0 / (theta * theta)))

@qm.qkernel
def evolve(
    time: qm.Param[qm.Real],
    hamiltonian: qm.Const[qm.Observable],
) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.pauli_evolve(qm.qubits(1), hamiltonian, time))

@qm.qkernel
def indexed_evolve(
    time: qm.Param[qm.Real],
    hamiltonian: qm.Const[qm.Observable],
) -> qm.Shot[qm.Bits[1]]:
    index = qm.get_current_shot()
    return qm.measure(
        qm.pauli_evolve(qm.qubits(1), hamiltonian, time * (index + 1))
    )

@qm.qkernel
def conditional(
    theta: qm.Param[qm.Real],
    rotate: qm.Param[qm.Bool],
) -> qm.Shot[qm.Bits[1]]:
    q = qm.qubits(1)
    if rotate:
        q = qm.ry(q, theta)
    return qm.measure(q)
"""

NONFINITE_VALUES = [math.nan, math.inf, -math.inf]


@pytest.fixture(scope="module")
def angle_programs():
    """Build the same small angle consumers for every execution route."""
    return qm.build_source(SOURCE, module_name="angle_validation")


@pytest.fixture(
    params=[
        pytest.param(("qamomile", "Reference", "qamomile"), id="reference"),
        pytest.param(("qamomile.qiskit", "Aer", "qiskit_aer"), id="qiskit"),
        pytest.param(
            ("qamomile.quri_parts", "Qulacs", "qulacs"),
            id="quri_parts",
            marks=pytest.mark.quri_parts,
        ),
        pytest.param(
            ("qamomile.cirq", "CirqSimulator", "cirq"),
            id="cirq",
            marks=pytest.mark.cirq,
        ),
        pytest.param(
            ("qamomile.braket", "BraketLocal", "braket"),
            id="braket",
            marks=pytest.mark.braket,
        ),
        pytest.param(
            ("qamomile.cudaq", "CudaqLocal", "cudaq"),
            id="cudaq",
            marks=pytest.mark.cudaq,
        ),
    ],
)
def angle_endpoint(request):
    """Select an available local endpoint without contacting a provider."""
    module_name, endpoint_name, dependency = request.param
    pytest.importorskip(dependency)
    return getattr(importlib.import_module(module_name), endpoint_name)()


@pytest.fixture
def angle_seed(angle_endpoint):
    """Preserve Braket's unseeded contract for deterministic witnesses."""
    return None if angle_endpoint.name.startswith("braket.") else 7


@pytest.fixture
def angle_failure_pattern(angle_endpoint):
    """Allow Reference's existing angle diagnostic alongside the SDK check."""
    return (
        "finite|non-real circuit angle"
        if isinstance(angle_endpoint, qm.Reference)
        else "finite"
    )


class UnreachableEndpoint:
    """Fail if invalid angles reach any endpoint execution entrypoint."""

    def __init__(self, endpoint=None):
        """Reuse a circuit target contract without its executor."""
        if endpoint is None:
            from qamomile.qiskit import Aer

            endpoint = Aer()
        self.target = endpoint.target
        self.name = endpoint.name
        self.capabilities = endpoint.capabilities

    def launch(self, artifact, invocation):
        """Require single-invocation validation before endpoint launch."""
        raise AssertionError("invalid angles reached endpoint launch")

    def launch_batch(self, artifact, invocations):
        """Require every batch row to be checked before endpoint launch."""
        raise AssertionError("invalid angles reached endpoint batch launch")


@pytest.mark.parametrize("value", NONFINITE_VALUES)
@pytest.mark.parametrize("operation", ["execute", "run", "execute_batch"])
def test_execution_rejects_nonfinite_angle_before_endpoint(
    angle_programs, value, operation
):
    """Every public execution route rejects invalid angles before submission."""
    endpoint = UnreachableEndpoint()
    program = angle_programs.specialize("scalar")
    artifact = qm.compile(program, to=endpoint.target)
    with pytest.raises(qm.ExecutionError, match="finite"):
        if operation == "run":
            qm.run(program, on=endpoint, parameters={"theta": value}, shots=4)
        elif operation == "execute_batch":
            qm.execute_batch(
                artifact,
                on=endpoint,
                parameters=[{"theta": 0.0}, {"theta": value}],
                shots=4,
            )
        else:
            qm.execute(artifact, on=endpoint, parameters={"theta": value}, shots=4)


@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_estimator_rejects_nonfinite_vector_angle_before_endpoint(
    angle_programs, value
):
    """A nested angle cannot reach the common sampling endpoint."""
    estimator = Estimator(angle_programs["state"], qm.Z(0))
    with pytest.raises(qm.ExecutionError, match="finite"):
        estimator.estimate(
            on=UnreachableEndpoint(),
            parameters={"thetas": (0.0, value)},
            shots=4,
        )


def test_estimator_remains_reusable_after_invalid_angle(
    angle_programs, angle_endpoint, angle_seed, angle_failure_pattern
):
    """Real endpoints preserve finite results around rejected vector inputs."""
    estimator = Estimator(angle_programs["state"], qm.Z(0))
    assert (
        estimator.estimate(
            on=angle_endpoint,
            parameters={"thetas": (0.0, 0.0)},
            shots=8,
            seed=angle_seed,
        ).value
        == 1.0
    )
    for value in NONFINITE_VALUES:
        with pytest.raises(qm.ExecutionError, match=angle_failure_pattern):
            estimator.estimate(
                on=angle_endpoint,
                parameters={"thetas": (0.0, value)},
                shots=8,
                seed=angle_seed,
            )
    result = estimator.estimate(
        on=angle_endpoint,
        parameters={"thetas": (0.0, math.pi)},
        shots=8,
        seed=angle_seed,
    )
    assert result.value == -1.0
    assert result.standard_error == 0.0


def test_backends_reject_overflowed_angle(
    angle_programs, angle_endpoint, angle_seed, angle_failure_pattern
):
    """Finite inputs are rejected when their evaluated gate angle overflows."""
    artifact = qm.compile(angle_programs.specialize("sample"), to=angle_endpoint.target)
    with pytest.raises(qm.ExecutionError, match=angle_failure_pattern):
        qm.execute(
            artifact,
            on=angle_endpoint,
            parameters={"thetas": (0.0, sys.float_info.max), "scale": 2.0},
            shots=4,
            seed=angle_seed,
        )
    result = qm.execute(
        artifact,
        on=angle_endpoint,
        parameters={"thetas": (0.0, math.pi), "scale": 1.0},
        shots=4,
        seed=angle_seed,
    )
    assert result.counts == {(1,): 4}


def test_backends_reject_overflowed_pauli_evolution_angle(
    angle_programs, angle_endpoint, angle_seed, angle_failure_pattern
):
    """Pauli coefficients participate in the effective angle's finite check."""
    program = angle_programs.specialize("evolve", hamiltonian=2.0 * qm.Z(0))
    artifact = qm.compile(program, to=angle_endpoint.target)
    with pytest.raises(qm.ExecutionError, match=angle_failure_pattern):
        qm.execute(
            artifact,
            on=angle_endpoint,
            parameters={"time": sys.float_info.max},
            shots=4,
            seed=angle_seed,
        )
    result = qm.execute(
        artifact,
        on=angle_endpoint,
        parameters={"time": 0.0},
        shots=4,
        seed=angle_seed,
    )
    assert result.counts == {(0,): 4}


@pytest.mark.parametrize("use_native_pauli_evolution", [False, True])
def test_qiskit_pauli_angle_check_is_independent_of_realization(
    angle_programs, use_native_pauli_evolution
):
    """Native Pauli composites and expanded rotations reject the same overflow."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import Qiskit, to_circuit

    with pytest.raises(qm.ExecutionError, match="finite"):
        to_circuit(
            angle_programs["evolve"],
            hamiltonian=2.0 * qm.Z(0),
            parameters={"time": sys.float_info.max},
            target=Qiskit(use_native_pauli_evolution=use_native_pauli_evolution),
        )


@pytest.mark.parametrize(
    "realization", [None, "pauli.native"], ids=["automatic", "explicit-native"]
)
def test_qiskit_indexed_pauli_angles_are_checked_before_simulation(
    monkeypatch, angle_programs, realization
):
    """Shot-indexed Pauli evolution validates each effective angle before launch."""
    qiskit_aer = pytest.importorskip("qiskit_aer")
    from qamomile.lowering import RealizationOptions
    from qamomile.qiskit import Aer, Qiskit

    options = (
        RealizationOptions()
        if realization is None
        else RealizationOptions({"quantum.pauli_evolution": realization})
    )
    endpoint = Aer(realizations=options)
    emitted_stages = []
    original_lower = Qiskit.lower_circuit

    def record_emission(target, logical):
        lowered = original_lower(target, logical)
        emitted_stages.append(lowered.stages)
        return lowered

    monkeypatch.setattr(Qiskit, "lower_circuit", record_emission)
    program = angle_programs.specialize("indexed_evolve", hamiltonian=2.0 * qm.X(0))
    artifact = qm.compile(program, to=endpoint.target)
    result = qm.execute(
        artifact,
        on=endpoint,
        parameters={"time": math.pi / 4},
        shots=2,
        seed=7,
    )
    assert result.counts == {(1,): 1, (0,): 1}
    if realization == "pauli.native":
        assert emitted_stages
        assert all("realization:pauli.native" in stages for stages in emitted_stages)

    def reject_sdk_run(*args, **kwargs):
        raise AssertionError("invalid indexed angle reached the simulator")

    monkeypatch.setattr(qiskit_aer.AerSimulator, "run", reject_sdk_run)
    with pytest.raises(qm.ExecutionError, match="finite"):
        qm.execute(
            artifact,
            on=endpoint,
            parameters={"time": sys.float_info.max},
            shots=2,
            seed=7,
        )


@pytest.mark.parametrize("value", NONFINITE_VALUES)
@pytest.mark.parametrize("operation", ["to_circuit", "estimate"])
def test_qiskit_rejects_nonfinite_angle_before_sdk_binding(
    monkeypatch, angle_programs, value, operation
):
    """Circuit inspection and native estimation validate before SDK assignment."""
    pytest.importorskip("qiskit")
    from qamomile.qiskit import Aer, _integration, to_circuit

    def reject_sdk_binding(*args, **kwargs):
        raise AssertionError("invalid angle reached Qiskit parameter assignment")

    monkeypatch.setattr(_integration, "_bind_circuit_parameters", reject_sdk_binding)
    with pytest.raises(qm.ExecutionError, match="Q4482:.*finite"):
        if operation == "to_circuit":
            to_circuit(angle_programs["state"], parameters={"thetas": (0.0, value)})
        else:
            Estimator(angle_programs["state"], qm.Z(0)).estimate(
                on=Aer(), parameters={"thetas": (0.0, value)}, shots=4, seed=7
            )


@pytest.mark.parametrize(
    "entry, parameters",
    [
        ("non_angle", {"value": math.inf}),
        ("reciprocal", {"theta": math.inf}),
        ("reciprocal_square", {"theta": 1e308}),
        ("conditional", {"theta": math.inf, "rotate": False}),
    ],
)
def test_nonfinite_real_is_allowed_when_consumed_angle_is_finite(
    angle_programs, angle_endpoint, angle_seed, entry, parameters
):
    """Only evaluated quantum angles restrict the ordinary Real value domain."""
    result = qm.run(
        angle_programs[entry],
        on=angle_endpoint,
        parameters=parameters,
        shots=4,
        seed=angle_seed,
    ).result
    assert result.counts == {(0,): 4}


@pytest.mark.parametrize(
    "module_name, endpoint_name, dependency",
    [
        ("qamomile.qiskit", "Aer", "qiskit_aer"),
        pytest.param(
            "qamomile.quri_parts",
            "Qulacs",
            "qulacs",
            marks=pytest.mark.quri_parts,
        ),
    ],
)
def test_restored_artifact_retains_angle_checks(
    angle_programs, module_name, endpoint_name, dependency
):
    """Saved artifacts preserve both valid execution and prelaunch rejection."""
    pytest.importorskip(dependency)
    endpoint = getattr(importlib.import_module(module_name), endpoint_name)()
    artifact = qm.compile(angle_programs.specialize("sample"), to=endpoint.target)
    restored = deserialize(serialize(artifact))
    result = qm.execute(
        restored,
        on=endpoint,
        parameters={"thetas": (0.0, math.pi)},
        shots=4,
        seed=7,
    )
    assert result.counts == {(1,): 4}
    with pytest.raises(qm.ExecutionError, match="finite"):
        qm.execute(
            restored,
            on=UnreachableEndpoint(endpoint),
            parameters={"thetas": (0.0, math.inf)},
            shots=4,
        )


def test_missing_native_angle_validator_cannot_reach_endpoint(
    monkeypatch, angle_programs
):
    """A missing native checker cannot silently permit unchecked SDK execution."""
    from qamomile._native import compiler as native

    endpoint = UnreachableEndpoint()
    artifact = qm.compile(angle_programs.specialize("scalar"), to=endpoint.target)
    monkeypatch.delattr(native, "validate_circuit_angles")
    with pytest.raises(qm.ExecutionError, match="native.*unavailable"):
        qm.execute(artifact, on=endpoint, parameters={"theta": 0.0}, shots=4)


@pytest.mark.parametrize(
    "module_name, endpoint_name, sdk_name",
    [
        ("qamomile.qiskit", "Aer", "qiskit"),
        pytest.param("qamomile.cirq", "CirqSimulator", "cirq", marks=pytest.mark.cirq),
    ],
)
def test_public_circuit_binding_checks_angles_before_sdk(
    monkeypatch, angle_programs, module_name, endpoint_name, sdk_name
):
    """Direct public binding cannot bypass the shared artifact angle check."""
    sdk = pytest.importorskip(sdk_name)
    adapter = importlib.import_module(module_name)
    endpoint = getattr(adapter, endpoint_name)()
    artifact = qm.compile(angle_programs.specialize("sample"), to=endpoint.target)
    circuit = adapter.load_circuit(artifact)
    invocation = Invocation.create(parameters={"thetas": (0.0, math.inf)}, shots=4)

    def reject_sdk_binding(*args, **kwargs):
        raise AssertionError("invalid angle reached direct SDK binding")

    if sdk_name == "qiskit":
        monkeypatch.setattr(sdk.QuantumCircuit, "assign_parameters", reject_sdk_binding)
    else:
        monkeypatch.setattr(sdk, "resolve_parameters", reject_sdk_binding)
    with pytest.raises(qm.ExecutionError, match="finite"):
        adapter.bind_circuit(circuit, artifact, invocation)


@pytest.mark.parametrize(
    "module_name, endpoint_name, sdk_name",
    [
        ("qamomile.qiskit", "Aer", "qiskit"),
        pytest.param("qamomile.cirq", "CirqSimulator", "cirq", marks=pytest.mark.cirq),
    ],
)
def test_public_circuit_binding_fills_angle_defaults(
    angle_programs, module_name, endpoint_name, sdk_name
):
    """Public binding fills declared defaults before validating gate angles."""
    sdk = pytest.importorskip(sdk_name)
    adapter = importlib.import_module(module_name)
    endpoint = getattr(adapter, endpoint_name)()
    artifact = qm.compile(angle_programs.specialize("defaulted"), to=endpoint.target)
    bound = adapter.bind_circuit(
        adapter.load_circuit(artifact), artifact, Invocation.create()
    )
    if sdk_name == "qiskit":
        assert bound.num_parameters == 0
    else:
        assert not sdk.is_parameterized(bound)


def test_legacy_artifact_without_angle_plan_requires_recompilation(angle_programs):
    """Restoring older artifacts cannot silently disable angle validation."""
    endpoint = UnreachableEndpoint()
    artifact = qm.compile(angle_programs.specialize("scalar"), to=endpoint.target)
    legacy = replace(
        artifact,
        metadata=FrozenMapping.from_mapping(
            {
                name: value
                for name, value in artifact.metadata.entries
                if name != "qamomile.circuit_angle_validation.v1"
            }
        ),
    )
    restored = deserialize(serialize(legacy))
    with pytest.raises(qm.ExecutionError, match="Q4483:.*recompile"):
        qm.execute(restored, on=endpoint, parameters={"theta": 0.0}, shots=4)


@pytest.mark.parametrize(
    "module_name, target_name, dependency",
    [
        ("qamomile", "Reference", "qamomile"),
        ("qamomile.qiskit", "Qiskit", "qiskit"),
        pytest.param(
            "qamomile.quri_parts",
            "QuriParts",
            "quri_parts",
            marks=pytest.mark.quri_parts,
        ),
        pytest.param("qamomile.cirq", "Cirq", "cirq", marks=pytest.mark.cirq),
        pytest.param("qamomile.braket", "Braket", "braket", marks=pytest.mark.braket),
        pytest.param("qamomile.cudaq", "Cudaq", "cudaq", marks=pytest.mark.cudaq),
        pytest.param("qamomile.qir", "QIR", "pyqir", marks=pytest.mark.qir),
        pytest.param(
            "qamomile.quration", "Quration", "pyqret", marks=pytest.mark.quration
        ),
    ],
)
def test_nonfinite_const_angle_is_rejected(
    angle_programs, module_name, target_name, dependency
):
    """Static angle bindings cannot bypass the finite-angle contract."""
    pytest.importorskip(dependency)
    target = getattr(importlib.import_module(module_name), target_name)()
    pattern = (
        "finite|non-real circuit angle"
        if isinstance(target, qm.Reference)
        else "finite"
    )
    with pytest.raises(qm.QamomileError, match=pattern):
        program = angle_programs.specialize("const_angle", theta=math.inf)
        artifact = qm.compile(program, to=target)
        if isinstance(target, qm.Reference):
            qm.execute(artifact, on=target, shots=1)


@pytest.mark.quration
def test_quration_executes_a_finite_const_angle(angle_programs):
    """The static-only Quration endpoint still executes valid concrete angles."""
    pytest.importorskip("pyqret")
    from qamomile.quration import QurationLocal

    result = qm.run(
        angle_programs["const_angle"],
        theta=math.pi,
        on=QurationLocal(),
        shots=4,
        seed=7,
    ).result
    assert result.counts == {(1,): 4}
