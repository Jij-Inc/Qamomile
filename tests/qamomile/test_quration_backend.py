"""Execution and native FTQC compilation against the published PyQret SDK."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import pytest

import qamomile as qm
from qamomile.persistence import deserialize, export_payload, serialize
from qamomile.quration import (
    Quration,
    QurationLocal,
    compile_resources,
    load_circuit,
)

from .conftest import BASIS_SOURCE, BELL_SOURCE, DERIVED_RESULT_SOURCE
from .test_stdlib import (
    amplitude_two_sample,
    exercise_static_product_formula_backend,
    exercise_static_variational_backend,
    grover_sample,
    qft_frequency_three,
    qft_roundtrip,
    qpe_sample,
    qsvt_phase_sample,
)

pytestmark = pytest.mark.quration
pytest.importorskip("pyqret")


def test_quration_native_artifact_roundtrip_is_deterministic() -> None:
    """Native module time metadata cannot perturb portable artifact identity."""
    program = qm.build_source(BASIS_SOURCE).specialize("basis")
    first = qm.compile(program, to=Quration())
    second = qm.compile(program, to=Quration())
    assert serialize(first) == serialize(second)
    assert "created_at" not in json.loads(export_payload(first))["metadata"]
    restored = deserialize(serialize(first))
    assert qm.execute(restored, on=QurationLocal(), shots=4, seed=7).counts == {
        (1, 0, 0): 4
    }
    assert qm.quration.Quration is Quration


def test_quration_sdk_circuit_retains_owners() -> None:
    """A restored circuit remains executable after the artifact goes away."""
    from pyqret.runtime import Simulator, SimulatorConfig

    circuit = load_circuit(
        qm.compile(qm.build_source(BASIS_SOURCE).specialize("basis"), to=Quration())
    )
    gc.collect()
    assert Simulator(SimulatorConfig(seed=1), circuit).run()["c"] == [
        True,
        False,
        False,
    ]


def test_quration_samples_bell_with_repeatable_seed() -> None:
    """Native measurements preserve entanglement and seeded reproducibility."""
    program = qm.build_source(BELL_SOURCE).specialize("bell")
    artifact = qm.compile(program, to=Quration())
    first = qm.execute(artifact, on=QurationLocal(), shots=32, seed=23)
    second = qm.execute(artifact, on=QurationLocal(), shots=32, seed=23)
    assert first.counts == second.counts
    assert set(first.counts) == {(0, 0), (1, 1)}
    assert sum(first.counts.values()) == 32


def test_quration_uses_native_derived_result_plan() -> None:
    """Sampled Boolean expressions use the shared Rust postprocessor."""
    program = qm.build_source(DERIVED_RESULT_SOURCE).specialize("sample")
    expected = qm.run(program, on=qm.Reference(), shots=8, seed=2).result.values
    assert (
        qm.run(program, on=QurationLocal(), shots=8, seed=2).result.values == expected
    )


@pytest.mark.parametrize(
    "function",
    [
        qft_roundtrip,
        qft_frequency_three,
        amplitude_two_sample,
        grover_sample,
        qpe_sample,
        qsvt_phase_sample,
    ],
)
def test_quration_shared_static_library_witnesses(function: qm.QKernel) -> None:
    """Existing Fourier, preparation, Grover, QPE and QSVT sources execute."""
    expected = qm.run(function, on=qm.Reference(), shots=8, seed=2).result.counts
    actual = qm.run(function, on=QurationLocal(), shots=8, seed=2).result.counts
    assert actual == expected


def test_quration_shared_static_variational_and_evolution() -> None:
    """QAOA, FQAOA and product formulas reuse the same public source witnesses."""
    exercise_static_variational_backend(endpoint=QurationLocal(), seed=2)
    exercise_static_product_formula_backend(endpoint=QurationLocal(), seed=2)


def test_quration_expectation_uses_portable_pauli_readout() -> None:
    """Expectation estimation uses ordinary endpoint measurement execution."""
    from .test_stdlib import amplitude_two_state

    result = qm.expectation(
        amplitude_two_state, qm.Z(0), on=QurationLocal(), shots=8, seed=2
    )
    reference = qm.expectation(
        amplitude_two_state, qm.Z(0), on=qm.Reference(), shots=8, seed=2
    )
    assert result.value == pytest.approx(reference.value)


def test_quration_resource_compile_uses_native_machine_topology() -> None:
    """Quration produces real machine instructions and a native resource report."""
    from pyqret.backend import CompileOption, ScLsFixedV0Option

    artifact = qm.compile(
        qm.build_source(BELL_SOURCE).specialize("bell"), to=Quration()
    )
    original = serialize(artifact)
    option = CompileOption(
        sc_ls_fixed_v0_option=ScLsFixedV0Option(
            topology=str(Path(__file__).parent / "data/quration-plane.yaml")
        )
    )
    result = compile_resources(artifact, option)
    assert result.circuit.has_mf()
    assert result.compile_result.get_run_order()
    assert result.compile_info.gate_count > 0
    assert serialize(artifact) == original
    assert qm.execute(artifact, on=QurationLocal(), shots=4, seed=1).shots == 4


@pytest.mark.parametrize(
    "body",
    [
        "state = qm.reset(state)",
        "bit, state = qm.measure_z(state, target=0)\n    if bit:\n        state = qm.x(state, target=1)",
    ],
)
def test_quration_rejects_unsupported_runtime_effects(body: str) -> None:
    """Reset and shot control fail at compilation rather than changing semantics."""
    source = f"import qamomile as qm\n@qm.qkernel\ndef sample() -> qm.Bits[2]:\n    state = qm.qubits(2)\n    {body}\n    return qm.measure(state)\n"
    with pytest.raises(qm.ArtifactError, match="Q7015"):
        qm.compile(qm.build_source(source).specialize("sample"), to=Quration())


def test_quration_requires_specialized_rotation_parameters() -> None:
    """Param angles are rejected until a native launch-binding profile is added."""
    source = """
import qamomile as qm
@qm.qkernel
def sample(angle: qm.Param[qm.Real]) -> qm.Bits[1]:
    state = qm.qubits(1)
    state = qm.rx(state, angle)
    return qm.measure(state)
"""
    with pytest.raises(qm.ArtifactError, match="parameterized_execution"):
        qm.compile(qm.build_source(source).specialize("sample"), to=Quration())


@pytest.mark.parametrize("precision", [0, -1, 1, float("nan"), float("inf"), True])
def test_quration_rejects_invalid_precision(precision: float) -> None:
    """Synthesis tolerance must be finite and strictly between zero and one."""
    with pytest.raises(ValueError, match="rotation_precision"):
        Quration(rotation_precision=precision)


def test_quration_statevector_limit_precedes_simulator_allocation() -> None:
    """The endpoint rejects over-budget state width before invoking the SDK."""
    artifact = qm.compile(
        qm.build_source(BASIS_SOURCE).specialize("basis"), to=Quration()
    )
    with pytest.raises(qm.ExecutionError, match="Q8908"):
        qm.execute(artifact, on=QurationLocal(max_qubits=2), shots=1)


def test_quration_rejects_foreign_artifacts_and_missing_machine() -> None:
    """Native inspection and FTQC compilation have explicit input contracts."""
    from pyqret.backend import CompileOption

    program = qm.build_source(BELL_SOURCE).specialize("bell")
    with pytest.raises(qm.ExecutionError, match="Q8905"):
        load_circuit(qm.compile(program, to=qm.Reference()))
    with pytest.raises(ValueError, match="machine configuration"):
        compile_resources(qm.compile(program, to=Quration()), CompileOption())


@pytest.mark.parametrize(
    "gate,control_width",
    [
        (gate, 1)
        for gate in (
            "h",
            "x",
            "y",
            "z",
            "s",
            "sdg",
            "t",
            "tdg",
            "rx",
            "ry",
            "rz",
            "p",
            "global_phase",
        )
    ]
    + [(gate, 2) for gate in ("x", "y", "z")],
)
def test_quration_controlled_gate_preserves_full_operator(
    gate: str, control_width: int
) -> None:
    """Controlled SDK decompositions match every matrix element, including phase."""
    import numpy as np
    from pyqret.runtime import Simulator, SimulatorConfig
    from qiskit.quantum_info import Operator

    from qamomile.qiskit import to_circuit
    from qamomile.quration._payload import QurationExecutable

    angle = ", 0.713" if gate in {"rx", "ry", "rz", "p", "global_phase"} else ""
    source = f"""
import qamomile as qm
@qm.qkernel
def operation(q: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.{gate}(q{angle})
@qm.qkernel
def state() -> qm.Q[qm.Bits[{control_width + 1}]]:
    q = qm.qubits({control_width + 1})
    q = qm.ry(q, 0.913)
    q = qm.rx(q, -0.437, target={control_width})
    control, target = qm.split_at(q, {control_width})
    control, target = qm.control(operation, control, target)
    return qm.concat(control, target)
"""
    program = qm.build_source(source).specialize("state")
    # Quantum-returning components use circuit inspection without an execution ABI.
    payload = Quration().lower(program).payload
    assert isinstance(payload, QurationExecutable)
    native = payload.circuit
    simulator = Simulator(SimulatorConfig(seed=7, save_operation_matrix=True), native)
    simulator.run()
    actual = simulator.get_full_quantum_state().get_operation_matrix()
    # PyQret stores input basis states in rows; verify that convention against
    # the separately executed zero-input state before comparing all columns.
    np.testing.assert_allclose(
        actual[0], simulator.get_full_quantum_state().get_state_vector(), atol=1e-12
    )
    expected = Operator(to_circuit(program)).data
    np.testing.assert_allclose(actual.T, expected, atol=1e-12, rtol=1e-12)


def test_quration_rejects_unimplemented_control_profile() -> None:
    """Three coherent controls fail before partially emitting an SDK circuit."""
    source = """
import qamomile as qm
@qm.qkernel
def operation(q: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(q)
@qm.qkernel
def sample() -> qm.Bits[4]:
    control = qm.qubits(3)
    target = qm.qubits(1)
    control, target = qm.control(operation, control, target)
    return qm.measure(qm.concat(control, target))
"""
    with pytest.raises(qm.CapabilityError, match="Q8904"):
        qm.compile(qm.build_source(source).specialize("sample"), to=Quration())


def test_quration_accepts_full_seed_domain_and_typed_uint() -> None:
    """Seed normalization and the shared result codec preserve integer meaning."""
    source = """
import qamomile as qm
@qm.qkernel
def sample() -> qm.UInt[3]:
    state = qm.prepare((1, 0, 1))
    return qm.measure(qm.as_uint(state))
"""
    program = qm.build_source(source).specialize("sample")
    result = qm.run(program, on=QurationLocal(), shots=2, seed=(1 << 64) - 1).result
    assert result.values == {5: 2}


def test_quration_reorders_chronological_measurements_into_typed_result() -> None:
    """Nondestructive measurements precede reordered logical result fields."""
    source = """
import qamomile as qm
@qm.qkernel
def sample() -> tuple[qm.Bits[3], qm.Bool]:
    state = qm.prepare((1, 0, 0))
    first, state = qm.measure_z(state, target=0)
    state = qm.x(state, target=0)
    last = qm.measure(state)
    return last, first
"""
    result = qm.run(
        qm.build_source(source).specialize("sample"),
        on=QurationLocal(),
        shots=4,
        seed=2,
    ).result
    assert result.values == {((0, 0, 0), True): 4}


def test_quration_zero_layer_qaoa() -> None:
    """Execute the shared zero layer qaoa contract locally."""
    from qamomile.quration import QurationLocal

    from .test_library_compatibility import exercise_zero_layer_qaoa_backend

    exercise_zero_layer_qaoa_backend(QurationLocal())


def test_quration_approximate_qft() -> None:
    """Execute the shared approximate qft contract locally."""
    from qamomile.quration import QurationLocal

    from .test_library_compatibility import exercise_approximate_qft_backend

    exercise_approximate_qft_backend(QurationLocal())


def test_quration_classical_array_compatibility() -> None:
    """Execute the shared classical array contract locally."""
    from qamomile.quration import QurationLocal

    from .test_classical_array_compatibility import exercise_classical_array_backend

    exercise_classical_array_backend(QurationLocal())


def test_quration_qubit_selection() -> None:
    """Execute owned noncontiguous selection on this local SDK."""
    from qamomile.quration import QurationLocal

    from .test_qubit_selection import exercise_qubit_selection_backend

    exercise_qubit_selection_backend(QurationLocal(), seed=7)


def test_quration_rejects_exact_fractional_gas_control_width() -> None:
    """Exact encoding cannot bypass the SDK's existing controlled-gate limit."""
    from qamomile.optimization import BinaryModel, GASExperiment

    experiment = GASExperiment(
        BinaryModel.from_hubo({(4,): 0.25, (9,): 0.5}),
        threshold=0.2,
        approximate_real_coefficients=False,
    )
    with pytest.raises(qm.CapabilityError, match="Q8904.*controls"):
        experiment.compile(to=Quration())


def test_quration_rejects_keyed_runtime_parameters() -> None:
    """Packing dictionaries cannot bypass the target's Param capability limit."""
    from qamomile.parameters import ParameterMap

    from .test_parameter_map import SOURCE

    layout = ParameterMap(((8, 2), (3, 7)))
    program = qm.build_source(SOURCE).specialize("sample", keys=layout.indices)
    with pytest.raises(qm.ArtifactError, match="parameterized_execution"):
        qm.compile(program, to=Quration())


def test_quration_quantum_records() -> None:
    """Execute whole-record ownership transfers and coherent transforms."""
    from qamomile.quration import QurationLocal

    from .test_quantum_records import exercise_quantum_record_backend

    exercise_quantum_record_backend(QurationLocal(), seed=7)


def test_quration_recursion() -> None:
    """Execute self, mutual, branching, and higher-order recursive helpers."""
    from qamomile.quration import QurationLocal

    from .test_recursion import exercise_recursive_backend

    exercise_recursive_backend(QurationLocal(), seed=7)


def test_quration_sqrt() -> None:
    """Execute Const and launch-bound square roots through the native algebra."""
    from qamomile.quration import QurationLocal

    from .test_sqrt import exercise_sqrt_backend

    exercise_sqrt_backend(QurationLocal(), seed=7, parameterized=False)
