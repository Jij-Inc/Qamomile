"""End-to-end Cirq JSON and simulator integration tests."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Callable
from unittest.mock import patch

import numpy as np
import pytest

import qamomile as qm
from qamomile.effects import Requirement
from qamomile.persistence import deserialize, export_payload, serialize

from .conftest import (
    BASIS_SOURCE,
    EXTENDED_GATE_SOURCE,
    FEEDBACK_SOURCE,
    JOINT_PRIVATE_MEASUREMENT_SOURCE,
    MULTI_INPUT_CLEANUP_SOURCE,
    ROTATION_SOURCE,
    TRAILING_MEASUREMENT_SOURCE,
    exercise_boolean_lift_backend,
    exercise_broadcast_gate_backend,
    exercise_collection_loops_backend,
    exercise_derived_result_backend,
    exercise_dynamic_classical_index_backend,
    exercise_dynamic_qubit_selection_backend,
    exercise_multi_owner_dispatch_backend,
    exercise_normalized_model_backend,
    exercise_structural_math_backend,
    exercise_symbolic_observable_backend,
    exercise_uint_measurement_backend,
)
from .test_documented_algorithms import (
    exercise_bit_flip_code_backend,
    exercise_classification_backend,
    exercise_documented_elliptic_curve_dlog_backend,
    exercise_documented_factorized_chemistry_backend,
    exercise_documented_fault_tolerant_data_loading_backend,
    exercise_documented_gf2_arithmetic_backend,
    exercise_documented_guided_hamiltonian_backend,
    exercise_documented_kikuchi_backend,
    exercise_documented_phase_gradient_backend,
    exercise_documented_reversible_arithmetic_backend,
    exercise_documented_reversible_indexing_backend,
    exercise_documented_reversible_trigonometry_backend,
    exercise_documented_windowed_qpe_backend,
    exercise_hqnn_backend,
    exercise_hydrogen_vqe_backend,
    exercise_mottonen_backend,
    exercise_multidimensional_qft_backend,
    exercise_option_pricing_backend,
    exercise_pce_backend,
    exercise_qemcmc_backend,
    exercise_qsci_backend,
    exercise_rabi_backend,
    exercise_steane_code_backend,
)
from .test_gas import exercise_gas_backend
from .test_stdlib import (
    amplitude_two_sample,
    amplitude_two_state,
    exercise_first_class_multi_exponentiation_backend,
    exercise_fixed_qpe_backend,
    exercise_functional_block_producers_backend,
    exercise_modular_arithmetic_backend,
    exercise_multi_exponentiation_realization_backend,
    exercise_nary_tensor_backend,
    exercise_pauli_block_backend,
    exercise_product_formula_backend,
    exercise_regev_factoring_backend,
    exercise_register_modulus_backend,
    exercise_runtime_amplitude_backend,
    exercise_runtime_matrix_backend,
    exercise_runtime_qsvt_backend,
    exercise_shor_order_finding_backend,
    exercise_sparse_lookup_backend,
    exercise_uniformly_controlled_backend,
    exercise_variational_backend,
    exercise_windowed_qpe_backend,
    exercise_xor_lookup_backend,
    grover_sample,
    grover_state,
    lcu_qsvt_sample,
    lcu_qsvt_state,
    multi_exponentiation_baseline,
    multi_exponentiation_product_tree,
    qft_frequency_three,
    qft_roundtrip,
    qpe_sample,
    qpe_state,
    qsvt_phase_sample,
    qsvt_phase_state,
    reciprocal_x_sample,
    reciprocal_x_state,
)

pytestmark = pytest.mark.cirq

pytest.importorskip("cirq")


def test_cirq_executes_signal_composition() -> None:
    """Execute shared region phases and editable signal schedules."""
    from qamomile.cirq import CirqSimulator

    from .test_signals import exercise_signals_backend

    exercise_signals_backend(endpoint=CirqSimulator())


def test_cirq_executes_standard_optimized_qcore() -> None:
    """Cirq executes the shared exact QCore optimization witness."""

    from qamomile.cirq import CirqSimulator

    from .test_standard_optimization import exercise_standard_optimization_backend

    exercise_standard_optimization_backend(endpoint=CirqSimulator())


def test_cirq_artifact_is_portable_and_preserves_source_bit_order() -> None:
    """Cirq JSON round-trips and executes asymmetric source-order bits."""

    from qamomile.cirq import Cirq, CirqSimulator, load_circuit

    program = qm.build_source(BASIS_SOURCE).specialize("basis")
    artifact = qm.compile(program, to=Cirq())
    repeated = qm.compile(program, to=Cirq())
    restored = deserialize(serialize(artifact))
    with patch(
        "subprocess.run",
        side_effect=AssertionError("worker used"),
    ):
        result = qm.execute(restored, on=CirqSimulator(), shots=16, seed=4)

    assert artifact.format == "cirq.json.v1"
    assert export_payload(artifact) == export_payload(repeated)
    assert len(load_circuit(restored).all_qubits()) == 3
    assert result.counts == {(1, 0, 0): 16}
    assert result.worker_pid is None


def test_cirq_executes_grover_adaptive_search() -> None:
    """Cirq executes GAS composed from ordinary reusable qkernels."""

    from qamomile.cirq import CirqSimulator

    exercise_gas_backend(endpoint=CirqSimulator(), seed=2)


def test_cirq_executes_derived_results_and_aggregate_parameters() -> None:
    """Cirq shares Rust-owned result and aggregate parameter projections."""

    from qamomile.cirq import CirqSimulator

    endpoint = CirqSimulator()
    exercise_derived_result_backend(endpoint=endpoint, seed=2)
    exercise_broadcast_gate_backend(endpoint=endpoint, seed=2)
    exercise_uint_measurement_backend(endpoint=endpoint, seed=2)
    exercise_dynamic_classical_index_backend(endpoint=endpoint, seed=2)


def test_cirq_executes_boolean_lifts_and_expectation_plans() -> None:
    """Cirq executes one shared sampling and grouped expectation witness."""

    from qamomile.cirq import CirqSimulator

    exercise_boolean_lift_backend(endpoint=CirqSimulator(), seed=2)


def test_cirq_executes_symbolic_observable_recipe() -> None:
    """Cirq executes an Observable assembled from Const coefficient data."""

    from qamomile.cirq import CirqSimulator

    exercise_symbolic_observable_backend(endpoint=CirqSimulator(), seed=2)


def test_cirq_executes_a_normalized_optimization_model() -> None:
    """Cirq executes the shared Rust-normalized model witness."""

    from qamomile.cirq import CirqSimulator

    exercise_normalized_model_backend(endpoint=CirqSimulator(), seed=2)


def test_cirq_executes_shared_static_control_witnesses() -> None:
    """Cirq executes the common loop, selector, math, and dispatch witnesses."""

    from qamomile.cirq import CirqSimulator

    endpoint = CirqSimulator()
    exercise_collection_loops_backend(endpoint=endpoint, seed=2)
    exercise_dynamic_qubit_selection_backend(endpoint=endpoint, seed=2)
    exercise_structural_math_backend(endpoint=endpoint, seed=2)
    exercise_multi_owner_dispatch_backend(endpoint=endpoint, seed=2)


@pytest.mark.parametrize(
    "exercise",
    [
        exercise_classification_backend,
        exercise_hydrogen_vqe_backend,
        exercise_qsci_backend,
        exercise_rabi_backend,
        exercise_mottonen_backend,
        exercise_multidimensional_qft_backend,
        exercise_hqnn_backend,
        exercise_option_pricing_backend,
        exercise_pce_backend,
        exercise_qemcmc_backend,
        exercise_bit_flip_code_backend,
        exercise_steane_code_backend,
    ],
    ids=[
        "classification",
        "hydrogen-vqe",
        "qsci",
        "hamiltonian-simulation",
        "mottonen",
        "multidimensional-qft",
        "hybrid-qnn",
        "option-pricing",
        "pce",
        "qemcmc",
        "bit-flip-code",
        "steane-code",
    ],
)
def test_cirq_executes_documented_application_matrix(
    exercise: Callable[..., None],
) -> None:
    """Cirq executes each shared documented application witness."""

    from qamomile.cirq import CirqSimulator

    exercise(endpoint=CirqSimulator(), seed=2)


@pytest.mark.parametrize(
    "exercise",
    [
        exercise_variational_backend,
        exercise_product_formula_backend,
        exercise_uniformly_controlled_backend,
        exercise_xor_lookup_backend,
        exercise_sparse_lookup_backend,
        exercise_runtime_matrix_backend,
        exercise_runtime_qsvt_backend,
        exercise_pauli_block_backend,
        exercise_functional_block_producers_backend,
        exercise_modular_arithmetic_backend,
        exercise_shor_order_finding_backend,
        exercise_regev_factoring_backend,
        exercise_register_modulus_backend,
    ],
    ids=[
        "variational",
        "product-formula",
        "uniformly-controlled",
        "xor-lookup",
        "sparse-lookup",
        "runtime-matrix",
        "runtime-qsvt",
        "pauli-block",
        "block-producers",
        "modular-arithmetic",
        "shor-order-finding",
        "regev-factoring",
        "register-modulus",
    ],
)
def test_cirq_executes_stdlib_backend_matrix(
    exercise: Callable[..., None],
) -> None:
    """Cirq executes each shared standard-library backend witness."""

    from qamomile.cirq import CirqSimulator

    exercise(endpoint=CirqSimulator(), seed=2)


def test_cirq_executes_multi_exponentiation_callable_realization() -> None:
    """Cirq executes the shared source after callable replacement."""

    from qamomile.cirq import CirqSimulator

    options = qm.RealizationOptions(
        {multi_exponentiation_baseline: multi_exponentiation_product_tree}
    )
    exercise_multi_exponentiation_realization_backend(
        endpoint=CirqSimulator(realizations=options), seed=2
    )


def test_cirq_executes_first_class_multi_exponentiation() -> None:
    """Cirq executes the semantic primitive through its independent realization."""

    from qamomile.cirq import CirqSimulator

    exercise_first_class_multi_exponentiation_backend(endpoint=CirqSimulator(), seed=2)


def test_cirq_executes_functional_stdlib_sampling_and_expectation() -> None:
    """Cirq samples and estimates shared QFT, QPE, Grover, and QSVT bodies."""

    from qamomile.cirq import CirqSimulator

    endpoint = CirqSimulator()
    exercise_nary_tensor_backend(endpoint=endpoint, seed=2)
    assert qm.run(qft_roundtrip, on=endpoint, shots=4, seed=2).result.counts == {
        (1, 0, 1): 4
    }
    assert qm.run(
        qft_frequency_three,
        on=endpoint,
        shots=4,
        seed=2,
    ).result.counts == {(1, 1, 0): 4}
    assert qm.run(qpe_sample, on=endpoint, shots=4, seed=2).result.counts == {
        (1, 0, 1): 4
    }
    assert (
        qm.expectation(qpe_state, qm.Z(0), on=endpoint, shots=4, seed=2).value == -1.0
    )
    exercise_fixed_qpe_backend(endpoint=endpoint, seed=2)
    exercise_windowed_qpe_backend(endpoint=endpoint, seed=2)
    exercise_documented_windowed_qpe_backend(endpoint=endpoint, seed=2)
    exercise_documented_reversible_arithmetic_backend(endpoint=endpoint, seed=2)
    exercise_documented_gf2_arithmetic_backend(endpoint=endpoint, seed=2)
    exercise_documented_phase_gradient_backend(endpoint=endpoint, seed=2)
    exercise_documented_fault_tolerant_data_loading_backend(endpoint=endpoint, seed=2)
    exercise_documented_reversible_indexing_backend(endpoint=endpoint, seed=2)
    exercise_documented_reversible_trigonometry_backend(endpoint=endpoint, seed=2)
    exercise_documented_kikuchi_backend(endpoint=endpoint, seed=2)
    exercise_documented_guided_hamiltonian_backend(endpoint=endpoint, seed=2)
    assert qm.run(grover_sample, on=endpoint, shots=4, seed=2).result.counts == {
        (1, 1): 4
    }
    assert (
        qm.expectation(grover_state, qm.Z(0), on=endpoint, shots=4, seed=2).value
        == -1.0
    )


def test_cirq_executes_documented_qualtran_gap_examples() -> None:
    """Cirq executes the bounded ECC and factorized-chemistry witnesses."""

    from qamomile.cirq import CirqSimulator

    endpoint = CirqSimulator()
    exercise_documented_elliptic_curve_dlog_backend(endpoint=endpoint, seed=2)
    exercise_documented_factorized_chemistry_backend(endpoint=endpoint, seed=2)
    for sample, state in (
        (qsvt_phase_sample, qsvt_phase_state),
        (lcu_qsvt_sample, lcu_qsvt_state),
    ):
        assert qm.run(sample, on=endpoint, shots=4, seed=2).result.counts == {(1, 0): 4}
        assert (
            qm.expectation(
                state,
                qm.Z(0),
                on=endpoint,
                shots=4,
                seed=2,
            ).value
            == -1.0
        )
    assert qm.run(reciprocal_x_sample, on=endpoint, shots=4, seed=2).result.counts == {
        (0, 0, 1): 4
    }
    assert (
        qm.expectation(
            reciprocal_x_state,
            qm.Z(2),
            on=endpoint,
            shots=4,
            seed=2,
        ).value
        == -1.0
    )


def test_cirq_executes_functional_amplitude_encoding() -> None:
    """Cirq samples and estimates the multilevel Gray-walk body."""

    from qamomile.cirq import CirqSimulator

    endpoint = CirqSimulator()
    assert qm.run(amplitude_two_sample, on=endpoint, shots=4, seed=2).result.counts == {
        (1, 1): 4
    }
    assert (
        qm.expectation(amplitude_two_state, qm.Z(0), on=endpoint, shots=4, seed=2).value
        == -1.0
    )


def test_cirq_reuses_runtime_parametric_amplitude_artifact() -> None:
    """Cirq binds aggregate coordinates through the common scalar ABI view."""

    from qamomile.cirq import Cirq, CirqSimulator

    exercise_runtime_amplitude_backend(
        endpoint=CirqSimulator(),
        target=Cirq(),
        seed=2,
    )


def test_cirq_executes_extended_and_parameterized_gate_families() -> None:
    """Cirq emits phase, rotation, ZZ, Toffoli, and launch expressions."""

    from qamomile.cirq import CirqSimulator

    endpoint = CirqSimulator()
    extended = qm.build_source(EXTENDED_GATE_SOURCE).specialize("extended_gates")
    rotation = qm.build_source(ROTATION_SOURCE).specialize("rotate")

    assert qm.run(extended, on=endpoint, shots=4, seed=2).result.counts == {
        (1, 1, 1): 4
    }
    assert qm.run(
        rotation,
        on=endpoint,
        parameters={"theta": 3.141592653589793},
        shots=4,
        seed=2,
    ).result.counts == {(1,): 4}


def test_cirq_zero_noise_matches_state_vector_execution() -> None:
    """A zero-strength channel preserves exact simulator results."""

    from qamomile.cirq import CirqSimulator, DepolarizingNoise

    program = qm.build_source(BASIS_SOURCE).specialize("basis")
    clean = qm.run(program, on=CirqSimulator(), shots=8, seed=7).result
    noisy = qm.run(
        program,
        on=CirqSimulator(DepolarizingNoise(0.0)),
        shots=8,
        seed=7,
    ).result

    assert clean.counts == noisy.counts == {(1, 0, 0): 8}


def test_cirq_preserves_cleanup_and_private_measurement_semantics() -> None:
    """Cirq retains cleanup and private measurements in closed segments."""

    from qamomile.cirq import CirqSimulator

    endpoint = CirqSimulator()
    cleanup = qm.build_source(MULTI_INPUT_CLEANUP_SOURCE).specialize(
        "multi_input_cleanup"
    )
    trailing = qm.build_source(TRAILING_MEASUREMENT_SOURCE).specialize(
        "trailing_measurement"
    )

    assert qm.run(cleanup, on=endpoint, shots=4, seed=7).result.counts == {(1,): 4}
    record = qm.run(trailing, on=endpoint, shots=4, seed=3)
    assert record.artifact.format == "qamomile.shot-execution.v1"
    assert record.artifact.result_abi.native_width == 1
    assert record.artifact.result_abi.source_order == (0,)
    assert record.result.counts == {(0,): 4}
    assert dict(record.result.execution_statistics.entries) == {
        "logical_shots": 4,
        "physical_segment_shots": 8,
        "submissions": 2,
    }


def test_cirq_aggregates_private_measurement_rows_after_result_projection() -> None:
    """Private measurement variation cannot discard public sample counts."""

    from qamomile.cirq import Cirq
    from qamomile.cirq._integration import _decode_cirq_result
    from qamomile.compilation import Invocation

    artifact = qm.compile(
        qm.build_source(JOINT_PRIVATE_MEASUREMENT_SOURCE).specialize(
            "joint_private_measurement"
        ),
        to=Cirq(),
    )
    assert artifact.format == "cirq.json.v1"
    assert artifact.result_abi.native_width == 2
    assert artifact.result_abi.source_order == (0,)
    native = SimpleNamespace(
        measurements={
            "m0": np.array([[0], [0], [0], [0]], dtype=np.int8),
            "m1": np.array([[0], [1], [0], [1]], dtype=np.int8),
        }
    )

    result = _decode_cirq_result(
        native,
        artifact,
        Invocation.create(shots=4),
        backend="cirq.test",
    )

    assert result.counts == {(0,): 4}


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan")])
def test_cirq_rejects_invalid_noise_probability(probability: float) -> None:
    """Portable depolarizing noise rejects values outside its closed domain."""

    from qamomile.cirq import DepolarizingNoise

    with pytest.raises(ValueError, match="between zero and one"):
        DepolarizingNoise(probability)


def test_cirq_accepts_closed_noise_probability_boundaries() -> None:
    """Portable depolarizing noise includes both probability endpoints."""

    from qamomile.cirq import DepolarizingNoise

    assert DepolarizingNoise(0).probability == 0.0
    assert DepolarizingNoise(1).probability == 1.0


def test_cirq_rejects_measurement_control_before_emission() -> None:
    """Static Cirq capability claims fail closed on Shot-time control."""

    from qamomile.cirq import Cirq

    program = qm.build_source(FEEDBACK_SOURCE).specialize("feedback")
    assert Requirement.DYNAMIC_BRANCH in program.entry.requirements
    with pytest.raises(qm.ArtifactError, match="Q7015"):
        qm.compile(program, to=Cirq())


def test_cirq_rejects_corrupt_json_artifact() -> None:
    """Cirq loading rejects a payload that no longer contains valid JSON."""

    from qamomile.cirq import Cirq, load_circuit

    artifact = qm.compile(
        qm.build_source(BASIS_SOURCE).specialize("basis"),
        to=Cirq(),
    )
    with pytest.raises(qm.ArtifactError, match="Q7001"):
        load_circuit(replace(artifact, payload=b"not-json"))


def test_cirq_google_qvm_automatically_routes_and_executes() -> None:
    """Willow QVM places a shared circuit and preserves logical result order."""

    pytest.importorskip("cirq_google")
    from qamomile.cirq import CirqSampler

    endpoint = CirqSampler.from_google_qvm(processor_id="willow_pink")
    program = qm.build_source(BASIS_SOURCE).specialize("basis")
    record = qm.run(program, on=endpoint, shots=4, seed=0)

    assert "cirq-routing" in record.artifact.report.stages
    assert record.result.counts == {(1, 0, 0): 4}


def test_cirq_google_qvm_loads_calibration_derived_noise() -> None:
    """Google's bundled calibrated QVM profile remains a first-class choice."""

    pytest.importorskip("cirq_google")
    from qamomile.cirq import GoogleQVMNoise, google_qvm

    endpoint = google_qvm(
        processor_id="willow_pink",
        noise=GoogleQVMNoise.CALIBRATED,
    )
    assert endpoint.name == "cirq.google-qvm.willow_pink"


def test_cirq_google_qvm_snapshots_and_restores_native_jobs() -> None:
    """Google QVM jobs restore from secret-free program and job identities."""

    pytest.importorskip("cirq_google")
    from qamomile.cirq import CirqSampler

    endpoint = CirqSampler.from_google_qvm(processor_id="willow_pink")
    program = qm.build_source(BASIS_SOURCE).specialize("basis")
    artifact = qm.compile(program, to=endpoint.target)
    job = qm.submit(artifact, on=endpoint, shots=4, seed=0)
    snapshot = job.snapshot()
    restored = endpoint.restore(snapshot, artifact=artifact)

    assert snapshot.kind is qm.ExecutionSnapshotKind.REMOTE
    assert restored.job_id == job.job_id
    assert restored.result().counts == {(1, 0, 0): 4}


def test_cirq_google_engine_keeps_configuration_at_endpoint_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Google Engine construction retains only live endpoint configuration."""

    cirq_google = pytest.importorskip("cirq_google")
    from qamomile.cirq import CirqProcessor, google_engine

    class Processor:
        """Provide the minimal processor surface used during construction."""

        processor_id = "processor"

        def get_device(self) -> object:
            """Return an inert target device."""

            return object()

        def run_sweep(self, **kwargs: object) -> object:
            """Reject accidental submission during endpoint construction."""

            raise AssertionError(kwargs)

    class Engine:
        """Resolve one deterministic fake processor."""

        def __init__(self, *, project_id: str) -> None:
            """Retain the requested project ID."""

            assert project_id == "project"

        def get_processor(self, processor_id: str) -> Processor:
            """Return the fake processor for the expected ID."""

            assert processor_id == "processor"
            return Processor()

    class Sampler:
        """Retain ProcessorSampler options without provider access."""

        def __init__(self, **kwargs: object) -> None:
            """Validate the selected run configuration."""

            assert kwargs["run_name"] == "run"
            assert kwargs["snapshot_id"] == ""
            assert kwargs["device_config_name"] == "config"

        def run(self, *args: object, **kwargs: object) -> object:
            """Reject accidental sampling during endpoint construction."""

            raise AssertionError((args, kwargs))

    monkeypatch.setattr(cirq_google, "Engine", Engine)
    monkeypatch.setattr(cirq_google, "ProcessorSampler", Sampler)

    endpoint = google_engine(
        "project",
        "processor",
        run_name="run",
        device_config_name="config",
    )
    assert isinstance(endpoint, CirqProcessor)
    assert endpoint.name == "cirq.google-engine.processor"
    with pytest.raises(ValueError, match="mutually exclusive"):
        google_engine(
            "project",
            "processor",
            run_name="run",
            snapshot_id="snapshot",
            device_config_name="config",
        )


@pytest.mark.parametrize("choice", ["add.increments", "add.qft", "add.ripple"])
def test_cirq_executes_late_realizations(choice: str) -> None:
    """One shared arithmetic source survives exact alternative realizations."""
    pytest.importorskip("cirq")
    from qamomile.cirq import CirqSimulator
    from qamomile.lowering import RealizationOptions

    from .test_realizations import exercise_realization_backend

    exercise_realization_backend(
        CirqSimulator(realizations=RealizationOptions({"quantum.add_into": choice}))
    )


def test_cirq_zero_layer_qaoa() -> None:
    """Execute the shared zero layer qaoa contract locally."""
    from qamomile.cirq import CirqSimulator

    from .test_library_compatibility import exercise_zero_layer_qaoa_backend

    exercise_zero_layer_qaoa_backend(CirqSimulator())


def test_cirq_approximate_qft() -> None:
    """Execute the shared approximate qft contract locally."""
    from qamomile.cirq import CirqSimulator

    from .test_library_compatibility import exercise_approximate_qft_backend

    exercise_approximate_qft_backend(CirqSimulator())


def test_cirq_classical_array_compatibility() -> None:
    """Execute the shared classical array contract locally."""
    from qamomile.cirq import CirqSimulator

    from .test_classical_array_compatibility import exercise_classical_array_backend

    exercise_classical_array_backend(CirqSimulator())


def test_cirq_qubit_selection() -> None:
    """Execute owned noncontiguous selection on this local SDK."""
    from qamomile.cirq import CirqSimulator

    from .test_qubit_selection import exercise_qubit_selection_backend

    exercise_qubit_selection_backend(CirqSimulator(), seed=7)


def test_cirq_keyed_parameters() -> None:
    """Rebind fixed-key dictionaries through the ordinary Param Vector ABI."""
    from qamomile.cirq import CirqSimulator

    from .test_parameter_map import exercise_parameter_map_backend

    exercise_parameter_map_backend(CirqSimulator(), seed=7)


def test_cirq_quantum_records() -> None:
    """Execute whole-record ownership transfers and coherent transforms."""
    from qamomile.cirq import CirqSimulator

    from .test_quantum_records import exercise_quantum_record_backend

    exercise_quantum_record_backend(CirqSimulator(), seed=7)


def test_cirq_recursion() -> None:
    """Execute self, mutual, branching, and higher-order recursive helpers."""
    from qamomile.cirq import CirqSimulator

    from .test_recursion import exercise_recursive_backend

    exercise_recursive_backend(CirqSimulator(), seed=7)


def test_cirq_sqrt() -> None:
    """Execute Const and launch-bound square roots through the native algebra."""
    from qamomile.cirq import CirqSimulator

    from .test_sqrt import exercise_sqrt_backend

    exercise_sqrt_backend(CirqSimulator(), seed=7)
