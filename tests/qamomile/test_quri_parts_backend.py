"""End-to-end portable QURI Parts and isolated Qulacs tests."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

import qamomile as qm
from qamomile.persistence import deserialize, serialize

from .conftest import (
    BASIS_SOURCE,
    CALLABLE_TRANSFORMS_SOURCE,
    COHERENT_ADJOINT_SOURCE,
    COHERENT_RESHAPE_SOURCE,
    CONTROLLED_PHASE_SOURCE,
    EXTENDED_GATE_SOURCE,
    HOST_DISPATCH_SOURCE,
    INDEXED_CONTROL_SOURCE,
    LAUNCH_LOOP_SOURCE,
    MULTI_INPUT_CLEANUP_SOURCE,
    PAULI_EVOLUTION_SOURCE,
    RECORD_RESULT_SOURCE,
    ROTATION_SOURCE,
    STATE_PREPARATION_SOURCE,
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


def test_quri_parts_executes_boolean_lifting() -> None:
    """Qulacs executes the shared native Boolean lift plans."""

    from qamomile.quri_parts import Qulacs

    exercise_boolean_lift_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_symbolic_observable_recipe() -> None:
    """Qulacs executes an Observable assembled from Const coefficient data."""

    from qamomile.quri_parts import Qulacs

    exercise_symbolic_observable_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_python_collection_loops() -> None:
    """Qulacs consumes enumerate/zip after shared Rust loop normalization."""

    from qamomile.quri_parts import Qulacs

    exercise_collection_loops_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_derived_measurement_results() -> None:
    """Qulacs evaluates the retained result tree over raw basis samples."""

    from qamomile.quri_parts import Qulacs

    endpoint = Qulacs()
    exercise_derived_result_backend(endpoint=endpoint, seed=2)
    exercise_broadcast_gate_backend(endpoint=endpoint, seed=2)


def test_quri_parts_measures_unsigned_integer_registers() -> None:
    """Qulacs preserves integer launch and typed result decoding."""

    from qamomile.quri_parts import Qulacs

    exercise_uint_measurement_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_materializes_dynamic_qubit_selection_at_launch() -> None:
    """Rust resolves a launch-time wire selector before logical emission."""

    from qamomile.quri_parts import Qulacs

    exercise_dynamic_qubit_selection_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_materializes_dynamic_classical_index_at_launch() -> None:
    """Rust resolves an aggregate path before QURI Parts emission."""

    from qamomile.quri_parts import Qulacs

    exercise_dynamic_classical_index_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_launch_bound_structural_math() -> None:
    """QURI Parts receives the loop after Rust evaluates structural math."""

    from qamomile.quri_parts import Qulacs

    exercise_structural_math_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_multi_owner_host_dispatch() -> None:
    """QURI Parts variants retain both independently updated owned outputs."""

    from qamomile.quri_parts import Qulacs

    exercise_multi_owner_dispatch_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_rust_lowered_host_dispatch_family() -> None:
    """One portable logical family evaluates an integer launch predicate."""

    from qamomile.quri_parts import Qulacs

    endpoint = Qulacs()
    program = qm.build_source(
        HOST_DISPATCH_SOURCE,
        module_name="quri_parts_host_dispatch_family",
    ).specialize("host_dispatch")
    artifact = qm.compile(program, to=endpoint.target)

    assert artifact.format == "qamomile.artifact-family.v3"
    for limit, expected in ((-2, 0), (3, 1)):
        result = qm.execute(
            artifact,
            on=endpoint,
            parameters={"limit": limit},
            shots=8,
            seed=3,
        )
        assert result.counts == {(expected,): 8}


def test_quri_parts_materializes_a_launch_dependent_loop() -> None:
    """Qulacs receives one Rust-expanded circuit for the requested loop count."""

    from qamomile.quri_parts import Qulacs

    endpoint = Qulacs()
    program = qm.build_source(
        LAUNCH_LOOP_SOURCE,
        module_name="quri_parts_launch_loop",
    ).specialize("launch_loop")
    artifact = qm.compile(program, to=endpoint.target)

    assert artifact.format == "qamomile.launch-materializer.v1"
    for rounds, expected in ((2, 0), (3, 1)):
        result = qm.execute(
            artifact,
            on=endpoint,
            parameters={"rounds": rounds},
            shots=8,
            seed=3,
        )
        assert result.counts == {(expected,): 8}


def test_quri_parts_executes_documented_quantum_kernel_classification() -> None:
    """Qulacs executes the documented overlap kernel without a custom path."""

    from qamomile.quri_parts import Qulacs

    exercise_classification_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_documented_hydrogen_vqe() -> None:
    """Qulacs estimates the documented hydrogen trial energy."""

    from qamomile.quri_parts import Qulacs

    exercise_hydrogen_vqe_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_documented_qsci_sampling() -> None:
    """Qulacs samples the documented QSCI trial-state configurations."""

    from qamomile.quri_parts import Qulacs

    exercise_qsci_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_documented_hamiltonian_simulation() -> None:
    """Qulacs executes the documented noncommuting Rabi evolution."""

    from qamomile.quri_parts import Qulacs

    exercise_rabi_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_documented_mottonen_encoding() -> None:
    """Qulacs prepares the documented complex Möttönen state."""

    from qamomile.quri_parts import Qulacs

    exercise_mottonen_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_documented_multidimensional_qft() -> None:
    """Qulacs executes the documented axiswise Fourier transform."""

    from qamomile.quri_parts import Qulacs

    exercise_multidimensional_qft_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_documented_hybrid_qnn() -> None:
    """Qulacs executes the documented runtime Hybrid QNN layer."""

    from qamomile.quri_parts import Qulacs

    exercise_hqnn_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_documented_option_pricing() -> None:
    """Qulacs executes the documented amplitude-estimation option pricer."""

    from qamomile.quri_parts import Qulacs

    exercise_option_pricing_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_documented_pce() -> None:
    """Qulacs estimates and decodes the documented PCE state."""

    from qamomile.quri_parts import Qulacs

    exercise_pce_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_accepts_the_full_unsigned_seed_domain() -> None:
    """Qulacs deterministically folds every invocation seed into its SDK domain."""

    from qamomile.quri_parts import Qulacs

    seed = (1 << 64) - 1
    first = qm.run(amplitude_two_sample, on=Qulacs(), shots=8, seed=seed)
    second = qm.run(amplitude_two_sample, on=Qulacs(), shots=8, seed=seed)

    assert first.invocation.seed == seed
    assert second.invocation.seed == seed
    assert first.result.counts == second.result.counts


def test_quri_parts_executes_documented_qemcmc() -> None:
    """Qulacs executes the documented quantum-enhanced MCMC proposal."""

    from qamomile.quri_parts import Qulacs

    exercise_qemcmc_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_documented_bit_flip_code() -> None:
    """Qulacs coherently corrects each single data-qubit bit flip."""

    from qamomile.quri_parts import Qulacs

    exercise_bit_flip_code_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_documented_steane_code() -> None:
    """Qulacs coherently corrects a bounded Steane error case."""

    from qamomile.quri_parts import Qulacs

    exercise_steane_code_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_variational_composition() -> None:
    """Qulacs executes QAOA and FQAOA without algorithm-specific lowering."""

    from qamomile.quri_parts import Qulacs

    exercise_variational_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_grover_adaptive_search() -> None:
    """Qulacs executes GAS composed from ordinary reusable qkernels."""

    from qamomile.quri_parts import Qulacs

    exercise_gas_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_product_formula_composition() -> None:
    """Qulacs executes the shared abstract-evolution schedule."""

    from qamomile.quri_parts import Qulacs

    exercise_product_formula_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_uniformly_controlled_rotation() -> None:
    """Qulacs executes the public function-composed Gray walk."""

    from qamomile.quri_parts import Qulacs

    exercise_uniformly_controlled_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_coherent_xor_lookup() -> None:
    """Qulacs executes lookup through total callable selection."""

    from qamomile.quri_parts import Qulacs

    exercise_xor_lookup_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_sparse_matrix_lookup() -> None:
    """Qulacs executes functional sparse row/slot access."""

    from qamomile.quri_parts import Qulacs

    exercise_sparse_lookup_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_runtime_matrix_access() -> None:
    """Qulacs binds a shaped Matrix while rotations use scalar leaves."""

    from qamomile.quri_parts import Qulacs

    exercise_runtime_matrix_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_runtime_qsvt_phases() -> None:
    """Qulacs reuses one QSVT circuit with a phase Vector launch argument."""

    from qamomile.quri_parts import Qulacs

    exercise_runtime_qsvt_backend(endpoint=Qulacs(), seed=2)


pytestmark = pytest.mark.quri_parts
pytest.importorskip("quri_parts")
pytest.importorskip("quri_parts.qulacs")


def test_quri_parts_executes_signal_composition() -> None:
    """Execute shared region phases and editable signal schedules."""
    from qamomile.quri_parts import Qulacs

    from .test_signals import exercise_signals_backend

    exercise_signals_backend(endpoint=Qulacs())


def test_quri_parts_executes_standard_optimized_qcore() -> None:
    """QURI Parts executes the shared exact QCore optimization witness."""

    from qamomile.quri_parts import Qulacs

    from .test_standard_optimization import exercise_standard_optimization_backend

    exercise_standard_optimization_backend(endpoint=Qulacs())


def test_quri_parts_executes_functional_amplitude_encoding() -> None:
    """Qulacs samples and estimates the multilevel Gray-walk body."""

    from qamomile.quri_parts import Qulacs

    endpoint = Qulacs()
    assert qm.run(amplitude_two_sample, on=endpoint, shots=4, seed=2).result.counts == {
        (1, 1): 4
    }
    assert (
        qm.expectation(amplitude_two_state, qm.Z(0), on=endpoint, shots=4, seed=2).value
        == -1.0
    )


def test_quri_parts_reuses_runtime_parametric_amplitude_artifact() -> None:
    """Qulacs resolves aggregate parameter projections from portable JSON."""

    from qamomile.quri_parts import Qulacs, QuriParts

    exercise_runtime_amplitude_backend(endpoint=Qulacs(), target=QuriParts(), seed=2)


def test_quri_parts_executes_functional_pauli_block_encoding() -> None:
    """Qulacs executes Pauli production through the existing LCU body."""

    from qamomile.quri_parts import Qulacs

    exercise_pauli_block_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_functional_block_producers() -> None:
    """Qulacs executes projected and periodic construction through common IR."""

    from qamomile.quri_parts import Qulacs

    exercise_functional_block_producers_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_standard_python_record_results() -> None:
    """Qulacs reconstructs field names from the shared product result ABI."""

    from qamomile.quri_parts import Qulacs

    program = qm.build_source(RECORD_RESULT_SOURCE).specialize("record_result")
    assert qm.run(program, on=Qulacs(), shots=4, seed=2).result.values == {
        (("first", (1,)), ("pair", (0, 1))): 4
    }


def test_quri_parts_executes_coherent_product_reassociation() -> None:
    """Qulacs receives only gates surrounding a structural coherent reshape."""

    from qamomile.quri_parts import Qulacs

    program = qm.build_source(COHERENT_RESHAPE_SOURCE).specialize("branched_adjoint")
    assert qm.run(program, on=Qulacs(), shots=4, seed=2).result.counts == {(1, 0, 1): 4}


def test_quri_parts_executes_compiler_derived_selected_adjoint() -> None:
    """Qulacs executes multiplex(U_i-adjoint) from the shared dagger pass."""

    from qamomile.quri_parts import Qulacs

    program = qm.build_source(COHERENT_ADJOINT_SOURCE).specialize(
        "selected_adjoint_roundtrip"
    )
    assert qm.run(program, on=Qulacs(), shots=4, seed=2).result.counts == {(1, 0): 4}


def test_quri_parts_executes_indexed_aggregate_control() -> None:
    """Qulacs samples and estimates selector-based coherent control."""

    from qamomile.quri_parts import Qulacs

    module = qm.build_source(INDEXED_CONTROL_SOURCE)
    assert qm.run(
        module["indexed_control_sample"], on=Qulacs(), shots=4, seed=2
    ).result.counts == {(1, 0, 1): 4}
    assert (
        qm.expectation(
            module["indexed_control_state"],
            qm.Z(2),
            on=Qulacs(),
            shots=4,
            seed=2,
        ).value
        == -1.0
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_quri_parts_artifact_executes_and_round_trips(seed: int) -> None:
    """Portable circuit bytes materialize and sample through seeded Qulacs."""

    from qamomile.quri_parts import Qulacs, QuriParts, load_logical_circuit

    program = qm.build_source(BASIS_SOURCE).specialize("basis")
    artifact = qm.compile(program, to=QuriParts())
    restored = deserialize(serialize(artifact))
    with patch(
        "subprocess.run",
        side_effect=AssertionError("worker used"),
    ):
        result = qm.execute(restored, on=Qulacs(), shots=8, seed=seed)

    assert artifact.format == "quri-parts.logical-circuit.v2"
    assert load_logical_circuit(restored)["width"] == 3
    assert result.counts == {(1, 0, 0): 8}
    assert result.worker_pid is None


def test_quri_parts_public_circuit_supports_both_estimator_paths() -> None:
    """The public SDK projection remains inspectable, bindable, and estimable."""

    from quri_parts.core.operator import pauli_label
    from quri_parts.core.state import (
        GeneralCircuitQuantumState,
        ParametricCircuitQuantumState,
    )
    from quri_parts.qulacs.estimator import (
        create_qulacs_vector_estimator,
        create_qulacs_vector_parametric_estimator,
    )

    from qamomile.quri_parts import QuriParts, load_circuit

    program = qm.build_source(ROTATION_SOURCE).specialize("rotate")
    artifact = qm.compile(program, to=QuriParts())
    circuit = load_circuit(artifact)
    parameter_names = tuple(
        parameter.name for parameter in circuit.param_mapping.in_params
    )

    assert circuit.qubit_count == 1
    assert circuit.parameter_count == 1
    assert parameter_names == ("theta",)

    observable = pauli_label("Z0")
    parametric_state = ParametricCircuitQuantumState(1, circuit)
    parametric_estimate = create_qulacs_vector_parametric_estimator()(
        observable,
        parametric_state,
        [math.pi],
    )

    bound = circuit.bind_parameters([math.pi])
    bound_state = GeneralCircuitQuantumState(1, circuit=bound)
    bound_estimate = create_qulacs_vector_estimator()(observable, bound_state)

    assert parametric_estimate.value.real == pytest.approx(-1.0)
    assert bound_estimate.value.real == pytest.approx(-1.0)


def test_quri_parts_public_circuit_flattens_qaoa_coordinates_by_first_use() -> None:
    """Aggregate QAOA parameters retain the documented SDK binding order."""

    from qamomile.optimization import BinaryModel, QAOAExperiment
    from qamomile.quri_parts import QuriParts, load_circuit

    model = BinaryModel.from_ising(
        {(0, 1): 0.5},
        constant=-0.5,
        variables=range(2),
    )
    artifact = QAOAExperiment(model, layers=2).compile(to=QuriParts()).artifact
    circuit = load_circuit(artifact)

    assert circuit.qubit_count == 2
    assert circuit.parameter_count == 4
    assert tuple(parameter.name for parameter in circuit.param_mapping.in_params) == (
        "gammas[0]",
        "betas[0]",
        "gammas[1]",
        "betas[1]",
    )
    assert circuit.bind_parameters([0.1, 0.2, 0.3, 0.4]).qubit_count == 2


def test_quri_parts_live_sampler_reuses_the_portable_artifact() -> None:
    """A live QURI Parts sampler receives one bound SDK circuit at launch."""

    from quri_parts.qulacs.sampler import create_qulacs_vector_sampler

    from qamomile.quri_parts import QuriPartsSampler

    endpoint = QuriPartsSampler(create_qulacs_vector_sampler(), label="vector")
    program = qm.build_source(ROTATION_SOURCE).specialize("rotate")
    artifact = qm.compile(program, to=endpoint.target)
    result = qm.execute(
        artifact,
        on=endpoint,
        parameters={"theta": math.pi},
        shots=8,
        seed=0,
    )

    assert result.backend == "quri-parts.sampler.vector"
    assert result.counts == {(1,): 8}
    assert result.worker_pid is None


def test_quri_parts_live_sampler_rejects_nonportable_seed() -> None:
    """Live samplers fail closed when asked for a portable seed contract."""

    from quri_parts.qulacs.sampler import create_qulacs_vector_sampler

    from qamomile.quri_parts import QuriPartsSampler

    endpoint = QuriPartsSampler(create_qulacs_vector_sampler())
    program = qm.build_source(BASIS_SOURCE).specialize("basis")

    with pytest.raises(qm.ExecutionError, match="use seed=0"):
        qm.run(program, on=endpoint, shots=4, seed=1)


def test_quri_parts_live_sampler_rejects_fractional_counts() -> None:
    """Sampler injection accepts observations rather than probability weights."""

    from qamomile.quri_parts import QuriPartsSampler

    endpoint = QuriPartsSampler(lambda circuit, shots: {0: float(shots)})
    program = qm.build_source(BASIS_SOURCE).specialize("basis")

    with pytest.raises(qm.ExecutionError, match="nonnegative integers"):
        qm.run(program, on=endpoint, shots=4, seed=0)


@pytest.mark.parametrize(
    ("theta", "expected"),
    [(0.0, (0,)), (math.pi, (1,)), (2 * math.pi, (0,))],
)
def test_quri_parts_executes_parameter_boundary_rotations(
    theta: float,
    expected: tuple[int, ...],
) -> None:
    """Runtime parameters survive the artifact and bind before QURI emission."""

    from qamomile.quri_parts import Qulacs

    program = qm.build_source(ROTATION_SOURCE).specialize("rotate")
    result = qm.run(
        program,
        on=Qulacs(),
        parameters={"theta": theta},
        shots=8,
        seed=2,
    ).result

    assert result.counts == {expected: 8}


def test_quri_parts_executes_gate_and_callable_transform_families() -> None:
    """Qulacs executes primitive, controlled-call, multiplex, and phase paths."""

    from qamomile.quri_parts import Qulacs

    extended = qm.build_source(EXTENDED_GATE_SOURCE).specialize("extended_gates")
    selected = qm.build_source(CALLABLE_TRANSFORMS_SOURCE).specialize(
        "callable_transforms"
    )
    phased = qm.build_source(CONTROLLED_PHASE_SOURCE).specialize("controlled_phase")

    assert qm.run(extended, on=Qulacs(), shots=4, seed=2).result.counts == {
        (1, 1, 1): 4
    }
    assert qm.run(selected, on=Qulacs(), shots=4, seed=2).result.values == {
        (1, (1,)): 4
    }
    assert qm.run(phased, on=Qulacs(), shots=4, seed=2).result.values == {(1, (0,)): 4}


def test_quri_parts_executes_pauli_evolution_and_expectation_paths() -> None:
    """Sampling and observable estimation both use real QURI/Qulacs execution."""

    from qamomile.quri_parts import Qulacs

    evolved = qm.build_source(PAULI_EVOLUTION_SOURCE).specialize(
        "pauli_evolution",
        **{"hamiltonian": 0.5 * qm.Z(0) * qm.Z(1)},
    )
    state = qm.build_source(STATE_PREPARATION_SOURCE)["plus_state"]

    assert qm.run(evolved, on=Qulacs(), shots=4, seed=2).result.counts == {(1, 0): 4}
    assert qm.expectation(state, qm.X(0), on=Qulacs(), shots=4, seed=2).value == 1.0
    exercise_normalized_model_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_imported_qft_roundtrip() -> None:
    """The linked functional QFT library executes through QURI Parts/Qulacs."""

    from qamomile.quri_parts import Qulacs

    assert qm.run(qft_roundtrip, on=Qulacs(), shots=4, seed=2).result.counts == {
        (1, 0, 1): 4
    }
    assert qm.run(qft_frequency_three, on=Qulacs(), shots=4, seed=2).result.counts == {
        (1, 1, 0): 4
    }


def test_quri_parts_executes_functional_phase_estimation() -> None:
    """Qulacs samples and estimates body-derived two-bit QPE."""

    from qamomile.quri_parts import Qulacs

    assert qm.run(qpe_sample, on=Qulacs(), shots=4, seed=2).result.counts == {
        (1, 0, 1): 4
    }
    assert (
        qm.expectation(qpe_state, qm.Z(0), on=Qulacs(), shots=4, seed=2).value == -1.0
    )
    exercise_nary_tensor_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_decodes_typed_fixed_point_after_measurement_mapping() -> None:
    """Qulacs maps measured qubits into source order before phase decoding."""

    from qamomile.quri_parts import Qulacs

    exercise_fixed_qpe_backend(endpoint=Qulacs(), seed=2)
    exercise_windowed_qpe_backend(endpoint=Qulacs(), seed=2)
    exercise_documented_windowed_qpe_backend(endpoint=Qulacs(), seed=2)
    exercise_documented_reversible_arithmetic_backend(endpoint=Qulacs(), seed=2)
    exercise_documented_gf2_arithmetic_backend(endpoint=Qulacs(), seed=2)
    exercise_documented_phase_gradient_backend(endpoint=Qulacs(), seed=2)
    exercise_documented_fault_tolerant_data_loading_backend(endpoint=Qulacs(), seed=2)
    exercise_documented_reversible_indexing_backend(endpoint=Qulacs(), seed=2)
    exercise_documented_reversible_trigonometry_backend(endpoint=Qulacs(), seed=2)
    exercise_documented_kikuchi_backend(endpoint=Qulacs(), seed=2)
    exercise_documented_guided_hamiltonian_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_documented_qualtran_gap_examples() -> None:
    """Qulacs executes the bounded ECC and factorized-chemistry witnesses."""

    from qamomile.quri_parts import Qulacs

    exercise_documented_elliptic_curve_dlog_backend(endpoint=Qulacs(), seed=2)
    exercise_documented_factorized_chemistry_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_functional_grover_search() -> None:
    """Qulacs samples and estimates body-composed amplitude amplification."""

    from qamomile.quri_parts import Qulacs

    assert qm.run(grover_sample, on=Qulacs(), shots=4, seed=2).result.counts == {
        (1, 1): 4
    }
    assert (
        qm.expectation(grover_state, qm.Z(0), on=Qulacs(), shots=4, seed=2).value
        == -1.0
    )


def test_quri_parts_executes_functional_modular_arithmetic() -> None:
    """Qulacs executes bounded sample/expectation arithmetic witnesses."""

    from qamomile.quri_parts import Qulacs

    exercise_modular_arithmetic_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_shor_order_finding() -> None:
    """Qulacs executes the shared Shor sampling and state witnesses."""

    from qamomile.quri_parts import Qulacs

    exercise_shor_order_finding_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_regev_factoring_sample() -> None:
    """Qulacs executes the shared bounded Regev sample."""

    from qamomile.quri_parts import Qulacs

    exercise_regev_factoring_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_multi_exponentiation_callable_realization() -> None:
    """Qulacs executes the shared source after callable replacement."""

    from qamomile.quri_parts import Qulacs

    options = qm.RealizationOptions(
        {multi_exponentiation_baseline: multi_exponentiation_product_tree}
    )
    exercise_multi_exponentiation_realization_backend(
        endpoint=Qulacs(realizations=options), seed=2
    )


def test_quri_parts_executes_first_class_multi_exponentiation() -> None:
    """Qulacs executes the semantic primitive through its independent realization."""

    from qamomile.quri_parts import Qulacs

    exercise_first_class_multi_exponentiation_backend(endpoint=Qulacs(), seed=2)


def test_quri_parts_executes_register_modulus_arithmetic() -> None:
    """Qulacs samples retained-modulus addition."""

    from qamomile.quri_parts import Qulacs

    exercise_register_modulus_backend(endpoint=Qulacs(), seed=0)


def test_quri_parts_executes_functional_qsvt() -> None:
    """Qulacs executes direct and PREPARE/SELECT-composed QSVT bodies."""

    from qamomile.quri_parts import Qulacs

    endpoint = Qulacs()
    assert qm.run(qsvt_phase_sample, on=endpoint, shots=4, seed=2).result.counts == {
        (1, 0): 4
    }
    assert (
        qm.expectation(qsvt_phase_state, qm.Z(0), on=endpoint, shots=4, seed=2).value
        == -1.0
    )
    assert qm.run(lcu_qsvt_sample, on=endpoint, shots=4, seed=2).result.counts == {
        (1, 0): 4
    }
    assert (
        qm.expectation(lcu_qsvt_state, qm.Z(0), on=endpoint, shots=4, seed=2).value
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


def test_quri_parts_executes_multi_input_automatic_uncomputation() -> None:
    """Qulacs executes a packed adjoint restoring two independent inputs."""

    from qamomile.quri_parts import Qulacs

    program = qm.build_source(MULTI_INPUT_CLEANUP_SOURCE).specialize(
        "multi_input_cleanup"
    )
    result = qm.run(program, on=Qulacs(), shots=4, seed=7).result

    assert result.counts == {(1,): 4}


def test_quri_parts_rejects_feedback_before_emission() -> None:
    """The static Qulacs profile never pretends to implement device feedback."""

    from qamomile.quri_parts import QuriParts

    source = """
import qamomile as qm

@qm.qkernel
def feedback() -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    bit, state = qm.measure_z(state, 0)
    if bit:
        state = qm.x(state)
    return qm.measure(state)
"""
    program = qm.build_source(source).specialize("feedback")

    with pytest.raises(qm.ArtifactError, match="dynamic_branch"):
        qm.compile(program, to=QuriParts())


@pytest.mark.parametrize("choice", ["add.increments", "add.qft", "add.ripple"])
def test_quri_parts_executes_late_realizations(choice: str) -> None:
    """One shared arithmetic source survives exact alternative realizations."""
    pytest.importorskip("quri_parts.qulacs")
    from qamomile.lowering import RealizationOptions
    from qamomile.quri_parts import Qulacs

    from .test_realizations import exercise_realization_backend

    exercise_realization_backend(
        Qulacs(realizations=RealizationOptions({"quantum.add_into": choice}))
    )


def test_quri_parts_zero_layer_qaoa() -> None:
    """Execute the shared zero layer qaoa contract locally."""
    from qamomile.quri_parts import Qulacs

    from .test_library_compatibility import exercise_zero_layer_qaoa_backend

    exercise_zero_layer_qaoa_backend(Qulacs())


def test_quri_parts_approximate_qft() -> None:
    """Execute the shared approximate qft contract locally."""
    from qamomile.quri_parts import Qulacs

    from .test_library_compatibility import exercise_approximate_qft_backend

    exercise_approximate_qft_backend(Qulacs())


def test_quri_parts_classical_array_compatibility() -> None:
    """Execute the shared classical array contract locally."""
    from qamomile.quri_parts import Qulacs

    from .test_classical_array_compatibility import exercise_classical_array_backend

    exercise_classical_array_backend(Qulacs())


def test_quri_parts_qubit_selection() -> None:
    """Execute owned noncontiguous selection on this local SDK."""
    from qamomile.quri_parts import Qulacs

    from .test_qubit_selection import exercise_qubit_selection_backend

    exercise_qubit_selection_backend(Qulacs(), seed=7)


def test_quri_parts_keyed_parameters() -> None:
    """Rebind fixed-key dictionaries through the ordinary Param Vector ABI."""
    from qamomile.quri_parts import Qulacs

    from .test_parameter_map import exercise_parameter_map_backend

    exercise_parameter_map_backend(Qulacs(), seed=7)


def test_quri_parts_quantum_records() -> None:
    """Execute whole-record ownership transfers and coherent transforms."""
    from qamomile.quri_parts import Qulacs

    from .test_quantum_records import exercise_quantum_record_backend

    exercise_quantum_record_backend(Qulacs(), seed=7)


def test_quri_parts_recursion() -> None:
    """Execute self, mutual, branching, and higher-order recursive helpers."""
    from qamomile.quri_parts import Qulacs

    from .test_recursion import exercise_recursive_backend

    exercise_recursive_backend(Qulacs(), seed=7)


def test_quri_parts_sqrt() -> None:
    """Execute Const and launch-bound square roots through the native algebra."""
    from qamomile.quri_parts import Qulacs

    from .test_sqrt import exercise_sqrt_backend

    exercise_sqrt_backend(Qulacs(), seed=7)
