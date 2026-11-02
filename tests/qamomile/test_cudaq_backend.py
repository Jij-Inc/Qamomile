"""End-to-end source-artifact and CUDA-Q runtime tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch

import pytest

import qamomile as qm
from qamomile.persistence import export_payload, serialize

from .conftest import (
    BASIS_SOURCE,
    BELL_SOURCE,
    CALLABLE_TRANSFORMS_SOURCE,
    COHERENT_ADJOINT_SOURCE,
    COHERENT_RESHAPE_SOURCE,
    CONTROLLED_PHASE_SOURCE,
    EXTENDED_GATE_SOURCE,
    FEEDBACK_SOURCE,
    HOST_DISPATCH_SOURCE,
    INDEXED_CONTROL_SOURCE,
    JOINT_PRIVATE_MEASUREMENT_SOURCE,
    LAUNCH_LOOP_SOURCE,
    MEASURE_AXES_SOURCE,
    MEASURE_RESET_SOURCE,
    MODULAR_CLEANUP_SOURCE,
    MULTI_INPUT_CLEANUP_SOURCE,
    NEGATED_FEEDBACK_SOURCE,
    PAULI_EVOLUTION_SOURCE,
    RECORD_RESULT_SOURCE,
    STATE_PREPARATION_SOURCE,
    STRUCTURED_RESULT_SOURCE,
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


def test_cudaq_executes_boolean_lifting() -> None:
    """CUDA-Q executes the shared native Boolean lift plans."""

    from qamomile.cudaq import CudaqLocal

    exercise_boolean_lift_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_symbolic_observable_recipe() -> None:
    """CUDA-Q executes an Observable assembled from Const coefficient data."""

    from qamomile.cudaq import CudaqLocal

    exercise_symbolic_observable_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_python_collection_loops() -> None:
    """CUDA-Q consumes enumerate/zip after shared Rust loop normalization."""

    from qamomile.cudaq import CudaqLocal

    exercise_collection_loops_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_derived_measurement_results() -> None:
    """CUDA-Q evaluates the retained result tree over raw kernel samples."""

    from qamomile.cudaq import CudaqLocal

    endpoint = CudaqLocal()
    exercise_derived_result_backend(endpoint=endpoint, seed=2)
    exercise_broadcast_gate_backend(endpoint=endpoint, seed=2)


def test_cudaq_measures_unsigned_integer_registers() -> None:
    """CUDA-Q preserves integer launch and typed result decoding."""

    from qamomile.cudaq import CudaqLocal

    exercise_uint_measurement_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_materializes_dynamic_qubit_selection_at_launch() -> None:
    """Rust resolves a launch-time wire selector before kernel emission."""

    from qamomile.cudaq import CudaqLocal

    exercise_dynamic_qubit_selection_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_materializes_dynamic_classical_index_at_launch() -> None:
    """Rust resolves an aggregate path before CUDA-Q kernel emission."""

    from qamomile.cudaq import CudaqLocal

    exercise_dynamic_classical_index_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_launch_bound_structural_math() -> None:
    """CUDA-Q receives the loop after Rust evaluates structural math."""

    from qamomile.cudaq import CudaqLocal

    exercise_structural_math_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_multi_owner_host_dispatch() -> None:
    """CUDA-Q variants retain both independently updated owned outputs."""

    from qamomile.cudaq import CudaqLocal

    exercise_multi_owner_dispatch_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_rust_lowered_host_dispatch_family() -> None:
    """One source-artifact family evaluates an integer launch predicate."""

    from qamomile.cudaq import CudaqLocal

    endpoint = CudaqLocal()
    program = qm.build_source(
        HOST_DISPATCH_SOURCE,
        module_name="cudaq_host_dispatch_family",
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


def test_cudaq_materializes_a_launch_dependent_loop() -> None:
    """CUDA-Q receives one Rust-expanded kernel for the requested loop count."""

    from qamomile.cudaq import CudaqLocal

    endpoint = CudaqLocal()
    program = qm.build_source(
        LAUNCH_LOOP_SOURCE,
        module_name="cudaq_launch_loop",
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


def test_cudaq_executes_documented_quantum_kernel_classification() -> None:
    """CUDA-Q executes the documented overlap kernel without a custom path."""

    from qamomile.cudaq import CudaqLocal

    exercise_classification_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_documented_hydrogen_vqe() -> None:
    """CUDA-Q estimates the documented hydrogen trial energy."""

    from qamomile.cudaq import CudaqLocal

    exercise_hydrogen_vqe_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_documented_qsci_sampling() -> None:
    """CUDA-Q samples the documented QSCI trial-state configurations."""

    from qamomile.cudaq import CudaqLocal

    exercise_qsci_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_documented_hamiltonian_simulation() -> None:
    """CUDA-Q executes the documented noncommuting Rabi evolution."""

    from qamomile.cudaq import CudaqLocal

    exercise_rabi_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_documented_mottonen_encoding() -> None:
    """CUDA-Q prepares the documented complex Möttönen state."""

    from qamomile.cudaq import CudaqLocal

    exercise_mottonen_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_documented_multidimensional_qft() -> None:
    """CUDA-Q executes the documented axiswise Fourier transform."""

    from qamomile.cudaq import CudaqLocal

    exercise_multidimensional_qft_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_documented_hybrid_qnn() -> None:
    """CUDA-Q executes the documented runtime Hybrid QNN layer."""

    from qamomile.cudaq import CudaqLocal

    exercise_hqnn_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_documented_option_pricing() -> None:
    """CUDA-Q executes the documented amplitude-estimation option pricer."""

    from qamomile.cudaq import CudaqLocal

    exercise_option_pricing_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_documented_pce() -> None:
    """CUDA-Q estimates and decodes the documented PCE state."""

    from qamomile.cudaq import CudaqLocal

    exercise_pce_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_documented_qemcmc() -> None:
    """CUDA-Q executes the documented quantum-enhanced MCMC proposal."""

    from qamomile.cudaq import CudaqLocal

    exercise_qemcmc_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_documented_bit_flip_code() -> None:
    """CUDA-Q coherently corrects each single data-qubit bit flip."""

    from qamomile.cudaq import CudaqLocal

    exercise_bit_flip_code_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_documented_steane_code() -> None:
    """CUDA-Q coherently corrects a bounded Steane error case."""

    from qamomile.cudaq import CudaqLocal

    exercise_steane_code_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_variational_composition() -> None:
    """CUDA-Q executes QAOA and FQAOA without algorithm-specific lowering."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    exercise_variational_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_grover_adaptive_search() -> None:
    """CUDA-Q executes GAS composed from ordinary reusable qkernels."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    exercise_gas_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_product_formula_composition() -> None:
    """CUDA-Q executes the shared abstract-evolution schedule."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    exercise_product_formula_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_uniformly_controlled_rotation() -> None:
    """CUDA-Q executes the public function-composed Gray walk."""

    from qamomile.cudaq import CudaqLocal

    exercise_uniformly_controlled_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_coherent_xor_lookup() -> None:
    """CUDA-Q executes lookup through total callable selection."""

    from qamomile.cudaq import CudaqLocal

    exercise_xor_lookup_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_sparse_matrix_lookup() -> None:
    """CUDA-Q executes functional sparse row/slot access."""

    from qamomile.cudaq import CudaqLocal

    exercise_sparse_lookup_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_runtime_matrix_access() -> None:
    """CUDA-Q binds a shaped Matrix while rotations use scalar leaves."""

    from qamomile.cudaq import CudaqLocal

    exercise_runtime_matrix_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_runtime_qsvt_phases() -> None:
    """CUDA-Q reuses one QSVT circuit with a phase Vector launch argument."""

    from qamomile.cudaq import CudaqLocal

    exercise_runtime_qsvt_backend(endpoint=CudaqLocal(), seed=2)


pytestmark = pytest.mark.cudaq


def test_cudaq_executes_signal_composition() -> None:
    """Execute shared region phases and editable signal schedules."""
    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    from .test_signals import exercise_signals_backend

    exercise_signals_backend(endpoint=CudaqLocal())


def test_cudaq_executes_standard_optimized_qcore() -> None:
    """CUDA-Q executes the shared exact QCore optimization witness."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    from .test_standard_optimization import exercise_standard_optimization_backend

    exercise_standard_optimization_backend(endpoint=CudaqLocal())


def test_cudaq_executes_functional_amplitude_encoding() -> None:
    """CUDA-Q samples and estimates the multilevel Gray-walk body."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    endpoint = CudaqLocal()
    assert qm.run(amplitude_two_sample, on=endpoint, shots=4, seed=2).result.counts == {
        (1, 1): 4
    }
    assert (
        qm.expectation(amplitude_two_state, qm.Z(0), on=endpoint, shots=4, seed=2).value
        == -1.0
    )


def test_cudaq_reuses_runtime_parametric_amplitude_artifact() -> None:
    """CUDA-Q receives scalar leaves in the common deterministic ABI order."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import Cudaq, CudaqLocal

    exercise_runtime_amplitude_backend(endpoint=CudaqLocal(), target=Cudaq(), seed=2)


def test_cudaq_executes_functional_pauli_block_encoding() -> None:
    """CUDA-Q executes Pauli production through the existing LCU body."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    exercise_pauli_block_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_functional_block_producers() -> None:
    """CUDA-Q executes projected and periodic construction through common IR."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    exercise_functional_block_producers_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_standard_python_record_results() -> None:
    """CUDA-Q reconstructs field names from the shared product result ABI."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    program = qm.build_source(RECORD_RESULT_SOURCE).specialize("record_result")
    assert qm.run(program, on=CudaqLocal(), shots=4, seed=2).result.values == {
        (("first", (1,)), ("pair", (0, 1))): 4
    }


def test_cudaq_executes_callable_transforms_and_exact_relative_phase() -> None:
    """CUDA-Q composes control/multiplex and retains controlled scalar phase."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    selected = qm.build_source(CALLABLE_TRANSFORMS_SOURCE).specialize(
        "callable_transforms"
    )
    phased = qm.build_source(CONTROLLED_PHASE_SOURCE).specialize("controlled_phase")

    assert qm.run(selected, on=CudaqLocal(), shots=4, seed=2).result.values == {
        (1, (1,)): 4
    }
    assert qm.run(phased, on=CudaqLocal(), shots=4, seed=2).result.values == {
        (1, (0,)): 4
    }


def test_cudaq_executes_compiler_derived_selected_adjoint() -> None:
    """CUDA-Q executes multiplex(U_i-adjoint) from the common dagger."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    program = qm.build_source(COHERENT_ADJOINT_SOURCE).specialize(
        "selected_adjoint_roundtrip"
    )
    assert qm.run(program, on=CudaqLocal(), shots=4, seed=2).result.counts == {
        (1, 0): 4
    }
    exercise_normalized_model_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_indexed_aggregate_control() -> None:
    """CUDA-Q samples and estimates selector-based coherent control."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    module = qm.build_source(INDEXED_CONTROL_SOURCE)
    assert qm.run(
        module["indexed_control_sample"], on=CudaqLocal(), shots=4, seed=2
    ).result.counts == {(1, 0, 1): 4}
    assert (
        qm.expectation(
            module["indexed_control_state"],
            qm.Z(2),
            on=CudaqLocal(),
            shots=4,
            seed=2,
        ).value
        == -1.0
    )


def test_cudaq_executes_abstract_pauli_evolution() -> None:
    """The shared circuit projection lowers a Pauli gadget once for CUDA-Q."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    program = qm.build_source(PAULI_EVOLUTION_SOURCE).specialize(
        "pauli_evolution",
        **{"hamiltonian": 0.5 * qm.Z(0) * qm.Z(1)},
    )
    assert qm.run(program, on=CudaqLocal(), shots=4, seed=2).result.counts == {
        (1, 0): 4
    }


def test_cudaq_estimates_a_backend_neutral_observable() -> None:
    """Expectation basis programs execute through the ordinary CUDA-Q endpoint."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    state = qm.build_source(STATE_PREPARATION_SOURCE)["plus_state"]
    assert qm.expectation(state, qm.X(0), on=CudaqLocal(), shots=4, seed=2).value == 1.0


def test_cudaq_executes_imported_qft_roundtrip() -> None:
    """The linked functional QFT library executes through CUDA-Q."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    assert qm.run(qft_roundtrip, on=CudaqLocal(), shots=4, seed=2).result.counts == {
        (1, 0, 1): 4
    }
    assert qm.run(
        qft_frequency_three,
        on=CudaqLocal(),
        shots=4,
        seed=2,
    ).result.counts == {(1, 1, 0): 4}


def test_cudaq_executes_functional_phase_estimation() -> None:
    """CUDA-Q samples and estimates body-derived two-bit QPE."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    assert qm.run(qpe_sample, on=CudaqLocal(), shots=4, seed=2).result.counts == {
        (1, 0, 1): 4
    }
    assert (
        qm.expectation(
            qpe_state,
            qm.Z(0),
            on=CudaqLocal(),
            shots=4,
            seed=2,
        ).value
        == -1.0
    )
    exercise_nary_tensor_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_decodes_typed_fixed_point_after_kernel_result_order() -> None:
    """CUDA-Q logical tuple results are normalized before phase decoding."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    exercise_fixed_qpe_backend(endpoint=CudaqLocal(), seed=2)
    exercise_windowed_qpe_backend(endpoint=CudaqLocal(), seed=2)
    exercise_documented_windowed_qpe_backend(endpoint=CudaqLocal(), seed=2)
    exercise_documented_reversible_arithmetic_backend(endpoint=CudaqLocal(), seed=2)
    exercise_documented_gf2_arithmetic_backend(endpoint=CudaqLocal(), seed=2)
    exercise_documented_phase_gradient_backend(endpoint=CudaqLocal(), seed=2)
    exercise_documented_fault_tolerant_data_loading_backend(
        endpoint=CudaqLocal(), seed=2
    )
    exercise_documented_reversible_indexing_backend(endpoint=CudaqLocal(), seed=2)
    exercise_documented_reversible_trigonometry_backend(endpoint=CudaqLocal(), seed=2)
    exercise_documented_kikuchi_backend(endpoint=CudaqLocal(), seed=2)
    exercise_documented_guided_hamiltonian_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_documented_qualtran_gap_examples() -> None:
    """CUDA-Q executes the bounded ECC and factorized-chemistry witnesses."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    exercise_documented_elliptic_curve_dlog_backend(endpoint=CudaqLocal(), seed=2)
    exercise_documented_factorized_chemistry_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_functional_grover_search() -> None:
    """CUDA-Q samples and estimates body-composed amplitude amplification."""

    from qamomile.cudaq import CudaqLocal

    endpoint = CudaqLocal()
    assert qm.run(grover_sample, on=endpoint, shots=4, seed=2).result.counts == {
        (1, 1): 4
    }
    assert (
        qm.expectation(grover_state, qm.Z(0), on=endpoint, shots=4, seed=2).value
        == -1.0
    )


def test_cudaq_executes_functional_modular_arithmetic() -> None:
    """CUDA-Q executes bounded sample/expectation arithmetic witnesses."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    exercise_modular_arithmetic_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_shor_order_finding() -> None:
    """CUDA-Q executes the shared Shor sampling and state witnesses."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    exercise_shor_order_finding_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_regev_factoring_sample() -> None:
    """CUDA-Q executes the shared bounded Regev sample."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    exercise_regev_factoring_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_multi_exponentiation_callable_realization() -> None:
    """CUDA-Q executes the shared source after callable replacement."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    options = qm.RealizationOptions(
        {multi_exponentiation_baseline: multi_exponentiation_product_tree}
    )
    exercise_multi_exponentiation_realization_backend(
        endpoint=CudaqLocal(realizations=options), seed=2
    )


def test_cudaq_executes_first_class_multi_exponentiation() -> None:
    """CUDA-Q executes the primitive through its independent realization."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    exercise_first_class_multi_exponentiation_backend(endpoint=CudaqLocal(), seed=2)


def test_cudaq_executes_register_modulus_arithmetic() -> None:
    """CUDA-Q samples retained-modulus addition."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    exercise_register_modulus_backend(endpoint=CudaqLocal(), seed=0)


def test_cudaq_executes_functional_qsvt() -> None:
    """CUDA-Q executes direct and PREPARE/SELECT-composed QSVT bodies."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    endpoint = CudaqLocal()
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


def test_cudaq_executes_coherent_product_reassociation() -> None:
    """CUDA-Q receives only the gates surrounding a structural reshape."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    program = qm.build_source(COHERENT_RESHAPE_SOURCE).specialize("branched_adjoint")
    assert qm.run(program, on=CudaqLocal(), shots=4, seed=2).result.counts == {
        (1, 0, 1): 4
    }


def test_cudaq_source_artifact_executes_basis_program() -> None:
    """CUDA-Q source round-trips and preserves source bit order."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import Cudaq, CudaqLocal, load_kernel

    program = qm.build_source(BASIS_SOURCE).specialize("basis")
    artifact = qm.compile(program, to=Cudaq())
    assert serialize(artifact) == serialize(qm.compile(program, to=Cudaq()))
    with patch("subprocess.run", side_effect=AssertionError("worker used")):
        result = qm.execute(artifact, on=CudaqLocal(), shots=16, seed=4)

    assert artifact.format == "cudaq.python-source.v1"
    assert load_kernel(artifact) is not None
    assert result.counts == {(1, 0, 0): 16}
    assert result.expectation_z(0, 2) == -1.0
    assert result.worker_pid is None


def test_cudaq_restores_target_after_success_and_failure() -> None:
    """A local launch restores CUDA-Q's target on both execution paths."""

    cudaq = pytest.importorskip("cudaq")
    from qamomile.cudaq import Cudaq, CudaqLocal
    from qamomile.errors import ExecutionError

    artifact = qm.compile(qm.build_source(BASIS_SOURCE).specialize("basis"), to=Cudaq())
    original = cudaq.get_target()
    try:
        cudaq.set_target("density-matrix-cpu")
        assert qm.execute(artifact, on=CudaqLocal(), shots=4).counts == {(1, 0, 0): 4}
        assert cudaq.get_target().name == "density-matrix-cpu"
        with patch.object(cudaq, "run", side_effect=RuntimeError("SDK failed")):
            with pytest.raises(ExecutionError, match="Q8518"):
                qm.execute(artifact, on=CudaqLocal(), shots=4)
        assert cudaq.get_target().name == "density-matrix-cpu"
        with patch(
            "qamomile.cudaq._integration.load_kernel",
            side_effect=ExecutionError("Q8504: invalid serialized CUDA-Q kernel"),
        ):
            with pytest.raises(ExecutionError, match="Q8504"):
                qm.execute(artifact, on=CudaqLocal(), shots=4)
        assert cudaq.get_target().name == "density-matrix-cpu"
        with pytest.raises(ExecutionError, match="Q8517"):
            qm.execute(artifact, on=CudaqLocal("invalid-target"), shots=4)
        assert cudaq.get_target().name == "density-matrix-cpu"
    finally:
        cudaq.set_target(original)


def test_cudaq_serializes_different_runtime_targets() -> None:
    """Concurrent Qamomile launches enter CUDA-Q with their own target."""

    cudaq = pytest.importorskip("cudaq")
    from qamomile.cudaq import Cudaq, CudaqLocal

    artifact = qm.compile(qm.build_source(BASIS_SOURCE).specialize("basis"), to=Cudaq())
    original_run = cudaq.run
    barrier = Barrier(2)
    seen: list[str] = []

    def observed_run(*args: object, **kwargs: object) -> object:
        seen.append(cudaq.get_target().name)
        return original_run(*args, **kwargs)

    def launch(target: str) -> tuple[str, dict[tuple[int, ...], int]]:
        barrier.wait(timeout=10)
        return target, qm.execute(artifact, on=CudaqLocal(target), shots=4).counts

    with patch.object(cudaq, "run", side_effect=observed_run):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(launch, ("qpp-cpu", "density-matrix-cpu")))

    assert {target: counts for target, counts in results} == {
        "qpp-cpu": {(1, 0, 0): 4},
        "density-matrix-cpu": {(1, 0, 0): 4},
    }
    assert seen == [target for target, _ in results] or seen == [
        target for target, _ in reversed(results)
    ]


def test_cudaq_reports_target_restoration_failure() -> None:
    """Restoration errors remain visible and release the reentrancy guard."""

    cudaq = pytest.importorskip("cudaq")
    from qamomile.cudaq import Cudaq, CudaqLocal
    from qamomile.errors import ExecutionError

    artifact = qm.compile(qm.build_source(BASIS_SOURCE).specialize("basis"), to=Cudaq())
    original_set_target = cudaq.set_target

    def fail_restoration(target: object) -> None:
        if not isinstance(target, str):
            raise RuntimeError("restore failed")
        original_set_target(target)

    with patch.object(cudaq, "set_target", side_effect=fail_restoration):
        with pytest.raises(ExecutionError, match="Q8519"):
            qm.execute(artifact, on=CudaqLocal(), shots=4)
    assert qm.execute(artifact, on=CudaqLocal(), shots=4).counts == {(1, 0, 0): 4}


def test_cudaq_target_callback_reentry_fails_without_deadlock() -> None:
    """An SDK target callback cannot nest another Qamomile launch."""

    cudaq = pytest.importorskip("cudaq")
    from qamomile.cudaq import Cudaq, CudaqLocal
    from qamomile.errors import ExecutionError

    artifact = qm.compile(qm.build_source(BASIS_SOURCE).specialize("basis"), to=Cudaq())
    original_set_target = cudaq.set_target
    observed: list[str] = []

    def set_target_with_callback(target: object) -> None:
        if isinstance(target, str) and target == "qpp-cpu":
            with pytest.raises(ExecutionError, match="Q8516") as error:
                qm.execute(artifact, on=CudaqLocal(), shots=4)
            observed.append(str(error.value))
        original_set_target(target)

    with patch.object(cudaq, "set_target", side_effect=set_target_with_callback):
        assert qm.execute(artifact, on=CudaqLocal(), shots=4).counts == {(1, 0, 0): 4}
    assert len(observed) == 1


def test_cudaq_batch_preserves_seeded_result_order() -> None:
    """Batch rows match serial CUDA-Q launches with advancing nonzero seeds."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import Cudaq, CudaqLocal

    artifact = qm.compile(qm.build_source(BELL_SOURCE).specialize("bell"), to=Cudaq())
    endpoint = CudaqLocal()
    batch = qm.execute_batch(
        artifact, on=endpoint, parameters=[{}, {}, {}], shots=64, seed=13
    )
    serial = tuple(
        qm.execute(artifact, on=endpoint, shots=64, seed=seed) for seed in range(13, 16)
    )
    assert tuple(result.counts for result in batch) == tuple(
        result.counts for result in serial
    )


def test_cudaq_coexists_with_other_local_sdk_endpoints() -> None:
    """All supported CPU endpoints execute the same source in one process."""

    pytest.importorskip("cudaq")
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("cirq")
    pytest.importorskip("qulacs")
    pytest.importorskip("braket")
    from qamomile.braket import BraketLocal
    from qamomile.cirq import CirqSimulator
    from qamomile.cudaq import CudaqLocal
    from qamomile.qiskit import Aer
    from qamomile.quri_parts import Qulacs

    program = qm.build_source(BASIS_SOURCE).specialize("basis")
    for endpoint in (CudaqLocal(), Aer(), CirqSimulator(), Qulacs(), BraketLocal()):
        assert qm.run(program, on=endpoint, shots=4, seed=0).result.counts == {
            (1, 0, 0): 4
        }


def test_cudaq_executes_extended_discrete_gate_family() -> None:
    """CUDA-Q preserves phase, decomposed ZZ, and multi-control semantics."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    program = qm.build_source(EXTENDED_GATE_SOURCE).specialize("extended_gates")
    assert qm.run(program, on=CudaqLocal(), shots=4).result.counts == {(1, 1, 1): 4}


def test_cudaq_executes_measure_reset() -> None:
    """CUDA-Q preserves measurement followed by reset."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    program = qm.build_source(MEASURE_RESET_SOURCE).specialize("measure_and_reset")
    assert qm.run(program, on=CudaqLocal(), shots=4).result.values == {((1,), (0,)): 4}


def test_cudaq_executes_nondestructive_pauli_measurements() -> None:
    """CUDA-Q preserves repeatable X, Y, and Z projective measurements."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    program = qm.build_source(MEASURE_AXES_SOURCE).specialize("measure_axes")
    result = qm.run(program, on=CudaqLocal(), shots=4, seed=2).result

    assert result.values == {(1, 1, 0, 0, 0, 0, (1,), (0,), (0,)): 4}


def test_cudaq_run_executes_measurement_feedback() -> None:
    """CUDA-Q run executes kernels that branch on measurement results."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import Cudaq, CudaqLocal

    program = qm.build_source(FEEDBACK_SOURCE).specialize("feedback")
    artifact = qm.compile(program, to=Cudaq())
    result = qm.execute(artifact, on=CudaqLocal(), shots=32, seed=5)

    generated = export_payload(artifact).decode("utf-8")
    assert isinstance(generated, str) and "if m0:" in generated
    assert set(result.counts) == {(0, 0), (1, 1)}


def test_cudaq_executes_boolean_measurement_expression() -> None:
    """CUDA-Q preserves ordinary ``not`` over a measured condition."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    result = qm.run(
        qm.build_source(NEGATED_FEEDBACK_SOURCE).specialize("negated_feedback"),
        on=CudaqLocal(),
        shots=32,
        seed=3,
    ).result

    assert set(result.counts) == {(0, 1), (1, 0)}


def test_cudaq_executes_modular_automatic_uncomputation() -> None:
    """CUDA-Q executes a kernel containing the common callable inverse."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import Cudaq, CudaqLocal

    program = qm.build_source(MODULAR_CLEANUP_SOURCE).specialize("modular_cleanup")
    artifact = qm.compile(program, to=Cudaq())
    result = qm.execute(artifact, on=CudaqLocal(), shots=8, seed=11)

    generated = export_payload(artifact).decode("utf-8")
    assert isinstance(generated, str) and ".adj(" in generated
    assert result.counts == {(1, 0, 0): 8}


def test_cudaq_executes_multi_input_automatic_uncomputation() -> None:
    """CUDA-Q executes a packed adjoint restoring independent inputs."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    program = qm.build_source(MULTI_INPUT_CLEANUP_SOURCE).specialize(
        "multi_input_cleanup"
    )
    result = qm.run(program, on=CudaqLocal(), shots=4, seed=7).result

    assert result.counts == {(1,): 4}


@pytest.mark.parametrize("segmented", [False, True], ids=["joint", "segmented"])
def test_cudaq_preserves_private_trailing_measurement_width(segmented: bool) -> None:
    """Joint circuits and closed segments execute non-result measurements."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import Cudaq, CudaqLocal

    source, entry = (
        (TRAILING_MEASUREMENT_SOURCE, "trailing_measurement")
        if segmented
        else (JOINT_PRIVATE_MEASUREMENT_SOURCE, "joint_private_measurement")
    )
    program = qm.build_source(source).specialize(entry)
    artifact = qm.compile(program, to=Cudaq())
    result = qm.execute(artifact, on=CudaqLocal(), shots=4, seed=3)

    if segmented:
        assert artifact.format == "qamomile.shot-execution.v1"
        assert artifact.result_abi.native_width == 1
        assert dict(result.execution_statistics.entries) == {
            "logical_shots": 4,
            "physical_segment_shots": 8,
            "submissions": 2,
        }
    else:
        assert artifact.format == "cudaq.python-source.v1"
        assert artifact.result_abi.native_width == 2
    assert artifact.result_abi.source_order == (0,)
    assert result.counts == {(0,): 4}


def test_cudaq_executes_typed_product_result() -> None:
    """CUDA-Q kernel returns flat transport bits with typed product decoding."""

    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    result = qm.run(
        qm.build_source(STRUCTURED_RESULT_SOURCE).specialize("structured_result"),
        on=CudaqLocal(),
        shots=4,
        seed=2,
    ).result

    assert result.counts == {(1, 0, 1): 4}
    assert result.values == {((1,), (0, 1)): 4}


@pytest.mark.parametrize("choice", ["add.increments", "add.qft"])
def test_cudaq_executes_late_realizations(choice: str) -> None:
    """One shared arithmetic source survives exact alternative realizations."""
    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal
    from qamomile.lowering import RealizationOptions

    from .test_realizations import exercise_realization_backend

    exercise_realization_backend(
        CudaqLocal(realizations=RealizationOptions({"quantum.add_into": choice}))
    )


def test_cudaq_zero_layer_qaoa() -> None:
    """Execute the shared zero layer qaoa contract locally."""
    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    from .test_library_compatibility import exercise_zero_layer_qaoa_backend

    exercise_zero_layer_qaoa_backend(CudaqLocal())


def test_cudaq_approximate_qft() -> None:
    """Execute the shared approximate qft contract locally."""
    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    from .test_library_compatibility import exercise_approximate_qft_backend

    exercise_approximate_qft_backend(CudaqLocal())


def test_cudaq_classical_array_compatibility() -> None:
    """Execute the shared classical array contract locally."""
    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqLocal

    from .test_classical_array_compatibility import exercise_classical_array_backend

    exercise_classical_array_backend(CudaqLocal())


def test_cudaq_qubit_selection() -> None:
    """Execute owned noncontiguous selection on this local SDK."""
    from qamomile.cudaq import CudaqLocal

    from .test_qubit_selection import exercise_qubit_selection_backend

    exercise_qubit_selection_backend(CudaqLocal(), seed=7)


def test_cudaq_keyed_parameters() -> None:
    """Rebind fixed-key dictionaries through the ordinary Param Vector ABI."""
    from qamomile.cudaq import CudaqLocal

    from .test_parameter_map import exercise_parameter_map_backend

    exercise_parameter_map_backend(CudaqLocal(), seed=7)


def test_cudaq_quantum_records() -> None:
    """Execute whole-record ownership transfers and coherent transforms."""
    from qamomile.cudaq import CudaqLocal

    from .test_quantum_records import exercise_quantum_record_backend

    exercise_quantum_record_backend(CudaqLocal(), seed=7)


def test_cudaq_recursion() -> None:
    """Execute self, mutual, branching, and higher-order recursive helpers."""
    from qamomile.cudaq import CudaqLocal

    from .test_recursion import exercise_recursive_backend

    exercise_recursive_backend(CudaqLocal(), seed=7)


def test_cudaq_sqrt() -> None:
    """Execute Const and launch-bound square roots through the native algebra."""
    from qamomile.cudaq import CudaqLocal

    from .test_sqrt import exercise_sqrt_backend

    exercise_sqrt_backend(CudaqLocal(), seed=7)
