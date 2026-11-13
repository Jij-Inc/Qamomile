"""End-to-end OpenQASM and Braket LocalSimulator tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

import qamomile as qm

from .conftest import (
    BASIS_SOURCE,
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
    subtract_register_sample,
)


def test_braket_executes_boolean_lifting() -> None:
    """LocalSimulator executes the shared native Boolean lift plans."""

    from qamomile.braket import BraketLocal

    exercise_boolean_lift_backend(endpoint=BraketLocal())


def test_braket_executes_symbolic_observable_recipe() -> None:
    """Braket executes an Observable assembled from Const coefficient data."""

    from qamomile.braket import BraketLocal

    exercise_symbolic_observable_backend(endpoint=BraketLocal())


def test_braket_executes_python_collection_loops() -> None:
    """LocalSimulator consumes enumerate/zip after shared Rust normalization."""

    from qamomile.braket import BraketLocal

    exercise_collection_loops_backend(endpoint=BraketLocal())


def test_braket_executes_derived_measurement_results() -> None:
    """LocalSimulator evaluates the retained result tree after sampling."""

    from qamomile.braket import BraketLocal

    endpoint = BraketLocal()
    exercise_derived_result_backend(endpoint=endpoint)
    exercise_broadcast_gate_backend(endpoint=endpoint)


def test_braket_measures_unsigned_integer_registers() -> None:
    """LocalSimulator preserves integer launch and typed result decoding."""

    from qamomile.braket import BraketLocal

    exercise_uint_measurement_backend(endpoint=BraketLocal())


def test_braket_materializes_dynamic_qubit_selection_at_launch() -> None:
    """Rust resolves a launch-time wire selector before OpenQASM emission."""

    from qamomile.braket import BraketLocal

    exercise_dynamic_qubit_selection_backend(endpoint=BraketLocal())


def test_braket_materializes_dynamic_classical_index_at_launch() -> None:
    """Rust resolves an aggregate path before OpenQASM emission."""

    from qamomile.braket import BraketLocal

    exercise_dynamic_classical_index_backend(endpoint=BraketLocal())


def test_braket_executes_launch_bound_structural_math() -> None:
    """Braket receives the loop after Rust evaluates structural math."""

    from qamomile.braket import BraketLocal

    exercise_structural_math_backend(endpoint=BraketLocal())


def test_braket_executes_multi_owner_host_dispatch() -> None:
    """Braket variants retain both independently updated owned outputs."""

    from qamomile.braket import BraketLocal

    exercise_multi_owner_dispatch_backend(endpoint=BraketLocal())


def test_braket_executes_rust_lowered_host_dispatch_family() -> None:
    """One OpenQASM family evaluates an integer launch predicate."""

    from qamomile.braket import BraketLocal

    endpoint = BraketLocal()
    program = qm.build_source(
        HOST_DISPATCH_SOURCE,
        module_name="braket_host_dispatch_family",
    ).specialize("host_dispatch")
    artifact = qm.compile(program, to=endpoint.target)

    assert artifact.format == "qamomile.artifact-family.v3"
    for limit, expected in ((-2, 0), (3, 1)):
        result = qm.execute(
            artifact,
            on=endpoint,
            parameters={"limit": limit},
            shots=8,
            seed=0,
        )
        assert result.counts == {(expected,): 8}


def test_braket_materializes_a_launch_dependent_loop() -> None:
    """Braket receives one Rust-expanded circuit for the requested loop count."""

    from qamomile.braket import BraketLocal

    endpoint = BraketLocal()
    program = qm.build_source(
        LAUNCH_LOOP_SOURCE,
        module_name="braket_launch_loop",
    ).specialize("launch_loop")
    artifact = qm.compile(program, to=endpoint.target)

    assert artifact.format == "qamomile.launch-materializer.v1"
    for rounds, expected in ((2, 0), (3, 1)):
        result = qm.execute(
            artifact,
            on=endpoint,
            parameters={"rounds": rounds},
            shots=8,
            seed=0,
        )
        assert result.counts == {(expected,): 8}


def test_braket_executes_documented_quantum_kernel_classification() -> None:
    """Braket executes the documented overlap kernel without a custom path."""

    from qamomile.braket import BraketLocal

    exercise_classification_backend(endpoint=BraketLocal(), seed=0)


def test_braket_executes_documented_hydrogen_vqe() -> None:
    """Braket estimates the documented hydrogen trial energy."""

    from qamomile.braket import BraketLocal

    exercise_hydrogen_vqe_backend(endpoint=BraketLocal(), seed=0)


def test_braket_executes_documented_qsci_sampling() -> None:
    """Braket samples the documented QSCI trial-state configurations."""

    from qamomile.braket import BraketLocal

    exercise_qsci_backend(endpoint=BraketLocal(), seed=0)


def test_braket_executes_documented_hamiltonian_simulation() -> None:
    """Braket executes the documented noncommuting Rabi evolution."""

    from qamomile.braket import BraketLocal

    exercise_rabi_backend(endpoint=BraketLocal(), seed=0)


def test_braket_executes_documented_mottonen_encoding() -> None:
    """Braket prepares the documented complex Möttönen state."""

    from qamomile.braket import BraketLocal

    exercise_mottonen_backend(endpoint=BraketLocal(), seed=0)


def test_braket_executes_documented_multidimensional_qft() -> None:
    """Braket executes the documented axiswise Fourier transform."""

    from qamomile.braket import BraketLocal

    exercise_multidimensional_qft_backend(endpoint=BraketLocal(), seed=0)


def test_braket_executes_documented_hybrid_qnn() -> None:
    """Braket executes the documented runtime Hybrid QNN layer."""

    from qamomile.braket import BraketLocal

    exercise_hqnn_backend(endpoint=BraketLocal(), seed=0)


def test_braket_executes_documented_option_pricing() -> None:
    """Braket executes the documented amplitude-estimation option pricer."""

    from qamomile.braket import BraketLocal

    exercise_option_pricing_backend(endpoint=BraketLocal(), seed=0)


def test_braket_executes_documented_pce() -> None:
    """Braket estimates and decodes the documented PCE state."""

    from qamomile.braket import BraketLocal

    exercise_pce_backend(endpoint=BraketLocal(), seed=0)


def test_braket_executes_documented_qemcmc() -> None:
    """Braket executes the documented quantum-enhanced MCMC proposal."""

    from qamomile.braket import BraketLocal

    exercise_qemcmc_backend(endpoint=BraketLocal(), seed=0)


def test_braket_executes_documented_bit_flip_code() -> None:
    """Braket coherently corrects each single data-qubit bit flip."""

    from qamomile.braket import BraketLocal

    exercise_bit_flip_code_backend(endpoint=BraketLocal(), seed=0)


def test_braket_executes_documented_steane_code() -> None:
    """Braket coherently corrects a bounded Steane error case."""

    from qamomile.braket import BraketLocal

    exercise_steane_code_backend(endpoint=BraketLocal(), seed=0)


def test_braket_executes_variational_composition() -> None:
    """Braket executes QAOA and FQAOA without algorithm-specific lowering."""

    from qamomile.braket import BraketLocal

    exercise_variational_backend(endpoint=BraketLocal())


def test_braket_executes_grover_adaptive_search() -> None:
    """LocalSimulator executes GAS composed from ordinary reusable qkernels."""

    from qamomile.braket import BraketLocal

    exercise_gas_backend(endpoint=BraketLocal(), seed=0)


def test_braket_executes_product_formula_composition() -> None:
    """Braket executes the shared abstract-evolution schedule."""

    from qamomile.braket import BraketLocal

    exercise_product_formula_backend(endpoint=BraketLocal())


def test_braket_executes_uniformly_controlled_rotation() -> None:
    """LocalSimulator executes the public function-composed Gray walk."""

    from qamomile.braket import BraketLocal

    exercise_uniformly_controlled_backend(endpoint=BraketLocal())


def test_braket_executes_coherent_xor_lookup() -> None:
    """LocalSimulator executes lookup through total callable selection."""

    from qamomile.braket import BraketLocal

    exercise_xor_lookup_backend(endpoint=BraketLocal())


def test_braket_executes_sparse_matrix_lookup() -> None:
    """LocalSimulator executes functional sparse row/slot access."""

    from qamomile.braket import BraketLocal

    exercise_sparse_lookup_backend(endpoint=BraketLocal())


def test_braket_executes_runtime_matrix_access() -> None:
    """Braket binds a shaped Matrix while rotations use scalar leaves."""

    from qamomile.braket import BraketLocal

    exercise_runtime_matrix_backend(endpoint=BraketLocal())


def test_braket_executes_runtime_qsvt_phases() -> None:
    """Braket reuses one QSVT circuit with a phase Vector launch argument."""

    from qamomile.braket import BraketLocal

    exercise_runtime_qsvt_backend(endpoint=BraketLocal())


@dataclass
class _FakeBraketResult:
    """Minimal native Braket result used at the endpoint contract boundary."""

    measurement_counts: dict[str, int]


class _FakeBraketTask:
    """Minimal completed AWS task shape used without network access."""

    id = "task-123"

    def state(self) -> str:
        """Return the native completed state."""

        return "COMPLETED"

    def result(self) -> _FakeBraketResult:
        """Return q0-first measurement counts."""

        return _FakeBraketResult({"100": 4})


class _FakeBraketDevice:
    """Capture one managed-device submission without an AWS session."""

    name = "fake-qpu"
    arn = "arn:aws:braket:::device/qpu/fake"
    properties = {"action": "OPENQASM"}

    def __init__(self) -> None:
        """Initialize an empty launch record."""

        self.launch: dict[str, Any] | None = None

    def run(self, program: object, **options: object) -> _FakeBraketTask:
        """Record the portable program and return a completed task."""

        self.launch = {"program": program, **options}
        return _FakeBraketTask()


pytestmark = pytest.mark.braket
pytest.importorskip("braket")


def test_braket_executes_signal_composition() -> None:
    """Execute shared region phases and editable signal schedules."""
    from qamomile.braket import BraketLocal

    from .test_signals import exercise_signals_backend

    exercise_signals_backend(endpoint=BraketLocal())


def test_braket_executes_standard_optimized_qcore() -> None:
    """Braket executes the shared exact QCore optimization witness."""

    from qamomile.braket import BraketLocal

    from .test_standard_optimization import exercise_standard_optimization_backend

    exercise_standard_optimization_backend(endpoint=BraketLocal())


def test_braket_executes_functional_amplitude_encoding() -> None:
    """LocalSimulator samples and estimates the multilevel Gray-walk body."""

    from qamomile.braket import BraketLocal

    endpoint = BraketLocal()
    assert qm.run(amplitude_two_sample, on=endpoint, shots=4).result.counts == {
        (1, 1): 4
    }
    assert (
        qm.expectation(amplitude_two_state, qm.Z(0), on=endpoint, shots=4).value == -1.0
    )


def test_braket_reuses_runtime_parametric_amplitude_artifact() -> None:
    """OpenQASM exposes scalar leaves while Qamomile keeps one aggregate ABI."""

    from qamomile.braket import Braket, BraketLocal

    exercise_runtime_amplitude_backend(endpoint=BraketLocal(), target=Braket())


def test_braket_executes_functional_pauli_block_encoding() -> None:
    """Braket executes Pauli production through the existing LCU body."""

    from qamomile.braket import BraketLocal

    exercise_pauli_block_backend(endpoint=BraketLocal())


def test_braket_executes_functional_block_producers() -> None:
    """Braket executes projected and periodic construction through common IR."""

    from qamomile.braket import BraketLocal

    exercise_functional_block_producers_backend(endpoint=BraketLocal())


def test_braket_executes_standard_python_record_results() -> None:
    """Braket reconstructs field names from the shared product result ABI."""

    from qamomile.braket import BraketLocal

    program = qm.build_source(RECORD_RESULT_SOURCE).specialize("record_result")
    assert qm.run(program, on=BraketLocal(), shots=4).result.values == {
        (("first", (1,)), ("pair", (0, 1))): 4
    }


def test_braket_executes_callable_transforms_and_exact_relative_phase() -> None:
    """OpenQASM composes control/multiplex and retains controlled scalar phase."""

    from qamomile.braket import BraketLocal

    selected = qm.build_source(CALLABLE_TRANSFORMS_SOURCE).specialize(
        "callable_transforms"
    )
    phased = qm.build_source(CONTROLLED_PHASE_SOURCE).specialize("controlled_phase")

    assert qm.run(selected, on=BraketLocal(), shots=4).result.values == {(1, (1,)): 4}
    assert qm.run(phased, on=BraketLocal(), shots=4).result.values == {(1, (0,)): 4}


def test_braket_executes_compiler_derived_selected_adjoint() -> None:
    """Braket executes multiplex(U_i-adjoint) from the common dagger."""

    from qamomile.braket import BraketLocal

    program = qm.build_source(COHERENT_ADJOINT_SOURCE).specialize(
        "selected_adjoint_roundtrip"
    )
    assert qm.run(program, on=BraketLocal(), shots=4).result.counts == {(1, 0): 4}
    exercise_normalized_model_backend(endpoint=BraketLocal(), seed=0)


def test_braket_executes_indexed_aggregate_control() -> None:
    """Braket samples and estimates selector-based coherent control."""

    from qamomile.braket import BraketLocal

    module = qm.build_source(INDEXED_CONTROL_SOURCE)
    assert qm.run(
        module["indexed_control_sample"], on=BraketLocal(), shots=4
    ).result.counts == {(1, 0, 1): 4}
    assert (
        qm.expectation(
            module["indexed_control_state"],
            qm.Z(2),
            on=BraketLocal(),
            shots=4,
        ).value
        == -1.0
    )


def test_braket_executes_abstract_pauli_evolution() -> None:
    """The shared circuit projection lowers a Pauli gadget once for Braket."""

    from qamomile.braket import BraketLocal

    program = qm.build_source(PAULI_EVOLUTION_SOURCE).specialize(
        "pauli_evolution",
        **{"hamiltonian": 0.5 * qm.Z(0) * qm.Z(1)},
    )
    assert qm.run(program, on=BraketLocal(), shots=4).result.counts == {(1, 0): 4}


def test_braket_estimates_a_backend_neutral_observable() -> None:
    """Expectation basis programs execute through the ordinary Braket endpoint."""

    from qamomile.braket import BraketLocal

    state = qm.build_source(STATE_PREPARATION_SOURCE)["plus_state"]
    assert qm.expectation(state, qm.X(0), on=BraketLocal(), shots=4).value == 1.0


def test_braket_executes_imported_qft_roundtrip() -> None:
    """The linked functional QFT library executes through Braket LocalSimulator."""

    from qamomile.braket import BraketLocal

    assert qm.run(qft_roundtrip, on=BraketLocal(), shots=4).result.counts == {
        (1, 0, 1): 4
    }
    assert qm.run(qft_frequency_three, on=BraketLocal(), shots=4).result.counts == {
        (1, 1, 0): 4
    }


def test_braket_executes_functional_phase_estimation() -> None:
    """Braket samples and estimates body-derived two-bit QPE."""

    from qamomile.braket import BraketLocal

    assert qm.run(qpe_sample, on=BraketLocal(), shots=4).result.counts == {(1, 0, 1): 4}
    assert qm.expectation(qpe_state, qm.Z(0), on=BraketLocal(), shots=4).value == -1.0
    exercise_nary_tensor_backend(endpoint=BraketLocal())


def test_braket_decodes_typed_fixed_point_after_q0_first_order() -> None:
    """Braket q0-first strings are normalized before numeric phase decoding."""

    from qamomile.braket import BraketLocal

    exercise_fixed_qpe_backend(endpoint=BraketLocal())
    exercise_windowed_qpe_backend(endpoint=BraketLocal())
    exercise_documented_windowed_qpe_backend(endpoint=BraketLocal())
    exercise_documented_reversible_arithmetic_backend(endpoint=BraketLocal())
    exercise_documented_gf2_arithmetic_backend(endpoint=BraketLocal())
    exercise_documented_phase_gradient_backend(endpoint=BraketLocal())
    exercise_documented_fault_tolerant_data_loading_backend(endpoint=BraketLocal())
    exercise_documented_reversible_indexing_backend(endpoint=BraketLocal())
    exercise_documented_reversible_trigonometry_backend(endpoint=BraketLocal())
    exercise_documented_kikuchi_backend(endpoint=BraketLocal())
    exercise_documented_guided_hamiltonian_backend(endpoint=BraketLocal())


def test_braket_executes_documented_qualtran_gap_examples() -> None:
    """Braket executes the bounded ECC and factorized-chemistry witnesses."""

    from qamomile.braket import BraketLocal

    exercise_documented_elliptic_curve_dlog_backend(endpoint=BraketLocal())
    exercise_documented_factorized_chemistry_backend(endpoint=BraketLocal())


def test_braket_executes_functional_grover_search() -> None:
    """LocalSimulator samples and estimates body-composed amplification."""

    from qamomile.braket import BraketLocal

    assert qm.run(grover_sample, on=BraketLocal(), shots=4).result.counts == {(1, 1): 4}
    assert (
        qm.expectation(grover_state, qm.Z(0), on=BraketLocal(), shots=4).value == -1.0
    )


def test_braket_executes_functional_modular_arithmetic() -> None:
    """Braket executes bounded sample/expectation arithmetic witnesses."""

    from qamomile.braket import Braket, BraketLocal, load_openqasm

    exercise_modular_arithmetic_backend(endpoint=BraketLocal())
    source = load_openqasm(qm.compile(subtract_register_sample, to=Braket()))
    assert "cnot q[1], q[4];" in source
    assert "ccnot q[1], q[4], q[5];" in source
    assert "ctrl @ ctrl @ x" not in source


def test_braket_executes_shor_order_finding() -> None:
    """Braket Local executes the shared Shor sampling and state witnesses."""

    from qamomile.braket import BraketLocal

    exercise_shor_order_finding_backend(endpoint=BraketLocal(), seed=0)


def test_braket_executes_regev_factoring_sample() -> None:
    """Braket Local executes the shared bounded Regev sample."""

    from qamomile.braket import BraketLocal

    exercise_regev_factoring_backend(endpoint=BraketLocal(), seed=0)


def test_braket_executes_multi_exponentiation_callable_realization() -> None:
    """Braket Local executes the shared source after callable replacement."""

    from qamomile.braket import BraketLocal

    options = qm.RealizationOptions(
        {multi_exponentiation_baseline: multi_exponentiation_product_tree}
    )
    exercise_multi_exponentiation_realization_backend(
        endpoint=BraketLocal(realizations=options), seed=0
    )


def test_braket_executes_first_class_multi_exponentiation() -> None:
    """Braket executes the primitive through its independent realization."""

    from qamomile.braket import BraketLocal

    exercise_first_class_multi_exponentiation_backend(endpoint=BraketLocal(), seed=0)


def test_braket_executes_register_modulus_arithmetic() -> None:
    """Braket samples retained-modulus addition."""

    from qamomile.braket import BraketLocal

    exercise_register_modulus_backend(endpoint=BraketLocal(), seed=0)


def test_braket_executes_functional_qsvt() -> None:
    """LocalSimulator executes direct and LCU-composed QSVT bodies."""

    from qamomile.braket import BraketLocal

    endpoint = BraketLocal()
    assert qm.run(qsvt_phase_sample, on=endpoint, shots=4).result.counts == {(1, 0): 4}
    assert (
        qm.expectation(
            qsvt_phase_state,
            qm.Z(0),
            on=endpoint,
            shots=4,
        ).value
        == -1.0
    )
    assert qm.run(lcu_qsvt_sample, on=endpoint, shots=4).result.counts == {(1, 0): 4}
    assert (
        qm.expectation(
            lcu_qsvt_state,
            qm.Z(0),
            on=endpoint,
            shots=4,
        ).value
        == -1.0
    )
    assert qm.run(reciprocal_x_sample, on=endpoint, shots=4).result.counts == {
        (0, 0, 1): 4
    }
    assert (
        qm.expectation(
            reciprocal_x_state,
            qm.Z(2),
            on=endpoint,
            shots=4,
        ).value
        == -1.0
    )


def test_braket_executes_coherent_product_reassociation() -> None:
    """Braket receives only the gates surrounding a structural reshape."""

    from qamomile.braket import BraketLocal

    program = qm.build_source(COHERENT_RESHAPE_SOURCE).specialize("branched_adjoint")
    assert qm.run(program, on=BraketLocal(), shots=4).result.counts == {(1, 0, 1): 4}


def test_braket_openqasm_executes_basis_program() -> None:
    """The Braket artifact is UTF-8 OpenQASM and preserves source bit order."""

    from qamomile.braket import Braket, BraketLocal, load_openqasm

    program = qm.build_source(BASIS_SOURCE).specialize("basis")
    artifact = qm.compile(program, to=Braket())
    with patch(
        "subprocess.run",
        side_effect=AssertionError("worker used"),
    ):
        result = qm.execute(artifact, on=BraketLocal(), shots=16)

    assert artifact.format == "braket.openqasm3.v1"
    assert load_openqasm(artifact).startswith("OPENQASM 3.0;")
    assert result.counts == {(1, 0, 0): 16}
    assert result.expectation_z(0, 2) == -1.0
    assert result.worker_pid is None


def test_braket_executes_extended_discrete_gate_family() -> None:
    """OpenQASM preserves phase, ZZ, and Toffoli primitives locally."""

    from qamomile.braket import BraketLocal

    program = qm.build_source(EXTENDED_GATE_SOURCE).specialize("extended_gates")
    assert qm.run(program, on=BraketLocal(), shots=4).result.counts == {(1, 1, 1): 4}


def test_braket_executes_measure_reset() -> None:
    """OpenQASM preserves measurement followed by reset."""

    from qamomile.braket import BraketLocal

    program = qm.build_source(MEASURE_RESET_SOURCE).specialize("measure_and_reset")
    assert qm.run(program, on=BraketLocal(), shots=4).result.values == {((1,), (0,)): 4}


def test_braket_executes_nondestructive_pauli_measurements() -> None:
    """OpenQASM preserves repeatable X, Y, and Z projective measurements."""

    from qamomile.braket import BraketLocal

    program = qm.build_source(MEASURE_AXES_SOURCE).specialize("measure_axes")
    result = qm.run(program, on=BraketLocal(), shots=4).result

    assert result.values == {(1, 1, 0, 0, 0, 0, (1,), (0,), (0,)): 4}


def test_braket_executes_measurement_feedback() -> None:
    """Shot-time Python if remains OpenQASM device control."""

    from qamomile.braket import Braket, BraketLocal, load_openqasm

    program = qm.build_source(FEEDBACK_SOURCE).specialize("feedback")
    artifact = qm.compile(program, to=Braket())
    result = qm.execute(artifact, on=BraketLocal(), shots=32)

    assert "if (c[0])" in load_openqasm(artifact)
    assert set(result.counts) <= {(0, 0), (1, 1)}


def test_braket_executes_boolean_measurement_expression() -> None:
    """OpenQASM preserves ordinary ``not`` over a measured condition."""

    from qamomile.braket import BraketLocal

    result = qm.run(
        qm.build_source(NEGATED_FEEDBACK_SOURCE).specialize("negated_feedback"),
        on=BraketLocal(),
        shots=32,
    ).result

    assert set(result.counts) <= {(0, 1), (1, 0)}


def test_braket_executes_modular_automatic_uncomputation() -> None:
    """OpenQASM maps callable S/T adjoints to Braket's inverse gates."""

    from qamomile.braket import Braket, BraketLocal, load_openqasm

    program = qm.build_source(MODULAR_CLEANUP_SOURCE).specialize("modular_cleanup")
    artifact = qm.compile(program, to=Braket())
    source = load_openqasm(artifact)
    result = qm.execute(artifact, on=BraketLocal(), shots=8)

    assert "ti q[1];" in source
    assert "si q[0];" in source
    assert source.index("ti q[1];") < source.index("c[0] = measure")
    assert result.counts == {(1, 0, 0): 8}


def test_braket_executes_multi_input_automatic_uncomputation() -> None:
    """OpenQASM executes a packed adjoint restoring independent inputs."""

    from qamomile.braket import BraketLocal

    program = qm.build_source(MULTI_INPUT_CLEANUP_SOURCE).specialize(
        "multi_input_cleanup"
    )
    result = qm.run(program, on=BraketLocal(), shots=4).result

    assert result.counts == {(1,): 4}


@pytest.mark.parametrize("segmented", [False, True], ids=["joint", "segmented"])
def test_braket_preserves_private_trailing_measurement_width(segmented: bool) -> None:
    """Joint circuits and closed segments execute non-result measurements."""

    from qamomile.braket import Braket, BraketLocal, load_openqasm

    source, entry = (
        (TRAILING_MEASUREMENT_SOURCE, "trailing_measurement")
        if segmented
        else (JOINT_PRIVATE_MEASUREMENT_SOURCE, "joint_private_measurement")
    )
    program = qm.build_source(source).specialize(entry)
    artifact = qm.compile(program, to=Braket())
    result = qm.execute(artifact, on=BraketLocal(), shots=4)

    if segmented:
        assert artifact.format == "qamomile.shot-execution.v1"
        assert artifact.result_abi.native_width == 1
        assert dict(result.execution_statistics.entries) == {
            "logical_shots": 4,
            "physical_segment_shots": 8,
            "submissions": 2,
        }
    else:
        assert "bit[2] c;" in load_openqasm(artifact)
        assert artifact.result_abi.native_width == 2
    assert artifact.result_abi.source_order == (0,)
    assert result.counts == {(0,): 4}


def test_braket_rejects_an_unavailable_seed_contract() -> None:
    """The local endpoint never silently drops a requested random seed."""

    from qamomile.braket import Braket, BraketLocal

    program = qm.build_source(BASIS_SOURCE).specialize("basis")
    artifact = qm.compile(program, to=Braket())

    with pytest.raises(qm.ExecutionError, match="Q8414"):
        qm.execute(artifact, on=BraketLocal(), shots=1, seed=9)


def test_braket_executes_typed_product_result() -> None:
    """OpenQASM sampling preserves the typed product result layout."""

    from qamomile.braket import BraketLocal

    result = qm.run(
        qm.build_source(STRUCTURED_RESULT_SOURCE).specialize("structured_result"),
        on=BraketLocal(),
        shots=4,
    ).result

    assert result.counts == {(1, 0, 1): 4}
    assert result.values == {((1,), (0, 1)): 4}


def test_braket_device_submits_portable_openqasm_through_common_job() -> None:
    """A managed device remains a thin live adapter outside the artifact."""

    from qamomile.braket import BraketDevice

    native = _FakeBraketDevice()
    endpoint = BraketDevice(native, poll_interval_seconds=0.001)
    program = qm.build_source(BASIS_SOURCE).specialize("basis")
    record = qm.run(program, on=endpoint, shots=4)

    assert record.result.counts == {(1, 0, 0): 4}
    assert native.launch is not None
    assert native.launch["shots"] == 4
    assert record.artifact.target_fingerprint == endpoint.target.fingerprint


@pytest.mark.parametrize("choice", ["add.increments", "add.qft", "add.ripple"])
def test_braket_executes_late_realizations(choice: str) -> None:
    """One shared arithmetic source survives exact alternative realizations."""
    pytest.importorskip("braket")
    from qamomile.braket import BraketLocal
    from qamomile.lowering import RealizationOptions

    from .test_realizations import exercise_realization_backend

    exercise_realization_backend(
        BraketLocal(realizations=RealizationOptions({"quantum.add_into": choice}))
    )


def test_braket_zero_layer_qaoa() -> None:
    """Execute the shared zero layer qaoa contract locally."""
    from qamomile.braket import BraketLocal

    from .test_library_compatibility import exercise_zero_layer_qaoa_backend

    exercise_zero_layer_qaoa_backend(BraketLocal(), seed=0)


def test_braket_approximate_qft() -> None:
    """Execute the shared approximate qft contract locally."""
    from qamomile.braket import BraketLocal

    from .test_library_compatibility import exercise_approximate_qft_backend

    exercise_approximate_qft_backend(BraketLocal(), seed=0)


def test_braket_classical_array_compatibility() -> None:
    """Execute the shared classical array contract locally."""
    from qamomile.braket import BraketLocal

    from .test_classical_array_compatibility import exercise_classical_array_backend

    exercise_classical_array_backend(BraketLocal(), seed=0)


def test_braket_qubit_selection() -> None:
    """Execute owned noncontiguous selection on this local SDK."""
    from qamomile.braket import BraketLocal

    from .test_qubit_selection import exercise_qubit_selection_backend

    exercise_qubit_selection_backend(BraketLocal(), seed=0)


def test_braket_keyed_parameters() -> None:
    """Rebind fixed-key dictionaries through the ordinary Param Vector ABI."""
    from qamomile.braket import BraketLocal

    from .test_parameter_map import exercise_parameter_map_backend

    exercise_parameter_map_backend(BraketLocal(), seed=0)


def test_braket_quantum_records() -> None:
    """Execute whole-record ownership transfers and coherent transforms."""
    from qamomile.braket import BraketLocal

    from .test_quantum_records import exercise_quantum_record_backend

    exercise_quantum_record_backend(BraketLocal(), seed=0)


def test_braket_recursion() -> None:
    """Execute self, mutual, branching, and higher-order recursive helpers."""
    from qamomile.braket import BraketLocal

    from .test_recursion import exercise_recursive_backend

    exercise_recursive_backend(BraketLocal(), seed=0)


def test_braket_sqrt() -> None:
    """Execute Const and launch-bound square roots through the native algebra."""
    from qamomile.braket import BraketLocal

    from .test_sqrt import exercise_sqrt_backend

    exercise_sqrt_backend(BraketLocal(), seed=0)
