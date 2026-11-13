"""End-to-end QPY and Aer tests for Qamomile."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

import qamomile as qm
from qamomile.effects import Requirement
from qamomile.persistence import deserialize, serialize
from qamomile.stdlib import qft

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
    LAUNCH_LOOP_SOURCE,
    MEASURE_AXES_SOURCE,
    MEASURE_RESET_SOURCE,
    MODULAR_CLEANUP_SOURCE,
    MULTI_INPUT_CLEANUP_SOURCE,
    NEGATED_FEEDBACK_SOURCE,
    PAULI_EVOLUTION_SOURCE,
    RECORD_RESULT_SOURCE,
    ROTATION_SOURCE,
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
    linear_system_reciprocal,
    linear_system_reciprocal_state,
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


@qm.composite_gate(name="entangle")
@qm.qkernel
def _boxed_entangle(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    """Return one body-backed named Bell entangler.

    Args:
        state: Two-qubit owned state.

    Returns:
        Entangled successor state.
    """

    state = qm.h(state, target=0)
    return qm.cx(state, control=0, target=1)


@qm.qkernel
def _boxed_bell_sample() -> qm.Shot[qm.Bits[2]]:
    """Measure one named composite invocation.

    Returns:
        Two correlated sample bits.
    """

    return qm.measure(_boxed_entangle(qm.qubits(2)))


_native_qft = qm.composite_gate(qft, name="qft")


@qm.composite_gate(name="qft")
@qm.qkernel
def _misleading_qft(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    """A label cannot replace an X body with a Fourier transform."""
    return qm.x(state, target=0)


@qm.qkernel
def _misleading_qft_sample() -> qm.Bits[2]:
    """Observe the verified body despite its deliberately misleading label."""
    return qm.measure(_misleading_qft(qm.qubits(2)))


@qm.qkernel
def _native_qft_state() -> qm.Q[qm.Bits[3]]:
    """Return a named native-QFT realization witness.

    Returns:
        Three-qubit Fourier-transformed zero state.
    """

    return _native_qft(qm.qubits(3), 3)


@qm.qkernel
def _decomposed_qft_state() -> qm.Q[qm.Bits[3]]:
    """Return the ordinary body-derived QFT witness.

    Returns:
        Three-qubit Fourier-transformed zero state.
    """

    return qft(qm.qubits(3), 3)


class _FakeRuntimeJob:
    """Small Runtime primitive job used by adapter contract tests."""

    def __init__(self, result: object, identity: str = "runtime-job") -> None:
        """Store one provider result and a stable identity."""

        self._result = result
        self._identity = identity
        self._status = "QUEUED"
        self.cancelled = False

    def job_id(self) -> str:
        """Return the stable fake provider identity."""

        return self._identity

    def status(self) -> str:
        """Return the current fake provider status."""

        return self._status

    def result(self) -> object:
        """Complete and return the configured primitive result."""

        self._status = "DONE"
        return self._result

    def cancel(self) -> None:
        """Record a provider cancellation request."""

        self.cancelled = True
        self._status = "CANCELLED"


class _FakeRuntimeSampler:
    """Capture Sampler V2 submissions without network access."""

    jobs: list[_FakeRuntimeJob] = []
    rows: list[str] | None = None

    def __init__(self, *, mode: object, options: object) -> None:
        """Retain the selected backend and options."""

        self._backend = mode
        self.options = options
        self.submissions = 0

    def backend(self) -> object:
        """Return the selected Runtime backend."""

        return self._backend

    def run(self, pubs: list[object], *, shots: int) -> _FakeRuntimeJob:
        """Return classified data for one submitted circuit."""

        from qiskit.primitives.containers import BitArray

        assert len(pubs) == 1
        self.submissions += 1
        circuit = pubs[0]
        rows = (
            [self.rows[index % len(self.rows)] for index in range(shots)]
            if self.rows
            else ["0" * len(circuit.clbits)] * (shots // 2)
            + ["1" * len(circuit.clbits)] * (shots - shots // 2)
        )
        registers = {}
        for register in circuit.cregs:
            indices = [circuit.find_bit(bit).index for bit in register]
            register_rows = [
                "".join(row[-1 - index] for index in reversed(indices)) for row in rows
            ]
            registers[register.name] = BitArray.from_samples(
                register_rows,
                num_bits=len(register),
            )
        result = [SimpleNamespace(data=SimpleNamespace(**registers))]
        job = _FakeRuntimeJob(result, f"runtime-job-{len(self.jobs)}")
        self.jobs.append(job)
        return job


def _basic_simulator_endpoint() -> Any:
    """Return the lightweight Qiskit endpoint used by the fast ABI smoke loop."""

    from qiskit.providers.basic_provider import BasicSimulator

    from qamomile.qiskit import QiskitBackend

    return QiskitBackend(BasicSimulator())


def test_qiskit_executes_signal_composition() -> None:
    """Execute shared region phases and editable signal schedules."""
    from qamomile.qiskit import Aer

    from .test_signals import exercise_signals_backend

    exercise_signals_backend(endpoint=Aer())


def test_qiskit_executes_standard_optimized_qcore() -> None:
    """Qiskit executes the shared exact QCore optimization witness."""

    from qamomile.qiskit import Aer

    from .test_standard_optimization import exercise_standard_optimization_backend

    exercise_standard_optimization_backend(endpoint=Aer())


def test_qiskit_ibm_runtime_submit_snapshot_cancel_and_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime jobs remain asynchronous, restorable, and ResultABI checked."""

    from qiskit.providers.basic_provider import BasicSimulator

    import qamomile.qiskit.runtime as runtime_module
    from qamomile.qiskit import IBMRuntime

    backend = BasicSimulator()
    restored_jobs: dict[str, _FakeRuntimeJob] = {}

    class Service:
        """Restore fake jobs by provider identity."""

        def job(self, identity: str) -> _FakeRuntimeJob:
            """Return the existing fake job without resubmission."""

            return restored_jobs[identity]

    monkeypatch.setattr(
        runtime_module,
        "_runtime_module",
        lambda: SimpleNamespace(SamplerV2=_FakeRuntimeSampler),
    )
    endpoint = IBMRuntime(backend, service=Service())
    program = qm.build_source(BELL_SOURCE).specialize("bell")
    artifact = qm.compile(program, to=endpoint.target)

    from qiskit import qpy

    def forbid_qpy(*args, **kwargs):
        raise AssertionError("Runtime submission must not serialize a circuit")

    monkeypatch.setattr(qpy, "dump", forbid_qpy)
    monkeypatch.setattr(qpy, "load", forbid_qpy)

    job = qm.submit(artifact, on=endpoint, shots=4, seed=0)
    native = endpoint._sampler.jobs[-1]
    restored_jobs[job.job_id] = native

    assert job.status is qm.JobStatus.SUBMITTED
    snapshot = qm.ExecutionSnapshot.from_bytes(job.snapshot().to_bytes())
    assert snapshot.references()[0].provider == "qiskit-runtime"
    assert "token" not in snapshot.references()[0].context_values
    restored = endpoint.restore(snapshot, artifact=artifact)
    assert endpoint._sampler.submissions == 1
    restored_result = restored.result()
    assert restored_result.counts == {(0, 0): 2, (1, 1): 2}
    assert restored_result.result_abi == artifact.result_abi
    cached = qm.ExecutionSnapshot.from_bytes(restored.snapshot().to_bytes())
    assert cached.kind is qm.ExecutionSnapshotKind.LOCAL
    detached = endpoint.restore(cached, artifact=artifact)
    assert detached.result() == restored_result

    other = qm.submit(artifact, on=endpoint, shots=2, seed=0)
    other_native = endpoint._sampler.jobs[-1]
    other.cancel()
    assert other.status is qm.JobStatus.CANCELLED
    assert other_native.cancelled


def test_qiskit_ibm_runtime_measures_unsigned_integer_registers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime transport preserves UInt decoding and Qiskit register order."""

    from qiskit.providers.basic_provider import BasicSimulator

    import qamomile.qiskit.runtime as runtime_module
    from qamomile.qiskit import IBMRuntime

    monkeypatch.setattr(_FakeRuntimeSampler, "rows", ["011"])
    monkeypatch.setattr(
        runtime_module,
        "_runtime_module",
        lambda: SimpleNamespace(SamplerV2=_FakeRuntimeSampler),
    )
    endpoint = IBMRuntime(BasicSimulator())
    exercise_uint_measurement_backend(endpoint=endpoint, seed=0)

    monkeypatch.setattr(_FakeRuntimeSampler, "rows", ["0"])
    state = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def zero() -> qm.Q[qm.Bits[1]]:
    return qm.qubits(1)
""",
        module_name="ibm_runtime_expectation",
    ).specialize("zero")
    estimate = qm.expectation(state, qm.Z(0), on=endpoint, shots=4, seed=0)
    assert estimate.value == 1.0
    assert estimate.shots == 4


def test_qiskit_ibm_runtime_rejects_nonportable_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime never pretends provider sampling implements seeded execution."""

    from qiskit.providers.basic_provider import BasicSimulator

    import qamomile.qiskit.runtime as runtime_module
    from qamomile.qiskit import IBMRuntime

    monkeypatch.setattr(
        runtime_module,
        "_runtime_module",
        lambda: SimpleNamespace(SamplerV2=_FakeRuntimeSampler),
    )
    endpoint = IBMRuntime(BasicSimulator())
    artifact = qm.compile(
        qm.build_source(BELL_SOURCE).specialize("bell"),
        to=endpoint.target,
    )

    with pytest.raises(qm.ExecutionError, match="portable seed contract"):
        qm.submit(artifact, on=endpoint, shots=2, seed=3)


def test_qiskit_executes_boolean_lifting() -> None:
    """Aer executes the shared native Boolean lift plans."""

    from qamomile.qiskit import Aer

    exercise_boolean_lift_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_symbolic_observable_recipe() -> None:
    """Aer executes an Observable assembled from Const coefficient data."""

    from qamomile.qiskit import Aer

    exercise_symbolic_observable_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_python_collection_loops() -> None:
    """Aer consumes enumerate/zip after shared Rust loop normalization."""

    from qamomile.qiskit import Aer

    exercise_collection_loops_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_derived_measurement_results() -> None:
    """Qiskit evaluates the retained result tree after each raw sample."""

    exercise_derived_result_backend(endpoint=_basic_simulator_endpoint(), seed=0)
    exercise_broadcast_gate_backend(endpoint=_basic_simulator_endpoint(), seed=0)


def test_qiskit_measures_unsigned_integer_registers() -> None:
    """Aer preserves integer launch and result UX across Qiskit bit order."""

    from qamomile.qiskit import Aer

    exercise_uint_measurement_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_launch_bound_structural_math() -> None:
    """Qiskit receives only the loop produced after Rust evaluates ceil/log2."""

    from qamomile.qiskit import Aer

    exercise_structural_math_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_multi_owner_host_dispatch() -> None:
    """Qiskit variants retain both independently updated owned outputs."""

    from qamomile.qiskit import Aer

    exercise_multi_owner_dispatch_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_rust_lowered_host_dispatch_family() -> None:
    """One portable QPY family evaluates an integer launch predicate."""

    from qamomile.qiskit import Aer

    endpoint = Aer()
    program = qm.build_source(
        HOST_DISPATCH_SOURCE,
        module_name="qiskit_host_dispatch_family",
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


def test_qiskit_materializes_a_launch_dependent_loop() -> None:
    """Aer receives one Rust-expanded circuit for the requested loop count."""

    from qamomile.qiskit import Aer

    endpoint = Aer()
    program = qm.build_source(
        LAUNCH_LOOP_SOURCE,
        module_name="qiskit_launch_loop",
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


def test_qiskit_executes_documented_quantum_kernel_classification() -> None:
    """Aer executes the documented overlap kernel without a custom path."""

    from qamomile.qiskit import Aer

    exercise_classification_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_documented_hydrogen_vqe() -> None:
    """Aer estimates the documented hydrogen trial energy."""

    from qamomile.qiskit import Aer

    exercise_hydrogen_vqe_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_documented_qsci_sampling() -> None:
    """Aer samples the documented QSCI trial-state configurations."""

    from qamomile.qiskit import Aer

    exercise_qsci_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_documented_hamiltonian_simulation() -> None:
    """Aer executes the documented noncommuting Rabi evolution."""

    from qamomile.qiskit import Aer

    exercise_rabi_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_documented_mottonen_encoding() -> None:
    """Aer prepares the documented complex Möttönen state."""

    from qamomile.qiskit import Aer

    exercise_mottonen_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_documented_multidimensional_qft() -> None:
    """Aer executes the documented axiswise Fourier transform."""

    from qamomile.qiskit import Aer

    exercise_multidimensional_qft_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_documented_hybrid_qnn() -> None:
    """Aer executes the documented runtime Hybrid QNN layer."""

    from qamomile.qiskit import Aer

    exercise_hqnn_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_documented_option_pricing() -> None:
    """Aer executes the documented amplitude-estimation option pricer."""

    from qamomile.qiskit import Aer

    exercise_option_pricing_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_documented_pce() -> None:
    """Aer estimates and decodes the documented PCE state."""

    from qamomile.qiskit import Aer

    exercise_pce_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_documented_qemcmc() -> None:
    """Aer executes the documented quantum-enhanced MCMC proposal."""

    from qamomile.qiskit import Aer

    exercise_qemcmc_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_documented_bit_flip_code() -> None:
    """Aer coherently corrects each single data-qubit bit flip."""

    from qamomile.qiskit import Aer

    exercise_bit_flip_code_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_documented_steane_code() -> None:
    """Aer coherently corrects a bounded Steane error case."""

    from qamomile.qiskit import Aer

    exercise_steane_code_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_variational_composition() -> None:
    """Aer executes QAOA and FQAOA without algorithm-specific lowering."""

    from qamomile.qiskit import Aer

    exercise_variational_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_grover_adaptive_search() -> None:
    """Aer executes GAS composed from ordinary reusable qkernels."""

    from qamomile.qiskit import Aer

    exercise_gas_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_product_formula_composition() -> None:
    """Aer executes the shared abstract-evolution schedule."""

    from qamomile.qiskit import Aer

    exercise_product_formula_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_uniformly_controlled_rotation() -> None:
    """Aer executes the public function-composed Gray walk."""

    from qamomile.qiskit import Aer

    exercise_uniformly_controlled_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_coherent_xor_lookup() -> None:
    """Aer executes lookup through total callable selection."""

    from qamomile.qiskit import Aer

    exercise_xor_lookup_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_sparse_matrix_lookup() -> None:
    """Aer executes functional sparse row/slot access."""

    from qamomile.qiskit import Aer

    exercise_sparse_lookup_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_runtime_matrix_access() -> None:
    """Aer binds a shaped Matrix while emitted rotations use scalar leaves."""

    from qamomile.qiskit import Aer

    exercise_runtime_matrix_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_runtime_qsvt_phases() -> None:
    """Aer reuses one QSVT circuit with a phase Vector launch argument."""

    from qamomile.qiskit import Aer

    exercise_runtime_qsvt_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_functional_amplitude_encoding() -> None:
    """Aer samples and estimates the multilevel Gray-walk body."""

    from qamomile.qiskit import Aer

    endpoint = Aer()
    assert qm.run(amplitude_two_sample, on=endpoint, shots=4, seed=2).result.counts == {
        (1, 1): 4
    }
    assert (
        qm.expectation(amplitude_two_state, qm.Z(0), on=endpoint, shots=4, seed=2).value
        == -1.0
    )


def test_qiskit_reuses_runtime_parametric_amplitude_artifact() -> None:
    """Aer receives aggregate coordinates through the common scalar ABI view."""

    from qamomile.qiskit import Aer, Qiskit

    exercise_runtime_amplitude_backend(endpoint=Aer(), target=Qiskit(), seed=2)


def test_qiskit_executes_functional_pauli_block_encoding() -> None:
    """Aer executes Pauli production through the existing LCU body."""

    from qamomile.qiskit import Aer

    exercise_pauli_block_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_functional_block_producers() -> None:
    """Aer executes projected and periodic construction through common IR."""

    from qamomile.qiskit import Aer

    exercise_functional_block_producers_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_standard_python_record_results() -> None:
    """Qiskit reconstructs field names from the shared product result ABI."""

    program = qm.build_source(RECORD_RESULT_SOURCE).specialize("record_result")
    assert qm.run(
        program,
        on=_basic_simulator_endpoint(),
        shots=4,
        seed=0,
    ).result.values == {(("first", (1,)), ("pair", (0, 1))): 4}


def test_qiskit_executes_callable_transforms_and_exact_relative_phase() -> None:
    """Aer executes shared control/multiplex semantics including scalar phase."""

    from qamomile.qiskit import Aer

    selected = qm.build_source(CALLABLE_TRANSFORMS_SOURCE).specialize(
        "callable_transforms"
    )
    phased = qm.build_source(CONTROLLED_PHASE_SOURCE).specialize("controlled_phase")

    assert qm.run(selected, on=Aer(), shots=4, seed=2).result.values == {(1, (1,)): 4}
    assert qm.run(phased, on=Aer(), shots=4, seed=2).result.values == {(1, (0,)): 4}


def test_qiskit_retains_opaque_multi_control_gate_until_endpoint_preparation() -> None:
    """A three-control H remains an SDK gate with the exact statevector action."""

    from qiskit.quantum_info import Statevector

    from qamomile.qiskit import Qiskit, to_circuit

    source = """
import qamomile as qm

@qm.qkernel
def hadamard(target: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.h(target)

@qm.qkernel
def controlled_h() -> qm.Q[qm.Bits[4]]:
    controls = qm.prepare((1, 1, 1))
    target = qm.qubits(1)
    controls, target = qm.control(hadamard, controls, target)
    return qm.concat(controls, target)
"""
    program = qm.build_source(source).specialize("controlled_h")
    circuit = to_circuit(program, target=Qiskit())

    from qiskit import transpile
    from qiskit.circuit import AnnotatedOperation

    assert sum(isinstance(row.operation, AnnotatedOperation) for row in circuit) == 1
    prepared = transpile(circuit, basis_gates=["u", "cx"], optimization_level=0)
    probabilities = abs(Statevector.from_instruction(prepared).data) ** 2
    assert probabilities[7] == pytest.approx(0.5)
    assert probabilities[15] == pytest.approx(0.5)


def test_qiskit_executes_compiler_derived_selected_adjoint() -> None:
    """Aer executes multiplex(U_i-adjoint) from the shared dagger pass."""

    from qamomile.qiskit import Aer

    program = qm.build_source(COHERENT_ADJOINT_SOURCE).specialize(
        "selected_adjoint_roundtrip"
    )
    assert qm.run(program, on=Aer(), shots=4, seed=2).result.counts == {(1, 0): 4}


def test_qiskit_executes_indexed_aggregate_control() -> None:
    """Aer samples and estimates selector-based coherent control."""

    from qamomile.qiskit import Aer

    module = qm.build_source(INDEXED_CONTROL_SOURCE)
    assert qm.run(
        module["indexed_control_sample"], on=Aer(), shots=4, seed=2
    ).result.counts == {(1, 0, 1): 4}
    assert (
        qm.expectation(
            module["indexed_control_state"],
            qm.Z(2),
            on=Aer(),
            shots=4,
            seed=2,
        ).value
        == -1.0
    )


def test_qiskit_executes_abstract_pauli_evolution() -> None:
    """The shared circuit projection lowers a Pauli gadget once for Qiskit."""

    from qamomile.qiskit import Aer, Qiskit, load_circuit

    program = qm.build_source(PAULI_EVOLUTION_SOURCE).specialize(
        "pauli_evolution",
        **{"hamiltonian": 0.5 * qm.Z(0) * qm.Z(1)},
    )
    artifact = qm.compile(program, to=Qiskit())
    native = [
        instruction.operation
        for instruction in load_circuit(artifact).data
        if instruction.operation.name == "PauliEvolution"
    ]

    assert len(native) == 1
    assert type(native[0]).__name__ == "PauliEvolutionGate"
    assert qm.run(artifact, on=Aer(), shots=4, seed=2).result.counts == {(1, 0): 4}
    exercise_normalized_model_backend(endpoint=Aer(), seed=2)


def test_qiskit_can_request_portable_pauli_evolution() -> None:
    """External Qiskit consumers can request the verified exact gate schedule."""

    from qamomile.qiskit import Qiskit, to_circuit

    program = qm.build_source(PAULI_EVOLUTION_SOURCE).specialize(
        "pauli_evolution",
        **{"hamiltonian": 0.5 * qm.Z(0) * qm.Z(1)},
    )
    native_target = Qiskit()
    portable_target = Qiskit(use_native_pauli_evolution=False)
    circuit = to_circuit(program, target=portable_target)

    assert native_target.fingerprint != portable_target.fingerprint
    assert "PauliEvolution" not in circuit.count_ops()
    assert circuit.count_ops()["rzz"] == 1


@pytest.mark.parametrize("choice", ["pauli.native", "pauli.rzz", "pauli.parity"])
def test_qiskit_executes_realization_with_composed_runtime_time(choice: str) -> None:
    """A runtime time expression survives QPY and SDK parameter binding."""
    from qamomile.lowering import RealizationOptions
    from qamomile.qiskit import Aer

    program = qm.build_source("""
import qamomile as qm
@qm.qkernel
def sample(theta: qm.Param[qm.Real], hamiltonian: qm.Const[qm.Observable]) -> qm.Bits[2]:
    state = qm.pauli_evolve(qm.qubits(2), hamiltonian, 2.0 * theta + 0.25)
    return qm.measure(state)
""").specialize("sample", hamiltonian=qm.X(0))
    endpoint = Aer(realizations=RealizationOptions({"quantum.pauli_evolution": choice}))
    artifact = qm.compile(program, to=endpoint.target)
    assert f"realization:{choice}" in artifact.report.stages
    restored = deserialize(serialize(artifact))
    assert qm.execute(
        restored,
        on=endpoint,
        parameters={"theta": (math.pi / 2 - 0.25) / 2},
        shots=4,
        seed=0,
    ).counts == {(1, 0): 4}


@pytest.mark.parametrize("expression", ["time / 1", "(time / 1) * 1"])
def test_qiskit_qpy_preserves_parameter_after_identity_arithmetic(
    expression: str,
) -> None:
    """QPY restores identity arithmetic without corrupting a parameter."""

    from qamomile.qiskit import Aer, load_circuit

    program = qm.build_source(f"""
import qamomile as qm

@qm.qkernel
def sample(time: qm.Param[qm.Real], hamiltonian: qm.Const[qm.Observable]) -> qm.Bits[1]:
    state = qm.pauli_evolve(qm.qubits(1), hamiltonian, {expression})
    return qm.measure(state)
""").specialize("sample", hamiltonian=qm.X(0))
    endpoint = Aer()
    artifact = qm.compile(program, to=endpoint.target)
    restored = deserialize(serialize(artifact))
    circuit = load_circuit(restored)
    evolution = next(
        instruction.operation
        for instruction in circuit.data
        if instruction.operation.name == "PauliEvolution"
    )

    assert type(evolution.time).__name__ == "Parameter"
    assert qm.execute(
        restored,
        on=endpoint,
        parameters={"time": math.pi / 2},
        shots=4,
        seed=2,
    ).counts == {(1,): 4}


def test_qiskit_estimates_a_backend_neutral_observable() -> None:
    """Expectation basis programs execute through the ordinary Aer endpoint."""

    from qamomile.qiskit import Aer

    state = qm.build_source(STATE_PREPARATION_SOURCE)["plus_state"]
    assert qm.expectation(state, qm.X(0), on=Aer(), shots=4, seed=2).value == 1.0


def test_qiskit_executes_imported_qft_roundtrip() -> None:
    """The linked functional QFT library executes through real Aer."""

    from qamomile.qiskit import Aer

    assert qm.run(qft_roundtrip, on=Aer(), shots=4, seed=2).result.counts == {
        (1, 0, 1): 4
    }
    assert qm.run(qft_frequency_three, on=Aer(), shots=4, seed=2).result.counts == {
        (1, 1, 0): 4
    }


def test_qiskit_preserves_named_qft_body_without_name_based_reinterpretation() -> None:
    """A callable label preserves its verified body and does not imply semantics."""

    from qiskit.quantum_info import Operator

    from qamomile.qiskit import Qiskit, to_circuit

    native = to_circuit(_native_qft_state, target=Qiskit())
    decomposed = to_circuit(_decomposed_qft_state, target=Qiskit())

    assert len(native.data) == 1
    assert native.data[0].operation.label == "qft"
    assert native.data[0].operation._definition is None
    assert len(decomposed.data) == 1
    body = decomposed.data[0].operation.definition
    assert body.count_ops()["h"] == 3
    assert body.count_ops()["cp"] == 3
    assert body.count_ops()["swap"] == 1
    assert Operator(native).equiv(Operator(decomposed))


def test_qiskit_does_not_infer_semantics_from_composite_labels() -> None:
    """A named body remains authoritative through QPY and Aer execution."""
    from qamomile.qiskit import Aer

    assert qm.run(_misleading_qft_sample, on=Aer(), shots=4, seed=0).result.counts == {
        (1, 0): 4
    }


def test_qiskit_preserves_named_composite_with_fallback_body() -> None:
    """A named qkernel becomes one labeled QPY gate with a decomposition."""

    from qamomile.qiskit import Aer, Qiskit, load_circuit

    artifact = qm.compile(_boxed_bell_sample, to=Qiskit())
    circuit = load_circuit(artifact)
    [boxed] = [
        instruction
        for instruction in circuit.data
        if instruction.operation.name != "measure"
    ]

    assert boxed.operation.label == "entangle"
    assert boxed.operation.definition.count_ops()["h"] == 1
    assert boxed.operation.definition.count_ops()["cx"] == 1
    assert set(qm.run(artifact, on=Aer(), shots=16, seed=3).result.counts) <= {
        (0, 0),
        (1, 1),
    }


def test_qiskit_preserves_index_independent_for_loop() -> None:
    """A structural loop that ignores its index remains one Qiskit for-loop."""

    from qamomile.qiskit import Aer, Qiskit, load_circuit

    source = """
import qamomile as qm

@qm.qkernel
def repeated_h(reps: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    for _ in range(reps):
        state = qm.h(state, target=0)
    return qm.measure(state)
"""
    program = qm.build_source(source).specialize(
        "repeated_h",
        **{"reps": 3},
    )
    artifact = qm.compile(program, to=Qiskit())

    assert load_circuit(artifact).count_ops()["for_loop"] == 1
    result = qm.run(artifact, on=Aer(), shots=16, seed=3).result
    assert result.shots == 16
    assert set(result.counts) <= {(0,), (1,)}


def test_qiskit_executes_functional_phase_estimation() -> None:
    """Aer samples and estimates body-derived two-bit QPE."""

    from qamomile.qiskit import Aer

    assert qm.run(qpe_sample, on=Aer(), shots=4, seed=2).result.counts == {(1, 0, 1): 4}
    assert qm.expectation(qpe_state, qm.Z(0), on=Aer(), shots=4, seed=2).value == -1.0
    exercise_nary_tensor_backend(endpoint=Aer(), seed=2)


def test_qiskit_decodes_typed_fixed_point_after_native_bit_order() -> None:
    """Aer normalizes Qiskit bitstrings before numeric phase decoding."""

    from qamomile.qiskit import Aer

    exercise_fixed_qpe_backend(endpoint=Aer(), seed=2)
    exercise_windowed_qpe_backend(endpoint=Aer(), seed=2)
    exercise_documented_windowed_qpe_backend(endpoint=Aer(), seed=2)
    exercise_documented_reversible_arithmetic_backend(endpoint=Aer(), seed=2)
    exercise_documented_gf2_arithmetic_backend(endpoint=Aer(), seed=2)
    exercise_documented_phase_gradient_backend(endpoint=Aer(), seed=2)
    exercise_documented_fault_tolerant_data_loading_backend(endpoint=Aer(), seed=2)
    exercise_documented_reversible_indexing_backend(endpoint=Aer(), seed=2)
    exercise_documented_reversible_trigonometry_backend(endpoint=Aer(), seed=2)
    exercise_documented_kikuchi_backend(endpoint=Aer(), seed=2)
    exercise_documented_guided_hamiltonian_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_documented_qualtran_gap_examples() -> None:
    """Aer executes the bounded ECC and factorized-chemistry witnesses."""

    from qamomile.qiskit import Aer

    exercise_documented_elliptic_curve_dlog_backend(endpoint=Aer(), seed=2)
    exercise_documented_factorized_chemistry_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_functional_grover_search() -> None:
    """Aer samples and estimates body-composed amplitude amplification."""

    from qamomile.qiskit import Aer

    assert qm.run(grover_sample, on=Aer(), shots=4, seed=2).result.counts == {(1, 1): 4}
    assert (
        qm.expectation(grover_state, qm.Z(0), on=Aer(), shots=4, seed=2).value == -1.0
    )


def test_qiskit_executes_functional_modular_arithmetic() -> None:
    """Aer executes bounded sample/expectation arithmetic witnesses."""

    from qamomile.qiskit import Aer

    exercise_modular_arithmetic_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_shor_order_finding() -> None:
    """Aer executes the shared Shor sampling and state witnesses."""

    from qamomile.qiskit import Aer

    exercise_shor_order_finding_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_regev_factoring_sample() -> None:
    """Aer executes the shared bounded multidimensional factoring sample."""

    from qamomile.qiskit import Aer

    exercise_regev_factoring_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_multi_exponentiation_callable_realization() -> None:
    """Aer executes the shared source after complete callable replacement."""

    from qamomile.qiskit import Aer

    options = qm.RealizationOptions(
        {multi_exponentiation_baseline: multi_exponentiation_product_tree}
    )
    exercise_multi_exponentiation_realization_backend(
        endpoint=Aer(realizations=options), seed=2
    )


def test_qiskit_executes_first_class_multi_exponentiation() -> None:
    """Aer executes the semantic primitive through its independent realization."""

    from qamomile.qiskit import Aer

    exercise_first_class_multi_exponentiation_backend(endpoint=Aer(), seed=2)


def test_qiskit_executes_register_modulus_arithmetic() -> None:
    """Aer samples retained-modulus addition."""

    from qamomile.qiskit import Aer

    exercise_register_modulus_backend(endpoint=Aer(), seed=0)


def test_qiskit_executes_functional_qsvt() -> None:
    """Aer executes direct and PREPARE/SELECT-composed QSVT bodies."""

    from qamomile.qiskit import Aer

    endpoint = Aer()
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


def test_qiskit_reciprocal_svt_realizes_the_certified_real_block() -> None:
    """The reciprocal signal-zero block applies the scaled matrix inverse."""

    import numpy as np
    from qiskit.quantum_info import Statevector

    circuit = qm.qiskit.to_circuit(linear_system_reciprocal_state)
    state = np.asarray(Statevector.from_instruction(circuit).data)
    signal_zero_branch = state[[0, 4]]
    phase = signal_zero_branch[0] / abs(signal_zero_branch[0])
    expected = (
        linear_system_reciprocal.operator_scale * np.array([1.0, 2.0]) / math.sqrt(2.0)
    )

    np.testing.assert_allclose(
        signal_zero_branch / phase,
        expected,
        atol=linear_system_reciprocal.total_error_bound,
        rtol=0.0,
    )


def test_qiskit_executes_coherent_product_reassociation() -> None:
    """Aer receives only the gates surrounding a zero-cost coherent reshape."""

    from qamomile.qiskit import Aer

    program = qm.build_source(COHERENT_RESHAPE_SOURCE).specialize("branched_adjoint")
    assert qm.run(program, on=Aer(), shots=4, seed=2).result.counts == {(1, 0, 1): 4}


def test_qiskit_materializes_dynamic_qubit_selection_at_launch() -> None:
    """Rust resolves a launch-time wire selector before QPY emission."""

    from qamomile.qiskit import Aer

    exercise_dynamic_qubit_selection_backend(endpoint=Aer(), seed=2)


def test_qiskit_materializes_dynamic_classical_index_at_launch() -> None:
    """Rust resolves an aggregate path before QPY parameter emission."""

    exercise_dynamic_classical_index_backend(
        endpoint=_basic_simulator_endpoint(), seed=0
    )


pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")


def test_qpy_artifact_is_portable_and_preserves_source_bit_order() -> None:
    """Qiskit compilation stores QPY bytes and Aer executes exact basis data."""

    from qamomile.qiskit import Aer, Qiskit, load_circuit

    program = qm.build_source(BASIS_SOURCE).specialize("basis")
    artifact = qm.compile(program, to=Qiskit())
    restored = deserialize(serialize(artifact))
    with patch(
        "subprocess.run",
        side_effect=AssertionError("worker used"),
    ):
        result = qm.execute(restored, on=Aer(), shots=16, seed=4)

    assert artifact.format == "qiskit.qpy.v1"
    assert load_circuit(restored).num_qubits == 3
    assert result.counts == {(1, 0, 0): 16}
    assert result.expectation_z(0, 2) == -1.0
    assert result.worker_pid is None


def test_qiskit_executes_extended_discrete_gate_family() -> None:
    """QPY preserves phase, ZZ, and Toffoli primitives through Aer execution."""

    from qamomile.qiskit import Aer

    program = qm.build_source(EXTENDED_GATE_SOURCE).specialize("extended_gates")
    assert qm.run(program, on=Aer(), shots=4).result.counts == {(1, 1, 1): 4}


def test_qiskit_executes_measure_reset() -> None:
    """Qiskit preserves measurement followed by reset as one source operation."""

    from qamomile.qiskit import Aer

    program = qm.build_source(MEASURE_RESET_SOURCE).specialize("measure_and_reset")
    record = qm.run(program, on=Aer(), shots=4)

    assert record.result.values == {((1,), (0,)): 4}
    assert (
        Requirement.MID_CIRCUIT_MEASUREMENT
        in record.artifact.execution_contract.requirements
    )


def test_qiskit_executes_nondestructive_pauli_measurements() -> None:
    """Aer preserves repeatable X, Y, and Z projective measurements."""

    from qamomile.qiskit import Aer

    program = qm.build_source(MEASURE_AXES_SOURCE).specialize("measure_axes")
    result = qm.run(program, on=Aer(), shots=4, seed=2).result

    assert result.values == {(1, 1, 0, 0, 0, 0, (1,), (0,), (0,)): 4}


@pytest.mark.parametrize(("theta", "expected"), [(0.0, (0,)), (math.pi, (1,))])
def test_qiskit_parameterized_rotation_executes(
    theta: float,
    expected: tuple[int, ...],
) -> None:
    """Param values bind after QPY compilation without changing the artifact."""

    from qamomile.qiskit import Aer, Qiskit

    program = qm.build_source(ROTATION_SOURCE).specialize("rotate")
    artifact = qm.compile(program, to=Qiskit())
    assert serialize(artifact) == serialize(qm.compile(program, to=Qiskit()))
    result = qm.execute(
        artifact,
        on=Aer(),
        parameters={"theta": theta},
        shots=16,
        seed=7,
    )
    assert result.counts == {expected: 16}
    if theta == 0.0:
        defaulted = qm.execute(artifact, on=Aer(), shots=4, seed=7)
        assert defaulted.counts == {(0,): 4}


def test_qiskit_accepts_the_full_unsigned_seed_domain() -> None:
    """Aer receives all invocation seed bits through its signed boundary."""

    from qamomile.qiskit import Aer

    seed = (1 << 64) - 1
    first = qm.run(amplitude_two_sample, on=Aer(), shots=8, seed=seed)
    second = qm.run(amplitude_two_sample, on=Aer(), shots=8, seed=seed)

    assert first.invocation.seed == seed
    assert second.invocation.seed == seed
    assert first.result.counts == second.result.counts


def test_qiskit_executes_measurement_feedback() -> None:
    """Shot-time Python if lowers to a Qiskit dynamic-control region."""

    from qamomile.qiskit import Aer, Qiskit, load_circuit

    program = qm.build_source(FEEDBACK_SOURCE).specialize("feedback")
    artifact = qm.compile(program, to=Qiskit())
    circuit = load_circuit(artifact)
    result = qm.execute(artifact, on=Aer(), shots=32, seed=3)

    assert "if_else" in circuit.count_ops()
    assert set(result.counts) == {(0, 0), (1, 1)}


def test_qiskit_executes_boolean_measurement_expression() -> None:
    """Qiskit lowers ordinary ``not`` over a measured Python condition."""

    from qamomile.qiskit import Aer

    result = qm.run(
        qm.build_source(NEGATED_FEEDBACK_SOURCE).specialize("negated_feedback"),
        on=Aer(),
        shots=32,
        seed=3,
    ).result

    assert set(result.counts) == {(0, 1), (1, 0)}


def test_qiskit_executes_modular_automatic_uncomputation() -> None:
    """Qiskit receives the common late lowering of a callable adjoint."""

    from qamomile.circuit import lower_to_circuit
    from qamomile.qiskit import Aer, load_circuit

    program = qm.build_source(MODULAR_CLEANUP_SOURCE).specialize("modular_cleanup")
    assert "qcore.adjoint_call" in program.inspect()
    assert lower_to_circuit(program).semantic_id == program.semantic_digest
    record = qm.run(program, on=Aer(), shots=8, seed=11)
    from qiskit.circuit import AnnotatedOperation, InverseModifier

    circuit = load_circuit(record.artifact)
    inverses = [
        row.operation
        for row in circuit
        if isinstance(row.operation, AnnotatedOperation)
    ]
    assert len(inverses) == 1
    assert inverses[0].modifiers == [InverseModifier()]
    assert inverses[0].base_op.definition.count_ops() == {
        "h": 1,
        "cx": 1,
        "s": 1,
        "t": 1,
    }
    assert record.result.counts == {(1, 0, 0): 8}


def test_qiskit_executes_multi_input_automatic_uncomputation() -> None:
    """Aer executes a packed adjoint restoring two independent inputs."""

    from qamomile.qiskit import Aer

    program = qm.build_source(MULTI_INPUT_CLEANUP_SOURCE).specialize(
        "multi_input_cleanup"
    )
    result = qm.run(program, on=Aer(), shots=4, seed=7).result

    assert result.counts == {(1,): 4}


def test_qiskit_preserves_private_trailing_measurement_width() -> None:
    """Inspection and closed child transports retain private measurements."""

    program = qm.build_source(TRAILING_MEASUREMENT_SOURCE).specialize(
        "trailing_measurement"
    )
    record = qm.run(program, on=_basic_simulator_endpoint(), shots=4, seed=0)

    assert record.artifact.result_abi.source_order == (0,)
    assert record.artifact.format == "qamomile.shot-execution.v1"
    assert record.artifact.result_abi.native_width == 1
    assert qm.qiskit.to_circuit(program).num_clbits == 2
    assert record.result.counts == {(0,): 4}


def test_qiskit_backend_routes_and_executes_through_the_common_job_contract() -> None:
    """A live BackendV2 stays behind one thin target/endpoint adapter."""

    from qiskit_aer import AerSimulator

    from qamomile.qiskit import QiskitBackend

    endpoint = QiskitBackend(AerSimulator())
    program = qm.build_source(BASIS_SOURCE).specialize("basis")
    record = qm.run(program, on=endpoint, shots=4)

    assert record.result.counts == {(1, 0, 0): 4}
    assert "qiskit-routing" in record.artifact.report.stages
    assert record.artifact.target_fingerprint == endpoint.target.fingerprint


def test_qiskit_executes_typed_product_result() -> None:
    """Aer transports flat bits and exposes the declared product shape."""

    from qamomile.qiskit import Aer

    record = qm.run(
        qm.build_source(STRUCTURED_RESULT_SOURCE).specialize("structured_result"),
        on=Aer(),
        shots=4,
        seed=2,
    )

    assert record.result.counts == {(1, 0, 1): 4}
    assert record.result.values == {((1,), (0, 1)): 4}


@pytest.mark.parametrize("choice", ["add.qft", "add.ripple", "add.native"])
def test_qiskit_executes_late_realizations(choice: str) -> None:
    """One shared arithmetic source survives exact alternative realizations."""
    pytest.importorskip("qiskit_aer")
    from qamomile.lowering import RealizationOptions
    from qamomile.qiskit import Aer

    from .test_realizations import exercise_realization_backend

    exercise_realization_backend(
        Aer(realizations=RealizationOptions({"quantum.add_into": choice}))
    )


def test_qiskit_zero_layer_qaoa() -> None:
    """Execute the shared zero layer qaoa contract locally."""
    from qamomile.qiskit import Aer

    from .test_library_compatibility import exercise_zero_layer_qaoa_backend

    exercise_zero_layer_qaoa_backend(Aer())


def test_qiskit_approximate_qft() -> None:
    """Execute the shared approximate qft contract locally."""
    from qamomile.qiskit import Aer

    from .test_library_compatibility import exercise_approximate_qft_backend

    exercise_approximate_qft_backend(Aer())


def test_qiskit_classical_array_compatibility() -> None:
    """Execute the shared classical array contract locally."""
    from qamomile.qiskit import Aer

    from .test_classical_array_compatibility import exercise_classical_array_backend

    exercise_classical_array_backend(Aer())


def test_qiskit_qubit_selection() -> None:
    """Execute owned noncontiguous selection on this local SDK."""
    from qamomile.qiskit import Aer

    from .test_qubit_selection import exercise_qubit_selection_backend

    exercise_qubit_selection_backend(Aer(), seed=7)


def test_qiskit_keyed_parameters() -> None:
    """Rebind fixed-key dictionaries through the ordinary Param Vector ABI."""
    from qamomile.qiskit import Aer

    from .test_parameter_map import exercise_parameter_map_backend

    exercise_parameter_map_backend(Aer(), seed=7)


def test_qiskit_quantum_records() -> None:
    """Execute whole-record ownership transfers and coherent transforms."""
    from qamomile.qiskit import Aer

    from .test_quantum_records import exercise_quantum_record_backend

    exercise_quantum_record_backend(Aer(), seed=7)


def test_qiskit_recursion() -> None:
    """Execute self, mutual, branching, and higher-order recursive helpers."""
    from qamomile.qiskit import Aer

    from .test_recursion import exercise_recursive_backend

    exercise_recursive_backend(Aer(), seed=7)


def test_qiskit_sqrt() -> None:
    """Execute Const and launch-bound square roots through the native algebra."""
    from qamomile.qiskit import Aer

    from .test_sqrt import exercise_sqrt_backend

    exercise_sqrt_backend(Aer(), seed=7)


def test_qiskit_target_profile_is_derived_from_the_backend_and_fingerprinted() -> None:
    """A live backend yields a validated profile that identifies the target."""
    from qiskit.providers.fake_provider import GenericBackendV2

    from qamomile.compilation import CouplingEdge, TargetProfile
    from qamomile.qiskit import Qiskit, QiskitBackend, profile_from_backend

    line = GenericBackendV2(num_qubits=3, coupling_map=[[0, 1], [1, 2]], seed=1)
    ring = GenericBackendV2(
        num_qubits=3, coupling_map=[[0, 1], [1, 0], [1, 2], [2, 0]], seed=1
    )
    profile = profile_from_backend(line)

    assert isinstance(profile, TargetProfile)
    assert profile.format == "qiskit.qpy.v1"
    assert profile.topology.num_qubits == 3
    assert profile.topology.edges == (CouplingEdge(0, 1), CouplingEdge(1, 2))
    assert profile.topology.two_qubit_operations == ("cx",)
    assert {"rz", "sx", "x"} <= set(profile.topology.one_qubit_operations)
    assert profile.backend_name == "generic_backend_3q"
    names = {row.name for row in profile.calibration.qubit_properties}
    assert {"t1", "t2", "frequency", "x.error", "measure.error"} <= names
    assert {
        (row.control, row.target) for row in profile.calibration.edge_properties
    } == {
        (0, 1),
        (1, 2),
    }
    assert {row.name for row in profile.calibration.edge_properties} == {
        "cx.error",
        "cx.duration",
    }
    assert TargetProfile.from_bytes(profile.to_bytes()) == profile
    assert profile_from_backend(ring).topology.edges == (
        CouplingEdge(0, 1, True),
        CouplingEdge(1, 2),
        CouplingEdge(2, 0),
    )

    target = Qiskit(backend=line)
    assert target.capabilities.profile == profile
    assert target.fingerprint != Qiskit(backend=ring).fingerprint
    assert (
        target.fingerprint
        != Qiskit(backend=line, backend_snapshot="pinned").fingerprint
    )
    assert Qiskit().capabilities.profile is None
    assert Qiskit(backend=line, profile=profile).fingerprint == target.fingerprint
    assert Qiskit(profile=profile).fingerprint != Qiskit().fingerprint
    endpoint = QiskitBackend(line)
    assert endpoint.target.capabilities.profile == profile
    assert endpoint.capabilities.profile == profile
    assert QiskitBackend(line, profile=profile).target.profile == profile
    with pytest.raises(ValueError, match="format"):
        Qiskit(profile=replace(profile, format="other"))

    from qiskit.providers.basic_provider import BasicSimulator

    assert profile_from_backend(BasicSimulator()) is None
