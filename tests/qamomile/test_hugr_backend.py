"""End-to-end direct HUGR and Selene tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import qamomile as qm
from qamomile.effects import Requirement

from .conftest import (
    BELL_SOURCE,
    BOOLEAN_LIFT_SOURCE,
    CALLABLE_TRANSFORMS_SOURCE,
    COHERENT_ADJOINT_SOURCE,
    COHERENT_RESHAPE_SOURCE,
    EXTENDED_DISCRETE_GATE_SOURCE,
    FEEDBACK_SOURCE,
    HOST_DISPATCH_SOURCE,
    INDEXED_CONTROL_SOURCE,
    LAUNCH_LOOP_SOURCE,
    MEASURE_AXES_SOURCE,
    MEASURE_RESET_SOURCE,
    MODULAR_CLEANUP_SOURCE,
    MULTI_INPUT_CLEANUP_SOURCE,
    PAULI_EVOLUTION_SOURCE,
    RECORD_RESULT_SOURCE,
    RETAINED_HUGR_SOURCE,
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
    exercise_static_bit_flip_code_backend,
    exercise_static_classification_backend,
    exercise_static_hqnn_backend,
    exercise_static_hydrogen_vqe_backend,
    exercise_static_mottonen_backend,
    exercise_static_multidimensional_qft_backend,
    exercise_static_option_pricing_backend,
    exercise_static_pce_backend,
    exercise_static_qemcmc_backend,
    exercise_static_qsci_backend,
    exercise_static_rabi_backend,
    exercise_steane_code_backend,
)
from .test_gas import exercise_gas_backend
from .test_stdlib import (
    add_modulo_sample,
    add_modulo_state,
    amplitude_two_sample,
    amplitude_two_state,
    decrement_sample,
    exercise_fixed_qpe_hugr_backend,
    exercise_functional_block_producers_backend,
    exercise_nary_tensor_backend,
    exercise_pauli_block_backend,
    exercise_register_modulus_backend,
    exercise_runtime_amplitude_backend,
    exercise_sparse_lookup_backend,
    exercise_static_matrix_backend,
    exercise_static_product_formula_backend,
    exercise_static_variational_backend,
    exercise_uniformly_controlled_backend,
    exercise_windowed_qpe_backend,
    exercise_xor_lookup_backend,
    grover_sample,
    grover_state,
    lcu_qsvt_sample,
    lcu_qsvt_state,
    qft_frequency_three,
    qpe_one_bit_sample,
    qpe_one_bit_state,
    qsvt_phase_sample,
    qsvt_phase_state,
    reciprocal_x_sample,
    reciprocal_x_state,
    shor_three_sample,
    static_vector_qsvt_sample,
)


class _FakeQsysShot:
    """Represent one QSystem shot containing ordered Qamomile bits."""

    def __init__(self, bits: tuple[int, ...]) -> None:
        """Store a logical source-ordered bit row."""

        self.bits = bits

    def collate_tags(self) -> dict[str, list[int]]:
        """Return the records emitted by the Helios HUGR target."""

        return {"qamomile": list(self.bits)}


class _FakeNexusJobs:
    """Implement the qnexus job surface without network access."""

    def __init__(self) -> None:
        """Initialize the provider job registry."""

        self.registry: dict[str, object] = {}
        self.rows: tuple[tuple[int, ...], ...] = (
            (0, 0),
            (1, 1),
            (0, 0),
            (1, 1),
        )

    def status(self, job: object) -> object:
        """Return the fake native status record."""

        return SimpleNamespace(status=SimpleNamespace(value=job.status_name))

    def wait_for(self, job: object, *, timeout: float | None) -> object:
        """Complete the fake job within the supplied local timeout."""

        assert timeout is None or timeout >= 0
        job.status_name = "COMPLETED"
        return self.status(job)

    def results(self, job: object, *, allow_incomplete: bool) -> list[object]:
        """Return exactly one QSystem result reference."""

        assert not allow_incomplete
        raw = SimpleNamespace(
            results=[
                _FakeQsysShot(self.rows[index % len(self.rows)])
                for index in range(job.shots)
            ]
        )
        return [SimpleNamespace(download_result=lambda: raw)]

    def cancel(self, job: object) -> None:
        """Cancel one fake provider job."""

        job.status_name = "CANCELLED"

    def get(self, *, id: str) -> object:
        """Restore an existing fake execute job by ID."""

        return self.registry[id]


class _FakeNexus:
    """Capture HUGR uploads and Helios submissions for contract tests."""

    class HeliosConfig:
        """Minimal native-compatible Helios configuration."""

        type = "HeliosConfig"

        def __init__(self, *, system_name: str) -> None:
            """Store the selected fake system name."""

            self.system_name = system_name

    def __init__(self) -> None:
        """Initialize upload and job registries."""

        self.jobs = _FakeNexusJobs()
        self.uploads: list[object] = []
        self.submissions = 0
        self.hugr = SimpleNamespace(upload=self._upload)

    def _upload(self, **kwargs: object) -> object:
        """Capture one package upload and return a provider reference."""

        self.uploads.append(kwargs["hugr_package"])
        return SimpleNamespace(id=f"upload-{len(self.uploads)}")

    def start_execute_job(self, **kwargs: object) -> object:
        """Capture one asynchronous Helios submission."""

        self.submissions += 1
        identity = f"helios-job-{self.submissions}"
        job = SimpleNamespace(
            id=identity,
            job_type=SimpleNamespace(value="execute"),
            shots=kwargs["n_shots"][0],
            status_name="SUBMITTED",
        )
        self.jobs.registry[identity] = job
        return job


def test_helios_submit_snapshot_cancel_and_restore() -> None:
    """Helios records logical bits and restores jobs without resubmission."""

    from qamomile.hugr import Helios, HeliosOptions

    client = _FakeNexus()
    endpoint = Helios(
        HeliosOptions(system_name="Helios-1E-lite", timeout=1.0),
        client=client,
    )
    program = qm.build_source(BELL_SOURCE).specialize("bell")
    artifact = qm.compile(program, to=endpoint.target)

    job = qm.submit(artifact, on=endpoint, shots=4, seed=0)
    assert job.status is qm.JobStatus.SUBMITTED
    snapshot = qm.ExecutionSnapshot.from_bytes(job.snapshot().to_bytes())
    reference = snapshot.references()[0]
    assert reference.provider == "quantinuum-nexus"
    assert "credential" not in reference.context_values

    restored = endpoint.restore(snapshot, artifact=artifact)
    assert client.submissions == 1
    restored_result = restored.result()
    assert restored_result.counts == {(0, 0): 2, (1, 1): 2}
    assert restored_result.result_abi == artifact.result_abi
    cached = qm.ExecutionSnapshot.from_bytes(restored.snapshot().to_bytes())
    assert cached.kind is qm.ExecutionSnapshotKind.LOCAL
    detached = endpoint.restore(cached, artifact=artifact)
    assert detached.result() == restored_result

    pending = qm.submit(artifact, on=endpoint, shots=2, seed=0)
    pending.cancel()
    assert pending.status is qm.JobStatus.CANCELLED


def test_helios_measures_unsigned_integer_registers() -> None:
    """Nexus transport preserves source-ordered Qsys UInt records."""

    from qamomile.hugr import Helios, HeliosOptions

    client = _FakeNexus()
    client.jobs.rows = ((1, 1, 0),)
    endpoint = Helios(
        HeliosOptions(system_name="Helios-1E-lite", timeout=1.0),
        client=client,
    )
    exercise_uint_measurement_backend(endpoint=endpoint, seed=0)

    client.jobs.rows = ((0,),)
    state = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def zero() -> qm.Q[qm.Bits[1]]:
    return qm.qubits(1)
""",
        module_name="helios_expectation",
    ).specialize("zero")
    estimate = qm.expectation(state, qm.Z(0), on=endpoint, shots=4, seed=0)
    assert estimate.value == 1.0
    assert estimate.shots == 4


def test_hugr_executes_boolean_lifting() -> None:
    """Selene executes the shared Rust-normalized Boolean lift plans."""

    from qamomile.hugr import Selene

    exercise_boolean_lift_backend(endpoint=Selene(), seed=2)


def test_hugr_executes_symbolic_observable_recipe() -> None:
    """Selene executes an Observable assembled from Const coefficient data."""

    from qamomile.hugr import Selene

    exercise_symbolic_observable_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_requires_native_boolean_lift_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct emission rejects an absent or malformed Rust lift authority."""

    import qamomile.hugr._module as hugr_module
    from qamomile.hugr import Hugr

    program = qm.build_source(BOOLEAN_LIFT_SOURCE).specialize("phase_sample")
    monkeypatch.setattr(hugr_module.native, "plan_boolean_lifts", None)
    with pytest.raises(qm.CapabilityError, match="Q8685:.*unavailable"):
        qm.compile(program, to=Hugr())
    monkeypatch.setattr(
        hugr_module.native,
        "plan_boolean_lifts",
        lambda _payload: [("predicate", 1, [[[True]]], None, None)],
    )
    with pytest.raises(qm.CapabilityError, match="Q8685: malformed"):
        qm.compile(program, to=Hugr())
    for bad_transpositions in ([], [(0, 4)]):
        monkeypatch.setattr(
            hugr_module.native,
            "plan_boolean_lifts",
            lambda _payload, rows=bad_transpositions: [
                ("predicate", 2, [[[0]], [[1]]], None, rows)
            ],
        )
        with pytest.raises(qm.CapabilityError, match="Q8685: malformed"):
            qm.compile(program, to=Hugr())


def test_hugr_executes_a_derived_sampled_result() -> None:
    """Selene reports a computed Shot Bool through the common result ABI."""

    from qamomile.hugr import Selene

    endpoint = Selene(timeout=30.0)
    exercise_derived_result_backend(endpoint=endpoint, seed=2)
    exercise_broadcast_gate_backend(endpoint=endpoint, seed=2)


def test_hugr_measures_unsigned_integer_registers() -> None:
    """Selene preserves integer launch and Qsys-compatible result order."""

    from qamomile.hugr import Selene

    exercise_uint_measurement_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_executes_python_collection_loops() -> None:
    """Selene consumes enumerate/zip after shared Rust loop normalization."""

    from qamomile.hugr import Selene

    exercise_collection_loops_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_propagates_a_dynamic_all_return_if_through_a_call() -> None:
    """Conditional return values remain usable by the HUGR caller."""

    from qamomile.hugr import Selene

    source = """
import qamomile as qm

@qm.qkernel
def choose() -> qm.Shot[qm.Bool]:
    candidate = qm.measure(qm.prepare(True))
    if candidate:
        return True
    return False

@qm.qkernel
def sample() -> qm.Bits[1]:
    enabled = choose()
    state = qm.qubits(1)
    if enabled:
        state = qm.x(state)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="hugr_dynamic_return_call",
    ).specialize("sample")

    result = qm.run(
        program,
        on=Selene(timeout=30.0),
        shots=4,
        seed=3,
    ).result

    assert result.counts == {(1,): 4}


@pytest.mark.parametrize("initial", [False, True])
def test_hugr_executes_measurement_sites_inside_a_dynamic_branch(
    initial: bool,
) -> None:
    """A trailing result row is independent of selected branch event count."""

    from qamomile.hugr import Hugr, Selene

    source = f"""
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bool]:
    selector = qm.measure(qm.prepare({initial!r}))
    if selector:
        ignored = qm.measure(qm.prepare(False))
        result = qm.measure(qm.prepare(True))
    else:
        result = qm.measure(qm.prepare(False))
    return result
"""
    program = qm.build_source(
        source,
        module_name=f"hugr_dynamic_branch_measurement_{initial}",
    ).specialize("sample")
    artifact = qm.compile(program, to=Hugr())
    expected = qm.run(
        program,
        on=qm.Reference(),
        shots=4,
        seed=3,
    ).result.counts
    result = qm.execute(
        artifact,
        on=Selene(timeout=30.0),
        shots=4,
        seed=3,
    )

    assert expected == {((1,) if initial else (0,)): 4}
    assert dict(artifact.metadata.entries)["result_measurements"] == "trailing"
    assert artifact.result_abi.native_width == 1
    assert artifact.result_abi.source_order == (0,)
    assert result.counts == expected


@pytest.mark.parametrize("initial", [False, True])
def test_hugr_composes_branch_measurements_with_an_early_return(
    initial: bool,
) -> None:
    """Branch-local return and continuation measurements share one value ABI."""

    from qamomile.hugr import Hugr, Selene

    source = f"""
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bool]:
    selector = qm.measure(qm.prepare({initial!r}))
    if selector:
        return qm.measure(qm.prepare(True))
    return qm.measure(qm.prepare(False))
"""
    program = qm.build_source(
        source,
        module_name=f"hugr_dynamic_branch_return_measurement_{initial}",
    ).specialize("sample")
    artifact = qm.compile(program, to=Hugr())
    expected = qm.run(
        program,
        on=qm.Reference(),
        shots=4,
        seed=3,
    ).result.counts
    result = qm.execute(
        artifact,
        on=Selene(timeout=30.0),
        shots=4,
        seed=3,
    )

    assert expected == {((1,) if initial else (0,)): 4}
    assert dict(artifact.metadata.entries)["result_measurements"] == "trailing"
    assert artifact.result_abi.native_width == 1
    assert result.counts == expected


@pytest.mark.parametrize("initial", [False, True])
def test_hugr_composes_a_one_arm_return_with_its_continuation(
    initial: bool,
) -> None:
    """A mixed return/yield conditional remains one typed callable flow."""

    from qamomile.hugr import Selene

    source = f"""
import qamomile as qm

@qm.qkernel
def choose() -> qm.Q[qm.Bool]:
    bit, state = qm.measure_reset(qm.prepare({initial!r}))
    if bit:
        return qm.x(state)
    state = qm.x(state)
    return state

@qm.qkernel
def sample() -> qm.Shot[qm.Bool]:
    return qm.measure(choose())
"""
    program = qm.build_source(
        source,
        module_name=f"hugr_one_arm_return_{initial}",
    ).specialize("sample")

    result = qm.run(
        program,
        on=Selene(timeout=30.0),
        shots=4,
        seed=3,
    ).result

    assert result.counts == {(1,): 4}


def test_hugr_propagates_return_from_a_statically_interpreted_for() -> None:
    """A selected finite-loop return exits before the post-loop continuation."""

    from qamomile.hugr import Selene

    source = """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Bits[1]:
    state = qm.qubits(1)
    for index in range(3):
        if index == 1:
            state = qm.x(state)
            return qm.measure(state)
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="hugr_static_for_return",
    ).specialize("sample")

    result = qm.run(
        program,
        on=Selene(timeout=30.0),
        shots=4,
        seed=3,
    ).result

    assert result.counts == {(1,): 4}


def test_hugr_materializes_dynamic_qubit_selection_at_launch() -> None:
    """Rust resolves a launch-time wire selector before HUGR emission."""

    from qamomile.hugr import Selene

    exercise_dynamic_qubit_selection_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_materializes_dynamic_classical_index_at_launch() -> None:
    """Rust resolves an aggregate path before structured HUGR emission."""

    from qamomile.hugr import Selene

    exercise_dynamic_classical_index_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_executes_launch_bound_structural_math() -> None:
    """HUGR receives the loop after Rust evaluates structural math."""

    from qamomile.hugr import Selene

    exercise_structural_math_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_executes_multi_owner_host_dispatch() -> None:
    """HUGR materialization retains both independently updated owned outputs."""

    from qamomile.hugr import Selene

    exercise_multi_owner_dispatch_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_executes_documented_quantum_kernel_classification() -> None:
    """Selene executes the closed documented overlap kernel without a custom path."""

    from qamomile.hugr import Selene

    exercise_static_classification_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_executes_documented_hydrogen_vqe() -> None:
    """Selene estimates the closed documented hydrogen trial energy."""

    from qamomile.hugr import Selene

    exercise_static_hydrogen_vqe_backend(
        endpoint=Selene(timeout=30.0),
        seed=2,
    )


def test_hugr_executes_documented_qsci_sampling() -> None:
    """Selene samples the closed documented QSCI trial state."""

    from qamomile.hugr import Selene

    exercise_static_qsci_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_executes_documented_hamiltonian_simulation() -> None:
    """Selene executes the closed documented noncommuting Rabi evolution."""

    from qamomile.hugr import Selene

    exercise_static_rabi_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_executes_documented_mottonen_encoding() -> None:
    """Selene prepares the closed documented complex Möttönen state."""

    from qamomile.hugr import Selene

    exercise_static_mottonen_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_executes_documented_multidimensional_qft() -> None:
    """Selene executes the closed documented axiswise Fourier transform."""

    from qamomile.hugr import Selene

    exercise_static_multidimensional_qft_backend(
        endpoint=Selene(timeout=30.0),
        seed=2,
    )


def test_hugr_executes_documented_hybrid_qnn() -> None:
    """Selene executes the closed documented Hybrid QNN layer."""

    from qamomile.hugr import Selene

    exercise_static_hqnn_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_executes_documented_option_pricing() -> None:
    """Selene executes the closed amplitude-estimation option pricer."""

    from qamomile.hugr import Selene

    exercise_static_option_pricing_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_executes_documented_pce() -> None:
    """Selene estimates and decodes the closed documented PCE state."""

    from qamomile.hugr import Selene

    exercise_static_pce_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_executes_documented_qemcmc() -> None:
    """Selene executes the closed quantum-enhanced MCMC proposal."""

    from qamomile.hugr import Selene

    exercise_static_qemcmc_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_executes_documented_bit_flip_code() -> None:
    """Selene coherently corrects each single data-qubit bit flip."""

    from qamomile.hugr import Selene

    exercise_static_bit_flip_code_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_executes_documented_steane_code() -> None:
    """Selene coherently corrects a bounded Steane error case."""

    from qamomile.hugr import Selene

    exercise_steane_code_backend(endpoint=Selene(timeout=60.0), seed=2)


def test_hugr_executes_variational_composition() -> None:
    """Selene executes QAOA and FQAOA without algorithm-specific lowering."""

    from qamomile.hugr import Selene

    exercise_static_variational_backend(endpoint=Selene(timeout=30.0), seed=2)


@pytest.mark.hugr_slow
def test_hugr_executes_grover_adaptive_search() -> None:
    """Selene executes closed GAS composition without custom lowering."""

    from qamomile.hugr import Selene

    # Shared inverse definitions add call instructions to the package. The exact
    # witness emits 323 instructions, above the 256-instruction smoke budget.
    exercise_gas_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_executes_product_formula_composition() -> None:
    """Selene executes the shared abstract-evolution schedule."""

    from qamomile.hugr import Selene

    exercise_static_product_formula_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_executes_abstract_pauli_evolution() -> None:
    """Selene consumes the same late Pauli-gadget schedule as circuit targets."""

    from qamomile.hugr import Selene

    program = qm.build_source(PAULI_EVOLUTION_SOURCE).specialize(
        "pauli_evolution",
        **{"hamiltonian": 0.5 * qm.Z(0) * qm.Z(1)},
    )

    assert qm.run(
        program,
        on=Selene(timeout=30.0),
        shots=4,
        seed=2,
    ).result.counts == {(1, 0): 4}
    exercise_normalized_model_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_executes_uniformly_controlled_rotation() -> None:
    """Selene executes the public function-composed Gray walk."""

    from qamomile.hugr import Selene

    exercise_uniformly_controlled_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_executes_coherent_xor_lookup() -> None:
    """Selene executes lookup through total callable selection."""

    from qamomile.hugr import Selene

    exercise_xor_lookup_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_executes_sparse_matrix_lookup() -> None:
    """Direct HUGR executes functional sparse row/slot access."""

    from qamomile.hugr import Selene

    exercise_sparse_lookup_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_executes_static_matrix_access() -> None:
    """Direct HUGR executes typed Matrix access after specialization."""

    from qamomile.hugr import Selene

    exercise_static_matrix_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_executes_vector_phase_qsvt() -> None:
    """Direct HUGR executes the vector-phase QSVT body with static phases."""

    from qamomile.hugr import Selene

    assert qm.run(
        static_vector_qsvt_sample,
        on=Selene(timeout=30.0),
        shots=4,
        seed=2,
    ).result.counts == {(0, 1): 4}


pytestmark = pytest.mark.hugr
pytest.importorskip("hugr")


def test_selene_endpoint_is_public() -> None:
    """Expose Selene as the supported local HUGR endpoint."""

    import qamomile.hugr as hugr

    assert "Selene" in hugr.__all__
    assert hugr.Selene().target == hugr.Hugr()


def test_selene_executes_shots_across_processes() -> None:
    """Delegate bounded multi-process shot execution to Selene."""

    from qamomile.hugr import Selene

    program = qm.build_source(RETAINED_HUGR_SOURCE).specialize("retained_basis")
    result = qm.run(
        program,
        on=Selene(n_processes=2),
        shots=4,
        seed=3,
    ).result

    assert result.counts == {(1, 0, 1): 4}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"simulator": "stim"},
        {"timeout": -1.0},
        {"timeout": float("inf")},
        {"n_processes": 0},
        {"n_processes": True},
    ],
)
def test_selene_rejects_invalid_options(kwargs: dict[str, object]) -> None:
    """Reject unsupported simulators and invalid local execution bounds."""

    from qamomile.hugr import Selene

    with pytest.raises((TypeError, ValueError)):
        Selene(**kwargs)


def test_selene_requires_result_instrumentation() -> None:
    """Reject HUGR artifacts that omit ordered QSystem result records."""

    from qamomile.compilation import Invocation
    from qamomile.hugr import Hugr, Selene

    program = qm.build_source(BELL_SOURCE).specialize("bell")
    artifact = qm.compile(program, to=Hugr(record_results=False))

    with pytest.raises(qm.ExecutionError, match="Q8690"):
        Selene().launch(artifact, Invocation.create(shots=1, seed=0))


def test_selene_decodes_only_well_formed_result_streams() -> None:
    """Validate Selene shot count, tag width, value domain, and stream shape."""

    from qamomile.hugr._selene import _decode_result_counts

    assert _decode_result_counts(
        [[("qamomile", True), ("qamomile", 0)]],
        shots=1,
        width=2,
    ) == {(1, 0): 1}
    for native_shots, shots, width in (
        ([[("qamomile", 1)]], 1, 2),
        ([[("qamomile", 2)]], 1, 1),
        ([[("qamomile", 1)]], 2, 1),
    ):
        with pytest.raises(qm.ExecutionError, match="Q8692"):
            _decode_result_counts(native_shots, shots=shots, width=width)
    with pytest.raises(qm.ExecutionError, match="Q8693"):
        _decode_result_counts([[("malformed",)]], shots=1, width=1)


def test_selene_fails_closed_without_the_optional_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report the HUGR extra when Selene is unavailable at execution time."""

    import sys

    from qamomile.hugr._selene import _load_selene

    monkeypatch.setitem(sys.modules, "selene_sim", None)
    with pytest.raises(ImportError, match=r"qamomile\[hugr\]"):
        _load_selene()


def test_hugr_executes_signal_composition() -> None:
    """Execute shared region phases and editable signal schedules."""
    from qamomile.hugr import Selene

    from .test_signals import exercise_signals_backend

    exercise_signals_backend(endpoint=Selene())


def test_hugr_executes_standard_optimized_qcore() -> None:
    """HUGR executes the shared exact QCore optimization witness."""

    from qamomile.hugr import Selene

    from .test_standard_optimization import exercise_standard_optimization_backend

    exercise_standard_optimization_backend(endpoint=Selene(timeout=30.0))


def test_hugr_executes_functional_amplitude_encoding() -> None:
    """Direct HUGR samples and estimates the multilevel Gray-walk body."""

    from qamomile.hugr import Selene

    endpoint = Selene(timeout=30.0)
    assert qm.run(amplitude_two_sample, on=endpoint, shots=4, seed=2).result.counts == {
        (1, 1): 4
    }
    assert (
        qm.expectation(amplitude_two_state, qm.Z(0), on=endpoint, shots=4, seed=2).value
        == -1.0
    )


def test_hugr_materializes_runtime_parametric_amplitude_at_launch() -> None:
    """One sealed source artifact accepts aggregate Param values on Selene."""

    from qamomile.hugr import Hugr, Selene

    exercise_runtime_amplitude_backend(
        endpoint=Selene(timeout=30.0),
        target=Hugr(),
        seed=2,
    )


@pytest.mark.parametrize(
    ("source", "entrypoint", "parameters", "expected"),
    [
        (HOST_DISPATCH_SOURCE, "host_dispatch", {"limit": 1}, (1,)),
        (LAUNCH_LOOP_SOURCE, "launch_loop", {"rounds": 3}, (1,)),
    ],
)
def test_hugr_materializes_launch_controlled_qcore(
    source: str,
    entrypoint: str,
    parameters: dict[str, object],
    expected: tuple[int, ...],
) -> None:
    """Rust closes exact predicates and loops before structured HUGR emission."""

    from qamomile.compilation import _materialize_launch_artifact
    from qamomile.hugr import Hugr, Selene

    program = qm.build_source(source).specialize(entrypoint)
    artifact = qm.compile(program, to=Hugr())
    child = _materialize_launch_artifact(artifact, parameters, target=Hugr())
    assert "rust-qcore-standard-optimization" not in artifact.report.stages
    assert child.report.stages.count("rust-qcore-standard-optimization") == 1
    result = qm.execute(
        artifact,
        on=Selene(timeout=30.0),
        parameters=parameters,
        shots=4,
        seed=2,
    )

    assert artifact.format == "qamomile.launch-materializer.v1"
    assert Requirement.HOST_DISPATCH in artifact.execution_contract.requirements
    assert result.counts == {expected: 4}


def test_hugr_executes_functional_pauli_block_encoding() -> None:
    """Direct HUGR executes Pauli production through the existing LCU body."""

    from qamomile.hugr import Selene

    exercise_pauli_block_backend(endpoint=Selene(timeout=30.0), seed=2)


def test_hugr_executes_functional_block_producers() -> None:
    """Direct HUGR executes both producers without specialized lowering."""

    from qamomile.hugr import Selene

    exercise_functional_block_producers_backend(
        endpoint=Selene(timeout=30.0),
        seed=2,
    )


def test_hugr_executes_standard_python_record_results() -> None:
    """Direct HUGR carries records through its existing product row types."""

    from qamomile.hugr import Selene

    program = qm.build_source(RECORD_RESULT_SOURCE).specialize("record_result")
    assert qm.run(
        program,
        on=Selene(timeout=30.0),
        shots=4,
        seed=2,
    ).result.values == {(("first", (1,)), ("pair", (0, 1))): 4}


def test_hugr_executes_coherent_product_reassociation() -> None:
    """Direct HUGR reassociates wires without emitting a native operation."""

    from qamomile.hugr import Selene

    program = qm.build_source(COHERENT_RESHAPE_SOURCE).specialize("branched_adjoint")
    assert qm.run(
        program,
        on=Selene(timeout=30.0),
        shots=4,
        seed=2,
    ).result.counts == {(1, 0, 1): 4}


def test_hugr_executes_callable_control_and_multiplex() -> None:
    """Direct HUGR composes multiplex from shared controlled-call semantics."""

    from qamomile.hugr import Selene

    selected = qm.build_source(CALLABLE_TRANSFORMS_SOURCE).specialize(
        "callable_transforms"
    )
    assert qm.run(
        selected,
        on=Selene(timeout=30.0),
        shots=4,
        seed=2,
    ).result.values == {(1, (1,)): 4}


def test_hugr_executes_compiler_derived_selected_adjoint() -> None:
    """Direct HUGR executes multiplex(U_i-adjoint) from the common dagger."""

    from qamomile.hugr import Selene

    program = qm.build_source(COHERENT_ADJOINT_SOURCE).specialize(
        "selected_adjoint_roundtrip"
    )
    assert qm.run(
        program,
        on=Selene(timeout=30.0),
        shots=4,
        seed=2,
    ).result.counts == {(1, 0): 4}


def test_hugr_executes_compiler_derived_coherent_cleanup() -> None:
    """Direct HUGR consumes Rust-materialized select-adjoint helpers."""

    from qamomile.hugr import Selene

    source = """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def coherent_cleanup() -> qm.Bits[1]:
    index = qm.qubits(1)
    target = qm.qubits(1)
    index, target = qm.multiplex(index, (unchanged, flipped), target)
    scratch = qm.concat(index, target)
    output = qm.prepare((1,))
    return qm.measure(output)
"""
    program = qm.build_source(source).specialize("coherent_cleanup")

    assert qm.run(
        program,
        on=Selene(timeout=30.0),
        shots=4,
        seed=2,
    ).result.counts == {(1,): 4}


def test_hugr_executes_indexed_aggregate_control() -> None:
    """Direct HUGR late-unrolls and executes aggregate control selection."""

    from qamomile.hugr import Selene

    module = qm.build_source(INDEXED_CONTROL_SOURCE)
    endpoint = Selene(timeout=30.0)
    assert qm.run(
        module["indexed_control_sample"],
        on=endpoint,
        shots=4,
        seed=2,
    ).result.counts == {(1, 0, 1): 4}
    assert (
        qm.expectation(
            module["indexed_control_state"],
            qm.Z(2),
            on=endpoint,
            shots=4,
            seed=2,
        ).value
        == -1.0
    )


def test_hugr_executes_discrete_functional_phase_estimation() -> None:
    """Direct HUGR samples and estimates the exact rotation-free QPE profile."""

    from qamomile.hugr import Selene

    endpoint = Selene(timeout=30.0)
    exercise_nary_tensor_backend(endpoint=endpoint, seed=2)
    assert qm.run(
        qft_frequency_three,
        on=endpoint,
        shots=4,
        seed=2,
    ).result.counts == {(1, 1, 0): 4}
    assert qm.run(
        qpe_one_bit_sample,
        on=endpoint,
        shots=4,
        seed=2,
    ).result.counts == {(1, 1): 4}
    assert (
        qm.expectation(
            qpe_one_bit_state,
            qm.Z(0),
            on=endpoint,
            shots=4,
            seed=2,
        ).value
        == -1.0
    )


def test_hugr_decodes_typed_fixed_point_after_measurement_log_order() -> None:
    """Selene logs are put in source order before numeric phase decoding."""

    from qamomile.hugr import Selene

    exercise_fixed_qpe_hugr_backend(endpoint=Selene(timeout=30.0), seed=2)
    exercise_windowed_qpe_backend(endpoint=Selene(timeout=30.0), seed=2)
    exercise_documented_windowed_qpe_backend(
        endpoint=Selene(timeout=30.0),
        seed=2,
    )


def test_hugr_executes_documented_reversible_arithmetic() -> None:
    """Selene executes the bounded two-bit comparator and multiplier."""

    from qamomile.hugr import Selene

    exercise_documented_reversible_arithmetic_backend(
        endpoint=Selene(timeout=30.0),
        seed=2,
    )
    exercise_documented_gf2_arithmetic_backend(
        endpoint=Selene(timeout=30.0),
        seed=2,
    )
    exercise_documented_phase_gradient_backend(
        endpoint=Selene(timeout=30.0),
        seed=2,
    )
    exercise_documented_fault_tolerant_data_loading_backend(
        endpoint=Selene(timeout=30.0),
        seed=2,
    )
    exercise_documented_reversible_indexing_backend(
        endpoint=Selene(timeout=30.0),
        seed=2,
    )
    exercise_documented_reversible_trigonometry_backend(
        endpoint=Selene(timeout=30.0),
        seed=2,
    )
    exercise_documented_kikuchi_backend(
        endpoint=Selene(timeout=30.0),
        seed=2,
    )
    exercise_documented_guided_hamiltonian_backend(
        endpoint=Selene(timeout=30.0),
        seed=2,
    )


def test_hugr_executes_documented_qualtran_gap_examples() -> None:
    """Selene executes the bounded ECC and factorized-chemistry witnesses."""

    from qamomile.hugr import Selene

    exercise_documented_elliptic_curve_dlog_backend(
        endpoint=Selene(timeout=30.0),
        seed=2,
    )
    exercise_documented_factorized_chemistry_backend(
        endpoint=Selene(timeout=30.0),
        seed=2,
    )


def test_hugr_executes_discrete_functional_grover_search() -> None:
    """Direct HUGR samples and estimates the discrete Grover composition."""

    from qamomile.hugr import Selene

    endpoint = Selene(timeout=30.0)
    assert qm.run(grover_sample, on=endpoint, shots=4, seed=2).result.counts == {
        (1, 1): 4
    }
    assert (
        qm.expectation(
            grover_state,
            qm.Z(0),
            on=endpoint,
            shots=4,
            seed=2,
        ).value
        == -1.0
    )


def test_hugr_executes_discrete_functional_modular_arithmetic() -> None:
    """Selene samples derived arithmetic and estimates one modular result."""

    from qamomile.hugr import Selene

    endpoint = Selene(timeout=30.0)
    assert qm.run(
        decrement_sample,
        on=endpoint,
        shots=4,
        seed=2,
    ).result.counts == {(1, 1, 0): 4}
    assert qm.run(
        add_modulo_sample,
        on=endpoint,
        shots=4,
        seed=2,
    ).result.counts == {(0, 0, 1, 0, 0): 4}
    assert (
        qm.expectation(
            add_modulo_state,
            qm.Z(2),
            on=endpoint,
            shots=4,
            seed=2,
        ).value
        == -1.0
    )


@pytest.mark.parametrize(
    "choice", ["mod_pow.square_and_multiply", "mod_pow.windowed_lookup"]
)
def test_hugr_compiles_shared_mod_pow_realization(choice: str) -> None:
    """Direct HUGR projects the compiler-owned exact modular-power plan."""

    from qamomile.hugr import Hugr
    from qamomile.lowering import RealizationOptions
    from qamomile.stdlib import regev_factoring

    # Squaring base 2 modulo 5 keeps a nonidentity modular power after optimization.
    regev_sample = regev_factoring(
        bases=(2,), modulus=5, coordinate_width=1, radius=0.75
    )
    for program in (shor_three_sample, regev_sample):
        artifact = qm.compile(
            program,
            to=Hugr(realizations=RealizationOptions({qm.mod_pow: choice})),
        )

        assert artifact.format == "hugr.package.v1"
        assert f"realization:{choice}" in artifact.report.stages


def test_hugr_compiles_multi_exponentiation_callable_realization() -> None:
    """Direct HUGR realizes and executes a nonidentity callable replacement."""

    from qamomile.hugr import Hugr, Selene
    from qamomile.stdlib import (
        modular_multi_exponentiation,
        product_tree_multi_exponentiation,
    )

    baseline = modular_multi_exponentiation(bases=(2,), modulus=3, coordinate_width=1)
    product_tree = product_tree_multi_exponentiation(
        bases=(2,), modulus=3, coordinate_width=1
    )

    @qm.qkernel
    def sample() -> qm.Bits[3]:
        """Measure the nonidentity one-coordinate callable replacement."""
        return qm.measure(baseline(qm.prepare((1,))))

    options = qm.RealizationOptions(
        {baseline: product_tree, qm.mod_pow: "mod_pow.square_and_multiply"}
    )
    artifact = qm.compile(sample, to=Hugr(realizations=options))
    assert artifact.format == "hugr.package.v1"
    assert "realization:mod_pow.square_and_multiply" in artifact.report.stages
    assert (
        f"user asserts {product_tree.symbol} implements {baseline.symbol}"
        in artifact.report.assumptions
    )
    assert qm.execute(
        artifact, on=Selene(realizations=options), shots=4, seed=2
    ).counts == {(1, 0, 1): 4}


def test_hugr_executes_register_modulus_arithmetic() -> None:
    """Selene samples retained-modulus addition."""

    from qamomile.hugr import Selene

    exercise_register_modulus_backend(endpoint=Selene(timeout=60.0), seed=0)


def test_hugr_multicontrol_y_preserves_coherent_relative_phase() -> None:
    """Basis conjugation and exact-phase lowering realize a coherent MCY."""

    from qamomile.hugr import Selene

    source = """
import qamomile as qm

@qm.qkernel
def flip_phase(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.y(state)

@qm.qkernel
def coherent_mcy() -> qm.Q[qm.Bits[4]]:
    state = qm.prepare((0, 1, 1, 0))
    state = qm.h(state, target=0)
    controls, target = qm.split_at(state, 3)
    controls, target = qm.control(flip_phase, controls, target)
    return qm.concat(controls, target)
"""
    program = qm.build_source(source).specialize("coherent_mcy")
    observable = qm.X(0) * qm.Y(3)

    assert (
        qm.expectation(program, observable, on=qm.Reference(), shots=4, seed=2).value
        == 1.0
    )
    assert (
        qm.expectation(program, observable, on=Selene(), shots=4, seed=2).value == 1.0
    )


def test_hugr_executes_functional_qsvt() -> None:
    """Direct HUGR executes direct and PREPARE/SELECT-composed QSVT bodies."""

    from qamomile.hugr import Selene

    endpoint = Selene(timeout=30.0)
    assert qm.run(qsvt_phase_sample, on=endpoint, shots=4, seed=2).result.counts == {
        (1, 0): 4
    }
    assert (
        qm.expectation(
            qsvt_phase_state,
            qm.Z(0),
            on=endpoint,
            shots=4,
            seed=2,
        ).value
        == -1.0
    )
    assert qm.run(lcu_qsvt_sample, on=endpoint, shots=4, seed=2).result.counts == {
        (1, 0): 4
    }
    assert (
        qm.expectation(
            lcu_qsvt_state,
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


def test_hugr_estimates_a_backend_neutral_observable() -> None:
    """Expectation basis programs execute through the ordinary Selene endpoint."""

    from qamomile.hugr import Selene

    state = qm.build_source(STATE_PREPARATION_SOURCE)["plus_state"]
    assert (
        qm.expectation(
            state,
            qm.X(0),
            on=Selene(timeout=30.0),
            shots=4,
            seed=2,
        ).value
        == 1.0
    )


def test_hugr_executes_entangled_partial_trace_out() -> None:
    """Discard lowers to an ignored measurement without leaking its result."""

    from qamomile.hugr import Hugr, Selene

    source = """
import qamomile as qm

@qm.qkernel
def discard_state() -> qm.Bits[1]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    state = qm.cx(state, control=0, target=1)
    discarded, retained = qm.split_at(state, 1)
    qm.discard(discarded)
    return qm.measure(retained)
"""
    program = qm.build_source(source).specialize("discard_state")
    artifact = qm.compile(program, to=Hugr())
    reference = qm.run(
        program,
        on=qm.Reference(),
        shots=8,
        seed=7,
    ).result
    result = qm.execute(
        artifact,
        on=Selene(timeout=30.0),
        shots=8,
        seed=7,
    )

    assert Requirement.DISCARD in program.entry.requirements
    assert Requirement.DISCARD in Selene().capabilities.requirements
    assert artifact.result_abi.native_width == 2
    assert artifact.result_abi.source_order == (1,)
    assert set(reference.counts) == {(0,), (1,)}
    assert set(result.counts) == {(0,), (1,)}
    assert result.shots == 8


@pytest.mark.parametrize("initial", [False, True])
def test_hugr_discards_a_branch_local_owner(initial: bool) -> None:
    """A selected discard participates in the dynamic trailing-event protocol."""

    from qamomile.hugr import Hugr, Selene

    source = f"""
import qamomile as qm

@qm.qkernel
def sample() -> qm.Shot[qm.Bool]:
    selector = qm.measure(qm.prepare({initial!r}))
    if selector:
        scratch = qm.prepare(True)
        qm.discard(scratch)
        result = qm.measure(qm.prepare(True))
    else:
        result = qm.measure(qm.prepare(False))
    return result
"""
    program = qm.build_source(
        source,
        module_name=f"hugr_dynamic_discard_{initial}",
    ).specialize("sample")
    artifact = qm.compile(program, to=Hugr())
    result = qm.execute(
        artifact,
        on=Selene(timeout=30.0),
        shots=4,
        seed=3,
    )

    assert dict(artifact.metadata.entries)["result_measurements"] == "trailing"
    assert artifact.result_abi.native_width == 1
    assert result.counts == {((1,) if initial else (0,)): 4}


def test_hugr_executes_rotation_family_from_one_angle_policy() -> None:
    """One radians-to-half-turn policy serves every direct rotation primitive."""

    from qamomile.hugr import Hugr, Selene

    source = """
import qamomile as qm

@qm.qkernel
def rotate() -> qm.Shot[qm.Bits[8]]:
    state = qm.qubits(8)
    state = qm.rx(state, 3.141592653589793, target=0)
    state = qm.ry(state, 3.141592653589793, target=1)
    state = qm.h(state, target=2)
    state = qm.rz(state, 3.141592653589793, target=2)
    state = qm.h(state, target=2)
    state = qm.h(state, target=3)
    state = qm.p(state, 3.141592653589793, target=3)
    state = qm.h(state, target=3)
    state = qm.x(state, target=4)
    state = qm.h(state, target=5)
    state = qm.cp(state, 3.141592653589793, control=4, target=5)
    state = qm.h(state, target=5)
    state = qm.h(state, target=6)
    state = qm.rzz(state, 3.141592653589793, left=6, right=7)
    state = qm.h(state, target=6)
    return qm.measure(state)
"""
    program = qm.build_source(source).specialize("rotate")

    assert Requirement.ROTATION in program.entry.requirements
    assert Requirement.ROTATION in Hugr().capabilities.requirements
    assert qm.run(
        program,
        on=Selene(timeout=30.0),
        shots=4,
        seed=2,
    ).result.counts == {(1, 1, 1, 1, 1, 1, 1, 0): 4}


def test_direct_hugr_package_validates_and_executes() -> None:
    """Direct QCore lowering produces a package executed by Selene Quest."""

    from qamomile.hugr import Hugr, Selene, load_package

    program = qm.build_source(RETAINED_HUGR_SOURCE).specialize("retained_basis")
    artifact = qm.compile(program, to=Hugr())
    result = qm.execute(
        artifact,
        on=Selene(timeout=30.0),
        shots=4,
        seed=4,
    )

    assert artifact.format == "hugr.package.v1"
    assert "direct-hugr-emission" in artifact.report.stages
    assert dict(artifact.metadata.entries)["retained_functions"] == 1
    assert load_package(artifact) is not None
    assert result.counts == {(1, 0, 1): 4}
    assert result.expectation_z(0, 2) == 1.0
    assert result.worker_pid is None


@pytest.mark.guppy
def test_hugr_exports_a_function_linkable_from_guppy() -> None:
    """A Qamomile owned-state ABI satisfies an owned Guppy declaration."""

    pytest.importorskip("guppylang")
    from guppylang import guppy
    from guppylang.std.lang import owned
    from guppylang.std.quantum import qubit
    from hugr.cli import validate

    from qamomile.hugr import Hugr

    @guppy.declare(link_name="qamomile.tests.flip")
    def qamomile_flip(q: qubit @ owned) -> qubit: ...

    @guppy
    def consumer(q: qubit @ owned) -> qubit:
        return qamomile_flip(q)

    source = """
import qamomile as qm

@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state, target=0)
"""
    program = qm.build_source(source, module_name="guppy_provider").specialize("flip")
    provider = Hugr().export_library(program, link_name="qamomile.tests.flip")
    linked = consumer.compile_function().link(provider)

    validate(linked.to_bytes())


@pytest.mark.guppy
def test_hugr_links_runtime_real_arithmetic_and_rotation_from_guppy() -> None:
    """A Guppy float satisfies QCore Real arithmetic and rotation inputs."""

    pytest.importorskip("guppylang")
    from guppylang import guppy
    from guppylang.std.builtins import float as guppy_float
    from guppylang.std.lang import owned
    from guppylang.std.quantum import qubit
    from hugr.cli import validate

    from qamomile.hugr import Hugr

    @guppy.declare(link_name="qamomile.tests.rotate")
    def qamomile_rotate(
        theta: guppy_float,
        q: qubit @ owned,
    ) -> tuple[guppy_float, qubit]: ...

    @guppy
    def consumer(
        theta: guppy_float,
        q: qubit @ owned,
    ) -> tuple[guppy_float, qubit]:
        return qamomile_rotate(theta, q)

    source = """
import qamomile as qm

@qm.qkernel
def rotate(
    theta: qm.Param[qm.Real],
    state: qm.Q[qm.Bits[1]],
) -> tuple[qm.Real, qm.Q[qm.Bits[1]]]:
    scaled = theta * 0.5
    angle = scaled + 0.25
    if angle > 0.0:
        state = qm.ry(state, scaled, target=0)
    return angle, state
"""
    program = qm.build_source(source, module_name="guppy_real_provider").specialize(
        "rotate"
    )
    provider = Hugr().export_library(program, link_name="qamomile.tests.rotate")
    linked = consumer.compile_function().link(provider)

    validate(linked.to_bytes())


def test_hugr_validates_runtime_real_comparison_chains() -> None:
    """Runtime Real chains remain one typed Boolean control expression."""

    from hugr.cli import validate

    from qamomile.hugr import Hugr

    source = """
import qamomile as qm

@qm.qkernel
def bounded_rotation(
    lower: qm.Param[qm.Real],
    theta: qm.Param[qm.Real],
    upper: qm.Param[qm.Real],
    state: qm.Q[qm.Bits[1]],
) -> qm.Q[qm.Bits[1]]:
    if lower < theta <= upper:
        state = qm.rz(state, -theta, target=0)
    return state
"""
    program = qm.build_source(source, module_name="hugr_real_chain").specialize(
        "bounded_rotation"
    )

    provider = Hugr().export_library(program)

    validate(provider.to_bytes())


def test_hugr_separates_safe_division_from_unproved_angle_finiteness() -> None:
    """A static divisor needs no zero guard while its angle result stays guarded."""

    from hugr.cli import validate

    from qamomile.hugr import Hugr

    source = """
import qamomile as qm

@qm.qkernel
def divide(
    theta: qm.Param[qm.Real],
    state: qm.Q[qm.Bits[1]],
) -> tuple[qm.Real, qm.Q[qm.Bits[1]]]:
    expanded = theta / 0.5
    state = qm.ry(state, theta / 2.0, target=0)
    return expanded, state
"""
    program = qm.build_source(source, module_name="hugr_real_division").specialize(
        "divide"
    )

    package = Hugr().export_library(program)

    validate(package.to_bytes())
    operations = "\n".join(str(data.op) for _, data in package.modules[0].nodes())
    assert operations.count("name='fdiv'") == 2
    assert "Q8611: Real division by zero" not in package.to_str()
    assert operations.count("name='panic'") == 1
    assert "Q8628: ry angle must remain finite" in package.to_str()


def test_hugr_preserves_unrestricted_binary64_real_inputs() -> None:
    """Real comparison does not invent a finite-input contract at HUGR export."""

    from hugr.cli import validate

    from qamomile.hugr import Hugr

    source = """
import qamomile as qm

@qm.qkernel
def positive(value: qm.Param[qm.Real]) -> qm.Bool:
    return value > 0.0
"""
    program = qm.build_source(source, module_name="hugr_binary64_real").specialize(
        "positive"
    )

    package = Hugr().export_library(program)

    validate(package.to_bytes())
    operations = "\n".join(str(data.op) for _, data in package.modules[0].nodes())
    assert "name='fgt'" in operations
    assert "name='panic'" not in operations


@pytest.mark.parametrize(
    ("case", "expression"),
    [
        pytest.param("positive_infinity", "1e309", id="positive-infinity"),
        pytest.param("negative_infinity", "-1e309", id="negative-infinity"),
        pytest.param("nan", "1e309 - 1e309", id="nan"),
    ],
)
def test_hugr_rejects_a_nonfinite_static_real_before_constant_construction(
    case: str,
    expression: str,
) -> None:
    """A nonfinite QCore constant never reaches HUGR's finite-only constructor."""

    from qamomile.hugr import Hugr

    source = f"""
import qamomile as qm

@qm.qkernel
def offset(value: qm.Param[qm.Real]) -> qm.Real:
    return value + ({expression})
"""
    program = qm.build_source(
        source, module_name=f"hugr_nonfinite_constant_{case}"
    ).specialize("offset")

    with pytest.raises(qm.CapabilityError, match="Q8611.*nonfinite Real constant"):
        Hugr().export_library(program)


def test_hugr_static_finite_angle_needs_no_runtime_guard() -> None:
    """A finite literal angle reaches HUGR rotation without a redundant guard."""

    from hugr.cli import validate

    from qamomile.hugr import Hugr

    source = """
import qamomile as qm

@qm.qkernel
def rotate(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.ry(state, 0.5, target=0)
"""
    program = qm.build_source(
        source, module_name="hugr_static_finite_angle"
    ).specialize("rotate")

    package = Hugr().export_library(program)

    validate(package.to_bytes())
    operations = "\n".join(str(data.op) for _, data in package.modules[0].nodes())
    assert "name='panic'" not in operations


def test_hugr_guards_a_runtime_divisor_without_a_totality_proof() -> None:
    """A possible QCore zero division becomes an explicit HUGR panic path."""

    from hugr.cli import validate

    from qamomile.hugr import Hugr

    source = """
import qamomile as qm

@qm.qkernel
def divide(
    theta: qm.Param[qm.Real],
    divisor: qm.Param[qm.Real],
) -> qm.Real:
    return theta / divisor
"""
    program = qm.build_source(source, module_name="hugr_dynamic_division").specialize(
        "divide"
    )

    package = Hugr().export_library(program)

    validate(package.to_bytes())
    operations = "\n".join(str(data.op) for _, data in package.modules[0].nodes())
    assert "name='feq'" in operations
    assert "name='fne'" not in operations
    assert "name='panic'" in operations
    assert "Q8611: Real division by zero" in package.to_str()


def test_hugr_emits_an_exact_dynamic_real_power_domain_guard() -> None:
    """Dynamic Real power uses an exact binary64 integrality guard."""

    from hugr.cli import validate

    from qamomile.hugr import Hugr

    source = """
import qamomile as qm

@qm.qkernel
def power(
    base: qm.Param[qm.Real],
    exponent: qm.Param[qm.Real],
) -> qm.Real:
    return base ** exponent
"""
    program = qm.build_source(source, module_name="hugr_guarded_power").specialize(
        "power"
    )

    package = Hugr().export_library(program)

    validate(package.to_bytes())
    operations = "\n".join(str(data.op) for _, data in package.modules[0].nodes())
    assert "name='bytecast_float64_to_int64'" in operations
    assert "name='ishr'" in operations
    assert "name='ishl'" in operations
    assert "name='iand'" in operations
    assert "name='fpow'" in operations
    assert "name='ffloor'" not in operations
    assert "name='panic'" in operations
    assert "Q8611: Real power is outside its domain" in package.to_str()


def test_hugr_omits_a_proved_unnecessary_real_power_guard() -> None:
    """A nonnegative integral exponent proves the Real power domain statically."""

    from hugr.cli import validate

    from qamomile.hugr import Hugr

    source = """
import qamomile as qm

@qm.qkernel
def safe_powers(
    base: qm.Param[qm.Real],
    exponent: qm.Param[qm.Real],
) -> tuple[qm.Real, qm.Real]:
    return base ** 2.0, 2.0 ** exponent
"""
    program = qm.build_source(source, module_name="hugr_safe_power").specialize(
        "safe_powers"
    )

    package = Hugr().export_library(program)

    validate(package.to_bytes())
    operations = "\n".join(str(data.op) for _, data in package.modules[0].nodes())
    assert operations.count("name='fpow'") == 2
    assert "name='panic'" not in operations


def test_hugr_executes_a_proved_real_power_value(
    tmp_path: Path,
) -> None:
    """Selene observes the binary64 result of a domain-proved HUGR power."""

    import selene_sim as selene
    from hugr import tys
    from hugr.build import Module
    from hugr.std.float import FLOAT_T, FloatVal
    from tket_exts import quantum

    from qamomile.hugr import Hugr
    from qamomile.hugr._module import _finish_hugr_package

    link_name = "qamomile.tests.proved_power"
    source = """
import qamomile as qm

@qm.qkernel
def mark(
    base: qm.Param[qm.Real],
    exponent: qm.Param[qm.Real],
    state: qm.Q[qm.Bits[1]],
) -> qm.Q[qm.Bits[1]]:
    if base ** 2.0 == 0.25 and 2.0 ** exponent == 4.0:
        state = qm.x(state, target=0)
    return state
"""
    program = qm.build_source(source, module_name="hugr_proved_power_value").specialize(
        "mark"
    )
    provider = Hugr().export_library(program, link_name=link_name)
    signature = tys.PolyFuncType(
        [],
        tys.FunctionType([FLOAT_T, FLOAT_T, tys.Qubit], [tys.Qubit]),
    )
    module = Module()
    declaration = module.declare_function(link_name, signature)
    main = module.define_function("main", [], [], visibility="Public")
    base = main.load(FloatVal(-0.5))[0]
    exponent = main.load(FloatVal(2.0))[0]
    state = main.add_op(quantum.qAlloc)[0]
    state = main.call(declaration, base, exponent, state)[0]
    post, _ = main.add_op(quantum.measure, state)
    main.add_op(quantum.qFree, post)
    main.set_outputs()
    consumer, _ = _finish_hugr_package(module)
    linked = consumer.link(provider)

    build_dir = tmp_path / "proved_power"
    build_dir.mkdir()
    runner = selene.build(linked, name="proved_power", build_dir=build_dir)
    measurements = selene.MeasurementExtractor()
    returned = list(
        runner.run(
            selene.Quest(),
            n_qubits=1,
            event_hook=measurements,
            random_seed=2,
            timeout=30.0,
        )
    )

    assert returned == []
    assert [entry.result_value for entry in measurements.log_entries[0]] == [1]


def test_hugr_guards_a_runtime_angle_without_a_finiteness_proof() -> None:
    """Unchecked HUGR rotations receive only runtime-guarded Real results."""

    from hugr.cli import validate

    from qamomile.hugr import Hugr

    source = """
import qamomile as qm

@qm.qkernel
def rotate(
    theta: qm.Param[qm.Real],
    scale: qm.Param[qm.Real],
    state: qm.Q[qm.Bits[1]],
) -> qm.Q[qm.Bits[1]]:
    return qm.ry(state, theta * scale, target=0)
"""
    program = qm.build_source(source, module_name="hugr_unbounded_angle").specialize(
        "rotate"
    )

    package = Hugr().export_library(program)

    validate(package.to_bytes())
    operations = "\n".join(str(data.op) for _, data in package.modules[0].nodes())
    assert "name='fge'" in operations
    assert "name='fle'" in operations
    assert "name='panic'" in operations
    assert "Q8628: ry angle must remain finite" in package.to_str()


def test_hugr_guards_a_real_leaf_inside_a_nested_public_product() -> None:
    """Nested public Real leaves retain unknown finiteness until angle use."""

    from hugr.cli import validate

    from qamomile.hugr import Hugr

    source = """
import qamomile as qm

@qm.qkernel
def rotate(
    values: qm.Param[tuple[qm.Real, tuple[qm.Real, qm.Real]]],
    state: qm.Q[qm.Bits[1]],
) -> qm.Q[qm.Bits[1]]:
    unused, pair = values
    angle, unused = pair
    return qm.ry(state, angle, target=0)
"""
    program = qm.build_source(source, module_name="hugr_nested_real_input").specialize(
        "rotate"
    )

    package = Hugr().export_library(program)

    validate(package.to_bytes())
    operations = "\n".join(str(data.op) for _, data in package.modules[0].nodes())
    assert "name='panic'" in operations
    assert "Q8628: ry angle must remain finite" in package.to_str()


@pytest.mark.parametrize(
    ("case", "expression", "theta", "scale", "overflow_input", "error_message"),
    [
        pytest.param("finite", "theta * scale", 0.5, 2.0, False, None, id="finite"),
        pytest.param(
            "nonfinite_input",
            "theta",
            float.fromhex("0x1.fffffffffffffp+1023"),
            2.0,
            True,
            "Q8628: ry angle must remain finite",
            id="nonfinite-input",
        ),
        pytest.param(
            "overflow",
            "theta * scale",
            float.fromhex("0x1.fffffffffffffp+1023"),
            2.0,
            False,
            "Q8628: ry angle must remain finite",
            id="overflow",
        ),
        pytest.param(
            "zero_divisor",
            "theta / scale",
            1.0,
            0.0,
            False,
            "Q8611: Real division by zero",
            id="zero-divisor",
        ),
        pytest.param(
            "finite_power",
            "theta ** 2.0",
            0.5,
            2.0,
            False,
            None,
            id="finite-power",
        ),
        pytest.param(
            "overflowing_power",
            "theta ** 2.0",
            float.fromhex("0x1.fffffffffffffp+1023"),
            2.0,
            False,
            "Q8628: ry angle must remain finite",
            id="overflowing-power",
        ),
        pytest.param(
            "dynamic_integral_power",
            "theta ** scale",
            -2.0,
            3.0,
            False,
            None,
            id="dynamic-integral-power",
        ),
        pytest.param(
            "dynamic_large_integral_power",
            "theta ** scale",
            1.0,
            float(1 << 52),
            False,
            None,
            id="dynamic-large-integral-power",
        ),
        pytest.param(
            "dynamic_negative_zero_exponent",
            "theta ** scale",
            -2.0,
            -0.0,
            False,
            None,
            id="dynamic-negative-zero-exponent",
        ),
        pytest.param(
            "dynamic_negative_zero_power",
            "theta ** scale",
            -0.0,
            -1.0,
            False,
            "Q8611: Real power is outside its domain",
            id="dynamic-negative-zero-power",
        ),
        pytest.param(
            "dynamic_fractional_power",
            "theta ** scale",
            -2.0,
            0.5,
            False,
            "Q8611: Real power is outside its domain",
            id="dynamic-fractional-power",
        ),
        pytest.param(
            "dynamic_subnormal_power",
            "theta ** scale",
            -2.0,
            float.fromhex("0x0.0000000000001p-1022"),
            False,
            "Q8611: Real power is outside its domain",
            id="dynamic-subnormal-power",
        ),
        pytest.param(
            "dynamic_largest_fractional_power",
            "theta ** scale",
            -2.0,
            float.fromhex("0x1.fffffffffffffp+51"),
            False,
            "Q8611: Real power is outside its domain",
            id="dynamic-largest-fractional-power",
        ),
        pytest.param(
            "dynamic_infinite_exponent",
            "theta ** (scale * 2.0)",
            -2.0,
            float.fromhex("0x1.fffffffffffffp+1023"),
            False,
            "Q8628: ry angle must remain finite",
            id="dynamic-infinite-exponent",
        ),
        pytest.param(
            "dynamic_nan_exponent",
            "theta ** ((scale * 2.0) - (scale * 2.0))",
            -2.0,
            float.fromhex("0x1.fffffffffffffp+1023"),
            False,
            "Q8628: ry angle must remain finite",
            id="dynamic-nan-exponent",
        ),
    ],
)
def test_hugr_executes_runtime_real_guards(
    case: str,
    expression: str,
    theta: float,
    scale: float,
    overflow_input: bool,
    error_message: str | None,
    tmp_path: Path,
) -> None:
    """Linked Qamomile Real guards either continue or panic in Selene."""

    import selene_sim as selene
    from hugr import tys
    from hugr.build import Module
    from hugr.std.float import FLOAT_T, FloatVal
    from selene_sim.exceptions import SelenePanicError
    from tket_exts import quantum

    from qamomile.hugr import Hugr
    from qamomile.hugr._compiler import _float_operation
    from qamomile.hugr._module import _finish_hugr_package

    link_name = f"qamomile.tests.real_guard.{case}"
    source = f"""
import qamomile as qm

@qm.qkernel
def rotate(
    theta: qm.Param[qm.Real],
    scale: qm.Param[qm.Real],
    state: qm.Q[qm.Bits[1]],
) -> qm.Q[qm.Bits[1]]:
    return qm.ry(state, {expression}, target=0)
"""
    program = qm.build_source(source, module_name=f"hugr_real_guard_{case}").specialize(
        "rotate"
    )
    provider = Hugr().export_library(program, link_name=link_name)
    signature = tys.PolyFuncType(
        [],
        tys.FunctionType([FLOAT_T, FLOAT_T, tys.Qubit], [tys.Qubit]),
    )
    module = Module()
    declaration = module.declare_function(link_name, signature)
    main = module.define_function("main", [], [], visibility="Public")
    theta_wire = main.load(FloatVal(theta))[0]
    scale_wire = main.load(FloatVal(scale))[0]
    if overflow_input:
        theta_wire = main.add_op(
            _float_operation("fmul", 2),
            theta_wire,
            scale_wire,
        )[0]
    state = main.add_op(quantum.qAlloc)[0]
    state = main.call(declaration, theta_wire, scale_wire, state)[0]
    main.add_op(quantum.qFree, state)
    main.set_outputs()
    consumer, _ = _finish_hugr_package(module)
    linked = consumer.link(provider)

    build_dir = tmp_path / case
    build_dir.mkdir()
    runner = selene.build(linked, name=f"real_guard_{case}", build_dir=build_dir)

    def run_program() -> list[object]:
        """Execute one linked guard package."""

        return list(
            runner.run(
                selene.Quest(),
                n_qubits=1,
                random_seed=2,
                timeout=30.0,
            )
        )

    if error_message is None:
        assert run_program() == []
    else:
        with pytest.raises(SelenePanicError, match=error_message):
            run_program()


@pytest.mark.parametrize(
    "replacement",
    [None, lambda *_: "not-a-proof", lambda *_: ("power_domain", False)],
)
def test_hugr_real_safety_plan_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    replacement: object,
) -> None:
    """HUGR never duplicates or guesses the native Real safety plan."""

    from qamomile._native import compiler as native
    from qamomile.hugr import Hugr

    source = """
import qamomile as qm

@qm.qkernel
def rotate(
    theta: qm.Param[qm.Real],
    state: qm.Q[qm.Bits[1]],
) -> qm.Q[qm.Bits[1]]:
    return qm.ry(state, theta * 0.5, target=0)
"""
    program = qm.build_source(source, module_name="hugr_native_finiteness").specialize(
        "rotate"
    )
    monkeypatch.setattr(
        native,
        "real_operation_facts",
        replacement,
        raising=False,
    )

    with pytest.raises(qm.CapabilityError, match="Q8685"):
        Hugr().export_library(program)


def test_hugr_library_rejects_arbitrary_precision_integer_abi() -> None:
    """QCore Int is not silently narrowed to HUGR's fixed-width integer."""

    from qamomile.hugr import Hugr

    source = """
import qamomile as qm

@qm.qkernel
def identity(value: qm.Param[qm.Int]) -> qm.Int:
    return value
"""
    program = qm.build_source(source, module_name="hugr_integer_abi").specialize(
        "identity"
    )

    with pytest.raises(qm.CapabilityError, match="Q8641"):
        Hugr().export_library(program)


@pytest.mark.guppy
def test_hugr_guppy_link_rejects_a_mismatched_linear_signature() -> None:
    """The HUGR linker rejects a declaration that consumes the owned qubit."""

    pytest.importorskip("guppylang")
    from guppylang import guppy
    from guppylang.std.lang import owned
    from guppylang.std.quantum import qubit
    from hugr._hugr.linking import HugrLinkingError

    from qamomile.hugr import Hugr

    @guppy.declare(link_name="qamomile.tests.mismatch")
    def qamomile_mismatch(q: qubit @ owned) -> None: ...

    @guppy
    def consumer(q: qubit @ owned) -> None:
        qamomile_mismatch(q)

    source = """
import qamomile as qm

@qm.qkernel
def identity(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state
"""
    program = qm.build_source(source, module_name="guppy_mismatch").specialize(
        "identity"
    )
    provider = Hugr().export_library(program, link_name="qamomile.tests.mismatch")

    with pytest.raises(HugrLinkingError, match="Conflicting signatures"):
        consumer.compile_function().link(provider)


@pytest.mark.parametrize("link_name", ["", "not\nprintable", "x" * 4097])
def test_hugr_rejects_invalid_library_link_names(link_name: str) -> None:
    """Public library symbols are printable and explicitly byte bounded."""

    from qamomile.hugr import Hugr

    source = """
import qamomile as qm

@qm.qkernel
def identity(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state
"""
    program = qm.build_source(source, module_name="invalid_link_name").specialize(
        "identity"
    )

    with pytest.raises(qm.CapabilityError, match="Q8680"):
        Hugr().export_library(program, link_name=link_name)


def test_hugr_rejects_a_static_library_loop_over_its_work_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Library emission fails before unrolling an excessive static loop."""

    import qamomile.hugr._compiler as hugr_compiler
    from qamomile.hugr import Hugr

    monkeypatch.setattr(hugr_compiler, "MAX_HUGR_STATIC_LOOP_ITERATIONS", 4)

    source = """
import qamomile as qm

@qm.qkernel
def repeated(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    for _ in range(5):
        state = qm.x(state, target=0)
    return state
"""
    program = qm.build_source(source, module_name="huge_hugr_loop").specialize(
        "repeated"
    )

    with pytest.raises(qm.CapabilityError, match="Q8682"):
        Hugr().export_library(program)


def test_hugr_rejects_a_library_type_over_its_port_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Library emission rejects an oversized type before allocating its row."""

    import qamomile.hugr._compiler as hugr_compiler
    from qamomile.hugr import Hugr

    monkeypatch.setattr(hugr_compiler, "MAX_HUGR_PORTS", 9)

    source = """
import qamomile as qm

@qm.qkernel
def identity(state: qm.Q[qm.Bits[10]]) -> qm.Q[qm.Bits[10]]:
    return state
"""
    program = qm.build_source(source, module_name="huge_hugr_type").specialize(
        "identity"
    )

    with pytest.raises(qm.CapabilityError, match="Q8681"):
        Hugr().export_library(program)


def test_hugr_library_requires_native_target_lowering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Library emission fails closed when Rust target lowering is unavailable."""

    from qamomile._native import compiler as native
    from qamomile.hugr import Hugr

    source = """
import qamomile as qm

@qm.qkernel
def identity(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state
"""
    program = qm.build_source(source, module_name="missing_native_hugr").specialize(
        "identity"
    )
    monkeypatch.setattr(native, "lower_optimized_adjoint_calls_for_target", None)

    with pytest.raises(qm.CapabilityError, match="Q6200"):
        Hugr().export_library(program)


def test_hugr_executes_extended_discrete_gate_family() -> None:
    """Direct HUGR retains adjoint phases and Toffoli as typed extension ops."""

    from qamomile.hugr import Selene

    program = qm.build_source(EXTENDED_DISCRETE_GATE_SOURCE).specialize(
        "extended_discrete_gates"
    )
    assert qm.run(program, on=Selene(timeout=30.0), shots=4).result.counts == {
        (1, 1, 1): 4
    }


def test_hugr_executes_measure_reset() -> None:
    """Direct HUGR preserves measurement followed by reset."""

    from qamomile.hugr import Selene

    program = qm.build_source(MEASURE_RESET_SOURCE).specialize("measure_and_reset")
    assert qm.run(
        program,
        on=Selene(timeout=30.0),
        shots=4,
    ).result.values == {((1,), (0,)): 4}


def test_hugr_executes_nondestructive_pauli_measurements() -> None:
    """Direct HUGR preserves repeatable X, Y, and Z projective measurements."""

    from qamomile.hugr import Selene

    program = qm.build_source(MEASURE_AXES_SOURCE).specialize("measure_axes")
    result = qm.run(program, on=Selene(timeout=30.0), shots=4, seed=2).result

    assert result.values == {(1, 1, 0, 0, 0, 0, (1,), (0,), (0,)): 4}


def test_hugr_preserves_and_executes_measurement_control() -> None:
    """A Shot-time Python if becomes an executable HUGR Conditional."""

    from qamomile.hugr import Hugr, Selene

    program = qm.build_source(FEEDBACK_SOURCE).specialize("feedback")
    artifact = qm.compile(program, to=Hugr())
    result = qm.execute(
        artifact,
        on=Selene(timeout=30.0),
        shots=4,
        seed=5,
    )

    assert artifact.result_abi.source_order == (1, 2)
    assert set(result.counts) <= {(0, 0), (1, 1)}


def test_hugr_executes_complete_boolean_measurement_dataflow() -> None:
    """Bitwise Bool operations, casts, and comparisons control Selene gates."""

    from qamomile.hugr import Selene

    source = """
import qamomile as qm

@qm.qkernel
def boolean_dataflow() -> qm.Bits[8]:
    state = qm.prepare((1, 0, 0, 0, 0, 0, 0, 0))
    left, state = qm.measure_z(state, 0)
    right, state = qm.measure_z(state, 1)
    if left & ~right:
        state = qm.x(state, target=1)
    if left | right:
        state = qm.x(state, target=2)
    if left ^ right:
        state = qm.x(state, target=3)
    if left != right:
        state = qm.x(state, target=4)
    if left == ~right:
        state = qm.x(state, target=5)
    if bool(left):
        state = qm.x(state, target=6)
    if bool(right):
        state = qm.x(state, target=7)
    return qm.measure(state)
"""
    program = qm.build_source(source, module_name="hugr_boolean_dataflow").specialize(
        "boolean_dataflow"
    )

    result = qm.run(
        program,
        on=Selene(timeout=30.0),
        shots=4,
        seed=2,
    ).result

    assert result.counts == {(1, 1, 1, 1, 1, 1, 1, 0): 4}


def test_hugr_executes_negative_indexed_boolean_result() -> None:
    """Rust normalizes a negative Shot aggregate index before HUGR selection."""

    from qamomile.hugr import Selene

    source = """
import qamomile as qm

@qm.qkernel
def selected() -> qm.Shot[qm.Bool]:
    bits = qm.measure(qm.prepare((1, 0)))
    return bits[0] ^ bits[-1]
"""
    program = qm.build_source(
        source, module_name="hugr_negative_bool_index"
    ).specialize("selected")

    result = qm.run(
        program,
        on=Selene(timeout=30.0),
        shots=4,
        seed=2,
    ).result

    assert result.values == {True: 4}


def test_hugr_preserves_binary64_boolean_semantics(tmp_path: Path) -> None:
    """One Selene build covers Real and fixed-word Boolean semantics."""

    import selene_sim as selene
    from hugr import tys, val
    from hugr.build import Module
    from hugr.std.float import FLOAT_T, FloatVal
    from tket_exts import quantum

    from qamomile.hugr import Hugr
    from qamomile.hugr._compiler import _float_operation
    from qamomile.hugr._module import _finish_hugr_package

    real_link_name = "qamomile.tests.real_boolean_semantics"
    real_source = """
import qamomile as qm

@qm.qkernel
def mark(
    value: qm.Param[qm.Real],
    divisor: qm.Param[qm.Real],
    state: qm.Q[qm.Bits[2]],
) -> qm.Q[qm.Bits[2]]:
    if bool(value):
        state = qm.x(state, target=0)
    if 1.0 / divisor != 0.0:
        state = qm.x(state, target=1)
    return state
"""
    real_program = qm.build_source(
        real_source,
        module_name="hugr_real_boolean_semantics",
    ).specialize("mark")
    real_provider = Hugr().export_library(real_program, link_name=real_link_name)
    real_signature = tys.PolyFuncType(
        [],
        tys.FunctionType(
            [FLOAT_T, FLOAT_T, tys.Qubit, tys.Qubit],
            [tys.Qubit, tys.Qubit],
        ),
    )
    word_link_name = "qamomile.tests.word_boolean_semantics"
    word_source = """
import qamomile as qm

@qm.qkernel
def mark(
    left: qm.Param[qm.UInt[3]],
    right: qm.Param[qm.UInt[3]],
    state: qm.Q[qm.Bits[4]],
) -> qm.Q[qm.Bits[4]]:
    if left == right:
        state = qm.x(state, target=0)
    if left != right:
        state = qm.x(state, target=1)
    if left[0] in right:
        state = qm.x(state, target=2)
    if (left, right) == (left, right):
        state = qm.x(state, target=3)
    return state
"""
    word_program = qm.build_source(
        word_source,
        module_name="hugr_word_boolean_semantics",
    ).specialize("mark")
    word_provider = Hugr().export_library(word_program, link_name=word_link_name)
    word_signature = tys.PolyFuncType(
        [],
        tys.FunctionType(
            [*[tys.Bool] * 6, *[tys.Qubit] * 4],
            [tys.Qubit] * 4,
        ),
    )
    module = Module()
    real_declaration = module.declare_function(real_link_name, real_signature)
    word_declaration = module.declare_function(word_link_name, word_signature)
    main = module.define_function("main", [], [], visibility="Public")
    zero = main.load(FloatVal(0.0))[0]
    negative_zero = main.load(FloatVal(-0.0))[0]
    nonzero = main.load(FloatVal(-2.0))[0]
    largest = main.load(FloatVal(float.fromhex("0x1.fffffffffffffp+1023")))[0]
    two = main.load(FloatVal(2.0))[0]
    infinity = main.add_op(_float_operation("fmul", 2), largest, two)[0]
    nan = main.add_op(_float_operation("fsub", 2), infinity, infinity)[0]
    words = tuple(
        main.load(value)[0]
        for value in (
            val.TRUE,
            val.FALSE,
            val.TRUE,
            val.TRUE,
            val.FALSE,
            val.FALSE,
        )
    )
    states = [main.add_op(quantum.qAlloc)[0] for _ in range(8)]
    states[:4] = main.call(word_declaration, *words, *states[:4])
    for index, value in enumerate((zero, negative_zero, nonzero, nan)):
        offset = index * 2
        states[offset : offset + 2] = main.call(
            real_declaration,
            value,
            nan,
            *states[offset : offset + 2],
        )
    previous_measurement = None
    for state in states:
        measurement = main.add_op(quantum.measure, state)
        if previous_measurement is not None:
            main.hugr.add_order_link(previous_measurement, measurement)
        previous_measurement = measurement
        post, _ = measurement
        main.add_op(quantum.qFree, post)
    main.set_outputs()
    consumer, _ = _finish_hugr_package(module)
    linked = consumer.link(real_provider).link(word_provider)

    build_dir = tmp_path / "real_boolean_semantics"
    build_dir.mkdir()
    runner = selene.build(
        linked,
        name="real_boolean_semantics",
        build_dir=build_dir,
    )
    measurements = selene.MeasurementExtractor()
    returned = list(
        runner.run(
            selene.Quest(),
            n_qubits=8,
            event_hook=measurements,
            random_seed=2,
            timeout=30.0,
        )
    )

    assert returned == []
    assert [entry.result_value for entry in measurements.log_entries[0]] == [
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        1,
    ]


@pytest.mark.parametrize(
    "replacement",
    [None, lambda *_: "not-a-position", lambda *_: 1],
)
def test_hugr_aggregate_projection_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    replacement: object,
) -> None:
    """HUGR never reimplements a missing or malformed native index rule."""

    from qamomile._native import compiler as native
    from qamomile.hugr import Hugr

    source = """
import qamomile as qm

@qm.qkernel
def selected() -> qm.Shot[qm.Bool]:
    bits = qm.measure(qm.qubits(1))
    return not bits[0]
"""
    program = qm.build_source(source, module_name="hugr_native_bool_index").specialize(
        "selected"
    )
    monkeypatch.setattr(
        native,
        "normalize_aggregate_index",
        replacement,
        raising=False,
    )

    with pytest.raises(qm.CapabilityError, match="Q8686"):
        Hugr().lower(program)


def test_hugr_executes_modular_automatic_uncomputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct HUGR consumes only the Rust-owned callable-adjoint lowering."""

    from qamomile._native import compiler as native
    from qamomile.hugr import Hugr, Selene

    calls: list[str] = []
    lower_adjoint = native.lower_optimized_adjoint_calls_for_target

    def record_native_lowering(payload: bytes) -> bytes:
        """Record and delegate one native target-adjoint transformation."""

        calls.append("native")
        return bytes(lower_adjoint(payload))

    monkeypatch.setattr(
        native, "lower_optimized_adjoint_calls_for_target", record_native_lowering
    )
    program = qm.build_source(MODULAR_CLEANUP_SOURCE).specialize("modular_cleanup")
    artifact = qm.compile(program, to=Hugr())
    result = qm.execute(
        artifact,
        on=Selene(timeout=30.0),
        shots=4,
        seed=11,
    )

    assert calls == ["native"]
    assert result.counts == {(1, 0, 0): 4}


def test_hugr_executes_multi_input_automatic_uncomputation() -> None:
    """Direct HUGR executes a packed adjoint restoring independent inputs."""

    from qamomile.hugr import Selene

    program = qm.build_source(MULTI_INPUT_CLEANUP_SOURCE).specialize(
        "multi_input_cleanup"
    )
    result = qm.run(
        program,
        on=Selene(timeout=30.0),
        shots=4,
        seed=7,
    ).result

    assert result.counts == {(1,): 4}


def test_hugr_preserves_private_trailing_measurement_width() -> None:
    """Selene receives the full measurement record and projects the result."""

    from qamomile.hugr import Hugr, Selene

    program = qm.build_source(TRAILING_MEASUREMENT_SOURCE).specialize(
        "trailing_measurement"
    )
    artifact = qm.compile(program, to=Hugr())
    result = qm.execute(
        artifact,
        on=Selene(timeout=30.0),
        shots=4,
        seed=3,
    )

    assert artifact.result_abi.native_width == 2
    assert artifact.result_abi.source_order == (0,)
    assert result.counts == {(0,): 4}


def test_hugr_tracks_measurement_provenance_through_conditional_phi() -> None:
    """A measured bit keeps semantic provenance after branch pass-through."""

    from qamomile.hugr import Hugr, Selene

    source = """
import qamomile as qm

@qm.qkernel
def passthrough() -> qm.Shot[qm.Bool]:
    state = qm.qubits(1)
    flag, state = qm.measure_z(state, 0)
    if flag:
        result = flag
    else:
        result = flag
    ignored = qm.measure(state)
    return result
"""
    artifact = qm.compile(qm.build_source(source).specialize("passthrough"), to=Hugr())
    result = qm.execute(artifact, on=Selene(timeout=30.0), shots=3, seed=9)

    assert artifact.result_abi.source_order == (0,)
    assert artifact.result_abi.native_width == 2
    assert result.counts == {(0,): 3}


def test_hugr_branch_resource_bound_includes_parent_allocations() -> None:
    """Conditional allocation slots are added to the parent resource bound."""

    from qamomile.hugr import Hugr, Selene

    source = """
import qamomile as qm

@qm.qkernel
def branch_resource() -> qm.Bits[2]:
    state = qm.qubits(2)
    flag, state = qm.measure_z(state, 0)
    if flag:
        scratch_true = qm.qubits(1)
        scratch_true = qm.x(scratch_true)
    else:
        scratch_false = qm.qubits(1)
        scratch_false = qm.h(scratch_false)
    return qm.measure(state)
"""
    artifact = qm.compile(
        qm.build_source(source).specialize("branch_resource"),
        to=Hugr(),
    )
    result = qm.execute(artifact, on=Selene(timeout=30.0), shots=3, seed=6)

    assert artifact.report.literal_int("qamomile.circuit.virtual_qubits") == 3
    assert set(result.counts) == {(0, 0)}


def test_hugr_executes_typed_product_result() -> None:
    """Direct HUGR lowering retains product fields through flat ports."""

    from qamomile.hugr import Selene

    result = qm.run(
        qm.build_source(STRUCTURED_RESULT_SOURCE).specialize("structured_result"),
        on=Selene(timeout=30.0),
        shots=3,
        seed=2,
    ).result

    assert result.counts == {(1, 0, 1): 3}
    assert result.values == {((1,), (0, 1)): 3}


@pytest.mark.parametrize("choice", ["add.qft", "add.ripple"])
def test_hugr_executes_late_realizations(choice: str) -> None:
    """One shared arithmetic source survives exact alternative realizations."""
    pytest.importorskip("selene_sim")
    from qamomile.hugr import Selene
    from qamomile.lowering import RealizationOptions

    from .test_realizations import exercise_realization_backend

    exercise_realization_backend(
        Selene(realizations=RealizationOptions({"quantum.add_into": choice})),
        supports_mod_pow=False,
    )


def test_hugr_zero_layer_qaoa() -> None:
    """Execute the shared zero layer qaoa contract locally."""
    from qamomile.hugr import Selene

    from .test_library_compatibility import exercise_zero_layer_qaoa_backend

    exercise_zero_layer_qaoa_backend(Selene(timeout=30.0))


def test_hugr_approximate_qft() -> None:
    """Execute the shared approximate qft contract locally."""
    from qamomile.hugr import Selene

    from .test_library_compatibility import exercise_approximate_qft_backend

    exercise_approximate_qft_backend(Selene(timeout=30.0))


def test_hugr_classical_array_compatibility() -> None:
    """Execute the shared classical array contract locally."""
    from qamomile.hugr import Selene

    from .test_classical_array_compatibility import exercise_classical_array_backend

    exercise_classical_array_backend(Selene(timeout=30.0))


def test_hugr_qubit_selection() -> None:
    """Execute owned noncontiguous selection on this local SDK."""
    from qamomile.hugr import Selene

    from .test_qubit_selection import exercise_qubit_selection_backend

    exercise_qubit_selection_backend(Selene(timeout=30.0), seed=7)


def test_hugr_keyed_parameters() -> None:
    """Rebind fixed-key dictionaries through the ordinary Param Vector ABI."""
    from qamomile.hugr import Selene

    from .test_parameter_map import exercise_parameter_map_backend

    exercise_parameter_map_backend(Selene(timeout=30.0), seed=7)


def test_hugr_quantum_records() -> None:
    """Execute whole-record ownership transfers and coherent transforms."""
    from qamomile.hugr import Selene

    from .test_quantum_records import exercise_quantum_record_backend

    exercise_quantum_record_backend(Selene(timeout=30.0), seed=7)


def test_hugr_recursion() -> None:
    """Execute self, mutual, branching, and higher-order recursive helpers."""
    from qamomile.hugr import Selene

    from .test_recursion import exercise_recursive_backend

    exercise_recursive_backend(Selene(timeout=30.0), seed=7)


def test_hugr_sqrt() -> None:
    """Execute Const and launch-bound square roots through the native algebra."""
    from qamomile.hugr import Selene

    from .test_sqrt import exercise_sqrt_backend

    exercise_sqrt_backend(Selene(timeout=30.0), seed=7)
