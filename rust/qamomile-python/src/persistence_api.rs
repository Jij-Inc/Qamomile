//! Persistence projections and entry points for the native Python module.

use super::abi_api::circuit_launch_bindings;
use super::circuit_api::{
    CircuitDispatchDecision, CircuitDispatchParameter, circuit_dispatch_decision,
    circuit_dispatch_parameter,
};
use crate::program_owner::program_owner;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};
use qamomile_compiler::{
    ArtifactFamilyPayload as NativeArtifactFamilyPayload, ArtifactFamilyVariantInput,
    LaunchMaterializerPayload as NativeLaunchMaterializerPayload,
    inspect_artifact as inspect_native_artifact,
    inspect_artifact_family_payload as inspect_native_artifact_family_payload,
    inspect_circuit_family_selection as inspect_native_circuit_family_selection,
    inspect_execution_snapshot as inspect_native_execution_snapshot,
    inspect_launch_materializer_artifact as inspect_native_launch_materializer_artifact,
    inspect_launch_materializer_payload as inspect_native_launch_materializer_payload,
    inspect_module_local_verification_work as inspect_native_module_local_verification_work,
    inspect_program_local_verification_work as inspect_native_program_local_verification_work,
    inspect_qcore_envelope as inspect_native_qcore_envelope,
    inspect_qcore_envelope_with_work as inspect_native_qcore_envelope_with_work,
    module_semantic_digest as native_module_semantic_digest,
    resolve_module_entrypoint as resolve_native_module_entrypoint,
    seal_artifact as seal_native_artifact,
    seal_artifact_family_payload as seal_native_artifact_family_payload,
    seal_execution_snapshot as seal_native_execution_snapshot,
    seal_hir_package as seal_native_hir_package,
    seal_launch_materializer_payload as seal_native_launch_materializer_payload,
    seal_qcore_package as seal_native_qcore_package,
    select_circuit_family_variant as select_native_circuit_family_variant,
    selected_module_semantic_digest as native_selected_module_semantic_digest,
};

/// One decision path projected from a checked persistent family selector.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct CircuitFamilySelectionVariant {
    #[pyo3(get)]
    decisions: Vec<CircuitDispatchDecision>,
}

/// Structural view of a checked persistent family selector.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct CircuitFamilySelectionProjection {
    #[pyo3(get)]
    dispatch_parameters: Vec<CircuitDispatchParameter>,
    #[pyo3(get)]
    variants: Vec<CircuitFamilySelectionVariant>,
}

/// One checked child retained by a persistent artifact-family payload.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct ArtifactFamilyVariantProjection {
    #[pyo3(get)]
    decisions: Vec<CircuitDispatchDecision>,
    artifact: Vec<u8>,
}

/// Structural view of a canonical persistent artifact-family payload.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct ArtifactFamilyPayloadProjection {
    #[pyo3(get)]
    target_format: String,
    #[pyo3(get)]
    dispatch_parameters: Vec<CircuitDispatchParameter>,
    selection_plan: Vec<u8>,
    #[pyo3(get)]
    variants: Vec<ArtifactFamilyVariantProjection>,
}

/// Structural view of a canonical persistent launch-materializer payload.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct LaunchMaterializerPayloadProjection {
    #[pyo3(get)]
    target_format: String,
    program: Vec<u8>,
}

/// Transient native work charged while inspecting one canonical `QCore` envelope.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct QCorePersistenceWorkProjection {
    #[pyo3(get)]
    input_bytes: usize,
    #[pyo3(get)]
    preflight_message_nodes: usize,
    #[pyo3(get)]
    canonical_encode_passes: usize,
    #[pyo3(get)]
    canonical_encoded_bytes: usize,
}

/// Transient native work charged by decoded `QCore` semantic verification.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct QCoreVerificationWorkProjection {
    #[pyo3(get)]
    literal_nodes: u128,
    #[pyo3(get)]
    type_components: u128,
    #[pyo3(get)]
    operation_kernel_nodes: u128,
    #[pyo3(get)]
    attribute_nodes: u128,
    #[pyo3(get)]
    call_graph_steps: u128,
    #[pyo3(get)]
    declaration_nodes: u128,
    #[pyo3(get)]
    static_domain_steps: u128,
    #[pyo3(get)]
    equation_steps: u128,
    #[pyo3(get)]
    call_abi_steps: u128,
    #[pyo3(get)]
    phase_lifting_steps: u128,
    #[pyo3(get)]
    control_dependence_steps: u128,
    #[pyo3(get)]
    contract_derivation_steps: u128,
    #[pyo3(get)]
    functions: u128,
    #[pyo3(get)]
    blocks: u128,
    #[pyo3(get)]
    operations: u128,
    #[pyo3(get)]
    values: u128,
    #[pyo3(get)]
    operands: u128,
    #[pyo3(get)]
    topology_comparisons: u128,
    #[pyo3(get)]
    ownership_path_steps: u128,
    #[pyo3(get)]
    control_flow_steps: u128,
    #[pyo3(get)]
    verification_steps: u128,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl ArtifactFamilyVariantProjection {
    #[getter]
    fn artifact<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.artifact)
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl ArtifactFamilyPayloadProjection {
    #[getter]
    fn selection_plan<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.selection_plan)
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[pymethods]
impl LaunchMaterializerPayloadProjection {
    #[getter]
    fn program<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.program)
    }
}

/// Inspect a canonical `QCore` envelope before Python Protobuf decoding.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[gen_stub(override_return_type(type_repr = "tuple[int, int, str, bytes]"))]
fn inspect_qcore_envelope(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<(u32, u32, String, Vec<u8>)> {
    inspect_native_qcore_envelope(payload)
        .map(|header| {
            (
                header.schema_major,
                header.schema_minor,
                header.artifact_kind,
                header.package_digest.to_vec(),
            )
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Inspect one canonical `QCore` envelope and return transient native work.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_qcore_persistence_work(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<QCorePersistenceWorkProjection> {
    inspect_native_qcore_envelope_with_work(payload)
        .map(|(_, work)| QCorePersistenceWorkProjection {
            input_bytes: work.input_bytes(),
            preflight_message_nodes: work.preflight_message_nodes(),
            canonical_encode_passes: work.canonical_encode_passes(),
            canonical_encoded_bytes: work.canonical_encoded_bytes(),
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn project_qcore_verification_work(
    work: &qamomile_compiler::QCoreVerificationWork,
) -> QCoreVerificationWorkProjection {
    QCoreVerificationWorkProjection {
        literal_nodes: work.literal_node_visits(),
        type_components: work.type_component_visits(),
        operation_kernel_nodes: work.operation_kernel_visits(),
        attribute_nodes: work.attribute_node_visits(),
        call_graph_steps: work.call_graph_steps(),
        declaration_nodes: work.declaration_node_visits(),
        static_domain_steps: work.static_domain_steps(),
        equation_steps: work.equation_steps(),
        call_abi_steps: work.call_abi_steps(),
        phase_lifting_steps: work.phase_lifting_steps(),
        control_dependence_steps: work.control_dependence_steps(),
        contract_derivation_steps: work.contract_derivation_steps(),
        functions: work.function_visits(),
        blocks: work.block_visits(),
        operations: work.operation_visits(),
        values: work.value_visits(),
        operands: work.operand_visits(),
        topology_comparisons: work.topology_comparisons(),
        ownership_path_steps: work.ownership_path_steps(),
        control_flow_steps: work.control_flow_steps(),
        verification_steps: work.verification_steps(),
    }
}

/// Verify one canonical module and return transient function-local work.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_module_verification_work(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<QCoreVerificationWorkProjection> {
    inspect_native_module_local_verification_work(payload)
        .map(|work| project_qcore_verification_work(&work))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Verify one canonical program and return transient function-local work.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_program_verification_work(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<QCoreVerificationWorkProjection> {
    inspect_native_program_local_verification_work(payload)
        .map(|work| project_qcore_verification_work(&work))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Seal one canonical unsealed `QCorePackageProto` through Rust.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn seal_qcore_package<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    seal_native_qcore_package(payload)
        .map(|sealed| PyBytes::new(py, &sealed))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Inspect a canonical target artifact before Python Protobuf decoding.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[gen_stub(override_return_type(type_repr = "tuple[int, int, str, str, bytes]"))]
fn inspect_artifact(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<(u32, u32, String, String, Vec<u8>)> {
    inspect_native_artifact(payload)
        .map(|header| {
            (
                header.schema_major,
                header.schema_minor,
                header.format,
                header.target_fingerprint,
                header.artifact_digest.to_vec(),
            )
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Seal one canonical unsealed `ArtifactProto` through Rust.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn seal_artifact<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    seal_native_artifact(payload)
        .map(|sealed| PyBytes::new(py, &sealed))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Inspect a canonical execution snapshot before Python Protobuf decoding.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[gen_stub(override_return_type(type_repr = "tuple[bytes, int, int]"))]
fn inspect_execution_snapshot(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<(Vec<u8>, usize, usize)> {
    inspect_native_execution_snapshot(payload)
        .map(|header| {
            (
                header.snapshot_digest.to_vec(),
                header.nodes,
                header.references,
            )
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Seal one canonical unsealed `ExecutionSnapshotProto` through Rust.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn seal_execution_snapshot<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    seal_native_execution_snapshot(payload)
        .map(|sealed| PyBytes::new(py, &sealed))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Seal one canonical unsealed `HIRPackageProto` through Rust.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn seal_hir_package<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    seal_native_hir_package(payload)
        .map(|sealed| PyBytes::new(py, &sealed))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Derive the debug-erased semantic identity of one verified module.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn module_semantic_digest<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    native_module_semantic_digest(payload)
        .map(|digest| PyBytes::new(py, &digest))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Derive the debug-erased semantic identity of one verified program.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn program_semantic_digest<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes | VerifiedProgramHandle"))] payload: &Bound<
        '_,
        PyAny,
    >,
) -> PyResult<Bound<'py, PyBytes>> {
    Ok(PyBytes::new(
        py,
        &program_owner(payload)?.verified.semantic_digest(),
    ))
}

/// Resolve one local or fully qualified callable selector in a verified module.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn resolve_module_entrypoint(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
    selector: &str,
) -> PyResult<String> {
    resolve_native_module_entrypoint(payload, selector)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Derive one selected-family identity from a verified module and symbol.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn selected_module_semantic_digest<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
    symbol: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    native_selected_module_semantic_digest(payload, symbol)
        .map(|digest| PyBytes::new(py, &digest))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Inspect one integrity-checked persistent artifact-family selector.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_circuit_family_selector(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<CircuitFamilySelectionProjection> {
    let plan = inspect_native_circuit_family_selection(payload)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(CircuitFamilySelectionProjection {
        dispatch_parameters: plan
            .dispatch_parameters()
            .iter()
            .map(circuit_dispatch_parameter)
            .collect(),
        variants: plan
            .variants()
            .iter()
            .map(|variant| CircuitFamilySelectionVariant {
                decisions: variant
                    .decisions()
                    .iter()
                    .map(circuit_dispatch_decision)
                    .collect(),
            })
            .collect(),
    })
}

type ArtifactFamilyDecisionRow = (usize, bool);
type ArtifactFamilyVariantRow = (Vec<ArtifactFamilyDecisionRow>, Vec<u8>);

/// Seal one portable artifact-family payload through the native compiler.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn seal_artifact_family_payload<'py>(
    py: Python<'py>,
    target_format: &str,
    dispatch_parameters: Vec<(usize, String)>,
    #[gen_stub(override_type(type_repr = "bytes"))] selection_plan: &[u8],
    variants: Vec<ArtifactFamilyVariantRow>,
) -> PyResult<Bound<'py, PyBytes>> {
    let variants = variants
        .into_iter()
        .map(|(decisions, artifact)| ArtifactFamilyVariantInput::new(decisions, artifact))
        .collect::<Vec<_>>();
    let payload = seal_native_artifact_family_payload(
        target_format,
        &dispatch_parameters,
        selection_plan,
        &variants,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyBytes::new(py, &payload))
}

/// Inspect one canonical persistent artifact-family payload.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_artifact_family_payload(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<ArtifactFamilyPayloadProjection> {
    let family = inspect_native_artifact_family_payload(payload)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(artifact_family_projection(&family))
}

fn artifact_family_projection(
    family: &NativeArtifactFamilyPayload,
) -> ArtifactFamilyPayloadProjection {
    ArtifactFamilyPayloadProjection {
        target_format: family.target_format().to_owned(),
        dispatch_parameters: family
            .dispatch_parameters()
            .iter()
            .map(circuit_dispatch_parameter)
            .collect(),
        selection_plan: family.selection_plan().to_vec(),
        variants: family
            .variants()
            .iter()
            .map(|variant| ArtifactFamilyVariantProjection {
                decisions: variant
                    .decisions()
                    .iter()
                    .map(circuit_dispatch_decision)
                    .collect(),
                artifact: variant.artifact().to_vec(),
            })
            .collect(),
    }
}

/// Seal one portable launch-materializer payload through the native compiler.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn seal_launch_materializer_payload<'py>(
    py: Python<'py>,
    target_format: &str,
    #[gen_stub(override_type(type_repr = "bytes"))] program: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let payload = seal_native_launch_materializer_payload(target_format, program)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyBytes::new(py, &payload))
}

/// Inspect one canonical persistent launch-materializer payload.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_launch_materializer_payload(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<LaunchMaterializerPayloadProjection> {
    let materializer = inspect_native_launch_materializer_payload(payload)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(launch_materializer_projection(&materializer))
}

/// Validate one complete launch-materializer artifact and its embedded program.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn inspect_launch_materializer_artifact(
    #[gen_stub(override_type(type_repr = "bytes"))] artifact: &[u8],
    target_format: &str,
    materialized_requirements: Vec<String>,
) -> PyResult<LaunchMaterializerPayloadProjection> {
    let materializer = inspect_native_launch_materializer_artifact(
        artifact,
        target_format,
        &materialized_requirements,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(launch_materializer_projection(&materializer))
}

fn launch_materializer_projection(
    materializer: &NativeLaunchMaterializerPayload,
) -> LaunchMaterializerPayloadProjection {
    LaunchMaterializerPayloadProjection {
        target_format: materializer.target_format().to_owned(),
        program: materializer.program().to_vec(),
    }
}

/// Select one zero-based child index through native scalar semantics.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn select_circuit_family_variant(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
    bindings: &Bound<'_, PyDict>,
) -> PyResult<usize> {
    let bindings = circuit_launch_bindings(bindings)?;
    select_native_circuit_family_variant(payload, &bindings)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<QCorePersistenceWorkProjection>()?;
    module.add_class::<QCoreVerificationWorkProjection>()?;
    module.add_class::<ArtifactFamilyVariantProjection>()?;
    module.add_class::<ArtifactFamilyPayloadProjection>()?;
    module.add_class::<LaunchMaterializerPayloadProjection>()?;
    module.add_class::<CircuitFamilySelectionVariant>()?;
    module.add_class::<CircuitFamilySelectionProjection>()?;
    module.add_function(wrap_pyfunction!(inspect_qcore_envelope, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_qcore_persistence_work, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_module_verification_work, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_program_verification_work, module)?)?;
    module.add_function(wrap_pyfunction!(seal_qcore_package, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_artifact, module)?)?;
    module.add_function(wrap_pyfunction!(seal_artifact, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_execution_snapshot, module)?)?;
    module.add_function(wrap_pyfunction!(seal_execution_snapshot, module)?)?;
    module.add_function(wrap_pyfunction!(seal_hir_package, module)?)?;
    module.add_function(wrap_pyfunction!(module_semantic_digest, module)?)?;
    module.add_function(wrap_pyfunction!(program_semantic_digest, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_module_entrypoint, module)?)?;
    module.add_function(wrap_pyfunction!(selected_module_semantic_digest, module)?)?;
    module.add_function(wrap_pyfunction!(seal_artifact_family_payload, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_artifact_family_payload, module)?)?;
    module.add_function(wrap_pyfunction!(seal_launch_materializer_payload, module)?)?;
    module.add_function(wrap_pyfunction!(
        inspect_launch_materializer_payload,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        inspect_launch_materializer_artifact,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(inspect_circuit_family_selector, module)?)?;
    module.add_function(wrap_pyfunction!(select_circuit_family_variant, module)?)?;
    Ok(())
}
