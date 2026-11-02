//! Compiler projections and entry points for the native Python module.

use super::conversion::package_digest_hex;
use crate::program_owner::program_owner;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};
use qamomile_compiler::{
    CallableContract, LocallySpecializedProgram, VerifiedModule, VerifiedProgram,
    VerifiedSpecializationRequest, inspect_hir_image, inspect_hir_image_with_work,
    lower_adjoint_calls_for_target as lower_native_adjoint_calls_for_target, lower_hir_image,
    lower_hir_image_with_normalization_work, lower_hir_image_with_work,
};

/// Validated header of a private typed-HIR compiler request.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct HirHeader {
    #[pyo3(get)]
    module_name: String,
    #[pyo3(get)]
    function_count: usize,
    #[pyo3(get)]
    constant_count: usize,
    #[pyo3(get)]
    expression_count: usize,
    #[pyo3(get)]
    statement_count: usize,
}

/// Transient native work charged while validating one typed-HIR image.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct HirValidationWorkProjection {
    #[pyo3(get)]
    input_bytes: usize,
    #[pyo3(get)]
    preflight_message_nodes: usize,
    #[pyo3(get)]
    semantic_node_visits: u128,
}

/// Transient work charged by final typed-HIR ownership-SSA lowering.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct HirSsaLoweringWorkProjection {
    #[pyo3(get)]
    functions: u128,
    #[pyo3(get)]
    statements: u128,
    #[pyo3(get)]
    expressions: u128,
    #[pyo3(get)]
    source_nodes: u128,
}

/// Transient work charged by pre-SSA typed-HIR normalization.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct HirNormalizationWorkProjection {
    #[pyo3(get)]
    function_results: u128,
    #[pyo3(get)]
    return_continuations: u128,
    #[pyo3(get)]
    higher_order: u128,
    #[pyo3(get)]
    function_aliases: u128,
    #[pyo3(get)]
    source_nodes: u128,
}

/// One compiler-derived callable contract projected into Python.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct FunctionContractProjection {
    #[pyo3(get)]
    symbol: String,
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    effects: Vec<String>,
    #[pyo3(get)]
    capabilities: Vec<String>,
    #[pyo3(get)]
    requirements: Vec<String>,
}

/// Validated header of a canonical Qamomile Module.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct ModuleHeader {
    #[pyo3(get)]
    schema_major: u32,
    #[pyo3(get)]
    schema_minor: u32,
    #[pyo3(get)]
    module_name: String,
    #[pyo3(get)]
    function_count: usize,
    #[pyo3(get)]
    constant_count: usize,
    #[pyo3(get)]
    type_count: usize,
    #[pyo3(get)]
    operation_count: usize,
    #[pyo3(get)]
    package_digest: String,
    #[pyo3(get)]
    verified: bool,
    #[pyo3(get)]
    pending_cleanup_obligation_count: usize,
    #[pyo3(get)]
    contracts: Vec<FunctionContractProjection>,
}

/// Verified metadata for a selected request that is not yet executable.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct SpecializationHeader {
    #[pyo3(get)]
    module_name: String,
    #[pyo3(get)]
    entrypoint: String,
    #[pyo3(get)]
    function_count: usize,
    #[pyo3(get)]
    operation_count: usize,
    #[pyo3(get)]
    verified: bool,
    #[pyo3(get)]
    pending_cleanup_obligation_count: usize,
    #[pyo3(get)]
    contracts: Vec<FunctionContractProjection>,
}

/// Validated header of a canonical Qamomile Program.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct ProgramHeader {
    #[pyo3(get)]
    schema_major: u32,
    #[pyo3(get)]
    schema_minor: u32,
    #[pyo3(get)]
    module_name: String,
    #[pyo3(get)]
    entrypoint: String,
    #[pyo3(get)]
    function_count: usize,
    #[pyo3(get)]
    constant_count: usize,
    #[pyo3(get)]
    type_count: usize,
    #[pyo3(get)]
    operation_count: usize,
    #[pyo3(get)]
    package_digest: String,
    #[pyo3(get)]
    verified: bool,
    #[pyo3(get)]
    discharged_cleanup_count: usize,
    #[pyo3(get)]
    entry_kind: String,
    #[pyo3(get)]
    entry_effects: Vec<String>,
    #[pyo3(get)]
    entry_capabilities: Vec<String>,
    #[pyo3(get)]
    entry_requirements: Vec<String>,
    /// Source-formal indices requiring runtime control bindings; sealed Const
    /// formals are excluded by the compiler's launch-materialization contract.
    #[pyo3(get)]
    entry_control_parameters: Vec<usize>,
    #[pyo3(get)]
    contracts: Vec<FunctionContractProjection>,
}

fn project_contract(symbol: &str, contract: CallableContract) -> FunctionContractProjection {
    FunctionContractProjection {
        symbol: symbol.to_owned(),
        kind: contract.kind().as_str().to_owned(),
        effects: contract
            .effects()
            .iter()
            .map(|effect| effect.as_str().to_owned())
            .collect(),
        capabilities: contract
            .capabilities()
            .iter()
            .map(|capability| capability.as_str().to_owned())
            .collect(),
        requirements: contract
            .requirements()
            .iter()
            .map(|requirement| requirement.as_str().to_owned())
            .collect(),
    }
}

/// Native target compatibility result in stable semantic order.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct TargetCapabilityPlan {
    #[pyo3(get)]
    required: Vec<String>,
    #[pyo3(get)]
    supported: Vec<String>,
    #[pyo3(get)]
    missing: Vec<String>,
    #[pyo3(get)]
    compatible: bool,
}

/// Native transient lowering decision in stable semantic order.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct TargetLoweringPlanProjection {
    #[pyo3(get)]
    required: Vec<String>,
    #[pyo3(get)]
    supported: Vec<String>,
    #[pyo3(get)]
    missing: Vec<String>,
    #[pyo3(get)]
    materialized: Vec<String>,
    #[pyo3(get)]
    unsupported: Vec<String>,
    #[pyo3(get)]
    route: String,
}

/// Inspect an integrity-checked canonical `Module` payload.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_module(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<ModuleHeader> {
    let module = VerifiedModule::from_bytes(payload)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let header = module.header();
    Ok(ModuleHeader {
        schema_major: header.schema_major,
        schema_minor: header.schema_minor,
        module_name: header.module_name.clone(),
        function_count: header.function_count,
        constant_count: header.constant_count,
        type_count: header.type_count,
        operation_count: header.operation_count,
        package_digest: package_digest_hex(&header.package_digest),
        verified: true,
        pending_cleanup_obligation_count: module.pending_cleanup_obligation_count(),
        contracts: module
            .callable_contracts()
            .map(|(symbol, contract)| project_contract(symbol, contract))
            .collect(),
    })
}

/// Inspect one fully bound, non-executable specialization request.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_specialization(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<SpecializationHeader> {
    let request = VerifiedSpecializationRequest::from_bytes(payload)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let header = request.header();
    Ok(SpecializationHeader {
        module_name: header.module_name.clone(),
        entrypoint: header.entrypoint.clone(),
        function_count: header.function_count,
        operation_count: header.operation_count,
        verified: true,
        pending_cleanup_obligation_count: request.pending_cleanup_obligation_count(),
        contracts: request
            .callable_contracts()
            .map(|(symbol, contract)| project_contract(symbol, contract))
            .collect(),
    })
}

/// Inspect one locally rechecked target-adjoint artifact.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_target_adjoint(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<SpecializationHeader> {
    let target = qamomile_compiler::TargetLoweredProgram::from_bytes(payload)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let header = target.header();
    Ok(SpecializationHeader {
        module_name: header.module_name.clone(),
        entrypoint: header.entrypoint.clone(),
        function_count: header.function_count,
        operation_count: header.operation_count,
        verified: true,
        pending_cleanup_obligation_count: target.pending_cleanup_obligation_count(),
        contracts: target
            .callable_contracts()
            .map(|(symbol, contract)| project_contract(symbol, contract))
            .collect(),
    })
}

/// Inspect one integrity-checked private typed-HIR compiler request.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_hir(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<HirHeader> {
    let header =
        inspect_hir_image(payload).map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(HirHeader {
        module_name: header.module_name,
        function_count: header.function_count,
        constant_count: header.constant_count,
        expression_count: header.expression_count,
        statement_count: header.statement_count,
    })
}

/// Inspect one typed-HIR image and return transient native validation work.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_hir_validation_work(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<HirValidationWorkProjection> {
    inspect_hir_image_with_work(payload)
        .map(|(_, work)| HirValidationWorkProjection {
            input_bytes: work.input_bytes(),
            preflight_message_nodes: work.preflight_message_nodes(),
            semantic_node_visits: work.semantic_node_visits(),
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Lower typed HIR and return transient final ownership-SSA work.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_hir_ssa_lowering_work(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<HirSsaLoweringWorkProjection> {
    lower_hir_image_with_work(payload)
        .map(|(_, work)| HirSsaLoweringWorkProjection {
            functions: work.function_visits(),
            statements: work.statement_visits(),
            expressions: work.expression_visits(),
            source_nodes: work.source_node_visits(),
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Lower typed HIR and return transient pre-SSA normalization work.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_hir_normalization_work(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<HirNormalizationWorkProjection> {
    lower_hir_image_with_normalization_work(payload)
        .map(|(_, work)| HirNormalizationWorkProjection {
            function_results: work.function_result_visits(),
            return_continuations: work.return_continuation_visits(),
            higher_order: work.higher_order_visits(),
            function_aliases: work.function_alias_visits(),
            source_nodes: work.source_node_visits(),
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Lower the migrated typed-HIR subset into a verified `QCore` module.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn lower_hir<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let module =
        lower_hir_image(payload).map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyBytes::new(py, &module))
}

/// Specialize and synthesize cleanup through the native proof pipeline.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn prepare_program<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let program = LocallySpecializedProgram::prepare_request_bytes(payload)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyBytes::new(py, program.canonical_bytes()))
}

/// Return `QCore` operation visits made by native Const specialization.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_program_preparation_operation_visits(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<u128> {
    LocallySpecializedProgram::prepare_request_bytes_with_work(payload)
        .map(|(_, work)| work.operation_visits())
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Lower callable adjoints into one locally rechecked transient target artifact.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn lower_adjoint_calls_for_target<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let target = py
        .detach(|| lower_native_adjoint_calls_for_target(payload))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyBytes::new(py, target.canonical_bytes()))
}

/// Optimize `QCore` before lowering callable adjoints for direct targets.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn lower_optimized_adjoint_calls_for_target<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes | VerifiedProgramHandle"))] payload: &Bound<
        '_,
        PyAny,
    >,
) -> PyResult<Bound<'py, PyBytes>> {
    let owner = program_owner(payload)?;
    let target = py.detach(|| {
        owner
            .prepared()?
            .lower_adjoint_calls()
            .map_err(|error| PyValueError::new_err(error.to_string()))
    })?;
    Ok(PyBytes::new(py, target.canonical_bytes()))
}

/// Return a fully reverified transient image after standard `QCore` optimization.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn optimize_standard_program<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes | VerifiedProgramHandle"))] payload: &Bound<
        '_,
        PyAny,
    >,
) -> PyResult<Bound<'py, PyBytes>> {
    let owner = program_owner(payload)?;
    let prepared = py.detach(|| owner.prepared())?;
    Ok(PyBytes::new(py, prepared.canonical_bytes()))
}

/// Return bounded operation visits made by standard `QCore` optimization.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_standard_optimization_operation_visits(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<u128> {
    let program = VerifiedProgram::from_bytes(payload)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    program
        .optimize_standard()
        .map(|(_, work)| work.operation_visits())
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Inspect an integrity-checked canonical `Program` payload.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_program(
    #[gen_stub(override_type(type_repr = "bytes | VerifiedProgramHandle"))] payload: &Bound<
        '_,
        PyAny,
    >,
) -> PyResult<ProgramHeader> {
    let owner = program_owner(payload)?;
    let program = &owner.verified;
    let header = program.header();
    let contract = program
        .callable_contract(&header.entrypoint)
        .ok_or_else(|| PyValueError::new_err("Q4002: missing verified entrypoint contract"))?;
    let entry_control_parameters = program
        .entry_unbound_control_parameters()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(ProgramHeader {
        schema_major: header.schema_major,
        schema_minor: header.schema_minor,
        module_name: header.module_name.clone(),
        entrypoint: header.entrypoint.clone(),
        function_count: header.function_count,
        constant_count: header.constant_count,
        type_count: header.type_count,
        operation_count: header.operation_count,
        package_digest: package_digest_hex(&header.package_digest),
        verified: true,
        discharged_cleanup_count: program.discharged_cleanup_count(),
        entry_kind: contract.kind().as_str().to_owned(),
        entry_effects: contract
            .effects()
            .iter()
            .map(|effect| effect.as_str().to_owned())
            .collect(),
        entry_capabilities: contract
            .capabilities()
            .iter()
            .map(|capability| capability.as_str().to_owned())
            .collect(),
        entry_requirements: contract
            .requirements()
            .iter()
            .map(|requirement| requirement.as_str().to_owned())
            .collect(),
        entry_control_parameters,
        contracts: program
            .callable_contracts()
            .map(|(symbol, contract)| project_contract(symbol, contract))
            .collect(),
    })
}

/// Verify a target capability profile against one canonical program.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn plan_target(
    #[gen_stub(override_type(type_repr = "bytes | VerifiedProgramHandle"))] payload: &Bound<
        '_,
        PyAny,
    >,
    supported: Vec<String>,
) -> PyResult<TargetCapabilityPlan> {
    let owner = program_owner(payload)?;
    let program = &owner.verified;
    let plan = program
        .plan_target(&supported)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(TargetCapabilityPlan {
        required: plan
            .required()
            .iter()
            .map(|requirement| requirement.as_str().to_owned())
            .collect(),
        supported: plan
            .supported()
            .iter()
            .map(|requirement| requirement.as_str().to_owned())
            .collect(),
        missing: plan
            .missing()
            .iter()
            .map(|requirement| requirement.as_str().to_owned())
            .collect(),
        compatible: plan.is_compatible(),
    })
}

/// Select direct, family, materialized, or unsupported target lowering.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn plan_target_lowering(
    #[gen_stub(override_type(type_repr = "bytes | VerifiedProgramHandle"))] payload: &Bound<
        '_,
        PyAny,
    >,
    supported: Vec<String>,
    materialized: Vec<String>,
    circuit_family: bool,
) -> PyResult<TargetLoweringPlanProjection> {
    let owner = program_owner(payload)?;
    let program = &owner.verified;
    let plan = program
        .plan_target_lowering(&supported, &materialized, circuit_family)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let capabilities = plan.capabilities();
    let labels = |requirements: qamomile_compiler::RequirementSet| {
        requirements
            .iter()
            .map(|requirement| requirement.as_str().to_owned())
            .collect()
    };
    Ok(TargetLoweringPlanProjection {
        required: labels(capabilities.required()),
        supported: labels(capabilities.supported()),
        missing: labels(capabilities.missing()),
        materialized: labels(plan.materialized()),
        unsupported: labels(plan.unsupported()),
        route: plan.route().as_str().to_owned(),
    })
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<HirHeader>()?;
    module.add_class::<HirValidationWorkProjection>()?;
    module.add_class::<HirNormalizationWorkProjection>()?;
    module.add_class::<HirSsaLoweringWorkProjection>()?;
    module.add_class::<FunctionContractProjection>()?;
    module.add_class::<ModuleHeader>()?;
    module.add_class::<SpecializationHeader>()?;
    module.add_class::<ProgramHeader>()?;
    module.add_class::<TargetCapabilityPlan>()?;
    module.add_class::<TargetLoweringPlanProjection>()?;
    module.add_function(wrap_pyfunction!(inspect_hir, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_hir_validation_work, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_hir_normalization_work, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_hir_ssa_lowering_work, module)?)?;
    module.add_function(wrap_pyfunction!(lower_hir, module)?)?;
    module.add_function(wrap_pyfunction!(
        inspect_program_preparation_operation_visits,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(inspect_module, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_specialization, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_target_adjoint, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_program, module)?)?;
    module.add_function(wrap_pyfunction!(lower_adjoint_calls_for_target, module)?)?;
    module.add_function(wrap_pyfunction!(
        lower_optimized_adjoint_calls_for_target,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(optimize_standard_program, module)?)?;
    module.add_function(wrap_pyfunction!(
        inspect_standard_optimization_operation_visits,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(inspect_program, module)?)?;
    module.add_function(wrap_pyfunction!(plan_target, module)?)?;
    module.add_function(wrap_pyfunction!(plan_target_lowering, module)?)?;
    Ok(())
}
