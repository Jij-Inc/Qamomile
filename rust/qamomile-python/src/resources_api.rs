//! Resources projections and entry points for the native Python module.

use super::abi_api::FourPythonBytes;
use crate::program_owner::program_owner;
use pyo3::exceptions::{PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};
use qamomile_compiler::{
    IterationAssumption, ResourceExpression as NativeResourceExpression,
    ResourceExpressionEvaluation as NativeResourceExpressionEvaluation,
    ResourceExpressionEvaluationError, ResourceExpressionSubstitutionError,
    ResourceHotspotCandidate as NativeResourceHotspotCandidate,
    ResourceScalar as NativeResourceScalar, SymbolicCount as NativeSymbolicCount, VerifiedProgram,
    VerifiedProgramFamily, VerifiedResourceAnalysisRequest,
    assume_resource_iterations as assume_native_resource_iterations,
    construct_resource_expression as construct_native_resource_expression,
    construct_resource_literal as construct_native_resource_literal,
    construct_resource_predicate as construct_native_resource_predicate,
    construct_resource_symbol as construct_native_resource_symbol,
    evaluate_resource_expression as evaluate_native_resource_expression,
    free_resource_expression_symbols as free_native_resource_expression_symbols,
    inspect_resource_study as inspect_native_resource_study,
    normalize_resource_expression as normalize_native_resource_expression,
    project_surface_code_resources as project_native_surface_code_resources,
    rank_resource_hotspots as rank_native_resource_hotspots,
    resolve_resource_symbol_bindings as resolve_native_resource_symbol_bindings,
    retag_resource_expression_unit as retag_native_resource_expression_unit,
    seal_resource_analysis_request as seal_native_resource_analysis_request,
    seal_resource_study as seal_native_resource_study,
    substitute_resource_expression as substitute_native_resource_expression,
    validate_resource_unit as validate_native_resource_unit,
    validate_resource_value as validate_native_resource_value,
};
use std::collections::BTreeMap;

/// Native lexical inventory for one verified Program.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct ProgramStructure {
    #[pyo3(get)]
    entrypoint: String,
    #[pyo3(get)]
    function_count: usize,
    #[pyo3(get)]
    block_count: usize,
    #[pyo3(get)]
    region_count: usize,
    #[pyo3(get)]
    operation_count: usize,
    #[pyo3(get)]
    call_site_count: usize,
    #[pyo3(get)]
    opcodes: BTreeMap<String, usize>,
    #[pyo3(get)]
    function_blocks: BTreeMap<String, usize>,
    #[pyo3(get)]
    function_regions: BTreeMap<String, usize>,
    #[pyo3(get)]
    function_operations: BTreeMap<String, usize>,
    #[pyo3(get)]
    function_call_sites: BTreeMap<String, usize>,
    #[pyo3(get)]
    function_opcodes: BTreeMap<String, BTreeMap<String, usize>>,
}

/// Exact dynamic quantum-operation counts for the native static subset.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct LogicalCounts {
    #[pyo3(get)]
    analysis_operation_visits: u128,
    #[pyo3(get)]
    work: u128,
    #[pyo3(get)]
    opcodes: BTreeMap<String, u128>,
    #[pyo3(get)]
    families: Option<BTreeMap<String, u128>>,
    #[pyo3(get)]
    measurements: Option<u128>,
    #[pyo3(get)]
    resets: Option<u128>,
    #[pyo3(get)]
    input_qubits: Option<u128>,
    #[pyo3(get)]
    allocated_qubits: Option<u128>,
    #[pyo3(get)]
    circuit_qubits: Option<u128>,
    #[pyo3(get)]
    peak_qubits: Option<u128>,
    #[pyo3(get)]
    depths: Option<BTreeMap<String, u128>>,
    #[pyo3(get)]
    cleanup_work: u128,
}

/// One node from the native symbolic resource-expression DAG.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct SymbolicExpressionNode {
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    domain: String,
    #[pyo3(get)]
    arguments: Vec<u32>,
    #[pyo3(get)]
    predicates: Vec<u32>,
    #[pyo3(get)]
    literal: Option<String>,
    #[pyo3(get)]
    literal_kind: Option<String>,
    #[pyo3(get)]
    rational_numerator: Option<String>,
    #[pyo3(get)]
    rational_denominator: Option<String>,
    #[pyo3(get)]
    real_bits: Option<u64>,
    #[pyo3(get)]
    symbol_scope: Option<String>,
    #[pyo3(get)]
    symbol_name: Option<String>,
    #[pyo3(get)]
    symbol_domain: Option<String>,
    #[pyo3(get)]
    symbol_origin: Option<String>,
    #[pyo3(get)]
    symbol_constraints: Vec<String>,
    #[pyo3(get)]
    function_name: Option<String>,
    #[pyo3(get)]
    binder: Option<u32>,
}

/// Result of native standalone resource-expression evaluation.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct ResourceExpressionEvaluation {
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    integer: Option<String>,
    #[pyo3(get)]
    rational_numerator: Option<String>,
    #[pyo3(get)]
    rational_denominator: Option<String>,
    #[pyo3(get)]
    real_bits: Option<u64>,
    #[pyo3(get)]
    boolean: Option<bool>,
}

/// One lexically free resource-symbol declaration.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct ResourceExpressionSymbolProjection {
    #[pyo3(get)]
    scope: String,
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    domain: String,
    #[pyo3(get)]
    unit: String,
    #[pyo3(get)]
    origin: String,
    #[pyo3(get)]
    constraints: Vec<String>,
}

/// One lower and optional upper symbolic count endpoint.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct SymbolicCountProjection {
    #[pyo3(get)]
    lower: u32,
    #[pyo3(get)]
    upper: Option<u32>,
    #[pyo3(get)]
    exact: bool,
    #[pyo3(get)]
    unknown: Vec<String>,
    #[pyo3(get)]
    rule_identities: Vec<String>,
    #[pyo3(get)]
    relation: String,
    #[pyo3(get)]
    dependency_span_relation: String,
}

/// One opcode and its symbolic count enclosure.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct OpcodeSymbolicCount {
    #[pyo3(get)]
    opcode: String,
    #[pyo3(get)]
    count: SymbolicCountProjection,
}

/// One logical gate family and its symbolic count enclosure.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct FamilySymbolicCount {
    #[pyo3(get)]
    family: String,
    #[pyo3(get)]
    count: SymbolicCountProjection,
}

/// One declarative metric and its symbolic enclosure.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct MetricSymbolicCount {
    #[pyo3(get)]
    identifier: String,
    #[pyo3(get)]
    count: SymbolicCountProjection,
}

/// One canonical source-local contribution from native symbolic analysis.
#[derive(Clone)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct SymbolicAnalysisFactProjection {
    #[pyo3(get)]
    owner: String,
    #[pyo3(get)]
    source: String,
    #[pyo3(get)]
    opcode: String,
    #[pyo3(get)]
    work: SymbolicCountProjection,
    #[pyo3(get)]
    opcodes: Vec<OpcodeSymbolicCount>,
    #[pyo3(get)]
    extra_metrics: Vec<MetricSymbolicCount>,
    #[pyo3(get)]
    families: Option<Vec<FamilySymbolicCount>>,
    #[pyo3(get)]
    measurements: Option<SymbolicCountProjection>,
    #[pyo3(get)]
    resets: Option<SymbolicCountProjection>,
    #[pyo3(get)]
    cleanup_work: SymbolicCountProjection,
}

/// Closed native symbolic work and opcode counts.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct SymbolicLogicalCountsProjection {
    #[pyo3(get)]
    analysis_operation_visits: u128,
    #[pyo3(get)]
    expressions: Vec<SymbolicExpressionNode>,
    #[pyo3(get)]
    unknown: Vec<String>,
    #[pyo3(get)]
    facts: Vec<SymbolicAnalysisFactProjection>,
    #[pyo3(get)]
    work: SymbolicCountProjection,
    #[pyo3(get)]
    dependency_span: SymbolicCountProjection,
    #[pyo3(get)]
    opcodes: Vec<OpcodeSymbolicCount>,
    #[pyo3(get)]
    extra_metrics: Vec<MetricSymbolicCount>,
    #[pyo3(get)]
    families: Option<Vec<FamilySymbolicCount>>,
    #[pyo3(get)]
    measurements: Option<SymbolicCountProjection>,
    #[pyo3(get)]
    resets: Option<SymbolicCountProjection>,
    #[pyo3(get)]
    input_qubits: Option<SymbolicCountProjection>,
    #[pyo3(get)]
    allocated_qubits: Option<SymbolicCountProjection>,
    #[pyo3(get)]
    circuit_qubits: Option<SymbolicCountProjection>,
    #[pyo3(get)]
    peak_qubits: Option<SymbolicCountProjection>,
    #[pyo3(get)]
    depths: Option<Vec<FamilySymbolicCount>>,
    #[pyo3(get)]
    cleanup_work: SymbolicCountProjection,
}

/// Return structural facts from an independently verified Program.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_structure(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<ProgramStructure> {
    let program = VerifiedProgram::from_bytes(payload)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let census = program.structural_census();
    let functions = census.functions();
    Ok(ProgramStructure {
        entrypoint: census.entrypoint().to_owned(),
        function_count: functions.len(),
        block_count: census.block_count(),
        region_count: census.region_count(),
        operation_count: census.operation_count(),
        call_site_count: census.call_site_count(),
        opcodes: census.opcodes().clone(),
        function_blocks: functions
            .iter()
            .map(|function| (function.symbol().to_owned(), function.block_count()))
            .collect(),
        function_regions: functions
            .iter()
            .map(|function| (function.symbol().to_owned(), function.region_count()))
            .collect(),
        function_operations: functions
            .iter()
            .map(|function| (function.symbol().to_owned(), function.operation_count()))
            .collect(),
        function_call_sites: functions
            .iter()
            .map(|function| (function.symbol().to_owned(), function.call_site_count()))
            .collect(),
        function_opcodes: functions
            .iter()
            .map(|function| (function.symbol().to_owned(), function.opcodes().clone()))
            .collect(),
    })
}

/// Derive exact body-composed logical counts or fail closed on dynamic control.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_exact_logical_counts(
    #[gen_stub(override_type(type_repr = "bytes | VerifiedProgramHandle"))] payload: &Bound<
        '_,
        PyAny,
    >,
) -> PyResult<LogicalCounts> {
    let owner = program_owner(payload)?;
    let program = &owner.verified;
    let counts = program
        .exact_logical_counts()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(LogicalCounts {
        analysis_operation_visits: counts.analysis_operation_visits(),
        work: counts.work(),
        opcodes: counts.opcodes().clone(),
        families: counts.families().cloned(),
        measurements: counts.measurements(),
        resets: counts.resets(),
        input_qubits: counts.input_qubits(),
        allocated_qubits: counts.allocated_qubits(),
        circuit_qubits: counts.circuit_qubits(),
        peak_qubits: counts.peak_qubits(),
        depths: counts.depths().cloned(),
        cleanup_work: counts.cleanup_work(),
    })
}

/// Derive a closed symbolic logical-count enclosure.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_symbolic_logical_counts(
    #[gen_stub(override_type(type_repr = "bytes | VerifiedProgramHandle"))] payload: &Bound<
        '_,
        PyAny,
    >,
) -> PyResult<SymbolicLogicalCountsProjection> {
    let owner = program_owner(payload)?;
    let program = &owner.verified;
    let counts = program
        .symbolic_logical_counts()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(symbolic_logical_counts_projection(&counts))
}

/// Evaluate one canonical standalone resource-expression graph in Rust.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn evaluate_resource_expression(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<ResourceExpressionEvaluation> {
    let result = evaluate_native_resource_expression(payload).map_err(|error| match error {
        ResourceExpressionEvaluationError::DivisionByZero(_) => {
            PyZeroDivisionError::new_err(error.to_string())
        }
        ResourceExpressionEvaluationError::Invalid(_)
        | ResourceExpressionEvaluationError::Domain(_)
        | ResourceExpressionEvaluationError::Budget(_) => PyValueError::new_err(error.to_string()),
    })?;
    let projection = match result {
        NativeResourceExpressionEvaluation::Numeric(NativeResourceScalar::Integer(value)) => {
            ResourceExpressionEvaluation {
                kind: "integer".to_owned(),
                integer: Some(value.to_string()),
                rational_numerator: None,
                rational_denominator: None,
                real_bits: None,
                boolean: None,
            }
        }
        NativeResourceExpressionEvaluation::Numeric(NativeResourceScalar::Rational {
            numerator,
            denominator,
        }) => ResourceExpressionEvaluation {
            kind: "rational".to_owned(),
            integer: None,
            rational_numerator: Some(numerator.to_string()),
            rational_denominator: Some(denominator.to_string()),
            real_bits: None,
            boolean: None,
        },
        NativeResourceExpressionEvaluation::Numeric(NativeResourceScalar::Real(value)) => {
            ResourceExpressionEvaluation {
                kind: "real".to_owned(),
                integer: None,
                rational_numerator: None,
                rational_denominator: None,
                real_bits: Some(value.to_bits()),
                boolean: None,
            }
        }
        NativeResourceExpressionEvaluation::Boolean(value) => ResourceExpressionEvaluation {
            kind: "boolean".to_owned(),
            integer: None,
            rational_numerator: None,
            rational_denominator: None,
            real_bits: None,
            boolean: Some(value),
        },
        NativeResourceExpressionEvaluation::Unknown => ResourceExpressionEvaluation {
            kind: "unknown".to_owned(),
            integer: None,
            rational_numerator: None,
            rational_denominator: None,
            real_bits: None,
            boolean: None,
        },
    };
    Ok(projection)
}

/// Return the lexically free symbols in one resource-expression graph.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_resource_expression_symbols(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<Vec<ResourceExpressionSymbolProjection>> {
    free_native_resource_expression_symbols(payload)
        .map(|symbols| {
            symbols
                .into_iter()
                .map(|symbol| ResourceExpressionSymbolProjection {
                    scope: symbol.scope().to_owned(),
                    name: symbol.name().to_owned(),
                    domain: symbol.domain().as_str().to_owned(),
                    unit: symbol.unit().to_owned(),
                    origin: symbol.origin().to_owned(),
                    constraints: symbol.constraints().to_vec(),
                })
                .collect()
        })
        .map_err(|error| match error {
            ResourceExpressionSubstitutionError::Invalid(_)
            | ResourceExpressionSubstitutionError::Budget(_) => {
                PyValueError::new_err(error.to_string())
            }
        })
}

/// Resolve public resource-substitution names across multiple expression roots.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts Python sequences into owned vectors.
fn resolve_resource_symbol_bindings(
    payloads: Vec<Vec<u8>>,
    supplied_names: Vec<String>,
) -> PyResult<Vec<(String, String)>> {
    resolve_native_resource_symbol_bindings(&payloads, &supplied_names)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Rank resource-study contributions by finite upper endpoint.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts transient rows into owned values.
fn rank_resource_hotspots(
    candidates: Vec<(Option<Vec<u8>>, String, String)>,
    limit: usize,
) -> PyResult<Vec<usize>> {
    let candidates = candidates
        .into_iter()
        .map(|(upper, owner, source)| NativeResourceHotspotCandidate::new(upper, owner, source))
        .collect::<Vec<_>>();
    rank_native_resource_hotspots(&candidates, limit)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Substitute qualified symbols in one standalone resource-expression graph.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn substitute_resource_expression<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
    bindings: Vec<(String, Vec<u8>, bool)>,
) -> PyResult<Bound<'py, PyBytes>> {
    let result =
        substitute_native_resource_expression(payload, bindings).map_err(|error| match error {
            ResourceExpressionSubstitutionError::Invalid(_)
            | ResourceExpressionSubstitutionError::Budget(_) => {
                PyValueError::new_err(error.to_string())
            }
        })?;
    Ok(PyBytes::new(py, &result))
}

/// Apply scenario values or asserted upper bounds to while iteration symbols.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts owned binding rows.
fn assume_resource_iterations<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
    bindings: Vec<(String, String)>,
    upper_bound: bool,
) -> PyResult<Bound<'py, PyBytes>> {
    let mode = if upper_bound {
        IterationAssumption::UpperBound
    } else {
        IterationAssumption::Scenario
    };
    let result = assume_native_resource_iterations(payload, &bindings, mode)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyBytes::new(py, &result))
}

/// Normalize one standalone resource-expression graph in Rust.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn normalize_resource_expression<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let result = normalize_native_resource_expression(payload).map_err(|error| match error {
        ResourceExpressionSubstitutionError::Invalid(_)
        | ResourceExpressionSubstitutionError::Budget(_) => {
            PyValueError::new_err(error.to_string())
        }
    })?;
    Ok(PyBytes::new(py, &result))
}

/// Construct one typed resource expression in Rust.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts Python graph sequences into owned vectors.
fn construct_resource_expression<'py>(
    py: Python<'py>,
    kind: &str,
    arguments: Vec<Vec<u8>>,
    predicates: Vec<Vec<u8>>,
    attributes: Vec<(String, String)>,
    binder_unit: Option<&str>,
    result_contract: Option<(i32, String)>,
) -> PyResult<Bound<'py, PyBytes>> {
    let (result_domain, result_unit) = result_contract
        .as_ref()
        .map_or((None, None), |(domain, unit)| {
            (Some(*domain), Some(unit.as_str()))
        });
    let result = construct_native_resource_expression(
        kind,
        &arguments,
        &predicates,
        &attributes,
        binder_unit,
        result_domain,
        result_unit,
    )
    .map_err(|error| match error {
        ResourceExpressionSubstitutionError::Invalid(_)
        | ResourceExpressionSubstitutionError::Budget(_) => {
            PyValueError::new_err(error.to_string())
        }
    })?;
    Ok(PyBytes::new(py, &result))
}

/// Construct one typed resource literal in Rust.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn construct_resource_literal<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
    unit: &str,
    domain: Option<i32>,
) -> PyResult<Bound<'py, PyBytes>> {
    let result =
        construct_native_resource_literal(payload, unit, domain).map_err(|error| match error {
            ResourceExpressionSubstitutionError::Invalid(_)
            | ResourceExpressionSubstitutionError::Budget(_) => {
                PyValueError::new_err(error.to_string())
            }
        })?;
    Ok(PyBytes::new(py, &result))
}

/// Validate one nominal resource-unit identity in Rust.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn validate_resource_unit(unit: &str) -> PyResult<()> {
    validate_native_resource_unit(unit).map_err(|error| match error {
        ResourceExpressionSubstitutionError::Invalid(_)
        | ResourceExpressionSubstitutionError::Budget(_) => {
            PyValueError::new_err(error.to_string())
        }
    })
}

/// Construct one scoped resource-symbol expression in Rust.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts Python sequences into owned vectors.
fn construct_resource_symbol<'py>(
    py: Python<'py>,
    scope: &str,
    name: &str,
    domain: i32,
    unit: &str,
    origin: &str,
    constraints: Vec<String>,
) -> PyResult<Bound<'py, PyBytes>> {
    let result = construct_native_resource_symbol(scope, name, domain, unit, origin, &constraints)
        .map_err(|error| match error {
            ResourceExpressionSubstitutionError::Invalid(_)
            | ResourceExpressionSubstitutionError::Budget(_) => {
                PyValueError::new_err(error.to_string())
            }
        })?;
    Ok(PyBytes::new(py, &result))
}

/// Construct one typed resource predicate in Rust.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts Python graph sequences into owned vectors.
fn construct_resource_predicate<'py>(
    py: Python<'py>,
    kind: &str,
    arguments: Vec<Vec<u8>>,
    predicates: Vec<Vec<u8>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let result =
        construct_native_resource_predicate(kind, &arguments, &predicates).map_err(|error| {
            match error {
                ResourceExpressionSubstitutionError::Invalid(_)
                | ResourceExpressionSubstitutionError::Budget(_) => {
                    PyValueError::new_err(error.to_string())
                }
            }
        })?;
    Ok(PyBytes::new(py, &result))
}

/// Attach one nominal unit to a standalone resource-expression root in Rust.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn retag_resource_expression_unit<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
    unit: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let result =
        retag_native_resource_expression_unit(payload, unit).map_err(|error| match error {
            ResourceExpressionSubstitutionError::Invalid(_)
            | ResourceExpressionSubstitutionError::Budget(_) => {
                PyValueError::new_err(error.to_string())
            }
        })?;
    Ok(PyBytes::new(py, &result))
}

/// Validate one complete public resource-value relation in Rust.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[pyo3(signature = (metric_domain, unit, relation, expression=None, lower=None, upper=None, unknown_contributors=0))]
fn validate_resource_value(
    metric_domain: &str,
    unit: &str,
    relation: &str,
    #[gen_stub(override_type(type_repr = "bytes | None"))] expression: Option<&[u8]>,
    #[gen_stub(override_type(type_repr = "bytes | None"))] lower: Option<&[u8]>,
    #[gen_stub(override_type(type_repr = "bytes | None"))] upper: Option<&[u8]>,
    unknown_contributors: usize,
) -> PyResult<()> {
    validate_native_resource_value(
        metric_domain,
        unit,
        relation,
        expression,
        lower,
        upper,
        unknown_contributors,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Project two logical resource expressions through the native surface-code model.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn project_surface_code_resources<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] logical_qubits: &[u8],
    #[gen_stub(override_type(type_repr = "bytes"))] non_clifford_gates: &[u8],
    alpha: f64,
    threshold: f64,
    physical_error_rate: f64,
    syndrome_cycle_seconds: f64,
) -> PyResult<FourPythonBytes<'py>> {
    let projection = project_native_surface_code_resources(
        logical_qubits,
        non_clifford_gates,
        alpha,
        threshold,
        physical_error_rate,
        syndrome_cycle_seconds,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        PyBytes::new(py, projection.code_distance()),
        PyBytes::new(py, projection.physical_qubits()),
        PyBytes::new(py, projection.duration()),
        PyBytes::new(py, projection.qubit_time()),
    ))
}

/// Derive symbolic logical counts for a selected open program family.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_symbolic_program_family_counts(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<SymbolicLogicalCountsProjection> {
    let family = VerifiedProgramFamily::from_bytes(payload)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let counts = family
        .symbolic_logical_counts()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(symbolic_logical_counts_projection(&counts))
}

/// Derive symbolic logical counts with canonical declarative external rules.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_symbolic_resource_analysis_request(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<SymbolicLogicalCountsProjection> {
    let request = VerifiedResourceAnalysisRequest::from_bytes(payload)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let counts = request
        .symbolic_logical_counts()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(symbolic_logical_counts_projection(&counts))
}

fn portable_basis(name: &str) -> PyResult<qamomile_compiler::GateBasis> {
    match name {
        "clifford_toffoli_rotations" => Ok(qamomile_compiler::GateBasis::CliffordToffoliRotations),
        "clifford_t_rotations" => Ok(qamomile_compiler::GateBasis::CliffordTRotations),
        _ => Err(PyValueError::new_err(
            "Q4260: unknown portable resource basis",
        )),
    }
}

/// Observe calls or price an explicit portable alphabet without circuit expansion.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[pyo3(signature = (payload, expanded=None, basis=None, realizations=Vec::new()))]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts the optional labels by value.
fn inspect_symbolic_resource_observation(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
    expanded: Option<Vec<String>>,
    basis: Option<&str>,
    realizations: Vec<String>,
) -> PyResult<SymbolicLogicalCountsProjection> {
    let expanded = expanded.map(Vec::into_boxed_slice);
    let request = VerifiedResourceAnalysisRequest::from_bytes(payload)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let counts = request
        .observed_counts(
            expanded.as_deref(),
            basis.map(portable_basis).transpose()?,
            &realizations,
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(symbolic_logical_counts_projection(&counts))
}

/// Materialize the bounded ordered recipe used by portable resource analysis.
type PortableGateStepWire = (String, Vec<u8>, String, i32, u32);

/// Materialize the bounded ordered recipe used by portable resource analysis.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_portable_gate_recipe(
    opcode: &str,
    controls: u32,
    basis: &str,
) -> PyResult<Option<Vec<PortableGateStepWire>>> {
    use qamomile_compiler::RecipeAngle;
    Ok(
        qamomile_compiler::portable_gate_recipe(opcode, controls, portable_basis(basis)?).map(
            |steps| {
                steps
                    .into_iter()
                    .map(|step| {
                        let (kind, numerator, denominator) = match step.angle() {
                            RecipeAngle::None => ("none", 0, 1),
                            RecipeAngle::Input(numerator, denominator) => {
                                ("input", numerator, denominator)
                            }
                            RecipeAngle::Pi(numerator, denominator) => {
                                ("pi", numerator, denominator)
                            }
                        };
                        (
                            step.opcode().to_owned(),
                            step.lanes().to_vec(),
                            kind.to_owned(),
                            numerator,
                            denominator,
                        )
                    })
                    .collect()
            },
        ),
    )
}

/// Seal one canonical unsealed symbolic resource-analysis request through Rust.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn seal_resource_analysis_request<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    seal_native_resource_analysis_request(payload)
        .map(|sealed| PyBytes::new(py, &sealed))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Seal one canonical unsealed resource study through Rust.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn seal_resource_study<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    seal_native_resource_study(payload)
        .map(|sealed| PyBytes::new(py, &sealed))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Verify one sealed resource study and return its stable digest.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_resource_study<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let header = inspect_native_resource_study(payload)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyBytes::new(py, header.study_digest()))
}

fn symbolic_logical_counts_projection(
    counts: &qamomile_compiler::SymbolicLogicalCounts,
) -> SymbolicLogicalCountsProjection {
    let expressions = counts.expressions();
    let project_count = |count| symbolic_count(count, expressions);
    SymbolicLogicalCountsProjection {
        analysis_operation_visits: counts.analysis_operation_visits(),
        expressions: counts
            .expressions()
            .iter()
            .map(symbolic_expression_node)
            .collect(),
        unknown: counts.unknown_contributors(),
        facts: counts
            .facts()
            .iter()
            .map(|fact| SymbolicAnalysisFactProjection {
                owner: fact.owner().to_owned(),
                source: fact.source().to_owned(),
                opcode: fact.opcode().to_owned(),
                work: project_count(fact.work()),
                opcodes: fact
                    .opcodes()
                    .iter()
                    .map(|(opcode, count)| OpcodeSymbolicCount {
                        opcode: opcode.clone(),
                        count: project_count(count),
                    })
                    .collect(),
                extra_metrics: fact
                    .extra_metrics()
                    .iter()
                    .map(|(identifier, count)| MetricSymbolicCount {
                        identifier: identifier.clone(),
                        count: project_count(count),
                    })
                    .collect(),
                families: fact.families().map(|families| {
                    families
                        .iter()
                        .map(|(family, count)| FamilySymbolicCount {
                            family: family.clone(),
                            count: project_count(count),
                        })
                        .collect()
                }),
                measurements: fact.measurements().map(project_count),
                resets: fact.resets().map(project_count),
                cleanup_work: project_count(fact.cleanup_work()),
            })
            .collect(),
        work: project_count(counts.work()),
        dependency_span: project_count(counts.dependency_span()),
        opcodes: counts
            .opcodes()
            .iter()
            .map(|(opcode, count)| OpcodeSymbolicCount {
                opcode: opcode.clone(),
                count: project_count(count),
            })
            .collect(),
        extra_metrics: counts
            .extra_metrics()
            .iter()
            .map(|(identifier, count)| MetricSymbolicCount {
                identifier: identifier.clone(),
                count: project_count(count),
            })
            .collect(),
        families: counts.families().map(|families| {
            families
                .iter()
                .map(|(family, count)| FamilySymbolicCount {
                    family: family.clone(),
                    count: project_count(count),
                })
                .collect()
        }),
        measurements: counts.measurements().map(project_count),
        resets: counts.resets().map(project_count),
        input_qubits: counts.input_qubits().map(project_count),
        allocated_qubits: counts.allocated_qubits().map(project_count),
        circuit_qubits: counts.circuit_qubits().map(project_count),
        peak_qubits: counts.peak_qubits().map(project_count),
        depths: counts.depths().map(|depths| {
            depths
                .iter()
                .map(|(family, count)| FamilySymbolicCount {
                    family: family.clone(),
                    count: project_count(count),
                })
                .collect()
        }),
        cleanup_work: project_count(counts.cleanup_work()),
    }
}

fn symbolic_expression_node(expression: &NativeResourceExpression) -> SymbolicExpressionNode {
    let symbol = expression.symbol();
    let rational = expression.rational_literal();
    SymbolicExpressionNode {
        kind: expression.kind().as_str().to_owned(),
        domain: expression.domain().as_str().to_owned(),
        arguments: expression
            .arguments()
            .into_iter()
            .map(qamomile_compiler::ResourceExpressionId::get)
            .collect(),
        predicates: expression
            .predicates()
            .into_iter()
            .map(qamomile_compiler::ResourceExpressionId::get)
            .collect(),
        literal: expression.literal().map(str::to_owned),
        literal_kind: expression
            .literal_kind()
            .map(|kind| kind.as_str().to_owned()),
        rational_numerator: rational.map(|value| value.numerator().to_owned()),
        rational_denominator: rational.map(|value| value.denominator().to_owned()),
        real_bits: expression.real_literal_bits(),
        symbol_scope: symbol.map(|value| value.scope().to_owned()),
        symbol_name: symbol.map(|value| value.name().to_owned()),
        symbol_domain: symbol.map(|value| value.domain().as_str().to_owned()),
        symbol_origin: symbol.map(|value| value.origin().to_owned()),
        symbol_constraints: symbol.map_or_else(Vec::new, |value| value.constraints().to_vec()),
        function_name: expression.function_name().map(str::to_owned),
        binder: expression
            .binder()
            .map(qamomile_compiler::ResourceExpressionId::get),
    }
}

fn symbolic_count(
    count: &NativeSymbolicCount,
    expressions: &[NativeResourceExpression],
) -> SymbolicCountProjection {
    SymbolicCountProjection {
        lower: count.lower().get(),
        upper: count
            .upper()
            .map(qamomile_compiler::ResourceExpressionId::get),
        exact: count.exact(),
        unknown: count.unknown().to_vec(),
        rule_identities: count.rule_identities().to_vec(),
        relation: count.relation().as_str().to_owned(),
        dependency_span_relation: count
            .dependency_span_relation(expressions)
            .as_str()
            .to_owned(),
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ResourceExpressionEvaluation>()?;
    module.add_class::<ResourceExpressionSymbolProjection>()?;
    module.add_class::<ProgramStructure>()?;
    module.add_class::<LogicalCounts>()?;
    module.add_class::<SymbolicExpressionNode>()?;
    module.add_class::<SymbolicCountProjection>()?;
    module.add_class::<OpcodeSymbolicCount>()?;
    module.add_class::<FamilySymbolicCount>()?;
    module.add_class::<MetricSymbolicCount>()?;
    module.add_class::<SymbolicAnalysisFactProjection>()?;
    module.add_class::<SymbolicLogicalCountsProjection>()?;
    module.add_function(wrap_pyfunction!(evaluate_resource_expression, module)?)?;
    module.add_function(wrap_pyfunction!(
        inspect_resource_expression_symbols,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(resolve_resource_symbol_bindings, module)?)?;
    module.add_function(wrap_pyfunction!(rank_resource_hotspots, module)?)?;
    module.add_function(wrap_pyfunction!(construct_resource_expression, module)?)?;
    module.add_function(wrap_pyfunction!(construct_resource_literal, module)?)?;
    module.add_function(wrap_pyfunction!(construct_resource_predicate, module)?)?;
    module.add_function(wrap_pyfunction!(construct_resource_symbol, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_resource_expression, module)?)?;
    module.add_function(wrap_pyfunction!(project_surface_code_resources, module)?)?;
    module.add_function(wrap_pyfunction!(retag_resource_expression_unit, module)?)?;
    module.add_function(wrap_pyfunction!(substitute_resource_expression, module)?)?;
    module.add_function(wrap_pyfunction!(assume_resource_iterations, module)?)?;
    module.add_function(wrap_pyfunction!(validate_resource_value, module)?)?;
    module.add_function(wrap_pyfunction!(validate_resource_unit, module)?)?;
    module.add_function(wrap_pyfunction!(seal_resource_study, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_resource_study, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_structure, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_exact_logical_counts, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_symbolic_logical_counts, module)?)?;
    module.add_function(wrap_pyfunction!(
        inspect_symbolic_resource_observation,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(inspect_portable_gate_recipe, module)?)?;
    module.add_function(wrap_pyfunction!(
        inspect_symbolic_program_family_counts,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        inspect_symbolic_resource_analysis_request,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(seal_resource_analysis_request, module)?)?;
    Ok(())
}
