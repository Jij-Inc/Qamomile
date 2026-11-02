//! Types projections and entry points for the native Python module.

use super::conversion::parse_signed_decimal;
use crate::program_owner::program_owner;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};
use qamomile_compiler::{
    PortableDimension as NativePortableDimension,
    RealFinitenessOperand as NativeRealFinitenessOperand, Stage as NativeStage,
    StaticDimensionCandidate as NativeStaticDimensionCandidate,
    TypeAdmissibilityProjection as NativeTypeAdmissibilityProjection,
    TypeShapeProjection as NativeTypeShapeProjection,
    bind_declared_function_arguments as bind_native_declared_function_arguments,
    common_type_stage as native_common_type_stage,
    evaluate_classical_literals as evaluate_native_classical_literals,
    infer_classical_type as infer_native_classical_type,
    infer_declared_function_application as infer_native_declared_function_application,
    infer_function_application as infer_native_function_application,
    infer_literal_type as infer_native_literal_type,
    infer_type_dimensions as infer_native_type_dimensions,
    inspect_type_admissibility as inspect_native_type_admissibility,
    inspect_type_shape as inspect_native_type_shape,
    literal_conforms_type as native_literal_conforms_type,
    normalize_aggregate_index as normalize_native_aggregate_index,
    normalize_aggregate_slice as normalize_native_aggregate_slice,
    operation_catalog as native_operation_catalog,
    real_operation_facts as native_real_operation_facts,
    substitute_type_dimensions as substitute_native_type_dimensions,
    terminator_catalog as native_terminator_catalog,
    type_is_assignable as native_type_is_assignable,
    type_symbolic_dimensions as native_type_symbolic_dimensions,
    types_are_local_peers as native_types_are_local_peers,
};

type GateCatalogProjection = (String, Vec<(String, usize)>, Vec<String>, bool);
type OptionalDimensionProjection = Option<(Option<String>, Option<String>)>;
type TypeDimensionProjection = (String, Option<u64>, Option<String>);
type TypeShapeProjection = (
    Option<u64>,
    Option<Vec<u64>>,
    Option<Vec<u8>>,
    Option<Vec<Vec<u8>>>,
);
type OperationCatalogProjection = (
    String,
    String,
    bool,
    Vec<String>,
    Vec<String>,
    Option<GateCatalogProjection>,
);
type NamedBooleanLiftProjection = (
    String,
    u64,
    Vec<Vec<Vec<u64>>>,
    Option<Vec<u64>>,
    Option<Vec<(u64, u64)>>,
);

/// Project the compiler-owned operation and terminator vocabulary.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn semantic_operation_catalog() -> (Vec<OperationCatalogProjection>, Vec<String>) {
    let operations = native_operation_catalog()
        .into_iter()
        .map(|entry| {
            let gate = entry.gate().map(|gate| {
                (
                    gate.name().to_owned(),
                    gate.selectors()
                        .iter()
                        .enumerate()
                        .map(|(index, value)| {
                            (
                                (*value).to_owned(),
                                gate.selector_default(index)
                                    .expect("declared selector has a default"),
                            )
                        })
                        .collect(),
                    gate.families()
                        .iter()
                        .map(|value| (*value).to_owned())
                        .collect(),
                    gate.has_angle(),
                )
            });
            (
                entry.opcode().to_owned(),
                entry.dialect().to_owned(),
                entry.is_structured(),
                entry
                    .effects()
                    .iter()
                    .map(|effect| effect.as_str().to_owned())
                    .collect(),
                entry
                    .requirements()
                    .iter()
                    .map(|requirement| requirement.as_str().to_owned())
                    .collect(),
                gate,
            )
        })
        .collect();
    let terminators = native_terminator_catalog()
        .into_iter()
        .map(str::to_owned)
        .collect();
    (operations, terminators)
}

/// Return every referenced exact Boolean lift plan after one program decode.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn plan_boolean_lifts(
    #[gen_stub(override_type(type_repr = "bytes | VerifiedProgramHandle"))] payload: &Bound<
        '_,
        PyAny,
    >,
) -> PyResult<Vec<NamedBooleanLiftProjection>> {
    let owner = program_owner(payload)?;
    let program = &owner.verified;
    Ok(program
        .boolean_lift_plans()
        .map(|(predicate, plan)| {
            (
                predicate.to_owned(),
                plan.input_width(),
                plan.outputs()
                    .iter()
                    .map(|output| {
                        output
                            .monomials()
                            .iter()
                            .map(|monomial| monomial.variables().to_vec())
                            .collect()
                    })
                    .collect(),
                plan.permutation_order().map(<[u64]>::to_vec),
                plan.basis_transpositions().map(|transpositions| {
                    transpositions
                        .iter()
                        .map(|transposition| (transposition.left(), transposition.right()))
                        .collect()
                }),
            )
        })
        .collect())
}

/// Evaluate one verified pure-classical operation over portable literals.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn evaluate_classical<'py>(
    py: Python<'py>,
    opcode: &str,
    operands: Vec<Vec<u8>>,
    attributes: Vec<(String, Vec<u8>)>,
) -> PyResult<Bound<'py, PyBytes>> {
    let value = py
        .detach(|| evaluate_native_classical_literals(opcode, &operands, &attributes))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyBytes::new(py, &value))
}

/// Derive the runtime guard and finiteness for one verified `QCore` Real operation.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn real_operation_facts(
    opcode: &str,
    operands: Vec<(bool, Option<f64>)>,
) -> PyResult<(String, bool)> {
    let operands = operands
        .into_iter()
        .map(|(guaranteed_finite, static_value)| {
            NativeRealFinitenessOperand::new(guaranteed_finite, static_value)
        })
        .collect::<Vec<_>>();
    let facts = native_real_operation_facts(opcode, &operands)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        facts.runtime_guard().as_str().to_owned(),
        facts.guarantees_finite(),
    ))
}

/// Check concrete direct-emission selectors through the native circuit rules.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn validate_circuit_selectors(opcode: &str, width: usize, selectors: Vec<i64>) -> PyResult<()> {
    qamomile_compiler::validate_circuit_selectors(opcode, width, &selectors)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Normalize one exact Python aggregate index through the native compiler.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn normalize_aggregate_index(extent: usize, index: &str) -> PyResult<Option<usize>> {
    let index = parse_signed_decimal(index, "Q4513: invalid aggregate index")?;
    Ok(normalize_native_aggregate_index(extent, &index))
}

/// Normalize exact Python aggregate slice bounds through the native compiler.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::similar_names)]
fn normalize_aggregate_slice(
    extent: usize,
    start: Option<&str>,
    stop: Option<&str>,
    step: Option<&str>,
) -> PyResult<Vec<usize>> {
    let parse = |value: Option<&str>| {
        value
            .map(|item| parse_signed_decimal(item, "Q4513: invalid aggregate slice bound"))
            .transpose()
    };
    let start = parse(start)?;
    let stop = parse(stop)?;
    let step = parse(step)?;
    normalize_native_aggregate_slice(extent, start.as_ref(), stop.as_ref(), step.as_ref())
        .map(<[usize]>::into_vec)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Derive one classical result type through the native verifier equation.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn infer_classical_result_type<'py>(
    py: Python<'py>,
    opcode: &str,
    operands: Vec<Vec<u8>>,
    attributes: Vec<(String, Vec<u8>)>,
) -> PyResult<Bound<'py, PyBytes>> {
    let result = py
        .detach(|| infer_native_classical_type(opcode, &operands, &attributes))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyBytes::new(py, &result))
}

/// Infer dependent dimensions through the native logical type relation.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn unify_type_dimensions(
    py: Python<'_>,
    expected: Vec<Vec<u8>>,
    actual: Vec<Vec<u8>>,
    initial: Vec<TypeDimensionProjection>,
) -> PyResult<Option<Vec<TypeDimensionProjection>>> {
    let initial = parse_type_dimension_projections(initial)?;
    let inferred = py
        .detach(|| infer_native_type_dimensions(&expected, &actual, &initial))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(inferred.map(|bindings| {
        bindings
            .iter()
            .map(|binding| match binding.value() {
                NativePortableDimension::Fixed(value) => {
                    (binding.name().to_owned(), Some(*value), None)
                }
                NativePortableDimension::Symbol(value) => {
                    (binding.name().to_owned(), None, Some(value.clone()))
                }
            })
            .collect()
    }))
}

/// Check one declared type boundary through the native assignability relation.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn is_type_assignable(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "bytes"))] expected: &[u8],
    #[gen_stub(override_type(type_repr = "bytes"))] actual: &[u8],
) -> PyResult<bool> {
    py.detach(|| native_type_is_assignable(expected, actual))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Bind a source-ordered closed-function call and infer its native result type.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn infer_function_application<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] function_type: &[u8],
    argument_types: Vec<Vec<u8>>,
    positional_count: usize,
    keyword_names: Vec<String>,
) -> PyResult<Bound<'py, PyBytes>> {
    let result = py
        .detach(|| {
            infer_native_function_application(
                function_type,
                &argument_types,
                positional_count,
                &keyword_names,
            )
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyBytes::new(py, &result))
}

/// Infer one declaration-ordered call using aligned static dimension evidence.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn infer_declared_function_application<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] function_type: &[u8],
    argument_types: Vec<Vec<u8>>,
    static_arguments: Vec<OptionalDimensionProjection>,
) -> PyResult<Bound<'py, PyBytes>> {
    let static_arguments = parse_optional_dimension_projections(static_arguments)?;
    let result = py
        .detach(|| {
            infer_native_declared_function_application(
                function_type,
                &argument_types,
                &static_arguments,
            )
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyBytes::new(py, &result))
}

/// Bind one raw source call into declaration order, retaining default slots.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn bind_declared_function_arguments(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "bytes"))] function_type: &[u8],
    argument_count: usize,
    positional_count: usize,
    keyword_names: Vec<String>,
    has_defaults: Vec<bool>,
) -> PyResult<Vec<Option<usize>>> {
    py.detach(|| {
        bind_native_declared_function_arguments(
            function_type,
            argument_count,
            positional_count,
            &keyword_names,
            &has_defaults,
        )
    })
    .map(<[Option<usize>]>::into_vec)
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Check one rigid local peer equation through the native type arena.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn are_local_peer_types(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "bytes"))] left: &[u8],
    #[gen_stub(override_type(type_repr = "bytes"))] right: &[u8],
) -> PyResult<bool> {
    py.detach(|| native_types_are_local_peers(left, right))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Check one canonical portable literal against one native semantic type.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn literal_conforms_type(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "bytes"))] type_source: &[u8],
    #[gen_stub(override_type(type_repr = "bytes"))] literal_source: &[u8],
) -> PyResult<bool> {
    py.detach(|| native_literal_conforms_type(type_source, literal_source))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Infer one source literal's unique semantic introduction type.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn infer_literal_type<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] literal_source: &[u8],
) -> PyResult<Option<Bound<'py, PyBytes>>> {
    let inferred = py
        .detach(|| infer_native_literal_type(literal_source))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(inferred.map(|payload| PyBytes::new(py, &payload)))
}

/// Substitute dependent dimensions through the native semantic type algebra.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn substitute_type_dimensions<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] source: &[u8],
    bindings: Vec<TypeDimensionProjection>,
) -> PyResult<Bound<'py, PyBytes>> {
    let bindings = parse_type_dimension_projections(bindings)?;
    let result = py
        .detach(|| substitute_native_type_dimensions(source, &bindings))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyBytes::new(py, &result))
}

/// Collect sorted dependent-dimension names through the native type algebra.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn type_symbolic_dimensions(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "bytes"))] source: &[u8],
) -> PyResult<Vec<String>> {
    py.detach(|| native_type_symbolic_dimensions(source))
        .map(<[String]>::into_vec)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Project logical widths and structural element types from one native type DAG.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[gen_stub(override_return_type(
    type_repr = "tuple[int | None, list[int] | None, bytes | None, list[bytes] | None]"
))]
fn inspect_type_shape(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "bytes"))] source: &[u8],
) -> PyResult<TypeShapeProjection> {
    let projection: NativeTypeShapeProjection = py
        .detach(|| inspect_native_type_shape(source))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        projection.logical_width(),
        projection.coherent_leaf_widths().map(ToOwned::to_owned),
        projection.sequence_element().map(ToOwned::to_owned),
        projection.aggregate_elements().map(ToOwned::to_owned),
    ))
}

/// Project coherent and staged constructor admissibility from one type DAG.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_type_admissibility(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "bytes"))] source: &[u8],
) -> PyResult<(bool, bool, bool)> {
    let projection: NativeTypeAdmissibilityProjection = py
        .detach(|| inspect_native_type_admissibility(source))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        projection.coherent_payload(),
        projection.stageable(),
        projection.classical(),
    ))
}

/// Return the latest outer availability stage in one canonical type row.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn common_type_stage(py: Python<'_>, sources: Vec<Vec<u8>>) -> PyResult<Option<&'static str>> {
    let stage = py
        .detach(|| native_common_type_stage(&sources))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(stage.map(NativeStage::as_str))
}

fn parse_type_dimension_projections(
    source: Vec<TypeDimensionProjection>,
) -> PyResult<Vec<(String, NativePortableDimension)>> {
    source
        .into_iter()
        .map(|(name, fixed, symbol)| {
            let value = match (fixed, symbol) {
                (Some(value), None) => NativePortableDimension::Fixed(value),
                (None, Some(value)) => NativePortableDimension::Symbol(value),
                _ => {
                    return Err(PyValueError::new_err(
                        "Q4510: invalid type-relation input: dimension must select exactly one representation",
                    ));
                }
            };
            Ok((name, value))
        })
        .collect()
}

fn parse_optional_dimension_projections(
    source: Vec<OptionalDimensionProjection>,
) -> PyResult<Vec<Option<NativeStaticDimensionCandidate>>> {
    source
        .into_iter()
        .map(|projection| {
            projection
                .map(|(fixed, symbol)| match (fixed, symbol) {
                    (Some(value), None) => Ok(NativeStaticDimensionCandidate::Integer(value)),
                    (None, Some(value)) => Ok(NativeStaticDimensionCandidate::Symbol(value)),
                    _ => Err(PyValueError::new_err(
                        "Q4510: invalid type-relation input: dimension must select exactly one representation",
                    )),
                })
                .transpose()
        })
        .collect()
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(unify_type_dimensions, module)?)?;
    module.add_function(wrap_pyfunction!(is_type_assignable, module)?)?;
    module.add_function(wrap_pyfunction!(infer_function_application, module)?)?;
    module.add_function(wrap_pyfunction!(
        infer_declared_function_application,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(bind_declared_function_arguments, module)?)?;
    module.add_function(wrap_pyfunction!(are_local_peer_types, module)?)?;
    module.add_function(wrap_pyfunction!(literal_conforms_type, module)?)?;
    module.add_function(wrap_pyfunction!(infer_literal_type, module)?)?;
    module.add_function(wrap_pyfunction!(substitute_type_dimensions, module)?)?;
    module.add_function(wrap_pyfunction!(type_symbolic_dimensions, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_type_shape, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_type_admissibility, module)?)?;
    module.add_function(wrap_pyfunction!(common_type_stage, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_classical, module)?)?;
    module.add_function(wrap_pyfunction!(real_operation_facts, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_aggregate_index, module)?)?;
    module.add_function(wrap_pyfunction!(validate_circuit_selectors, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_aggregate_slice, module)?)?;
    module.add_function(wrap_pyfunction!(infer_classical_result_type, module)?)?;
    module.add_function(wrap_pyfunction!(plan_boolean_lifts, module)?)?;
    module.add_function(wrap_pyfunction!(semantic_operation_catalog, module)?)?;
    Ok(())
}
