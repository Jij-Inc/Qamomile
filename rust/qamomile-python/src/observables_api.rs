//! Observables projections and entry points for the native Python module.

use super::conversion::parse_unsigned_decimal;
use crate::program_owner::program_owner;
use pyo3::exceptions::{PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};
use qamomile_compiler::{
    DensePauliDecomposition, MAX_DENSE_MATRIX_ENTRIES, NormalizedObservable, Pauli,
    ProjectedSubspace, add_scaled_hermitian_matrices as add_scaled_native_hermitian_matrices,
    add_scaled_observable as add_scaled_native_observable,
    affine_observable as affine_native_observable,
    decompose_hermitian_matrix as decompose_native_hermitian_matrix,
    decompose_pauli_matrix as decompose_native_pauli_matrix,
    multiply_observables as multiply_native_observables,
    normalize_observable as normalize_native_observable,
    observable_is_pairwise_commuting as native_observable_is_pairwise_commuting,
    observable_lcu_one_norm as native_observable_lcu_one_norm,
    pauli_words_commute as native_pauli_words_commute,
    plan_commuting_pauli_expectation as plan_native_commuting_pauli_expectation,
    plan_qwc_pauli_expectation as plan_native_qwc_pauli_expectation,
    prepare_commuting_pauli_measurement as prepare_native_commuting_pauli_measurement,
    prepare_pauli_measurement as prepare_native_pauli_measurement,
    prepare_qwc_pauli_measurement as prepare_native_qwc_pauli_measurement,
    project_pauli_subspace as project_native_pauli_subspace,
    remap_observable as remap_native_observable,
    scale_hermitian_matrix as scale_native_hermitian_matrix,
    validate_hermitian_matrix as validate_native_hermitian_matrix,
};

/// Derive one verified Pauli-basis sampling program.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn prepare_pauli_measurement<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
    factors: Vec<(u64, String)>,
) -> PyResult<Bound<'py, PyBytes>> {
    let factors = parse_pauli_factors(factors)?;
    let transformed = py
        .detach(|| prepare_native_pauli_measurement(payload, &factors))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyBytes::new(py, &transformed))
}

/// Derive one verified QWC-group basis sampling program from term positions.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn prepare_qwc_pauli_measurement<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
    #[gen_stub(override_type(type_repr = "bytes"))] observable: &[u8],
    term_indices: Vec<usize>,
) -> PyResult<Bound<'py, PyBytes>> {
    let transformed = py
        .detach(|| prepare_native_qwc_pauli_measurement(payload, observable, &term_indices))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyBytes::new(py, &transformed))
}

/// Derive one verified generally commuting Clifford sampling program.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn prepare_commuting_pauli_measurement<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
    #[gen_stub(override_type(type_repr = "bytes"))] observable: &[u8],
    term_indices: Vec<usize>,
) -> PyResult<Bound<'py, PyBytes>> {
    let transformed = py
        .detach(|| prepare_native_commuting_pauli_measurement(payload, observable, &term_indices))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyBytes::new(py, &transformed))
}

type PauliMeasurementGroupProjection = (Vec<(u64, String)>, Vec<usize>, String, u64);
type CommutingPauliReadoutProjection = (usize, Vec<u64>, bool);
type CommutingPauliMeasurementGroupProjection = (
    Vec<usize>,
    Vec<CommutingPauliReadoutProjection>,
    usize,
    String,
    u64,
);
/// Validate the source and ABI boundary of one live executable artifact.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn validate_execution_interface(
    #[gen_stub(override_type(type_repr = "bytes | VerifiedProgramHandle"))] payload: &Bound<
        '_,
        PyAny,
    >,
    #[gen_stub(override_type(type_repr = "bytes"))] parameters: &[u8],
    #[gen_stub(override_type(type_repr = "bytes"))] result: &[u8],
) -> PyResult<()> {
    program_owner(payload)?
        .verified
        .validate_execution_interface(parameters, result)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Validate a reusable estimator source through the compiler type arena.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn validate_estimator_source(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
    #[gen_stub(override_type(type_repr = "bytes"))] observable: &[u8],
) -> PyResult<()> {
    py.detach(|| qamomile_compiler::validate_estimator_source(payload, observable))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Plan independent terms with a total shot budget.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn plan_independent_pauli_expectation(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "bytes"))] observable: &[u8],
    total_shots: &str,
    base_seed: u64,
) -> PyResult<Vec<PauliMeasurementGroupProjection>> {
    let total_shots =
        parse_unsigned_decimal(total_shots, "Q7311: expectation shots must be positive")?;
    let plan = py
        .detach(|| {
            qamomile_compiler::plan_independent_pauli_expectation(
                observable,
                &total_shots,
                base_seed,
            )
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(plan
        .groups()
        .iter()
        .map(|group| {
            (
                group
                    .basis()
                    .iter()
                    .map(|(index, pauli)| (*index, pauli_name(*pauli).to_owned()))
                    .collect(),
                group.term_indices().to_vec(),
                group.shots().to_str_radix(10),
                group.seed(),
            )
        })
        .collect())
}

/// Plan deterministic QWC groups and one exact total-shot allocation.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn plan_qwc_pauli_expectation(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "bytes"))] observable: &[u8],
    total_shots: &str,
    base_seed: u64,
) -> PyResult<Vec<PauliMeasurementGroupProjection>> {
    let total_shots =
        parse_unsigned_decimal(total_shots, "Q7311: expectation shots must be positive")?;
    let plan = py
        .detach(|| plan_native_qwc_pauli_expectation(observable, &total_shots, base_seed))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(plan
        .groups()
        .iter()
        .map(|group| {
            (
                group
                    .basis()
                    .iter()
                    .map(|(index, pauli)| (*index, pauli_name(*pauli).to_owned()))
                    .collect(),
                group.term_indices().to_vec(),
                group.shots().to_str_radix(10),
                group.seed(),
            )
        })
        .collect())
}

/// Plan deterministic generally commuting groups and exact shot allocation.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn plan_commuting_pauli_expectation(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "bytes"))] observable: &[u8],
    total_shots: &str,
    base_seed: u64,
) -> PyResult<Vec<CommutingPauliMeasurementGroupProjection>> {
    let total_shots =
        parse_unsigned_decimal(total_shots, "Q7311: expectation shots must be positive")?;
    let plan = py
        .detach(|| plan_native_commuting_pauli_expectation(observable, &total_shots, base_seed))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(plan
        .groups()
        .iter()
        .map(|group| {
            (
                group.term_indices().to_vec(),
                group
                    .readouts()
                    .iter()
                    .map(|readout| {
                        (
                            readout.term_index(),
                            readout.factors().to_vec(),
                            readout.is_negative(),
                        )
                    })
                    .collect(),
                group.gate_count(),
                group.shots().to_str_radix(10),
                group.seed(),
            )
        })
        .collect())
}

/// Normalize one portable Hamiltonian and return its semantic facts.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn normalize_observable<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<(Bound<'py, PyBytes>, bool)> {
    let observable = py
        .detach(|| normalize_native_observable(payload))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(observable_projection(py, &observable))
}

/// Return the exact-sum coefficient one-norm of one canonical observable.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn observable_lcu_one_norm(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<f64> {
    py.detach(|| native_observable_lcu_one_norm(payload))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Return the compiler-owned dense-matrix transfer budget.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
const fn dense_matrix_entry_budget() -> usize {
    MAX_DENSE_MATRIX_ENTRIES
}

/// Validate one dense Hermitian-matrix boundary through the compiler core.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn validate_hermitian_matrix(
    py: Python<'_>,
    shape: Vec<usize>,
    entries: Vec<(f64, f64)>,
    require_hermitian: bool,
    atol: f64,
) -> PyResult<u64> {
    let entries = entries
        .into_iter()
        .map(|(real, imaginary)| num_complex::Complex64::new(real, imaginary))
        .collect::<Vec<_>>();
    py.detach(|| validate_native_hermitian_matrix(&shape, &entries, require_hermitian, atol))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Add a real multiple of one Hermitian matrix through compiler-owned algebra.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn add_scaled_hermitian_matrices(
    py: Python<'_>,
    left_shape: Vec<usize>,
    left_entries: Vec<(f64, f64)>,
    right_shape: Vec<usize>,
    right_entries: Vec<(f64, f64)>,
    right_scale: f64,
) -> PyResult<Vec<(f64, f64)>> {
    let left_entries = complex_entries(left_entries);
    let right_entries = complex_entries(right_entries);
    let result = py
        .detach(|| {
            add_scaled_native_hermitian_matrices(
                &left_shape,
                &left_entries,
                &right_shape,
                &right_entries,
                right_scale,
            )
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(complex_projection(&result))
}

/// Scale or divide one Hermitian matrix through compiler-owned algebra.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn scale_hermitian_matrix(
    py: Python<'_>,
    shape: Vec<usize>,
    entries: Vec<(f64, f64)>,
    scalar: f64,
    divide: bool,
) -> PyResult<Vec<(f64, f64)>> {
    let entries = complex_entries(entries);
    let result = py
        .detach(|| scale_native_hermitian_matrix(&shape, &entries, scalar, divide))
        .map_err(|error| match error {
            qamomile_compiler::ObservableError::MatrixDivisionByZero => {
                PyZeroDivisionError::new_err(error.to_string())
            }
            _ => PyValueError::new_err(error.to_string()),
        })?;
    Ok(complex_projection(&result))
}

fn complex_entries(entries: Vec<(f64, f64)>) -> Vec<num_complex::Complex64> {
    entries
        .into_iter()
        .map(|(real, imaginary)| num_complex::Complex64::new(real, imaginary))
        .collect()
}

fn complex_projection(entries: &[num_complex::Complex64]) -> Vec<(f64, f64)> {
    entries.iter().map(|value| (value.re, value.im)).collect()
}

/// Decompose one dense matrix into the compiler-owned Pauli normal form.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn decompose_hermitian_matrix(
    py: Python<'_>,
    shape: Vec<usize>,
    entries: Vec<(f64, f64)>,
    tol: f64,
) -> PyResult<(Bound<'_, PyBytes>, bool)> {
    let entries = entries
        .into_iter()
        .map(|(real, imaginary)| num_complex::Complex64::new(real, imaginary))
        .collect::<Vec<_>>();
    let observable = py
        .detach(|| decompose_native_hermitian_matrix(&shape, &entries, tol))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(observable_projection(py, &observable))
}

/// Decompose one arbitrary dense matrix into a complex Pauli sum.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn decompose_pauli_matrix(
    py: Python<'_>,
    shape: Vec<usize>,
    entries: Vec<(f64, f64)>,
    atol: f64,
) -> PyResult<(Bound<'_, PyBytes>, bool, f64)> {
    let entries = entries
        .into_iter()
        .map(|(real, imaginary)| num_complex::Complex64::new(real, imaginary))
        .collect::<Vec<_>>();
    let decomposition = py
        .detach(|| decompose_native_pauli_matrix(&shape, &entries, atol))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(pauli_decomposition_projection(py, &decomposition))
}

fn pauli_decomposition_projection<'py>(
    py: Python<'py>,
    decomposition: &DensePauliDecomposition,
) -> (Bound<'py, PyBytes>, bool, f64) {
    let (literal, hermitian) = observable_projection(py, decomposition.observable());
    (literal, hermitian, decomposition.truncation_error_bound())
}

/// Multiply two portable Hamiltonians through the compiler-owned algebra.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn multiply_observables<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] left: &[u8],
    #[gen_stub(override_type(type_repr = "bytes"))] right: &[u8],
) -> PyResult<(Bound<'py, PyBytes>, bool)> {
    let observable = py
        .detach(|| multiply_native_observables(left, right))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(observable_projection(py, &observable))
}

/// Apply one complex scale and identity shift through the compiler-owned algebra.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn affine_observable<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
    scale: (f64, f64),
    shift: (f64, f64),
) -> PyResult<(Bound<'py, PyBytes>, bool)> {
    let observable = py
        .detach(|| {
            affine_native_observable(
                payload,
                num_complex::Complex64::new(scale.0, scale.1),
                num_complex::Complex64::new(shift.0, shift.1),
            )
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(observable_projection(py, &observable))
}

/// Add one complex-scaled observable through the compiler-owned algebra.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn add_scaled_observable<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] left: &[u8],
    #[gen_stub(override_type(type_repr = "bytes"))] right: &[u8],
    scale: (f64, f64),
) -> PyResult<(Bound<'py, PyBytes>, bool)> {
    let observable = py
        .detach(|| {
            add_scaled_native_observable(left, right, num_complex::Complex64::new(scale.0, scale.1))
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(observable_projection(py, &observable))
}

/// Remap logical indices through the compiler-owned observable algebra.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn remap_observable<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
    indices: Vec<(u64, u64)>,
) -> PyResult<(Bound<'py, PyBytes>, bool)> {
    let observable = py
        .detach(|| remap_native_observable(payload, &indices))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(observable_projection(py, &observable))
}

pub(super) fn observable_projection<'py>(
    py: Python<'py>,
    observable: &NormalizedObservable,
) -> (Bound<'py, PyBytes>, bool) {
    (
        PyBytes::new(py, observable.literal()),
        observable.is_hermitian(),
    )
}

/// Compute the deliberately lazy pairwise-commutation property.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn observable_is_pairwise_commuting(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<bool> {
    py.detach(|| native_observable_is_pairwise_commuting(payload))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Project a canonical observable onto sampled Pauli product states.
type ComplexCoordinates = Vec<(f64, f64)>;
type ProjectedSubspaceCoordinates = (usize, ComplexCoordinates, ComplexCoordinates);

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn project_pauli_subspace(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
    samples: Vec<Vec<u8>>,
    bases: Vec<Vec<u8>>,
) -> PyResult<ProjectedSubspaceCoordinates> {
    let projected = py
        .detach(|| project_native_pauli_subspace(payload, &samples, &bases))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(projected_subspace_projection(&projected))
}

fn projected_subspace_projection(projected: &ProjectedSubspace) -> ProjectedSubspaceCoordinates {
    let coordinates = |values: &[num_complex::Complex64]| {
        values
            .iter()
            .map(|value| (value.re, value.im))
            .collect::<Vec<_>>()
    };
    (
        projected.size(),
        coordinates(projected.hamiltonian()),
        coordinates(projected.overlap()),
    )
}

/// Determine whether two canonical Pauli words commute.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn pauli_words_commute(left: Vec<(u64, String)>, right: Vec<(u64, String)>) -> PyResult<bool> {
    let left = parse_pauli_factors(left)?;
    let right = parse_pauli_factors(right)?;
    Ok(native_pauli_words_commute(&left, &right))
}

fn parse_pauli_factors(factors: Vec<(u64, String)>) -> PyResult<Vec<(u64, Pauli)>> {
    factors
        .into_iter()
        .map(|(index, basis)| {
            let basis = match basis.as_str() {
                "X" => Pauli::X,
                "Y" => Pauli::Y,
                "Z" => Pauli::Z,
                _ => return Err(PyValueError::new_err("Q7302: unknown Pauli basis")),
            };
            Ok((index, basis))
        })
        .collect()
}

const fn pauli_name(pauli: Pauli) -> &'static str {
    match pauli {
        Pauli::X => "X",
        Pauli::Y => "Y",
        Pauli::Z => "Z",
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(pauli_words_commute, module)?)?;
    module.add_function(wrap_pyfunction!(
        prepare_commuting_pauli_measurement,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(prepare_pauli_measurement, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_qwc_pauli_measurement, module)?)?;
    module.add_function(wrap_pyfunction!(plan_commuting_pauli_expectation, module)?)?;
    module.add_function(wrap_pyfunction!(plan_qwc_pauli_expectation, module)?)?;
    module.add_function(wrap_pyfunction!(
        plan_independent_pauli_expectation,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(validate_estimator_source, module)?)?;
    module.add_function(wrap_pyfunction!(validate_execution_interface, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_observable, module)?)?;
    module.add_function(wrap_pyfunction!(observable_lcu_one_norm, module)?)?;
    module.add_function(wrap_pyfunction!(dense_matrix_entry_budget, module)?)?;
    module.add_function(wrap_pyfunction!(validate_hermitian_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(add_scaled_hermitian_matrices, module)?)?;
    module.add_function(wrap_pyfunction!(scale_hermitian_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(decompose_hermitian_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(decompose_pauli_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(multiply_observables, module)?)?;
    module.add_function(wrap_pyfunction!(affine_observable, module)?)?;
    module.add_function(wrap_pyfunction!(add_scaled_observable, module)?)?;
    module.add_function(wrap_pyfunction!(remap_observable, module)?)?;
    module.add_function(wrap_pyfunction!(observable_is_pairwise_commuting, module)?)?;
    module.add_function(wrap_pyfunction!(project_pauli_subspace, module)?)?;
    Ok(())
}
