//! Algorithms projections and entry points for the native Python module.

use super::conversion::{
    decimal_rows, parse_signed_decimal, parse_unsigned_decimal, parse_unsigned_decimal_matrix,
    parse_unsigned_decimals, parse_width,
};
use num_bigint::{BigInt, BigUint};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use qamomile_compiler::{
    AmplitudeCoordinates, EkeraHastadPlan as NativeEkeraHastadPlan,
    LcuCoefficientPlan as NativeLcuCoefficientPlan, MAX_GROVER_QUBITS,
    ModularMultiplicationPlan as NativeModularMultiplicationPlan, PeriodicShiftDecomposition,
    SparseLookupPlan as NativeSparseLookupPlan, XorLookupPlan as NativeXorLookupPlan,
    cyclic_gray_control_schedule as native_cyclic_gray_control_schedule,
    decompose_periodic_shift_matrix as decompose_native_periodic_shift_matrix,
    grover_iteration_count as native_grover_iteration_count,
    modular_addition_residue as native_modular_addition_residue,
    mottonen_amplitude_coordinates as native_mottonen_amplitude_coordinates,
    mottonen_rotation_schedule as native_mottonen_rotation_schedule,
    normalize_periodic_shift_terms as normalize_native_periodic_shift_terms,
    plan_ekera_hastad as plan_native_ekera_hastad,
    plan_lcu_coefficients as plan_native_lcu_coefficients,
    plan_modular_multiplication as plan_native_modular_multiplication,
    plan_sparse_lookup as plan_native_sparse_lookup, plan_xor_lookup as plan_native_xor_lookup,
    suzuki_product_formula_schedule as native_suzuki_product_formula_schedule,
    uniformly_controlled_rotation_coordinates as native_uniformly_controlled_rotation_coordinates,
    wrapping_addition_offsets as native_wrapping_addition_offsets,
};

/// Derive exact-phase Möttönen coordinates from one complex amplitude vector.
type AmplitudeCoordinateProjection = (usize, Vec<f64>, Vec<f64>, f64);

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn mottonen_amplitude_coordinates(
    py: Python<'_>,
    amplitudes: Vec<(f64, f64)>,
) -> PyResult<AmplitudeCoordinateProjection> {
    let amplitudes = amplitudes
        .into_iter()
        .map(|(real, imaginary)| num_complex::Complex64::new(real, imaginary))
        .collect::<Vec<_>>();
    let coordinates = py
        .detach(|| native_mottonen_amplitude_coordinates(&amplitudes))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(amplitude_coordinate_projection(&coordinates))
}

fn amplitude_coordinate_projection(
    coordinates: &AmplitudeCoordinates,
) -> AmplitudeCoordinateProjection {
    (
        coordinates.width(),
        coordinates.ry().to_vec(),
        coordinates.rz().to_vec(),
        coordinates.phase(),
    )
}

/// Convert address-ordered angles into cyclic Gray-walk coordinates.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn uniformly_controlled_rotation_coordinates(
    py: Python<'_>,
    branch_angles: Vec<f64>,
) -> PyResult<Vec<f64>> {
    py.detach(|| native_uniformly_controlled_rotation_coordinates(&branch_angles))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Return the control components for one complete cyclic Gray walk.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn cyclic_gray_control_schedule(py: Python<'_>, width: usize) -> PyResult<Vec<usize>> {
    py.detach(|| native_cyclic_gray_control_schedule(width))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Return angle, target, and control indices for Möttönen's Gray walks.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn mottonen_rotation_schedule(
    py: Python<'_>,
    width: usize,
) -> PyResult<Vec<(usize, usize, usize)>> {
    py.detach(|| native_mottonen_rotation_schedule(width))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Construct one bounded adjacent-normalized Suzuki schedule.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn suzuki_product_formula_schedule(
    py: Python<'_>,
    term_count: usize,
    order: usize,
) -> PyResult<Vec<(usize, f64)>> {
    py.detach(|| native_suzuki_product_formula_schedule(term_count, order))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// One immutable native LCU coefficient and selector plan.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct LcuCoefficientPlan {
    plan: NativeLcuCoefficientPlan,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl LcuCoefficientPlan {
    /// Return canonical nonzero groups and their coefficient coordinates.
    #[getter]
    fn terms(&self) -> Vec<(usize, f64, f64, f64, f64)> {
        self.plan
            .terms()
            .iter()
            .map(|term| {
                let coefficient = term.coefficient();
                (
                    term.group(),
                    coefficient.re,
                    coefficient.im,
                    term.amplitude(),
                    term.phase(),
                )
            })
            .collect()
    }

    /// Return the coefficient one-norm, or one for the zero operator.
    #[getter]
    fn normalization(&self) -> f64 {
        self.plan.normalization()
    }

    /// Return the positive selector-register width.
    #[getter]
    fn signal_width(&self) -> usize {
        self.plan.signal_width()
    }

    /// Return the power-of-two padded SELECT case count.
    #[getter]
    fn case_count(&self) -> usize {
        self.plan.case_count()
    }
}

/// Derive one bounded LCU coefficient plan from callable-group rows.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn plan_lcu_coefficients(
    py: Python<'_>,
    group_count: usize,
    rows: Vec<(usize, f64, f64)>,
) -> PyResult<LcuCoefficientPlan> {
    let coefficients = rows
        .into_iter()
        .map(|(group, real, imaginary)| (group, num_complex::Complex64::new(real, imaginary)))
        .collect::<Vec<_>>();
    let plan = py
        .detach(|| plan_native_lcu_coefficients(group_count, &coefficients))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(LcuCoefficientPlan { plan })
}

/// One immutable native XOR lookup schedule.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct XorLookupPlan {
    plan: NativeXorLookupPlan,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl XorLookupPlan {
    /// Return the coherent address-register width.
    #[getter]
    fn address_width(&self) -> usize {
        self.plan.address_width()
    }

    /// Return the output-register width.
    #[getter]
    fn value_width(&self) -> usize {
        self.plan.value_width()
    }

    /// Return one ascending X-target row per address.
    #[getter]
    fn flip_targets(&self) -> Vec<Vec<usize>> {
        self.plan.flip_targets().to_vec()
    }
}

/// One immutable native sparse lookup schedule.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct SparseLookupPlan {
    plan: NativeSparseLookupPlan,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl SparseLookupPlan {
    /// Return the packed row/slot address-register width.
    #[getter]
    fn address_width(&self) -> usize {
        self.plan.address_width()
    }

    /// Return the column output-register width.
    #[getter]
    fn column_width(&self) -> usize {
        self.plan.column_width()
    }

    /// Return the encoded-value output-register width.
    #[getter]
    fn value_width(&self) -> usize {
        self.plan.value_width()
    }

    /// Return one packed ascending X-target row per address.
    #[getter]
    fn flip_targets(&self) -> Vec<Vec<usize>> {
        self.plan.flip_targets().to_vec()
    }
}

/// Derive one bounded XOR lookup schedule.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn plan_xor_lookup(
    py: Python<'_>,
    values: Vec<String>,
    value_width: Option<&str>,
) -> PyResult<XorLookupPlan> {
    let values = parse_unsigned_decimals(&values, "Q7654: invalid lookup value")?;
    let value_width = value_width
        .map(|value| parse_width(value, "Q7653: invalid lookup value width"))
        .transpose()?;
    let plan = py
        .detach(|| plan_native_xor_lookup(&values, value_width))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(XorLookupPlan { plan })
}

/// Derive one bounded sparse row/slot lookup schedule.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn plan_sparse_lookup(
    py: Python<'_>,
    columns: Vec<Vec<String>>,
    values: Vec<Vec<String>>,
    column_width: &str,
    value_width: Option<&str>,
) -> PyResult<SparseLookupPlan> {
    let columns = parse_unsigned_decimal_matrix(columns, "Q7658: invalid column index")?;
    let values = parse_unsigned_decimal_matrix(values, "Q7654: invalid encoded value")?;
    let column_width = parse_width(column_width, "Q7653: invalid column width")?;
    let value_width = value_width
        .map(|value| parse_width(value, "Q7653: invalid lookup value width"))
        .transpose()?;
    let plan = py
        .detach(|| plan_native_sparse_lookup(&columns, &values, column_width, value_width))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(SparseLookupPlan { plan })
}

/// One immutable native modular-multiplication schedule.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct ModularMultiplicationPlan {
    plan: NativeModularMultiplicationPlan,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl ModularMultiplicationPlan {
    /// Return the value-register width.
    #[getter]
    fn width(&self) -> usize {
        self.plan.width()
    }

    /// Return the canonical positive modulus as decimal text.
    #[getter]
    fn modulus(&self) -> String {
        self.plan.modulus().to_str_radix(10)
    }

    /// Return the canonical multiplier residue as decimal text.
    #[getter]
    fn multiplier(&self) -> String {
        self.plan.multiplier().to_str_radix(10)
    }

    /// Return the modular inverse as decimal text.
    #[getter]
    fn inverse(&self) -> String {
        self.plan.inverse().to_str_radix(10)
    }

    /// Return shifted multiplier residues as decimal text.
    #[getter]
    fn accumulate(&self) -> Vec<String> {
        decimal_rows(self.plan.accumulate())
    }

    /// Return shifted inverse residues as decimal text.
    #[getter]
    fn erase(&self) -> Vec<String> {
        decimal_rows(self.plan.erase())
    }
}

/// One immutable native Ekerå–Håstad coordinate plan.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct EkeraHastadPlan {
    plan: NativeEkeraHastadPlan,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl EkeraHastadPlan {
    /// Return the modular value width.
    #[getter]
    fn width(&self) -> usize {
        self.plan.width()
    }

    /// Return the selected short parameter width.
    #[getter]
    fn parameter_width(&self) -> usize {
        self.plan.parameter_width()
    }

    /// Return the long counting-register width.
    #[getter]
    fn long_precision(&self) -> usize {
        self.plan.long_precision()
    }

    /// Return the short counting-register width.
    #[getter]
    fn short_precision(&self) -> usize {
        self.plan.short_precision()
    }

    /// Return the complete multiplication target width.
    #[getter]
    fn target_width(&self) -> usize {
        self.plan.target_width()
    }

    /// Return the short-DLP inverse multiplier as decimal text.
    #[getter]
    fn inverse_y(&self) -> String {
        self.plan.inverse_y().to_str_radix(10)
    }
}

/// Return native set-bit offsets for signed wrapping addition.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn wrapping_addition_offsets(py: Python<'_>, width: &str, value: &str) -> PyResult<Vec<usize>> {
    let width = parse_width(width, "Q7631: invalid arithmetic width")?;
    let value = parse_signed_decimal(value, "Q7635: invalid arithmetic constant")?;
    py.detach(|| native_wrapping_addition_offsets(width, &value))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Return one native canonical modular-addition residue as decimal text.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn modular_addition_residue(
    py: Python<'_>,
    width: &str,
    value: &str,
    modulus: &str,
) -> PyResult<String> {
    let width = parse_width(width, "Q7631: invalid arithmetic width")?;
    let value = parse_signed_decimal(value, "Q7635: invalid arithmetic constant")?;
    let modulus = parse_unsigned_decimal(modulus, "Q7632: invalid arithmetic modulus")?;
    py.detach(|| native_modular_addition_residue(width, &value, &modulus))
        .map(|residue| residue.to_str_radix(10))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Derive one native modular-multiplication schedule.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn plan_modular_multiplication(
    py: Python<'_>,
    width: &str,
    value: &str,
    modulus: &str,
) -> PyResult<ModularMultiplicationPlan> {
    let width = parse_width(width, "Q7634: invalid modular multiplication width")?;
    let value = parse_signed_decimal(value, "Q7635: invalid arithmetic constant")?;
    let modulus = parse_unsigned_decimal(modulus, "Q7632: invalid arithmetic modulus")?;
    let plan = py
        .detach(|| plan_native_modular_multiplication(width, &value, &modulus))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(ModularMultiplicationPlan { plan })
}

/// Derive one native Ekerå–Håstad coordinate plan.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn plan_ekera_hastad(
    py: Python<'_>,
    generator: &str,
    modulus: &str,
    parameter_width: Option<&str>,
) -> PyResult<EkeraHastadPlan> {
    let generator = parse_unsigned_decimal(generator, "Q7640: invalid short-DLP generator")?;
    let modulus = parse_unsigned_decimal(modulus, "Q7641: invalid short-DLP modulus")?;
    let parameter_width = parameter_width
        .map(|value| parse_width(value, "Q7643: invalid short-DLP parameter width"))
        .transpose()?;
    let plan = py
        .detach(|| plan_native_ekera_hastad(&generator, &modulus, parameter_width))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(EkeraHastadPlan { plan })
}

/// Derive the bounded arbitrary-precision Grover iteration count.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn grover_iteration_count(py: Python<'_>, num_qubits: &str, num_marked: &str) -> PyResult<String> {
    let qubits = num_qubits.parse::<usize>().map_err(|_| {
        PyValueError::new_err(format!(
            "Q7591: Grover search width exceeds the {MAX_GROVER_QUBITS}-qubit budget"
        ))
    })?;
    let marked = BigUint::parse_bytes(num_marked.as_bytes(), 10)
        .ok_or_else(|| PyValueError::new_err("Q7592: invalid Grover marked-state count"))?;
    py.detach(|| native_grover_iteration_count(qubits, &marked))
        .map(|value| value.to_str_radix(10))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Canonicalize arbitrary-precision signed periodic offsets and coefficients.
type PeriodicShiftTermProjection = (Vec<String>, f64, f64);

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn normalize_periodic_shift_terms(
    py: Python<'_>,
    register_sizes: Vec<usize>,
    offsets: Vec<Vec<String>>,
    coefficients: Vec<(f64, f64)>,
) -> PyResult<Vec<PeriodicShiftTermProjection>> {
    let offsets = parse_periodic_offsets(offsets)?;
    let coefficients = coefficients
        .into_iter()
        .map(|(real, imaginary)| num_complex::Complex64::new(real, imaginary))
        .collect::<Vec<_>>();
    let decomposition = py
        .detach(|| normalize_native_periodic_shift_terms(&register_sizes, &offsets, &coefficients))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(periodic_shift_projection(&decomposition))
}

/// Recognize one exact dense periodic-shift matrix.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn decompose_periodic_shift_matrix(
    py: Python<'_>,
    register_sizes: Vec<usize>,
    shape: Vec<usize>,
    entries: Vec<(f64, f64)>,
) -> PyResult<Vec<PeriodicShiftTermProjection>> {
    let entries = entries
        .into_iter()
        .map(|(real, imaginary)| num_complex::Complex64::new(real, imaginary))
        .collect::<Vec<_>>();
    let decomposition = py
        .detach(|| decompose_native_periodic_shift_matrix(&register_sizes, &shape, &entries))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(periodic_shift_projection(&decomposition))
}

fn parse_periodic_offsets(offsets: Vec<Vec<String>>) -> PyResult<Vec<Vec<BigInt>>> {
    offsets
        .into_iter()
        .map(|offset| {
            offset
                .into_iter()
                .map(|component| {
                    BigInt::parse_bytes(component.as_bytes(), 16).ok_or_else(|| {
                        PyValueError::new_err("Q7534: invalid periodic shift offset integer")
                    })
                })
                .collect()
        })
        .collect()
}

fn periodic_shift_projection(
    decomposition: &PeriodicShiftDecomposition,
) -> Vec<PeriodicShiftTermProjection> {
    decomposition
        .terms()
        .iter()
        .map(|term| {
            let coefficient = term.coefficient();
            (
                term.offset()
                    .iter()
                    .map(|component| component.to_str_radix(16))
                    .collect(),
                coefficient.re,
                coefficient.im,
            )
        })
        .collect()
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<LcuCoefficientPlan>()?;
    module.add_class::<XorLookupPlan>()?;
    module.add_class::<SparseLookupPlan>()?;
    module.add_class::<ModularMultiplicationPlan>()?;
    module.add_class::<EkeraHastadPlan>()?;
    module.add_function(wrap_pyfunction!(mottonen_amplitude_coordinates, module)?)?;
    module.add_function(wrap_pyfunction!(mottonen_rotation_schedule, module)?)?;
    module.add_function(wrap_pyfunction!(
        uniformly_controlled_rotation_coordinates,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(cyclic_gray_control_schedule, module)?)?;
    module.add_function(wrap_pyfunction!(suzuki_product_formula_schedule, module)?)?;
    module.add_function(wrap_pyfunction!(plan_lcu_coefficients, module)?)?;
    module.add_function(wrap_pyfunction!(plan_xor_lookup, module)?)?;
    module.add_function(wrap_pyfunction!(plan_sparse_lookup, module)?)?;
    module.add_function(wrap_pyfunction!(grover_iteration_count, module)?)?;
    module.add_function(wrap_pyfunction!(wrapping_addition_offsets, module)?)?;
    module.add_function(wrap_pyfunction!(modular_addition_residue, module)?)?;
    module.add_function(wrap_pyfunction!(plan_modular_multiplication, module)?)?;
    module.add_function(wrap_pyfunction!(plan_ekera_hastad, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_periodic_shift_terms, module)?)?;
    module.add_function(wrap_pyfunction!(decompose_periodic_shift_matrix, module)?)?;
    Ok(())
}
