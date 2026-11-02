//! Qsvt projections and entry points for the native Python module.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use qamomile_compiler::{
    CertifiedQsvtPolynomial as NativeCertifiedQsvtPolynomial,
    CertifiedReciprocalQsvt as NativeCertifiedReciprocalQsvt, SignalRegion as NativeSignalRegion,
    convert_qsp_wx_phases as convert_native_qsp_wx_phases,
    synthesize_qsvt_polynomial as synthesize_native_qsvt_polynomial,
    synthesize_reciprocal_qsvt as synthesize_native_reciprocal_qsvt,
    verify_qsvt_polynomial as verify_native_qsvt_polynomial,
    verify_reciprocal_qsvt as verify_native_reciprocal_qsvt,
};

/// Immutable native layout of a fixed-basis signal region.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct SignalRegion {
    region: NativeSignalRegion,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl SignalRegion {
    /// Return the source-ordered basis pattern.
    #[getter]
    fn pattern(&self) -> &str {
        self.region.pattern()
    }

    /// Return the complete coherent width.
    #[getter]
    fn width(&self) -> usize {
        self.region.width()
    }

    /// Return source-ordered free coordinates.
    #[getter]
    fn payload_indices(&self) -> Vec<usize> {
        self.region.payload_indices().to_vec()
    }

    /// Return the phase target, absent for the full region.
    #[getter]
    fn target(&self) -> Option<usize> {
        self.region.target()
    }

    /// Return whether the target is conjugated with X.
    #[getter]
    fn target_is_zero(&self) -> bool {
        self.region.target_is_zero()
    }

    /// Return source-ordered control positions after target removal.
    #[getter]
    fn controls(&self) -> Vec<usize> {
        self.region.controls().to_vec()
    }

    /// Return the size of the routed fixed-basis prefix.
    #[getter]
    fn phase_width(&self) -> usize {
        self.region.phase_width()
    }

    /// Return the bounded coordinate exchanges for ordinary source composition.
    #[getter]
    fn routing_swaps(&self) -> Vec<(usize, usize)> {
        self.region
            .routing_swaps()
            .iter()
            .map(|swap| (swap.left(), swap.right()))
            .collect()
    }

    /// Return the arbitrary-width control state as decimal text.
    #[getter]
    fn control_state(&self) -> String {
        self.region.control_state().to_string()
    }
}

/// Check one bounded basis pattern before allocating its native layout.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn signal_region(pattern: &str) -> PyResult<SignalRegion> {
    let region = NativeSignalRegion::new(pattern)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(SignalRegion { region })
}

/// Convert finite QSP `Wx` phases into canonical projector coordinates.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn convert_qsp_wx_phases(py: Python<'_>, phases: Vec<f64>) -> PyResult<Vec<f64>> {
    py.detach(|| convert_native_qsp_wx_phases(&phases))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// One immutable native certificate for a real QSVT polynomial.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct QsvtPolynomialCertificate {
    certificate: NativeCertifiedQsvtPolynomial,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl QsvtPolynomialCertificate {
    /// Return ascending first-kind Chebyshev coefficients.
    #[getter]
    fn chebyshev_coefficients(&self) -> Vec<f64> {
        self.certificate.chebyshev_coefficients().to_vec()
    }

    /// Return the replayed symmetric QSP `Wx` phases.
    #[getter]
    fn qsp_phases(&self) -> Vec<f64> {
        self.certificate.qsp_phases().to_vec()
    }

    /// Return canonical projector phases consumed by Qamomile.
    #[getter]
    fn projector_phases(&self) -> Vec<f64> {
        self.certificate.projector_phases().to_vec()
    }

    /// Return the target polynomial degree and block-encoding query count.
    #[getter]
    fn degree(&self) -> usize {
        self.certificate.degree()
    }

    /// Return whether the target polynomial is odd.
    #[getter]
    fn odd(&self) -> bool {
        self.certificate.is_odd()
    }

    /// Return the proved uniform approximation error over `[-1, 1]`.
    #[getter]
    fn uniform_error_bound(&self) -> f64 {
        self.certificate.uniform_error_bound()
    }

    /// Return the coefficient-L1 target magnitude bound.
    #[getter]
    fn target_magnitude_bound(&self) -> f64 {
        self.certificate.target_magnitude_bound()
    }

    /// Return the proved realized block magnitude bound.
    #[getter]
    fn realized_magnitude_bound(&self) -> f64 {
        self.certificate.realized_magnitude_bound()
    }

    /// Return the exact block-encoding query count.
    #[getter]
    fn queries(&self) -> usize {
        self.certificate.degree()
    }

    /// Return the exact projector-rotation count.
    #[getter]
    fn projector_rotations(&self) -> usize {
        self.certificate.degree() + 1
    }

    /// Return the postselection contract for the transformed block.
    #[getter]
    fn success(&self) -> &'static str {
        if self.certificate.is_heralded() {
            "heralded"
        } else {
            "deterministic"
        }
    }
}

/// One immutable native certificate for a reciprocal singular-value transform.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct ReciprocalQsvtCertificate {
    certificate: NativeCertifiedReciprocalQsvt,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl ReciprocalQsvtCertificate {
    /// Return the block-encoding normalization used by synthesis.
    #[getter]
    fn normalization(&self) -> f64 {
        self.certificate.normalization()
    }

    /// Return the promised physical minimum singular value.
    #[getter]
    fn minimum_singular_value(&self) -> f64 {
        self.certificate.minimum_singular_value()
    }

    /// Return the normalized minimum singular value.
    #[getter]
    fn normalized_gap(&self) -> f64 {
        self.certificate.normalized_gap()
    }

    /// Return the numerator of the normalized reciprocal target.
    #[getter]
    fn reciprocal_scale(&self) -> f64 {
        self.certificate.reciprocal_scale()
    }

    /// Return the numerator multiplying physical reciprocal singular values.
    #[getter]
    fn operator_scale(&self) -> f64 {
        self.certificate.operator_scale()
    }

    /// Return the proved polynomial approximation error.
    #[getter]
    fn approximation_error_bound(&self) -> f64 {
        self.certificate.approximation_error_bound()
    }

    /// Return the combined approximation and phase-realization error.
    #[getter]
    fn total_error_bound(&self) -> f64 {
        self.certificate.total_error_bound()
    }

    /// Return the nested independently replayed QSVT polynomial certificate.
    #[getter]
    fn polynomial(&self) -> QsvtPolynomialCertificate {
        QsvtPolynomialCertificate {
            certificate: self.certificate.polynomial().clone(),
        }
    }
}

/// Synthesize and certify one real Chebyshev polynomial for QSVT.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn synthesize_qsvt_polynomial(
    py: Python<'_>,
    chebyshev_coefficients: Vec<f64>,
    max_error: f64,
    max_iterations: usize,
) -> PyResult<QsvtPolynomialCertificate> {
    let certificate = py
        .detach(|| {
            synthesize_native_qsvt_polynomial(&chebyshev_coefficients, max_error, max_iterations)
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(QsvtPolynomialCertificate { certificate })
}

/// Replay one symmetric QSP candidate and its uniform error bound.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn verify_qsvt_polynomial(
    py: Python<'_>,
    chebyshev_coefficients: Vec<f64>,
    qsp_phases: Vec<f64>,
    max_error: f64,
) -> PyResult<QsvtPolynomialCertificate> {
    let certificate = py
        .detach(|| verify_native_qsvt_polynomial(&chebyshev_coefficients, &qsp_phases, max_error))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(QsvtPolynomialCertificate { certificate })
}

/// Synthesize and certify one reciprocal singular-value transform.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn synthesize_reciprocal_qsvt(
    py: Python<'_>,
    normalization: f64,
    minimum_singular_value: f64,
    max_error: f64,
    max_degree: usize,
    max_iterations: usize,
) -> PyResult<ReciprocalQsvtCertificate> {
    let certificate = py
        .detach(|| {
            synthesize_native_reciprocal_qsvt(
                normalization,
                minimum_singular_value,
                max_error,
                max_degree,
                max_iterations,
            )
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(ReciprocalQsvtCertificate { certificate })
}

/// Replay one reciprocal polynomial and phase certificate.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn verify_reciprocal_qsvt(
    py: Python<'_>,
    normalization: f64,
    minimum_singular_value: f64,
    reciprocal_scale: f64,
    chebyshev_coefficients: Vec<f64>,
    qsp_phases: Vec<f64>,
    max_error: f64,
) -> PyResult<ReciprocalQsvtCertificate> {
    let certificate = py
        .detach(|| {
            verify_native_reciprocal_qsvt(
                normalization,
                minimum_singular_value,
                reciprocal_scale,
                &chebyshev_coefficients,
                &qsp_phases,
                max_error,
            )
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(ReciprocalQsvtCertificate { certificate })
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SignalRegion>()?;
    module.add_class::<QsvtPolynomialCertificate>()?;
    module.add_class::<ReciprocalQsvtCertificate>()?;
    module.add_function(wrap_pyfunction!(signal_region, module)?)?;
    module.add_function(wrap_pyfunction!(convert_qsp_wx_phases, module)?)?;
    module.add_function(wrap_pyfunction!(synthesize_reciprocal_qsvt, module)?)?;
    module.add_function(wrap_pyfunction!(synthesize_qsvt_polynomial, module)?)?;
    module.add_function(wrap_pyfunction!(verify_reciprocal_qsvt, module)?)?;
    module.add_function(wrap_pyfunction!(verify_qsvt_polynomial, module)?)?;
    Ok(())
}
