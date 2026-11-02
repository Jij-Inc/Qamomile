//! Narrow immutable projections for selected-map library constructions.

use num_complex::Complex64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use qamomile_compiler::{
    SignalContract as Contract, SignalError, SignalRegion, signal_embedding, signal_product,
    signal_sum,
};

#[allow(clippy::needless_pass_by_value)] // Result::map_err transfers the diagnostic.
fn error(value: SignalError) -> PyErr {
    PyValueError::new_err(value.to_string())
}

/// Native selected-map shape, scale, error, and construction-budget contract.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
pub(crate) struct SignalContract {
    contract: Contract,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl SignalContract {
    fn rescale(&self, scale: f64, max_error: f64) -> PyResult<Self> {
        Ok(Self {
            contract: self.contract.rescale(scale, max_error).map_err(error)?,
        })
    }
    #[getter]
    fn input(&self) -> &str {
        self.contract.input().pattern()
    }
    #[getter]
    fn output(&self) -> &str {
        self.contract.output().pattern()
    }
    #[getter]
    fn normalization(&self) -> f64 {
        self.contract.scale()
    }
    #[getter]
    fn error_bound(&self) -> f64 {
        self.contract.error()
    }
    #[getter]
    fn hermitian(&self) -> bool {
        self.contract.is_hermitian()
    }
    #[getter]
    fn nodes(&self) -> usize {
        self.contract.nodes()
    }
    #[getter]
    fn depth(&self) -> usize {
        self.contract.depth()
    }
    fn adjoint(&self) -> PyResult<Self> {
        Ok(Self {
            contract: self.contract.adjoint().map_err(error)?,
        })
    }
    fn prepared(&self) -> PyResult<Self> {
        Ok(Self {
            contract: self.contract.prepared().map_err(error)?,
        })
    }
    fn hermitian_part(&self) -> PyResult<Self> {
        Ok(Self {
            contract: self.contract.hermitian_part().map_err(error)?,
        })
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn signal_contract(
    input: &str,
    output: &str,
    scale: f64,
    error_bound: f64,
) -> PyResult<SignalContract> {
    Ok(SignalContract {
        contract: Contract::new(input, output, scale, error_bound).map_err(error)?,
    })
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn signal_product_plan(
    left: &SignalContract,
    right: &SignalContract,
) -> PyResult<(SignalContract, Vec<(usize, usize)>)> {
    let plan = signal_product(&left.contract, &right.contract).map_err(error)?;
    Ok((
        SignalContract {
            contract: plan.contract().clone(),
        },
        plan.swaps().to_vec(),
    ))
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::type_complexity)] // Tuple projection is confined to the PyO3 boundary.
fn signal_embedding_plan(pattern: &str) -> PyResult<(Vec<(usize, usize)>, Vec<usize>)> {
    let region = SignalRegion::new(pattern).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let plan = signal_embedding(&region);
    Ok((plan.swaps().to_vec(), plan.flips().to_vec()))
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value, clippy::type_complexity)]
fn signal_sum_plan(
    operands: Vec<PyRef<'_, SignalContract>>,
    rows: Vec<(usize, f64, f64)>,
) -> PyResult<(SignalContract, usize, usize, Vec<(usize, f64, f64)>)> {
    let operands = operands
        .iter()
        .map(|v| v.contract.clone())
        .collect::<Vec<_>>();
    let rows = rows
        .into_iter()
        .map(|(i, r, j)| (i, Complex64::new(r, j)))
        .collect::<Vec<_>>();
    let plan = signal_sum(&operands, &rows).map_err(error)?;
    let terms = plan
        .coefficients()
        .terms()
        .iter()
        .map(|t| (t.group(), t.amplitude(), t.phase()))
        .collect();
    Ok((
        SignalContract {
            contract: plan.contract().clone(),
        },
        plan.workspace(),
        plan.coefficients().signal_width(),
        terms,
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SignalEstimate>()?;
    module.add_function(wrap_pyfunction!(signal_estimate, module)?)?;
    module.add_function(wrap_pyfunction!(signal_schedule, module)?)?;
    module.add_function(wrap_pyfunction!(signal_schedule_response, module)?)?;
    module.add_class::<SignalResponse>()?;
    module.add_class::<SignalResponsePlan>()?;
    module.add_function(wrap_pyfunction!(signal_response, module)?)?;
    module.add_function(wrap_pyfunction!(signal_response_plan, module)?)?;
    module.add_class::<SignalContract>()?;
    module.add_function(wrap_pyfunction!(signal_contract, module)?)?;
    module.add_function(wrap_pyfunction!(signal_product_plan, module)?)?;
    module.add_function(wrap_pyfunction!(signal_embedding_plan, module)?)?;
    module.add_function(wrap_pyfunction!(signal_sum_plan, module)?)?;
    Ok(())
}

/// Success-conditioned measurement estimates; intervals assume independent shots.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct SignalEstimate {
    estimate: qamomile_compiler::SignalEstimate,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl SignalEstimate {
    #[getter]
    fn shots(&self) -> u64 {
        self.estimate.shots()
    }
    #[getter]
    fn successes(&self) -> u64 {
        self.estimate.successes()
    }
    #[getter]
    fn probability(&self) -> f64 {
        self.estimate.probability()
    }
    #[getter]
    fn probability_interval(&self) -> (f64, f64) {
        self.estimate.probability_interval()
    }
    #[getter]
    fn conditional(&self) -> Option<f64> {
        self.estimate.conditional()
    }
    #[getter]
    fn conditional_interval(&self) -> (f64, f64) {
        self.estimate.conditional_interval()
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn signal_estimate(
    rows: Vec<(Vec<u8>, u64)>,
    indices: Vec<usize>,
    confidence: f64,
) -> PyResult<SignalEstimate> {
    Ok(SignalEstimate {
        estimate: qamomile_compiler::signal_estimate(&rows, &indices, confidence).map_err(error)?,
    })
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn signal_schedule(phases: Vec<(f64, f64)>) -> PyResult<Vec<(f64, f64)>> {
    qamomile_compiler::signal_schedule(&phases).map_err(error)
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn signal_schedule_response(phases: Vec<(f64, f64)>, amplitude: f64) -> PyResult<(f64, f64)> {
    let value = qamomile_compiler::signal_schedule_response(&phases, amplitude).map_err(error)?;
    Ok((value.re, value.im))
}

/// Checked real polynomial target in normalized spectral coordinates.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct SignalResponse {
    spec: qamomile_compiler::SignalResponse,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl SignalResponse {
    #[allow(clippy::type_complexity)]
    fn lcu_terms(
        &self,
        source: &SignalContract,
        max_error: f64,
    ) -> PyResult<(Vec<(usize, f64)>, f64, f64)> {
        let plan = self
            .spec
            .lcu_terms(&source.contract, max_error)
            .map_err(error)?;
        Ok((plan.terms().to_vec(), plan.margin(), plan.tolerance()))
    }
    #[getter]
    fn coefficients(&self) -> Vec<f64> {
        self.spec.coefficients().to_vec()
    }
    #[getter]
    fn scale(&self) -> f64 {
        self.spec.scale()
    }
    #[getter]
    fn degree(&self) -> usize {
        self.spec.degree()
    }
    fn evaluate(&self, x: f64) -> PyResult<f64> {
        self.spec.evaluate(x).map_err(error)
    }
}

/// Native-replayed two-branch response certificate and error enclosure.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct SignalResponsePlan {
    plan: qamomile_compiler::SignalResponsePlan,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl SignalResponsePlan {
    #[getter]
    fn contract(&self) -> SignalContract {
        SignalContract {
            contract: self.plan.contract().clone(),
        }
    }
    #[getter]
    fn phases(&self) -> Vec<(f64, f64)> {
        self.plan.phases().to_vec()
    }
    #[getter]
    fn reverse(&self) -> bool {
        self.plan.reverse()
    }
    #[getter]
    fn synthesis_error(&self) -> f64 {
        self.plan.synthesis_error()
    }
    #[getter]
    fn input_error(&self) -> f64 {
        self.plan.input_error()
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn signal_response(coefficients: Vec<f64>, scale: f64) -> PyResult<SignalResponse> {
    Ok(SignalResponse {
        spec: qamomile_compiler::SignalResponse::new(&coefficients, scale).map_err(error)?,
    })
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn signal_response_plan(
    spec: &SignalResponse,
    source: &SignalContract,
    kind: &str,
    side: &str,
    max_error: f64,
    max_iterations: usize,
) -> PyResult<SignalResponsePlan> {
    Ok(SignalResponsePlan {
        plan: qamomile_compiler::signal_response_plan(
            &spec.spec,
            &source.contract,
            kind,
            side,
            max_error,
            max_iterations,
        )
        .map_err(error)?,
    })
}
