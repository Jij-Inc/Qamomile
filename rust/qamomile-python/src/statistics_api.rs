//! Typed native projections for classical sample statistics.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use qamomile_compiler::{
    MAX_STATISTICS_PATH, MAX_STATISTICS_TERMS, SampleReadout as Readout,
    ScalarParameterPathComponent as Component, StatisticalEstimate as Estimate, StatisticsError,
    combine_independent_estimates, sample_mean,
};

use crate::results_api::SampleCountSummary;

#[allow(clippy::needless_pass_by_value)]
fn error(error: StatisticsError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Checked classical field/parity selector with no quantum-basis claim.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct StatisticsReadout {
    readout: Readout,
}

/// Immutable checked statistical result; construction is owned by Rust reductions.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct StatisticsEstimate {
    estimate: Estimate,
    sample_variance: Option<f64>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl StatisticsEstimate {
    #[getter]
    fn value(&self) -> f64 {
        self.estimate.value()
    }
    #[getter]
    fn standard_error(&self) -> Option<f64> {
        self.estimate.standard_error()
    }
    #[getter]
    fn sample_variance(&self) -> Option<f64> {
        self.sample_variance
    }
    #[getter]
    fn shots(&self) -> String {
        self.estimate.shots().to_str_radix(10)
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[pyo3(signature = (path, indices=None))]
fn statistics_readout(
    #[gen_stub(override_type(type_repr = "list[tuple[int | None, str | None]]"))] path: &Bound<
        '_,
        PyList,
    >,
    #[gen_stub(override_type(type_repr = "list[int] | None"))] indices: Option<&Bound<'_, PyList>>,
) -> PyResult<StatisticsReadout> {
    if path.len() > MAX_STATISTICS_PATH || indices.is_some_and(|i| i.len() > MAX_STATISTICS_TERMS) {
        return Err(error(StatisticsError::Budget));
    }
    let path: Vec<(Option<u64>, Option<String>)> = path.extract()?;
    let indices: Option<Vec<usize>> = indices.map(|i| i.extract()).transpose()?;
    let path = path
        .into_iter()
        .map(|component| match component {
            (Some(i), None) => Ok(Component::Index(i)),
            (None, Some(s)) => Ok(Component::Field(s)),
            _ => Err(PyValueError::new_err(
                "Q7400: invalid statistics path component",
            )),
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(StatisticsReadout {
        readout: match indices {
            None => Readout::value(path),
            Some(indices) => Readout::parity(path, indices),
        }
        .map_err(error)?,
    })
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn statistics_mean(
    py: Python<'_>,
    samples: PyRef<'_, SampleCountSummary>,
    #[gen_stub(override_type(type_repr = "bytes"))] result_abi: &[u8],
    #[gen_stub(override_type(type_repr = "list[tuple[float, StatisticsReadout]]"))] terms: &Bound<
        '_,
        PyList,
    >,
    offset: f64,
) -> PyResult<StatisticsEstimate> {
    if terms.len() > MAX_STATISTICS_TERMS {
        return Err(error(StatisticsError::Budget));
    }
    let terms: Vec<(f64, PyRef<'_, StatisticsReadout>)> = terms.extract()?;
    let terms = terms
        .iter()
        .map(|(c, r)| (*c, r.readout.clone()))
        .collect::<Vec<_>>();
    let samples = samples.summary();
    let value = py
        .detach(|| sample_mean(samples, result_abi, &terms, offset))
        .map_err(error)?;
    Ok(StatisticsEstimate {
        estimate: value.estimate().clone(),
        sample_variance: value.sample_variance(),
    })
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn statistics_combine_independent(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "list[tuple[float, StatisticsEstimate]]"))] terms: &Bound<
        '_,
        PyList,
    >,
    offset: f64,
) -> PyResult<StatisticsEstimate> {
    if terms.len() > MAX_STATISTICS_TERMS {
        return Err(error(StatisticsError::Budget));
    }
    let terms: Vec<(f64, PyRef<'_, StatisticsEstimate>)> = terms.extract()?;
    let terms = terms
        .iter()
        .map(|(c, e)| (*c, e.estimate.clone()))
        .collect::<Vec<_>>();
    let estimate = py
        .detach(|| combine_independent_estimates(&terms, offset))
        .map_err(error)?;
    Ok(StatisticsEstimate {
        estimate,
        sample_variance: None,
    })
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn statistics_from_summary(
    value: f64,
    standard_error: Option<f64>,
    shots: &str,
) -> PyResult<StatisticsEstimate> {
    if shots.len() > 1234 {
        return Err(error(StatisticsError::Budget));
    }
    let shots = shots.parse().map_err(|_| error(StatisticsError::Budget))?;
    Ok(StatisticsEstimate {
        estimate: Estimate::from_summary(value, standard_error, shots).map_err(error)?,
        sample_variance: None,
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<StatisticsReadout>()?;
    module.add_class::<StatisticsEstimate>()?;
    module.add_function(wrap_pyfunction!(statistics_readout, module)?)?;
    module.add_function(wrap_pyfunction!(statistics_mean, module)?)?;
    module.add_function(wrap_pyfunction!(statistics_from_summary, module)?)?;
    module.add_function(wrap_pyfunction!(statistics_combine_independent, module)?)?;
    Ok(())
}
