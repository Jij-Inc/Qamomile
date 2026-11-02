//! Results projections and entry points for the native Python module.

use super::abi_api::circuit_launch_bindings;
use super::conversion::{parse_signed_decimal, parse_width};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use qamomile_compiler::{
    NativeBitOrder, SampleCountSummary as NativeSampleCountSummary,
    decode_native_result_bitstring as decode_native_result_bitstring_impl,
    decode_result_value as decode_native_result_value,
    decode_sample_values as decode_native_sample_values,
    encode_result_value as encode_native_result_value,
    estimate_sample_parity as estimate_native_sample_parity,
    normalize_sample_counts as normalize_native_sample_counts,
    postprocess_static_counts as postprocess_native_static_counts,
    validate_result_value as validate_native_result_value,
};

/// Decode source-ordered logical bits through one canonical result ABI.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn decode_result_value<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] result_abi: &[u8],
    #[gen_stub(override_type(type_repr = "bytes"))] bits: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    decode_native_result_value(result_abi, bits)
        .map(|value| PyBytes::new(py, &value))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Validate one canonical portable value through its complete result ABI.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn validate_result_value(
    #[gen_stub(override_type(type_repr = "bytes"))] result_abi: &[u8],
    #[gen_stub(override_type(type_repr = "bytes"))] value: &[u8],
) -> PyResult<()> {
    validate_native_result_value(result_abi, value)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Decode one SDK-native textual bit row into source order.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn decode_native_result_bitstring(
    #[gen_stub(override_type(type_repr = "bytes"))] result_abi: &[u8],
    bitstring: &str,
    q0_first: bool,
) -> PyResult<Vec<u32>> {
    let order = if q0_first {
        NativeBitOrder::Q0First
    } else {
        NativeBitOrder::BigEndian
    };
    decode_native_result_bitstring_impl(result_abi, bitstring, order)
        .map(|bits| bits.into_iter().map(u32::from).collect())
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// One immutable native logical histogram summary.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
pub(super) struct SampleCountSummary {
    summary: NativeSampleCountSummary,
}

impl SampleCountSummary {
    /// Borrow the checked histogram without exposing mutable state.
    pub(super) fn summary(&self) -> &NativeSampleCountSummary {
        &self.summary
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl SampleCountSummary {
    /// Return canonical rows with arbitrary-precision decimal counts.
    #[getter]
    fn entries(&self) -> Vec<(Vec<u32>, String)> {
        self.summary
            .entries()
            .iter()
            .map(|(bits, count)| {
                (
                    bits.iter().copied().map(u32::from).collect(),
                    count.to_str_radix(10),
                )
            })
            .collect()
    }

    /// Return the exact total observation count as decimal text.
    #[getter]
    fn shots(&self) -> String {
        self.summary.shots().to_str_radix(10)
    }

    /// Return the deterministic most-frequent logical row.
    #[getter]
    fn most_likely(&self) -> Vec<u32> {
        self.summary
            .most_likely()
            .iter()
            .copied()
            .map(u32::from)
            .collect()
    }

    /// Return an overflow-safe empirical probability for one logical row.
    fn probability(
        &self,
        #[gen_stub(override_type(type_repr = "bytes"))] bits: &[u8],
    ) -> PyResult<f64> {
        self.summary.probability(bits).ok_or_else(|| {
            PyValueError::new_err("Q8130: probability row disagrees with the sample width")
        })
    }

    /// Decode every logical row through one canonical result ABI.
    fn decode_values<'py>(
        &self,
        py: Python<'py>,
        #[gen_stub(override_type(type_repr = "bytes"))] result_abi: &[u8],
    ) -> PyResult<Vec<(Bound<'py, PyBytes>, String)>> {
        decode_native_sample_values(result_abi, &self.summary)
            .map(|values| {
                values
                    .into_iter()
                    .map(|(value, count)| (PyBytes::new(py, &value), count.to_str_radix(10)))
                    .collect()
            })
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Estimate one Pauli-Z product without reprojecting the histogram.
    #[allow(clippy::needless_pass_by_value)]
    fn expectation_z(&self, factors: Vec<usize>) -> PyResult<f64> {
        let estimate = estimate_native_sample_parity(&self.summary, &factors)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(estimate)
    }
}

/// Normalize logical sample rows and derive their shared summary.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn normalize_sample_counts(
    py: Python<'_>,
    rows: Vec<(Vec<u8>, String)>,
    expected_width: Option<&str>,
) -> PyResult<SampleCountSummary> {
    let expected_width = expected_width
        .map(|value| parse_width(value, "Q8110: invalid expected sample width"))
        .transpose()?;
    let rows = rows
        .into_iter()
        .map(|(bits, count)| {
            parse_signed_decimal(&count, "Q8112: invalid sampling count")
                .map(|count| (bits.into_boxed_slice(), count))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let summary = py
        .detach(|| normalize_native_sample_counts(&rows, expected_width))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(SampleCountSummary { summary })
}

/// Encode one canonical portable value through one canonical result ABI.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn encode_result_value<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] result_abi: &[u8],
    #[gen_stub(override_type(type_repr = "bytes"))] value: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    encode_native_result_value(result_abi, value)
        .map(|bits| PyBytes::new(py, &bits))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Evaluate one backend distribution through a sealed native result plan.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[gen_stub(override_return_type(type_repr = "list[tuple[bytes, int]]"))]
fn postprocess_circuit_counts(
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
    counts: Vec<(Vec<u8>, u64)>,
    bindings: &Bound<'_, PyDict>,
    #[gen_stub(override_type(type_repr = "bytes"))] result_abi: &[u8],
) -> PyResult<Vec<(Vec<u8>, u64)>> {
    let counts = counts
        .into_iter()
        .map(|(bits, count)| (bits.into_boxed_slice(), count))
        .collect::<Vec<_>>();
    let bindings = circuit_launch_bindings(bindings)?;
    postprocess_native_static_counts(payload, &counts, &bindings, result_abi)
        .map(|counts| {
            counts
                .entries()
                .iter()
                .map(|(bits, count)| (bits.to_vec(), *count))
                .collect()
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SampleCountSummary>()?;
    module.add_function(wrap_pyfunction!(normalize_sample_counts, module)?)?;
    module.add_function(wrap_pyfunction!(decode_native_result_bitstring, module)?)?;
    module.add_function(wrap_pyfunction!(decode_result_value, module)?)?;
    module.add_function(wrap_pyfunction!(encode_result_value, module)?)?;
    module.add_function(wrap_pyfunction!(validate_result_value, module)?)?;
    module.add_function(wrap_pyfunction!(postprocess_circuit_counts, module)?)?;
    Ok(())
}
