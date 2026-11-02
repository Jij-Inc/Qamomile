//! Target profile validation entry point for the native Python module.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use qamomile_compiler::validate_target_profile as validate_native_target_profile;

/// Validate one target profile and return its canonical bytes and digest.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn validate_target_profile<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let profile = py
        .detach(|| validate_native_target_profile(payload))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        PyBytes::new(py, profile.canonical_bytes()),
        PyBytes::new(py, profile.digest()),
    ))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(validate_target_profile, module)?)?;
    Ok(())
}
