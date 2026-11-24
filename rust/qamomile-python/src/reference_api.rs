//! Reference projections and entry points for the native Python module.

use super::abi_api::circuit_launch_bindings;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyList};
use qamomile_compiler::{
    ReferenceExecution as NativeReferenceExecution, ReferenceValueError, StaticCircuitError,
    StaticSimulationError, TargetPreparedProgram, VerifiedProgram,
    sample_static_circuit as sample_native_static_circuit,
};

/// Resolve an omitted seed from operating-system entropy at the execution edge.
fn reference_seed(py: Python<'_>, seed: Option<u64>) -> PyResult<u64> {
    if let Some(seed) = seed {
        return Ok(seed);
    }
    let entropy = py.import("os")?.call_method1("urandom", (8,))?;
    let bytes: [u8; 8] = entropy.extract()?;
    Ok(u64::from_le_bytes(bytes))
}

/// One result from the unified native Reference execution boundary.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct ReferenceExecutionProjection {
    #[pyo3(get)]
    kind: String,
    counts: Vec<(Vec<u8>, u64)>,
    value: Option<Vec<u8>>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl ReferenceExecutionProjection {
    /// Return packed sample rows as bytes, matching `PyO3`'s `Vec<u8>` conversion.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "list[tuple[bytes, int]]"))]
    fn counts(&self) -> Vec<(Vec<u8>, u64)> {
        self.counts.clone()
    }

    /// Return the canonical portable literal as immutable bytes when present.
    #[getter]
    fn value<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.value
            .as_ref()
            .map(|value| PyBytes::new(py, value.as_slice()))
    }
}

/// Materialize and sample one verified static-circuit invocation in Rust.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction(signature = (payload, bindings, shots, seed=None))]
#[gen_stub(override_return_type(type_repr = "list[tuple[bytes, int]]"))]
fn sample_static_circuit_at(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
    bindings: &Bound<'_, PyDict>,
    shots: u64,
    seed: Option<u64>,
) -> PyResult<Vec<(Vec<u8>, u64)>> {
    let program = VerifiedProgram::from_bytes(payload)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let launch_bindings = circuit_launch_bindings(bindings)?;
    program
        .validate_launch_parameters(&launch_bindings)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let seed = reference_seed(py, seed)?;
    py.detach(move || {
        let circuit = program
            .static_circuit_for_invocation(&launch_bindings)
            .map_err(|error| match error {
                StaticCircuitError::Unsupported(_)
                | StaticCircuitError::LaunchMaterialization(_) => {
                    PyNotImplementedError::new_err(error.to_string())
                }
                _ => PyValueError::new_err(error.to_string()),
            })?;
        sample_native_static_circuit(&circuit, &launch_bindings, shots, seed)
            .map(|counts| {
                counts
                    .entries()
                    .iter()
                    .map(|(bits, count)| (bits.to_vec(), *count))
                    .collect()
            })
            .map_err(|error| match error {
                StaticSimulationError::Unsupported(_) => {
                    PyNotImplementedError::new_err(error.to_string())
                }
                _ => PyValueError::new_err(error.to_string()),
            })
    })
}

/// Evaluate one supported non-sampled Reference value invocation in Rust.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction(signature = (payload, bindings, seed = None))]
fn evaluate_reference_value_at<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
    bindings: &Bound<'_, PyDict>,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyBytes>> {
    let program = VerifiedProgram::from_bytes(payload)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let prepared = TargetPreparedProgram::from_verified(program)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let launch_bindings = circuit_launch_bindings(bindings)
        .map_err(|error| PyNotImplementedError::new_err(error.to_string()))?;
    let seed = reference_seed(py, seed)?;
    py.detach(move || prepared.reference_value_for_invocation(&launch_bindings, seed))
        .map(|value| PyBytes::new(py, &value))
        .map_err(|error| match error {
            ReferenceValueError::Unsupported(_) => {
                PyNotImplementedError::new_err(error.to_string())
            }
            ReferenceValueError::Invalid(_) | ReferenceValueError::WorkBudgetExceeded => {
                PyValueError::new_err(error.to_string())
            }
        })
}

/// Execute one sampled, classical, or hybrid Reference invocation in Rust.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction(signature = (payload, bindings, shots, seed, single_value = false))]
fn execute_reference_at(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "bytes | ReferenceProgramHandle"))] payload: &Bound<
        '_,
        PyAny,
    >,
    bindings: &Bound<'_, PyDict>,
    shots: u64,
    seed: Option<u64>,
    single_value: bool,
) -> PyResult<ReferenceExecutionProjection> {
    if let Ok(handle) = payload.extract::<PyRef<'_, ReferenceProgramHandle>>() {
        handle.execute(py, bindings, shots, seed, single_value)
    } else {
        ReferenceProgramHandle::new(payload.extract::<&[u8]>()?)?.execute(
            py,
            bindings,
            shots,
            seed,
            single_value,
        )
    }
}

/// Execute an ordered batch through one restored Reference program.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction(signature = (payload, bindings, shots, base_seed=None))]
fn execute_reference_batch_at(
    py: Python<'_>,
    #[gen_stub(override_type(type_repr = "bytes | ReferenceProgramHandle"))] payload: &Bound<
        '_,
        PyAny,
    >,
    bindings: &Bound<'_, PyList>,
    shots: u64,
    base_seed: Option<u64>,
) -> PyResult<Vec<ReferenceExecutionProjection>> {
    if let Ok(handle) = payload.extract::<PyRef<'_, ReferenceProgramHandle>>() {
        handle.execute_batch(py, bindings, shots, base_seed)
    } else {
        ReferenceProgramHandle::new(payload.extract::<&[u8]>()?)?
            .execute_batch(py, bindings, shots, base_seed)
    }
}

/// Process-local owner of one verified and target-prepared Reference program.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
struct ReferenceProgramHandle {
    prepared: TargetPreparedProgram,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl ReferenceProgramHandle {
    /// Verify a portable program once before retaining its typed native state.
    #[new]
    fn new(#[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8]) -> PyResult<Self> {
        let program = VerifiedProgram::from_bytes(payload)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            prepared: TargetPreparedProgram::from_verified(program)
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    /// Execute with fresh invocation and quantum state, borrowing the program.
    #[pyo3(signature = (bindings, shots, seed, single_value = false))]
    fn execute(
        &self,
        py: Python<'_>,
        bindings: &Bound<'_, PyDict>,
        shots: u64,
        seed: Option<u64>,
        single_value: bool,
    ) -> PyResult<ReferenceExecutionProjection> {
        if single_value && shots != 1 {
            return Err(PyValueError::new_err(
                "Q8145: value evaluation requires one shot",
            ));
        }
        let bindings = circuit_launch_bindings(bindings)?;
        let seed = reference_seed(py, seed)?;
        py.detach(move || {
            if single_value {
                self.prepared
                    .reference_value_for_invocation(&bindings, seed)
                    .map(NativeReferenceExecution::Value)
            } else {
                self.prepared.execute_reference(&bindings, shots, seed)
            }
        })
        .map(|execution| match execution {
            NativeReferenceExecution::Samples(counts) => ReferenceExecutionProjection {
                kind: "samples".to_owned(),
                counts: counts
                    .entries()
                    .iter()
                    .map(|(bits, count)| (bits.to_vec(), *count))
                    .collect(),
                value: None,
            },
            NativeReferenceExecution::Value(value) => ReferenceExecutionProjection {
                kind: "value".to_owned(),
                counts: Vec::new(),
                value: Some(value),
            },
        })
        .map_err(|error| match error {
            ReferenceValueError::Unsupported(_) => {
                PyNotImplementedError::new_err(error.to_string())
            }
            ReferenceValueError::Invalid(_) | ReferenceValueError::WorkBudgetExceeded => {
                PyValueError::new_err(error.to_string())
            }
        })
    }

    /// Execute a bounded batch without restoring the program again.
    #[pyo3(signature = (bindings, shots, base_seed=None))]
    fn execute_batch(
        &self,
        py: Python<'_>,
        bindings: &Bound<'_, PyList>,
        shots: u64,
        base_seed: Option<u64>,
    ) -> PyResult<Vec<ReferenceExecutionProjection>> {
        const MAX_REFERENCE_BATCH_INVOCATIONS: usize = 4096;

        let row_count = bindings.len();
        if row_count > MAX_REFERENCE_BATCH_INVOCATIONS {
            return Err(PyValueError::new_err(format!(
                "Q8145: Reference batch exceeds {MAX_REFERENCE_BATCH_INVOCATIONS} invocations"
            )));
        }
        let mut launch_rows = Vec::with_capacity(row_count);
        for item in bindings.iter() {
            let row = item.cast::<PyDict>().map_err(|_| {
                PyValueError::new_err("Q8145: Reference batch rows must be mappings")
            })?;
            launch_rows.push(circuit_launch_bindings(row)?);
        }
        let seeds = (0..row_count)
            .map(|index| {
                let seed = base_seed
                    .map(|seed| {
                        seed.checked_add(index as u64).ok_or_else(|| {
                            PyValueError::new_err("Q8145: Reference batch seed range overflows u64")
                        })
                    })
                    .transpose()?;
                reference_seed(py, seed)
            })
            .collect::<PyResult<Vec<_>>>()?;
        py.detach(move || {
            launch_rows
                .iter()
                .zip(seeds)
                .map(|(row, seed)| {
                    self.prepared
                        .execute_reference(row, shots, seed)
                        .map(|execution| match execution {
                            NativeReferenceExecution::Samples(counts) => {
                                ReferenceExecutionProjection {
                                    kind: "samples".to_owned(),
                                    counts: counts
                                        .entries()
                                        .iter()
                                        .map(|(bits, count)| (bits.to_vec(), *count))
                                        .collect(),
                                    value: None,
                                }
                            }
                            NativeReferenceExecution::Value(value) => {
                                ReferenceExecutionProjection {
                                    kind: "value".to_owned(),
                                    counts: Vec::new(),
                                    value: Some(value),
                                }
                            }
                        })
                        .map_err(|error| match error {
                            ReferenceValueError::Unsupported(_) => {
                                PyNotImplementedError::new_err(error.to_string())
                            }
                            ReferenceValueError::Invalid(_)
                            | ReferenceValueError::WorkBudgetExceeded => {
                                PyValueError::new_err(error.to_string())
                            }
                        })
                })
                .collect()
        })
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ReferenceExecutionProjection>()?;
    module.add_class::<ReferenceProgramHandle>()?;
    module.add_function(wrap_pyfunction!(sample_static_circuit_at, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_reference_value_at, module)?)?;
    module.add_function(wrap_pyfunction!(execute_reference_at, module)?)?;
    module.add_function(wrap_pyfunction!(execute_reference_batch_at, module)?)?;
    Ok(())
}
