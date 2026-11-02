//! Thin projections of compiler-owned shot cohorts and closed child circuits.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};
use qamomile_compiler::{
    ShotExecutionState, ShotSeedPolicy, inspect_shot_execution_recipe as inspect_recipe,
    seal_shot_execution_recipe as seal_recipe, shot_segment_result_abi,
    validate_shot_segment_interface as validate_interface,
};

use crate::abi_api::circuit_launch_bindings;
use crate::circuit_api::{CircuitProjection, circuit_projection};
use crate::program_owner::program_owner;
use crate::realization_api::lowering_config;

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native", skip_from_py_object)]
struct ShotExecutionRequest {
    #[pyo3(get)]
    circuit: CircuitProjection,
    #[pyo3(get)]
    shots: u64,
    #[pyo3(get)]
    seed: u64,
    result_abi: Vec<u8>,
    interface: Vec<u8>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl ShotExecutionRequest {
    #[getter]
    fn result_abi<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.result_abi)
    }
    #[getter]
    fn interface<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.interface)
    }
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(module = "qamomile._qamomile_native")]
struct ShotExecutionHandle {
    state: ShotExecutionState,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl ShotExecutionHandle {
    #[new]
    #[pyo3(signature = (payload, bindings, shots, seed, realization_capabilities=None, lowering=None, seeded_execution=false))]
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        #[gen_stub(override_type(type_repr = "bytes | VerifiedProgramHandle"))] payload: &Bound<
            '_,
            PyAny,
        >,
        bindings: &Bound<'_, PyDict>,
        shots: u64,
        seed: u64,
        realization_capabilities: Option<Vec<String>>,
        lowering: Option<Vec<String>>,
        seeded_execution: bool,
    ) -> PyResult<Self> {
        let owner = program_owner(payload)?;
        let bindings = circuit_launch_bindings(bindings)?;
        // Host materialization replaces selectors before SDK emission. Inserting
        // device store slots beforehand would invalidate source cut positions.
        // The initial continuation evaluator requires finite inlined calls/loops;
        // retained runtime call/index state has no portable cohort meaning yet.
        let capabilities = realization_capabilities.map(|labels| {
            labels
                .into_iter()
                .filter(|label| {
                    !matches!(
                        label.as_str(),
                        "boolean_store" | "named_call" | "native_for" | "indexed_for"
                    )
                })
                .collect::<Vec<_>>()
        });
        let config = lowering_config(capabilities.as_deref(), lowering.as_deref())?;
        let program = owner.prepared()?;
        program
            .validate_launch_parameters(&bindings)
            .map_err(value_error)?;
        let circuit = program
            .static_circuit_with_lowering(config)
            .map_err(value_error)?;
        let abi = owner.verified.artifact_abi().map_err(value_error)?;
        let seed = if seeded_execution || seed != 0 {
            ShotSeedPolicy::Seeded(seed)
        } else {
            ShotSeedPolicy::ProviderDefault
        };
        let state =
            ShotExecutionState::new(circuit, bindings, abi.result_abi().to_vec(), shots, seed)
                .map_err(value_error)?;
        Ok(Self { state })
    }

    fn next_request(&mut self) -> PyResult<Option<ShotExecutionRequest>> {
        self.state
            .next_request()
            .map_err(value_error)?
            .map(|request| {
                let width = request.result_width();
                Ok(ShotExecutionRequest {
                    circuit: circuit_projection(
                        request.circuit(),
                        u32::try_from(width).map_err(value_error)?,
                    )?,
                    shots: request.shots(),
                    seed: request.seed(),
                    result_abi: shot_segment_result_abi(width).map_err(value_error)?,
                    interface: request.interface(),
                })
            })
            .transpose()
    }

    #[allow(clippy::needless_pass_by_value)] // PyO3 owns the extracted count rows.
    fn accept(&mut self, counts: Vec<(Vec<u8>, u64)>) -> PyResult<()> {
        self.state.accept(&counts).map_err(value_error)
    }

    fn finish(&self) -> PyResult<Vec<(Vec<u8>, u64)>> {
        self.state.finish().map_err(value_error)
    }

    #[getter]
    fn physical_shots(&self) -> u64 {
        self.state.physical_shots()
    }
    #[getter]
    fn submissions(&self) -> u64 {
        self.state.submissions()
    }
}

#[allow(clippy::needless_pass_by_value)] // map_err transfers heterogeneous native errors.
fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn seal_shot_execution_recipe<'py>(
    py: Python<'py>,
    target: &str,
    #[gen_stub(override_type(type_repr = "bytes"))] source: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    Ok(PyBytes::new(
        py,
        &seal_recipe(target, source).map_err(value_error)?,
    ))
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn inspect_shot_execution_recipe<'py>(
    py: Python<'py>,
    #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
) -> PyResult<(String, Bound<'py, PyBytes>)> {
    let recipe = inspect_recipe(payload).map_err(value_error)?;
    Ok((
        recipe.target().to_owned(),
        PyBytes::new(py, recipe.source()),
    ))
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn validate_shot_segment_interface(
    #[gen_stub(override_type(type_repr = "bytes | VerifiedProgramHandle"))] payload: &Bound<
        '_,
        PyAny,
    >,
    #[gen_stub(override_type(type_repr = "bytes"))] receipt: &[u8],
    #[gen_stub(override_type(type_repr = "bytes"))] parameter_abi: &[u8],
    #[gen_stub(override_type(type_repr = "bytes"))] result_abi: &[u8],
) -> PyResult<()> {
    let owner = program_owner(payload)?;
    validate_interface(&owner.verified, receipt, parameter_abi, result_abi).map_err(value_error)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ShotExecutionRequest>()?;
    module.add_class::<ShotExecutionHandle>()?;
    module.add_function(wrap_pyfunction!(validate_shot_segment_interface, module)?)?;
    module.add_function(wrap_pyfunction!(seal_shot_execution_recipe, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_shot_execution_recipe, module)?)?;
    Ok(())
}
