//! Checked native projections for endpoint shot conventions.

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use qamomile_compiler::CurrentShotPolicy;

/// Immutable endpoint normalization and exact target integer limits.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
pub(crate) struct NativeCurrentShotPolicy {
    #[pyo3(get)]
    origin: u64,
    #[pyo3(get)]
    integer_min: i64,
    #[pyo3(get)]
    integer_max: i64,
    #[pyo3(get)]
    max_shots: u64,
}

/// Validate one explicit native origin and expose its checked numeric policy.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn current_shot_policy(origin: u64) -> PyResult<NativeCurrentShotPolicy> {
    let policy =
        CurrentShotPolicy::new(origin).map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(NativeCurrentShotPolicy {
        origin: policy.origin(),
        integer_min: policy.integer_min(),
        integer_max: policy.integer_max(),
        max_shots: policy.max_shots(),
    })
}

/// Reject an unsupported native repetition count before submission.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
fn check_current_shot_invocation(origin: u64, shots: u64) -> PyResult<()> {
    CurrentShotPolicy::new(origin)
        .and_then(|policy| policy.validate(shots))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCurrentShotPolicy>()?;
    module.add_class::<NativeShotCircuitTemplate>()?;
    module.add_class::<NativeShotLaunchRows>()?;
    module.add_function(wrap_pyfunction!(inspect_indexed_circuit, module)?)?;
    module.add_function(wrap_pyfunction!(current_shot_policy, module)?)?;
    module.add_function(wrap_pyfunction!(check_current_shot_invocation, module)?)?;
    Ok(())
}

/// Immutable checked row sequence with no retained per-shot Python objects.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
pub(crate) struct NativeShotLaunchRows {
    owner: qamomile_compiler::ShotLaunchRows,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl NativeShotLaunchRows {
    /// Return the checked number of logical rows.
    fn __len__(&self) -> usize {
        self.owner.row_count()
    }

    /// Return the uniform compiler slot count without visiting rows.
    #[getter]
    fn slot_count(&self) -> usize {
        self.owner.slot_count()
    }

    /// Materialize only the requested SDK row, supporting negative indices.
    fn __getitem__(&self, index: isize) -> PyResult<Vec<f64>> {
        let index = if index < 0 {
            self.owner.row_count().checked_add_signed(index)
        } else {
            usize::try_from(index).ok()
        };
        index
            .and_then(|index| self.owner.row(index))
            .map(<[f64]>::to_vec)
            .ok_or_else(|| PyIndexError::new_err("shot row index out of range"))
    }
}

/// Immutable native owner of one fixed-topology numeric launch plan.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
pub(crate) struct NativeShotCircuitTemplate {
    owner: qamomile_compiler::ShotCircuitTemplate,
    projection: Option<super::circuit_api::CircuitProjection>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl NativeShotCircuitTemplate {
    /// Borrow the compiler-certified SDK emission view once at compilation.
    #[getter]
    fn projection(&self) -> PyResult<super::circuit_api::CircuitProjection> {
        self.projection.clone().ok_or_else(|| {
            PyValueError::new_err("Q4481: restored scalar plan has no SDK emission view")
        })
    }

    /// Return compiler numeric slot count independently of SDK parameter order.
    #[getter]
    fn slot_count(&self) -> usize {
        self.owner.slot_count()
    }

    /// Check invocation work and numeric storage without allocating rows.
    fn check_invocation(&self, shots: u64) -> PyResult<()> {
        self.owner.check_invocation(shots).map_err(value_error)
    }

    /// Evaluate all rows through native checked numeric semantics.
    fn bind_rows(
        &self,
        bindings: &Bound<'_, pyo3::types::PyDict>,
        shots: u64,
    ) -> PyResult<NativeShotLaunchRows> {
        let bindings = super::abi_api::circuit_launch_bindings(bindings)?;
        self.owner
            .bind_rows(&bindings, shots)
            .map(|owner| NativeShotLaunchRows { owner })
            .map_err(value_error)
    }

    /// Validate complete logical coordinates before native result evaluation.
    #[gen_stub(override_return_type(type_repr = "list[tuple[bytes, int]]"))]
    fn postprocess(
        &self,
        indexed_rows: Vec<(u64, Vec<u8>, u64)>,
        bindings: &Bound<'_, pyo3::types::PyDict>,
        shots: u64,
        #[gen_stub(override_type(type_repr = "bytes"))] result_abi: &[u8],
    ) -> PyResult<Vec<(Vec<u8>, u64)>> {
        let rows = indexed_rows
            .into_iter()
            .map(|(index, bits, count)| (index, bits.into_boxed_slice(), count))
            .collect::<Vec<_>>();
        let bindings = super::abi_api::circuit_launch_bindings(bindings)?;
        qamomile_compiler::postprocess_indexed_counts(
            &self.owner.result_plan().map_err(value_error)?,
            &rows,
            &bindings,
            shots,
            result_abi,
        )
        .map(|counts| {
            counts
                .entries()
                .iter()
                .map(|(bits, count)| (bits.to_vec(), *count))
                .collect()
        })
        .map_err(value_error)
    }

    /// Evaluate every unmeasured coordinate without allocating empty input rows.
    #[gen_stub(override_return_type(type_repr = "list[tuple[bytes, int]]"))]
    fn postprocess_unmeasured(
        &self,
        bindings: &Bound<'_, pyo3::types::PyDict>,
        shots: u64,
        #[gen_stub(override_type(type_repr = "bytes"))] result_abi: &[u8],
    ) -> PyResult<Vec<(Vec<u8>, u64)>> {
        self.owner.check_invocation(shots).map_err(value_error)?;
        let bindings = super::abi_api::circuit_launch_bindings(bindings)?;
        qamomile_compiler::postprocess_unmeasured_indexed_counts(
            &self.owner.result_plan().map_err(value_error)?,
            &bindings,
            shots,
            result_abi,
        )
        .map(|counts| {
            counts
                .entries()
                .iter()
                .map(|(bits, count)| (bits.to_vec(), *count))
                .collect()
        })
        .map_err(value_error)
    }

    /// Seal native scalar plans with enclosing artifact identities.
    fn to_bytes<'py>(
        &self,
        py: Python<'py>,
        #[gen_stub(override_type(type_repr = "bytes"))] parameter_abi: &[u8],
        #[gen_stub(override_type(type_repr = "bytes"))] result_abi: &[u8],
        target_fingerprint: &str,
        source_digest: &str,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        self.owner
            .to_bytes(parameter_abi, result_abi, target_fingerprint, source_digest)
            .map(|payload| pyo3::types::PyBytes::new(py, &payload))
            .map_err(value_error)
    }

    /// Restore a persisted native evaluator after matching enclosing identities.
    #[staticmethod]
    fn from_bytes(
        #[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8],
        #[gen_stub(override_type(type_repr = "bytes"))] parameter_abi: &[u8],
        #[gen_stub(override_type(type_repr = "bytes"))] result_abi: &[u8],
        target_fingerprint: &str,
        source_digest: &str,
    ) -> PyResult<Self> {
        qamomile_compiler::ShotCircuitTemplate::from_bytes(
            payload,
            parameter_abi,
            result_abi,
            target_fingerprint,
            source_digest,
        )
        .map(|owner| Self {
            owner,
            projection: None,
        })
        .map_err(value_error)
    }
}

/// Select a native fixed-topology plan, or retain the checked source recipe.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyfunction]
#[pyo3(signature = (payload, realization_capabilities=None, lowering=None))]
#[allow(clippy::needless_pass_by_value)]
fn inspect_indexed_circuit(
    #[gen_stub(override_type(type_repr = "bytes | VerifiedProgramHandle"))] payload: &Bound<
        '_,
        PyAny,
    >,
    realization_capabilities: Option<Vec<String>>,
    lowering: Option<Vec<String>>,
) -> PyResult<Option<NativeShotCircuitTemplate>> {
    let program = super::program_owner::program_owner(payload)?;
    let lowering = super::realization_api::lowering_config(
        realization_capabilities.as_deref(),
        lowering.as_deref(),
    )?;
    let abi = program.verified.artifact_abi().map_err(value_error)?;
    match program
        .prepared()?
        .indexed_circuit_plan(lowering)
        .map_err(value_error)?
    {
        qamomile_compiler::ShotCircuitPlan::Recipe(_) => Ok(None),
        qamomile_compiler::ShotCircuitPlan::Template(owner) => {
            let projection =
                super::circuit_api::circuit_projection(owner.circuit(), abi.sampled_width())?;
            Ok(Some(NativeShotCircuitTemplate {
                owner: *owner,
                projection: Some(projection),
            }))
        }
    }
}

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}
