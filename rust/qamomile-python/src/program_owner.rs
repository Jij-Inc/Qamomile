//! Process-local ownership of checked source and target-prepared programs.

use std::sync::{Arc, OnceLock};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyTuple};
use qamomile_compiler::{
    GateRewriteRule, OptimizationPassIdentity, OptimizationPolicy, OptimizationPreparation,
    StandardOptimizationError, StandardOptimizationOptions, TargetPreparedProgram, VerifiedProgram,
    validate_optimization_pipeline,
};

/// Retained verified symbolic module; passes do not round-trip through bytes.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
pub(crate) struct VerifiedModuleHandle {
    verified: Arc<qamomile_compiler::VerifiedModule>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl VerifiedModuleHandle {
    /// Decode and verify an independent symbolic module once.
    #[new]
    fn new(#[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8]) -> PyResult<Self> {
        Ok(Self {
            verified: Arc::new(
                qamomile_compiler::VerifiedModule::from_bytes(payload).map_err(value_error)?,
            ),
        })
    }

    /// Export at a persistence or independent-provider boundary.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.verified.canonical_bytes())
    }

    /// Resolve a family entry through the retained native declaration table.
    fn resolve_entrypoint(&self, entrypoint: &str) -> PyResult<String> {
        self.verified
            .resolve_entrypoint(entrypoint)
            .map(str::to_owned)
            .map_err(value_error)
    }

    /// Apply the shared bounded CSE driver without evaluating symbolic extents.
    #[pyo3(signature = (max_work=1_000_000))]
    fn eliminate_common_expressions(&self, py: Python<'_>, max_work: u64) -> PyResult<Self> {
        let result = py
            .detach(|| self.verified.eliminate_common_expressions(max_work))
            .map_err(value_error)?;
        Ok(Self {
            verified: Arc::new(result.program),
        })
    }

    /// Check public declarations without serializing either native graph.
    fn validate_result(&self, result: &Self) -> PyResult<()> {
        self.verified
            .validate_optimization_result(&result.verified)
            .map_err(value_error)
    }
}

/// Verified `QCore` with either lazy standard or explicitly selected preparation.
pub(crate) struct ProgramOwner {
    pub(crate) verified: VerifiedProgram,
    preparation: Preparation,
}

/// Mutually exclusive preparation modes; an explicit pipeline never runs defaults.
enum Preparation {
    Standard(OnceLock<Result<TargetPreparedProgram, StandardOptimizationError>>),
    Selected(Box<OptimizationPreparation>),
}

impl ProgramOwner {
    fn from_bytes(payload: &[u8]) -> PyResult<Arc<Self>> {
        let verified = VerifiedProgram::from_bytes(payload)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Arc::new(Self::standard(verified)))
    }

    /// Own a verified candidate whose target preparation is the lazy standard pipeline.
    fn standard(verified: VerifiedProgram) -> Self {
        Self {
            verified,
            preparation: Preparation::Standard(OnceLock::new()),
        }
    }

    pub(crate) fn prepared(&self) -> PyResult<&TargetPreparedProgram> {
        match &self.preparation {
            Preparation::Selected(optimization) => Ok(optimization.prepared()),
            Preparation::Standard(cache) => cache
                .get_or_init(|| self.verified.prepare_standard().map(|(program, _)| program))
                .as_ref()
                .map_err(value_error),
        }
    }

    fn optimization(&self) -> Option<&OptimizationPreparation> {
        match &self.preparation {
            Preparation::Selected(optimization) => Some(optimization),
            Preparation::Standard(_) => None,
        }
    }
}

/// Verified candidate shared by ABI, planning, analysis and target projection.
/// Explicit preparation additionally retains its unchanged original source.
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pyclass(frozen, module = "qamomile._qamomile_native")]
pub(crate) struct VerifiedProgramHandle {
    owner: Arc<ProgramOwner>,
}

#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl VerifiedProgramHandle {
    /// Decode and fully verify untrusted persistent input once.
    #[new]
    fn new(#[gen_stub(override_type(type_repr = "bytes"))] payload: &[u8]) -> PyResult<Self> {
        Ok(Self {
            owner: ProgramOwner::from_bytes(payload)?,
        })
    }

    /// Export the current verified `QCore`, excluding its source association.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.owner.verified.canonical_bytes())
    }

    /// Reuse dominance-local classical CSE on the selected program.
    #[pyo3(signature = (max_work=1_000_000))]
    fn eliminate_common_expressions(&self, py: Python<'_>, max_work: u64) -> PyResult<Self> {
        let result = py
            .detach(|| self.owner.verified.eliminate_common_expressions(max_work))
            .map_err(value_error)?;
        Ok(Self::standard(result.program))
    }

    /// Reuse native standard simplification without invoking target preparation.
    #[pyo3(signature = (max_work=1_000_000, options=None))]
    fn simplify(
        &self,
        py: Python<'_>,
        max_work: u64,
        #[gen_stub(override_type(type_repr = "list[bool] | None"))] options: Option<
            &Bound<'_, PyList>,
        >,
    ) -> PyResult<Self> {
        let options = match options {
            None => StandardOptimizationOptions::default(),
            Some(flags) => simplification_options(flags)?,
        };
        let (verified, _) = py
            .detach(|| {
                self.owner
                    .verified
                    .optimize_standard_with_budget(max_work, options)
            })
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self::standard(verified))
    }

    /// Apply an author-asserted contracting unary rewrite through native SSA checks.
    #[pyo3(signature = (pattern, replacement, max_work=1_000_000))]
    fn rewrite(
        &self,
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "list[str]"))] pattern: &Bound<'_, PyList>,
        #[gen_stub(override_type(type_repr = "list[str]"))] replacement: &Bound<'_, PyList>,
        max_work: u64,
    ) -> PyResult<Self> {
        let rule = GateRewriteRule::new(gate_names(pattern)?, gate_names(replacement)?)
            .map_err(value_error)?;
        let verified = py
            .detach(|| rule.apply(&self.owner.verified, max_work))
            .map_err(value_error)?;
        Ok(Self::standard(verified))
    }

    /// Preflight stable pipeline identities before extension invocation.
    #[staticmethod]
    fn validate_pipeline(
        #[gen_stub(override_type(type_repr = "list[tuple[str, str, bytes]]"))] passes: &Bound<
            '_,
            PyList,
        >,
    ) -> PyResult<()> {
        validate_optimization_pipeline(&pass_identities(passes)?).map_err(value_error)
    }

    /// Associate a completely checked extension result with this source authority.
    #[pyo3(signature = (result, passes, seed=0, max_work=1_000_000, target=""))]
    fn with_optimization(
        &self,
        result: &Self,
        #[gen_stub(override_type(type_repr = "list[tuple[str, str, bytes]]"))] passes: &Bound<
            '_,
            PyList,
        >,
        seed: u64,
        max_work: u64,
        target: &str,
    ) -> PyResult<Self> {
        if target.len() > 4096 {
            return Err(value_error("Q7802: target snapshot exceeds budget"));
        }
        let policy =
            OptimizationPolicy::new(seed, max_work, target.to_owned()).map_err(value_error)?;
        let optimization = OptimizationPreparation::new(
            self.owner.verified.clone(),
            result.owner.verified.clone(),
            pass_identities(passes)?,
            &policy,
        )
        .map_err(value_error)?;
        Ok(Self::from_optimization(optimization))
    }

    /// Check an intermediate extension result without encoding or copying `QCore`.
    fn validate_result(&self, result: &Self) -> PyResult<()> {
        self.owner
            .verified
            .validate_optimization_result(&result.owner.verified)
            .map_err(value_error)
    }

    /// Preserve selected source and late policies through realization substitution.
    fn preserve_optimization(&self, result: &Self) -> PyResult<Self> {
        let optimization = self
            .owner
            .optimization()
            .ok_or_else(|| value_error("Q7802: no selected preparation"))?;
        Ok(Self::from_optimization(
            optimization
                .with_result(result.owner.verified.clone())
                .map_err(value_error)?,
        ))
    }

    /// Restore a portable preparation while retaining mandatory native checking.
    fn restore_optimization(
        &self,
        #[gen_stub(override_type(type_repr = "bytes"))] receipt: &[u8],
    ) -> PyResult<Self> {
        let optimization =
            OptimizationPreparation::from_receipt(self.owner.verified.clone(), receipt)
                .map_err(value_error)?;
        Ok(Self::from_optimization(optimization))
    }

    /// Export the source-associated selected result, excluding extension callbacks.
    fn optimization_receipt<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.owner
            .optimization()
            .map(|value| PyBytes::new(py, value.receipt()))
    }

    /// Return explicit stages, or None when ordinary target preparation is pending.
    fn optimization_stages(&self) -> Option<Vec<String>> {
        self.owner
            .optimization()
            .map(|value| value.stages().collect())
    }

    /// Return the authoritative original source semantic identity.
    fn compilation_source_digest<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let source = self
            .owner
            .optimization()
            .map_or(&self.owner.verified, OptimizationPreparation::source);
        PyBytes::new(py, &source.semantic_digest())
    }

    /// Return a checked canonical identity for the built-in simplifier options.
    #[staticmethod]
    fn simplification_configuration<'py>(
        py: Python<'py>,
        #[gen_stub(override_type(type_repr = "list[bool]"))] flags: &Bound<'_, PyList>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let options = simplification_options(flags)?;
        Ok(PyBytes::new(py, &options.configuration()))
    }
}

impl VerifiedProgramHandle {
    fn standard(verified: VerifiedProgram) -> Self {
        Self {
            owner: Arc::new(ProgramOwner::standard(verified)),
        }
    }

    fn from_optimization(optimization: OptimizationPreparation) -> Self {
        Self {
            owner: Arc::new(ProgramOwner {
                verified: optimization.prepared().as_verified().clone(),
                preparation: Preparation::Selected(Box::new(optimization)),
            }),
        }
    }
}

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn pass_identities(passes: &Bound<'_, PyList>) -> PyResult<Vec<OptimizationPassIdentity>> {
    if passes.len() > qamomile_compiler::MAX_OPTIMIZATION_PASSES {
        return Err(value_error("Q7802: pipeline exceeds 64 passes"));
    }
    passes
        .iter()
        .map(|item| {
            let row = item.cast::<PyTuple>()?;
            if row.len() != 3 {
                return Err(value_error("Q7802: invalid pass identity row"));
            }
            let name = row.get_item(0)?;
            let version = row.get_item(1)?;
            let configuration = row.get_item(2)?;
            let configuration = configuration.cast::<PyBytes>()?.as_bytes();
            if name.len()? > 128 || version.len()? > 128 || configuration.len() > 65_536 {
                return Err(value_error("Q7802: pass identity exceeds budget"));
            }
            OptimizationPassIdentity::new(
                name.extract()?,
                version.extract()?,
                configuration.to_vec(),
            )
            .map_err(value_error)
        })
        .collect()
}

fn simplification_options(flags: &Bound<'_, PyList>) -> PyResult<StandardOptimizationOptions> {
    if flags.len() != 5 {
        return Err(value_error("Q7802: expected five simplification flags"));
    }
    let flags = flags.extract::<Vec<bool>>()?;
    StandardOptimizationOptions::from_flags(&flags)
        .ok_or_else(|| value_error("Q7802: invalid simplification flags"))
}

fn gate_names(names: &Bound<'_, PyList>) -> PyResult<Vec<String>> {
    if names.is_empty() || names.len() > 16 {
        return Err(value_error("Q7802: rewrite requires one to sixteen gates"));
    }
    names
        .iter()
        .map(|name| {
            if name.len()? > 128 {
                return Err(value_error("Q7802: gate opcode exceeds budget"));
            }
            name.extract()
        })
        .collect()
}

/// Accept verified owners directly; portable bytes always undergo full checking.
pub(crate) fn program_owner(payload: &Bound<'_, PyAny>) -> PyResult<Arc<ProgramOwner>> {
    if let Ok(handle) = payload.extract::<PyRef<'_, VerifiedProgramHandle>>() {
        Ok(Arc::clone(&handle.owner))
    } else {
        ProgramOwner::from_bytes(payload.extract::<&[u8]>()?)
    }
}
