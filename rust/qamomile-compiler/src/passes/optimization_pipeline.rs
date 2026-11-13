//! Checked pipeline identities, source/result association, and late-pass policy.

use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{TargetPreparedProgram, VerifiedProgram, wire};

mod receipt;

/// Maximum number of explicitly ordered optimization passes.
pub const MAX_OPTIMIZATION_PASSES: usize = 64;
/// Maximum native rewrite operation visits per pass invocation.
pub(crate) const MAX_REWRITE_WORK: u64 = 1_000_000;
const MAX_CONFIGURATION_BYTES: usize = 65_536;
const MAX_REWRITE_DEPTH: usize = 256;
const CIRCUIT_SIMPLIFICATION_PASS: &str = "circuit.exact-local";
const STANDARD_SIMPLIFICATION_PASS: &str = "qcore.simplify";

/// Why a bounded rewrite cannot start on a retained graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WorkPreflightError {
    /// The budget is outside `1..=MAX_REWRITE_WORK` or smaller than the graph.
    Budget,
    /// Nested regions exceed the shared rewrite depth bound.
    Depth,
}

/// Charge every retained operation before cloning; symbolic extents are never
/// enumerated, so the cost is linear in the persisted graph.
pub(crate) fn preflight_work(
    package: &wire::QCorePackageProto,
    max_work: u64,
    cost_per_operation: u64,
) -> Result<(), WorkPreflightError> {
    if max_work == 0 || max_work > MAX_REWRITE_WORK {
        return Err(WorkPreflightError::Budget);
    }
    let mut remaining = max_work;
    let mut blocks = package
        .functions
        .iter()
        .filter_map(|function| function.body.as_ref())
        .map(|block| (block, 0))
        .collect::<Vec<_>>();
    while let Some((block, depth)) = blocks.pop() {
        if depth >= MAX_REWRITE_DEPTH {
            return Err(WorkPreflightError::Depth);
        }
        for operation in &block.operations {
            remaining = remaining
                .checked_sub(cost_per_operation)
                .ok_or(WorkPreflightError::Budget)?;
            for region in &operation.regions {
                blocks.extend(region.blocks.iter().map(|block| (block, depth + 1)));
            }
        }
    }
    Ok(())
}

impl From<WorkPreflightError> for OptimizationError {
    fn from(error: WorkPreflightError) -> Self {
        Self::invalid(match error {
            WorkPreflightError::Budget => "rewrite work budget exceeded",
            WorkPreflightError::Depth => "rewrite depth exceeds 256",
        })
    }
}

/// Invalid optimization configuration, association, or rewrite.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("Q7802: {0}")]
pub struct OptimizationError(String);

impl OptimizationError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

/// Immutable identity and opaque deterministic configuration of one pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OptimizationPassIdentity {
    name: String,
    version: String,
    configuration: Vec<u8>,
}

impl OptimizationPassIdentity {
    /// Validate a bounded pass identity before invoking extension code.
    ///
    /// # Errors
    /// Rejects malformed identifiers, oversized versions, and configurations.
    pub fn new(
        name: String,
        version: String,
        configuration: Vec<u8>,
    ) -> Result<Self, OptimizationError> {
        if name.is_empty()
            || name.len() > 128
            || !name.contains('.')
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
            || version.is_empty()
            || version.len() > 128
            || !version
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
            || configuration.len() > MAX_CONFIGURATION_BYTES
        {
            return Err(OptimizationError::invalid(
                "invalid pass identity or configuration budget",
            ));
        }
        Ok(Self {
            name,
            version,
            configuration,
        })
    }

    /// Return the stable public pass identifier.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the implementation version.
    #[must_use]
    pub fn version(&self) -> &str {
        &self.version
    }
}

/// Check an ordered pipeline before executing any local or remote pass.
///
/// # Errors
/// Rejects excessive or duplicate passes.
pub fn validate_optimization_pipeline(
    passes: &[OptimizationPassIdentity],
) -> Result<(), OptimizationError> {
    if passes.len() > MAX_OPTIMIZATION_PASSES {
        return Err(OptimizationError::invalid("pipeline exceeds 64 passes"));
    }
    let mut names = std::collections::HashSet::new();
    for pass in passes {
        if !names.insert(pass.name()) {
            return Err(OptimizationError::invalid(
                "duplicate optimization pass identifier",
            ));
        }
    }
    Ok(())
}

/// Native association of an original source with a selected verified result.
#[derive(Clone, Debug)]
pub struct OptimizationPreparation {
    source: VerifiedProgram,
    prepared: TargetPreparedProgram,
    passes: Vec<OptimizationPassIdentity>,
    receipt: Vec<u8>,
    policy: OptimizationPolicy,
}

/// Portable execution policy included even for an explicitly empty pipeline.
#[derive(Clone, Debug)]
pub struct OptimizationPolicy {
    seed: u64,
    max_work: u64,
    target: String,
}

/// Closed, portable late circuit policy restored without Python callbacks.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LateOptimization {
    exact_circuit: bool,
    max_work: u64,
}

impl LateOptimization {
    pub(crate) const NONE: Self = Self {
        exact_circuit: false,
        max_work: MAX_REWRITE_WORK,
    };

    fn from_passes(
        passes: &[OptimizationPassIdentity],
        max_work: u64,
    ) -> Result<Self, OptimizationError> {
        let mut exact_circuit = false;
        for pass in passes {
            if pass.name != CIRCUIT_SIMPLIFICATION_PASS {
                continue;
            }
            if pass.version != "1" || !pass.configuration.is_empty() {
                return Err(OptimizationError::invalid(
                    "unsupported late optimization version or configuration",
                ));
            }
            exact_circuit = true;
        }
        Ok(Self {
            exact_circuit,
            max_work,
        })
    }

    pub(crate) fn finish(
        self,
        circuit: crate::StaticCircuit,
    ) -> Result<crate::StaticCircuit, crate::StaticCircuitError> {
        if self.exact_circuit {
            circuit.simplify_exact(self.max_work)
        } else {
            Ok(circuit)
        }
    }
}

impl OptimizationPolicy {
    /// Check optimization policy bounds before producing portable provenance.
    ///
    /// # Errors
    /// Rejects invalid work budgets and oversized target snapshots.
    pub fn new(seed: u64, max_work: u64, target: String) -> Result<Self, OptimizationError> {
        if max_work == 0 || max_work > MAX_REWRITE_WORK || target.len() > 4096 {
            return Err(OptimizationError::invalid(
                "invalid optimization policy budget",
            ));
        }
        Ok(Self {
            seed,
            max_work,
            target,
        })
    }
}

impl OptimizationPreparation {
    /// Adopt an explicitly selected pipeline result without standard optimization.
    ///
    /// Graph checking does not prove equivalence to the source.
    ///
    /// # Errors
    /// Rejects changed public ABIs, entrypoints, or invalid pipeline identities.
    pub fn new(
        source: VerifiedProgram,
        result: VerifiedProgram,
        passes: Vec<OptimizationPassIdentity>,
        policy: &OptimizationPolicy,
    ) -> Result<Self, OptimizationError> {
        validate_optimization_pipeline(&passes)?;
        super::static_diagnostics::check(&source)
            .map_err(|error| OptimizationError::invalid(error.to_string()))?;
        validate_result(&source, &result)?;
        let late = LateOptimization::from_passes(&passes, policy.max_work)?;
        let receipt = receipt::encode(&source, &result, &passes, policy);
        Ok(Self {
            source,
            prepared: TargetPreparedProgram {
                verified: result,
                late,
            },
            passes,
            receipt,
            policy: policy.clone(),
        })
    }

    /// Restore a selected preparation at the native persistent boundary.
    ///
    /// # Errors
    /// Rejects corruption, budgets, source mismatch, invalid graphs, and ABI changes.
    pub fn from_receipt(
        source: VerifiedProgram,
        receipt: &[u8],
    ) -> Result<Self, OptimizationError> {
        let receipt::Decoded {
            result,
            passes,
            policy,
        } = receipt::decode(&source, receipt)?;
        Self::new(source, result, passes, &policy)
    }

    /// Borrow the unchanged source authority.
    #[must_use]
    pub const fn source(&self) -> &VerifiedProgram {
        &self.source
    }

    /// Borrow the selected target-facing result.
    #[must_use]
    pub const fn prepared(&self) -> &TargetPreparedProgram {
        &self.prepared
    }

    /// Borrow the canonical portable receipt, including the verified result.
    #[must_use]
    pub fn receipt(&self) -> &[u8] {
        &self.receipt
    }

    /// Retain source and late strategies across a separately checked realization.
    ///
    /// # Errors
    /// Rejects a changed public interface or invalid preparation policy.
    pub fn with_result(&self, result: VerifiedProgram) -> Result<Self, OptimizationError> {
        Self::new(
            self.source.clone(),
            result,
            self.passes.clone(),
            &self.policy,
        )
    }

    /// Return source, pipeline, configuration, and result provenance identity.
    #[must_use]
    pub fn identity(&self) -> [u8; 32] {
        Sha256::digest(&self.receipt).into()
    }

    /// Project only successfully invoked pass stages.
    pub fn stages(&self) -> impl Iterator<Item = String> + '_ {
        self.passes
            .iter()
            .filter(|pass| pass.name != CIRCUIT_SIMPLIFICATION_PASS)
            .map(|pass| {
                if pass.name == STANDARD_SIMPLIFICATION_PASS && pass.version == "1" {
                    "rust-qcore-standard-optimization".to_owned()
                } else {
                    format!("optimization:{}@{}", pass.name, pass.version)
                }
            })
    }
}

pub(crate) fn validate_result(
    source: &VerifiedProgram,
    result: &VerifiedProgram,
) -> Result<(), OptimizationError> {
    let left = source
        .artifact_abi()
        .map_err(|error| OptimizationError::invalid(error.to_string()))?;
    let right = result
        .artifact_abi()
        .map_err(|error| OptimizationError::invalid(error.to_string()))?;
    if source.local.header.entrypoint != result.local.header.entrypoint
        || left.parameter_abi() != right.parameter_abi()
        || left.result_abi() != right.result_abi()
    {
        return Err(OptimizationError::invalid(
            "optimization changed the entrypoint or public ABI",
        ));
    }
    Ok(())
}
