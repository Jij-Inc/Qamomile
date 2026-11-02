//! Target capability planning over compiler-derived entry requirements.
//!
//! Target identities and payload formats remain host/plugin protocol facts.
//! This module owns the semantic compatibility question: whether one closed
//! target capability set contains every requirement inferred for the verified
//! entrypoint.

use thiserror::Error;

use crate::{Requirement, RequirementSet, VerifiedProgram};

/// Immutable compatibility result for one verified program and target profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TargetCapabilityPlan {
    required: RequirementSet,
    supported: RequirementSet,
    missing: RequirementSet,
}

/// One compiler-selected lowering route for a target capability profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TargetLoweringRoute {
    /// Lower the verified program directly to the target.
    Direct,
    /// Emit a finite family selected by a launch-time host predicate.
    CircuitFamily,
    /// Seal the program for exact launch-time materialization.
    LaunchMaterialized,
    /// Execute source-proved closed circuits with correlated shot cohorts.
    ShotExecution,
    /// Reject because some required capability cannot be discharged.
    Unsupported,
}

impl TargetLoweringRoute {
    /// Return the stable transient bridge spelling of this route.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::CircuitFamily => "circuit_family",
            Self::LaunchMaterialized => "launch_materialized",
            Self::ShotExecution => "shot_execution",
            Self::Unsupported => "unsupported",
        }
    }
}

/// Complete transient lowering decision derived from one target profile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TargetLoweringPlan {
    capabilities: TargetCapabilityPlan,
    materialized: RequirementSet,
    unsupported: RequirementSet,
    route: TargetLoweringRoute,
}

impl TargetLoweringPlan {
    /// Return the underlying requirement compatibility facts.
    #[must_use]
    pub const fn capabilities(self) -> TargetCapabilityPlan {
        self.capabilities
    }

    /// Return missing requirements discharged by the selected launch route.
    #[must_use]
    pub const fn materialized(self) -> RequirementSet {
        self.materialized
    }

    /// Return requirements with no implementation in this target profile.
    #[must_use]
    pub const fn unsupported(self) -> RequirementSet {
        self.unsupported
    }

    /// Return the compiler-selected transient lowering route.
    #[must_use]
    pub const fn route(self) -> TargetLoweringRoute {
        self.route
    }
}

impl TargetCapabilityPlan {
    /// Return the compiler-derived entrypoint requirements.
    #[must_use]
    pub const fn required(self) -> RequirementSet {
        self.required
    }

    /// Return the target capabilities supplied to planning.
    #[must_use]
    pub const fn supported(self) -> RequirementSet {
        self.supported
    }

    /// Return requirements that the target cannot satisfy.
    #[must_use]
    pub const fn missing(self) -> RequirementSet {
        self.missing
    }

    /// Return whether lowering may cross this capability boundary.
    #[must_use]
    pub const fn is_compatible(self) -> bool {
        self.missing.is_empty()
    }
}

/// Rejection at the closed target-capability request boundary.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum TargetPlanningError {
    /// Only classical interfaces may become executable artifacts.
    #[error(transparent)]
    InvalidExecutionEntry(#[from] crate::ArtifactAbiError),
    /// A closed continuation cannot preserve source numeric failure ordering.
    #[error(transparent)]
    InvalidShotPlacement(#[from] super::shot_execution::ShotExecutionError),
    /// A capability label is outside the stable requirement vocabulary.
    #[error("Q6301: unknown target capability {0:?}")]
    UnknownCapability(String),
    /// A capability was supplied more than once.
    #[error("Q6302: duplicate target capability {0:?}")]
    DuplicateCapability(String),
    /// A target promises to materialize a requirement outside its protocol.
    #[error("Q6303: target cannot materialize capability {0:?}")]
    InvalidMaterializedCapability(String),
}

/// Plan one target profile against a verified entrypoint.
pub(crate) fn plan(
    program: &VerifiedProgram,
    labels: &[String],
) -> Result<TargetCapabilityPlan, TargetPlanningError> {
    let supported = parse_requirements(labels)?;
    let required = program
        .callable_contract(&program.header().entrypoint)
        .expect("verified entrypoint has a callable contract")
        .requirements();
    let missing = required.difference(supported);
    Ok(TargetCapabilityPlan {
        required,
        supported,
        missing,
    })
}

/// Select a lowering route from verified requirements and transient target facts.
pub(crate) fn plan_lowering(
    program: &VerifiedProgram,
    supported_labels: &[String],
    materialized_labels: &[String],
    circuit_family: bool,
) -> Result<TargetLoweringPlan, TargetPlanningError> {
    program.validate_execution_entry()?;
    let mut capabilities = plan(program, supported_labels)?;
    let mut segmented = false;
    if circuit_family && let Ok(circuit) = program.static_circuit() {
        let (required, split) = circuit.shot_placement(capabilities.required);
        capabilities.required = required;
        capabilities.missing = required.difference(capabilities.supported);
        if split {
            circuit.validate_shot_checks()?;
        }
        segmented = split;
    }
    let promised = parse_requirements(materialized_labels)?;
    let allowed = launch_closable_requirements(circuit_family);
    if let Some(invalid) = promised.difference(allowed).iter().next() {
        return Err(TargetPlanningError::InvalidMaterializedCapability(
            invalid.as_str().to_owned(),
        ));
    }
    let missing = capabilities.missing();
    let unsupported = missing.difference(promised);
    let materialized = missing.difference(unsupported);
    let route = if !unsupported.is_empty() {
        TargetLoweringRoute::Unsupported
    } else if segmented {
        TargetLoweringRoute::ShotExecution
    } else if circuit_family && missing.contains(Requirement::HostDispatch) {
        TargetLoweringRoute::CircuitFamily
    } else if !missing.is_empty() {
        TargetLoweringRoute::LaunchMaterialized
    } else {
        TargetLoweringRoute::Direct
    };
    Ok(TargetLoweringPlan {
        capabilities,
        materialized,
        unsupported,
        route,
    })
}

fn parse_requirements(labels: &[String]) -> Result<RequirementSet, TargetPlanningError> {
    let mut requirements = RequirementSet::EMPTY;
    for label in labels {
        let requirement = Requirement::from_contract_str(label)
            .ok_or_else(|| TargetPlanningError::UnknownCapability(label.clone()))?;
        if requirements.contains(requirement) {
            return Err(TargetPlanningError::DuplicateCapability(label.clone()));
        }
        requirements = requirements.with(requirement);
    }
    Ok(requirements)
}

const fn launch_closable_requirements(circuit_family: bool) -> RequirementSet {
    let requirements = RequirementSet::EMPTY
        .with(Requirement::HostDispatch)
        .with(Requirement::DynamicClassicalIndex)
        .with(Requirement::DynamicQubitSelection);
    if circuit_family {
        requirements
    } else {
        requirements.with(Requirement::ParameterizedExecution)
    }
}
