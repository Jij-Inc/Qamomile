//! Canonical persistence for launch-time materialization recipes.
//!
//! The public payload remains the existing portable `LiteralProto` tuple. This
//! module owns its introduction and elimination and verifies the embedded
//! `QCore` program before Python or a target adapter may inspect it.

use prost::Message;
use thiserror::Error;

use crate::VerifiedProgram;
use crate::artifact::decode_checked_artifact;
use crate::budget;
use crate::effects::{Requirement, RequirementSet};
use crate::literals::{
    bytes_literal, bytes_value, int_literal, int_value, string_literal, string_value,
    tuple_literal, tuple_values,
};
use crate::resource_study::inspect_resource_study;
use crate::wire;

const LAUNCH_MATERIALIZER_KIND: &str = "qamomile.launch-materializer";
const LAUNCH_MATERIALIZER_VERSION: i64 = 1;
const LAUNCH_MATERIALIZER_FORMAT: &str = "qamomile.launch-materializer.v1";

/// Maximum encoded size of one launch-materializer payload.
pub const MAX_LAUNCH_MATERIALIZER_BYTES: usize = 64 * 1024 * 1024;

/// One fully checked persistent launch-materializer payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LaunchMaterializerPayload {
    target_format: String,
    program: Box<[u8]>,
}

impl LaunchMaterializerPayload {
    /// Return the target format selected before launch.
    #[must_use]
    pub fn target_format(&self) -> &str {
        &self.target_format
    }

    /// Return the canonical verified `QCore` program materialized at launch.
    #[must_use]
    pub fn program(&self) -> &[u8] {
        &self.program
    }
}

/// Failure to seal or inspect a persistent launch materializer.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum LaunchMaterializerError {
    /// The payload is malformed, noncanonical, inconsistent, or excessive.
    #[error("Q4464: invalid native launch-materializer payload: {0}")]
    Invalid(String),
}

/// Seal one launch materializer in the existing portable v1 representation.
///
/// # Errors
///
/// Returns [`LaunchMaterializerError`] when the target is missing, the program
/// is not a canonical verified executable, or the resulting payload exceeds a
/// persistence budget.
pub fn seal_launch_materializer_payload(
    target_format: &str,
    program: &[u8],
) -> Result<Vec<u8>, LaunchMaterializerError> {
    ensure_component_sizes(target_format.len(), program.len())?;
    let payload = tuple_literal(vec![
        string_literal(LAUNCH_MATERIALIZER_KIND),
        int_literal(LAUNCH_MATERIALIZER_VERSION),
        string_literal(target_format),
        bytes_literal(program),
    ])
    .encode_to_vec();
    inspect_launch_materializer_payload(&payload)?;
    Ok(payload)
}

fn ensure_component_sizes(
    target_size: usize,
    program_size: usize,
) -> Result<(), LaunchMaterializerError> {
    if target_size > budget::MAX_STRING_BYTES {
        return Err(invalid("target format exceeds its string budget"));
    }
    if program_size > budget::MAX_BYTES_FIELD {
        return Err(invalid("program exceeds its bytes-field budget"));
    }
    Ok(())
}

/// Decode and fully validate one untrusted launch-materializer payload.
///
/// # Errors
///
/// Returns [`LaunchMaterializerError`] when the payload exceeds a structural
/// budget, is noncanonical, has the wrong identity, or embeds an invalid
/// executable program.
pub fn inspect_launch_materializer_payload(
    payload: &[u8],
) -> Result<LaunchMaterializerPayload, LaunchMaterializerError> {
    let (target_format, program) = decode_launch_materializer_payload(payload)?;
    Ok(checked_payload(target_format, &program))
}

/// Validate one complete launch-materializer artifact and its embedded program.
///
/// # Errors
///
/// Returns [`LaunchMaterializerError`] when the outer artifact, inner payload,
/// resource identity, compiler-derived ABIs, target, or materialized
/// requirements disagree.
pub fn inspect_launch_materializer_artifact(
    artifact_bytes: &[u8],
    target_format: &str,
    materialized_requirements: &[String],
) -> Result<LaunchMaterializerPayload, LaunchMaterializerError> {
    let (artifact, _) =
        decode_checked_artifact(artifact_bytes).map_err(|error| invalid(&error.to_string()))?;
    if artifact.format != LAUNCH_MATERIALIZER_FORMAT {
        return Err(invalid("outer artifact has the wrong format"));
    }
    let (embedded_target, program) = decode_launch_materializer_payload(&artifact.payload)?;
    if embedded_target != target_format {
        return Err(invalid("target format disagrees with the materializer"));
    }
    let derived = program
        .artifact_abi()
        .map_err(|error| invalid(&error.to_string()))?;
    let parameter_abi = artifact
        .parameter_abi
        .as_ref()
        .map(Message::encode_to_vec)
        .unwrap_or_default();
    let result_abi = artifact
        .result_abi
        .as_ref()
        .ok_or_else(|| invalid("outer artifact has no result ABI"))?
        .encode_to_vec();
    if parameter_abi != derived.parameter_abi() || result_abi != derived.result_abi() {
        return Err(invalid("outer artifact ABI disagrees with the program"));
    }
    let resource_study = artifact
        .report
        .as_ref()
        .ok_or_else(|| invalid("outer artifact has no resource report"))?
        .resource_study
        .as_slice();
    let study = inspect_resource_study(resource_study)
        .map_err(|error| invalid(&format!("invalid resource study: {error}")))?;
    if study.semantic_digest() != &program.semantic_digest() {
        return Err(invalid("resource identity disagrees with the program"));
    }
    let outer_requirements = requirement_set(&artifact.execution_requirements)?;
    let materialized = requirement_set(materialized_requirements)?;
    if materialized.is_empty() || !materialized.difference(outer_requirements).is_empty() {
        return Err(invalid(
            "materialized requirements are not a nonempty contract subset",
        ));
    }
    Ok(checked_payload(embedded_target, &program))
}

fn decode_launch_materializer_payload(
    payload: &[u8],
) -> Result<(String, VerifiedProgram), LaunchMaterializerError> {
    ensure_payload_size(payload.len())?;
    budget::preflight_literal(payload).map_err(|error| invalid(&preflight_reason(error)))?;
    let encoded =
        wire::LiteralProto::decode(payload).map_err(|error| invalid(&error.to_string()))?;
    if encoded.encode_to_vec() != payload {
        return Err(invalid("payload is not canonically encoded"));
    }
    let root = tuple_values(&encoded).ok_or_else(|| invalid("header is not a tuple"))?;
    if root.len() != 4
        || string_value(&root[0]) != Some(LAUNCH_MATERIALIZER_KIND)
        || int_value(&root[1]) != Some(LAUNCH_MATERIALIZER_VERSION)
    {
        return Err(invalid("unsupported or malformed header"));
    }
    let target_format = string_value(&root[2])
        .filter(|value| !value.is_empty())
        .ok_or_else(|| invalid("target format is missing"))?;
    let program = bytes_value(&root[3])
        .filter(|value| !value.is_empty())
        .ok_or_else(|| invalid("program is missing"))?;
    let program =
        VerifiedProgram::from_bytes(program).map_err(|error| invalid(&error.to_string()))?;
    Ok((target_format.to_owned(), program))
}

fn checked_payload(target_format: String, program: &VerifiedProgram) -> LaunchMaterializerPayload {
    LaunchMaterializerPayload {
        target_format,
        program: program.canonical_bytes().to_vec().into_boxed_slice(),
    }
}

fn requirement_set(labels: &[String]) -> Result<RequirementSet, LaunchMaterializerError> {
    let mut requirements = RequirementSet::EMPTY;
    for label in labels {
        let requirement = Requirement::from_contract_str(label)
            .ok_or_else(|| invalid("unknown materialization requirement"))?;
        if requirements.contains(requirement) {
            return Err(invalid("duplicate materialization requirement"));
        }
        requirements = requirements.with(requirement);
    }
    Ok(requirements)
}

fn ensure_payload_size(size: usize) -> Result<(), LaunchMaterializerError> {
    if size > MAX_LAUNCH_MATERIALIZER_BYTES {
        return Err(invalid("payload exceeds its byte budget"));
    }
    Ok(())
}

fn invalid(reason: &str) -> LaunchMaterializerError {
    LaunchMaterializerError::Invalid(reason.to_owned())
}

fn preflight_reason(error: budget::PreflightError) -> String {
    match error {
        budget::PreflightError::Budget(reason) | budget::PreflightError::Malformed(reason) => {
            reason
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_malformed_headers_before_program_adoption() {
        let wrong_kind = tuple_literal(vec![
            string_literal("qamomile.other"),
            int_literal(LAUNCH_MATERIALIZER_VERSION),
            string_literal("probe.v1"),
            bytes_literal(b"program"),
        ])
        .encode_to_vec();
        let wrong_version = tuple_literal(vec![
            string_literal(LAUNCH_MATERIALIZER_KIND),
            int_literal(2),
            string_literal("probe.v1"),
            bytes_literal(b"program"),
        ])
        .encode_to_vec();

        assert!(inspect_launch_materializer_payload(&wrong_kind).is_err());
        assert!(inspect_launch_materializer_payload(&wrong_version).is_err());
        assert!(seal_launch_materializer_payload("", b"program").is_err());
    }

    #[test]
    fn borrowed_preflight_rejects_excessive_literal_nesting() {
        let mut nested = int_literal(0);
        for _ in 0..=budget::MAX_NESTING {
            nested = tuple_literal(vec![nested]);
        }

        assert!(inspect_launch_materializer_payload(&nested.encode_to_vec()).is_err());
    }

    #[test]
    fn rejects_the_total_byte_budget_without_allocating_the_payload() {
        assert!(ensure_component_sizes(budget::MAX_STRING_BYTES, budget::MAX_BYTES_FIELD).is_ok());
        assert!(ensure_component_sizes(budget::MAX_STRING_BYTES + 1, 0).is_err());
        assert!(ensure_component_sizes(0, budget::MAX_BYTES_FIELD + 1).is_err());
        assert!(ensure_payload_size(MAX_LAUNCH_MATERIALIZER_BYTES).is_ok());
        assert!(ensure_payload_size(MAX_LAUNCH_MATERIALIZER_BYTES + 1).is_err());
    }
}
