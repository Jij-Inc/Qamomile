//! Canonical persistence policy for target artifacts.

use prost::Message;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::budget;
use crate::effects::Requirement;
use crate::literals::validate_literal_references;
use crate::result_codec::checked_result_abi;
use crate::runtime::validate_parameter_abi;
use crate::wire;
use crate::{SCHEMA_MAJOR, SCHEMA_MINOR};

/// Maximum encoded size of one portable target artifact.
pub const MAX_ARTIFACT_BYTES: usize = 128 * 1024 * 1024;

/// Integrity-checked metadata from one canonical target artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactHeader {
    /// Schema major carried by the artifact.
    pub schema_major: u32,
    /// Schema minor carried by the artifact.
    pub schema_minor: u32,
    /// Target artifact format selected during compilation.
    pub format: String,
    /// Target identity against which the payload was compiled.
    pub target_fingerprint: String,
    /// SHA-256 digest covering the canonical unsealed artifact.
    pub artifact_digest: [u8; 32],
}

/// Failure to seal or inspect one portable target artifact.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ArtifactError {
    /// The encoded artifact exceeds its total wire-size limit.
    #[error("Q7004: artifact exceeds the wire-size limit")]
    TooLarge,
    /// Borrowed wire traversal found excessive or malformed structure.
    #[error("Q7004: invalid artifact: {0}")]
    InvalidWire(String),
    /// Protobuf decoding failed after bounded preflight.
    #[error("Q7004: malformed artifact: {0}")]
    Malformed(String),
    /// The artifact schema is incompatible with this compiler.
    #[error("Q7005: unsupported artifact schema {major}.{minor}")]
    UnsupportedSchema {
        /// Schema major observed on the wire.
        major: u32,
        /// Schema minor observed on the wire.
        minor: u32,
    },
    /// The digest does not cover the canonical unsealed artifact.
    #[error("Q7006: artifact digest mismatch")]
    DigestMismatch,
    /// The message or one repeated semantic set has an alternative encoding.
    #[error("Q7004: artifact is not canonically encoded")]
    NonCanonical,
    /// The target format or target fingerprint is absent.
    #[error("Q7003: artifact format and fingerprint are required")]
    MissingIdentity,
    /// The ordered launch ABI is malformed or semantically invalid.
    #[error("Q7007: invalid artifact parameter ABI: {0}")]
    InvalidParameterAbi(String),
    /// The result layout is malformed or semantically invalid.
    #[error("Q7010: invalid artifact result ABI: {0}")]
    InvalidResultAbi(String),
    /// A runtime requirement lies outside the closed compiler vocabulary.
    #[error("Q7013: unknown execution requirement {0:?}")]
    UnknownRequirement(String),
    /// The compilation report is missing from the sealed artifact.
    #[error("Q7021: artifact has no resource report")]
    MissingReport,
    /// Metadata is unordered, duplicated, or contains an invalid literal.
    #[error("Q7023: invalid artifact metadata: {0}")]
    InvalidMetadata(String),
}

/// Seal one canonical unsealed `ArtifactProto` through Rust.
///
/// # Errors
///
/// Returns [`ArtifactError`] when the message is oversized, malformed,
/// incompatible, noncanonical, or contains an invalid ABI or contract.
pub fn seal_artifact(payload: &[u8]) -> Result<Vec<u8>, ArtifactError> {
    let artifact = decode_message(payload)?;
    validate_schema(&artifact)?;
    if !artifact.artifact_digest.is_empty() || artifact.encode_to_vec() != payload {
        return Err(ArtifactError::NonCanonical);
    }
    validate_structure(&artifact)?;
    Ok(seal(artifact))
}

/// Inspect one canonical integrity-sealed target artifact.
///
/// # Errors
///
/// Returns [`ArtifactError`] when the artifact is oversized, malformed,
/// incompatible, corrupted, noncanonical, or contains an invalid contract.
pub fn inspect_artifact(payload: &[u8]) -> Result<ArtifactHeader, ArtifactError> {
    let (artifact, artifact_digest) = decode_checked_artifact(payload)?;
    Ok(ArtifactHeader {
        schema_major: artifact.schema_major,
        schema_minor: artifact.schema_minor,
        format: artifact.format,
        target_fingerprint: artifact.target_fingerprint,
        artifact_digest,
    })
}

fn decode_message(payload: &[u8]) -> Result<wire::ArtifactProto, ArtifactError> {
    if payload.len() > MAX_ARTIFACT_BYTES {
        return Err(ArtifactError::TooLarge);
    }
    budget::preflight_artifact(payload)
        .map_err(|error| ArtifactError::InvalidWire(preflight_reason(error)))?;
    wire::ArtifactProto::decode(payload)
        .map_err(|error| ArtifactError::Malformed(error.to_string()))
}

pub(crate) fn decode_checked_artifact(
    payload: &[u8],
) -> Result<(wire::ArtifactProto, [u8; 32]), ArtifactError> {
    let mut artifact = decode_message(payload)?;
    validate_schema(&artifact)?;
    let observed: [u8; 32] = artifact
        .artifact_digest
        .as_slice()
        .try_into()
        .map_err(|_| ArtifactError::DigestMismatch)?;
    artifact.artifact_digest.clear();
    let expected: [u8; 32] = Sha256::digest(artifact.encode_to_vec()).into();
    if observed != expected {
        return Err(ArtifactError::DigestMismatch);
    }
    artifact.artifact_digest.extend_from_slice(&observed);
    if artifact.encode_to_vec() != payload {
        return Err(ArtifactError::NonCanonical);
    }
    validate_structure(&artifact)?;
    Ok((artifact, observed))
}

fn validate_schema(artifact: &wire::ArtifactProto) -> Result<(), ArtifactError> {
    if artifact.schema_major != SCHEMA_MAJOR || artifact.schema_minor > SCHEMA_MINOR {
        return Err(ArtifactError::UnsupportedSchema {
            major: artifact.schema_major,
            minor: artifact.schema_minor,
        });
    }
    Ok(())
}

fn validate_structure(artifact: &wire::ArtifactProto) -> Result<(), ArtifactError> {
    if artifact.format.is_empty() || artifact.target_fingerprint.is_empty() {
        return Err(ArtifactError::MissingIdentity);
    }
    let parameter_abi = match artifact.parameter_abi.as_ref() {
        Some(parameter_abi) if parameter_abi.slots.is_empty() => {
            return Err(ArtifactError::NonCanonical);
        }
        Some(parameter_abi) => parameter_abi.encode_to_vec(),
        None => Vec::new(),
    };
    validate_parameter_abi(&parameter_abi)
        .map_err(|error| ArtifactError::InvalidParameterAbi(error.to_string()))?;
    let result_abi = artifact
        .result_abi
        .as_ref()
        .ok_or_else(|| ArtifactError::InvalidResultAbi("missing ABI".to_owned()))?;
    checked_result_abi(&result_abi.encode_to_vec())
        .map_err(|error| ArtifactError::InvalidResultAbi(error.to_string()))?;
    if artifact.report.is_none() {
        return Err(ArtifactError::MissingReport);
    }
    if !strictly_sorted(&artifact.execution_requirements) {
        return Err(ArtifactError::NonCanonical);
    }
    for requirement in &artifact.execution_requirements {
        if Requirement::from_contract_str(requirement).is_none() {
            return Err(ArtifactError::UnknownRequirement(requirement.clone()));
        }
    }
    let metadata_names = artifact
        .metadata
        .iter()
        .map(|attribute| attribute.name.as_str())
        .collect::<Vec<_>>();
    if !strictly_sorted(&metadata_names) {
        return Err(ArtifactError::InvalidMetadata(
            "keys are not in canonical order".to_owned(),
        ));
    }
    let metadata_values = artifact
        .metadata
        .iter()
        .map(|attribute| {
            attribute.value.as_ref().ok_or_else(|| {
                ArtifactError::InvalidMetadata(format!("entry {:?} has no value", attribute.name))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_literal_references(metadata_values).map_err(ArtifactError::InvalidMetadata)?;
    Ok(())
}

fn strictly_sorted<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn seal(mut artifact: wire::ArtifactProto) -> Vec<u8> {
    artifact.artifact_digest.clear();
    let digest: [u8; 32] = Sha256::digest(artifact.encode_to_vec()).into();
    artifact.artifact_digest.extend_from_slice(&digest);
    artifact.encode_to_vec()
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

    fn scalar(name: &str) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Scalar.into(),
            name: name.to_owned(),
            ..Default::default()
        }
    }

    fn unsealed_envelope() -> wire::ArtifactProto {
        wire::ArtifactProto {
            schema_major: SCHEMA_MAJOR,
            schema_minor: SCHEMA_MINOR,
            format: "qamomile.qcore.program".to_owned(),
            payload: b"target-payload".to_vec(),
            target_fingerprint: "reference-v1".to_owned(),
            parameter_abi: None,
            result_abi: Some(wire::ResultAbiProto {
                r#type: Some(scalar("Unit")),
                ..Default::default()
            }),
            report: Some(wire::CompilationReportProto {
                resource_study: b"resource-study".to_vec(),
                ..Default::default()
            }),
            execution_requirements: vec!["quantum".to_owned()],
            ..Default::default()
        }
    }

    #[test]
    fn seals_and_inspects_one_canonical_artifact() {
        let unsealed = unsealed_envelope().encode_to_vec();
        let sealed = seal_artifact(&unsealed).expect("native seal");
        let header = inspect_artifact(&sealed).expect("native inspect");

        assert_eq!(header.schema_major, SCHEMA_MAJOR);
        assert_eq!(header.schema_minor, SCHEMA_MINOR);
        assert_eq!(header.format, "qamomile.qcore.program");
        assert_eq!(header.target_fingerprint, "reference-v1");
        assert_eq!(header.artifact_digest.len(), 32);
    }

    #[test]
    fn rejects_unknown_requirements_and_noncanonical_metadata() {
        let mut unknown = unsealed_envelope();
        unknown.execution_requirements = vec!["future-runtime".to_owned()];
        assert_eq!(
            seal_artifact(&unknown.encode_to_vec()),
            Err(ArtifactError::UnknownRequirement(
                "future-runtime".to_owned()
            ))
        );

        let literal = || wire::LiteralProto {
            value: Some(wire::literal_proto::Value::IntValue(1)),
        };
        let mut unordered = unsealed_envelope();
        unordered.metadata = vec![
            wire::AttributeProto {
                name: "z".to_owned(),
                value: Some(literal()),
            },
            wire::AttributeProto {
                name: "a".to_owned(),
                value: Some(literal()),
            },
        ];
        assert!(matches!(
            seal_artifact(&unordered.encode_to_vec()),
            Err(ArtifactError::InvalidMetadata(_))
        ));
    }

    #[test]
    fn rejects_oversized_strings_during_borrowed_preflight() {
        let mut oversized = unsealed_envelope();
        oversized.target_fingerprint = "x".repeat(budget::MAX_STRING_BYTES + 1);

        assert!(matches!(
            seal_artifact(&oversized.encode_to_vec()),
            Err(ArtifactError::InvalidWire(_))
        ));
    }
}
