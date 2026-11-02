//! Canonical persistence for local, remote, and composite execution state.

use prost::Message;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::budget;
use crate::literals::validate_literal_references;
use crate::wire;
use crate::{SCHEMA_MAJOR, SCHEMA_MINOR};

/// Stable artifact kind carried by every execution snapshot.
pub const EXECUTION_SNAPSHOT_KIND: &str = "qamomile.execution-snapshot";
/// Maximum encoded size of one execution snapshot.
pub const MAX_EXECUTION_SNAPSHOT_BYTES: usize = 1024 * 1024;
/// Maximum total message and repeated-value nodes in one snapshot.
pub const MAX_EXECUTION_SNAPSHOT_NODES: usize = 4096;
/// Maximum nesting depth of one composite execution tree.
pub const MAX_EXECUTION_SNAPSHOT_DEPTH: usize = 64;
/// Maximum number of provider references in one snapshot.
pub const MAX_EXECUTION_SNAPSHOT_REFERENCES: usize = 256;
/// Maximum context entries carried by one provider reference.
pub const MAX_EXECUTION_CONTEXT_ENTRIES: usize = 256;
/// Maximum cumulative UTF-8 bytes across execution identifiers and context.
pub const MAX_EXECUTION_STRING_BYTES: usize = 256 * 1024;

/// Integrity-checked summary of one canonical execution snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionSnapshotHeader {
    /// SHA-256 digest covering the canonical unsealed snapshot.
    pub snapshot_digest: [u8; 32],
    /// Number of local, remote, and composite execution nodes.
    pub nodes: usize,
    /// Number of remote provider references.
    pub references: usize,
}

/// Failure to seal or inspect one portable execution snapshot.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ExecutionSnapshotError {
    /// The encoded snapshot exceeds its total wire-size limit.
    #[error("Q8401: execution snapshot exceeds the wire-size limit")]
    TooLarge,
    /// Borrowed wire traversal found excessive or malformed structure.
    #[error("Q8402: invalid execution snapshot wire: {0}")]
    InvalidWire(String),
    /// Protobuf decoding failed after bounded preflight.
    #[error("Q8403: malformed execution snapshot: {0}")]
    Malformed(String),
    /// The snapshot schema is incompatible with this compiler.
    #[error("Q8404: unsupported execution snapshot schema {major}.{minor}")]
    UnsupportedSchema {
        /// Schema major observed on the wire.
        major: u32,
        /// Schema minor observed on the wire.
        minor: u32,
    },
    /// The persistent message has a noncanonical equivalent encoding.
    #[error("Q8405: execution snapshot is not canonically encoded")]
    NonCanonical,
    /// The stored integrity digest does not match the payload.
    #[error("Q8406: execution snapshot digest mismatch")]
    DigestMismatch,
    /// A node kind or its selected fields violate the closed snapshot grammar.
    #[error("Q8407: invalid execution snapshot structure: {0}")]
    InvalidStructure(String),
    /// A provider reference is incomplete, unordered, or over budget.
    #[error("Q8408: invalid execution reference: {0}")]
    InvalidReference(String),
    /// A local value is not a canonical portable literal.
    #[error("Q8409: invalid local execution value: {0}")]
    InvalidLocalValue(String),
}

/// Seal one canonical unsealed `ExecutionSnapshotProto` through Rust.
///
/// # Errors
///
/// Returns [`ExecutionSnapshotError`] when the snapshot is oversized,
/// malformed, incompatible, noncanonical, or structurally invalid.
pub fn seal_execution_snapshot(payload: &[u8]) -> Result<Vec<u8>, ExecutionSnapshotError> {
    let snapshot = decode_message(payload)?;
    validate_schema(&snapshot)?;
    if !snapshot.snapshot_digest.is_empty() || snapshot.encode_to_vec() != payload {
        return Err(ExecutionSnapshotError::NonCanonical);
    }
    validate_structure(&snapshot)?;
    Ok(seal(snapshot))
}

/// Inspect one integrity-sealed canonical execution snapshot.
///
/// # Errors
///
/// Returns [`ExecutionSnapshotError`] when the snapshot is oversized,
/// malformed, incompatible, corrupted, noncanonical, or invalid.
pub fn inspect_execution_snapshot(
    payload: &[u8],
) -> Result<ExecutionSnapshotHeader, ExecutionSnapshotError> {
    let mut snapshot = decode_message(payload)?;
    validate_schema(&snapshot)?;
    let observed: [u8; 32] = snapshot
        .snapshot_digest
        .as_slice()
        .try_into()
        .map_err(|_| ExecutionSnapshotError::DigestMismatch)?;
    snapshot.snapshot_digest.clear();
    let unsealed = snapshot.encode_to_vec();
    let expected: [u8; 32] = Sha256::digest(&unsealed).into();
    if observed != expected {
        return Err(ExecutionSnapshotError::DigestMismatch);
    }
    let sealed = seal(snapshot.clone());
    if sealed != payload {
        return Err(ExecutionSnapshotError::NonCanonical);
    }
    let (nodes, references) = validate_structure(&snapshot)?;
    Ok(ExecutionSnapshotHeader {
        snapshot_digest: observed,
        nodes,
        references,
    })
}

fn decode_message(payload: &[u8]) -> Result<wire::ExecutionSnapshotProto, ExecutionSnapshotError> {
    if payload.len() > MAX_EXECUTION_SNAPSHOT_BYTES {
        return Err(ExecutionSnapshotError::TooLarge);
    }
    budget::preflight_execution_snapshot(
        payload,
        MAX_EXECUTION_SNAPSHOT_NODES,
        MAX_EXECUTION_SNAPSHOT_DEPTH + 8,
        MAX_EXECUTION_STRING_BYTES,
    )
    .map_err(|error| ExecutionSnapshotError::InvalidWire(format!("{error:?}")))?;
    wire::ExecutionSnapshotProto::decode(payload)
        .map_err(|error| ExecutionSnapshotError::Malformed(error.to_string()))
}

fn validate_schema(snapshot: &wire::ExecutionSnapshotProto) -> Result<(), ExecutionSnapshotError> {
    if snapshot.schema_major != SCHEMA_MAJOR || snapshot.schema_minor > SCHEMA_MINOR {
        return Err(ExecutionSnapshotError::UnsupportedSchema {
            major: snapshot.schema_major,
            minor: snapshot.schema_minor,
        });
    }
    if snapshot.artifact_kind != EXECUTION_SNAPSHOT_KIND {
        return Err(ExecutionSnapshotError::InvalidStructure(
            "unknown artifact kind".to_owned(),
        ));
    }
    Ok(())
}

fn validate_structure(
    snapshot: &wire::ExecutionSnapshotProto,
) -> Result<(usize, usize), ExecutionSnapshotError> {
    use wire::execution_snapshot_node_proto::Kind;

    let root = snapshot.root.as_ref().ok_or_else(|| {
        ExecutionSnapshotError::InvalidStructure("snapshot has no root".to_owned())
    })?;
    let mut pending = vec![(root, 0_usize)];
    let mut nodes = 0_usize;
    let mut references = 0_usize;
    let mut string_bytes = snapshot.artifact_kind.len();
    let mut local_values = Vec::new();
    while let Some((node, depth)) = pending.pop() {
        nodes = nodes.checked_add(1).ok_or_else(|| {
            ExecutionSnapshotError::InvalidStructure("node count overflow".to_owned())
        })?;
        if nodes > MAX_EXECUTION_SNAPSHOT_NODES || depth > MAX_EXECUTION_SNAPSHOT_DEPTH {
            return Err(ExecutionSnapshotError::InvalidStructure(
                "execution tree exceeds its node or depth budget".to_owned(),
            ));
        }
        let kind = Kind::try_from(node.kind).map_err(|_| {
            ExecutionSnapshotError::InvalidStructure("unknown node kind".to_owned())
        })?;
        match kind {
            Kind::Remote => {
                if node.local_value.is_some() || !node.children.is_empty() {
                    return Err(ExecutionSnapshotError::InvalidStructure(
                        "remote node carries local value or children".to_owned(),
                    ));
                }
                let reference = node.reference.as_ref().ok_or_else(|| {
                    ExecutionSnapshotError::InvalidStructure(
                        "remote node has no reference".to_owned(),
                    )
                })?;
                references = references.checked_add(1).ok_or_else(|| {
                    ExecutionSnapshotError::InvalidReference("reference count overflow".to_owned())
                })?;
                if references > MAX_EXECUTION_SNAPSHOT_REFERENCES {
                    return Err(ExecutionSnapshotError::InvalidReference(
                        "snapshot exceeds the reference budget".to_owned(),
                    ));
                }
                string_bytes = validate_reference(reference, string_bytes)?;
            }
            Kind::Local => {
                if node.reference.is_some() || !node.children.is_empty() {
                    return Err(ExecutionSnapshotError::InvalidStructure(
                        "local node carries reference or children".to_owned(),
                    ));
                }
                local_values.push(node.local_value.as_ref().ok_or_else(|| {
                    ExecutionSnapshotError::InvalidStructure("local node has no value".to_owned())
                })?);
            }
            Kind::Composite => {
                if node.reference.is_some()
                    || node.local_value.is_some()
                    || node.children.is_empty()
                {
                    return Err(ExecutionSnapshotError::InvalidStructure(
                        "composite node must contain only nonempty children".to_owned(),
                    ));
                }
                pending.extend(node.children.iter().map(|child| (child, depth + 1)));
            }
            Kind::Unspecified => {
                return Err(ExecutionSnapshotError::InvalidStructure(
                    "unspecified node kind".to_owned(),
                ));
            }
        }
    }
    validate_literal_references(local_values).map_err(ExecutionSnapshotError::InvalidLocalValue)?;
    Ok((nodes, references))
}

fn validate_reference(
    reference: &wire::ExecutionReferenceProto,
    mut string_bytes: usize,
) -> Result<usize, ExecutionSnapshotError> {
    if reference.provider.is_empty()
        || reference.job_ids.is_empty()
        || reference.job_ids.iter().any(String::is_empty)
        || reference.has_target == reference.target.is_empty()
        || reference.has_group_id == reference.group_id.is_empty()
    {
        return Err(ExecutionSnapshotError::InvalidReference(
            "missing or contradictory identifier".to_owned(),
        ));
    }
    if reference.context.len() > MAX_EXECUTION_CONTEXT_ENTRIES {
        return Err(ExecutionSnapshotError::InvalidReference(
            "context exceeds the entry budget".to_owned(),
        ));
    }
    let mut previous: Option<&str> = None;
    for entry in &reference.context {
        if entry.key.is_empty() || previous.is_some_and(|key| key >= entry.key.as_str()) {
            return Err(ExecutionSnapshotError::InvalidReference(
                "context keys must be unique, nonempty, and sorted".to_owned(),
            ));
        }
        previous = Some(&entry.key);
        string_bytes = add_string_bytes(string_bytes, entry.key.len())?;
        string_bytes = add_string_bytes(string_bytes, entry.value.len())?;
    }
    string_bytes = add_string_bytes(string_bytes, reference.provider.len())?;
    string_bytes = add_string_bytes(string_bytes, reference.target.len())?;
    string_bytes = add_string_bytes(string_bytes, reference.group_id.len())?;
    for job_id in &reference.job_ids {
        string_bytes = add_string_bytes(string_bytes, job_id.len())?;
    }
    Ok(string_bytes)
}

fn add_string_bytes(total: usize, count: usize) -> Result<usize, ExecutionSnapshotError> {
    let total = total.checked_add(count).ok_or_else(|| {
        ExecutionSnapshotError::InvalidReference("string-byte count overflow".to_owned())
    })?;
    if total > MAX_EXECUTION_STRING_BYTES {
        return Err(ExecutionSnapshotError::InvalidReference(
            "snapshot exceeds the cumulative string budget".to_owned(),
        ));
    }
    Ok(total)
}

fn seal(mut snapshot: wire::ExecutionSnapshotProto) -> Vec<u8> {
    snapshot.snapshot_digest.clear();
    let digest = Sha256::digest(snapshot.encode_to_vec());
    snapshot.snapshot_digest = digest.to_vec();
    snapshot.encode_to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn remote_node() -> wire::ExecutionSnapshotNodeProto {
        wire::ExecutionSnapshotNodeProto {
            kind: wire::execution_snapshot_node_proto::Kind::Remote.into(),
            reference: Some(wire::ExecutionReferenceProto {
                provider: "provider".to_owned(),
                job_ids: vec!["job".to_owned()],
                target: "target".to_owned(),
                has_target: true,
                context: vec![wire::ExecutionContextEntryProto {
                    key: "kind".to_owned(),
                    value: "sample".to_owned(),
                }],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn unsealed(root: wire::ExecutionSnapshotNodeProto) -> Vec<u8> {
        wire::ExecutionSnapshotProto {
            schema_major: SCHEMA_MAJOR,
            schema_minor: SCHEMA_MINOR,
            artifact_kind: EXECUTION_SNAPSHOT_KIND.to_owned(),
            root: Some(root),
            snapshot_digest: Vec::new(),
        }
        .encode_to_vec()
    }

    #[test]
    fn remote_snapshot_round_trips_deterministically() {
        let first = seal_execution_snapshot(&unsealed(remote_node())).expect("seal snapshot");
        let second = seal_execution_snapshot(&unsealed(remote_node())).expect("reseal snapshot");

        assert_eq!(first, second);
        assert_eq!(
            inspect_execution_snapshot(&first)
                .expect("inspect")
                .references,
            1
        );
    }

    #[test]
    fn rejects_corruption_and_unknown_kinds() {
        let mut corrupted = seal_execution_snapshot(&unsealed(remote_node())).expect("seal");
        let last = corrupted.last_mut().expect("digest byte");
        *last ^= 1;
        assert!(matches!(
            inspect_execution_snapshot(&corrupted),
            Err(ExecutionSnapshotError::DigestMismatch)
        ));

        let mut unknown = remote_node();
        unknown.kind = 99;
        assert!(matches!(
            seal_execution_snapshot(&unsealed(unknown)),
            Err(ExecutionSnapshotError::InvalidStructure(_))
        ));
    }

    #[test]
    fn rejects_unordered_context_and_excessive_depth() {
        let mut unordered = remote_node();
        unordered.reference.as_mut().expect("reference").context = vec![
            wire::ExecutionContextEntryProto {
                key: "z".to_owned(),
                value: String::new(),
            },
            wire::ExecutionContextEntryProto {
                key: "a".to_owned(),
                value: String::new(),
            },
        ];
        assert!(matches!(
            seal_execution_snapshot(&unsealed(unordered)),
            Err(ExecutionSnapshotError::InvalidReference(_))
        ));

        let mut nested = remote_node();
        for _ in 0..=MAX_EXECUTION_SNAPSHOT_DEPTH {
            nested = wire::ExecutionSnapshotNodeProto {
                kind: wire::execution_snapshot_node_proto::Kind::Composite.into(),
                children: vec![nested],
                ..Default::default()
            };
        }
        assert!(seal_execution_snapshot(&unsealed(nested)).is_err());
    }

    #[test]
    fn rejects_unknown_schema_and_oversized_payload() {
        let mut schema = wire::ExecutionSnapshotProto::decode(unsealed(remote_node()).as_slice())
            .expect("decode helper");
        schema.schema_major += 1;
        assert!(matches!(
            seal_execution_snapshot(&schema.encode_to_vec()),
            Err(ExecutionSnapshotError::UnsupportedSchema { .. })
        ));
        assert_eq!(
            seal_execution_snapshot(&vec![0; MAX_EXECUTION_SNAPSHOT_BYTES + 1]),
            Err(ExecutionSnapshotError::TooLarge)
        );
    }
}
