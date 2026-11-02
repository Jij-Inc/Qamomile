//! Immutable device facts a compilation target can supply to extension passes.
//!
//! A target profile is an input to pass context and target identity, never an
//! IR element: no lowering stage reads it. This module owns the schema bounds,
//! the canonical row order, and the digest; Python retains canonical bytes
//! behind a frozen facade, and adapters derive profiles from SDK objects without
//! defining what a profile means. Absent facets are unknown rather than empty.

use std::collections::HashSet;

use prost::Message;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::budget::MAX_STRING_BYTES;
use crate::wire;

/// Maximum encoded size of one target profile.
pub const MAX_TARGET_PROFILE_BYTES: usize = 16 * 1024 * 1024;
/// Maximum qubit count one target profile may describe.
pub const MAX_TARGET_PROFILE_QUBITS: u32 = 65_536;
/// Maximum coupling edges, and property rows per table, in one profile.
pub const MAX_TARGET_PROFILE_EDGES: usize = 4_194_304;
const SCHEMA_VERSION: u32 = 1;
const DIGEST_DOMAIN: &[u8] = b"qamomile.target-profile.v1\0";

/// Why a target profile payload was rejected.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum TargetProfileError {
    /// The bytes do not decode as a supported profile, or a required part is missing.
    #[error("Q7030: malformed target profile: {0}")]
    Malformed(String),
    /// A count or string exceeds the profile budget.
    #[error("Q7031: target profile exceeds its budget: {0}")]
    Budget(String),
    /// A row references a qubit or edge the topology does not declare, or repeats one.
    #[error("Q7032: invalid target profile reference: {0}")]
    Reference(String),
    /// A measured or modeled value is not a finite number in its allowed range.
    #[error("Q7033: invalid target profile value: {0}")]
    Value(String),
}

/// One validated profile in canonical byte order with its digest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TargetProfile {
    canonical: Vec<u8>,
    digest: [u8; 32],
    num_qubits: u32,
    edge_count: usize,
}

impl TargetProfile {
    /// Return the canonical encoding whose bytes identify this profile.
    #[must_use]
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical
    }

    /// Return the SHA-256 identity of the canonical encoding.
    #[must_use]
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// Return the declared qubit count.
    #[must_use]
    pub const fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// Return the number of declared coupling rows.
    #[must_use]
    pub const fn edge_count(&self) -> usize {
        self.edge_count
    }
}

/// Validate one profile payload and return its canonical form.
///
/// Rows are sorted into canonical order and re-encoded, so equal facts in any
/// input order yield equal bytes. Work is linear in the payload plus sorting.
///
/// # Errors
///
/// Returns [`TargetProfileError`] when the payload is malformed, exceeds a
/// bound, references undeclared qubits or edges, repeats a row, or holds a
/// non-finite value.
pub fn validate_target_profile(payload: &[u8]) -> Result<TargetProfile, TargetProfileError> {
    if payload.len() > MAX_TARGET_PROFILE_BYTES {
        return Err(TargetProfileError::Budget(format!(
            "encoded size exceeds {MAX_TARGET_PROFILE_BYTES} bytes"
        )));
    }
    let mut profile = wire::TargetProfileProto::decode(payload)
        .map_err(|error| TargetProfileError::Malformed(error.to_string()))?;
    if profile.schema_version != SCHEMA_VERSION {
        return Err(TargetProfileError::Malformed(format!(
            "unsupported schema version {}",
            profile.schema_version
        )));
    }
    if profile.format.is_empty() {
        return Err(TargetProfileError::Malformed("format is empty".to_owned()));
    }
    for (label, text) in [
        ("format", &profile.format),
        ("backend_name", &profile.backend_name),
        ("calibration_time", &profile.calibration_time),
        ("provenance", &profile.provenance),
    ] {
        require_bounded_text(label, text)?;
    }
    let topology = profile
        .topology
        .as_mut()
        .ok_or_else(|| TargetProfileError::Malformed("topology is missing".to_owned()))?;
    canonicalize_topology(topology)?;
    let num_qubits = topology.num_qubits;
    let edges: HashSet<(u32, u32)> = topology
        .edges
        .iter()
        .flat_map(|edge| {
            let reverse = edge.symmetric.then_some((edge.target, edge.control));
            std::iter::once((edge.control, edge.target)).chain(reverse)
        })
        .collect();
    let edge_count = topology.edges.len();
    if let Some(calibration) = profile.calibration.as_mut() {
        canonicalize_calibration(calibration, num_qubits, &edges)?;
    }
    if let Some(logical) = profile.logical.as_mut() {
        canonicalize_logical(logical)?;
    }
    let canonical = profile.encode_to_vec();
    let digest = Sha256::new()
        .chain_update(DIGEST_DOMAIN)
        .chain_update(&canonical)
        .finalize()
        .into();
    Ok(TargetProfile {
        canonical,
        digest,
        num_qubits,
        edge_count,
    })
}

fn require_bounded_text(label: &str, text: &str) -> Result<(), TargetProfileError> {
    if text.len() > MAX_STRING_BYTES {
        return Err(TargetProfileError::Budget(format!(
            "{label} exceeds {MAX_STRING_BYTES} bytes"
        )));
    }
    Ok(())
}

fn require_named(label: &str, text: &str) -> Result<(), TargetProfileError> {
    if text.is_empty() {
        return Err(TargetProfileError::Malformed(format!("{label} is empty")));
    }
    require_bounded_text(label, text)
}

fn require_finite(label: &str, value: f64) -> Result<(), TargetProfileError> {
    if !value.is_finite() {
        return Err(TargetProfileError::Value(format!("{label} is not finite")));
    }
    Ok(())
}

fn require_qubit(label: &str, qubit: u32, num_qubits: u32) -> Result<(), TargetProfileError> {
    if qubit >= num_qubits {
        return Err(TargetProfileError::Reference(format!(
            "{label} names qubit {qubit} outside {num_qubits} qubits"
        )));
    }
    Ok(())
}

fn canonicalize_names(label: &str, names: &mut [String]) -> Result<(), TargetProfileError> {
    for name in names.iter() {
        require_named(label, name)?;
    }
    names.sort_unstable();
    if names.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(TargetProfileError::Reference(format!(
            "{label} repeats a name"
        )));
    }
    Ok(())
}

fn canonicalize_topology(topology: &mut wire::TopologyProto) -> Result<(), TargetProfileError> {
    let num_qubits = topology.num_qubits;
    if num_qubits == 0 {
        return Err(TargetProfileError::Malformed(
            "num_qubits must be positive".to_owned(),
        ));
    }
    if num_qubits > MAX_TARGET_PROFILE_QUBITS {
        return Err(TargetProfileError::Budget(format!(
            "num_qubits exceeds {MAX_TARGET_PROFILE_QUBITS}"
        )));
    }
    let pair_bound = (num_qubits as usize)
        .saturating_mul(num_qubits as usize - 1)
        .min(MAX_TARGET_PROFILE_EDGES);
    if topology.edges.len() > pair_bound {
        return Err(TargetProfileError::Budget(format!(
            "edge count exceeds {pair_bound}"
        )));
    }
    for edge in &topology.edges {
        require_qubit("edge", edge.control, num_qubits)?;
        require_qubit("edge", edge.target, num_qubits)?;
        if edge.control == edge.target {
            return Err(TargetProfileError::Reference(format!(
                "edge ({}, {}) is a self loop",
                edge.control, edge.target
            )));
        }
        if edge.symmetric && edge.control > edge.target {
            return Err(TargetProfileError::Malformed(format!(
                "symmetric edge ({}, {}) must list the smaller qubit first",
                edge.control, edge.target
            )));
        }
    }
    topology
        .edges
        .sort_unstable_by_key(|edge| (edge.control, edge.target));
    let mut seen = HashSet::with_capacity(topology.edges.len());
    for edge in &topology.edges {
        if !seen.insert((edge.control, edge.target)) {
            return Err(TargetProfileError::Reference(format!(
                "edge ({}, {}) is repeated",
                edge.control, edge.target
            )));
        }
    }
    for edge in &topology.edges {
        if seen.contains(&(edge.target, edge.control)) {
            let symmetric = topology
                .edges
                .binary_search_by_key(&(edge.target, edge.control), |other| {
                    (other.control, other.target)
                })
                .is_ok_and(|index| topology.edges[index].symmetric);
            if edge.symmetric || symmetric {
                return Err(TargetProfileError::Reference(format!(
                    "edge ({}, {}) duplicates a symmetric edge",
                    edge.control, edge.target
                )));
            }
        }
    }
    canonicalize_names("one_qubit_operations", &mut topology.one_qubit_operations)?;
    canonicalize_names("two_qubit_operations", &mut topology.two_qubit_operations)
}

fn canonicalize_calibration(
    calibration: &mut wire::CalibrationProto,
    num_qubits: u32,
    edges: &HashSet<(u32, u32)>,
) -> Result<(), TargetProfileError> {
    require_bounded_text("calibration provenance", &calibration.provenance)?;
    if calibration.qubit_properties.len() > MAX_TARGET_PROFILE_EDGES
        || calibration.edge_properties.len() > MAX_TARGET_PROFILE_EDGES
    {
        return Err(TargetProfileError::Budget(format!(
            "property rows exceed {MAX_TARGET_PROFILE_EDGES}"
        )));
    }
    for row in &calibration.qubit_properties {
        require_qubit("qubit property", row.qubit, num_qubits)?;
        require_named("qubit property name", &row.name)?;
        require_bounded_text("qubit property unit", &row.unit)?;
        require_finite("qubit property value", row.value)?;
    }
    calibration
        .qubit_properties
        .sort_by(|left, right| (left.qubit, &left.name).cmp(&(right.qubit, &right.name)));
    if calibration
        .qubit_properties
        .windows(2)
        .any(|pair| (pair[0].qubit, &pair[0].name) == (pair[1].qubit, &pair[1].name))
    {
        return Err(TargetProfileError::Reference(
            "qubit property is repeated".to_owned(),
        ));
    }
    for row in &calibration.edge_properties {
        if !edges.contains(&(row.control, row.target)) {
            return Err(TargetProfileError::Reference(format!(
                "edge property names undeclared edge ({}, {})",
                row.control, row.target
            )));
        }
        require_named("edge property name", &row.name)?;
        require_bounded_text("edge property unit", &row.unit)?;
        require_finite("edge property value", row.value)?;
    }
    calibration.edge_properties.sort_by(|left, right| {
        (left.control, left.target, &left.name).cmp(&(right.control, right.target, &right.name))
    });
    if calibration.edge_properties.windows(2).any(|pair| {
        (pair[0].control, pair[0].target, &pair[0].name)
            == (pair[1].control, pair[1].target, &pair[1].name)
    }) {
        return Err(TargetProfileError::Reference(
            "edge property is repeated".to_owned(),
        ));
    }
    Ok(())
}

fn canonicalize_logical(logical: &mut wire::LogicalProto) -> Result<(), TargetProfileError> {
    require_named("code_family", &logical.code_family)?;
    if logical.code_distance == 0 {
        return Err(TargetProfileError::Value(
            "code_distance must be positive".to_owned(),
        ));
    }
    for (label, value) in [
        ("cycle_time", logical.cycle_time),
        ("magic_state_cost", logical.magic_state_cost),
    ] {
        require_finite(label, value)?;
        if value < 0.0 {
            return Err(TargetProfileError::Value(format!("{label} is negative")));
        }
    }
    for (label, text) in [
        ("cycle_time_unit", &logical.cycle_time_unit),
        ("magic_state_cost_unit", &logical.magic_state_cost_unit),
        (
            "rotation_synthesis_model",
            &logical.rotation_synthesis_model,
        ),
    ] {
        require_bounded_text(label, text)?;
    }
    if logical.rotation_synthesis_constants.len() > MAX_TARGET_PROFILE_EDGES {
        return Err(TargetProfileError::Budget(format!(
            "rotation constants exceed {MAX_TARGET_PROFILE_EDGES}"
        )));
    }
    for constant in &logical.rotation_synthesis_constants {
        require_named("rotation constant name", &constant.name)?;
        require_finite("rotation constant", constant.value)?;
    }
    logical
        .rotation_synthesis_constants
        .sort_by(|left, right| left.name.cmp(&right.name));
    if logical
        .rotation_synthesis_constants
        .windows(2)
        .any(|pair| pair[0].name == pair[1].name)
    {
        return Err(TargetProfileError::Reference(
            "rotation constant is repeated".to_owned(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(control: u32, target: u32, symmetric: bool) -> wire::CouplingEdgeProto {
        wire::CouplingEdgeProto {
            control,
            target,
            symmetric,
        }
    }

    fn line_profile() -> wire::TargetProfileProto {
        wire::TargetProfileProto {
            schema_version: SCHEMA_VERSION,
            format: "qiskit.qpy.v1".to_owned(),
            topology: Some(wire::TopologyProto {
                num_qubits: 3,
                edges: vec![edge(1, 2, true), edge(0, 1, false)],
                one_qubit_operations: vec!["x".to_owned(), "rz".to_owned()],
                two_qubit_operations: vec!["cx".to_owned()],
            }),
            calibration: Some(wire::CalibrationProto {
                qubit_properties: vec![
                    wire::QubitPropertyProto {
                        qubit: 1,
                        name: "t1".to_owned(),
                        value: 1e-4,
                        unit: "s".to_owned(),
                    },
                    wire::QubitPropertyProto {
                        qubit: 0,
                        name: "t1".to_owned(),
                        value: 2e-4,
                        unit: "s".to_owned(),
                    },
                ],
                edge_properties: vec![wire::EdgePropertyProto {
                    control: 2,
                    target: 1,
                    name: "cx.error".to_owned(),
                    value: 0.01,
                    unit: "probability".to_owned(),
                }],
                provenance: "test".to_owned(),
            }),
            logical: None,
            backend_name: "fake_line".to_owned(),
            calibration_time: String::new(),
            provenance: String::new(),
        }
    }

    fn validate(profile: &wire::TargetProfileProto) -> Result<TargetProfile, TargetProfileError> {
        validate_target_profile(&profile.encode_to_vec())
    }

    #[test]
    fn canonical_order_is_independent_of_input_order() {
        let shuffled = validate(&line_profile()).expect("valid profile");
        let mut ordered = line_profile();
        let topology = ordered.topology.as_mut().expect("topology");
        topology.edges.reverse();
        topology.one_qubit_operations.reverse();
        ordered
            .calibration
            .as_mut()
            .expect("calibration")
            .qubit_properties
            .reverse();
        let sorted = validate(&ordered).expect("valid profile");
        assert_eq!(shuffled, sorted);
        assert_eq!(shuffled.num_qubits(), 3);
        assert_eq!(shuffled.edge_count(), 2);
        let decoded = wire::TargetProfileProto::decode(shuffled.canonical_bytes()).expect("decode");
        let topology = decoded.topology.expect("topology");
        assert_eq!(topology.edges, vec![edge(0, 1, false), edge(1, 2, true)]);
        assert_eq!(topology.one_qubit_operations, vec!["rz", "x"]);
        assert_eq!(
            validate_target_profile(shuffled.canonical_bytes())
                .expect("canonical input")
                .canonical_bytes(),
            shuffled.canonical_bytes()
        );
    }

    #[test]
    fn rejects_undeclared_references_and_bad_values() {
        let mut out_of_range = line_profile();
        out_of_range.topology.as_mut().expect("topology").edges[0].target = 3;
        assert!(matches!(
            validate(&out_of_range),
            Err(TargetProfileError::Reference(_))
        ));

        let mut duplicate = line_profile();
        let topology = duplicate.topology.as_mut().expect("topology");
        topology.edges.push(edge(2, 1, false));
        assert!(matches!(
            validate(&duplicate),
            Err(TargetProfileError::Reference(_))
        ));

        let mut dangling = line_profile();
        dangling
            .calibration
            .as_mut()
            .expect("calibration")
            .edge_properties[0]
            .target = 0;
        assert!(matches!(
            validate(&dangling),
            Err(TargetProfileError::Reference(_))
        ));

        let mut nan = line_profile();
        nan.calibration
            .as_mut()
            .expect("calibration")
            .qubit_properties[0]
            .value = f64::NAN;
        assert!(matches!(validate(&nan), Err(TargetProfileError::Value(_))));

        let mut oversized = line_profile();
        oversized.topology.as_mut().expect("topology").num_qubits = MAX_TARGET_PROFILE_QUBITS + 1;
        assert!(matches!(
            validate(&oversized),
            Err(TargetProfileError::Budget(_))
        ));

        let mut missing = line_profile();
        missing.topology = None;
        assert!(matches!(
            validate(&missing),
            Err(TargetProfileError::Malformed(_))
        ));
        assert!(matches!(
            validate_target_profile(b"\xff\xff"),
            Err(TargetProfileError::Malformed(_))
        ));
    }

    #[test]
    fn logical_facet_is_checked_and_sorted() {
        let mut profile = line_profile();
        profile.logical = Some(wire::LogicalProto {
            code_family: "surface".to_owned(),
            code_distance: 11,
            cycle_time: 1e-6,
            cycle_time_unit: "s".to_owned(),
            magic_state_cost: 100.0,
            magic_state_cost_unit: "cycles".to_owned(),
            rotation_synthesis_model: "ross-selinger".to_owned(),
            rotation_synthesis_constants: vec![
                wire::NamedConstantProto {
                    name: "b".to_owned(),
                    value: 1.0,
                },
                wire::NamedConstantProto {
                    name: "a".to_owned(),
                    value: 3.0,
                },
            ],
        });
        let valid = validate(&profile).expect("valid logical facet");
        let decoded = wire::TargetProfileProto::decode(valid.canonical_bytes()).expect("decode");
        let names: Vec<_> = decoded
            .logical
            .expect("logical")
            .rotation_synthesis_constants
            .iter()
            .map(|constant| constant.name.clone())
            .collect();
        assert_eq!(names, vec!["a", "b"]);
        profile.logical.as_mut().expect("logical").code_distance = 0;
        assert!(matches!(
            validate(&profile),
            Err(TargetProfileError::Value(_))
        ));
    }
}
