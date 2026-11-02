//! Canonical persistence for target artifact families.
//!
//! The public payload format deliberately remains the existing portable
//! `LiteralProto` tuple. This module owns its introduction and elimination so
//! Python never assigns a second meaning to the selector, variant order, or
//! child artifact bytes.

use prost::Message;
use thiserror::Error;

use crate::artifact::inspect_artifact;
use crate::budget;
use crate::family_selection::inspect_circuit_family_selection;
use crate::literals::{
    bool_literal, bool_value, bytes_literal, bytes_value, int_literal, int_value, string_literal,
    string_value, tuple_literal, tuple_values,
};
use crate::wire;
use crate::{CircuitDispatchDecision, CircuitDispatchParameter};

const ARTIFACT_FAMILY_KIND: &str = "qamomile.artifact-family";
const ARTIFACT_FAMILY_VERSION: i64 = 3;

/// Maximum encoded size of one artifact-family payload.
pub const MAX_ARTIFACT_FAMILY_BYTES: usize = 64 * 1024 * 1024;

/// One selector path paired with an integrity-sealed child artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactFamilyVariantInput {
    decisions: Vec<(usize, bool)>,
    artifact: Vec<u8>,
}

impl ArtifactFamilyVariantInput {
    /// Create one unverified child input for native family sealing.
    #[must_use]
    pub const fn new(decisions: Vec<(usize, bool)>, artifact: Vec<u8>) -> Self {
        Self {
            decisions,
            artifact,
        }
    }

    /// Return the redundant selector path that must match the native plan.
    #[must_use]
    pub fn decisions(&self) -> &[(usize, bool)] {
        &self.decisions
    }

    /// Return the integrity-sealed child artifact bytes.
    #[must_use]
    pub fn artifact(&self) -> &[u8] {
        &self.artifact
    }
}

/// One checked backend child and its canonical selector path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactFamilyVariant {
    decisions: Box<[CircuitDispatchDecision]>,
    artifact: Box<[u8]>,
}

impl ArtifactFamilyVariant {
    /// Return the canonical predicate decisions selecting this child.
    #[must_use]
    pub fn decisions(&self) -> &[CircuitDispatchDecision] {
        &self.decisions
    }

    /// Return one integrity-sealed child artifact.
    #[must_use]
    pub fn artifact(&self) -> &[u8] {
        &self.artifact
    }
}

/// One fully checked persistent artifact-family payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactFamilyPayload {
    target_format: String,
    dispatch_parameters: Box<[CircuitDispatchParameter]>,
    selection_plan: Box<[u8]>,
    variants: Box<[ArtifactFamilyVariant]>,
}

impl ArtifactFamilyPayload {
    /// Return the backend format shared by every child artifact.
    #[must_use]
    pub fn target_format(&self) -> &str {
        &self.target_format
    }

    /// Return the source-ordered parameters consumed by family selection.
    #[must_use]
    pub fn dispatch_parameters(&self) -> &[CircuitDispatchParameter] {
        &self.dispatch_parameters
    }

    /// Return the canonical native selector bytes.
    #[must_use]
    pub fn selection_plan(&self) -> &[u8] {
        &self.selection_plan
    }

    /// Return every child in canonical false-first selector order.
    #[must_use]
    pub fn variants(&self) -> &[ArtifactFamilyVariant] {
        &self.variants
    }
}

/// Failure to seal or inspect a persistent artifact family.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ArtifactFamilyError {
    /// The payload is malformed, noncanonical, inconsistent, or excessive.
    #[error("Q4463: invalid native artifact-family payload: {0}")]
    Invalid(String),
}

/// Seal one artifact family in the existing portable v3 representation.
///
/// The redundant dispatch identities and decision rows are accepted only when
/// they exactly match the native selection plan. Every child must already be a
/// canonical artifact for `target_format`.
///
/// # Errors
///
/// Returns [`ArtifactFamilyError`] when any supplied fact disagrees with the
/// selector or a child artifact is invalid.
pub fn seal_artifact_family_payload(
    target_format: &str,
    dispatch_parameters: &[(usize, String)],
    selection_plan: &[u8],
    variants: &[ArtifactFamilyVariantInput],
) -> Result<Vec<u8>, ArtifactFamilyError> {
    let parameters = dispatch_parameters
        .iter()
        .map(|(index, name)| {
            Ok(tuple_literal(vec![
                int_literal(portable_index(*index)?),
                string_literal(name),
            ]))
        })
        .collect::<Result<Vec<_>, ArtifactFamilyError>>()?;
    let variants = variants
        .iter()
        .map(|variant| {
            let decisions = variant
                .decisions()
                .iter()
                .map(|(index, value)| {
                    Ok(tuple_literal(vec![
                        int_literal(portable_index(*index)?),
                        bool_literal(*value),
                    ]))
                })
                .collect::<Result<Vec<_>, ArtifactFamilyError>>()?;
            Ok(tuple_literal(vec![
                tuple_literal(decisions),
                bytes_literal(variant.artifact()),
            ]))
        })
        .collect::<Result<Vec<_>, ArtifactFamilyError>>()?;
    let payload = tuple_literal(vec![
        string_literal(ARTIFACT_FAMILY_KIND),
        int_literal(ARTIFACT_FAMILY_VERSION),
        string_literal(target_format),
        tuple_literal(parameters),
        bytes_literal(selection_plan),
        tuple_literal(variants),
    ])
    .encode_to_vec();
    inspect_artifact_family_payload(&payload)?;
    Ok(payload)
}

/// Decode and fully validate one untrusted artifact-family payload.
///
/// # Errors
///
/// Returns [`ArtifactFamilyError`] when the payload exceeds a structural
/// budget, is noncanonical, disagrees with its selector, or contains an
/// invalid child artifact.
pub fn inspect_artifact_family_payload(
    payload: &[u8],
) -> Result<ArtifactFamilyPayload, ArtifactFamilyError> {
    if payload.len() > MAX_ARTIFACT_FAMILY_BYTES {
        return Err(invalid("payload exceeds its byte budget"));
    }
    budget::preflight_literal(payload).map_err(|error| invalid(&preflight_reason(error)))?;
    let encoded =
        wire::LiteralProto::decode(payload).map_err(|error| invalid(&error.to_string()))?;
    if encoded.encode_to_vec() != payload {
        return Err(invalid("payload is not canonically encoded"));
    }
    let root = tuple_values(&encoded).ok_or_else(|| invalid("header is not a tuple"))?;
    if root.len() != 6
        || string_value(&root[0]) != Some(ARTIFACT_FAMILY_KIND)
        || int_value(&root[1]) != Some(ARTIFACT_FAMILY_VERSION)
    {
        return Err(invalid("unsupported or malformed header"));
    }
    let target_format = string_value(&root[2])
        .filter(|value| !value.is_empty())
        .ok_or_else(|| invalid("target format is missing"))?;
    let encoded_parameters =
        tuple_values(&root[3]).ok_or_else(|| invalid("dispatch parameters are not a tuple"))?;
    let selection_plan = bytes_value(&root[4])
        .filter(|value| !value.is_empty())
        .ok_or_else(|| invalid("selection plan is missing"))?;
    let encoded_variants = tuple_values(&root[5])
        .filter(|variants| !variants.is_empty())
        .ok_or_else(|| invalid("variants are missing"))?;
    let plan = inspect_circuit_family_selection(selection_plan)
        .map_err(|error| invalid(&error.to_string()))?;
    if encoded_parameters.len() != plan.dispatch_parameters().len()
        || encoded_variants.len() != plan.variants().len()
    {
        return Err(invalid("selector dimensions disagree with the payload"));
    }
    for (encoded, expected) in encoded_parameters.iter().zip(plan.dispatch_parameters()) {
        let row = tuple_values(encoded)
            .filter(|row| row.len() == 2)
            .ok_or_else(|| invalid("dispatch parameter is malformed"))?;
        let index = int_value(&row[0])
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| invalid("dispatch parameter index is invalid"))?;
        let name = string_value(&row[1])
            .filter(|value| !value.is_empty())
            .ok_or_else(|| invalid("dispatch parameter name is invalid"))?;
        if index != expected.parameter_index() || name != expected.name() {
            return Err(invalid("dispatch parameters disagree with the selector"));
        }
    }
    let mut variants = Vec::with_capacity(encoded_variants.len());
    for (encoded, expected) in encoded_variants.iter().zip(plan.variants()) {
        let row = tuple_values(encoded)
            .filter(|row| row.len() == 2)
            .ok_or_else(|| invalid("variant is malformed"))?;
        let decisions =
            tuple_values(&row[0]).ok_or_else(|| invalid("variant decisions are not a tuple"))?;
        if decisions.len() != expected.decisions().len() {
            return Err(invalid("variant decisions disagree with the selector"));
        }
        for (encoded, expected) in decisions.iter().zip(expected.decisions()) {
            let decision = tuple_values(encoded)
                .filter(|decision| decision.len() == 2)
                .ok_or_else(|| invalid("variant decision is malformed"))?;
            let index = int_value(&decision[0])
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| invalid("variant decision index is invalid"))?;
            let value = bool_value(&decision[1])
                .ok_or_else(|| invalid("variant decision value is invalid"))?;
            if index != expected.predicate_index() || value != expected.value() {
                return Err(invalid("variant decisions disagree with the selector"));
            }
        }
        let artifact = bytes_value(&row[1])
            .filter(|value| !value.is_empty())
            .ok_or_else(|| invalid("child artifact is missing"))?;
        let header = inspect_artifact(artifact)
            .map_err(|error| invalid(&format!("invalid child artifact: {error}")))?;
        if header.format != target_format {
            return Err(invalid("child artifact format disagrees with the family"));
        }
        variants.push(ArtifactFamilyVariant {
            decisions: expected.decisions().to_vec().into_boxed_slice(),
            artifact: artifact.to_vec().into_boxed_slice(),
        });
    }
    Ok(ArtifactFamilyPayload {
        target_format: target_format.to_owned(),
        dispatch_parameters: plan.dispatch_parameters().to_vec().into_boxed_slice(),
        selection_plan: selection_plan.to_vec().into_boxed_slice(),
        variants: variants.into_boxed_slice(),
    })
}

fn portable_index(index: usize) -> Result<i64, ArtifactFamilyError> {
    i64::try_from(index).map_err(|_| invalid("index exceeds the portable integer range"))
}

fn invalid(reason: &str) -> ArtifactFamilyError {
    ArtifactFamilyError::Invalid(reason.to_owned())
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
    use crate::artifact::seal_artifact;
    use crate::result_postprocessing::encode_scalar;
    use crate::{CircuitScalarNode, CircuitScalarType, SCHEMA_MAJOR, SCHEMA_MINOR};

    fn scalar(name: &str) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Scalar.into(),
            name: name.to_owned(),
            ..Default::default()
        }
    }

    fn child_artifact(format: &str) -> Vec<u8> {
        seal_artifact(
            &wire::ArtifactProto {
                schema_major: SCHEMA_MAJOR,
                schema_minor: SCHEMA_MINOR,
                format: format.to_owned(),
                payload: b"child".to_vec(),
                target_fingerprint: "probe-v1".to_owned(),
                result_abi: Some(wire::ResultAbiProto {
                    r#type: Some(scalar("Unit")),
                    ..Default::default()
                }),
                report: Some(wire::CompilationReportProto {
                    resource_study: b"study".to_vec(),
                    ..Default::default()
                }),
                ..Default::default()
            }
            .encode_to_vec(),
        )
        .expect("child artifact seals")
    }

    fn selection_plan() -> Vec<u8> {
        let parameter = CircuitScalarNode::Parameter {
            name: "enabled".to_owned(),
            path: Box::new([]),
            scalar_type: CircuitScalarType::Bool,
        };
        wire::CircuitFamilySelectionProto {
            schema_major: 1,
            schema_minor: 0,
            dispatch_parameters: vec![wire::CircuitFamilyDispatchParameterProto {
                parameter_index: 0,
                name: "enabled".to_owned(),
                uint_width: 0,
            }],
            predicates: vec![wire::CircuitFamilyPredicateProto {
                scalars: vec![encode_scalar(&parameter).expect("parameter encodes")],
                root: 0,
            }],
            variants: vec![
                wire::CircuitFamilyVariantProto {
                    decisions: vec![wire::CircuitFamilyDecisionProto {
                        predicate_index: 0,
                        value: false,
                    }],
                },
                wire::CircuitFamilyVariantProto {
                    decisions: vec![wire::CircuitFamilyDecisionProto {
                        predicate_index: 0,
                        value: true,
                    }],
                },
            ],
        }
        .encode_to_vec()
    }

    fn family_parameters() -> Vec<(usize, String)> {
        vec![(0, "enabled".to_owned())]
    }

    fn family_variants() -> Vec<ArtifactFamilyVariantInput> {
        vec![
            ArtifactFamilyVariantInput::new(vec![(0, false)], child_artifact("probe.v1")),
            ArtifactFamilyVariantInput::new(vec![(0, true)], child_artifact("probe.v1")),
        ]
    }

    #[test]
    fn seals_and_inspects_the_existing_artifact_family_format() {
        let plan = selection_plan();
        let parameters = family_parameters();
        let variants = family_variants();
        let payload = seal_artifact_family_payload("probe.v1", &parameters, &plan, &variants)
            .expect("family seals");
        let family = inspect_artifact_family_payload(&payload).expect("family inspects");

        assert_eq!(family.target_format(), "probe.v1");
        assert_eq!(family.selection_plan(), plan);
        assert_eq!(family.dispatch_parameters()[0].name(), "enabled");
        assert_eq!(family.variants().len(), 2);
        assert!(!family.variants()[0].decisions()[0].value());
        assert!(family.variants()[1].decisions()[0].value());
    }

    #[test]
    fn rejects_selector_disagreement_and_invalid_children() {
        let plan = selection_plan();
        let parameters = family_parameters();
        let variants = family_variants();
        assert!(
            seal_artifact_family_payload("probe.v1", &[(0, "other".to_owned())], &plan, &variants,)
                .is_err()
        );

        let mut reordered = variants.clone();
        reordered.swap(0, 1);
        assert!(seal_artifact_family_payload("probe.v1", &parameters, &plan, &reordered).is_err());

        let mut invalid_child = variants;
        invalid_child[0].artifact = b"not-an-artifact".to_vec();
        assert!(
            seal_artifact_family_payload("probe.v1", &parameters, &plan, &invalid_child).is_err()
        );
    }

    #[test]
    fn borrowed_preflight_rejects_excessive_literal_nesting() {
        let mut nested = int_literal(0);
        for _ in 0..=budget::MAX_NESTING {
            nested = tuple_literal(vec![nested]);
        }

        assert!(inspect_artifact_family_payload(&nested.encode_to_vec()).is_err());
    }
}
