//! Native persistence and conditional iteration assumptions for resource studies.

use std::collections::{HashMap, HashSet};

use prost::Message;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    VerifiedModule, VerifiedProgram, budget,
    resource_expression_evaluation::validate_resource_value,
    resource_rules::resource_unit_name_is_valid, wire,
};

const SCHEMA_MAJOR: u32 = 1;
const SCHEMA_MINOR: u32 = 0;
const MAX_STUDY_BYTES: usize = 64 * 1024 * 1024;

/// Integrity-checked metadata from one canonical resource study.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceStudyHeader {
    schema_major: u32,
    schema_minor: u32,
    study_digest: [u8; 32],
    semantic_digest: [u8; 32],
    metric_count: usize,
    fact_count: usize,
}

impl ResourceStudyHeader {
    /// Return the accepted resource-study schema major.
    #[must_use]
    pub const fn schema_major(&self) -> u32 {
        self.schema_major
    }

    /// Return the accepted resource-study schema minor.
    #[must_use]
    pub const fn schema_minor(&self) -> u32 {
        self.schema_minor
    }

    /// Return the SHA-256 identity of the canonical unsealed study.
    #[must_use]
    pub const fn study_digest(&self) -> &[u8; 32] {
        &self.study_digest
    }

    /// Return the selected `QCore` semantic identity retained by this study.
    #[must_use]
    pub const fn semantic_digest(&self) -> &[u8; 32] {
        &self.semantic_digest
    }

    /// Return the number of checked metric values.
    #[must_use]
    pub const fn metric_count(&self) -> usize {
        self.metric_count
    }

    /// Return the number of retained source-addressable facts.
    #[must_use]
    pub const fn fact_count(&self) -> usize {
        self.fact_count
    }
}

/// Failure at the canonical resource-study persistence boundary.
#[derive(Debug, Error)]
pub enum ResourceStudyError {
    /// The study or one of its embedded semantic values is invalid.
    #[error("{0}")]
    Invalid(String),
}

/// Seal one canonical, unsealed resource study through the native verifier.
///
/// # Errors
///
/// Returns [`ResourceStudyError`] when the study is malformed, noncanonical,
/// oversized, semantically invalid, or already sealed.
pub fn seal_resource_study(payload: &[u8]) -> Result<Vec<u8>, ResourceStudyError> {
    let mut study = decode_study(payload)?;
    validate_schema(&study)?;
    if !study.study_digest.is_empty() || study.encode_to_vec() != payload {
        return invalid("Q4255: resource study must be canonical and unsealed");
    }
    verify_study(&study)?;
    study.study_digest = Sha256::digest(study.encode_to_vec()).to_vec();
    let sealed = study.encode_to_vec();
    if sealed.len() > MAX_STUDY_BYTES {
        return invalid("Q4250: resource study exceeds the wire-size limit");
    }
    Ok(sealed)
}

/// Inspect one integrity-sealed canonical resource study.
///
/// # Errors
///
/// Returns [`ResourceStudyError`] when schema, integrity, canonicality, or any
/// embedded semantic invariant fails.
pub fn inspect_resource_study(payload: &[u8]) -> Result<ResourceStudyHeader, ResourceStudyError> {
    let mut study = decode_study(payload)?;
    validate_schema(&study)?;
    let observed: [u8; 32] = study
        .study_digest
        .as_slice()
        .try_into()
        .map_err(|_| ResourceStudyError::Invalid("Q4254: invalid study digest".to_owned()))?;
    study.study_digest.clear();
    let expected: [u8; 32] = Sha256::digest(study.encode_to_vec()).into();
    if observed != expected {
        return invalid("Q4254: resource study digest mismatch");
    }
    study.study_digest.extend_from_slice(&observed);
    if study.encode_to_vec() != payload {
        return invalid("Q4255: resource study is not canonically encoded");
    }
    let (metric_count, fact_count, semantic_digest) = verify_study(&study)?;
    Ok(ResourceStudyHeader {
        schema_major: study.schema_major,
        schema_minor: study.schema_minor,
        study_digest: observed,
        semantic_digest,
        metric_count,
        fact_count,
    })
}

/// Interpretation of supplied while iteration counts in a resource study.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IterationAssumption {
    /// A user-selected scenario, conditional on the stated execution counts.
    Scenario,
    /// A user assertion that each selected count is at most the supplied value.
    UpperBound,
}

/// Bind while resource symbols without changing the retained executable.
///
/// # Errors
///
/// Rejects invalid studies, duplicate or absent symbols, non-natural counts,
/// and oversized input. Unsupported monotonicity retains an explicit unknown.
#[allow(clippy::too_many_lines)]
pub fn assume_resource_iterations(
    payload: &[u8],
    bindings: &[(String, String)],
    mode: IterationAssumption,
) -> Result<Vec<u8>, ResourceStudyError> {
    use crate::resource_expression_evaluation::{
        free_resource_expression_symbols, substitute_resource_expression,
    };
    inspect_resource_study(payload)?;
    if bindings.is_empty() || bindings.len() > 4096 {
        return invalid("Q4256: iteration assumptions require 1..4096 bindings");
    }
    let mut study = decode_study(payload)?;
    let mut selected = HashMap::new();
    let mut replacements = Vec::new();
    for (identity, count) in bindings {
        if identity.len() > 65536
            || count.len() > 4096
            || count.is_empty()
            || !count.bytes().all(|byte| byte.is_ascii_digit())
            || (count.len() > 1 && count.starts_with('0'))
            || selected.insert(identity.clone(), count.clone()).is_some()
        {
            return invalid("Q4256: invalid or duplicate iteration assumption");
        }
        let graph = wire::ResourceExpressionGraphProto {
            nodes: vec![wire::ResourceNodeProto {
                node: Some(wire::resource_node_proto::Node::Expression(
                    wire::ResourceExpressionNodeProto {
                        kind: "literal".to_owned(),
                        domain: wire::ResourceDomainProto::ResourceNatural as i32,
                        unit: "1".to_owned(),
                        literal: Some(wire::ResourceNumberProto {
                            value: Some(wire::resource_number_proto::Value::Integer(count.clone())),
                        }),
                        ..Default::default()
                    },
                )),
            }],
            root: 0,
        };
        replacements.push((identity.clone(), graph.encode_to_vec(), true));
    }
    let mut observed = HashSet::new();
    for item in &mut study.metrics {
        let value = item
            .value
            .as_mut()
            .ok_or_else(|| ResourceStudyError::Invalid("Q4256: missing metric".to_owned()))?;
        let graphs = [&value.expression, &value.lower, &value.upper];
        let mut affected = Vec::new();
        for graph in graphs.into_iter().flatten() {
            let symbols = free_resource_expression_symbols(&graph.encode_to_vec())
                .map_err(|error| ResourceStudyError::Invalid(error.to_string()))?;
            for symbol in symbols {
                let identity = format!("{}::{}", symbol.scope(), symbol.name());
                if selected.contains_key(&identity) {
                    if symbol.origin() != "while_iterations" {
                        return invalid("Q4256: assumption does not name a while iteration symbol");
                    }
                    observed.insert(identity.clone());
                    affected.push(identity);
                }
            }
        }
        affected.sort();
        affected.dedup();
        if affected.is_empty() {
            continue;
        }
        if mode == IterationAssumption::UpperBound {
            let source = match wire::ResourceRelationProto::try_from(value.relation) {
                Ok(
                    wire::ResourceRelationProto::ResourceEquality
                    | wire::ResourceRelationProto::ResourceUpperBound,
                ) => value.expression.as_ref(),
                _ => value.upper.as_ref(),
            };
            if source.is_some_and(|graph| iteration_monotone(graph, &selected)) {
                value.expression = source.cloned();
                value.upper = None;
                value.lower = None;
                value.relation = wire::ResourceRelationProto::ResourceUpperBound as i32;
            } else {
                value.expression = None;
                value.lower = None;
                value.upper = None;
                value.relation = wire::ResourceRelationProto::ResourceUnknown as i32;
                value.unknown_contributors.push(
                    "while iteration upper bound requires a monotonic resource expression"
                        .to_owned(),
                );
            }
        }
        for graph in [&mut value.expression, &mut value.lower, &mut value.upper]
            .into_iter()
            .flatten()
        {
            let result =
                substitute_resource_expression(&graph.encode_to_vec(), replacements.clone())
                    .map_err(|error| ResourceStudyError::Invalid(error.to_string()))?;
            *graph = wire::ResourceExpressionGraphProto::decode(result.as_slice())
                .map_err(|error| ResourceStudyError::Invalid(error.to_string()))?;
        }
        for identity in affected {
            let relation = if mode == IterationAssumption::Scenario {
                "="
            } else {
                "<="
            };
            value.assumptions.push(format!(
                "{identity} {relation} {} (user assumption)",
                selected[&identity]
            ));
        }
        value.evidence.push(wire::ResourceEvidenceProto {
            authority: if mode == IterationAssumption::Scenario {
                wire::ResourceEvidenceAuthorityProto::ResourceHeuristicModel as i32
            } else {
                wire::ResourceEvidenceAuthorityProto::ResourceTrustedAxiom as i32
            },
            reference: "qamomile.while.iteration-assumption.v1".to_owned(),
            ..Default::default()
        });
    }
    if observed.len() != selected.len() {
        return invalid("Q4256: absent while iteration symbol");
    }
    study.derivations.push("display-evaluation".to_owned());
    study.derivations.push(
        if mode == IterationAssumption::Scenario {
            "while-iteration-scenario"
        } else {
            "while-iteration-upper-bound"
        }
        .to_owned(),
    );
    study.study_digest.clear();
    seal_resource_study(&study.encode_to_vec())
}

/// Check a deliberately small nonnegative monotone expression vocabulary.
fn iteration_monotone(
    graph: &wire::ResourceExpressionGraphProto,
    selected: &HashMap<String, String>,
) -> bool {
    let mut dependent = Vec::with_capacity(graph.nodes.len());
    let mut monotone = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        let Some(wire::resource_node_proto::Node::Expression(expression)) = &node.node else {
            return false;
        };
        let direct = expression.symbol.as_ref().is_some_and(|symbol| {
            selected.contains_key(&format!("{}::{}", symbol.scope, symbol.name))
        });
        let depends = direct
            || expression
                .arguments
                .iter()
                .any(|id| dependent[*id as usize]);
        let valid = !depends || (expression.domain == wire::ResourceDomainProto::ResourceNatural as i32
            && expression.predicates.is_empty()
            && (direct || (matches!(expression.kind.as_str(), "add" | "multiply" | "minimum" | "maximum")
                && expression.arguments.iter().all(|id| monotone[*id as usize]
                    && matches!(&graph.nodes[*id as usize].node, Some(wire::resource_node_proto::Node::Expression(arg)) if arg.domain == wire::ResourceDomainProto::ResourceNatural as i32)))));
        dependent.push(depends);
        monotone.push(valid);
    }
    monotone[graph.root as usize]
}

fn decode_study(payload: &[u8]) -> Result<wire::ResourceStudyProto, ResourceStudyError> {
    if payload.len() > MAX_STUDY_BYTES {
        return invalid("Q4250: resource study exceeds the wire-size limit");
    }
    budget::preflight_resource_study(payload).map_err(|error| match error {
        budget::PreflightError::Budget(detail) => ResourceStudyError::Invalid(format!(
            "Q4250: resource study violates its structural budget: {detail}"
        )),
        budget::PreflightError::Malformed(detail) => {
            ResourceStudyError::Invalid(format!("Q4251: {detail}"))
        }
    })?;
    wire::ResourceStudyProto::decode(payload)
        .map_err(|error| ResourceStudyError::Invalid(format!("Q4251: {error}")))
}

fn validate_schema(study: &wire::ResourceStudyProto) -> Result<(), ResourceStudyError> {
    if study.schema_major != SCHEMA_MAJOR {
        return invalid(&format!(
            "Q4252: unsupported resource study schema major {}; expected {SCHEMA_MAJOR}",
            study.schema_major
        ));
    }
    if study.schema_minor > SCHEMA_MINOR {
        return invalid(&format!(
            "Q4253: unsupported resource study schema minor {}",
            study.schema_minor
        ));
    }
    Ok(())
}

fn verify_study(
    study: &wire::ResourceStudyProto,
) -> Result<(usize, usize, [u8; 32]), ResourceStudyError> {
    let bundle = study
        .bundle
        .as_ref()
        .ok_or_else(|| ResourceStudyError::Invalid("Q4256: missing analysis bundle".to_owned()))?;
    let mut previous: Option<&str> = None;
    let mut metric_definitions = HashMap::with_capacity(study.metrics.len());
    for item in &study.metrics {
        if item.identifier.is_empty()
            || previous.is_some_and(|identifier| identifier >= item.identifier.as_str())
        {
            return invalid("Q4256: resource metrics are not strictly identity-ordered");
        }
        previous = Some(&item.identifier);
        let value = item.value.as_ref().ok_or_else(|| {
            ResourceStudyError::Invalid("Q4256: resource metric has no value".to_owned())
        })?;
        let metric = value.metric.as_ref().ok_or_else(|| {
            ResourceStudyError::Invalid("Q4256: resource value has no metric".to_owned())
        })?;
        if item.identifier != metric.identifier {
            return invalid("Q4256: resource metric key disagrees with its definition");
        }
        validate_value(value, metric)?;
        metric_definitions.insert(item.identifier.as_str(), metric);
    }
    validate_bundle(bundle, &metric_definitions)?;
    if study.derivations.iter().any(String::is_empty) {
        return invalid("Q4256: resource derivations must be nonempty");
    }
    let semantic_digest = validate_bundle_qcore(bundle)?;
    Ok((study.metrics.len(), bundle.facts.len(), semantic_digest))
}

fn validate_bundle(
    bundle: &wire::ResourceAnalysisBundleProto,
    metrics: &HashMap<&str, &wire::ResourceMetricProto>,
) -> Result<(), ResourceStudyError> {
    if bundle.entrypoint.is_empty()
        || bundle.analysis_version.is_empty()
        || bundle.qcore_payload.is_empty()
    {
        return invalid("Q4256: analysis bundle identities must be nonempty");
    }
    let mut identities = HashSet::with_capacity(bundle.facts.len());
    for fact in &bundle.facts {
        if fact.kind.is_empty() || fact.owner.is_empty() || fact.source.is_empty() {
            return invalid("Q4256: analysis fact identities must be nonempty");
        }
        let Some(wire::resource_analysis_fact_proto::Value::Operation(operation)) =
            fact.value.as_ref()
        else {
            return invalid("Q4256: analysis fact has no typed value");
        };
        if fact.kind != "operation" || operation.opcode.is_empty() {
            return invalid("Q4256: typed operation fact identity is invalid");
        }
        if !identities.insert((
            fact.owner.as_str(),
            fact.source.as_str(),
            operation.opcode.as_str(),
        )) {
            return invalid("Q4256: duplicate source-local operation fact");
        }
    }
    for fact in &bundle.facts {
        let Some(wire::resource_analysis_fact_proto::Value::Operation(operation)) =
            fact.value.as_ref()
        else {
            return invalid("Q4256: analysis fact has no typed value");
        };
        validate_contributions(&operation.contributions, metrics)?;
    }
    Ok(())
}

fn validate_contributions(
    contributions: &[wire::ResourceContributionProto],
    metrics: &HashMap<&str, &wire::ResourceMetricProto>,
) -> Result<(), ResourceStudyError> {
    if contributions.is_empty() {
        return invalid("Q4256: operation fact has no resource contributions");
    }
    let mut previous: Option<&str> = None;
    for contribution in contributions {
        if contribution.identifier.is_empty()
            || previous.is_some_and(|identifier| identifier >= contribution.identifier.as_str())
        {
            return invalid("Q4256: fact contributions are not strictly identity-ordered");
        }
        previous = Some(&contribution.identifier);
    }
    for contribution in contributions {
        let metric = metrics
            .get(contribution.identifier.as_str())
            .ok_or_else(|| {
                ResourceStudyError::Invalid(
                    "Q4256: fact contribution names an unknown metric".to_owned(),
                )
            })?;
        validate_contribution(contribution, metric)?;
    }
    Ok(())
}

fn validate_contribution(
    contribution: &wire::ResourceContributionProto,
    metric: &wire::ResourceMetricProto,
) -> Result<(), ResourceStudyError> {
    if contribution.identifier != metric.identifier {
        return invalid("Q4256: fact contribution metric identity disagrees");
    }
    let mut unknown = HashSet::with_capacity(contribution.unknown_contributors.len());
    if contribution
        .unknown_contributors
        .iter()
        .any(|reason| reason.is_empty() || !unknown.insert(reason))
    {
        return invalid("Q4256: fact contribution unknowns must be nonempty and unique");
    }
    let relation = enum_value::<wire::ResourceRelationProto>(contribution.relation)?;
    if !matches!(
        relation,
        wire::ResourceRelationProto::ResourceEquality
            | wire::ResourceRelationProto::ResourceEnclosure
            | wire::ResourceRelationProto::ResourcePartialEnclosure
    ) {
        return invalid("Q4256: fact contribution relation is invalid");
    }
    if relation == wire::ResourceRelationProto::ResourcePartialEnclosure
        && contribution.lower.is_none()
    {
        return invalid("Q4256: partial fact contribution requires a lower endpoint");
    }
    let domain = resource_domain(metric)?;
    let expression = contribution.expression.as_ref().map(Message::encode_to_vec);
    let lower = contribution.lower.as_ref().map(Message::encode_to_vec);
    let upper = contribution.upper.as_ref().map(Message::encode_to_vec);
    validate_resource_value(
        domain,
        &metric.unit,
        resource_relation(relation)?,
        expression.as_deref(),
        lower.as_deref(),
        upper.as_deref(),
        contribution.unknown_contributors.len(),
    )
    .map_err(|error| ResourceStudyError::Invalid(format!("Q4256: {error}")))
}

fn validate_bundle_qcore(
    bundle: &wire::ResourceAnalysisBundleProto,
) -> Result<[u8; 32], ResourceStudyError> {
    if let Ok(program) = VerifiedProgram::from_bytes(&bundle.qcore_payload) {
        if program.header().entrypoint != bundle.entrypoint {
            return invalid("Q4256: analysis bundle entrypoint disagrees with QCore");
        }
        return Ok(program.semantic_digest());
    }
    let module = VerifiedModule::from_bytes(&bundle.qcore_payload)
        .map_err(|error| ResourceStudyError::Invalid(format!("Q4256: {error}")))?;
    module
        .selected_semantic_digest(&bundle.entrypoint)
        .map_err(|error| ResourceStudyError::Invalid(format!("Q4256: {error}")))
}

fn validate_value(
    value: &wire::ResourceValueProto,
    metric: &wire::ResourceMetricProto,
) -> Result<(), ResourceStudyError> {
    validate_metric(metric)?;
    if value
        .unknown_contributors
        .iter()
        .chain(value.assumptions.iter())
        .any(String::is_empty)
    {
        return invalid("Q4256: resource explanations must be nonempty");
    }
    for evidence in &value.evidence {
        let authority = enum_value::<wire::ResourceEvidenceAuthorityProto>(evidence.authority)?;
        if authority == wire::ResourceEvidenceAuthorityProto::ResourceEvidenceAuthorityUnspecified
            || evidence.reference.is_empty()
        {
            return invalid("Q4256: resource evidence is invalid");
        }
        validate_attributes(&evidence.details, "resource evidence details")?;
    }
    let relation = resource_relation(enum_value::<wire::ResourceRelationProto>(value.relation)?)?;
    let domain = resource_domain(metric)?;
    let expression = value.expression.as_ref().map(Message::encode_to_vec);
    let lower = value.lower.as_ref().map(Message::encode_to_vec);
    let upper = value.upper.as_ref().map(Message::encode_to_vec);
    validate_resource_value(
        domain,
        &metric.unit,
        relation,
        expression.as_deref(),
        lower.as_deref(),
        upper.as_deref(),
        value.unknown_contributors.len(),
    )
    .map_err(|error| ResourceStudyError::Invalid(format!("Q4256: {error}")))
}

fn resource_relation(
    relation: wire::ResourceRelationProto,
) -> Result<&'static str, ResourceStudyError> {
    Ok(match relation {
        wire::ResourceRelationProto::ResourceEquality => "equality",
        wire::ResourceRelationProto::ResourceLowerBound => "lower_bound",
        wire::ResourceRelationProto::ResourceUpperBound => "upper_bound",
        wire::ResourceRelationProto::ResourceEnclosure => "enclosure",
        wire::ResourceRelationProto::ResourcePartialEnclosure => "partial_enclosure",
        wire::ResourceRelationProto::ResourceUnknown => "unknown",
        wire::ResourceRelationProto::ResourceRelationUnspecified => {
            return invalid("Q4256: resource relation is unspecified");
        }
    })
}

fn resource_domain(metric: &wire::ResourceMetricProto) -> Result<&'static str, ResourceStudyError> {
    Ok(
        match enum_value::<wire::ResourceDomainProto>(metric.domain)? {
            wire::ResourceDomainProto::ResourceNatural => "natural",
            wire::ResourceDomainProto::ResourceInteger => "integer",
            wire::ResourceDomainProto::ResourceRational => "rational",
            wire::ResourceDomainProto::ResourceReal => "real",
            wire::ResourceDomainProto::ResourceProbability => "probability",
            wire::ResourceDomainProto::ResourceBoolean => "boolean",
            wire::ResourceDomainProto::ResourceDomainUnspecified => {
                return invalid("Q4256: resource metric domain is unspecified");
            }
        },
    )
}

fn validate_attributes(
    attributes: &[wire::ResourceAttributeProto],
    context: &str,
) -> Result<(), ResourceStudyError> {
    let mut names = HashSet::with_capacity(attributes.len());
    if attributes
        .iter()
        .any(|attribute| attribute.name.is_empty() || !names.insert(&attribute.name))
    {
        return invalid(&format!(
            "Q4256: {context} contains duplicate or empty names"
        ));
    }
    Ok(())
}

fn validate_metric(metric: &wire::ResourceMetricProto) -> Result<(), ResourceStudyError> {
    if metric.identifier.is_empty()
        || !metric.identifier.contains('.')
        || metric.version.is_empty()
        || !resource_unit_name_is_valid(&metric.unit)
    {
        return invalid("Q4256: resource metric identity, version, or unit is invalid");
    }
    if enum_value::<wire::ResourceCoordinateProto>(metric.coordinate)?
        == wire::ResourceCoordinateProto::ResourceCoordinateUnspecified
        || enum_value::<wire::ResourceCompositionProto>(metric.composition)?
            == wire::ResourceCompositionProto::ResourceCompositionUnspecified
    {
        return invalid("Q4256: resource metric coordinate or composition is unspecified");
    }
    Ok(())
}

fn enum_value<T>(value: i32) -> Result<T, ResourceStudyError>
where
    T: TryFrom<i32>,
{
    T::try_from(value)
        .map_err(|_| ResourceStudyError::Invalid("Q4256: unknown resource enum value".to_owned()))
}

fn invalid<T>(message: &str) -> Result<T, ResourceStudyError> {
    Err(ResourceStudyError::Invalid(message.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unknown_metric(identifier: &str) -> wire::ResourceMetricValueProto {
        wire::ResourceMetricValueProto {
            identifier: identifier.to_owned(),
            value: Some(wire::ResourceValueProto {
                metric: Some(wire::ResourceMetricProto {
                    identifier: identifier.to_owned(),
                    coordinate: wire::ResourceCoordinateProto::ResourceLogical.into(),
                    domain: wire::ResourceDomainProto::ResourceNatural.into(),
                    unit: "semantic_operations".to_owned(),
                    composition: wire::ResourceCompositionProto::ResourceAdditiveWork.into(),
                    version: "1".to_owned(),
                }),
                relation: wire::ResourceRelationProto::ResourceUnknown.into(),
                unknown_contributors: vec!["opaque".to_owned()],
                ..Default::default()
            }),
        }
    }

    fn typed_study(
        contributions: Vec<wire::ResourceContributionProto>,
    ) -> wire::ResourceStudyProto {
        let metrics = contributions
            .iter()
            .map(|contribution| unknown_metric(&contribution.identifier))
            .collect();
        wire::ResourceStudyProto {
            schema_major: SCHEMA_MAJOR,
            schema_minor: SCHEMA_MINOR,
            bundle: Some(wire::ResourceAnalysisBundleProto {
                entrypoint: "test.main".to_owned(),
                analysis_version: "1".to_owned(),
                qcore_payload: vec![1],
                facts: vec![wire::ResourceAnalysisFactProto {
                    kind: "operation".to_owned(),
                    owner: "test.main".to_owned(),
                    source: "test.py:1:0-1:1".to_owned(),
                    value: Some(wire::resource_analysis_fact_proto::Value::Operation(
                        wire::ResourceOperationFactProto {
                            opcode: "quantum.h".to_owned(),
                            contributions,
                        },
                    )),
                }],
            }),
            metrics,
            ..Default::default()
        }
    }

    #[test]
    fn rejects_unknown_schema_before_decoding_embedded_values() {
        let payload = wire::ResourceStudyProto {
            schema_major: 2,
            ..Default::default()
        }
        .encode_to_vec();

        assert!(
            seal_resource_study(&payload)
                .expect_err("unknown schema must fail")
                .to_string()
                .starts_with("Q4252:")
        );

        let payload = wire::ResourceStudyProto {
            schema_major: SCHEMA_MAJOR,
            schema_minor: SCHEMA_MINOR + 1,
            ..Default::default()
        }
        .encode_to_vec();
        assert!(
            seal_resource_study(&payload)
                .expect_err("future schema minor must fail")
                .to_string()
                .starts_with("Q4253:")
        );
    }

    #[test]
    fn rejects_noncanonical_metric_order_before_qcore_verification() {
        let payload = wire::ResourceStudyProto {
            schema_major: SCHEMA_MAJOR,
            schema_minor: SCHEMA_MINOR,
            bundle: Some(wire::ResourceAnalysisBundleProto {
                entrypoint: "test.main".to_owned(),
                analysis_version: "1".to_owned(),
                qcore_payload: vec![1],
                ..Default::default()
            }),
            metrics: vec![unknown_metric("z.metric"), unknown_metric("a.metric")],
            ..Default::default()
        }
        .encode_to_vec();

        assert!(
            seal_resource_study(&payload)
                .expect_err("unordered metrics must fail")
                .to_string()
                .contains("strictly identity-ordered")
        );
    }

    #[test]
    fn rejects_noncanonical_and_duplicate_typed_facts() {
        let contribution = |identifier: &str| wire::ResourceContributionProto {
            identifier: identifier.to_owned(),
            relation: wire::ResourceRelationProto::ResourcePartialEnclosure.into(),
            unknown_contributors: vec!["opaque".to_owned()],
            ..Default::default()
        };
        let mut unordered = typed_study(vec![contribution("z.metric"), contribution("a.metric")]);
        unordered
            .metrics
            .sort_by(|left, right| left.identifier.cmp(&right.identifier));
        assert!(
            seal_resource_study(&unordered.encode_to_vec())
                .expect_err("unordered contributions must fail")
                .to_string()
                .contains("strictly identity-ordered")
        );

        let mut duplicate = typed_study(vec![contribution("a.metric")]);
        let repeated = duplicate.bundle.as_ref().expect("bundle").facts[0].clone();
        duplicate
            .bundle
            .as_mut()
            .expect("bundle")
            .facts
            .push(repeated);
        assert!(
            seal_resource_study(&duplicate.encode_to_vec())
                .expect_err("duplicate lexical fact must fail")
                .to_string()
                .contains("duplicate source-local")
        );
    }

    #[test]
    fn rejects_contradictory_typed_contribution_relations() {
        let study = typed_study(vec![wire::ResourceContributionProto {
            identifier: "a.metric".to_owned(),
            relation: wire::ResourceRelationProto::ResourceEquality.into(),
            unknown_contributors: vec!["opaque".to_owned()],
            ..Default::default()
        }]);

        assert!(
            seal_resource_study(&study.encode_to_vec())
                .expect_err("contradictory equality must fail")
                .to_string()
                .starts_with("Q4256:")
        );
    }
}
