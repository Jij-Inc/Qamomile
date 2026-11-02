//! Canonical native boundary for declarative external resource contracts.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use num_bigint::BigInt;
use num_traits::{Signed, Zero};
use prost::Message;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    CompilerError, LoweringConfig, ProgramHeader, SymbolicLogicalCounts, SymbolicResourceError,
    VerifiedProgram, VerifiedProgramFamily, budget, symbolic_resources, wire,
};

use super::timing;

const SCHEMA_MAJOR: u32 = 1;
const SCHEMA_MINOR: u32 = 2;
const MAX_REQUEST_BYTES: usize = 64 * 1024 * 1024;
const MAX_EXPRESSION_NODES: usize = 100_000;

/// One integrity-checked analysis request and its independently verified family.
#[derive(Debug)]
pub struct VerifiedResourceAnalysisRequest {
    program: VerifiedResourceProgram,
    rules: Box<[ResourceRuleContract]>,
}

#[derive(Debug)]
enum VerifiedResourceProgram {
    Closed(VerifiedProgram),
    Family(VerifiedProgramFamily),
}

impl VerifiedResourceAnalysisRequest {
    /// Observe a selected abstraction boundary using the same resource visitor.
    ///
    /// # Errors
    /// Rejects unknown/duplicate callable selections and unsupported analysis.
    pub fn observed_counts(
        &self,
        expanded: Option<&[String]>,
        basis: Option<symbolic_resources::GateBasis>,
        realizations: &[String],
    ) -> Result<SymbolicLogicalCounts, SymbolicResourceError> {
        let local = match &self.program {
            VerifiedResourceProgram::Closed(program) => &program.local,
            VerifiedResourceProgram::Family(family) => &family.local,
        };
        let expanded = expanded
            .map(|names| {
                if names.len() > 4096 {
                    return Err(SymbolicResourceError::Unsupported(
                        "Q4261: observation selection exceeds the callable budget".to_owned(),
                    ));
                }
                let unique = names.iter().cloned().collect::<BTreeSet<_>>();
                let available = local
                    .checked
                    .package
                    .functions
                    .iter()
                    .map(|function| &function.symbol)
                    .collect::<BTreeSet<_>>();
                if unique.len() != names.len()
                    || unique.iter().any(|name| !available.contains(name))
                {
                    return Err(SymbolicResourceError::Unsupported(
                        "Q4260: invalid observation callable selection".to_owned(),
                    ));
                }
                Ok(unique)
            })
            .transpose()?;
        let lowering = LoweringConfig::from_labels(&[], realizations).map_err(|error| {
            SymbolicResourceError::Unsupported(format!(
                "Q4260: invalid resource realization selection: {error}"
            ))
        })?;
        let observation = symbolic_resources::ResourceObservation::new(expanded, basis, lowering);
        let mut counts = symbolic_resources::symbolic_observed_counts(
            &local.checked.package,
            &local.checked.type_arena,
            &local.checked.operations,
            self.rules(),
            Some(&observation),
        )?;
        if let Some(basis) = basis
            && let Ok(circuit) = crate::lowering::circuit::project_with_lowering(
                &local.checked.package,
                &local.checked.boolean_plans,
                lowering,
            )
            && let Ok(schedule) = timing::schedule_portable_circuit(&circuit, basis)
        {
            counts.refine_portable_fixed_graph(schedule.gate_count(), schedule.makespan())?;
        }
        Ok(counts)
    }

    /// Decode and verify a canonical native resource-analysis request.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceRuleError`] when the request, selected `QCore` family,
    /// expression graph, or rule-to-callable contract is malformed.
    pub fn from_bytes(payload: &[u8]) -> Result<Self, ResourceRuleError> {
        let mut request = decode_request(payload)?;
        validate_schema(&request)?;
        let observed: [u8; 32] =
            request.request_digest.as_slice().try_into().map_err(|_| {
                ResourceRuleError::Invalid("Q4224: invalid request digest".to_owned())
            })?;
        request.request_digest.clear();
        let expected: [u8; 32] = Sha256::digest(request.encode_to_vec()).into();
        if observed != expected {
            return Err(ResourceRuleError::Invalid(
                "Q4224: resource analysis request digest mismatch".to_owned(),
            ));
        }
        request.request_digest.extend_from_slice(&observed);
        if request.encode_to_vec() != payload {
            return Err(ResourceRuleError::Invalid(
                "Q4225: resource analysis request is not canonically encoded".to_owned(),
            ));
        }
        Self::verify(&request)
    }

    fn verify(request: &wire::ResourceAnalysisRequestProto) -> Result<Self, ResourceRuleError> {
        let program = match VerifiedProgramFamily::from_bytes(&request.program_family) {
            Ok(family) => VerifiedResourceProgram::Family(family),
            Err(_) => VerifiedResourceProgram::Closed(VerifiedProgram::from_bytes(
                &request.program_family,
            )?),
        };
        let functions = match &program {
            VerifiedResourceProgram::Closed(program) => &program.local.checked.package.functions,
            VerifiedResourceProgram::Family(family) => &family.local.checked.package.functions,
        };
        let mut previous_identity: Option<&str> = None;
        let mut coverage = BTreeSet::new();
        let mut footprints = BTreeMap::new();
        let mut rules = Vec::with_capacity(request.rules.len());
        for rule in &request.rules {
            if previous_identity.is_some_and(|previous| previous >= rule.identity.as_str()) {
                return Err(ResourceRuleError::Invalid(
                    "Q4226: resource rules are not strictly identity-ordered".to_owned(),
                ));
            }
            previous_identity = Some(&rule.identity);
            let contract = ResourceRuleContract::decode(rule, functions)?;
            for mode in &contract.modes {
                if !coverage.insert((
                    contract.callable.clone(),
                    *mode,
                    contract.metric.identifier.clone(),
                )) {
                    return invalid("overlapping resource-rule transform coverage");
                }
                if let Some(parameters) = contract.scheduling_footprint.as_ref()
                    && footprints
                        .insert((contract.callable.clone(), *mode), parameters.clone())
                        .is_some_and(|existing| existing != *parameters)
                {
                    return invalid("conflicting scheduling footprints for one callable transform");
                }
            }
            rules.push(contract);
        }
        Ok(Self {
            program,
            rules: rules.into_boxed_slice(),
        })
    }

    /// Return the independently verified selected program header.
    #[must_use]
    pub fn header(&self) -> &ProgramHeader {
        match &self.program {
            VerifiedResourceProgram::Closed(program) => program.header(),
            VerifiedResourceProgram::Family(family) => family.header(),
        }
    }

    /// Return the number of checked external resource rules.
    #[must_use]
    pub const fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Return the total number of checked expression and predicate DAG nodes.
    #[must_use]
    pub fn expression_node_count(&self) -> usize {
        self.rules
            .iter()
            .map(|rule| rule.expression.nodes.len())
            .sum()
    }

    /// Return checked rule identities in canonical request order.
    #[must_use]
    pub fn rule_identities(&self) -> impl ExactSizeIterator<Item = &str> {
        self.rules.iter().map(|rule| rule.identity.as_str())
    }

    /// Derive symbolic logical counts using the request's checked external rules.
    ///
    /// # Errors
    ///
    /// Returns [`SymbolicResourceError`] when a reachable construct or rule
    /// expression cannot be represented by the native count algebra.
    pub fn symbolic_logical_counts(&self) -> Result<SymbolicLogicalCounts, SymbolicResourceError> {
        match &self.program {
            VerifiedResourceProgram::Closed(program) => {
                symbolic_resources::symbolic_logical_counts_with_rules(
                    &program.local.checked.package,
                    &program.local.checked.type_arena,
                    &program.local.checked.operations,
                    self.rules(),
                )
            }
            VerifiedResourceProgram::Family(family) => {
                symbolic_resources::symbolic_logical_counts_with_rules(
                    &family.local.checked.package,
                    &family.local.checked.type_arena,
                    &family.local.checked.operations,
                    self.rules(),
                )
            }
        }
    }

    fn rules(&self) -> &[ResourceRuleContract] {
        &self.rules
    }
}

/// Seal one canonical unsealed symbolic resource-analysis request.
///
/// The request is structurally preflighted and semantically verified together
/// with its selected `QCore` program family before Rust assigns its digest.
///
/// # Errors
///
/// Returns [`ResourceRuleError`] when the request is oversized, malformed,
/// already sealed, noncanonical, incompatible, or semantically invalid.
pub fn seal_resource_analysis_request(payload: &[u8]) -> Result<Vec<u8>, ResourceRuleError> {
    let mut request = decode_request(payload)?;
    validate_schema(&request)?;
    if !request.request_digest.is_empty() || request.encode_to_vec() != payload {
        return Err(ResourceRuleError::Invalid(
            "Q4225: resource analysis request must be canonical and unsealed".to_owned(),
        ));
    }
    VerifiedResourceAnalysisRequest::verify(&request)?;
    request.request_digest = Sha256::digest(request.encode_to_vec()).to_vec();
    let sealed = request.encode_to_vec();
    if sealed.len() > MAX_REQUEST_BYTES {
        return Err(ResourceRuleError::Invalid(
            "Q4220: resource analysis request exceeds the wire-size limit".to_owned(),
        ));
    }
    Ok(sealed)
}

fn decode_request(payload: &[u8]) -> Result<wire::ResourceAnalysisRequestProto, ResourceRuleError> {
    if payload.len() > MAX_REQUEST_BYTES {
        return Err(ResourceRuleError::Invalid(
            "Q4220: resource analysis request exceeds the wire-size limit".to_owned(),
        ));
    }
    budget::preflight_resource_analysis_request(payload).map_err(|error| match error {
        budget::PreflightError::Budget(detail) => ResourceRuleError::Invalid(format!(
            "Q4220: resource analysis request violates its structural budget: {detail}"
        )),
        budget::PreflightError::Malformed(detail) => {
            ResourceRuleError::Invalid(format!("Q4221: {detail}"))
        }
    })?;
    wire::ResourceAnalysisRequestProto::decode(payload)
        .map_err(|error| ResourceRuleError::Invalid(format!("Q4221: {error}")))
}

fn validate_schema(request: &wire::ResourceAnalysisRequestProto) -> Result<(), ResourceRuleError> {
    if request.schema_major != SCHEMA_MAJOR {
        return Err(ResourceRuleError::Invalid(format!(
            "Q4222: unsupported resource analysis schema major {}; expected {SCHEMA_MAJOR}",
            request.schema_major
        )));
    }
    if request.schema_minor > SCHEMA_MINOR {
        return Err(ResourceRuleError::Invalid(format!(
            "Q4223: unsupported resource analysis schema minor {}",
            request.schema_minor
        )));
    }
    Ok(())
}

/// Failure at the closed external resource-rule boundary.
#[derive(Debug, Error)]
pub enum ResourceRuleError {
    /// The embedded `QCore` family did not pass the compiler verifier.
    #[error(transparent)]
    Compiler(#[from] CompilerError),
    /// The resource request or one of its closed contracts is invalid.
    #[error("{0}")]
    Invalid(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuleRelation {
    Equality,
    LowerBound,
    UpperBound,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum TransformMode {
    Direct,
    Controlled,
    Adjoint,
    ControlledAdjoint,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Composition {
    AdditiveWork,
    DependencySpan,
    PeakLiveness,
    ErrorBound,
}

/// Closed mathematical domain of a resource expression or metric.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Domain {
    /// Nonnegative integers.
    Natural,
    /// Signed integers.
    Integer,
    /// Exact rational numbers.
    Rational,
    /// Finite real numbers.
    Real,
    /// Real values constrained to the probability interval.
    Probability,
    /// Boolean predicates.
    Boolean,
}

#[derive(Clone, Debug)]
pub(crate) struct MetricContract {
    pub(crate) identifier: String,
    pub(crate) domain: Domain,
    pub(crate) composition: Composition,
}

#[derive(Clone, Debug)]
pub(crate) struct ResourceRuleContract {
    pub(crate) identity: String,
    pub(crate) callable: String,
    pub(crate) metric: MetricContract,
    pub(crate) relation: RuleRelation,
    pub(crate) expression: ExpressionGraph,
    pub(crate) modes: Box<[TransformMode]>,
    pub(crate) scheduling_footprint: Option<Box<[String]>>,
}

impl ResourceRuleContract {
    fn decode(
        source: &wire::ResourceRuleProto,
        functions: &[wire::FunctionProto],
    ) -> Result<Self, ResourceRuleError> {
        if source.identity.is_empty() || source.callable.is_empty() || source.provenance.is_empty()
        {
            return invalid("resource rules require identity, callable, and provenance");
        }
        let function = functions
            .iter()
            .find(|function| function.symbol == source.callable)
            .ok_or_else(|| {
                ResourceRuleError::Invalid(format!(
                    "Q4227: resource rule {:?} names unknown callable {:?}",
                    source.identity, source.callable
                ))
            })?;
        if !function.external {
            return invalid("resource rules cannot override concrete callables");
        }
        let metric = MetricContract::decode(
            source
                .metric
                .as_ref()
                .ok_or_else(|| ResourceRuleError::Invalid("Q4227: missing metric".to_owned()))?,
        )?;
        if matches!(
            metric.identifier.as_str(),
            "qamomile.logical.input_qubits" | "qamomile.logical.static_circuit_qubits"
        ) || metric
            .identifier
            .starts_with("qamomile.algorithmic.call_count.")
            || metric
                .identifier
                .starts_with("qamomile.algorithmic.query_count.")
        {
            return invalid("resource rule cannot override a structural metric");
        }
        let relation = match enum_value::<wire::ResourceRelationProto>(source.relation)? {
            wire::ResourceRelationProto::ResourceEquality => RuleRelation::Equality,
            wire::ResourceRelationProto::ResourceLowerBound => RuleRelation::LowerBound,
            wire::ResourceRelationProto::ResourceUpperBound => RuleRelation::UpperBound,
            wire::ResourceRelationProto::ResourceEnclosure
            | wire::ResourceRelationProto::ResourcePartialEnclosure
            | wire::ResourceRelationProto::ResourceUnknown
            | wire::ResourceRelationProto::ResourceRelationUnspecified => {
                return invalid("resource rule relation is unspecified");
            }
        };
        if metric.composition == Composition::PeakLiveness && relation != RuleRelation::Equality {
            return invalid("peak-liveness rules require an exact equality");
        }
        let modes = source
            .modes
            .iter()
            .map(
                |value| match enum_value::<wire::ResourceTransformModeProto>(*value)? {
                    wire::ResourceTransformModeProto::ResourceDirect => Ok(TransformMode::Direct),
                    wire::ResourceTransformModeProto::ResourceControlled => {
                        Ok(TransformMode::Controlled)
                    }
                    wire::ResourceTransformModeProto::ResourceAdjoint => Ok(TransformMode::Adjoint),
                    wire::ResourceTransformModeProto::ResourceControlledAdjoint => {
                        Ok(TransformMode::ControlledAdjoint)
                    }
                    wire::ResourceTransformModeProto::ResourceTransformModeUnspecified => {
                        invalid("resource transform mode is unspecified")
                    }
                },
            )
            .collect::<Result<Vec<_>, _>>()?;
        if modes.is_empty() || modes.windows(2).any(|pair| pair[0] >= pair[1]) {
            return invalid("resource transform modes must be nonempty and strictly ordered");
        }
        if enum_value::<wire::ResourceEvidenceAuthorityProto>(source.authority)?
            == wire::ResourceEvidenceAuthorityProto::ResourceEvidenceAuthorityUnspecified
        {
            return invalid("resource evidence authority is unspecified");
        }
        let expression = ExpressionGraph::decode(source.expression.as_ref().ok_or_else(|| {
            ResourceRuleError::Invalid("Q4227: missing resource expression".to_owned())
        })?)?;
        let root = expression.expression(expression.root)?;
        if root.unit != "1" {
            return invalid("resource rule formulas must be dimensionless");
        }
        if !metric.domain.contains(root.domain) {
            return invalid("resource rule formula domain is incompatible with its metric");
        }
        expression.validate_symbols(function)?;
        let scheduling_footprint =
            decode_scheduling_footprint(source.scheduling_footprint.as_ref(), function)?;
        Ok(Self {
            identity: source.identity.clone(),
            callable: source.callable.clone(),
            metric,
            relation,
            expression,
            modes: modes.into_boxed_slice(),
            scheduling_footprint,
        })
    }
}

fn decode_scheduling_footprint(
    source: Option<&wire::ResourceSchedulingFootprintProto>,
    function: &wire::FunctionProto,
) -> Result<Option<Box<[String]>>, ResourceRuleError> {
    let Some(footprint) = source else {
        return Ok(None);
    };
    if footprint.parameters.is_empty() {
        return invalid("scheduling footprints require at least one parameter");
    }
    let positions = footprint
        .parameters
        .iter()
        .map(|name| {
            function
                .parameters
                .iter()
                .position(|parameter| {
                    parameter.name == *name
                        && parameter
                            .value
                            .as_ref()
                            .is_some_and(|value| value.has_region)
                })
                .ok_or_else(|| {
                    ResourceRuleError::Invalid(format!(
                        "Q4227: scheduling footprint names non-quantum parameter {name:?}"
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if positions.windows(2).any(|pair| pair[0] >= pair[1]) {
        return invalid("scheduling footprint parameters must follow declaration order");
    }
    Ok(Some(footprint.parameters.clone().into_boxed_slice()))
}

impl MetricContract {
    fn decode(source: &wire::ResourceMetricProto) -> Result<Self, ResourceRuleError> {
        if source.identifier.is_empty()
            || !source.identifier.contains('.')
            || source.version.is_empty()
            || !resource_unit_name_is_valid(&source.unit)
        {
            return invalid("resource metric identity, version, or unit is invalid");
        }
        let coordinate = enum_value::<wire::ResourceCoordinateProto>(source.coordinate)?;
        if coordinate == wire::ResourceCoordinateProto::ResourceCoordinateUnspecified {
            return invalid("resource metric coordinate is unspecified");
        }
        let domain = Domain::decode(source.domain)?;
        let composition = match enum_value::<wire::ResourceCompositionProto>(source.composition)? {
            wire::ResourceCompositionProto::ResourceAdditiveWork => Composition::AdditiveWork,
            wire::ResourceCompositionProto::ResourceDependencySpan => Composition::DependencySpan,
            wire::ResourceCompositionProto::ResourcePeakLiveness => Composition::PeakLiveness,
            wire::ResourceCompositionProto::ResourceErrorBound => Composition::ErrorBound,
            wire::ResourceCompositionProto::ResourceReplacement
            | wire::ResourceCompositionProto::ResourceCompositionUnspecified => {
                return invalid("unsupported resource rule composition");
            }
        };
        Ok(Self {
            identifier: source.identifier.clone(),
            domain,
            composition,
        })
    }
}

impl Domain {
    /// Return the stable Python-facing domain spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Natural => "natural",
            Self::Integer => "integer",
            Self::Rational => "rational",
            Self::Real => "real",
            Self::Probability => "probability",
            Self::Boolean => "boolean",
        }
    }

    pub(crate) const fn promotion_rank(self) -> u8 {
        match self {
            Self::Natural => 0,
            Self::Integer => 1,
            Self::Rational => 2,
            Self::Probability => 3,
            Self::Real => 4,
            Self::Boolean => 5,
        }
    }

    fn decode(value: i32) -> Result<Self, ResourceRuleError> {
        match enum_value::<wire::ResourceDomainProto>(value)? {
            wire::ResourceDomainProto::ResourceNatural => Ok(Self::Natural),
            wire::ResourceDomainProto::ResourceInteger => Ok(Self::Integer),
            wire::ResourceDomainProto::ResourceRational => Ok(Self::Rational),
            wire::ResourceDomainProto::ResourceReal => Ok(Self::Real),
            wire::ResourceDomainProto::ResourceProbability => Ok(Self::Probability),
            wire::ResourceDomainProto::ResourceBoolean => Ok(Self::Boolean),
            wire::ResourceDomainProto::ResourceDomainUnspecified => {
                invalid("resource domain is unspecified")
            }
        }
    }

    pub(crate) const fn contains(self, expression: Self) -> bool {
        match self {
            Self::Natural => matches!(expression, Self::Natural),
            Self::Integer => matches!(expression, Self::Natural | Self::Integer),
            Self::Rational => {
                matches!(expression, Self::Natural | Self::Integer | Self::Rational)
            }
            Self::Real => !matches!(expression, Self::Boolean),
            Self::Probability => matches!(expression, Self::Probability),
            Self::Boolean => matches!(expression, Self::Boolean),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ExpressionGraph {
    nodes: Box<[ExpressionGraphNode]>,
    root: u32,
}

#[derive(Clone, Debug)]
pub(crate) enum ExpressionGraphNode {
    Expression(Box<ExpressionNode>),
    Predicate(PredicateNode),
}

#[derive(Clone, Debug)]
pub(crate) struct ExpressionNode {
    pub(crate) kind: String,
    pub(crate) domain: Domain,
    pub(crate) unit: String,
    pub(crate) arguments: Box<[u32]>,
    pub(crate) literal: Option<Number>,
    pub(crate) symbol: Option<SymbolContract>,
    pub(crate) attributes: Box<[(String, String)]>,
    pub(crate) predicates: Box<[u32]>,
}

#[derive(Clone, Debug)]
pub(crate) enum Number {
    Integer(BigInt),
    Rational(BigInt, BigInt),
    RealBits(u64),
}

#[derive(Clone, Debug)]
pub(crate) struct SymbolContract {
    pub(crate) scope: String,
    pub(crate) name: String,
    pub(crate) domain: Domain,
    pub(crate) unit: String,
    pub(crate) origin: String,
    pub(crate) constraints: Box<[String]>,
}

#[derive(Clone, Debug)]
pub(crate) struct PredicateNode {
    pub(crate) kind: String,
    pub(crate) arguments: Box<[u32]>,
    pub(crate) predicates: Box<[u32]>,
}

impl ExpressionGraph {
    fn decode(source: &wire::ResourceExpressionGraphProto) -> Result<Self, ResourceRuleError> {
        let graph = Self::decode_any(source)?;
        if !matches!(
            graph.nodes[graph.root as usize],
            ExpressionGraphNode::Expression(_)
        ) {
            return invalid("resource expression root must denote an expression");
        }
        Ok(graph)
    }

    pub(crate) fn decode_any(
        source: &wire::ResourceExpressionGraphProto,
    ) -> Result<Self, ResourceRuleError> {
        if source.nodes.is_empty() || source.nodes.len() > MAX_EXPRESSION_NODES {
            return invalid("resource expression graph violates the node budget");
        }
        let _root = usize::try_from(source.root)
            .ok()
            .filter(|root| *root < source.nodes.len())
            .ok_or_else(|| {
                ResourceRuleError::Invalid("Q4228: invalid expression root".to_owned())
            })?;
        let mut encodings = HashSet::with_capacity(source.nodes.len());
        let mut nodes = Vec::with_capacity(source.nodes.len());
        for (index, source_node) in source.nodes.iter().enumerate() {
            if !encodings.insert(source_node.encode_to_vec()) {
                return invalid("resource expression graph is not hash-consed");
            }
            let node = match source_node.node.as_ref() {
                Some(wire::resource_node_proto::Node::Expression(expression)) => {
                    ExpressionGraphNode::Expression(Box::new(ExpressionNode::decode(
                        expression, index, &nodes,
                    )?))
                }
                Some(wire::resource_node_proto::Node::Predicate(predicate)) => {
                    ExpressionGraphNode::Predicate(PredicateNode::decode(predicate, index, &nodes)?)
                }
                None => return invalid("resource expression node has no selected constructor"),
            };
            nodes.push(node);
        }
        super::resource_expression_evaluation::validate_recurrence_graph(source)
            .map_err(|error| ResourceRuleError::Invalid(error.to_string()))?;
        Ok(Self {
            nodes: nodes.into_boxed_slice(),
            root: source.root,
        })
    }

    pub(crate) fn expression(&self, identifier: u32) -> Result<&ExpressionNode, ResourceRuleError> {
        match self.nodes.get(identifier as usize) {
            Some(ExpressionGraphNode::Expression(node)) => Ok(node),
            _ => invalid("resource expression reference does not denote an expression"),
        }
    }

    pub(crate) const fn root(&self) -> u32 {
        self.root
    }

    pub(crate) fn nodes(&self) -> &[ExpressionGraphNode] {
        &self.nodes
    }

    pub(crate) fn node(&self, identifier: u32) -> Option<&ExpressionGraphNode> {
        self.nodes.get(identifier as usize)
    }

    fn validate_symbols(&self, function: &wire::FunctionProto) -> Result<(), ResourceRuleError> {
        let mut allowed = function
            .parameters
            .iter()
            .map(|parameter| parameter.name.clone())
            .collect::<BTreeSet<_>>();
        if let Some(result) = function.result_type.as_ref() {
            collect_dimensions(result, &mut allowed);
        }
        for parameter in &function.parameters {
            if let Some(value) = parameter.value.as_ref()
                && let Some(r#type) = value.r#type.as_ref()
            {
                collect_dimensions(r#type, &mut allowed);
            }
        }
        self.visit_symbols(self.root, function, &allowed, &BTreeSet::new())
    }

    fn visit_symbols(
        &self,
        identifier: u32,
        function: &wire::FunctionProto,
        allowed: &BTreeSet<String>,
        bound: &BTreeSet<(String, String)>,
    ) -> Result<(), ResourceRuleError> {
        match self.nodes.get(identifier as usize) {
            Some(ExpressionGraphNode::Expression(node)) => {
                if let Some(symbol) = node.symbol.as_ref()
                    && !bound.contains(&(symbol.scope.clone(), symbol.name.clone()))
                    && (symbol.scope != function.symbol || !allowed.contains(&symbol.name))
                {
                    return invalid("resource rule contains an unbound symbol");
                }
                let mut body_bound = bound.clone();
                if matches!(node.kind.as_str(), "bounded_sum" | "bounded_product") {
                    let attributes = node
                        .attributes
                        .iter()
                        .cloned()
                        .collect::<std::collections::BTreeMap<_, _>>();
                    let scope = attributes.get("binder_scope").ok_or_else(|| {
                        ResourceRuleError::Invalid("Q4228: missing binder scope".to_owned())
                    })?;
                    let name = attributes.get("binder_name").ok_or_else(|| {
                        ResourceRuleError::Invalid("Q4228: missing binder name".to_owned())
                    })?;
                    body_bound.insert((scope.clone(), name.clone()));
                }
                for (index, argument) in node.arguments.iter().enumerate() {
                    let active = if index == 2
                        && matches!(node.kind.as_str(), "bounded_sum" | "bounded_product")
                    {
                        &body_bound
                    } else {
                        bound
                    };
                    self.visit_symbols(*argument, function, allowed, active)?;
                }
                for predicate in &node.predicates {
                    self.visit_symbols(*predicate, function, allowed, bound)?;
                }
            }
            Some(ExpressionGraphNode::Predicate(node)) => {
                for argument in &node.arguments {
                    self.visit_symbols(*argument, function, allowed, bound)?;
                }
                for predicate in &node.predicates {
                    self.visit_symbols(*predicate, function, allowed, bound)?;
                }
            }
            None => return invalid("resource expression traversal lost a node"),
        }
        Ok(())
    }
}

impl ExpressionNode {
    fn decode(
        source: &wire::ResourceExpressionNodeProto,
        index: usize,
        nodes: &[ExpressionGraphNode],
    ) -> Result<Self, ResourceRuleError> {
        const KINDS: &[&str] = &[
            "literal",
            "symbol",
            "add",
            "multiply",
            "power",
            "absolute",
            "minimum",
            "maximum",
            "floor",
            "ceiling",
            "floor_divide",
            "ceiling_divide",
            "exp",
            "log",
            "log2",
            "square_root",
            "piecewise",
            "bounded_sum",
            "bounded_product",
            "contracted_function",
            "recurrence",
            "recurrence_equation",
            "recurrence_system",
        ];
        if !KINDS.contains(&source.kind.as_str()) || !resource_unit_name_is_valid(&source.unit) {
            return invalid("unknown resource expression constructor or empty unit");
        }
        validate_references(&source.arguments, index, nodes, true)?;
        validate_references(&source.predicates, index, nodes, false)?;
        let domain = Domain::decode(source.domain)?;
        let literal = source.literal.as_ref().map(Number::decode).transpose()?;
        let symbol = source
            .symbol
            .as_ref()
            .map(SymbolContract::decode)
            .transpose()?;
        let attributes = source
            .attributes
            .iter()
            .map(|attribute| (attribute.name.clone(), attribute.value.clone()))
            .collect::<Vec<_>>();
        if attributes.windows(2).any(|pair| pair[0].0 >= pair[1].0)
            || attributes.iter().any(|(name, _)| name.is_empty())
        {
            return invalid("resource expression attributes are not canonical");
        }
        validate_expression_shape(source, literal.is_some(), symbol.is_some())?;
        validate_expression_attributes(&source.kind, &attributes)?;
        if let Some(number) = literal.as_ref() {
            validate_number_domain(number, domain)?;
        }
        if let Some(symbol) = symbol.as_ref()
            && (symbol.domain != domain || symbol.unit != source.unit)
        {
            return invalid("symbol expression disagrees with its declaration");
        }
        Ok(Self {
            kind: source.kind.clone(),
            domain,
            unit: source.unit.clone(),
            arguments: source.arguments.clone().into_boxed_slice(),
            literal,
            symbol,
            attributes: attributes.into_boxed_slice(),
            predicates: source.predicates.clone().into_boxed_slice(),
        })
    }
}

impl PredicateNode {
    fn decode(
        source: &wire::ResourcePredicateNodeProto,
        index: usize,
        nodes: &[ExpressionGraphNode],
    ) -> Result<Self, ResourceRuleError> {
        const COMPARISONS: &[&str] = &[
            "equal",
            "not_equal",
            "less",
            "less_equal",
            "greater",
            "greater_equal",
        ];
        validate_references(&source.arguments, index, nodes, true)?;
        validate_references(&source.predicates, index, nodes, false)?;
        let valid = match source.kind.as_str() {
            "true" | "false" => source.arguments.is_empty() && source.predicates.is_empty(),
            kind if COMPARISONS.contains(&kind) || kind == "divisible" => {
                source.arguments.len() == 2 && source.predicates.is_empty()
            }
            "and" | "or" => source.arguments.is_empty() && !source.predicates.is_empty(),
            "not" => source.arguments.is_empty() && source.predicates.len() == 1,
            _ => false,
        };
        if !valid {
            return invalid("invalid resource predicate constructor shape");
        }
        Ok(Self {
            kind: source.kind.clone(),
            arguments: source.arguments.clone().into_boxed_slice(),
            predicates: source.predicates.clone().into_boxed_slice(),
        })
    }
}

impl Number {
    pub(crate) fn decode(source: &wire::ResourceNumberProto) -> Result<Self, ResourceRuleError> {
        use wire::resource_number_proto::Value;
        match source.value.as_ref() {
            Some(Value::Integer(value)) => parse_integer(value).map(Self::Integer),
            Some(Value::Rational(value)) => {
                let numerator = parse_integer(&value.numerator)?;
                let denominator = parse_integer(&value.denominator)?;
                if denominator <= BigInt::zero()
                    || gcd(numerator.clone(), denominator.clone()) != BigInt::from(1)
                {
                    return invalid("resource rational literal is not reduced and positive");
                }
                Ok(Self::Rational(numerator, denominator))
            }
            Some(Value::RealBits(bits)) if f64::from_bits(*bits).is_finite() => {
                Ok(Self::RealBits(*bits))
            }
            Some(Value::RealBits(_)) => invalid("resource real literal is not finite"),
            None => invalid("resource literal has no selected numeric value"),
        }
    }
}

impl SymbolContract {
    fn decode(source: &wire::ResourceSymbolProto) -> Result<Self, ResourceRuleError> {
        if source.scope.is_empty()
            || source.name.is_empty()
            || source.scope.contains("::")
            || source.name.contains("::")
            || !resource_unit_name_is_valid(&source.unit)
        {
            return invalid("resource symbol identity or unit is invalid");
        }
        Ok(Self {
            scope: source.scope.clone(),
            name: source.name.clone(),
            domain: Domain::decode(source.domain)?,
            unit: source.unit.clone(),
            origin: source.origin.clone(),
            constraints: source.constraints.clone().into_boxed_slice(),
        })
    }
}

fn validate_expression_shape(
    source: &wire::ResourceExpressionNodeProto,
    has_literal: bool,
    has_symbol: bool,
) -> Result<(), ResourceRuleError> {
    let ordinary = !has_literal && !has_symbol;
    let valid = match source.kind.as_str() {
        "literal" => has_literal && !has_symbol && source.arguments.is_empty(),
        "symbol" => has_symbol && !has_literal && source.arguments.is_empty(),
        "add" | "multiply" | "minimum" | "maximum" => ordinary && !source.arguments.is_empty(),
        "power" | "floor_divide" | "ceiling_divide" => ordinary && source.arguments.len() == 2,
        "absolute" | "floor" | "ceiling" | "exp" | "log2" | "square_root" => {
            ordinary && source.arguments.len() == 1
        }
        "log" => ordinary && matches!(source.arguments.len(), 1 | 2),
        "piecewise" => {
            ordinary
                && !source.predicates.is_empty()
                && source.arguments.len() == source.predicates.len() + 1
        }
        "bounded_sum" | "bounded_product" => ordinary && source.arguments.len() == 3,
        "contracted_function" | "recurrence" => ordinary,
        "recurrence_equation" => {
            ordinary && !source.arguments.is_empty() && source.predicates.len() == 1
        }
        "recurrence_system" => ordinary && source.arguments.len() >= 2,
        _ => false,
    };
    if !valid
        || (!matches!(source.kind.as_str(), "piecewise" | "recurrence_equation")
            && !source.predicates.is_empty())
    {
        return invalid("invalid resource expression constructor shape");
    }
    Ok(())
}

fn validate_expression_attributes(
    kind: &str,
    attributes: &[(String, String)],
) -> Result<(), ResourceRuleError> {
    let names = attributes
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>();
    let valid = match kind {
        "bounded_sum" | "bounded_product" => {
            let shape = names
                == [
                    "binder_domain",
                    "binder_name",
                    "binder_origin",
                    "binder_scope",
                ];
            let values = attributes
                .iter()
                .cloned()
                .collect::<std::collections::BTreeMap<_, _>>();
            shape
                && matches!(
                    values.get("binder_domain").map(String::as_str),
                    Some("natural" | "integer")
                )
                && values
                    .get("binder_name")
                    .is_some_and(|value| !value.contains("::"))
                && values
                    .get("binder_scope")
                    .is_some_and(|value| !value.contains("::"))
        }
        "contracted_function" | "recurrence" | "recurrence_equation" => {
            names == ["name", "version"]
        }
        _ => attributes.is_empty(),
    };
    if !valid || attributes.iter().any(|(_, value)| value.is_empty()) {
        return invalid("resource expression attributes have an invalid shape");
    }
    Ok(())
}

fn validate_number_domain(number: &Number, domain: Domain) -> Result<(), ResourceRuleError> {
    let valid = match (number, domain) {
        (Number::Integer(value), Domain::Natural) => !value.is_negative(),
        (Number::Integer(_), Domain::Integer | Domain::Rational | Domain::Real)
        | (Number::Rational(_, _), Domain::Rational | Domain::Real)
        | (Number::RealBits(_), Domain::Real) => true,
        (Number::Integer(value), Domain::Probability) => {
            *value == BigInt::zero() || *value == BigInt::from(1)
        }
        (Number::Rational(numerator, denominator), Domain::Probability) => {
            !numerator.is_negative() && numerator <= denominator
        }
        (Number::RealBits(bits), Domain::Probability) => {
            (0.0..=1.0).contains(&f64::from_bits(*bits))
        }
        _ => false,
    };
    if !valid {
        return invalid("resource literal is outside its declared domain");
    }
    Ok(())
}

fn validate_references(
    identifiers: &[u32],
    index: usize,
    nodes: &[ExpressionGraphNode],
    expression: bool,
) -> Result<(), ResourceRuleError> {
    for identifier in identifiers {
        let target = usize::try_from(*identifier)
            .ok()
            .filter(|value| *value < index)
            .and_then(|value| nodes.get(value));
        if !matches!(
            (expression, target),
            (true, Some(ExpressionGraphNode::Expression(_)))
                | (false, Some(ExpressionGraphNode::Predicate(_)))
        ) {
            return invalid("resource expression graph is not topologically typed");
        }
    }
    Ok(())
}

fn parse_integer(value: &str) -> Result<BigInt, ResourceRuleError> {
    let integer = value
        .parse::<BigInt>()
        .map_err(|_| ResourceRuleError::Invalid("Q4228: malformed integer literal".to_owned()))?;
    if integer.to_string() != value {
        return invalid("resource integer literal is not canonical");
    }
    Ok(integer)
}

fn gcd(mut left: BigInt, mut right: BigInt) -> BigInt {
    left = left.abs();
    right = right.abs();
    while !right.is_zero() {
        let remainder = left % &right;
        left = right;
        right = remainder;
    }
    left
}

fn collect_dimensions(r#type: &wire::TypeProto, names: &mut BTreeSet<String>) {
    for dimension in &r#type.dimensions {
        if let Some(wire::dimension_proto::Value::Symbol(name)) = dimension.value.as_ref() {
            names.insert(name.clone());
        }
    }
    for argument in &r#type.arguments {
        collect_dimensions(argument, names);
    }
}

pub(crate) fn resource_unit_name_is_valid(unit: &str) -> bool {
    !unit.is_empty() && !unit.chars().any(char::is_whitespace)
}

fn enum_value<T>(value: i32) -> Result<T, ResourceRuleError>
where
    T: TryFrom<i32>,
{
    T::try_from(value)
        .map_err(|_| ResourceRuleError::Invalid("Q4228: unknown resource enum value".to_owned()))
}

fn invalid<T>(message: &str) -> Result<T, ResourceRuleError> {
    Err(ResourceRuleError::Invalid(format!("Q4228: {message}")))
}
