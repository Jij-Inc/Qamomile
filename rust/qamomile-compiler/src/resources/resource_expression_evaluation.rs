//! Native evaluation for the closed portable resource-expression language.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use num_bigint::BigInt;
use num_traits::{FromPrimitive, One, Signed, ToPrimitive, Zero};
use prost::Message;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::resource_rules::{
    Domain as ResourceDomain, ExpressionGraph, ExpressionGraphNode, Number, SymbolContract,
    resource_unit_name_is_valid,
};
use crate::wire;

const MAX_EXPRESSION_BYTES: usize = 64 * 1024 * 1024;
const MAX_EVALUATION_STEPS: usize = 1_000_000;
const MAX_EVALUATION_DEPTH: usize = 1_024;
const MAX_BOUNDED_TERMS: usize = 100_000;
const MAX_HOTSPOT_CANDIDATES: usize = 1_000_000;
const MAX_INTEGER_BITS: u64 = 1_048_576;
const MAX_INTEGER_DECIMAL_DIGITS: usize = 315_654;
const MAX_POWER_EXPONENT: u32 = 1_000_000;

/// A fully evaluated scalar from the portable resource-expression language.
#[derive(Clone, Debug, PartialEq)]
pub enum ResourceScalar {
    /// An arbitrary-precision integer.
    Integer(BigInt),
    /// An exact reduced rational with a positive denominator.
    Rational {
        /// Reduced signed numerator.
        numerator: BigInt,
        /// Reduced positive denominator.
        denominator: BigInt,
    },
    /// A finite IEEE-754 binary64 value.
    Real(f64),
}

/// The result of evaluating one standalone resource-expression graph.
#[derive(Clone, Debug, PartialEq)]
pub enum ResourceExpressionEvaluation {
    /// The root reduced to a numeric scalar.
    Numeric(ResourceScalar),
    /// The root reduced to a Boolean predicate.
    Boolean(bool),
    /// At least one free symbol or opaque contract prevented evaluation.
    Unknown,
}

/// Failure to validate or safely evaluate a resource-expression graph.
#[derive(Debug, Error)]
pub enum ResourceExpressionEvaluationError {
    /// The encoded graph is malformed or not canonical.
    #[error("Q4230: invalid resource-expression graph: {0}")]
    Invalid(String),
    /// An otherwise valid expression is undefined for its literal operands.
    #[error("Q4231: resource-expression domain error: {0}")]
    Domain(String),
    /// A division-like operation has a zero divisor.
    #[error("Q4232: resource-expression division by zero: {0}")]
    DivisionByZero(String),
    /// Evaluation exceeded a deterministic complexity budget.
    #[error("Q4233: resource-expression evaluation exceeds the {0} budget")]
    Budget(&'static str),
}

/// Failure to validate or substitute a standalone resource-expression graph.
#[derive(Debug, Error)]
pub enum ResourceExpressionSubstitutionError {
    /// The source graph, binding identity, or replacement graph is invalid.
    #[error("Q4235: invalid resource-expression substitution: {0}")]
    Invalid(String),
    /// Substitution exceeded a deterministic graph traversal budget.
    #[error("Q4236: resource-expression substitution exceeds the {0} budget")]
    Budget(&'static str),
}

/// Failure to resolve public resource-substitution names across expression roots.
#[derive(Debug, Error)]
pub enum ResourceSymbolResolutionError {
    /// One source graph or its free-symbol declarations is invalid.
    #[error(transparent)]
    Expression(#[from] ResourceExpressionSubstitutionError),
    /// One convenient short name denotes declarations in multiple scopes.
    #[error("ambiguous resource symbol '{name}'; use one of {qualified}")]
    Ambiguous {
        /// Supplied unqualified name.
        name: String,
        /// Deterministically ordered qualified alternatives.
        qualified: String,
    },
}

/// Failure to rank resource-study contributions by their finite upper endpoint.
#[derive(Debug, Error)]
pub enum ResourceHotspotRankingError {
    /// The candidate table is malformed or exceeds its aggregate budget.
    #[error("Q4243: invalid resource hotspot ranking: {0}")]
    Invalid(String),
    /// One supplied upper endpoint is not a valid resource expression.
    #[error("Q4244: invalid resource contribution projection: {0}")]
    Expression(#[from] ResourceExpressionEvaluationError),
}

/// One source-local contribution projected for native hotspot ranking.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceHotspotCandidate {
    upper: Option<Vec<u8>>,
    owner: String,
    source: String,
}

impl ResourceHotspotCandidate {
    /// Construct one transient resource-hotspot candidate.
    #[must_use]
    pub fn new(upper: Option<Vec<u8>>, owner: String, source: String) -> Self {
        Self {
            upper,
            owner,
            source,
        }
    }
}

/// One lexically free symbol declaration in a resource-expression graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceExpressionSymbol {
    scope: String,
    name: String,
    domain: ResourceDomain,
    unit: String,
    origin: String,
    constraints: Box<[String]>,
}

impl ResourceExpressionSymbol {
    /// Return the stable scope owning this symbol.
    #[must_use]
    pub fn scope(&self) -> &str {
        &self.scope
    }

    /// Return the source-facing unqualified symbol name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the declared mathematical domain.
    #[must_use]
    pub const fn domain(&self) -> ResourceDomain {
        self.domain
    }

    /// Return the nominal resource unit.
    #[must_use]
    pub fn unit(&self) -> &str {
        &self.unit
    }

    /// Return the declaration origin.
    #[must_use]
    pub fn origin(&self) -> &str {
        &self.origin
    }

    /// Return the canonical declaration constraints.
    #[must_use]
    pub fn constraints(&self) -> &[String] {
        &self.constraints
    }

    fn from_contract(symbol: &SymbolContract) -> Self {
        Self {
            scope: symbol.scope.clone(),
            name: symbol.name.clone(),
            domain: symbol.domain,
            unit: symbol.unit.clone(),
            origin: symbol.origin.clone(),
            constraints: symbol.constraints.clone(),
        }
    }

    fn identity(&self) -> String {
        format!("{}::{}", self.scope, self.name)
    }
}

/// Failure to validate a public resource value against its metric contract.
#[derive(Debug, Error)]
pub enum ResourceValueValidationError {
    /// The relation shape or mathematical value violates the metric contract.
    #[error("Q4241: invalid resource value: {0}")]
    Invalid(String),
    /// The supplied expression graph is malformed or exceeds its budget.
    #[error(transparent)]
    Expression(#[from] ResourceExpressionEvaluationError),
}

/// Failure to construct the explicit toy surface-code resource projection.
#[derive(Debug, Error)]
pub enum SurfaceCodeProjectionError {
    /// A physical model coefficient is non-finite or outside its domain.
    #[error("Q4240: invalid surface-code resource projection: {0}")]
    Invalid(String),
    /// An input or generated resource-expression graph is invalid or excessive.
    #[error(transparent)]
    Expression(#[from] ResourceExpressionSubstitutionError),
}

/// Canonical resource-expression roots produced by the surface-code model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SurfaceCodeResourceProjection {
    code_distance: Vec<u8>,
    physical_qubits: Vec<u8>,
    duration: Vec<u8>,
    qubit_time: Vec<u8>,
}

impl SurfaceCodeResourceProjection {
    /// Return the canonical code-distance expression graph.
    #[must_use]
    pub fn code_distance(&self) -> &[u8] {
        &self.code_distance
    }

    /// Return the canonical physical-qubit expression graph.
    #[must_use]
    pub fn physical_qubits(&self) -> &[u8] {
        &self.physical_qubits
    }

    /// Return the canonical duration expression graph.
    #[must_use]
    pub fn duration(&self) -> &[u8] {
        &self.duration
    }

    /// Return the canonical physical-qubit-time expression graph.
    #[must_use]
    pub fn qubit_time(&self) -> &[u8] {
        &self.qubit_time
    }
}

/// Validate and evaluate one canonical portable resource-expression graph.
///
/// Both expression and predicate roots are accepted. Free symbols, recurrence
/// contracts, and unknown contracted functions produce
/// [`ResourceExpressionEvaluation::Unknown`].
///
/// # Errors
///
/// Returns [`ResourceExpressionEvaluationError`] when the wire graph is
/// malformed, a numeric operation is undefined, or evaluation exceeds a
/// deterministic work budget.
pub fn evaluate_resource_expression(
    payload: &[u8],
) -> Result<ResourceExpressionEvaluation, ResourceExpressionEvaluationError> {
    let graph = decode_canonical_graph(payload)?;
    evaluate_graph(&graph)
}

fn decode_canonical_graph(
    payload: &[u8],
) -> Result<ExpressionGraph, ResourceExpressionEvaluationError> {
    if payload.len() > MAX_EXPRESSION_BYTES {
        return Err(ResourceExpressionEvaluationError::Budget("wire-size"));
    }
    let source = wire::ResourceExpressionGraphProto::decode(payload)
        .map_err(|error| ResourceExpressionEvaluationError::Invalid(error.to_string()))?;
    if source.encode_to_vec() != payload {
        return Err(ResourceExpressionEvaluationError::Invalid(
            "encoding is not canonical".to_owned(),
        ));
    }
    let graph = ExpressionGraph::decode_any(&source)
        .map_err(|error| ResourceExpressionEvaluationError::Invalid(error.to_string()))?;
    Ok(graph)
}

fn evaluate_graph(
    graph: &ExpressionGraph,
) -> Result<ResourceExpressionEvaluation, ResourceExpressionEvaluationError> {
    let mut steps = 0;
    evaluate_graph_with_steps(graph, &mut steps)
}

fn evaluate_graph_with_steps(
    graph: &ExpressionGraph,
    steps: &mut usize,
) -> Result<ResourceExpressionEvaluation, ResourceExpressionEvaluationError> {
    let root = graph.root();
    let mut evaluator = Evaluator {
        graph,
        steps: *steps,
        recurrence: None,
    };
    let mut memo = HashMap::new();
    let result = match evaluator.evaluate(root, &BTreeMap::new(), 0, &mut memo)? {
        Value::Numeric(number) => Ok(ResourceExpressionEvaluation::Numeric(number.into_public())),
        Value::Boolean(value) => Ok(ResourceExpressionEvaluation::Boolean(value)),
        Value::Unknown => Ok(ResourceExpressionEvaluation::Unknown),
    };
    *steps = evaluator.steps;
    result
}

/// Validate one public resource-value relation and all of its endpoints.
///
/// # Errors
///
/// Returns [`ResourceValueValidationError`] when the relation shape is invalid,
/// an expression graph is malformed, or an endpoint violates the metric.
pub fn validate_resource_value(
    metric_domain: &str,
    unit: &str,
    relation: &str,
    expression: Option<&[u8]>,
    lower: Option<&[u8]>,
    upper: Option<&[u8]>,
    unknown_contributors: usize,
) -> Result<(), ResourceValueValidationError> {
    validate_relation_shape(
        relation,
        expression.is_some(),
        lower.is_some(),
        upper.is_some(),
        unknown_contributors,
    )?;
    let domain = metric_domain_from_name(metric_domain)?;
    let mut lower_value = None;
    let mut upper_value = None;
    for (context, payload, evaluated) in [
        ("resource expression", expression, None),
        ("resource lower endpoint", lower, Some(&mut lower_value)),
        ("resource upper endpoint", upper, Some(&mut upper_value)),
    ] {
        let Some(payload) = payload else {
            continue;
        };
        let graph = decode_canonical_graph(payload)?;
        let value = validate_metric_graph(domain, unit, context, &graph)?;
        if let Some(target) = evaluated {
            *target = value;
        }
    }
    if let (Some(low), Some(high)) = (&lower_value, &upper_value)
        && compare_scalars(low, high) == Ordering::Greater
    {
        return resource_value_invalid("resource enclosure lower endpoint exceeds upper");
    }
    Ok(())
}

fn validate_relation_shape(
    relation: &str,
    has_expression: bool,
    has_lower: bool,
    has_upper: bool,
    unknown_contributors: usize,
) -> Result<(), ResourceValueValidationError> {
    match relation {
        "equality" | "lower_bound" | "upper_bound" => {
            if !has_expression {
                return resource_value_invalid(format!("{relation} requires an expression"));
            }
            if has_lower || has_upper {
                return resource_value_invalid(format!(
                    "{relation} cannot carry enclosure endpoints"
                ));
            }
            if unknown_contributors != 0 {
                return resource_value_invalid(format!(
                    "{relation} cannot carry unknown contributors"
                ));
            }
        }
        "enclosure" => {
            if has_expression || !has_lower || !has_upper {
                return resource_value_invalid(
                    "an enclosure requires lower and upper endpoints only",
                );
            }
            if unknown_contributors != 0 {
                return resource_value_invalid("an enclosure cannot carry unknown contributors");
            }
        }
        "partial_enclosure" => {
            if has_expression {
                return resource_value_invalid("a partial enclosure cannot carry an expression");
            }
            if !has_lower && !has_upper {
                return resource_value_invalid("a partial enclosure requires a finite endpoint");
            }
            if unknown_contributors == 0 {
                return resource_value_invalid(
                    "a partial enclosure must name unknown contributors",
                );
            }
        }
        "unknown" => {
            if has_expression || has_lower || has_upper {
                return resource_value_invalid("an unknown resource value cannot carry endpoints");
            }
            if unknown_contributors == 0 {
                return resource_value_invalid("an unknown resource value must explain why");
            }
        }
        _ => return resource_value_invalid("resource value relation is unknown"),
    }
    Ok(())
}

fn metric_domain_from_name(
    name: &str,
) -> Result<crate::resource_rules::Domain, ResourceValueValidationError> {
    use crate::resource_rules::Domain;
    match name {
        "natural" => Ok(Domain::Natural),
        "integer" => Ok(Domain::Integer),
        "rational" => Ok(Domain::Rational),
        "real" => Ok(Domain::Real),
        "probability" => Ok(Domain::Probability),
        "boolean" => Ok(Domain::Boolean),
        _ => resource_value_invalid("resource metric domain is unknown"),
    }
}

fn validate_metric_graph(
    metric_domain: crate::resource_rules::Domain,
    expected_unit: &str,
    context: &str,
    graph: &ExpressionGraph,
) -> Result<Option<ResourceScalar>, ResourceValueValidationError> {
    use crate::resource_rules::ExpressionGraphNode;
    let Some(ExpressionGraphNode::Expression(root)) = graph.node(graph.root()) else {
        return resource_value_invalid(format!("{context} must be a resource expression"));
    };
    if root.unit != expected_unit {
        return resource_value_invalid(format!("{context} has the wrong nominal unit"));
    }
    match evaluate_graph(graph)? {
        ResourceExpressionEvaluation::Numeric(value) => {
            validate_concrete_metric_value(metric_domain, &value, context)?;
            Ok(Some(value))
        }
        ResourceExpressionEvaluation::Boolean(_) => {
            if metric_domain != crate::resource_rules::Domain::Boolean {
                return resource_value_invalid(format!(
                    "{context} must evaluate to a real resource quantity"
                ));
            }
            Ok(None)
        }
        ResourceExpressionEvaluation::Unknown => {
            if !metric_domain.contains(root.domain) {
                return resource_value_invalid(
                    format!(
                        "{context} domain {:?} is not contained in metric domain {:?}",
                        domain_name(root.domain),
                        domain_name(metric_domain),
                    )
                    .replace('"', "'"),
                );
            }
            Ok(None)
        }
    }
}

fn validate_concrete_metric_value(
    domain: crate::resource_rules::Domain,
    value: &ResourceScalar,
    context: &str,
) -> Result<(), ResourceValueValidationError> {
    use crate::resource_rules::Domain;
    if domain == Domain::Boolean {
        return resource_value_invalid(format!("{context} is outside metric domain 'boolean'"));
    }
    if scalar_is_negative(value) {
        return resource_value_invalid(format!("{context} must be nonnegative"));
    }
    let belongs = match domain {
        Domain::Natural | Domain::Integer => scalar_is_integral(value),
        Domain::Rational => !matches!(value, ResourceScalar::Real(_)),
        Domain::Real => true,
        Domain::Probability => {
            compare_scalars(value, &ResourceScalar::Integer(BigInt::zero())) != Ordering::Less
                && compare_scalars(value, &ResourceScalar::Integer(BigInt::one()))
                    != Ordering::Greater
        }
        Domain::Boolean => false,
    };
    if !belongs {
        return resource_value_invalid(format!(
            "{context} is outside metric domain '{}'",
            domain_name(domain)
        ));
    }
    Ok(())
}

const fn domain_name(domain: crate::resource_rules::Domain) -> &'static str {
    domain.as_str()
}

fn scalar_is_negative(value: &ResourceScalar) -> bool {
    match value {
        ResourceScalar::Integer(value) => value.is_negative(),
        ResourceScalar::Rational { numerator, .. } => numerator.is_negative(),
        ResourceScalar::Real(value) => *value < 0.0,
    }
}

fn scalar_is_integral(value: &ResourceScalar) -> bool {
    match value {
        ResourceScalar::Integer(_) => true,
        ResourceScalar::Rational { denominator, .. } => denominator.is_one(),
        ResourceScalar::Real(_) => false,
    }
}

fn compare_scalars(left: &ResourceScalar, right: &ResourceScalar) -> Ordering {
    let (left_numerator, left_denominator) = scalar_ratio(left);
    let (right_numerator, right_denominator) = scalar_ratio(right);
    (left_numerator * right_denominator).cmp(&(right_numerator * left_denominator))
}

fn scalar_ratio(value: &ResourceScalar) -> (BigInt, BigInt) {
    match value {
        ResourceScalar::Integer(value) => (value.clone(), BigInt::one()),
        ResourceScalar::Rational {
            numerator,
            denominator,
        } => (numerator.clone(), denominator.clone()),
        ResourceScalar::Real(value) => finite_f64_ratio(*value),
    }
}

fn finite_f64_ratio(value: f64) -> (BigInt, BigInt) {
    let bits = value.to_bits();
    let negative = bits >> 63 != 0;
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (significand, binary_exponent) = if exponent == 0 {
        (fraction, -1074)
    } else {
        ((1_u64 << 52) | fraction, exponent - 1023 - 52)
    };
    let mut numerator = BigInt::from(significand);
    if negative {
        numerator = -numerator;
    }
    let shift = usize::try_from(binary_exponent.unsigned_abs())
        .expect("an IEEE-754 binary exponent fits usize");
    if binary_exponent >= 0 {
        (numerator << shift, BigInt::one())
    } else {
        (numerator, BigInt::one() << shift)
    }
}

fn resource_value_invalid<T>(
    message: impl Into<String>,
) -> Result<T, ResourceValueValidationError> {
    Err(ResourceValueValidationError::Invalid(message.into()))
}

/// Return the lexically free symbol declarations in one expression graph.
///
/// Bounded-expression binders shadow a matching qualified identity only in
/// their body. The result is sorted by qualified identity and contains one
/// declaration per identity.
///
/// # Errors
///
/// Returns [`ResourceExpressionSubstitutionError`] when the graph is malformed,
/// conflicting declarations use the same identity, or traversal exceeds a
/// deterministic complexity budget.
#[expect(
    clippy::too_many_lines,
    reason = "Keep lexical scope traversal and its bounded context arena together"
)]
pub fn free_resource_expression_symbols(
    payload: &[u8],
) -> Result<Vec<ResourceExpressionSymbol>, ResourceExpressionSubstitutionError> {
    let source = PreparedGraph::decode(payload)?;
    let mut free = BTreeMap::<String, ResourceExpressionSymbol>::new();
    let mut contexts = vec![ResourceLexicalContext {
        parent: None,
        identity: String::new(),
    }];
    let mut visited = HashSet::<(u32, usize)>::new();
    let mut stack = vec![(source.graph.root(), 0_usize, 0_usize)];
    let mut steps = 0_usize;

    while let Some((identifier, context_id, depth)) = stack.pop() {
        steps = steps
            .checked_add(1)
            .ok_or(ResourceExpressionSubstitutionError::Budget(
                "operation-count",
            ))?;
        if steps > MAX_EVALUATION_STEPS {
            return Err(ResourceExpressionSubstitutionError::Budget(
                "operation-count",
            ));
        }
        if depth > MAX_EVALUATION_DEPTH {
            return Err(ResourceExpressionSubstitutionError::Budget("nesting-depth"));
        }
        if !visited.insert((identifier, context_id)) {
            continue;
        }
        let next_depth = depth
            .checked_add(1)
            .ok_or(ResourceExpressionSubstitutionError::Budget("nesting-depth"))?;
        match source.graph.node(identifier) {
            Some(ExpressionGraphNode::Expression(node)) => {
                if let Some(contract) = node.symbol.as_ref() {
                    let symbol = ResourceExpressionSymbol::from_contract(contract);
                    let identity = symbol.identity();
                    if !resource_symbol_is_bound(&contexts, context_id, &identity, &mut steps)? {
                        if let Some(existing) = free.get(&identity)
                            && existing != &symbol
                        {
                            return Err(ResourceExpressionSubstitutionError::Invalid(format!(
                                "conflicting declarations for symbol {identity:?}"
                            )));
                        }
                        free.insert(identity, symbol);
                    }
                }

                let mut equation_context_id = context_id;
                if node.kind == "recurrence_equation" {
                    for parameter in node.arguments.iter().take(node.arguments.len() - 1) {
                        let declaration = source.graph.expression(*parameter).map_err(|error| {
                            ResourceExpressionSubstitutionError::Invalid(error.to_string())
                        })?;
                        let symbol = declaration.symbol.as_ref().ok_or_else(|| {
                            ResourceExpressionSubstitutionError::Invalid(
                                "recurrence parameter is not a symbol".to_owned(),
                            )
                        })?;
                        let new_id = contexts.len();
                        contexts.push(ResourceLexicalContext {
                            parent: Some(equation_context_id),
                            identity: format!("{}::{}", symbol.scope, symbol.name),
                        });
                        equation_context_id = new_id;
                    }
                }
                let body_context_id =
                    if matches!(node.kind.as_str(), "bounded_sum" | "bounded_product") {
                        let identity = bounded_expression_identity(node)?;
                        let new_id = contexts.len();
                        contexts.push(ResourceLexicalContext {
                            parent: Some(context_id),
                            identity,
                        });
                        new_id
                    } else {
                        equation_context_id
                    };

                for predicate in node.predicates.iter().rev() {
                    stack.push((*predicate, equation_context_id, next_depth));
                }
                for (index, argument) in node.arguments.iter().enumerate().rev() {
                    let argument_context = if node.kind == "recurrence_equation" {
                        equation_context_id
                    } else if index == 2
                        && matches!(node.kind.as_str(), "bounded_sum" | "bounded_product")
                    {
                        body_context_id
                    } else {
                        context_id
                    };
                    stack.push((*argument, argument_context, next_depth));
                }
            }
            Some(ExpressionGraphNode::Predicate(node)) => {
                for predicate in node.predicates.iter().rev() {
                    stack.push((*predicate, context_id, next_depth));
                }
                for argument in node.arguments.iter().rev() {
                    stack.push((*argument, context_id, next_depth));
                }
            }
            None => {
                return Err(ResourceExpressionSubstitutionError::Invalid(
                    "resource expression reference is out of range".to_owned(),
                ));
            }
        }
    }
    Ok(free.into_values().collect())
}

/// Resolve public resource-substitution names across one evaluation boundary.
///
/// Qualified names remain authoritative even when the same mapping also
/// supplies their unqualified spelling. Unknown names are ignored, matching
/// substitution of a mapping into an expression that does not mention them.
///
/// # Errors
///
/// Returns [`ResourceSymbolResolutionError`] when the aggregate input exceeds
/// a deterministic budget, a graph contains invalid symbol declarations, or
/// an unqualified name is ambiguous across scopes.
pub fn resolve_resource_symbol_bindings(
    payloads: &[Vec<u8>],
    supplied_names: &[String],
) -> Result<Vec<(String, String)>, ResourceSymbolResolutionError> {
    let item_count = payloads.len().checked_add(supplied_names.len()).ok_or(
        ResourceExpressionSubstitutionError::Budget("operation-count"),
    )?;
    if item_count > MAX_EVALUATION_STEPS {
        return Err(ResourceExpressionSubstitutionError::Budget("operation-count").into());
    }
    let byte_count = payloads
        .iter()
        .map(Vec::len)
        .chain(supplied_names.iter().map(String::len))
        .try_fold(0_usize, usize::checked_add)
        .ok_or(ResourceExpressionSubstitutionError::Budget("wire-size"))?;
    if byte_count > MAX_EXPRESSION_BYTES {
        return Err(ResourceExpressionSubstitutionError::Budget("wire-size").into());
    }

    let mut declarations = BTreeMap::<String, String>::new();
    for payload in payloads {
        for symbol in free_resource_expression_symbols(payload)? {
            let identity = symbol.identity();
            declarations
                .entry(identity)
                .or_insert_with(|| symbol.name().to_owned());
        }
    }

    let mut identities_by_name = BTreeMap::<&str, Vec<&str>>::new();
    for (identity, name) in &declarations {
        identities_by_name.entry(name).or_default().push(identity);
    }
    let mut resolved = BTreeMap::<String, String>::new();
    for name in supplied_names.iter().filter(|name| name.contains("::")) {
        resolved.insert(name.clone(), name.clone());
    }
    for name in supplied_names.iter().filter(|name| !name.contains("::")) {
        let identities = identities_by_name
            .get(name.as_str())
            .map_or(&[][..], Vec::as_slice);
        if identities.len() > 1 {
            return Err(ResourceSymbolResolutionError::Ambiguous {
                name: name.clone(),
                qualified: identities
                    .iter()
                    .map(|identity| format!("'{identity}'"))
                    .collect::<Vec<_>>()
                    .join(", "),
            });
        }
        if let Some(identity) = identities.first() {
            resolved
                .entry((*identity).to_owned())
                .or_insert_with(|| name.clone());
        }
    }
    Ok(resolved.into_iter().collect())
}

/// Rank resource-study contributions by their evaluated upper endpoints.
///
/// Numeric candidates precede unresolved candidates and are ordered by exact
/// mathematical value in descending order. Owner, source, and original row
/// position provide a deterministic total tie break.
///
/// # Errors
///
/// Returns [`ResourceHotspotRankingError`] when the aggregate candidate table
/// exceeds its byte, item, or evaluation-work budget; an endpoint graph is
/// malformed; or an endpoint root denotes a Boolean predicate.
pub fn rank_resource_hotspots(
    candidates: &[ResourceHotspotCandidate],
    limit: usize,
) -> Result<Vec<usize>, ResourceHotspotRankingError> {
    if candidates.len() > MAX_HOTSPOT_CANDIDATES {
        return Err(ResourceHotspotRankingError::Invalid(
            "candidate count exceeds the item budget".to_owned(),
        ));
    }
    let byte_count = candidates
        .iter()
        .flat_map(|candidate| {
            [
                candidate.upper.as_ref().map_or(0, Vec::len),
                candidate.owner.len(),
                candidate.source.len(),
            ]
        })
        .try_fold(0_usize, usize::checked_add)
        .ok_or_else(|| {
            ResourceHotspotRankingError::Invalid(
                "candidate table exceeds the byte budget".to_owned(),
            )
        })?;
    if byte_count > MAX_EXPRESSION_BYTES {
        return Err(ResourceHotspotRankingError::Invalid(
            "candidate table exceeds the byte budget".to_owned(),
        ));
    }

    let mut steps = 0_usize;
    let mut upper_values = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let value = match candidate.upper.as_deref() {
            None => None,
            Some(payload) => {
                let graph = decode_canonical_graph(payload)?;
                if !matches!(
                    graph.node(graph.root()),
                    Some(ExpressionGraphNode::Expression(_))
                ) {
                    return Err(ResourceHotspotRankingError::Invalid(
                        "upper endpoint must denote a resource expression".to_owned(),
                    ));
                }
                match evaluate_graph_with_steps(&graph, &mut steps)? {
                    ResourceExpressionEvaluation::Numeric(value) => Some(value),
                    ResourceExpressionEvaluation::Unknown => None,
                    ResourceExpressionEvaluation::Boolean(_) => {
                        return Err(ResourceHotspotRankingError::Invalid(
                            "upper endpoint evaluated to a Boolean".to_owned(),
                        ));
                    }
                }
            }
        };
        upper_values.push(value);
    }

    let mut order = (0..candidates.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        let numeric = match (&upper_values[*left], &upper_values[*right]) {
            (Some(left_value), Some(right_value)) => compare_scalars(right_value, left_value),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        };
        numeric
            .then_with(|| candidates[*left].owner.cmp(&candidates[*right].owner))
            .then_with(|| candidates[*left].source.cmp(&candidates[*right].source))
            .then_with(|| left.cmp(right))
    });
    order.truncate(limit.min(order.len()));
    Ok(order)
}

struct ResourceLexicalContext {
    parent: Option<usize>,
    identity: String,
}

fn resource_symbol_is_bound(
    contexts: &[ResourceLexicalContext],
    mut context_id: usize,
    identity: &str,
    steps: &mut usize,
) -> Result<bool, ResourceExpressionSubstitutionError> {
    while let Some(parent) = contexts[context_id].parent {
        *steps = steps
            .checked_add(1)
            .ok_or(ResourceExpressionSubstitutionError::Budget(
                "operation-count",
            ))?;
        if *steps > MAX_EVALUATION_STEPS {
            return Err(ResourceExpressionSubstitutionError::Budget(
                "operation-count",
            ));
        }
        if contexts[context_id].identity == identity {
            return Ok(true);
        }
        context_id = parent;
    }
    Ok(false)
}

fn bounded_expression_identity(
    node: &crate::resource_rules::ExpressionNode,
) -> Result<String, ResourceExpressionSubstitutionError> {
    let scope = node
        .attributes
        .iter()
        .find_map(|(key, value)| (key == "binder_scope").then_some(value));
    let name = node
        .attributes
        .iter()
        .find_map(|(key, value)| (key == "binder_name").then_some(value));
    match (scope, name) {
        (Some(scope), Some(name)) => Ok(format!("{scope}::{name}")),
        _ => Err(ResourceExpressionSubstitutionError::Invalid(
            "bounded expression is missing its binder identity".to_owned(),
        )),
    }
}

/// Substitute qualified symbols in one canonical resource-expression graph.
///
/// Replacement expressions are inserted as values and are not recursively
/// rewritten by other bindings. Bounded-expression binders shadow a matching
/// external identity only in their body, matching lexical substitution.
///
/// # Errors
///
/// Returns [`ResourceExpressionSubstitutionError`] when an input graph or
/// binding is malformed, a replacement changes a nominal unit, or traversal
/// exceeds a deterministic complexity budget.
pub fn substitute_resource_expression(
    payload: &[u8],
    bindings: Vec<(String, Vec<u8>, bool)>,
) -> Result<Vec<u8>, ResourceExpressionSubstitutionError> {
    let source = PreparedGraph::decode(payload)?;
    let mut replacements = BTreeMap::new();
    for (identity, payload, adapt_literal) in bindings {
        if !valid_qualified_identity(&identity) || replacements.contains_key(&identity) {
            return Err(ResourceExpressionSubstitutionError::Invalid(format!(
                "invalid or duplicate binding identity {identity:?}"
            )));
        }
        let replacement = PreparedGraph::decode(&payload)?;
        let root = replacement.root_expression()?;
        if adapt_literal && root.kind != "literal" {
            return Err(ResourceExpressionSubstitutionError::Invalid(
                "an adaptive host-number binding is not a literal".to_owned(),
            ));
        }
        replacements.insert(identity, (replacement, adapt_literal));
    }
    let mut builder = SubstitutionBuilder::default();
    let root = builder.import(
        &source,
        source.graph.root(),
        &replacements,
        &BTreeSet::new(),
        0,
    )?;
    let (nodes, root) = canonical_reachable_graph(&builder.nodes, root)?;
    let result = wire::ResourceExpressionGraphProto { nodes, root };
    let payload = result.encode_to_vec();
    if payload.len() > MAX_EXPRESSION_BYTES {
        return Err(ResourceExpressionSubstitutionError::Budget("wire-size"));
    }
    ExpressionGraph::decode_any(&result)
        .map_err(|error| ResourceExpressionSubstitutionError::Invalid(error.to_string()))?;
    Ok(payload)
}

/// Validate and normalize one standalone resource-expression graph.
///
/// Literal subexpressions and predicates are evaluated through the same exact
/// constructor semantics used by [`evaluate_resource_expression`].
///
/// # Errors
///
/// Returns [`ResourceExpressionSubstitutionError`] when validation,
/// normalization, or a deterministic complexity budget fails.
pub fn normalize_resource_expression(
    payload: &[u8],
) -> Result<Vec<u8>, ResourceExpressionSubstitutionError> {
    substitute_resource_expression(payload, Vec::new())
}

/// Construct one numeric resource literal with a Rust-derived domain.
///
/// `payload` is the canonical encoding of the existing
/// `ResourceNumberProto` wire message. An explicit domain is checked through the
/// same graph contract as persisted resource expressions; otherwise Rust
/// introduces the narrowest portable numeric domain.
///
/// # Errors
///
/// Returns [`ResourceExpressionSubstitutionError`] when the numeric payload,
/// requested domain, unit, or deterministic complexity budget is invalid.
pub fn construct_resource_literal(
    payload: &[u8],
    unit: &str,
    domain: Option<i32>,
) -> Result<Vec<u8>, ResourceExpressionSubstitutionError> {
    if payload
        .len()
        .checked_add(unit.len())
        .is_none_or(|size| size > MAX_EXPRESSION_BYTES)
    {
        return Err(ResourceExpressionSubstitutionError::Budget("wire-size"));
    }
    let source = wire::ResourceNumberProto::decode(payload).map_err(|error| {
        ResourceExpressionSubstitutionError::Invalid(format!("malformed resource literal: {error}"))
    })?;
    if source.encode_to_vec() != payload {
        return Err(ResourceExpressionSubstitutionError::Invalid(
            "resource literal encoding is not canonical".to_owned(),
        ));
    }
    preflight_resource_number(&source)?;
    let parsed = Number::decode(&source)
        .map_err(|error| ResourceExpressionSubstitutionError::Invalid(error.to_string()))?;
    let inferred = match &parsed {
        Number::Integer(value) => {
            if value.is_negative() {
                wire::ResourceDomainProto::ResourceInteger
            } else {
                wire::ResourceDomainProto::ResourceNatural
            }
        }
        Number::Rational(_, _) => wire::ResourceDomainProto::ResourceRational,
        Number::RealBits(_) => wire::ResourceDomainProto::ResourceReal,
    };
    let resolved = domain
        .map_or(Ok(inferred), wire_domain)
        .map_err(substitution_evaluation_error)?;
    let literal = match (&source.value, resolved) {
        (
            Some(wire::resource_number_proto::Value::Integer(value)),
            wire::ResourceDomainProto::ResourceRational,
        ) => wire::ResourceNumberProto {
            value: Some(wire::resource_number_proto::Value::Rational(
                wire::ResourceRationalProto {
                    numerator: value.clone(),
                    denominator: "1".to_owned(),
                },
            )),
        },
        _ => source,
    };
    let mut builder = SubstitutionBuilder::default();
    let root = builder.normalize(wire::ResourceNodeProto {
        node: Some(wire::resource_node_proto::Node::Expression(
            wire::ResourceExpressionNodeProto {
                kind: "literal".to_owned(),
                domain: resolved.into(),
                unit: unit.to_owned(),
                literal: Some(literal),
                ..Default::default()
            },
        )),
    })?;
    builder.encode_root(root)
}

/// Validate one nominal resource-unit identity.
///
/// # Errors
///
/// Returns [`ResourceExpressionSubstitutionError`] when the unit is empty,
/// contains whitespace, or exceeds the deterministic wire-size budget.
pub fn validate_resource_unit(unit: &str) -> Result<(), ResourceExpressionSubstitutionError> {
    if unit.len() > MAX_EXPRESSION_BYTES {
        return Err(ResourceExpressionSubstitutionError::Budget("wire-size"));
    }
    if !resource_unit_name_is_valid(unit) {
        return Err(ResourceExpressionSubstitutionError::Invalid(
            "resource unit names must be nonempty and whitespace-free".to_owned(),
        ));
    }
    Ok(())
}

/// Construct one scoped resource-symbol expression through the Rust contract.
///
/// # Errors
///
/// Returns [`ResourceExpressionSubstitutionError`] when the declaration is
/// malformed or exceeds a deterministic complexity budget.
pub fn construct_resource_symbol(
    scope: &str,
    name: &str,
    domain: i32,
    unit: &str,
    origin: &str,
    constraints: &[String],
) -> Result<Vec<u8>, ResourceExpressionSubstitutionError> {
    if constraints.len() > 100_000 {
        return Err(ResourceExpressionSubstitutionError::Budget("node-count"));
    }
    let input_bytes = [scope, name, unit, origin]
        .into_iter()
        .chain(constraints.iter().map(String::as_str))
        .try_fold(0_usize, |total, value| total.checked_add(value.len()))
        .ok_or(ResourceExpressionSubstitutionError::Budget("wire-size"))?;
    if input_bytes > MAX_EXPRESSION_BYTES {
        return Err(ResourceExpressionSubstitutionError::Budget("wire-size"));
    }
    if scope.is_empty() || name.is_empty() {
        return Err(ResourceExpressionSubstitutionError::Invalid(
            "symbol scope and name must be nonempty".to_owned(),
        ));
    }
    if scope.contains("::") || name.contains("::") {
        return Err(ResourceExpressionSubstitutionError::Invalid(
            "'::' is reserved for qualified symbol identities".to_owned(),
        ));
    }
    validate_resource_unit(unit)?;
    wire_domain(domain).map_err(substitution_evaluation_error)?;

    let mut builder = SubstitutionBuilder::default();
    let root = builder.normalize(wire::ResourceNodeProto {
        node: Some(wire::resource_node_proto::Node::Expression(
            wire::ResourceExpressionNodeProto {
                kind: "symbol".to_owned(),
                domain,
                unit: unit.to_owned(),
                symbol: Some(wire::ResourceSymbolProto {
                    scope: scope.to_owned(),
                    name: name.to_owned(),
                    domain,
                    unit: unit.to_owned(),
                    origin: origin.to_owned(),
                    constraints: constraints.to_vec(),
                }),
                ..Default::default()
            },
        )),
    })?;
    builder.encode_root(root)
}

fn preflight_resource_number(
    source: &wire::ResourceNumberProto,
) -> Result<(), ResourceExpressionSubstitutionError> {
    use wire::resource_number_proto::Value as WireValue;

    let oversized = match source.value.as_ref() {
        Some(WireValue::Integer(value)) => value.len() > MAX_INTEGER_DECIMAL_DIGITS,
        Some(WireValue::Rational(value)) => {
            value.numerator.len() > MAX_INTEGER_DECIMAL_DIGITS
                || value.denominator.len() > MAX_INTEGER_DECIMAL_DIGITS
        }
        Some(WireValue::RealBits(_)) | None => false,
    };
    if oversized {
        return Err(ResourceExpressionSubstitutionError::Budget("numeric-size"));
    }
    Ok(())
}

/// Construct one algebraic resource-expression node with a Rust-derived contract.
///
/// Each argument and predicate is an independently canonical expression graph.
/// Rust imports the roots into one hash-consed DAG, derives the result domain and
/// nominal unit, and applies the ordinary constructor normalizer.
///
/// # Errors
///
/// Returns [`ResourceExpressionSubstitutionError`] when an input graph,
/// constructor contract, or deterministic complexity budget is invalid.
pub fn construct_resource_expression(
    kind: &str,
    arguments: &[Vec<u8>],
    predicates: &[Vec<u8>],
    attributes: &[(String, String)],
    binder_unit: Option<&str>,
    result_domain: Option<i32>,
    result_unit: Option<&str>,
) -> Result<Vec<u8>, ResourceExpressionSubstitutionError> {
    let input_count = arguments
        .len()
        .checked_add(predicates.len())
        .ok_or(ResourceExpressionSubstitutionError::Budget("node-count"))?;
    if input_count > 100_000 {
        return Err(ResourceExpressionSubstitutionError::Budget("node-count"));
    }
    let input_bytes = arguments
        .iter()
        .chain(predicates)
        .try_fold(0_usize, |total, payload| total.checked_add(payload.len()))
        .and_then(|total| total.checked_add(kind.len()))
        .and_then(|total| {
            attributes.iter().try_fold(total, |size, (name, value)| {
                size.checked_add(name.len())?.checked_add(value.len())
            })
        })
        .and_then(|total| total.checked_add(binder_unit.map_or(0, str::len)))
        .and_then(|total| total.checked_add(result_unit.map_or(0, str::len)))
        .ok_or(ResourceExpressionSubstitutionError::Budget("wire-size"))?;
    if input_bytes > MAX_EXPRESSION_BYTES {
        return Err(ResourceExpressionSubstitutionError::Budget("wire-size"));
    }

    let mut builder = SubstitutionBuilder::default();
    let imported_arguments = import_construction_roots(&mut builder, arguments, true)?;
    let imported_predicates = import_construction_roots(&mut builder, predicates, false)?;

    let mut expression = wire::ResourceExpressionNodeProto {
        kind: kind.to_owned(),
        arguments: imported_arguments,
        predicates: imported_predicates,
        attributes: attributes
            .iter()
            .map(|(name, value)| wire::ResourceAttributeProto {
                name: name.clone(),
                value: value.clone(),
            })
            .collect(),
        ..Default::default()
    };
    let bounded = matches!(kind, "bounded_sum" | "bounded_product");
    if (bounded && binder_unit != Some("1")) || (!bounded && binder_unit.is_some()) {
        return Err(ResourceExpressionSubstitutionError::Invalid(
            "bounded-expression binder must be dimensionless".to_owned(),
        ));
    }
    let inferred = infer_wire_expression_contract(&expression, &builder.nodes, &builder.values)
        .map_err(substitution_evaluation_error)?;
    let explicit = match (result_domain, result_unit) {
        (None, None) => None,
        (Some(domain), Some(unit)) => {
            wire_domain(domain).map_err(substitution_evaluation_error)?;
            Some((domain, unit.to_owned()))
        }
        _ => {
            return Err(ResourceExpressionSubstitutionError::Invalid(
                "resource expression result domain and unit must be supplied together".to_owned(),
            ));
        }
    };
    let (domain, unit) = match (inferred, explicit) {
        (Some(contract), None) => contract,
        (Some(_), Some(_)) => {
            return Err(ResourceExpressionSubstitutionError::Invalid(
                "derived resource expression constructors cannot override their result contract"
                    .to_owned(),
            ));
        }
        (None, Some(contract)) if matches!(kind, "contracted_function" | "recurrence") => contract,
        (None, _) => {
            return Err(ResourceExpressionSubstitutionError::Invalid(format!(
                "resource expression constructor {kind:?} requires an explicit contract"
            )));
        }
    };
    expression.domain = domain;
    expression.unit = unit;
    let root = builder.normalize(wire::ResourceNodeProto {
        node: Some(wire::resource_node_proto::Node::Expression(expression)),
    })?;
    builder.encode_root(root)
}

/// Construct one Boolean resource predicate through the Rust contract owner.
///
/// # Errors
///
/// Returns [`ResourceExpressionSubstitutionError`] when an input graph,
/// predicate contract, or deterministic complexity budget is invalid.
pub fn construct_resource_predicate(
    kind: &str,
    arguments: &[Vec<u8>],
    predicates: &[Vec<u8>],
) -> Result<Vec<u8>, ResourceExpressionSubstitutionError> {
    let input_count = arguments
        .len()
        .checked_add(predicates.len())
        .ok_or(ResourceExpressionSubstitutionError::Budget("node-count"))?;
    if input_count > 100_000 {
        return Err(ResourceExpressionSubstitutionError::Budget("node-count"));
    }
    let input_bytes = arguments
        .iter()
        .chain(predicates)
        .try_fold(kind.len(), |total, payload| {
            total.checked_add(payload.len())
        })
        .ok_or(ResourceExpressionSubstitutionError::Budget("wire-size"))?;
    if input_bytes > MAX_EXPRESSION_BYTES {
        return Err(ResourceExpressionSubstitutionError::Budget("wire-size"));
    }

    let mut builder = SubstitutionBuilder::default();
    let imported_arguments = import_construction_roots(&mut builder, arguments, true)?;
    let imported_predicates = import_construction_roots(&mut builder, predicates, false)?;
    let predicate = wire::ResourcePredicateNodeProto {
        kind: kind.to_owned(),
        arguments: imported_arguments,
        predicates: imported_predicates,
    };
    validate_wire_predicate_contract(&predicate, &builder.nodes)
        .map_err(substitution_evaluation_error)?;
    let root = builder.normalize(wire::ResourceNodeProto {
        node: Some(wire::resource_node_proto::Node::Predicate(predicate)),
    })?;
    builder.encode_root(root)
}

fn import_construction_roots(
    builder: &mut SubstitutionBuilder,
    payloads: &[Vec<u8>],
    expressions: bool,
) -> Result<Vec<u32>, ResourceExpressionSubstitutionError> {
    let empty_replacements = BTreeMap::new();
    let empty_bound = BTreeSet::new();
    let mut imported = Vec::with_capacity(payloads.len());
    for payload in payloads {
        let source = PreparedGraph::decode(payload)?;
        let root_is_expression = matches!(
            source.graph.node(source.graph.root()),
            Some(ExpressionGraphNode::Expression(_))
        );
        if root_is_expression != expressions {
            return Err(ResourceExpressionSubstitutionError::Invalid(
                "resource constructor input has the wrong root kind".to_owned(),
            ));
        }
        for node in &source.source.nodes {
            if let Some(wire::resource_node_proto::Node::Predicate(predicate)) = node.node.as_ref()
            {
                validate_wire_predicate_contract(predicate, &source.source.nodes)
                    .map_err(substitution_evaluation_error)?;
            }
        }
        imported.push(builder.import(
            &source,
            source.graph.root(),
            &empty_replacements,
            &empty_bound,
            0,
        )?);
    }
    Ok(imported)
}

/// Attach one nominal unit to a standalone resource-expression root.
///
/// Nested piecewise branch roots receive the same unit so every selected value
/// retains the public expression contract. Other descendants keep their own
/// algebraic units.
///
/// # Errors
///
/// Returns [`ResourceExpressionSubstitutionError`] when the unit or graph is
/// malformed, or traversal exceeds a deterministic complexity budget.
pub fn retag_resource_expression_unit(
    payload: &[u8],
    unit: &str,
) -> Result<Vec<u8>, ResourceExpressionSubstitutionError> {
    if !resource_unit_name_is_valid(unit) {
        return Err(ResourceExpressionSubstitutionError::Invalid(
            "resource expression unit is invalid".to_owned(),
        ));
    }
    let source = PreparedGraph::decode(payload)?;
    source.root_expression()?;
    let states = resource_unit_retag_states(&source)?;
    let mut builder = SubstitutionBuilder::default();
    let mut mapped = HashMap::<(u32, bool), u32>::with_capacity(states.len());
    for state in states {
        let node = rebuild_resource_unit_node(&source, state, unit, &mapped)?;
        let target = builder.intern(node)?;
        mapped.insert(state, target);
    }
    let root = mapped
        .get(&(source.graph.root(), true))
        .copied()
        .ok_or_else(|| {
            ResourceExpressionSubstitutionError::Invalid(
                "resource unit projection lost its root".to_owned(),
            )
        })?;
    builder.encode_root(root)
}

fn resource_unit_retag_states(
    source: &PreparedGraph,
) -> Result<Vec<(u32, bool)>, ResourceExpressionSubstitutionError> {
    let mut states = HashSet::new();
    let mut stack = vec![(source.graph.root(), true, 0_usize)];
    let mut steps = 0_usize;
    while let Some((identifier, retag, depth)) = stack.pop() {
        steps = steps
            .checked_add(1)
            .ok_or(ResourceExpressionSubstitutionError::Budget(
                "operation-count",
            ))?;
        if steps > MAX_EVALUATION_STEPS {
            return Err(ResourceExpressionSubstitutionError::Budget(
                "operation-count",
            ));
        }
        if depth > MAX_EVALUATION_DEPTH {
            return Err(ResourceExpressionSubstitutionError::Budget("nesting-depth"));
        }
        if !states.insert((identifier, retag)) {
            continue;
        }
        let next_depth = depth
            .checked_add(1)
            .ok_or(ResourceExpressionSubstitutionError::Budget("nesting-depth"))?;
        match source.graph.node(identifier) {
            Some(ExpressionGraphNode::Expression(node)) => {
                if retag
                    && matches!(
                        node.kind.as_str(),
                        "symbol" | "recurrence_system" | "recurrence_equation"
                    )
                {
                    stack.push((identifier, false, next_depth));
                }
                let branch_retag = retag && node.kind == "piecewise";
                stack.extend(
                    node.arguments
                        .iter()
                        .map(|child| (*child, branch_retag, next_depth)),
                );
                stack.extend(
                    node.predicates
                        .iter()
                        .map(|child| (*child, false, next_depth)),
                );
            }
            Some(ExpressionGraphNode::Predicate(node)) if !retag => {
                stack.extend(
                    node.arguments
                        .iter()
                        .map(|child| (*child, false, next_depth)),
                );
                stack.extend(
                    node.predicates
                        .iter()
                        .map(|child| (*child, false, next_depth)),
                );
            }
            Some(ExpressionGraphNode::Predicate(_)) => {
                return Err(ResourceExpressionSubstitutionError::Invalid(
                    "resource unit projection received a predicate root".to_owned(),
                ));
            }
            None => {
                return Err(ResourceExpressionSubstitutionError::Invalid(
                    "resource unit projection lost a node".to_owned(),
                ));
            }
        }
    }
    let mut ordered = states.into_iter().collect::<Vec<_>>();
    ordered.sort_unstable();
    Ok(ordered)
}

fn rebuild_resource_unit_node(
    source: &PreparedGraph,
    state: (u32, bool),
    unit: &str,
    mapped: &HashMap<(u32, bool), u32>,
) -> Result<wire::ResourceNodeProto, ResourceExpressionSubstitutionError> {
    let (identifier, retag) = state;
    let decoded = source.graph.node(identifier).ok_or_else(|| {
        ResourceExpressionSubstitutionError::Invalid(
            "resource unit projection lost a decoded node".to_owned(),
        )
    })?;
    let mut rebuilt = source
        .source
        .nodes
        .get(identifier as usize)
        .cloned()
        .ok_or_else(|| {
            ResourceExpressionSubstitutionError::Invalid(
                "resource unit projection lost an encoded node".to_owned(),
            )
        })?;
    match (decoded, rebuilt.node.as_mut()) {
        (
            ExpressionGraphNode::Expression(expression),
            Some(wire::resource_node_proto::Node::Expression(node)),
        ) => {
            if retag
                && matches!(
                    expression.kind.as_str(),
                    "symbol" | "recurrence_system" | "recurrence_equation"
                )
            {
                let original = mapped.get(&(identifier, false)).copied().ok_or_else(|| {
                    ResourceExpressionSubstitutionError::Invalid(
                        "resource unit projection lost a symbol declaration".to_owned(),
                    )
                })?;
                return Ok(wire::ResourceNodeProto {
                    node: Some(wire::resource_node_proto::Node::Expression(
                        wire::ResourceExpressionNodeProto {
                            kind: "multiply".to_owned(),
                            domain: node.domain,
                            unit: unit.to_owned(),
                            arguments: vec![original],
                            ..Default::default()
                        },
                    )),
                });
            }
            let branch_retag = retag && expression.kind == "piecewise";
            node.arguments = mapped_resource_children(&expression.arguments, branch_retag, mapped)?;
            node.predicates = mapped_resource_children(&expression.predicates, false, mapped)?;
            if retag {
                unit.clone_into(&mut node.unit);
            }
        }
        (
            ExpressionGraphNode::Predicate(predicate),
            Some(wire::resource_node_proto::Node::Predicate(node)),
        ) if !retag => {
            node.arguments = mapped_resource_children(&predicate.arguments, false, mapped)?;
            node.predicates = mapped_resource_children(&predicate.predicates, false, mapped)?;
        }
        _ => {
            return Err(ResourceExpressionSubstitutionError::Invalid(
                "resource unit projection changed node kind".to_owned(),
            ));
        }
    }
    Ok(rebuilt)
}

fn mapped_resource_children(
    children: &[u32],
    retag: bool,
    mapped: &HashMap<(u32, bool), u32>,
) -> Result<Vec<u32>, ResourceExpressionSubstitutionError> {
    children
        .iter()
        .map(|identifier| {
            mapped.get(&(*identifier, retag)).copied().ok_or_else(|| {
                ResourceExpressionSubstitutionError::Invalid(
                    "resource unit projection is not topological".to_owned(),
                )
            })
        })
        .collect()
}

/// Project logical width and non-Clifford work through the toy surface-code model.
///
/// Both inputs remain symbolic expression DAGs. The returned roots use the same
/// exact native constructor algebra as evaluation and substitution.
///
/// # Errors
///
/// Returns [`SurfaceCodeProjectionError`] when a model coefficient is invalid,
/// an input graph is malformed, or construction exceeds an expression budget.
pub fn project_surface_code_resources(
    logical_qubits: &[u8],
    non_clifford_gates: &[u8],
    alpha: f64,
    threshold: f64,
    physical_error_rate: f64,
    syndrome_cycle_seconds: f64,
) -> Result<SurfaceCodeResourceProjection, SurfaceCodeProjectionError> {
    for (name, value) in [
        ("alpha", alpha),
        ("threshold", threshold),
        ("physical_error_rate", physical_error_rate),
        ("syndrome_cycle_seconds", syndrome_cycle_seconds),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(SurfaceCodeProjectionError::Invalid(format!(
                "{name} must be positive and finite"
            )));
        }
    }
    if threshold <= physical_error_rate {
        return Err(SurfaceCodeProjectionError::Invalid(
            "threshold must exceed physical_error_rate".to_owned(),
        ));
    }
    let logarithm_base = threshold / physical_error_rate;
    if !logarithm_base.is_finite() {
        return Err(SurfaceCodeProjectionError::Invalid(
            "threshold ratio must be finite".to_owned(),
        ));
    }

    let logical_source = PreparedGraph::decode(logical_qubits)?;
    let logical_contract = logical_source.root_expression()?;
    let work_source = PreparedGraph::decode(non_clifford_gates)?;
    let work_contract = work_source.root_expression()?;
    if logical_contract.domain != crate::resource_rules::Domain::Natural
        || logical_contract.unit != "logical_qubits"
    {
        return Err(SurfaceCodeProjectionError::Invalid(
            "logical width must be a natural logical_qubits expression".to_owned(),
        ));
    }
    if work_contract.domain != crate::resource_rules::Domain::Natural
        || work_contract.unit != "semantic_operations"
    {
        return Err(SurfaceCodeProjectionError::Invalid(
            "non-Clifford work must be a natural semantic_operations expression".to_owned(),
        ));
    }
    let mut builder = SubstitutionBuilder::default();
    let logical_root = builder.import(
        &logical_source,
        logical_source.graph.root(),
        &BTreeMap::new(),
        &BTreeSet::new(),
        0,
    )?;
    let work_root = builder.import(
        &work_source,
        work_source.graph.root(),
        &BTreeMap::new(),
        &BTreeSet::new(),
        0,
    )?;
    let real = wire::ResourceDomainProto::ResourceReal.into();
    let natural = wire::ResourceDomainProto::ResourceNatural.into();
    let integer = wire::ResourceDomainProto::ResourceInteger.into();
    let logical = builder.retag_expression(logical_root, real, "1")?;
    let work = builder.retag_expression(work_root, real, "1")?;
    let one = builder.integer_literal(1, natural, "1")?;
    let two = builder.integer_literal(2, natural, "1")?;
    let three = builder.integer_literal(3, natural, "1")?;
    let negative_one = builder.integer_literal(-1, integer, "1")?;
    let alpha = builder.real_literal(alpha, real, "1")?;
    let logarithm_base = builder.real_literal(logarithm_base, real, "1")?;
    let cycle = builder.real_literal(syndrome_cycle_seconds, real, "1")?;

    let scaled_logical = builder.operation("multiply", real, "1", vec![alpha, logical])?;
    let scaled_volume = builder.operation("multiply", real, "1", vec![scaled_logical, work])?;
    let scaled_volume = builder.operation("maximum", real, "1", vec![one, scaled_volume])?;
    let logarithm = builder.operation("log", real, "1", vec![scaled_volume, logarithm_base])?;
    let doubled_logarithm = builder.operation("multiply", real, "1", vec![two, logarithm])?;
    let raw_distance = builder.operation("ceiling", integer, "1", vec![doubled_logarithm])?;
    let minimum_distance = builder.operation("maximum", natural, "1", vec![raw_distance, three])?;
    let shifted_distance =
        builder.operation("add", integer, "1", vec![minimum_distance, negative_one])?;
    let half_distance =
        builder.operation("ceiling_divide", integer, "1", vec![shifted_distance, two])?;
    let even_distance = builder.operation("multiply", integer, "1", vec![two, half_distance])?;
    let distance = builder.operation("add", integer, "1", vec![even_distance, one])?;

    let distance_squared = builder.operation("power", integer, "1", vec![distance, two])?;
    let four = builder.integer_literal(4, natural, "1")?;
    let physical_qubits =
        builder.operation("multiply", real, "1", vec![four, logical, distance_squared])?;
    let physical_qubits = builder.retag_expression(physical_qubits, natural, "physical_qubits")?;
    let duration = builder.operation("multiply", real, "1", vec![work, distance, cycle])?;
    let duration = builder.retag_expression(duration, real, "seconds")?;
    let qubit_time = builder.operation("multiply", real, "1", vec![physical_qubits, duration])?;
    let qubit_time = builder.retag_expression(qubit_time, real, "physical_qubit_seconds")?;
    let distance = builder.retag_expression(distance, natural, "1")?;

    Ok(SurfaceCodeResourceProjection {
        code_distance: builder.encode_root(distance)?,
        physical_qubits: builder.encode_root(physical_qubits)?,
        duration: builder.encode_root(duration)?,
        qubit_time: builder.encode_root(qubit_time)?,
    })
}

fn infer_wire_expression_contract(
    node: &wire::ResourceExpressionNodeProto,
    nodes: &[wire::ResourceNodeProto],
    values: &[Value],
) -> Result<Option<(i32, String)>, ResourceExpressionEvaluationError> {
    let arguments = node
        .arguments
        .iter()
        .map(|identifier| wire_expression(nodes, *identifier))
        .collect::<Result<Vec<_>, _>>()?;
    let contract = match node.kind.as_str() {
        "literal" | "symbol" | "contracted_function" | "recurrence" => return Ok(None),
        "recurrence_equation" => {
            validate_recurrence_equation(node, &arguments)?;
            let body = arguments.last().expect("checked body");
            (wire_domain(body.domain)?, body.unit.clone())
        }
        "recurrence_system" => {
            validate_recurrence_system(node, nodes, &mut 0)?;
            (wire_domain(arguments[0].domain)?, arguments[0].unit.clone())
        }
        "add" => {
            require_expression_arity("add", arguments.len(), 1, None)?;
            let unit = common_wire_unit(
                &arguments,
                "cannot add resource expressions with different units",
            )?;
            (common_wire_domain(&arguments)?, unit)
        }
        "multiply" => {
            require_expression_arity("multiply", arguments.len(), 1, None)?;
            (
                common_wire_domain(&arguments)?,
                product_wire_unit(&arguments),
            )
        }
        "minimum" | "maximum" => infer_extremum_contract(node, &arguments)?,
        "power" => infer_power_contract(node, &arguments, values)?,
        "absolute" => {
            require_expression_arity("absolute", arguments.len(), 1, Some(1))?;
            numeric_wire_domain(arguments[0])?;
            (wire_domain(arguments[0].domain)?, arguments[0].unit.clone())
        }
        "floor" | "ceiling" => infer_rounding_contract(node, &arguments)?,
        "floor_divide" | "ceiling_divide" => infer_division_contract(node, &arguments)?,
        "exp" | "log" | "log2" | "square_root" => infer_transcendental_contract(node, &arguments)?,
        "piecewise" => infer_piecewise_contract(node, nodes, &arguments)?,
        "bounded_sum" | "bounded_product" => infer_bounded_contract(node, &arguments)?,
        kind => {
            return invalid_value(&format!(
                "unsupported resource expression constructor {kind:?}"
            ));
        }
    };
    Ok(Some((contract.0.into(), contract.1)))
}

fn validate_recurrence_equation(
    node: &wire::ResourceExpressionNodeProto,
    arguments: &[&wire::ResourceExpressionNodeProto],
) -> Result<(), ResourceExpressionEvaluationError> {
    if !node
        .attributes
        .iter()
        .any(|attribute| attribute.name == "version" && attribute.value == "1")
    {
        return invalid_value("unsupported recurrence equation version");
    }
    let Some(body) = arguments.last() else {
        return invalid_value("recurrence equation has no body");
    };
    if node.predicates.len() != 1
        || node.domain != body.domain
        || node.unit != body.unit
        || wire_domain(body.domain)? != wire::ResourceDomainProto::ResourceNatural
    {
        return invalid_value(
            "recurrence equation requires a natural body and one applicability condition",
        );
    }
    let mut names = BTreeSet::new();
    for parameter in arguments.iter().take(arguments.len() - 1) {
        let Some(symbol) = parameter.symbol.as_ref() else {
            return invalid_value("recurrence formal parameters must be symbols");
        };
        if parameter.kind != "symbol"
            || parameter.unit != "1"
            || !matches!(
                wire_domain(parameter.domain)?,
                wire::ResourceDomainProto::ResourceInteger
                    | wire::ResourceDomainProto::ResourceNatural
            )
            || !names.insert(format!("{}::{}", symbol.scope, symbol.name))
        {
            return invalid_value(
                "recurrence formal parameters must be distinct dimensionless integer symbols",
            );
        }
    }
    Ok(())
}

fn validate_recurrence_system(
    system: &wire::ResourceExpressionNodeProto,
    nodes: &[wire::ResourceNodeProto],
    steps: &mut usize,
) -> Result<(), ResourceExpressionEvaluationError> {
    if system.arguments.len() < 2 {
        return invalid_value("recurrence system requires an invocation and equations");
    }
    let root = wire_expression(nodes, system.arguments[0])?;
    if system.domain != root.domain || system.unit != root.unit {
        return invalid_value("recurrence system disagrees with its invocation contract");
    }
    let mut definitions = BTreeMap::new();
    for identifier in system.arguments.iter().skip(1) {
        let equation = wire_expression(nodes, *identifier)?;
        if equation.kind != "recurrence_equation"
            || equation.unit != system.unit
            || equation.domain != system.domain
        {
            return invalid_value("recurrence system contains an incompatible equation");
        }
        let name = equation
            .attributes
            .iter()
            .find(|attribute| attribute.name == "name")
            .ok_or_else(|| {
                ResourceExpressionEvaluationError::Invalid(
                    "recurrence equation has no identity".to_owned(),
                )
            })?;
        if definitions.insert(name.value.as_str(), equation).is_some() {
            return invalid_value("duplicate recurrence equation identity");
        }
    }
    let mut pending = system.arguments.clone();
    let mut visited = HashSet::new();
    while let Some(identifier) = pending.pop() {
        *steps += 1;
        if *steps > MAX_EVALUATION_STEPS {
            return Err(ResourceExpressionEvaluationError::Budget(
                "recurrence-definition-check",
            ));
        }
        if !visited.insert(identifier) {
            continue;
        }
        match nodes
            .get(identifier as usize)
            .and_then(|node| node.node.as_ref())
        {
            Some(wire::resource_node_proto::Node::Expression(node)) => {
                if node.kind == "recurrence_system" {
                    continue;
                }
                if node.kind == "recurrence" {
                    let name = node
                        .attributes
                        .iter()
                        .find(|attribute| attribute.name == "name")
                        .ok_or_else(|| {
                            ResourceExpressionEvaluationError::Invalid(
                                "recurrence call has no identity".to_owned(),
                            )
                        })?;
                    let equation = definitions.get(name.value.as_str()).ok_or_else(|| {
                        ResourceExpressionEvaluationError::Invalid(
                            "recurrence call refers to an undefined equation".to_owned(),
                        )
                    })?;
                    validate_recurrence_call_contract(node, equation, nodes)?;
                }
                pending.extend(&node.arguments);
                pending.extend(&node.predicates);
            }
            Some(wire::resource_node_proto::Node::Predicate(node)) => {
                pending.extend(&node.arguments);
                pending.extend(&node.predicates);
            }
            None => return invalid_value("recurrence system lost a node"),
        }
    }
    Ok(())
}

fn validate_recurrence_call_contract(
    node: &wire::ResourceExpressionNodeProto,
    equation: &wire::ResourceExpressionNodeProto,
    nodes: &[wire::ResourceNodeProto],
) -> Result<(), ResourceExpressionEvaluationError> {
    if node
        .attributes
        .iter()
        .find(|attribute| attribute.name == "version")
        .map(|attribute| &attribute.value)
        != equation
            .attributes
            .iter()
            .find(|attribute| attribute.name == "version")
            .map(|attribute| &attribute.value)
    {
        return invalid_value("recurrence call disagrees with its equation version");
    }
    if node.arguments.len() + 1 != equation.arguments.len()
        || node.domain != equation.domain
        || node.unit != equation.unit
    {
        return invalid_value("recurrence call disagrees with its equation signature");
    }
    for argument in &node.arguments {
        let argument = wire_expression(nodes, *argument)?;
        if argument.unit != "1"
            || !matches!(
                wire_domain(argument.domain)?,
                wire::ResourceDomainProto::ResourceInteger
                    | wire::ResourceDomainProto::ResourceNatural
            )
        {
            return invalid_value("recurrence call arguments must be dimensionless integers");
        }
    }
    Ok(())
}

pub(crate) fn validate_recurrence_graph(
    source: &wire::ResourceExpressionGraphProto,
) -> Result<(), ResourceExpressionEvaluationError> {
    let mut steps = 0;
    for node in &source.nodes {
        if let Some(wire::resource_node_proto::Node::Expression(node)) = node.node.as_ref() {
            if node.kind == "recurrence_equation" {
                let arguments = node
                    .arguments
                    .iter()
                    .map(|id| wire_expression(&source.nodes, *id))
                    .collect::<Result<Vec<_>, _>>()?;
                validate_recurrence_equation(node, &arguments)?;
            } else if node.kind == "recurrence_system" {
                validate_recurrence_system(node, &source.nodes, &mut steps)?;
            }
        }
    }
    Ok(())
}

fn infer_extremum_contract(
    node: &wire::ResourceExpressionNodeProto,
    arguments: &[&wire::ResourceExpressionNodeProto],
) -> Result<(wire::ResourceDomainProto, String), ResourceExpressionEvaluationError> {
    use wire::ResourceDomainProto as WireDomain;

    require_expression_arity(node.kind.as_str(), arguments.len(), 1, None)?;
    let unit = common_wire_unit(
        arguments,
        &format!("{} operands must have the same unit", node.kind),
    )?;
    let mut domain = common_wire_domain(arguments)?;
    if node.kind == "maximum"
        && matches!(
            domain,
            WireDomain::ResourceNatural | WireDomain::ResourceInteger
        )
        && arguments
            .iter()
            .any(|argument| wire_domain(argument.domain).ok() == Some(WireDomain::ResourceNatural))
    {
        domain = WireDomain::ResourceNatural;
    }
    Ok((domain, unit))
}

fn infer_power_contract(
    node: &wire::ResourceExpressionNodeProto,
    arguments: &[&wire::ResourceExpressionNodeProto],
    values: &[Value],
) -> Result<(wire::ResourceDomainProto, String), ResourceExpressionEvaluationError> {
    use wire::ResourceDomainProto as WireDomain;

    require_expression_arity("power", arguments.len(), 2, Some(2))?;
    let base = arguments[0];
    let exponent = arguments[1];
    let exponent_domain = wire_domain(exponent.domain)?;
    if exponent.unit != "1"
        || !matches!(
            exponent_domain,
            WireDomain::ResourceNatural
                | WireDomain::ResourceInteger
                | WireDomain::ResourceRational
        )
    {
        return invalid_value("power exponent must be dimensionless and integral or rational");
    }
    let exponent_value = values
        .get(node.arguments[1] as usize)
        .and_then(|value| match value {
            Value::Numeric(value) => Some(value),
            Value::Boolean(_) | Value::Unknown => None,
        });
    let exponent_integer = exponent_value.and_then(Numeric::as_integer);
    if base.unit != "1" && exponent_integer.is_none() {
        return invalid_value("dimensioned powers require a literal integer exponent");
    }
    let common = common_wire_domain(arguments)?;
    let domain = match exponent_value {
        Some(value) if value.as_integer().is_none() => WireDomain::ResourceReal,
        Some(value)
            if value.as_integer().is_some_and(Signed::is_negative)
                && !matches!(
                    wire_domain(base.domain)?,
                    WireDomain::ResourceReal | WireDomain::ResourceProbability
                ) =>
        {
            WireDomain::ResourceRational
        }
        Some(value) if value.as_integer().is_some_and(Signed::is_negative) => {
            WireDomain::ResourceReal
        }
        _ => common,
    };
    Ok((domain, power_wire_unit(&base.unit, exponent_integer)))
}

fn infer_rounding_contract(
    node: &wire::ResourceExpressionNodeProto,
    arguments: &[&wire::ResourceExpressionNodeProto],
) -> Result<(wire::ResourceDomainProto, String), ResourceExpressionEvaluationError> {
    require_expression_arity(node.kind.as_str(), arguments.len(), 1, Some(1))?;
    numeric_wire_domain(arguments[0])?;
    Ok((
        wire::ResourceDomainProto::ResourceInteger,
        arguments[0].unit.clone(),
    ))
}

fn infer_division_contract(
    node: &wire::ResourceExpressionNodeProto,
    arguments: &[&wire::ResourceExpressionNodeProto],
) -> Result<(wire::ResourceDomainProto, String), ResourceExpressionEvaluationError> {
    require_expression_arity(node.kind.as_str(), arguments.len(), 2, Some(2))?;
    numeric_wire_domain(arguments[0])?;
    numeric_wire_domain(arguments[1])?;
    if arguments[1].unit != "1" {
        return invalid_value(if node.kind == "floor_divide" {
            "floor-divide denominator must be dimensionless"
        } else {
            "ceiling-divide denominator must be dimensionless"
        });
    }
    Ok((
        wire::ResourceDomainProto::ResourceInteger,
        arguments[0].unit.clone(),
    ))
}

fn infer_transcendental_contract(
    node: &wire::ResourceExpressionNodeProto,
    arguments: &[&wire::ResourceExpressionNodeProto],
) -> Result<(wire::ResourceDomainProto, String), ResourceExpressionEvaluationError> {
    let maximum = if node.kind == "log" { Some(2) } else { Some(1) };
    require_expression_arity(node.kind.as_str(), arguments.len(), 1, maximum)?;
    for argument in arguments {
        numeric_wire_domain(argument)?;
        if argument.unit != "1" {
            return invalid_value(match node.kind.as_str() {
                "exp" => "exponential operand must be dimensionless",
                "log" => "logarithm operand and base must be dimensionless",
                "log2" => "log2 operand must be dimensionless",
                _ => "square-root operand must be dimensionless",
            });
        }
    }
    Ok((wire::ResourceDomainProto::ResourceReal, "1".to_owned()))
}

fn infer_piecewise_contract(
    node: &wire::ResourceExpressionNodeProto,
    nodes: &[wire::ResourceNodeProto],
    arguments: &[&wire::ResourceExpressionNodeProto],
) -> Result<(wire::ResourceDomainProto, String), ResourceExpressionEvaluationError> {
    if node.predicates.is_empty() || arguments.len() != node.predicates.len() + 1 {
        return invalid_value("piecewise requires one value per predicate and a default");
    }
    for identifier in &node.predicates {
        wire_predicate(nodes, *identifier)?;
    }
    let unit = common_wire_unit(arguments, "piecewise branch units must agree")?;
    Ok((common_wire_domain(arguments)?, unit))
}

fn infer_bounded_contract(
    node: &wire::ResourceExpressionNodeProto,
    arguments: &[&wire::ResourceExpressionNodeProto],
) -> Result<(wire::ResourceDomainProto, String), ResourceExpressionEvaluationError> {
    use wire::ResourceDomainProto as WireDomain;

    require_expression_arity(node.kind.as_str(), arguments.len(), 3, Some(3))?;
    let binder_domain = node
        .attributes
        .iter()
        .find(|attribute| attribute.name == "binder_domain")
        .map(|attribute| attribute.value.as_str());
    if !matches!(binder_domain, Some("natural" | "integer")) {
        return invalid_value("bounded-expression binder must be integral");
    }
    for bound in &arguments[..2] {
        let domain = wire_domain(bound.domain)?;
        if bound.unit != "1"
            || !matches!(
                domain,
                WireDomain::ResourceNatural | WireDomain::ResourceInteger
            )
        {
            return invalid_value("bounded-expression limits must be dimensionless and integral");
        }
    }
    let body = arguments[2];
    numeric_wire_domain(body)?;
    if node.kind == "bounded_product" && body.unit != "1" {
        return invalid_value("bounded products currently require a dimensionless body");
    }
    Ok((wire_domain(body.domain)?, body.unit.clone()))
}

fn validate_wire_predicate_contract(
    node: &wire::ResourcePredicateNodeProto,
    nodes: &[wire::ResourceNodeProto],
) -> Result<(), ResourceExpressionEvaluationError> {
    let arguments = node
        .arguments
        .iter()
        .map(|identifier| wire_expression(nodes, *identifier))
        .collect::<Result<Vec<_>, _>>()?;
    match node.kind.as_str() {
        "equal" | "not_equal" | "less" | "less_equal" | "greater" | "greater_equal" => {
            if arguments.len() != 2 || !node.predicates.is_empty() {
                return invalid_value("comparison predicates require two expressions");
            }
            if arguments[0].unit != arguments[1].unit {
                return invalid_value("comparison predicate units must agree");
            }
        }
        "divisible" => {
            if arguments.len() != 2 || !node.predicates.is_empty() {
                return invalid_value("divisibility requires a value and divisor");
            }
            if arguments.iter().any(|argument| argument.unit != "1") {
                return invalid_value("divisibility operands must be dimensionless and integral");
            }
            for argument in &arguments {
                if !matches!(
                    wire_domain(argument.domain)?,
                    wire::ResourceDomainProto::ResourceNatural
                        | wire::ResourceDomainProto::ResourceInteger
                ) {
                    return invalid_value(
                        "divisibility operands must be dimensionless and integral",
                    );
                }
            }
        }
        "true" | "false" if arguments.is_empty() && node.predicates.is_empty() => {}
        "and" | "or" if arguments.is_empty() && !node.predicates.is_empty() => {
            for identifier in &node.predicates {
                wire_predicate(nodes, *identifier)?;
            }
        }
        "not" if arguments.is_empty() && node.predicates.len() == 1 => {
            wire_predicate(nodes, node.predicates[0])?;
        }
        "true" | "false" | "and" | "or" | "not" => {
            return invalid_value("invalid resource predicate constructor shape");
        }
        kind => return invalid_value(&format!("unsupported resource predicate {kind:?}")),
    }
    Ok(())
}

fn wire_expression(
    nodes: &[wire::ResourceNodeProto],
    identifier: u32,
) -> Result<&wire::ResourceExpressionNodeProto, ResourceExpressionEvaluationError> {
    match nodes
        .get(identifier as usize)
        .and_then(|node| node.node.as_ref())
    {
        Some(wire::resource_node_proto::Node::Expression(expression)) => Ok(expression),
        _ => invalid_value("resource expression reference does not denote an expression"),
    }
}

fn wire_predicate(
    nodes: &[wire::ResourceNodeProto],
    identifier: u32,
) -> Result<&wire::ResourcePredicateNodeProto, ResourceExpressionEvaluationError> {
    match nodes
        .get(identifier as usize)
        .and_then(|node| node.node.as_ref())
    {
        Some(wire::resource_node_proto::Node::Predicate(predicate)) => Ok(predicate),
        _ => invalid_value("resource predicate reference does not denote a predicate"),
    }
}

fn wire_domain(value: i32) -> Result<wire::ResourceDomainProto, ResourceExpressionEvaluationError> {
    let domain = wire::ResourceDomainProto::try_from(value).map_err(|_| {
        ResourceExpressionEvaluationError::Invalid("invalid resource domain".to_owned())
    })?;
    if domain == wire::ResourceDomainProto::ResourceDomainUnspecified {
        return invalid_value("resource domain is unspecified");
    }
    Ok(domain)
}

fn numeric_wire_domain(
    expression: &wire::ResourceExpressionNodeProto,
) -> Result<wire::ResourceDomainProto, ResourceExpressionEvaluationError> {
    let domain = wire_domain(expression.domain)?;
    if domain == wire::ResourceDomainProto::ResourceBoolean {
        return invalid_value("numeric resource expression contains a Boolean operand");
    }
    Ok(domain)
}

fn common_wire_domain(
    expressions: &[&wire::ResourceExpressionNodeProto],
) -> Result<wire::ResourceDomainProto, ResourceExpressionEvaluationError> {
    let mut common = None;
    for expression in expressions {
        let domain = numeric_wire_domain(expression)?;
        if common.is_none_or(|current| wire_domain_rank(domain) > wire_domain_rank(current)) {
            common = Some(domain);
        }
    }
    common.ok_or_else(|| {
        ResourceExpressionEvaluationError::Invalid(
            "resource expression requires at least one operand".to_owned(),
        )
    })
}

const fn wire_domain_rank(domain: wire::ResourceDomainProto) -> u8 {
    match domain {
        wire::ResourceDomainProto::ResourceNatural => 0,
        wire::ResourceDomainProto::ResourceInteger => 1,
        wire::ResourceDomainProto::ResourceRational => 2,
        wire::ResourceDomainProto::ResourceProbability => 3,
        wire::ResourceDomainProto::ResourceReal => 4,
        wire::ResourceDomainProto::ResourceBoolean
        | wire::ResourceDomainProto::ResourceDomainUnspecified => 5,
    }
}

fn common_wire_unit(
    expressions: &[&wire::ResourceExpressionNodeProto],
    message: &str,
) -> Result<String, ResourceExpressionEvaluationError> {
    let Some(first) = expressions.first() else {
        return invalid_value("resource expression requires at least one operand");
    };
    if expressions
        .iter()
        .any(|expression| expression.unit != first.unit)
    {
        return invalid_value(message);
    }
    Ok(first.unit.clone())
}

fn product_wire_unit(expressions: &[&wire::ResourceExpressionNodeProto]) -> String {
    let mut units = expressions
        .iter()
        .filter(|expression| expression.unit != "1")
        .map(|expression| expression.unit.as_str())
        .collect::<Vec<_>>();
    units.sort_unstable();
    if units.is_empty() {
        "1".to_owned()
    } else {
        units.join("*")
    }
}

fn power_wire_unit(unit: &str, exponent: Option<&BigInt>) -> String {
    let Some(exponent) = exponent else {
        return "1".to_owned();
    };
    if unit == "1" || exponent.is_zero() {
        "1".to_owned()
    } else if exponent.is_one() {
        unit.to_owned()
    } else {
        format!("{unit}^{exponent}")
    }
}

fn require_expression_arity(
    kind: &str,
    actual: usize,
    minimum: usize,
    maximum: Option<usize>,
) -> Result<(), ResourceExpressionEvaluationError> {
    if actual < minimum || maximum.is_some_and(|maximum| actual > maximum) {
        return invalid_value(&format!(
            "resource expression constructor {kind:?} has invalid arity"
        ));
    }
    Ok(())
}

struct PreparedGraph {
    source: wire::ResourceExpressionGraphProto,
    graph: ExpressionGraph,
}

impl PreparedGraph {
    fn decode(payload: &[u8]) -> Result<Self, ResourceExpressionSubstitutionError> {
        if payload.len() > MAX_EXPRESSION_BYTES {
            return Err(ResourceExpressionSubstitutionError::Budget("wire-size"));
        }
        let source = wire::ResourceExpressionGraphProto::decode(payload)
            .map_err(|error| ResourceExpressionSubstitutionError::Invalid(error.to_string()))?;
        if source.encode_to_vec() != payload {
            return Err(ResourceExpressionSubstitutionError::Invalid(
                "encoding is not canonical".to_owned(),
            ));
        }
        let graph = ExpressionGraph::decode_any(&source)
            .map_err(|error| ResourceExpressionSubstitutionError::Invalid(error.to_string()))?;
        Ok(Self { source, graph })
    }

    fn root_expression(
        &self,
    ) -> Result<&crate::resource_rules::ExpressionNode, ResourceExpressionSubstitutionError> {
        self.graph
            .expression(self.graph.root())
            .map_err(|error| ResourceExpressionSubstitutionError::Invalid(error.to_string()))
    }
}

#[derive(Default)]
struct SubstitutionBuilder {
    nodes: Vec<wire::ResourceNodeProto>,
    identifiers: HashMap<Vec<u8>, u32>,
    structural_keys: Vec<[u8; 32]>,
    values: Vec<Value>,
    steps: usize,
}

enum NormalizedNode {
    Existing(u32),
    Node(Box<wire::ResourceNodeProto>),
}

impl SubstitutionBuilder {
    fn import(
        &mut self,
        source: &PreparedGraph,
        identifier: u32,
        replacements: &BTreeMap<String, (PreparedGraph, bool)>,
        bound: &BTreeSet<String>,
        depth: usize,
    ) -> Result<u32, ResourceExpressionSubstitutionError> {
        self.steps =
            self.steps
                .checked_add(1)
                .ok_or(ResourceExpressionSubstitutionError::Budget(
                    "operation-count",
                ))?;
        if self.steps > MAX_EVALUATION_STEPS {
            return Err(ResourceExpressionSubstitutionError::Budget(
                "operation-count",
            ));
        }
        if depth > MAX_EVALUATION_DEPTH {
            return Err(ResourceExpressionSubstitutionError::Budget("nesting-depth"));
        }
        let decoded = source.graph.node(identifier).ok_or_else(|| {
            ResourceExpressionSubstitutionError::Invalid("substitution lost a node".to_owned())
        })?;
        let raw = source
            .source
            .nodes
            .get(identifier as usize)
            .ok_or_else(|| {
                ResourceExpressionSubstitutionError::Invalid(
                    "substitution lost an encoded node".to_owned(),
                )
            })?;
        if let ExpressionGraphNode::Expression(expression) = decoded
            && expression.kind == "symbol"
        {
            let symbol = expression.symbol.as_ref().ok_or_else(|| {
                ResourceExpressionSubstitutionError::Invalid("symbol has no declaration".to_owned())
            })?;
            let identity = format!("{}::{}", symbol.scope, symbol.name);
            if !bound.contains(&identity)
                && let Some((replacement, adapt_literal)) = replacements.get(&identity)
            {
                let root = replacement.root_expression()?;
                if *adapt_literal {
                    return self.import_adapted_literal(replacement, raw);
                }
                if root.unit != expression.unit {
                    return Err(ResourceExpressionSubstitutionError::Invalid(format!(
                        "symbol substitution changes a nominal unit for {identity:?} from {:?} to {:?}",
                        expression.unit, root.unit
                    )));
                }
                return self.import(
                    replacement,
                    replacement.graph.root(),
                    &BTreeMap::new(),
                    &BTreeSet::new(),
                    depth + 1,
                );
            }
        }
        let rebuilt = match decoded {
            ExpressionGraphNode::Expression(expression) => {
                self.rebuild_expression(source, raw, expression, replacements, bound, depth)?
            }
            ExpressionGraphNode::Predicate(predicate) => {
                self.rebuild_predicate(source, raw, predicate, replacements, bound, depth)?
            }
        };
        self.normalize(rebuilt)
    }

    fn rebuild_expression(
        &mut self,
        source: &PreparedGraph,
        raw: &wire::ResourceNodeProto,
        expression: &crate::resource_rules::ExpressionNode,
        replacements: &BTreeMap<String, (PreparedGraph, bool)>,
        bound: &BTreeSet<String>,
        depth: usize,
    ) -> Result<wire::ResourceNodeProto, ResourceExpressionSubstitutionError> {
        let bounded = matches!(expression.kind.as_str(), "bounded_sum" | "bounded_product");
        let mut active = bound.clone();
        if bounded {
            let attributes = expression
                .attributes
                .iter()
                .cloned()
                .collect::<BTreeMap<_, _>>();
            let scope = attributes.get("binder_scope").ok_or_else(|| {
                ResourceExpressionSubstitutionError::Invalid(
                    "bounded expression has no binder scope".to_owned(),
                )
            })?;
            let name = attributes.get("binder_name").ok_or_else(|| {
                ResourceExpressionSubstitutionError::Invalid(
                    "bounded expression has no binder name".to_owned(),
                )
            })?;
            active.insert(format!("{scope}::{name}"));
        }
        if expression.kind == "recurrence_equation" {
            for parameter in expression
                .arguments
                .iter()
                .take(expression.arguments.len() - 1)
            {
                let parameter = source.graph.expression(*parameter).map_err(|error| {
                    ResourceExpressionSubstitutionError::Invalid(error.to_string())
                })?;
                let symbol = parameter.symbol.as_ref().ok_or_else(|| {
                    ResourceExpressionSubstitutionError::Invalid(
                        "recurrence parameter is not a symbol".to_owned(),
                    )
                })?;
                active.insert(format!("{}::{}", symbol.scope, symbol.name));
            }
        }
        let arguments = expression
            .arguments
            .iter()
            .enumerate()
            .map(|(index, child)| {
                let child_bound =
                    if expression.kind == "recurrence_equation" || (bounded && index == 2) {
                        &active
                    } else {
                        bound
                    };
                self.import(source, *child, replacements, child_bound, depth + 1)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let predicates = expression
            .predicates
            .iter()
            .map(|child| {
                self.import(
                    source,
                    *child,
                    replacements,
                    if expression.kind == "recurrence_equation" {
                        &active
                    } else {
                        bound
                    },
                    depth + 1,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut rebuilt = raw.clone();
        let Some(wire::resource_node_proto::Node::Expression(node)) = rebuilt.node.as_mut() else {
            return Err(ResourceExpressionSubstitutionError::Invalid(
                "expression projection changed node kind".to_owned(),
            ));
        };
        node.arguments = arguments;
        node.predicates = predicates;
        Ok(rebuilt)
    }

    fn rebuild_predicate(
        &mut self,
        source: &PreparedGraph,
        raw: &wire::ResourceNodeProto,
        predicate: &crate::resource_rules::PredicateNode,
        replacements: &BTreeMap<String, (PreparedGraph, bool)>,
        bound: &BTreeSet<String>,
        depth: usize,
    ) -> Result<wire::ResourceNodeProto, ResourceExpressionSubstitutionError> {
        let arguments = predicate
            .arguments
            .iter()
            .map(|child| self.import(source, *child, replacements, bound, depth + 1))
            .collect::<Result<Vec<_>, _>>()?;
        let predicates = predicate
            .predicates
            .iter()
            .map(|child| self.import(source, *child, replacements, bound, depth + 1))
            .collect::<Result<Vec<_>, _>>()?;
        let mut rebuilt = raw.clone();
        let Some(wire::resource_node_proto::Node::Predicate(node)) = rebuilt.node.as_mut() else {
            return Err(ResourceExpressionSubstitutionError::Invalid(
                "predicate projection changed node kind".to_owned(),
            ));
        };
        node.arguments = arguments;
        node.predicates = predicates;
        Ok(rebuilt)
    }

    fn intern(
        &mut self,
        node: wire::ResourceNodeProto,
    ) -> Result<u32, ResourceExpressionSubstitutionError> {
        let encoding = node.encode_to_vec();
        if let Some(identifier) = self.identifiers.get(&encoding) {
            return Ok(*identifier);
        }
        if self.nodes.len() >= 100_000 {
            return Err(ResourceExpressionSubstitutionError::Budget("node-count"));
        }
        let identifier = u32::try_from(self.nodes.len())
            .map_err(|_| ResourceExpressionSubstitutionError::Budget("node-count"))?;
        let value = known_wire_value(&node, &self.values).map_err(substitution_evaluation_error)?;
        let structural_key = wire_structural_key(&node, &self.structural_keys)?;
        self.nodes.push(node);
        self.structural_keys.push(structural_key);
        self.values.push(value);
        self.identifiers.insert(encoding, identifier);
        Ok(identifier)
    }

    fn normalize(
        &mut self,
        node: wire::ResourceNodeProto,
    ) -> Result<u32, ResourceExpressionSubstitutionError> {
        let node = match self.normalize_structure(node)? {
            NormalizedNode::Existing(identifier) => return Ok(identifier),
            NormalizedNode::Node(node) => *node,
        };
        let value = known_wire_value(&node, &self.values).map_err(substitution_evaluation_error)?;
        match (&node.node, value) {
            (
                Some(wire::resource_node_proto::Node::Expression(expression)),
                Value::Numeric(number),
            ) if expression.kind != "literal" => self.intern(numeric_wire_node(
                number,
                expression.domain,
                &expression.unit,
            )),
            (Some(wire::resource_node_proto::Node::Predicate(_)), Value::Boolean(value)) => self
                .intern(wire::ResourceNodeProto {
                    node: Some(wire::resource_node_proto::Node::Predicate(
                        wire::ResourcePredicateNodeProto {
                            kind: if value { "true" } else { "false" }.to_owned(),
                            arguments: Vec::new(),
                            predicates: Vec::new(),
                        },
                    )),
                }),
            _ => self.intern(node),
        }
    }

    fn normalize_structure(
        &mut self,
        mut node: wire::ResourceNodeProto,
    ) -> Result<NormalizedNode, ResourceExpressionSubstitutionError> {
        let Some(wire::resource_node_proto::Node::Expression(expression)) = node.node.as_mut()
        else {
            return Ok(NormalizedNode::Node(Box::new(node)));
        };
        let replacement = match expression.kind.as_str() {
            "add" | "multiply" => self.normalize_associative(expression)?,
            "minimum" | "maximum" => self.normalize_extremum(expression)?,
            "power" => self.normalize_power(expression)?,
            "piecewise" => self.normalize_piecewise(expression)?,
            _ => None,
        };
        Ok(replacement.map_or(
            NormalizedNode::Node(Box::new(node)),
            NormalizedNode::Existing,
        ))
    }

    fn normalize_associative(
        &mut self,
        node: &mut wire::ResourceExpressionNodeProto,
    ) -> Result<Option<u32>, ResourceExpressionSubstitutionError> {
        let kind = node.kind.as_str();
        let mut arguments = Vec::new();
        for identifier in std::mem::take(&mut node.arguments) {
            match self.expression_node(identifier) {
                Some(child) if child.kind == kind => arguments.extend_from_slice(&child.arguments),
                _ => arguments.push(identifier),
            }
        }
        let mut literals = Vec::new();
        let mut symbolic = Vec::new();
        for identifier in arguments {
            match self.values.get(identifier as usize) {
                Some(Value::Numeric(value)) => literals.push(value.clone()),
                Some(Value::Unknown) => symbolic.push(identifier),
                Some(Value::Boolean(_)) | None => {
                    return Err(ResourceExpressionSubstitutionError::Invalid(
                        "numeric expression contains a non-numeric operand".to_owned(),
                    ));
                }
            }
        }
        let identity = if kind == "add" {
            Numeric::integer(BigInt::zero())
        } else {
            Numeric::integer(BigInt::one())
        };
        let folded = literals.into_iter().try_fold(identity, |left, right| {
            if kind == "add" {
                left.add(right)
            } else {
                left.multiply(right)
            }
        });
        let folded = folded.map_err(substitution_evaluation_error)?;
        if kind == "multiply" && folded.is_zero() {
            return self
                .intern(numeric_wire_node(folded, node.domain, &node.unit))
                .map(Some);
        }
        if symbolic.is_empty() {
            return self
                .intern(numeric_wire_node(folded, node.domain, &node.unit))
                .map(Some);
        }
        if (kind == "add" && !folded.is_zero()) || (kind == "multiply" && !folded.is_one()) {
            let literal_unit = if kind == "add" { &node.unit } else { "1" };
            symbolic.push(self.intern(numeric_wire_node(folded, node.domain, literal_unit))?);
        }
        self.sort_identifiers(&mut symbolic)?;
        if symbolic.len() == 1
            && self
                .expression_node(symbolic[0])
                .is_some_and(|child| child.unit == node.unit)
        {
            return Ok(Some(symbolic[0]));
        }
        node.arguments = symbolic;
        Ok(None)
    }

    fn normalize_extremum(
        &self,
        node: &mut wire::ResourceExpressionNodeProto,
    ) -> Result<Option<u32>, ResourceExpressionSubstitutionError> {
        self.sort_identifiers(&mut node.arguments)?;
        node.arguments.dedup();
        Ok((node.arguments.len() == 1
            && self
                .expression_node(node.arguments[0])
                .is_some_and(|child| child.unit == node.unit))
        .then_some(node.arguments[0]))
    }

    fn normalize_power(
        &mut self,
        node: &wire::ResourceExpressionNodeProto,
    ) -> Result<Option<u32>, ResourceExpressionSubstitutionError> {
        let exponent = self.values.get(node.arguments[1] as usize).ok_or_else(|| {
            ResourceExpressionSubstitutionError::Invalid(
                "power exponent reference is invalid".to_owned(),
            )
        })?;
        let Value::Numeric(exponent) = exponent else {
            return Ok(None);
        };
        let Some(exponent) = exponent.as_integer() else {
            return Ok(None);
        };
        if exponent.is_zero() {
            return self
                .intern(numeric_wire_node(
                    Numeric::integer(BigInt::one()),
                    wire::ResourceDomainProto::ResourceNatural.into(),
                    &node.unit,
                ))
                .map(Some);
        }
        Ok((exponent.is_one()
            && self
                .expression_node(node.arguments[0])
                .is_some_and(|child| child.unit == node.unit))
        .then_some(node.arguments[0]))
    }

    fn normalize_piecewise(
        &self,
        node: &mut wire::ResourceExpressionNodeProto,
    ) -> Result<Option<u32>, ResourceExpressionSubstitutionError> {
        let mut retained_arguments = Vec::new();
        let mut retained_predicates = Vec::new();
        let mut otherwise = *node.arguments.last().ok_or_else(|| {
            ResourceExpressionSubstitutionError::Invalid(
                "piecewise expression has no default".to_owned(),
            )
        })?;
        for (predicate, argument) in node.predicates.iter().zip(&node.arguments) {
            match self.values.get(*predicate as usize) {
                Some(Value::Boolean(true)) => {
                    if retained_predicates.is_empty()
                        && self
                            .expression_node(*argument)
                            .is_some_and(|child| child.unit == node.unit)
                    {
                        return Ok(Some(*argument));
                    }
                    otherwise = *argument;
                    break;
                }
                Some(Value::Boolean(false)) => {}
                Some(Value::Unknown) => {
                    retained_predicates.push(*predicate);
                    retained_arguments.push(*argument);
                }
                Some(Value::Numeric(_)) | None => {
                    return Err(ResourceExpressionSubstitutionError::Invalid(
                        "piecewise expression contains a non-Boolean predicate".to_owned(),
                    ));
                }
            }
        }
        if retained_predicates.is_empty()
            && self
                .expression_node(otherwise)
                .is_some_and(|child| child.unit == node.unit)
        {
            return Ok(Some(otherwise));
        }
        if retained_predicates.is_empty() {
            return Ok(None);
        }
        retained_arguments.push(otherwise);
        node.arguments = retained_arguments;
        node.predicates = retained_predicates;
        Ok(None)
    }

    fn expression_node(&self, identifier: u32) -> Option<&wire::ResourceExpressionNodeProto> {
        match self.nodes.get(identifier as usize)?.node.as_ref()? {
            wire::resource_node_proto::Node::Expression(expression) => Some(expression),
            wire::resource_node_proto::Node::Predicate(_) => None,
        }
    }

    fn sort_identifiers(
        &self,
        identifiers: &mut [u32],
    ) -> Result<(), ResourceExpressionSubstitutionError> {
        for identifier in identifiers.iter() {
            if self.structural_keys.get(*identifier as usize).is_none() {
                return Err(ResourceExpressionSubstitutionError::Invalid(
                    "normalization lost an operand structural key".to_owned(),
                ));
            }
        }
        identifiers.sort_by(|left, right| {
            self.structural_keys[*left as usize].cmp(&self.structural_keys[*right as usize])
        });
        Ok(())
    }

    fn integer_literal(
        &mut self,
        value: i64,
        domain: i32,
        unit: &str,
    ) -> Result<u32, ResourceExpressionSubstitutionError> {
        self.intern(numeric_wire_node(
            Numeric::integer(BigInt::from(value)),
            domain,
            unit,
        ))
    }

    fn real_literal(
        &mut self,
        value: f64,
        domain: i32,
        unit: &str,
    ) -> Result<u32, ResourceExpressionSubstitutionError> {
        self.intern(numeric_wire_node(Numeric::Real(value), domain, unit))
    }

    fn operation(
        &mut self,
        kind: &str,
        domain: i32,
        unit: &str,
        arguments: Vec<u32>,
    ) -> Result<u32, ResourceExpressionSubstitutionError> {
        self.normalize(wire::ResourceNodeProto {
            node: Some(wire::resource_node_proto::Node::Expression(
                wire::ResourceExpressionNodeProto {
                    kind: kind.to_owned(),
                    domain,
                    unit: unit.to_owned(),
                    arguments,
                    ..Default::default()
                },
            )),
        })
    }

    fn retag_expression(
        &mut self,
        identifier: u32,
        domain: i32,
        unit: &str,
    ) -> Result<u32, ResourceExpressionSubstitutionError> {
        let mut node = self
            .nodes
            .get(identifier as usize)
            .cloned()
            .ok_or_else(|| {
                ResourceExpressionSubstitutionError::Invalid(
                    "coordinate conversion lost an expression".to_owned(),
                )
            })?;
        let Some(wire::resource_node_proto::Node::Expression(expression)) = node.node.as_mut()
        else {
            return Err(ResourceExpressionSubstitutionError::Invalid(
                "coordinate conversion received a predicate".to_owned(),
            ));
        };
        expression.domain = domain;
        unit.clone_into(&mut expression.unit);
        if let Some(symbol) = expression.symbol.as_mut() {
            symbol.domain = domain;
            unit.clone_into(&mut symbol.unit);
        }
        self.normalize(node)
    }

    fn encode_root(&self, root: u32) -> Result<Vec<u8>, ResourceExpressionSubstitutionError> {
        let (nodes, root) = canonical_reachable_graph(&self.nodes, root)?;
        let graph = wire::ResourceExpressionGraphProto { nodes, root };
        let payload = graph.encode_to_vec();
        if payload.len() > MAX_EXPRESSION_BYTES {
            return Err(ResourceExpressionSubstitutionError::Budget("wire-size"));
        }
        ExpressionGraph::decode_any(&graph)
            .map_err(|error| ResourceExpressionSubstitutionError::Invalid(error.to_string()))?;
        Ok(payload)
    }

    fn import_adapted_literal(
        &mut self,
        replacement: &PreparedGraph,
        target: &wire::ResourceNodeProto,
    ) -> Result<u32, ResourceExpressionSubstitutionError> {
        let mut literal = replacement.source.nodes[replacement.graph.root() as usize].clone();
        let Some(wire::resource_node_proto::Node::Expression(literal_node)) = literal.node.as_mut()
        else {
            return Err(ResourceExpressionSubstitutionError::Invalid(
                "adaptive binding root is not an expression".to_owned(),
            ));
        };
        let Some(wire::resource_node_proto::Node::Expression(target_node)) = target.node.as_ref()
        else {
            return Err(ResourceExpressionSubstitutionError::Invalid(
                "adaptive binding target is not an expression".to_owned(),
            ));
        };
        literal_node.domain = target_node.domain;
        literal_node.unit.clone_from(&target_node.unit);
        self.normalize(literal)
    }
}

fn wire_structural_key(
    node: &wire::ResourceNodeProto,
    keys: &[[u8; 32]],
) -> Result<[u8; 32], ResourceExpressionSubstitutionError> {
    let mut root = node.clone();
    let (arguments, predicates) = match root.node.as_mut() {
        Some(wire::resource_node_proto::Node::Expression(expression)) => (
            std::mem::take(&mut expression.arguments),
            std::mem::take(&mut expression.predicates),
        ),
        Some(wire::resource_node_proto::Node::Predicate(predicate)) => (
            std::mem::take(&mut predicate.arguments),
            std::mem::take(&mut predicate.predicates),
        ),
        None => {
            return Err(ResourceExpressionSubstitutionError::Invalid(
                "resource expression node has no constructor".to_owned(),
            ));
        }
    };
    let mut digest = Sha256::new();
    digest.update(root.encode_to_vec());
    append_structural_children(&mut digest, 0xa1, &arguments, keys)?;
    append_structural_children(&mut digest, 0xa2, &predicates, keys)?;
    Ok(digest.finalize().into())
}

fn append_structural_children(
    target: &mut Sha256,
    marker: u8,
    identifiers: &[u32],
    keys: &[[u8; 32]],
) -> Result<(), ResourceExpressionSubstitutionError> {
    let count = u64::try_from(identifiers.len())
        .map_err(|_| ResourceExpressionSubstitutionError::Budget("node-count"))?;
    target.update([marker]);
    target.update(count.to_le_bytes());
    for identifier in identifiers {
        let key = keys.get(*identifier as usize).ok_or_else(|| {
            ResourceExpressionSubstitutionError::Invalid(
                "structural key contains an invalid child reference".to_owned(),
            )
        })?;
        target.update(key);
    }
    Ok(())
}

fn canonical_reachable_graph(
    nodes: &[wire::ResourceNodeProto],
    root: u32,
) -> Result<(Vec<wire::ResourceNodeProto>, u32), ResourceExpressionSubstitutionError> {
    if nodes.get(root as usize).is_none() {
        return Err(ResourceExpressionSubstitutionError::Invalid(
            "normalization produced an invalid root".to_owned(),
        ));
    }
    let mut states = vec![0_u8; nodes.len()];
    let mut remapped = vec![None; nodes.len()];
    let mut result = Vec::new();
    let mut stack = vec![(root, false)];
    while let Some((identifier, exiting)) = stack.pop() {
        let index = identifier as usize;
        if exiting {
            let mut node = nodes[index].clone();
            remap_wire_references(&mut node, &remapped)?;
            let replacement = u32::try_from(result.len())
                .map_err(|_| ResourceExpressionSubstitutionError::Budget("node-count"))?;
            result.push(node);
            remapped[index] = Some(replacement);
            states[index] = 2;
            continue;
        }
        match states[index] {
            2 => continue,
            1 => {
                return Err(ResourceExpressionSubstitutionError::Invalid(
                    "normalization produced a cyclic graph".to_owned(),
                ));
            }
            _ => states[index] = 1,
        }
        stack.push((identifier, true));
        for child in wire_references(&nodes[index])?.into_iter().rev() {
            let child_index = child as usize;
            if child_index >= nodes.len() {
                return Err(ResourceExpressionSubstitutionError::Invalid(
                    "normalization produced an invalid child reference".to_owned(),
                ));
            }
            if states[child_index] == 0 {
                stack.push((child, false));
            } else if states[child_index] == 1 {
                return Err(ResourceExpressionSubstitutionError::Invalid(
                    "normalization produced a cyclic graph".to_owned(),
                ));
            }
        }
    }
    let root = remapped[root as usize].ok_or_else(|| {
        ResourceExpressionSubstitutionError::Invalid(
            "normalization failed to retain its root".to_owned(),
        )
    })?;
    Ok((result, root))
}

fn wire_references(
    node: &wire::ResourceNodeProto,
) -> Result<Vec<u32>, ResourceExpressionSubstitutionError> {
    match node.node.as_ref() {
        Some(wire::resource_node_proto::Node::Expression(expression)) => Ok(expression
            .arguments
            .iter()
            .chain(&expression.predicates)
            .copied()
            .collect()),
        Some(wire::resource_node_proto::Node::Predicate(predicate)) => Ok(predicate
            .arguments
            .iter()
            .chain(&predicate.predicates)
            .copied()
            .collect()),
        None => Err(ResourceExpressionSubstitutionError::Invalid(
            "normalization produced a node without a constructor".to_owned(),
        )),
    }
}

fn remap_wire_references(
    node: &mut wire::ResourceNodeProto,
    remapped: &[Option<u32>],
) -> Result<(), ResourceExpressionSubstitutionError> {
    let (arguments, predicates) = match node.node.as_mut() {
        Some(wire::resource_node_proto::Node::Expression(expression)) => {
            (&mut expression.arguments, &mut expression.predicates)
        }
        Some(wire::resource_node_proto::Node::Predicate(predicate)) => {
            (&mut predicate.arguments, &mut predicate.predicates)
        }
        None => {
            return Err(ResourceExpressionSubstitutionError::Invalid(
                "normalization produced a node without a constructor".to_owned(),
            ));
        }
    };
    for identifier in arguments.iter_mut().chain(predicates) {
        *identifier = remapped
            .get(*identifier as usize)
            .and_then(|value| *value)
            .ok_or_else(|| {
                ResourceExpressionSubstitutionError::Invalid(
                    "normalization emitted a parent before its child".to_owned(),
                )
            })?;
    }
    Ok(())
}

fn known_wire_value(
    node: &wire::ResourceNodeProto,
    values: &[Value],
) -> Result<Value, ResourceExpressionEvaluationError> {
    match node.node.as_ref() {
        Some(wire::resource_node_proto::Node::Expression(expression)) => {
            known_wire_expression(expression, values)
        }
        Some(wire::resource_node_proto::Node::Predicate(predicate)) => {
            known_wire_predicate(predicate, values)
        }
        None => invalid_value("resource expression node has no constructor"),
    }
}

fn known_wire_expression(
    node: &wire::ResourceExpressionNodeProto,
    values: &[Value],
) -> Result<Value, ResourceExpressionEvaluationError> {
    if node.kind == "literal" {
        return Ok(Value::Numeric(numeric_wire_literal(
            node.literal.as_ref().ok_or_else(|| {
                ResourceExpressionEvaluationError::Invalid("literal has no value".to_owned())
            })?,
        )?));
    }
    if node.kind == "symbol"
        || matches!(
            node.kind.as_str(),
            "bounded_sum" | "bounded_product" | "recurrence_equation" | "recurrence_system"
        )
    {
        return Ok(Value::Unknown);
    }
    if node.kind == "piecewise" {
        for (predicate, argument) in node.predicates.iter().zip(&node.arguments) {
            match wire_value(values, *predicate)? {
                Value::Boolean(true) => return Ok(wire_value(values, *argument)?.clone()),
                Value::Boolean(false) => {}
                Value::Unknown => return Ok(Value::Unknown),
                Value::Numeric(_) => return invalid_value("piecewise predicate is numeric"),
            }
        }
        return Ok(wire_value(
            values,
            *node.arguments.last().ok_or_else(|| {
                ResourceExpressionEvaluationError::Invalid(
                    "piecewise expression has no default".to_owned(),
                )
            })?,
        )?
        .clone());
    }
    let Some(arguments) = numeric_wire_arguments(&node.arguments, values)? else {
        return Ok(Value::Unknown);
    };
    let attributes = node
        .attributes
        .iter()
        .map(|item| (item.name.clone(), item.value.clone()))
        .collect::<Vec<_>>();
    Ok(
        evaluate_numeric_constructor(&node.kind, &attributes, arguments)?
            .map_or(Value::Unknown, Value::Numeric),
    )
}

fn known_wire_predicate(
    node: &wire::ResourcePredicateNodeProto,
    values: &[Value],
) -> Result<Value, ResourceExpressionEvaluationError> {
    match node.kind.as_str() {
        "true" => return Ok(Value::Boolean(true)),
        "false" => return Ok(Value::Boolean(false)),
        _ => {}
    }
    if !node.arguments.is_empty() {
        let Some(arguments) = numeric_wire_arguments(&node.arguments, values)? else {
            return Ok(Value::Unknown);
        };
        return known_comparison(&node.kind, &arguments);
    }
    let nested = node
        .predicates
        .iter()
        .map(|identifier| wire_value(values, *identifier))
        .collect::<Result<Vec<_>, _>>()?;
    match node.kind.as_str() {
        "not" => match nested[0] {
            Value::Boolean(value) => Ok(Value::Boolean(!value)),
            Value::Unknown => Ok(Value::Unknown),
            Value::Numeric(_) => invalid_value("negated predicate is numeric"),
        },
        "and"
            if nested
                .iter()
                .any(|value| matches!(value, Value::Boolean(false))) =>
        {
            Ok(Value::Boolean(false))
        }
        "and"
            if nested
                .iter()
                .all(|value| matches!(value, Value::Boolean(true))) =>
        {
            Ok(Value::Boolean(true))
        }
        "or" if nested
            .iter()
            .any(|value| matches!(value, Value::Boolean(true))) =>
        {
            Ok(Value::Boolean(true))
        }
        "or" if nested
            .iter()
            .all(|value| matches!(value, Value::Boolean(false))) =>
        {
            Ok(Value::Boolean(false))
        }
        "and" | "or" => Ok(Value::Unknown),
        kind => Err(ResourceExpressionEvaluationError::Invalid(format!(
            "unsupported predicate {kind:?}"
        ))),
    }
}

fn known_comparison(
    kind: &str,
    arguments: &[Numeric],
) -> Result<Value, ResourceExpressionEvaluationError> {
    let left = &arguments[0];
    let right = &arguments[1];
    if kind == "divisible" {
        let (Some(left), Some(right)) = (left.as_integer(), right.as_integer()) else {
            return Ok(Value::Unknown);
        };
        return Ok(if right.is_zero() {
            Value::Unknown
        } else {
            Value::Boolean((left % right).is_zero())
        });
    }
    let ordering = left.compare(right)?;
    let value = match kind {
        "equal" => ordering.is_eq(),
        "not_equal" => !ordering.is_eq(),
        "less" => ordering.is_lt(),
        "less_equal" => !ordering.is_gt(),
        "greater" => ordering.is_gt(),
        "greater_equal" => !ordering.is_lt(),
        _ => return invalid_value("unknown comparison constructor"),
    };
    Ok(Value::Boolean(value))
}

fn numeric_wire_arguments(
    identifiers: &[u32],
    values: &[Value],
) -> Result<Option<Vec<Numeric>>, ResourceExpressionEvaluationError> {
    let mut arguments = Vec::with_capacity(identifiers.len());
    for identifier in identifiers {
        match wire_value(values, *identifier)? {
            Value::Numeric(value) => arguments.push(value.clone()),
            Value::Unknown => return Ok(None),
            Value::Boolean(_) => return invalid_value("numeric operand is Boolean"),
        }
    }
    Ok(Some(arguments))
}

fn wire_value(
    values: &[Value],
    identifier: u32,
) -> Result<&Value, ResourceExpressionEvaluationError> {
    values.get(identifier as usize).ok_or_else(|| {
        ResourceExpressionEvaluationError::Invalid("wire value reference is invalid".to_owned())
    })
}

fn numeric_wire_literal(
    value: &wire::ResourceNumberProto,
) -> Result<Numeric, ResourceExpressionEvaluationError> {
    use wire::resource_number_proto::Value as WireValue;
    let result = match value.value.as_ref() {
        Some(WireValue::Integer(value)) => Numeric::integer(value.parse().map_err(|_| {
            ResourceExpressionEvaluationError::Invalid("malformed integer literal".to_owned())
        })?),
        Some(WireValue::Rational(value)) => Numeric::rational(
            value.numerator.parse().map_err(|_| {
                ResourceExpressionEvaluationError::Invalid(
                    "malformed rational numerator".to_owned(),
                )
            })?,
            value.denominator.parse().map_err(|_| {
                ResourceExpressionEvaluationError::Invalid(
                    "malformed rational denominator".to_owned(),
                )
            })?,
        ),
        Some(WireValue::RealBits(bits)) => Numeric::Real(f64::from_bits(*bits)),
        None => return invalid_value("literal has no numeric payload"),
    };
    result.checked()
}

fn numeric_wire_node(value: Numeric, domain: i32, unit: &str) -> wire::ResourceNodeProto {
    use wire::resource_number_proto::Value as WireValue;
    let value = match value {
        Numeric::Rational(numerator, denominator) if denominator.is_one() => {
            WireValue::Integer(numerator.to_string())
        }
        Numeric::Rational(numerator, denominator) => {
            WireValue::Rational(wire::ResourceRationalProto {
                numerator: numerator.to_string(),
                denominator: denominator.to_string(),
            })
        }
        Numeric::Real(value) => WireValue::RealBits(value.to_bits()),
    };
    wire::ResourceNodeProto {
        node: Some(wire::resource_node_proto::Node::Expression(
            wire::ResourceExpressionNodeProto {
                kind: "literal".to_owned(),
                domain,
                unit: unit.to_owned(),
                literal: Some(wire::ResourceNumberProto { value: Some(value) }),
                ..Default::default()
            },
        )),
    }
}

fn substitution_evaluation_error(
    error: ResourceExpressionEvaluationError,
) -> ResourceExpressionSubstitutionError {
    match error {
        ResourceExpressionEvaluationError::Budget(name) => {
            ResourceExpressionSubstitutionError::Budget(name)
        }
        error @ (ResourceExpressionEvaluationError::Invalid(_)
        | ResourceExpressionEvaluationError::Domain(_)
        | ResourceExpressionEvaluationError::DivisionByZero(_)) => {
            ResourceExpressionSubstitutionError::Invalid(error.to_string())
        }
    }
}

fn valid_qualified_identity(identity: &str) -> bool {
    let Some((scope, name)) = identity.split_once("::") else {
        return false;
    };
    !scope.is_empty() && !name.is_empty() && !name.contains("::")
}

#[derive(Clone, Debug)]
enum Value {
    Numeric(Numeric),
    Boolean(bool),
    Unknown,
}

#[derive(Clone, Debug)]
enum Numeric {
    Rational(BigInt, BigInt),
    Real(f64),
}

impl Numeric {
    fn integer(value: BigInt) -> Self {
        Self::Rational(value, BigInt::one())
    }

    fn rational(numerator: BigInt, denominator: BigInt) -> Self {
        debug_assert!(!denominator.is_zero());
        let (mut numerator, mut denominator) = if denominator.is_negative() {
            (-numerator, -denominator)
        } else {
            (numerator, denominator)
        };
        let divisor = gcd(numerator.clone(), denominator.clone());
        numerator /= &divisor;
        denominator /= divisor;
        Self::Rational(numerator, denominator)
    }

    fn into_public(self) -> ResourceScalar {
        match self {
            Self::Rational(numerator, denominator) if denominator.is_one() => {
                ResourceScalar::Integer(numerator)
            }
            Self::Rational(numerator, denominator) => ResourceScalar::Rational {
                numerator,
                denominator,
            },
            Self::Real(value) => ResourceScalar::Real(value),
        }
    }

    fn checked(self) -> Result<Self, ResourceExpressionEvaluationError> {
        let within_limit = match &self {
            Self::Rational(numerator, denominator) => {
                numerator.bits() <= MAX_INTEGER_BITS && denominator.bits() <= MAX_INTEGER_BITS
            }
            Self::Real(value) => value.is_finite(),
        };
        if within_limit {
            Ok(self)
        } else {
            Err(ResourceExpressionEvaluationError::Budget("numeric-size"))
        }
    }

    fn as_integer(&self) -> Option<&BigInt> {
        match self {
            Self::Rational(value, denominator) if denominator.is_one() => Some(value),
            Self::Rational(_, _) | Self::Real(_) => None,
        }
    }

    fn is_zero(&self) -> bool {
        match self {
            Self::Rational(value, _) => value.is_zero(),
            Self::Real(value) => value.to_bits() << 1 == 0,
        }
    }

    fn is_one(&self) -> bool {
        match self {
            Self::Rational(value, denominator) => value == denominator,
            Self::Real(value) => value.to_bits() == 1.0_f64.to_bits(),
        }
    }

    fn to_real(&self) -> Result<f64, ResourceExpressionEvaluationError> {
        let value = match self {
            Self::Rational(numerator, denominator) => {
                let numerator = numerator.to_f64().ok_or_else(|| {
                    ResourceExpressionEvaluationError::Domain(
                        "exact value is too large for a real operation".to_owned(),
                    )
                })?;
                let denominator = denominator.to_f64().ok_or_else(|| {
                    ResourceExpressionEvaluationError::Domain(
                        "exact value is too large for a real operation".to_owned(),
                    )
                })?;
                numerator / denominator
            }
            Self::Real(value) => *value,
        };
        finite(value)
    }

    fn add(self, other: Self) -> Result<Self, ResourceExpressionEvaluationError> {
        match (self, other) {
            (Self::Rational(left_n, left_d), Self::Rational(right_n, right_d)) => {
                Self::rational(left_n * &right_d + right_n * &left_d, left_d * right_d).checked()
            }
            (left, right) => Ok(Self::Real(finite(left.to_real()? + right.to_real()?)?)),
        }
    }

    fn multiply(self, other: Self) -> Result<Self, ResourceExpressionEvaluationError> {
        match (self, other) {
            (Self::Rational(left_n, left_d), Self::Rational(right_n, right_d)) => {
                Self::rational(left_n * right_n, left_d * right_d).checked()
            }
            (left, right) => Ok(Self::Real(finite(left.to_real()? * right.to_real()?)?)),
        }
    }

    fn compare(
        &self,
        other: &Self,
    ) -> Result<std::cmp::Ordering, ResourceExpressionEvaluationError> {
        match (self, other) {
            (Self::Rational(left_n, left_d), Self::Rational(right_n, right_d)) => {
                Ok((left_n * right_d).cmp(&(right_n * left_d)))
            }
            _ => self
                .to_real()?
                .partial_cmp(&other.to_real()?)
                .ok_or_else(|| {
                    ResourceExpressionEvaluationError::Domain("unordered real value".to_owned())
                }),
        }
    }

    fn floor(&self) -> Result<BigInt, ResourceExpressionEvaluationError> {
        match self {
            Self::Rational(numerator, denominator) => Ok(floor_ratio(numerator, denominator)),
            Self::Real(value) => BigInt::from_f64(value.floor()).ok_or_else(|| {
                ResourceExpressionEvaluationError::Domain("non-finite floor".to_owned())
            }),
        }
    }

    fn ceiling(&self) -> Result<BigInt, ResourceExpressionEvaluationError> {
        match self {
            Self::Rational(numerator, denominator) => Ok(-floor_ratio(&-numerator, denominator)),
            Self::Real(value) => BigInt::from_f64(value.ceil()).ok_or_else(|| {
                ResourceExpressionEvaluationError::Domain("non-finite ceiling".to_owned())
            }),
        }
    }
}

struct RecurrenceInvocation {
    environment: BTreeMap<String, Numeric>,
    body: u32,
    condition: Option<u32>,
}

struct Evaluator<'a> {
    graph: &'a ExpressionGraph,
    steps: usize,
    recurrence: Option<RecurrenceRuntime>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct RecurrenceCall {
    equation: u32,
    arguments: Vec<BigInt>,
}

#[derive(Default)]
struct RecurrenceRuntime {
    definitions: BTreeMap<String, u32>,
    values: HashMap<RecurrenceCall, Value>,
    pending: Vec<RecurrenceCall>,
    storage_bytes: usize,
}

impl Evaluator<'_> {
    fn evaluate(
        &mut self,
        identifier: u32,
        environment: &BTreeMap<String, Numeric>,
        depth: usize,
        memo: &mut HashMap<u32, Value>,
    ) -> Result<Value, ResourceExpressionEvaluationError> {
        self.steps = self
            .steps
            .checked_add(1)
            .ok_or(ResourceExpressionEvaluationError::Budget("operation-count"))?;
        if self.steps > MAX_EVALUATION_STEPS {
            return Err(ResourceExpressionEvaluationError::Budget("operation-count"));
        }
        if depth > MAX_EVALUATION_DEPTH {
            return Err(ResourceExpressionEvaluationError::Budget("nesting-depth"));
        }
        if let Some(value) = memo.get(&identifier) {
            return Ok(value.clone());
        }
        let value = match self.graph.node(identifier).ok_or_else(|| {
            ResourceExpressionEvaluationError::Invalid("evaluation lost a node".to_owned())
        })? {
            ExpressionGraphNode::Expression(node) => {
                if node.kind == "literal" {
                    Value::Numeric(number(node.literal.as_ref().ok_or_else(|| {
                        ResourceExpressionEvaluationError::Invalid(
                            "literal has no value".to_owned(),
                        )
                    })?)?)
                } else if node.kind == "symbol" {
                    let symbol = node.symbol.as_ref().ok_or_else(|| {
                        ResourceExpressionEvaluationError::Invalid(
                            "symbol has no declaration".to_owned(),
                        )
                    })?;
                    environment
                        .get(&format!("{}::{}", symbol.scope, symbol.name))
                        .cloned()
                        .map_or(Value::Unknown, Value::Numeric)
                } else {
                    self.evaluate_expression(node, environment, depth + 1, memo)?
                }
            }
            ExpressionGraphNode::Predicate(node) => {
                self.evaluate_predicate(node, environment, depth + 1, memo)?
            }
        };
        memo.insert(identifier, value.clone());
        Ok(value)
    }

    fn evaluate_expression(
        &mut self,
        node: &crate::resource_rules::ExpressionNode,
        environment: &BTreeMap<String, Numeric>,
        depth: usize,
        memo: &mut HashMap<u32, Value>,
    ) -> Result<Value, ResourceExpressionEvaluationError> {
        if node.kind == "recurrence_system" {
            return self.evaluate_recurrence_system(node, environment, depth);
        }
        if node.kind == "recurrence_equation" {
            return Ok(Value::Unknown);
        }
        if matches!(node.kind.as_str(), "bounded_sum" | "bounded_product") {
            return self.evaluate_bounded(node, environment, depth, memo);
        }
        if node.kind == "piecewise" {
            for (predicate, argument) in node.predicates.iter().zip(&node.arguments) {
                match self.evaluate(*predicate, environment, depth, memo)? {
                    Value::Boolean(true) => {
                        return self.evaluate(*argument, environment, depth, memo);
                    }
                    Value::Boolean(false) => {}
                    Value::Unknown => return Ok(Value::Unknown),
                    Value::Numeric(_) => return invalid_value("predicate reduced to a number"),
                }
            }
            return self.evaluate(
                *node.arguments.last().ok_or_else(|| {
                    ResourceExpressionEvaluationError::Invalid(
                        "piecewise expression has no default".to_owned(),
                    )
                })?,
                environment,
                depth,
                memo,
            );
        }
        let mut arguments = Vec::with_capacity(node.arguments.len());
        for identifier in &node.arguments {
            match self.evaluate(*identifier, environment, depth, memo)? {
                Value::Numeric(value) => arguments.push(value),
                Value::Unknown => return Ok(Value::Unknown),
                Value::Boolean(_) => return invalid_value("numeric operand reduced to Boolean"),
            }
        }
        if node.kind == "recurrence" {
            return Ok(self.evaluate_recurrence_call(node, &arguments));
        }
        evaluate_numeric_constructor(&node.kind, &node.attributes, arguments)
            .map(|value| value.map_or(Value::Unknown, Value::Numeric))
    }

    fn evaluate_recurrence_call(
        &mut self,
        node: &crate::resource_rules::ExpressionNode,
        arguments: &[Numeric],
    ) -> Value {
        let Some(runtime) = self.recurrence.as_mut() else {
            return Value::Unknown;
        };
        let name = node
            .attributes
            .iter()
            .find(|(name, _)| name == "name")
            .map(|(_, value)| value);
        let Some(equation) = name.and_then(|name| runtime.definitions.get(name)).copied() else {
            return Value::Unknown;
        };
        let Some(arguments) = arguments
            .iter()
            .map(|value| value.as_integer().cloned())
            .collect::<Option<Vec<_>>>()
        else {
            return Value::Unknown;
        };
        let definition = self
            .graph
            .expression(equation)
            .expect("checked recurrence equation");
        for (parameter, argument) in definition.arguments.iter().zip(&arguments) {
            let parameter = self
                .graph
                .expression(*parameter)
                .expect("checked recurrence parameter");
            if parameter.domain == ResourceDomain::Natural && argument.is_negative() {
                return Value::Unknown;
            }
        }
        let call = RecurrenceCall {
            equation,
            arguments,
        };
        if let Some(value) = runtime.values.get(&call) {
            return value.clone();
        }
        runtime.pending.push(call);
        Value::Unknown
    }

    /// Evaluate a partial recurrence by memoized, iterative dependency traversal.
    /// Repeated active calls and finite work limits preserve the unresolved expression.
    fn evaluate_recurrence_system(
        &mut self,
        node: &crate::resource_rules::ExpressionNode,
        outer: &BTreeMap<String, Numeric>,
        depth: usize,
    ) -> Result<Value, ResourceExpressionEvaluationError> {
        if self.recurrence.is_some() {
            return Ok(Value::Unknown);
        }
        let mut runtime = RecurrenceRuntime::default();
        for identifier in node.arguments.iter().skip(1) {
            let equation = self
                .graph
                .expression(*identifier)
                .map_err(|error| ResourceExpressionEvaluationError::Invalid(error.to_string()))?;
            let name = equation
                .attributes
                .iter()
                .find(|(name, _)| name == "name")
                .ok_or_else(|| {
                    ResourceExpressionEvaluationError::Invalid(
                        "recurrence equation has no identity".to_owned(),
                    )
                })?;
            runtime.definitions.insert(name.1.clone(), *identifier);
        }
        self.recurrence = Some(runtime);
        let result = match self.solve_recurrence(node.arguments[0], outer, depth) {
            Err(
                ResourceExpressionEvaluationError::Budget(_)
                | ResourceExpressionEvaluationError::Domain(_)
                | ResourceExpressionEvaluationError::DivisionByZero(_),
            ) => Ok(Value::Unknown),
            result => result,
        };
        self.recurrence = None;
        result
    }

    fn solve_recurrence(
        &mut self,
        root: u32,
        outer: &BTreeMap<String, Numeric>,
        depth: usize,
    ) -> Result<Value, ResourceExpressionEvaluationError> {
        const MAX_RECURRENCE_CALLS: usize = 100_000;
        let mut jobs: Vec<RecurrenceCall> = Vec::new();
        let mut active = HashSet::new();
        let initial_steps = self.steps;
        loop {
            if self.steps - initial_steps >= MAX_EVALUATION_STEPS / 2 {
                return Ok(Value::Unknown);
            }
            if self
                .recurrence
                .as_ref()
                .expect("active system")
                .values
                .len()
                + jobs.len()
                > MAX_RECURRENCE_CALLS
            {
                return Ok(Value::Unknown);
            }
            let invocation = if let Some(call) = jobs.last() {
                self.bind_recurrence_parameters(call, outer)?
            } else {
                RecurrenceInvocation {
                    environment: outer.clone(),
                    body: root,
                    condition: None,
                }
            };
            self.recurrence
                .as_mut()
                .expect("active system")
                .pending
                .clear();
            let mut memo = HashMap::new();
            let value = if let Some(condition) = invocation.condition {
                match self.evaluate(condition, &invocation.environment, depth, &mut memo)? {
                    Value::Boolean(true) => {
                        self.evaluate(invocation.body, &invocation.environment, depth, &mut memo)?
                    }
                    _ => Value::Unknown,
                }
            } else {
                self.evaluate(invocation.body, &invocation.environment, depth, &mut memo)?
            };
            let pending =
                std::mem::take(&mut self.recurrence.as_mut().expect("active system").pending);
            if let Some(child) = pending.into_iter().next() {
                let bytes = child
                    .arguments
                    .iter()
                    .try_fold(64_usize, |total, argument| {
                        total
                            .checked_add(32)?
                            .checked_add(usize::try_from(argument.bits().div_ceil(8)).ok()?)
                    })
                    .and_then(|bytes| bytes.checked_mul(4));
                if !self.charge_recurrence_storage(bytes) {
                    return Ok(Value::Unknown);
                }
                if !active.insert(child.clone()) {
                    return Ok(Value::Unknown);
                }
                jobs.push(child);
                continue;
            }
            let Some(call) = jobs.pop() else {
                return Ok(value);
            };
            active.remove(&call);
            let value_bytes = match &value {
                Value::Numeric(Numeric::Rational(numerator, denominator)) => numerator
                    .bits()
                    .checked_add(denominator.bits())
                    .and_then(|bits| usize::try_from(bits.div_ceil(8)).ok())
                    .and_then(|bytes| bytes.checked_add(64)),
                _ => Some(16),
            };
            if !self.charge_recurrence_storage(value_bytes) {
                return Ok(Value::Unknown);
            }
            self.recurrence
                .as_mut()
                .expect("active system")
                .values
                .insert(call, value);
        }
    }

    fn bind_recurrence_parameters(
        &self,
        call: &RecurrenceCall,
        outer: &BTreeMap<String, Numeric>,
    ) -> Result<RecurrenceInvocation, ResourceExpressionEvaluationError> {
        let equation = self
            .graph
            .expression(call.equation)
            .map_err(|error| ResourceExpressionEvaluationError::Invalid(error.to_string()))?;
        let mut environment = outer.clone();
        for (parameter, value) in equation.arguments.iter().zip(&call.arguments) {
            let declaration = self
                .graph
                .expression(*parameter)
                .map_err(|error| ResourceExpressionEvaluationError::Invalid(error.to_string()))?;
            let symbol = declaration
                .symbol
                .as_ref()
                .expect("checked recurrence parameter");
            environment.insert(
                format!("{}::{}", symbol.scope, symbol.name),
                Numeric::integer(value.clone()),
            );
        }
        Ok(RecurrenceInvocation {
            environment,
            body: *equation.arguments.last().expect("checked equation body"),
            condition: Some(equation.predicates[0]),
        })
    }

    fn charge_recurrence_storage(&mut self, bytes: Option<usize>) -> bool {
        let runtime = self.recurrence.as_mut().expect("active system");
        let Some(total) = bytes.and_then(|bytes| runtime.storage_bytes.checked_add(bytes)) else {
            return false;
        };
        if total > MAX_EXPRESSION_BYTES {
            return false;
        }
        runtime.storage_bytes = total;
        true
    }

    fn evaluate_bounded(
        &mut self,
        node: &crate::resource_rules::ExpressionNode,
        environment: &BTreeMap<String, Numeric>,
        depth: usize,
        memo: &mut HashMap<u32, Value>,
    ) -> Result<Value, ResourceExpressionEvaluationError> {
        let lower = match self.evaluate(node.arguments[0], environment, depth, memo)? {
            Value::Numeric(value) => value,
            Value::Unknown => return Ok(Value::Unknown),
            Value::Boolean(_) => return invalid_value("bounded lower limit is Boolean"),
        };
        let upper = match self.evaluate(node.arguments[1], environment, depth, memo)? {
            Value::Numeric(value) => value,
            Value::Unknown => return Ok(Value::Unknown),
            Value::Boolean(_) => return invalid_value("bounded upper limit is Boolean"),
        };
        let lower = lower.as_integer().ok_or_else(|| {
            ResourceExpressionEvaluationError::Domain(
                "bounded lower limit is not integral".to_owned(),
            )
        })?;
        let upper = upper.as_integer().ok_or_else(|| {
            ResourceExpressionEvaluationError::Domain(
                "bounded upper limit is not integral".to_owned(),
            )
        })?;
        if upper < lower {
            return Ok(Value::Numeric(Numeric::integer(
                if node.kind == "bounded_sum" {
                    BigInt::zero()
                } else {
                    BigInt::one()
                },
            )));
        }
        let count = (upper - lower + BigInt::one()).to_usize().ok_or(
            ResourceExpressionEvaluationError::Budget("bounded-term-count"),
        )?;
        if count > MAX_BOUNDED_TERMS {
            return Ok(Value::Unknown);
        }
        let attributes = node.attributes.iter().cloned().collect::<BTreeMap<_, _>>();
        let scope = attributes.get("binder_scope").ok_or_else(|| {
            ResourceExpressionEvaluationError::Invalid(
                "bounded expression has no binder scope".to_owned(),
            )
        })?;
        let name = attributes.get("binder_name").ok_or_else(|| {
            ResourceExpressionEvaluationError::Invalid(
                "bounded expression has no binder name".to_owned(),
            )
        })?;
        let identity = format!("{scope}::{name}");
        let mut active = environment.clone();
        let mut total = Numeric::integer(if node.kind == "bounded_sum" {
            BigInt::zero()
        } else {
            BigInt::one()
        });
        let mut index = lower.clone();
        for _ in 0..count {
            active.insert(identity.clone(), Numeric::integer(index.clone()));
            let mut iteration_memo = HashMap::new();
            let value =
                match self.evaluate(node.arguments[2], &active, depth, &mut iteration_memo)? {
                    Value::Numeric(value) => value,
                    Value::Unknown => return Ok(Value::Unknown),
                    Value::Boolean(_) => return invalid_value("bounded body is Boolean"),
                };
            total = if node.kind == "bounded_sum" {
                total.add(value)?
            } else {
                total.multiply(value)?
            };
            index += 1;
        }
        Ok(Value::Numeric(total))
    }

    fn evaluate_predicate(
        &mut self,
        node: &crate::resource_rules::PredicateNode,
        environment: &BTreeMap<String, Numeric>,
        depth: usize,
        memo: &mut HashMap<u32, Value>,
    ) -> Result<Value, ResourceExpressionEvaluationError> {
        match node.kind.as_str() {
            "true" => return Ok(Value::Boolean(true)),
            "false" => return Ok(Value::Boolean(false)),
            _ => {}
        }
        if matches!(
            node.kind.as_str(),
            "equal"
                | "not_equal"
                | "less"
                | "less_equal"
                | "greater"
                | "greater_equal"
                | "divisible"
        ) {
            let left = match self.evaluate(node.arguments[0], environment, depth, memo)? {
                Value::Numeric(value) => value,
                Value::Unknown => return Ok(Value::Unknown),
                Value::Boolean(_) => return invalid_value("comparison operand is Boolean"),
            };
            let right = match self.evaluate(node.arguments[1], environment, depth, memo)? {
                Value::Numeric(value) => value,
                Value::Unknown => return Ok(Value::Unknown),
                Value::Boolean(_) => return invalid_value("comparison operand is Boolean"),
            };
            let value = if node.kind == "divisible" {
                let (Some(left), Some(right)) = (left.as_integer(), right.as_integer()) else {
                    return Ok(Value::Unknown);
                };
                if right.is_zero() {
                    return Ok(Value::Unknown);
                }
                (left % right).is_zero()
            } else {
                let ordering = left.compare(&right)?;
                match node.kind.as_str() {
                    "equal" => ordering.is_eq(),
                    "not_equal" => !ordering.is_eq(),
                    "less" => ordering.is_lt(),
                    "less_equal" => !ordering.is_gt(),
                    "greater" => ordering.is_gt(),
                    "greater_equal" => !ordering.is_lt(),
                    _ => unreachable!(),
                }
            };
            return Ok(Value::Boolean(value));
        }
        let mut unknown = false;
        match node.kind.as_str() {
            "not" => match self.evaluate(node.predicates[0], environment, depth, memo)? {
                Value::Boolean(value) => Ok(Value::Boolean(!value)),
                Value::Unknown => Ok(Value::Unknown),
                Value::Numeric(_) => invalid_value("negated predicate is numeric"),
            },
            "and" => {
                for predicate in &node.predicates {
                    match self.evaluate(*predicate, environment, depth, memo)? {
                        Value::Boolean(false) => return Ok(Value::Boolean(false)),
                        Value::Boolean(true) => {}
                        Value::Unknown => unknown = true,
                        Value::Numeric(_) => return invalid_value("predicate operand is numeric"),
                    }
                }
                Ok(if unknown {
                    Value::Unknown
                } else {
                    Value::Boolean(true)
                })
            }
            "or" => {
                for predicate in &node.predicates {
                    match self.evaluate(*predicate, environment, depth, memo)? {
                        Value::Boolean(true) => return Ok(Value::Boolean(true)),
                        Value::Boolean(false) => {}
                        Value::Unknown => unknown = true,
                        Value::Numeric(_) => return invalid_value("predicate operand is numeric"),
                    }
                }
                Ok(if unknown {
                    Value::Unknown
                } else {
                    Value::Boolean(false)
                })
            }
            kind => Err(ResourceExpressionEvaluationError::Invalid(format!(
                "unsupported predicate {kind:?}"
            ))),
        }
    }
}

fn evaluate_numeric_constructor(
    kind: &str,
    attributes: &[(String, String)],
    arguments: Vec<Numeric>,
) -> Result<Option<Numeric>, ResourceExpressionEvaluationError> {
    let result = match kind {
        "add" => arguments
            .into_iter()
            .try_fold(Numeric::integer(BigInt::zero()), Numeric::add)?,
        "multiply" => arguments
            .into_iter()
            .try_fold(Numeric::integer(BigInt::one()), Numeric::multiply)?,
        "power" => power(&arguments[0], &arguments[1])?,
        "absolute" => match &arguments[0] {
            Numeric::Rational(numerator, denominator) => {
                Numeric::Rational(numerator.abs(), denominator.clone())
            }
            Numeric::Real(value) => Numeric::Real(value.abs()),
        },
        "minimum" | "maximum" => extremum(&arguments, kind == "maximum")?,
        "floor" => Numeric::integer(arguments[0].floor()?),
        "ceiling" => Numeric::integer(arguments[0].ceiling()?),
        "floor_divide" | "ceiling_divide" => {
            divide_to_integer(&arguments[0], &arguments[1], kind == "ceiling_divide")?
        }
        "exp" => Numeric::Real(finite(arguments[0].to_real()?.exp())?),
        "log" => logarithm(&arguments)?,
        "log2" => {
            let operand = positive_real(&arguments[0], "log2 operand must be positive")?;
            Numeric::Real(finite(operand.log2())?)
        }
        "square_root" => {
            let operand = arguments[0].to_real()?;
            if operand < 0.0 {
                return Err(ResourceExpressionEvaluationError::Domain(
                    "square-root operand must be nonnegative".to_owned(),
                ));
            }
            Numeric::Real(finite(operand.sqrt())?)
        }
        "contracted_function" => return evaluate_contracted(attributes, &arguments),
        "recurrence" => return Ok(None),
        kind => {
            return Err(ResourceExpressionEvaluationError::Invalid(format!(
                "unsupported constructor {kind:?}"
            )));
        }
    };
    Ok(Some(result))
}

fn extremum(
    arguments: &[Numeric],
    maximum: bool,
) -> Result<Numeric, ResourceExpressionEvaluationError> {
    let mut result = arguments[0].clone();
    for argument in &arguments[1..] {
        let ordering = result.compare(argument)?;
        if (maximum && ordering.is_lt()) || (!maximum && ordering.is_gt()) {
            result = argument.clone();
        }
    }
    Ok(result)
}

fn logarithm(arguments: &[Numeric]) -> Result<Numeric, ResourceExpressionEvaluationError> {
    let operand = positive_real(&arguments[0], "logarithm operand must be positive")?;
    let value = if arguments.len() == 1 {
        operand.ln()
    } else {
        let base = arguments[1].to_real()?;
        if base <= 0.0 || base.to_bits() == 1.0_f64.to_bits() {
            return Err(ResourceExpressionEvaluationError::Domain(
                "logarithm base must be positive and not one".to_owned(),
            ));
        }
        operand.log(base)
    };
    Ok(Numeric::Real(finite(value)?))
}

fn positive_real(value: &Numeric, message: &str) -> Result<f64, ResourceExpressionEvaluationError> {
    let value = value.to_real()?;
    if value <= 0.0 {
        return Err(ResourceExpressionEvaluationError::Domain(
            message.to_owned(),
        ));
    }
    Ok(value)
}

fn evaluate_contracted(
    attributes: &[(String, String)],
    arguments: &[Numeric],
) -> Result<Option<Numeric>, ResourceExpressionEvaluationError> {
    let name = attributes
        .iter()
        .find_map(|(key, value)| (key == "name").then_some(value.as_str()));
    match name {
        Some("qamomile.range_trip_count") => {
            let [range_start, range_stop, range_stride] = integer_arguments(arguments)?;
            if range_stride.is_zero() {
                return Err(ResourceExpressionEvaluationError::Domain(
                    "range trip count step cannot be zero".to_owned(),
                ));
            }
            Ok(Some(Numeric::integer(range_length(
                range_start,
                range_stop,
                range_stride,
            ))))
        }
        Some("qamomile.floor_divide") => {
            let [left, right] = integer_arguments(arguments)?;
            if right.is_zero() {
                return Err(ResourceExpressionEvaluationError::DivisionByZero(
                    "floor division".to_owned(),
                ));
            }
            Ok(Some(Numeric::integer(floor_ratio(left, right))))
        }
        Some("qamomile.positive_dimension") => {
            let [value] = integer_arguments(arguments)?;
            if value.is_positive() {
                Ok(Some(Numeric::integer(value.clone())))
            } else {
                Err(ResourceExpressionEvaluationError::Domain(
                    "a shape dimension must be positive".to_owned(),
                ))
            }
        }
        Some("qamomile.modulo") => {
            let [left, right] = integer_arguments(arguments)?;
            if right.is_zero() {
                return Err(ResourceExpressionEvaluationError::DivisionByZero(
                    "modulo".to_owned(),
                ));
            }
            let quotient = floor_ratio(left, right);
            Ok(Some(Numeric::integer(left - quotient * right)))
        }
        _ => Ok(None),
    }
}

fn number(value: &Number) -> Result<Numeric, ResourceExpressionEvaluationError> {
    let number = match value {
        Number::Integer(value) => Numeric::integer(value.clone()),
        Number::Rational(numerator, denominator) => {
            Numeric::Rational(numerator.clone(), denominator.clone())
        }
        Number::RealBits(bits) => Numeric::Real(f64::from_bits(*bits)),
    };
    number.checked()
}

fn power(base: &Numeric, exponent: &Numeric) -> Result<Numeric, ResourceExpressionEvaluationError> {
    if let Some(exponent) = exponent.as_integer() {
        let negative = exponent.is_negative();
        let magnitude = exponent
            .abs()
            .to_u32()
            .ok_or(ResourceExpressionEvaluationError::Budget("power-exponent"))?;
        if magnitude > MAX_POWER_EXPONENT {
            return Err(ResourceExpressionEvaluationError::Budget("power-exponent"));
        }
        return match base {
            Numeric::Rational(numerator, denominator) => {
                if negative && numerator.is_zero() {
                    return Err(ResourceExpressionEvaluationError::DivisionByZero(
                        "negative power of zero".to_owned(),
                    ));
                }
                if [numerator, denominator].iter().any(|value| {
                    value.bits().saturating_mul(u64::from(magnitude)) > MAX_INTEGER_BITS
                }) {
                    return Err(ResourceExpressionEvaluationError::Budget("numeric-size"));
                }
                let numerator_power = numerator.pow(magnitude);
                let denominator_power = denominator.pow(magnitude);
                (if negative {
                    Numeric::rational(denominator_power, numerator_power)
                } else {
                    Numeric::rational(numerator_power, denominator_power)
                })
                .checked()
            }
            Numeric::Real(value) => {
                let exponent = exponent
                    .to_f64()
                    .ok_or(ResourceExpressionEvaluationError::Budget("power-exponent"))?;
                Ok(Numeric::Real(finite(value.powf(exponent))?))
            }
        };
    }
    let base = base.to_real()?;
    let exponent = exponent.to_real()?;
    Ok(Numeric::Real(finite(base.powf(exponent))?))
}

fn divide_to_integer(
    numerator: &Numeric,
    denominator: &Numeric,
    ceiling: bool,
) -> Result<Numeric, ResourceExpressionEvaluationError> {
    if let (Numeric::Rational(left_n, left_d), Numeric::Rational(right_n, right_d)) =
        (numerator, denominator)
    {
        if right_n.is_zero() {
            return Err(ResourceExpressionEvaluationError::DivisionByZero(
                "integer division".to_owned(),
            ));
        }
        let quotient_n = left_n * right_d;
        let quotient_d = left_d * right_n;
        let value = if ceiling {
            -floor_ratio(&-quotient_n, &quotient_d)
        } else {
            floor_ratio(&quotient_n, &quotient_d)
        };
        return Ok(Numeric::integer(value));
    }
    let denominator = denominator.to_real()?;
    if denominator.is_zero() {
        return Err(ResourceExpressionEvaluationError::DivisionByZero(
            "integer division".to_owned(),
        ));
    }
    let quotient = numerator.to_real()? / denominator;
    let value = if ceiling {
        quotient.ceil()
    } else {
        quotient.floor()
    };
    Ok(Numeric::integer(BigInt::from_f64(value).ok_or_else(
        || {
            ResourceExpressionEvaluationError::Domain(
                "integer division produced a non-finite value".to_owned(),
            )
        },
    )?))
}

fn integer_arguments<const N: usize>(
    arguments: &[Numeric],
) -> Result<[&BigInt; N], ResourceExpressionEvaluationError> {
    let values = arguments
        .iter()
        .map(|value| {
            value.as_integer().ok_or_else(|| {
                ResourceExpressionEvaluationError::Domain(
                    "contracted function requires integral operands".to_owned(),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    values.try_into().map_err(|_| {
        ResourceExpressionEvaluationError::Invalid(
            "contracted function has the wrong operand count".to_owned(),
        )
    })
}

fn range_length(range_start: &BigInt, range_stop: &BigInt, range_stride: &BigInt) -> BigInt {
    if (range_stride.is_positive() && range_start >= range_stop)
        || (range_stride.is_negative() && range_start <= range_stop)
    {
        return BigInt::zero();
    }
    let distance = if range_stride.is_positive() {
        range_stop - range_start
    } else {
        range_start - range_stop
    };
    let magnitude = range_stride.abs();
    (distance + &magnitude - 1) / magnitude
}

fn floor_ratio(numerator: &BigInt, denominator: &BigInt) -> BigInt {
    debug_assert!(!denominator.is_zero());
    let mut quotient = numerator / denominator;
    let remainder = numerator % denominator;
    if !remainder.is_zero() && numerator.is_negative() != denominator.is_negative() {
        quotient -= 1;
    }
    quotient
}

fn gcd(mut left: BigInt, mut right: BigInt) -> BigInt {
    left = left.abs();
    right = right.abs();
    while !right.is_zero() {
        let remainder = left % &right;
        left = right;
        right = remainder;
    }
    if left.is_zero() { BigInt::one() } else { left }
}

fn finite(value: f64) -> Result<f64, ResourceExpressionEvaluationError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ResourceExpressionEvaluationError::Domain(
            "real operation produced NaN or infinity".to_owned(),
        ))
    }
}

fn invalid_value<T>(message: &str) -> Result<T, ResourceExpressionEvaluationError> {
    Err(ResourceExpressionEvaluationError::Invalid(
        message.to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integer(value: &str) -> wire::ResourceNodeProto {
        wire::ResourceNodeProto {
            node: Some(wire::resource_node_proto::Node::Expression(
                wire::ResourceExpressionNodeProto {
                    kind: "literal".to_owned(),
                    domain: wire::ResourceDomainProto::ResourceInteger.into(),
                    unit: "1".to_owned(),
                    literal: Some(wire::ResourceNumberProto {
                        value: Some(wire::resource_number_proto::Value::Integer(
                            value.to_owned(),
                        )),
                    }),
                    ..Default::default()
                },
            )),
        }
    }

    fn natural_with_unit(value: &str, unit: &str) -> wire::ResourceNodeProto {
        wire::ResourceNodeProto {
            node: Some(wire::resource_node_proto::Node::Expression(
                wire::ResourceExpressionNodeProto {
                    kind: "literal".to_owned(),
                    domain: wire::ResourceDomainProto::ResourceNatural.into(),
                    unit: unit.to_owned(),
                    literal: Some(wire::ResourceNumberProto {
                        value: Some(wire::resource_number_proto::Value::Integer(
                            value.to_owned(),
                        )),
                    }),
                    ..Default::default()
                },
            )),
        }
    }

    fn real_with_unit(value: f64, unit: &str) -> wire::ResourceNodeProto {
        wire::ResourceNodeProto {
            node: Some(wire::resource_node_proto::Node::Expression(
                wire::ResourceExpressionNodeProto {
                    kind: "literal".to_owned(),
                    domain: wire::ResourceDomainProto::ResourceReal.into(),
                    unit: unit.to_owned(),
                    literal: Some(wire::ResourceNumberProto {
                        value: Some(wire::resource_number_proto::Value::RealBits(
                            value.to_bits(),
                        )),
                    }),
                    ..Default::default()
                },
            )),
        }
    }

    fn rational(numerator: &str, denominator: &str) -> wire::ResourceNodeProto {
        wire::ResourceNodeProto {
            node: Some(wire::resource_node_proto::Node::Expression(
                wire::ResourceExpressionNodeProto {
                    kind: "literal".to_owned(),
                    domain: wire::ResourceDomainProto::ResourceRational.into(),
                    unit: "1".to_owned(),
                    literal: Some(wire::ResourceNumberProto {
                        value: Some(wire::resource_number_proto::Value::Rational(
                            wire::ResourceRationalProto {
                                numerator: numerator.to_owned(),
                                denominator: denominator.to_owned(),
                            },
                        )),
                    }),
                    ..Default::default()
                },
            )),
        }
    }

    fn typed_symbol(
        scope: &str,
        name: &str,
        domain: wire::ResourceDomainProto,
        unit: &str,
    ) -> wire::ResourceNodeProto {
        wire::ResourceNodeProto {
            node: Some(wire::resource_node_proto::Node::Expression(
                wire::ResourceExpressionNodeProto {
                    kind: "symbol".to_owned(),
                    domain: domain.into(),
                    unit: unit.to_owned(),
                    symbol: Some(wire::ResourceSymbolProto {
                        scope: scope.to_owned(),
                        name: name.to_owned(),
                        domain: domain.into(),
                        unit: unit.to_owned(),
                        origin: "program_parameter".to_owned(),
                        constraints: Vec::new(),
                    }),
                    ..Default::default()
                },
            )),
        }
    }

    fn symbol(scope: &str, name: &str) -> wire::ResourceNodeProto {
        wire::ResourceNodeProto {
            node: Some(wire::resource_node_proto::Node::Expression(
                wire::ResourceExpressionNodeProto {
                    kind: "symbol".to_owned(),
                    domain: wire::ResourceDomainProto::ResourceInteger.into(),
                    unit: "1".to_owned(),
                    symbol: Some(wire::ResourceSymbolProto {
                        scope: scope.to_owned(),
                        name: name.to_owned(),
                        domain: wire::ResourceDomainProto::ResourceInteger.into(),
                        unit: "1".to_owned(),
                        origin: "program_parameter".to_owned(),
                        constraints: Vec::new(),
                    }),
                    ..Default::default()
                },
            )),
        }
    }

    fn encode(nodes: Vec<wire::ResourceNodeProto>, root: u32) -> Vec<u8> {
        wire::ResourceExpressionGraphProto { nodes, root }.encode_to_vec()
    }

    fn expression(kind: &str, arguments: Vec<u32>) -> wire::ResourceNodeProto {
        wire::ResourceNodeProto {
            node: Some(wire::resource_node_proto::Node::Expression(
                wire::ResourceExpressionNodeProto {
                    kind: kind.to_owned(),
                    domain: wire::ResourceDomainProto::ResourceInteger.into(),
                    unit: "1".to_owned(),
                    arguments,
                    ..Default::default()
                },
            )),
        }
    }

    fn evaluate(nodes: Vec<wire::ResourceNodeProto>, root: u32) -> ResourceExpressionEvaluation {
        let graph = wire::ResourceExpressionGraphProto { nodes, root };
        evaluate_resource_expression(&graph.encode_to_vec()).expect("evaluation")
    }

    fn decrement_recurrence(input: &str, cyclic: bool) -> Vec<u8> {
        let mut nodes = vec![
            typed_symbol(
                "equation",
                "n",
                wire::ResourceDomainProto::ResourceInteger,
                "1",
            ),
            natural_with_unit("0", "1"),
            integer("-1"),
            natural_with_unit("1", "1"),
            expression("add", vec![0, 2]),
        ];
        let named =
            |kind: &str, arguments: Vec<u32>, predicates: Vec<u32>| wire::ResourceNodeProto {
                node: Some(wire::resource_node_proto::Node::Expression(
                    wire::ResourceExpressionNodeProto {
                        kind: kind.to_owned(),
                        domain: wire::ResourceDomainProto::ResourceNatural.into(),
                        unit: "1".to_owned(),
                        arguments,
                        predicates,
                        attributes: if kind == "recurrence_system" {
                            vec![]
                        } else {
                            vec![
                                wire::ResourceAttributeProto {
                                    name: "name".to_owned(),
                                    value: "step".to_owned(),
                                },
                                wire::ResourceAttributeProto {
                                    name: "version".to_owned(),
                                    value: "1".to_owned(),
                                },
                            ]
                        },
                        ..Default::default()
                    },
                )),
            };
        nodes.push(named(
            "recurrence",
            vec![if cyclic { 0 } else { 4 }],
            vec![],
        ));
        nodes.push(expression("add", vec![3, 5]));
        nodes.push(wire::ResourceNodeProto {
            node: Some(wire::resource_node_proto::Node::Predicate(
                wire::ResourcePredicateNodeProto {
                    kind: "equal".to_owned(),
                    arguments: vec![0, 1],
                    predicates: vec![],
                },
            )),
        });
        nodes.push(wire::ResourceNodeProto {
            node: Some(wire::resource_node_proto::Node::Expression(
                wire::ResourceExpressionNodeProto {
                    kind: "piecewise".to_owned(),
                    domain: wire::ResourceDomainProto::ResourceNatural.into(),
                    unit: "1".to_owned(),
                    arguments: vec![1, 6],
                    predicates: vec![7],
                    ..Default::default()
                },
            )),
        });
        nodes.push(wire::ResourceNodeProto {
            node: Some(wire::resource_node_proto::Node::Predicate(
                wire::ResourcePredicateNodeProto {
                    kind: "true".to_owned(),
                    arguments: vec![],
                    predicates: vec![],
                },
            )),
        });
        nodes.push(named("recurrence_equation", vec![0, 8], vec![9]));
        let input_id = match input {
            "0" => 1,
            "-1" => 2,
            "1" => 3,
            _ => {
                nodes.push(integer(input));
                11
            }
        };
        let invocation = u32::try_from(nodes.len()).expect("small test graph");
        nodes.push(named("recurrence", vec![input_id], vec![]));
        let root = u32::try_from(nodes.len()).expect("small test graph");
        nodes.push(named("recurrence_system", vec![invocation, 10], vec![]));
        wire::ResourceExpressionGraphProto { nodes, root }.encode_to_vec()
    }

    #[test]
    fn recurrence_evaluation_is_iterative_and_preserves_partial_domains() {
        for n in [0, 3, 10_000] {
            assert_eq!(
                evaluate_resource_expression(&decrement_recurrence(&n.to_string(), false))
                    .expect("valid recurrence"),
                ResourceExpressionEvaluation::Numeric(ResourceScalar::Integer(BigInt::from(n)))
            );
        }
        assert_eq!(
            evaluate_resource_expression(&decrement_recurrence("3", true))
                .expect("cyclic partial recurrence"),
            ResourceExpressionEvaluation::Unknown
        );
        assert_eq!(
            evaluate_resource_expression(&decrement_recurrence("-1", false))
                .expect("work-bounded partial recurrence"),
            ResourceExpressionEvaluation::Unknown
        );
    }

    #[test]
    fn recurrence_memoization_has_an_aggregate_numeric_storage_budget() {
        let huge = (BigInt::one() << 900_000_usize).to_string();
        assert_eq!(
            evaluate_resource_expression(&decrement_recurrence(&huge, false))
                .expect("bounded recurrence storage"),
            ResourceExpressionEvaluation::Unknown
        );
    }

    #[test]
    fn resource_value_relations_enforce_their_payload_shapes() {
        let one = encode(vec![natural_with_unit("1", "semantic_ops")], 0);
        assert!(
            validate_resource_value(
                "natural",
                "semantic_ops",
                "equality",
                Some(&one),
                None,
                None,
                0,
            )
            .is_ok()
        );
        assert!(
            validate_resource_value("natural", "semantic_ops", "equality", None, None, None, 0,)
                .unwrap_err()
                .to_string()
                .contains("requires an expression")
        );
        assert!(
            validate_resource_value(
                "natural",
                "semantic_ops",
                "partial_enclosure",
                None,
                Some(&one),
                None,
                0,
            )
            .unwrap_err()
            .to_string()
            .contains("must name unknown contributors")
        );
        assert!(
            validate_resource_value("natural", "semantic_ops", "unknown", None, None, None, 1,)
                .is_ok()
        );
    }

    #[test]
    fn resource_value_validation_enforces_symbolic_and_concrete_domains() {
        let probability_symbol = encode(
            vec![typed_symbol(
                "test",
                "p",
                wire::ResourceDomainProto::ResourceProbability,
                "semantic_ops",
            )],
            0,
        );
        assert!(
            validate_resource_value(
                "natural",
                "semantic_ops",
                "equality",
                Some(&probability_symbol),
                None,
                None,
                0,
            )
            .unwrap_err()
            .to_string()
            .contains("is not contained in metric domain 'natural'")
        );

        let negative = encode(vec![real_with_unit(-0.5, "seconds")], 0);
        assert!(
            validate_resource_value(
                "real",
                "seconds",
                "equality",
                Some(&negative),
                None,
                None,
                0,
            )
            .unwrap_err()
            .to_string()
            .contains("must be nonnegative")
        );

        let excessive_probability = encode(vec![real_with_unit(1.5, "1")], 0);
        assert!(
            validate_resource_value(
                "probability",
                "1",
                "equality",
                Some(&excessive_probability),
                None,
                None,
                0,
            )
            .unwrap_err()
            .to_string()
            .contains("outside metric domain 'probability'")
        );
    }

    #[test]
    fn resource_value_interval_order_uses_exact_cross_domain_comparison() {
        let lower = encode(vec![natural_with_unit("9007199254740993", "seconds")], 0);
        let upper = encode(vec![real_with_unit(9_007_199_254_740_992.0, "seconds")], 0);
        assert!(
            validate_resource_value(
                "real",
                "seconds",
                "enclosure",
                None,
                Some(&lower),
                Some(&upper),
                0,
            )
            .unwrap_err()
            .to_string()
            .contains("lower endpoint exceeds upper")
        );
    }

    #[test]
    fn exact_addition_preserves_arbitrary_precision() {
        let mut nodes = vec![integer("999999999999999999999999999999"), integer("1")];
        nodes.push(wire::ResourceNodeProto {
            node: Some(wire::resource_node_proto::Node::Expression(
                wire::ResourceExpressionNodeProto {
                    kind: "add".to_owned(),
                    domain: wire::ResourceDomainProto::ResourceInteger.into(),
                    unit: "1".to_owned(),
                    arguments: vec![0, 1],
                    ..Default::default()
                },
            )),
        });
        assert_eq!(
            evaluate(nodes, 2),
            ResourceExpressionEvaluation::Numeric(ResourceScalar::Integer(
                "1000000000000000000000000000000".parse().expect("integer")
            ))
        );
    }

    #[test]
    fn shared_dag_nodes_are_evaluated_once_per_environment() {
        let mut nodes = vec![integer("1")];
        for identifier in 1..=20 {
            nodes.push(expression("add", vec![identifier - 1, identifier - 1]));
        }

        assert_eq!(
            evaluate(nodes, 20),
            ResourceExpressionEvaluation::Numeric(ResourceScalar::Integer(BigInt::one() << 20))
        );
    }

    #[test]
    fn shared_dag_memoization_is_scoped_to_each_binder_value() {
        let mut nodes = vec![
            integer("1"),
            integer("3"),
            symbol("loop", "index"),
            expression("add", vec![2, 2]),
        ];
        nodes.push(wire::ResourceNodeProto {
            node: Some(wire::resource_node_proto::Node::Expression(
                wire::ResourceExpressionNodeProto {
                    kind: "bounded_sum".to_owned(),
                    domain: wire::ResourceDomainProto::ResourceInteger.into(),
                    unit: "1".to_owned(),
                    arguments: vec![0, 1, 3],
                    attributes: vec![
                        wire::ResourceAttributeProto {
                            name: "binder_domain".to_owned(),
                            value: "integer".to_owned(),
                        },
                        wire::ResourceAttributeProto {
                            name: "binder_name".to_owned(),
                            value: "index".to_owned(),
                        },
                        wire::ResourceAttributeProto {
                            name: "binder_origin".to_owned(),
                            value: "bound_variable".to_owned(),
                        },
                        wire::ResourceAttributeProto {
                            name: "binder_scope".to_owned(),
                            value: "loop".to_owned(),
                        },
                    ],
                    ..Default::default()
                },
            )),
        });

        assert_eq!(
            evaluate(nodes, 4),
            ResourceExpressionEvaluation::Numeric(ResourceScalar::Integer(BigInt::from(12)))
        );
    }

    #[test]
    fn predicate_roots_use_exact_comparison() {
        let mut nodes = vec![integer("3"), integer("4")];
        nodes.push(wire::ResourceNodeProto {
            node: Some(wire::resource_node_proto::Node::Predicate(
                wire::ResourcePredicateNodeProto {
                    kind: "less".to_owned(),
                    arguments: vec![0, 1],
                    predicates: Vec::new(),
                },
            )),
        });
        assert_eq!(
            evaluate(nodes, 2),
            ResourceExpressionEvaluation::Boolean(true)
        );
    }

    #[test]
    fn free_symbols_are_unknown() {
        let graph = wire::ResourceExpressionGraphProto {
            nodes: vec![wire::ResourceNodeProto {
                node: Some(wire::resource_node_proto::Node::Expression(
                    wire::ResourceExpressionNodeProto {
                        kind: "symbol".to_owned(),
                        domain: wire::ResourceDomainProto::ResourceNatural.into(),
                        unit: "1".to_owned(),
                        symbol: Some(wire::ResourceSymbolProto {
                            scope: "example".to_owned(),
                            name: "n".to_owned(),
                            domain: wire::ResourceDomainProto::ResourceNatural.into(),
                            unit: "1".to_owned(),
                            origin: "program_parameter".to_owned(),
                            constraints: Vec::new(),
                        }),
                        ..Default::default()
                    },
                )),
            }],
            root: 0,
        };
        assert_eq!(
            evaluate_resource_expression(&graph.encode_to_vec()).expect("evaluation"),
            ResourceExpressionEvaluation::Unknown
        );
    }

    #[test]
    fn free_symbol_projection_respects_body_only_binder_scope() {
        let mut nodes = vec![
            integer("2"),
            symbol("loop", "index"),
            symbol("example", "n"),
            expression("add", vec![1, 2]),
        ];
        nodes.push(wire::ResourceNodeProto {
            node: Some(wire::resource_node_proto::Node::Expression(
                wire::ResourceExpressionNodeProto {
                    kind: "bounded_sum".to_owned(),
                    domain: wire::ResourceDomainProto::ResourceInteger.into(),
                    unit: "1".to_owned(),
                    arguments: vec![1, 0, 3],
                    attributes: vec![
                        wire::ResourceAttributeProto {
                            name: "binder_domain".to_owned(),
                            value: "integer".to_owned(),
                        },
                        wire::ResourceAttributeProto {
                            name: "binder_name".to_owned(),
                            value: "index".to_owned(),
                        },
                        wire::ResourceAttributeProto {
                            name: "binder_origin".to_owned(),
                            value: "bound_variable".to_owned(),
                        },
                        wire::ResourceAttributeProto {
                            name: "binder_scope".to_owned(),
                            value: "loop".to_owned(),
                        },
                    ],
                    ..Default::default()
                },
            )),
        });

        let symbols = free_resource_expression_symbols(&encode(nodes, 4)).expect("projection");
        assert_eq!(
            symbols
                .iter()
                .map(ResourceExpressionSymbol::identity)
                .collect::<Vec<_>>(),
            vec!["example::n", "loop::index"]
        );
    }

    #[test]
    fn resolves_resource_names_once_across_expression_roots() {
        let left = encode(vec![symbol("left", "n")], 0);
        let right = encode(vec![symbol("right", "n")], 0);
        let unique = resolve_resource_symbol_bindings(
            std::slice::from_ref(&left),
            &["n".to_owned(), "missing".to_owned()],
        )
        .expect("unique short name");
        assert_eq!(unique, [("left::n".to_owned(), "n".to_owned())]);

        let qualified = resolve_resource_symbol_bindings(
            std::slice::from_ref(&left),
            &["left::n".to_owned(), "n".to_owned()],
        )
        .expect("qualified name precedence");
        assert_eq!(qualified, [("left::n".to_owned(), "left::n".to_owned())]);

        assert!(matches!(
            resolve_resource_symbol_bindings(&[left, right], &["n".to_owned()]),
            Err(ResourceSymbolResolutionError::Ambiguous { name, qualified })
                if name == "n" && qualified == "'left::n', 'right::n'"
        ));
    }

    #[test]
    fn resource_name_resolution_coalesces_stage_specific_declarations() {
        let integer = encode(vec![symbol("shared", "n")], 0);
        let real = encode(
            vec![typed_symbol(
                "shared",
                "n",
                wire::ResourceDomainProto::ResourceReal,
                "1",
            )],
            0,
        );

        assert_eq!(
            resolve_resource_symbol_bindings(&[integer, real], &["n".to_owned()])
                .expect("one semantic identity"),
            [("shared::n".to_owned(), "n".to_owned())]
        );
    }

    #[test]
    fn resource_hotspots_use_exact_numeric_order_and_stable_symbolic_ties() {
        let candidates = vec![
            ResourceHotspotCandidate::new(
                Some(encode(vec![symbol("program", "n")], 0)),
                "z".to_owned(),
                "symbolic".to_owned(),
            ),
            ResourceHotspotCandidate::new(
                Some(encode(
                    vec![real_with_unit(9_007_199_254_740_992.0, "1")],
                    0,
                )),
                "z".to_owned(),
                "real".to_owned(),
            ),
            ResourceHotspotCandidate::new(
                Some(encode(vec![integer("9007199254740993")], 0)),
                "z".to_owned(),
                "integer".to_owned(),
            ),
            ResourceHotspotCandidate::new(
                Some(encode(vec![rational("3", "2")], 0)),
                "b".to_owned(),
                "rational".to_owned(),
            ),
            ResourceHotspotCandidate::new(
                Some(encode(vec![real_with_unit(1.5, "1")], 0)),
                "a".to_owned(),
                "real-tie".to_owned(),
            ),
        ];

        assert_eq!(
            rank_resource_hotspots(&candidates, usize::MAX).expect("hotspot order"),
            [2, 1, 4, 3, 0]
        );
        assert_eq!(
            rank_resource_hotspots(&candidates, 3).expect("bounded hotspot order"),
            [2, 1, 4]
        );
    }

    #[test]
    fn unit_projection_retags_nested_piecewise_values_not_predicates() {
        let mut nodes = vec![symbol("example", "n"), integer("0")];
        nodes.push(wire::ResourceNodeProto {
            node: Some(wire::resource_node_proto::Node::Predicate(
                wire::ResourcePredicateNodeProto {
                    kind: "greater".to_owned(),
                    arguments: vec![0, 1],
                    predicates: Vec::new(),
                },
            )),
        });
        for arguments in [vec![0, 0], vec![3, 0]] {
            nodes.push(wire::ResourceNodeProto {
                node: Some(wire::resource_node_proto::Node::Expression(
                    wire::ResourceExpressionNodeProto {
                        kind: "piecewise".to_owned(),
                        domain: wire::ResourceDomainProto::ResourceInteger.into(),
                        unit: "1".to_owned(),
                        arguments,
                        predicates: vec![2],
                        ..Default::default()
                    },
                )),
            });
        }

        let payload = retag_resource_expression_unit(&encode(nodes, 4), "semantic_operations")
            .expect("unit projection");
        let projected = PreparedGraph::decode(&payload).expect("projected graph");
        let outer = projected.root_expression().expect("outer expression");
        let inner = projected
            .graph
            .expression(outer.arguments[0])
            .expect("inner expression");
        let selected = projected
            .graph
            .expression(inner.arguments[0])
            .expect("selected value");
        let selected_symbol = projected
            .graph
            .expression(selected.arguments[0])
            .expect("selected symbol");
        let Some(ExpressionGraphNode::Predicate(predicate)) =
            projected.graph.node(outer.predicates[0])
        else {
            panic!("predicate projection");
        };
        let compared = projected
            .graph
            .expression(predicate.arguments[0])
            .expect("predicate symbol");

        assert_eq!(outer.unit, "semantic_operations");
        assert_eq!(inner.unit, "semantic_operations");
        assert_eq!(selected.unit, "semantic_operations");
        assert_eq!(selected.kind, "multiply");
        assert_eq!(selected_symbol.unit, "1");
        assert_eq!(selected_symbol.symbol.as_ref().expect("symbol").unit, "1");
        assert_eq!(compared.unit, "1");
        assert_eq!(compared.symbol.as_ref().expect("symbol").unit, "1");
        assert_eq!(
            free_resource_expression_symbols(&payload)
                .expect("free symbols")
                .iter()
                .map(ResourceExpressionSymbol::identity)
                .collect::<Vec<_>>(),
            vec!["example::n"]
        );
        assert!(matches!(
            retag_resource_expression_unit(&payload, "bad unit"),
            Err(ResourceExpressionSubstitutionError::Invalid(_))
        ));
    }

    #[test]
    fn substitution_replaces_only_the_named_qualified_symbol() {
        let source = encode(vec![symbol("left", "n"), symbol("right", "n")], 0);
        let replacement = encode(vec![integer("7")], 0);
        let substituted = substitute_resource_expression(
            &source,
            vec![("left::n".to_owned(), replacement, false)],
        )
        .expect("substitution");

        assert_eq!(
            evaluate_resource_expression(&substituted).expect("evaluation"),
            ResourceExpressionEvaluation::Numeric(ResourceScalar::Integer(BigInt::from(7)))
        );
    }

    #[test]
    fn normalization_flattens_and_folds_partial_associative_expressions() {
        let source = encode(
            vec![
                symbol("example", "n"),
                integer("2"),
                expression("add", vec![0, 1]),
                integer("3"),
                expression("add", vec![2, 3]),
            ],
            4,
        );
        let normalized = normalize_resource_expression(&source).expect("normalization");
        let graph = wire::ResourceExpressionGraphProto::decode(normalized.as_slice())
            .expect("normalized graph");

        assert_eq!(graph.nodes.len(), 3);
        let Some(wire::resource_node_proto::Node::Expression(root)) =
            &graph.nodes[graph.root as usize].node
        else {
            panic!("normalized root is not an expression");
        };
        assert_eq!(root.kind, "add");
        assert_eq!(root.arguments.len(), 2);
        assert!(graph.nodes.iter().any(|node| node == &integer("5")));
    }

    #[test]
    fn normalization_canonicalizes_commutative_operand_order() {
        let left = encode(
            vec![
                symbol("example", "b"),
                symbol("example", "a"),
                expression("maximum", vec![0, 1]),
            ],
            2,
        );
        let right = encode(
            vec![
                symbol("example", "a"),
                symbol("example", "b"),
                expression("maximum", vec![0, 1]),
            ],
            2,
        );

        assert_eq!(
            normalize_resource_expression(&left).expect("left normalization"),
            normalize_resource_expression(&right).expect("right normalization")
        );
    }

    #[test]
    fn construction_derives_expression_contracts_and_rejects_unit_mismatch() {
        let quantity = encode(
            vec![typed_symbol(
                "example",
                "n",
                wire::ResourceDomainProto::ResourceInteger,
                "semantic_operations",
            )],
            0,
        );
        let scale = encode(vec![integer("2")], 0);
        let product = construct_resource_expression(
            "multiply",
            &[quantity.clone(), scale],
            &[],
            &[],
            None,
            None,
            None,
        )
        .expect("typed product");
        let graph = wire::ResourceExpressionGraphProto::decode(product.as_slice())
            .expect("constructed graph");
        let Some(wire::resource_node_proto::Node::Expression(root)) =
            &graph.nodes[graph.root as usize].node
        else {
            panic!("constructed root is not an expression");
        };
        assert_eq!(
            root.domain,
            i32::from(wire::ResourceDomainProto::ResourceInteger)
        );
        assert_eq!(root.unit, "semantic_operations");

        let seconds = encode(vec![real_with_unit(1.0, "seconds")], 0);
        assert!(matches!(
            construct_resource_expression(
                "add",
                &[quantity, seconds],
                &[],
                &[],
                None,
                None,
                None,
            ),
            Err(ResourceExpressionSubstitutionError::Invalid(message))
                if message.contains("different units")
        ));
    }

    #[test]
    fn literal_construction_derives_and_checks_numeric_domains() {
        let positive = wire::ResourceNumberProto {
            value: Some(wire::resource_number_proto::Value::Integer("2".to_owned())),
        }
        .encode_to_vec();
        let literal = construct_resource_literal(&positive, "1", None).expect("literal");
        let graph = wire::ResourceExpressionGraphProto::decode(literal.as_slice())
            .expect("constructed literal graph");
        let Some(wire::resource_node_proto::Node::Expression(root)) =
            &graph.nodes[graph.root as usize].node
        else {
            panic!("constructed root is not an expression");
        };
        assert_eq!(
            root.domain,
            i32::from(wire::ResourceDomainProto::ResourceNatural)
        );
        assert!(matches!(
            construct_resource_literal(
                &positive,
                "1",
                Some(wire::ResourceDomainProto::ResourceProbability.into()),
            ),
            Err(ResourceExpressionSubstitutionError::Invalid(message))
                if message.contains("outside its declared domain")
        ));
    }

    #[test]
    fn unit_and_symbol_construction_are_native_and_bounded() {
        validate_resource_unit("semantic_operations").expect("valid unit");
        assert!(matches!(
            validate_resource_unit("bad unit"),
            Err(ResourceExpressionSubstitutionError::Invalid(message))
                if message.contains("nonempty and whitespace-free")
        ));

        let payload = construct_resource_symbol(
            "example",
            "n",
            wire::ResourceDomainProto::ResourceInteger.into(),
            "semantic_operations",
            "program_parameter",
            &["n >= 0".to_owned()],
        )
        .expect("symbol");
        let graph = wire::ResourceExpressionGraphProto::decode(payload.as_slice())
            .expect("constructed symbol graph");
        let Some(wire::resource_node_proto::Node::Expression(root)) =
            &graph.nodes[graph.root as usize].node
        else {
            panic!("constructed root is not an expression");
        };
        let declaration = root.symbol.as_ref().expect("symbol declaration");
        assert_eq!(root.kind, "symbol");
        assert_eq!(declaration.scope, "example");
        assert_eq!(declaration.name, "n");
        assert_eq!(declaration.constraints, ["n >= 0"]);

        assert!(matches!(
            construct_resource_symbol(
                "example::nested",
                "n",
                wire::ResourceDomainProto::ResourceNatural.into(),
                "1",
                "program_parameter",
                &[],
            ),
            Err(ResourceExpressionSubstitutionError::Invalid(message))
                if message.contains("reserved for qualified")
        ));
        assert!(matches!(
            construct_resource_symbol(
                "example",
                "n",
                wire::ResourceDomainProto::ResourceNatural.into(),
                "bad unit",
                "program_parameter",
                &[],
            ),
            Err(ResourceExpressionSubstitutionError::Invalid(message))
                if message.contains("nonempty and whitespace-free")
        ));
    }

    #[test]
    fn predicate_construction_checks_shape_units_and_domains() {
        let quantity = encode(
            vec![typed_symbol(
                "example",
                "n",
                wire::ResourceDomainProto::ResourceInteger,
                "semantic_operations",
            )],
            0,
        );
        let seconds = encode(vec![real_with_unit(1.0, "seconds")], 0);
        assert!(matches!(
            construct_resource_predicate("equal", &[quantity, seconds], &[]),
            Err(ResourceExpressionSubstitutionError::Invalid(message))
                if message.contains("units must agree")
        ));
        assert!(matches!(
            construct_resource_predicate("and", &[], &[]),
            Err(ResourceExpressionSubstitutionError::Invalid(message))
                if message.contains("constructor shape")
        ));

        let truth = construct_resource_predicate("true", &[], &[]).expect("truth predicate");
        let graph = wire::ResourceExpressionGraphProto::decode(truth.as_slice())
            .expect("constructed predicate graph");
        let Some(wire::resource_node_proto::Node::Predicate(root)) =
            &graph.nodes[graph.root as usize].node
        else {
            panic!("constructed root is not a predicate");
        };
        assert_eq!(root.kind, "true");
    }

    #[test]
    fn bounded_construction_checks_the_ephemeral_binder_contract() {
        let lower = encode(vec![integer("0")], 0);
        let upper = encode(vec![integer("3")], 0);
        let body = encode(
            vec![typed_symbol(
                "example",
                "i",
                wire::ResourceDomainProto::ResourceNatural,
                "1",
            )],
            0,
        );
        let attributes = vec![
            ("binder_domain".to_owned(), "natural".to_owned()),
            ("binder_name".to_owned(), "i".to_owned()),
            ("binder_origin".to_owned(), "program_parameter".to_owned()),
            ("binder_scope".to_owned(), "example".to_owned()),
        ];
        let bounded = construct_resource_expression(
            "bounded_sum",
            &[lower.clone(), upper.clone(), body.clone()],
            &[],
            &attributes,
            Some("1"),
            None,
            None,
        )
        .expect("bounded sum");
        let graph = wire::ResourceExpressionGraphProto::decode(bounded.as_slice())
            .expect("constructed bounded graph");
        let Some(wire::resource_node_proto::Node::Expression(root)) =
            &graph.nodes[graph.root as usize].node
        else {
            panic!("constructed root is not an expression");
        };
        assert_eq!(root.kind, "bounded_sum");
        assert!(matches!(
            construct_resource_expression(
                "bounded_sum",
                &[lower, upper, body],
                &[],
                &attributes,
                Some("seconds"),
                None,
                None,
            ),
            Err(ResourceExpressionSubstitutionError::Invalid(message))
                if message.contains("binder must be dimensionless")
        ));
    }

    #[test]
    fn opaque_construction_requires_one_explicit_native_contract() {
        let argument = encode(vec![integer("2")], 0);
        let attributes = vec![
            ("name".to_owned(), "example.cost".to_owned()),
            ("version".to_owned(), "1".to_owned()),
        ];
        let recurrence = construct_resource_expression(
            "recurrence",
            std::slice::from_ref(&argument),
            &[],
            &attributes,
            None,
            Some(wire::ResourceDomainProto::ResourceNatural.into()),
            Some("semantic_operations"),
        )
        .expect("typed recurrence");
        let graph = wire::ResourceExpressionGraphProto::decode(recurrence.as_slice())
            .expect("constructed recurrence graph");
        let Some(wire::resource_node_proto::Node::Expression(root)) =
            &graph.nodes[graph.root as usize].node
        else {
            panic!("constructed root is not an expression");
        };
        assert_eq!(root.kind, "recurrence");
        assert_eq!(root.unit, "semantic_operations");

        assert!(matches!(
            construct_resource_expression(
                "recurrence",
                std::slice::from_ref(&argument),
                &[],
                &attributes,
                None,
                None,
                None,
            ),
            Err(ResourceExpressionSubstitutionError::Invalid(message))
                if message.contains("requires an explicit contract")
        ));
        assert!(matches!(
            construct_resource_expression(
                "add",
                &[argument.clone(), argument],
                &[],
                &[],
                None,
                Some(wire::ResourceDomainProto::ResourceNatural.into()),
                Some("1"),
            ),
            Err(ResourceExpressionSubstitutionError::Invalid(message))
                if message.contains("cannot override")
        ));
    }

    #[test]
    fn surface_code_projection_matches_the_reference_model() {
        let projection = project_surface_code_resources(
            &encode(vec![natural_with_unit("6144", "logical_qubits")], 0),
            &encode(
                vec![natural_with_unit("2576980378", "semantic_operations")],
                0,
            ),
            0.05,
            0.01,
            0.001,
            0.000_001,
        )
        .expect("surface-code projection");

        assert_eq!(
            evaluate_resource_expression(projection.code_distance()).expect("distance"),
            ResourceExpressionEvaluation::Numeric(ResourceScalar::Integer(BigInt::from(25)))
        );
        assert_eq!(
            evaluate_resource_expression(projection.physical_qubits()).expect("qubits"),
            ResourceExpressionEvaluation::Numeric(ResourceScalar::Integer(BigInt::from(
                15_360_000
            )))
        );
        let ResourceExpressionEvaluation::Numeric(ResourceScalar::Real(duration)) =
            evaluate_resource_expression(projection.duration()).expect("duration")
        else {
            panic!("duration is not real");
        };
        assert!((duration - 64_424.509_45).abs() < 1.0e-9);
    }

    #[test]
    fn surface_code_projection_rejects_invalid_inputs() {
        let logical = encode(vec![natural_with_unit("1", "logical_qubits")], 0);
        let work = encode(vec![natural_with_unit("1", "semantic_operations")], 0);
        let dimensionless = encode(vec![natural_with_unit("1", "1")], 0);
        assert!(matches!(
            project_surface_code_resources(&dimensionless, &work, 0.05, 0.01, 0.001, 0.000_001,),
            Err(SurfaceCodeProjectionError::Invalid(_))
        ));
        assert!(matches!(
            project_surface_code_resources(&logical, &work, 0.05, 0.001, 0.01, 0.000_001),
            Err(SurfaceCodeProjectionError::Invalid(_))
        ));
        assert!(matches!(
            project_surface_code_resources(&logical, &work, f64::INFINITY, 0.01, 0.001, 0.000_001,),
            Err(SurfaceCodeProjectionError::Invalid(_))
        ));
    }
}
