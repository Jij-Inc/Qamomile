//! Native symbolic logical-operation counts over verified `QCore`.
//!
//! Expressions form a closed, hash-consed, topologically ordered DAG. The same
//! algebra analyzes closed and open programs so static specialization does not
//! select a second implementation of logical resource semantics.
//! Structured exits retain all possible paths. Measurement-dependent loop exits
//! make subsequent iterations and loop-else work optional, yielding enclosures
//! without expanding loops beyond the bounded interpretation budget.
//! Proven adjacent-lane chains retain compact per-lane completion times rather
//! than materializing their qubits or imposing an artificial global barrier.

mod count_algebra;
mod expression_arena;
mod external_rules;
mod observation;
mod recurrences;
mod timing;
mod timing_chain;
mod values;
mod visitor;
mod widths;

#[cfg(test)]
mod tests;

use count_algebra::{
    external_metrics_with_structural_counts, external_operation_counts, merge_optional_symbolic,
};
use timing::{
    add_symbolic_timing_footprints, append_symbolic_timing, instantiate_symbolic_timing,
    natural_literal, reverse_symbolic_timing, schedule_symbolic_depth_bounds,
    small_permutation_step_bound, symbolic_adjoint_roots, symbolic_call_roots,
    symbolic_coherent_roots, symbolic_depth_families, symbolic_quantum_multiplicity,
    symbolic_timing_root, symbolic_timing_selectors, symbolic_unit_durations,
};

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use num_bigint::BigInt;
use num_traits::Zero;
use thiserror::Error;

use crate::lowering::approximation::OUTPUT_TRACE_DISTANCE_BOUND_METRIC;
use crate::lowering::realization::LoweringConfig;
use crate::operations::{LocallyCheckedOperations, TerminatorKind};
use crate::resource_rules::{
    Composition as MetricComposition, Domain as ResourceDomain, ResourceRuleContract, TransformMode,
};
use crate::resources::analysis::function_value_regions;
use crate::static_values::{StaticScalar, StaticValue, normalize_aggregate_index};
use crate::types::{Dimension, TypeArena};
use crate::wire;

const MAX_SYMBOLIC_LOOP_INTERPRETATIONS: u128 = 4_096;
const MAX_SYMBOLIC_POLYNOMIAL_TERMS: usize = 512;
const MAX_SYMBOLIC_POLYNOMIAL_FACTORS: usize = 4_096;

/// Explicit portable counting alphabet; arbitrary rotations remain unsynthesized.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GateBasis {
    /// Clifford primitives, Toffoli, and arbitrary single-qubit rotations.
    CliffordToffoliRotations,
    /// Clifford and T primitives, plus unsynthesized arbitrary rotations.
    CliffordTRotations,
}

/// Transient observation policy consumed by the existing symbolic visitor.
#[derive(Clone, Debug, Default)]
pub(crate) struct ResourceObservation {
    expanded: Option<BTreeSet<String>>,
    basis: Option<GateBasis>,
    realizations: LoweringConfig,
}

impl ResourceObservation {
    /// Observe selected callable bodies, or all bodies when `expanded` is absent.
    #[must_use]
    pub(crate) fn new(
        expanded: Option<BTreeSet<String>>,
        basis: Option<GateBasis>,
        realizations: LoweringConfig,
    ) -> Self {
        Self {
            expanded,
            basis,
            realizations,
        }
    }
}

const FRONTIER_PREFIX: &str = "qamomile.algorithmic.frontier.";
const PORTABLE_PREFIX: &str = "qamomile.circuit.portable.";

/// Stable index into one topologically ordered resource-expression DAG.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ResourceExpressionId(u32);

impl ResourceExpressionId {
    /// Return the portable zero-based node index.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    fn from_index(index: usize) -> Result<Self, SymbolicResourceError> {
        u32::try_from(index)
            .map(Self)
            .map_err(|_| SymbolicResourceError::Overflow)
    }

    fn index(self) -> Option<usize> {
        usize::try_from(self.0).ok()
    }
}

type ExprId = ResourceExpressionId;

/// Closed Boolean predicate constructor in the resource-expression algebra.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResourcePredicateKind {
    /// Logical conjunction.
    And,
    /// Logical disjunction.
    Or,
    /// Logical negation.
    Not,
    /// Numeric equality.
    Equal,
    /// Numeric inequality.
    NotEqual,
    /// Strict less-than comparison.
    Less,
    /// Less-than-or-equal comparison.
    LessEqual,
    /// Strict greater-than comparison.
    Greater,
    /// Greater-than-or-equal comparison.
    GreaterEqual,
    /// Exact integer divisibility.
    Divisible,
}

impl ResourcePredicateKind {
    /// Return the stable Python-facing predicate spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::And => "and",
            Self::Or => "or",
            Self::Not => "not",
            Self::Equal => "equal",
            Self::NotEqual => "not_equal",
            Self::Less => "less",
            Self::LessEqual => "less_equal",
            Self::Greater => "greater",
            Self::GreaterEqual => "greater_equal",
            Self::Divisible => "divisible",
        }
    }
}

/// Closed constructor of one projected resource-expression node.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResourceExpressionKind {
    /// Numeric literal.
    Literal,
    /// Scoped symbolic value.
    Symbol,
    /// Associative sum.
    Add,
    /// Associative product.
    Multiply,
    /// Pointwise minimum.
    Minimum,
    /// Pointwise maximum.
    Maximum,
    /// Integer power.
    Power,
    /// Absolute value.
    Absolute,
    /// Mathematical floor.
    Floor,
    /// Mathematical ceiling.
    Ceiling,
    /// Integer floor division.
    FloorDivide,
    /// Integer ceiling division.
    CeilingDivide,
    /// Natural exponential.
    Exponential,
    /// Logarithm.
    Logarithm,
    /// Base-two logarithm.
    Log2,
    /// Real square root.
    SquareRoot,
    /// Named contracted function.
    ContractedFunction,
    /// Named recurrence.
    Recurrence,
    /// A lexically bound recurrence equation.
    RecurrenceEquation,
    /// A closed simultaneous recurrence system.
    RecurrenceSystem,
    /// Inclusive bounded sum.
    BoundedSum,
    /// Inclusive bounded product.
    BoundedProduct,
    /// Boolean literal.
    PredicateLiteral(bool),
    /// Boolean predicate constructor.
    Predicate(ResourcePredicateKind),
    /// Ordered conditional expression.
    Piecewise,
}

impl ResourceExpressionKind {
    /// Return the stable Python-facing constructor spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Literal => "literal",
            Self::Symbol => "symbol",
            Self::Add => "add",
            Self::Multiply => "multiply",
            Self::Minimum => "minimum",
            Self::Maximum => "maximum",
            Self::Power => "power",
            Self::Absolute => "absolute",
            Self::Floor => "floor",
            Self::Ceiling => "ceiling",
            Self::FloorDivide => "floor_divide",
            Self::CeilingDivide => "ceiling_divide",
            Self::Exponential => "exp",
            Self::Logarithm => "log",
            Self::Log2 => "log2",
            Self::SquareRoot => "square_root",
            Self::ContractedFunction => "contracted_function",
            Self::Recurrence => "recurrence",
            Self::RecurrenceEquation => "recurrence_equation",
            Self::RecurrenceSystem => "recurrence_system",
            Self::BoundedSum => "bounded_sum",
            Self::BoundedProduct => "bounded_product",
            Self::PredicateLiteral(true) => "true",
            Self::PredicateLiteral(false) => "false",
            Self::Predicate(kind) => kind.as_str(),
            Self::Piecewise => "piecewise",
        }
    }
}

/// Closed numeric literal representation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResourceLiteralKind {
    /// Arbitrary-precision integer literal.
    Integer,
    /// Exact reduced rational literal.
    Rational,
    /// Exact binary64 literal.
    Real,
}

impl ResourceLiteralKind {
    /// Return the stable Python-facing literal spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Integer => "integer",
            Self::Rational => "rational",
            Self::Real => "real",
        }
    }
}

/// One immutable node projected from a checked resource-expression DAG.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ResourceExpression {
    node: ResourceExpressionNode,
}

/// Borrowed metadata for an exact rational resource literal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceRationalLiteral<'a> {
    numerator: &'a str,
    denominator: &'a str,
}

impl<'a> ResourceRationalLiteral<'a> {
    /// Return the canonical signed numerator.
    #[must_use]
    pub const fn numerator(self) -> &'a str {
        self.numerator
    }

    /// Return the canonical positive denominator.
    #[must_use]
    pub const fn denominator(self) -> &'a str {
        self.denominator
    }
}

/// Borrowed metadata for a scoped resource symbol or binder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceSymbol<'a> {
    scope: &'a str,
    name: &'a str,
    domain: ResourceDomain,
    origin: &'a str,
    constraints: &'a [String],
}

impl<'a> ResourceSymbol<'a> {
    /// Return the stable scope owning this symbol.
    #[must_use]
    pub const fn scope(self) -> &'a str {
        self.scope
    }

    /// Return the source-facing symbol name.
    #[must_use]
    pub const fn name(self) -> &'a str {
        self.name
    }

    /// Return the symbol's mathematical domain.
    #[must_use]
    pub const fn domain(self) -> ResourceDomain {
        self.domain
    }

    /// Return the declaration origin.
    #[must_use]
    pub const fn origin(self) -> &'a str {
        self.origin
    }

    /// Return the canonical declaration constraints.
    #[must_use]
    pub const fn constraints(self) -> &'a [String] {
        self.constraints
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum ResourceExpressionNode {
    /// An integral literal encoded canonically in base ten.
    Literal {
        /// Canonical signed or unsigned decimal spelling.
        value: String,
        /// Mathematical domain name.
        domain: ResourceDomain,
    },
    /// An exact reduced rational literal.
    RationalLiteral {
        /// Canonical signed numerator.
        numerator: String,
        /// Canonical positive denominator.
        denominator: String,
        /// Mathematical domain name.
        domain: ResourceDomain,
    },
    /// A finite IEEE-754 binary64 literal.
    RealLiteral {
        /// Exact binary64 representation.
        bits: u64,
        /// Mathematical domain name.
        domain: ResourceDomain,
    },
    /// A scoped program or loop symbol.
    Symbol {
        /// Stable scope identity.
        scope: String,
        /// Source-facing symbol name.
        name: String,
        /// Mathematical domain name.
        domain: ResourceDomain,
        /// Provenance category.
        origin: String,
        /// Declarative constraints.
        constraints: Box<[String]>,
    },
    /// A canonical associative sum.
    Add {
        /// Mathematical result domain.
        domain: ResourceDomain,
        /// Sum operands.
        arguments: Box<[ExprId]>,
    },
    /// A canonical associative product.
    Multiply {
        /// Mathematical result domain.
        domain: ResourceDomain,
        /// Product operands.
        arguments: Box<[ExprId]>,
    },
    /// A pointwise minimum over control alternatives.
    Minimum {
        /// Mathematical result domain.
        domain: ResourceDomain,
        /// Alternative operands.
        arguments: Box<[ExprId]>,
    },
    /// A pointwise maximum over control alternatives.
    Maximum {
        /// Mathematical result domain.
        domain: ResourceDomain,
        /// Alternative operands.
        arguments: Box<[ExprId]>,
    },
    /// An integer power.
    Power {
        /// Mathematical result domain.
        domain: ResourceDomain,
        /// Base and exponent operands.
        arguments: Box<[ExprId]>,
    },
    /// An absolute value.
    Absolute {
        /// Mathematical result domain.
        domain: ResourceDomain,
        /// Value operand.
        argument: ExprId,
    },
    /// A mathematical floor.
    Floor {
        /// Value operand.
        argument: ExprId,
    },
    /// A mathematical ceiling.
    Ceiling {
        /// Value operand.
        argument: ExprId,
    },
    /// Integer floor division.
    FloorDivide {
        /// Numerator and denominator operands.
        arguments: Box<[ExprId; 2]>,
    },
    /// Integer ceiling division.
    CeilingDivide {
        /// Numerator and denominator operands.
        arguments: Box<[ExprId; 2]>,
    },
    /// A natural exponential.
    Exponential {
        /// Value operand.
        argument: ExprId,
    },
    /// A logarithm with an optional explicit base.
    Logarithm {
        /// Value followed by an optional base.
        arguments: Box<[ExprId]>,
    },
    /// A base-two logarithm.
    Log2 {
        /// Value operand.
        argument: ExprId,
    },
    /// A real square root.
    SquareRoot {
        /// Value operand.
        argument: ExprId,
    },
    /// A namespaced declarative function.
    Contracted {
        /// Stable function identity.
        name: String,
        /// Mathematical result domain.
        domain: ResourceDomain,
        /// Function operands.
        arguments: Box<[ExprId]>,
    },
    /// A namespaced declarative recurrence.
    Recurrence {
        /// Stable recurrence identity.
        name: String,
        /// Mathematical result domain.
        domain: ResourceDomain,
        /// Recurrence operands.
        arguments: Box<[ExprId]>,
    },
    /// An inclusive finite symbolic summation.
    BoundedSum {
        /// Mathematical result domain.
        domain: ResourceDomain,
        /// Loop binder symbol node.
        binder: ExprId,
        /// Inclusive lower bound.
        lower: ExprId,
        /// Inclusive upper bound.
        upper: ExprId,
        /// Summand expression.
        body: ExprId,
    },
    /// An inclusive finite symbolic product.
    BoundedProduct {
        /// Mathematical result domain.
        domain: ResourceDomain,
        /// Loop binder symbol node.
        binder: ExprId,
        /// Inclusive lower bound.
        lower: ExprId,
        /// Inclusive upper bound.
        upper: ExprId,
        /// Factor expression.
        body: ExprId,
    },
    /// Equation arguments are formal integer symbols followed by its body.
    RecurrenceEquation {
        name: String,
        domain: ResourceDomain,
        arguments: Box<[ExprId]>,
        condition: ExprId,
    },
    /// Invocation followed by a finite set of simultaneous equations.
    RecurrenceSystem {
        domain: ResourceDomain,
        arguments: Box<[ExprId]>,
    },
    /// A canonical Boolean literal used by an ordered piecewise expression.
    PredicateLiteral(bool),
    /// A closed predicate constructor.
    Predicate {
        /// Predicate kind (`and`, `or`, `not`, or a comparison spelling).
        kind: ResourcePredicateKind,
        /// Numeric expression operands for comparisons.
        arguments: Box<[ExprId]>,
        /// Nested predicate operands for Boolean connectives.
        predicates: Box<[ExprId]>,
    },
    /// An ordered symbolic conditional expression with a final default.
    Piecewise {
        /// Mathematical result domain.
        domain: ResourceDomain,
        /// Guarded results followed by the default result.
        arguments: Box<[ExprId]>,
        /// Ordered guards corresponding to all non-default results.
        predicates: Box<[ExprId]>,
    },
}

impl ResourceExpressionNode {
    const fn domain(&self) -> ResourceDomain {
        match self {
            Self::Literal { domain, .. }
            | Self::RationalLiteral { domain, .. }
            | Self::RealLiteral { domain, .. }
            | Self::Symbol { domain, .. }
            | Self::Add { domain, .. }
            | Self::Multiply { domain, .. }
            | Self::Power { domain, .. }
            | Self::Absolute { domain, .. }
            | Self::Minimum { domain, .. }
            | Self::Maximum { domain, .. }
            | Self::Contracted { domain, .. }
            | Self::Recurrence { domain, .. }
            | Self::RecurrenceEquation { domain, .. }
            | Self::RecurrenceSystem { domain, .. }
            | Self::Piecewise { domain, .. }
            | Self::BoundedSum { domain, .. }
            | Self::BoundedProduct { domain, .. } => *domain,
            Self::Floor { .. }
            | Self::Ceiling { .. }
            | Self::FloorDivide { .. }
            | Self::CeilingDivide { .. } => ResourceDomain::Integer,
            Self::Exponential { .. }
            | Self::Logarithm { .. }
            | Self::Log2 { .. }
            | Self::SquareRoot { .. } => ResourceDomain::Real,
            Self::PredicateLiteral(_) | Self::Predicate { .. } => ResourceDomain::Boolean,
        }
    }

    fn arguments(&self) -> Vec<ExprId> {
        match self {
            Self::Add { arguments, .. }
            | Self::Multiply { arguments, .. }
            | Self::Power { arguments, .. }
            | Self::Piecewise { arguments, .. }
            | Self::Minimum { arguments, .. }
            | Self::Maximum { arguments, .. }
            | Self::Contracted { arguments, .. }
            | Self::Recurrence { arguments, .. }
            | Self::RecurrenceEquation { arguments, .. }
            | Self::RecurrenceSystem { arguments, .. }
            | Self::Logarithm { arguments }
            | Self::Predicate { arguments, .. } => arguments.to_vec(),
            Self::Absolute { argument, .. }
            | Self::Floor { argument }
            | Self::Ceiling { argument }
            | Self::Exponential { argument }
            | Self::Log2 { argument }
            | Self::SquareRoot { argument } => vec![*argument],
            Self::FloorDivide { arguments } | Self::CeilingDivide { arguments } => {
                arguments.to_vec()
            }
            Self::BoundedSum {
                lower, upper, body, ..
            }
            | Self::BoundedProduct {
                lower, upper, body, ..
            } => vec![*lower, *upper, *body],
            Self::Literal { .. }
            | Self::RationalLiteral { .. }
            | Self::RealLiteral { .. }
            | Self::Symbol { .. }
            | Self::PredicateLiteral(_) => Vec::new(),
        }
    }

    fn predicates(&self) -> Vec<ExprId> {
        match self {
            Self::Predicate { predicates, .. } | Self::Piecewise { predicates, .. } => {
                predicates.to_vec()
            }
            Self::RecurrenceEquation { condition, .. } => vec![*condition],
            _ => Vec::new(),
        }
    }

    fn dependencies(&self) -> Vec<ExprId> {
        let mut dependencies = self.arguments();
        dependencies.extend(self.predicates());
        dependencies
    }
}

fn resource_domain(name: &str) -> Result<ResourceDomain, SymbolicResourceError> {
    match name {
        "natural" => Ok(ResourceDomain::Natural),
        "integer" => Ok(ResourceDomain::Integer),
        "rational" => Ok(ResourceDomain::Rational),
        "real" => Ok(ResourceDomain::Real),
        "probability" => Ok(ResourceDomain::Probability),
        "boolean" => Ok(ResourceDomain::Boolean),
        _ => Err(SymbolicResourceError::Missing(format!(
            "verified resource domain {name:?}"
        ))),
    }
}

fn resource_predicate_kind(name: &str) -> Result<ResourcePredicateKind, SymbolicResourceError> {
    match name {
        "and" => Ok(ResourcePredicateKind::And),
        "or" => Ok(ResourcePredicateKind::Or),
        "not" => Ok(ResourcePredicateKind::Not),
        "equal" => Ok(ResourcePredicateKind::Equal),
        "not_equal" => Ok(ResourcePredicateKind::NotEqual),
        "less" => Ok(ResourcePredicateKind::Less),
        "less_equal" => Ok(ResourcePredicateKind::LessEqual),
        "greater" => Ok(ResourcePredicateKind::Greater),
        "greater_equal" => Ok(ResourcePredicateKind::GreaterEqual),
        "divisible" => Ok(ResourcePredicateKind::Divisible),
        _ => Err(SymbolicResourceError::Missing(format!(
            "verified resource predicate {name:?}"
        ))),
    }
}

impl ResourceExpression {
    fn from_node(node: ResourceExpressionNode) -> Self {
        Self { node }
    }

    /// Return the closed constructor kind.
    #[must_use]
    pub const fn kind(&self) -> ResourceExpressionKind {
        match &self.node {
            ResourceExpressionNode::Literal { .. }
            | ResourceExpressionNode::RationalLiteral { .. }
            | ResourceExpressionNode::RealLiteral { .. } => ResourceExpressionKind::Literal,
            ResourceExpressionNode::Symbol { .. } => ResourceExpressionKind::Symbol,
            ResourceExpressionNode::Add { .. } => ResourceExpressionKind::Add,
            ResourceExpressionNode::Multiply { .. } => ResourceExpressionKind::Multiply,
            ResourceExpressionNode::Minimum { .. } => ResourceExpressionKind::Minimum,
            ResourceExpressionNode::Maximum { .. } => ResourceExpressionKind::Maximum,
            ResourceExpressionNode::Power { .. } => ResourceExpressionKind::Power,
            ResourceExpressionNode::Absolute { .. } => ResourceExpressionKind::Absolute,
            ResourceExpressionNode::Floor { .. } => ResourceExpressionKind::Floor,
            ResourceExpressionNode::Ceiling { .. } => ResourceExpressionKind::Ceiling,
            ResourceExpressionNode::FloorDivide { .. } => ResourceExpressionKind::FloorDivide,
            ResourceExpressionNode::CeilingDivide { .. } => ResourceExpressionKind::CeilingDivide,
            ResourceExpressionNode::Exponential { .. } => ResourceExpressionKind::Exponential,
            ResourceExpressionNode::Logarithm { .. } => ResourceExpressionKind::Logarithm,
            ResourceExpressionNode::Log2 { .. } => ResourceExpressionKind::Log2,
            ResourceExpressionNode::SquareRoot { .. } => ResourceExpressionKind::SquareRoot,
            ResourceExpressionNode::Contracted { .. } => ResourceExpressionKind::ContractedFunction,
            ResourceExpressionNode::Recurrence { .. } => ResourceExpressionKind::Recurrence,
            ResourceExpressionNode::RecurrenceEquation { .. } => {
                ResourceExpressionKind::RecurrenceEquation
            }
            ResourceExpressionNode::RecurrenceSystem { .. } => {
                ResourceExpressionKind::RecurrenceSystem
            }
            ResourceExpressionNode::BoundedSum { .. } => ResourceExpressionKind::BoundedSum,
            ResourceExpressionNode::BoundedProduct { .. } => ResourceExpressionKind::BoundedProduct,
            ResourceExpressionNode::PredicateLiteral(value) => {
                ResourceExpressionKind::PredicateLiteral(*value)
            }
            ResourceExpressionNode::Predicate { kind, .. } => {
                ResourceExpressionKind::Predicate(*kind)
            }
            ResourceExpressionNode::Piecewise { .. } => ResourceExpressionKind::Piecewise,
        }
    }

    /// Return operand node identifiers in constructor order.
    #[must_use]
    pub fn arguments(&self) -> Vec<ResourceExpressionId> {
        self.node.arguments()
    }

    /// Return predicate-node identifiers in constructor order.
    #[must_use]
    pub fn predicates(&self) -> Vec<ResourceExpressionId> {
        self.node.predicates()
    }

    /// Return the mathematical domain.
    #[must_use]
    pub const fn domain(&self) -> ResourceDomain {
        match &self.node {
            ResourceExpressionNode::Literal { domain, .. }
            | ResourceExpressionNode::RationalLiteral { domain, .. }
            | ResourceExpressionNode::RealLiteral { domain, .. }
            | ResourceExpressionNode::Symbol { domain, .. }
            | ResourceExpressionNode::Add { domain, .. }
            | ResourceExpressionNode::Multiply { domain, .. }
            | ResourceExpressionNode::Power { domain, .. }
            | ResourceExpressionNode::Absolute { domain, .. }
            | ResourceExpressionNode::Minimum { domain, .. }
            | ResourceExpressionNode::Maximum { domain, .. }
            | ResourceExpressionNode::Contracted { domain, .. }
            | ResourceExpressionNode::Recurrence { domain, .. }
            | ResourceExpressionNode::RecurrenceEquation { domain, .. }
            | ResourceExpressionNode::RecurrenceSystem { domain, .. }
            | ResourceExpressionNode::Piecewise { domain, .. }
            | ResourceExpressionNode::BoundedSum { domain, .. }
            | ResourceExpressionNode::BoundedProduct { domain, .. } => *domain,
            ResourceExpressionNode::Floor { .. }
            | ResourceExpressionNode::Ceiling { .. }
            | ResourceExpressionNode::FloorDivide { .. }
            | ResourceExpressionNode::CeilingDivide { .. } => ResourceDomain::Integer,
            ResourceExpressionNode::Exponential { .. }
            | ResourceExpressionNode::Logarithm { .. }
            | ResourceExpressionNode::Log2 { .. }
            | ResourceExpressionNode::SquareRoot { .. } => ResourceDomain::Real,
            ResourceExpressionNode::PredicateLiteral(_)
            | ResourceExpressionNode::Predicate { .. } => ResourceDomain::Boolean,
        }
    }

    /// Return the literal payload when this is a literal node.
    #[must_use]
    pub fn literal(&self) -> Option<&str> {
        match &self.node {
            ResourceExpressionNode::Literal { value, .. } => Some(value),
            _ => None,
        }
    }

    /// Return the closed literal payload discriminator.
    #[must_use]
    pub const fn literal_kind(&self) -> Option<ResourceLiteralKind> {
        match &self.node {
            ResourceExpressionNode::Literal { .. } => Some(ResourceLiteralKind::Integer),
            ResourceExpressionNode::RationalLiteral { .. } => Some(ResourceLiteralKind::Rational),
            ResourceExpressionNode::RealLiteral { .. } => Some(ResourceLiteralKind::Real),
            _ => None,
        }
    }

    /// Return the exact rational numerator and denominator when present.
    #[must_use]
    pub fn rational_literal(&self) -> Option<ResourceRationalLiteral<'_>> {
        match &self.node {
            ResourceExpressionNode::RationalLiteral {
                numerator,
                denominator,
                ..
            } => Some(ResourceRationalLiteral {
                numerator,
                denominator,
            }),
            _ => None,
        }
    }

    /// Return the exact binary64 payload when present.
    #[must_use]
    pub const fn real_literal_bits(&self) -> Option<u64> {
        match &self.node {
            ResourceExpressionNode::RealLiteral { bits, .. } => Some(*bits),
            _ => None,
        }
    }

    /// Return symbol metadata when this node declares a symbol or binder.
    #[must_use]
    pub fn symbol(&self) -> Option<ResourceSymbol<'_>> {
        match &self.node {
            ResourceExpressionNode::Symbol {
                scope,
                name,
                domain,
                origin,
                constraints,
            } => Some(ResourceSymbol {
                scope,
                name,
                domain: *domain,
                origin,
                constraints,
            }),
            _ => None,
        }
    }

    /// Return a function, recurrence, or equation identity when present.
    #[must_use]
    pub fn function_name(&self) -> Option<&str> {
        match &self.node {
            ResourceExpressionNode::Contracted { name, .. }
            | ResourceExpressionNode::Recurrence { name, .. }
            | ResourceExpressionNode::RecurrenceEquation { name, .. } => Some(name),
            _ => None,
        }
    }

    /// Return the binder node for a bounded expression.
    #[must_use]
    pub const fn binder(&self) -> Option<ResourceExpressionId> {
        match &self.node {
            ResourceExpressionNode::BoundedSum { binder, .. }
            | ResourceExpressionNode::BoundedProduct { binder, .. } => Some(*binder),
            _ => None,
        }
    }
}

/// One symbolic nonnegative count enclosure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SymbolicCount {
    lower: ExprId,
    upper: Option<ExprId>,
    exact: bool,
    unknown: Box<[String]>,
    rule_identities: Box<[String]>,
}

/// Public bound relation implied by one native symbolic count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SymbolicCountRelation {
    /// Both endpoints denote the same exact value.
    Equality,
    /// Only a finite lower endpoint is known.
    LowerBound,
    /// Only a finite upper endpoint is semantically informative.
    UpperBound,
    /// Both finite endpoints are known but may differ.
    Enclosure,
    /// At least one unknown contribution prevents a closed enclosure.
    PartialEnclosure,
}

impl SymbolicCountRelation {
    /// Return the stable Python-facing relation spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Equality => "equality",
            Self::LowerBound => "lower_bound",
            Self::UpperBound => "upper_bound",
            Self::Enclosure => "enclosure",
            Self::PartialEnclosure => "partial_enclosure",
        }
    }
}

impl SymbolicCount {
    /// Return the lower-endpoint expression identifier.
    #[must_use]
    pub const fn lower(&self) -> ResourceExpressionId {
        self.lower
    }

    /// Return the optional upper-endpoint expression identifier.
    #[must_use]
    pub const fn upper(&self) -> Option<ResourceExpressionId> {
        self.upper
    }

    /// Return whether both endpoints denote one exact symbolic value.
    #[must_use]
    pub const fn exact(&self) -> bool {
        self.exact
    }

    /// Return stable reasons that prevent a finite exact enclosure.
    #[must_use]
    pub fn unknown(&self) -> &[String] {
        &self.unknown
    }

    /// Return checked resource-rule identities contributing to this count.
    #[must_use]
    pub fn rule_identities(&self) -> &[String] {
        &self.rule_identities
    }

    /// Classify this count using ordinary endpoint semantics.
    #[must_use]
    pub fn relation(&self) -> SymbolicCountRelation {
        if self.exact {
            SymbolicCountRelation::Equality
        } else if self.upper.is_none() && self.unknown.is_empty() {
            SymbolicCountRelation::LowerBound
        } else if !self.unknown.is_empty() || self.upper.is_none() {
            SymbolicCountRelation::PartialEnclosure
        } else {
            SymbolicCountRelation::Enclosure
        }
    }

    /// Classify this count when the metric composes as a dependency span.
    #[must_use]
    pub fn dependency_span_relation(
        &self,
        expressions: &[ResourceExpression],
    ) -> SymbolicCountRelation {
        let lower_is_zero = self
            .lower
            .index()
            .and_then(|identifier| expressions.get(identifier))
            .is_some_and(|expression| expression.literal() == Some("0"));
        if !self.exact && self.upper.is_some() && self.unknown.is_empty() && lower_is_zero {
            SymbolicCountRelation::UpperBound
        } else {
            self.relation()
        }
    }
}

/// Whole-program symbolic work and opcode counts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SymbolicLogicalCounts {
    analysis_operation_visits: u128,
    expressions: Box<[ResourceExpression]>,
    facts: Box<[SymbolicAnalysisFact]>,
    work: SymbolicCount,
    dependency_span: SymbolicCount,
    opcodes: BTreeMap<String, SymbolicCount>,
    extra_metrics: BTreeMap<String, SymbolicCount>,
    families: Option<BTreeMap<String, SymbolicCount>>,
    measurements: Option<SymbolicCount>,
    resets: Option<SymbolicCount>,
    input_qubits: Option<SymbolicCount>,
    allocated_qubits: Option<SymbolicCount>,
    circuit_qubits: Option<SymbolicCount>,
    peak_qubits: Option<SymbolicCount>,
    depths: Option<BTreeMap<String, SymbolicCount>>,
    cleanup_work: SymbolicCount,
}

/// One canonical source-local resource contribution from symbolic analysis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SymbolicAnalysisFact {
    owner: String,
    source: String,
    opcode: String,
    work: SymbolicCount,
    opcodes: BTreeMap<String, SymbolicCount>,
    extra_metrics: BTreeMap<String, SymbolicCount>,
    families: Option<BTreeMap<String, SymbolicCount>>,
    measurements: Option<SymbolicCount>,
    resets: Option<SymbolicCount>,
    cleanup_work: SymbolicCount,
}

impl SymbolicAnalysisFact {
    /// Return the function that owns this lexical operation.
    #[must_use]
    pub fn owner(&self) -> &str {
        &self.owner
    }

    /// Return the normalized source identity of this lexical operation.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Return the operation's canonical opcode.
    #[must_use]
    pub fn opcode(&self) -> &str {
        &self.opcode
    }

    /// Return this operation's local semantic-work contribution.
    #[must_use]
    pub const fn work(&self) -> &SymbolicCount {
        &self.work
    }

    /// Return this operation's local opcode contributions.
    #[must_use]
    pub fn opcodes(&self) -> &BTreeMap<String, SymbolicCount> {
        &self.opcodes
    }

    /// Return this operation's local declarative metric contributions.
    #[must_use]
    pub fn extra_metrics(&self) -> &BTreeMap<String, SymbolicCount> {
        &self.extra_metrics
    }

    /// Return this operation's local logical gate-family contributions.
    #[must_use]
    pub fn families(&self) -> Option<&BTreeMap<String, SymbolicCount>> {
        self.families.as_ref()
    }

    /// Return this operation's local measurement contribution.
    #[must_use]
    pub const fn measurements(&self) -> Option<&SymbolicCount> {
        self.measurements.as_ref()
    }

    /// Return this operation's local reset contribution.
    #[must_use]
    pub const fn resets(&self) -> Option<&SymbolicCount> {
        self.resets.as_ref()
    }

    /// Return this operation's local cleanup-work contribution.
    #[must_use]
    pub const fn cleanup_work(&self) -> &SymbolicCount {
        &self.cleanup_work
    }
}

impl SymbolicLogicalCounts {
    /// Refine portable gate count and depth from one checked fixed graph.
    ///
    /// The materialized gate count must lie inside the independently derived
    /// symbolic enclosure before refinement, so a circuit/counting
    /// disagreement leaves the existing conservative result untouched.
    pub(crate) fn refine_portable_fixed_graph(
        &mut self,
        gate_count: u128,
        makespan: u128,
    ) -> Result<bool, SymbolicResourceError> {
        let gate_key = format!("{PORTABLE_PREFIX}gates");
        let Some(gates) = self.extra_metrics.get(&gate_key) else {
            return Ok(false);
        };
        let endpoint = |identifier| {
            self.expression(identifier)
                .filter(|expression| expression.domain() == ResourceDomain::Natural)
                .and_then(ResourceExpression::literal)
                .and_then(|value| value.parse::<u128>().ok())
        };
        let Some((lower, upper)) = endpoint(gates.lower).zip(gates.upper.and_then(endpoint)) else {
            return Ok(false);
        };
        if !gates.unknown.is_empty() || !(lower..=upper).contains(&gate_count) {
            return Ok(false);
        }
        let gate_identifier = self.intern_natural_literal(gate_count)?;
        let depth_identifier = self.intern_natural_literal(makespan)?;
        self.extra_metrics
            .insert(gate_key, SymbolicCount::exact_value(gate_identifier));
        self.extra_metrics.insert(
            format!("{PORTABLE_PREFIX}gate_depth"),
            SymbolicCount::exact_value(depth_identifier),
        );
        Ok(true)
    }

    fn intern_natural_literal(
        &mut self,
        natural: u128,
    ) -> Result<ResourceExpressionId, SymbolicResourceError> {
        let value = natural.to_string();
        if let Some(index) = self.expressions.iter().position(|expression| {
            expression.domain() == ResourceDomain::Natural
                && expression.literal() == Some(value.as_str())
        }) {
            ResourceExpressionId::from_index(index)
        } else {
            let identifier = ResourceExpressionId::from_index(self.expressions.len())?;
            let mut expressions = self.expressions.to_vec();
            expressions.push(ResourceExpression::from_node(
                ResourceExpressionNode::Literal {
                    value,
                    domain: ResourceDomain::Natural,
                },
            ));
            self.expressions = expressions.into_boxed_slice();
            Ok(identifier)
        }
    }

    /// Return `QCore` operation visits made by symbolic summary traversals.
    ///
    /// This diagnostic work charge covers count, allocation, and
    /// peak-liveness traversal. It is not a logical resource metric.
    #[must_use]
    pub const fn analysis_operation_visits(&self) -> u128 {
        self.analysis_operation_visits
    }

    /// Return the topologically ordered expression DAG.
    #[must_use]
    pub fn expressions(&self) -> &[ResourceExpression] {
        &self.expressions
    }

    /// Resolve one expression identifier in this DAG.
    #[must_use]
    pub fn expression(&self, identifier: ResourceExpressionId) -> Option<&ResourceExpression> {
        identifier
            .index()
            .and_then(|index| self.expressions.get(index))
    }

    /// Return canonical source-local contribution facts.
    #[must_use]
    pub fn facts(&self) -> &[SymbolicAnalysisFact] {
        &self.facts
    }

    /// Return total logical work.
    #[must_use]
    pub const fn work(&self) -> &SymbolicCount {
        &self.work
    }

    /// Return the scalar dependency-span enclosure.
    ///
    /// This remains compositional when an opaque external call prevents the
    /// lane scheduler from proving every depth family.
    #[must_use]
    pub const fn dependency_span(&self) -> &SymbolicCount {
        &self.dependency_span
    }

    /// Return counts keyed by canonical quantum opcode.
    #[must_use]
    pub fn opcodes(&self) -> &BTreeMap<String, SymbolicCount> {
        &self.opcodes
    }

    /// Return declarative rule metrics not covered by a structural projection.
    #[must_use]
    pub fn extra_metrics(&self) -> &BTreeMap<String, SymbolicCount> {
        &self.extra_metrics
    }

    /// Return counts keyed by canonical logical gate family.
    ///
    /// Returns `None` when a coherent transform preserves semantic work but
    /// does not prove that the callee's primitive gate families are unchanged.
    #[must_use]
    pub fn families(&self) -> Option<&BTreeMap<String, SymbolicCount>> {
        self.families.as_ref()
    }

    /// Return per-qubit measurement events when their widths are representable.
    #[must_use]
    pub const fn measurements(&self) -> Option<&SymbolicCount> {
        self.measurements.as_ref()
    }

    /// Return per-qubit reset events when their widths are representable.
    #[must_use]
    pub const fn resets(&self) -> Option<&SymbolicCount> {
        self.resets.as_ref()
    }

    /// Return the entrypoint's caller-supplied logical width when representable.
    #[must_use]
    pub const fn input_qubits(&self) -> Option<&SymbolicCount> {
        self.input_qubits.as_ref()
    }

    /// Return statically reserved logical width when representable.
    #[must_use]
    pub const fn allocated_qubits(&self) -> Option<&SymbolicCount> {
        self.allocated_qubits.as_ref()
    }

    /// Return caller-supplied plus statically allocated width when representable.
    #[must_use]
    pub const fn circuit_qubits(&self) -> Option<&SymbolicCount> {
        self.circuit_qubits.as_ref()
    }

    /// Return the maximum simultaneously live logical width when proved.
    #[must_use]
    pub const fn peak_qubits(&self) -> Option<&SymbolicCount> {
        self.peak_qubits.as_ref()
    }

    /// Return symbolic dependency span and depth-family maxima when proved.
    #[must_use]
    pub fn depths(&self) -> Option<&BTreeMap<String, SymbolicCount>> {
        self.depths.as_ref()
    }

    /// Return work proven to discharge cleanup obligations.
    #[must_use]
    pub const fn cleanup_work(&self) -> &SymbolicCount {
        &self.cleanup_work
    }

    /// Return every unresolved contributor carried by the public projection.
    ///
    /// Reasons are deduplicated and sorted so strict resource analysis does not
    /// need to duplicate the compiler's metric inventory or ordering policy.
    #[must_use]
    pub fn unknown_contributors(&self) -> Vec<String> {
        fn include<'a>(unknown: &mut BTreeSet<&'a str>, count: &'a SymbolicCount) {
            unknown.extend(count.unknown().iter().map(String::as_str));
        }

        let mut unknown = BTreeSet::new();

        include(&mut unknown, &self.work);
        include(&mut unknown, &self.dependency_span);
        include(&mut unknown, &self.cleanup_work);
        for count in self.opcodes.values().chain(self.extra_metrics.values()) {
            include(&mut unknown, count);
        }
        for counts in [&self.families, &self.depths].into_iter().flatten() {
            for count in counts.values() {
                include(&mut unknown, count);
            }
        }
        for count in [
            &self.measurements,
            &self.resets,
            &self.input_qubits,
            &self.allocated_qubits,
            &self.circuit_qubits,
            &self.peak_qubits,
        ]
        .into_iter()
        .flatten()
        {
            include(&mut unknown, count);
        }

        unknown.into_iter().map(str::to_owned).collect()
    }
}

/// Failure to derive a finite symbolic logical-count enclosure.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum SymbolicResourceError {
    /// A verified construct is outside the finite symbolic subset.
    #[error("Q4211: symbolic native logical counts do not support {0}")]
    Unsupported(String),
    /// A structural reference was unexpectedly unavailable.
    #[error("Q4212: symbolic native logical analysis lost {0}")]
    Missing(String),
    /// A node identifier or exact arithmetic result cannot be represented.
    #[error("Q4213: symbolic native logical analysis overflow")]
    Overflow,
    /// A checked rule cannot be instantiated by the selected program value.
    #[error("Q4229: invalid resource-rule specialization: {0}")]
    InvalidRuleValue(String),
}

#[derive(Default)]
struct ExpressionArena {
    nodes: Vec<ResourceExpressionNode>,
    ids: HashMap<ResourceExpressionNode, ExprId>,
}

#[derive(Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
struct SymbolicMonomial(BTreeMap<ExprId, u32>);

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct SymbolicPolynomial {
    terms: BTreeMap<SymbolicMonomial, BigInt>,
}

impl SymbolicPolynomial {
    fn constant(value: BigInt) -> Self {
        let mut result = Self::default();
        if !value.is_zero() {
            result.terms.insert(SymbolicMonomial::default(), value);
        }
        result
    }

    fn atom(identifier: ExprId) -> Self {
        let mut powers = BTreeMap::new();
        powers.insert(identifier, 1);
        let mut terms = BTreeMap::new();
        terms.insert(SymbolicMonomial(powers), BigInt::from(1));
        Self { terms }
    }

    fn add_assign(&mut self, source: Self) -> Option<()> {
        for (monomial, coefficient) in source.terms {
            let combined = self.terms.get(&monomial).cloned().unwrap_or_default() + coefficient;
            if combined.is_zero() {
                self.terms.remove(&monomial);
            } else {
                self.terms.insert(monomial, combined);
            }
            if self.terms.len() > MAX_SYMBOLIC_POLYNOMIAL_TERMS {
                return None;
            }
        }
        Some(())
    }

    fn multiplied(&self, right: &Self) -> Option<Self> {
        if self.terms.is_empty() || right.terms.is_empty() {
            return Some(Self::default());
        }
        let work = self.terms.len().checked_mul(right.terms.len())?;
        if work > MAX_SYMBOLIC_POLYNOMIAL_TERMS {
            return None;
        }
        let mut result = Self::default();
        for (left_monomial, left_coefficient) in &self.terms {
            for (right_monomial, right_coefficient) in &right.terms {
                let mut powers = left_monomial.0.clone();
                for (identifier, exponent) in &right_monomial.0 {
                    let combined = powers
                        .get(identifier)
                        .copied()
                        .unwrap_or(0)
                        .checked_add(*exponent)?;
                    powers.insert(*identifier, combined);
                }
                if powers.values().try_fold(0_usize, |total, exponent| {
                    total.checked_add(usize::try_from(*exponent).ok()?)
                })? > MAX_SYMBOLIC_POLYNOMIAL_FACTORS
                {
                    return None;
                }
                let coefficient = left_coefficient * right_coefficient;
                let entry = result.terms.entry(SymbolicMonomial(powers)).or_default();
                *entry += coefficient;
            }
        }
        result.terms.retain(|_, coefficient| !coefficient.is_zero());
        (result.terms.len() <= MAX_SYMBOLIC_POLYNOMIAL_TERMS).then_some(result)
    }

    fn power(&self, exponent: u32) -> Option<Self> {
        let mut result = Self::constant(BigInt::from(1));
        for _ in 0..exponent {
            result = result.multiplied(self)?;
        }
        Some(result)
    }
}

const fn transform_mode_name(mode: TransformMode) -> &'static str {
    match mode {
        TransformMode::Direct => "direct",
        TransformMode::Controlled => "controlled",
        TransformMode::Adjoint => "adjoint",
        TransformMode::ControlledAdjoint => "controlled-adjoint",
    }
}

impl TransformMode {
    const fn adjoint(self) -> Self {
        match self {
            Self::Direct => Self::Adjoint,
            Self::Adjoint => Self::Direct,
            Self::Controlled => Self::ControlledAdjoint,
            Self::ControlledAdjoint => Self::Controlled,
        }
    }

    const fn controlled(self) -> Self {
        match self {
            Self::Direct | Self::Controlled => Self::Controlled,
            Self::Adjoint | Self::ControlledAdjoint => Self::ControlledAdjoint,
        }
    }
}

fn is_coherent_external(function: &wire::FunctionProto) -> bool {
    matches!(function.asserted_kind.as_str(), "unitary" | "isometry")
}

fn consume_live_values(live: &mut HashSet<u64>, values: &[u64]) -> bool {
    values.iter().all(|value| live.remove(value))
}

fn is_structural_owner_isomorphism(
    operation: &wire::OperationProto,
    regions: &HashMap<u64, wire::RegionRefProto>,
    live: &mut HashSet<u64>,
) -> bool {
    if !operation.regions.is_empty() {
        return false;
    }
    match operation.opcode.as_str() {
        "qcore.reshape" => {
            let ([source], [target]) =
                (operation.operands.as_slice(), operation.results.as_slice())
            else {
                return false;
            };
            if regions.get(source) != target.region.as_ref() || !live.remove(source) {
                return false;
            }
            live.insert(target.id);
        }
        "qcore.unpack" => {
            let [source] = operation.operands.as_slice() else {
                return false;
            };
            let Some(parent) = regions.get(source) else {
                return false;
            };
            if operation.results.is_empty()
                || operation.results.iter().enumerate().any(|(index, result)| {
                    let mut expected = parent.clone();
                    if operation.results.len() > 1 {
                        expected.projections.push(index.to_string());
                    }
                    result.region.as_ref() != Some(&expected)
                })
                || !live.remove(source)
            {
                return false;
            }
            live.extend(operation.results.iter().map(|result| result.id));
        }
        "qcore.tuple" => {
            let [target] = operation.results.as_slice() else {
                return false;
            };
            let Some(parent) = target.region.as_ref() else {
                return false;
            };
            if operation.operands.is_empty()
                || operation
                    .operands
                    .iter()
                    .enumerate()
                    .any(|(index, source)| {
                        let mut expected = parent.clone();
                        if operation.operands.len() > 1 {
                            expected.projections.push(index.to_string());
                        }
                        regions.get(source) != Some(&expected)
                    })
                || !consume_live_values(live, &operation.operands)
            {
                return false;
            }
            live.insert(target.id);
        }
        _ => return false,
    }
    true
}

fn is_structural_coherent_wrapper(
    function: &wire::FunctionProto,
    functions: &HashMap<&str, &wire::FunctionProto>,
    visiting: &mut HashSet<String>,
) -> bool {
    if function.external {
        return is_coherent_external(function);
    }
    if !visiting.insert(function.symbol.clone()) {
        return false;
    }
    let result = (|| {
        let body = function.body.as_ref()?;
        let quantum_parameters = function
            .parameters
            .iter()
            .filter_map(|parameter| parameter.value.as_ref().filter(|value| value.has_region))
            .collect::<Vec<_>>();
        let [quantum] = quantum_parameters.as_slice() else {
            return None;
        };
        let classical = function
            .parameters
            .iter()
            .filter_map(|parameter| parameter.value.as_ref().filter(|value| !value.has_region))
            .map(|value| value.id)
            .collect::<HashSet<_>>();
        let regions = function_value_regions(function);
        let mut live = HashSet::from([quantum.id]);
        for operation in &body.operations {
            if operation.opcode != "qcore.call" {
                if is_structural_owner_isomorphism(operation, &regions, &mut live) {
                    continue;
                }
                return None;
            }
            if !operation.regions.is_empty() {
                return None;
            }
            let callee = string_attribute(operation, "callee")
                .and_then(|symbol| functions.get(symbol).copied())?;
            let quantum_index = callee
                .parameters
                .iter()
                .enumerate()
                .filter_map(|(index, parameter)| {
                    parameter
                        .value
                        .as_ref()
                        .is_some_and(|value| value.has_region)
                        .then_some(index)
                })
                .collect::<Vec<_>>();
            let [quantum_index] = quantum_index.as_slice() else {
                return None;
            };
            let source = *operation.operands.get(*quantum_index)?;
            let [target] = operation.results.as_slice() else {
                return None;
            };
            if operation
                .operands
                .iter()
                .enumerate()
                .any(|(index, operand)| index != *quantum_index && !classical.contains(operand))
                || target.region.as_ref() != regions.get(&source)
                || !live.remove(&source)
                || !is_structural_coherent_wrapper(callee, functions, visiting)
            {
                return None;
            }
            live.insert(target.id);
        }
        let terminator = body.terminator.as_ref()?;
        let [returned] = terminator.operands.as_slice() else {
            return None;
        };
        (terminator.kind == "return" && live == HashSet::from([*returned])).then_some(())
    })()
    .is_some();
    visiting.remove(&function.symbol);
    result
}

fn apply_control_footprints(
    timing: &mut Option<Vec<SymbolicTimingNode>>,
    control_footprints: Option<&[SymbolicTimingFootprint]>,
) {
    if let (Some(nodes), Some(footprints)) = (timing.as_mut(), control_footprints) {
        add_symbolic_timing_footprints(nodes, footprints);
    } else if control_footprints.is_none() {
        *timing = None;
    }
}

const fn external_family_metrics() -> [(&'static str, &'static str); 9] {
    [
        ("gates", "qamomile.logical.gates"),
        ("single_qubit_gates", "qamomile.logical.single_qubit_gates"),
        ("two_qubit_gates", "qamomile.logical.two_qubit_gates"),
        ("multi_qubit_gates", "qamomile.logical.multi_qubit_gates"),
        ("clifford_gates", "qamomile.logical.clifford_gates"),
        ("rotation_gates", "qamomile.logical.rotation_gates"),
        ("t_gates", "qamomile.logical.t_gates"),
        ("toffoli_gates", "qamomile.logical.toffoli_gates"),
        ("non_clifford_gates", "qamomile.logical.non_clifford_gates"),
    ]
}

const fn external_depth_metrics() -> [(&'static str, &'static str); 8] {
    [
        ("gates", "qamomile.logical.gate_depth"),
        ("clifford_gates", "qamomile.logical.clifford_depth"),
        ("rotation_gates", "qamomile.logical.rotation_depth"),
        ("t_gates", "qamomile.logical.t_depth"),
        ("toffoli_gates", "qamomile.logical.toffoli_depth"),
        ("non_clifford_gates", "qamomile.logical.non_clifford_depth"),
        ("measurements", "qamomile.logical.measurement_depth"),
        ("resets", "qamomile.logical.reset_depth"),
    ]
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Counts {
    work: SymbolicCount,
    span: SymbolicCount,
    depths: Option<BTreeMap<String, SymbolicCount>>,
    opcodes: BTreeMap<String, SymbolicCount>,
    extra_metrics: BTreeMap<String, SymbolicCount>,
    families: Option<BTreeMap<String, SymbolicCount>>,
    measurements: Option<SymbolicCount>,
    resets: Option<SymbolicCount>,
    cleanup_work: SymbolicCount,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum ScalarFact {
    Int {
        expression: ExprId,
        exact: Option<i64>,
    },
    Bool(Option<bool>),
    Predicate(ExprId),
    Real {
        expression: Option<ExprId>,
        exact: Option<u64>,
    },
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum ValueFact {
    Scalar(ScalarFact),
    Aggregate(Box<[Option<Self>]>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BlockSummary {
    counts: Counts,
    exit: BlockExit,
    values: Vec<Option<ValueFact>>,
    timing: Option<Vec<SymbolicTimingNode>>,
}

/// Possible structured exits. Joining paths retains every exit until its
/// enclosing block or loop consumes it; resource bounds must not erase exits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BlockExit(u8);

impl BlockExit {
    const RETURN: Self = Self(1);
    const YIELD: Self = Self(2);
    const BREAK: Self = Self(4);
    const CONTINUE: Self = Self(8);
    const UNREACHABLE: Self = Self(16);
    const EXTERNAL: Self = Self(32);

    const fn contains(self, exit: Self) -> bool {
        self.0 & exit.0 != 0
    }

    const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    const fn without(self, exit: Self) -> Self {
        Self(self.0 & !exit.0)
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::RETURN => "return",
            Self::YIELD => "yield",
            Self::BREAK => "break",
            Self::CONTINUE => "continue",
            Self::UNREACHABLE => "unreachable",
            Self::EXTERNAL => "external",
            _ => "path_dependent",
        }
    }
}

impl From<TerminatorKind> for BlockExit {
    fn from(kind: TerminatorKind) -> Self {
        match kind {
            TerminatorKind::Return => Self::RETURN,
            TerminatorKind::Yield => Self::YIELD,
            TerminatorKind::Break => Self::BREAK,
            TerminatorKind::Continue => Self::CONTINUE,
            TerminatorKind::Unreachable => Self::UNREACHABLE,
            TerminatorKind::External => Self::EXTERNAL,
        }
    }
}

fn symbolic_block_exit(
    terminator: &wire::TerminatorProto,
) -> Result<BlockExit, SymbolicResourceError> {
    TerminatorKind::try_from(terminator.kind.as_str())
        .map(BlockExit::from)
        .map_err(|_| {
            SymbolicResourceError::Missing(format!(
                "verified terminator kind {:?}",
                terminator.kind
            ))
        })
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct SymbolicTimingRoot {
    scope: String,
    root: u64,
    projections: Vec<String>,
    instance: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SymbolicTimingFootprint {
    root: SymbolicTimingRoot,
    selectors: Option<Box<[ExprId]>>,
    width: Option<ExprId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SymbolicTimingStep {
    footprints: Box<[SymbolicTimingFootprint]>,
    durations: BTreeMap<String, ExprId>,
}

enum SymbolicTimingSelectors {
    Whole,
    Selected(Box<[ExprId]>),
    Unsupported,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum SymbolicTimingNode {
    Step(SymbolicTimingStep),
    Repeat {
        body: Box<[Self]>,
        binder: ExprId,
        trip_count: ExprId,
    },
    /// A region whose internal schedule is summarized by an upper bound.
    ///
    /// It occupies every lane of each listed root for at most `durations`
    /// steps per depth family, which keeps one unresolved region from
    /// discarding the schedule of the whole program.
    Barrier {
        roots: Box<[SymbolicTimingRoot]>,
        durations: BTreeMap<String, ExprId>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SymbolicLaneTimeline {
    barrier: ExprId,
    lanes: HashMap<ExprId, ExprId>,
    chain: Option<timing_chain::AdjacentChainFrontier>,
}

/// Compact earliest-start scheduler for one parametric realization graph.
///
/// Ports are semantic resource identities, not physical wire numbers.  A step
/// advances exactly the ports it occupies; disjoint steps therefore join by a
/// maximum.  Repetition is retained algebraically instead of materializing a
/// node per iteration.
struct ParametricDependencyScheduler {
    frontier: HashMap<u32, ExprId>,
    makespan: ExprId,
    has_steps: bool,
}

/// Gate work and dependency makespan for one parametric portable subgraph.
///
/// Both fields are upper expressions. Keeping them together prevents callers
/// from accidentally using total work as the duration of a realization node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ParametricGateProfile {
    gates: ExprId,
    makespan: ExprId,
}

impl ParametricDependencyScheduler {
    fn new(arena: &mut ExpressionArena) -> Result<Self, SymbolicResourceError> {
        let zero = arena.literal_u128(0)?;
        Ok(Self {
            frontier: HashMap::new(),
            makespan: zero,
            has_steps: false,
        })
    }

    fn step(
        &mut self,
        ports: &[u32],
        duration: ExprId,
        arena: &mut ExpressionArena,
    ) -> Result<ExprId, SymbolicResourceError> {
        if arena
            .node(duration)
            .is_none_or(|node| node.domain() != ResourceDomain::Natural)
        {
            return Err(SymbolicResourceError::Unsupported(
                "parametric schedule duration must be natural".to_owned(),
            ));
        }
        let predecessors = ports
            .iter()
            .filter_map(|port| self.frontier.get(port).copied())
            .collect::<Vec<_>>();
        let start = arena.extremum(true, predecessors)?;
        let starts_at_makespan = start == self.makespan;
        let finish = arena.add([start, duration])?;
        for port in ports {
            self.frontier.insert(*port, finish);
        }
        self.makespan = if !self.has_steps || starts_at_makespan {
            // Natural durations are nonnegative. A successor of a current
            // critical-port event cannot finish before the current makespan.
            finish
        } else {
            arena.extremum(true, [self.makespan, finish])?
        };
        self.has_steps = true;
        Ok(finish)
    }

    fn repeat_step(
        &mut self,
        ports: &[u32],
        duration: ExprId,
        repetitions: ExprId,
        arena: &mut ExpressionArena,
    ) -> Result<ExprId, SymbolicResourceError> {
        let repeated = arena.multiply([duration, repetitions])?;
        self.step(ports, repeated, arena)
    }

    const fn makespan(&self) -> ExprId {
        self.makespan
    }
}

struct SymbolicAnalyzer<'a> {
    functions: HashMap<&'a str, &'a wire::FunctionProto>,
    arena: ExpressionArena,
    rules: HashMap<(&'a str, TransformMode, &'a str), &'a ResourceRuleContract>,
    error_bound_metrics: BTreeSet<String>,
    facts: Vec<SymbolicAnalysisFact>,
    fact_keys: HashSet<(String, String, String)>,
    timing_aliases: HashMap<SymbolicTimingRoot, SymbolicTimingRoot>,
    types: &'a TypeArena,
    operations: &'a LocallyCheckedOperations,
    analysis_cache: HashMap<SymbolicAnalysisKey, BlockSummary>,
    recurrences: recurrences::RecurrenceAnalysis,
    while_contexts: usize,
    while_functions: HashSet<String>,
    operation_visits: u128,
    observation: Option<&'a ResourceObservation>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct SymbolicAnalysisKey {
    symbol: String,
    arguments: Box<[Option<ValueFact>]>,
    dimensions: Box<[(Dimension, ExprId)]>,
    mode: TransformMode,
    controls: Option<u32>,
}

struct SymbolicDimensionBindings {
    values: HashMap<Dimension, ExprId>,
}

#[derive(Clone, Copy)]
struct SymbolicFunctionContext<'a> {
    owner: &'a str,
    dimensions: &'a SymbolicDimensionBindings,
    mode: TransformMode,
    controls: Option<u32>,
}

/// Derive finite symbolic logical counts for one verified package.
pub(crate) fn symbolic_logical_counts(
    package: &wire::QCorePackageProto,
    types: &TypeArena,
    operations: &LocallyCheckedOperations,
) -> Result<SymbolicLogicalCounts, SymbolicResourceError> {
    symbolic_logical_counts_with_rules(package, types, operations, &[])
}

pub(crate) fn symbolic_logical_counts_with_rules<'a>(
    package: &'a wire::QCorePackageProto,
    types: &'a TypeArena,
    operations: &'a LocallyCheckedOperations,
    rules: &'a [ResourceRuleContract],
) -> Result<SymbolicLogicalCounts, SymbolicResourceError> {
    symbolic_observed_counts(package, types, operations, rules, None)
}

#[expect(
    clippy::too_many_lines,
    reason = "Keep analyzer initialization and final projection together"
)]
pub(crate) fn symbolic_observed_counts<'a>(
    package: &'a wire::QCorePackageProto,
    types: &'a TypeArena,
    operations: &'a LocallyCheckedOperations,
    rules: &'a [ResourceRuleContract],
    observation: Option<&'a ResourceObservation>,
) -> Result<SymbolicLogicalCounts, SymbolicResourceError> {
    let graph = crate::call_graph::build(package)
        .map_err(|error| SymbolicResourceError::Missing(error.to_string()))?;
    let indexed_rules = rules
        .iter()
        .flat_map(|rule| {
            rule.modes.iter().map(move |mode| {
                (
                    (
                        rule.callable.as_str(),
                        *mode,
                        rule.metric.identifier.as_str(),
                    ),
                    rule,
                )
            })
        })
        .collect();
    let error_bound_metrics = std::iter::once(OUTPUT_TRACE_DISTANCE_BOUND_METRIC.to_owned())
        .chain(
            rules
                .iter()
                .filter(|rule| rule.metric.composition == MetricComposition::ErrorBound)
                .map(|rule| rule.metric.identifier.clone()),
        )
        .collect();
    let mut analyzer = SymbolicAnalyzer {
        functions: package
            .functions
            .iter()
            .map(|function| (function.symbol.as_str(), function))
            .collect(),
        arena: ExpressionArena::default(),
        rules: indexed_rules,
        error_bound_metrics,
        facts: Vec::new(),
        fact_keys: HashSet::new(),
        timing_aliases: HashMap::new(),
        types,
        operations,
        analysis_cache: HashMap::new(),
        recurrences: recurrences::RecurrenceAnalysis::new(&graph, package),
        while_contexts: 0,
        while_functions: while_function_closure(package, &graph),
        operation_visits: 0,
        observation,
    };
    let entry = analyzer
        .functions
        .get(package.entrypoint.as_str())
        .copied()
        .ok_or_else(|| SymbolicResourceError::Missing("entrypoint".to_owned()))?;
    let bindings = package
        .const_bindings
        .iter()
        .filter_map(|binding| {
            binding
                .value
                .as_ref()
                .map(|value| (binding.name.as_str(), value))
        })
        .collect::<HashMap<_, _>>();
    let mut arguments = Vec::with_capacity(entry.parameters.len());
    for parameter in &entry.parameters {
        let argument = if let Some(value) = bindings.get(parameter.name.as_str()) {
            analyzer.literal_fact(value)?
        } else {
            analyzer.parameter_fact(entry, parameter)?
        };
        arguments.push(argument);
    }
    let dimensions = symbolic_dimension_bindings(entry, &arguments, &HashMap::new());
    let input_qubits = analyzer.input_width(entry, &dimensions)?;
    let mut summary = analyzer.analyze_function(entry, &arguments, &HashMap::new())?;
    if entry.external {
        analyzer.observe_call(&mut summary.counts, entry, TransformMode::Direct, Some(0))?;
    }
    analyzer.extract_recurrence_equations()?;
    let recursive = analyzer.recurrences.encountered;
    let allocated_qubits = if recursive {
        Some(SymbolicCount::unresolved(
            "recursive allocation summaries are not yet supported".to_owned(),
            &mut analyzer.arena,
        )?)
    } else {
        analyzer
            .allocated_function(entry, &arguments, &HashMap::new())
            .unwrap_or_default()
    };
    let circuit_qubits = match (&input_qubits, &allocated_qubits) {
        (Some(input), Some(allocated)) => Some(input.then(allocated, &mut analyzer.arena)?),
        _ => None,
    };
    let peak_qubits = if recursive {
        Some(SymbolicCount::unresolved(
            "recursive peak liveness requires a liveness recurrence".to_owned(),
            &mut analyzer.arena,
        )?)
    } else {
        analyzer
            .peak_function(entry, &arguments, &HashMap::new())
            .unwrap_or_default()
    };
    let (dependency_span, depths) = final_symbolic_depths(&summary, &mut analyzer.arena);
    analyzer.finish_portable_counts(&mut summary.counts)?;
    analyzer.close_recurrence_counts(&mut summary.counts)?;
    analyzer.close_recurrence_facts()?;
    let facts = exposed_analysis_facts(analyzer.facts, &summary.counts);
    let opcodes = complete_reachable_opcodes(
        summary.counts.opcodes,
        entry,
        &analyzer.functions,
        &mut analyzer.arena,
    )?;
    let expressions = analyzer.arena.into_projected_nodes();
    Ok(SymbolicLogicalCounts {
        analysis_operation_visits: analyzer.operation_visits,
        expressions,
        facts,
        work: summary.counts.work,
        dependency_span,
        opcodes,
        extra_metrics: summary.counts.extra_metrics,
        families: summary.counts.families,
        measurements: summary.counts.measurements,
        resets: summary.counts.resets,
        input_qubits,
        allocated_qubits,
        circuit_qubits,
        peak_qubits,
        depths,
        cleanup_work: summary.counts.cleanup_work,
    })
}

fn final_symbolic_depths(
    summary: &BlockSummary,
    arena: &mut ExpressionArena,
) -> (SymbolicCount, Option<BTreeMap<String, SymbolicCount>>) {
    let scheduled = summary
        .timing
        .as_deref()
        .and_then(|timing| schedule_symbolic_depth_bounds(timing, arena).ok())
        .and_then(|(maxima, exact)| {
            if exact {
                return Some(
                    maxima
                        .into_iter()
                        .map(|(family, value)| (family, SymbolicCount::exact_value(value)))
                        .collect::<BTreeMap<_, _>>(),
                );
            }
            // A conservative schedule only tightens the algebraic upper bound:
            // keep the algebraic lower bound and report the smaller upper bound.
            let mut tightened = BTreeMap::new();
            for (family, value) in maxima {
                let algebraic = if family == "span" {
                    Some(&summary.counts.span)
                } else {
                    summary
                        .counts
                        .depths
                        .as_ref()
                        .and_then(|depths| depths.get(&family))
                };
                let Some(algebraic) = algebraic.filter(|count| count.unknown.is_empty()) else {
                    continue;
                };
                let upper = match algebraic.upper {
                    Some(upper) if upper == value => value,
                    Some(upper) => arena.extremum(false, [upper, value]).ok()?,
                    None => value,
                };
                tightened.insert(
                    family,
                    SymbolicCount {
                        lower: algebraic.lower,
                        upper: Some(upper),
                        exact: algebraic.exact && algebraic.upper == Some(upper),
                        unknown: algebraic.unknown.clone(),
                        rule_identities: algebraic.rule_identities.clone(),
                    },
                );
            }
            Some(tightened)
        });
    let span = scheduled
        .as_ref()
        .and_then(|depths| depths.get("span"))
        .cloned()
        .unwrap_or_else(|| summary.counts.span.clone());
    (span, merge_symbolic_depths(&summary.counts, scheduled))
}

fn exposed_analysis_facts(
    mut facts: Vec<SymbolicAnalysisFact>,
    counts: &Counts,
) -> Box<[SymbolicAnalysisFact]> {
    for fact in &mut facts {
        fact.opcodes
            .retain(|name, _| counts.opcodes.contains_key(name));
        fact.extra_metrics
            .retain(|name, _| counts.extra_metrics.contains_key(name));
        if let Some(local) = fact.families.as_mut() {
            if let Some(exposed) = counts.families.as_ref() {
                local.retain(|name, _| exposed.contains_key(name));
            } else {
                fact.families = None;
            }
        }
        if counts.measurements.is_none() {
            fact.measurements = None;
        }
        if counts.resets.is_none() {
            fact.resets = None;
        }
    }
    facts.into_boxed_slice()
}

fn complete_reachable_opcodes(
    mut counts: BTreeMap<String, SymbolicCount>,
    entry: &wire::FunctionProto,
    functions: &HashMap<&str, &wire::FunctionProto>,
    arena: &mut ExpressionArena,
) -> Result<BTreeMap<String, SymbolicCount>, SymbolicResourceError> {
    let declared = statically_reachable_quantum_opcodes(entry, functions);
    if declared.iter().any(|opcode| !counts.contains_key(opcode)) {
        let zero = SymbolicCount::exact_value(arena.literal_u128(0)?);
        for opcode in declared {
            counts.entry(opcode).or_insert_with(|| zero.clone());
        }
    }
    Ok(counts)
}

/// Find transitive callers whose costs contain invocation-specific while symbols.
fn while_function_closure(
    package: &wire::QCorePackageProto,
    graph: &crate::call_graph::StaticCallGraph,
) -> HashSet<String> {
    let mut parents = vec![Vec::new(); package.functions.len()];
    let mut pending = Vec::new();
    for (index, function) in package.functions.iter().enumerate() {
        for callee in graph.callees(index).unwrap_or_default() {
            parents[*callee].push(index);
        }
        let mut blocks = function.body.iter().collect::<Vec<_>>();
        while let Some(block) = blocks.pop() {
            for operation in &block.operations {
                if operation.opcode == "qcore.while" {
                    pending.push(index);
                }
                for region in &operation.regions {
                    blocks.extend(&region.blocks);
                }
            }
        }
    }
    let mut visited = HashSet::new();
    while let Some(index) = pending.pop() {
        if visited.insert(index) {
            pending.extend(&parents[index]);
        }
    }
    visited
        .into_iter()
        .map(|index| package.functions[index].symbol.clone())
        .collect()
}

fn statically_reachable_quantum_opcodes(
    entry: &wire::FunctionProto,
    functions: &HashMap<&str, &wire::FunctionProto>,
) -> BTreeSet<String> {
    let mut opcodes = BTreeSet::new();
    let mut visited = HashSet::from([entry.symbol.as_str()]);
    let mut pending = entry.body.iter().collect::<Vec<_>>();
    while let Some(block) = pending.pop() {
        for operation in &block.operations {
            if operation.opcode.starts_with("quantum.")
                || matches!(
                    operation.opcode.as_str(),
                    "qcore.phase_by" | "qcore.xor_by" | "qcore.permute_by"
                )
            {
                opcodes.insert(operation.opcode.clone());
            } else if operation.opcode == "qcore.adjoint"
                && string_attribute(operation, "original").is_some_and(|name| {
                    name.starts_with("quantum.")
                        || matches!(
                            name,
                            "qcore.control_call"
                                | "qcore.phase_by"
                                | "qcore.xor_by"
                                | "qcore.permute_by"
                        )
                })
            {
                opcodes.insert("qcore.adjoint".to_owned());
            }
            let callees = if operation.opcode == "qcore.select_call" {
                string_tuple_attribute(operation, "callees").unwrap_or_default()
            } else if matches!(
                operation.opcode.as_str(),
                "qcore.call" | "qcore.adjoint_call" | "qcore.control_call"
            ) {
                string_attribute(operation, "callee").into_iter().collect()
            } else {
                Vec::new()
            };
            for callee in callees {
                if let Some(function) = functions.get(callee).copied()
                    && visited.insert(function.symbol.as_str())
                {
                    pending.extend(function.body.iter());
                }
            }
            pending.extend(
                operation
                    .regions
                    .iter()
                    .flat_map(|region| region.blocks.iter()),
            );
        }
    }
    opcodes
}

fn merge_symbolic_depths(
    counts: &Counts,
    scheduled: Option<BTreeMap<String, SymbolicCount>>,
) -> Option<BTreeMap<String, SymbolicCount>> {
    counts.depths.clone().map(|mut depths| {
        depths.insert("span".to_owned(), counts.span.clone());
        if let Some(scheduled) = scheduled {
            depths.extend(scheduled);
        }
        depths
    })
}

#[derive(Clone, Copy)]
struct IntFact {
    expression: ExprId,
    exact: Option<i64>,
}

fn int_fact(value: Option<&ValueFact>) -> Option<IntFact> {
    match value? {
        ValueFact::Scalar(ScalarFact::Int { expression, exact }) => Some(IntFact {
            expression: *expression,
            exact: *exact,
        }),
        ValueFact::Scalar(_) | ValueFact::Aggregate(_) => None,
    }
}

fn scalar_expression(value: &ValueFact) -> Option<(ExprId, bool)> {
    match value {
        ValueFact::Scalar(ScalarFact::Int { expression, .. }) => Some((*expression, true)),
        ValueFact::Scalar(ScalarFact::Real {
            expression: Some(expression),
            ..
        }) => Some((*expression, false)),
        ValueFact::Scalar(_) | ValueFact::Aggregate(_) => None,
    }
}

fn symbolic_dimension_bindings(
    function: &wire::FunctionProto,
    arguments: &[Option<ValueFact>],
    inherited: &HashMap<Dimension, ExprId>,
) -> SymbolicDimensionBindings {
    let mut values = inherited.clone();
    values.extend(
        function
            .parameters
            .iter()
            .zip(arguments)
            .filter_map(|(parameter, value)| {
                int_fact(value.as_ref()).map(|fact| {
                    (
                        Dimension::Symbol {
                            scope: Arc::from(function.symbol.as_str()),
                            name: Arc::from(parameter.name.as_str()),
                        },
                        fact.expression,
                    )
                })
            }),
    );
    SymbolicDimensionBindings { values }
}

fn bool_fact(value: &ValueFact) -> Option<bool> {
    match value {
        ValueFact::Scalar(ScalarFact::Bool(value)) => *value,
        _ => None,
    }
}

fn exact_static_value(value: &ValueFact, arena: &ExpressionArena) -> Option<StaticValue> {
    match value {
        ValueFact::Scalar(ScalarFact::Int {
            exact: Some(value), ..
        }) => Some(StaticValue::Scalar(StaticScalar::Int(*value))),
        ValueFact::Scalar(ScalarFact::Int {
            expression,
            exact: None,
        }) => {
            let ResourceExpressionNode::Literal {
                value,
                domain: ResourceDomain::Natural | ResourceDomain::Integer,
            } = arena.node(*expression)?
            else {
                return None;
            };
            let value = value.parse::<BigInt>().ok()?;
            Some(StaticValue::BigInt(value.to_signed_bytes_be().into()))
        }
        ValueFact::Scalar(ScalarFact::Bool(Some(value))) => {
            Some(StaticValue::Scalar(StaticScalar::Bool(*value)))
        }
        ValueFact::Scalar(ScalarFact::Real {
            exact: Some(value), ..
        }) => Some(StaticValue::Scalar(StaticScalar::RealBits(*value))),
        ValueFact::Aggregate(values) => Some(StaticValue::Aggregate(
            values
                .iter()
                .map(|value| {
                    value
                        .as_ref()
                        .and_then(|value| exact_static_value(value, arena))
                })
                .collect::<Vec<_>>()
                .into(),
        )),
        ValueFact::Scalar(_) => None,
    }
}

fn bind_results(
    operation: &wire::OperationProto,
    environment: &mut HashMap<u64, Option<ValueFact>>,
    value: Option<&ValueFact>,
) {
    for result in &operation.results {
        environment.insert(result.id, value.cloned());
    }
}

fn bind_exact_results(
    operation: &wire::OperationProto,
    environment: &mut HashMap<u64, Option<ValueFact>>,
    values: &[Option<ValueFact>],
) -> Result<(), SymbolicResourceError> {
    if operation.results.len() != values.len() {
        return Err(SymbolicResourceError::Missing(format!(
            "result ABI for {:?}",
            operation.opcode
        )));
    }
    for (result, value) in operation.results.iter().zip(values) {
        environment.insert(result.id, value.clone());
    }
    Ok(())
}

fn bind_unpack(
    operation: &wire::OperationProto,
    environment: &mut HashMap<u64, Option<ValueFact>>,
) -> Result<(), SymbolicResourceError> {
    let values = operation
        .operands
        .first()
        .and_then(|operand| environment.get(operand))
        .and_then(Option::as_ref)
        .and_then(|value| match value {
            ValueFact::Aggregate(values) => Some(values.to_vec()),
            ValueFact::Scalar(_) => None,
        })
        .filter(|values| values.len() == operation.results.len())
        .unwrap_or_else(|| vec![None; operation.results.len()]);
    bind_exact_results(operation, environment, &values)
}

fn first_operand(
    operation: &wire::OperationProto,
    environment: &HashMap<u64, Option<ValueFact>>,
) -> Option<ValueFact> {
    operation
        .operands
        .first()
        .and_then(|operand| environment.get(operand))
        .cloned()
        .flatten()
}

fn bind_region(
    block: &wire::BlockProto,
    operands: &[u64],
    environment: &HashMap<u64, Option<ValueFact>>,
    induction: bool,
    facts: Option<(ExprId, &[bool])>,
) -> Result<HashMap<u64, Option<ValueFact>>, SymbolicResourceError> {
    let invariants = facts.map(|(_, invariants)| invariants);
    let mut values = Vec::with_capacity(operands.len() + usize::from(induction));
    if induction {
        let expression = facts
            .map(|(expression, _)| expression)
            .ok_or_else(|| SymbolicResourceError::Missing("loop induction fact".to_owned()))?;
        values.push(Some(ValueFact::Scalar(ScalarFact::Int {
            expression,
            exact: None,
        })));
    }
    values.extend(operands.iter().enumerate().map(|(index, operand)| {
        if invariants.is_none_or(|items| items.get(index).copied().unwrap_or(false)) {
            environment.get(operand).cloned().flatten()
        } else {
            None
        }
    }));
    bind_values(block, values)
}

fn bind_values(
    block: &wire::BlockProto,
    values: Vec<Option<ValueFact>>,
) -> Result<HashMap<u64, Option<ValueFact>>, SymbolicResourceError> {
    if block.arguments.len() != values.len() {
        return Err(SymbolicResourceError::Missing(
            "structured region argument ABI".to_owned(),
        ));
    }
    Ok(block
        .arguments
        .iter()
        .zip(values)
        .map(|(argument, value)| (argument.id, value))
        .collect())
}

fn invariant_values(
    operands: &[u64],
    environment: &HashMap<u64, Option<ValueFact>>,
    invariants: &[bool],
) -> Vec<Option<ValueFact>> {
    operands
        .iter()
        .zip(invariants)
        .map(|(operand, invariant)| {
            invariant
                .then(|| environment.get(operand).cloned().flatten())
                .flatten()
        })
        .collect()
}

fn join_values<'a>(
    branches: impl Iterator<Item = &'a [Option<ValueFact>]>,
) -> Result<Vec<Option<ValueFact>>, SymbolicResourceError> {
    let values = branches.collect::<Vec<_>>();
    let Some(first) = values.first() else {
        return Ok(Vec::new());
    };
    if values.iter().any(|items| items.len() != first.len()) {
        return Err(SymbolicResourceError::Missing(
            "branch result ABI".to_owned(),
        ));
    }
    Ok((0..first.len())
        .map(|index| {
            let value = first[index].clone();
            values
                .iter()
                .all(|items| items[index] == value)
                .then_some(value)
                .flatten()
        })
        .collect())
}

fn project_aggregate(
    operation: &wire::OperationProto,
    environment: &HashMap<u64, Option<ValueFact>>,
) -> Option<ValueFact> {
    let source = operation.operands.first()?;
    let ValueFact::Aggregate(values) = environment.get(source)?.as_ref()? else {
        return None;
    };
    let index = if operation.opcode == "classical.index" {
        let selector = operation.operands.get(1)?;
        int_fact(environment.get(selector)?.as_ref())?.exact?
    } else {
        int_attribute(operation, "position")?
    };
    aggregate_index(values, index)
}

fn aggregate_index(values: &[Option<ValueFact>], index: i64) -> Option<ValueFact> {
    let normalized = normalize_aggregate_index(values.len(), &num_bigint::BigInt::from(index))?;
    values.get(normalized)?.clone()
}

fn scalar_type(source: Option<&wire::TypeProto>) -> Option<(&str, Option<&str>)> {
    use wire::type_proto::Kind;

    let mut current = source?;
    let mut stage = None;
    while current.kind() == Kind::Staged {
        stage = Some(current.name.as_str());
        let [inner] = current.arguments.as_slice() else {
            return None;
        };
        current = inner;
    }
    (current.kind() == Kind::Scalar).then_some((current.name.as_str(), stage))
}

fn attribute<'a>(
    operation: &'a wire::OperationProto,
    name: &str,
) -> Option<&'a wire::LiteralProto> {
    operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .and_then(|attribute| attribute.value.as_ref())
}

fn coherent_callees(operation: &wire::OperationProto) -> Result<Vec<&str>, SymbolicResourceError> {
    if operation.opcode == "qcore.control_call" {
        Ok(vec![string_attribute(operation, "callee").ok_or_else(
            || SymbolicResourceError::Missing("control callee".to_owned()),
        )?])
    } else {
        string_tuple_attribute(operation, "callees")
            .ok_or_else(|| SymbolicResourceError::Missing("select callees".to_owned()))
    }
}

fn string_attribute<'a>(operation: &'a wire::OperationProto, name: &str) -> Option<&'a str> {
    use wire::literal_proto::Value;

    match attribute(operation, name)?.value.as_ref()? {
        Value::StringValue(value) => Some(value),
        _ => None,
    }
}

fn bool_attribute(operation: &wire::OperationProto, name: &str) -> Option<bool> {
    use wire::literal_proto::Value;

    match attribute(operation, name)?.value.as_ref()? {
        Value::BoolValue(value) => Some(*value),
        _ => None,
    }
}

fn int_attribute(operation: &wire::OperationProto, name: &str) -> Option<i64> {
    use wire::literal_proto::Value;

    match attribute(operation, name)?.value.as_ref()? {
        Value::IntValue(value) => Some(*value),
        _ => None,
    }
}

fn string_tuple_attribute<'a>(
    operation: &'a wire::OperationProto,
    name: &str,
) -> Option<Vec<&'a str>> {
    use wire::literal_proto::Value;

    let Value::TupleValue(sequence) = attribute(operation, name)?.value.as_ref()? else {
        return None;
    };
    sequence
        .values
        .iter()
        .map(|value| match value.value.as_ref()? {
            Value::StringValue(value) => Some(value.as_str()),
            _ => None,
        })
        .collect()
}

fn mapping_string_tuple_attribute<'a>(
    operation: &'a wire::OperationProto,
    attribute_name: &str,
    key: &str,
) -> Option<Vec<&'a str>> {
    use wire::literal_proto::Value;

    let Value::MappingValue(mapping) = attribute(operation, attribute_name)?.value.as_ref()? else {
        return None;
    };
    let Value::TupleValue(sequence) = mapping
        .entries
        .iter()
        .find(|entry| entry.key == key)?
        .value
        .as_ref()?
        .value
        .as_ref()?
    else {
        return None;
    };
    sequence
        .values
        .iter()
        .map(|value| match value.value.as_ref()? {
            Value::StringValue(value) => Some(value.as_str()),
            _ => None,
        })
        .collect()
}

fn mapping_string_attribute<'a>(
    operation: &'a wire::OperationProto,
    attribute_name: &str,
    key: &str,
) -> Option<&'a str> {
    use wire::literal_proto::Value;

    let Value::MappingValue(mapping) = attribute(operation, attribute_name)?.value.as_ref()? else {
        return None;
    };
    match mapping
        .entries
        .iter()
        .find(|entry| entry.key == key)?
        .value
        .as_ref()?
        .value
        .as_ref()?
    {
        Value::StringValue(value) => Some(value),
        _ => None,
    }
}

fn span_identity(operation: &wire::OperationProto) -> String {
    operation.span.as_ref().map_or_else(
        || "unknown:0:0".to_owned(),
        |span| format!("{}:{}:{}", span.uri, span.start_line, span.start_column),
    )
}
