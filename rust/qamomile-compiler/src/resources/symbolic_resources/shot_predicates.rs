//! Bounded proof of correlations in one logical trial's ambient shot context.
//!
//! Only total integer predicates over the private shot handle are admitted.
//! Boolean assignments over their canonical atoms overapproximate executions;
//! they never enumerate shot indices. Counts keep ordinary piecewise expressions
//! until composition finishes, then publish their marginal bounds. Recognition,
//! traversal, temporary storage, and rewriting share one precharged budget.
//! The limits are eight atoms, 1,048,576 work units, and 262,144 storage units.
//! Units include graph edges and cloned declaration bytes, not just cache keys.
//! Unsupported conditions use ordinary branch enclosures; a failed admitted
//! proof preserves provenance and a guarded lower bound for signed metrics.

use std::collections::{BTreeMap, HashMap};

use crate::resource_rules::MetricContract;

mod projection;
use projection::Projection;

use super::{
    ExprId, ExpressionArena, MetricComposition, PORTABLE_PREFIX, ResourceDomain,
    ResourceExpressionNode as Node, ResourcePredicateKind as Predicate, ScalarFact,
    SymbolicAnalyzer, SymbolicCount, SymbolicLogicalCounts, SymbolicResourceError, ValueFact,
};

const MAX_ATOMS: usize = 8;
const MAX_WORK: usize = 1_048_576;
const MAX_STORAGE: usize = 262_144;
const BUDGET_REASON: &str = "shot predicate analysis budget exceeded";
const TOTALITY_REASON: &str = "shot predicate projection cannot prove the remaining cost total";
const COMPOSITION_REASON: &str =
    "custom metric composition is unproved in an analysis with current shot context";

#[cfg(test)]
mod tests;

pub(super) fn unproved_metric_compositions<'a>(
    metrics: impl IntoIterator<Item = &'a MetricContract>,
) -> Result<BTreeMap<String, ResourceDomain>, SymbolicResourceError> {
    let mut declarations = BTreeMap::new();
    for metric in metrics {
        let definition = (metric.domain, metric.composition);
        if declarations
            .insert(metric.identifier.as_str(), definition)
            .is_some_and(|previous| previous != definition)
        {
            return Err(SymbolicResourceError::Unsupported(format!(
                "conflicting domain or composition declarations for resource metric {:?}",
                metric.identifier
            )));
        }
    }
    Ok(declarations
        .into_iter()
        .filter(|(identifier, (_, composition))| {
            matches!(
                composition,
                MetricComposition::DependencySpan | MetricComposition::PeakLiveness
            ) && *identifier != format!("{PORTABLE_PREFIX}clean_ancilla_qubits")
        })
        .map(|(identifier, (domain, _))| (identifier.to_owned(), domain))
        .collect())
}

#[derive(Clone, Copy)]
pub(super) struct CurrentShot(ExprId);

#[derive(Clone, Copy)]
struct Truth([u64; 4]);

impl Truth {
    const fn constant(value: bool) -> Self {
        Self([if value { u64::MAX } else { 0 }; 4])
    }

    fn atom(index: usize) -> Self {
        let mut result = Self::constant(false);
        for assignment in 0..256 {
            if assignment & (1 << index) != 0 {
                result.0[assignment / 64] |= 1 << (assignment % 64);
            }
        }
        result
    }

    fn get(self, assignment: u8) -> bool {
        self.0[usize::from(assignment) / 64] & (1 << (assignment % 64)) != 0
    }

    fn not(self) -> Self {
        Self(self.0.map(|word| !word))
    }

    fn combine(self, other: Self, conjunction: bool) -> Self {
        Self(std::array::from_fn(|index| {
            if conjunction {
                self.0[index] & other.0[index]
            } else {
                self.0[index] | other.0[index]
            }
        }))
    }
}

struct Budget {
    work: usize,
    storage: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            work: MAX_WORK,
            storage: MAX_STORAGE,
        }
    }
}

impl Budget {
    fn charge(&mut self, work: usize, storage: usize) -> Result<(), &'static str> {
        let remaining_work = self.work.checked_sub(work).ok_or(BUDGET_REASON)?;
        let remaining_storage = self.storage.checked_sub(storage).ok_or(BUDGET_REASON)?;
        self.work = remaining_work;
        self.storage = remaining_storage;
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Proof {
    Integer,
    Predicate { expression: ExprId, truth: Truth },
    Unsupported,
}

#[derive(Default)]
pub(super) struct ShotPredicates {
    budget: Budget,
    proofs: HashMap<ExprId, Proof>,
    atoms: HashMap<(Predicate, [ExprId; 2]), usize>,
    predicates: HashMap<ExprId, Truth>,
}

impl SymbolicAnalyzer<'_> {
    pub(super) fn current_shot_fact(&mut self) -> Result<ValueFact, SymbolicResourceError> {
        let shot = if let Some(shot) = self.current_shot {
            shot
        } else {
            let shot = CurrentShot(self.arena.current_shot()?);
            self.current_shot = Some(shot);
            shot
        };
        Ok(ValueFact::Scalar(ScalarFact::Int {
            expression: shot.0,
            exact: None,
        }))
    }

    pub(super) fn certified_shot_predicate(
        &mut self,
        predicate: ExprId,
    ) -> Result<Option<ExprId>, SymbolicResourceError> {
        let Some(shot) = self.current_shot else {
            return Ok(None);
        };
        self.shot_predicates
            .recognize(predicate, shot, &mut self.arena)
    }
}

impl ShotPredicates {
    fn recognize(
        &mut self,
        root: ExprId,
        shot: CurrentShot,
        arena: &mut ExpressionArena,
    ) -> Result<Option<ExprId>, SymbolicResourceError> {
        let boundary = arena.nodes.len();
        if self.budget.charge(1, 1).is_err() {
            return Ok(None);
        }
        let mut pending = vec![(root, false)];
        while let Some((identifier, expanded)) = pending.pop() {
            if self.proofs.contains_key(&identifier) {
                continue;
            }
            let Some(node) = original_node(arena, identifier, boundary) else {
                return Ok(None);
            };
            if self
                .budget
                .charge(node_weight(node), node_weight(node))
                .is_err()
            {
                return Ok(None);
            }
            if !expanded {
                pending.push((identifier, true));
                pending.extend(node.dependencies().into_iter().map(|child| (child, false)));
                continue;
            }
            let node = node.clone();
            let proof = self.prove_node(identifier, &node, shot, arena)?;
            self.proofs.insert(identifier, proof);
        }
        Ok(match self.proofs.get(&root) {
            Some(Proof::Predicate { expression, .. }) => Some(*expression),
            _ => None,
        })
    }

    fn prove_node(
        &mut self,
        identifier: ExprId,
        node: &Node,
        shot: CurrentShot,
        arena: &mut ExpressionArena,
    ) -> Result<Proof, SymbolicResourceError> {
        let integers = |arguments: &[ExprId]| {
            arguments
                .iter()
                .all(|argument| matches!(self.proofs.get(argument), Some(Proof::Integer)))
        };
        match node {
            Node::Symbol { .. } if identifier == shot.0 => Ok(Proof::Integer),
            Node::Literal {
                domain: ResourceDomain::Natural | ResourceDomain::Integer,
                ..
            } => Ok(Proof::Integer),
            Node::Add { arguments, .. } | Node::Multiply { arguments, .. }
                if integers(arguments) =>
            {
                Ok(Proof::Integer)
            }
            Node::FloorDivide { arguments }
                if integers(arguments.as_slice()) && positive_literal(arena, arguments[1]) =>
            {
                Ok(Proof::Integer)
            }
            Node::Contracted {
                name, arguments, ..
            } if matches!(name.as_str(), "qamomile.modulo" | "qamomile.floor_divide")
                && arguments.len() == 2
                && integers(arguments)
                && positive_literal(arena, arguments[1]) =>
            {
                Ok(Proof::Integer)
            }
            Node::PredicateLiteral(value) => {
                Ok(self.remember_predicate(identifier, Truth::constant(*value)))
            }
            Node::Predicate {
                kind,
                arguments,
                predicates,
            } if arguments.is_empty() => {
                let mut truth = Truth::constant(*kind == Predicate::And);
                let mut normalized = Vec::with_capacity(predicates.len());
                for predicate in predicates {
                    let Some(Proof::Predicate {
                        expression,
                        truth: child,
                    }) = self.proofs.get(predicate)
                    else {
                        return Ok(Proof::Unsupported);
                    };
                    normalized.push(*expression);
                    truth = match kind {
                        Predicate::And => truth.combine(*child, true),
                        Predicate::Or => truth.combine(*child, false),
                        Predicate::Not if predicates.len() == 1 => child.not(),
                        _ => return Ok(Proof::Unsupported),
                    };
                }
                let expression = arena.predicate_connective(*kind, normalized)?;
                Ok(self.remember_predicate(expression, truth))
            }
            Node::Predicate {
                kind,
                arguments,
                predicates,
            } if predicates.is_empty() && arguments.len() == 2 && integers(arguments) => {
                self.comparison(*kind, [arguments[0], arguments[1]], arena)
            }
            _ => Ok(Proof::Unsupported),
        }
    }

    fn remember_predicate(&mut self, expression: ExprId, truth: Truth) -> Proof {
        if self.budget.charge(1, 1).is_err() {
            return Proof::Unsupported;
        }
        self.predicates.insert(expression, truth);
        Proof::Predicate { expression, truth }
    }

    fn comparison(
        &mut self,
        kind: Predicate,
        mut arguments: [ExprId; 2],
        arena: &mut ExpressionArena,
    ) -> Result<Proof, SymbolicResourceError> {
        let (canonical, negated) = match kind {
            Predicate::Equal | Predicate::NotEqual => {
                arguments.sort_unstable();
                (Predicate::Equal, kind == Predicate::NotEqual)
            }
            Predicate::Less | Predicate::GreaterEqual => {
                (Predicate::Less, kind == Predicate::GreaterEqual)
            }
            Predicate::Greater | Predicate::LessEqual => {
                arguments.swap(0, 1);
                (Predicate::Less, kind == Predicate::LessEqual)
            }
            _ => return Ok(Proof::Unsupported),
        };
        if arguments[0] == arguments[1] {
            let value = (canonical == Predicate::Equal) != negated;
            let expression = arena.predicate_literal(value)?;
            return Ok(self.remember_predicate(expression, Truth::constant(value)));
        }
        let key = (canonical, arguments);
        let atom = if let Some(atom) = self.atoms.get(&key) {
            *atom
        } else {
            if self.atoms.len() == MAX_ATOMS || self.budget.charge(256, 1).is_err() {
                return Ok(Proof::Unsupported);
            }
            let atom = self.atoms.len();
            self.atoms.insert(key, atom);
            atom
        };
        let mut expression = arena.predicate_comparison(canonical, arguments)?;
        let mut truth = arena
            .predicate_value(expression)
            .map_or_else(|| Truth::atom(atom), Truth::constant);
        if negated {
            expression = arena.predicate_connective(Predicate::Not, [expression])?;
            truth = truth.not();
        }
        Ok(self.remember_predicate(expression, truth))
    }

    pub(super) fn project(
        &mut self,
        result: &mut SymbolicLogicalCounts,
        unproved_composition_metrics: &BTreeMap<String, ResourceDomain>,
        arena: &mut ExpressionArena,
    ) -> Result<(), SymbolicResourceError> {
        if arena.private_shot.is_some() {
            for metrics in std::iter::once(&mut result.extra_metrics)
                .chain(result.facts.iter_mut().map(|fact| &mut fact.extra_metrics))
            {
                for (name, count) in metrics {
                    if let Some(domain) = unproved_composition_metrics.get(name) {
                        fence_unproved_composition(name, *domain, count, arena)?;
                    }
                }
            }
        }
        if self.atoms.is_empty() {
            return Ok(());
        }
        let boundary = arena.nodes.len();
        let mut projection = Projection::new(self, boundary);
        for count in [
            &mut result.work,
            &mut result.dependency_span,
            &mut result.cleanup_work,
        ] {
            projection.count(count, arena)?;
        }
        for count in result.opcodes.values_mut() {
            projection.count(count, arena)?;
        }
        for (name, count) in &mut result.extra_metrics {
            if !unproved_composition_metrics.contains_key(name) {
                projection.count(count, arena)?;
            }
        }
        for counts in [&mut result.families, &mut result.depths]
            .into_iter()
            .flatten()
        {
            for count in counts.values_mut() {
                projection.count(count, arena)?;
            }
        }
        for count in [
            &mut result.measurements,
            &mut result.resets,
            &mut result.input_qubits,
            &mut result.allocated_qubits,
            &mut result.circuit_qubits,
            &mut result.peak_qubits,
        ]
        .into_iter()
        .flatten()
        {
            projection.count(count, arena)?;
        }
        for fact in &mut result.facts {
            for count in [&mut fact.work, &mut fact.cleanup_work] {
                projection.count(count, arena)?;
            }
            for count in fact.opcodes.values_mut() {
                projection.count(count, arena)?;
            }
            for (name, count) in &mut fact.extra_metrics {
                if !unproved_composition_metrics.contains_key(name) {
                    projection.count(count, arena)?;
                }
            }
            if let Some(families) = &mut fact.families {
                for count in families.values_mut() {
                    projection.count(count, arena)?;
                }
            }
            for count in [&mut fact.measurements, &mut fact.resets]
                .into_iter()
                .flatten()
            {
                projection.count(count, arena)?;
            }
        }
        Ok(())
    }
}

fn fence_unproved_composition(
    name: &str,
    domain: ResourceDomain,
    count: &mut SymbolicCount,
    arena: &mut ExpressionArena,
) -> Result<(), SymbolicResourceError> {
    if !matches!(
        domain,
        ResourceDomain::Natural | ResourceDomain::Probability
    ) {
        return Err(SymbolicResourceError::Unsupported(format!(
            "{COMPOSITION_REASON} for {name:?}; its declared domain has no proved lower bound"
        )));
    }
    let lower = arena.intern(Node::Literal {
        value: "0".to_owned(),
        domain,
    })?;
    *count = SymbolicCount::invalidated(count, lower, COMPOSITION_REASON);
    Ok(())
}

fn original_node(arena: &ExpressionArena, identifier: ExprId, boundary: usize) -> Option<&Node> {
    (identifier.index()? < boundary)
        .then(|| arena.node(identifier))
        .flatten()
}

fn positive_literal(arena: &ExpressionArena, identifier: ExprId) -> bool {
    matches!(arena.node(identifier), Some(Node::Literal {
        value, domain: ResourceDomain::Natural | ResourceDomain::Integer,
    }) if value != "0" && !value.starts_with('-'))
}

fn node_weight(node: &Node) -> usize {
    let mut weight = 4usize;
    match node {
        Node::Literal { value, .. } => weight += value.len(),
        Node::RationalLiteral {
            numerator,
            denominator,
            ..
        } => weight += numerator.len() + denominator.len(),
        Node::Symbol {
            scope,
            name,
            origin,
            constraints,
            ..
        } => {
            weight += scope.len()
                + name.len()
                + origin.len()
                + constraints.len()
                + constraints.iter().map(String::len).sum::<usize>();
        }
        _ => {}
    }
    match node {
        Node::Add { arguments, .. }
        | Node::Multiply { arguments, .. }
        | Node::Power { arguments, .. }
        | Node::Minimum { arguments, .. }
        | Node::Maximum { arguments, .. }
        | Node::Logarithm { arguments }
        | Node::RecurrenceSystem { arguments, .. } => weight += arguments.len(),
        Node::Contracted {
            name, arguments, ..
        }
        | Node::Recurrence {
            name, arguments, ..
        }
        | Node::RecurrenceEquation {
            name, arguments, ..
        } => weight += name.len() + arguments.len(),
        Node::Predicate {
            arguments,
            predicates,
            ..
        }
        | Node::Piecewise {
            arguments,
            predicates,
            ..
        } => weight += arguments.len() + predicates.len(),
        _ => {}
    }
    weight
}
