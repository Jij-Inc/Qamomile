//! Lazy, bounded cofactoring and totality checks for published resource bounds.

use super::super::{
    ExprId, ExpressionArena, ResourceDomain, ResourceExpressionNode as Node,
    ResourcePredicateKind as Predicate, SymbolicCount, SymbolicResourceError,
};
use super::{
    BUDGET_REASON, Budget, ShotPredicates, TOTALITY_REASON, Truth, node_weight, original_node,
    positive_literal,
};
use std::collections::HashMap;

#[derive(Clone, Copy)]
struct Rewritten {
    expression: ExprId,
    total: bool,
}

pub(super) struct Projection<'a> {
    predicates: &'a HashMap<ExprId, Truth>,
    budget: &'a mut Budget,
    boundary: usize,
    assignments: usize,
    affected: HashMap<ExprId, bool>,
    rewritten: HashMap<(ExprId, u8), Rewritten>,
}

impl<'a> Projection<'a> {
    pub(super) fn new(proofs: &'a mut ShotPredicates, boundary: usize) -> Self {
        Self {
            predicates: &proofs.predicates,
            budget: &mut proofs.budget,
            boundary,
            assignments: 1 << proofs.atoms.len(),
            affected: HashMap::new(),
            rewritten: HashMap::new(),
        }
    }
}

impl Projection<'_> {
    pub(super) fn count(
        &mut self,
        count: &mut SymbolicCount,
        arena: &mut ExpressionArena,
    ) -> Result<(), SymbolicResourceError> {
        let result = self.project_count(count, arena);
        match result {
            Ok(Some(projected)) => *count = projected,
            Ok(None) => {}
            Err(ProjectionError::Proof(reason)) => {
                let nonnegative = arena.node(count.lower).is_some_and(|node| {
                    matches!(
                        node.domain(),
                        ResourceDomain::Natural | ResourceDomain::Probability
                    )
                });
                *count = if nonnegative {
                    SymbolicCount::unresolved_from(count, reason.to_owned(), arena)?
                } else {
                    SymbolicCount::invalidated(count, count.lower, reason)
                };
            }
            Err(ProjectionError::Resource(error)) => return Err(error),
        }
        Ok(())
    }

    fn project_count(
        &mut self,
        count: &SymbolicCount,
        arena: &mut ExpressionArena,
    ) -> Result<Option<SymbolicCount>, ProjectionError> {
        let affected = self.is_affected(count.lower, arena)?
            | count
                .upper
                .map(|upper| self.is_affected(upper, arena))
                .transpose()?
                .unwrap_or(false);
        if !affected {
            return Ok(None);
        }
        self.budget.charge(self.assignments, self.assignments * 2)?;
        let mut lower = Vec::with_capacity(self.assignments);
        let mut upper = Vec::with_capacity(self.assignments);
        for assignment in 0..self.assignments {
            let assignment = u8::try_from(assignment).map_err(|_| BUDGET_REASON)?;
            let value = self.rewrite(count.lower, assignment, arena)?;
            if !value.total {
                return Err(TOTALITY_REASON.into());
            }
            lower.push(value.expression);
            if let Some(endpoint) = count.upper {
                let value = self.rewrite(endpoint, assignment, arena)?;
                if !value.total {
                    return Err(TOTALITY_REASON.into());
                }
                upper.push(value.expression);
            }
        }
        let exact = count.exact
            && count.unknown.is_empty()
            && !upper.is_empty()
            && lower.iter().chain(&upper).all(|value| *value == lower[0]);
        self.budget
            .charge(lower.len() + upper.len(), lower.len() + upper.len())?;
        Ok(Some(SymbolicCount {
            lower: arena.extremum(false, lower)?,
            upper: count
                .upper
                .map(|_| arena.extremum(true, upper))
                .transpose()?,
            exact,
            unknown: count.unknown.clone(),
            rule_identities: count.rule_identities.clone(),
        }))
    }

    fn is_affected(
        &mut self,
        root: ExprId,
        arena: &ExpressionArena,
    ) -> Result<bool, ProjectionError> {
        if let Some(affected) = self.affected.get(&root) {
            return Ok(*affected);
        }
        if matches!(
            original_node(arena, root, self.boundary),
            Some(
                Node::Literal { .. }
                    | Node::RationalLiteral { .. }
                    | Node::RealLiteral { .. }
                    | Node::Symbol { .. }
                    | Node::PredicateLiteral(_)
            )
        ) {
            return Ok(false);
        }
        self.budget.charge(1, 1)?;
        let mut pending = vec![(root, false)];
        while let Some((identifier, expanded)) = pending.pop() {
            if self.affected.contains_key(&identifier) {
                continue;
            }
            let node = original_node(arena, identifier, self.boundary).ok_or(BUDGET_REASON)?;
            self.budget.charge(node_weight(node), node_weight(node))?;
            let direct = matches!(node, Node::Piecewise { predicates, .. }
                if predicates.iter().any(|predicate| self.predicates.contains_key(predicate)));
            let dependencies = node.dependencies();
            if !expanded && !direct {
                pending.push((identifier, true));
                pending.extend(dependencies.into_iter().map(|child| (child, false)));
            } else {
                let affected = direct
                    || dependencies
                        .iter()
                        .any(|child| self.affected.get(child) == Some(&true));
                self.affected.insert(identifier, affected);
            }
        }
        Ok(self.affected[&root])
    }

    fn rewrite(
        &mut self,
        root: ExprId,
        assignment: u8,
        arena: &mut ExpressionArena,
    ) -> Result<Rewritten, ProjectionError> {
        self.budget.charge(1, 1)?;
        let mut pending = vec![(root, false)];
        while let Some((identifier, expanded)) = pending.pop() {
            if self.rewritten.contains_key(&(identifier, assignment)) {
                continue;
            }
            let original = original_node(arena, identifier, self.boundary).ok_or(BUDGET_REASON)?;
            self.budget
                .charge(node_weight(original), node_weight(original))?;
            let mut node = self.selected_node(original, assignment);
            let dependencies = node.dependencies();
            if !expanded {
                pending.push((identifier, true));
                pending.extend(dependencies.into_iter().map(|child| (child, false)));
                continue;
            }
            let mut changed = node != *original;
            let mut children_total = true;
            map_dependencies(&mut node, |child| {
                let rewritten = self.rewritten[&(*child, assignment)];
                changed |= rewritten.expression != *child;
                *child = rewritten.expression;
                children_total &= rewritten.total;
            });
            let total = children_total && total_constructor(&node, arena);
            let expression = if changed {
                self.budget.charge(node_weight(&node), node_weight(&node))?;
                rebuild(node, arena)?
            } else {
                identifier
            };
            self.rewritten
                .insert((identifier, assignment), Rewritten { expression, total });
        }
        Ok(self.rewritten[&(root, assignment)])
    }

    fn selected_node(&self, original: &Node, assignment: u8) -> Node {
        let Node::Piecewise {
            domain,
            arguments,
            predicates,
        } = original
        else {
            return original.clone();
        };
        let mut retained_arguments = Vec::new();
        let mut retained_predicates = Vec::new();
        for (argument, predicate) in arguments.iter().zip(predicates) {
            match self
                .predicates
                .get(predicate)
                .map(|truth| truth.get(assignment))
            {
                Some(false) => {}
                Some(true) => {
                    retained_arguments.push(*argument);
                    return Node::Piecewise {
                        domain: *domain,
                        arguments: retained_arguments.into_boxed_slice(),
                        predicates: retained_predicates.into_boxed_slice(),
                    };
                }
                None => {
                    retained_arguments.push(*argument);
                    retained_predicates.push(*predicate);
                }
            }
        }
        retained_arguments.extend(arguments.last());
        Node::Piecewise {
            domain: *domain,
            arguments: retained_arguments.into_boxed_slice(),
            predicates: retained_predicates.into_boxed_slice(),
        }
    }
}

enum ProjectionError {
    Proof(&'static str),
    Resource(SymbolicResourceError),
}

impl From<&'static str> for ProjectionError {
    fn from(value: &'static str) -> Self {
        Self::Proof(value)
    }
}

impl From<SymbolicResourceError> for ProjectionError {
    fn from(value: SymbolicResourceError) -> Self {
        Self::Resource(value)
    }
}

fn nonzero_literal(arena: &ExpressionArena, identifier: ExprId) -> bool {
    matches!(arena.node(identifier), Some(Node::Literal { value, .. })
        if value != "0")
}

fn map_dependencies(node: &mut Node, mut map: impl FnMut(&mut ExprId)) {
    match node {
        Node::Add { arguments, .. }
        | Node::Multiply { arguments, .. }
        | Node::Minimum { arguments, .. }
        | Node::Maximum { arguments, .. }
        | Node::Logarithm { arguments }
        | Node::Contracted { arguments, .. }
        | Node::Power { arguments, .. }
        | Node::Recurrence { arguments, .. }
        | Node::RecurrenceSystem { arguments, .. } => {
            arguments.iter_mut().for_each(&mut map);
        }
        Node::FloorDivide { arguments } | Node::CeilingDivide { arguments } => {
            arguments.iter_mut().for_each(&mut map);
        }
        Node::Absolute { argument, .. }
        | Node::Floor { argument }
        | Node::Ceiling { argument }
        | Node::Exponential { argument }
        | Node::Log2 { argument }
        | Node::SquareRoot { argument } => map(argument),
        Node::BoundedSum {
            lower, upper, body, ..
        }
        | Node::BoundedProduct {
            lower, upper, body, ..
        } => {
            map(lower);
            map(upper);
            map(body);
        }
        Node::RecurrenceEquation {
            arguments,
            condition,
            ..
        } => {
            arguments.iter_mut().for_each(&mut map);
            map(condition);
        }
        Node::Predicate {
            arguments,
            predicates,
            ..
        }
        | Node::Piecewise {
            arguments,
            predicates,
            ..
        } => {
            arguments
                .iter_mut()
                .chain(predicates.iter_mut())
                .for_each(&mut map);
        }
        Node::Literal { .. }
        | Node::RationalLiteral { .. }
        | Node::RealLiteral { .. }
        | Node::Symbol { .. }
        | Node::PredicateLiteral(_) => {}
    }
}

fn total_constructor(node: &Node, arena: &ExpressionArena) -> bool {
    match node {
        Node::Predicate {
            kind: Predicate::Divisible,
            arguments,
            ..
        } => arguments.len() == 2 && nonzero_literal(arena, arguments[1]),
        Node::Literal { .. }
        | Node::RationalLiteral { .. }
        | Node::RealLiteral { .. }
        | Node::Symbol { .. }
        | Node::PredicateLiteral(_)
        | Node::Add { .. }
        | Node::Multiply { .. }
        | Node::Minimum { .. }
        | Node::Maximum { .. }
        | Node::Absolute { .. }
        | Node::Floor { .. }
        | Node::Ceiling { .. }
        | Node::Piecewise { .. }
        | Node::Predicate { .. }
        | Node::Exponential { .. } => true,
        Node::BoundedSum {
            binder,
            lower,
            upper,
            ..
        }
        | Node::BoundedProduct {
            binder,
            lower,
            upper,
            ..
        } => {
            let (Some(Node::Symbol { domain, .. }), Some(lower), Some(upper)) =
                (arena.node(*binder), arena.node(*lower), arena.node(*upper))
            else {
                return false;
            };
            matches!(
                lower.domain(),
                ResourceDomain::Natural | ResourceDomain::Integer
            ) && matches!(
                upper.domain(),
                ResourceDomain::Natural | ResourceDomain::Integer
            ) && (*domain == ResourceDomain::Integer
                || (*domain == ResourceDomain::Natural
                    && lower.domain() == ResourceDomain::Natural))
        }
        Node::Power { arguments, .. } => {
            matches!(
                arena.node(arguments[1]),
                Some(Node::Literal {
                    domain: ResourceDomain::Natural,
                    ..
                })
            ) || positive_literal(arena, arguments[0])
        }
        Node::FloorDivide { arguments } | Node::CeilingDivide { arguments } => {
            nonzero_literal(arena, arguments[1])
        }
        Node::Contracted {
            name, arguments, ..
        } => match name.as_str() {
            "qamomile.modulo" | "qamomile.floor_divide" => {
                arguments.len() == 2 && nonzero_literal(arena, arguments[1])
            }
            "qamomile.range_trip_count" => {
                arguments.len() == 3 && nonzero_literal(arena, arguments[2])
            }
            _ => false,
        },
        Node::SquareRoot { argument } => arena.node(*argument).is_some_and(|value| {
            matches!(
                value.domain(),
                ResourceDomain::Natural | ResourceDomain::Probability
            )
        }),
        Node::Logarithm { .. }
        | Node::Log2 { .. }
        | Node::Recurrence { .. }
        | Node::RecurrenceEquation { .. }
        | Node::RecurrenceSystem { .. } => false,
    }
}

fn rebuild(node: Node, arena: &mut ExpressionArena) -> Result<ExprId, SymbolicResourceError> {
    match node {
        Node::Piecewise {
            domain,
            arguments,
            predicates,
        } => arena.piecewise(&predicates, &arguments, domain),
        Node::Add { domain, arguments } => {
            let mut arguments = arguments.into_vec();
            arguments.retain(|argument| !arena.is_literal(*argument, "0"));
            arguments.sort_unstable();
            if arguments.len() <= 1
                || arguments
                    .iter()
                    .all(|argument| matches!(arena.node(*argument), Some(Node::Literal { .. })))
            {
                arena.add(arguments)
            } else {
                arena.intern(Node::Add {
                    domain,
                    arguments: arguments.into_boxed_slice(),
                })
            }
        }
        Node::Multiply { domain, arguments } => {
            if arguments
                .iter()
                .any(|argument| arena.is_literal(*argument, "0"))
            {
                return arena.literal_u128(0);
            }
            let mut arguments = arguments.into_vec();
            arguments.retain(|argument| !arena.is_literal(*argument, "1"));
            arguments.sort_unstable();
            match arguments.as_slice() {
                [] => arena.literal_u128(1),
                [only] => Ok(*only),
                _ => arena.intern(Node::Multiply {
                    domain,
                    arguments: arguments.into_boxed_slice(),
                }),
            }
        }
        Node::Minimum { arguments, .. } => arena.extremum(false, arguments),
        Node::Maximum { arguments, .. } => arena.extremum(true, arguments),
        node => arena.intern(node),
    }
}
