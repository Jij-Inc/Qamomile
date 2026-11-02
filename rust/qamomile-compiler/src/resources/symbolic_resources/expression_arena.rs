//! Hash-consed expression construction and bounded polynomial simplification.

use num_traits::Zero;

use crate::resource_rules::Domain as ResourceDomain;
use num_bigint::{BigInt, Sign};
use std::collections::HashSet;

use super::{
    ExprId, ExpressionArena, MAX_SYMBOLIC_POLYNOMIAL_FACTORS, ResourceExpression,
    ResourceExpressionId, ResourceExpressionNode, ResourcePredicateKind, SymbolicPolynomial,
    SymbolicResourceError,
};

impl ExpressionArena {
    pub(super) fn intern(
        &mut self,
        node: ResourceExpressionNode,
    ) -> Result<ExprId, SymbolicResourceError> {
        if let Some(identifier) = self.ids.get(&node) {
            return Ok(*identifier);
        }
        let identifier = ResourceExpressionId::from_index(self.nodes.len())?;
        self.nodes.push(node.clone());
        self.ids.insert(node, identifier);
        Ok(identifier)
    }

    pub(super) fn node(&self, identifier: ExprId) -> Option<&ResourceExpressionNode> {
        self.nodes.get(identifier.index()?)
    }

    #[cfg(test)]
    pub(super) fn projected_nodes(&self) -> Vec<ResourceExpression> {
        self.nodes
            .iter()
            .cloned()
            .map(ResourceExpression::from_node)
            .collect()
    }

    pub(super) fn into_projected_nodes(self) -> Box<[ResourceExpression]> {
        self.nodes
            .into_iter()
            .map(ResourceExpression::from_node)
            .collect()
    }

    pub(super) fn literal_i64(&mut self, value: i64) -> Result<ExprId, SymbolicResourceError> {
        self.intern(ResourceExpressionNode::Literal {
            value: value.to_string(),
            domain: if value < 0 {
                ResourceDomain::Integer
            } else {
                ResourceDomain::Natural
            },
        })
    }

    pub(super) fn literal_u128(&mut self, value: u128) -> Result<ExprId, SymbolicResourceError> {
        self.intern(ResourceExpressionNode::Literal {
            value: value.to_string(),
            domain: ResourceDomain::Natural,
        })
    }

    pub(super) fn literal_bigint(
        &mut self,
        encoded: &[u8],
    ) -> Result<ExprId, SymbolicResourceError> {
        let value = BigInt::from_signed_bytes_be(encoded);
        self.intern(ResourceExpressionNode::Literal {
            domain: if value.sign() == Sign::Minus {
                ResourceDomain::Integer
            } else {
                ResourceDomain::Natural
            },
            value: value.to_string(),
        })
    }

    pub(super) fn symbol(
        &mut self,
        scope: &str,
        name: &str,
        domain: ResourceDomain,
        origin: &str,
        constraints: &[&str],
    ) -> Result<ExprId, SymbolicResourceError> {
        self.intern(ResourceExpressionNode::Symbol {
            scope: scope.to_owned(),
            name: name.to_owned(),
            domain,
            origin: origin.to_owned(),
            constraints: constraints
                .iter()
                .map(|item| (*item).to_owned())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        })
    }

    pub(super) fn predicate_literal(
        &mut self,
        value: bool,
    ) -> Result<ExprId, SymbolicResourceError> {
        self.intern(ResourceExpressionNode::PredicateLiteral(value))
    }

    pub(super) fn predicate_value(&self, identifier: ExprId) -> Option<bool> {
        match self.node(identifier) {
            Some(ResourceExpressionNode::PredicateLiteral(value)) => Some(*value),
            _ => None,
        }
    }

    pub(super) fn predicate_connective(
        &mut self,
        kind: ResourcePredicateKind,
        predicates: impl IntoIterator<Item = ExprId>,
    ) -> Result<ExprId, SymbolicResourceError> {
        let mut predicates = predicates.into_iter().collect::<Vec<_>>();
        match kind {
            ResourcePredicateKind::And
                if predicates
                    .iter()
                    .any(|value| self.predicate_value(*value) == Some(false)) =>
            {
                return self.predicate_literal(false);
            }
            ResourcePredicateKind::Or
                if predicates
                    .iter()
                    .any(|value| self.predicate_value(*value) == Some(true)) =>
            {
                return self.predicate_literal(true);
            }
            ResourcePredicateKind::And => {
                predicates.retain(|value| self.predicate_value(*value) != Some(true));
            }
            ResourcePredicateKind::Or => {
                predicates.retain(|value| self.predicate_value(*value) != Some(false));
            }
            ResourcePredicateKind::Not => {
                let [predicate] = predicates.as_slice() else {
                    return Err(SymbolicResourceError::Missing(
                        "resource-rule negation operand".to_owned(),
                    ));
                };
                if let Some(value) = self.predicate_value(*predicate) {
                    return self.predicate_literal(!value);
                }
            }
            _ => {}
        }
        predicates.sort_unstable();
        predicates.dedup();
        match predicates.as_slice() {
            [] if kind == ResourcePredicateKind::And => self.predicate_literal(true),
            [] if kind == ResourcePredicateKind::Or => self.predicate_literal(false),
            [only] if kind != ResourcePredicateKind::Not => Ok(*only),
            _ => self.intern(ResourceExpressionNode::Predicate {
                kind,
                arguments: Box::new([]),
                predicates: predicates.into_boxed_slice(),
            }),
        }
    }

    pub(super) fn predicate_comparison(
        &mut self,
        kind: ResourcePredicateKind,
        arguments: [ExprId; 2],
    ) -> Result<ExprId, SymbolicResourceError> {
        if kind == ResourcePredicateKind::Divisible && self.is_literal(arguments[1], "0") {
            return Err(SymbolicResourceError::Unsupported(
                "zero resource-rule divisor".to_owned(),
            ));
        }
        let exact: [Option<num_bigint::BigInt>; 2] =
            arguments.map(|argument| match self.node(argument) {
                Some(ResourceExpressionNode::Literal {
                    value,
                    domain: ResourceDomain::Natural | ResourceDomain::Integer,
                }) => value.parse().ok(),
                _ => None,
            });
        if let [Some(left), Some(right)] = exact {
            let value = match kind {
                ResourcePredicateKind::Equal => left == right,
                ResourcePredicateKind::NotEqual => left != right,
                ResourcePredicateKind::Less => left < right,
                ResourcePredicateKind::LessEqual => left <= right,
                ResourcePredicateKind::Greater => left > right,
                ResourcePredicateKind::GreaterEqual => left >= right,
                ResourcePredicateKind::Divisible if !right.is_zero() => {
                    left % right == num_bigint::BigInt::from(0)
                }
                ResourcePredicateKind::Divisible => {
                    return Err(SymbolicResourceError::Unsupported(
                        "zero resource-rule divisor".to_owned(),
                    ));
                }
                ResourcePredicateKind::And
                | ResourcePredicateKind::Or
                | ResourcePredicateKind::Not => {
                    return Err(SymbolicResourceError::Missing(
                        "resource-rule predicate constructor".to_owned(),
                    ));
                }
            };
            return self.predicate_literal(value);
        }
        self.intern(ResourceExpressionNode::Predicate {
            kind,
            arguments: arguments.into(),
            predicates: Box::new([]),
        })
    }

    pub(super) fn piecewise(
        &mut self,
        predicates: &[ExprId],
        arguments: &[ExprId],
        domain: ResourceDomain,
    ) -> Result<ExprId, SymbolicResourceError> {
        let mut retained_predicates = Vec::new();
        let mut retained_arguments = Vec::new();
        for (predicate, argument) in predicates.iter().zip(arguments) {
            match self.predicate_value(*predicate) {
                Some(false) => {}
                Some(true) if retained_predicates.is_empty() => return Ok(*argument),
                Some(true) => {
                    retained_arguments.push(*argument);
                    return self.intern(ResourceExpressionNode::Piecewise {
                        domain,
                        arguments: retained_arguments.into_boxed_slice(),
                        predicates: retained_predicates.into_boxed_slice(),
                    });
                }
                None => {
                    retained_predicates.push(*predicate);
                    retained_arguments.push(*argument);
                }
            }
        }
        let otherwise = *arguments.last().ok_or_else(|| {
            SymbolicResourceError::Missing("resource-rule piecewise default".to_owned())
        })?;
        if retained_predicates.is_empty() {
            return Ok(otherwise);
        }
        retained_arguments.push(otherwise);
        if retained_arguments.iter().all(|value| *value == otherwise) {
            return Ok(otherwise);
        }
        self.intern(ResourceExpressionNode::Piecewise {
            domain,
            arguments: retained_arguments.into_boxed_slice(),
            predicates: retained_predicates.into_boxed_slice(),
        })
    }

    pub(super) fn add(
        &mut self,
        arguments: impl IntoIterator<Item = ExprId>,
    ) -> Result<ExprId, SymbolicResourceError> {
        let mut flattened = Vec::new();
        for argument in arguments {
            match self.node(argument) {
                Some(ResourceExpressionNode::Add { arguments, .. }) => {
                    flattened.extend_from_slice(arguments);
                }
                _ if self.is_literal(argument, "0") => {}
                _ => flattened.push(argument),
            }
        }
        flattened.sort_unstable();
        if flattened.len() > 1
            && let Some(polynomial) = self.polynomial_sum(&flattened)
        {
            let domain = self.aggregate_domain(&flattened);
            return self.polynomial_expression(polynomial, Some(domain));
        }
        match flattened.as_slice() {
            [] => self.literal_u128(0),
            [only] => Ok(*only),
            _ => {
                let domain = self.aggregate_domain(&flattened);
                self.intern(ResourceExpressionNode::Add {
                    domain,
                    arguments: flattened.into_boxed_slice(),
                })
            }
        }
    }

    pub(super) fn multiply(
        &mut self,
        arguments: impl IntoIterator<Item = ExprId>,
    ) -> Result<ExprId, SymbolicResourceError> {
        let mut flattened = Vec::new();
        for argument in arguments {
            if self.is_literal(argument, "0") {
                return self.literal_u128(0);
            }
            match self.node(argument) {
                Some(ResourceExpressionNode::Multiply { arguments, .. }) => {
                    flattened.extend_from_slice(arguments);
                }
                _ if self.is_literal(argument, "1") => {}
                _ => flattened.push(argument),
            }
        }
        flattened.sort_unstable();
        match flattened.as_slice() {
            [] => self.literal_u128(1),
            [only] => Ok(*only),
            _ => {
                let domain = self.aggregate_domain(&flattened);
                self.intern(ResourceExpressionNode::Multiply {
                    domain,
                    arguments: flattened.into_boxed_slice(),
                })
            }
        }
    }

    pub(super) fn extremum(
        &mut self,
        maximum: bool,
        arguments: impl IntoIterator<Item = ExprId>,
    ) -> Result<ExprId, SymbolicResourceError> {
        let mut values = arguments.into_iter().collect::<Vec<_>>();
        values.sort_unstable();
        values.dedup();
        match values.as_slice() {
            [] => self.literal_u128(0),
            [only] => Ok(*only),
            _ => {
                let domain = self.aggregate_domain(&values);
                let arguments = values.into_boxed_slice();
                if maximum {
                    self.intern(ResourceExpressionNode::Maximum { domain, arguments })
                } else {
                    self.intern(ResourceExpressionNode::Minimum { domain, arguments })
                }
            }
        }
    }

    pub(super) fn power(
        &mut self,
        base: ExprId,
        exponent: ExprId,
    ) -> Result<ExprId, SymbolicResourceError> {
        let base_domain = self
            .node(base)
            .map_or(ResourceDomain::Natural, ResourceExpressionNode::domain);
        let exponent_domain = self
            .node(exponent)
            .map_or(ResourceDomain::Natural, ResourceExpressionNode::domain);
        let domain = match exponent_domain {
            ResourceDomain::Integer => match base_domain {
                ResourceDomain::Real | ResourceDomain::Probability => ResourceDomain::Real,
                _ => ResourceDomain::Rational,
            },
            ResourceDomain::Rational => ResourceDomain::Real,
            _ => self.aggregate_domain(&[base, exponent]),
        };
        self.intern(ResourceExpressionNode::Power {
            domain,
            arguments: Box::new([base, exponent]),
        })
    }

    pub(super) fn absolute(
        &mut self,
        argument: ExprId,
        domain: ResourceDomain,
    ) -> Result<ExprId, SymbolicResourceError> {
        self.intern(ResourceExpressionNode::Absolute { domain, argument })
    }

    pub(super) fn floor(&mut self, argument: ExprId) -> Result<ExprId, SymbolicResourceError> {
        self.intern(ResourceExpressionNode::Floor { argument })
    }

    pub(super) fn ceiling(&mut self, argument: ExprId) -> Result<ExprId, SymbolicResourceError> {
        self.intern(ResourceExpressionNode::Ceiling { argument })
    }

    pub(super) fn floor_divide(
        &mut self,
        numerator: ExprId,
        denominator: ExprId,
    ) -> Result<ExprId, SymbolicResourceError> {
        self.intern(ResourceExpressionNode::FloorDivide {
            arguments: Box::new([numerator, denominator]),
        })
    }

    pub(super) fn ceiling_divide(
        &mut self,
        numerator: ExprId,
        denominator: ExprId,
    ) -> Result<ExprId, SymbolicResourceError> {
        self.intern(ResourceExpressionNode::CeilingDivide {
            arguments: Box::new([numerator, denominator]),
        })
    }

    pub(super) fn exponential(
        &mut self,
        argument: ExprId,
    ) -> Result<ExprId, SymbolicResourceError> {
        self.intern(ResourceExpressionNode::Exponential { argument })
    }

    pub(super) fn logarithm(
        &mut self,
        arguments: impl IntoIterator<Item = ExprId>,
    ) -> Result<ExprId, SymbolicResourceError> {
        self.intern(ResourceExpressionNode::Logarithm {
            arguments: arguments.into_iter().collect::<Vec<_>>().into_boxed_slice(),
        })
    }

    pub(super) fn log2(&mut self, argument: ExprId) -> Result<ExprId, SymbolicResourceError> {
        self.intern(ResourceExpressionNode::Log2 { argument })
    }

    pub(super) fn square_root(
        &mut self,
        argument: ExprId,
    ) -> Result<ExprId, SymbolicResourceError> {
        self.intern(ResourceExpressionNode::SquareRoot { argument })
    }

    pub(super) fn contracted(
        &mut self,
        name: &str,
        domain: ResourceDomain,
        arguments: impl IntoIterator<Item = ExprId>,
    ) -> Result<ExprId, SymbolicResourceError> {
        let arguments = arguments.into_iter().collect::<Vec<_>>();
        if name == "qamomile.range_trip_count"
            && let [start, stop, step] = arguments.as_slice()
            && self.is_literal(*start, "0")
            && self.is_literal(*step, "1")
            && self.is_nonnegative(*stop)
        {
            return self.naturalize_nonnegative(*stop);
        }
        self.intern(ResourceExpressionNode::Contracted {
            name: name.to_owned(),
            domain,
            arguments: arguments.into_boxed_slice(),
        })
    }

    pub(super) fn recurrence(
        &mut self,
        name: &str,
        domain: ResourceDomain,
        arguments: impl IntoIterator<Item = ExprId>,
    ) -> Result<ExprId, SymbolicResourceError> {
        self.intern(ResourceExpressionNode::Recurrence {
            name: name.to_owned(),
            domain,
            arguments: arguments.into_iter().collect::<Vec<_>>().into_boxed_slice(),
        })
    }

    pub(super) fn bounded_sum(
        &mut self,
        binder: ExprId,
        lower: ExprId,
        upper: ExprId,
        body: ExprId,
    ) -> Result<ExprId, SymbolicResourceError> {
        let domain = self
            .node(body)
            .ok_or_else(|| SymbolicResourceError::Missing("bounded-sum body".to_owned()))?
            .domain();
        self.intern(ResourceExpressionNode::BoundedSum {
            domain,
            binder,
            lower,
            upper,
            body,
        })
    }

    pub(super) fn polynomial_sum(&self, arguments: &[ExprId]) -> Option<SymbolicPolynomial> {
        let mut result = SymbolicPolynomial::default();
        for argument in arguments {
            result.add_assign(self.polynomial(*argument)?)?;
        }
        Some(result)
    }

    pub(super) fn polynomial(&self, root: ExprId) -> Option<SymbolicPolynomial> {
        match self.node(root)? {
            ResourceExpressionNode::Literal {
                value,
                domain: ResourceDomain::Natural | ResourceDomain::Integer,
            } => Some(SymbolicPolynomial::constant(value.parse().ok()?)),
            ResourceExpressionNode::Add { arguments, .. } => {
                let mut result = SymbolicPolynomial::default();
                for argument in arguments {
                    result.add_assign(self.polynomial(*argument)?)?;
                }
                Some(result)
            }
            ResourceExpressionNode::Multiply { arguments, .. } => {
                let mut result = SymbolicPolynomial::constant(BigInt::from(1));
                for argument in arguments {
                    result = result.multiplied(&self.polynomial(*argument)?)?;
                }
                Some(result)
            }
            ResourceExpressionNode::Power { arguments, .. } => {
                let exponent = self.integer_value(arguments[1])?;
                let exponent = u32::try_from(exponent).ok()?;
                if usize::try_from(exponent).ok()? > MAX_SYMBOLIC_POLYNOMIAL_FACTORS {
                    return None;
                }
                self.polynomial(arguments[0])?.power(exponent)
            }
            _ => Some(SymbolicPolynomial::atom(root)),
        }
    }

    pub(super) fn polynomial_expression(
        &mut self,
        polynomial: SymbolicPolynomial,
        root_domain: Option<ResourceDomain>,
    ) -> Result<ExprId, SymbolicResourceError> {
        let mut terms = Vec::with_capacity(polynomial.terms.len());
        for (monomial, coefficient) in polynomial.terms {
            let mut factors = Vec::new();
            if monomial.0.is_empty() || coefficient != BigInt::from(1) {
                factors.push(self.literal_bigint(&coefficient.to_signed_bytes_be())?);
            }
            for (identifier, exponent) in monomial.0 {
                if exponent == 1 {
                    factors.push(identifier);
                } else {
                    let exponent = self.literal_u128(u128::from(exponent))?;
                    factors.push(self.power(identifier, exponent)?);
                }
            }
            factors.sort_unstable();
            let term = match factors.as_slice() {
                [] => self.literal_u128(1)?,
                [only] => *only,
                _ => {
                    let domain = self.aggregate_domain(&factors);
                    self.intern(ResourceExpressionNode::Multiply {
                        domain,
                        arguments: factors.into_boxed_slice(),
                    })?
                }
            };
            terms.push(term);
        }
        terms.sort_unstable();
        match terms.as_slice() {
            [] => self.literal_u128(0),
            [only] => root_domain.map_or(Ok(*only), |domain| {
                self.retag_polynomial_root(*only, domain)
            }),
            _ => {
                let domain = root_domain.unwrap_or_else(|| self.aggregate_domain(&terms));
                self.intern(ResourceExpressionNode::Add {
                    domain,
                    arguments: terms.into_boxed_slice(),
                })
            }
        }
    }

    pub(super) fn retag_polynomial_root(
        &mut self,
        identifier: ExprId,
        domain: ResourceDomain,
    ) -> Result<ExprId, SymbolicResourceError> {
        let Some(node) = self.node(identifier).cloned() else {
            return Err(SymbolicResourceError::Missing(
                "normalized polynomial root".to_owned(),
            ));
        };
        let node = match node {
            ResourceExpressionNode::Literal { value, .. } => {
                ResourceExpressionNode::Literal { value, domain }
            }
            ResourceExpressionNode::Add { arguments, .. } => {
                ResourceExpressionNode::Add { domain, arguments }
            }
            ResourceExpressionNode::Multiply { arguments, .. } => {
                ResourceExpressionNode::Multiply { domain, arguments }
            }
            ResourceExpressionNode::Power { arguments, .. } => {
                ResourceExpressionNode::Power { domain, arguments }
            }
            node if node.domain() == domain => return Ok(identifier),
            _ => return Ok(identifier),
        };
        self.intern(node)
    }

    pub(super) fn is_nonnegative(&self, identifier: ExprId) -> bool {
        let Some(node) = self.node(identifier) else {
            return false;
        };
        if node.domain() == ResourceDomain::Natural {
            return true;
        }
        match node {
            ResourceExpressionNode::Literal { value, .. } => value
                .parse::<BigInt>()
                .is_ok_and(|value| value >= BigInt::from(0)),
            ResourceExpressionNode::Add { arguments, .. }
            | ResourceExpressionNode::Multiply { arguments, .. } => arguments
                .iter()
                .all(|argument| self.is_nonnegative(*argument)),
            ResourceExpressionNode::FloorDivide { arguments }
            | ResourceExpressionNode::CeilingDivide { arguments } => {
                self.is_nonnegative(arguments[0]) && self.is_positive(arguments[1])
            }
            ResourceExpressionNode::Contracted {
                name, arguments, ..
            } if name == "qamomile.floor_divide" => {
                let [numerator, denominator] = arguments.as_ref() else {
                    return false;
                };
                self.is_nonnegative(*numerator) && self.is_positive(*denominator)
            }
            ResourceExpressionNode::Contracted { name, .. } => {
                name == "qamomile.positive_dimension"
            }
            _ => false,
        }
    }

    pub(super) fn naturalize_nonnegative(
        &mut self,
        identifier: ExprId,
    ) -> Result<ExprId, SymbolicResourceError> {
        let Some(node) = self.node(identifier).cloned() else {
            return Err(SymbolicResourceError::Missing(
                "nonnegative resource expression".to_owned(),
            ));
        };
        if node.domain() == ResourceDomain::Natural {
            return Ok(identifier);
        }
        let refined = match node {
            ResourceExpressionNode::Contracted {
                name, arguments, ..
            } if name == "qamomile.floor_divide" => ResourceExpressionNode::Contracted {
                name,
                domain: ResourceDomain::Natural,
                arguments,
            },
            _ => return Ok(identifier),
        };
        self.intern(refined)
    }

    pub(super) fn is_positive(&self, identifier: ExprId) -> bool {
        match self.node(identifier) {
            Some(ResourceExpressionNode::Literal { value, .. }) => value
                .parse::<BigInt>()
                .is_ok_and(|value| value > BigInt::from(0)),
            Some(ResourceExpressionNode::Symbol {
                domain: ResourceDomain::Natural,
                constraints,
                name,
                ..
            }) => constraints.iter().any(|constraint| {
                constraint == &format!("{name} >= 1") || constraint == "positive"
            }),
            Some(ResourceExpressionNode::Contracted { name, .. }) => {
                name == "qamomile.positive_dimension"
            }
            _ => false,
        }
    }

    pub(super) fn bounded_product(
        &mut self,
        binder: ExprId,
        lower: ExprId,
        upper: ExprId,
        body: ExprId,
    ) -> Result<ExprId, SymbolicResourceError> {
        let domain = self
            .node(body)
            .ok_or_else(|| SymbolicResourceError::Missing("bounded-product body".to_owned()))?
            .domain();
        self.intern(ResourceExpressionNode::BoundedProduct {
            domain,
            binder,
            lower,
            upper,
            body,
        })
    }

    pub(super) fn is_literal(&self, identifier: ExprId, expected: &str) -> bool {
        matches!(
            self.node(identifier),
            Some(ResourceExpressionNode::Literal { value, .. }) if value == expected
        )
    }

    pub(super) fn integer_value(&self, identifier: ExprId) -> Option<i128> {
        match self.node(identifier)? {
            ResourceExpressionNode::Literal {
                value,
                domain: ResourceDomain::Natural | ResourceDomain::Integer,
            } => value.parse().ok(),
            ResourceExpressionNode::Add { arguments, .. } => {
                arguments.iter().try_fold(0_i128, |total, argument| {
                    total.checked_add(self.integer_value(*argument)?)
                })
            }
            ResourceExpressionNode::Multiply { arguments, .. } => {
                arguments.iter().try_fold(1_i128, |total, argument| {
                    total.checked_mul(self.integer_value(*argument)?)
                })
            }
            ResourceExpressionNode::Minimum { arguments, .. } => arguments
                .iter()
                .map(|argument| self.integer_value(*argument))
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .min(),
            ResourceExpressionNode::Maximum { arguments, .. } => arguments
                .iter()
                .map(|argument| self.integer_value(*argument))
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .max(),
            ResourceExpressionNode::Absolute { argument, .. } => {
                self.integer_value(*argument)?.checked_abs()
            }
            _ => None,
        }
    }

    pub(super) fn aggregate_domain(&self, arguments: &[ExprId]) -> ResourceDomain {
        arguments
            .iter()
            .filter_map(|identifier| self.node(*identifier).map(ResourceExpressionNode::domain))
            .max_by_key(|domain| domain.promotion_rank())
            .unwrap_or(ResourceDomain::Natural)
    }

    pub(super) fn depends_on(&self, root: ExprId, needle: ExprId) -> bool {
        let mut pending = vec![root];
        let mut visited = HashSet::new();
        while let Some(identifier) = pending.pop() {
            if identifier == needle {
                return true;
            }
            if visited.insert(identifier)
                && let Some(node) = self.node(identifier)
            {
                pending.extend(node.dependencies());
            }
        }
        false
    }

    pub(super) fn depends_on_symbol_origin(&self, root: ExprId, origin: &str) -> bool {
        let mut pending = vec![root];
        let mut visited = HashSet::new();
        while let Some(identifier) = pending.pop() {
            if !visited.insert(identifier) {
                continue;
            }
            let Some(node) = self.node(identifier) else {
                continue;
            };
            if matches!(
                node,
                ResourceExpressionNode::Symbol {
                    origin: candidate,
                    ..
                } if candidate == origin
            ) {
                return true;
            }
            pending.extend(node.dependencies());
        }
        false
    }
}
