//! Composition of count bounds, provenance, and metric families.

use crate::operations::{defers_gate_families, logical_gate_families};
use crate::resource_rules::{Domain as ResourceDomain, RuleRelation};
use crate::resources::analysis::{is_measurement, is_reset};
use std::collections::{BTreeMap, BTreeSet};

use super::{
    Counts, ExprId, ExpressionArena, PORTABLE_PREFIX, ResourceExpressionNode, SymbolicCount,
    SymbolicResourceError, external_depth_metrics, external_family_metrics,
    symbolic_depth_families,
};

impl SymbolicCount {
    pub(super) fn exact_value(expression: ExprId) -> Self {
        Self {
            lower: expression,
            upper: Some(expression),
            exact: true,
            unknown: Box::new([]),
            rule_identities: Box::new([]),
        }
    }

    pub(super) fn from_rule(
        identity: &str,
        relation: RuleRelation,
        expression: ExprId,
        arena: &mut ExpressionArena,
    ) -> Result<Self, SymbolicResourceError> {
        let zero = arena.literal_u128(0)?;
        let mut count = match relation {
            RuleRelation::Equality => Self::exact_value(expression),
            RuleRelation::LowerBound => Self {
                lower: expression,
                upper: None,
                exact: false,
                unknown: Box::new([]),
                rule_identities: Box::new([]),
            },
            RuleRelation::UpperBound => Self {
                lower: zero,
                upper: Some(expression),
                exact: false,
                unknown: Box::new([]),
                rule_identities: Box::new([]),
            },
        };
        count.rule_identities = Box::new([identity.to_owned()]);
        Ok(count)
    }

    pub(super) fn unresolved(
        reason: String,
        arena: &mut ExpressionArena,
    ) -> Result<Self, SymbolicResourceError> {
        Ok(Self {
            lower: arena.literal_u128(0)?,
            upper: None,
            exact: false,
            unknown: vec![reason].into_boxed_slice(),
            rule_identities: Box::new([]),
        })
    }

    pub(super) fn unresolved_from(
        source: &Self,
        reason: String,
        arena: &mut ExpressionArena,
    ) -> Result<Self, SymbolicResourceError> {
        let mut result = Self::unresolved(reason, arena)?;
        result.unknown = merge_unknown([source.unknown.as_ref(), result.unknown.as_ref()]);
        result.rule_identities.clone_from(&source.rule_identities);
        Ok(result)
    }

    pub(super) fn bounded_by(
        upper: &Self,
        arena: &mut ExpressionArena,
    ) -> Result<Self, SymbolicResourceError> {
        Ok(Self {
            lower: arena.literal_u128(0)?,
            upper: upper.upper,
            exact: false,
            unknown: upper.unknown.clone(),
            rule_identities: upper.rule_identities.clone(),
        })
    }

    pub(super) fn dependency_bounded_by(
        upper: &Self,
        arena: &mut ExpressionArena,
    ) -> Result<Self, SymbolicResourceError> {
        if upper.exact
            && upper
                .upper
                .is_some_and(|expression| arena.is_literal(expression, "0"))
        {
            return Ok(upper.clone());
        }
        Self::bounded_by(upper, arena)
    }

    pub(super) fn peak(
        &self,
        other: &Self,
        arena: &mut ExpressionArena,
    ) -> Result<Self, SymbolicResourceError> {
        Ok(Self {
            lower: arena.extremum(true, [self.lower, other.lower])?,
            upper: self
                .upper
                .zip(other.upper)
                .map(|(left, right)| arena.extremum(true, [left, right]))
                .transpose()?,
            exact: self.exact && other.exact,
            unknown: merge_unknown([self.unknown.as_ref(), other.unknown.as_ref()]),
            rule_identities: merge_rule_identities([
                self.rule_identities.as_ref(),
                other.rule_identities.as_ref(),
            ]),
        })
    }

    pub(super) fn then(
        &self,
        other: &Self,
        arena: &mut ExpressionArena,
    ) -> Result<Self, SymbolicResourceError> {
        Ok(Self {
            lower: arena.add([self.lower, other.lower])?,
            upper: self
                .upper
                .zip(other.upper)
                .map(|(left, right)| arena.add([left, right]))
                .transpose()?,
            exact: self.exact && other.exact,
            unknown: merge_unknown([self.unknown.as_ref(), other.unknown.as_ref()]),
            rule_identities: merge_rule_identities([
                self.rule_identities.as_ref(),
                other.rule_identities.as_ref(),
            ]),
        })
    }

    pub(super) fn probability_then(
        &self,
        other: &Self,
        arena: &mut ExpressionArena,
    ) -> Result<Self, SymbolicResourceError> {
        let exact_zero = self.exact
            && other.exact
            && arena.is_literal(self.lower, "0")
            && arena.is_literal(other.lower, "0");
        if exact_zero {
            let mut result = Self::exact_value(arena.literal_u128(0)?);
            result.rule_identities = merge_rule_identities([
                self.rule_identities.as_ref(),
                other.rule_identities.as_ref(),
            ]);
            return Ok(result);
        }
        let one = arena.literal_u128(1)?;
        Ok(Self {
            lower: arena.literal_u128(0)?,
            upper: self
                .upper
                .zip(other.upper)
                .map(|(left, right)| {
                    let sum = arena.add([left, right])?;
                    arena.extremum(false, [one, sum])
                })
                .transpose()?,
            exact: false,
            unknown: merge_unknown([self.unknown.as_ref(), other.unknown.as_ref()]),
            rule_identities: merge_rule_identities([
                self.rule_identities.as_ref(),
                other.rule_identities.as_ref(),
            ]),
        })
    }

    pub(super) fn dependency_then(
        &self,
        other: &Self,
        arena: &mut ExpressionArena,
    ) -> Result<Self, SymbolicResourceError> {
        if self.upper.is_some_and(|upper| arena.is_literal(upper, "0")) {
            let mut result = other.clone();
            result.rule_identities = merge_rule_identities([
                self.rule_identities.as_ref(),
                other.rule_identities.as_ref(),
            ]);
            return Ok(result);
        }
        if other
            .upper
            .is_some_and(|upper| arena.is_literal(upper, "0"))
        {
            let mut result = self.clone();
            result.rule_identities = merge_rule_identities([
                self.rule_identities.as_ref(),
                other.rule_identities.as_ref(),
            ]);
            return Ok(result);
        }
        Ok(Self {
            lower: arena.extremum(true, [self.lower, other.lower])?,
            upper: self
                .upper
                .zip(other.upper)
                .map(|(left, right)| arena.add([left, right]))
                .transpose()?,
            exact: false,
            unknown: merge_unknown([self.unknown.as_ref(), other.unknown.as_ref()]),
            rule_identities: merge_rule_identities([
                self.rule_identities.as_ref(),
                other.rule_identities.as_ref(),
            ]),
        })
    }

    pub(super) fn repeat(
        &self,
        trip_count: ExprId,
        binder: ExprId,
        arena: &mut ExpressionArena,
    ) -> Result<Self, SymbolicResourceError> {
        if arena.is_literal(trip_count, "0") {
            return Ok(Self::exact_value(arena.literal_u128(0)?));
        }
        if !arena.is_literal(trip_count, "1")
            && (arena.depends_on_symbol_origin(self.lower, "while_iterations")
                || self
                    .upper
                    .is_some_and(|value| arena.depends_on_symbol_origin(value, "while_iterations")))
        {
            return Self::unresolved_from(
                self,
                "nested while counts may vary between invocations".to_owned(),
                arena,
            );
        }
        let zero = arena.literal_u128(0)?;
        let negative_one = arena.literal_i64(-1)?;
        let upper_bound = arena.add([trip_count, negative_one])?;
        let scale = |expression: ExprId, arena: &mut ExpressionArena| {
            if arena.depends_on(expression, binder) {
                arena.bounded_sum(binder, zero, upper_bound, expression)
            } else {
                arena.multiply([expression, trip_count])
            }
        };
        Ok(Self {
            lower: scale(self.lower, arena)?,
            upper: self.upper.map(|value| scale(value, arena)).transpose()?,
            exact: self.exact,
            unknown: self.unknown.clone(),
            rule_identities: self.rule_identities.clone(),
        })
    }

    pub(super) fn probability_repeat(
        &self,
        trip_count: ExprId,
        binder: ExprId,
        arena: &mut ExpressionArena,
    ) -> Result<Self, SymbolicResourceError> {
        let repeated = self.repeat(trip_count, binder, arena)?;
        if repeated.exact && arena.is_literal(repeated.lower, "0") {
            let mut result = Self::exact_value(arena.literal_u128(0)?);
            result.rule_identities = repeated.rule_identities;
            return Ok(result);
        }
        let one = arena.literal_u128(1)?;
        Ok(Self {
            lower: arena.literal_u128(0)?,
            upper: repeated
                .upper
                .map(|upper| arena.extremum(false, [one, upper]))
                .transpose()?,
            exact: false,
            unknown: repeated.unknown,
            rule_identities: repeated.rule_identities,
        })
    }

    pub(super) fn dependency_repeat(
        &self,
        trip_count: ExprId,
        binder: ExprId,
        arena: &mut ExpressionArena,
    ) -> Result<Self, SymbolicResourceError> {
        if arena.is_literal(trip_count, "0") {
            return Ok(Self::exact_value(arena.literal_u128(0)?));
        }
        if arena.is_literal(trip_count, "1") {
            return Ok(self.clone());
        }
        let repeated = self.repeat(trip_count, binder, arena)?;
        let lower = if arena.node(trip_count).is_some_and(|node| {
            matches!(
                node,
                ResourceExpressionNode::Literal {
                    domain: ResourceDomain::Natural,
                    ..
                }
            )
        }) {
            self.lower
        } else {
            arena.literal_u128(0)?
        };
        Ok(Self {
            lower,
            upper: repeated.upper,
            exact: false,
            unknown: repeated.unknown,
            rule_identities: repeated.rule_identities,
        })
    }

    pub(super) fn branch(
        alternatives: &[Self],
        arena: &mut ExpressionArena,
    ) -> Result<Self, SymbolicResourceError> {
        let Some(first) = alternatives.first() else {
            return Ok(Self::exact_value(arena.literal_u128(0)?));
        };
        if alternatives
            .iter()
            .all(|item| item.exact && item.lower == first.lower)
        {
            let mut result = first.clone();
            result.rule_identities = merge_rule_identities(
                alternatives
                    .iter()
                    .map(|item| item.rule_identities.as_ref()),
            );
            return Ok(result);
        }
        Ok(Self {
            lower: arena.extremum(false, alternatives.iter().map(|item| item.lower))?,
            upper: alternatives
                .iter()
                .map(|item| item.upper)
                .collect::<Option<Vec<_>>>()
                .map(|values| arena.extremum(true, values))
                .transpose()?,
            exact: false,
            unknown: merge_unknown(alternatives.iter().map(|item| item.unknown.as_ref())),
            rule_identities: merge_rule_identities(
                alternatives
                    .iter()
                    .map(|item| item.rule_identities.as_ref()),
            ),
        })
    }

    pub(super) fn unbounded(required: &Self, reason: &str) -> Self {
        let mut unknown = required.unknown.to_vec();
        if !unknown.iter().any(|item| item == reason) {
            unknown.push(reason.to_owned());
        }
        Self {
            lower: required.lower,
            upper: None,
            exact: false,
            unknown: unknown.into_boxed_slice(),
            rule_identities: required.rule_identities.clone(),
        }
    }

    pub(super) fn invalidated(source: &Self, lower: ExprId, reason: &str) -> Self {
        let mut unknown = source.unknown.to_vec();
        if !unknown.iter().any(|item| item == reason) {
            unknown.push(reason.to_owned());
        }
        Self {
            lower,
            upper: None,
            exact: false,
            unknown: unknown.into_boxed_slice(),
            rule_identities: source.rule_identities.clone(),
        }
    }
}

pub(super) fn merge_unknown<'a>(values: impl IntoIterator<Item = &'a [String]>) -> Box<[String]> {
    let mut merged = Vec::new();
    for reason in values.into_iter().flatten() {
        if !merged.contains(reason) {
            merged.push(reason.clone());
        }
    }
    merged.into_boxed_slice()
}

pub(super) fn merge_rule_identities<'a>(
    values: impl IntoIterator<Item = &'a [String]>,
) -> Box<[String]> {
    values
        .into_iter()
        .flatten()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

pub(super) fn branch_named_counts(
    alternatives: &[Counts],
    select: impl Fn(&Counts) -> &BTreeMap<String, SymbolicCount>,
    arena: &mut ExpressionArena,
) -> Result<BTreeMap<String, SymbolicCount>, SymbolicResourceError> {
    let names = alternatives
        .iter()
        .flat_map(|item| select(item).keys().cloned())
        .collect::<std::collections::BTreeSet<_>>();
    let zero = SymbolicCount::exact_value(arena.literal_u128(0)?);
    names
        .into_iter()
        .map(|name| {
            let values = alternatives
                .iter()
                .map(|item| {
                    select(item)
                        .get(&name)
                        .cloned()
                        .unwrap_or_else(|| zero.clone())
                })
                .collect::<Vec<_>>();
            Ok((name, SymbolicCount::branch(&values, arena)?))
        })
        .collect()
}

impl Counts {
    pub(super) fn zero(arena: &mut ExpressionArena) -> Result<Self, SymbolicResourceError> {
        let zero = SymbolicCount::exact_value(arena.literal_u128(0)?);
        Ok(Self {
            work: zero.clone(),
            span: zero.clone(),
            depths: Some(
                external_depth_metrics()
                    .iter()
                    .map(|(family, _)| ((*family).to_owned(), zero.clone()))
                    .collect(),
            ),
            opcodes: BTreeMap::new(),
            extra_metrics: BTreeMap::new(),
            families: Some(BTreeMap::new()),
            measurements: Some(zero.clone()),
            resets: Some(zero.clone()),
            cleanup_work: zero,
        })
    }

    pub(super) fn quantum(
        opcode: &str,
        effective_opcode: &str,
        event_width: Option<ExprId>,
        operation_count: ExprId,
        cleanup: bool,
        arena: &mut ExpressionArena,
    ) -> Result<Self, SymbolicResourceError> {
        let one = SymbolicCount::exact_value(arena.literal_u128(1)?);
        let zero = SymbolicCount::exact_value(arena.literal_u128(0)?);
        let active_depths = symbolic_depth_families(effective_opcode);
        let deferred = defers_gate_families(effective_opcode);
        let unresolved = SymbolicCount::unbounded(
            &zero,
            &format!("{effective_opcode}: gate cost requires realization"),
        );
        let depths = external_depth_metrics()
            .iter()
            .map(|(family, _)| {
                (
                    (*family).to_owned(),
                    if deferred && !matches!(*family, "measurements" | "resets") {
                        unresolved.clone()
                    } else if active_depths.contains(family) {
                        one.clone()
                    } else {
                        zero.clone()
                    },
                )
            })
            .collect();
        Ok(Self {
            work: SymbolicCount::exact_value(operation_count),
            span: one.clone(),
            depths: Some(depths),
            opcodes: BTreeMap::from([(
                opcode.to_owned(),
                SymbolicCount::exact_value(operation_count),
            )]),
            extra_metrics: BTreeMap::new(),
            families: Some(if deferred {
                external_family_metrics()
                    .into_iter()
                    .map(|(family, _)| (family.to_owned(), unresolved.clone()))
                    .collect()
            } else {
                logical_gate_families(effective_opcode)
                    .iter()
                    .map(|family| {
                        (
                            (*family).to_owned(),
                            SymbolicCount::exact_value(operation_count),
                        )
                    })
                    .collect()
            }),
            measurements: event_symbolic_count(is_measurement(opcode), opcode, event_width, arena)?,
            resets: event_symbolic_count(is_reset(opcode), opcode, event_width, arena)?,
            cleanup_work: if cleanup {
                SymbolicCount::exact_value(operation_count)
            } else {
                SymbolicCount::exact_value(arena.literal_u128(0)?)
            },
        })
    }

    pub(super) fn then(
        &mut self,
        other: &Self,
        error_bound_metrics: &BTreeSet<String>,
        arena: &mut ExpressionArena,
    ) -> Result<(), SymbolicResourceError> {
        self.work = self.work.then(&other.work, arena)?;
        self.span = self.span.dependency_then(&other.span, arena)?;
        merge_optional_depths(&mut self.depths, other.depths.as_ref(), arena)?;
        for (opcode, count) in &other.opcodes {
            let merged = if let Some(current) = self.opcodes.get(opcode) {
                current.then(count, arena)?
            } else {
                count.clone()
            };
            self.opcodes.insert(opcode.clone(), merged);
        }
        for (identifier, count) in &other.extra_metrics {
            let merged = if let Some(current) = self.extra_metrics.get(identifier) {
                if identifier == &format!("{PORTABLE_PREFIX}clean_ancilla_qubits") {
                    current.peak(count, arena)?
                } else if error_bound_metrics.contains(identifier) {
                    current.probability_then(count, arena)?
                } else {
                    current.then(count, arena)?
                }
            } else {
                count.clone()
            };
            self.extra_metrics.insert(identifier.clone(), merged);
        }
        match (&mut self.families, &other.families) {
            (Some(destination), Some(source)) => {
                for (family, count) in source {
                    let merged = if let Some(current) = destination.get(family) {
                        current.then(count, arena)?
                    } else {
                        count.clone()
                    };
                    destination.insert(family.clone(), merged);
                }
            }
            (_, None) => self.families = None,
            (None, Some(_)) => {}
        }
        merge_optional_symbolic(&mut self.measurements, other.measurements.as_ref(), arena)?;
        merge_optional_symbolic(&mut self.resets, other.resets.as_ref(), arena)?;
        self.cleanup_work = self.cleanup_work.then(&other.cleanup_work, arena)?;
        Ok(())
    }

    pub(super) fn repeat(
        &self,
        trip_count: ExprId,
        binder: ExprId,
        error_bound_metrics: &BTreeSet<String>,
        arena: &mut ExpressionArena,
    ) -> Result<Self, SymbolicResourceError> {
        Ok(Self {
            work: self.work.repeat(trip_count, binder, arena)?,
            span: self.span.dependency_repeat(trip_count, binder, arena)?,
            depths: self
                .depths
                .as_ref()
                .map(|depths| {
                    depths
                        .iter()
                        .map(|(family, count)| {
                            Ok((
                                family.clone(),
                                count.dependency_repeat(trip_count, binder, arena)?,
                            ))
                        })
                        .collect()
                })
                .transpose()?,
            opcodes: self
                .opcodes
                .iter()
                .map(|(opcode, count)| {
                    Ok((opcode.clone(), count.repeat(trip_count, binder, arena)?))
                })
                .collect::<Result<_, SymbolicResourceError>>()?,
            extra_metrics: self
                .extra_metrics
                .iter()
                .map(|(identifier, count)| {
                    Ok((
                        identifier.clone(),
                        if identifier == &format!("{PORTABLE_PREFIX}clean_ancilla_qubits") {
                            count.clone()
                        } else if error_bound_metrics.contains(identifier) {
                            count.probability_repeat(trip_count, binder, arena)?
                        } else {
                            count.repeat(trip_count, binder, arena)?
                        },
                    ))
                })
                .collect::<Result<_, SymbolicResourceError>>()?,
            families: self
                .families
                .as_ref()
                .map(|families| {
                    families
                        .iter()
                        .map(|(family, count)| {
                            Ok((family.clone(), count.repeat(trip_count, binder, arena)?))
                        })
                        .collect()
                })
                .transpose()?,
            measurements: self
                .measurements
                .as_ref()
                .map(|count| count.repeat(trip_count, binder, arena))
                .transpose()?,
            resets: self
                .resets
                .as_ref()
                .map(|count| count.repeat(trip_count, binder, arena))
                .transpose()?,
            cleanup_work: self.cleanup_work.repeat(trip_count, binder, arena)?,
        })
    }

    pub(super) fn cleanup_adjoint(
        &self,
        arena: &mut ExpressionArena,
    ) -> Result<Self, SymbolicResourceError> {
        let zero = SymbolicCount::exact_value(arena.literal_u128(0)?);
        Ok(Self {
            work: self.work.clone(),
            span: self.span.clone(),
            depths: self.depths.clone(),
            opcodes: BTreeMap::from([("qcore.adjoint".to_owned(), self.work.clone())]),
            extra_metrics: self.extra_metrics.clone(),
            families: self.families.clone(),
            measurements: Some(zero.clone()),
            resets: Some(zero),
            cleanup_work: self.work.clone(),
        })
    }

    pub(super) fn invalidate_body_backed_coherent(
        &mut self,
        reason: &str,
        arena: &mut ExpressionArena,
    ) -> Result<(), SymbolicResourceError> {
        let zero = SymbolicCount::exact_value(arena.literal_u128(0)?);
        self.span = SymbolicCount::invalidated(&self.span, zero.lower, reason);
        let direct_families = self.families.take().unwrap_or_default();
        self.families = Some(
            external_family_metrics()
                .iter()
                .map(|(family, _)| {
                    let source = direct_families.get(*family).unwrap_or(&zero);
                    let lower = if *family == "gates" {
                        source.lower
                    } else {
                        zero.lower
                    };
                    (
                        (*family).to_owned(),
                        SymbolicCount::invalidated(source, lower, reason),
                    )
                })
                .collect(),
        );
        let direct_depths = self.depths.take().unwrap_or_default();
        self.depths = Some(
            external_depth_metrics()
                .iter()
                .map(|(family, _)| {
                    let source = direct_depths.get(*family).unwrap_or(&zero);
                    (
                        (*family).to_owned(),
                        SymbolicCount::invalidated(source, zero.lower, reason),
                    )
                })
                .collect(),
        );
        Ok(())
    }

    pub(super) fn branch(
        alternatives: &[Self],
        arena: &mut ExpressionArena,
    ) -> Result<Self, SymbolicResourceError> {
        let opcodes = branch_named_counts(alternatives, |item| &item.opcodes, arena)?;
        let extra_metrics = branch_named_counts(alternatives, |item| &item.extra_metrics, arena)?;
        let zero = SymbolicCount::exact_value(arena.literal_u128(0)?);
        let families = if alternatives.iter().all(|item| item.families.is_some()) {
            let mut families = BTreeMap::new();
            let names = alternatives
                .iter()
                .filter_map(|item| item.families.as_ref())
                .flat_map(|item| item.keys().cloned())
                .collect::<std::collections::BTreeSet<_>>();
            for name in names {
                let values = alternatives
                    .iter()
                    .filter_map(|item| item.families.as_ref())
                    .map(|item| item.get(&name).cloned().unwrap_or_else(|| zero.clone()))
                    .collect::<Vec<_>>();
                families.insert(name, SymbolicCount::branch(&values, arena)?);
            }
            Some(families)
        } else {
            None
        };
        let measurements = branch_optional_symbolic(
            alternatives.iter().map(|item| item.measurements.as_ref()),
            arena,
        )?;
        let resets =
            branch_optional_symbolic(alternatives.iter().map(|item| item.resets.as_ref()), arena)?;
        let depths = if alternatives.iter().all(|item| item.depths.is_some()) {
            let mut depths = BTreeMap::new();
            for (family, _) in external_depth_metrics() {
                let values = alternatives
                    .iter()
                    .filter_map(|item| item.depths.as_ref())
                    .map(|item| item.get(family).cloned())
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        SymbolicResourceError::Missing("scalar dependency-depth family".to_owned())
                    })?;
                depths.insert(family.to_owned(), SymbolicCount::branch(&values, arena)?);
            }
            Some(depths)
        } else {
            None
        };
        Ok(Self {
            work: SymbolicCount::branch(
                &alternatives
                    .iter()
                    .map(|item| item.work.clone())
                    .collect::<Vec<_>>(),
                arena,
            )?,
            span: SymbolicCount::branch(
                &alternatives
                    .iter()
                    .map(|item| item.span.clone())
                    .collect::<Vec<_>>(),
                arena,
            )?,
            depths,
            opcodes,
            extra_metrics,
            families,
            measurements,
            resets,
            cleanup_work: SymbolicCount::branch(
                &alternatives
                    .iter()
                    .map(|item| item.cleanup_work.clone())
                    .collect::<Vec<_>>(),
                arena,
            )?,
        })
    }
}

pub(super) fn event_symbolic_count(
    relevant: bool,
    opcode: &str,
    event_width: Option<ExprId>,
    arena: &mut ExpressionArena,
) -> Result<Option<SymbolicCount>, SymbolicResourceError> {
    if !relevant {
        return Ok(Some(SymbolicCount::exact_value(arena.literal_u128(0)?)));
    }
    let expression = if matches!(
        opcode,
        "quantum.measure_x" | "quantum.measure_y" | "quantum.measure_z"
    ) {
        Some(arena.literal_u128(1)?)
    } else {
        event_width
    };
    Ok(expression.map(SymbolicCount::exact_value))
}

pub(super) fn merge_optional_symbolic(
    total: &mut Option<SymbolicCount>,
    value: Option<&SymbolicCount>,
    arena: &mut ExpressionArena,
) -> Result<(), SymbolicResourceError> {
    *total = match (total.as_ref(), value) {
        (Some(left), Some(right)) => Some(left.then(right, arena)?),
        _ => None,
    };
    Ok(())
}

pub(super) fn merge_optional_depths(
    total: &mut Option<BTreeMap<String, SymbolicCount>>,
    value: Option<&BTreeMap<String, SymbolicCount>>,
    arena: &mut ExpressionArena,
) -> Result<(), SymbolicResourceError> {
    let (Some(destination), Some(source)) = (total.as_mut(), value) else {
        *total = None;
        return Ok(());
    };
    if destination.keys().ne(source.keys()) {
        return Err(SymbolicResourceError::Missing(
            "scalar dependency-depth family".to_owned(),
        ));
    }
    for (family, incoming) in source {
        let current = destination.get(family).cloned().ok_or_else(|| {
            SymbolicResourceError::Missing("scalar dependency-depth family".to_owned())
        })?;
        destination.insert(family.clone(), current.dependency_then(incoming, arena)?);
    }
    Ok(())
}

pub(super) fn branch_optional_symbolic<'a>(
    alternatives: impl Iterator<Item = Option<&'a SymbolicCount>>,
    arena: &mut ExpressionArena,
) -> Result<Option<SymbolicCount>, SymbolicResourceError> {
    let Some(alternatives) = alternatives
        .map(Option::<&SymbolicCount>::cloned)
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    SymbolicCount::branch(&alternatives, arena).map(Some)
}

pub(super) fn external_operation_counts(
    available: &BTreeMap<String, SymbolicCount>,
) -> BTreeMap<String, SymbolicCount> {
    let operation_prefix = "qamomile.logical.operation_count.";
    available
        .iter()
        .filter_map(|(identifier, count)| {
            identifier
                .strip_prefix(operation_prefix)
                .map(|opcode| (opcode.to_owned(), count.clone()))
        })
        .collect()
}

pub(super) fn is_structural_logical_metric(identifier: &str) -> bool {
    identifier.starts_with("qamomile.logical.operation_count.")
        || matches!(
            identifier,
            "qamomile.logical.semantic_work"
                | "qamomile.logical.dependency_span"
                | "qamomile.logical.measurements"
                | "qamomile.logical.resets"
                | "qamomile.logical.cleanup_work"
                | "qamomile.logical.input_qubits"
                | "qamomile.logical.allocated_qubits"
                | "qamomile.logical.static_circuit_qubits"
                | "qamomile.logical.peak_live_qubits"
                | "qamomile.logical.gates"
                | "qamomile.logical.single_qubit_gates"
                | "qamomile.logical.two_qubit_gates"
                | "qamomile.logical.multi_qubit_gates"
                | "qamomile.logical.clifford_gates"
                | "qamomile.logical.rotation_gates"
                | "qamomile.logical.t_gates"
                | "qamomile.logical.toffoli_gates"
                | "qamomile.logical.non_clifford_gates"
                | "qamomile.logical.gate_depth"
                | "qamomile.logical.clifford_depth"
                | "qamomile.logical.rotation_depth"
                | "qamomile.logical.t_depth"
                | "qamomile.logical.toffoli_depth"
                | "qamomile.logical.non_clifford_depth"
                | "qamomile.logical.measurement_depth"
                | "qamomile.logical.reset_depth"
        )
}

pub(super) fn external_extra_metrics(
    available: &BTreeMap<String, SymbolicCount>,
) -> BTreeMap<String, SymbolicCount> {
    available
        .iter()
        .filter(|(identifier, _)| !is_structural_logical_metric(identifier))
        .map(|(identifier, count)| (identifier.clone(), count.clone()))
        .collect()
}

pub(super) fn external_metrics_with_structural_counts(
    available: &BTreeMap<String, SymbolicCount>,
    callable: &str,
    arena: &mut ExpressionArena,
) -> Result<BTreeMap<String, SymbolicCount>, SymbolicResourceError> {
    let mut metrics = external_extra_metrics(available);
    let one = SymbolicCount::exact_value(arena.literal_u128(1)?);
    metrics.insert(
        format!("qamomile.algorithmic.call_count.{callable}"),
        one.clone(),
    );
    metrics.insert(format!("qamomile.algorithmic.query_count.{callable}"), one);
    Ok(metrics)
}
