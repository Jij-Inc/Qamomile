//! Import and interpretation of declarative external resource contracts.

use crate::resource_rules::{
    Domain as ResourceDomain, ExpressionGraph, ExpressionGraphNode, Number as RuleNumber,
    ResourceRuleContract, TransformMode,
};
use crate::types::Dimension;
use crate::wire;
use std::collections::{BTreeMap, HashMap};

use super::{
    Counts, ExprId, ExpressionArena, ResourceExpressionNode, SymbolicAnalyzer, SymbolicCount,
    SymbolicDimensionBindings, SymbolicResourceError, SymbolicTimingFootprint, SymbolicTimingNode,
    SymbolicTimingRoot, SymbolicTimingStep, ValueFact, external_depth_metrics,
    external_family_metrics, external_metrics_with_structural_counts, external_operation_counts,
    int_fact, is_coherent_external, resource_domain, resource_predicate_kind, transform_mode_name,
};

impl SymbolicAnalyzer<'_> {
    pub(super) fn charge_block(&mut self, block: &wire::BlockProto) {
        self.operation_visits = self
            .operation_visits
            .saturating_add(block.operations.len() as u128);
    }

    pub(super) fn instantiate_rule_count(
        &mut self,
        rule: &ResourceRuleContract,
        function: &wire::FunctionProto,
        arguments: &[Option<ValueFact>],
        dimensions: &SymbolicDimensionBindings,
    ) -> Result<SymbolicCount, SymbolicResourceError> {
        let expression =
            self.import_rule_expression(&rule.expression, function, arguments, dimensions)?;
        SymbolicCount::from_rule(&rule.identity, rule.relation, expression, &mut self.arena)
    }

    pub(super) fn external_rule_count(
        &mut self,
        function: &wire::FunctionProto,
        arguments: &[Option<ValueFact>],
        dimensions: &SymbolicDimensionBindings,
        mode: TransformMode,
        identifier: &str,
    ) -> Result<Option<SymbolicCount>, SymbolicResourceError> {
        let rule = self
            .rules
            .get(&(function.symbol.as_str(), mode, identifier))
            .copied();
        let Some(rule) = rule else {
            return Ok(None);
        };
        self.instantiate_rule_count(rule, function, arguments, dimensions)
            .map(Some)
    }

    pub(super) fn external_counts(
        &mut self,
        function: &wire::FunctionProto,
        arguments: &[Option<ValueFact>],
        dimensions: &SymbolicDimensionBindings,
        mode: TransformMode,
    ) -> Result<Counts, SymbolicResourceError> {
        let mut available = BTreeMap::new();
        let matching = self
            .rules
            .iter()
            .filter(|((callable, rule_mode, _), _)| {
                *callable == function.symbol && *rule_mode == mode
            })
            .map(|((_, _, identifier), rule)| ((*identifier).to_owned(), *rule))
            .collect::<Vec<_>>();
        for (identifier, rule) in matching {
            available.insert(
                identifier,
                self.instantiate_rule_count(rule, function, arguments, dimensions)?,
            );
        }
        let unresolved = |label: &str, arena: &mut ExpressionArena| {
            SymbolicCount::unresolved(
                format!(
                    "external {} callable {} has no {label} resource rule",
                    transform_mode_name(mode),
                    function.symbol
                ),
                arena,
            )
        };
        let total_gates = available.get("qamomile.logical.gates").cloned();
        let work = if let Some(work) = available.get("qamomile.logical.semantic_work") {
            work.clone()
        } else if is_coherent_external(function) {
            total_gates
                .clone()
                .map_or_else(|| unresolved("semantic-work", &mut self.arena), Ok)?
        } else {
            unresolved("semantic-work", &mut self.arena)?
        };
        let span = if let Some(span) = available.get("qamomile.logical.dependency_span") {
            span.clone()
        } else if work.upper.is_some() {
            SymbolicCount::bounded_by(&work, &mut self.arena)?
        } else {
            unresolved("dependency-span", &mut self.arena)?
        };
        let zero = SymbolicCount::exact_value(self.arena.literal_u128(0)?);
        let measurements = if let Some(value) = available.get("qamomile.logical.measurements") {
            value.clone()
        } else if is_coherent_external(function) {
            zero.clone()
        } else {
            unresolved("measurement", &mut self.arena)?
        };
        let resets = if let Some(value) = available.get("qamomile.logical.resets") {
            value.clone()
        } else if is_coherent_external(function) {
            zero.clone()
        } else {
            unresolved("reset", &mut self.arena)?
        };
        let cleanup_work = available
            .get("qamomile.logical.cleanup_work")
            .cloned()
            .unwrap_or_else(|| zero.clone());
        let mut families = BTreeMap::new();
        for (family, identifier) in external_family_metrics() {
            let value = if let Some(value) = available.get(identifier) {
                value.clone()
            } else if family == "gates" {
                total_gates
                    .clone()
                    .map_or_else(|| unresolved("gates", &mut self.arena), Ok)?
            } else if let Some(total) = total_gates.as_ref().filter(|total| total.upper.is_some()) {
                SymbolicCount::bounded_by(total, &mut self.arena)?
            } else {
                unresolved(&family.replace('_', "-"), &mut self.arena)?
            };
            families.insert(family.to_owned(), value);
        }
        let opcodes = external_operation_counts(&available);
        let extra_metrics =
            external_metrics_with_structural_counts(&available, &function.symbol, &mut self.arena)?;
        let depths = self.external_depth_counts(
            &available,
            &families,
            &measurements,
            &resets,
            function,
            mode,
        )?;
        Ok(Counts {
            work,
            span,
            depths: Some(depths),
            opcodes,
            extra_metrics,
            families: Some(families),
            measurements: Some(measurements),
            resets: Some(resets),
            cleanup_work,
        })
    }

    pub(super) fn external_timing(
        &mut self,
        function: &wire::FunctionProto,
        dimensions: &SymbolicDimensionBindings,
        mode: TransformMode,
        counts: &Counts,
    ) -> Result<Option<Vec<SymbolicTimingNode>>, SymbolicResourceError> {
        let Some(parameter_names) =
            self.rules
                .iter()
                .find_map(|((callable, rule_mode, _), rule)| {
                    (*callable == function.symbol && *rule_mode == mode)
                        .then_some(rule.scheduling_footprint.as_deref())
                        .flatten()
                })
        else {
            return Ok(None);
        };
        let types = self
            .operations
            .value_types(&function.symbol)
            .ok_or_else(|| {
                SymbolicResourceError::Missing(format!(
                    "type roots for external callable {:?}",
                    function.symbol
                ))
            })?;
        let mut footprints = Vec::with_capacity(parameter_names.len());
        for name in parameter_names {
            let parameter = function
                .parameters
                .iter()
                .find(|parameter| parameter.name == *name)
                .ok_or_else(|| {
                    SymbolicResourceError::Missing(format!(
                        "verified scheduling parameter {name:?}"
                    ))
                })?;
            let value = parameter.value.as_ref().ok_or_else(|| {
                SymbolicResourceError::Missing("external scheduling value".to_owned())
            })?;
            let region = value.region.as_ref().ok_or_else(|| {
                SymbolicResourceError::Missing("external scheduling region".to_owned())
            })?;
            let type_id = types.get(&value.id).copied().ok_or_else(|| {
                SymbolicResourceError::Missing("external scheduling type".to_owned())
            })?;
            footprints.push(SymbolicTimingFootprint {
                root: SymbolicTimingRoot {
                    scope: function.symbol.clone(),
                    root: region.root,
                    projections: region.projections.clone(),
                    instance: Vec::new(),
                },
                selectors: None,
                width: self.coherent_width(type_id, dimensions)?,
            });
        }
        let mut durations = BTreeMap::new();
        if counts.span.exact {
            durations.insert("span".to_owned(), counts.span.lower);
        }
        if let Some(depths) = counts.depths.as_ref() {
            durations.extend(
                depths
                    .iter()
                    .filter(|(_, count)| count.exact)
                    .map(|(family, count)| (family.clone(), count.lower)),
            );
        }
        if durations.is_empty() {
            return Ok(None);
        }
        Ok(Some(vec![SymbolicTimingNode::Step(SymbolicTimingStep {
            footprints: footprints.into_boxed_slice(),
            durations,
        })]))
    }

    pub(super) fn external_depth_counts(
        &mut self,
        available: &BTreeMap<String, SymbolicCount>,
        families: &BTreeMap<String, SymbolicCount>,
        measurements: &SymbolicCount,
        resets: &SymbolicCount,
        function: &wire::FunctionProto,
        mode: TransformMode,
    ) -> Result<BTreeMap<String, SymbolicCount>, SymbolicResourceError> {
        let mut depths = BTreeMap::new();
        for (family, identifier) in external_depth_metrics() {
            let value = if let Some(value) = available.get(identifier) {
                value.clone()
            } else {
                let count = match family {
                    "measurements" => measurements,
                    "resets" => resets,
                    name => families.get(name).ok_or_else(|| {
                        SymbolicResourceError::Missing("external depth fallback family".to_owned())
                    })?,
                };
                if count.upper.is_some() {
                    SymbolicCount::dependency_bounded_by(count, &mut self.arena)?
                } else {
                    SymbolicCount::unresolved_from(
                        count,
                        format!(
                            "{identifier} cannot be bounded from an operation count without a finite upper endpoint for external {} callable {}",
                            transform_mode_name(mode),
                            function.symbol
                        ),
                        &mut self.arena,
                    )?
                }
            };
            depths.insert(family.to_owned(), value);
        }
        Ok(depths)
    }

    pub(super) fn import_rule_expression(
        &mut self,
        graph: &ExpressionGraph,
        function: &wire::FunctionProto,
        arguments: &[Option<ValueFact>],
        dimensions: &SymbolicDimensionBindings,
    ) -> Result<ExprId, SymbolicResourceError> {
        let mut substitutions = function
            .parameters
            .iter()
            .zip(arguments)
            .filter_map(|(parameter, value)| {
                int_fact(value.as_ref()).map(|fact| {
                    (
                        (function.symbol.clone(), parameter.name.clone()),
                        fact.expression,
                    )
                })
            })
            .collect::<HashMap<_, _>>();
        for (dimension, expression) in &dimensions.values {
            if let Dimension::Symbol { scope, name } = dimension {
                substitutions.insert((scope.to_string(), name.to_string()), *expression);
            }
        }
        let mut imported = Vec::with_capacity(graph.nodes().len());
        let mut predicate_values = Vec::with_capacity(graph.nodes().len());
        for node in graph.nodes() {
            let (expression, predicate) = match node {
                ExpressionGraphNode::Predicate(node) => (
                    None,
                    Some(self.evaluate_rule_predicate(node, &imported, &predicate_values)?),
                ),
                ExpressionGraphNode::Expression(node) => {
                    let arguments = node
                        .arguments
                        .iter()
                        .map(|identifier| {
                            imported
                                .get(*identifier as usize)
                                .copied()
                                .flatten()
                                .ok_or_else(|| {
                                    SymbolicResourceError::Missing(
                                        "resource-rule expression operand".to_owned(),
                                    )
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let predicates = node
                        .predicates
                        .iter()
                        .map(|identifier| {
                            predicate_values
                                .get(*identifier as usize)
                                .copied()
                                .flatten()
                                .ok_or_else(|| {
                                    SymbolicResourceError::Missing(
                                        "resource-rule predicate value".to_owned(),
                                    )
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    (
                        Some(self.import_rule_node(
                            node,
                            arguments,
                            &predicates,
                            &substitutions,
                        )?),
                        None,
                    )
                }
            };
            imported.push(expression);
            predicate_values.push(predicate);
        }
        let root = imported
            .get(graph.root() as usize)
            .copied()
            .flatten()
            .ok_or_else(|| {
                SymbolicResourceError::Missing("resource-rule expression root".to_owned())
            })?;
        let declared = graph.expression(graph.root()).map_err(|error| {
            SymbolicResourceError::Missing(format!("checked resource-rule root: {error}"))
        })?;
        if self
            .arena
            .node(root)
            .is_none_or(|node| node.domain() != declared.domain)
        {
            return Err(SymbolicResourceError::Unsupported(
                "resource-rule expression changes its declared domain".to_owned(),
            ));
        }
        Ok(root)
    }

    pub(super) fn import_rule_node(
        &mut self,
        node: &crate::resource_rules::ExpressionNode,
        arguments: Vec<ExprId>,
        predicates: &[ExprId],
        substitutions: &HashMap<(String, String), ExprId>,
    ) -> Result<ExprId, SymbolicResourceError> {
        if node.unit != "1" {
            return Err(SymbolicResourceError::Unsupported(
                "dimensioned resource-rule intermediate".to_owned(),
            ));
        }
        match node.kind.as_str() {
            "literal" => self.import_rule_literal(node),
            "symbol" => self.import_rule_symbol(node, substitutions),
            "add" => self.arena.add(arguments),
            "multiply" => self.arena.multiply(arguments),
            "minimum" => self.arena.extremum(false, arguments),
            "maximum" => self.arena.extremum(true, arguments),
            "power" => self.arena.power(arguments[0], arguments[1]),
            "absolute" => self.arena.absolute(arguments[0], node.domain),
            "floor" => self.arena.floor(arguments[0]),
            "ceiling" => self.arena.ceiling(arguments[0]),
            "floor_divide" => self.arena.floor_divide(arguments[0], arguments[1]),
            "ceiling_divide" => self.arena.ceiling_divide(arguments[0], arguments[1]),
            "exp" => self.arena.exponential(arguments[0]),
            "log" => self.arena.logarithm(arguments),
            "log2" => self.arena.log2(arguments[0]),
            "square_root" => self.arena.square_root(arguments[0]),
            "piecewise" => self.arena.piecewise(predicates, &arguments, node.domain),
            "contracted_function" => self.import_rule_named(node, arguments, false),
            "recurrence" => self.import_rule_named(node, arguments, true),
            "bounded_sum" => self.import_rule_bounded(node, &arguments, false),
            "bounded_product" => self.import_rule_bounded(node, &arguments, true),
            unsupported => Err(SymbolicResourceError::Unsupported(format!(
                "resource-rule expression constructor {unsupported:?}"
            ))),
        }
    }

    pub(super) fn evaluate_rule_predicate(
        &mut self,
        node: &crate::resource_rules::PredicateNode,
        expressions: &[Option<ExprId>],
        predicates: &[Option<ExprId>],
    ) -> Result<ExprId, SymbolicResourceError> {
        let nested = node
            .predicates
            .iter()
            .map(|identifier| predicates.get(*identifier as usize).copied().flatten())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                SymbolicResourceError::Missing("resource-rule predicate operand".to_owned())
            })?;
        match node.kind.as_str() {
            "true" => self.arena.predicate_literal(true),
            "false" => self.arena.predicate_literal(false),
            "and" | "or" | "not" => self
                .arena
                .predicate_connective(resource_predicate_kind(&node.kind)?, nested),
            kind => {
                let arguments = node
                    .arguments
                    .iter()
                    .map(|identifier| {
                        expressions
                            .get(*identifier as usize)
                            .copied()
                            .flatten()
                            .ok_or_else(|| {
                                SymbolicResourceError::Missing(
                                    "resource-rule comparison operand".to_owned(),
                                )
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let [left, right] = arguments.as_slice() else {
                    return Err(SymbolicResourceError::Missing(
                        "resource-rule comparison operands".to_owned(),
                    ));
                };
                self.arena
                    .predicate_comparison(resource_predicate_kind(kind)?, [*left, *right])
            }
        }
    }

    pub(super) fn import_rule_symbol(
        &mut self,
        node: &crate::resource_rules::ExpressionNode,
        substitutions: &HashMap<(String, String), ExprId>,
    ) -> Result<ExprId, SymbolicResourceError> {
        let symbol = node.symbol.as_ref().ok_or_else(|| {
            SymbolicResourceError::Missing("resource-rule symbol declaration".to_owned())
        })?;
        if let Some(substitution) = substitutions
            .get(&(symbol.scope.clone(), symbol.name.clone()))
            .copied()
        {
            if node.domain == ResourceDomain::Natural
                && self
                    .arena
                    .node(substitution)
                    .is_some_and(|value| value.domain() != ResourceDomain::Natural)
            {
                let Some(ResourceExpressionNode::Symbol {
                    scope,
                    name,
                    origin,
                    constraints,
                    ..
                }) = self.arena.node(substitution).cloned()
                else {
                    return Err(SymbolicResourceError::InvalidRuleValue(format!(
                        "Natural parameter {:?} must be nonnegative",
                        symbol.name
                    )));
                };
                if name != symbol.name {
                    return Err(SymbolicResourceError::InvalidRuleValue(
                        "Natural refinement changes symbol identity".to_owned(),
                    ));
                }
                let mut refined_constraints = constraints.into_vec();
                refined_constraints.extend(symbol.constraints.iter().cloned());
                refined_constraints.sort();
                refined_constraints.dedup();
                return self.arena.symbol(
                    &scope,
                    &name,
                    ResourceDomain::Natural,
                    &origin,
                    &refined_constraints
                        .iter()
                        .map(String::as_str)
                        .collect::<Vec<_>>(),
                );
            }
            return Ok(substitution);
        }
        self.arena.symbol(
            &symbol.scope,
            &symbol.name,
            symbol.domain,
            &symbol.origin,
            &symbol
                .constraints
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
        )
    }

    pub(super) fn import_rule_named(
        &mut self,
        node: &crate::resource_rules::ExpressionNode,
        arguments: Vec<ExprId>,
        recurrence: bool,
    ) -> Result<ExprId, SymbolicResourceError> {
        let attributes = node.attributes.iter().cloned().collect::<BTreeMap<_, _>>();
        let name = attributes.get("name").ok_or_else(|| {
            SymbolicResourceError::Missing("resource-rule named expression identity".to_owned())
        })?;
        if attributes.get("version").map(String::as_str) != Some("1") {
            return Err(SymbolicResourceError::Unsupported(
                "non-v1 resource-rule named expression".to_owned(),
            ));
        }
        if recurrence {
            self.arena.recurrence(name, node.domain, arguments)
        } else {
            self.arena.contracted(name, node.domain, arguments)
        }
    }

    pub(super) fn import_rule_bounded(
        &mut self,
        node: &crate::resource_rules::ExpressionNode,
        arguments: &[ExprId],
        product: bool,
    ) -> Result<ExprId, SymbolicResourceError> {
        let attributes = node.attributes.iter().cloned().collect::<BTreeMap<_, _>>();
        let attribute = |name: &str| {
            attributes
                .get(name)
                .ok_or_else(|| SymbolicResourceError::Missing(format!("resource-rule {name}")))
        };
        let binder = self.arena.symbol(
            attribute("binder_scope")?,
            attribute("binder_name")?,
            resource_domain(attribute("binder_domain")?)?,
            attribute("binder_origin")?,
            &[],
        )?;
        if product {
            self.arena
                .bounded_product(binder, arguments[0], arguments[1], arguments[2])
        } else {
            self.arena
                .bounded_sum(binder, arguments[0], arguments[1], arguments[2])
        }
    }

    pub(super) fn import_rule_literal(
        &mut self,
        node: &crate::resource_rules::ExpressionNode,
    ) -> Result<ExprId, SymbolicResourceError> {
        match node.literal.as_ref() {
            Some(RuleNumber::Integer(value)) => {
                self.arena.intern(ResourceExpressionNode::Literal {
                    value: value.to_string(),
                    domain: node.domain,
                })
            }
            Some(RuleNumber::Rational(numerator, denominator)) => {
                self.arena.intern(ResourceExpressionNode::RationalLiteral {
                    numerator: numerator.to_string(),
                    denominator: denominator.to_string(),
                    domain: node.domain,
                })
            }
            Some(RuleNumber::RealBits(bits)) => {
                self.arena.intern(ResourceExpressionNode::RealLiteral {
                    bits: *bits,
                    domain: node.domain,
                })
            }
            None => Err(SymbolicResourceError::Missing(
                "resource-rule literal".to_owned(),
            )),
        }
    }
}
