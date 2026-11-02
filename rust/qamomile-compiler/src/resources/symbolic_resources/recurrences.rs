//! Finite, scoped additive recurrence extraction, without circuit expansion.

use super::{
    BlockExit, BlockSummary, Counts, ExprId, ExpressionArena, MetricComposition,
    ResourceExpressionNode, ResourcePredicateKind, ScalarFact, SymbolicAnalyzer, SymbolicCount,
    SymbolicResourceError, ValueFact, external_depth_metrics, external_family_metrics, int_fact,
    scalar_type, statically_reachable_quantum_opcodes,
};
use crate::call_graph::StaticCallGraph;
use crate::resource_rules::{Domain as ResourceDomain, TransformMode};
use crate::wire;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

const MAX_RECURRENCE_CONTEXTS: usize = 4096;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct RecurrenceKey {
    symbol: String,
    mode: TransformMode,
    controls: Option<u32>,
}

impl RecurrenceKey {
    fn identity(&self, metric: &str) -> String {
        format!(
            "qcore:{}:{:?}:{:?}:{metric}",
            self.symbol, self.mode, self.controls
        )
    }
}

#[derive(Clone)]
pub(super) struct RecurrenceRequest {
    parameters: Vec<ExprId>,
    arguments: Vec<Option<ValueFact>>,
    counts: Option<Counts>,
    condition: ExprId,
}

struct RecurrenceEquation {
    parameters: Box<[ExprId]>,
    condition: ExprId,
    body: SymbolicCount,
}

pub(super) struct RecurrenceAnalysis {
    symbols: HashSet<String>,
    definitions: BTreeMap<String, RecurrenceEquation>,
    closure_work: u128,
    opcodes: BTreeSet<String>,
    external_metrics: BTreeSet<String>,
    pub(super) encountered: bool,

    pub(super) requests: BTreeMap<RecurrenceKey, RecurrenceRequest>,
}

impl RecurrenceAnalysis {
    pub(super) fn new(graph: &StaticCallGraph, package: &wire::QCorePackageProto) -> Self {
        let mut reachable = HashSet::new();
        let mut pending = graph
            .index(&package.entrypoint)
            .into_iter()
            .collect::<Vec<_>>();
        while let Some(index) = pending.pop() {
            if reachable.insert(index) {
                pending.extend(graph.callees(index).unwrap_or_default());
            }
        }
        let external_metrics = reachable
            .into_iter()
            .filter_map(|index| {
                let function = &package.functions[index];
                function.external.then_some(function.symbol.as_str())
            })
            .flat_map(|symbol| {
                [
                    format!("qamomile.algorithmic.call_count.{symbol}"),
                    format!("qamomile.algorithmic.query_count.{symbol}"),
                ]
            })
            .collect();
        Self {
            external_metrics,
            symbols: if graph.has_recursion() {
                graph
                    .components()
                    .iter()
                    .filter(|component| component.is_recursive())
                    .flat_map(crate::call_graph::CallComponent::members)
                    .map(|index| package.functions[*index].symbol.clone())
                    .collect()
            } else {
                HashSet::new()
            },
            opcodes: package
                .functions
                .iter()
                .filter(|function| function.symbol == package.entrypoint)
                .flat_map(|function| {
                    let functions = package
                        .functions
                        .iter()
                        .map(|function| (function.symbol.as_str(), function))
                        .collect();
                    statically_reachable_quantum_opcodes(function, &functions)
                })
                .collect(),
            encountered: false,

            requests: BTreeMap::new(),
            definitions: BTreeMap::new(),
            closure_work: 0,
        }
    }

    pub(super) fn contains(&self, symbol: &str) -> bool {
        self.symbols.contains(symbol)
    }
}

impl Counts {
    /// Visit additive coordinates only; scheduling and liveness are separate algebras.
    fn map_additive(
        &mut self,
        mut map: impl FnMut(&str, &mut SymbolicCount) -> Result<(), SymbolicResourceError>,
    ) -> Result<(), SymbolicResourceError> {
        map("work", &mut self.work)?;
        map("cleanup_work", &mut self.cleanup_work)?;
        for (name, value) in &mut self.opcodes {
            map(&format!("opcode:{name}"), value)?;
        }
        for (name, value) in &mut self.extra_metrics {
            map(&format!("metric:{name}"), value)?;
        }
        if let Some(families) = &mut self.families {
            for (name, value) in families {
                map(&format!("family:{name}"), value)?;
            }
        }
        if let Some(value) = &mut self.measurements {
            map("measurements", value)?;
        }
        if let Some(value) = &mut self.resets {
            map("resets", value)?;
        }
        Ok(())
    }

    fn additive(&self) -> Result<BTreeMap<String, SymbolicCount>, SymbolicResourceError> {
        let mut counts = self.clone();
        let mut result = BTreeMap::new();
        counts.map_additive(|name, count| {
            result.insert(name.to_owned(), count.clone());
            Ok(())
        })?;
        Ok(result)
    }

    pub(super) fn conditional(
        left: &Self,
        right: &Self,
        predicate: ExprId,
        arena: &mut ExpressionArena,
    ) -> Result<Self, SymbolicResourceError> {
        let mut result = Self::branch(&[left.clone(), right.clone()], arena)?;
        let left = left.additive()?;
        let right = right.additive()?;
        let zero = SymbolicCount::exact_value(arena.literal_u128(0)?);
        result.map_additive(|name, count| {
            let left = left.get(name).unwrap_or(&zero);
            let right = right.get(name).unwrap_or(&zero);
            count.lower = arena.piecewise(
                &[predicate],
                &[left.lower, right.lower],
                ResourceDomain::Natural,
            )?;
            count.upper = left
                .upper
                .zip(right.upper)
                .map(|(left, right)| {
                    arena.piecewise(&[predicate], &[left, right], ResourceDomain::Natural)
                })
                .transpose()?;
            count.exact = left.exact && right.exact;
            Ok(())
        })?;
        Ok(result)
    }
}

impl SymbolicAnalyzer<'_> {
    pub(super) fn guard_continuation(
        &mut self,
        counts: &Counts,
        optional: bool,
        predicate: Option<ExprId>,
    ) -> Result<Counts, SymbolicResourceError> {
        if let Some(predicate) = predicate {
            let zero = Counts::zero(&mut self.arena)?;
            Counts::conditional(counts, &zero, predicate, &mut self.arena)
        } else {
            self.optional_counts(counts, optional)
        }
    }

    pub(super) fn fallthrough_predicate(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ValueFact>>,
        previous: Option<ExprId>,
    ) -> Result<Option<ExprId>, SymbolicResourceError> {
        if !self.recurrences.encountered {
            return Ok(None);
        }
        let Some(ValueFact::Scalar(ScalarFact::Predicate(predicate))) = operation
            .operands
            .first()
            .and_then(|operand| environment.get(operand))
            .and_then(Option::as_ref)
        else {
            return Ok(None);
        };
        let exits = operation
            .regions
            .iter()
            .map(|region| {
                region
                    .blocks
                    .first()
                    .and_then(|block| block.terminator.as_ref())
                    .map(|terminator| terminator.kind.as_str())
            })
            .collect::<Vec<_>>();
        let predicate = match exits.as_slice() {
            [Some("yield"), Some("return")] => *predicate,
            [Some("return"), Some("yield")] => self
                .arena
                .predicate_connective(ResourcePredicateKind::Not, [*predicate])?,
            _ => return Ok(None),
        };
        let predicate = if let Some(previous) = previous {
            self.arena
                .predicate_connective(ResourcePredicateKind::And, [previous, predicate])?
        } else {
            predicate
        };
        Ok(Some(predicate))
    }

    #[expect(
        clippy::too_many_lines,
        reason = "Keep recurrence signature reservation and its complete count projection together"
    )]
    pub(super) fn recursive_summary(
        &mut self,
        function: &wire::FunctionProto,
        arguments: &[Option<ValueFact>],
        mode: TransformMode,
        controls: Option<u32>,
    ) -> Result<BlockSummary, SymbolicResourceError> {
        self.recurrences.encountered = true;
        let key = RecurrenceKey {
            symbol: function.symbol.clone(),
            mode,
            controls,
        };
        let mut operands = Vec::new();
        let mut formals = Vec::new();
        let mut formal_arguments = Vec::new();
        let mut reason = None;
        for (parameter, argument) in function.parameters.iter().zip(arguments) {
            let value = parameter
                .value
                .as_ref()
                .ok_or_else(|| SymbolicResourceError::Missing("recursive parameter".to_owned()))?;
            if value.has_region {
                formal_arguments.push(None);
                continue;
            }
            if scalar_type(value.r#type.as_ref()) != Some(("Int", Some("Const"))) {
                reason = Some("recursive additive estimation currently requires Const integer classical parameters".to_owned());
                break;
            }
            let Some(argument) = int_fact(argument.as_ref()) else {
                reason = Some(
                    "recursive Const argument is not a supported symbolic integer expression"
                        .to_owned(),
                );
                break;
            };
            operands.push(argument.expression);
            let domain = self
                .parameter_fact(function, parameter)?
                .as_ref()
                .and_then(|fact| int_fact(Some(fact)))
                .and_then(|fact| self.arena.node(fact.expression))
                .map_or(ResourceDomain::Integer, ResourceExpressionNode::domain);
            let formal = self.arena.symbol(
                &format!("resource.recurrence.{}", key.identity("parameters")),
                &parameter.name,
                domain,
                "recurrence_parameter",
                &[],
            )?;
            formals.push(formal);
            formal_arguments.push(Some(ValueFact::Scalar(ScalarFact::Int {
                expression: formal,
                exact: None,
            })));
        }
        let mut counts = Counts::zero(&mut self.arena)?;
        // A recursive call contributes every reachable additive coordinate, including
        // zero families. This preserves the termination condition even for zero cost.
        for opcode in self.recurrences.opcodes.iter().cloned() {
            counts.opcodes.insert(
                opcode,
                SymbolicCount::exact_value(self.arena.literal_u128(0)?),
            );
        }
        counts.families = Some(
            external_family_metrics()
                .iter()
                .map(|(name, _)| ((*name).to_owned(), counts.work.clone()))
                .collect(),
        );
        for metric in &self.recurrences.external_metrics {
            counts
                .extra_metrics
                .insert(metric.clone(), counts.work.clone());
        }
        for rule in self.rules.values() {
            let count = if rule.metric.composition == MetricComposition::AdditiveWork
                && rule.metric.domain == ResourceDomain::Natural
            {
                counts.work.clone()
            } else {
                SymbolicCount::unresolved(
                    "recursive metric requires a supported natural additive recurrence".to_owned(),
                    &mut self.arena,
                )?
            };
            counts
                .extra_metrics
                .insert(rule.metric.identifier.clone(), count);
        }
        if let Some(reason) = reason {
            counts.map_additive(|_, count| {
                *count = SymbolicCount::unresolved(reason.clone(), &mut self.arena)?;
                Ok(())
            })?;
        } else {
            if !self.recurrences.requests.contains_key(&key) {
                if self.recurrences.requests.len() >= MAX_RECURRENCE_CONTEXTS {
                    return Err(SymbolicResourceError::Unsupported(
                        "recursive resource context budget exceeded".to_owned(),
                    ));
                }
                let zero = self.arena.literal_u128(0)?;
                let mut conditions = Vec::new();
                for formal in &formals {
                    if self
                        .arena
                        .node(*formal)
                        .is_some_and(|node| node.domain() == ResourceDomain::Natural)
                    {
                        conditions.push(self.arena.predicate_comparison(
                            ResourcePredicateKind::GreaterEqual,
                            [*formal, zero],
                        )?);
                    }
                }
                let condition = self
                    .arena
                    .predicate_connective(ResourcePredicateKind::And, conditions)?;
                self.recurrences.requests.insert(
                    key.clone(),
                    RecurrenceRequest {
                        parameters: formals,
                        arguments: formal_arguments,
                        counts: None,
                        condition,
                    },
                );
            }
            counts.map_additive(|name, count| {
                if !count.exact {
                    return Ok(());
                }
                *count = SymbolicCount::exact_value(self.arena.recurrence(
                    &key.identity(name),
                    ResourceDomain::Natural,
                    operands.iter().copied(),
                )?);
                Ok(())
            })?;
        }
        let unavailable = SymbolicCount::unresolved(
            "recursive dependency depth requires a scheduling recurrence".to_owned(),
            &mut self.arena,
        )?;
        counts.span = unavailable.clone();
        counts.depths = Some(
            external_depth_metrics()
                .iter()
                .map(|(name, _)| ((*name).to_owned(), unavailable.clone()))
                .collect(),
        );
        Ok(BlockSummary {
            counts,
            exit: BlockExit::RETURN,
            values: vec![None; 1],
            timing: None,
        })
    }

    pub(super) fn extract_recurrence_equations(&mut self) -> Result<(), SymbolicResourceError> {
        while let Some((key, request)) = self
            .recurrences
            .requests
            .iter()
            .find(|(_, request)| request.counts.is_none())
            .map(|(key, request)| (key.clone(), request.clone()))
        {
            let function = self.functions[key.symbol.as_str()];
            let mut summary = self.analyze_function_mode_uncached(
                function,
                &request.arguments,
                &HashMap::new(),
                key.mode,
                key.controls,
            )?;
            self.finish_portable_counts(&mut summary.counts)?;
            self.recurrences
                .requests
                .get_mut(&key)
                .expect("reserved recurrence")
                .counts = Some(summary.counts);
        }
        for (key, request) in &self.recurrences.requests {
            for (metric, body) in request
                .counts
                .as_ref()
                .expect("extracted equation")
                .additive()?
            {
                self.recurrences.definitions.insert(
                    key.identity(&metric),
                    RecurrenceEquation {
                        parameters: request.parameters.clone().into_boxed_slice(),
                        condition: request.condition,
                        body,
                    },
                );
            }
        }
        Ok(())
    }

    pub(super) fn close_recurrence_counts(
        &mut self,
        counts: &mut Counts,
    ) -> Result<(), SymbolicResourceError> {
        if self.recurrences.encountered && self.observation.is_some() {
            for count in counts.extra_metrics.values_mut() {
                *count = SymbolicCount::unresolved_from(
                    count,
                    "recursive observation and portable projection summaries are not yet supported"
                        .to_owned(),
                    &mut self.arena,
                )?;
            }
        }
        if self.recurrences.encountered {
            for rule in self.rules.values().filter(|rule| {
                rule.metric.composition != MetricComposition::AdditiveWork
                    || rule.metric.domain != ResourceDomain::Natural
            }) {
                if let Some(count) = counts.extra_metrics.get_mut(&rule.metric.identifier) {
                    *count = SymbolicCount::unresolved_from(
                        count,
                        "recursive metric requires a supported natural additive recurrence"
                            .to_owned(),
                        &mut self.arena,
                    )?;
                }
            }
        }
        if self.recurrences.requests.is_empty() {
            return Ok(());
        }
        counts.map_additive(|_, count| self.close_recurrence_count(count))
    }

    fn close_recurrence_count(
        &mut self,
        count: &mut SymbolicCount,
    ) -> Result<(), SymbolicResourceError> {
        let mut pending = vec![count.lower];
        pending.extend(count.upper);
        let mut visited = HashSet::new();
        let mut required = BTreeSet::new();
        let mut rule_identities = count
            .rule_identities
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        while let Some(identifier) = pending.pop() {
            self.recurrences.closure_work += 1;
            if self.recurrences.closure_work > crate::budget::MAX_RECURSIVE_ANALYSIS_WORK {
                return Err(SymbolicResourceError::Unsupported(
                    "recursive equation closure exceeds the work budget".to_owned(),
                ));
            }
            if !visited.insert(identifier) {
                continue;
            }
            let node = self.arena.node(identifier).ok_or_else(|| {
                SymbolicResourceError::Missing("recurrence expression".to_owned())
            })?;
            if let ResourceExpressionNode::RecurrenceSystem { .. } = node {
                continue;
            }
            if let ResourceExpressionNode::Recurrence { name, .. } = node
                && required.insert(name.clone())
            {
                let Some(definition) = self.recurrences.definitions.get(name) else {
                    *count = SymbolicCount::unresolved_from(
                        count,
                        format!("recursive resource equation {name} has no supported definition"),
                        &mut self.arena,
                    )?;
                    return Ok(());
                };
                let body = &definition.body;
                if !body.exact {
                    *count = SymbolicCount::unresolved_from(
                        count,
                        format!(
                            "recursive resource equation {name} has unresolved additive contributors: {}",
                            body.unknown.join("; ")
                        ),
                        &mut self.arena,
                    )?;
                    count.rule_identities = super::count_algebra::merge_rule_identities([
                        count.rule_identities.as_ref(),
                        body.rule_identities.as_ref(),
                    ]);
                    return Ok(());
                }
                rule_identities.extend(body.rule_identities.iter().cloned());
                pending.push(body.lower);
                pending.push(definition.condition);
            }
            pending.extend(node.dependencies());
        }
        if required.is_empty() {
            return Ok(());
        }
        let mut equations = Vec::new();
        for name in required {
            let definition = &self.recurrences.definitions[&name];
            let mut arguments = definition.parameters.to_vec();
            arguments.push(definition.body.lower);
            equations.push(
                self.arena
                    .intern(ResourceExpressionNode::RecurrenceEquation {
                        name,
                        domain: ResourceDomain::Natural,
                        arguments: arguments.into_boxed_slice(),
                        condition: definition.condition,
                    })?,
            );
        }
        let wrap = |root, arena: &mut ExpressionArena| {
            arena.intern(ResourceExpressionNode::RecurrenceSystem {
                domain: ResourceDomain::Natural,
                arguments: std::iter::once(root)
                    .chain(equations.iter().copied())
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            })
        };
        count.rule_identities = rule_identities
            .into_iter()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        count.lower = wrap(count.lower, &mut self.arena)?;
        count.upper = count
            .upper
            .map(|root| wrap(root, &mut self.arena))
            .transpose()?;
        Ok(())
    }

    pub(super) fn close_recurrence_facts(&mut self) -> Result<(), SymbolicResourceError> {
        let mut facts = std::mem::take(&mut self.facts);
        for fact in &mut facts {
            let mut counts = Counts::zero(&mut self.arena)?;
            counts.work = fact.work.clone();
            counts.opcodes = fact.opcodes.clone();
            counts.extra_metrics = fact.extra_metrics.clone();
            counts.families.clone_from(&fact.families);
            counts.measurements.clone_from(&fact.measurements);
            counts.resets.clone_from(&fact.resets);
            counts.cleanup_work = fact.cleanup_work.clone();
            self.close_recurrence_counts(&mut counts)?;
            fact.work = counts.work;
            fact.opcodes = counts.opcodes;
            fact.extra_metrics = counts.extra_metrics;
            fact.families = counts.families;
            fact.measurements = counts.measurements;
            fact.resets = counts.resets;
            fact.cleanup_work = counts.cleanup_work;
        }
        self.facts = facts;
        Ok(())
    }
}
