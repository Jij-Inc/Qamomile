//! Bounded exact simplification of verified `QCore` before target lowering.
//!
//! The persistent program remains the semantic and resource authority. This
//! pass rewrites a transient copy, seals it again, and requires the complete
//! verifier (including cleanup-proof replay) to accept the result before any
//! target may consume it.

use std::collections::{HashMap, HashSet};

use thiserror::Error;

use super::optimization_pipeline::{WorkPreflightError, preflight_work};
use crate::number_theory::{CheckedModulus, ModularUnit};
use crate::operations::{
    GateOpcode, QCoreOpcode, QuantumOpcode, SelectorOperand, primitive_gate_spec, selector_operands,
};
use crate::static_values::{self, StaticScalar};
use crate::types::{SemanticType, TypeArena};
use crate::wire;
use crate::{CompilerError, VerifiedProgram, seal_package};

const MAX_OPTIMIZATION_DEPTH: usize = 256;
const MAX_GATE_SELECTORS: usize = 3;

/// Independent optional standard rewrites, shared by built-in and extension passes.
#[derive(Clone, Copy, Debug)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Independent optional rewrite switches are an orthogonal product, not lifecycle states"
)]
pub struct StandardOptimizationOptions {
    /// Remove proven identity operations.
    pub identities: bool,
    /// Cancel inverse gates and callables.
    pub inverses: bool,
    /// Simplify tuple, unpack, and reshape round trips.
    pub structure: bool,
    /// Remove immediately released quantum roots.
    pub vacuous: bool,
    /// Remove dead total constant and stage-lift operations.
    pub dead_constants: bool,
}

impl Default for StandardOptimizationOptions {
    fn default() -> Self {
        Self {
            identities: true,
            inverses: true,
            structure: true,
            vacuous: true,
            dead_constants: true,
        }
    }
}

impl StandardOptimizationOptions {
    /// Decode the fixed native option projection without another semantics owner.
    #[must_use]
    pub fn from_flags(flags: &[bool]) -> Option<Self> {
        let [identities, inverses, structure, vacuous, dead_constants] = flags else {
            return None;
        };
        Some(Self {
            identities: *identities,
            inverses: *inverses,
            structure: *structure,
            vacuous: *vacuous,
            dead_constants: *dead_constants,
        })
    }

    /// Return the canonical option identity used by pipeline provenance.
    #[must_use]
    pub fn configuration(self) -> Vec<u8> {
        [
            self.identities,
            self.inverses,
            self.structure,
            self.vacuous,
            self.dead_constants,
        ]
        .map(u8::from)
        .to_vec()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CanonicalSelectors {
    values: [StaticScalar; MAX_GATE_SELECTORS],
    len: u8,
}

impl CanonicalSelectors {
    fn iter(self) -> impl Iterator<Item = StaticScalar> {
        self.values.into_iter().take(usize::from(self.len))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GateSupport {
    selectors: [i64; MAX_GATE_SELECTORS],
    len: u8,
}

impl GateSupport {
    fn from_selectors(selectors: CanonicalSelectors) -> Option<Self> {
        let mut support = Self {
            selectors: [0; MAX_GATE_SELECTORS],
            len: 0,
        };
        for selector in selectors.iter() {
            let StaticScalar::Int(selector) = selector else {
                return None;
            };
            if selector < 0 {
                return None;
            }
            if !support.iter().any(|existing| existing == selector) {
                support.selectors[usize::from(support.len)] = selector;
                support.len += 1;
            }
        }
        Some(support)
    }

    fn iter(self) -> impl Iterator<Item = i64> {
        self.selectors.into_iter().take(usize::from(self.len))
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct OperationAnalysis {
    selectors: Option<CanonicalSelectors>,
    support: Option<GateSupport>,
}

/// Failure of the standard exact `QCore` optimization pipeline.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum StandardOptimizationError {
    /// The input or transformed program failed a canonical compiler boundary.
    #[error("{0}")]
    Compiler(#[from] CompilerError),
    /// A selected source operation has a statically known domain error.
    #[error(transparent)]
    InvalidDomain(#[from] super::static_diagnostics::StaticDomainError),
    /// A verified graph exceeded the explicit nested-region work bound.
    #[error("Q7801: standard optimization exceeded the region-depth limit")]
    RegionDepthExceeded,
    /// The requested native operation-visit budget cannot cover this graph.
    #[error("Q7803: standard optimization exceeded the work budget")]
    WorkBudgetExceeded,
}

/// Diagnostic work and changes made by standard optimization.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StandardOptimizationWork {
    operation_visits: u128,
    removed_operations: u128,
}

impl StandardOptimizationWork {
    /// Return the number of visited `QCore` operations.
    #[must_use]
    pub const fn operation_visits(self) -> u128 {
        self.operation_visits
    }

    /// Return the number of operations absent from the optimized image.
    #[must_use]
    pub const fn removed_operations(self) -> u128 {
        self.removed_operations
    }

    fn visit(&mut self) {
        self.operation_visits = self.operation_visits.saturating_add(1);
    }

    fn remove(&mut self, count: usize) {
        self.removed_operations = self.removed_operations.saturating_add(count as u128);
    }
}

/// Optimize one verified program and independently verify the transient result.
///
/// # Errors
///
/// Returns [`StandardOptimizationError`] if nested work exceeds the explicit
/// bound or the transformed package fails sealing or semantic verification.
pub fn optimize_standard(
    program: &VerifiedProgram,
) -> Result<(VerifiedProgram, StandardOptimizationWork), StandardOptimizationError> {
    optimize_with_options(program, StandardOptimizationOptions::default())
}

fn optimize_with_options(
    program: &VerifiedProgram,
    options: StandardOptimizationOptions,
) -> Result<(VerifiedProgram, StandardOptimizationWork), StandardOptimizationError> {
    super::static_diagnostics::check(program)?;
    let mut package = program.local.checked.package.clone();
    let mut work = StandardOptimizationWork::default();
    for function in &mut package.functions {
        let facts = function_facts(function);
        if let Some(body) = function.body.as_mut() {
            optimize_block_with_options(
                body,
                &facts.values,
                &facts.owned_values,
                0,
                &mut work,
                options,
            )?;
        }
    }
    let payload = seal_package(package)?;
    Ok((VerifiedProgram::from_bytes(&payload)?, work))
}

pub(crate) fn optimize_standard_bounded(
    program: &VerifiedProgram,
    max_work: u64,
    options: StandardOptimizationOptions,
) -> Result<(VerifiedProgram, StandardOptimizationWork), StandardOptimizationError> {
    // Two visits per operation: the rewrite walk and dead-constant elimination.
    preflight_work(&program.local.checked.package, max_work, 2).map_err(|error| match error {
        WorkPreflightError::Budget => StandardOptimizationError::WorkBudgetExceeded,
        WorkPreflightError::Depth => StandardOptimizationError::RegionDepthExceeded,
    })?;
    optimize_with_options(program, options)
}

#[cfg(test)]
fn optimize_block(
    block: &mut wire::BlockProto,
    values: &HashMap<u64, wire::ValueProto>,
    owned_values: &HashSet<u64>,
    depth: usize,
    work: &mut StandardOptimizationWork,
) -> Result<(), StandardOptimizationError> {
    optimize_block_with_options(
        block,
        values,
        owned_values,
        depth,
        work,
        StandardOptimizationOptions::default(),
    )
}

fn optimize_block_with_options(
    block: &mut wire::BlockProto,
    values: &HashMap<u64, wire::ValueProto>,
    owned_values: &HashSet<u64>,
    depth: usize,
    work: &mut StandardOptimizationWork,
    options: StandardOptimizationOptions,
) -> Result<(), StandardOptimizationError> {
    if depth >= MAX_OPTIMIZATION_DEPTH {
        return Err(StandardOptimizationError::RegionDepthExceeded);
    }

    for operation in &mut block.operations {
        for region in &mut operation.regions {
            for nested in &mut region.blocks {
                optimize_block_with_options(
                    nested,
                    values,
                    owned_values,
                    depth + 1,
                    work,
                    options,
                )?;
            }
        }
    }

    let source = std::mem::take(&mut block.operations);
    let preserves_cleanup_trace = source.iter().any(|operation| {
        string_attribute(operation, "proof") == Some("modular-adjoint")
            && !matches!(
                operation.opcode.as_str(),
                "quantum.release" | "quantum.unprepare"
            )
    });
    let mut state = BlockRewriteState::new(values, owned_values, source.len(), work);

    for mut operation in source {
        state.work.visit();
        state.rewrite_operands(&mut operation.operands);

        if options.identities && !preserves_cleanup_trace && state.eliminate_identity(&operation) {
            continue;
        }

        if options.structure && !preserves_cleanup_trace && state.simplify_structure(&operation) {
            continue;
        }

        let analysis = if is_single_owned_transform(&operation, state.owned_values) {
            operation_analysis(&operation, &state.constants)
        } else {
            OperationAnalysis::default()
        };
        if options.inverses
            && !preserves_cleanup_trace
            && state.cancel_inverse(&operation, analysis)
        {
            continue;
        }

        if options.vacuous && state.eliminate_vacuous_owner(&operation) {
            continue;
        }

        state.retain(operation, analysis, !preserves_cleanup_trace);
    }

    state.finish(block, options);
    Ok(())
}

struct BlockRewriteState<'a> {
    values: &'a HashMap<u64, wire::ValueProto>,
    owned_values: &'a HashSet<u64>,
    operations: Vec<Option<wire::OperationProto>>,
    retained_links: Vec<Option<RetainedLinks>>,
    retained_tail: Option<usize>,
    producers: HashMap<u64, usize>,
    replacements: HashMap<u64, u64>,
    constants: HashMap<u64, wire::LiteralProto>,
    frontiers: DependencyFrontiers,
    work: &'a mut StandardOptimizationWork,
}

impl<'a> BlockRewriteState<'a> {
    fn new(
        values: &'a HashMap<u64, wire::ValueProto>,
        owned_values: &'a HashSet<u64>,
        capacity: usize,
        work: &'a mut StandardOptimizationWork,
    ) -> Self {
        Self {
            values,
            owned_values,
            operations: Vec::with_capacity(capacity),
            retained_links: Vec::with_capacity(capacity),
            retained_tail: None,
            producers: HashMap::new(),
            replacements: HashMap::new(),
            constants: HashMap::new(),
            frontiers: DependencyFrontiers::default(),
            work,
        }
    }

    fn rewrite_operands(&self, operands: &mut [u64]) {
        rewrite_operands(operands, &self.replacements);
    }

    fn replace(&mut self, result: u64, replacement: u64) {
        self.replacements.insert(result, replacement);
    }

    fn eliminate_identity(&mut self, operation: &wire::OperationProto) -> bool {
        if operation.results.len() != 1 || operation.operands.is_empty() {
            return false;
        }
        let zero_angle = primitive_gate_spec(&operation.opcode)
            .filter(|spec| spec.has_angle())
            .and_then(|_| operation.operands.get(1))
            .and_then(|identifier| self.constants.get(identifier))
            .is_some_and(literal_is_zero);
        let zero_angle = zero_angle && primitive_selectors_are_valid(operation, &self.constants);
        if !zero_angle && !self.arithmetic_identity(operation) {
            return false;
        }
        self.replace(operation.results[0].id, operation.operands[0]);
        self.work.remove(1);
        true
    }

    fn arithmetic_identity(&self, operation: &wire::OperationProto) -> bool {
        let Some(opcode) = QuantumOpcode::parse(&operation.opcode) else {
            return false;
        };
        let parameter_count = match opcode {
            QuantumOpcode::AddConstant | QuantumOpcode::SubtractConstant => 1,
            QuantumOpcode::ModAddConstant
            | QuantumOpcode::ModSubConstant
            | QuantumOpcode::ModMul
            | QuantumOpcode::ModDiv
            | QuantumOpcode::ModPow
            | QuantumOpcode::ModPowInverse => 2,
            _ => return false,
        };
        if operation.operands.len() != parameter_count + 1
            || !is_single_owned_transform(operation, self.owned_values)
            || !values_are_substitutable(&operation.results[0], operation.operands[0], self.values)
        {
            return false;
        }
        let integer = |index: usize| {
            self.constants
                .get(&operation.operands[index])
                .and_then(static_values::literal)
                .and_then(|value| match value {
                    StaticScalar::Int(value) => Some(value),
                    _ => None,
                })
        };
        let Some(value) = integer(1) else {
            return false;
        };
        let Some(width) = arithmetic_target_width(operation, opcode) else {
            return false;
        };
        if parameter_count == 1 {
            // Divisibility by 2**width needs neither a shift nor a domain table.
            return value == 0 || width <= u64::from(value.unsigned_abs().trailing_zeros());
        }
        let Some(modulus) = integer(2).and_then(|modulus| CheckedModulus::new(width, modulus))
        else {
            return false;
        };
        match opcode {
            QuantumOpcode::ModAddConstant | QuantumOpcode::ModSubConstant => {
                modulus.is_additive_identity(value)
            }
            _ => modulus.unit(value).is_some_and(ModularUnit::is_identity),
        }
    }

    fn simplify_structure(&mut self, operation: &wire::OperationProto) -> bool {
        self.simplify_reshape(operation)
            || self.simplify_unpack(operation)
            || self.simplify_tuple(operation)
    }

    fn simplify_reshape(&mut self, operation: &wire::OperationProto) -> bool {
        if operation.opcode != "qcore.reshape" || operation.results.len() != 1 {
            return false;
        }
        let Some(input) = operation.operands.first().copied() else {
            return false;
        };
        if values_are_substitutable(&operation.results[0], input, self.values) {
            self.replace(operation.results[0].id, input);
            self.discard_current(operation);
            self.work.remove(1);
            return true;
        }
        let Some(index) = self.producers.get(&input).copied() else {
            return false;
        };
        if self.retained_tail != Some(index) {
            return false;
        }
        let Some(previous) = self.operations.get(index).and_then(Option::as_ref) else {
            return false;
        };
        if previous.opcode != "qcore.reshape"
            || previous.operands.len() != 1
            || previous.results.len() != 1
        {
            return false;
        }
        let original = previous.operands[0];
        if values_are_substitutable(&operation.results[0], original, self.values) {
            self.remove_retained(index);
            self.replace(operation.results[0].id, original);
            self.discard_current(operation);
            self.work.remove(2);
            return true;
        }
        let previous = self.operations[index]
            .as_mut()
            .expect("retained reshape was checked above");
        previous.results[0].clone_from(&operation.results[0]);
        self.producers.remove(&input);
        self.producers.insert(operation.results[0].id, index);
        self.discard_current(operation);
        self.work.remove(1);
        true
    }

    fn simplify_unpack(&mut self, operation: &wire::OperationProto) -> bool {
        if operation.opcode != "qcore.unpack" || operation.operands.len() != 1 {
            return false;
        }
        let input = operation.operands[0];
        let Some(index) = self.producers.get(&input).copied() else {
            return false;
        };
        if self.retained_tail != Some(index) {
            return false;
        }
        let Some(previous) = self.operations.get(index).and_then(Option::as_ref) else {
            return false;
        };
        if previous.opcode != "qcore.tuple"
            || previous.results.len() != 1
            || !replace_results(
                &operation.results,
                &previous.operands,
                self.values,
                &mut self.replacements,
            )
        {
            return false;
        }
        self.remove_retained(index);
        self.discard_current(operation);
        self.work.remove(2);
        true
    }

    fn simplify_tuple(&mut self, operation: &wire::OperationProto) -> bool {
        if operation.opcode != "qcore.tuple" || operation.results.len() != 1 {
            return false;
        }
        let Some(first) = operation.operands.first() else {
            return false;
        };
        let Some(index) = self.producers.get(first).copied() else {
            return false;
        };
        if self.retained_tail != Some(index) {
            return false;
        }
        let Some(previous) = self.operations.get(index).and_then(Option::as_ref) else {
            return false;
        };
        if previous.opcode != "qcore.unpack"
            || previous.operands.len() != 1
            || previous
                .results
                .iter()
                .map(|value| value.id)
                .ne(operation.operands.iter().copied())
            || !values_are_substitutable(&operation.results[0], previous.operands[0], self.values)
        {
            return false;
        }
        let original = previous.operands[0];
        self.remove_retained(index);
        self.replace(operation.results[0].id, original);
        self.discard_current(operation);
        self.work.remove(2);
        true
    }

    fn cancel_inverse(
        &mut self,
        operation: &wire::OperationProto,
        analysis: OperationAnalysis,
    ) -> bool {
        let Some(region) = owned_transform_region(operation, self.values, self.owned_values) else {
            return false;
        };
        self.work.visit();
        let Some(index) = self.frontiers.inverse_candidate(&region, analysis.support) else {
            return false;
        };
        let Some(previous) = self.operations.get(index).and_then(Option::as_ref) else {
            return false;
        };
        let previous_analysis = self.frontiers.analysis(index).unwrap_or_default();
        if !operations_are_inverse(
            previous,
            operation,
            previous_analysis,
            analysis,
            self.owned_values,
            &self.constants,
        ) {
            return false;
        }
        self.cancel_frontier_pair(index, operation)
    }

    fn cancel_frontier_pair(&mut self, index: usize, operation: &wire::OperationProto) -> bool {
        let Some(previous) = self.operations.get(index).and_then(Option::as_ref) else {
            return false;
        };
        let Some(predecessor) = previous.operands.first().copied() else {
            return false;
        };
        let Some(node) = self.frontiers.nodes.get(&index) else {
            return false;
        };
        if !self.frontiers.owners.contains_key(&node.region)
            || node.next.is_some_and(|next| {
                self.operations
                    .get(next)
                    .and_then(Option::as_ref)
                    .and_then(|operation| operation.operands.first())
                    .is_none()
            })
        {
            return false;
        }

        let Some(node) = self.frontiers.nodes.remove(&index) else {
            return false;
        };
        let Some(frontier) = self.frontiers.owners.get_mut(&node.region) else {
            self.frontiers.nodes.insert(index, node);
            return false;
        };
        if let Some(previous) = node.previous
            && let Some(previous_node) = self.frontiers.nodes.get_mut(&previous)
        {
            previous_node.next = node.next;
        }
        if let Some(next) = node.next {
            if let Some(next_node) = self.frontiers.nodes.get_mut(&next) {
                next_node.previous = node.previous;
            }
            if let Some(next_operation) = self.operations.get_mut(next).and_then(Option::as_mut) {
                next_operation.operands[0] = predecessor;
            }
        }

        if node.next.is_none() {
            frontier.tail = node.previous;
        }
        match node.update {
            FrontierUpdate::Static(update) => {
                for position in 0..usize::from(update.len) {
                    let selector = update.selectors[position];
                    if let Some(previous) = update.previous[position] {
                        frontier.last_touch.insert(selector, previous);
                    } else {
                        frontier.last_touch.remove(&selector);
                    }
                }
            }
            FrontierUpdate::Barrier(previous) => frontier.last_touch = previous,
        }

        let replacement = if node.next.is_some() {
            operation.operands[0]
        } else {
            predecessor
        };
        if let Some(removed) = self.operations.get_mut(index).and_then(Option::take) {
            self.unlink_retained(index);
            for result in removed.results {
                self.producers.remove(&result.id);
            }
        }
        self.replace(operation.results[0].id, replacement);
        self.work.remove(2);
        true
    }

    fn eliminate_vacuous_owner(&mut self, operation: &wire::OperationProto) -> bool {
        let (producer_opcode, extra_operands) = match operation.opcode.as_str() {
            "quantum.release" => ("quantum.alloc", &[][..]),
            "quantum.unprepare" => ("quantum.prepare", &operation.operands[1..]),
            _ => return false,
        };
        let Some(input) = operation.operands.first() else {
            return false;
        };
        let Some(index) = self.producers.get(input).copied() else {
            return false;
        };
        let Some(producer) = self.operations.get(index).and_then(Option::as_ref) else {
            return false;
        };
        if producer.opcode != producer_opcode
            || (producer_opcode == "quantum.prepare"
                && !equivalent_operands(&producer.operands, extra_operands, &self.constants))
        {
            return false;
        }
        self.remove_retained(index);
        self.discard_current(operation);
        self.work.remove(2);
        true
    }

    fn remove_retained(&mut self, index: usize) -> Option<wire::OperationProto> {
        let operation = self.operations.get_mut(index)?.take()?;
        self.unlink_retained(index);
        self.frontiers.invalidate_operation(&operation, self.values);
        for result in &operation.results {
            self.producers.remove(&result.id);
        }
        Some(operation)
    }

    fn discard_current(&mut self, operation: &wire::OperationProto) {
        self.frontiers.invalidate_operation(operation, self.values);
    }

    fn retain(
        &mut self,
        operation: wire::OperationProto,
        analysis: OperationAnalysis,
        track_frontier: bool,
    ) {
        let index = self.operations.len();
        let previous = self.retained_tail;
        if let Some(previous) = previous
            && let Some(links) = self
                .retained_links
                .get_mut(previous)
                .and_then(Option::as_mut)
        {
            links.next = Some(index);
        }
        self.retained_links.push(Some(RetainedLinks {
            previous,
            next: None,
        }));
        self.retained_tail = Some(index);
        if let Some((identifier, literal)) = constant_definition(&operation) {
            self.constants.insert(identifier, literal.clone());
        }
        for result in &operation.results {
            self.producers.insert(result.id, index);
        }
        if track_frontier {
            self.frontiers
                .record(index, &operation, self.values, self.owned_values, analysis);
        }
        self.operations.push(Some(operation));
    }

    fn unlink_retained(&mut self, index: usize) {
        let Some(links) = self.retained_links.get_mut(index).and_then(Option::take) else {
            return;
        };
        if let Some(previous) = links.previous
            && let Some(previous_links) = self
                .retained_links
                .get_mut(previous)
                .and_then(Option::as_mut)
        {
            previous_links.next = links.next;
        }
        if let Some(next) = links.next
            && let Some(next_links) = self.retained_links.get_mut(next).and_then(Option::as_mut)
        {
            next_links.previous = links.previous;
        } else if self.retained_tail == Some(index) {
            self.retained_tail = links.previous;
        }
    }

    fn finish(self, block: &mut wire::BlockProto, options: StandardOptimizationOptions) {
        let mut retained = self.operations.into_iter().flatten().collect::<Vec<_>>();
        for operation in &mut retained {
            rewrite_operands(&mut operation.operands, &self.replacements);
        }
        if let Some(terminator) = block.terminator.as_mut() {
            rewrite_operands(&mut terminator.operands, &self.replacements);
        }
        if options.dead_constants {
            eliminate_dead_total_operations(&mut retained, block.terminator.as_ref(), self.work);
        }
        block.operations = retained;
    }
}

#[derive(Clone, Copy)]
struct RetainedLinks {
    previous: Option<usize>,
    next: Option<usize>,
}

struct FunctionFacts {
    values: HashMap<u64, wire::ValueProto>,
    owned_values: HashSet<u64>,
}

fn function_facts(function: &wire::FunctionProto) -> FunctionFacts {
    let mut facts = FunctionFacts {
        values: HashMap::new(),
        owned_values: HashSet::new(),
    };
    for parameter in &function.parameters {
        if let Some(value) = &parameter.value {
            record_function_value(&mut facts, value);
        }
    }
    let mut blocks = function.body.iter().collect::<Vec<_>>();
    while let Some(block) = blocks.pop() {
        for value in &block.arguments {
            record_function_value(&mut facts, value);
        }
        for operation in &block.operations {
            for value in &operation.results {
                record_function_value(&mut facts, value);
            }
            for region in &operation.regions {
                blocks.extend(&region.blocks);
            }
        }
    }
    facts
}

fn record_function_value(facts: &mut FunctionFacts, value: &wire::ValueProto) {
    if value.has_region {
        facts.owned_values.insert(value.id);
    }
    facts.values.insert(value.id, value.clone());
}

#[cfg(test)]
fn function_owned_values(function: &wire::FunctionProto) -> HashSet<u64> {
    function_facts(function).owned_values
}

#[cfg(test)]
fn function_values(function: &wire::FunctionProto) -> HashMap<u64, wire::ValueProto> {
    function_facts(function).values
}

fn resolve(mut identifier: u64, replacements: &HashMap<u64, u64>) -> u64 {
    let mut remaining = replacements.len().saturating_add(1);
    while let Some(replacement) = replacements.get(&identifier).copied() {
        if replacement == identifier || remaining == 0 {
            break;
        }
        identifier = replacement;
        remaining -= 1;
    }
    identifier
}

fn rewrite_operands(operands: &mut [u64], replacements: &HashMap<u64, u64>) {
    for operand in operands {
        *operand = resolve(*operand, replacements);
    }
}

fn replace_results(
    results: &[wire::ValueProto],
    operands: &[u64],
    values: &HashMap<u64, wire::ValueProto>,
    replacements: &mut HashMap<u64, u64>,
) -> bool {
    if results.len() != operands.len()
        || results
            .iter()
            .zip(operands)
            .any(|(result, operand)| !values_are_substitutable(result, *operand, values))
    {
        return false;
    }
    for (result, operand) in results.iter().zip(operands) {
        replacements.insert(result.id, *operand);
    }
    true
}

fn values_are_substitutable(
    expected: &wire::ValueProto,
    replacement: u64,
    values: &HashMap<u64, wire::ValueProto>,
) -> bool {
    values.get(&replacement).is_some_and(|candidate| {
        expected.r#type == candidate.r#type
            && expected.has_region == candidate.has_region
            && expected.region == candidate.region
    })
}

fn literal_is_zero(literal: &wire::LiteralProto) -> bool {
    match static_values::literal(literal) {
        Some(StaticScalar::Int(0)) => true,
        Some(StaticScalar::RealBits(bits)) => bits == 0 || bits == 0x8000_0000_0000_0000,
        _ => false,
    }
}

#[derive(Default)]
struct DependencyFrontiers {
    owners: HashMap<wire::RegionRefProto, OwnerFrontier>,
    nodes: HashMap<usize, FrontierNode>,
}

#[derive(Default)]
struct OwnerFrontier {
    tail: Option<usize>,
    last_touch: HashMap<i64, usize>,
}

struct FrontierNode {
    region: wire::RegionRefProto,
    previous: Option<usize>,
    next: Option<usize>,
    update: FrontierUpdate,
    analysis: OperationAnalysis,
}

enum FrontierUpdate {
    Static(StaticFrontierUpdate),
    Barrier(HashMap<i64, usize>),
}

struct StaticFrontierUpdate {
    selectors: [i64; MAX_GATE_SELECTORS],
    previous: [Option<usize>; MAX_GATE_SELECTORS],
    len: u8,
}

impl DependencyFrontiers {
    fn inverse_candidate(
        &self,
        region: &wire::RegionRefProto,
        support: Option<GateSupport>,
    ) -> Option<usize> {
        let frontier = self.owners.get(region)?;
        let Some(support) = support else {
            return frontier.tail;
        };
        let mut selectors = support.iter();
        let candidate = frontier.last_touch.get(&selectors.next()?).copied()?;
        selectors
            .all(|selector| frontier.last_touch.get(&selector) == Some(&candidate))
            .then_some(candidate)
    }

    fn record(
        &mut self,
        index: usize,
        operation: &wire::OperationProto,
        values: &HashMap<u64, wire::ValueProto>,
        owned_values: &HashSet<u64>,
        analysis: OperationAnalysis,
    ) {
        let Some(region) = owned_transform_region(operation, values, owned_values) else {
            self.invalidate_operation(operation, values);
            return;
        };
        let frontier = self.owners.entry(region.clone()).or_default();
        let previous = frontier.tail;
        if let Some(previous) = previous
            && let Some(node) = self.nodes.get_mut(&previous)
        {
            node.next = Some(index);
        }
        let update = match analysis.support {
            Some(support) => {
                let mut update = StaticFrontierUpdate {
                    selectors: [0; MAX_GATE_SELECTORS],
                    previous: [None; MAX_GATE_SELECTORS],
                    len: support.len,
                };
                for (position, selector) in support.iter().enumerate() {
                    update.selectors[position] = selector;
                    update.previous[position] = frontier.last_touch.insert(selector, index);
                }
                FrontierUpdate::Static(update)
            }
            None => FrontierUpdate::Barrier(std::mem::take(&mut frontier.last_touch)),
        };
        frontier.tail = Some(index);
        self.nodes.insert(
            index,
            FrontierNode {
                region,
                previous,
                next: None,
                update,
                analysis,
            },
        );
    }

    fn analysis(&self, index: usize) -> Option<OperationAnalysis> {
        self.nodes.get(&index).map(|node| node.analysis)
    }

    fn invalidate_operation(
        &mut self,
        operation: &wire::OperationProto,
        values: &HashMap<u64, wire::ValueProto>,
    ) {
        for region in operation
            .operands
            .iter()
            .filter_map(|identifier| values.get(identifier))
            .chain(operation.results.iter())
            .filter_map(|value| value.region.as_ref())
        {
            self.invalidate_region(region);
        }
    }

    fn invalidate_region(&mut self, region: &wire::RegionRefProto) {
        self.owners.remove(region);
    }
}

fn owned_transform_region(
    operation: &wire::OperationProto,
    values: &HashMap<u64, wire::ValueProto>,
    owned_values: &HashSet<u64>,
) -> Option<wire::RegionRefProto> {
    if !is_single_owned_transform(operation, owned_values) {
        return None;
    }
    let input = operation
        .operands
        .first()
        .and_then(|identifier| values.get(identifier))?
        .region
        .as_ref()?;
    let output = operation.results[0].region.as_ref()?;
    (input == output).then(|| input.clone())
}

fn is_single_owned_transform(
    operation: &wire::OperationProto,
    owned_values: &HashSet<u64>,
) -> bool {
    operation.results.len() == 1
        && operation.results[0].has_region
        && operation
            .operands
            .first()
            .is_some_and(|identifier| owned_values.contains(identifier))
        && operation
            .operands
            .iter()
            .skip(1)
            .all(|identifier| !owned_values.contains(identifier))
}

fn operations_are_inverse(
    left: &wire::OperationProto,
    right: &wire::OperationProto,
    left_analysis: OperationAnalysis,
    right_analysis: OperationAnalysis,
    owned_values: &HashSet<u64>,
    constants: &HashMap<u64, wire::LiteralProto>,
) -> bool {
    if !is_single_owned_transform(left, owned_values) {
        return false;
    }
    let directly_adjacent =
        left.results.first().map(|result| result.id) == right.operands.first().copied();
    if !directly_adjacent
        && left_analysis
            .support
            .zip(right_analysis.support)
            .is_none_or(|(left, right)| left != right)
    {
        return false;
    }

    if let (Some(left_gate), Some(right_gate)) = (
        primitive_gate_spec(&left.opcode),
        primitive_gate_spec(&right.opcode),
    ) {
        if left_gate.has_angle() || right_gate.has_angle() {
            return false;
        }
        return GateOpcode::parse(&left.opcode)
            .zip(GateOpcode::parse(&right.opcode))
            .is_some_and(|(left, right)| left.adjoint() == right)
            && primitive_selectors_equivalent(
                left,
                right,
                left_analysis.selectors,
                right_analysis.selectors,
            );
    }

    if let (Some(left_opcode), Some(right_opcode)) = (
        QuantumOpcode::parse(&left.opcode),
        QuantumOpcode::parse(&right.opcode),
    ) && matches!(
        left_opcode,
        QuantumOpcode::AddInto
            | QuantumOpcode::SubtractInto
            | QuantumOpcode::AddConstant
            | QuantumOpcode::SubtractConstant
            | QuantumOpcode::ModAdd
            | QuantumOpcode::ModSub
            | QuantumOpcode::ModAddConstant
            | QuantumOpcode::ModSubConstant
            | QuantumOpcode::MulAdd
            | QuantumOpcode::MulSub
            | QuantumOpcode::ModMul
            | QuantumOpcode::ModDiv
            | QuantumOpcode::ModPow
            | QuantumOpcode::ModPowInverse
            | QuantumOpcode::LessThanXor
    ) && left_opcode.exact_adjoint() == Some(right_opcode)
    {
        return equivalent_operands(&left.operands[1..], &right.operands[1..], constants);
    }

    if let (Some(left_opcode), Some(right_opcode)) = (
        QCoreOpcode::parse(&left.opcode),
        QCoreOpcode::parse(&right.opcode),
    ) && left_opcode.exact_adjoint() == Some(right_opcode)
    {
        return string_attribute(left, "callee") == string_attribute(right, "callee")
            && equivalent_operands(&left.operands[1..], &right.operands[1..], constants);
    }
    false
}

fn primitive_selectors_equivalent(
    left: &wire::OperationProto,
    right: &wire::OperationProto,
    left_selectors: Option<CanonicalSelectors>,
    right_selectors: Option<CanonicalSelectors>,
) -> bool {
    match (left_selectors, right_selectors) {
        (Some(left), Some(right)) => left == right,
        _ => {
            left.operands.get(1..) == right.operands.get(1..)
                && attribute(left, "keywords") == attribute(right, "keywords")
        }
    }
}

fn canonical_selectors(
    operation: &wire::OperationProto,
    constants: &HashMap<u64, wire::LiteralProto>,
) -> Option<CanonicalSelectors> {
    let operands = selector_operands(operation)?;
    if operands.len() > MAX_GATE_SELECTORS {
        return None;
    }
    let mut selectors = CanonicalSelectors {
        values: [StaticScalar::Int(0); MAX_GATE_SELECTORS],
        len: u8::try_from(operands.len()).ok()?,
    };
    for (index, operand) in operands.into_iter().enumerate() {
        selectors.values[index] = match operand {
            SelectorOperand::Value(id) => constants.get(&id).and_then(static_values::literal)?,
            SelectorOperand::Default(value) => StaticScalar::Int(value),
        };
    }
    Some(selectors)
}

/// An identity still evaluates its selectors. Only erase that check when the
/// existing native selector validator proves it cannot fail.
fn primitive_selectors_are_valid(
    operation: &wire::OperationProto,
    constants: &HashMap<u64, wire::LiteralProto>,
) -> bool {
    let Some(spec) = primitive_gate_spec(&operation.opcode) else {
        return false;
    };
    if spec.broadcasts_when_unselected()
        && operation.operands.len() == 1 + usize::from(spec.has_angle())
    {
        return true;
    }
    let Some(selectors) = canonical_selectors(operation, constants) else {
        return false;
    };
    let Some(source) = operation.results.first().and_then(|v| v.r#type.as_ref()) else {
        return false;
    };
    let mut arena = TypeArena::default();
    let scope = arena.intern_scope("optimization-selector-proof");
    let Ok(source) = arena.resolve_wire_type(source, &scope) else {
        return false;
    };
    let Some(SemanticType::Quantum(payload)) = arena.get(arena.strip_stage(source)) else {
        return false;
    };
    let Some(width) = arena
        .coherent_payload_width(*payload)
        .ok()
        .flatten()
        .and_then(|v| usize::try_from(v).ok())
    else {
        return false;
    };
    let indices = selectors
        .iter()
        .map(|v| match v {
            StaticScalar::Int(v) => Some(v),
            _ => None,
        })
        .collect::<Option<Vec<_>>>();
    indices.is_some_and(|indices| {
        crate::validate_circuit_selectors(&operation.opcode, width, &indices).is_ok()
    })
}

/// Project the existing coherent type meaning without expanding any register.
fn arithmetic_target_width(operation: &wire::OperationProto, opcode: QuantumOpcode) -> Option<u64> {
    let source = operation.results.first()?.r#type.as_ref()?;
    let mut arena = TypeArena::default();
    let scope = arena.intern_scope("standard-arithmetic-identity");
    let source = arena.resolve_wire_type(source, &scope).ok()?;
    let SemanticType::Quantum(payload) = arena.get(arena.strip_stage(source))? else {
        return None;
    };
    let payload = if matches!(opcode, QuantumOpcode::ModPow | QuantumOpcode::ModPowInverse) {
        let SemanticType::Product(elements) = arena.get(arena.strip_stage(*payload))? else {
            return None;
        };
        let [target, exponent] = elements.as_ref() else {
            return None;
        };
        arena
            .coherent_payload_width(*exponent)
            .ok()??
            .checked_sub(1)?;
        *target
    } else {
        *payload
    };
    arena
        .coherent_payload_width(payload)
        .ok()?
        .filter(|width| *width > 0)
}

fn operation_analysis(
    operation: &wire::OperationProto,
    constants: &HashMap<u64, wire::LiteralProto>,
) -> OperationAnalysis {
    let selectors = canonical_selectors(operation, constants);
    OperationAnalysis {
        selectors,
        support: selectors.and_then(GateSupport::from_selectors),
    }
}

fn equivalent_operands(
    left: &[u64],
    right: &[u64],
    constants: &HashMap<u64, wire::LiteralProto>,
) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left == right
                || constants
                    .get(left)
                    .zip(constants.get(right))
                    .is_some_and(|(left, right)| left == right)
        })
}

fn eliminate_dead_total_operations(
    operations: &mut Vec<wire::OperationProto>,
    terminator: Option<&wire::TerminatorProto>,
    work: &mut StandardOptimizationWork,
) {
    let mut used = terminator
        .into_iter()
        .flat_map(|terminator| terminator.operands.iter().copied())
        .collect::<HashSet<_>>();
    let mut retained = Vec::with_capacity(operations.len());
    for operation in operations.drain(..).rev() {
        let removable = matches!(
            operation.opcode.as_str(),
            "qcore.constant" | "qcore.stage_lift"
        ) && operation
            .results
            .iter()
            .all(|result| !used.contains(&result.id));
        if removable {
            work.remove(1);
            continue;
        }
        for result in &operation.results {
            used.remove(&result.id);
        }
        used.extend(operation.operands.iter().copied());
        retained.push(operation);
    }
    retained.reverse();
    *operations = retained;
}

fn constant_definition(operation: &wire::OperationProto) -> Option<(u64, &wire::LiteralProto)> {
    if operation.opcode != "qcore.constant" || operation.results.len() != 1 {
        return None;
    }
    Some((operation.results[0].id, attribute(operation, "value")?))
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

fn string_attribute<'a>(operation: &'a wire::OperationProto, name: &str) -> Option<&'a str> {
    match attribute(operation, name)?.value.as_ref()? {
        wire::literal_proto::Value::StringValue(value) => Some(value),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(name: &str) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Scalar.into(),
            name: name.to_owned(),
            ..Default::default()
        }
    }

    fn quantum_with_width(width: u64) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Quantum.into(),
            arguments: vec![wire::TypeProto {
                kind: wire::type_proto::Kind::Array.into(),
                name: "Bits".to_owned(),
                arguments: vec![scalar("Bit")],
                dimensions: vec![wire::DimensionProto {
                    value: Some(wire::dimension_proto::Value::Fixed(width)),
                }],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn quantum() -> wire::TypeProto {
        quantum_with_width(2)
    }

    fn value(id: u64, type_spec: wire::TypeProto, has_region: bool) -> wire::ValueProto {
        value_in_region(id, type_spec, has_region.then_some(0))
    }

    fn value_in_region(
        id: u64,
        type_spec: wire::TypeProto,
        region_root: Option<u64>,
    ) -> wire::ValueProto {
        wire::ValueProto {
            id,
            r#type: Some(type_spec),
            has_region: region_root.is_some(),
            region: region_root.map(|root| wire::RegionRefProto {
                root,
                projections: Vec::new(),
            }),
            ..Default::default()
        }
    }

    fn literal(value: i64) -> wire::LiteralProto {
        wire::LiteralProto {
            value: Some(wire::literal_proto::Value::IntValue(value)),
        }
    }

    fn strings(values: &[&str]) -> wire::LiteralProto {
        wire::LiteralProto {
            value: Some(wire::literal_proto::Value::TupleValue(
                wire::literal_proto::Sequence {
                    values: values
                        .iter()
                        .map(|value| wire::LiteralProto {
                            value: Some(wire::literal_proto::Value::StringValue(
                                (*value).to_owned(),
                            )),
                        })
                        .collect(),
                },
            )),
        }
    }

    fn constant(id: u64, literal_value: i64) -> wire::OperationProto {
        wire::OperationProto {
            opcode: "qcore.constant".to_owned(),
            results: vec![value(id, scalar("Int"), false)],
            attributes: vec![wire::AttributeProto {
                name: "value".to_owned(),
                value: Some(literal(literal_value)),
            }],
            ..Default::default()
        }
    }

    fn gate(opcode: &str, input: u64, selector: u64, output: u64) -> wire::OperationProto {
        gate_with_width(opcode, input, selector, output, 2)
    }

    fn gate_with_width(
        opcode: &str,
        input: u64,
        selector: u64,
        output: u64,
        width: u64,
    ) -> wire::OperationProto {
        gate_with_region(opcode, input, selector, output, width, 0)
    }

    fn gate_with_region(
        opcode: &str,
        input: u64,
        selector: u64,
        output: u64,
        width: u64,
        region_root: u64,
    ) -> wire::OperationProto {
        wire::OperationProto {
            opcode: opcode.to_owned(),
            operands: vec![input, selector],
            results: vec![value_in_region(
                output,
                quantum_with_width(width),
                Some(region_root),
            )],
            attributes: vec![wire::AttributeProto {
                name: "keywords".to_owned(),
                value: Some(strings(&["target"])),
            }],
            ..Default::default()
        }
    }

    fn controlled_gate(
        opcode: &str,
        input: u64,
        control: u64,
        target: u64,
        output: u64,
        width: u64,
        region_root: u64,
    ) -> wire::OperationProto {
        wire::OperationProto {
            opcode: opcode.to_owned(),
            operands: vec![input, control, target],
            results: vec![value_in_region(
                output,
                quantum_with_width(width),
                Some(region_root),
            )],
            attributes: vec![wire::AttributeProto {
                name: "keywords".to_owned(),
                value: Some(strings(&["control", "target"])),
            }],
            ..Default::default()
        }
    }

    fn call(opcode: &str, input: u64, output: u64, region_root: u64) -> wire::OperationProto {
        wire::OperationProto {
            opcode: opcode.to_owned(),
            operands: vec![input],
            results: vec![value_in_region(output, quantum(), Some(region_root))],
            attributes: vec![wire::AttributeProto {
                name: "callee".to_owned(),
                value: Some(wire::LiteralProto {
                    value: Some(wire::literal_proto::Value::StringValue(
                        "unitary".to_owned(),
                    )),
                }),
            }],
            ..Default::default()
        }
    }

    fn arithmetic_block(opcode: &str, width: u64, parameters: &[i64]) -> wire::BlockProto {
        let mut type_spec = quantum_with_width(width);
        if matches!(opcode, "quantum.mod_pow" | "quantum.mod_pow_inverse") {
            type_spec.arguments = vec![wire::TypeProto {
                kind: wire::type_proto::Kind::Product.into(),
                arguments: vec![
                    type_spec.arguments[0].clone(),
                    quantum().arguments[0].clone(),
                ],
                ..Default::default()
            }];
        }
        let mut operations = parameters
            .iter()
            .enumerate()
            .map(|(index, parameter)| constant(u64::try_from(index + 1).unwrap(), *parameter))
            .collect::<Vec<_>>();
        operations.push(wire::OperationProto {
            opcode: opcode.to_owned(),
            operands: (0..=u64::try_from(parameters.len()).unwrap()).collect(),
            results: vec![value(10, type_spec.clone(), true)],
            ..Default::default()
        });
        wire::BlockProto {
            arguments: vec![value(0, type_spec, true)],
            operations,
            terminator: Some(wire::TerminatorProto {
                kind: "return".to_owned(),
                operands: vec![10],
                ..Default::default()
            }),
        }
    }

    fn optimize_test_block(block: &mut wire::BlockProto) -> StandardOptimizationWork {
        let function = wire::FunctionProto {
            parameters: block
                .arguments
                .iter()
                .cloned()
                .map(|value| wire::ParameterProto {
                    value: Some(value),
                    ..Default::default()
                })
                .collect(),
            body: Some(block.clone()),
            ..Default::default()
        };
        let mut work = StandardOptimizationWork::default();
        optimize_block(
            block,
            &function_values(&function),
            &function_owned_values(&function),
            0,
            &mut work,
        )
        .unwrap();
        work
    }

    #[test]
    fn removes_exact_arithmetic_identities_without_domain_expansion() {
        for (opcode, width, parameters) in [
            ("quantum.add_constant", 3, vec![-8]),
            ("quantum.subtract_constant", 63, vec![i64::MIN]),
            ("quantum.add_constant", 4096, vec![0]),
            ("quantum.mod_add_constant", 3, vec![-15, 5]),
            ("quantum.mod_sub_constant", 3, vec![i64::MIN, 8]),
            ("quantum.mod_mul", 3, vec![-4, 5]),
            ("quantum.mod_div", 3, vec![6, 5]),
            ("quantum.mod_mul", 3, vec![0, 1]),
            ("quantum.mod_pow", 3, vec![6, 5]),
            ("quantum.mod_pow_inverse", 3, vec![i64::MIN, 1]),
        ] {
            let mut block = arithmetic_block(opcode, width, &parameters);
            let original = block.clone();
            let work = optimize_test_block(&mut block);
            assert!(block.operations.is_empty(), "{opcode}: {block:?}");
            assert_eq!(block.terminator.as_ref().unwrap().operands, [0]);
            assert_eq!(work.removed_operations(), original.operations.len() as u128);
            let mut second = original;
            assert_eq!(optimize_test_block(&mut second), work);
            assert_eq!(second, block);
        }
    }

    #[test]
    fn retains_invalid_unresolved_and_nonidentity_arithmetic() {
        for (opcode, width, parameters) in [
            ("quantum.add_constant", 0, vec![0]),
            ("quantum.add_constant", 64, vec![i64::MIN]),
            ("quantum.mod_add_constant", 3, vec![0, 0]),
            ("quantum.mod_sub_constant", 3, vec![0, 9]),
            ("quantum.mod_mul", 3, vec![1, -3]),
            ("quantum.mod_div", 3, vec![0, 5]),
            ("quantum.mod_pow", 3, vec![2, 3]),
        ] {
            let mut block = arithmetic_block(opcode, width, &parameters);
            let original = block.clone();
            assert_eq!(optimize_test_block(&mut block).removed_operations(), 0);
            assert_eq!(block, original);
        }
        let mut block = arithmetic_block("quantum.mod_mul", 3, &[1, 5]);
        block.operations.remove(0);
        let original = block.clone();
        optimize_test_block(&mut block);
        assert_eq!(block, original);
    }

    #[test]
    fn arithmetic_identity_keeps_modular_cleanup_trace_and_malformed_layout() {
        let mut block = arithmetic_block("quantum.mod_mul", 3, &[1, 5]);
        block
            .operations
            .last_mut()
            .unwrap()
            .attributes
            .push(wire::AttributeProto {
                name: "proof".to_owned(),
                value: Some(wire::LiteralProto {
                    value: Some(wire::literal_proto::Value::StringValue(
                        "modular-adjoint".to_owned(),
                    )),
                }),
            });
        let original = block.clone();
        assert_eq!(optimize_test_block(&mut block).removed_operations(), 0);
        assert_eq!(block, original);

        let mut malformed = arithmetic_block("quantum.mod_mul", 3, &[1, 5]);
        malformed.operations.last_mut().unwrap().opcode = "quantum.mod_pow".to_owned();
        let original = malformed.clone();
        assert_eq!(optimize_test_block(&mut malformed).removed_operations(), 0);
        assert_eq!(malformed, original);
    }

    #[test]
    fn cancels_inverse_gates_across_disjoint_support() {
        let mut block = wire::BlockProto {
            arguments: vec![value(0, quantum(), true)],
            operations: vec![
                constant(1, 0),
                gate("quantum.x", 0, 1, 2),
                constant(3, 1),
                gate("quantum.h", 2, 3, 4),
                constant(5, 0),
                gate("quantum.x", 4, 5, 6),
            ],
            terminator: Some(wire::TerminatorProto {
                kind: "return".to_owned(),
                operands: vec![6],
                ..Default::default()
            }),
        };
        let function = wire::FunctionProto {
            parameters: vec![wire::ParameterProto {
                value: Some(value(0, quantum(), true)),
                ..Default::default()
            }],
            body: Some(block.clone()),
            ..Default::default()
        };
        let values = function_values(&function);
        let owned = function_owned_values(&function);
        let mut work = StandardOptimizationWork::default();
        optimize_block(&mut block, &values, &owned, 0, &mut work).expect("optimization succeeds");

        assert_eq!(
            block
                .operations
                .iter()
                .map(|operation| operation.opcode.as_str())
                .collect::<Vec<_>>(),
            ["qcore.constant", "quantum.h"]
        );
        assert_eq!(block.operations[1].operands[0], 0);
        assert_eq!(block.terminator.unwrap().operands, [4]);
        assert_eq!(work.removed_operations(), 4);
    }

    #[test]
    fn cancels_two_qubit_inverse_across_disjoint_support() {
        let width = 3;
        let mut block = wire::BlockProto {
            arguments: vec![value(0, quantum_with_width(width), true)],
            operations: vec![
                constant(1, 0),
                constant(2, 1),
                controlled_gate("quantum.cx", 0, 1, 2, 3, width, 0),
                constant(4, 2),
                gate_with_width("quantum.h", 3, 4, 5, width),
                constant(6, 0),
                constant(7, 1),
                controlled_gate("quantum.cx", 5, 6, 7, 8, width, 0),
            ],
            terminator: Some(wire::TerminatorProto {
                kind: "return".to_owned(),
                operands: vec![8],
                ..Default::default()
            }),
        };
        let function = wire::FunctionProto {
            parameters: vec![wire::ParameterProto {
                value: Some(value(0, quantum_with_width(width), true)),
                ..Default::default()
            }],
            body: Some(block.clone()),
            ..Default::default()
        };
        let values = function_values(&function);
        let owned = function_owned_values(&function);
        optimize_block(
            &mut block,
            &values,
            &owned,
            0,
            &mut StandardOptimizationWork::default(),
        )
        .expect("optimization succeeds");

        assert_eq!(
            block
                .operations
                .iter()
                .map(|operation| operation.opcode.as_str())
                .collect::<Vec<_>>(),
            ["qcore.constant", "quantum.h"]
        );
        assert_eq!(block.terminator.unwrap().operands, [5]);
    }

    #[test]
    fn retains_two_qubit_inverse_across_partial_overlap() {
        let width = 3;
        let mut block = wire::BlockProto {
            arguments: vec![value(0, quantum_with_width(width), true)],
            operations: vec![
                constant(1, 0),
                constant(2, 1),
                controlled_gate("quantum.cx", 0, 1, 2, 3, width, 0),
                constant(4, 1),
                gate_with_width("quantum.h", 3, 4, 5, width),
                constant(6, 0),
                constant(7, 1),
                controlled_gate("quantum.cx", 5, 6, 7, 8, width, 0),
            ],
            terminator: Some(wire::TerminatorProto {
                kind: "return".to_owned(),
                operands: vec![8],
                ..Default::default()
            }),
        };
        let function = wire::FunctionProto {
            parameters: vec![wire::ParameterProto {
                value: Some(value(0, quantum_with_width(width), true)),
                ..Default::default()
            }],
            body: Some(block.clone()),
            ..Default::default()
        };
        let values = function_values(&function);
        let owned = function_owned_values(&function);
        optimize_block(
            &mut block,
            &values,
            &owned,
            0,
            &mut StandardOptimizationWork::default(),
        )
        .expect("optimization succeeds");

        assert_eq!(
            block
                .operations
                .iter()
                .filter(|operation| operation.opcode.starts_with("quantum."))
                .count(),
            3
        );
    }

    #[test]
    fn restores_selector_frontier_after_barrier_inverse_cancellation() {
        let mut block = wire::BlockProto {
            arguments: vec![value(0, quantum(), true)],
            operations: vec![
                constant(1, 0),
                gate("quantum.x", 0, 1, 2),
                call("qcore.call", 2, 3, 0),
                call("qcore.adjoint_call", 3, 4, 0),
                constant(5, 0),
                gate("quantum.x", 4, 5, 6),
            ],
            terminator: Some(wire::TerminatorProto {
                kind: "return".to_owned(),
                operands: vec![6],
                ..Default::default()
            }),
        };
        let function = wire::FunctionProto {
            parameters: vec![wire::ParameterProto {
                value: Some(value(0, quantum(), true)),
                ..Default::default()
            }],
            body: Some(block.clone()),
            ..Default::default()
        };
        let values = function_values(&function);
        let owned = function_owned_values(&function);
        optimize_block_with_options(
            &mut block,
            &values,
            &owned,
            0,
            &mut StandardOptimizationWork::default(),
            StandardOptimizationOptions::default(),
        )
        .expect("optimization succeeds");

        assert!(block.operations.is_empty());
        assert_eq!(block.terminator.unwrap().operands, [0]);
    }

    #[test]
    fn keeps_equal_selectors_separate_across_ownership_roots() {
        let mut block = wire::BlockProto {
            arguments: vec![
                value_in_region(0, quantum(), Some(10)),
                value_in_region(1, quantum(), Some(20)),
            ],
            operations: vec![
                constant(2, 0),
                gate_with_region("quantum.x", 0, 2, 3, 2, 10),
                constant(4, 0),
                gate_with_region("quantum.x", 1, 4, 5, 2, 20),
                constant(6, 0),
                gate_with_region("quantum.x", 3, 6, 7, 2, 10),
            ],
            terminator: Some(wire::TerminatorProto {
                kind: "return".to_owned(),
                operands: vec![7, 5],
                ..Default::default()
            }),
        };
        let function = wire::FunctionProto {
            parameters: block
                .arguments
                .iter()
                .cloned()
                .map(|value| wire::ParameterProto {
                    value: Some(value),
                    ..Default::default()
                })
                .collect(),
            body: Some(block.clone()),
            ..Default::default()
        };
        let values = function_values(&function);
        let owned = function_owned_values(&function);
        optimize_block(
            &mut block,
            &values,
            &owned,
            0,
            &mut StandardOptimizationWork::default(),
        )
        .expect("optimization succeeds");

        assert_eq!(
            block
                .operations
                .iter()
                .filter(|operation| operation.opcode == "quantum.x")
                .count(),
            1
        );
        assert_eq!(block.terminator.unwrap().operands, [0, 5]);
    }

    #[test]
    fn cancels_inverse_gates_beyond_the_old_lookback_distance() {
        const SEPARATION: u64 = 300;
        let width = SEPARATION + 1;
        let mut operations = Vec::new();
        let mut state = 0;
        let mut next_identifier = 1;
        let mut append_gate = |opcode: &str, target: i64| {
            let selector = next_identifier;
            next_identifier += 1;
            operations.push(constant(selector, target));
            let output = next_identifier;
            next_identifier += 1;
            operations.push(gate_with_width(opcode, state, selector, output, width));
            state = output;
        };
        append_gate("quantum.x", 0);
        for target in 1..=SEPARATION {
            append_gate(
                "quantum.h",
                i64::try_from(target).expect("test selector fits i64"),
            );
        }
        append_gate("quantum.x", 0);

        let mut block = wire::BlockProto {
            arguments: vec![value(0, quantum_with_width(width), true)],
            operations,
            terminator: Some(wire::TerminatorProto {
                kind: "return".to_owned(),
                operands: vec![state],
                ..Default::default()
            }),
        };
        let function = wire::FunctionProto {
            parameters: vec![wire::ParameterProto {
                value: Some(value(0, quantum_with_width(width), true)),
                ..Default::default()
            }],
            body: Some(block.clone()),
            ..Default::default()
        };
        let source_operations = block.operations.len();
        let values = function_values(&function);
        let owned = function_owned_values(&function);
        let mut work = StandardOptimizationWork::default();
        optimize_block(&mut block, &values, &owned, 0, &mut work).expect("optimization succeeds");

        assert_eq!(
            block
                .operations
                .iter()
                .filter(|operation| operation.opcode == "quantum.x")
                .count(),
            0
        );
        assert_eq!(
            block
                .operations
                .iter()
                .filter(|operation| operation.opcode == "quantum.h")
                .count(),
            usize::try_from(SEPARATION).expect("test separation fits usize")
        );
        assert!(work.operation_visits() <= (source_operations as u128).saturating_mul(2));
    }

    #[test]
    fn retains_inverse_gates_with_overlapping_support() {
        let mut block = wire::BlockProto {
            arguments: vec![value(0, quantum(), true)],
            operations: vec![
                constant(1, 0),
                gate("quantum.x", 0, 1, 2),
                constant(3, 0),
                gate("quantum.h", 2, 3, 4),
                constant(5, 0),
                gate("quantum.x", 4, 5, 6),
            ],
            terminator: Some(wire::TerminatorProto {
                kind: "return".to_owned(),
                operands: vec![6],
                ..Default::default()
            }),
        };
        let function = wire::FunctionProto {
            parameters: vec![wire::ParameterProto {
                value: Some(value(0, quantum(), true)),
                ..Default::default()
            }],
            body: Some(block.clone()),
            ..Default::default()
        };
        let values = function_values(&function);
        let owned = function_owned_values(&function);
        optimize_block(
            &mut block,
            &values,
            &owned,
            0,
            &mut StandardOptimizationWork::default(),
        )
        .expect("optimization succeeds");

        assert_eq!(
            block
                .operations
                .iter()
                .filter(|operation| operation.opcode.starts_with("quantum."))
                .count(),
            3
        );
    }

    #[test]
    fn retains_dynamic_selectors_with_different_order() {
        let controlled = |input, control, target, output| wire::OperationProto {
            opcode: "quantum.cx".to_owned(),
            operands: vec![input, control, target],
            results: vec![value(output, quantum(), true)],
            attributes: vec![wire::AttributeProto {
                name: "keywords".to_owned(),
                value: Some(strings(&["control", "target"])),
            }],
            ..Default::default()
        };
        let mut block = wire::BlockProto {
            arguments: vec![
                value(0, quantum(), true),
                value(1, scalar("Int"), false),
                value(2, scalar("Int"), false),
            ],
            operations: vec![controlled(0, 1, 2, 3), controlled(3, 2, 1, 4)],
            terminator: Some(wire::TerminatorProto {
                kind: "return".to_owned(),
                operands: vec![4],
                ..Default::default()
            }),
        };
        let function = wire::FunctionProto {
            parameters: block
                .arguments
                .iter()
                .cloned()
                .map(|value| wire::ParameterProto {
                    value: Some(value),
                    ..Default::default()
                })
                .collect(),
            body: Some(block.clone()),
            ..Default::default()
        };
        let values = function_values(&function);
        let owned = function_owned_values(&function);
        optimize_block(
            &mut block,
            &values,
            &owned,
            0,
            &mut StandardOptimizationWork::default(),
        )
        .expect("optimization succeeds");

        assert_eq!(block.operations.len(), 2);
    }

    #[test]
    fn preserves_forward_trace_when_modular_cleanup_is_present() {
        let mut block = wire::BlockProto {
            arguments: vec![value(0, quantum(), true)],
            operations: vec![
                constant(1, 0),
                gate("quantum.x", 0, 1, 2),
                constant(3, 1),
                gate("quantum.h", 2, 3, 4),
                constant(5, 0),
                gate("quantum.x", 4, 5, 6),
                wire::OperationProto {
                    opcode: "qcore.adjoint".to_owned(),
                    operands: vec![6],
                    results: vec![value(7, quantum(), true)],
                    attributes: vec![wire::AttributeProto {
                        name: "proof".to_owned(),
                        value: Some(wire::LiteralProto {
                            value: Some(wire::literal_proto::Value::StringValue(
                                "modular-adjoint".to_owned(),
                            )),
                        }),
                    }],
                    ..Default::default()
                },
            ],
            terminator: Some(wire::TerminatorProto {
                kind: "return".to_owned(),
                operands: vec![7],
                ..Default::default()
            }),
        };
        let function = wire::FunctionProto {
            parameters: vec![wire::ParameterProto {
                value: Some(value(0, quantum(), true)),
                ..Default::default()
            }],
            body: Some(block.clone()),
            ..Default::default()
        };
        let values = function_values(&function);
        let owned = function_owned_values(&function);
        optimize_block(
            &mut block,
            &values,
            &owned,
            0,
            &mut StandardOptimizationWork::default(),
        )
        .expect("optimization succeeds");

        assert_eq!(
            block
                .operations
                .iter()
                .filter(|operation| operation.opcode == "quantum.x")
                .count(),
            2
        );
    }
}
