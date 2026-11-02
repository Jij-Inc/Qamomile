//! Independent cleanup proof-term reconstruction.
//!
//! Cleanup provenance strings are never evidence. This pass replays the owned
//! producer chain as a hash-consed proof DAG and accepts release/unprepare only
//! when exact inverse steps normalize back to the original allocation or
//! preparation root. Hash-consing keeps long generated inverse chains
//! iterative and bounded by the already validated operation budget.

use std::collections::{HashMap, HashSet};

use prost::Message;
use thiserror::Error;

use crate::call_graph::StaticCallGraph;
use crate::contracts::ProgramContracts;
use crate::effects::Capability;
use crate::equations::LocalEquationFacts;
use crate::operations::{LocallyCheckedOperations, is_primitive_adjointable};
use crate::types::{TypeArena, TypeId};
use crate::wire;

/// Failure to reconstruct one producer-independent cleanup proof.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub(crate) enum CleanupError {
    /// Frontend-only forget markers cannot cross the verified boundary.
    #[error("Q1103: unresolved Forget remains in {function:?}")]
    UnresolvedForget { function: String },
    /// A release does not carry compiler-generated proof provenance.
    #[error("Q1104: uncertified release in {function:?}")]
    UncertifiedRelease { function: String },
    /// An inverse edge differs from the exact retained producer edge.
    #[error("Q1113: inverse operation in {function:?} does not match its producer")]
    InvalidInverse { function: String },
    /// A terminal cleanup does not normalize to its claimed producer root.
    #[error("Q1112: release in {function:?} is not the inverse of its producer chain")]
    InvalidRelease { function: String },
    /// An adjoint cleanup references a callable without a body-derived proof.
    #[error("Q1111: adjoint cleanup callable {callee:?} is unresolved")]
    InvalidAdjointCallable { callee: String },
    /// Retained values or semantic facts are unexpectedly incomplete.
    #[error("Q4002: missing retained cleanup facts in {function:?}")]
    MissingFacts { function: String },
    /// A proof DAG identifier cannot be represented safely.
    #[error("Q4007: cleanup proof DAG exceeds its supported capacity")]
    ProofCapacityExceeded,
    /// The replayed terminal obligations differ from the local equation pass.
    #[error("Q1114: cleanup proof obligation accounting is inconsistent")]
    ObligationMismatch,
}

/// Facts established by independent proof replay.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct VerifiedCleanupFacts {
    discharged: usize,
}

impl VerifiedCleanupFacts {
    /// Return the number of terminal cleanup claims independently discharged.
    #[must_use]
    pub(crate) const fn discharged_count(self) -> usize {
        self.discharged
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct TermId(u32);

type Term = Option<TermId>;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct RegionKey {
    root: u64,
    projections: Box<[String]>,
}

impl From<&wire::RegionRefProto> for RegionKey {
    fn from(region: &wire::RegionRefProto) -> Self {
        Self {
            root: region.root,
            projections: region.projections.clone().into_boxed_slice(),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct AttributeKey {
    name: String,
    value: Box<[u8]>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct StepKey {
    opcode: String,
    operands: Box<[u64]>,
    attributes: Box<[AttributeKey]>,
    regions: Box<[Box<[u8]>]>,
    history_origin: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum RootKind {
    Alloc,
    Prepare,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum TermNode {
    Root {
        kind: RootKind,
        value: u64,
        operands: Box<[u64]>,
        region: Option<RegionKey>,
    },
    Apply {
        step: StepKey,
        source: Term,
    },
    Tensor(Box<[Term]>),
    Projection {
        source: Term,
        index: usize,
        arity: usize,
    },
}

#[derive(Default)]
struct TermArena {
    nodes: Vec<TermNode>,
    indices: HashMap<TermNode, TermId>,
}

impl TermArena {
    fn intern(&mut self, node: TermNode) -> Result<TermId, CleanupError> {
        if let Some(term) = self.indices.get(&node) {
            return Ok(*term);
        }
        let index =
            u32::try_from(self.nodes.len()).map_err(|_| CleanupError::ProofCapacityExceeded)?;
        let term = TermId(index);
        self.nodes.push(node.clone());
        self.indices.insert(node, term);
        Ok(term)
    }

    fn get(&self, term: TermId) -> Option<&TermNode> {
        self.nodes.get(term.0 as usize)
    }
}

struct ReplayContext<'a> {
    function: &'a str,
    values: &'a HashMap<u64, TypeId>,
    arena: &'a TypeArena,
    package: &'a wire::QCorePackageProto,
    call_graph: &'a StaticCallGraph,
    contracts: &'a ProgramContracts,
    equations: &'a LocalEquationFacts,
    history_call_origins: &'a HashSet<u64>,
    history_replayable: &'a HashSet<usize>,
    terms: TermArena,
    discharged: usize,
}

/// Replay every cleanup proof against exact persistent producer chains.
///
/// # Errors
///
/// Returns [`CleanupError`] if a forget remains, an inverse step differs from
/// its producer, a terminal cleanup has the wrong origin/data/owner, or the
/// replayed obligation total differs from the local equation pass.
pub(crate) fn verify(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
    equations: &LocalEquationFacts,
    call_graph: &StaticCallGraph,
    contracts: &ProgramContracts,
) -> Result<VerifiedCleanupFacts, CleanupError> {
    let mut discharged = 0_usize;
    let history_replayable = crate::cleanup_synthesis::history_replayable_functions(
        package, operations, arena, call_graph, contracts,
    );
    for function in &package.functions {
        let values =
            operations
                .value_types(&function.symbol)
                .ok_or_else(|| CleanupError::MissingFacts {
                    function: function.symbol.clone(),
                })?;
        let mut traces = function
            .parameters
            .iter()
            .filter_map(|parameter| parameter.value.as_ref())
            .filter(|value| {
                values
                    .get(&value.id)
                    .is_some_and(|id| arena.is_quantum(*id))
            })
            .map(|value| (value.id, None))
            .collect::<HashMap<_, _>>();
        let history_call_origins = function
            .body
            .as_ref()
            .map(history_call_origins)
            .unwrap_or_default();
        let mut context = ReplayContext {
            function: &function.symbol,
            values,
            arena,
            package,
            call_graph,
            contracts,
            equations,
            history_call_origins: &history_call_origins,
            history_replayable: &history_replayable,
            terms: TermArena::default(),
            discharged: 0,
        };
        if let Some(body) = &function.body {
            context.verify_block(body, &mut traces)?;
        }
        discharged = discharged
            .checked_add(context.discharged)
            .ok_or(CleanupError::ProofCapacityExceeded)?;
    }
    if discharged != equations.proof_obligation_count() {
        return Err(CleanupError::ObligationMismatch);
    }
    Ok(VerifiedCleanupFacts { discharged })
}

impl ReplayContext<'_> {
    fn is_linear(&self, value: u64) -> Result<bool, CleanupError> {
        self.values
            .get(&value)
            .copied()
            .map(|type_id| self.arena.is_quantum(type_id))
            .ok_or_else(|| CleanupError::MissingFacts {
                function: self.function.to_owned(),
            })
    }

    fn verify_block(
        &mut self,
        block: &wire::BlockProto,
        traces: &mut HashMap<u64, Term>,
    ) -> Result<(), CleanupError> {
        for argument in &block.arguments {
            if self.is_linear(argument.id)? {
                traces.insert(argument.id, None);
            }
        }
        let mut definitions = block
            .arguments
            .iter()
            .map(|value| (value.id, value))
            .collect::<HashMap<_, _>>();
        for operation in &block.operations {
            definitions.extend(operation.results.iter().map(|value| (value.id, value)));
        }
        for operation in &block.operations {
            for region in &operation.regions {
                for nested in &region.blocks {
                    let mut nested_traces = HashMap::new();
                    self.verify_block(nested, &mut nested_traces)?;
                }
            }
            self.verify_operation(operation, traces, &definitions)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn verify_operation(
        &mut self,
        operation: &wire::OperationProto,
        traces: &mut HashMap<u64, Term>,
        definitions: &HashMap<u64, &wire::ValueProto>,
    ) -> Result<(), CleanupError> {
        match operation.opcode.as_str() {
            "qcore.forget" => {
                return Err(CleanupError::UnresolvedForget {
                    function: self.function.to_owned(),
                });
            }
            "quantum.alloc" | "quantum.prepare" if operation.results.len() == 1 => {
                let result = &operation.results[0];
                let node = TermNode::Root {
                    kind: if operation.opcode == "quantum.alloc" {
                        RootKind::Alloc
                    } else {
                        RootKind::Prepare
                    },
                    value: result.id,
                    operands: operation.operands.clone().into_boxed_slice(),
                    region: result.region.as_ref().map(RegionKey::from),
                };
                let term = self.terms.intern(node)?;
                traces.insert(result.id, Some(term));
            }
            "qcore.tuple" | "qcore.unpack" | "qcore.reshape" => {
                self.verify_structural(operation, traces)?;
            }
            opcode if is_primitive_adjointable(opcode) && operation.results.len() == 1 => {
                let source = operation
                    .operands
                    .first()
                    .and_then(|value| trace(traces, *value));
                let step = operation_step(operation, operation.operands.get(1..).unwrap_or(&[]));
                let term = self.terms.intern(TermNode::Apply { step, source })?;
                traces.insert(operation.results[0].id, Some(term));
            }
            "qcore.call" if operation.results.len() == 1 => {
                self.verify_call(operation, traces)?;
            }
            "qcore.control_call" | "qcore.select_call" if operation.results.len() == 1 => {
                self.verify_coherent_call(operation, traces)?;
            }
            "qcore.adjoint" if operation.results.len() == 1 => {
                let expected = adjoint_original_step(operation);
                let source = operation
                    .operands
                    .first()
                    .and_then(|value| trace(traces, *value));
                let term = self.cancel(source, expected.as_ref(), operation)?;
                traces.insert(operation.results[0].id, term);
            }
            "qcore.adjoint_call" if operation.results.len() == 1 => {
                self.verify_adjoint_call(operation, traces)?;
            }
            "qcore.if" | "qcore.for" => {
                self.verify_control_step(operation, traces)?;
            }
            "quantum.release" | "quantum.unprepare" => {
                self.verify_release(operation, traces, definitions)?;
            }
            opcode if opcode.starts_with("quantum.") || !operation.regions.is_empty() => {
                for result in &operation.results {
                    if self.is_linear(result.id)? {
                        traces.insert(result.id, None);
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn verify_structural(
        &mut self,
        operation: &wire::OperationProto,
        traces: &mut HashMap<u64, Term>,
    ) -> Result<(), CleanupError> {
        let operand_facts = operation
            .operands
            .iter()
            .copied()
            .map(|value| self.is_linear(value).map(|linear| (value, linear)))
            .collect::<Result<Vec<_>, _>>()?;
        let linear_operands = operand_facts
            .iter()
            .filter_map(|(value, linear)| linear.then_some(*value))
            .collect::<Vec<_>>();
        let result_facts = operation
            .results
            .iter()
            .map(|result| self.is_linear(result.id).map(|linear| (result.id, linear)))
            .collect::<Result<Vec<_>, _>>()?;
        let linear_results = result_facts
            .iter()
            .filter_map(|(value, linear)| linear.then_some(*value))
            .collect::<Vec<_>>();
        match operation.opcode.as_str() {
            "qcore.tuple" => {
                let elements = linear_operands
                    .iter()
                    .map(|value| trace(traces, *value))
                    .collect::<Vec<_>>();
                let term = if elements.len() == 1 {
                    elements[0]
                } else if self.complete_projection_row(&elements) {
                    let Some(first) = elements[0] else {
                        return Err(CleanupError::MissingFacts {
                            function: self.function.to_owned(),
                        });
                    };
                    match self.terms.get(first) {
                        Some(TermNode::Projection { source, .. }) => *source,
                        _ => None,
                    }
                } else {
                    Some(
                        self.terms
                            .intern(TermNode::Tensor(elements.into_boxed_slice()))?,
                    )
                };
                for result in linear_results {
                    traces.insert(result, term);
                }
            }
            "qcore.unpack" => {
                let source = linear_operands
                    .first()
                    .and_then(|value| trace(traces, *value));
                if self.equations.requires_proof(
                    self.function,
                    "qcore.unpack",
                    operation.operands.first().copied(),
                ) {
                    self.verify_unpack_rebase(operation, source)?;
                }
                let elements = if linear_results.len() == 1 {
                    vec![source]
                } else if let Some(term) = source {
                    match self.terms.get(term) {
                        Some(TermNode::Tensor(elements))
                            if elements.len() == linear_results.len() =>
                        {
                            elements.to_vec()
                        }
                        _ => self.projections(source, linear_results.len())?,
                    }
                } else {
                    self.projections(source, linear_results.len())?
                };
                traces.extend(linear_results.into_iter().zip(elements));
            }
            "qcore.reshape" => {
                let term = linear_operands
                    .first()
                    .and_then(|value| trace(traces, *value));
                for result in linear_results {
                    traces.insert(result, term);
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn verify_unpack_rebase(
        &mut self,
        operation: &wire::OperationProto,
        source: Term,
    ) -> Result<(), CleanupError> {
        if string_attribute(operation, "proof") != Some("modular-adjoint") {
            return Err(CleanupError::InvalidInverse {
                function: self.function.to_owned(),
            });
        }
        let elements = source
            .and_then(|term| match self.terms.get(term) {
                Some(TermNode::Tensor(elements)) => Some(elements.to_vec()),
                _ => None,
            })
            .filter(|elements| elements.len() == operation.results.len())
            .ok_or_else(|| CleanupError::InvalidInverse {
                function: self.function.to_owned(),
            })?;
        let regions_match = elements
            .iter()
            .zip(&operation.results)
            .all(|(term, result)| {
                self.preserved_region(*term).as_ref()
                    == result.region.as_ref().map(RegionKey::from).as_ref()
            });
        if !regions_match {
            return Err(CleanupError::InvalidInverse {
                function: self.function.to_owned(),
            });
        }
        self.discharged = self
            .discharged
            .checked_add(1)
            .ok_or(CleanupError::ProofCapacityExceeded)?;
        Ok(())
    }

    fn preserved_region(&self, mut term: Term) -> Option<RegionKey> {
        while let Some(current) = term {
            match self.terms.get(current)? {
                TermNode::Root { region, .. } => return region.clone(),
                TermNode::Apply { source, .. } | TermNode::Projection { source, .. } => {
                    term = *source;
                }
                TermNode::Tensor(_) => return None,
            }
        }
        None
    }

    fn complete_projection_row(&self, elements: &[Term]) -> bool {
        if elements.is_empty() || elements.iter().any(Option::is_none) {
            return false;
        }
        let projections = elements
            .iter()
            .filter_map(|term| term.and_then(|term| self.terms.get(term)))
            .collect::<Vec<_>>();
        let Some(TermNode::Projection {
            source,
            arity,
            index: _,
        }) = projections.first()
        else {
            return false;
        };
        *arity == elements.len()
            && projections.iter().enumerate().all(|(expected, node)| {
                matches!(
                    node,
                    TermNode::Projection { source: candidate, index, arity: candidate_arity }
                        if candidate == source && *index == expected && candidate_arity == arity
                )
            })
    }

    fn projections(&mut self, source: Term, arity: usize) -> Result<Vec<Term>, CleanupError> {
        (0..arity)
            .map(|index| {
                self.terms
                    .intern(TermNode::Projection {
                        source,
                        index,
                        arity,
                    })
                    .map(Some)
            })
            .collect()
    }

    fn verify_call(
        &mut self,
        operation: &wire::OperationProto,
        traces: &mut HashMap<u64, Term>,
    ) -> Result<(), CleanupError> {
        let linear = operation
            .operands
            .iter()
            .copied()
            .map(|value| self.is_linear(value).map(|is_linear| (value, is_linear)))
            .collect::<Result<Vec<_>, _>>()?;
        let linear_operands = linear
            .iter()
            .filter_map(|(value, is_linear)| is_linear.then_some(*value))
            .collect::<Vec<_>>();
        if linear_operands.is_empty() || !self.is_linear(operation.results[0].id)? {
            return Ok(());
        }
        let sources = linear_operands
            .iter()
            .map(|value| trace(traces, *value))
            .collect::<Vec<_>>();
        let source = if sources.len() == 1 {
            sources[0]
        } else {
            Some(
                self.terms
                    .intern(TermNode::Tensor(sources.into_boxed_slice()))?,
            )
        };
        let callee = string_attribute(operation, "callee").unwrap_or_default();
        let classical = linear
            .iter()
            .filter_map(|(value, is_linear)| (!is_linear).then_some(*value))
            .collect::<Vec<_>>();
        let mut forward = callable_step("call", callee, &classical);
        if self.history_call_origins.contains(&operation.results[0].id) {
            forward.history_origin = Some(operation.results[0].id);
        }
        let inverse = callable_step("adjoint_call", callee, &classical);
        let term = match source.and_then(|term| self.terms.get(term)) {
            Some(TermNode::Apply { step, source }) if step == &inverse => *source,
            _ => Some(self.terms.intern(TermNode::Apply {
                step: forward,
                source,
            })?),
        };
        traces.insert(operation.results[0].id, term);
        Ok(())
    }

    fn verify_coherent_call(
        &mut self,
        operation: &wire::OperationProto,
        traces: &mut HashMap<u64, Term>,
    ) -> Result<(), CleanupError> {
        let linear = operation
            .operands
            .iter()
            .copied()
            .map(|value| self.is_linear(value).map(|is_linear| (value, is_linear)))
            .collect::<Result<Vec<_>, _>>()?;
        let linear_operands = linear
            .iter()
            .filter_map(|(value, is_linear)| is_linear.then_some(*value))
            .collect::<Vec<_>>();
        if linear_operands.len() != 2 || !self.is_linear(operation.results[0].id)? {
            return Ok(());
        }
        let source = Some(
            self.terms.intern(TermNode::Tensor(
                linear_operands
                    .iter()
                    .map(|value| trace(traces, *value))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ))?,
        );
        let classical = linear
            .iter()
            .filter_map(|(value, is_linear)| (!is_linear).then_some(*value))
            .collect::<Vec<_>>();
        let step = operation_step(operation, &classical);
        let term = self.terms.intern(TermNode::Apply { step, source })?;
        traces.insert(operation.results[0].id, Some(term));
        Ok(())
    }

    fn verify_adjoint_call(
        &mut self,
        operation: &wire::OperationProto,
        traces: &mut HashMap<u64, Term>,
    ) -> Result<(), CleanupError> {
        let callee = string_attribute(operation, "callee").unwrap_or_default();
        let history_origin = integer_attribute(operation, "history_origin");
        let history_adjoint = history_origin.is_some();
        if history_adjoint
            && !self
                .call_graph
                .index(callee)
                .is_some_and(|index| self.history_replayable.contains(&index))
        {
            return Err(CleanupError::InvalidAdjointCallable {
                callee: callee.to_owned(),
            });
        }
        if !history_adjoint && !self.adjointable(callee) {
            return Err(CleanupError::InvalidAdjointCallable {
                callee: callee.to_owned(),
            });
        }
        let classical = operation.operands.get(1..).unwrap_or(&[]);
        let mut expected = callable_step("call", callee, classical);
        expected.history_origin = history_origin;
        let source = operation
            .operands
            .first()
            .and_then(|value| trace(traces, *value));
        let term = if string_attribute(operation, "proof") == Some("explicit-adjoint") {
            match source.and_then(|term| self.terms.get(term)) {
                Some(TermNode::Apply { step, source }) if step == &expected => *source,
                _ => Some(self.terms.intern(TermNode::Apply {
                    step: callable_step("adjoint_call", callee, classical),
                    source,
                })?),
            }
        } else {
            self.cancel(source, Some(&expected), operation)?
        };
        traces.insert(operation.results[0].id, term);
        Ok(())
    }

    fn verify_control_step(
        &mut self,
        operation: &wire::OperationProto,
        traces: &mut HashMap<u64, Term>,
    ) -> Result<(), CleanupError> {
        let operand_facts = operation
            .operands
            .iter()
            .copied()
            .map(|value| self.is_linear(value).map(|linear| (value, linear)))
            .collect::<Result<Vec<_>, _>>()?;
        let linear_operands = operand_facts
            .iter()
            .filter_map(|(value, linear)| linear.then_some(*value))
            .collect::<Vec<_>>();
        let result_facts = operation
            .results
            .iter()
            .map(|result| self.is_linear(result.id).map(|linear| (result, linear)))
            .collect::<Result<Vec<_>, _>>()?;
        let linear_results = result_facts
            .iter()
            .filter_map(|(result, linear)| linear.then_some(*result))
            .collect::<Vec<_>>();
        if linear_operands.len() != 1 || linear_results.len() != 1 {
            for result in linear_results {
                traces.insert(result.id, None);
            }
            return Ok(());
        }
        let source = trace(traces, linear_operands[0]);
        let classical = operation
            .operands
            .iter()
            .copied()
            .zip(operand_facts.iter().map(|(_, linear)| linear))
            .filter_map(|(value, linear)| (!linear).then_some(value))
            .collect::<Vec<_>>();
        if string_attribute(operation, "body_mode") == Some("adjoint") {
            let expected = match operation.opcode.as_str() {
                "qcore.if" => if_original_step(operation, &classical, self)?,
                "qcore.for" => for_original_step(operation, self)?,
                _ => unreachable!("dispatcher selects only structured control"),
            };
            let term = self.cancel(source, Some(&expected), operation)?;
            traces.insert(linear_results[0].id, term);
            return Ok(());
        }
        let step = operation_step(operation, &classical);
        let term = self.terms.intern(TermNode::Apply { step, source })?;
        traces.insert(linear_results[0].id, Some(term));
        Ok(())
    }

    fn verify_release(
        &mut self,
        operation: &wire::OperationProto,
        traces: &HashMap<u64, Term>,
        definitions: &HashMap<u64, &wire::ValueProto>,
    ) -> Result<(), CleanupError> {
        if string_attribute(operation, "proof") != Some("modular-adjoint") {
            return Err(CleanupError::UncertifiedRelease {
                function: self.function.to_owned(),
            });
        }
        let Some(state) = operation.operands.first().copied() else {
            return Err(CleanupError::InvalidRelease {
                function: self.function.to_owned(),
            });
        };
        let Some(term) = trace(traces, state) else {
            return Err(CleanupError::InvalidRelease {
                function: self.function.to_owned(),
            });
        };
        let expected_kind = if operation.opcode == "quantum.release" {
            RootKind::Alloc
        } else {
            RootKind::Prepare
        };
        let Some(TermNode::Root {
            kind,
            value,
            operands,
            region,
        }) = self.terms.get(term)
        else {
            return Err(CleanupError::InvalidRelease {
                function: self.function.to_owned(),
            });
        };
        let actual_region = definitions
            .get(&state)
            .and_then(|value| value.region.as_ref())
            .map(RegionKey::from);
        let valid = *kind == expected_kind
            && (expected_kind == RootKind::Alloc
                || operands.as_ref() == operation.operands.get(1..).unwrap_or(&[]))
            && integer_attribute(operation, "origin") == Some(*value)
            && actual_region.as_ref() == region.as_ref();
        if !valid {
            return Err(CleanupError::InvalidRelease {
                function: self.function.to_owned(),
            });
        }
        self.discharged = self
            .discharged
            .checked_add(1)
            .ok_or(CleanupError::ProofCapacityExceeded)?;
        Ok(())
    }

    fn cancel(
        &self,
        term: Term,
        expected: Option<&StepKey>,
        operation: &wire::OperationProto,
    ) -> Result<Term, CleanupError> {
        let valid_proof = matches!(
            string_attribute(operation, "proof"),
            Some("modular-adjoint" | "explicit-adjoint")
        );
        let matched = term.and_then(|term| match self.terms.get(term) {
            Some(TermNode::Apply { step, source }) if Some(step) == expected => Some(*source),
            _ => None,
        });
        if !valid_proof || matched.is_none() {
            return Err(CleanupError::InvalidInverse {
                function: self.function.to_owned(),
            });
        }
        Ok(matched.flatten())
    }

    fn adjointable(&self, callee: &str) -> bool {
        let Some(index) = self.call_graph.index(callee) else {
            return false;
        };
        self.package.functions.get(index).is_some()
            && self
                .contracts
                .contract(index)
                .is_some_and(|contract| contract.capabilities().contains(Capability::Adjointable))
    }
}

fn history_call_origins(block: &wire::BlockProto) -> HashSet<u64> {
    let mut origins = HashSet::new();
    let mut pending = vec![block];
    while let Some(block) = pending.pop() {
        for operation in &block.operations {
            pending.extend(
                operation
                    .regions
                    .iter()
                    .flat_map(|region| region.blocks.iter()),
            );
            if operation.opcode == "qcore.adjoint_call"
                && string_attribute(operation, "proof") == Some("modular-adjoint")
                && let Some(origin) = integer_attribute(operation, "history_origin")
            {
                origins.insert(origin);
            }
        }
    }
    origins
}

fn trace(traces: &HashMap<u64, Term>, value: u64) -> Term {
    traces.get(&value).copied().flatten()
}

fn operation_step(operation: &wire::OperationProto, operands: &[u64]) -> StepKey {
    StepKey {
        opcode: operation.opcode.clone(),
        operands: operands.to_vec().into_boxed_slice(),
        attributes: attribute_keys(&operation.attributes),
        regions: region_keys(&operation.regions),
        history_origin: None,
    }
}

fn callable_step(flavor: &str, callee: &str, operands: &[u64]) -> StepKey {
    StepKey {
        opcode: format!("{flavor}:{callee}"),
        operands: operands.to_vec().into_boxed_slice(),
        attributes: Box::default(),
        regions: Box::default(),
        history_origin: None,
    }
}

fn adjoint_original_step(operation: &wire::OperationProto) -> Option<StepKey> {
    let opcode = string_attribute(operation, "original")?;
    let attributes = operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == "original_attributes")?
        .value
        .as_ref()?;
    let wire::literal_proto::Value::MappingValue(mapping) = attributes.value.as_ref()? else {
        return None;
    };
    let attributes = mapping
        .entries
        .iter()
        .map(|entry| {
            Some(AttributeKey {
                name: entry.key.clone(),
                value: entry.value.as_ref()?.encode_to_vec().into_boxed_slice(),
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(StepKey {
        opcode: opcode.to_owned(),
        operands: operation
            .operands
            .get(1..)
            .unwrap_or(&[])
            .to_vec()
            .into_boxed_slice(),
        attributes: attributes.into_boxed_slice(),
        regions: Box::default(),
        history_origin: None,
    })
}

fn inverse_original_attributes(
    operation: &wire::OperationProto,
    context: &ReplayContext<'_>,
) -> Result<Vec<AttributeKey>, CleanupError> {
    let invalid = || CleanupError::InvalidInverse {
        function: context.function.to_owned(),
    };
    if string_attribute(operation, "proof") != Some("modular-adjoint") {
        return Err(invalid());
    }
    let original = operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == "original_attributes")
        .and_then(|attribute| attribute.value.as_ref())
        .and_then(|literal| match literal.value.as_ref() {
            Some(wire::literal_proto::Value::MappingValue(mapping)) => Some(mapping),
            _ => None,
        })
        .ok_or_else(&invalid)?;
    original
        .entries
        .iter()
        .map(|entry| {
            Some(AttributeKey {
                name: entry.key.clone(),
                value: entry.value.as_ref()?.encode_to_vec().into_boxed_slice(),
            })
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(invalid)
}

fn if_original_step(
    operation: &wire::OperationProto,
    operands: &[u64],
    context: &ReplayContext<'_>,
) -> Result<StepKey, CleanupError> {
    let original_attributes = inverse_original_attributes(operation, context)?;
    if !valid_if_inverse_attributes(operation, &original_attributes) {
        return Err(CleanupError::InvalidInverse {
            function: context.function.to_owned(),
        });
    }
    Ok(StepKey {
        opcode: "qcore.if".to_owned(),
        operands: operands.to_vec().into_boxed_slice(),
        attributes: original_attributes.into_boxed_slice(),
        regions: region_keys(&operation.regions),
        history_origin: None,
    })
}

fn valid_if_inverse_attributes(
    operation: &wire::OperationProto,
    original: &[AttributeKey],
) -> bool {
    let actual = operation
        .attributes
        .iter()
        .filter(|attribute| {
            !matches!(
                attribute.name.as_str(),
                "body_mode" | "original_attributes" | "proof"
            )
        })
        .map(|attribute| AttributeKey {
            name: attribute.name.clone(),
            value: attribute
                .value
                .as_ref()
                .map_or_else(Vec::new, Message::encode_to_vec)
                .into_boxed_slice(),
        })
        .collect::<Vec<_>>();
    actual == original
}

fn for_original_step(
    operation: &wire::OperationProto,
    context: &ReplayContext<'_>,
) -> Result<StepKey, CleanupError> {
    let original_attributes = inverse_original_attributes(operation, context)?;
    if !valid_for_inverse_attributes(operation, &original_attributes) {
        return Err(CleanupError::InvalidInverse {
            function: context.function.to_owned(),
        });
    }
    let operands = operation
        .operands
        .iter()
        .copied()
        .filter(|value| !context.is_linear(*value).unwrap_or(true))
        .collect::<Vec<_>>();
    Ok(StepKey {
        opcode: "qcore.for".to_owned(),
        operands: operands.into_boxed_slice(),
        attributes: original_attributes.into_boxed_slice(),
        regions: region_keys(&operation.regions),
        history_origin: None,
    })
}

fn valid_for_inverse_attributes(
    operation: &wire::OperationProto,
    original: &[AttributeKey],
) -> bool {
    let original_reverse = match original
        .iter()
        .find(|attribute| attribute.name == "reverse")
    {
        None => false,
        Some(attribute) => {
            let Ok(literal) = wire::LiteralProto::decode(attribute.value.as_ref()) else {
                return false;
            };
            let Some(wire::literal_proto::Value::BoolValue(value)) = literal.value else {
                return false;
            };
            value
        }
    };
    let actual_reverse = operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == "reverse")
        .and_then(|attribute| attribute.value.as_ref())
        .and_then(|literal| match literal.value {
            Some(wire::literal_proto::Value::BoolValue(value)) => Some(value),
            _ => None,
        });
    if actual_reverse != Some(!original_reverse) {
        return false;
    }
    let actual = operation
        .attributes
        .iter()
        .filter(|attribute| {
            !matches!(
                attribute.name.as_str(),
                "body_mode" | "original_attributes" | "proof" | "reverse"
            )
        })
        .map(|attribute| AttributeKey {
            name: attribute.name.clone(),
            value: attribute
                .value
                .as_ref()
                .map_or_else(Vec::new, Message::encode_to_vec)
                .into_boxed_slice(),
        })
        .collect::<Vec<_>>();
    let expected = original
        .iter()
        .filter(|attribute| attribute.name != "reverse")
        .cloned()
        .collect::<Vec<_>>();
    actual == expected
}

fn region_keys(regions: &[wire::RegionProto]) -> Box<[Box<[u8]>]> {
    let mut canonical = regions.to_vec();
    let mut identifiers = HashMap::new();
    let mut roots = HashMap::new();
    let mut next_identifier = 0_u64;
    let mut next_root = 0_u64;
    for region in &mut canonical {
        canonicalize_region_definitions(region, &mut identifiers, &mut next_identifier);
    }
    for region in &mut canonical {
        canonicalize_region_uses(region, &identifiers, &mut roots, &mut next_root);
    }
    canonical
        .into_iter()
        .map(|region| region.encode_to_vec().into_boxed_slice())
        .collect()
}

fn canonicalize_region_definitions(
    region: &mut wire::RegionProto,
    identifiers: &mut HashMap<u64, u64>,
    next_identifier: &mut u64,
) {
    for block in &mut region.blocks {
        for argument in &mut block.arguments {
            canonicalize_value_definition(argument, identifiers, next_identifier);
        }
        for operation in &mut block.operations {
            for result in &mut operation.results {
                canonicalize_value_definition(result, identifiers, next_identifier);
            }
            for nested in &mut operation.regions {
                canonicalize_region_definitions(nested, identifiers, next_identifier);
            }
        }
    }
}

fn canonicalize_value_definition(
    value: &mut wire::ValueProto,
    identifiers: &mut HashMap<u64, u64>,
    next_identifier: &mut u64,
) {
    let canonical = *next_identifier;
    *next_identifier = next_identifier.saturating_add(1);
    identifiers.insert(value.id, canonical);
    value.id = canonical;
}

fn canonicalize_region_uses(
    region: &mut wire::RegionProto,
    identifiers: &HashMap<u64, u64>,
    roots: &mut HashMap<u64, u64>,
    next_root: &mut u64,
) {
    for block in &mut region.blocks {
        for argument in &mut block.arguments {
            canonicalize_value_region(argument, roots, next_root);
        }
        for operation in &mut block.operations {
            for operand in &mut operation.operands {
                if let Some(canonical) = identifiers.get(operand) {
                    *operand = *canonical;
                }
            }
            for result in &mut operation.results {
                canonicalize_value_region(result, roots, next_root);
            }
            for nested in &mut operation.regions {
                canonicalize_region_uses(nested, identifiers, roots, next_root);
            }
        }
        if let Some(terminator) = &mut block.terminator {
            for operand in &mut terminator.operands {
                if let Some(canonical) = identifiers.get(operand) {
                    *operand = *canonical;
                }
            }
        }
    }
}

fn canonicalize_value_region(
    value: &mut wire::ValueProto,
    roots: &mut HashMap<u64, u64>,
    next_root: &mut u64,
) {
    let Some(region) = &mut value.region else {
        return;
    };
    let canonical = *roots.entry(region.root).or_insert_with(|| {
        let root = *next_root;
        *next_root = next_root.saturating_add(1);
        root
    });
    region.root = canonical;
}

fn attribute_keys(attributes: &[wire::AttributeProto]) -> Box<[AttributeKey]> {
    attributes
        .iter()
        .map(|attribute| AttributeKey {
            name: attribute.name.clone(),
            value: attribute
                .value
                .as_ref()
                .map_or_else(Vec::new, Message::encode_to_vec)
                .into_boxed_slice(),
        })
        .collect()
}

fn string_attribute<'a>(operation: &'a wire::OperationProto, name: &str) -> Option<&'a str> {
    let literal = operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)?
        .value
        .as_ref()?;
    match literal.value.as_ref()? {
        wire::literal_proto::Value::StringValue(value) => Some(value),
        _ => None,
    }
}

fn integer_attribute(operation: &wire::OperationProto, name: &str) -> Option<u64> {
    let literal = operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)?
        .value
        .as_ref()?;
    match literal.value.as_ref()? {
        wire::literal_proto::Value::IntValue(value) => u64::try_from(*value).ok(),
        wire::literal_proto::Value::BigintValue(encoded) if encoded.len() == 9 => {
            let (&0, bytes) = encoded.split_first()? else {
                return None;
            };
            Some(u64::from_be_bytes(bytes.try_into().ok()?))
        }
        _ => None,
    }
}
