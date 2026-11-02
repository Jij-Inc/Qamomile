//! Native synthesis of modular cleanup proofs for finite owned diagrams.
//!
//! This pass never treats provenance attributes as evidence. It constructs an
//! inverse chain from locally verified producers, then requires the independent
//! cleanup replay kernel to prove the resulting executable program. Finite
//! structural diagrams reverse tuple, unpack, reshape, primitive, ordinary,
//! inverse-call, control, and select morphisms in one ownership DAG. Unsupported
//! structured morphisms fail closed at the native preparation boundary.

use std::collections::{HashMap, HashSet};

use thiserror::Error;

use crate::call_graph::StaticCallGraph;
use crate::contracts::ProgramContracts;
use crate::effects::{CallableKind, Capability};
use crate::operations::{LocallyCheckedOperations, Opcode, QCoreOpcode, is_primitive_adjointable};
use crate::reversible_regions::{
    ReversibleFunctionPlan, ReversibleRegionError, ReversibleRegionPlan, ReversibleRootKind,
    ReversibleStepClassifier, ReversibleStepKind, classify_structural_step, form_program_plan,
    owned_operands, owned_results,
};
use crate::types::TypeArena;
use crate::wire;

/// Failure to synthesize an independently checkable cleanup program.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum CleanupSynthesisError {
    /// A producer requires a structured inverse not yet migrated.
    #[error("Q6201: native cleanup synthesis does not yet support producer {opcode:?}")]
    UnsupportedProducer {
        /// Producer opcode that requires a later native migration slice.
        opcode: String,
    },
    /// A structural diagram cannot be inverted as one closed owned morphism.
    #[error("Q6111: cleanup diagram {reason}")]
    InvalidDiagram {
        /// Stable description of the failed ownership condition.
        reason: &'static str,
    },
    /// A forget does not refer to one locally defined owned value.
    #[error("Q1106: Forget operand has no local definition")]
    MissingDefinition,
    /// An ordinary call cannot be inverted from its verified contract.
    #[error("Q6102: callable {callee:?} has no modular adjoint summary")]
    MissingAdjoint {
        /// Callable whose verified contract lacks coherent inversion.
        callee: String,
    },
    /// Fresh SSA identifiers cannot be represented without collision.
    #[error("Q4007: cleanup synthesis exhausted the SSA identifier space")]
    IdentifierCapacityExceeded,
    /// Reversible-region formation exceeded its explicit compiler work budget.
    #[error("Q4007: reversible-region analysis exceeds its compiler work limit")]
    RegionWorkBudgetExceeded,
}

fn cleanup_region_error(error: ReversibleRegionError) -> CleanupSynthesisError {
    match error {
        ReversibleRegionError::UnsupportedProducer { opcode } => {
            CleanupSynthesisError::UnsupportedProducer { opcode }
        }
        ReversibleRegionError::InvalidDiagram { reason } => {
            CleanupSynthesisError::InvalidDiagram { reason }
        }
        ReversibleRegionError::MissingDefinition => CleanupSynthesisError::MissingDefinition,
        ReversibleRegionError::MissingAdjoint { callee } => {
            CleanupSynthesisError::MissingAdjoint { callee }
        }
        ReversibleRegionError::WorkBudgetExceeded => {
            CleanupSynthesisError::RegionWorkBudgetExceeded
        }
    }
}

struct CleanupStepClassifier<'a> {
    history_replayable: &'a HashSet<usize>,
    call_graph: &'a StaticCallGraph,
    contracts: &'a ProgramContracts,
}

impl ReversibleStepClassifier for CleanupStepClassifier<'_> {
    fn classify(
        &self,
        operation: &wire::OperationProto,
        definitions: &HashMap<u64, &wire::ValueProto>,
        value_types: &HashMap<u64, crate::types::TypeId>,
        arena: &TypeArena,
    ) -> Result<ReversibleStepKind, ReversibleRegionError> {
        let structural = classify_structural_step(operation, definitions, value_types, arena)?;
        match structural {
            ReversibleStepKind::Call => {
                let callee = string_attribute(operation, "callee").unwrap_or_default();
                if callable_is_adjointable(callee, self.call_graph, self.contracts) {
                    Ok(ReversibleStepKind::Call)
                } else if self
                    .call_graph
                    .index(callee)
                    .is_some_and(|index| self.history_replayable.contains(&index))
                {
                    Ok(ReversibleStepKind::HistoryCall)
                } else {
                    Err(ReversibleRegionError::MissingAdjoint {
                        callee: callee.to_owned(),
                    })
                }
            }
            ReversibleStepKind::ControlCall => {
                let callee = string_attribute(operation, "callee").unwrap_or_default();
                require_adjointable(callee, self.call_graph, self.contracts)?;
                Ok(structural)
            }
            ReversibleStepKind::SelectCall => {
                for callee in coherent_callees(operation) {
                    require_adjointable(callee, self.call_graph, self.contracts)?;
                }
                Ok(structural)
            }
            ReversibleStepKind::If => {
                validate_reversible_if(
                    operation,
                    value_types,
                    arena,
                    self.call_graph,
                    self.contracts,
                    &HashSet::new(),
                )?;
                Ok(structural)
            }
            ReversibleStepKind::For => {
                validate_reversible_for(
                    operation,
                    value_types,
                    arena,
                    self.call_graph,
                    self.contracts,
                    &HashSet::new(),
                )?;
                Ok(structural)
            }
            _ => Ok(structural),
        }
    }
}

fn require_adjointable(
    callee: &str,
    call_graph: &StaticCallGraph,
    contracts: &ProgramContracts,
) -> Result<(), ReversibleRegionError> {
    if callable_is_adjointable(callee, call_graph, contracts) {
        Ok(())
    } else {
        Err(ReversibleRegionError::MissingAdjoint {
            callee: callee.to_owned(),
        })
    }
}

/// Rewrite every supported lexical forget into a modular inverse proof.
///
/// # Errors
///
/// Returns [`CleanupSynthesisError`] when a producer chain is malformed or
/// belongs to the structured subset whose native inversion is not implemented
/// yet.
pub(crate) fn synthesize(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
    call_graph: &StaticCallGraph,
    contracts: &ProgramContracts,
) -> Result<wire::QCorePackageProto, CleanupSynthesisError> {
    let history_replayable =
        history_replayable_functions(package, operations, arena, call_graph, contracts);
    let classifier = CleanupStepClassifier {
        history_replayable: &history_replayable,
        call_graph,
        contracts,
    };
    let region_plan =
        form_program_plan(package, operations, arena, &classifier).map_err(cleanup_region_error)?;
    let mut restored = package.clone();
    let mut next_id = maximum_value_id(package)
        .checked_add(1)
        .ok_or(CleanupSynthesisError::IdentifierCapacityExceeded)?;
    for function in &mut restored.functions {
        let values = operations
            .value_types(&function.symbol)
            .ok_or(CleanupSynthesisError::MissingDefinition)?;
        let function_plan = region_plan
            .function(&function.symbol)
            .ok_or(CleanupSynthesisError::MissingDefinition)?;
        if let Some(body) = function.body.as_ref() {
            function.body = Some(synthesize_block(
                body,
                function_plan,
                values,
                arena,
                &mut next_id,
            )?);
        }
    }
    Ok(restored)
}

fn synthesize_block(
    source: &wire::BlockProto,
    function_plan: &ReversibleFunctionPlan<'_>,
    value_types: &HashMap<u64, crate::types::TypeId>,
    arena: &TypeArena,
    next_id: &mut u64,
) -> Result<wire::BlockProto, CleanupSynthesisError> {
    let mut rewritten = Vec::new();
    for source_operation in &source.operations {
        let mut operation = source_operation.clone();
        for region in &mut operation.regions {
            for block in &mut region.blocks {
                *block = synthesize_block(block, function_plan, value_types, arena, next_id)?;
            }
        }
        if !matches!(
            Opcode::try_from(operation.opcode.as_str()),
            Ok(Opcode::QCore(QCoreOpcode::Forget))
        ) {
            rewritten.push(operation);
            continue;
        }
        let value_id = operation
            .operands
            .first()
            .copied()
            .ok_or(CleanupSynthesisError::MissingDefinition)?;
        let current = function_plan
            .definition(value_id)
            .cloned()
            .ok_or(CleanupSynthesisError::MissingDefinition)?;
        let region = function_plan
            .region(value_id)
            .ok_or(CleanupSynthesisError::MissingDefinition)?;
        synthesize_owned_diagram(
            &current,
            region,
            function_plan,
            value_types,
            arena,
            next_id,
            &mut rewritten,
        )?;
    }
    Ok(wire::BlockProto {
        arguments: source.arguments.clone(),
        operations: rewritten,
        terminator: source.terminator.clone(),
    })
}

#[allow(clippy::too_many_lines)]
fn synthesize_owned_diagram(
    current: &wire::ValueProto,
    region: &ReversibleRegionPlan,
    function_plan: &ReversibleFunctionPlan<'_>,
    value_types: &HashMap<u64, crate::types::TypeId>,
    arena: &TypeArena,
    next_id: &mut u64,
    output: &mut Vec<wire::OperationProto>,
) -> Result<(), CleanupSynthesisError> {
    let mut available = HashMap::from([(region.origin(), current.clone())]);
    for step in region.steps().iter().rev() {
        let operation = function_plan
            .producer(step.producer_key())
            .ok_or(CleanupSynthesisError::MissingDefinition)?;
        let linear_operands = owned_operands(operation, value_types, arena);
        let linear_results = owned_results(operation, value_types, arena);
        let current_values = linear_results
            .iter()
            .map(|result| {
                available
                    .get(&result.id)
                    .cloned()
                    .ok_or(CleanupSynthesisError::InvalidDiagram {
                        reason: "has a linear branch outside its returned value",
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let restored = linear_operands
            .iter()
            .map(|identifier| {
                function_plan
                    .definition(*identifier)
                    .ok_or(CleanupSynthesisError::MissingDefinition)
                    .and_then(|value| fresh_value(value, "uncomputed_diagram", next_id))
            })
            .collect::<Result<Vec<_>, _>>()?;
        match step.kind() {
            ReversibleStepKind::Tuple => output.push(wire::OperationProto {
                opcode: "qcore.unpack".to_owned(),
                operands: vec![current_values[0].id],
                results: restored.clone(),
                attributes: string_attributes(&[("proof", "modular-adjoint")]),
                span: operation.span.clone(),
                ..Default::default()
            }),
            ReversibleStepKind::Unpack => output.push(wire::OperationProto {
                opcode: "qcore.tuple".to_owned(),
                operands: current_values.iter().map(|value| value.id).collect(),
                results: vec![restored[0].clone()],
                attributes: string_attributes(&[("proof", "modular-adjoint")]),
                span: operation.span.clone(),
                ..Default::default()
            }),
            ReversibleStepKind::Reshape => output.push(wire::OperationProto {
                opcode: "qcore.reshape".to_owned(),
                operands: vec![current_values[0].id],
                results: vec![restored[0].clone()],
                attributes: string_attributes(&[("proof", "modular-adjoint")]),
                span: operation.span.clone(),
                ..Default::default()
            }),
            ReversibleStepKind::Primitive => {
                output.push(wire::OperationProto {
                    opcode: "qcore.adjoint".to_owned(),
                    operands: std::iter::once(current_values[0].id)
                        .chain(
                            operation
                                .operands
                                .iter()
                                .copied()
                                .filter(|operand| !linear_operands.contains(operand)),
                        )
                        .collect(),
                    results: vec![restored[0].clone()],
                    attributes: adjoint_attributes(operation),
                    span: operation.span.clone(),
                    ..Default::default()
                });
            }
            ReversibleStepKind::Call | ReversibleStepKind::HistoryCall => {
                output.extend(invert_diagram_call(
                    operation,
                    &current_values[0],
                    &restored,
                    value_types,
                    arena,
                    step.kind() == ReversibleStepKind::HistoryCall,
                    next_id,
                )?);
            }
            ReversibleStepKind::AdjointCall => {
                let callee = string_attribute(operation, "callee").unwrap_or_default();
                let classical = operation.operands.iter().copied().filter(|operand| {
                    value_types
                        .get(operand)
                        .is_none_or(|type_id| !arena.is_quantum(*type_id))
                });
                output.push(wire::OperationProto {
                    opcode: "qcore.call".to_owned(),
                    operands: std::iter::once(current_values[0].id)
                        .chain(classical)
                        .collect(),
                    results: vec![restored[0].clone()],
                    attributes: string_attributes(&[("callee", callee)]),
                    span: operation.span.clone(),
                    ..Default::default()
                });
            }
            ReversibleStepKind::ControlCall | ReversibleStepKind::SelectCall => {
                invert_coherent_call(
                    operation,
                    &current_values[0],
                    &restored,
                    value_types,
                    arena,
                    next_id,
                    output,
                )?;
            }
            ReversibleStepKind::If => invert_if(
                operation,
                &current_values[0],
                &restored,
                value_types,
                arena,
                next_id,
                output,
            )?,
            ReversibleStepKind::For => invert_for(
                operation,
                &current_values[0],
                &restored,
                value_types,
                arena,
                next_id,
                output,
            )?,
        }
        available.extend(linear_operands.into_iter().zip(restored));
    }
    for root in region.roots().iter().copied() {
        let root_id = root.value();
        let restored = available
            .remove(&root_id)
            .ok_or(CleanupSynthesisError::InvalidDiagram {
                reason: "does not restore every owned root",
            })?;
        let producer = function_plan
            .producer(root_id)
            .ok_or(CleanupSynthesisError::MissingDefinition)?;
        let opcode = match root.kind() {
            ReversibleRootKind::Allocation => "quantum.release",
            ReversibleRootKind::Preparation => "quantum.unprepare",
        };
        output.push(wire::OperationProto {
            opcode: opcode.to_owned(),
            operands: std::iter::once(restored.id)
                .chain(
                    (root.kind() == ReversibleRootKind::Preparation)
                        .then_some(producer.operands.as_slice())
                        .into_iter()
                        .flatten()
                        .copied(),
                )
                .collect(),
            attributes: cleanup_attributes(root_id),
            span: producer.span.clone(),
            ..Default::default()
        });
    }
    Ok(())
}

fn invert_if(
    operation: &wire::OperationProto,
    current: &wire::ValueProto,
    restored: &[wire::ValueProto],
    value_types: &HashMap<u64, crate::types::TypeId>,
    arena: &TypeArena,
    next_id: &mut u64,
    output: &mut Vec<wire::OperationProto>,
) -> Result<(), CleanupSynthesisError> {
    let [restored] = restored else {
        return Err(CleanupSynthesisError::InvalidDiagram {
            reason: "contains a conditional without one owned carry",
        });
    };
    let operands = operation
        .operands
        .iter()
        .map(|operand| {
            if value_types
                .get(operand)
                .is_some_and(|type_id| arena.is_quantum(*type_id))
            {
                current.id
            } else {
                *operand
            }
        })
        .collect();
    let results = operation
        .results
        .iter()
        .map(|result| {
            if value_types
                .get(&result.id)
                .is_some_and(|type_id| arena.is_quantum(*type_id))
            {
                Ok(restored.clone())
            } else {
                fresh_value(result, "uncomputed_if_carry", next_id)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let rebase = RegionRebase::new(restored.region.as_ref(), current.region.as_ref());
    let regions = operation
        .regions
        .iter()
        .map(|region| freshen_region(region, rebase.as_ref(), next_id))
        .collect::<Result<Vec<_>, _>>()?;
    output.push(wire::OperationProto {
        opcode: "qcore.if".to_owned(),
        operands,
        results,
        attributes: adjoint_if_attributes(operation),
        regions,
        span: operation.span.clone(),
        intent: Vec::new(),
    });
    Ok(())
}

fn validate_reversible_if(
    operation: &wire::OperationProto,
    value_types: &HashMap<u64, crate::types::TypeId>,
    arena: &TypeArena,
    call_graph: &StaticCallGraph,
    contracts: &ProgramContracts,
    history_replayable: &HashSet<usize>,
) -> Result<(), ReversibleRegionError> {
    let captures = operation.operands.get(1..).unwrap_or_default();
    if captures.len() != operation.results.len() || operation.regions.len() != 2 {
        return invalid_if();
    }
    for region in &operation.regions {
        let [branch] = region.blocks.as_slice() else {
            return invalid_if();
        };
        let Some(terminator) = branch.terminator.as_ref() else {
            return invalid_if();
        };
        if branch.arguments.len() != captures.len()
            || terminator.kind != "yield"
            || terminator.operands.len() != operation.results.len()
        {
            return invalid_if();
        }
        for (index, (capture, result)) in captures.iter().zip(&operation.results).enumerate() {
            let linear = value_types
                .get(capture)
                .is_some_and(|type_id| arena.is_quantum(*type_id));
            if !linear
                && (terminator.operands[index] != branch.arguments[index].id
                    || value_types.get(capture) != value_types.get(&result.id))
            {
                return Err(ReversibleRegionError::InvalidDiagram {
                    reason: "contains a conditional that mutates a classical carry",
                });
            }
        }
        if !branch.operations.iter().all(|nested| {
            reversible_loop_operation(nested, call_graph, contracts, history_replayable)
        }) {
            return Err(ReversibleRegionError::InvalidDiagram {
                reason: "contains a conditional branch outside the adjointable subset",
            });
        }
    }
    Ok(())
}

fn invalid_if() -> Result<(), ReversibleRegionError> {
    Err(ReversibleRegionError::InvalidDiagram {
        reason: "contains a conditional without a reversible carry shape",
    })
}

fn invert_for(
    operation: &wire::OperationProto,
    current: &wire::ValueProto,
    restored: &[wire::ValueProto],
    value_types: &HashMap<u64, crate::types::TypeId>,
    arena: &TypeArena,
    next_id: &mut u64,
    output: &mut Vec<wire::OperationProto>,
) -> Result<(), CleanupSynthesisError> {
    let [restored] = restored else {
        return Err(CleanupSynthesisError::InvalidDiagram {
            reason: "contains a loop without one owned carry",
        });
    };
    let operands = operation
        .operands
        .iter()
        .map(|operand| {
            if value_types
                .get(operand)
                .is_some_and(|type_id| arena.is_quantum(*type_id))
            {
                current.id
            } else {
                *operand
            }
        })
        .collect();
    let results = operation
        .results
        .iter()
        .map(|result| {
            if value_types
                .get(&result.id)
                .is_some_and(|type_id| arena.is_quantum(*type_id))
            {
                Ok(restored.clone())
            } else {
                fresh_value(result, "uncomputed_loop_carry", next_id)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let regions = freshen_for_regions(operation, current, restored, next_id)?;
    output.push(wire::OperationProto {
        opcode: "qcore.for".to_owned(),
        operands,
        results,
        attributes: adjoint_for_attributes(operation),
        regions,
        span: operation.span.clone(),
        intent: Vec::new(),
    });
    Ok(())
}

fn freshen_for_regions(
    operation: &wire::OperationProto,
    current: &wire::ValueProto,
    restored: &wire::ValueProto,
    next_id: &mut u64,
) -> Result<Vec<wire::RegionProto>, CleanupSynthesisError> {
    let body_rebase = RegionRebase::new(restored.region.as_ref(), current.region.as_ref());
    let else_rebase = RegionRebase::new(current.region.as_ref(), restored.region.as_ref());
    operation
        .regions
        .iter()
        .enumerate()
        .map(|(index, region)| {
            freshen_region(
                region,
                if index == 0 {
                    body_rebase.as_ref()
                } else {
                    else_rebase.as_ref()
                },
                next_id,
            )
        })
        .collect()
}

#[derive(Clone, Debug)]
pub(crate) struct RegionRebase {
    source: wire::RegionRefProto,
    target: wire::RegionRefProto,
}

impl RegionRebase {
    pub(crate) fn new(
        source: Option<&wire::RegionRefProto>,
        target: Option<&wire::RegionRefProto>,
    ) -> Option<Self> {
        source.zip(target).map(|(source, target)| Self {
            source: source.clone(),
            target: target.clone(),
        })
    }

    pub(crate) fn apply(&self, region: &mut wire::RegionRefProto) {
        if region.root != self.source.root
            || !region.projections.starts_with(&self.source.projections)
        {
            return;
        }
        let relative = region.projections[self.source.projections.len()..].to_vec();
        region.root = self.target.root;
        region.projections = self
            .target
            .projections
            .iter()
            .cloned()
            .chain(relative)
            .collect();
    }
}

pub(crate) fn freshen_region(
    region: &wire::RegionProto,
    rebase: Option<&RegionRebase>,
    next_id: &mut u64,
) -> Result<wire::RegionProto, CleanupSynthesisError> {
    let mut identifiers = HashMap::new();
    Ok(wire::RegionProto {
        blocks: region
            .blocks
            .iter()
            .map(|block| freshen_block(block, &mut identifiers, rebase, next_id))
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn freshen_block(
    block: &wire::BlockProto,
    identifiers: &mut HashMap<u64, u64>,
    rebase: Option<&RegionRebase>,
    next_id: &mut u64,
) -> Result<wire::BlockProto, CleanupSynthesisError> {
    let arguments = block
        .arguments
        .iter()
        .map(|value| freshen_region_value(value, identifiers, rebase, next_id))
        .collect::<Result<Vec<_>, _>>()?;
    let mut operations = Vec::with_capacity(block.operations.len());
    for operation in &block.operations {
        let operands = operation
            .operands
            .iter()
            .map(|operand| {
                identifiers
                    .get(operand)
                    .copied()
                    .ok_or(CleanupSynthesisError::MissingDefinition)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let results = operation
            .results
            .iter()
            .map(|value| freshen_region_value(value, identifiers, rebase, next_id))
            .collect::<Result<Vec<_>, _>>()?;
        let regions = operation
            .regions
            .iter()
            .map(|region| freshen_region(region, rebase, next_id))
            .collect::<Result<Vec<_>, _>>()?;
        operations.push(wire::OperationProto {
            opcode: operation.opcode.clone(),
            operands,
            results,
            attributes: operation.attributes.clone(),
            regions,
            span: operation.span.clone(),
            intent: Vec::new(),
        });
    }
    let terminator = block
        .terminator
        .as_ref()
        .map(|terminator| {
            Ok(wire::TerminatorProto {
                kind: terminator.kind.clone(),
                operands: terminator
                    .operands
                    .iter()
                    .map(|operand| {
                        identifiers
                            .get(operand)
                            .copied()
                            .ok_or(CleanupSynthesisError::MissingDefinition)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                attributes: terminator.attributes.clone(),
                span: terminator.span.clone(),
            })
        })
        .transpose()?;
    Ok(wire::BlockProto {
        arguments,
        operations,
        terminator,
    })
}

fn freshen_region_value(
    value: &wire::ValueProto,
    identifiers: &mut HashMap<u64, u64>,
    rebase: Option<&RegionRebase>,
    next_id: &mut u64,
) -> Result<wire::ValueProto, CleanupSynthesisError> {
    let mut fresh = value.clone();
    fresh.id = take_id(next_id)?;
    identifiers.insert(value.id, fresh.id);
    if let Some(region) = &mut fresh.region
        && let Some(rebase) = rebase
    {
        rebase.apply(region);
    }
    Ok(fresh)
}

fn validate_reversible_for(
    operation: &wire::OperationProto,
    value_types: &HashMap<u64, crate::types::TypeId>,
    arena: &TypeArena,
    call_graph: &StaticCallGraph,
    contracts: &ProgramContracts,
    history_replayable: &HashSet<usize>,
) -> Result<(), ReversibleRegionError> {
    let carries = operation.operands.get(3..).unwrap_or_default();
    let [body_region, else_region] = operation.regions.as_slice() else {
        return invalid_loop();
    };
    let [body] = body_region.blocks.as_slice() else {
        return invalid_loop();
    };
    let [otherwise] = else_region.blocks.as_slice() else {
        return invalid_loop();
    };
    let Some(body_terminator) = body.terminator.as_ref() else {
        return invalid_loop();
    };
    let Some(else_terminator) = otherwise.terminator.as_ref() else {
        return invalid_loop();
    };
    if operation.operands.len() < 4
        || carries.len() != operation.results.len()
        || body.arguments.len() != carries.len() + 1
        || otherwise.arguments.len() != carries.len()
        || body_terminator.kind != "yield"
        || else_terminator.kind != "yield"
        || !otherwise.operations.is_empty()
        || body_terminator.operands.len() != carries.len()
        || else_terminator.operands.len() != carries.len()
    {
        return invalid_loop();
    }
    for (index, (carry, result)) in carries.iter().zip(&operation.results).enumerate() {
        let linear = value_types
            .get(carry)
            .is_some_and(|type_id| arena.is_quantum(*type_id));
        if !linear
            && (body_terminator.operands[index] != body.arguments[index + 1].id
                || else_terminator.operands[index] != otherwise.arguments[index].id
                || value_types.get(carry) != value_types.get(&result.id))
        {
            return Err(ReversibleRegionError::InvalidDiagram {
                reason: "contains a loop that mutates a classical carry",
            });
        }
    }
    if !body
        .operations
        .iter()
        .all(|nested| reversible_loop_operation(nested, call_graph, contracts, history_replayable))
    {
        return Err(ReversibleRegionError::InvalidDiagram {
            reason: "contains a loop body outside the finite adjointable subset",
        });
    }
    Ok(())
}

fn history_reversible_operation(
    operation: &wire::OperationProto,
    value_types: &HashMap<u64, crate::types::TypeId>,
    arena: &TypeArena,
    call_graph: &StaticCallGraph,
    contracts: &ProgramContracts,
    history_replayable: &HashSet<usize>,
) -> bool {
    if operation.regions.is_empty() {
        return reversible_loop_operation(operation, call_graph, contracts, history_replayable);
    }
    match operation.opcode.as_str() {
        "qcore.if" => history_reversible_if(
            operation,
            value_types,
            arena,
            call_graph,
            contracts,
            history_replayable,
        ),
        "qcore.for" => validate_reversible_for(
            operation,
            value_types,
            arena,
            call_graph,
            contracts,
            history_replayable,
        )
        .is_ok(),
        _ => false,
    }
}

fn history_reversible_block(
    block: &wire::BlockProto,
    value_types: &HashMap<u64, crate::types::TypeId>,
    arena: &TypeArena,
    call_graph: &StaticCallGraph,
    contracts: &ProgramContracts,
    history_replayable: &HashSet<usize>,
) -> bool {
    let mut used = block
        .terminator
        .iter()
        .flat_map(|terminator| terminator.operands.iter().copied())
        .collect::<HashSet<_>>();
    for operation in block.operations.iter().rev() {
        if !history_reversible_operation(
            operation,
            value_types,
            arena,
            call_graph,
            contracts,
            history_replayable,
        ) {
            return false;
        }
        if !operation.regions.is_empty() {
            let identity_results = structured_classical_identity_results(operation, value_types);
            if operation.results.iter().enumerate().any(|(index, result)| {
                !value_types
                    .get(&result.id)
                    .is_some_and(|type_id| arena.is_quantum(*type_id))
                    && used.contains(&result.id)
                    && !identity_results.get(index).copied().unwrap_or(false)
            }) {
                return false;
            }
        }
        for result in &operation.results {
            used.remove(&result.id);
        }
        used.extend(operation.operands.iter().copied());
    }
    true
}

pub(crate) fn structured_classical_identity_results(
    operation: &wire::OperationProto,
    value_types: &HashMap<u64, crate::types::TypeId>,
) -> Vec<bool> {
    let mut memo = HashMap::new();
    operation
        .results
        .iter()
        .enumerate()
        .map(|(index, _)| {
            structured_classical_result_is_identity(operation, index, value_types, &mut memo)
        })
        .collect()
}

fn structured_classical_result_is_identity(
    operation: &wire::OperationProto,
    index: usize,
    value_types: &HashMap<u64, crate::types::TypeId>,
    memo: &mut HashMap<u64, bool>,
) -> bool {
    let Some(operand_offset) = structured_operand_offset(operation) else {
        return false;
    };
    let Some(operand) = operation.operands.get(operand_offset + index) else {
        return false;
    };
    let Some(result) = operation.results.get(index) else {
        return false;
    };
    if let Some(identity) = memo.get(&result.id) {
        return *identity;
    }
    if value_types.get(operand) != value_types.get(&result.id) {
        memo.insert(result.id, false);
        return false;
    }
    let identity = operation
        .regions
        .iter()
        .enumerate()
        .all(|(region_index, region)| {
            let [block] = region.blocks.as_slice() else {
                return false;
            };
            let argument_index = usize::from(operation.opcode == "qcore.for" && region_index == 0);
            block_identity_alias(block, argument_index + index, index, value_types, memo)
        });
    memo.insert(result.id, identity);
    identity
}

fn structured_operand_offset(operation: &wire::OperationProto) -> Option<usize> {
    match operation.opcode.as_str() {
        "qcore.if" => Some(1),
        "qcore.for" => Some(3),
        _ => None,
    }
}

fn block_identity_alias(
    block: &wire::BlockProto,
    argument_index: usize,
    terminator_index: usize,
    value_types: &HashMap<u64, crate::types::TypeId>,
    memo: &mut HashMap<u64, bool>,
) -> bool {
    let Some(argument) = block.arguments.get(argument_index) else {
        return false;
    };
    let mut aliases = block
        .arguments
        .iter()
        .map(|argument| (argument.id, argument.id))
        .collect::<HashMap<_, _>>();
    for operation in &block.operations {
        let Some(offset) = structured_operand_offset(operation) else {
            continue;
        };
        for (index, result) in operation.results.iter().enumerate() {
            if !structured_classical_result_is_identity(operation, index, value_types, memo) {
                continue;
            }
            let Some(source) = operation.operands.get(offset + index) else {
                return false;
            };
            if let Some(alias) = aliases.get(source).copied() {
                aliases.insert(result.id, alias);
            }
        }
    }
    block
        .terminator
        .as_ref()
        .and_then(|terminator| terminator.operands.get(terminator_index))
        .and_then(|yielded| aliases.get(yielded))
        == Some(&argument.id)
}

fn history_reversible_if(
    operation: &wire::OperationProto,
    value_types: &HashMap<u64, crate::types::TypeId>,
    arena: &TypeArena,
    call_graph: &StaticCallGraph,
    contracts: &ProgramContracts,
    history_replayable: &HashSet<usize>,
) -> bool {
    let captures = operation.operands.get(1..).unwrap_or_default();
    if captures.len() != operation.results.len() || operation.regions.len() != 2 {
        return false;
    }
    let owned_captures = captures
        .iter()
        .filter(|capture| {
            value_types
                .get(capture)
                .is_some_and(|type_id| arena.is_quantum(*type_id))
        })
        .count();
    let owned_results = operation
        .results
        .iter()
        .filter(|result| {
            value_types
                .get(&result.id)
                .is_some_and(|type_id| arena.is_quantum(*type_id))
        })
        .count();
    owned_captures == 1
        && owned_results == 1
        && operation.regions.iter().all(|region| {
            let [branch] = region.blocks.as_slice() else {
                return false;
            };
            branch.arguments.len() == captures.len()
                && branch.terminator.as_ref().is_some_and(|terminator| {
                    terminator.kind == "yield"
                        && terminator.operands.len() == operation.results.len()
                })
                && history_reversible_block(
                    branch,
                    value_types,
                    arena,
                    call_graph,
                    contracts,
                    history_replayable,
                )
        })
}

fn invalid_loop() -> Result<(), ReversibleRegionError> {
    Err(ReversibleRegionError::InvalidDiagram {
        reason: "contains a loop without a reversible carry shape",
    })
}

fn reversible_loop_operation(
    operation: &wire::OperationProto,
    call_graph: &StaticCallGraph,
    contracts: &ProgramContracts,
    history_replayable: &HashSet<usize>,
) -> bool {
    if !operation.regions.is_empty() {
        return false;
    }
    match operation.opcode.as_str() {
        opcode if is_primitive_adjointable(opcode) => true,
        "qcore.constant" | "qcore.stage_lift" | "qcore.tuple" | "qcore.unpack"
        | "qcore.reshape" | "qcore.adjoint" => true,
        "qcore.call" => string_attribute(operation, "callee").is_some_and(|callee| {
            callable_is_adjointable(callee, call_graph, contracts)
                || call_graph
                    .index(callee)
                    .is_some_and(|index| history_replayable.contains(&index))
        }),
        "qcore.adjoint_call" | "qcore.control_call" => string_attribute(operation, "callee")
            .is_some_and(|callee| callable_is_adjointable(callee, call_graph, contracts)),
        "qcore.select_call" => coherent_callees(operation)
            .into_iter()
            .all(|callee| callable_is_adjointable(callee, call_graph, contracts)),
        opcode if opcode.starts_with("classical.") || opcode == "array" || opcode == "tuple" => {
            true
        }
        _ => false,
    }
}

fn invert_coherent_call(
    operation: &wire::OperationProto,
    current: &wire::ValueProto,
    restored: &[wire::ValueProto],
    value_types: &HashMap<u64, crate::types::TypeId>,
    arena: &TypeArena,
    next_id: &mut u64,
    output: &mut Vec<wire::OperationProto>,
) -> Result<(), CleanupSynthesisError> {
    if restored.len() != 2 {
        return Err(CleanupSynthesisError::InvalidDiagram {
            reason: "contains a coherent call without selector and target outputs",
        });
    }
    let classical = operation.operands.iter().copied().filter(|operand| {
        value_types
            .get(operand)
            .is_none_or(|type_id| !arena.is_quantum(*type_id))
    });
    let mut packed = current.clone();
    packed.id = take_id(next_id)?;
    "uncomputed_coherent_call".clone_into(&mut packed.hint);
    output.push(wire::OperationProto {
        opcode: "qcore.adjoint".to_owned(),
        operands: std::iter::once(current.id).chain(classical).collect(),
        results: vec![packed.clone()],
        attributes: adjoint_attributes(operation),
        span: operation.span.clone(),
        ..Default::default()
    });
    output.push(wire::OperationProto {
        opcode: "qcore.unpack".to_owned(),
        operands: vec![packed.id],
        results: restored.to_vec(),
        attributes: string_attributes(&[("proof", "modular-adjoint")]),
        span: operation.span.clone(),
        ..Default::default()
    });
    Ok(())
}

fn coherent_callees(operation: &wire::OperationProto) -> Vec<&str> {
    if operation.opcode == "qcore.control_call" {
        string_attribute(operation, "callee").into_iter().collect()
    } else {
        string_tuple_attribute(operation, "callees")
    }
}

fn invert_diagram_call(
    operation: &wire::OperationProto,
    current: &wire::ValueProto,
    restored: &[wire::ValueProto],
    value_types: &HashMap<u64, crate::types::TypeId>,
    arena: &TypeArena,
    history_adjoint: bool,
    next_id: &mut u64,
) -> Result<Vec<wire::OperationProto>, CleanupSynthesisError> {
    let callee = string_attribute(operation, "callee").unwrap_or_default();
    let packed = if restored.len() == 1 {
        restored[0].clone()
    } else {
        let mut packed = current.clone();
        packed.id = take_id(next_id)?;
        "uncomputed_diagram_call".clone_into(&mut packed.hint);
        packed.r#type = Some(wire::TypeProto {
            kind: wire::type_proto::Kind::Product.into(),
            arguments: restored
                .iter()
                .map(|value| {
                    value
                        .r#type
                        .clone()
                        .ok_or(CleanupSynthesisError::MissingDefinition)
                })
                .collect::<Result<Vec<_>, _>>()?,
            ..Default::default()
        });
        packed
    };
    let classical = operation.operands.iter().copied().filter(|operand| {
        value_types
            .get(operand)
            .is_none_or(|type_id| !arena.is_quantum(*type_id))
    });
    let mut inverse = vec![wire::OperationProto {
        opcode: "qcore.adjoint_call".to_owned(),
        operands: std::iter::once(current.id).chain(classical).collect(),
        results: vec![packed.clone()],
        attributes: if history_adjoint {
            history_adjoint_call_attributes(callee, operation.results[0].id)
        } else {
            string_attributes(&[("callee", callee), ("proof", "modular-adjoint")])
        },
        span: operation.span.clone(),
        ..Default::default()
    }];
    if restored.len() > 1 {
        inverse.push(wire::OperationProto {
            opcode: "qcore.unpack".to_owned(),
            operands: vec![packed.id],
            results: restored.to_vec(),
            attributes: string_attributes(&[("proof", "modular-adjoint")]),
            span: operation.span.clone(),
            ..Default::default()
        });
    }
    Ok(inverse)
}

pub(crate) fn history_replayable_functions(
    package: &wire::QCorePackageProto,
    operations: &LocallyCheckedOperations,
    arena: &TypeArena,
    call_graph: &StaticCallGraph,
    contracts: &ProgramContracts,
) -> HashSet<usize> {
    let mut replayable = HashSet::new();
    loop {
        let previous = replayable.len();
        for &index in call_graph.callee_first_order() {
            if callable_is_history_replayable(
                index,
                package,
                operations,
                arena,
                call_graph,
                contracts,
                &replayable,
            ) {
                replayable.insert(index);
            }
        }
        if replayable.len() == previous {
            break;
        }
    }
    replayable
}

#[allow(clippy::too_many_arguments)]
fn callable_is_history_replayable(
    index: usize,
    package: &wire::QCorePackageProto,
    operations: &LocallyCheckedOperations,
    arena: &TypeArena,
    call_graph: &StaticCallGraph,
    contracts: &ProgramContracts,
    replayable: &HashSet<usize>,
) -> bool {
    let Some(function) = package.functions.get(index) else {
        return false;
    };
    let Some(contract) = contracts.contract(index) else {
        return false;
    };
    let Some(value_types) = operations.value_types(&function.symbol) else {
        return false;
    };
    let Some(result_type) = operations.result_type(&function.symbol) else {
        return false;
    };
    let Some(body) = function.body.as_ref() else {
        return false;
    };
    let owned_parameters = function
        .parameters
        .iter()
        .filter(|parameter| {
            parameter
                .value
                .as_ref()
                .and_then(|value| value_types.get(&value.id))
                .is_some_and(|type_id| arena.is_quantum(*type_id))
        })
        .count();
    let owned_returns = body
        .terminator
        .as_ref()
        .filter(|terminator| terminator.kind == "return")
        .into_iter()
        .flat_map(|terminator| terminator.operands.iter())
        .filter(|value| {
            value_types
                .get(value)
                .is_some_and(|type_id| arena.is_quantum(*type_id))
        })
        .count();
    !function.external
        && contract.kind() == CallableKind::Unitary
        && owned_parameters == 1
        && arena.is_quantum(result_type)
        && owned_returns == 1
        && history_reversible_block(body, value_types, arena, call_graph, contracts, replayable)
}

fn callable_is_adjointable(
    callee: &str,
    call_graph: &StaticCallGraph,
    contracts: &ProgramContracts,
) -> bool {
    call_graph.index(callee).is_some_and(|index| {
        contracts
            .contract(index)
            .is_some_and(|contract| contract.capabilities().contains(Capability::Adjointable))
    })
}

fn fresh_value(
    source: &wire::ValueProto,
    hint: &str,
    next_id: &mut u64,
) -> Result<wire::ValueProto, CleanupSynthesisError> {
    let mut value = source.clone();
    value.id = take_id(next_id)?;
    hint.clone_into(&mut value.hint);
    Ok(value)
}

fn take_id(next_id: &mut u64) -> Result<u64, CleanupSynthesisError> {
    let value = *next_id;
    *next_id = next_id
        .checked_add(1)
        .ok_or(CleanupSynthesisError::IdentifierCapacityExceeded)?;
    Ok(value)
}

fn cleanup_attributes(origin: u64) -> Vec<wire::AttributeProto> {
    vec![
        wire::AttributeProto {
            name: "origin".to_owned(),
            value: Some(nonnegative_integer(origin)),
        },
        string_attribute_value("proof", "modular-adjoint"),
    ]
}

fn history_adjoint_call_attributes(callee: &str, origin: u64) -> Vec<wire::AttributeProto> {
    vec![
        string_attribute_value("callee", callee),
        wire::AttributeProto {
            name: "history_origin".to_owned(),
            value: Some(nonnegative_integer(origin)),
        },
        string_attribute_value("proof", "modular-adjoint"),
    ]
}

fn adjoint_attributes(producer: &wire::OperationProto) -> Vec<wire::AttributeProto> {
    let entries = producer
        .attributes
        .iter()
        .map(|attribute| wire::literal_proto::Entry {
            key: attribute.name.clone(),
            value: attribute.value.clone(),
        })
        .collect();
    vec![
        string_attribute_value("original", &producer.opcode),
        wire::AttributeProto {
            name: "original_attributes".to_owned(),
            value: Some(wire::LiteralProto {
                value: Some(wire::literal_proto::Value::MappingValue(
                    wire::literal_proto::Mapping { entries },
                )),
            }),
        },
        string_attribute_value("proof", "modular-adjoint"),
    ]
}

pub(crate) fn adjoint_if_attributes(producer: &wire::OperationProto) -> Vec<wire::AttributeProto> {
    let entries = producer
        .attributes
        .iter()
        .map(|attribute| wire::literal_proto::Entry {
            key: attribute.name.clone(),
            value: attribute.value.clone(),
        })
        .collect();
    let mut attributes = producer.attributes.clone();
    attributes.extend([
        string_attribute_value("body_mode", "adjoint"),
        wire::AttributeProto {
            name: "original_attributes".to_owned(),
            value: Some(wire::LiteralProto {
                value: Some(wire::literal_proto::Value::MappingValue(
                    wire::literal_proto::Mapping { entries },
                )),
            }),
        },
        string_attribute_value("proof", "modular-adjoint"),
    ]);
    attributes.sort_by(|left, right| left.name.cmp(&right.name));
    attributes
}

pub(crate) fn adjoint_for_attributes(producer: &wire::OperationProto) -> Vec<wire::AttributeProto> {
    let entries = producer
        .attributes
        .iter()
        .map(|attribute| wire::literal_proto::Entry {
            key: attribute.name.clone(),
            value: attribute.value.clone(),
        })
        .collect();
    let mut attributes = producer
        .attributes
        .iter()
        .filter(|attribute| attribute.name != "reverse")
        .cloned()
        .collect::<Vec<_>>();
    attributes.extend([
        string_attribute_value("body_mode", "adjoint"),
        wire::AttributeProto {
            name: "original_attributes".to_owned(),
            value: Some(wire::LiteralProto {
                value: Some(wire::literal_proto::Value::MappingValue(
                    wire::literal_proto::Mapping { entries },
                )),
            }),
        },
        string_attribute_value("proof", "modular-adjoint"),
        bool_attribute_value(
            "reverse",
            !bool_attribute(producer, "reverse").unwrap_or(false),
        ),
    ]);
    attributes.sort_by(|left, right| left.name.cmp(&right.name));
    attributes
}

fn string_attributes(values: &[(&str, &str)]) -> Vec<wire::AttributeProto> {
    values
        .iter()
        .map(|(name, value)| string_attribute_value(name, value))
        .collect()
}

fn string_attribute_value(name: &str, value: &str) -> wire::AttributeProto {
    wire::AttributeProto {
        name: name.to_owned(),
        value: Some(wire::LiteralProto {
            value: Some(wire::literal_proto::Value::StringValue(value.to_owned())),
        }),
    }
}

fn bool_attribute_value(name: &str, value: bool) -> wire::AttributeProto {
    wire::AttributeProto {
        name: name.to_owned(),
        value: Some(wire::LiteralProto {
            value: Some(wire::literal_proto::Value::BoolValue(value)),
        }),
    }
}

fn nonnegative_integer(value: u64) -> wire::LiteralProto {
    let encoded = if let Ok(value) = i64::try_from(value) {
        wire::literal_proto::Value::IntValue(value)
    } else {
        let bytes = value.to_be_bytes();
        let first = bytes
            .iter()
            .position(|byte| *byte != 0)
            .unwrap_or(bytes.len() - 1);
        let mut minimal = bytes[first..].to_vec();
        if minimal[0] >= 0x80 {
            minimal.insert(0, 0);
        }
        wire::literal_proto::Value::BigintValue(minimal)
    };
    wire::LiteralProto {
        value: Some(encoded),
    }
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

fn bool_attribute(operation: &wire::OperationProto, name: &str) -> Option<bool> {
    let literal = operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)?
        .value
        .as_ref()?;
    match literal.value.as_ref()? {
        wire::literal_proto::Value::BoolValue(value) => Some(*value),
        _ => None,
    }
}

fn string_tuple_attribute<'a>(operation: &'a wire::OperationProto, name: &str) -> Vec<&'a str> {
    use wire::literal_proto::Value;

    let Some(Value::TupleValue(sequence)) = operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .and_then(|attribute| attribute.value.as_ref())
        .and_then(|literal| literal.value.as_ref())
    else {
        return Vec::new();
    };
    sequence
        .values
        .iter()
        .filter_map(|item| match item.value.as_ref() {
            Some(Value::StringValue(value)) => Some(value.as_str()),
            _ => None,
        })
        .collect()
}

pub(crate) fn maximum_value_id(package: &wire::QCorePackageProto) -> u64 {
    let mut maximum = 0;
    for function in &package.functions {
        maximum = maximum.max(
            function
                .parameters
                .iter()
                .filter_map(|parameter| parameter.value.as_ref())
                .map(|value| value.id)
                .max()
                .unwrap_or(0),
        );
        if let Some(body) = &function.body {
            maximum = maximum.max(maximum_block_value_id(body));
        }
    }
    maximum
}

fn maximum_block_value_id(block: &wire::BlockProto) -> u64 {
    let arguments = block
        .arguments
        .iter()
        .map(|value| value.id)
        .max()
        .unwrap_or(0);
    let mut maximum = arguments;
    for operation in &block.operations {
        maximum = maximum.max(
            operation
                .results
                .iter()
                .map(|value| value.id)
                .max()
                .unwrap_or(0),
        );
        for region in &operation.regions {
            for nested in &region.blocks {
                maximum = maximum.max(maximum_block_value_id(nested));
            }
        }
    }
    maximum
}
