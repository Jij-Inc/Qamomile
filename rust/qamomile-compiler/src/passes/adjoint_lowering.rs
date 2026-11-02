//! Transient target lowering for callable adjoints.
//!
//! Persistent `QCore` retains `qcore.adjoint_call` as an abstract semantic
//! operation. Targets without callable inversion receive a separately typed,
//! locally rechecked artifact with shared typed inverse functions. Calls retain
//! formal availability stages, dependent types, and recursive graph structure.

mod helpers;
mod returns;

use helpers::{adjoint_dependency_closure, adjoint_helper_symbols, build_adjoint_helper};

use std::collections::{BTreeSet, HashMap, HashSet};

use thiserror::Error;

use crate::cleanup_synthesis::{
    RegionRebase, adjoint_for_attributes, adjoint_if_attributes, freshen_region, maximum_value_id,
    structured_classical_identity_results,
};
use crate::operations::is_primitive_adjointable;
use crate::wire;
use crate::{
    CallableContract, CompilerError, LocallyCheckedProgram, ProgramHeader,
    StandardOptimizationError, TARGET_ADJOINT_KIND, VerifiedProgram, seal_package,
};

const MAX_TARGET_OPERATIONS: usize = 1_000_000;
const MAX_ADJOINT_REGION_DEPTH: usize = 256;

/// Failure to derive one transient target-level adjoint expansion.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum AdjointLoweringError {
    /// Exact standard optimization failed before target-specific expansion.
    #[error(transparent)]
    Optimization(#[from] StandardOptimizationError),
    /// The persistent input or transient result failed a compiler boundary.
    #[error("{0}")]
    Compiler(#[from] CompilerError),
    /// The callable uses a reversible construct not yet migrated to this pass.
    #[error("Q7601: native target adjoint lowering does not support {0}")]
    Unsupported(String),
    /// A locally verified adjoint ABI cannot be reconstructed.
    #[error("Q7602: malformed verified adjoint lowering: {0}")]
    Malformed(&'static str),
    /// Fresh SSA identifiers cannot be represented safely.
    #[error("Q4007: target adjoint lowering exhausted the SSA identifier space")]
    IdentifierCapacityExceeded,
    /// Recursive or expanded target IR exceeded a compiler work budget.
    #[error("Q4007: target adjoint lowering exceeded the {0}")]
    ResourceLimitExceeded(&'static str),
}

/// Locally rechecked transient program produced only for target emission.
#[derive(Clone, Debug)]
pub struct TargetLoweredProgram {
    local: LocallyCheckedProgram,
}

impl TargetLoweredProgram {
    /// Decode and locally recheck one canonical target-adjoint artifact.
    ///
    /// # Errors
    ///
    /// Returns [`CompilerError`] when the payload is malformed, uses another
    /// artifact kind, or violates a local compiler invariant.
    pub fn from_bytes(payload: &[u8]) -> Result<Self, CompilerError> {
        Ok(Self {
            local: LocallyCheckedProgram::from_target_adjoint_bytes(payload)?,
        })
    }

    /// Return immutable target-artifact metadata.
    #[must_use]
    pub fn header(&self) -> &ProgramHeader {
        self.local.header()
    }

    /// Return cleanup obligations retained in the transient artifact.
    #[must_use]
    pub fn pending_cleanup_obligation_count(&self) -> usize {
        self.local.unverified_proof_obligation_count()
    }

    /// Iterate over every compiler-derived callable contract.
    pub fn callable_contracts(&self) -> impl Iterator<Item = (&str, CallableContract)> + '_ {
        self.local.callable_contracts()
    }

    /// Return the canonical transient artifact bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> &[u8] {
        self.local.canonical_bytes()
    }
}

/// Derive shared typed inverse functions for a target.
///
/// The returned artifact uses the distinct target-adjoint boundary and cannot
/// be confused with either a specialization request or a persistent executable
/// proof. Reversible conditionals and finite `for` loops preserve their abstract
/// regions; self and mutual recursion retain call edges. Unsupported structured
/// control still fails closed.
///
/// # Errors
///
/// Returns [`AdjointLoweringError`] when the source is invalid, the callable is
/// outside the migrated subset, or the transformed artifact fails local
/// compiler rechecking.
pub fn lower_adjoint_calls_for_target(
    payload: &[u8],
) -> Result<TargetLoweredProgram, AdjointLoweringError> {
    let program = VerifiedProgram::from_bytes(payload)?;
    lower_adjoint_calls(&program)
}

pub(crate) fn lower_adjoint_calls(
    program: &VerifiedProgram,
) -> Result<TargetLoweredProgram, AdjointLoweringError> {
    let source = &program.local.checked.package;
    let functions = source
        .functions
        .iter()
        .map(|function| (function.symbol.as_str(), function))
        .collect::<HashMap<_, _>>();
    let adjoint_helpers = adjoint_helper_symbols(source)?;
    let mut required_helpers = BTreeSet::new();
    let mut next_id = maximum_value_id(source)
        .checked_add(1)
        .ok_or(AdjointLoweringError::IdentifierCapacityExceeded)?;
    let mut budget = LoweringBudget::default();
    let mut package = source.clone();
    for function in &mut package.functions {
        if let Some(body) = function.body.as_ref() {
            function.body = Some(lower_block(
                body,
                &function.symbol,
                &functions,
                &adjoint_helpers,
                &mut required_helpers,
                program,
                &mut next_id,
                &mut budget,
            )?);
        }
    }
    // Normalize only definitions needed to derive inverses. Forward definitions
    // keep their original control flow in the emitted package.
    let closure = adjoint_dependency_closure(&required_helpers, &functions)?;
    let normalized = returns::normalize(program, &closure)?;
    let program = &normalized;
    let functions = program
        .local
        .checked
        .package
        .functions
        .iter()
        .map(|function| (function.symbol.as_str(), function))
        .collect::<HashMap<_, _>>();
    let mut generated_helpers = HashSet::new();
    while let Some(callee) = required_helpers.pop_first() {
        if !generated_helpers.insert(callee.clone()) {
            continue;
        }
        package.functions.push(build_adjoint_helper(
            &callee,
            &adjoint_helpers,
            &mut required_helpers,
            &functions,
            program,
            &mut next_id,
            &mut budget,
        )?);
    }
    package
        .functions
        .sort_by(|left, right| left.symbol.cmp(&right.symbol));
    TARGET_ADJOINT_KIND.clone_into(&mut package.artifact_kind);
    let payload = seal_package(package)?;
    Ok(TargetLoweredProgram::from_bytes(&payload)?)
}

/// Optimize a verified program before target-level callable-adjoint expansion.
///
/// # Errors
///
/// Returns [`AdjointLoweringError`] when standard optimization, source
/// verification, callable expansion, or transient local rechecking fails.
pub fn lower_optimized_adjoint_calls_for_target(
    payload: &[u8],
) -> Result<TargetLoweredProgram, AdjointLoweringError> {
    let program = VerifiedProgram::from_bytes(payload)?;
    let (prepared, _) = program.prepare_standard()?;
    lower_adjoint_calls(prepared.as_verified())
}

#[allow(clippy::too_many_arguments)]
fn lower_block(
    source: &wire::BlockProto,
    owner: &str,
    functions: &HashMap<&str, &wire::FunctionProto>,
    adjoint_helpers: &HashMap<String, String>,
    required_helpers: &mut BTreeSet<String>,
    program: &VerifiedProgram,
    next_id: &mut u64,
    budget: &mut LoweringBudget,
) -> Result<wire::BlockProto, AdjointLoweringError> {
    let mut operations = Vec::new();
    for source_operation in &source.operations {
        if source_operation.opcode == "qcore.adjoint"
            && matches!(
                string_attribute(source_operation, "original"),
                Some("qcore.control_call" | "qcore.select_call")
            )
        {
            operations.extend(materialize_coherent_adjoint(
                source_operation,
                functions,
                adjoint_helpers,
                required_helpers,
                next_id,
                budget,
            )?);
            continue;
        }
        if matches!(source_operation.opcode.as_str(), "qcore.if" | "qcore.for")
            && string_attribute(source_operation, "body_mode") == Some("adjoint")
        {
            operations.push(materialize_structured_body_mode(
                source_operation,
                owner,
                functions,
                adjoint_helpers,
                required_helpers,
                program,
                next_id,
                budget,
            )?);
            continue;
        }
        let mut operation = source_operation.clone();
        for region in &mut operation.regions {
            for block in &mut region.blocks {
                *block = lower_block(
                    block,
                    owner,
                    functions,
                    adjoint_helpers,
                    required_helpers,
                    program,
                    next_id,
                    budget,
                )?;
            }
        }
        if operation.opcode == "qcore.adjoint_call" {
            budget.charge_operation()?;
            operations.push(shared_adjoint_call(
                &operation,
                adjoint_helpers,
                required_helpers,
            )?);
        } else {
            budget.charge_operation()?;
            operations.push(operation);
        }
    }
    Ok(wire::BlockProto {
        arguments: source.arguments.clone(),
        operations,
        terminator: source.terminator.clone(),
    })
}

#[allow(clippy::too_many_arguments)]
fn derive_adjoint_body(
    marker: &wire::OperationProto,
    functions: &HashMap<&str, &wire::FunctionProto>,
    adjoint_helpers: &HashMap<String, String>,
    required_helpers: &mut BTreeSet<String>,
    program: &VerifiedProgram,
    next_id: &mut u64,
    budget: &mut LoweringBudget,
) -> Result<Vec<wire::OperationProto>, AdjointLoweringError> {
    let callee_name = string_attribute(marker, "callee")
        .ok_or(AdjointLoweringError::Malformed("missing adjoint callee"))?;
    let callee = functions
        .get(callee_name)
        .copied()
        .ok_or(AdjointLoweringError::Malformed("unresolved adjoint callee"))?;
    let arena = &program.local.checked.type_arena;
    let value_types = program
        .local
        .checked
        .operations
        .value_types(callee_name)
        .ok_or(AdjointLoweringError::Malformed(
            "missing callable value types",
        ))?;
    let linear_parameters = callee
        .parameters
        .iter()
        .filter(|parameter| {
            parameter
                .value
                .as_ref()
                .and_then(|value| value_types.get(&value.id))
                .is_some_and(|type_id| arena.is_quantum(*type_id))
        })
        .collect::<Vec<_>>();
    if linear_parameters.is_empty() || marker.operands.is_empty() || marker.results.len() != 1 {
        return Err(AdjointLoweringError::Unsupported(format!(
            "non-owned callable {callee_name:?}"
        )));
    }
    let linear_parameters = linear_parameters
        .into_iter()
        .map(|parameter| {
            parameter
                .value
                .as_ref()
                .ok_or(AdjointLoweringError::Malformed("missing linear parameter"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let body = callee
        .body
        .as_ref()
        .ok_or_else(|| AdjointLoweringError::Unsupported(callee_name.to_owned()))?;
    let mut mapping = classical_argument_mapping(callee, &linear_parameters, marker)?;
    let CallableTrace { mut prelude, steps } = trace_callable_body(
        callee_name,
        body,
        value_types,
        arena,
        &mut mapping,
        next_id,
        budget,
    )?;
    let returned = returned_quantum_value(body, value_types, arena)?;
    if steps.is_empty() && linear_parameters.len() == 1 && returned == linear_parameters[0].id {
        return emit_identity_adjoint(marker, prelude, budget);
    }
    let definitions = callable_definitions(callee);
    let (inverse_outputs, repack) = inverse_outputs(marker, &linear_parameters, next_id)?;
    let inverse = invert_trace(
        steps,
        returned,
        marker,
        &linear_parameters,
        &inverse_outputs,
        &definitions,
        &mapping,
        functions,
        adjoint_helpers,
        required_helpers,
        program,
        next_id,
        budget,
        0,
        callee_name,
    )?;
    prelude.extend(inverse);
    if let Some(repack) = repack {
        budget.charge_operation()?;
        prelude.push(repack);
    }
    Ok(prelude)
}

fn emit_identity_adjoint(
    marker: &wire::OperationProto,
    mut prelude: Vec<wire::OperationProto>,
    budget: &mut LoweringBudget,
) -> Result<Vec<wire::OperationProto>, AdjointLoweringError> {
    budget.charge_operation()?;
    prelude.push(wire::OperationProto {
        opcode: "qcore.reshape".to_owned(),
        operands: vec![marker.operands[0]],
        results: marker.results.clone(),
        attributes: string_attributes(&[("proof", "modular-adjoint")]),
        span: marker.span.clone(),
        ..Default::default()
    });
    Ok(prelude)
}

#[allow(clippy::too_many_arguments)]
fn invert_trace(
    steps: Vec<AdjointStep<'_>>,
    returned: u64,
    marker: &wire::OperationProto,
    linear_parameters: &[&wire::ValueProto],
    inverse_outputs: &[wire::ValueProto],
    definitions: &HashMap<u64, wire::ValueProto>,
    mapping: &HashMap<u64, u64>,
    functions: &HashMap<&str, &wire::FunctionProto>,
    adjoint_helpers: &HashMap<String, String>,
    required_helpers: &mut BTreeSet<String>,
    program: &VerifiedProgram,
    next_id: &mut u64,
    budget: &mut LoweringBudget,
    depth: usize,
    callee_name: &str,
) -> Result<Vec<wire::OperationProto>, AdjointLoweringError> {
    let output_by_input = linear_parameters
        .iter()
        .zip(inverse_outputs)
        .map(|(parameter, output)| (parameter.id, output.clone()))
        .collect::<HashMap<_, _>>();
    let mut available = HashMap::from([(returned, marker.operands[0])]);
    let mut inverse = Vec::new();
    let region_pairs = boundary_region_pairs(
        linear_parameters,
        inverse_outputs,
        definitions,
        returned,
        marker.results[0].region.as_ref(),
    )?;
    for step in steps.into_iter().rev() {
        let current = step
            .linear_results
            .iter()
            .map(|result| {
                available.remove(result).ok_or_else(|| {
                    AdjointLoweringError::Unsupported(format!(
                        "disconnected diagram in {callee_name:?}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let current_regions = step
            .linear_results
            .iter()
            .map(|result| {
                rebase_region(
                    definitions
                        .get(result)
                        .and_then(|value| value.region.as_ref()),
                    &region_pairs,
                    callee_name,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let restored = step
            .linear_operands
            .iter()
            .map(|operand| {
                if let Some(output) = output_by_input.get(operand) {
                    return Ok(output.clone());
                }
                let source = definitions
                    .get(operand)
                    .ok_or(AdjointLoweringError::Malformed("missing adjoint operand"))?;
                let region = rebase_region(source.region.as_ref(), &region_pairs, callee_name)?;
                fresh_value(source, region, next_id)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let classical = step
            .operation
            .operands
            .iter()
            .filter(|identifier| !step.linear_operands.contains(identifier))
            .map(|identifier| mapping.get(identifier).copied())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                AdjointLoweringError::Unsupported(format!("open gate operands in {callee_name:?}"))
            })?;
        let structured_rebases =
            structured_step_rebases(&step, &restored, definitions, &region_pairs, callee_name)?;
        inverse.extend(invert_step(
            &step,
            current,
            &current_regions,
            restored.clone(),
            classical,
            &structured_rebases,
            functions,
            adjoint_helpers,
            required_helpers,
            program,
            next_id,
            budget,
            depth,
            callee_name,
        )?);
        available.extend(
            step.linear_operands
                .iter()
                .copied()
                .zip(restored.iter().map(|value| value.id)),
        );
    }
    if output_by_input
        .iter()
        .any(|(input, output)| available.get(input) != Some(&output.id))
    {
        return Err(AdjointLoweringError::Unsupported(format!(
            "incomplete diagram in {callee_name:?}"
        )));
    }
    Ok(inverse)
}

fn structured_step_rebases(
    step: &AdjointStep<'_>,
    restored: &[wire::ValueProto],
    definitions: &HashMap<u64, wire::ValueProto>,
    region_pairs: &[(wire::RegionRefProto, wire::RegionRefProto)],
    callee_name: &str,
) -> Result<Vec<Option<RegionRebase>>, AdjointLoweringError> {
    if !matches!(step.operation.opcode.as_str(), "qcore.if" | "qcore.for") {
        return Ok(Vec::new());
    }
    let [linear_operand] = step.linear_operands.as_slice() else {
        return Err(AdjointLoweringError::Malformed("structured linear operand"));
    };
    let [linear_result] = step.linear_results.as_slice() else {
        return Err(AdjointLoweringError::Malformed("structured linear result"));
    };
    let source = definitions
        .get(linear_operand)
        .and_then(|value| value.region.as_ref())
        .ok_or(AdjointLoweringError::Malformed(
            "structured input ownership region",
        ))?;
    let target = rebase_region(
        definitions
            .get(linear_result)
            .and_then(|value| value.region.as_ref()),
        region_pairs,
        callee_name,
    )?
    .ok_or(AdjointLoweringError::Malformed(
        "structured output ownership region",
    ))?;
    let body_rebase = RegionRebase::new(Some(source), Some(&target));
    if step.operation.opcode == "qcore.if" {
        return Ok(vec![body_rebase; step.operation.regions.len()]);
    }
    let restored = restored
        .first()
        .and_then(|value| value.region.as_ref())
        .ok_or(AdjointLoweringError::Malformed(
            "structured inverse ownership region",
        ))?;
    let result = definitions
        .get(linear_result)
        .and_then(|value| value.region.as_ref())
        .ok_or(AdjointLoweringError::Malformed(
            "structured result ownership region",
        ))?;
    Ok(vec![
        body_rebase,
        RegionRebase::new(Some(result), Some(restored)),
    ])
}

fn classical_argument_mapping(
    callee: &wire::FunctionProto,
    linear_parameters: &[&wire::ValueProto],
    marker: &wire::OperationProto,
) -> Result<HashMap<u64, u64>, AdjointLoweringError> {
    let linear_parameters = linear_parameters
        .iter()
        .map(|value| value.id)
        .collect::<std::collections::HashSet<_>>();
    let classical_parameters = callee.parameters.iter().filter(|parameter| {
        parameter
            .value
            .as_ref()
            .is_some_and(|value| !linear_parameters.contains(&value.id))
    });
    let mapping = classical_parameters
        .zip(&marker.operands[1..])
        .filter_map(|(parameter, operand)| {
            parameter.value.as_ref().map(|value| (value.id, *operand))
        })
        .collect::<HashMap<_, _>>();
    if mapping.len() + 1 != marker.operands.len() {
        return Err(AdjointLoweringError::Malformed(
            "adjoint classical argument arity",
        ));
    }
    Ok(mapping)
}

fn inverse_outputs(
    marker: &wire::OperationProto,
    parameters: &[&wire::ValueProto],
    next_id: &mut u64,
) -> Result<(Vec<wire::ValueProto>, Option<wire::OperationProto>), AdjointLoweringError> {
    if parameters.len() == 1 {
        return Ok((marker.results.clone(), None));
    }
    let parent = marker.results[0]
        .region
        .as_ref()
        .ok_or(AdjointLoweringError::Malformed(
            "packed adjoint result region",
        ))?;
    let result_types = marker.results[0]
        .r#type
        .as_ref()
        .filter(|result_type| {
            result_type.kind == wire::type_proto::Kind::Product as i32
                && result_type.arguments.len() == parameters.len()
        })
        .ok_or(AdjointLoweringError::Malformed(
            "packed adjoint result type",
        ))?;
    let outputs = parameters
        .iter()
        .enumerate()
        .map(|(index, parameter)| {
            let mut region = parent.clone();
            region.projections.push(index.to_string());
            let mut output = fresh_value(parameter, Some(region), next_id)?;
            output.r#type = Some(result_types.arguments[index].clone());
            Ok::<_, AdjointLoweringError>(output)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let repack = wire::OperationProto {
        opcode: "qcore.tuple".to_owned(),
        operands: outputs.iter().map(|value| value.id).collect(),
        results: marker.results.clone(),
        attributes: string_attributes(&[("proof", "modular-adjoint")]),
        span: marker.span.clone(),
        ..Default::default()
    };
    Ok((outputs, Some(repack)))
}

struct CallableTrace<'a> {
    prelude: Vec<wire::OperationProto>,
    steps: Vec<AdjointStep<'a>>,
}

struct AdjointStep<'a> {
    operation: &'a wire::OperationProto,
    linear_operands: Vec<u64>,
    linear_results: Vec<u64>,
}

fn trace_callable_body<'a>(
    callee_name: &str,
    body: &'a wire::BlockProto,
    value_types: &HashMap<u64, crate::types::TypeId>,
    arena: &crate::types::TypeArena,
    mapping: &mut HashMap<u64, u64>,
    next_id: &mut u64,
    budget: &mut LoweringBudget,
) -> Result<CallableTrace<'a>, AdjointLoweringError> {
    let mut prelude = Vec::new();
    let mut steps = Vec::new();
    let required_values = block_required_values(body);
    for operation in &body.operations {
        let linear_operands = quantum_operands(operation, value_types, arena);
        let linear_results = quantum_results(operation, value_types, arena);
        if linear_operands.is_empty() && linear_results.is_empty() {
            if !operation.regions.is_empty() {
                return Err(AdjointLoweringError::Unsupported(format!(
                    "structured classical prelude in {callee_name:?}"
                )));
            }
            let operands = operation
                .operands
                .iter()
                .map(|operand| mapping.get(operand).copied())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    AdjointLoweringError::Unsupported(format!(
                        "open classical prelude in {callee_name:?}"
                    ))
                })?;
            let results = operation
                .results
                .iter()
                .map(|result| fresh_value(result, result.region.clone(), next_id))
                .collect::<Result<Vec<_>, _>>()?;
            mapping.extend(
                operation
                    .results
                    .iter()
                    .zip(&results)
                    .map(|(left, right)| (left.id, right.id)),
            );
            let mut replay = operation.clone();
            replay.operands = operands;
            replay.results = results;
            budget.charge_operation()?;
            prelude.push(replay);
            continue;
        }
        if !(is_primitive_adjointable(&operation.opcode)
            || matches!(
                operation.opcode.as_str(),
                "qcore.adjoint"
                    | "qcore.reshape"
                    | "qcore.tuple"
                    | "qcore.unpack"
                    | "qcore.call"
                    | "qcore.adjoint_call"
                    | "qcore.control_call"
                    | "qcore.select_call"
                    | "qcore.if"
                    | "qcore.for"
            ))
        {
            return Err(AdjointLoweringError::Unsupported(format!(
                "{} in {callee_name:?}",
                operation.opcode
            )));
        }
        validate_step_shape(operation, &linear_operands, &linear_results, callee_name)?;
        match operation.opcode.as_str() {
            "qcore.if" => {
                validate_if_regions(operation, callee_name)?;
                map_identity_structured_results(
                    operation,
                    1,
                    "conditional",
                    value_types,
                    arena,
                    mapping,
                    &required_values,
                    callee_name,
                )?;
            }
            "qcore.for" => {
                validate_for_regions(operation, callee_name)?;
                map_identity_structured_results(
                    operation,
                    3,
                    "loop",
                    value_types,
                    arena,
                    mapping,
                    &required_values,
                    callee_name,
                )?;
            }
            _ => {}
        }
        steps.push(AdjointStep {
            operation,
            linear_operands,
            linear_results,
        });
    }
    Ok(CallableTrace { prelude, steps })
}

fn block_required_values(body: &wire::BlockProto) -> HashSet<u64> {
    body.operations
        .iter()
        .flat_map(|operation| operation.operands.iter().copied())
        .chain(
            body.terminator
                .iter()
                .flat_map(|terminator| terminator.operands.iter().copied()),
        )
        .collect()
}

fn validate_if_regions(
    operation: &wire::OperationProto,
    callee_name: &str,
) -> Result<(), AdjointLoweringError> {
    if operation.regions.len() != 2
        || operation.regions.iter().any(|region| {
            let [block] = region.blocks.as_slice() else {
                return true;
            };
            block
                .terminator
                .as_ref()
                .is_none_or(|terminator| terminator.kind != "yield")
        })
    {
        return Err(AdjointLoweringError::Unsupported(format!(
            "malformed or exiting conditional in {callee_name:?}"
        )));
    }
    Ok(())
}

fn validate_for_regions(
    operation: &wire::OperationProto,
    callee_name: &str,
) -> Result<(), AdjointLoweringError> {
    let [body_region, else_region] = operation.regions.as_slice() else {
        return unsupported_loop(callee_name);
    };
    let ([body], [otherwise]) = (body_region.blocks.as_slice(), else_region.blocks.as_slice())
    else {
        return unsupported_loop(callee_name);
    };
    if body
        .terminator
        .as_ref()
        .is_none_or(|terminator| terminator.kind != "yield")
        || otherwise
            .terminator
            .as_ref()
            .is_none_or(|terminator| terminator.kind != "yield")
        || !otherwise.operations.is_empty()
    {
        return unsupported_loop(callee_name);
    }
    Ok(())
}

fn unsupported_loop(callee_name: &str) -> Result<(), AdjointLoweringError> {
    Err(AdjointLoweringError::Unsupported(format!(
        "malformed, exiting, or nonidentity loop in {callee_name:?}"
    )))
}

#[allow(clippy::too_many_arguments)]
fn map_identity_structured_results(
    operation: &wire::OperationProto,
    operand_offset: usize,
    construct: &str,
    value_types: &HashMap<u64, crate::types::TypeId>,
    arena: &crate::types::TypeArena,
    mapping: &mut HashMap<u64, u64>,
    required_values: &HashSet<u64>,
    callee_name: &str,
) -> Result<(), AdjointLoweringError> {
    let identity_results = structured_classical_identity_results(operation, value_types);
    for (index, result) in operation.results.iter().enumerate() {
        if value_types
            .get(&result.id)
            .is_some_and(|type_id| arena.is_quantum(*type_id))
            || !required_values.contains(&result.id)
        {
            continue;
        }
        if !identity_results.get(index).copied().unwrap_or(false) {
            return Err(AdjointLoweringError::Unsupported(format!(
                "{construct} classical carry in {callee_name:?}"
            )));
        }
        let capture = operation.operands.get(index + operand_offset).ok_or(
            AdjointLoweringError::Malformed("structured classical capture"),
        )?;
        let mapped = mapping.get(capture).copied().ok_or_else(|| {
            AdjointLoweringError::Unsupported(format!("open structured capture in {callee_name:?}"))
        })?;
        mapping.insert(result.id, mapped);
    }
    Ok(())
}

fn validate_step_shape(
    operation: &wire::OperationProto,
    linear_operands: &[u64],
    linear_results: &[u64],
    callee_name: &str,
) -> Result<(), AdjointLoweringError> {
    let valid = match operation.opcode.as_str() {
        "qcore.tuple" => {
            !linear_operands.is_empty()
                && linear_results.len() == 1
                && linear_operands.len() == operation.operands.len()
                && linear_results.len() == operation.results.len()
        }
        "qcore.unpack" => {
            linear_operands.len() == 1
                && !linear_results.is_empty()
                && linear_operands.len() == operation.operands.len()
                && linear_results.len() == operation.results.len()
        }
        "qcore.call" => !linear_operands.is_empty() && linear_results.len() == 1,
        "qcore.control_call" | "qcore.select_call" => {
            linear_operands.len() == 2 && linear_results.len() == 1
        }
        _ => linear_operands.len() == 1 && linear_results.len() == 1,
    };
    if !valid {
        return Err(AdjointLoweringError::Unsupported(format!(
            "branched {} diagram in {callee_name:?}",
            operation.opcode
        )));
    }
    Ok(())
}

fn boundary_region_pairs(
    parameters: &[&wire::ValueProto],
    outputs: &[wire::ValueProto],
    definitions: &HashMap<u64, wire::ValueProto>,
    returned: u64,
    inverse_input: Option<&wire::RegionRefProto>,
) -> Result<Vec<(wire::RegionRefProto, wire::RegionRefProto)>, AdjointLoweringError> {
    let mut pairs = parameters
        .iter()
        .zip(outputs)
        .map(|(parameter, output)| -> Result<_, AdjointLoweringError> {
            Ok((
                parameter
                    .region
                    .clone()
                    .ok_or(AdjointLoweringError::Malformed("linear parameter region"))?,
                output
                    .region
                    .clone()
                    .ok_or(AdjointLoweringError::Malformed("inverse output region"))?,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    pairs.push((
        definitions
            .get(&returned)
            .and_then(|value| value.region.clone())
            .ok_or(AdjointLoweringError::Malformed("callable result region"))?,
        inverse_input
            .cloned()
            .ok_or(AdjointLoweringError::Malformed("adjoint input region"))?,
    ));
    Ok(pairs)
}

fn rebase_region(
    region: Option<&wire::RegionRefProto>,
    pairs: &[(wire::RegionRefProto, wire::RegionRefProto)],
    callee_name: &str,
) -> Result<Option<wire::RegionRefProto>, AdjointLoweringError> {
    let Some(region) = region else {
        return Ok(None);
    };
    let (source, target) = pairs
        .iter()
        .filter(|(source, _)| {
            source.root == region.root
                && region
                    .projections
                    .starts_with(source.projections.as_slice())
        })
        .max_by_key(|(source, _)| source.projections.len())
        .ok_or_else(|| {
            AdjointLoweringError::Unsupported(format!(
                "foreign ownership region in {callee_name:?}"
            ))
        })?;
    let mut rebased = target.clone();
    rebased
        .projections
        .extend_from_slice(&region.projections[source.projections.len()..]);
    Ok(Some(rebased))
}

fn returned_quantum_value(
    body: &wire::BlockProto,
    value_types: &HashMap<u64, crate::types::TypeId>,
    arena: &crate::types::TypeArena,
) -> Result<u64, AdjointLoweringError> {
    body.terminator
        .as_ref()
        .filter(|terminator| terminator.kind == "return")
        .and_then(|terminator| {
            terminator.operands.iter().copied().find(|identifier| {
                value_types
                    .get(identifier)
                    .is_some_and(|type_id| arena.is_quantum(*type_id))
            })
        })
        .ok_or(AdjointLoweringError::Malformed("missing owned return"))
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn invert_step(
    step: &AdjointStep<'_>,
    current: Vec<u64>,
    current_regions: &[Option<wire::RegionRefProto>],
    restored: Vec<wire::ValueProto>,
    classical: Vec<u64>,
    structured_rebases: &[Option<RegionRebase>],
    functions: &HashMap<&str, &wire::FunctionProto>,
    adjoint_helpers: &HashMap<String, String>,
    required_helpers: &mut BTreeSet<String>,
    program: &VerifiedProgram,
    next_id: &mut u64,
    budget: &mut LoweringBudget,
    depth: usize,
    owner: &str,
) -> Result<Vec<wire::OperationProto>, AdjointLoweringError> {
    let operation = step.operation;
    if matches!(operation.opcode.as_str(), "qcore.if" | "qcore.for") {
        return invert_structured_step(
            step,
            &current,
            &restored,
            classical,
            structured_rebases,
            functions,
            adjoint_helpers,
            required_helpers,
            program,
            next_id,
            budget,
            depth,
            owner,
        );
    }
    if operation.opcode == "qcore.tuple" {
        let [current] = current.as_slice() else {
            return Err(AdjointLoweringError::Malformed("tuple inverse results"));
        };
        budget.charge_operation()?;
        return Ok(vec![wire::OperationProto {
            opcode: "qcore.unpack".to_owned(),
            operands: vec![*current],
            results: restored,
            attributes: string_attributes(&[("proof", "modular-adjoint")]),
            span: operation.span.clone(),
            ..Default::default()
        }]);
    }
    if operation.opcode == "qcore.unpack" {
        let [restored] = restored.as_slice() else {
            return Err(AdjointLoweringError::Malformed("unpack inverse operand"));
        };
        budget.charge_operation()?;
        return Ok(vec![wire::OperationProto {
            opcode: "qcore.tuple".to_owned(),
            operands: current,
            results: vec![restored.clone()],
            attributes: string_attributes(&[("proof", "modular-adjoint")]),
            span: operation.span.clone(),
            ..Default::default()
        }]);
    }
    if matches!(
        operation.opcode.as_str(),
        "qcore.control_call" | "qcore.select_call"
    ) {
        return invert_coherent_step(
            operation,
            &step.linear_operands,
            &current,
            current_regions,
            &restored,
            classical,
            adjoint_helpers,
            required_helpers,
            next_id,
            budget,
        );
    }
    if operation.opcode == "qcore.call" {
        return invert_call_step(
            operation,
            &current,
            &restored,
            classical,
            adjoint_helpers,
            required_helpers,
            next_id,
            budget,
        );
    }
    let ([current], [restored]) = (current.as_slice(), restored.as_slice()) else {
        return Err(AdjointLoweringError::Malformed(
            "non-structural inverse arity",
        ));
    };
    if operation.opcode == "qcore.adjoint_call" {
        budget.charge_operation()?;
        return Ok(vec![restore_direct_call(
            operation,
            *current,
            restored.clone(),
            classical,
            functions,
            program,
        )?]);
    }
    budget.charge_operation()?;
    Ok(vec![invert_operation(
        operation,
        *current,
        restored.clone(),
        classical,
    )?])
}

#[allow(clippy::too_many_arguments)]
fn invert_call_step(
    operation: &wire::OperationProto,
    current: &[u64],
    restored: &[wire::ValueProto],
    classical: Vec<u64>,
    adjoint_helpers: &HashMap<String, String>,
    required_helpers: &mut BTreeSet<String>,
    next_id: &mut u64,
    budget: &mut LoweringBudget,
) -> Result<Vec<wire::OperationProto>, AdjointLoweringError> {
    let [current] = current else {
        return Err(AdjointLoweringError::Malformed("nested call result"));
    };
    let callee = string_attribute(operation, "callee")
        .ok_or(AdjointLoweringError::Malformed("nested call callee"))?;
    let (result, unpack) = if let [restored] = restored {
        (restored.clone(), None)
    } else {
        let source = operation
            .results
            .first()
            .ok_or(AdjointLoweringError::Malformed("nested call packed result"))?;
        let mut packed = fresh_value(source, source.region.clone(), next_id)?;
        "nested_adjoint_result".clone_into(&mut packed.hint);
        let unpack = wire::OperationProto {
            opcode: "qcore.unpack".to_owned(),
            operands: vec![packed.id],
            results: restored.to_vec(),
            attributes: string_attributes(&[("proof", "modular-adjoint")]),
            span: operation.span.clone(),
            ..Default::default()
        };
        (packed, Some(unpack))
    };
    let marker = wire::OperationProto {
        opcode: "qcore.adjoint_call".to_owned(),
        operands: std::iter::once(*current).chain(classical).collect(),
        results: vec![result],
        attributes: string_attributes(&[("callee", callee), ("proof", "modular-adjoint")]),
        span: operation.span.clone(),
        ..Default::default()
    };
    budget.charge_operation()?;
    let mut inverse = vec![shared_adjoint_call(
        &marker,
        adjoint_helpers,
        required_helpers,
    )?];
    if let Some(unpack) = unpack {
        budget.charge_operation()?;
        inverse.push(unpack);
    }
    Ok(inverse)
}

#[allow(clippy::too_many_arguments)]
fn invert_coherent_step(
    operation: &wire::OperationProto,
    linear_operands: &[u64],
    current: &[u64],
    current_regions: &[Option<wire::RegionRefProto>],
    restored: &[wire::ValueProto],
    classical: Vec<u64>,
    adjoint_helpers: &HashMap<String, String>,
    required_helpers: &mut BTreeSet<String>,
    next_id: &mut u64,
    budget: &mut LoweringBudget,
) -> Result<Vec<wire::OperationProto>, AdjointLoweringError> {
    let ([current], [current_region]) = (current, current_regions) else {
        return Err(AdjointLoweringError::Malformed(
            "coherent inverse packed input",
        ));
    };
    if restored.len() != linear_operands.len() {
        return Err(AdjointLoweringError::Malformed(
            "coherent inverse restored arity",
        ));
    }
    let temporary = restored
        .iter()
        .enumerate()
        .map(|(index, restored)| {
            let mut region = current_region
                .clone()
                .ok_or(AdjointLoweringError::Malformed(
                    "coherent inverse ownership region",
                ))?;
            region.projections.push(index.to_string());
            fresh_value(restored, Some(region), next_id)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut classical = classical.into_iter();
    let mut linear = temporary.iter();
    let operands = operation
        .operands
        .iter()
        .map(|operand| {
            if linear_operands.contains(operand) {
                linear
                    .next()
                    .map(|value| value.id)
                    .ok_or(AdjointLoweringError::Malformed(
                        "coherent inverse linear operand",
                    ))
            } else {
                classical.next().ok_or(AdjointLoweringError::Malformed(
                    "coherent inverse classical operand",
                ))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    if linear.next().is_some() || classical.next().is_some() {
        return Err(AdjointLoweringError::Malformed(
            "coherent inverse operand arity",
        ));
    }
    let attributes = coherent_adjoint_attributes(operation, adjoint_helpers, required_helpers)?;
    let source_result = operation
        .results
        .first()
        .ok_or(AdjointLoweringError::Malformed("coherent inverse result"))?;
    let coherent_result = fresh_value(source_result, current_region.clone(), next_id)?;
    budget.charge_operation()?;
    budget.charge_operation()?;
    budget.charge_operation()?;
    Ok(vec![
        wire::OperationProto {
            opcode: "qcore.unpack".to_owned(),
            operands: vec![*current],
            results: temporary,
            attributes: string_attributes(&[("proof", "modular-adjoint")]),
            span: operation.span.clone(),
            ..Default::default()
        },
        wire::OperationProto {
            opcode: operation.opcode.clone(),
            operands,
            results: vec![coherent_result.clone()],
            attributes,
            span: operation.span.clone(),
            ..Default::default()
        },
        wire::OperationProto {
            opcode: "qcore.unpack".to_owned(),
            operands: vec![coherent_result.id],
            results: restored.to_vec(),
            attributes: string_attributes(&[("proof", "modular-adjoint")]),
            span: operation.span.clone(),
            ..Default::default()
        },
    ])
}

fn coherent_adjoint_attributes(
    operation: &wire::OperationProto,
    adjoint_helpers: &HashMap<String, String>,
    required_helpers: &mut BTreeSet<String>,
) -> Result<Vec<wire::AttributeProto>, AdjointLoweringError> {
    let mut attributes = operation.attributes.clone();
    let name = if operation.opcode == "qcore.control_call" {
        "callee"
    } else {
        "callees"
    };
    let attribute = attributes
        .iter_mut()
        .find(|attribute| attribute.name == name)
        .and_then(|attribute| attribute.value.as_mut())
        .ok_or(AdjointLoweringError::Malformed(
            "coherent inverse callee attribute",
        ))?;
    if name == "callee" {
        let wire::literal_proto::Value::StringValue(callee) = attribute
            .value
            .as_mut()
            .ok_or(AdjointLoweringError::Malformed("coherent inverse callee"))?
        else {
            return Err(AdjointLoweringError::Malformed("coherent inverse callee"));
        };
        *callee = require_adjoint_helper(callee, adjoint_helpers, required_helpers)?;
        return Ok(attributes);
    }
    let wire::literal_proto::Value::TupleValue(callees) = attribute
        .value
        .as_mut()
        .ok_or(AdjointLoweringError::Malformed("coherent inverse callees"))?
    else {
        return Err(AdjointLoweringError::Malformed("coherent inverse callees"));
    };
    for value in &mut callees.values {
        let wire::literal_proto::Value::StringValue(callee) = value.value.as_mut().ok_or(
            AdjointLoweringError::Malformed("coherent inverse selected callee"),
        )?
        else {
            return Err(AdjointLoweringError::Malformed(
                "coherent inverse selected callee",
            ));
        };
        *callee = require_adjoint_helper(callee, adjoint_helpers, required_helpers)?;
    }
    Ok(attributes)
}

fn materialize_coherent_adjoint(
    operation: &wire::OperationProto,
    functions: &HashMap<&str, &wire::FunctionProto>,
    adjoint_helpers: &HashMap<String, String>,
    required_helpers: &mut BTreeSet<String>,
    next_id: &mut u64,
    budget: &mut LoweringBudget,
) -> Result<Vec<wire::OperationProto>, AdjointLoweringError> {
    let original = string_attribute(operation, "original")
        .ok_or(AdjointLoweringError::Malformed("coherent adjoint original"))?;
    let original_attributes = mapping_attribute(operation, "original_attributes").ok_or(
        AdjointLoweringError::Malformed("coherent adjoint attributes"),
    )?;
    let [result] = operation.results.as_slice() else {
        return Err(AdjointLoweringError::Malformed("coherent adjoint result"));
    };
    let temporary = unpacked_product_values(result, next_id)?;
    let synthetic = wire::OperationProto {
        opcode: original.to_owned(),
        attributes: original_attributes,
        ..Default::default()
    };
    let attributes = coherent_adjoint_attributes(&synthetic, adjoint_helpers, required_helpers)?;
    let operands =
        coherent_adjoint_operands(operation, original, &synthetic, &temporary, functions)?;
    budget.charge_operation()?;
    budget.charge_operation()?;
    Ok(vec![
        wire::OperationProto {
            opcode: "qcore.unpack".to_owned(),
            operands: operation.operands[..1].to_vec(),
            results: temporary,
            attributes: string_attributes(&[("proof", "modular-adjoint")]),
            span: operation.span.clone(),
            ..Default::default()
        },
        wire::OperationProto {
            opcode: original.to_owned(),
            operands,
            results: operation.results.clone(),
            attributes,
            span: operation.span.clone(),
            ..Default::default()
        },
    ])
}

fn unpacked_product_values(
    product: &wire::ValueProto,
    next_id: &mut u64,
) -> Result<Vec<wire::ValueProto>, AdjointLoweringError> {
    let product_type = product
        .r#type
        .as_ref()
        .filter(|r#type| r#type.kind == wire::type_proto::Kind::Product as i32)
        .ok_or(AdjointLoweringError::Malformed(
            "coherent adjoint product type",
        ))?;
    if product_type.arguments.len() != 2 {
        return Err(AdjointLoweringError::Malformed(
            "coherent adjoint product arity",
        ));
    }
    product_type
        .arguments
        .iter()
        .enumerate()
        .map(|(index, r#type)| {
            let mut region = product
                .region
                .clone()
                .ok_or(AdjointLoweringError::Malformed(
                    "coherent adjoint product region",
                ))?;
            region.projections.push(index.to_string());
            Ok(wire::ValueProto {
                id: take_id(next_id)?,
                r#type: Some(r#type.clone()),
                hint: "coherent_adjoint_input".to_owned(),
                has_region: true,
                region: Some(region),
            })
        })
        .collect()
}

fn coherent_adjoint_operands(
    operation: &wire::OperationProto,
    original: &str,
    synthetic: &wire::OperationProto,
    temporary: &[wire::ValueProto],
    functions: &HashMap<&str, &wire::FunctionProto>,
) -> Result<Vec<u64>, AdjointLoweringError> {
    let [selector, target] = temporary else {
        return Err(AdjointLoweringError::Malformed(
            "coherent adjoint unpack arity",
        ));
    };
    let tail = operation.operands.get(1..).unwrap_or_default();
    if original == "qcore.select_call" {
        return Ok(std::iter::once(selector.id)
            .chain(std::iter::once(target.id))
            .chain(tail.iter().copied())
            .collect());
    }
    let callee = string_attribute(synthetic, "callee")
        .and_then(|callee| functions.get(callee).copied())
        .ok_or(AdjointLoweringError::Malformed(
            "coherent adjoint controlled callee",
        ))?;
    if operation.operands.len() == callee.parameters.len() + 1 {
        let component = *tail.first().ok_or(AdjointLoweringError::Malformed(
            "coherent adjoint component",
        ))?;
        return Ok(std::iter::once(selector.id)
            .chain(std::iter::once(component))
            .chain(std::iter::once(target.id))
            .chain(tail[1..].iter().copied())
            .collect());
    }
    if operation.operands.len() != callee.parameters.len() {
        return Err(AdjointLoweringError::Malformed(
            "coherent adjoint operand arity",
        ));
    }
    Ok(std::iter::once(selector.id)
        .chain(std::iter::once(target.id))
        .chain(tail.iter().copied())
        .collect())
}

fn require_adjoint_helper(
    callee: &str,
    adjoint_helpers: &HashMap<String, String>,
    required_helpers: &mut BTreeSet<String>,
) -> Result<String, AdjointLoweringError> {
    let helper = adjoint_helpers.get(callee).cloned().ok_or_else(|| {
        AdjointLoweringError::Unsupported(format!("coherent callable {callee:?}"))
    })?;
    required_helpers.insert(callee.to_owned());
    Ok(helper)
}

fn invert_structured_attributes(
    operation: &wire::OperationProto,
) -> Result<Vec<wire::AttributeProto>, AdjointLoweringError> {
    if string_attribute(operation, "body_mode") == Some("adjoint") {
        return mapping_attribute(operation, "original_attributes").ok_or(
            AdjointLoweringError::Malformed("structured inverse attributes"),
        );
    }
    let attributes = if operation.opcode == "qcore.if" {
        adjoint_if_attributes(operation)
    } else {
        adjoint_for_attributes(operation)
    };
    Ok(strip_structured_adjoint_metadata(attributes))
}

fn strip_structured_adjoint_metadata(
    attributes: Vec<wire::AttributeProto>,
) -> Vec<wire::AttributeProto> {
    attributes
        .into_iter()
        .filter(|attribute| {
            !matches!(
                attribute.name.as_str(),
                "body_mode" | "original_attributes" | "proof"
            )
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn invert_structured_step(
    step: &AdjointStep<'_>,
    current: &[u64],
    restored: &[wire::ValueProto],
    classical: Vec<u64>,
    rebases: &[Option<RegionRebase>],
    functions: &HashMap<&str, &wire::FunctionProto>,
    adjoint_helpers: &HashMap<String, String>,
    required_helpers: &mut BTreeSet<String>,
    program: &VerifiedProgram,
    next_id: &mut u64,
    budget: &mut LoweringBudget,
    depth: usize,
    owner: &str,
) -> Result<Vec<wire::OperationProto>, AdjointLoweringError> {
    let operation = step.operation;
    let ([current], [restored]) = (current, restored) else {
        return Err(AdjointLoweringError::Malformed("structured inverse arity"));
    };
    if rebases.len() != operation.regions.len() {
        return Err(AdjointLoweringError::Malformed(
            "structured region rebase arity",
        ));
    }
    let mut classical = classical.into_iter();
    let operands = operation
        .operands
        .iter()
        .map(|operand| {
            if step.linear_operands.contains(operand) {
                Ok(*current)
            } else {
                classical.next().ok_or(AdjointLoweringError::Malformed(
                    "structured classical operand arity",
                ))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    if classical.next().is_some() {
        return Err(AdjointLoweringError::Malformed(
            "structured classical operand arity",
        ));
    }
    let results = operation
        .results
        .iter()
        .map(|result| {
            if step.linear_results.contains(&result.id) {
                Ok(restored.clone())
            } else {
                fresh_value(result, result.region.clone(), next_id)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let already_adjoint = string_attribute(operation, "body_mode") == Some("adjoint");
    let attributes = invert_structured_attributes(operation)?;
    let regions = if already_adjoint {
        budget.charge_operation_tree(&operation.regions)?;
        freshen_structured_regions(operation, rebases, next_id)?
    } else {
        invert_structured_regions(
            operation,
            &results,
            rebases,
            owner,
            functions,
            adjoint_helpers,
            required_helpers,
            program,
            next_id,
            budget,
            depth,
        )?
    };
    if !already_adjoint {
        budget.charge_operation()?;
    }
    Ok(vec![wire::OperationProto {
        opcode: operation.opcode.clone(),
        operands,
        results,
        attributes,
        regions,
        span: operation.span.clone(),
        intent: Vec::new(),
    }])
}

#[allow(clippy::too_many_arguments)]
fn materialize_structured_body_mode(
    operation: &wire::OperationProto,
    owner: &str,
    functions: &HashMap<&str, &wire::FunctionProto>,
    adjoint_helpers: &HashMap<String, String>,
    required_helpers: &mut BTreeSet<String>,
    program: &VerifiedProgram,
    next_id: &mut u64,
    budget: &mut LoweringBudget,
) -> Result<wire::OperationProto, AdjointLoweringError> {
    let value_types = program.local.checked.operations.value_types(owner).ok_or(
        AdjointLoweringError::Malformed("missing structured value types"),
    )?;
    let arena = &program.local.checked.type_arena;
    let linear_operands = quantum_operands(operation, value_types, arena);
    let linear_results = quantum_results(operation, value_types, arena);
    validate_step_shape(operation, &linear_operands, &linear_results, owner)?;
    let rebases = vec![None; operation.regions.len()];
    let regions = invert_structured_regions(
        operation,
        &operation.results,
        &rebases,
        owner,
        functions,
        adjoint_helpers,
        required_helpers,
        program,
        next_id,
        budget,
        0,
    )?;
    budget.charge_operation()?;
    Ok(wire::OperationProto {
        opcode: operation.opcode.clone(),
        operands: operation.operands.clone(),
        results: operation.results.clone(),
        attributes: strip_structured_adjoint_metadata(operation.attributes.clone()),
        regions,
        span: operation.span.clone(),
        intent: Vec::new(),
    })
}

#[allow(clippy::too_many_arguments)]
fn invert_structured_regions(
    operation: &wire::OperationProto,
    results: &[wire::ValueProto],
    rebases: &[Option<RegionRebase>],
    owner: &str,
    functions: &HashMap<&str, &wire::FunctionProto>,
    adjoint_helpers: &HashMap<String, String>,
    required_helpers: &mut BTreeSet<String>,
    program: &VerifiedProgram,
    next_id: &mut u64,
    budget: &mut LoweringBudget,
    depth: usize,
) -> Result<Vec<wire::RegionProto>, AdjointLoweringError> {
    if depth >= MAX_ADJOINT_REGION_DEPTH {
        return Err(AdjointLoweringError::ResourceLimitExceeded(
            "structured inverse depth",
        ));
    }
    if rebases.len() != operation.regions.len() {
        return Err(AdjointLoweringError::Malformed(
            "structured region rebase arity",
        ));
    }
    let value_types = program.local.checked.operations.value_types(owner).ok_or(
        AdjointLoweringError::Malformed("missing structured value types"),
    )?;
    let arena = &program.local.checked.type_arena;
    let linear_indices = operation
        .results
        .iter()
        .enumerate()
        .filter_map(|(index, result)| {
            value_types
                .get(&result.id)
                .is_some_and(|type_id| arena.is_quantum(*type_id))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let [linear_index] = linear_indices.as_slice() else {
        return Err(AdjointLoweringError::Malformed(
            "structured linear result index",
        ));
    };
    let result = results
        .get(*linear_index)
        .ok_or(AdjointLoweringError::Malformed(
            "structured inverse result index",
        ))?;
    if operation.opcode == "qcore.if" {
        validate_if_regions(operation, owner)?;
        return operation
            .regions
            .iter()
            .zip(rebases)
            .map(|(region, rebase)| {
                invert_owned_region(
                    region,
                    0,
                    *linear_index,
                    result,
                    rebase.as_ref(),
                    owner,
                    functions,
                    adjoint_helpers,
                    required_helpers,
                    program,
                    next_id,
                    budget,
                    depth,
                )
            })
            .collect();
    }
    validate_for_regions(operation, owner)?;
    let body = invert_owned_region(
        &operation.regions[0],
        1,
        *linear_index,
        result,
        rebases[0].as_ref(),
        owner,
        functions,
        adjoint_helpers,
        required_helpers,
        program,
        next_id,
        budget,
        depth,
    )?;
    let otherwise = identity_else_region(operation, results, next_id)?;
    Ok(vec![body, otherwise])
}

#[allow(clippy::too_many_arguments)]
fn invert_owned_region(
    region: &wire::RegionProto,
    argument_offset: usize,
    linear_index: usize,
    inverse_output: &wire::ValueProto,
    rebase: Option<&RegionRebase>,
    owner: &str,
    functions: &HashMap<&str, &wire::FunctionProto>,
    adjoint_helpers: &HashMap<String, String>,
    required_helpers: &mut BTreeSet<String>,
    program: &VerifiedProgram,
    next_id: &mut u64,
    budget: &mut LoweringBudget,
    depth: usize,
) -> Result<wire::RegionProto, AdjointLoweringError> {
    let [source] = region.blocks.as_slice() else {
        return Err(AdjointLoweringError::Malformed(
            "structured inverse block arity",
        ));
    };
    let linear_argument_index =
        linear_index
            .checked_add(argument_offset)
            .ok_or(AdjointLoweringError::Malformed(
                "structured inverse argument index",
            ))?;
    let original_linear =
        source
            .arguments
            .get(linear_argument_index)
            .ok_or(AdjointLoweringError::Malformed(
                "structured inverse linear argument",
            ))?;
    let (arguments, mut mapping) =
        fresh_region_arguments(source, linear_argument_index, rebase, next_id)?;
    let inverse_input = arguments
        .get(linear_argument_index)
        .ok_or(AdjointLoweringError::Malformed("structured inverse input"))?;
    let value_types = program.local.checked.operations.value_types(owner).ok_or(
        AdjointLoweringError::Malformed("missing structured value types"),
    )?;
    let arena = &program.local.checked.type_arena;
    let CallableTrace { mut prelude, steps } = trace_callable_body(
        owner,
        source,
        value_types,
        arena,
        &mut mapping,
        next_id,
        budget,
    )?;
    let terminator = source
        .terminator
        .as_ref()
        .ok_or(AdjointLoweringError::Malformed(
            "structured inverse terminator",
        ))?;
    let returned =
        *terminator
            .operands
            .get(linear_index)
            .ok_or(AdjointLoweringError::Malformed(
                "structured inverse yielded value",
            ))?;
    let yielded = if steps.is_empty() && returned == original_linear.id {
        inverse_input.id
    } else {
        let mut restored = original_linear.clone();
        restored.id = take_id(next_id)?;
        "structured_adjoint".clone_into(&mut restored.hint);
        restored.region.clone_from(&inverse_output.region);
        let mut boundary = restored.clone();
        boundary.region.clone_from(&inverse_input.region);
        let marker = wire::OperationProto {
            operands: vec![inverse_input.id],
            results: vec![boundary],
            span: terminator.span.clone(),
            ..Default::default()
        };
        let definitions = block_definitions(source);
        prelude.extend(invert_trace(
            steps,
            returned,
            &marker,
            &[original_linear],
            &[restored.clone()],
            &definitions,
            &mapping,
            functions,
            adjoint_helpers,
            required_helpers,
            program,
            next_id,
            budget,
            depth + 1,
            owner,
        )?);
        restored.id
    };
    let yielded_values = inverse_region_yields(
        &arguments,
        terminator.operands.len(),
        argument_offset,
        linear_index,
        yielded,
    )?;
    Ok(single_yield_region(
        arguments,
        prelude,
        yielded_values,
        terminator,
    ))
}

fn single_yield_region(
    arguments: Vec<wire::ValueProto>,
    operations: Vec<wire::OperationProto>,
    operands: Vec<u64>,
    source: &wire::TerminatorProto,
) -> wire::RegionProto {
    wire::RegionProto {
        blocks: vec![wire::BlockProto {
            arguments,
            operations,
            terminator: Some(wire::TerminatorProto {
                kind: "yield".to_owned(),
                operands,
                attributes: source.attributes.clone(),
                span: source.span.clone(),
            }),
        }],
    }
}

fn fresh_region_arguments(
    source: &wire::BlockProto,
    linear_argument_index: usize,
    rebase: Option<&RegionRebase>,
    next_id: &mut u64,
) -> Result<(Vec<wire::ValueProto>, HashMap<u64, u64>), AdjointLoweringError> {
    let arguments = source
        .arguments
        .iter()
        .map(|argument| fresh_rebased_value(argument, rebase, next_id))
        .collect::<Result<Vec<_>, _>>()?;
    let mapping = source
        .arguments
        .iter()
        .zip(&arguments)
        .enumerate()
        .filter(|(index, _)| *index != linear_argument_index)
        .map(|(_, (original, fresh))| (original.id, fresh.id))
        .collect();
    Ok((arguments, mapping))
}

fn inverse_region_yields(
    arguments: &[wire::ValueProto],
    result_count: usize,
    argument_offset: usize,
    linear_index: usize,
    yielded: u64,
) -> Result<Vec<u64>, AdjointLoweringError> {
    (0..result_count)
        .map(|index| {
            if index == linear_index {
                return Ok(yielded);
            }
            arguments
                .get(index + argument_offset)
                .map(|argument| argument.id)
                .ok_or(AdjointLoweringError::Malformed(
                    "structured inverse classical yield",
                ))
        })
        .collect()
}

fn identity_else_region(
    operation: &wire::OperationProto,
    results: &[wire::ValueProto],
    next_id: &mut u64,
) -> Result<wire::RegionProto, AdjointLoweringError> {
    let span = operation
        .regions
        .get(1)
        .and_then(|region| region.blocks.first())
        .and_then(|block| block.terminator.as_ref())
        .and_then(|terminator| terminator.span.clone());
    let arguments = results
        .iter()
        .map(|result| fresh_value(result, result.region.clone(), next_id))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(wire::RegionProto {
        blocks: vec![wire::BlockProto {
            arguments: arguments.clone(),
            operations: Vec::new(),
            terminator: Some(wire::TerminatorProto {
                kind: "yield".to_owned(),
                operands: arguments.iter().map(|argument| argument.id).collect(),
                span,
                ..Default::default()
            }),
        }],
    })
}

fn fresh_rebased_value(
    source: &wire::ValueProto,
    rebase: Option<&RegionRebase>,
    next_id: &mut u64,
) -> Result<wire::ValueProto, AdjointLoweringError> {
    let mut value = fresh_value(source, source.region.clone(), next_id)?;
    if let Some(region) = &mut value.region
        && let Some(rebase) = rebase
    {
        rebase.apply(region);
    }
    Ok(value)
}

fn freshen_structured_regions(
    operation: &wire::OperationProto,
    rebases: &[Option<RegionRebase>],
    next_id: &mut u64,
) -> Result<Vec<wire::RegionProto>, AdjointLoweringError> {
    operation
        .regions
        .iter()
        .zip(rebases)
        .map(|(region, rebase)| {
            freshen_region(region, rebase.as_ref(), next_id).map_err(|error| match error {
                crate::cleanup_synthesis::CleanupSynthesisError::IdentifierCapacityExceeded => {
                    AdjointLoweringError::IdentifierCapacityExceeded
                }
                _ => AdjointLoweringError::Malformed("structured region"),
            })
        })
        .collect()
}

fn restore_direct_call(
    operation: &wire::OperationProto,
    current: u64,
    restored: wire::ValueProto,
    classical: Vec<u64>,
    functions: &HashMap<&str, &wire::FunctionProto>,
    program: &VerifiedProgram,
) -> Result<wire::OperationProto, AdjointLoweringError> {
    let callee_name = string_attribute(operation, "callee")
        .ok_or(AdjointLoweringError::Malformed("nested adjoint callee"))?;
    let callee = functions
        .get(callee_name)
        .copied()
        .ok_or(AdjointLoweringError::Malformed(
            "unresolved nested adjoint callee",
        ))?;
    let value_types = program
        .local
        .checked
        .operations
        .value_types(callee_name)
        .ok_or(AdjointLoweringError::Malformed(
            "missing nested callable value types",
        ))?;
    let arena = &program.local.checked.type_arena;
    let linear_count = callee
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
    if linear_count != 1 {
        return Err(AdjointLoweringError::Unsupported(format!(
            "non-unary nested callable {callee_name:?}"
        )));
    }
    let mut classical = classical.into_iter();
    let operands = callee
        .parameters
        .iter()
        .map(|parameter| {
            let value = parameter.value.as_ref()?;
            let type_id = value_types.get(&value.id)?;
            if arena.is_quantum(*type_id) {
                Some(current)
            } else {
                classical.next()
            }
        })
        .collect::<Option<Vec<_>>>()
        .ok_or(AdjointLoweringError::Malformed(
            "nested direct-call parameter ABI",
        ))?;
    if classical.next().is_some() {
        return Err(AdjointLoweringError::Malformed(
            "nested direct-call classical arity",
        ));
    }
    Ok(wire::OperationProto {
        opcode: "qcore.call".to_owned(),
        operands,
        results: vec![restored],
        attributes: string_attributes(&[("callee", callee_name)]),
        span: operation.span.clone(),
        ..Default::default()
    })
}

fn invert_operation(
    operation: &wire::OperationProto,
    current: u64,
    restored: wire::ValueProto,
    classical: Vec<u64>,
) -> Result<wire::OperationProto, AdjointLoweringError> {
    let operands = std::iter::once(current).chain(classical).collect();
    if is_primitive_adjointable(&operation.opcode) {
        return Ok(wire::OperationProto {
            opcode: "qcore.adjoint".to_owned(),
            operands,
            results: vec![restored],
            attributes: adjoint_attributes(operation),
            span: operation.span.clone(),
            ..Default::default()
        });
    }
    if operation.opcode == "qcore.reshape" {
        return Ok(wire::OperationProto {
            opcode: "qcore.reshape".to_owned(),
            operands: vec![current],
            results: vec![restored],
            attributes: string_attributes(&[("proof", "modular-adjoint")]),
            span: operation.span.clone(),
            ..Default::default()
        });
    }
    let original = string_attribute(operation, "original")
        .ok_or(AdjointLoweringError::Malformed("nested adjoint original"))?;
    let attributes = mapping_attribute(operation, "original_attributes")
        .ok_or(AdjointLoweringError::Malformed("nested adjoint attributes"))?;
    Ok(wire::OperationProto {
        opcode: original.to_owned(),
        operands,
        results: vec![restored],
        attributes,
        span: operation.span.clone(),
        ..Default::default()
    })
}

fn shared_adjoint_call(
    operation: &wire::OperationProto,
    helpers: &HashMap<String, String>,
    required: &mut BTreeSet<String>,
) -> Result<wire::OperationProto, AdjointLoweringError> {
    let callee = string_attribute(operation, "callee")
        .ok_or(AdjointLoweringError::Malformed("adjoint callee"))?;
    let helper = require_adjoint_helper(callee, helpers, required)?;
    Ok(wire::OperationProto {
        opcode: "qcore.call".into(),
        operands: operation.operands.clone(),
        results: operation.results.clone(),
        attributes: string_attributes(&[("callee", &helper)]),
        span: operation.span.clone(),
        ..Default::default()
    })
}

#[derive(Default)]
struct LoweringBudget {
    operations: usize,
}

impl LoweringBudget {
    fn charge_operation(&mut self) -> Result<(), AdjointLoweringError> {
        self.operations =
            self.operations
                .checked_add(1)
                .ok_or(AdjointLoweringError::ResourceLimitExceeded(
                    "operation limit",
                ))?;
        if self.operations > MAX_TARGET_OPERATIONS {
            return Err(AdjointLoweringError::ResourceLimitExceeded(
                "operation limit",
            ));
        }
        Ok(())
    }

    fn charge_operation_tree(
        &mut self,
        regions: &[wire::RegionProto],
    ) -> Result<(), AdjointLoweringError> {
        self.charge_operation()?;
        let mut pending = regions
            .iter()
            .flat_map(|region| &region.blocks)
            .flat_map(|block| &block.operations)
            .collect::<Vec<_>>();
        while let Some(operation) = pending.pop() {
            self.charge_operation()?;
            pending.extend(
                operation
                    .regions
                    .iter()
                    .flat_map(|region| &region.blocks)
                    .flat_map(|block| &block.operations),
            );
        }
        Ok(())
    }
}

fn callable_definitions(function: &wire::FunctionProto) -> HashMap<u64, wire::ValueProto> {
    function
        .parameters
        .iter()
        .filter_map(|parameter| parameter.value.clone())
        .chain(function.body.iter().flat_map(block_values))
        .map(|value| (value.id, value))
        .collect()
}

fn block_definitions(block: &wire::BlockProto) -> HashMap<u64, wire::ValueProto> {
    block_values(block).map(|value| (value.id, value)).collect()
}

fn block_values(block: &wire::BlockProto) -> impl Iterator<Item = wire::ValueProto> + '_ {
    block.arguments.iter().cloned().chain(
        block
            .operations
            .iter()
            .flat_map(|operation| operation.results.iter().cloned()),
    )
}

fn quantum_operands(
    operation: &wire::OperationProto,
    value_types: &HashMap<u64, crate::types::TypeId>,
    arena: &crate::types::TypeArena,
) -> Vec<u64> {
    operation
        .operands
        .iter()
        .copied()
        .filter(|identifier| {
            value_types
                .get(identifier)
                .is_some_and(|value| arena.is_quantum(*value))
        })
        .collect()
}

fn quantum_results(
    operation: &wire::OperationProto,
    value_types: &HashMap<u64, crate::types::TypeId>,
    arena: &crate::types::TypeArena,
) -> Vec<u64> {
    operation
        .results
        .iter()
        .map(|result| result.id)
        .filter(|identifier| {
            value_types
                .get(identifier)
                .is_some_and(|value| arena.is_quantum(*value))
        })
        .collect()
}

fn fresh_value(
    source: &wire::ValueProto,
    region: Option<wire::RegionRefProto>,
    next_id: &mut u64,
) -> Result<wire::ValueProto, AdjointLoweringError> {
    let mut value = source.clone();
    value.id = take_id(next_id)?;
    "target_adjoint".clone_into(&mut value.hint);
    value.region = region;
    Ok(value)
}

fn take_id(next_id: &mut u64) -> Result<u64, AdjointLoweringError> {
    let value = *next_id;
    *next_id = next_id
        .checked_add(1)
        .ok_or(AdjointLoweringError::IdentifierCapacityExceeded)?;
    Ok(value)
}

fn adjoint_attributes(operation: &wire::OperationProto) -> Vec<wire::AttributeProto> {
    let entries = operation
        .attributes
        .iter()
        .map(|attribute| wire::literal_proto::Entry {
            key: attribute.name.clone(),
            value: attribute.value.clone(),
        })
        .collect();
    vec![
        string_attribute_value("original", &operation.opcode),
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

fn mapping_attribute(
    operation: &wire::OperationProto,
    name: &str,
) -> Option<Vec<wire::AttributeProto>> {
    let wire::literal_proto::Value::MappingValue(mapping) =
        attribute(operation, name)?.value.as_ref()?
    else {
        return None;
    };
    mapping
        .entries
        .iter()
        .map(|entry| {
            Some(wire::AttributeProto {
                name: entry.key.clone(),
                value: Some(entry.value.clone()?),
            })
        })
        .collect::<Option<Vec<_>>>()
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
    let wire::literal_proto::Value::StringValue(value) =
        attribute(operation, name)?.value.as_ref()?
    else {
        return None;
    };
    Some(value)
}

fn string_tuple_attribute(operation: &wire::OperationProto, name: &str) -> Option<Vec<String>> {
    let wire::literal_proto::Value::TupleValue(values) =
        attribute(operation, name)?.value.as_ref()?
    else {
        return None;
    };
    values
        .values
        .iter()
        .map(|value| {
            let wire::literal_proto::Value::StringValue(value) = value.value.as_ref()? else {
                return None;
            };
            Some(value.clone())
        })
        .collect()
}
