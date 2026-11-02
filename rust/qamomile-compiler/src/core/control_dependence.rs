//! Whole-program formal-parameter control dependence.
//!
//! The analysis is reconstructed after local SSA and structured-region
//! verification. It follows explicit captures, value dataflow, loop-carried
//! values, and the normalized static call graph. Producer metadata is never
//! accepted as a control-dependence assertion.
//! Only classical formals become host-control inputs. Quantum ownership may
//! contaminate conservative dataflow joins, but never supplies a launch selector.

use std::collections::{BTreeSet, HashMap};

use thiserror::Error;

use crate::call_graph::{CallFlavor, StaticCallGraph};
use crate::contracts::launch_selector_ids;
use crate::operations::LocallyCheckedOperations;
use crate::types::TypeArena;
use crate::verification::QCoreVerificationWork;
use crate::wire;

type Dependencies = BTreeSet<usize>;

/// Classical formal parameters whose values may reach host control in each callable.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ControlDependenceSummary {
    controlled_parameters: Box<[Box<[usize]>]>,
}

impl ControlDependenceSummary {
    /// Return the sorted controlled formal indices for one function.
    #[must_use]
    pub(crate) fn parameters(&self, function: usize) -> Option<&[usize]> {
        self.controlled_parameters.get(function).map(AsRef::as_ref)
    }
}

/// Failure to reconstruct whole-program control dependence.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub(crate) enum ControlDependenceError {
    /// Recursive closure exhausted its bounded compiler work allowance.
    #[error("Q4007: recursive control-dependence analysis exceeds its work budget")]
    RecursiveWorkBudget,
    /// The retained graph and package no longer identify the same function row.
    #[error("Q4002: missing control-dependence function row {function}")]
    MissingFunction { function: usize },
    /// A retained formal has no canonical semantic type.
    #[error("Q4002: invalid control-dependence parameter {parameter} in {function:?}: {reason}")]
    InvalidParameterType {
        /// Function containing the invalid parameter.
        function: String,
        /// Zero-based formal parameter index.
        parameter: usize,
        /// Canonical type-decoder failure.
        reason: String,
    },
    /// A verified call graph site names an unavailable formal parameter.
    #[error(
        "Q1052: transformed call from {caller:?} cannot map controlled formal {formal} of {callee:?}"
    )]
    InvalidCallOperandMapping {
        /// Caller containing the malformed transformed call site.
        caller: String,
        /// Statically resolved callee whose formal cannot be mapped.
        callee: String,
        /// Zero-based controlled formal index.
        formal: usize,
    },
    /// A controlled call cannot identify whether it carries a host selector.
    #[error("Q1052: controlled call from {caller:?} has an invalid selector ABI for {callee:?}")]
    InvalidControlSelectorMapping {
        /// Caller containing the malformed controlled call site.
        caller: String,
        /// Statically resolved controlled callable.
        callee: String,
    },
}

/// Reconstruct formal parameters that can affect local or transitive control.
///
/// Opaque external declarations conservatively treat every formal as
/// control-sensitive. Concrete functions are analyzed in deterministic
/// callee-before-caller component order. Recursive components propagate formal
/// dependencies to a finite fixed point before callers consume their summaries.
///
/// # Errors
///
/// Returns [`ControlDependenceError`] if retained graph indices do not match
/// the checked package or a call site cannot map a controlled formal to an
/// actual SSA operand.
pub(crate) fn analyze_with_work(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
    call_graph: &StaticCallGraph,
    verification_work: &mut QCoreVerificationWork,
) -> Result<ControlDependenceSummary, ControlDependenceError> {
    verification_work.charge_control_dependence(call_graph.function_count());
    let linear_parameters =
        checked_parameter_linearity(package, arena, operations, verification_work)?;
    analyze_with_linearity(package, call_graph, &linear_parameters, verification_work)
}

fn analyze_with_linearity(
    package: &wire::QCorePackageProto,
    call_graph: &StaticCallGraph,
    linear_parameters: &[Box<[bool]>],
    verification_work: &mut QCoreVerificationWork,
) -> Result<ControlDependenceSummary, ControlDependenceError> {
    let mut summaries = vec![Box::<[usize]>::default(); call_graph.function_count()];
    for component in call_graph.components() {
        let start_work = verification_work.control_dependence_steps();
        loop {
            let mut changed = false;
            for &index in component.members() {
                let derived = analyze_function_control(
                    index,
                    package,
                    call_graph,
                    linear_parameters,
                    &summaries,
                    verification_work,
                )?;
                changed |= summaries[index] != derived;
                summaries[index] = derived;
            }
            if component.is_recursive()
                && verification_work
                    .control_dependence_steps()
                    .saturating_sub(start_work)
                    > crate::budget::MAX_RECURSIVE_ANALYSIS_WORK
            {
                return Err(ControlDependenceError::RecursiveWorkBudget);
            }
            if !component.is_recursive() || !changed {
                break;
            }
        }
    }
    Ok(ControlDependenceSummary {
        controlled_parameters: summaries.into_boxed_slice(),
    })
}

/// Reconstruct one formal control row from local dataflow and current callees.
fn analyze_function_control(
    function_index: usize,
    package: &wire::QCorePackageProto,
    call_graph: &StaticCallGraph,
    linear_parameters: &[Box<[bool]>],
    summaries: &[Box<[usize]>],
    verification_work: &mut QCoreVerificationWork,
) -> Result<Box<[usize]>, ControlDependenceError> {
    verification_work.charge_control_dependence(1);
    let function =
        package
            .functions
            .get(function_index)
            .ok_or(ControlDependenceError::MissingFunction {
                function: function_index,
            })?;
    let function_linearity =
        linear_parameters
            .get(function_index)
            .ok_or(ControlDependenceError::MissingFunction {
                function: function_index,
            })?;
    if function.external {
        verification_work.charge_control_dependence(function_linearity.len());
        return Ok(classical_control_parameters(
            0..function_linearity.len(),
            function_linearity,
        ));
    }

    let mut value_dependencies = seed_parameter_dependencies(function);
    verification_work.charge_control_dependence(function.parameters.len());
    let mut controlled = Dependencies::new();
    if let Some(body) = &function.body {
        analyze_block(
            body,
            &mut value_dependencies,
            &mut controlled,
            verification_work,
        );
    }

    let sites = call_graph.sites_for_caller(function_index).ok_or(
        ControlDependenceError::MissingFunction {
            function: function_index,
        },
    )?;
    for site in sites {
        verification_work.charge_control_dependence(1);
        if site.flavor() == CallFlavor::BooleanLift {
            continue;
        }
        for &callee_index in site.callees() {
            verification_work.charge_control_dependence(1);
            let callee = package.functions.get(callee_index).ok_or(
                ControlDependenceError::MissingFunction {
                    function: callee_index,
                },
            )?;
            let callee_linearity = linear_parameters.get(callee_index).ok_or(
                ControlDependenceError::MissingFunction {
                    function: callee_index,
                },
            )?;
            if let Some(actual) = control_selector_operand(site, callee).map_err(|()| {
                ControlDependenceError::InvalidControlSelectorMapping {
                    caller: function.symbol.clone(),
                    callee: callee.symbol.clone(),
                }
            })? && let Some(dependencies) = value_dependencies.get(actual)
            {
                controlled.extend(dependencies);
            }
            for &formal in summaries
                .get(callee_index)
                .ok_or(ControlDependenceError::MissingFunction {
                    function: callee_index,
                })?
                .as_ref()
            {
                verification_work.charge_control_dependence(1);
                let actual = actual_operand_with_linearity(
                    site.flavor(),
                    site.operands(),
                    callee,
                    callee_linearity,
                    formal,
                )
                .ok_or_else(|| {
                    ControlDependenceError::InvalidCallOperandMapping {
                        caller: function.symbol.clone(),
                        callee: callee.symbol.clone(),
                        formal,
                    }
                })?;
                if let Some(dependencies) = value_dependencies.get(actual) {
                    controlled.extend(dependencies);
                }
            }
        }
    }
    Ok(classical_control_parameters(controlled, function_linearity))
}

/// Exclude quantum owners from formal host-control inputs at the summary boundary.
fn classical_control_parameters(
    controlled: impl IntoIterator<Item = usize>,
    linear_parameters: &[bool],
) -> Box<[usize]> {
    controlled
        .into_iter()
        .filter(|index| !linear_parameters[*index])
        .collect()
}

fn seed_parameter_dependencies(function: &wire::FunctionProto) -> HashMap<u64, Dependencies> {
    function
        .parameters
        .iter()
        .enumerate()
        .filter_map(|(index, parameter)| {
            parameter
                .value
                .as_ref()
                .map(|value| (value.id, Dependencies::from([index])))
        })
        .collect()
}

fn checked_parameter_linearity(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
    work: &mut QCoreVerificationWork,
) -> Result<Vec<Box<[bool]>>, ControlDependenceError> {
    package
        .functions
        .iter()
        .enumerate()
        .map(|(function_index, function)| {
            work.charge_control_dependence(1_usize.saturating_add(function.parameters.len()));
            let value_types = operations.value_types(&function.symbol).ok_or(
                ControlDependenceError::MissingFunction {
                    function: function_index,
                },
            )?;
            function
                .parameters
                .iter()
                .enumerate()
                .map(|(parameter, declaration)| {
                    let value = declaration.value.as_ref().ok_or_else(|| {
                        ControlDependenceError::InvalidParameterType {
                            function: function.symbol.clone(),
                            parameter,
                            reason: "parameter has no semantic type".to_owned(),
                        }
                    })?;
                    let type_id = value_types.get(&value.id).copied().ok_or_else(|| {
                        ControlDependenceError::InvalidParameterType {
                            function: function.symbol.clone(),
                            parameter,
                            reason: "parameter has no checked semantic type".to_owned(),
                        }
                    })?;
                    Ok(arena.is_quantum(type_id))
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Vec::into_boxed_slice)
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn analyze(
    package: &wire::QCorePackageProto,
    call_graph: &StaticCallGraph,
) -> Result<ControlDependenceSummary, ControlDependenceError> {
    let mut arena = TypeArena::default();
    let linear_parameters = package
        .functions
        .iter()
        .map(|function| {
            let scope = arena.intern_scope(&function.symbol);
            function
                .parameters
                .iter()
                .enumerate()
                .map(|(parameter, declaration)| {
                    let source = declaration
                        .value
                        .as_ref()
                        .and_then(|value| value.r#type.as_ref())
                        .ok_or_else(|| ControlDependenceError::InvalidParameterType {
                            function: function.symbol.clone(),
                            parameter,
                            reason: "parameter has no semantic type".to_owned(),
                        })?;
                    let type_id = arena.resolve_wire_type(source, &scope).map_err(|reason| {
                        ControlDependenceError::InvalidParameterType {
                            function: function.symbol.clone(),
                            parameter,
                            reason,
                        }
                    })?;
                    Ok(arena.is_quantum(type_id))
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Vec::into_boxed_slice)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut work = QCoreVerificationWork::default();
    work.charge_control_dependence(call_graph.function_count());
    analyze_with_linearity(package, call_graph, &linear_parameters, &mut work)
}

fn control_selector_operand<'a>(
    site: &'a crate::call_graph::StaticCallSite,
    callee: &wire::FunctionProto,
) -> Result<Option<&'a u64>, ()> {
    if site.flavor() != CallFlavor::Control {
        return Ok(None);
    }
    match site.operands().len().checked_sub(callee.parameters.len()) {
        Some(1) => Ok(None),
        Some(2) => site.operands().get(1).map(Some).ok_or(()),
        _ => Err(()),
    }
}

fn actual_operand_with_linearity<'a>(
    flavor: CallFlavor,
    operands: &'a [u64],
    callee: &wire::FunctionProto,
    linear_parameters: &[bool],
    formal: usize,
) -> Option<&'a u64> {
    match flavor {
        CallFlavor::BooleanLift => None,
        CallFlavor::Ordinary => operands.get(formal),
        CallFlavor::Adjoint | CallFlavor::HistoryAdjoint => {
            if *linear_parameters.get(formal)? {
                return None;
            }
            let classical_index = linear_parameters
                .get(..formal)?
                .iter()
                .filter(|linear| !**linear)
                .count();
            classical_index
                .checked_add(1)
                .and_then(|index| operands.get(index))
        }
        CallFlavor::Control | CallFlavor::Select => operands
            .len()
            .checked_sub(callee.parameters.len())?
            .checked_add(formal)
            .and_then(|index| operands.get(index)),
    }
}

fn analyze_block(
    block: &wire::BlockProto,
    dependencies: &mut HashMap<u64, Dependencies>,
    controlled: &mut Dependencies,
    work: &mut QCoreVerificationWork,
) {
    work.charge_control_dependence(1);
    for operation in &block.operations {
        work.charge_control_dependence(
            1_usize
                .saturating_add(operation.operands.len())
                .saturating_add(operation.results.len())
                .saturating_add(operation.regions.len()),
        );
        let operands = operation
            .operands
            .iter()
            .map(|value_id| dependencies.get(value_id).cloned().unwrap_or_default())
            .collect::<Vec<_>>();
        let combined = union_all(&operands);

        let launch_selectors = launch_selector_ids(operation);
        work.charge_control_dependence(launch_selectors.len());
        for selector in launch_selectors {
            if let Some(selector_dependencies) = dependencies.get(&selector) {
                controlled.extend(selector_dependencies);
            }
        }

        match operation.opcode.as_str() {
            "qcore.if" => {
                if let Some(condition) = operands.first() {
                    controlled.extend(condition);
                }
                for region in &operation.regions {
                    for nested in &region.blocks {
                        bind_arguments(
                            &nested.arguments,
                            operands.get(1..).unwrap_or_default(),
                            dependencies,
                        );
                        analyze_block(nested, dependencies, controlled, work);
                    }
                }
            }
            "qcore.while" => {
                let mut carries = operands.clone();
                loop {
                    work.charge_control_dependence(1);
                    let test = &operation.regions[0].blocks[0];
                    bind_arguments(&test.arguments, &carries, dependencies);
                    analyze_block(test, dependencies, controlled, work);
                    let exits = continuation_dependencies(test, dependencies);
                    if let Some(condition) = exits.first() {
                        controlled.extend(condition);
                    }
                    let mut changed = merge_rows(&mut carries, exits.get(1..).unwrap_or_default());
                    let body = &operation.regions[1].blocks[0];
                    bind_arguments(&body.arguments, &carries, dependencies);
                    analyze_block(body, dependencies, controlled, work);
                    changed |=
                        merge_rows(&mut carries, &continuation_dependencies(body, dependencies));
                    if !changed {
                        break;
                    }
                }
                let otherwise = &operation.regions[2].blocks[0];
                bind_arguments(&otherwise.arguments, &carries, dependencies);
                analyze_block(otherwise, dependencies, controlled, work);
            }
            "qcore.for" => {
                for dependency in operands.iter().take(3) {
                    controlled.extend(dependency);
                }
                analyze_for_regions(operation, &operands, dependencies, controlled, work);
            }
            _ => {
                for region in &operation.regions {
                    for nested in &region.blocks {
                        bind_arguments(&nested.arguments, &operands, dependencies);
                        analyze_block(nested, dependencies, controlled, work);
                    }
                }
            }
        }

        let result_dependencies = match operation.opcode.as_str() {
            "qcore.if" => if_result_dependencies_with_work(operation, &operands, &combined, work),
            "qcore.for" => for_result_dependencies_with_work(operation, &operands, &combined, work),
            "classical.len" if is_shape_only_length(operation) => {
                // Static type extents do not depend on launch-time element values.
                vec![Dependencies::new(); operation.results.len()]
            }
            _ => vec![combined; operation.results.len()],
        };
        for (result, dependency) in operation.results.iter().zip(result_dependencies) {
            merge_dependency(dependencies, result.id, &dependency);
        }
    }
}

fn if_result_dependencies_with_work(
    operation: &wire::OperationProto,
    operands: &[Dependencies],
    fallback: &Dependencies,
    work: &mut QCoreVerificationWork,
) -> Vec<Dependencies> {
    (0..operation.results.len())
        .map(|index| {
            work.charge_control_dependence(1);
            if if_result_is_forwarded(operation, index, work) {
                operands.get(index + 1).cloned().unwrap_or_default()
            } else {
                fallback.clone()
            }
        })
        .collect()
}

fn if_result_is_forwarded(
    operation: &wire::OperationProto,
    index: usize,
    work: &mut QCoreVerificationWork,
) -> bool {
    for region in &operation.regions {
        work.charge_control_dependence(1);
        for block in &region.blocks {
            work.charge_control_dependence(1);
            let Some(terminator) = block
                .terminator
                .as_ref()
                .filter(|terminator| terminator.kind == "yield")
            else {
                return false;
            };
            let Some((source, yielded)) = block
                .arguments
                .get(index)
                .zip(terminator.operands.get(index))
            else {
                return false;
            };
            if !block_value_forwards(block, *yielded, source.id, work) {
                return false;
            }
        }
    }
    true
}

fn analyze_for_regions(
    operation: &wire::OperationProto,
    operands: &[Dependencies],
    dependencies: &mut HashMap<u64, Dependencies>,
    controlled: &mut Dependencies,
    work: &mut QCoreVerificationWork,
) {
    let induction = union_all(operands.get(..3).unwrap_or_default());
    let initial_carries = operands.get(3..).unwrap_or_default();
    let mut loop_carries = initial_carries.to_vec();
    if let Some(body) = operation
        .regions
        .first()
        .and_then(|region| region.blocks.first())
    {
        loop {
            work.charge_control_dependence(1);
            let incoming = std::iter::once(induction.clone())
                .chain(loop_carries.iter().cloned())
                .collect::<Vec<_>>();
            bind_arguments(&body.arguments, &incoming, dependencies);
            analyze_block(body, dependencies, controlled, work);
            let next = continuation_dependencies(body, dependencies);
            if !merge_rows(&mut loop_carries, &next) {
                break;
            }
        }
    }
    for region in operation.regions.iter().skip(1) {
        for block in &region.blocks {
            bind_arguments(&block.arguments, initial_carries, dependencies);
            analyze_block(block, dependencies, controlled, work);
        }
    }
}

fn for_result_dependencies_with_work(
    operation: &wire::OperationProto,
    operands: &[Dependencies],
    fallback: &Dependencies,
    work: &mut QCoreVerificationWork,
) -> Vec<Dependencies> {
    (0..operation.results.len())
        .map(|index| {
            work.charge_control_dependence(1);
            if for_result_is_forwarded(operation, index, work) {
                operands.get(index + 3).cloned().unwrap_or_default()
            } else {
                fallback.clone()
            }
        })
        .collect()
}

#[cfg(test)]
fn for_result_dependencies(
    operation: &wire::OperationProto,
    operands: &[Dependencies],
    fallback: &Dependencies,
) -> Vec<Dependencies> {
    let mut work = QCoreVerificationWork::default();
    for_result_dependencies_with_work(operation, operands, fallback, &mut work)
}

fn for_result_is_forwarded(
    operation: &wire::OperationProto,
    index: usize,
    work: &mut QCoreVerificationWork,
) -> bool {
    for (region_index, region) in operation.regions.iter().enumerate() {
        work.charge_control_dependence(1);
        for block in &region.blocks {
            work.charge_control_dependence(1);
            let argument_index = index + usize::from(region_index == 0);
            let Some((source, yielded)) = block.arguments.get(argument_index).zip(
                block
                    .terminator
                    .as_ref()
                    .and_then(|terminator| terminator.operands.get(index)),
            ) else {
                return false;
            };
            if !block_value_forwards(block, *yielded, source.id, work) {
                return false;
            }
        }
    }
    true
}

fn block_value_forwards(
    block: &wire::BlockProto,
    value_id: u64,
    source_id: u64,
    work: &mut QCoreVerificationWork,
) -> bool {
    work.charge_control_dependence(1);
    if value_id == source_id {
        return true;
    }
    let mut producer = None;
    'operations: for operation in &block.operations {
        work.charge_control_dependence(1);
        for (result_index, result) in operation.results.iter().enumerate() {
            work.charge_control_dependence(1);
            if result.id == value_id {
                producer = Some((operation, result_index));
                break 'operations;
            }
        }
    }
    let Some((operation, result_index)) = producer else {
        return false;
    };
    match operation.opcode.as_str() {
        "qcore.stage_lift" | "qcore.reshape" => operation
            .operands
            .first()
            .is_some_and(|operand| block_value_forwards(block, *operand, source_id, work)),
        "qcore.for" => {
            operation
                .operands
                .get(result_index + 3)
                .is_some_and(|operand| block_value_forwards(block, *operand, source_id, work))
                && for_result_is_forwarded(operation, result_index, work)
        }
        _ => false,
    }
}

fn bind_arguments(
    arguments: &[wire::ValueProto],
    incoming: &[Dependencies],
    dependencies: &mut HashMap<u64, Dependencies>,
) {
    for (argument, dependency) in arguments.iter().zip(incoming) {
        merge_dependency(dependencies, argument.id, dependency);
    }
}

fn continuation_dependencies(
    block: &wire::BlockProto,
    dependencies: &HashMap<u64, Dependencies>,
) -> Vec<Dependencies> {
    let Some(terminator) = &block.terminator else {
        return Vec::new();
    };
    if !matches!(terminator.kind.as_str(), "yield" | "continue") {
        return Vec::new();
    }
    terminator
        .operands
        .iter()
        .map(|value_id| dependencies.get(value_id).cloned().unwrap_or_default())
        .collect()
}

fn merge_rows(target: &mut [Dependencies], source: &[Dependencies]) -> bool {
    let mut changed = false;
    for (target_row, source_row) in target.iter_mut().zip(source) {
        let previous = target_row.len();
        target_row.extend(source_row);
        changed |= target_row.len() != previous;
    }
    changed
}

fn merge_dependency(
    dependencies: &mut HashMap<u64, Dependencies>,
    value_id: u64,
    incoming: &Dependencies,
) {
    dependencies.entry(value_id).or_default().extend(incoming);
}

fn union_all(rows: &[Dependencies]) -> Dependencies {
    rows.iter().flatten().copied().collect()
}

fn is_shape_only_length(operation: &wire::OperationProto) -> bool {
    matches!(
        operation.attributes.as_slice(),
        [attribute]
            if attribute.name == "shape_only"
                && matches!(
                    attribute.value.as_ref().and_then(|value| value.value.as_ref()),
                    Some(wire::literal_proto::Value::BoolValue(true))
                )
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(id: u64) -> wire::ValueProto {
        wire::ValueProto {
            id,
            ..Default::default()
        }
    }

    fn parameter(name: &str, id: u64) -> wire::ParameterProto {
        typed_parameter(name, id, scalar("Int"))
    }

    fn typed_parameter(name: &str, id: u64, type_spec: wire::TypeProto) -> wire::ParameterProto {
        wire::ParameterProto {
            name: name.to_owned(),
            value: Some(wire::ValueProto {
                id,
                r#type: Some(type_spec),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn scalar(name: &str) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Scalar.into(),
            name: name.to_owned(),
            ..Default::default()
        }
    }

    fn quantum_bit() -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Quantum.into(),
            arguments: vec![scalar("Bit")],
            ..Default::default()
        }
    }

    fn product(elements: Vec<wire::TypeProto>) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Product.into(),
            arguments: elements,
            ..Default::default()
        }
    }

    fn block(
        arguments: Vec<wire::ValueProto>,
        operations: Vec<wire::OperationProto>,
        kind: &str,
        operands: Vec<u64>,
    ) -> wire::BlockProto {
        wire::BlockProto {
            arguments,
            operations,
            terminator: Some(wire::TerminatorProto {
                kind: kind.to_owned(),
                operands,
                ..Default::default()
            }),
        }
    }

    fn analyze_function(function: wire::FunctionProto) -> ControlDependenceSummary {
        let package = wire::QCorePackageProto {
            functions: vec![function],
            ..Default::default()
        };
        let graph = crate::call_graph::build(&package).expect("closed test graph");
        analyze(&package, &graph).expect("valid test control dependence")
    }

    #[test]
    fn external_control_dependence_excludes_linear_formals() {
        let function = wire::FunctionProto {
            symbol: "test.external".to_owned(),
            parameters: vec![
                typed_parameter("choice", 1, scalar("Int")),
                typed_parameter("state", 2, quantum_bit()),
                typed_parameter("owned_pair", 3, product(vec![scalar("Int"), quantum_bit()])),
            ],
            external: true,
            ..Default::default()
        };

        let summary = analyze_function(function);

        assert_eq!(summary.parameters(0), Some([0].as_slice()));
    }

    #[test]
    fn concrete_control_dependence_retains_classical_inputs_without_quantum_owners() {
        let function = wire::FunctionProto {
            symbol: "test.measured_control".to_owned(),
            parameters: vec![
                typed_parameter("state", 1, quantum_bit()),
                typed_parameter("flag", 2, scalar("Bool")),
            ],
            body: Some(block(
                vec![],
                vec![
                    wire::OperationProto {
                        opcode: "quantum.measure".to_owned(),
                        operands: vec![1],
                        results: vec![value(3)],
                        ..Default::default()
                    },
                    wire::OperationProto {
                        opcode: "classical.and".to_owned(),
                        operands: vec![2, 3],
                        results: vec![value(4)],
                        ..Default::default()
                    },
                    wire::OperationProto {
                        opcode: "qcore.if".to_owned(),
                        operands: vec![4],
                        regions: vec![
                            wire::RegionProto {
                                blocks: vec![block(vec![], vec![], "yield", vec![])],
                            },
                            wire::RegionProto {
                                blocks: vec![block(vec![], vec![], "yield", vec![])],
                            },
                        ],
                        ..Default::default()
                    },
                ],
                "return",
                vec![],
            )),
            ..Default::default()
        };
        let summary = analyze_function(function);
        assert_eq!(summary.parameters(0), Some([1].as_slice()));
    }

    #[test]
    fn controlled_component_is_a_host_control_dependency() {
        let callee = wire::FunctionProto {
            symbol: "test.flip".to_owned(),
            parameters: vec![typed_parameter("state", 1, quantum_bit())],
            body: Some(block(vec![], vec![], "return", vec![])),
            ..Default::default()
        };
        let entry = wire::FunctionProto {
            symbol: "test.main".to_owned(),
            parameters: vec![
                typed_parameter("controls", 1, quantum_bit()),
                typed_parameter("component", 2, scalar("Int")),
                typed_parameter("target", 3, quantum_bit()),
            ],
            body: Some(block(
                vec![],
                vec![wire::OperationProto {
                    opcode: "qcore.control_call".to_owned(),
                    operands: vec![1, 2, 3],
                    attributes: vec![
                        wire::AttributeProto {
                            name: "callee".to_owned(),
                            value: Some(wire::LiteralProto {
                                value: Some(wire::literal_proto::Value::StringValue(
                                    "test.flip".to_owned(),
                                )),
                            }),
                        },
                        wire::AttributeProto {
                            name: "control_state".to_owned(),
                            value: Some(wire::LiteralProto {
                                value: Some(wire::literal_proto::Value::IntValue(1)),
                            }),
                        },
                    ],
                    ..Default::default()
                }],
                "return",
                vec![],
            )),
            ..Default::default()
        };
        let package = wire::QCorePackageProto {
            functions: vec![callee, entry],
            ..Default::default()
        };
        let graph = crate::call_graph::build(&package).expect("closed control graph");
        let summary = analyze(&package, &graph).expect("valid controlled selector mapping");

        assert_eq!(summary.parameters(1), Some([1].as_slice()));
    }

    #[test]
    fn primitive_selector_is_a_topology_control_dependency() {
        let function = wire::FunctionProto {
            symbol: "test.main".to_owned(),
            parameters: vec![
                typed_parameter("state", 1, quantum_bit()),
                typed_parameter("target", 2, scalar("Int")),
            ],
            body: Some(block(
                vec![],
                vec![wire::OperationProto {
                    opcode: "quantum.x".to_owned(),
                    operands: vec![1, 2],
                    attributes: vec![wire::AttributeProto {
                        name: "keywords".to_owned(),
                        value: Some(wire::LiteralProto {
                            value: Some(wire::literal_proto::Value::TupleValue(
                                wire::literal_proto::Sequence {
                                    values: vec![wire::LiteralProto {
                                        value: Some(wire::literal_proto::Value::StringValue(
                                            "target".to_owned(),
                                        )),
                                    }],
                                },
                            )),
                        }),
                    }],
                    ..Default::default()
                }],
                "return",
                vec![],
            )),
            ..Default::default()
        };

        let summary = analyze_function(function);

        assert_eq!(summary.parameters(0), Some([1].as_slice()));
    }

    #[test]
    fn aggregate_index_is_a_launch_control_dependency() {
        let function = wire::FunctionProto {
            symbol: "test.main".to_owned(),
            parameters: vec![
                parameter("values", 1),
                parameter("index", 2),
                parameter("theta", 3),
            ],
            body: Some(block(
                vec![],
                vec![wire::OperationProto {
                    opcode: "classical.index".to_owned(),
                    operands: vec![1, 2],
                    results: vec![value(4)],
                    ..Default::default()
                }],
                "return",
                vec![3],
            )),
            ..Default::default()
        };

        let summary = analyze_function(function);

        assert_eq!(summary.parameters(0), Some([1].as_slice()));
    }

    #[test]
    fn shape_only_length_excludes_elements_but_preserves_runtime_control() {
        for shape_only in [false, true] {
            let attributes = shape_only
                .then(|| wire::AttributeProto {
                    name: "shape_only".to_owned(),
                    value: Some(wire::LiteralProto {
                        value: Some(wire::literal_proto::Value::BoolValue(true)),
                    }),
                })
                .into_iter()
                .collect();
            let function = wire::FunctionProto {
                symbol: "test.main".to_owned(),
                parameters: vec![
                    typed_parameter("values", 1, product(vec![scalar("Real"), scalar("Real")])),
                    typed_parameter("flag", 2, scalar("Bool")),
                ],
                body: Some(block(
                    vec![],
                    vec![
                        wire::OperationProto {
                            opcode: "classical.len".to_owned(),
                            operands: vec![1],
                            results: vec![value(3)],
                            attributes,
                            ..Default::default()
                        },
                        wire::OperationProto {
                            opcode: "qcore.for".to_owned(),
                            operands: vec![100, 3, 101],
                            ..Default::default()
                        },
                        wire::OperationProto {
                            opcode: "qcore.if".to_owned(),
                            operands: vec![2],
                            ..Default::default()
                        },
                    ],
                    "return",
                    vec![],
                )),
                ..Default::default()
            };
            let summary = analyze_function(function);
            let expected = if shape_only { vec![1] } else { vec![0, 1] };
            assert_eq!(summary.parameters(0), Some(expected.as_slice()));
        }
    }

    #[test]
    fn forwarded_if_captures_do_not_taint_later_control() {
        let forwarding_region = |start: u64| wire::RegionProto {
            blocks: vec![block(
                vec![value(start), value(start + 1), value(start + 2)],
                vec![],
                "yield",
                vec![start, start + 1, start + 2],
            )],
        };
        let first_if = wire::OperationProto {
            opcode: "qcore.if".to_owned(),
            operands: vec![1, 1, 3, 2],
            results: vec![value(10), value(11), value(12)],
            regions: vec![forwarding_region(20), forwarding_region(30)],
            ..Default::default()
        };
        let second_if = wire::OperationProto {
            opcode: "qcore.if".to_owned(),
            operands: vec![11],
            ..Default::default()
        };
        let function = wire::FunctionProto {
            symbol: "test.main".to_owned(),
            parameters: vec![
                parameter("first", 1),
                parameter("theta", 2),
                parameter("second", 3),
            ],
            body: Some(block(vec![], vec![first_if, second_if], "return", vec![])),
            ..Default::default()
        };

        let summary = analyze_function(function);

        assert_eq!(summary.parameters(0), Some([0, 2].as_slice()));
    }

    #[test]
    fn reaches_a_for_loop_carry_fixed_point() {
        let nested_control = wire::OperationProto {
            opcode: "qcore.if".to_owned(),
            operands: vec![12],
            ..Default::default()
        };
        let loop_operation = wire::OperationProto {
            opcode: "qcore.for".to_owned(),
            operands: vec![1, 1, 1, 2, 3],
            regions: vec![wire::RegionProto {
                blocks: vec![block(
                    vec![value(10), value(11), value(12)],
                    vec![nested_control],
                    "continue",
                    vec![12, 11],
                )],
            }],
            ..Default::default()
        };
        let function = wire::FunctionProto {
            symbol: "test.main".to_owned(),
            parameters: vec![
                parameter("bound", 1),
                parameter("left", 2),
                parameter("right", 3),
            ],
            body: Some(block(vec![], vec![loop_operation], "return", vec![])),
            ..Default::default()
        };

        let summary = analyze_function(function);
        assert_eq!(summary.parameters(0), Some([0, 1, 2].as_slice()));
    }

    #[test]
    fn retains_precise_for_result_provenance_through_nested_computation() {
        let nested = wire::OperationProto {
            opcode: "qcore.for".to_owned(),
            operands: vec![100, 101, 102, 11, 12],
            results: vec![value(20), value(21)],
            regions: vec![
                wire::RegionProto {
                    blocks: vec![block(
                        vec![value(110), value(111), value(112)],
                        vec![],
                        "yield",
                        vec![111, 112],
                    )],
                },
                wire::RegionProto {
                    blocks: vec![block(
                        vec![value(113), value(114)],
                        vec![],
                        "yield",
                        vec![113, 114],
                    )],
                },
            ],
            ..Default::default()
        };
        let operation = wire::OperationProto {
            opcode: "qcore.for".to_owned(),
            results: vec![value(30), value(31)],
            regions: vec![
                wire::RegionProto {
                    blocks: vec![block(
                        vec![value(10), value(11), value(12)],
                        vec![nested],
                        "yield",
                        vec![20, 21],
                    )],
                },
                wire::RegionProto {
                    blocks: vec![block(
                        vec![value(13), value(14)],
                        vec![],
                        "yield",
                        vec![13, 14],
                    )],
                },
            ],
            ..Default::default()
        };
        let operands = vec![
            Dependencies::from([2]),
            Dependencies::from([2]),
            Dependencies::from([2]),
            Dependencies::from([0]),
            Dependencies::from([1]),
        ];
        let fallback = Dependencies::from([0, 1, 2]);

        assert_eq!(
            for_result_dependencies(&operation, &operands, &fallback),
            vec![Dependencies::from([0]), Dependencies::from([1])]
        );
    }

    #[test]
    fn retains_loop_bounds_when_a_carry_is_transformed() {
        let operation = wire::OperationProto {
            opcode: "qcore.for".to_owned(),
            results: vec![value(30)],
            regions: vec![
                wire::RegionProto {
                    blocks: vec![block(
                        vec![value(10), value(11)],
                        vec![wire::OperationProto {
                            opcode: "classical.not".to_owned(),
                            operands: vec![11],
                            results: vec![value(20)],
                            ..Default::default()
                        }],
                        "yield",
                        vec![20],
                    )],
                },
                wire::RegionProto {
                    blocks: vec![block(vec![value(12)], vec![], "yield", vec![12])],
                },
            ],
            ..Default::default()
        };
        let operands = vec![
            Dependencies::from([1]),
            Dependencies::from([1]),
            Dependencies::from([1]),
            Dependencies::from([0]),
        ];
        let fallback = Dependencies::from([0, 1]);

        assert_eq!(
            for_result_dependencies(&operation, &operands, &fallback),
            vec![Dependencies::from([0, 1])]
        );
    }

    #[test]
    fn maps_adjoint_classical_formals_around_packed_linear_inputs() {
        let callee = wire::FunctionProto {
            symbol: "test.callee".to_owned(),
            parameters: vec![
                typed_parameter("flag", 1, scalar("Bool")),
                typed_parameter("left", 2, quantum_bit()),
                typed_parameter("right", 3, quantum_bit()),
            ],
            body: Some(block(
                vec![],
                vec![wire::OperationProto {
                    opcode: "qcore.if".to_owned(),
                    operands: vec![1],
                    ..Default::default()
                }],
                "return",
                vec![],
            )),
            ..Default::default()
        };
        let entry = wire::FunctionProto {
            symbol: "test.main".to_owned(),
            parameters: vec![
                typed_parameter("packed", 10, quantum_bit()),
                typed_parameter("flag", 11, scalar("Bool")),
            ],
            body: Some(block(
                vec![],
                vec![wire::OperationProto {
                    opcode: "qcore.adjoint_call".to_owned(),
                    operands: vec![10, 11],
                    attributes: vec![wire::AttributeProto {
                        name: "callee".to_owned(),
                        value: Some(wire::LiteralProto {
                            value: Some(wire::literal_proto::Value::StringValue(
                                "test.callee".to_owned(),
                            )),
                        }),
                    }],
                    ..Default::default()
                }],
                "return",
                vec![],
            )),
            ..Default::default()
        };
        let package = wire::QCorePackageProto {
            functions: vec![callee, entry],
            ..Default::default()
        };
        let graph = crate::call_graph::build(&package).expect("closed adjoint graph");
        let summary = analyze(&package, &graph).expect("valid transformed mapping");

        assert_eq!(summary.parameters(1), Some([1].as_slice()));
    }
}
