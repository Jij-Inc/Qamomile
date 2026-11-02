//! Dependent, stage-aware ABI verification for ordinary static calls.
//!
//! Callable dimensions are scoped by declaration. Each call instantiates that
//! scope from operand types and direct natural-number binders. Result types are
//! checked only after every result dimension has been derived from operands;
//! producer-claimed result shapes can therefore never invent an ABI binding.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use num_bigint::BigUint;
use thiserror::Error;

use crate::control_state::ControlState;
use crate::operations::LocallyCheckedOperations;
use crate::type_relations::unify_assignable;
use crate::types::{
    Dimension, ScalarType, SemanticType, Stage, TypeArena, TypeId, dimension_symbols,
    instantiate_dimension,
};
use crate::verification::QCoreVerificationWork;
use crate::wire;

/// Failure of one ordinary `qcore.call` declaration boundary.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub(crate) enum CallAbiError {
    /// Operands do not instantiate the declared parameter list.
    #[error("Q1027: call to {callee:?} has an invalid parameter ABI")]
    InvalidParameterAbi { callee: String },
    /// The single result is not the exact operand-derived instantiation.
    #[error("Q1028: call to {callee:?} has an invalid result ABI")]
    InvalidResultAbi { callee: String },
    /// An adjoint call does not implement the inverse callable ABI.
    #[error("Q1030: adjoint call to {callee:?} has an invalid ABI")]
    InvalidAdjointAbi { callee: String },
    /// A coherent control or selection call has an invalid packed ABI.
    #[error("Q1052: {transform} call has an invalid ABI")]
    InvalidCoherentAbi { transform: &'static str },
    /// A coherent Boolean lift does not match one pure fixed-width predicate.
    #[error("Q7680: Boolean lift predicate {predicate:?} has an invalid coherent ABI")]
    InvalidBooleanPredicateAbi { predicate: String },
    /// A retained function-local type index is unexpectedly incomplete.
    #[error("Q4002: missing retained call ABI type roots for {function:?}")]
    MissingTypeRoots { function: String },
}

/// Verify every ordinary call against its statically resolved declaration.
///
/// Static symbol and exact attribute-schema checks must run before this pass.
/// Adjoint, controlled, and selected call capability/ABI rules intentionally
/// remain separate; this pass only establishes ordinary functional calls.
///
/// # Errors
///
/// Returns [`CallAbiError`] when an operand list is not assignable under one
/// coherent dependent-dimension substitution, a result binder is not derived
/// from operands, or the exact one-result type/stage contract is violated.
#[cfg(test)]
pub(crate) fn verify_ordinary_calls(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
) -> Result<(), CallAbiError> {
    let mut work = QCoreVerificationWork::default();
    let functions = package
        .functions
        .iter()
        .map(|function| (function.symbol.as_str(), function))
        .collect::<HashMap<_, _>>();
    let entry_bindings = entry_dimension_bindings(package, true, &mut work);
    verify_ordinary_calls_with_context(
        package,
        arena,
        operations,
        &functions,
        &entry_bindings,
        &mut work,
    )
}

pub(crate) fn verify_calls_with_work(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
    use_entry_bindings: bool,
    work: &mut QCoreVerificationWork,
) -> Result<(), CallAbiError> {
    work.charge_call_abi(package.functions.len());
    let functions = package
        .functions
        .iter()
        .map(|function| (function.symbol.as_str(), function))
        .collect::<HashMap<_, _>>();
    let entry_bindings = entry_dimension_bindings(package, use_entry_bindings, work);
    verify_ordinary_calls_with_context(
        package,
        arena,
        operations,
        &functions,
        &entry_bindings,
        work,
    )?;
    verify_transformed_calls_with_context(
        package,
        arena,
        operations,
        &functions,
        &entry_bindings,
        work,
    )
}

fn verify_ordinary_calls_with_context(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
    functions: &HashMap<&str, &wire::FunctionProto>,
    entry_bindings: &HashMap<&str, u64>,
    work: &mut QCoreVerificationWork,
) -> Result<(), CallAbiError> {
    for caller in &package.functions {
        work.charge_call_abi(1_usize.saturating_add(caller.parameters.len()));
        let caller_values = operations.value_types(&caller.symbol).ok_or_else(|| {
            CallAbiError::MissingTypeRoots {
                function: caller.symbol.clone(),
            }
        })?;
        let caller_dimensions = caller_parameter_dimensions(
            caller,
            caller_values,
            arena,
            (caller.symbol == package.entrypoint).then_some(entry_bindings),
            work,
        );
        let caller_operations = walk_operations(caller, work);
        let producers = collect_producers(&caller_operations, work);
        for operation in caller_operations {
            if operation.opcode != "qcore.call" {
                continue;
            }
            work.charge_call_abi(
                1_usize
                    .saturating_add(operation.operands.len())
                    .saturating_add(operation.results.len()),
            );
            let callee_name = string_attribute(operation, "callee")
                .expect("static call graph validates ordinary call attributes");
            let callee = functions
                .get(callee_name)
                .copied()
                .expect("static call graph resolves ordinary callees");
            verify_one_call(
                operation,
                callee,
                caller_values,
                &caller_dimensions,
                &producers,
                arena,
                operations,
            )?;
        }
    }
    Ok(())
}

/// Verify adjoint, coherent-control, and coherent-selection call ABIs.
///
/// Capability proofs are derived later from closed callable contracts. This
/// pass establishes the transformation-independent type, dependent-dimension,
/// selector, and ownership equations first, so contract derivation never has
/// to reason about a malformed call layout.
///
/// # Errors
///
/// Returns [`CallAbiError`] when an adjoint input/output is not the exact
/// inverse signature, or a controlled/selected call does not preserve its
/// coherent selector and callable result in one fresh owned product.
#[cfg(test)]
pub(crate) fn verify_transformed_calls(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
) -> Result<(), CallAbiError> {
    let mut work = QCoreVerificationWork::default();
    let functions = package
        .functions
        .iter()
        .map(|function| (function.symbol.as_str(), function))
        .collect::<HashMap<_, _>>();
    let entry_bindings = entry_dimension_bindings(package, true, &mut work);
    verify_transformed_calls_with_context(
        package,
        arena,
        operations,
        &functions,
        &entry_bindings,
        &mut work,
    )
}

fn verify_transformed_calls_with_context(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
    functions: &HashMap<&str, &wire::FunctionProto>,
    entry_bindings: &HashMap<&str, u64>,
    work: &mut QCoreVerificationWork,
) -> Result<(), CallAbiError> {
    for caller in &package.functions {
        work.charge_call_abi(1_usize.saturating_add(caller.parameters.len()));
        let caller_values = operations.value_types(&caller.symbol).ok_or_else(|| {
            CallAbiError::MissingTypeRoots {
                function: caller.symbol.clone(),
            }
        })?;
        let definitions = collect_definitions(caller, work);
        let caller_dimensions = caller_parameter_dimensions(
            caller,
            caller_values,
            arena,
            (caller.symbol == package.entrypoint).then_some(entry_bindings),
            work,
        );
        let caller_operations = walk_operations(caller, work);
        let producers = collect_producers(&caller_operations, work);
        for operation in caller_operations {
            work.charge_call_abi(
                1_usize
                    .saturating_add(operation.operands.len())
                    .saturating_add(operation.results.len()),
            );
            match operation.opcode.as_str() {
                "qcore.adjoint_call" => verify_adjoint_call(
                    operation,
                    functions,
                    caller_values,
                    &definitions,
                    &caller_dimensions,
                    &producers,
                    arena,
                    operations,
                )?,
                "qcore.control_call" | "qcore.select_call" => verify_coherent_call(
                    operation,
                    functions,
                    caller_values,
                    &definitions,
                    &caller_dimensions,
                    &producers,
                    arena,
                    operations,
                )?,
                "qcore.phase_by" => verify_phase_by(
                    operation,
                    functions,
                    caller_values,
                    &definitions,
                    arena,
                    operations,
                )?,
                "qcore.xor_by" => verify_xor_by(
                    operation,
                    functions,
                    caller_values,
                    &definitions,
                    arena,
                    operations,
                )?,
                "qcore.permute_by" => verify_permute_by(
                    operation,
                    functions,
                    caller_values,
                    &definitions,
                    arena,
                    operations,
                )?,
                _ => {}
            }
        }
    }
    Ok(())
}

fn verify_phase_by(
    operation: &wire::OperationProto,
    functions: &HashMap<&str, &wire::FunctionProto>,
    caller_values: &HashMap<u64, TypeId>,
    definitions: &HashMap<u64, &wire::ValueProto>,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
) -> Result<(), CallAbiError> {
    let predicate_name = string_attribute(operation, "predicate")
        .expect("the static call graph validates Boolean predicate attributes");
    let invalid = || CallAbiError::InvalidBooleanPredicateAbi {
        predicate: predicate_name.to_owned(),
    };
    let (predicate, predicate_input, predicate_result) =
        boolean_predicate_abi(predicate_name, functions, operations)?;
    let ([state], [result]) = (operation.operands.as_slice(), operation.results.as_slice()) else {
        return Err(invalid());
    };
    let state_type = caller_values.get(state).copied().ok_or_else(invalid)?;
    let result_type = caller_values.get(&result.id).copied().ok_or_else(invalid)?;
    let payload = match arena.get(state_type) {
        Some(SemanticType::Quantum(payload)) => *payload,
        _ => return Err(invalid()),
    };
    if predicate.external
        || predicate_input != payload
        || !matches!(
            arena.get(predicate_input),
            Some(SemanticType::Word {
                kind: crate::types::WordKind::Bits,
                width: Dimension::Fixed(_),
            })
        )
        || arena.get(predicate_result) != Some(&SemanticType::Scalar(ScalarType::Bool))
        || result_type != state_type
        || definitions
            .get(state)
            .and_then(|value| value.region.as_ref())
            != result.region.as_ref()
    {
        return Err(invalid());
    }
    Ok(())
}

fn verify_xor_by(
    operation: &wire::OperationProto,
    functions: &HashMap<&str, &wire::FunctionProto>,
    caller_values: &HashMap<u64, TypeId>,
    definitions: &HashMap<u64, &wire::ValueProto>,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
) -> Result<(), CallAbiError> {
    let predicate_name = string_attribute(operation, "predicate")
        .expect("the static call graph validates Boolean predicate attributes");
    let invalid = || CallAbiError::InvalidBooleanPredicateAbi {
        predicate: predicate_name.to_owned(),
    };
    let (predicate, predicate_input, predicate_result) =
        boolean_predicate_abi(predicate_name, functions, operations)?;
    let ([state], [result]) = (operation.operands.as_slice(), operation.results.as_slice()) else {
        return Err(invalid());
    };
    let state_type = caller_values.get(state).copied().ok_or_else(invalid)?;
    let result_type = caller_values.get(&result.id).copied().ok_or_else(invalid)?;
    let Some(SemanticType::Product(elements)) = arena.get(state_type) else {
        return Err(invalid());
    };
    let [source_type, target_type] = elements.as_ref() else {
        return Err(invalid());
    };
    let source_payload = match arena.get(*source_type) {
        Some(SemanticType::Quantum(payload)) => *payload,
        _ => return Err(invalid()),
    };
    let target_payload = match arena.get(*target_type) {
        Some(SemanticType::Quantum(payload)) => *payload,
        _ => return Err(invalid()),
    };
    let output_width =
        crate::phase_lifting::boolean_output_width(arena, predicate_result).ok_or_else(invalid)?;
    if predicate.external
        || predicate_input != source_payload
        || !matches!(
            arena.get(predicate_input),
            Some(SemanticType::Word {
                kind: crate::types::WordKind::Bits,
                width: Dimension::Fixed(_),
            })
        )
        || !matches!(
            arena.get(target_payload),
            Some(SemanticType::Word {
                kind: crate::types::WordKind::Bits,
                width: Dimension::Fixed(width),
            }) if *width == output_width
        )
        || result_type != state_type
        || definitions
            .get(state)
            .and_then(|value| value.region.as_ref())
            != result.region.as_ref()
    {
        return Err(invalid());
    }
    Ok(())
}

fn verify_permute_by(
    operation: &wire::OperationProto,
    functions: &HashMap<&str, &wire::FunctionProto>,
    caller_values: &HashMap<u64, TypeId>,
    definitions: &HashMap<u64, &wire::ValueProto>,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
) -> Result<(), CallAbiError> {
    let predicate_name = string_attribute(operation, "predicate")
        .expect("the static call graph validates Boolean function attributes");
    let invalid = || CallAbiError::InvalidBooleanPredicateAbi {
        predicate: predicate_name.to_owned(),
    };
    let (predicate, predicate_input, predicate_result) =
        boolean_predicate_abi(predicate_name, functions, operations)?;
    let ([state], [result]) = (operation.operands.as_slice(), operation.results.as_slice()) else {
        return Err(invalid());
    };
    let state_type = caller_values.get(state).copied().ok_or_else(invalid)?;
    let result_type = caller_values.get(&result.id).copied().ok_or_else(invalid)?;
    let payload = match arena.get(state_type) {
        Some(SemanticType::Quantum(payload)) => *payload,
        _ => return Err(invalid()),
    };
    let input_width = match arena.get(predicate_input) {
        Some(SemanticType::Word {
            kind: crate::types::WordKind::Bits,
            width: Dimension::Fixed(width),
        }) => *width,
        _ => return Err(invalid()),
    };
    if predicate.external
        || predicate_input != payload
        || crate::phase_lifting::boolean_output_width(arena, predicate_result) != Some(input_width)
        || result_type != state_type
        || definitions
            .get(state)
            .and_then(|value| value.region.as_ref())
            != result.region.as_ref()
    {
        return Err(invalid());
    }
    Ok(())
}

fn boolean_predicate_abi<'a>(
    predicate_name: &str,
    functions: &'a HashMap<&str, &'a wire::FunctionProto>,
    operations: &LocallyCheckedOperations,
) -> Result<(&'a wire::FunctionProto, TypeId, TypeId), CallAbiError> {
    let invalid = || CallAbiError::InvalidBooleanPredicateAbi {
        predicate: predicate_name.to_owned(),
    };
    let predicate = functions.get(predicate_name).copied().ok_or_else(invalid)?;
    let predicate_types =
        operations
            .value_types(predicate_name)
            .ok_or_else(|| CallAbiError::MissingTypeRoots {
                function: predicate_name.to_owned(),
            })?;
    let [parameter] = predicate.parameters.as_slice() else {
        return Err(invalid());
    };
    let predicate_input = parameter
        .value
        .as_ref()
        .and_then(|value| predicate_types.get(&value.id))
        .copied()
        .ok_or_else(invalid)?;
    let predicate_result = operations.result_type(predicate_name).ok_or_else(invalid)?;
    Ok((predicate, predicate_input, predicate_result))
}

#[allow(clippy::too_many_arguments)]
fn verify_adjoint_call(
    operation: &wire::OperationProto,
    functions: &HashMap<&str, &wire::FunctionProto>,
    caller_values: &HashMap<u64, TypeId>,
    definitions: &HashMap<u64, &wire::ValueProto>,
    caller_dimensions: &HashMap<u64, Dimension>,
    producers: &HashMap<u64, &wire::OperationProto>,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
) -> Result<(), CallAbiError> {
    let callee_name = string_attribute(operation, "callee")
        .expect("the static call graph validates adjoint attributes");
    let invalid = || CallAbiError::InvalidAdjointAbi {
        callee: callee_name.to_owned(),
    };
    let callee = functions.get(callee_name).copied().ok_or_else(invalid)?;
    let declared_values =
        operations
            .value_types(callee_name)
            .ok_or_else(|| CallAbiError::MissingTypeRoots {
                function: callee_name.to_owned(),
            })?;
    let callee_result =
        operations
            .result_type(callee_name)
            .ok_or_else(|| CallAbiError::MissingTypeRoots {
                function: callee_name.to_owned(),
            })?;
    let parameter_types = parameter_types(callee, declared_values).ok_or_else(invalid)?;
    let linear = parameter_types
        .iter()
        .enumerate()
        .filter_map(|(index, type_id)| arena.is_quantum(*type_id).then_some((index, *type_id)))
        .collect::<Vec<_>>();
    let classical = parameter_types
        .iter()
        .enumerate()
        .filter_map(|(index, type_id)| (!arena.is_quantum(*type_id)).then_some((index, *type_id)))
        .collect::<Vec<_>>();
    if linear.is_empty()
        || operation.operands.len() != classical.len() + 1
        || operation.results.len() != 1
    {
        return Err(invalid());
    }
    let actual_state = caller_values
        .get(&operation.operands[0])
        .copied()
        .ok_or_else(invalid)?;
    let mut bindings = HashMap::new();
    if !unify_assignable(arena, callee_result, actual_state, &mut bindings) {
        return Err(invalid());
    }
    for ((_, expected), actual_id) in classical.iter().zip(&operation.operands[1..]) {
        let actual = caller_values.get(actual_id).copied().ok_or_else(invalid)?;
        if !unify_assignable(arena, *expected, actual, &mut bindings) {
            return Err(invalid());
        }
    }
    let classical_parameters = classical
        .iter()
        .zip(&operation.operands[1..])
        .map(|((index, expected), actual)| (&callee.parameters[*index], *expected, *actual));
    bind_direct_dimensions(
        classical_parameters,
        callee_result,
        &parameter_types,
        caller_dimensions,
        producers,
        &mut bindings,
        arena,
    )
    .ok_or_else(invalid)?;
    let actual_result = caller_values
        .get(&operation.results[0].id)
        .copied()
        .ok_or_else(invalid)?;
    if !all_dimensions_bound(arena, linear.iter().map(|(_, type_id)| *type_id), &bindings)
        || !matches_adjoint_result(arena, &linear, actual_result, &bindings)
        || definitions
            .get(&operation.operands[0])
            .and_then(|value| value.region.as_ref())
            != operation.results[0].region.as_ref()
    {
        return Err(invalid());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn verify_coherent_call(
    operation: &wire::OperationProto,
    functions: &HashMap<&str, &wire::FunctionProto>,
    caller_values: &HashMap<u64, TypeId>,
    definitions: &HashMap<u64, &wire::ValueProto>,
    caller_dimensions: &HashMap<u64, Dimension>,
    producers: &HashMap<u64, &wire::OperationProto>,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
) -> Result<(), CallAbiError> {
    let transform = if operation.opcode == "qcore.control_call" {
        "control"
    } else {
        "select"
    };
    let invalid = || CallAbiError::InvalidCoherentAbi { transform };
    let selector_id = *operation.operands.first().ok_or_else(invalid)?;
    let selector_type = caller_values
        .get(&selector_id)
        .copied()
        .ok_or_else(invalid)?;
    let selector_width = arena.coherent_width(selector_type).ok().flatten();
    let callee_names = if operation.opcode == "qcore.control_call" {
        vec![
            string_attribute(operation, "callee")
                .expect("the static call graph validates control attributes"),
        ]
    } else {
        string_tuple_attribute(operation, "callees").ok_or_else(invalid)?
    };
    let first_callee = callee_names
        .first()
        .and_then(|name| functions.get(name).copied())
        .ok_or_else(invalid)?;
    let offset = operation
        .operands
        .len()
        .checked_sub(first_callee.parameters.len())
        .ok_or_else(invalid)?;
    if operation.opcode == "qcore.control_call" {
        if arena
            .coherent_width_with(selector_type, |_| Some(1))
            .ok()
            .flatten()
            .is_none()
        {
            return Err(invalid());
        }
        if !matches!(offset, 1 | 2) {
            return Err(invalid());
        }
        if offset == 2 {
            let component_id = operation.operands[1];
            let index_type = operation
                .operands
                .get(1)
                .and_then(|value_id| caller_values.get(value_id))
                .copied()
                .ok_or_else(invalid)?;
            if !is_unstaged_int(arena, index_type) {
                return Err(invalid());
            }
            if let Some(producer) = producers.get(&component_id)
                && producer.opcode == "qcore.constant"
            {
                let component = producer
                    .attributes
                    .iter()
                    .find(|attribute| attribute.name == "value")
                    .and_then(|attribute| attribute.value.as_ref())
                    .and_then(ControlState::decode);
                let Some(ControlState::Exact(component)) = component else {
                    return Err(invalid());
                };
                if selector_width.is_some_and(|width| component >= BigUint::from(width)) {
                    return Err(invalid());
                }
            }
        }
        let control_state = operation
            .attributes
            .iter()
            .find(|attribute| attribute.name == "control_state")
            .and_then(|attribute| attribute.value.as_ref())
            .and_then(ControlState::decode)
            .ok_or_else(invalid)?;
        let effective_width = if offset == 2 { Some(1) } else { selector_width };
        if effective_width.is_some_and(|width| !control_state.fits(width)) {
            return Err(invalid());
        }
    } else if offset != 1 || selector_width.and_then(power_of_two_count) != Some(callee_names.len())
    {
        return Err(invalid());
    }
    if operation.results.len() != 1 {
        return Err(invalid());
    }
    let result_type = caller_values
        .get(&operation.results[0].id)
        .copied()
        .ok_or_else(invalid)?;
    let Some(SemanticType::Product(result_elements)) = arena.get(result_type) else {
        return Err(invalid());
    };
    if result_elements.len() != 2 || result_elements[0] != selector_type {
        return Err(invalid());
    }
    let callable_result = result_elements[1];
    let actual_ids = operation.operands.get(offset..).ok_or_else(invalid)?;
    let actual_types = actual_ids
        .iter()
        .map(|value_id| caller_values.get(value_id).copied().ok_or_else(invalid))
        .collect::<Result<Vec<_>, _>>()?;
    for callee_name in &callee_names {
        let callee = functions.get(callee_name).copied().ok_or_else(invalid)?;
        let declared_values =
            operations
                .value_types(callee_name)
                .ok_or_else(|| CallAbiError::MissingTypeRoots {
                    function: (*callee_name).to_owned(),
                })?;
        let parameter_types = parameter_types(callee, declared_values).ok_or_else(invalid)?;
        if parameter_types.len() != actual_types.len()
            || !parameter_types
                .first()
                .is_some_and(|type_id| arena.is_quantum(*type_id))
            || parameter_types
                .iter()
                .skip(1)
                .any(|type_id| arena.is_quantum(*type_id))
        {
            return Err(invalid());
        }
        let mut bindings = HashMap::new();
        if parameter_types
            .iter()
            .zip(&actual_types)
            .any(|(expected, actual)| !unify_assignable(arena, *expected, *actual, &mut bindings))
        {
            return Err(invalid());
        }
        let callee_result_type =
            operations
                .result_type(callee_name)
                .ok_or_else(|| CallAbiError::MissingTypeRoots {
                    function: (*callee_name).to_owned(),
                })?;
        bind_direct_dimensions(
            callee
                .parameters
                .iter()
                .zip(&parameter_types)
                .zip(actual_ids)
                .map(|((parameter, expected), actual)| (parameter, *expected, *actual)),
            callee_result_type,
            &parameter_types,
            caller_dimensions,
            producers,
            &mut bindings,
            arena,
        )
        .ok_or_else(invalid)?;
        if !all_dimensions_bound(arena, [callee_result_type], &bindings)
            || !matches_instantiated(arena, callee_result_type, callable_result, &bindings)
        {
            return Err(invalid());
        }
    }
    let selector_region = definitions
        .get(&selector_id)
        .and_then(|value| value.region.as_ref());
    let target_region = actual_ids
        .first()
        .and_then(|value_id| definitions.get(value_id))
        .and_then(|value| value.region.as_ref());
    let result_region = operation.results[0].region.as_ref();
    if selector_region.is_none()
        || target_region.is_none()
        || result_region.is_none()
        || result_region == selector_region
        || result_region == target_region
    {
        return Err(invalid());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn verify_one_call(
    operation: &wire::OperationProto,
    callee: &wire::FunctionProto,
    caller_values: &HashMap<u64, TypeId>,
    caller_dimensions: &HashMap<u64, Dimension>,
    producers: &HashMap<u64, &wire::OperationProto>,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
) -> Result<(), CallAbiError> {
    let invalid_parameters = || CallAbiError::InvalidParameterAbi {
        callee: callee.symbol.clone(),
    };
    let invalid_result = || CallAbiError::InvalidResultAbi {
        callee: callee.symbol.clone(),
    };
    if operation.operands.len() != callee.parameters.len() {
        return Err(invalid_parameters());
    }
    let callee_roots =
        operations
            .value_types(&callee.symbol)
            .ok_or_else(|| CallAbiError::MissingTypeRoots {
                function: callee.symbol.clone(),
            })?;
    let callee_result =
        operations
            .result_type(&callee.symbol)
            .ok_or_else(|| CallAbiError::MissingTypeRoots {
                function: callee.symbol.clone(),
            })?;
    let actual = operation
        .operands
        .iter()
        .map(|value_id| {
            caller_values
                .get(value_id)
                .copied()
                .ok_or_else(invalid_parameters)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expected = callee
        .parameters
        .iter()
        .map(|parameter| {
            parameter
                .value
                .as_ref()
                .and_then(|value| callee_roots.get(&value.id))
                .copied()
                .ok_or_else(invalid_parameters)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut bindings = HashMap::new();
    for (&declared, &supplied) in expected.iter().zip(&actual) {
        if !unify_assignable(arena, declared, supplied, &mut bindings) {
            return Err(invalid_parameters());
        }
    }

    let mut callable_binders = HashSet::new();
    collect_dimensions(arena, callee_result, &mut callable_binders);
    for &parameter_type in &expected {
        collect_dimensions(arena, parameter_type, &mut callable_binders);
    }
    for ((parameter, &declared), &operand_id) in callee
        .parameters
        .iter()
        .zip(&expected)
        .zip(&operation.operands)
    {
        if !is_unstaged_int(arena, declared) {
            continue;
        }
        let binder = callable_binders.iter().find(|dimension| {
            matches!(
                dimension,
                Dimension::Symbol { name, .. } if name.as_ref() == parameter.name
            )
        });
        let Some(binder) = binder else {
            continue;
        };
        let Some(observed) = recover_dimension(
            operand_id,
            caller_dimensions,
            producers,
            &mut HashSet::new(),
        ) else {
            continue;
        };
        if bindings
            .get(binder)
            .is_some_and(|previous| previous != &observed)
        {
            return Err(invalid_parameters());
        }
        bindings.insert(binder.clone(), observed);
    }

    if operation.results.len() != 1 {
        return Err(invalid_result());
    }
    let result_binders = {
        let mut dimensions = HashSet::new();
        collect_dimensions(arena, callee_result, &mut dimensions);
        dimensions
    };
    if result_binders
        .iter()
        .any(|dimension| !bindings.contains_key(dimension))
    {
        return Err(invalid_result());
    }
    let result = caller_values
        .get(&operation.results[0].id)
        .copied()
        .ok_or_else(invalid_result)?;
    if !matches_call_result(arena, callee_result, result, &actual, &bindings) {
        return Err(invalid_result());
    }
    Ok(())
}

fn caller_parameter_dimensions(
    function: &wire::FunctionProto,
    values: &HashMap<u64, TypeId>,
    arena: &TypeArena,
    bindings: Option<&HashMap<&str, u64>>,
    work: &mut QCoreVerificationWork,
) -> HashMap<u64, Dimension> {
    work.charge_call_abi(function.parameters.len());
    function
        .parameters
        .iter()
        .filter_map(|parameter| {
            let value = parameter.value.as_ref()?;
            let type_id = values.get(&value.id).copied()?;
            is_unstaged_int(arena, type_id).then(|| {
                (
                    value.id,
                    bindings
                        .and_then(|values| values.get(parameter.name.as_str()))
                        .map_or_else(
                            || Dimension::Symbol {
                                scope: Arc::from(function.symbol.as_str()),
                                name: Arc::from(parameter.name.as_str()),
                            },
                            |fixed| Dimension::Fixed(*fixed),
                        ),
                )
            })
        })
        .collect()
}

fn entry_dimension_bindings<'a>(
    package: &'a wire::QCorePackageProto,
    use_entry_bindings: bool,
    work: &mut QCoreVerificationWork,
) -> HashMap<&'a str, u64> {
    if !use_entry_bindings {
        return HashMap::new();
    }
    work.charge_call_abi(package.const_bindings.len());
    package
        .const_bindings
        .iter()
        .filter_map(|binding| {
            positive_u64_literal(binding.value.as_ref()?)
                .map(|value| (binding.name.as_str(), value))
        })
        .collect()
}

fn collect_producers<'a>(
    operations: &[&'a wire::OperationProto],
    work: &mut QCoreVerificationWork,
) -> HashMap<u64, &'a wire::OperationProto> {
    let mut producers = HashMap::new();
    for operation in operations {
        work.charge_call_abi(operation.results.len());
        for result in &operation.results {
            producers.insert(result.id, *operation);
        }
    }
    producers
}

fn collect_definitions<'a>(
    function: &'a wire::FunctionProto,
    work: &mut QCoreVerificationWork,
) -> HashMap<u64, &'a wire::ValueProto> {
    work.charge_call_abi(function.parameters.len());
    let mut definitions = function
        .parameters
        .iter()
        .filter_map(|parameter| parameter.value.as_ref())
        .map(|value| (value.id, value))
        .collect::<HashMap<_, _>>();
    let mut blocks = function.body.iter().collect::<Vec<_>>();
    while let Some(block) = blocks.pop() {
        work.charge_call_abi(1_usize.saturating_add(block.arguments.len()));
        definitions.extend(block.arguments.iter().map(|value| (value.id, value)));
        for operation in &block.operations {
            work.charge_call_abi(operation.results.len());
            definitions.extend(operation.results.iter().map(|value| (value.id, value)));
            for region in operation.regions.iter().rev() {
                blocks.extend(region.blocks.iter().rev());
            }
        }
    }
    definitions
}

fn parameter_types(
    function: &wire::FunctionProto,
    values: &HashMap<u64, TypeId>,
) -> Option<Vec<TypeId>> {
    function
        .parameters
        .iter()
        .map(|parameter| {
            parameter
                .value
                .as_ref()
                .and_then(|value| values.get(&value.id))
                .copied()
        })
        .collect()
}

fn walk_operations<'a>(
    function: &'a wire::FunctionProto,
    work: &mut QCoreVerificationWork,
) -> Vec<&'a wire::OperationProto> {
    let mut operations = Vec::new();
    let mut blocks = function.body.iter().collect::<Vec<_>>();
    while let Some(block) = blocks.pop() {
        work.charge_call_abi(1);
        for operation in &block.operations {
            work.charge_call_abi(1);
            operations.push(operation);
            for region in operation.regions.iter().rev() {
                blocks.extend(region.blocks.iter().rev());
            }
        }
    }
    operations
}

fn string_attribute<'a>(operation: &'a wire::OperationProto, name: &str) -> Option<&'a str> {
    use wire::literal_proto::Value;

    operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .and_then(|attribute| attribute.value.as_ref())
        .and_then(|literal| match literal.value.as_ref() {
            Some(Value::StringValue(value)) => Some(value.as_str()),
            _ => None,
        })
}

fn string_tuple_attribute<'a>(
    operation: &'a wire::OperationProto,
    name: &str,
) -> Option<Vec<&'a str>> {
    use wire::literal_proto::Value;

    let literal = operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)?
        .value
        .as_ref()?;
    let Value::TupleValue(sequence) = literal.value.as_ref()? else {
        return None;
    };
    sequence
        .values
        .iter()
        .map(|item| match item.value.as_ref() {
            Some(Value::StringValue(value)) if !value.is_empty() => Some(value.as_str()),
            _ => None,
        })
        .collect()
}

fn is_unstaged_int(arena: &TypeArena, type_id: TypeId) -> bool {
    arena.get(arena.strip_stage(type_id)) == Some(&SemanticType::Scalar(ScalarType::Int))
}

fn recover_dimension(
    value_id: u64,
    parameters: &HashMap<u64, Dimension>,
    producers: &HashMap<u64, &wire::OperationProto>,
    visiting: &mut HashSet<u64>,
) -> Option<Dimension> {
    if let Some(dimension) = parameters.get(&value_id) {
        return Some(dimension.clone());
    }
    if !visiting.insert(value_id) {
        return None;
    }
    let producer = producers.get(&value_id)?;
    let dimension = match producer.opcode.as_str() {
        "qcore.constant"
            if producer.operands.is_empty()
                && producer.results.len() == 1
                && producer.attributes.len() == 1
                && producer.attributes[0].name == "value" =>
        {
            positive_u64_attribute(producer, "value").map(Dimension::Fixed)
        }
        "qcore.stage_lift"
            if producer.operands.len() == 1
                && producer.results.len() == 1
                && producer.attributes.is_empty() =>
        {
            recover_dimension(producer.operands[0], parameters, producers, visiting)
        }
        _ => None,
    };
    visiting.remove(&value_id);
    dimension
}

fn positive_u64_attribute(operation: &wire::OperationProto, name: &str) -> Option<u64> {
    let literal = operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)?
        .value
        .as_ref()?;
    positive_u64_literal(literal)
}

fn positive_u64_literal(literal: &wire::LiteralProto) -> Option<u64> {
    use wire::literal_proto::Value;

    let value = literal.value.as_ref()?;
    match value {
        Value::IntValue(value) => u64::try_from(*value).ok().filter(|value| *value > 0),
        Value::BigintValue(encoded) if encoded.len() == 9 && encoded[0] == 0 => {
            let bytes: [u8; 8] = encoded[1..].try_into().ok()?;
            Some(u64::from_be_bytes(bytes))
        }
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn bind_direct_dimensions<'a, I>(
    parameters: I,
    result_type: TypeId,
    parameter_types: &[TypeId],
    caller_dimensions: &HashMap<u64, Dimension>,
    producers: &HashMap<u64, &wire::OperationProto>,
    bindings: &mut HashMap<Dimension, Dimension>,
    arena: &TypeArena,
) -> Option<()>
where
    I: IntoIterator<Item = (&'a wire::ParameterProto, TypeId, u64)>,
{
    let mut callable_binders = HashSet::new();
    collect_dimensions(arena, result_type, &mut callable_binders);
    for &parameter_type in parameter_types {
        collect_dimensions(arena, parameter_type, &mut callable_binders);
    }
    for (parameter, declared, operand_id) in parameters {
        if !is_unstaged_int(arena, declared) {
            continue;
        }
        let binder = callable_binders.iter().find(|dimension| {
            matches!(
                dimension,
                Dimension::Symbol { name, .. } if name.as_ref() == parameter.name
            )
        });
        let Some(binder) = binder else {
            continue;
        };
        let Some(observed) = recover_dimension(
            operand_id,
            caller_dimensions,
            producers,
            &mut HashSet::new(),
        ) else {
            continue;
        };
        if bindings
            .get(binder)
            .is_some_and(|previous| previous != &observed)
        {
            return None;
        }
        bindings.insert(binder.clone(), observed);
    }
    Some(())
}

fn all_dimensions_bound<I>(
    arena: &TypeArena,
    roots: I,
    bindings: &HashMap<Dimension, Dimension>,
) -> bool
where
    I: IntoIterator<Item = TypeId>,
{
    let mut dimensions = HashSet::new();
    for root in roots {
        collect_dimensions(arena, root, &mut dimensions);
    }
    dimensions
        .iter()
        .all(|dimension| bindings.contains_key(dimension))
}

fn matches_adjoint_result(
    arena: &TypeArena,
    linear: &[(usize, TypeId)],
    actual: TypeId,
    bindings: &HashMap<Dimension, Dimension>,
) -> bool {
    if linear.len() == 1 {
        return matches_instantiated(arena, linear[0].1, actual, bindings);
    }
    let Some(SemanticType::Product(elements)) = arena.get(actual) else {
        return false;
    };
    elements.len() == linear.len()
        && linear.iter().zip(elements).all(|((_, expected), actual)| {
            matches_instantiated(arena, *expected, *actual, bindings)
        })
}

fn power_of_two_count(width: u64) -> Option<usize> {
    let shift = u32::try_from(width).ok()?;
    1_usize.checked_shl(shift)
}

fn matches_call_result(
    arena: &TypeArena,
    declared: TypeId,
    actual: TypeId,
    operands: &[TypeId],
    bindings: &HashMap<Dimension, Dimension>,
) -> bool {
    if arena.is_quantum(declared) {
        return matches_instantiated(arena, declared, actual, bindings);
    }
    let expected_stage = operands
        .iter()
        .fold(arena.stage_of(declared), |stage, operand| {
            Stage::join(stage, arena.stage_of(*operand))
        });
    if arena.stage_of(actual) != expected_stage {
        return false;
    }
    matches_instantiated(
        arena,
        arena.strip_stage(declared),
        arena.strip_stage(actual),
        bindings,
    )
}

/// Recover the dependent-dimension substitution already proved by a call ABI.
pub(crate) fn infer_call_dimensions(
    arena: &TypeArena,
    expected: &[TypeId],
    actual: &[TypeId],
) -> Option<HashMap<Dimension, Dimension>> {
    if expected.len() != actual.len() {
        return None;
    }
    let mut bindings = HashMap::new();
    expected
        .iter()
        .zip(actual)
        .all(|(expected, actual)| unify_assignable(arena, *expected, *actual, &mut bindings))
        .then_some(bindings)
}

fn collect_dimensions(arena: &TypeArena, root: TypeId, dimensions: &mut HashSet<Dimension>) {
    let mut pending = vec![root];
    let mut visited = HashSet::new();
    while let Some(type_id) = pending.pop() {
        if !visited.insert(type_id) {
            continue;
        }
        match arena.get(type_id) {
            Some(SemanticType::Array { element, shape }) => {
                dimensions.extend(shape.iter().flat_map(dimension_symbols));
                pending.push(*element);
            }
            Some(SemanticType::Word { width, .. }) => {
                dimensions.extend(dimension_symbols(width));
            }
            Some(SemanticType::Fixed {
                integer_bits,
                fractional_bits,
            }) => {
                dimensions.extend(
                    [integer_bits, fractional_bits]
                        .into_iter()
                        .flat_map(dimension_symbols),
                );
            }
            Some(SemanticType::Product(elements)) => pending.extend(elements.iter().copied()),
            Some(SemanticType::Record { fields, .. }) => {
                pending.extend(fields.iter().map(|(_, field_type)| *field_type));
            }
            Some(SemanticType::Quantum(inner) | SemanticType::Staged { inner, .. }) => {
                pending.push(*inner);
            }
            Some(SemanticType::Function { inputs, output, .. }) => {
                pending.extend(inputs.iter().copied());
                pending.push(*output);
            }
            Some(SemanticType::Scalar(_)) | None => {}
        }
    }
}

fn matches_instantiated(
    arena: &TypeArena,
    expected: TypeId,
    actual: TypeId,
    bindings: &HashMap<Dimension, Dimension>,
) -> bool {
    match (arena.get(expected), arena.get(actual)) {
        (Some(SemanticType::Scalar(left)), Some(SemanticType::Scalar(right))) => left == right,
        (
            Some(SemanticType::Array {
                element: left_element,
                shape: left_shape,
            }),
            Some(SemanticType::Array {
                element: right_element,
                shape: right_shape,
            }),
        ) => {
            left_shape.len() == right_shape.len()
                && left_shape
                    .iter()
                    .zip(right_shape)
                    .all(|(left, right)| instantiated_dimension(left, right, bindings))
                && matches_instantiated(arena, *left_element, *right_element, bindings)
        }
        (
            Some(SemanticType::Word {
                kind: left_kind,
                width: left_width,
            }),
            Some(SemanticType::Word {
                kind: right_kind,
                width: right_width,
            }),
        ) => left_kind == right_kind && instantiated_dimension(left_width, right_width, bindings),
        (
            Some(SemanticType::Fixed {
                integer_bits: left_integer,
                fractional_bits: left_fractional,
            }),
            Some(SemanticType::Fixed {
                integer_bits: right_integer,
                fractional_bits: right_fractional,
            }),
        ) => {
            instantiated_dimension(left_integer, right_integer, bindings)
                && instantiated_dimension(left_fractional, right_fractional, bindings)
        }
        (Some(SemanticType::Product(left)), Some(SemanticType::Product(right))) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| matches_instantiated(arena, *left, *right, bindings))
        }
        (
            Some(SemanticType::Record {
                name: left_name,
                fields: left_fields,
            }),
            Some(SemanticType::Record {
                name: right_name,
                fields: right_fields,
            }),
        ) => {
            left_name == right_name
                && left_fields.len() == right_fields.len()
                && left_fields.iter().zip(right_fields).all(
                    |((left_name, left), (right_name, right))| {
                        left_name == right_name
                            && matches_instantiated(arena, *left, *right, bindings)
                    },
                )
        }
        (Some(SemanticType::Quantum(left)), Some(SemanticType::Quantum(right))) => {
            matches_instantiated(arena, *left, *right, bindings)
        }
        (
            Some(SemanticType::Staged {
                stage: left_stage,
                inner: left,
            }),
            Some(SemanticType::Staged {
                stage: right_stage,
                inner: right,
            }),
        ) => left_stage == right_stage && matches_instantiated(arena, *left, *right, bindings),
        _ => false,
    }
}

fn instantiated_dimension(
    expected: &Dimension,
    actual: &Dimension,
    bindings: &HashMap<Dimension, Dimension>,
) -> bool {
    match expected {
        Dimension::Fixed(_) => expected == actual,
        Dimension::Symbol { .. } => instantiate_dimension(expected, bindings) == *actual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::validate_program_operations_with_types;
    use crate::types::validate_program_types;

    fn scalar(name: &str) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Scalar.into(),
            name: name.to_owned(),
            ..Default::default()
        }
    }

    fn staged(stage: &str, inner: wire::TypeProto) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Staged.into(),
            name: stage.to_owned(),
            arguments: vec![inner],
            ..Default::default()
        }
    }

    fn word(width: wire::dimension_proto::Value) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Array.into(),
            name: "Bits".to_owned(),
            arguments: vec![scalar("Bit")],
            dimensions: vec![wire::DimensionProto { value: Some(width) }],
            ..Default::default()
        }
    }

    fn quantum(inner: wire::TypeProto) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Quantum.into(),
            arguments: vec![inner],
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

    fn value(id: u64, type_spec: wire::TypeProto) -> wire::ValueProto {
        wire::ValueProto {
            id,
            r#type: Some(type_spec),
            ..Default::default()
        }
    }

    fn owned(id: u64, type_spec: wire::TypeProto, root: u64) -> wire::ValueProto {
        wire::ValueProto {
            id,
            r#type: Some(type_spec),
            region: Some(wire::RegionRefProto {
                root,
                ..Default::default()
            }),
            has_region: true,
            ..Default::default()
        }
    }

    fn parameter(name: &str, value: wire::ValueProto) -> wire::ParameterProto {
        wire::ParameterProto {
            name: name.to_owned(),
            value: Some(value),
            ..Default::default()
        }
    }

    fn block(operations: Vec<wire::OperationProto>) -> wire::BlockProto {
        wire::BlockProto {
            operations,
            terminator: Some(wire::TerminatorProto {
                kind: "return".to_owned(),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn function(
        symbol: &str,
        parameters: Vec<wire::ParameterProto>,
        result_type: wire::TypeProto,
        operations: Vec<wire::OperationProto>,
    ) -> wire::FunctionProto {
        wire::FunctionProto {
            symbol: symbol.to_owned(),
            parameters,
            result_type: Some(result_type),
            body: Some(block(operations)),
            ..Default::default()
        }
    }

    fn string(value: &str) -> wire::LiteralProto {
        wire::LiteralProto {
            value: Some(wire::literal_proto::Value::StringValue(value.to_owned())),
        }
    }

    fn integer(value: i64) -> wire::LiteralProto {
        wire::LiteralProto {
            value: Some(wire::literal_proto::Value::IntValue(value)),
        }
    }

    fn string_tuple(values: &[&str]) -> wire::LiteralProto {
        wire::LiteralProto {
            value: Some(wire::literal_proto::Value::TupleValue(
                wire::literal_proto::Sequence {
                    values: values.iter().map(|value| string(value)).collect(),
                },
            )),
        }
    }

    fn attribute(name: &str, literal: wire::LiteralProto) -> wire::AttributeProto {
        wire::AttributeProto {
            name: name.to_owned(),
            value: Some(literal),
        }
    }

    fn call(operands: Vec<u64>, result: wire::ValueProto) -> wire::OperationProto {
        wire::OperationProto {
            opcode: "qcore.call".to_owned(),
            operands,
            results: vec![result],
            attributes: vec![wire::AttributeProto {
                name: "callee".to_owned(),
                value: Some(string("test.callee")),
            }],
            ..Default::default()
        }
    }

    fn check(package: &wire::QCorePackageProto) -> Result<(), CallAbiError> {
        let mut arena = validate_program_types(package).expect("test types are valid");
        let operations = validate_program_operations_with_types(package, &mut arena)
            .expect("test operations are locally valid");
        verify_ordinary_calls(package, &arena, &operations)
    }

    fn check_transformed(package: &wire::QCorePackageProto) -> Result<(), CallAbiError> {
        let mut arena = validate_program_types(package).expect("test types are valid");
        let operations = validate_program_operations_with_types(package, &mut arena)
            .expect("test operations are locally valid");
        verify_transformed_calls(package, &arena, &operations)
    }

    #[test]
    fn accepts_an_operand_derived_result_dimension() {
        use wire::dimension_proto::Value::{Fixed, Symbol};

        let package = wire::QCorePackageProto {
            functions: vec![
                function(
                    "test.callee",
                    vec![parameter("n", value(1, scalar("Int")))],
                    word(Symbol("n".to_owned())),
                    vec![],
                ),
                function(
                    "test.main",
                    vec![parameter("width", value(10, scalar("Int")))],
                    scalar("Unit"),
                    vec![call(vec![10], value(11, word(Symbol("width".to_owned()))))],
                ),
            ],
            ..Default::default()
        };

        assert_eq!(check(&package), Ok(()));

        let fixed_package = wire::QCorePackageProto {
            functions: vec![
                function(
                    "test.callee",
                    vec![parameter("n", value(1, scalar("Int")))],
                    word(Symbol("n".to_owned())),
                    vec![],
                ),
                function(
                    "test.main",
                    vec![],
                    scalar("Unit"),
                    vec![
                        wire::OperationProto {
                            opcode: "qcore.constant".to_owned(),
                            results: vec![value(10, scalar("Int"))],
                            attributes: vec![wire::AttributeProto {
                                name: "value".to_owned(),
                                value: Some(wire::LiteralProto {
                                    value: Some(wire::literal_proto::Value::IntValue(7)),
                                }),
                            }],
                            ..Default::default()
                        },
                        call(vec![10], value(11, word(Fixed(7)))),
                    ],
                ),
            ],
            ..Default::default()
        };
        assert_eq!(check(&fixed_package), Ok(()));
    }

    #[test]
    fn entry_const_binding_is_a_concrete_dimension_argument() {
        use wire::dimension_proto::Value::{Fixed, Symbol};

        let mut package = wire::QCorePackageProto {
            functions: vec![
                function(
                    "test.callee",
                    vec![parameter("n", value(1, staged("Const", scalar("Int"))))],
                    quantum(word(Symbol("n".to_owned()))),
                    vec![],
                ),
                function(
                    "test.main",
                    vec![parameter(
                        "width",
                        value(10, staged("Const", scalar("Int"))),
                    )],
                    quantum(word(Fixed(3))),
                    vec![call(vec![10], value(11, quantum(word(Fixed(3)))))],
                ),
            ],
            entrypoint: "test.main".to_owned(),
            const_bindings: vec![wire::BindingProto {
                name: "width".to_owned(),
                value: Some(wire::LiteralProto {
                    value: Some(wire::literal_proto::Value::IntValue(3)),
                }),
            }],
            ..Default::default()
        };
        package
            .functions
            .sort_by(|left, right| left.symbol.cmp(&right.symbol));

        assert_eq!(check(&package), Ok(()));
    }

    #[test]
    fn rejects_a_result_only_forged_dimension() {
        use wire::dimension_proto::Value::{Fixed, Symbol};

        let package = wire::QCorePackageProto {
            functions: vec![
                function(
                    "test.callee",
                    vec![parameter("n", value(1, scalar("Int")))],
                    word(Symbol("n".to_owned())),
                    vec![],
                ),
                function(
                    "test.main",
                    vec![],
                    scalar("Unit"),
                    vec![
                        wire::OperationProto {
                            opcode: "qcore.constant".to_owned(),
                            results: vec![value(10, scalar("Int"))],
                            attributes: vec![wire::AttributeProto {
                                name: "value".to_owned(),
                                value: Some(wire::LiteralProto {
                                    value: Some(wire::literal_proto::Value::IntValue(-1)),
                                }),
                            }],
                            ..Default::default()
                        },
                        call(vec![10], value(11, word(Fixed(7)))),
                    ],
                ),
            ],
            ..Default::default()
        };

        assert_eq!(
            check(&package),
            Err(CallAbiError::InvalidResultAbi {
                callee: "test.callee".to_owned(),
            })
        );
    }

    #[test]
    fn requires_the_exact_derived_result_stage() {
        let package = wire::QCorePackageProto {
            functions: vec![
                function(
                    "test.callee",
                    vec![parameter("value", value(1, scalar("Int")))],
                    scalar("Int"),
                    vec![],
                ),
                function(
                    "test.main",
                    vec![parameter(
                        "value",
                        value(10, staged("Const", scalar("Int"))),
                    )],
                    scalar("Unit"),
                    vec![call(vec![10], value(11, staged("Param", scalar("Int"))))],
                ),
            ],
            ..Default::default()
        };

        assert_eq!(
            check(&package),
            Err(CallAbiError::InvalidResultAbi {
                callee: "test.callee".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_conflicting_dependent_argument_shapes() {
        use wire::dimension_proto::Value::{Fixed, Symbol};

        let package = wire::QCorePackageProto {
            functions: vec![
                function(
                    "test.callee",
                    vec![
                        parameter("left", value(1, word(Symbol("n".to_owned())))),
                        parameter("right", value(2, word(Symbol("n".to_owned())))),
                    ],
                    scalar("Unit"),
                    vec![],
                ),
                function(
                    "test.main",
                    vec![
                        parameter("left", value(10, word(Fixed(2)))),
                        parameter("right", value(11, word(Fixed(3)))),
                    ],
                    scalar("Unit"),
                    vec![call(vec![10, 11], value(12, scalar("Unit")))],
                ),
            ],
            ..Default::default()
        };

        assert_eq!(
            check(&package),
            Err(CallAbiError::InvalidParameterAbi {
                callee: "test.callee".to_owned(),
            })
        );
    }

    #[test]
    fn accepts_an_exact_dependent_adjoint_and_rejects_a_forged_owner() {
        use wire::dimension_proto::Value::{Fixed, Symbol};

        let target = quantum(word(Symbol("n".to_owned())));
        let concrete_target = quantum(word(Fixed(3)));
        let callee = function(
            "test.callee",
            vec![
                parameter("state", owned(1, target.clone(), 1)),
                parameter("n", value(2, staged("Const", scalar("Int")))),
            ],
            target,
            vec![],
        );
        let adjoint = wire::OperationProto {
            opcode: "qcore.adjoint_call".to_owned(),
            operands: vec![10, 11],
            results: vec![owned(12, concrete_target.clone(), 10)],
            attributes: vec![attribute("callee", string("test.callee"))],
            ..Default::default()
        };
        let width = wire::OperationProto {
            opcode: "qcore.constant".to_owned(),
            results: vec![value(11, staged("Const", scalar("Int")))],
            attributes: vec![attribute("value", integer(3))],
            ..Default::default()
        };
        let main = function(
            "test.main",
            vec![parameter("state", owned(10, concrete_target.clone(), 10))],
            concrete_target,
            vec![width, adjoint],
        );
        let package = wire::QCorePackageProto {
            functions: vec![callee, main.clone()],
            ..Default::default()
        };
        assert_eq!(check_transformed(&package), Ok(()));

        let mut forged = main;
        forged.body.as_mut().expect("body").operations[1].results[0].region =
            Some(wire::RegionRefProto {
                root: 99,
                ..Default::default()
            });
        let forged_package = wire::QCorePackageProto {
            functions: vec![package.functions[0].clone(), forged],
            ..Default::default()
        };
        assert_eq!(
            check_transformed(&forged_package),
            Err(CallAbiError::InvalidAdjointAbi {
                callee: "test.callee".to_owned(),
            })
        );
    }

    #[test]
    fn packs_every_linear_adjoint_input_into_one_owned_result() {
        use wire::dimension_proto::Value::Fixed;

        let qubit = quantum(word(Fixed(1)));
        let pair = product(vec![qubit.clone(), qubit.clone()]);
        let package = wire::QCorePackageProto {
            functions: vec![
                function(
                    "test.callee",
                    vec![
                        parameter("left", owned(1, qubit.clone(), 1)),
                        parameter("right", owned(2, qubit.clone(), 2)),
                    ],
                    pair.clone(),
                    vec![],
                ),
                function(
                    "test.main",
                    vec![parameter("pair", owned(10, pair.clone(), 10))],
                    pair.clone(),
                    vec![wire::OperationProto {
                        opcode: "qcore.adjoint_call".to_owned(),
                        operands: vec![10],
                        results: vec![owned(11, pair, 10)],
                        attributes: vec![attribute("callee", string("test.callee"))],
                        ..Default::default()
                    }],
                ),
            ],
            ..Default::default()
        };

        assert_eq!(check_transformed(&package), Ok(()));
    }

    #[test]
    fn boolean_xor_requires_one_bit_target_and_preserves_the_product_owner() {
        use wire::dimension_proto::Value::Fixed;

        let source = quantum(word(Fixed(2)));
        let target = quantum(word(Fixed(2)));
        let pair = product(vec![source.clone(), target]);
        let predicate_result = product(vec![scalar("Bool"), scalar("Bool")]);
        let predicate = function(
            "test.predicate",
            vec![parameter("bits", value(1, word(Fixed(2))))],
            predicate_result,
            vec![],
        );
        let xor = wire::OperationProto {
            opcode: "qcore.xor_by".to_owned(),
            operands: vec![10],
            results: vec![owned(11, pair.clone(), 10)],
            attributes: vec![attribute("predicate", string("test.predicate"))],
            ..Default::default()
        };
        let main = function(
            "test.main",
            vec![parameter("state", owned(10, pair.clone(), 10))],
            pair.clone(),
            vec![xor],
        );
        let package = wire::QCorePackageProto {
            functions: vec![main.clone(), predicate.clone()],
            ..Default::default()
        };
        assert_eq!(check_transformed(&package), Ok(()));

        let forged_pair = product(vec![source, quantum(word(Fixed(1)))]);
        let forged = function(
            "test.main",
            vec![parameter("state", owned(10, forged_pair.clone(), 10))],
            forged_pair.clone(),
            vec![wire::OperationProto {
                opcode: "qcore.xor_by".to_owned(),
                operands: vec![10],
                results: vec![owned(11, forged_pair, 10)],
                attributes: vec![attribute("predicate", string("test.predicate"))],
                ..Default::default()
            }],
        );
        let forged_package = wire::QCorePackageProto {
            functions: vec![forged, predicate],
            ..Default::default()
        };
        assert_eq!(
            check_transformed(&forged_package),
            Err(CallAbiError::InvalidBooleanPredicateAbi {
                predicate: "test.predicate".to_owned(),
            })
        );
    }

    #[test]
    fn boolean_permutation_requires_matching_input_output_and_owner_widths() {
        use wire::dimension_proto::Value::Fixed;

        let state = quantum(word(Fixed(2)));
        let predicate = function(
            "test.permutation",
            vec![parameter("bits", value(1, word(Fixed(2))))],
            product(vec![scalar("Bool"), scalar("Bool")]),
            vec![],
        );
        let operation = wire::OperationProto {
            opcode: "qcore.permute_by".to_owned(),
            operands: vec![10],
            results: vec![owned(11, state.clone(), 10)],
            attributes: vec![attribute("predicate", string("test.permutation"))],
            ..Default::default()
        };
        let main = function(
            "test.main",
            vec![parameter("state", owned(10, state.clone(), 10))],
            state,
            vec![operation],
        );
        let package = wire::QCorePackageProto {
            functions: vec![main, predicate.clone()],
            ..Default::default()
        };
        assert_eq!(check_transformed(&package), Ok(()));

        let forged_state = quantum(word(Fixed(1)));
        let forged = function(
            "test.main",
            vec![parameter("state", owned(10, forged_state.clone(), 10))],
            forged_state.clone(),
            vec![wire::OperationProto {
                opcode: "qcore.permute_by".to_owned(),
                operands: vec![10],
                results: vec![owned(11, forged_state, 10)],
                attributes: vec![attribute("predicate", string("test.permutation"))],
                ..Default::default()
            }],
        );
        let forged_package = wire::QCorePackageProto {
            functions: vec![forged, predicate],
            ..Default::default()
        };
        assert_eq!(
            check_transformed(&forged_package),
            Err(CallAbiError::InvalidBooleanPredicateAbi {
                predicate: "test.permutation".to_owned(),
            })
        );
    }

    #[test]
    fn verifies_control_selector_range_packed_result_and_fresh_owner() {
        use wire::dimension_proto::Value::Fixed;

        let selector = quantum(word(Fixed(2)));
        let target = quantum(word(Fixed(1)));
        let result = product(vec![selector.clone(), target.clone()]);
        let controlled = wire::OperationProto {
            opcode: "qcore.control_call".to_owned(),
            operands: vec![10, 11],
            results: vec![owned(12, result.clone(), 12)],
            attributes: vec![
                attribute("callee", string("test.callee")),
                attribute("control_state", integer(3)),
            ],
            ..Default::default()
        };
        let main = function(
            "test.main",
            vec![
                parameter("selector", owned(10, selector, 10)),
                parameter("target", owned(11, target.clone(), 11)),
            ],
            result,
            vec![controlled],
        );
        let callee = function(
            "test.callee",
            vec![parameter("target", owned(1, target.clone(), 1))],
            target,
            vec![],
        );
        let package = wire::QCorePackageProto {
            functions: vec![callee.clone(), main.clone()],
            ..Default::default()
        };
        assert_eq!(check_transformed(&package), Ok(()));

        let mut out_of_range = main;
        out_of_range.body.as_mut().expect("body").operations[0].attributes[1].value =
            Some(integer(4));
        assert_eq!(
            check_transformed(&wire::QCorePackageProto {
                functions: vec![callee, out_of_range],
                ..Default::default()
            }),
            Err(CallAbiError::InvalidCoherentAbi {
                transform: "control",
            })
        );
    }

    #[test]
    fn rejects_unrepresentable_select_cardinality_without_shifting() {
        use wire::dimension_proto::Value::Fixed;

        let selector = quantum(word(Fixed(u64::MAX)));
        let target = quantum(word(Fixed(1)));
        let result = product(vec![selector.clone(), target.clone()]);
        let package = wire::QCorePackageProto {
            functions: vec![
                function(
                    "test.callee",
                    vec![parameter("target", owned(1, target.clone(), 1))],
                    target.clone(),
                    vec![],
                ),
                function(
                    "test.main",
                    vec![
                        parameter("selector", owned(10, selector, 10)),
                        parameter("target", owned(11, target, 11)),
                    ],
                    result.clone(),
                    vec![wire::OperationProto {
                        opcode: "qcore.select_call".to_owned(),
                        operands: vec![10, 11],
                        results: vec![owned(12, result, 12)],
                        attributes: vec![attribute("callees", string_tuple(&["test.callee"]))],
                        ..Default::default()
                    }],
                ),
            ],
            ..Default::default()
        };

        assert_eq!(
            check_transformed(&package),
            Err(CallAbiError::InvalidCoherentAbi {
                transform: "select",
            })
        );
    }
}
