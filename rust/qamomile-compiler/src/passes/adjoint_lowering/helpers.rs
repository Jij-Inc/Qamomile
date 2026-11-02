//! Deterministic inverse-helper discovery and declaration-order ABI construction.

use super::{
    AdjointLoweringError, LoweringBudget, derive_adjoint_body, mapping_attribute, string_attribute,
    string_attributes, string_tuple_attribute, take_id,
};
use crate::{VerifiedProgram, wire};
use std::collections::{BTreeSet, HashMap, HashSet};

pub(super) fn adjoint_helper_symbols(
    package: &wire::QCorePackageProto,
) -> Result<HashMap<String, String>, AdjointLoweringError> {
    let mut callees = HashSet::new();
    for function in &package.functions {
        collect_adjoint_callees(
            function.body.iter().flat_map(|block| &block.operations),
            &mut callees,
        )?;
    }
    let mut occupied = package
        .functions
        .iter()
        .map(|function| function.symbol.clone())
        .collect::<HashSet<_>>();
    let mut callees = callees.into_iter().collect::<Vec<_>>();
    callees.sort();
    let mut helpers = HashMap::new();
    for callee in callees {
        let base = format!("{callee}.__qamomile_adjoint");
        let mut suffix = 0_u64;
        let helper = loop {
            let candidate = if suffix == 0 {
                base.clone()
            } else {
                format!("{base}_{suffix}")
            };
            if occupied.insert(candidate.clone()) {
                break candidate;
            }
            suffix = suffix
                .checked_add(1)
                .ok_or(AdjointLoweringError::IdentifierCapacityExceeded)?;
        };
        helpers.insert(callee, helper);
    }
    Ok(helpers)
}

pub(super) fn adjoint_dependency_closure(
    roots: &BTreeSet<String>,
    functions: &HashMap<&str, &wire::FunctionProto>,
) -> Result<HashSet<String>, AdjointLoweringError> {
    let mut closure = HashSet::new();
    let mut pending = roots.iter().cloned().collect::<Vec<_>>();
    while let Some(symbol) = pending.pop() {
        if !closure.insert(symbol.clone()) {
            continue;
        }
        let function = functions.get(symbol.as_str()).ok_or_else(|| {
            AdjointLoweringError::Unsupported(format!("external inverse {symbol:?}"))
        })?;
        let mut dependencies = HashSet::new();
        collect_adjoint_callees(
            function.body.iter().flat_map(|body| &body.operations),
            &mut dependencies,
        )?;
        pending.extend(
            dependencies
                .into_iter()
                .filter(|name| !closure.contains(name)),
        );
    }
    Ok(closure)
}

fn collect_adjoint_callees<'a>(
    operations: impl Iterator<Item = &'a wire::OperationProto>,
    callees: &mut HashSet<String>,
) -> Result<(), AdjointLoweringError> {
    for operation in operations {
        match operation.opcode.as_str() {
            "qcore.call" | "qcore.adjoint_call" | "qcore.control_call" => {
                callees.insert(
                    string_attribute(operation, "callee")
                        .ok_or(AdjointLoweringError::Malformed("controlled callee"))?
                        .to_owned(),
                );
            }
            "qcore.select_call" => {
                callees.extend(
                    string_tuple_attribute(operation, "callees")
                        .ok_or(AdjointLoweringError::Malformed("selected callees"))?,
                );
            }
            "qcore.adjoint" => {
                let original = string_attribute(operation, "original")
                    .ok_or(AdjointLoweringError::Malformed("adjoint original"))?;
                if matches!(original, "qcore.control_call" | "qcore.select_call") {
                    let attributes = mapping_attribute(operation, "original_attributes").ok_or(
                        AdjointLoweringError::Malformed("coherent adjoint attributes"),
                    )?;
                    let synthetic = wire::OperationProto {
                        opcode: original.to_owned(),
                        attributes,
                        ..Default::default()
                    };
                    collect_adjoint_callees(std::iter::once(&synthetic), callees)?;
                }
            }
            _ => {}
        }
        for region in &operation.regions {
            for block in &region.blocks {
                collect_adjoint_callees(block.operations.iter(), callees)?;
            }
        }
    }
    Ok(())
}

/// A declaration-order inverse ABI, independent of any call site's operands.
struct AdjointBoundary {
    parameters: Vec<wire::ParameterProto>,
    result: wire::ValueProto,
}

fn adjoint_boundary(
    callee: &wire::FunctionProto,
    linear_indices: &[usize],
    next_id: &mut u64,
) -> Result<AdjointBoundary, AdjointLoweringError> {
    let Some(&first_linear) = linear_indices.first() else {
        return Err(AdjointLoweringError::Unsupported(
            "adjoint callable has no owned input".into(),
        ));
    };
    // The inverse signature is (forward result, classical inputs) -> owned
    // forward inputs, packed in declaration order. Calls retain this ABI rather
    // than substituting raw operand IDs across differing availability stages.
    let mut inverse_input = callee.parameters[first_linear].clone();
    let input = inverse_input
        .value
        .as_mut()
        .ok_or(AdjointLoweringError::Malformed("helper input"))?;
    input.id = take_id(next_id)?;
    input.r#type.clone_from(&callee.result_type);
    input.has_region = true;
    input.region = Some(wire::RegionRefProto {
        root: input.id,
        ..Default::default()
    });
    let mut parameters = vec![inverse_input];
    for (index, parameter) in callee.parameters.iter().enumerate() {
        if linear_indices.contains(&index) {
            continue;
        }
        let mut parameter = parameter.clone();
        parameter
            .value
            .as_mut()
            .ok_or(AdjointLoweringError::Malformed(
                "helper classical parameter",
            ))?
            .id = take_id(next_id)?;
        parameters.push(parameter);
    }
    let input_types = linear_indices
        .iter()
        .map(|index| {
            callee.parameters[*index]
                .value
                .as_ref()
                .and_then(|v| v.r#type.clone())
                .ok_or(AdjointLoweringError::Malformed("helper output type"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output_type = if input_types.len() == 1 {
        input_types.into_iter().next().expect("one input")
    } else {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Product.into(),
            arguments: input_types,
            ..Default::default()
        }
    };
    let output = wire::ValueProto {
        id: take_id(next_id)?,
        r#type: Some(output_type),
        hint: "adjoint_result".into(),
        has_region: true,
        region: parameters[0].value.as_ref().and_then(|v| v.region.clone()),
    };
    Ok(AdjointBoundary {
        parameters,
        result: output,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_adjoint_helper(
    callee_name: &str,
    adjoint_helpers: &HashMap<String, String>,
    required_helpers: &mut BTreeSet<String>,
    functions: &HashMap<&str, &wire::FunctionProto>,
    program: &VerifiedProgram,
    next_id: &mut u64,
    budget: &mut LoweringBudget,
) -> Result<wire::FunctionProto, AdjointLoweringError> {
    let callee = functions
        .get(callee_name)
        .copied()
        .ok_or_else(|| AdjointLoweringError::Unsupported(callee_name.to_owned()))?;
    let value_types = program
        .local
        .checked
        .operations
        .value_types(callee_name)
        .ok_or(AdjointLoweringError::Malformed("helper value types"))?;
    let arena = &program.local.checked.type_arena;
    let linear_indices = callee
        .parameters
        .iter()
        .enumerate()
        .filter_map(|(index, parameter)| {
            parameter
                .value
                .as_ref()
                .and_then(|value| value_types.get(&value.id))
                .is_some_and(|type_id| arena.is_quantum(*type_id))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let AdjointBoundary {
        parameters,
        result: output,
    } = adjoint_boundary(callee, &linear_indices, next_id)?;
    let marker = wire::OperationProto {
        opcode: "qcore.adjoint_call".to_owned(),
        operands: parameters
            .iter()
            .filter_map(|parameter| parameter.value.as_ref().map(|value| value.id))
            .collect(),
        results: vec![output.clone()],
        attributes: string_attributes(&[("callee", callee_name), ("proof", "modular-adjoint")]),
        span: callee.span.clone(),
        ..Default::default()
    };
    let operations = derive_adjoint_body(
        &marker,
        functions,
        adjoint_helpers,
        required_helpers,
        program,
        next_id,
        budget,
    )?;
    let mut helper = callee.clone();
    helper.symbol = adjoint_helpers
        .get(callee_name)
        .cloned()
        .ok_or(AdjointLoweringError::Malformed("helper symbol"))?;
    helper.parameters = parameters;
    helper.result_type.clone_from(&output.r#type);
    helper.body = Some(wire::BlockProto {
        operations,
        terminator: Some(wire::TerminatorProto {
            kind: "return".to_owned(),
            operands: vec![output.id],
            span: callee.span.clone(),
            ..Default::default()
        }),
        ..Default::default()
    });
    helper.external = false;
    Ok(helper)
}
