//! Normalize terminal owned returns without expanding control or recursion.

use std::collections::{HashMap, HashSet};

use super::{
    AdjointLoweringError, LoweringBudget, MAX_ADJOINT_REGION_DEPTH, fresh_value, maximum_value_id,
};
use crate::{VerifiedProgram, seal_package, wire};

pub(super) fn normalize(
    program: &VerifiedProgram,
    required: &HashSet<String>,
) -> Result<VerifiedProgram, AdjointLoweringError> {
    if required.is_empty() {
        return Ok(program.clone());
    }
    let mut package = program.local.checked.package.clone();
    let mut next_id = maximum_value_id(&package)
        .checked_add(1)
        .ok_or(AdjointLoweringError::IdentifierCapacityExceeded)?;
    let mut budget = LoweringBudget::default();
    let mut changed = false;
    let unitary = program
        .callable_contracts()
        .filter_map(|(name, contract)| {
            (contract.kind() == crate::CallableKind::Unitary).then_some(name)
        })
        .collect::<HashSet<_>>();
    for function in &mut package.functions {
        if !required.contains(&function.symbol) || !unitary.contains(function.symbol.as_str()) {
            continue;
        }
        let mut definitions = function
            .parameters
            .iter()
            .filter_map(|p| p.value.as_ref())
            .map(|v| (v.id, v.clone()))
            .collect::<HashMap<_, _>>();
        let mut pending = function.body.iter().collect::<Vec<_>>();
        while let Some(block) = pending.pop() {
            definitions.extend(block.arguments.iter().map(|v| (v.id, v.clone())));
            for operation in &block.operations {
                definitions.extend(operation.results.iter().map(|v| (v.id, v.clone())));
                pending.extend(operation.regions.iter().flat_map(|r| &r.blocks));
            }
        }
        if let Some(body) = &mut function.body {
            changed |= normalize_tail(body, &definitions, &mut next_id, &mut budget, 0)?;
        }
    }
    if !changed {
        return Ok(program.clone());
    }
    Ok(VerifiedProgram::from_bytes(&seal_package(package)?)?)
}

fn normalize_tail(
    block: &mut wire::BlockProto,
    definitions: &HashMap<u64, wire::ValueProto>,
    next_id: &mut u64,
    budget: &mut LoweringBudget,
    depth: usize,
) -> Result<bool, AdjointLoweringError> {
    if depth >= MAX_ADJOINT_REGION_DEPTH {
        return Err(AdjointLoweringError::ResourceLimitExceeded(
            "return normalization depth",
        ));
    }
    if block
        .terminator
        .as_ref()
        .is_none_or(|t| t.kind != "unreachable")
    {
        return Ok(false);
    }
    let Some(operation) = block.operations.last_mut() else {
        return Ok(false);
    };
    if operation.opcode != "qcore.if"
        || !operation.results.is_empty()
        || operation.regions.len() != 2
    {
        return Ok(false);
    }
    let captures = operation.operands[1..]
        .iter()
        .map(|id| {
            definitions
                .get(id)
                .cloned()
                .ok_or(AdjointLoweringError::Malformed("return capture"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let owners = captures
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.has_region.then_some(i))
        .collect::<Vec<_>>();
    let [owner] = owners.as_slice() else {
        return Ok(false);
    };
    for region in &mut operation.regions {
        let [branch] = region.blocks.as_mut_slice() else {
            return Err(AdjointLoweringError::Malformed("return branch"));
        };
        normalize_tail(branch, definitions, next_id, budget, depth + 1)?;
        if branch.arguments.len() != captures.len()
            || branch
                .terminator
                .as_ref()
                .is_none_or(|t| t.kind != "return" || t.operands.len() != 1)
        {
            return Err(AdjointLoweringError::Unsupported(
                "nonterminal owned conditional return".into(),
            ));
        }
        let terminator = branch.terminator.as_mut().expect("validated return");
        let returned = terminator.operands[0];
        terminator.kind = "yield".into();
        terminator.operands = branch
            .arguments
            .iter()
            .enumerate()
            .map(|(i, v)| if i == *owner { returned } else { v.id })
            .collect();
    }
    budget.charge_operation()?;
    operation.results = captures
        .iter()
        .map(|capture| fresh_value(capture, capture.region.clone(), next_id))
        .collect::<Result<_, _>>()?;
    let names = operation
        .attributes
        .iter()
        .find(|a| a.name == "capture_names")
        .and_then(|a| a.value.clone())
        .ok_or(AdjointLoweringError::Malformed("return capture names"))?;
    let outputs = operation
        .attributes
        .iter_mut()
        .find(|a| a.name == "output_names")
        .ok_or(AdjointLoweringError::Malformed("return output names"))?;
    outputs.value = Some(names);
    let terminator = block.terminator.as_mut().expect("validated terminator");
    terminator.kind = "return".into();
    terminator.operands = vec![operation.results[*owner].id];
    Ok(true)
}
