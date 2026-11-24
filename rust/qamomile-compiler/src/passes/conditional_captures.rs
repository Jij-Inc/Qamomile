//! Prune unused Const captures before target lowering.

use std::collections::HashMap;

use crate::types::{Stage, TypeArena, TypeId};
use crate::{CompilerError, VerifiedProgram, seal_package, wire};

pub(crate) fn prepare(program: VerifiedProgram) -> Result<VerifiedProgram, CompilerError> {
    let checked = &program.local.checked;
    let source = &checked.package;
    if !source
        .functions
        .iter()
        .filter_map(|f| f.body.as_ref())
        .any(contains_measured_conditional)
    {
        return Ok(program);
    }

    let mut package = source.clone();
    let mut changed = false;
    for function in &mut package.functions {
        if let Some(body) = &mut function.body {
            let value_types = checked
                .operations
                .value_types(&function.symbol)
                .expect("verified function value types");
            changed |= prune_block(body, false, &checked.type_arena, value_types);
        }
    }
    if changed {
        VerifiedProgram::from_bytes(&seal_package(package)?)
    } else {
        Ok(program)
    }
}

fn contains_measured_conditional(block: &wire::BlockProto) -> bool {
    block.operations.iter().any(|operation| {
        is_measured_conditional(operation)
            || operation
                .regions
                .iter()
                .flat_map(|region| &region.blocks)
                .any(contains_measured_conditional)
    })
}

fn is_measured_conditional(operation: &wire::OperationProto) -> bool {
    operation.opcode == "qcore.if" && operation.attributes.iter().any(|attribute| {
        attribute.name == "stage"
            && matches!(
                attribute.value.as_ref().and_then(|literal| literal.value.as_ref()),
                Some(wire::literal_proto::Value::StringValue(stage)) if stage == Stage::Shot.as_str()
            )
    })
}

fn prune_block(
    block: &mut wire::BlockProto,
    inside_feedback: bool,
    arena: &TypeArena,
    value_types: &HashMap<u64, TypeId>,
) -> bool {
    let mut changed = false;
    let mut forwarded = HashMap::new();
    for operation in &mut block.operations {
        substitute_forwarded_captures(&mut operation.operands, &forwarded);
        let feedback = inside_feedback || is_measured_conditional(operation);
        for nested in operation
            .regions
            .iter_mut()
            .flat_map(|region| &mut region.blocks)
        {
            changed |= prune_block(nested, feedback, arena, value_types);
        }
        if feedback && operation.opcode == "qcore.if" {
            changed |= prune_unused_const_captures(operation, &mut forwarded, arena, value_types);
        }
    }
    if let Some(terminator) = &mut block.terminator {
        substitute_forwarded_captures(&mut terminator.operands, &forwarded);
    }
    changed
}

/// Replace uses of results already resolved to a dominating capture.
fn substitute_forwarded_captures(operands: &mut [u64], forwarded: &HashMap<u64, u64>) {
    if !forwarded.is_empty() {
        for operand in operands {
            if let Some(source) = forwarded.get(operand) {
                *operand = *source;
            }
        }
    }
}

/// Prune unused Const captures and any unchanged outputs that forward them.
///
/// Branches have explicit SSA capture scopes and have already been pruned.
/// Scan their immediate uses once. The transient result is rechecked before
/// emission, including the arities and types of the remaining rows.
fn prune_unused_const_captures(
    operation: &mut wire::OperationProto,
    forwarded: &mut HashMap<u64, u64>,
    arena: &TypeArena,
    value_types: &HashMap<u64, TypeId>,
) -> bool {
    let then_block = &operation.regions[0].blocks[0];
    let else_block = &operation.regions[1].blocks[0];
    let then_exit = then_block.terminator.as_ref().expect("verified then exit");
    let else_exit = else_block.terminator.as_ref().expect("verified else exit");

    let mut keep_captures = vec![true; then_block.arguments.len()];
    let mut const_slots = HashMap::new();
    for (slot, (left, right)) in then_block
        .arguments
        .iter()
        .zip(&else_block.arguments)
        .enumerate()
    {
        let ty = value_types[&left.id];
        if !arena.is_quantum(ty) && arena.latest_availability(ty) == Some(Stage::Const) {
            const_slots.insert(left.id, slot);
            const_slots.insert(right.id, slot);
            keep_captures[slot] = false;
        }
    }
    if const_slots.is_empty() {
        return false;
    }

    let mut output_sources = vec![None; operation.results.len()];
    if then_exit.kind == "yield" && else_exit.kind == "yield" {
        for (index, (&left, &right)) in then_exit
            .operands
            .iter()
            .zip(&else_exit.operands)
            .enumerate()
        {
            if let Some(&slot) = const_slots.get(&left)
                && right == else_block.arguments[slot].id
                && operation.results[index].r#type == then_block.arguments[slot].r#type
            {
                output_sources[index] = Some(slot);
            }
        }
    }

    for (block, exit) in [(then_block, then_exit), (else_block, else_exit)] {
        let operand_uses = block.operations.iter().flat_map(|nested| &nested.operands);
        let exit_uses = exit
            .operands
            .iter()
            .enumerate()
            .filter_map(|(index, operand)| {
                (exit.kind != "yield" || output_sources[index].is_none()).then_some(operand)
            });
        for operand in operand_uses.chain(exit_uses) {
            if let Some(&slot) = const_slots.get(operand) {
                keep_captures[slot] = true;
            }
        }
    }
    if keep_captures.iter().all(|&keep| keep) {
        return false;
    }

    let keep_outputs = operation
        .results
        .iter()
        .zip(output_sources)
        .map(|(result, source)| {
            if let Some(slot) = source
                && !keep_captures[slot]
            {
                forwarded.insert(result.id, operation.operands[slot + 1]);
                false
            } else {
                true
            }
        })
        .collect::<Vec<_>>();
    retain_conditional_slots(operation, &keep_captures, &keep_outputs);
    true
}

fn retain_conditional_slots(
    operation: &mut wire::OperationProto,
    keep_captures: &[bool],
    keep_outputs: &[bool],
) {
    for block in operation
        .regions
        .iter_mut()
        .flat_map(|region| &mut region.blocks)
    {
        retain_slots(&mut block.arguments, keep_captures.iter().copied());
        let exit = block
            .terminator
            .as_mut()
            .expect("verified conditional exit");
        if exit.kind == "yield" {
            retain_slots(&mut exit.operands, keep_outputs.iter().copied());
        }
    }
    retain_slots(
        &mut operation.operands,
        std::iter::once(true).chain(keep_captures.iter().copied()),
    );
    retain_slots(&mut operation.results, keep_outputs.iter().copied());
    // Cleanup replay compares the inverse's saved producer interface with the
    // forward conditional. Both copies of each name row follow the same mask.
    let retain_names = |name: &str, literal: &mut Option<wire::LiteralProto>| {
        let keep = match name {
            "capture_names" => keep_captures,
            "output_names" => keep_outputs,
            _ => return,
        };
        if let Some(wire::literal_proto::Value::TupleValue(names)) =
            literal.as_mut().and_then(|literal| literal.value.as_mut())
            && names.values.len() == keep.len()
        {
            retain_slots(&mut names.values, keep.iter().copied());
        }
    };
    for attribute in &mut operation.attributes {
        if attribute.name == "original_attributes"
            && let Some(wire::literal_proto::Value::MappingValue(original)) = attribute
                .value
                .as_mut()
                .and_then(|literal| literal.value.as_mut())
        {
            for entry in &mut original.entries {
                retain_names(&entry.key, &mut entry.value);
            }
        } else {
            retain_names(&attribute.name, &mut attribute.value);
        }
    }
}

fn retain_slots<T>(row: &mut Vec<T>, mut keep: impl Iterator<Item = bool>) {
    row.retain(|_| keep.next().expect("verified conditional row length"));
}
