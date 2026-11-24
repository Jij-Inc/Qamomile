//! Logical and symbolic resource analysis and persistent studies.

pub(crate) mod analysis;
pub(crate) mod portable;
pub(crate) mod resource_expression_evaluation;
pub(crate) mod resource_rules;
pub(crate) mod resource_study;
pub(crate) mod symbolic_resources;
pub(crate) mod timing;

/// Read the extent of a verified collection without inspecting its elements.
fn length_dimension(
    types: &crate::types::TypeArena,
    operations: &crate::operations::LocallyCheckedOperations,
    owner: &str,
    operation: &crate::wire::OperationProto,
) -> Option<crate::types::Dimension> {
    use crate::types::{Dimension, SemanticType};

    let source = operations
        .value_types(owner)?
        .get(operation.operands.first()?)?;
    let base = types.strip_stage(*source);
    match types.get(base)? {
        SemanticType::Array { shape, .. } => shape.first().cloned(),
        SemanticType::Word { width, .. } => Some(width.clone()),
        _ => Some(Dimension::Fixed(
            u64::try_from(types.aggregate_len(base)?).ok()?,
        )),
    }
}

fn dimension_conflict(parameter: &str, expected: u64) -> String {
    format!("dimension argument {parameter:?} disagrees with its verified width {expected}")
}
