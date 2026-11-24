use super::{HirError, HirValidationWork, validate_expressions};
use crate::{
    MAX_QCORE_BYTES, budget,
    equations::literal_conforms,
    literals,
    types::{Dimension, ScalarType, SemanticType, TypeArena, TypeId},
    wire,
};
use prost::Message;
use std::sync::Arc;

/// Infer preparation from the original serialized source operand.
///
/// # Errors
///
/// Rejects malformed HIR and unsupported logical preparation data.
pub fn infer_preparation_result_type(source: &[u8]) -> Result<Vec<u8>, HirError> {
    if source.len() > MAX_QCORE_BYTES {
        return Err(HirError::PackageTooLarge);
    }
    budget::preflight_hir_expression(source).map_err(|error| match error {
        budget::PreflightError::Budget(reason) => HirError::PackageBudgetExceeded(reason),
        budget::PreflightError::Malformed(reason) => HirError::MalformedPackage(reason),
    })?;
    let operand = wire::HirExpressionProto::decode(source)
        .map_err(|error| HirError::MalformedPackage(error.to_string()))?;
    if operand.encode_to_vec() != source {
        return Err(HirError::NonCanonicalEncoding);
    }
    let mut arena = TypeArena::default();
    let scope = arena.intern_scope("source");
    let mut literal_values = Vec::new();
    validate_expressions(
        vec![&operand],
        None,
        &scope,
        &mut arena,
        &mut literal_values,
        &mut HirValidationWork::default(),
    )?;
    literals::validate_literal_references(literal_values)
        .map_err(HirError::InvalidPortableLiteral)?;
    let result = infer_preparation_result(&mut arena, &operand, &scope)?;
    arena
        .encode_wire_type(result)
        .map(|value| value.encode_to_vec())
        .map_err(HirError::InvalidSemanticType)
}

pub(super) fn infer_preparation_result(
    arena: &mut TypeArena,
    operand: &wire::HirExpressionProto,
    scope: &Arc<str>,
) -> Result<TypeId, HirError> {
    let invalid = || {
        HirError::InvalidStructure(
            "qm.prepare requires logical bit data or a binary integer".to_owned(),
        )
    };
    validate_operand_literals(arena, operand, scope)?;
    let original = arena
        .resolve_wire_type(operand.r#type.as_ref().ok_or_else(invalid)?, scope)
        .map_err(HirError::InvalidSemanticType)?;
    if arena.is_quantum(original) {
        return Err(invalid());
    }
    let inner = arena.strip_stage(original);
    let payload = if arena.get(inner) == Some(&SemanticType::Scalar(ScalarType::Int)) {
        arena
            .intern_derived(SemanticType::Scalar(ScalarType::Bit))
            .map_err(HirError::InvalidSemanticType)?
    } else if matches!(operand.opcode.as_str(), "tuple" | "array") {
        if operand.operands.is_empty() || !operand.attributes.is_empty() {
            return Err(invalid());
        }
        if arena.aggregate_len(original) != Some(operand.operands.len()) {
            return Err(invalid());
        }
        for (index, item) in operand.operands.iter().enumerate() {
            let element = arena
                .aggregate_element_at(original, index)
                .map_err(HirError::InvalidSemanticType)?;
            let item_type = arena
                .resolve_wire_type(item.r#type.as_ref().ok_or_else(invalid)?, scope)
                .map_err(HirError::InvalidSemanticType)?;
            if element != Some(item_type) || !binary_literal(arena, item, scope)? {
                return Err(invalid());
            }
        }
        arena
            .intern_derived(SemanticType::Word {
                kind: crate::types::WordKind::Bits,
                width: Dimension::Fixed(
                    u64::try_from(operand.operands.len()).map_err(|_| invalid())?,
                ),
            })
            .map_err(HirError::InvalidSemanticType)?
    } else {
        if arena
            .logical_width(inner)
            .map_err(HirError::InvalidSemanticType)?
            .is_none()
        {
            return Err(invalid());
        }
        inner
    };
    arena
        .intern_derived(SemanticType::Quantum(payload))
        .map_err(HirError::InvalidSemanticType)
}

fn binary_literal(
    arena: &mut TypeArena,
    operand: &wire::HirExpressionProto,
    scope: &Arc<str>,
) -> Result<bool, HirError> {
    if operand.opcode != "literal"
        || !operand.operands.is_empty()
        || !operand.regions.is_empty()
        || operand.attributes.len() != 1
        || operand.attributes[0].name != "value"
    {
        return Ok(false);
    }
    let Some(value) = operand.attributes[0].value.as_ref() else {
        return Ok(false);
    };
    let Some(source_type) = operand.r#type.as_ref() else {
        return Ok(false);
    };
    let type_id = arena
        .resolve_wire_type(source_type, scope)
        .map_err(HirError::InvalidSemanticType)?;
    Ok(matches!(
        arena.get(type_id),
        Some(SemanticType::Scalar(
            ScalarType::Bool | ScalarType::Bit | ScalarType::Int
        ))
    ) && literal_conforms(arena, type_id, value)
        && matches!(
            value.value,
            Some(
                wire::literal_proto::Value::BoolValue(_)
                    | wire::literal_proto::Value::IntValue(0 | 1)
            )
        ))
}

fn validate_operand_literals(
    arena: &mut TypeArena,
    root: &wire::HirExpressionProto,
    scope: &Arc<str>,
) -> Result<(), HirError> {
    let mut pending = vec![root];
    while let Some(operand) = pending.pop() {
        if matches!(operand.opcode.as_str(), "literal" | "module_constant") {
            let value = operand
                .attributes
                .iter()
                .find(|attribute| attribute.name == "value")
                .and_then(|attribute| attribute.value.as_ref())
                .ok_or_else(|| {
                    HirError::InvalidStructure("source literal has no value".to_owned())
                })?;
            let type_id = arena
                .resolve_wire_type(
                    operand.r#type.as_ref().ok_or_else(|| {
                        HirError::InvalidStructure("source literal has no type".to_owned())
                    })?,
                    scope,
                )
                .map_err(HirError::InvalidSemanticType)?;
            if !operand.operands.is_empty()
                || !operand.regions.is_empty()
                || (operand.opcode == "literal" && operand.attributes.len() != 1)
                || !literal_conforms(arena, type_id, value)
            {
                return Err(HirError::InvalidStructure(
                    "source literal disagrees with its type".to_owned(),
                ));
            }
        }
        pending.extend(&operand.operands);
        pending.extend(&operand.regions);
    }
    Ok(())
}

pub(super) fn lower_integer_operand(
    arena: &mut TypeArena,
    operand: &wire::ValueProto,
    scope: &Arc<str>,
    next_id: &mut u64,
    operations: &mut Vec<wire::OperationProto>,
    span: Option<&wire::SourceSpanProto>,
) -> Result<Option<wire::ValueProto>, super::HirLoweringError> {
    let invalid = super::HirLoweringError::InvalidQCore;
    let original = arena
        .resolve_wire_type(
            operand
                .r#type
                .as_ref()
                .ok_or_else(|| invalid("preparation operand has no type".to_owned()))?,
            scope,
        )
        .map_err(invalid)?;
    if arena.get(arena.strip_stage(original)) != Some(&SemanticType::Scalar(ScalarType::Int)) {
        return Ok(None);
    }
    let bit = arena
        .intern_derived(SemanticType::Scalar(ScalarType::Bit))
        .map_err(invalid)?;
    let result = if let Some(stage) = arena.stage_of(original) {
        arena
            .intern_derived(SemanticType::Staged { stage, inner: bit })
            .map_err(invalid)?
    } else {
        bit
    };
    let value = wire::ValueProto {
        id: *next_id,
        r#type: Some(arena.encode_wire_type(result).map_err(invalid)?),
        hint: "prepared_bit".to_owned(),
        ..Default::default()
    };
    *next_id = next_id
        .checked_add(1)
        .ok_or_else(|| invalid("preparation identifier overflow".to_owned()))?;
    operations.push(wire::OperationProto {
        opcode: "classical.cast".to_owned(),
        operands: vec![operand.id],
        results: vec![value.clone()],
        attributes: vec![wire::AttributeProto {
            name: "target".to_owned(),
            value: Some(literals::string_literal("Bit")),
        }],
        span: span.cloned(),
        ..Default::default()
    });
    Ok(Some(value))
}
