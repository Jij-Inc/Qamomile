//! Resolve primitive selector syntax independently of optional optimizer passes.

use super::primitive_gate_spec;
use crate::wire;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum SelectorOperand {
    Value(u64),
    Default(i64),
}

/// Resolve positional, keyword, and default selectors without requiring values.
pub(crate) fn selector_operands(operation: &wire::OperationProto) -> Option<Vec<SelectorOperand>> {
    let spec = primitive_gate_spec(&operation.opcode)?;
    let classical = operation
        .operands
        .get(1 + usize::from(spec.has_angle())..)?;
    let keyword_names = string_tuple_attribute(operation, "keywords")?;
    if keyword_names.values.len() > classical.len() {
        return None;
    }
    let positional_count = classical.len() - keyword_names.values.len();
    let any_selector = !classical.is_empty();
    if spec.broadcasts_when_unselected() && !any_selector {
        return None;
    }
    let mut selectors = Vec::with_capacity(spec.selectors().len());
    for (index, name) in spec.selectors().iter().enumerate() {
        let identifier = if index < positional_count {
            classical.get(index)
        } else {
            keyword_names
                .values
                .iter()
                .position(|literal| {
                    matches!(
                        literal.value.as_ref(),
                        Some(wire::literal_proto::Value::StringValue(value)) if value == name
                    )
                })
                .and_then(|position| classical.get(positional_count + position))
        };
        selectors.push(match identifier {
            Some(identifier) => SelectorOperand::Value(*identifier),
            None => SelectorOperand::Default(i64::try_from(spec.selector_default(index)?).ok()?),
        });
    }
    Some(selectors)
}

fn string_tuple_attribute<'a>(
    operation: &'a wire::OperationProto,
    name: &str,
) -> Option<&'a wire::literal_proto::Sequence> {
    let wire::literal_proto::Value::TupleValue(sequence) = operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)?
        .value
        .as_ref()?
        .value
        .as_ref()?
    else {
        return None;
    };
    sequence
        .values
        .iter()
        .all(|literal| {
            matches!(
                literal.value.as_ref(),
                Some(wire::literal_proto::Value::StringValue(_))
            )
        })
        .then_some(sequence)
}
