//! Unambiguous attribute and intent rows at the persistent `QCore` boundary.
//!
//! This pass is deliberately syntax-only. Callable attribute schemas and
//! symbol resolution belong exclusively to the normalized static call graph.
//! Intent rows are bounded here and interpreted nowhere in the compiler: they
//! are erased from semantic identity and never read by verification, lowering,
//! or emission.

use std::collections::HashSet;

use prost::Message;

use crate::verification::QCoreVerificationWork;
use crate::wire;

/// Maximum intent rows on one function or operation.
pub const MAX_INTENT_ROWS: usize = 16;
/// Maximum bytes of one intent name.
pub const MAX_INTENT_NAME_BYTES: usize = 128;
/// Maximum encoded bytes of one intent literal.
pub const MAX_INTENT_VALUE_BYTES: usize = 4096;

/// Require unique operation and terminator attribute names throughout a program.
#[cfg(test)]
pub(crate) fn validate_unique_attributes(package: &wire::QCorePackageProto) -> Result<(), String> {
    let mut work = QCoreVerificationWork::default();
    validate_unique_attributes_with_work(package, &mut work)
}

pub(crate) fn validate_unique_attributes_with_work(
    package: &wire::QCorePackageProto,
    work: &mut QCoreVerificationWork,
) -> Result<(), String> {
    for function in &package.functions {
        require_bounded_intent(&function.intent, &function.symbol, work)?;
    }
    let mut blocks = package
        .functions
        .iter()
        .filter_map(|function| function.body.as_ref())
        .collect::<Vec<_>>();
    while let Some(block) = blocks.pop() {
        work.charge_attributes(1);
        if let Some(terminator) = &block.terminator {
            require_unique(&terminator.attributes, "terminator", work)?;
        }
        for operation in &block.operations {
            require_unique(&operation.attributes, &operation.opcode, work)?;
            require_bounded_intent(&operation.intent, &operation.opcode, work)?;
            for region in &operation.regions {
                blocks.extend(&region.blocks);
            }
        }
    }
    Ok(())
}

/// Bound intent rows without reading them: unique lowercase dotted names and
/// literals within a fixed encoded size.
fn require_bounded_intent(
    intent: &[wire::AttributeProto],
    owner: &str,
    work: &mut QCoreVerificationWork,
) -> Result<(), String> {
    work.charge_attributes(1_usize.saturating_add(intent.len()));
    if intent.len() > MAX_INTENT_ROWS {
        return Err(format!(
            "Q1060: invalid intent on {owner:?}: more than {MAX_INTENT_ROWS} rows"
        ));
    }
    let mut names = HashSet::with_capacity(intent.len());
    for row in intent {
        if row.name.is_empty()
            || row.name.len() > MAX_INTENT_NAME_BYTES
            || !row.name.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_' || byte == b'.'
            })
        {
            return Err(format!(
                "Q1060: invalid intent on {owner:?}: invalid name {:?}",
                row.name
            ));
        }
        if !names.insert(row.name.as_str()) {
            return Err(format!(
                "Q1060: invalid intent on {owner:?}: duplicate name {:?}",
                row.name
            ));
        }
        let encoded = row.value.as_ref().map_or(0, Message::encoded_len);
        if encoded > MAX_INTENT_VALUE_BYTES {
            return Err(format!(
                "Q1060: invalid intent on {owner:?}: value of {:?} exceeds {MAX_INTENT_VALUE_BYTES} bytes",
                row.name
            ));
        }
    }
    Ok(())
}

fn require_unique(
    attributes: &[wire::AttributeProto],
    owner: &str,
    work: &mut QCoreVerificationWork,
) -> Result<(), String> {
    work.charge_attributes(1_usize.saturating_add(attributes.len()));
    let mut names = HashSet::with_capacity(attributes.len());
    if attributes
        .iter()
        .any(|attribute| !names.insert(attribute.name.as_str()))
    {
        return Err(format!("Q1025: duplicate attributes on {owner:?}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn string(value: &str) -> wire::LiteralProto {
        wire::LiteralProto {
            value: Some(wire::literal_proto::Value::StringValue(value.to_owned())),
        }
    }

    fn program_with(operation: wire::OperationProto) -> wire::QCorePackageProto {
        wire::QCorePackageProto {
            functions: vec![wire::FunctionProto {
                body: Some(wire::BlockProto {
                    operations: vec![operation],
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn accepts_unique_attributes_without_interpreting_them() {
        let package = program_with(wire::OperationProto {
            opcode: "qcore.call".to_owned(),
            attributes: vec![wire::AttributeProto {
                name: "callee".to_owned(),
                value: Some(string("resolved.by.call.graph")),
            }],
            ..Default::default()
        });

        assert_eq!(validate_unique_attributes(&package), Ok(()));
    }

    #[test]
    fn rejects_duplicate_operation_attributes() {
        let attribute = wire::AttributeProto {
            name: "value".to_owned(),
            value: Some(string("one")),
        };
        let package = program_with(wire::OperationProto {
            opcode: "qcore.constant".to_owned(),
            attributes: vec![attribute.clone(), attribute],
            ..Default::default()
        });

        assert_eq!(
            validate_unique_attributes(&package),
            Err("Q1025: duplicate attributes on \"qcore.constant\"".to_owned())
        );
    }

    #[test]
    fn rejects_duplicate_terminator_attributes() {
        let attribute = wire::AttributeProto {
            name: "stage".to_owned(),
            value: Some(string("Shot")),
        };
        let package = wire::QCorePackageProto {
            functions: vec![wire::FunctionProto {
                body: Some(wire::BlockProto {
                    terminator: Some(wire::TerminatorProto {
                        attributes: vec![attribute.clone(), attribute],
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };

        assert_eq!(
            validate_unique_attributes(&package),
            Err("Q1025: duplicate attributes on \"terminator\"".to_owned())
        );
    }

    fn intent_row(name: &str, value: &str) -> wire::AttributeProto {
        wire::AttributeProto {
            name: name.to_owned(),
            value: Some(string(value)),
        }
    }

    fn function_with_intent(intent: Vec<wire::AttributeProto>) -> wire::QCorePackageProto {
        wire::QCorePackageProto {
            functions: vec![wire::FunctionProto {
                symbol: "m.f".to_owned(),
                intent,
                body: Some(wire::BlockProto::default()),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn accepts_bounded_intent_rows_on_functions_and_operations() {
        let rows = (0..MAX_INTENT_ROWS)
            .map(|index| intent_row(&format!("tolerance.row_{index}"), "x"))
            .collect::<Vec<_>>();
        assert_eq!(
            validate_unique_attributes(&function_with_intent(rows.clone())),
            Ok(())
        );
        let package = program_with(wire::OperationProto {
            opcode: "qcore.constant".to_owned(),
            intent: rows,
            ..Default::default()
        });
        assert_eq!(validate_unique_attributes(&package), Ok(()));
    }

    #[test]
    fn rejects_intent_that_exceeds_its_bounds() {
        let too_many = (0..=MAX_INTENT_ROWS)
            .map(|index| intent_row(&format!("row_{index}"), "x"))
            .collect::<Vec<_>>();
        assert_eq!(
            validate_unique_attributes(&function_with_intent(too_many)),
            Err("Q1060: invalid intent on \"m.f\": more than 16 rows".to_owned())
        );
        assert_eq!(
            validate_unique_attributes(&function_with_intent(vec![intent_row("Origin Name", "x")])),
            Err("Q1060: invalid intent on \"m.f\": invalid name \"Origin Name\"".to_owned())
        );
        assert_eq!(
            validate_unique_attributes(&function_with_intent(vec![
                intent_row("origin", "x"),
                intent_row("origin", "y"),
            ])),
            Err("Q1060: invalid intent on \"m.f\": duplicate name \"origin\"".to_owned())
        );
        let oversized = program_with(wire::OperationProto {
            opcode: "qcore.constant".to_owned(),
            intent: vec![intent_row("doc", &"x".repeat(MAX_INTENT_VALUE_BYTES))],
            ..Default::default()
        });
        assert_eq!(
            validate_unique_attributes(&oversized),
            Err(
                "Q1060: invalid intent on \"qcore.constant\": value of \"doc\" exceeds 4096 bytes"
                    .to_owned()
            )
        );
    }
}
