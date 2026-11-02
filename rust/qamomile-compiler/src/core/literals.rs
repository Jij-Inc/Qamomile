//! Canonical closed literal validation for persistent programs.

use crate::{verification::QCoreVerificationWork, wire};

pub(crate) fn tuple_literal(
    values: impl IntoIterator<Item = wire::LiteralProto>,
) -> wire::LiteralProto {
    wire::LiteralProto {
        value: Some(wire::literal_proto::Value::TupleValue(
            wire::literal_proto::Sequence {
                values: values.into_iter().collect(),
            },
        )),
    }
}

pub(crate) fn string_literal(value: &str) -> wire::LiteralProto {
    wire::LiteralProto {
        value: Some(wire::literal_proto::Value::StringValue(value.to_owned())),
    }
}

pub(crate) fn bytes_literal(value: &[u8]) -> wire::LiteralProto {
    wire::LiteralProto {
        value: Some(wire::literal_proto::Value::BytesValue(value.to_vec())),
    }
}

pub(crate) fn int_literal(value: i64) -> wire::LiteralProto {
    wire::LiteralProto {
        value: Some(wire::literal_proto::Value::IntValue(value)),
    }
}

pub(crate) fn bool_literal(value: bool) -> wire::LiteralProto {
    wire::LiteralProto {
        value: Some(wire::literal_proto::Value::BoolValue(value)),
    }
}

pub(crate) fn tuple_values(value: &wire::LiteralProto) -> Option<&[wire::LiteralProto]> {
    match value.value.as_ref()? {
        wire::literal_proto::Value::TupleValue(sequence) => Some(&sequence.values),
        _ => None,
    }
}

pub(crate) fn string_value(value: &wire::LiteralProto) -> Option<&str> {
    match value.value.as_ref()? {
        wire::literal_proto::Value::StringValue(value) => Some(value),
        _ => None,
    }
}

pub(crate) fn bytes_value(value: &wire::LiteralProto) -> Option<&[u8]> {
    match value.value.as_ref()? {
        wire::literal_proto::Value::BytesValue(value) => Some(value),
        _ => None,
    }
}

pub(crate) fn int_value(value: &wire::LiteralProto) -> Option<i64> {
    match value.value.as_ref()? {
        wire::literal_proto::Value::IntValue(value) => Some(*value),
        _ => None,
    }
}

pub(crate) fn bool_value(value: &wire::LiteralProto) -> Option<bool> {
    match value.value.as_ref()? {
        wire::literal_proto::Value::BoolValue(value) => Some(*value),
        _ => None,
    }
}

/// Validate every portable literal occurrence without materializing Python data.
#[cfg(test)]
pub(crate) fn validate_program_literals(package: &wire::QCorePackageProto) -> Result<(), String> {
    let mut work = QCoreVerificationWork::default();
    validate_program_literals_with_work(package, &mut work)
}

pub(crate) fn validate_program_literals_with_work(
    package: &wire::QCorePackageProto,
    work: &mut QCoreVerificationWork,
) -> Result<(), String> {
    let mut literals = Vec::new();
    for constant in &package.constants {
        literals.push(required(
            constant.value.as_ref(),
            "Q4007: module constant has no selected portable value",
        )?);
    }
    for binding in &package.const_bindings {
        literals.push(required(
            binding.value.as_ref(),
            "Q4007: constant binding has no selected portable value",
        )?);
    }

    let mut blocks = Vec::new();
    for function in &package.functions {
        for parameter in &function.parameters {
            match (parameter.has_default, parameter.default_value.as_ref()) {
                (true, Some(value)) => literals.push(value),
                (true, None) => {
                    return Err(
                        "Q4007: parameter default has no selected portable value".to_owned()
                    );
                }
                (false, Some(_)) => {
                    return Err(
                        "Q4008: parameter carries a default while has_default is false".to_owned(),
                    );
                }
                (false, None) => {}
            }
        }
        if let Some(body) = &function.body {
            blocks.push(body);
        }
    }
    while let Some(block) = blocks.pop() {
        if let Some(terminator) = &block.terminator {
            collect_attributes(&terminator.attributes, &mut literals)?;
        }
        for operation in &block.operations {
            collect_attributes(&operation.attributes, &mut literals)?;
            for region in &operation.regions {
                blocks.extend(&region.blocks);
            }
        }
    }

    validate_literal_references_with_work(literals, work)
}

/// Validate a collection of portable literals from another private compiler boundary.
pub(crate) fn validate_literal_references(
    literals: Vec<&wire::LiteralProto>,
) -> Result<(), String> {
    let mut work = QCoreVerificationWork::default();
    validate_literal_references_with_work(literals, &mut work)
}

fn validate_literal_references_with_work(
    mut literals: Vec<&wire::LiteralProto>,
    work: &mut QCoreVerificationWork,
) -> Result<(), String> {
    use wire::literal_proto::Value;

    while let Some(literal) = literals.pop() {
        work.visit_literal();

        match literal.value.as_ref() {
            None => return Err("Q4007: portable literal has no selected value".to_owned()),
            Some(
                Value::UnitValue(true)
                | Value::BoolValue(_)
                | Value::IntValue(_)
                | Value::RealValue(_)
                | Value::ComplexValue(_)
                | Value::StringValue(_)
                | Value::BytesValue(_),
            ) => {}
            Some(Value::UnitValue(false)) => {
                return Err("Q4008: unit literal is not canonical".to_owned());
            }
            Some(Value::BigintValue(encoded)) => validate_bigint(encoded)?,
            Some(Value::TupleValue(sequence)) => literals.extend(&sequence.values),
            Some(Value::MappingValue(mapping)) => {
                let mut previous: Option<&str> = None;
                for entry in &mapping.entries {
                    work.visit_literal();
                    if previous.is_some_and(|key| key >= entry.key.as_str()) {
                        return Err("Q4006: mapping literal is not canonical".to_owned());
                    }
                    previous = Some(&entry.key);
                    literals.push(required(
                        entry.value.as_ref(),
                        "Q4007: mapping entry has no selected portable value",
                    )?);
                }
            }
        }
    }
    Ok(())
}

fn collect_attributes<'a>(
    attributes: &'a [wire::AttributeProto],
    literals: &mut Vec<&'a wire::LiteralProto>,
) -> Result<(), String> {
    for attribute in attributes {
        literals.push(required(
            attribute.value.as_ref(),
            "Q4007: operation attribute has no selected portable value",
        )?);
    }
    Ok(())
}

fn required<'a, T>(value: Option<&'a T>, error: &str) -> Result<&'a T, String> {
    value.ok_or_else(|| error.to_owned())
}

fn validate_bigint(encoded: &[u8]) -> Result<(), String> {
    if encoded.is_empty() {
        return Err("Q4008: bigint literal is empty".to_owned());
    }
    let redundant_positive = encoded.len() > 1 && encoded[0] == 0 && encoded[1] < 0x80;
    let redundant_negative = encoded.len() > 1 && encoded[0] == 0xff && encoded[1] >= 0x80;
    if redundant_positive || redundant_negative || encoded.len() <= size_of::<i64>() {
        return Err("Q4008: bigint literal is not canonical".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn package_with_binding(value: wire::LiteralProto) -> wire::QCorePackageProto {
        wire::QCorePackageProto {
            const_bindings: vec![wire::BindingProto {
                name: "value".to_owned(),
                value: Some(value),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn rejects_redundant_bigints() {
        let package = package_with_binding(wire::LiteralProto {
            value: Some(wire::literal_proto::Value::BigintValue(vec![0, 0x80])),
        });

        assert_eq!(
            validate_program_literals(&package),
            Err("Q4008: bigint literal is not canonical".to_owned())
        );
    }

    #[test]
    fn accepts_minimal_values_outside_i64() {
        let package = package_with_binding(wire::LiteralProto {
            value: Some(wire::literal_proto::Value::BigintValue(vec![
                0, 0x80, 0, 0, 0, 0, 0, 0, 0,
            ])),
        });

        assert_eq!(validate_program_literals(&package), Ok(()));
    }

    #[test]
    fn rejects_unsorted_mapping_keys() {
        let scalar = || wire::LiteralProto {
            value: Some(wire::literal_proto::Value::IntValue(1)),
        };
        let package = package_with_binding(wire::LiteralProto {
            value: Some(wire::literal_proto::Value::MappingValue(
                wire::literal_proto::Mapping {
                    entries: vec![
                        wire::literal_proto::Entry {
                            key: "z".to_owned(),
                            value: Some(scalar()),
                        },
                        wire::literal_proto::Entry {
                            key: "a".to_owned(),
                            value: Some(scalar()),
                        },
                    ],
                },
            )),
        });

        assert_eq!(
            validate_program_literals(&package),
            Err("Q4006: mapping literal is not canonical".to_owned())
        );
    }
}
