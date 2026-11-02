//! Lightweight constant dataflow for primitive static semantic domains.
//!
//! This pass intentionally tracks only the sign of canonical integer
//! constants. That is sufficient to reject zero loop steps and non-positive
//! allocation widths without materializing arbitrary-precision integers.

use std::collections::HashMap;

use thiserror::Error;

use crate::verification::QCoreVerificationWork;
use crate::wire;

/// A primitive static-domain violation.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub(crate) enum StaticDomainError {
    /// A statically known range step is zero.
    #[error("Q1045: for-loop step cannot be zero")]
    ZeroForStep,
    /// A statically known quantum allocation width is not positive.
    #[error("Q1046: allocation width must be positive")]
    NonPositiveAllocationWidth,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IntegerSign {
    Negative,
    Zero,
    Positive,
}

/// Reject statically known values outside primitive semantic domains.
///
/// Direct constants survive availability lifts and explicit structured-region
/// captures. Unknown computations remain unknown; this pass never guesses a
/// value from its declared type.
///
/// # Errors
///
/// Returns [`StaticDomainError`] for a known-zero `qcore.for` step or a known
/// non-positive `quantum.alloc` width at any region nesting depth.
#[cfg(test)]
pub(crate) fn verify_static_domains(
    package: &wire::QCorePackageProto,
) -> Result<(), StaticDomainError> {
    let mut work = QCoreVerificationWork::default();
    verify_static_domains_with_work(package, true, &mut work)
}

pub(crate) fn verify_static_domains_with_work(
    package: &wire::QCorePackageProto,
    use_entry_bindings: bool,
    work: &mut QCoreVerificationWork,
) -> Result<(), StaticDomainError> {
    for function in &package.functions {
        work.charge_static_domains(1);
        let Some(body) = &function.body else {
            continue;
        };
        let initial = if use_entry_bindings && function.symbol == package.entrypoint {
            entry_constants(function, package, work)
        } else {
            HashMap::new()
        };
        let mut pending = vec![(body, initial)];
        while let Some((block, mut known)) = pending.pop() {
            work.charge_static_domains(1);
            for operation in &block.operations {
                work.charge_static_domains(1);
                match operation.opcode.as_str() {
                    "qcore.constant"
                        if operation.operands.is_empty() && operation.results.len() == 1 =>
                    {
                        if let Some(sign) = integer_attribute(operation, "value") {
                            known.insert(operation.results[0].id, sign);
                        }
                    }
                    "qcore.stage_lift"
                        if operation.operands.len() == 1 && operation.results.len() == 1 =>
                    {
                        if let Some(sign) = known.get(&operation.operands[0]).copied() {
                            known.insert(operation.results[0].id, sign);
                        }
                    }
                    "qcore.for" if operation.operands.len() >= 3 => {
                        if known.get(&operation.operands[2]) == Some(&IntegerSign::Zero) {
                            return Err(StaticDomainError::ZeroForStep);
                        }
                    }
                    "quantum.alloc" if !operation.operands.is_empty() => {
                        if matches!(
                            known.get(&operation.operands[0]),
                            Some(IntegerSign::Negative | IntegerSign::Zero)
                        ) {
                            return Err(StaticDomainError::NonPositiveAllocationWidth);
                        }
                    }
                    _ => {}
                }
                enqueue_regions(operation, &known, &mut pending, work);
            }
        }
    }
    Ok(())
}

fn entry_constants(
    function: &wire::FunctionProto,
    package: &wire::QCorePackageProto,
    work: &mut QCoreVerificationWork,
) -> HashMap<u64, IntegerSign> {
    work.charge_static_domains(
        package
            .const_bindings
            .len()
            .saturating_add(function.parameters.len()),
    );
    let bindings = package
        .const_bindings
        .iter()
        .filter_map(|binding| {
            integer_literal(binding.value.as_ref()?).map(|sign| (&binding.name, sign))
        })
        .collect::<HashMap<_, _>>();
    function
        .parameters
        .iter()
        .filter_map(|parameter| {
            let sign = bindings.get(&parameter.name)?;
            Some((parameter.value.as_ref()?.id, *sign))
        })
        .collect()
}

fn enqueue_regions<'a>(
    operation: &'a wire::OperationProto,
    known: &HashMap<u64, IntegerSign>,
    pending: &mut Vec<(&'a wire::BlockProto, HashMap<u64, IntegerSign>)>,
    work: &mut QCoreVerificationWork,
) {
    match operation.opcode.as_str() {
        "qcore.if" => {
            for region in operation.regions.iter().rev() {
                for block in region.blocks.iter().rev() {
                    pending.push((
                        block,
                        captured_constants(
                            &block.arguments,
                            operation.operands.get(1..).unwrap_or_default(),
                            known,
                            work,
                        ),
                    ));
                }
            }
        }
        "qcore.for" => {
            for (region_index, region) in operation.regions.iter().enumerate().rev() {
                for block in region.blocks.iter().rev() {
                    let nested = if region_index == 0 && !block.arguments.is_empty() {
                        captured_constants(
                            &block.arguments[1..],
                            operation.operands.get(3..).unwrap_or_default(),
                            known,
                            work,
                        )
                    } else {
                        HashMap::new()
                    };
                    pending.push((block, nested));
                }
            }
        }
        _ => {
            for region in operation.regions.iter().rev() {
                for block in region.blocks.iter().rev() {
                    pending.push((block, HashMap::new()));
                }
            }
        }
    }
}

fn captured_constants(
    arguments: &[wire::ValueProto],
    sources: &[u64],
    known: &HashMap<u64, IntegerSign>,
    work: &mut QCoreVerificationWork,
) -> HashMap<u64, IntegerSign> {
    work.charge_static_domains(arguments.len().min(sources.len()));
    arguments
        .iter()
        .zip(sources)
        .filter_map(|(argument, source)| known.get(source).map(|sign| (argument.id, *sign)))
        .collect()
}

fn integer_attribute(operation: &wire::OperationProto, name: &str) -> Option<IntegerSign> {
    if operation.attributes.len() != 1 || operation.attributes[0].name != name {
        return None;
    }
    integer_literal(operation.attributes[0].value.as_ref()?)
}

fn integer_literal(literal: &wire::LiteralProto) -> Option<IntegerSign> {
    use wire::literal_proto::Value;

    let value = literal.value.as_ref()?;
    match value {
        Value::IntValue(value) => Some(match value.cmp(&0) {
            std::cmp::Ordering::Less => IntegerSign::Negative,
            std::cmp::Ordering::Equal => IntegerSign::Zero,
            std::cmp::Ordering::Greater => IntegerSign::Positive,
        }),
        Value::BigintValue(encoded) => Some(if encoded.first()? & 0x80 == 0 {
            IntegerSign::Positive
        } else {
            IntegerSign::Negative
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integer(value: i64) -> wire::LiteralProto {
        wire::LiteralProto {
            value: Some(wire::literal_proto::Value::IntValue(value)),
        }
    }

    fn value(id: u64) -> wire::ValueProto {
        wire::ValueProto {
            id,
            ..Default::default()
        }
    }

    fn constant(id: u64, literal: wire::LiteralProto) -> wire::OperationProto {
        wire::OperationProto {
            opcode: "qcore.constant".to_owned(),
            results: vec![value(id)],
            attributes: vec![wire::AttributeProto {
                name: "value".to_owned(),
                value: Some(literal),
            }],
            ..Default::default()
        }
    }

    fn function(operations: Vec<wire::OperationProto>) -> wire::FunctionProto {
        wire::FunctionProto {
            body: Some(wire::BlockProto {
                operations,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn rejects_a_zero_step_after_an_availability_lift() {
        let package = wire::QCorePackageProto {
            functions: vec![function(vec![
                constant(1, integer(0)),
                wire::OperationProto {
                    opcode: "qcore.stage_lift".to_owned(),
                    operands: vec![1],
                    results: vec![value(2)],
                    ..Default::default()
                },
                wire::OperationProto {
                    opcode: "qcore.for".to_owned(),
                    operands: vec![10, 11, 2],
                    ..Default::default()
                },
            ])],
            ..Default::default()
        };

        assert_eq!(
            verify_static_domains(&package),
            Err(StaticDomainError::ZeroForStep)
        );
    }

    #[test]
    fn rejects_a_non_positive_width_through_an_explicit_capture() {
        let allocate = wire::OperationProto {
            opcode: "quantum.alloc".to_owned(),
            operands: vec![4],
            ..Default::default()
        };
        let nested = || wire::RegionProto {
            blocks: vec![wire::BlockProto {
                arguments: vec![value(4)],
                operations: vec![allocate.clone()],
                ..Default::default()
            }],
        };
        let package = wire::QCorePackageProto {
            functions: vec![function(vec![
                constant(1, integer(-2)),
                wire::OperationProto {
                    opcode: "qcore.stage_lift".to_owned(),
                    operands: vec![1],
                    results: vec![value(2)],
                    ..Default::default()
                },
                wire::OperationProto {
                    opcode: "qcore.if".to_owned(),
                    operands: vec![3, 2],
                    regions: vec![nested(), nested()],
                    ..Default::default()
                },
            ])],
            ..Default::default()
        };

        assert_eq!(
            verify_static_domains(&package),
            Err(StaticDomainError::NonPositiveAllocationWidth)
        );
    }

    #[test]
    fn does_not_treat_unknown_integer_computation_as_a_constant() {
        let package = wire::QCorePackageProto {
            functions: vec![function(vec![wire::OperationProto {
                opcode: "quantum.alloc".to_owned(),
                operands: vec![99],
                ..Default::default()
            }])],
            ..Default::default()
        };

        assert_eq!(verify_static_domains(&package), Ok(()));
    }

    #[test]
    fn rejects_a_zero_step_from_the_entry_const_binding() {
        let package = wire::QCorePackageProto {
            functions: vec![wire::FunctionProto {
                symbol: "test.main".to_owned(),
                parameters: vec![wire::ParameterProto {
                    name: "step".to_owned(),
                    value: Some(value(1)),
                    ..Default::default()
                }],
                body: Some(wire::BlockProto {
                    operations: vec![wire::OperationProto {
                        opcode: "qcore.for".to_owned(),
                        operands: vec![10, 11, 1],
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
                ..Default::default()
            }],
            entrypoint: "test.main".to_owned(),
            const_bindings: vec![wire::BindingProto {
                name: "step".to_owned(),
                value: Some(integer(0)),
            }],
            ..Default::default()
        };

        assert_eq!(
            verify_static_domains(&package),
            Err(StaticDomainError::ZeroForStep)
        );
    }
}
