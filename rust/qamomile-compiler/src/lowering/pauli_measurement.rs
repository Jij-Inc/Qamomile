//! Verified Pauli-basis measurement transformation.
//!
//! Expectation estimation remains a composition of ordinary target-neutral
//! programs: this pass converts a state-producing executable into a sampled
//! executable by appending exact basis changes and full-state measurement.

use std::collections::BTreeMap;

use thiserror::Error;

use crate::expectation::{CommutingMeasurementGate, MAX_COMMUTING_MEASUREMENT_GATES};
use crate::observable::Pauli;
use crate::types::{SemanticType, TypeArena};
use crate::wire;
use crate::{CompilerError, VerifiedProgram, seal_package};

/// Failure to derive a sampled Pauli-word program.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PauliMeasurementError {
    /// The estimator source violates the state-preparation signature.
    #[error("Q7304: Estimator requires classical inputs and a fixed-width Q[Bits[N]] return")]
    InvalidEstimatorSource,
    /// Bound host measurement expansion before creating group programs.
    #[error("Q7305: Estimator measurement expansion exceeds its source or statistics budget")]
    EstimatorBudget,
    /// The source executable failed its persistent semantic boundary.
    #[error("{0}")]
    Compiler(#[from] CompilerError),
    /// The entrypoint does not return exactly one fixed-width quantum value.
    #[error("Q7301: expectation source must return one fixed-width quantum value")]
    InvalidSource,
    /// A Pauli lane lies outside the returned state or appears more than once.
    #[error("Q7302: Pauli factors must be unique, ordered, and within the returned state")]
    InvalidFactors,
    /// Group term positions do not derive one valid QWC basis.
    #[error("Q7303: invalid QWC measurement group")]
    InvalidGroup,
    /// Fresh SSA identifiers cannot be represented safely.
    #[error("Q4007: Pauli measurement exhausted the SSA identifier space")]
    IdentifierCapacityExceeded,
}

/// Check the classical-input, quantum-bit-vector contract of an estimator source.
///
/// # Errors
/// Returns a source error for other signatures and a factor error for an
/// observable wider than the state. Ordinary verification proves ownership.
pub fn validate_estimator_source(
    payload: &[u8],
    observable: &[u8],
) -> Result<(), PauliMeasurementError> {
    let program = VerifiedProgram::from_bytes(payload)?;
    let package = &program.local.checked.package;
    let entry = package
        .functions
        .iter()
        .find(|f| f.symbol == package.entrypoint)
        .ok_or(PauliMeasurementError::InvalidEstimatorSource)?;
    let result = entry
        .result_type
        .as_ref()
        .ok_or(PauliMeasurementError::InvalidEstimatorSource)?;
    let (inner, width) = quantum_result_shape(result, &entry.symbol)
        .map_err(|_| PauliMeasurementError::InvalidEstimatorSource)?;
    if inner.name != "Bits" {
        return Err(PauliMeasurementError::InvalidEstimatorSource);
    }
    let mut arena = TypeArena::default();
    let scope = arena.intern_scope(&entry.symbol);
    for parameter in &entry.parameters {
        let ty = parameter
            .value
            .as_ref()
            .and_then(|v| v.r#type.as_ref())
            .ok_or(PauliMeasurementError::InvalidEstimatorSource)?;
        let id = arena
            .resolve_wire_type(ty, &scope)
            .map_err(|_| PauliMeasurementError::InvalidEstimatorSource)?;
        if arena.is_quantum(id) {
            return Err(PauliMeasurementError::InvalidEstimatorSource);
        }
    }
    let observable = crate::observable::Observable::decode(observable)
        .map_err(|_| PauliMeasurementError::InvalidGroup)?;
    if observable.terms.len() > crate::statistics::MAX_STATISTICS_TERMS
        || payload
            .len()
            .checked_mul(observable.terms.len().max(1))
            .is_none_or(|n| n > (1 << 26))
    {
        return Err(PauliMeasurementError::EstimatorBudget);
    }
    if observable.num_qubits > width {
        return Err(PauliMeasurementError::InvalidFactors);
    }
    if observable.constant.im != 0.0 || observable.terms.values().any(|c| c.im != 0.0) {
        return Err(PauliMeasurementError::InvalidGroup);
    }
    Ok(())
}

/// Append exact Pauli-basis measurement to a verified state program.
///
/// # Errors
///
/// Returns [`PauliMeasurementError`] when the input is not a verified
/// executable returning one fixed-width quantum value, a factor is malformed,
/// or the transformed program does not pass ordinary executable verification.
pub fn prepare_pauli_measurement(
    payload: &[u8],
    factors: &[(u64, Pauli)],
) -> Result<Vec<u8>, PauliMeasurementError> {
    let program = VerifiedProgram::from_bytes(payload)?;
    let mut package = program.local.checked.package.clone();
    transform(&mut package, factors)?;
    let transformed = seal_package(package)?;
    Ok(VerifiedProgram::from_bytes(&transformed)?
        .canonical_bytes()
        .to_vec())
}

/// Append the Rust-derived basis measurement for one QWC term group.
///
/// # Errors
///
/// Returns [`PauliMeasurementError`] when the observable term positions do not
/// form one QWC group or ordinary measurement preparation fails.
pub fn prepare_qwc_pauli_measurement(
    payload: &[u8],
    observable: &[u8],
    term_indices: &[usize],
) -> Result<Vec<u8>, PauliMeasurementError> {
    let basis = crate::expectation::qwc_group_basis(observable, term_indices)
        .map_err(|_| PauliMeasurementError::InvalidGroup)?;
    prepare_pauli_measurement(payload, &basis)
}

/// Append the Rust-derived Clifford measurement for one generally commuting group.
///
/// # Errors
///
/// Returns [`PauliMeasurementError`] when term positions do not form a
/// commuting group or the ordinary gate/measurement transformation fails.
pub fn prepare_commuting_pauli_measurement(
    payload: &[u8],
    observable: &[u8],
    term_indices: &[usize],
) -> Result<Vec<u8>, PauliMeasurementError> {
    let (gates, _) = crate::expectation::commuting_group_diagonalization(observable, term_indices)
        .map_err(|_| PauliMeasurementError::InvalidGroup)?;
    let program = VerifiedProgram::from_bytes(payload)?;
    let mut package = program.local.checked.package.clone();
    transform_gates(&mut package, &gates)?;
    let transformed = seal_package(package)?;
    Ok(VerifiedProgram::from_bytes(&transformed)?
        .canonical_bytes()
        .to_vec())
}

fn transform(
    package: &mut wire::QCorePackageProto,
    factors: &[(u64, Pauli)],
) -> Result<(), PauliMeasurementError> {
    let gates = factors
        .iter()
        .flat_map(|(target, basis)| match basis {
            Pauli::X => vec![CommutingMeasurementGate::Hadamard { target: *target }],
            Pauli::Y => vec![
                CommutingMeasurementGate::Sdg { target: *target },
                CommutingMeasurementGate::Hadamard { target: *target },
            ],
            Pauli::Z => Vec::new(),
        })
        .collect::<Vec<_>>();
    let entry_index = package
        .functions
        .iter()
        .position(|function| function.symbol == package.entrypoint)
        .ok_or(PauliMeasurementError::InvalidSource)?;
    let entry = &package.functions[entry_index];
    let terminator = entry
        .body
        .as_ref()
        .and_then(|body| body.terminator.as_ref())
        .filter(|terminator| terminator.kind == "return" && terminator.operands.len() == 1)
        .ok_or(PauliMeasurementError::InvalidSource)?;
    let state =
        find_value(entry, terminator.operands[0]).ok_or(PauliMeasurementError::InvalidSource)?;
    let (_, width) = quantum_result_shape(
        state
            .r#type
            .as_ref()
            .ok_or(PauliMeasurementError::InvalidSource)?,
        &entry.symbol,
    )?;
    validate_factors(factors, width)?;
    transform_gates(package, &gates)
}

fn transform_gates(
    package: &mut wire::QCorePackageProto,
    gates: &[CommutingMeasurementGate],
) -> Result<(), PauliMeasurementError> {
    if gates.len() > MAX_COMMUTING_MEASUREMENT_GATES {
        return Err(PauliMeasurementError::InvalidGroup);
    }
    let entry_index = package
        .functions
        .iter()
        .position(|function| function.symbol == package.entrypoint)
        .ok_or(PauliMeasurementError::InvalidSource)?;
    let entry = &package.functions[entry_index];
    let terminator = entry
        .body
        .as_ref()
        .and_then(|body| body.terminator.as_ref())
        .filter(|terminator| terminator.kind == "return" && terminator.operands.len() == 1)
        .ok_or(PauliMeasurementError::InvalidSource)?;
    let state = find_value(entry, terminator.operands[0])
        .cloned()
        .ok_or(PauliMeasurementError::InvalidSource)?;
    let (inner, width) = quantum_result_shape(
        state
            .r#type
            .as_ref()
            .ok_or(PauliMeasurementError::InvalidSource)?,
        &entry.symbol,
    )?;
    validate_gates(gates, width)?;

    let span = terminator.span.clone();
    let mut next_id = maximum_function_value_id(entry).checked_add(1);
    let entry = &mut package.functions[entry_index];
    let body = entry
        .body
        .as_mut()
        .ok_or(PauliMeasurementError::InvalidSource)?;
    let current = append_gate_schedule(body, gates, state, &mut next_id, span.as_ref())?;

    let measured_id = take_id(&mut next_id)?;
    let result_type = wire::TypeProto {
        kind: wire::type_proto::Kind::Staged.into(),
        name: "Shot".to_owned(),
        arguments: vec![inner],
        ..Default::default()
    };
    body.operations.push(wire::OperationProto {
        opcode: "quantum.measure".to_owned(),
        operands: vec![current.id],
        results: vec![wire::ValueProto {
            id: measured_id,
            r#type: Some(result_type.clone()),
            hint: "observable_measurement".to_owned(),
            ..Default::default()
        }],
        span,
        ..Default::default()
    });
    body.terminator
        .as_mut()
        .ok_or(PauliMeasurementError::InvalidSource)?
        .operands[0] = measured_id;
    entry.result_type = Some(result_type);
    entry.asserted_kind.clear();
    Ok(())
}

fn validate_gates(
    gates: &[CommutingMeasurementGate],
    width: u64,
) -> Result<(), PauliMeasurementError> {
    for gate in gates {
        match gate {
            CommutingMeasurementGate::Hadamard { target }
            | CommutingMeasurementGate::Sdg { target } => {
                if *target >= width {
                    return Err(PauliMeasurementError::InvalidFactors);
                }
            }
            CommutingMeasurementGate::Cx { control, target } => {
                if *control >= width || *target >= width || control == target {
                    return Err(PauliMeasurementError::InvalidFactors);
                }
            }
        }
    }
    Ok(())
}

fn append_gate_schedule(
    body: &mut wire::BlockProto,
    gates: &[CommutingMeasurementGate],
    mut current: wire::ValueProto,
    next_id: &mut Option<u64>,
    span: Option<&wire::SourceSpanProto>,
) -> Result<wire::ValueProto, PauliMeasurementError> {
    let mut selectors = BTreeMap::<u64, u64>::new();
    for gate in gates {
        let (opcode, coordinates, keywords): (&str, Vec<u64>, &[&str]) = match gate {
            CommutingMeasurementGate::Hadamard { target } => {
                ("quantum.h", vec![*target], &["target"])
            }
            CommutingMeasurementGate::Sdg { target } => ("quantum.sdg", vec![*target], &["target"]),
            CommutingMeasurementGate::Cx { control, target } => (
                "quantum.cx",
                vec![*control, *target],
                &["control", "target"],
            ),
        };
        let mut operands = vec![current.id];
        for coordinate in coordinates {
            let selector = if let Some(selector) = selectors.get(&coordinate) {
                *selector
            } else {
                let selector = take_id(next_id)?;
                body.operations.push(integer_constant(
                    selector,
                    i64::try_from(coordinate).map_err(|_| PauliMeasurementError::InvalidFactors)?,
                    span.cloned(),
                ));
                selectors.insert(coordinate, selector);
                selector
            };
            operands.push(selector);
        }
        let result_id = take_id(next_id)?;
        let mut successor = current.clone();
        successor.id = result_id;
        "observable_basis".clone_into(&mut successor.hint);
        body.operations.push(wire::OperationProto {
            opcode: opcode.to_owned(),
            operands,
            results: vec![successor.clone()],
            attributes: vec![keywords_attribute(keywords)],
            span: span.cloned(),
            ..Default::default()
        });
        current = successor;
    }
    Ok(current)
}

fn validate_factors(factors: &[(u64, Pauli)], width: u64) -> Result<(), PauliMeasurementError> {
    let mut previous = None;
    for (index, _) in factors {
        if *index >= width || previous.is_some_and(|value| value >= *index) {
            return Err(PauliMeasurementError::InvalidFactors);
        }
        previous = Some(*index);
    }
    Ok(())
}

fn quantum_result_shape(
    type_spec: &wire::TypeProto,
    scope: &str,
) -> Result<(wire::TypeProto, u64), PauliMeasurementError> {
    let mut arena = TypeArena::default();
    let scope = arena.intern_scope(scope);
    let type_id = arena
        .resolve_wire_type(type_spec, &scope)
        .map_err(|_| PauliMeasurementError::InvalidSource)?;
    let Some(SemanticType::Quantum(inner)) = arena.get(type_id) else {
        return Err(PauliMeasurementError::InvalidSource);
    };
    let inner = *inner;
    let width = arena
        .coherent_width(type_id)
        .map_err(|_| PauliMeasurementError::InvalidSource)?
        .ok_or(PauliMeasurementError::InvalidSource)?;
    let inner = arena
        .encode_wire_type(inner)
        .map_err(|_| PauliMeasurementError::InvalidSource)?;
    Ok((inner, width))
}

fn find_value(function: &wire::FunctionProto, id: u64) -> Option<&wire::ValueProto> {
    function
        .parameters
        .iter()
        .filter_map(|parameter| parameter.value.as_ref())
        .find(|value| value.id == id)
        .or_else(|| {
            function
                .body
                .as_ref()
                .and_then(|body| find_block_value(body, id))
        })
}

fn find_block_value(block: &wire::BlockProto, id: u64) -> Option<&wire::ValueProto> {
    block
        .arguments
        .iter()
        .find(|value| value.id == id)
        .or_else(|| {
            block.operations.iter().find_map(|operation| {
                operation
                    .results
                    .iter()
                    .find(|value| value.id == id)
                    .or_else(|| {
                        operation.regions.iter().find_map(|region| {
                            region
                                .blocks
                                .iter()
                                .find_map(|nested| find_block_value(nested, id))
                        })
                    })
            })
        })
}

fn maximum_function_value_id(function: &wire::FunctionProto) -> u64 {
    let parameters = function
        .parameters
        .iter()
        .filter_map(|parameter| parameter.value.as_ref())
        .map(|value| value.id)
        .max()
        .unwrap_or(0);
    function
        .body
        .as_ref()
        .map_or(0, maximum_block_value_id)
        .max(parameters)
}

fn maximum_block_value_id(block: &wire::BlockProto) -> u64 {
    let arguments = block.arguments.iter().map(|value| value.id);
    let operations = block.operations.iter().flat_map(|operation| {
        operation.results.iter().map(|value| value.id).chain(
            operation
                .regions
                .iter()
                .flat_map(|region| region.blocks.iter().map(maximum_block_value_id)),
        )
    });
    arguments.chain(operations).max().unwrap_or(0)
}

fn take_id(next_id: &mut Option<u64>) -> Result<u64, PauliMeasurementError> {
    let current = next_id
        .take()
        .ok_or(PauliMeasurementError::IdentifierCapacityExceeded)?;
    *next_id = current.checked_add(1);
    Ok(current)
}

fn integer_constant(
    id: u64,
    value: i64,
    span: Option<wire::SourceSpanProto>,
) -> wire::OperationProto {
    wire::OperationProto {
        opcode: "qcore.constant".to_owned(),
        results: vec![wire::ValueProto {
            id,
            r#type: Some(wire::TypeProto {
                kind: wire::type_proto::Kind::Scalar.into(),
                name: "Int".to_owned(),
                ..Default::default()
            }),
            hint: "observable_target".to_owned(),
            ..Default::default()
        }],
        attributes: vec![wire::AttributeProto {
            name: "value".to_owned(),
            value: Some(wire::LiteralProto {
                value: Some(wire::literal_proto::Value::IntValue(value)),
            }),
        }],
        span,
        ..Default::default()
    }
}

fn keywords_attribute(keywords: &[&str]) -> wire::AttributeProto {
    wire::AttributeProto {
        name: "keywords".to_owned(),
        value: Some(wire::LiteralProto {
            value: Some(wire::literal_proto::Value::TupleValue(
                wire::literal_proto::Sequence {
                    values: keywords
                        .iter()
                        .map(|keyword| wire::LiteralProto {
                            value: Some(wire::literal_proto::Value::StringValue(
                                (*keyword).to_owned(),
                            )),
                        })
                        .collect(),
                },
            )),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PROGRAM_KIND, SCHEMA_MAJOR, SCHEMA_MINOR};
    use prost::Message;

    fn scalar(name: &str) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Scalar.into(),
            name: name.to_owned(),
            ..Default::default()
        }
    }

    fn bits(width: u64) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Array.into(),
            name: "Bits".to_owned(),
            arguments: vec![scalar("Bit")],
            dimensions: vec![wire::DimensionProto {
                value: Some(wire::dimension_proto::Value::Fixed(width)),
            }],
            ..Default::default()
        }
    }

    fn fixed(integer_bits: u64, fractional_bits: u64) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Fixed.into(),
            name: "UFixed".to_owned(),
            dimensions: vec![
                wire::DimensionProto {
                    value: Some(wire::dimension_proto::Value::Fixed(integer_bits)),
                },
                wire::DimensionProto {
                    value: Some(wire::dimension_proto::Value::Fixed(fractional_bits)),
                },
            ],
            ..Default::default()
        }
    }

    fn span() -> wire::SourceSpanProto {
        wire::SourceSpanProto {
            uri: "qmi://pauli-measurement-test".to_owned(),
            start_line: 1,
            end_line: 1,
            ..Default::default()
        }
    }

    fn state_program() -> Vec<u8> {
        let quantum_bool = wire::TypeProto {
            kind: wire::type_proto::Kind::Quantum.into(),
            arguments: vec![scalar("Bool")],
            ..Default::default()
        };
        let state = wire::ValueProto {
            id: 2,
            r#type: Some(quantum_bool.clone()),
            hint: "state".to_owned(),
            region: Some(wire::RegionRefProto {
                root: 2,
                ..Default::default()
            }),
            has_region: true,
        };
        seal_package(wire::QCorePackageProto {
            schema_major: SCHEMA_MAJOR,
            schema_minor: SCHEMA_MINOR,
            artifact_kind: PROGRAM_KIND.to_owned(),
            module_name: "pauli_measurement_test".to_owned(),
            functions: vec![wire::FunctionProto {
                symbol: "pauli_measurement_test.state".to_owned(),
                result_type: Some(quantum_bool),
                body: Some(wire::BlockProto {
                    operations: vec![
                        wire::OperationProto {
                            opcode: "qcore.constant".to_owned(),
                            results: vec![wire::ValueProto {
                                id: 1,
                                r#type: Some(scalar("Bool")),
                                hint: "basis".to_owned(),
                                ..Default::default()
                            }],
                            attributes: vec![wire::AttributeProto {
                                name: "value".to_owned(),
                                value: Some(wire::LiteralProto {
                                    value: Some(wire::literal_proto::Value::BoolValue(false)),
                                }),
                            }],
                            span: Some(span()),
                            ..Default::default()
                        },
                        wire::OperationProto {
                            opcode: "quantum.prepare".to_owned(),
                            operands: vec![1],
                            results: vec![state],
                            span: Some(span()),
                            ..Default::default()
                        },
                    ],
                    terminator: Some(wire::TerminatorProto {
                        kind: "return".to_owned(),
                        operands: vec![2],
                        span: Some(span()),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                span: Some(span()),
                ..Default::default()
            }],
            entrypoint: "pauli_measurement_test.state".to_owned(),
            ..Default::default()
        })
        .expect("seal state program")
    }

    #[test]
    fn appends_basis_change_and_measurement_then_reverifies() {
        let transformed = prepare_pauli_measurement(&state_program(), &[(0, Pauli::X)])
            .expect("valid Pauli measurement");
        let program = VerifiedProgram::from_bytes(&transformed).expect("verified result");
        let entry = &program.local.checked.package.functions[0];
        let opcodes = entry
            .body
            .as_ref()
            .expect("body")
            .operations
            .iter()
            .map(|operation| operation.opcode.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            opcodes,
            [
                "qcore.constant",
                "quantum.prepare",
                "qcore.constant",
                "quantum.h",
                "quantum.measure",
            ]
        );
        assert_eq!(entry.result_type.as_ref().expect("result").name, "Shot");
    }

    #[test]
    fn derives_composite_coherent_width_from_the_shared_type_arena() {
        let inner = wire::TypeProto {
            kind: wire::type_proto::Kind::Product.into(),
            arguments: vec![bits(2), fixed(1, 2)],
            ..Default::default()
        };
        let state = wire::TypeProto {
            kind: wire::type_proto::Kind::Quantum.into(),
            arguments: vec![inner.clone()],
            ..Default::default()
        };

        assert_eq!(quantum_result_shape(&state, "test.state"), Ok((inner, 5)));
    }

    #[test]
    fn rejects_a_factor_outside_the_returned_state() {
        assert_eq!(
            prepare_pauli_measurement(&state_program(), &[(1, Pauli::Z)]),
            Err(PauliMeasurementError::InvalidFactors)
        );
    }

    #[test]
    fn rotates_y_with_sdg_before_h() {
        let transformed = prepare_pauli_measurement(&state_program(), &[(0, Pauli::Y)])
            .expect("valid Y measurement");
        let program = VerifiedProgram::from_bytes(&transformed).expect("verified result");
        let opcodes = program.local.checked.package.functions[0]
            .body
            .as_ref()
            .expect("body")
            .operations
            .iter()
            .map(|operation| operation.opcode.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            &opcodes[2..],
            [
                "qcore.constant",
                "quantum.sdg",
                "quantum.h",
                "quantum.measure",
            ]
        );
    }

    #[test]
    fn uses_the_last_representable_identifier_for_the_final_measurement() {
        let mut package = wire::QCorePackageProto::decode(state_program().as_slice())
            .expect("decode state program");
        let body = package.functions[0].body.as_mut().expect("body");
        let state = &mut body.operations[1].results[0];
        state.id = u64::MAX - 1;
        state.region.as_mut().expect("region").root = u64::MAX - 1;
        body.terminator.as_mut().expect("terminator").operands[0] = u64::MAX - 1;
        let payload = seal_package(package).expect("reseal state program");

        prepare_pauli_measurement(&payload, &[(0, Pauli::Z)])
            .expect("the final identifier remains representable");
    }
}
