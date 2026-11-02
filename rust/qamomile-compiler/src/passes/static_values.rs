//! Shared checked scalar abstraction for native specialization-time passes.

use std::collections::HashMap;
use std::sync::Arc;

use num_bigint::{BigInt, Sign};
use num_complex::Complex64;
use num_traits::ToPrimitive;
use prost::Message;
use thiserror::Error;

use crate::numeric_values::{self, NumericEvaluationError};
use crate::observable::{
    NormalizedObservable, ObservableError, Pauli, add_scaled_observable, affine_observable,
    multiply_observables, normalize_observable, normalize_real_observable,
};
use crate::wire;

/// Maximum number of element projections materialized by one fixed slice.
pub const MAX_AGGREGATE_SLICE_ITEMS: usize = 1_024;

/// Failure to normalize one exact fixed aggregate slice.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum AggregateSliceError {
    /// Python slice strides cannot be zero.
    #[error("Q4513: aggregate slice step cannot be zero")]
    ZeroStep,
    /// Source expansion must remain within the bounded HIR work budget.
    #[error("Q4514: aggregate slice exceeds {MAX_AGGREGATE_SLICE_ITEMS} selected items")]
    OutputBudget,
    /// A normalized source position cannot enter the native host index domain.
    #[error("Q4515: aggregate slice position is not representable")]
    UnrepresentablePosition,
}

/// One scalar known exactly without executing Python.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum StaticScalar {
    /// A classical truth value.
    Bool(bool),
    /// A signed integer representable by the current native exact slice.
    Int(i64),
    /// An IEEE-754 binary64 value stored by bits for total equality.
    RealBits(u64),
}

/// One target-independent classical value known without executing Python.
///
/// Aggregate elements are individually optional so analyses can retain exact
/// projections from a value whose other coordinates remain dynamic.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum StaticValue {
    /// The sole unit value.
    Unit,
    /// One exact scalar.
    Scalar(StaticScalar),
    /// One canonical signed arbitrary-precision integer encoding.
    BigInt(Arc<[u8]>),
    /// One complex binary64 pair stored by bits for total equality.
    ComplexBits(u64, u64),
    /// One immutable Unicode string.
    String(Arc<str>),
    /// One immutable byte string.
    Bytes(Arc<[u8]>),
    /// One positional product, record, word, or array value.
    Aggregate(Arc<[Option<Self>]>),
    /// One labelled product retained in semantic declaration order.
    Record(Arc<[(Arc<str>, Option<Self>)]>),
    /// One canonical lexical string-keyed mapping.
    Mapping(Arc<[(Arc<str>, Self)]>),
}

impl StaticValue {
    /// Return the scalar leaf, if this value is scalar.
    pub(crate) const fn scalar(&self) -> Option<StaticScalar> {
        match self {
            Self::Scalar(value) => Some(*value),
            Self::Unit
            | Self::BigInt(_)
            | Self::ComplexBits(_, _)
            | Self::String(_)
            | Self::Bytes(_)
            | Self::Aggregate(_)
            | Self::Record(_)
            | Self::Mapping(_) => None,
        }
    }

    /// Return whether every aggregate coordinate is known recursively.
    pub(crate) fn is_complete(&self) -> bool {
        match self {
            Self::Aggregate(values) => values
                .iter()
                .all(|value| value.as_ref().is_some_and(Self::is_complete)),
            Self::Record(fields) => fields
                .iter()
                .all(|(_, value)| value.as_ref().is_some_and(Self::is_complete)),
            _ => true,
        }
    }
}

/// Decode one scalar literal in the current exact native domain.
pub(crate) fn literal(literal: &wire::LiteralProto) -> Option<StaticScalar> {
    use wire::literal_proto::Value;

    match literal.value.as_ref()? {
        Value::BoolValue(value) => Some(StaticScalar::Bool(*value)),
        Value::IntValue(value) => Some(StaticScalar::Int(*value)),
        Value::RealValue(value) => Some(StaticScalar::RealBits(value.to_bits())),
        _ => None,
    }
}

/// Encode one exact scalar back into the canonical portable literal domain.
pub(crate) fn to_literal(value: StaticScalar) -> wire::LiteralProto {
    use wire::literal_proto::Value;

    wire::LiteralProto {
        value: Some(match value {
            StaticScalar::Bool(value) => Value::BoolValue(value),
            StaticScalar::Int(value) => Value::IntValue(value),
            StaticScalar::RealBits(value) => Value::RealValue(f64::from_bits(value)),
        }),
    }
}

/// Decode one recursively exact positional literal.
pub(crate) fn value_literal(source: &wire::LiteralProto) -> Option<StaticValue> {
    use wire::literal_proto::Value;

    match source.value.as_ref()? {
        Value::UnitValue(_) => Some(StaticValue::Unit),
        Value::BigintValue(value) => Some(StaticValue::BigInt(Arc::from(value.as_slice()))),
        Value::ComplexValue(value) => Some(StaticValue::ComplexBits(
            value.real.to_bits(),
            value.imaginary.to_bits(),
        )),
        Value::StringValue(value) => Some(StaticValue::String(Arc::from(value.as_str()))),
        Value::BytesValue(value) => Some(StaticValue::Bytes(Arc::from(value.as_slice()))),
        Value::TupleValue(sequence) => Some(StaticValue::Aggregate(
            sequence
                .values
                .iter()
                .map(value_literal)
                .collect::<Vec<_>>()
                .into(),
        )),
        Value::MappingValue(mapping) => Some(StaticValue::Mapping(
            mapping
                .entries
                .iter()
                .map(|entry| {
                    Some((
                        Arc::from(entry.key.as_str()),
                        value_literal(entry.value.as_ref()?)?,
                    ))
                })
                .collect::<Option<Vec<_>>>()?
                .into(),
        )),
        _ => literal(source).map(StaticValue::Scalar),
    }
}

/// Decode a portable literal using its verified semantic type.
///
/// Record literals are canonicalized lexically on the wire, while record
/// projection follows declaration order. This decoder restores that typed
/// ordering once at the boundary instead of making evaluators guess it.
pub(crate) fn typed_value_literal(
    source: &wire::LiteralProto,
    type_spec: &wire::TypeProto,
) -> Option<StaticValue> {
    use wire::literal_proto::Value;
    use wire::type_proto::Kind;

    match Kind::try_from(type_spec.kind).ok()? {
        Kind::Staged => typed_value_literal(source, type_spec.arguments.first()?),
        Kind::Record => {
            let Value::MappingValue(mapping) = source.value.as_ref()? else {
                return None;
            };
            if type_spec.field_names.len() != type_spec.arguments.len()
                || mapping.entries.len() != type_spec.field_names.len()
            {
                return None;
            }
            let entries = mapping
                .entries
                .iter()
                .map(|entry| (entry.key.as_str(), entry.value.as_ref()))
                .collect::<HashMap<_, _>>();
            if entries.len() != mapping.entries.len() {
                return None;
            }
            Some(StaticValue::Record(
                type_spec
                    .field_names
                    .iter()
                    .zip(&type_spec.arguments)
                    .map(|(name, field_type)| {
                        Some((
                            Arc::from(name.as_str()),
                            Some(typed_value_literal(
                                entries.get(name.as_str()).copied().flatten()?,
                                field_type,
                            )?),
                        ))
                    })
                    .collect::<Option<Vec<_>>>()?
                    .into(),
            ))
        }
        Kind::Product => {
            let Value::TupleValue(sequence) = source.value.as_ref()? else {
                return None;
            };
            if sequence.values.len() != type_spec.arguments.len() {
                return None;
            }
            Some(StaticValue::Aggregate(
                sequence
                    .values
                    .iter()
                    .zip(&type_spec.arguments)
                    .map(|(value, element_type)| typed_value_literal(value, element_type).map(Some))
                    .collect::<Option<Vec<_>>>()?
                    .into(),
            ))
        }
        Kind::Array if type_spec.name == "Array" => typed_array_literal(
            source,
            type_spec.arguments.first()?,
            type_spec.dimensions.len(),
        ),
        _ => value_literal(source),
    }
}

/// Build one structural static fact using the verified result type.
pub(crate) fn structural_value(
    result: Option<&wire::ValueProto>,
    values: Vec<Option<StaticValue>>,
) -> StaticValue {
    let record = result
        .and_then(|value| value.r#type.as_ref())
        .and_then(record_type)
        .filter(|type_spec| type_spec.field_names.len() == values.len());
    if let Some(type_spec) = record {
        StaticValue::Record(
            type_spec
                .field_names
                .iter()
                .zip(values)
                .map(|(name, value)| (Arc::from(name.as_str()), value))
                .collect::<Vec<_>>()
                .into(),
        )
    } else {
        StaticValue::Aggregate(values.into())
    }
}

/// Return positional components from one product-like static value.
pub(crate) fn structural_components(value: &StaticValue) -> Option<Vec<Option<StaticValue>>> {
    match value {
        StaticValue::Aggregate(values) => Some(values.to_vec()),
        StaticValue::Record(fields) => {
            Some(fields.iter().map(|(_, value)| value.clone()).collect())
        }
        _ => None,
    }
}

fn typed_array_literal(
    source: &wire::LiteralProto,
    element_type: &wire::TypeProto,
    rank: usize,
) -> Option<StaticValue> {
    use wire::literal_proto::Value;

    let Value::TupleValue(sequence) = source.value.as_ref()? else {
        return None;
    };
    if rank == 0 {
        return None;
    }
    Some(StaticValue::Aggregate(
        sequence
            .values
            .iter()
            .map(|value| {
                let decoded = if rank == 1 {
                    typed_value_literal(value, element_type)
                } else {
                    typed_array_literal(value, element_type, rank - 1)
                }?;
                Some(Some(decoded))
            })
            .collect::<Option<Vec<_>>>()?
            .into(),
    ))
}

fn record_type(mut type_spec: &wire::TypeProto) -> Option<&wire::TypeProto> {
    use wire::type_proto::Kind;

    while Kind::try_from(type_spec.kind).ok()? == Kind::Staged {
        type_spec = type_spec.arguments.first()?;
    }
    (Kind::try_from(type_spec.kind).ok()? == Kind::Record).then_some(type_spec)
}

/// Encode one complete static value into the canonical portable literal domain.
pub(crate) fn value_to_literal(value: &StaticValue) -> Option<wire::LiteralProto> {
    use wire::literal_proto::Value;

    match value {
        StaticValue::Unit => Some(wire::LiteralProto {
            value: Some(Value::UnitValue(true)),
        }),
        StaticValue::Scalar(value) => Some(to_literal(*value)),
        StaticValue::BigInt(value) => Some(wire::LiteralProto {
            value: Some(Value::BigintValue(value.to_vec())),
        }),
        StaticValue::ComplexBits(real, imaginary) => Some(wire::LiteralProto {
            value: Some(Value::ComplexValue(wire::literal_proto::Complex {
                real: f64::from_bits(*real),
                imaginary: f64::from_bits(*imaginary),
            })),
        }),
        StaticValue::String(value) => Some(wire::LiteralProto {
            value: Some(Value::StringValue(value.to_string())),
        }),
        StaticValue::Bytes(value) => Some(wire::LiteralProto {
            value: Some(Value::BytesValue(value.to_vec())),
        }),
        StaticValue::Aggregate(values) => Some(wire::LiteralProto {
            value: Some(Value::TupleValue(wire::literal_proto::Sequence {
                values: values
                    .iter()
                    .map(|value| value.as_ref().and_then(value_to_literal))
                    .collect::<Option<Vec<_>>>()?,
            })),
        }),
        StaticValue::Record(fields) => {
            let mut entries = fields
                .iter()
                .map(|(key, value)| {
                    Some(wire::literal_proto::Entry {
                        key: key.to_string(),
                        value: Some(value_to_literal(value.as_ref()?)?),
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            entries.sort_by(|left, right| left.key.cmp(&right.key));
            Some(wire::LiteralProto {
                value: Some(Value::MappingValue(wire::literal_proto::Mapping {
                    entries,
                })),
            })
        }
        StaticValue::Mapping(entries) => Some(wire::LiteralProto {
            value: Some(Value::MappingValue(wire::literal_proto::Mapping {
                entries: entries
                    .iter()
                    .map(|(key, value)| {
                        Some(wire::literal_proto::Entry {
                            key: key.to_string(),
                            value: Some(value_to_literal(value)?),
                        })
                    })
                    .collect::<Option<Vec<_>>>()?,
            })),
        }),
    }
}

/// Evaluate one verified classical operation over recursively exact facts.
pub(crate) fn evaluate_value(
    operation: &wire::OperationProto,
    environment: &HashMap<u64, Option<StaticValue>>,
) -> Option<StaticValue> {
    evaluate_value_checked(operation, environment)
        .ok()
        .flatten()
}

/// Evaluate one verified classical operation and retain selected domain errors.
pub(crate) fn evaluate_value_checked(
    operation: &wire::OperationProto,
    environment: &HashMap<u64, Option<StaticValue>>,
) -> Result<Option<StaticValue>, NumericEvaluationError> {
    if let Some(value) = evaluate_observable_value(operation, environment)? {
        return Ok(Some(value));
    }
    match operation.opcode.as_str() {
        "classical.index" => return Ok(evaluate_index_value(operation, environment)),
        "classical.slice" => return evaluate_slice_value(operation, environment),
        "classical.update" => return evaluate_update_value(operation, environment),
        "classical.map_get" => return evaluate_map_get_value(operation, environment),
        "classical.project" => return Ok(evaluate_project_value(operation, environment)),
        "classical.len" => return Ok(evaluate_len_value(operation, environment)),
        "classical.select" => return Ok(evaluate_select_value(operation, environment)),
        "classical.compare_chain" => return Ok(compare_value_chain(operation, environment)),
        _ => {}
    }
    let values = operation
        .operands
        .iter()
        .map(|operand| environment.get(operand)?.as_ref())
        .collect::<Option<Vec<_>>>();
    if let Some(values) = values
        && let Some(value) = numeric_values::evaluate(operation, &values)?
    {
        return Ok(Some(value));
    }
    let scalars = operation
        .operands
        .iter()
        .map(|operand| {
            let value = environment
                .get(operand)
                .and_then(Option::as_ref)
                .and_then(StaticValue::scalar);
            (*operand, value)
        })
        .collect::<HashMap<_, _>>();
    Ok(evaluate_boolean(operation, &scalars).map(StaticValue::Scalar))
}

fn evaluate_observable_value(
    operation: &wire::OperationProto,
    environment: &HashMap<u64, Option<StaticValue>>,
) -> Result<Option<StaticValue>, NumericEvaluationError> {
    let operands = operation
        .operands
        .iter()
        .map(|operand| environment.get(operand)?.as_ref())
        .collect::<Option<Vec<_>>>();
    let Some(operands) = operands else {
        return Ok(None);
    };
    match operation.opcode.as_str() {
        "classical.observable_zero" => {
            let [width] = operands.as_slice() else {
                return Ok(None);
            };
            let width = nonnegative_index(width)?;
            normalize_real_observable(width, 0.0, &[])
                .map(|observable| normalized_observable_value(&observable))
                .map(Some)
                .map_err(|error| observable_evaluation_error(&error))
        }
        "classical.observable_pauli" => {
            let [index] = operands.as_slice() else {
                return Ok(None);
            };
            let index = nonnegative_index(index)?;
            let width = index.checked_add(1).ok_or(NumericEvaluationError::Limit(
                "Q7404: observable logical width is not representable",
            ))?;
            let pauli = match string_attribute(operation, "axis") {
                Some("X") => Pauli::X,
                Some("Y") => Pauli::Y,
                Some("Z") => Pauli::Z,
                _ => return Ok(None),
            };
            normalize_real_observable(width, 0.0, &[(vec![(index, pauli)], 1.0)])
                .map(|observable| normalized_observable_value(&observable))
                .map(Some)
                .map_err(|error| observable_evaluation_error(&error))
        }
        "classical.add" | "classical.sub" | "classical.mul" => {
            let [left, right] = operands.as_slice() else {
                return Ok(None);
            };
            evaluate_observable_binary(operation.opcode.as_str(), left, right)
        }
        "classical.neg" => {
            let [operand] = operands.as_slice() else {
                return Ok(None);
            };
            let Some(payload) = observable_payload(operand) else {
                return Ok(None);
            };
            affine_observable(
                &payload,
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.0, 0.0),
            )
            .map(|observable| normalized_observable_value(&observable))
            .map(Some)
            .map_err(|error| observable_evaluation_error(&error))
        }
        _ => Ok(None),
    }
}

fn evaluate_observable_binary(
    opcode: &str,
    left: &StaticValue,
    right: &StaticValue,
) -> Result<Option<StaticValue>, NumericEvaluationError> {
    let left_observable = observable_payload(left);
    let right_observable = observable_payload(right);
    let result = match (opcode, left_observable, right_observable) {
        ("classical.add", Some(left), Some(right)) => {
            add_scaled_observable(&left, &right, Complex64::new(1.0, 0.0))
        }
        ("classical.sub", Some(left), Some(right)) => {
            add_scaled_observable(&left, &right, Complex64::new(-1.0, 0.0))
        }
        ("classical.mul", Some(left), Some(right)) => multiply_observables(&left, &right),
        ("classical.add", Some(observable), None) => {
            affine_observable(&observable, Complex64::new(1.0, 0.0), coefficient(right)?)
        }
        ("classical.add", None, Some(observable)) => {
            affine_observable(&observable, Complex64::new(1.0, 0.0), coefficient(left)?)
        }
        ("classical.sub", Some(observable), None) => {
            affine_observable(&observable, Complex64::new(1.0, 0.0), -coefficient(right)?)
        }
        ("classical.sub", None, Some(observable)) => {
            affine_observable(&observable, Complex64::new(-1.0, 0.0), coefficient(left)?)
        }
        ("classical.mul", Some(observable), None) => {
            affine_observable(&observable, coefficient(right)?, Complex64::new(0.0, 0.0))
        }
        ("classical.mul", None, Some(observable)) => {
            affine_observable(&observable, coefficient(left)?, Complex64::new(0.0, 0.0))
        }
        _ => return Ok(None),
    };
    result
        .map(|observable| normalized_observable_value(&observable))
        .map(Some)
        .map_err(|error| observable_evaluation_error(&error))
}

fn observable_payload(value: &StaticValue) -> Option<Box<[u8]>> {
    let StaticValue::Mapping(_) = value else {
        return None;
    };
    let payload = value_to_literal(value)?.encode_to_vec().into_boxed_slice();
    normalize_observable(&payload).ok()?;
    Some(payload)
}

fn normalized_observable_value(observable: &NormalizedObservable) -> StaticValue {
    let literal = wire::LiteralProto::decode(observable.literal())
        .expect("compiler-produced observable literal must decode");
    value_literal(&literal).expect("compiler-produced observable literal must be portable")
}

fn coefficient(value: &StaticValue) -> Result<Complex64, NumericEvaluationError> {
    let coefficient = match value {
        StaticValue::Scalar(StaticScalar::Int(value)) => Complex64::new(
            value.to_f64().ok_or(NumericEvaluationError::Limit(
                "Q7411: Pauli coefficient exceeds the finite binary64 range",
            ))?,
            0.0,
        ),
        StaticValue::Scalar(StaticScalar::RealBits(value)) => {
            Complex64::new(f64::from_bits(*value), 0.0)
        }
        StaticValue::BigInt(value) => Complex64::new(
            BigInt::from_signed_bytes_be(value)
                .to_f64()
                .ok_or(NumericEvaluationError::Limit(
                    "Q7411: Pauli coefficient exceeds the finite binary64 range",
                ))?,
            0.0,
        ),
        StaticValue::ComplexBits(real, imaginary) => {
            Complex64::new(f64::from_bits(*real), f64::from_bits(*imaginary))
        }
        _ => {
            return Err(NumericEvaluationError::Undefined(
                "Q7403: observable coefficient must be numeric",
            ));
        }
    };
    if coefficient.re.is_finite() && coefficient.im.is_finite() {
        Ok(coefficient)
    } else {
        Err(NumericEvaluationError::Limit(
            "Q7411: Pauli coefficient exceeds the finite binary64 range",
        ))
    }
}

fn nonnegative_index(value: &StaticValue) -> Result<usize, NumericEvaluationError> {
    let value = match value {
        StaticValue::Scalar(StaticScalar::Int(value)) => usize::try_from(*value).ok(),
        StaticValue::BigInt(value) => BigInt::from_signed_bytes_be(value).to_usize(),
        _ => None,
    };
    value.ok_or(NumericEvaluationError::Undefined(
        "Q7404: observable logical indices and widths must be non-negative and representable",
    ))
}

const fn observable_evaluation_error(error: &ObservableError) -> NumericEvaluationError {
    match error {
        ObservableError::Budget => {
            NumericEvaluationError::Limit("Q7401: observable literal exceeds the structural budget")
        }
        ObservableError::WidthOverflow => {
            NumericEvaluationError::Limit("Q7404: observable logical width is not representable")
        }
        ObservableError::CoefficientRange => NumericEvaluationError::Limit(
            "Q7411: Pauli coefficient exceeds the finite binary64 range",
        ),
        _ => NumericEvaluationError::Undefined("Q7403: malformed symbolic observable construction"),
    }
}

fn evaluate_slice_value(
    operation: &wire::OperationProto,
    environment: &HashMap<u64, Option<StaticValue>>,
) -> Result<Option<StaticValue>, NumericEvaluationError> {
    use crate::core::slice_shape::{SliceBound, operation_bounds};
    let Some(StaticValue::Aggregate(values)) = operation
        .operands
        .first()
        .and_then(|id| environment.get(id))
        .and_then(Option::as_ref)
    else {
        return Ok(None);
    };
    let descriptors = operation_bounds(operation)
        .map_err(|_| NumericEvaluationError::Undefined("invalid slice bounds"))?;
    let mut operands = operation.operands.iter().skip(1);
    let mut bounds = Vec::new();
    for descriptor in descriptors {
        if descriptor == SliceBound::Omitted {
            bounds.push(None);
        } else {
            let Some(value) = operands
                .next()
                .and_then(|id| environment.get(id))
                .and_then(Option::as_ref)
                .and_then(StaticValue::scalar)
                .and_then(integer)
            else {
                return Ok(None);
            };
            bounds.push(Some(BigInt::from(value)));
        }
    }
    let positions = normalize_aggregate_slice(
        values.len(),
        bounds[0].as_ref(),
        bounds[1].as_ref(),
        bounds[2].as_ref(),
    )
    .map_err(|_| NumericEvaluationError::Undefined("invalid or excessive aggregate slice"))?;
    Ok(Some(StaticValue::Aggregate(
        positions.iter().map(|i| values[*i].clone()).collect(),
    )))
}

fn evaluate_update_value(
    operation: &wire::OperationProto,
    environment: &HashMap<u64, Option<StaticValue>>,
) -> Result<Option<StaticValue>, NumericEvaluationError> {
    let [collection, index, replacement] = operation.operands.as_slice() else {
        return Ok(None);
    };
    let Some(StaticValue::Aggregate(values)) = environment.get(collection).and_then(Option::as_ref)
    else {
        return Ok(None);
    };
    let Some(index) = environment
        .get(index)
        .and_then(Option::as_ref)
        .and_then(StaticValue::scalar)
        .and_then(integer)
    else {
        return Ok(None);
    };
    let index = normalize_aggregate_index(values.len(), &BigInt::from(index)).ok_or(
        NumericEvaluationError::Undefined("aggregate update index is out of bounds"),
    )?;
    if values.len() > MAX_AGGREGATE_SLICE_ITEMS {
        return Err(NumericEvaluationError::Limit(
            "aggregate update exceeds its item budget",
        ));
    }
    let mut updated = values.to_vec();
    updated[index] = environment.get(replacement).cloned().flatten();
    Ok(Some(StaticValue::Aggregate(updated.into())))
}

fn evaluate_index_value(
    operation: &wire::OperationProto,
    environment: &HashMap<u64, Option<StaticValue>>,
) -> Option<StaticValue> {
    let [aggregate, selector] = operation.operands.as_slice() else {
        return None;
    };
    let aggregate = environment.get(aggregate)?.as_ref()?;
    let index = environment
        .get(selector)?
        .as_ref()?
        .scalar()
        .and_then(integer)?;
    match aggregate {
        StaticValue::Aggregate(values) => aggregate_index(values, index),
        StaticValue::Record(fields) => record_index(fields, index),
        _ => None,
    }
}

fn evaluate_map_get_value(
    operation: &wire::OperationProto,
    environment: &HashMap<u64, Option<StaticValue>>,
) -> Result<Option<StaticValue>, NumericEvaluationError> {
    let [mapping, key] = operation.operands.as_slice() else {
        return Ok(None);
    };
    let Some(StaticValue::Aggregate(entries)) = environment.get(mapping).and_then(Option::as_ref)
    else {
        return Ok(None);
    };
    let Some(key) = environment.get(key).and_then(Option::as_ref) else {
        return Ok(None);
    };
    let mut found = None;
    for entry in entries.iter() {
        let Some(entry) = entry else {
            return Ok(None);
        };
        let (entry_key, entry_value) = match entry {
            StaticValue::Record(fields) => {
                let [(_, Some(entry_key)), (_, Some(entry_value))] = fields.as_ref() else {
                    return Ok(None);
                };
                (entry_key, entry_value)
            }
            StaticValue::Aggregate(values) => {
                let [Some(entry_key), Some(entry_value)] = values.as_ref() else {
                    return Ok(None);
                };
                (entry_key, entry_value)
            }
            StaticValue::Mapping(fields) => {
                let Some((_, entry_key)) = fields.iter().find(|(name, _)| name.as_ref() == "key")
                else {
                    return Ok(None);
                };
                let Some((_, entry_value)) =
                    fields.iter().find(|(name, _)| name.as_ref() == "value")
                else {
                    return Ok(None);
                };
                (entry_key, entry_value)
            }
            _ => return Ok(None),
        };
        if entry_key == key {
            if found.is_some() {
                return Err(NumericEvaluationError::Undefined(
                    "Q6204: immutable mapping contains a duplicate key",
                ));
            }
            found = Some(entry_value.clone());
        }
    }
    found.map(Some).ok_or(NumericEvaluationError::Undefined(
        "Q6204: immutable mapping key is missing",
    ))
}

fn evaluate_project_value(
    operation: &wire::OperationProto,
    environment: &HashMap<u64, Option<StaticValue>>,
) -> Option<StaticValue> {
    let [aggregate] = operation.operands.as_slice() else {
        return None;
    };
    let source = environment.get(aggregate)?.as_ref()?;
    let position = int_attribute(operation, "position")?;
    match source {
        StaticValue::Aggregate(values) => aggregate_index(values, position),
        StaticValue::Record(fields) => record_index(fields, position),
        _ => None,
    }
}

fn evaluate_len_value(
    operation: &wire::OperationProto,
    environment: &HashMap<u64, Option<StaticValue>>,
) -> Option<StaticValue> {
    let [aggregate] = operation.operands.as_slice() else {
        return None;
    };
    let length = match environment.get(aggregate)?.as_ref()? {
        StaticValue::Aggregate(values) => values.len(),
        StaticValue::Record(fields) => fields.len(),
        _ => return None,
    };
    i64::try_from(length)
        .ok()
        .map(StaticScalar::Int)
        .map(StaticValue::Scalar)
}

fn evaluate_select_value(
    operation: &wire::OperationProto,
    environment: &HashMap<u64, Option<StaticValue>>,
) -> Option<StaticValue> {
    let [condition, when_true, when_false] = operation.operands.as_slice() else {
        return None;
    };
    let condition = environment.get(condition)?.as_ref()?.scalar()?;
    environment
        .get(if truth(condition) {
            when_true
        } else {
            when_false
        })?
        .clone()
}

/// Select one positional aggregate coordinate using Python index semantics.
pub(crate) fn aggregate_index(values: &[Option<StaticValue>], index: i64) -> Option<StaticValue> {
    values
        .get(normalize_aggregate_index(
            values.len(),
            &BigInt::from(index),
        )?)?
        .clone()
}

/// Return whether an index lies within one positional aggregate extent.
pub(crate) fn aggregate_index_is_in_bounds(extent: usize, index: i64) -> bool {
    normalize_aggregate_index(extent, &BigInt::from(index)).is_some()
}

fn record_index(fields: &[(Arc<str>, Option<StaticValue>)], index: i64) -> Option<StaticValue> {
    fields
        .get(normalize_aggregate_index(
            fields.len(),
            &BigInt::from(index),
        )?)?
        .1
        .clone()
}

/// Normalize one arbitrary-precision integer with Python aggregate-index rules.
///
/// The result is a zero-based position only when the index lies in
/// ``[-extent, extent)``. Empty aggregates and every more-negative or
/// nonnegative out-of-range value return ``None``.
#[must_use]
pub fn normalize_aggregate_index(extent: usize, index: &BigInt) -> Option<usize> {
    let extent_value = BigInt::from(extent);
    let normalized = if index.sign() == Sign::Minus {
        extent_value.clone() + index
    } else {
        index.clone()
    };
    normalized.to_usize().filter(|position| *position < extent)
}

/// Normalize exact bounds with Python aggregate-slice rules.
///
/// Omitted bounds retain their direction-dependent sentinel semantics. Every
/// explicit bound is shifted and clamped to the aggregate extent before the
/// selected zero-based positions are materialized. Every successful result
/// contains at most ``extent`` positions.
///
/// # Errors
///
/// Returns [`AggregateSliceError`] for a zero stride, excessive selected-item
/// count, or an unrepresentable normalized position.
pub fn normalize_aggregate_slice(
    extent: usize,
    start_bound: Option<&BigInt>,
    stop_bound: Option<&BigInt>,
    stride_bound: Option<&BigInt>,
) -> Result<Box<[usize]>, AggregateSliceError> {
    let extent = BigInt::from(extent);
    let stride = stride_bound.cloned().unwrap_or_else(|| BigInt::from(1_u8));
    if stride.sign() == Sign::NoSign {
        return Err(AggregateSliceError::ZeroStep);
    }

    let negative_step = stride.sign() == Sign::Minus;
    let lower = if negative_step {
        BigInt::from(-1_i8)
    } else {
        BigInt::from(0_u8)
    };
    let upper = if negative_step {
        &extent - 1_u8
    } else {
        extent.clone()
    };
    let normalize = |value: &BigInt| {
        let shifted = if value.sign() == Sign::Minus {
            &extent + value
        } else {
            value.clone()
        };
        shifted.max(lower.clone()).min(upper.clone())
    };
    let mut position = start_bound.map_or_else(
        || {
            if negative_step {
                &extent - 1_u8
            } else {
                BigInt::from(0_u8)
            }
        },
        normalize,
    );
    let stop = stop_bound.map_or_else(
        || {
            if negative_step {
                BigInt::from(-1_i8)
            } else {
                extent.clone()
            }
        },
        normalize,
    );

    let mut result = Vec::new();
    while if negative_step {
        position > stop
    } else {
        position < stop
    } {
        if result.len() == MAX_AGGREGATE_SLICE_ITEMS {
            return Err(AggregateSliceError::OutputBudget);
        }
        result.push(
            position
                .to_usize()
                .ok_or(AggregateSliceError::UnrepresentablePosition)?,
        );
        position += &stride;
    }
    Ok(result.into_boxed_slice())
}

/// Diagnose domain violations established by exact operands.
pub(crate) fn known_domain_error(
    operation: &wire::OperationProto,
    known: &HashMap<u64, Option<StaticValue>>,
) -> Option<&'static str> {
    if matches!(
        operation.opcode.as_str(),
        "classical.index" | "classical.update"
    ) {
        let aggregate = operation.operands.first()?;
        let selector = operation.operands.get(1)?;
        let StaticValue::Aggregate(values) = known.get(aggregate)?.as_ref()? else {
            return None;
        };
        let index = known.get(selector)?.as_ref()?.scalar().and_then(integer)?;
        if !aggregate_index_is_in_bounds(values.len(), index) {
            return Some("index out of bounds");
        }
    }
    let denominator = known.get(operation.operands.get(1)?)?.as_ref()?.scalar()?;
    let zero = match denominator {
        StaticScalar::Int(value) => value == 0,
        StaticScalar::RealBits(value) => f64::from_bits(value) == 0.0,
        StaticScalar::Bool(_) => false,
    };
    (zero
        && matches!(
            operation.opcode.as_str(),
            "classical.truediv" | "classical.floordiv" | "classical.mod"
        ))
    .then_some("division by zero")
}

#[cfg(test)]
fn evaluate(
    operation: &wire::OperationProto,
    environment: &HashMap<u64, Option<StaticScalar>>,
) -> Option<StaticScalar> {
    let values = environment
        .iter()
        .map(|(identifier, value)| (*identifier, value.map(StaticValue::Scalar)))
        .collect::<HashMap<_, _>>();
    evaluate_value_checked(operation, &values)
        .ok()
        .flatten()?
        .scalar()
}

fn evaluate_boolean(
    operation: &wire::OperationProto,
    environment: &HashMap<u64, Option<StaticScalar>>,
) -> Option<StaticScalar> {
    let operands = operation
        .operands
        .iter()
        .map(|operand| environment.get(operand).copied().flatten())
        .collect::<Option<Vec<_>>>()?;
    match operation.opcode.as_str() {
        "classical.and_bits" => binary_bits(&operands, |left, right| left & right),
        "classical.or_bits" => binary_bits(&operands, |left, right| left | right),
        "classical.xor" => binary_bits(&operands, |left, right| left ^ right),
        "classical.and" => Some(StaticScalar::Bool(operands.iter().copied().all(truth))),
        "classical.or" => Some(StaticScalar::Bool(operands.iter().copied().any(truth))),
        "classical.not" => unary(&operands).map(|value| StaticScalar::Bool(!truth(value))),
        "classical.invert" => match unary(&operands)? {
            StaticScalar::Bool(value) => Some(StaticScalar::Bool(!value)),
            StaticScalar::Int(value) => Some(StaticScalar::Int(!value)),
            StaticScalar::RealBits(_) => None,
        },
        "classical.cast" => match string_attribute(operation, "target")? {
            "Bool" => unary(&operands).map(|value| StaticScalar::Bool(truth(value))),
            "Int" => unary(&operands).and_then(|value| match value {
                StaticScalar::Bool(value) => Some(StaticScalar::Int(i64::from(value))),
                StaticScalar::Int(_) | StaticScalar::RealBits(_) => None,
            }),
            _ => None,
        },
        _ => None,
    }
}

/// Return Python truthiness for one exact scalar.
pub(crate) const fn truth(value: StaticScalar) -> bool {
    match value {
        StaticScalar::Bool(value) => value,
        StaticScalar::Int(value) => value != 0,
        StaticScalar::RealBits(value) => f64::from_bits(value) != 0.0,
    }
}

/// Return an exact integer, rejecting Boolean values.
pub(crate) const fn integer(value: StaticScalar) -> Option<i64> {
    match value {
        StaticScalar::Int(value) => Some(value),
        StaticScalar::Bool(_) | StaticScalar::RealBits(_) => None,
    }
}

/// Compute Python `range` cardinality without enumerating its elements.
pub(crate) fn range_len(lower: i64, upper: i64, stride: i64) -> Option<u128> {
    if stride == 0 {
        return None;
    }
    let (distance, step_size) = if stride > 0 {
        if lower >= upper {
            return Some(0);
        }
        (i128::from(upper) - i128::from(lower), i128::from(stride))
    } else {
        if lower <= upper {
            return Some(0);
        }
        (i128::from(lower) - i128::from(upper), -i128::from(stride))
    };
    u128::try_from((distance + step_size - 1) / step_size).ok()
}

fn unary(operands: &[StaticScalar]) -> Option<StaticScalar> {
    let [operand] = operands else {
        return None;
    };
    Some(*operand)
}

fn binary_bits(operands: &[StaticScalar], operation: fn(i64, i64) -> i64) -> Option<StaticScalar> {
    let [left, right] = operands else {
        return None;
    };
    match (*left, *right) {
        (StaticScalar::Bool(left), StaticScalar::Bool(right)) => Some(StaticScalar::Bool(
            operation(i64::from(left), i64::from(right)) != 0,
        )),
        (left, right) => Some(StaticScalar::Int(operation(
            scalar_bits(left)?,
            scalar_bits(right)?,
        ))),
    }
}

const fn scalar_bits(value: StaticScalar) -> Option<i64> {
    match value {
        StaticScalar::Bool(value) => Some(value as i64),
        StaticScalar::Int(value) => Some(value),
        StaticScalar::RealBits(_) => None,
    }
}

fn compare_value_chain(
    operation: &wire::OperationProto,
    environment: &HashMap<u64, Option<StaticValue>>,
) -> Option<StaticValue> {
    use wire::literal_proto::Value;

    let Value::TupleValue(sequence) = attribute(operation, "operators")?.value.as_ref()? else {
        return None;
    };
    if operation.operands.len() != sequence.values.len() + 1 {
        return None;
    }
    for (operator, operands) in sequence.values.iter().zip(operation.operands.windows(2)) {
        let Some(Value::StringValue(operator)) = operator.value.as_ref() else {
            return None;
        };
        let left = environment.get(&operands[0])?.as_ref()?;
        let right = environment.get(&operands[1])?.as_ref()?;
        if !compare_value(operator, left, right)? {
            return Some(StaticValue::Scalar(StaticScalar::Bool(false)));
        }
    }
    Some(StaticValue::Scalar(StaticScalar::Bool(true)))
}

fn compare_value(operator: &str, left: &StaticValue, right: &StaticValue) -> Option<bool> {
    match operator {
        "eq" => value_equal(left, right),
        "ne" => value_equal(left, right).map(|equal| !equal),
        "in" | "not_in" => {
            let StaticValue::Aggregate(values) = right else {
                return None;
            };
            let mut unknown = false;
            for candidate in values.iter() {
                let Some(candidate) = candidate else {
                    unknown = true;
                    continue;
                };
                if value_equal(left, candidate)? {
                    return Some(operator == "in");
                }
            }
            (!unknown).then_some(operator == "not_in")
        }
        "lt" => Some(numeric_values::order(left, right)? == std::cmp::Ordering::Less),
        "le" => Some(numeric_values::order(left, right)? != std::cmp::Ordering::Greater),
        "gt" => Some(numeric_values::order(left, right)? == std::cmp::Ordering::Greater),
        "ge" => Some(numeric_values::order(left, right)? != std::cmp::Ordering::Less),
        _ => compare_static(operator, left.scalar()?, right.scalar()?),
    }
}

#[allow(clippy::float_cmp)]
fn value_equal(left: &StaticValue, right: &StaticValue) -> Option<bool> {
    if let Some(equal) = numeric_values::equal(left, right) {
        return Some(equal);
    }
    match (left, right) {
        (StaticValue::Unit, StaticValue::Unit) => Some(true),
        (StaticValue::Scalar(left), StaticValue::Scalar(right)) => scalar_equal(*left, *right),
        (StaticValue::BigInt(left), StaticValue::BigInt(right))
        | (StaticValue::Bytes(left), StaticValue::Bytes(right)) => Some(left == right),
        (
            StaticValue::ComplexBits(left_real, left_imaginary),
            StaticValue::ComplexBits(right_real, right_imaginary),
        ) => Some(
            f64::from_bits(*left_real) == f64::from_bits(*right_real)
                && f64::from_bits(*left_imaginary) == f64::from_bits(*right_imaginary),
        ),
        (StaticValue::String(left), StaticValue::String(right)) => Some(left == right),
        (StaticValue::Aggregate(left), StaticValue::Aggregate(right)) => {
            if left.len() != right.len() {
                return Some(false);
            }
            for (left, right) in left.iter().zip(right.iter()) {
                let (Some(left), Some(right)) = (left, right) else {
                    return None;
                };
                if !value_equal(left, right)? {
                    return Some(false);
                }
            }
            Some(true)
        }
        (StaticValue::Record(left), StaticValue::Record(right)) => {
            if left.len() != right.len() {
                return Some(false);
            }
            for ((left_name, left_value), (right_name, right_value)) in
                left.iter().zip(right.iter())
            {
                let (Some(left_value), Some(right_value)) = (left_value, right_value) else {
                    return None;
                };
                if left_name != right_name || !value_equal(left_value, right_value)? {
                    return Some(false);
                }
            }
            Some(true)
        }
        (StaticValue::Mapping(left), StaticValue::Mapping(right)) => {
            if left.len() != right.len() {
                return Some(false);
            }
            for ((left_key, left_value), (right_key, right_value)) in left.iter().zip(right.iter())
            {
                if left_key != right_key || !value_equal(left_value, right_value)? {
                    return Some(false);
                }
            }
            Some(true)
        }
        _ => None,
    }
}

fn compare_static(operator: &str, left: StaticScalar, right: StaticScalar) -> Option<bool> {
    match operator {
        "eq" => scalar_equal(left, right),
        "ne" => scalar_equal(left, right).map(|equal| !equal),
        "is" => Some(left == right),
        "is_not" => Some(left != right),
        "lt" => Some(scalar_order(left, right)? == std::cmp::Ordering::Less),
        "le" => Some(scalar_order(left, right)? != std::cmp::Ordering::Greater),
        "gt" => Some(scalar_order(left, right)? == std::cmp::Ordering::Greater),
        "ge" => Some(scalar_order(left, right)? != std::cmp::Ordering::Less),
        _ => None,
    }
}

#[allow(clippy::float_cmp)]
fn scalar_equal(left: StaticScalar, right: StaticScalar) -> Option<bool> {
    match (left, right) {
        (StaticScalar::Bool(left), StaticScalar::Bool(right)) => Some(left == right),
        _ => None,
    }
}

fn scalar_order(left: StaticScalar, right: StaticScalar) -> Option<std::cmp::Ordering> {
    match (left, right) {
        (StaticScalar::Bool(left), StaticScalar::Bool(right)) => Some(left.cmp(&right)),
        _ => None,
    }
}

fn attribute<'a>(
    operation: &'a wire::OperationProto,
    name: &str,
) -> Option<&'a wire::LiteralProto> {
    operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .and_then(|attribute| attribute.value.as_ref())
}

fn string_attribute<'a>(operation: &'a wire::OperationProto, name: &str) -> Option<&'a str> {
    use wire::literal_proto::Value;

    match attribute(operation, name)?.value.as_ref()? {
        Value::StringValue(value) => Some(value),
        _ => None,
    }
}

fn int_attribute(operation: &wire::OperationProto, name: &str) -> Option<i64> {
    use wire::literal_proto::Value;

    match attribute(operation, name)?.value.as_ref()? {
        Value::IntValue(value) => Some(*value),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observable_operation(
        opcode: &str,
        operands: Vec<u64>,
        axis: Option<&str>,
    ) -> wire::OperationProto {
        wire::OperationProto {
            opcode: opcode.to_owned(),
            operands,
            attributes: axis
                .map(|axis| wire::AttributeProto {
                    name: "axis".to_owned(),
                    value: Some(wire::LiteralProto {
                        value: Some(wire::literal_proto::Value::StringValue(axis.to_owned())),
                    }),
                })
                .into_iter()
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn evaluates_symbolic_observable_construction_through_canonical_algebra() {
        let mut environment = HashMap::from([
            (1, Some(StaticValue::Scalar(StaticScalar::Int(2)))),
            (2, Some(StaticValue::Scalar(StaticScalar::Int(0)))),
            (
                3,
                Some(StaticValue::Scalar(StaticScalar::RealBits(
                    0.5_f64.to_bits(),
                ))),
            ),
        ]);
        let zero = evaluate_value_checked(
            &observable_operation("classical.observable_zero", vec![1], None),
            &environment,
        )
        .expect("valid zero")
        .expect("known zero");
        let pauli = evaluate_value_checked(
            &observable_operation("classical.observable_pauli", vec![2], Some("X")),
            &environment,
        )
        .expect("valid Pauli")
        .expect("known Pauli");
        environment.insert(4, Some(pauli));
        let scaled = evaluate_value_checked(
            &observable_operation("classical.mul", vec![3, 4], None),
            &environment,
        )
        .expect("valid scale")
        .expect("known scaled Pauli");
        environment.insert(5, Some(zero));
        environment.insert(6, Some(scaled));
        let sum = evaluate_value_checked(
            &observable_operation("classical.add", vec![5, 6], None),
            &environment,
        )
        .expect("valid sum")
        .expect("known sum");
        let actual = value_to_literal(&sum)
            .expect("portable sum")
            .encode_to_vec();
        let expected = normalize_real_observable(2, 0.0, &[(vec![(0, Pauli::X)], 0.5)])
            .expect("expected observable");

        assert_eq!(actual, expected.literal());
    }

    #[test]
    fn aggregate_index_normalization_matches_python_for_arbitrary_integers() {
        let position = |index: i64| normalize_aggregate_index(3, &BigInt::from(index));

        assert_eq!(position(0), Some(0));
        assert_eq!(position(2), Some(2));
        assert_eq!(position(-1), Some(2));
        assert_eq!(position(-3), Some(0));
        assert_eq!(position(3), None);
        assert_eq!(position(-4), None);
        assert_eq!(normalize_aggregate_index(0, &BigInt::from(0)), None);

        let huge = BigInt::from(1_u8) << 1024;
        assert_eq!(normalize_aggregate_index(3, &huge), None);
        assert_eq!(normalize_aggregate_index(3, &-huge), None);
    }

    #[test]
    fn aggregate_slice_normalization_matches_python_direction_and_clamping() {
        let value = |value: i64| BigInt::from(value);
        let positions = |start: Option<i64>, stop: Option<i64>, step: Option<i64>| {
            normalize_aggregate_slice(
                5,
                start.map(value).as_ref(),
                stop.map(value).as_ref(),
                step.map(value).as_ref(),
            )
            .map(<[usize]>::into_vec)
        };

        assert_eq!(positions(None, None, None), Ok(vec![0, 1, 2, 3, 4]));
        assert_eq!(positions(None, None, Some(-1)), Ok(vec![4, 3, 2, 1, 0]));
        assert_eq!(positions(Some(1), Some(4), Some(2)), Ok(vec![1, 3]));
        assert_eq!(positions(Some(-4), Some(-1), None), Ok(vec![1, 2, 3]));
        assert_eq!(positions(Some(99), Some(-99), Some(-2)), Ok(vec![4, 2, 0]));
        assert_eq!(positions(None, Some(-1), None), Ok(vec![0, 1, 2, 3]));
        assert_eq!(positions(Some(1), Some(1), None), Ok(vec![]));
        assert_eq!(
            positions(None, None, Some(0)),
            Err(AggregateSliceError::ZeroStep)
        );
        assert_eq!(
            normalize_aggregate_slice(MAX_AGGREGATE_SLICE_ITEMS + 1, None, None, None),
            Err(AggregateSliceError::OutputBudget)
        );
    }

    #[test]
    fn scalar_projection_shares_numeric_semantics_and_retains_boolean_invert() {
        let operation = |opcode: &str| wire::OperationProto {
            opcode: opcode.to_owned(),
            operands: vec![1],
            ..Default::default()
        };

        assert_eq!(
            evaluate(
                &operation("classical.neg"),
                &HashMap::from([(1, Some(StaticScalar::Int(i64::MIN)))])
            ),
            None
        );
        assert_eq!(
            evaluate(
                &operation("classical.invert"),
                &HashMap::from([(1, Some(StaticScalar::Bool(true)))])
            ),
            Some(StaticScalar::Bool(false))
        );
    }

    #[test]
    fn range_cardinality_matches_python_sign_rules() {
        assert_eq!(range_len(0, 10, 3), Some(4));
        assert_eq!(range_len(10, 0, -3), Some(4));
        assert_eq!(range_len(10, 0, 3), Some(0));
        assert_eq!(range_len(0, 10, -3), Some(0));
        assert_eq!(range_len(0, 10, 0), None);
    }

    #[test]
    fn portable_value_round_trip_preserves_nested_non_scalar_literals() {
        use wire::literal_proto::Value;

        let literal = wire::LiteralProto {
            value: Some(Value::MappingValue(wire::literal_proto::Mapping {
                entries: vec![wire::literal_proto::Entry {
                    key: "payload".to_owned(),
                    value: Some(wire::LiteralProto {
                        value: Some(Value::TupleValue(wire::literal_proto::Sequence {
                            values: vec![
                                wire::LiteralProto {
                                    value: Some(Value::BytesValue(vec![1, 2, 3])),
                                },
                                wire::LiteralProto {
                                    value: Some(Value::ComplexValue(
                                        wire::literal_proto::Complex {
                                            real: -0.0,
                                            imaginary: 2.5,
                                        },
                                    )),
                                },
                            ],
                        })),
                    }),
                }],
            })),
        };

        let value = value_literal(&literal).expect("closed portable value");
        assert!(value.is_complete());
        assert_eq!(value_to_literal(&value), Some(literal));
    }

    #[test]
    fn aggregate_length_does_not_require_known_elements() {
        let operation = wire::OperationProto {
            opcode: "classical.len".to_owned(),
            operands: vec![1],
            ..Default::default()
        };
        let environment = HashMap::from([(
            1,
            Some(StaticValue::Aggregate(
                vec![Some(StaticValue::Scalar(StaticScalar::Int(3))), None].into(),
            )),
        )]);

        assert_eq!(
            evaluate_value(&operation, &environment),
            Some(StaticValue::Scalar(StaticScalar::Int(2)))
        );
    }

    #[test]
    fn typed_record_access_uses_declaration_order_not_wire_key_order() {
        use wire::literal_proto::Value;

        let integer_type = || wire::TypeProto {
            kind: wire::type_proto::Kind::Scalar.into(),
            name: "Int".to_owned(),
            ..Default::default()
        };
        let record_type = wire::TypeProto {
            kind: wire::type_proto::Kind::Record.into(),
            name: "test.Coordinates".to_owned(),
            arguments: vec![integer_type(), integer_type()],
            field_names: vec!["zeta".to_owned(), "alpha".to_owned()],
            ..Default::default()
        };
        let literal = wire::LiteralProto {
            value: Some(Value::MappingValue(wire::literal_proto::Mapping {
                entries: vec![
                    wire::literal_proto::Entry {
                        key: "alpha".to_owned(),
                        value: Some(wire::LiteralProto {
                            value: Some(Value::IntValue(2)),
                        }),
                    },
                    wire::literal_proto::Entry {
                        key: "zeta".to_owned(),
                        value: Some(wire::LiteralProto {
                            value: Some(Value::IntValue(9)),
                        }),
                    },
                ],
            })),
        };
        let value = typed_value_literal(&literal, &record_type).expect("typed record");
        let operation = wire::OperationProto {
            opcode: "classical.project".to_owned(),
            operands: vec![1],
            attributes: vec![wire::AttributeProto {
                name: "position".to_owned(),
                value: Some(wire::LiteralProto {
                    value: Some(Value::IntValue(0)),
                }),
            }],
            ..Default::default()
        };

        assert_eq!(
            evaluate_value(&operation, &HashMap::from([(1, Some(value.clone()))])),
            Some(StaticValue::Scalar(StaticScalar::Int(9)))
        );
        let index = wire::OperationProto {
            opcode: "classical.index".to_owned(),
            operands: vec![1, 2],
            ..Default::default()
        };
        let indexed_environment = HashMap::from([
            (1, Some(value.clone())),
            (2, Some(StaticValue::Scalar(StaticScalar::Int(1)))),
        ]);

        assert_eq!(
            evaluate_value(&index, &indexed_environment),
            Some(StaticValue::Scalar(StaticScalar::Int(2)))
        );
        assert_eq!(value_to_literal(&value), Some(literal));
    }
}
