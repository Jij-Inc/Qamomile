//! Canonical sampled-result plans shared by every static backend.

use std::collections::{BTreeMap, HashSet};

use prost::Message;
use thiserror::Error;

use crate::budget;
use crate::circuit::StaticCircuitValue;
use crate::literals::validate_literal_references;
use crate::result_codec::checked_result_abi;
use crate::simulation::{StaticSampleCounts, evaluate_retained_result};
use crate::static_values;
use crate::wire;
use crate::{
    CircuitLaunchBinding, CircuitParameterPath, CircuitScalarAttribute,
    CircuitScalarAttributeValue, CircuitScalarId, CircuitScalarNode, CircuitScalarType,
    StaticCircuit,
};

const PLAN_SCHEMA_MAJOR: u32 = 1;
const PLAN_SCHEMA_MINOR: u32 = 1;
const MAX_PLAN_BYTES: usize = 64 * 1024 * 1024;
const MAX_MEASUREMENTS: u64 = 1_000_000;
const MAX_DISTRIBUTION_ROWS: usize = 1_000_000;

/// Failure to seal or execute one backend-neutral sampled-result plan.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ResultPostprocessingError {
    /// The persistent plan is malformed, non-canonical, or unsupported.
    #[error("Q4451: invalid native result postprocessing plan: {0}")]
    InvalidPlan(String),
    /// A backend distribution row is not a valid measurement sample.
    #[error("Q4452: invalid native measurement distribution: {0}")]
    InvalidCounts(String),
    /// The retained scalar DAG cannot be evaluated for one backend row.
    #[error("Q4453: native result postprocessing failed: {0}")]
    Evaluation(String),
}

/// Compiler-derived execution route for one static circuit result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StaticCircuitResultRoute {
    /// The entrypoint returns an ordinary value instead of sampled data.
    Value,
    /// The logical result is the source-ordered direct measurement row.
    DirectMeasurements,
    /// The logical result requires the enclosed portable native plan.
    Postprocessing,
    /// The result requires runtime-local call state unavailable to a static backend.
    RuntimeLocal,
}

impl StaticCircuitResultRoute {
    /// Return the closed bridge label for this route.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Value => "value",
            Self::DirectMeasurements => "direct_measurements",
            Self::Postprocessing => "postprocessing",
            Self::RuntimeLocal => "runtime_local",
        }
    }
}

/// One checked static-result route and its optional portable evaluator.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticCircuitResultProjection {
    route: StaticCircuitResultRoute,
    postprocessing: Option<Vec<u8>>,
}

impl StaticCircuitResultProjection {
    /// Return the compiler-derived result route.
    #[must_use]
    pub const fn route(&self) -> StaticCircuitResultRoute {
        self.route
    }

    /// Consume the projection and return its optional portable evaluator.
    #[must_use]
    pub fn into_postprocessing(self) -> Option<Vec<u8>> {
        self.postprocessing
    }
}

/// Classify one static result through its verified sampled width.
///
/// # Errors
///
/// Returns [`ResultPostprocessingError`] if direct measurement metadata
/// disagrees with the verified result ABI or a retained result cannot be
/// represented canonically.
pub fn project_result_postprocessing(
    circuit: &StaticCircuit,
    sampled_width: u32,
) -> Result<StaticCircuitResultProjection, ResultPostprocessingError> {
    let direct_width = u32::try_from(circuit.result_measurements().len()).map_err(|_| {
        ResultPostprocessingError::InvalidPlan(
            "direct measurement width exceeds the result ABI".to_owned(),
        )
    })?;
    if sampled_width == 0 {
        if direct_width != 0 {
            return Err(ResultPostprocessingError::InvalidPlan(
                "value result retains direct measurement metadata".to_owned(),
            ));
        }
        return Ok(StaticCircuitResultProjection {
            route: StaticCircuitResultRoute::Value,
            postprocessing: None,
        });
    }
    if direct_width != 0 && !circuit.has_terminal_checks() {
        if direct_width != sampled_width {
            return Err(ResultPostprocessingError::InvalidPlan(format!(
                "direct measurement width {direct_width} disagrees with sampled width {sampled_width}"
            )));
        }
        return Ok(StaticCircuitResultProjection {
            route: StaticCircuitResultRoute::DirectMeasurements,
            postprocessing: None,
        });
    }
    let postprocessing = encode_result_postprocessing(circuit)?;
    Ok(StaticCircuitResultProjection {
        route: if postprocessing.is_some() {
            StaticCircuitResultRoute::Postprocessing
        } else {
            StaticCircuitResultRoute::RuntimeLocal
        },
        postprocessing,
    })
}

/// Encode the result-reachable scalar DAG of one compiler-owned circuit.
///
/// # Errors
///
/// Returns [`ResultPostprocessingError`] if the retained result depends on a
/// runtime-only call slot or cannot be represented canonically.
pub fn encode_result_postprocessing(
    circuit: &StaticCircuit,
) -> Result<Option<Vec<u8>>, ResultPostprocessingError> {
    if !circuit.result_measurements().is_empty() && !circuit.has_terminal_checks() {
        return Ok(None);
    }
    let Some((scalars, result)) = circuit
        .compact_result_plan()
        .map_err(|error| ResultPostprocessingError::InvalidPlan(error.to_string()))?
    else {
        return Ok(None);
    };
    if scalars
        .iter()
        .any(|scalar| matches!(scalar, CircuitScalarNode::CallValue { .. }))
    {
        return Ok(None);
    }
    let scalars = scalars
        .iter()
        .map(encode_scalar)
        .collect::<Result<Vec<_>, _>>()?;
    let result = encode_result(&result)?;
    Ok(Some(
        wire::CircuitResultPostprocessingProto {
            schema_major: PLAN_SCHEMA_MAJOR,
            schema_minor: u32::from(circuit.has_terminal_checks()),
            measurement_count: circuit.measurement_count(),
            scalars,
            result: Some(result),
        }
        .encode_to_vec(),
    ))
}

/// Evaluate raw backend counts and return logical source-ordered counts.
///
/// # Errors
///
/// Returns [`ResultPostprocessingError`] if the plan, distribution, launch
/// values, retained expression, or result ABI is invalid.
pub fn postprocess_static_counts(
    payload: &[u8],
    counts: &[(Box<[u8]>, u64)],
    bindings: &[CircuitLaunchBinding],
    result_abi: &[u8],
) -> Result<StaticSampleCounts, ResultPostprocessingError> {
    if counts.len() > MAX_DISTRIBUTION_ROWS {
        return Err(ResultPostprocessingError::InvalidCounts(
            "distribution exceeds its row budget".to_owned(),
        ));
    }
    let decoded = decode_plan(payload)?;
    let circuit = &decoded.circuit;
    let result_abi = checked_result_abi(result_abi)
        .map_err(|error| ResultPostprocessingError::Evaluation(error.to_string()))?;
    let measurement_count = usize::try_from(circuit.measurement_count()).map_err(|_| {
        ResultPostprocessingError::InvalidPlan("measurement width exceeds usize".to_owned())
    })?;
    let mut logical = BTreeMap::<Box<[u8]>, u64>::new();
    for (measurements, count) in counts {
        if measurements.len() != measurement_count || measurements.iter().any(|bit| *bit > 1) {
            return Err(ResultPostprocessingError::InvalidCounts(
                "row width or bit value disagrees with the plan".to_owned(),
            ));
        }
        let mut literal = evaluate_retained_result(circuit, bindings, measurements)
            .map_err(|error| ResultPostprocessingError::Evaluation(error.to_string()))?;
        if decoded.checked_result {
            let Some(wire::literal_proto::Value::TupleValue(mut tuple)) = literal.value else {
                return Err(ResultPostprocessingError::InvalidPlan(
                    "checked result tuple".to_owned(),
                ));
            };
            if tuple.values.len() != 2 {
                return Err(ResultPostprocessingError::InvalidPlan(
                    "checked result arity".to_owned(),
                ));
            }
            literal = tuple.values.pop().ok_or_else(|| {
                ResultPostprocessingError::InvalidPlan("checked result arity".to_owned())
            })?;
        }
        let bits = result_abi
            .encode_literal(&literal)
            .map_err(|error| ResultPostprocessingError::Evaluation(error.to_string()))?
            .into_boxed_slice();
        let total = logical.entry(bits).or_default();
        *total = total.checked_add(*count).ok_or_else(|| {
            ResultPostprocessingError::InvalidCounts("count total overflows u64".to_owned())
        })?;
    }
    Ok(StaticSampleCounts::from_entries(logical))
}

pub(crate) fn encode_scalar(
    scalar: &CircuitScalarNode,
) -> Result<wire::CircuitResultScalarProto, ResultPostprocessingError> {
    use wire::circuit_result_scalar_proto::Node;

    let node = match scalar {
        CircuitScalarNode::LoopIndex(_) => {
            return Err(ResultPostprocessingError::InvalidPlan(
                "scoped loop index escaped its body".to_owned(),
            ));
        }
        CircuitScalarNode::Bool(value) => Node::BoolValue(*value),
        CircuitScalarNode::Int(value) => Node::IntValue(*value),
        CircuitScalarNode::RealBits(value) => Node::RealBits(*value),
        CircuitScalarNode::Measurement(index) => Node::Measurement(*index),
        CircuitScalarNode::Parameter {
            name,
            path,
            scalar_type,
        } => Node::Parameter(wire::CircuitResultParameterProto {
            name: name.clone(),
            path: path
                .iter()
                .map(|component| wire::CircuitResultPathProto {
                    component: Some(match component {
                        CircuitParameterPath::Index(index) => {
                            wire::circuit_result_path_proto::Component::Index(*index)
                        }
                        CircuitParameterPath::Field(field) => {
                            wire::circuit_result_path_proto::Component::Field(field.clone())
                        }
                    }),
                })
                .collect(),
            r#type: encode_scalar_type(*scalar_type).into(),
        }),
        CircuitScalarNode::Apply {
            opcode,
            result_type,
            attributes,
            operands,
        } => {
            let mut attributes = attributes.iter().map(encode_attribute).collect::<Vec<_>>();
            attributes.sort_by(|left, right| left.name.cmp(&right.name));
            Node::Apply(wire::CircuitResultApplyProto {
                opcode: opcode.clone(),
                r#type: encode_scalar_type(*result_type).into(),
                attributes,
                operands: operands.iter().map(|operand| operand.index()).collect(),
            })
        }
        CircuitScalarNode::CallValue { .. } => {
            return Err(ResultPostprocessingError::InvalidPlan(
                "result depends on a runtime-local scalar".to_owned(),
            ));
        }
    };
    Ok(wire::CircuitResultScalarProto { node: Some(node) })
}

fn encode_attribute(attribute: &CircuitScalarAttribute) -> wire::AttributeProto {
    use wire::literal_proto::Value;

    let value = match attribute.value() {
        CircuitScalarAttributeValue::Int(value) => Value::IntValue(*value),
        CircuitScalarAttributeValue::String(value) => Value::StringValue(value.clone()),
        CircuitScalarAttributeValue::Strings(values) => {
            Value::TupleValue(wire::literal_proto::Sequence {
                values: values
                    .iter()
                    .map(|value| wire::LiteralProto {
                        value: Some(Value::StringValue(value.clone())),
                    })
                    .collect(),
            })
        }
    };
    wire::AttributeProto {
        name: attribute.name().to_owned(),
        value: Some(wire::LiteralProto { value: Some(value) }),
    }
}

fn encode_result(
    result: &StaticCircuitValue,
) -> Result<wire::CircuitResultValueProto, ResultPostprocessingError> {
    use wire::circuit_result_value_proto::Value;

    let value = match result {
        StaticCircuitValue::Unit => Value::Unit(true),
        StaticCircuitValue::Portable(value) => {
            Value::Literal(static_values::value_to_literal(value).ok_or_else(|| {
                ResultPostprocessingError::InvalidPlan(
                    "result contains an incomplete portable value".to_owned(),
                )
            })?)
        }
        StaticCircuitValue::Scalar(identifier) => Value::Scalar(identifier.index()),
        StaticCircuitValue::Aggregate(values) => {
            Value::Aggregate(wire::CircuitResultSequenceProto {
                values: values
                    .iter()
                    .map(encode_result)
                    .collect::<Result<Vec<_>, _>>()?,
            })
        }
        StaticCircuitValue::Record(fields) => Value::Record(wire::CircuitResultRecordProto {
            fields: fields
                .iter()
                .map(|(name, value)| {
                    Ok(wire::CircuitResultFieldProto {
                        name: name.clone(),
                        value: Some(encode_result(value)?),
                    })
                })
                .collect::<Result<Vec<_>, ResultPostprocessingError>>()?,
        }),
        StaticCircuitValue::Wires(_) => {
            return Err(ResultPostprocessingError::InvalidPlan(
                "result contains coherent wires".to_owned(),
            ));
        }
    };
    Ok(wire::CircuitResultValueProto { value: Some(value) })
}

struct DecodedResultPlan {
    circuit: StaticCircuit,
    checked_result: bool,
}

fn decode_plan(payload: &[u8]) -> Result<DecodedResultPlan, ResultPostprocessingError> {
    if payload.len() > MAX_PLAN_BYTES {
        return Err(ResultPostprocessingError::InvalidPlan(
            "plan exceeds its byte budget".to_owned(),
        ));
    }
    budget::preflight_result_postprocessing(payload)
        .map_err(|error| ResultPostprocessingError::InvalidPlan(preflight_reason(error)))?;
    let plan = wire::CircuitResultPostprocessingProto::decode(payload)
        .map_err(|error| ResultPostprocessingError::InvalidPlan(error.to_string()))?;
    if plan.encode_to_vec() != payload {
        return Err(ResultPostprocessingError::InvalidPlan(
            "plan is not canonical".to_owned(),
        ));
    }
    if plan.schema_major != PLAN_SCHEMA_MAJOR || plan.schema_minor > PLAN_SCHEMA_MINOR {
        return Err(ResultPostprocessingError::InvalidPlan(format!(
            "unsupported schema {}.{}",
            plan.schema_major, plan.schema_minor
        )));
    }
    if plan.measurement_count > MAX_MEASUREMENTS {
        return Err(ResultPostprocessingError::InvalidPlan(
            "measurement width exceeds its budget".to_owned(),
        ));
    }
    validate_plan_literals(&plan)?;
    let scalars = plan
        .scalars
        .iter()
        .enumerate()
        .map(|(index, scalar)| decode_scalar(scalar, index, plan.measurement_count))
        .collect::<Result<Vec<_>, _>>()?
        .into_boxed_slice();
    let result = decode_result(
        plan.result.as_ref().ok_or_else(|| {
            ResultPostprocessingError::InvalidPlan("missing result root".to_owned())
        })?,
        scalars.len(),
    )?;
    if plan.schema_minor == 1 {
        let StaticCircuitValue::Aggregate(values) = &result else {
            return Err(ResultPostprocessingError::InvalidPlan(
                "checked result structure".to_owned(),
            ));
        };
        if values.len() != 2 || !matches!(values[0], StaticCircuitValue::Aggregate(_)) {
            return Err(ResultPostprocessingError::InvalidPlan(
                "checked result structure".to_owned(),
            ));
        }
    }
    Ok(DecodedResultPlan {
        circuit: StaticCircuit::from_result_plan(plan.measurement_count, scalars, result),
        checked_result: plan.schema_minor == 1,
    })
}

fn validate_plan_literals(
    plan: &wire::CircuitResultPostprocessingProto,
) -> Result<(), ResultPostprocessingError> {
    let mut literals = plan
        .scalars
        .iter()
        .filter_map(|scalar| match scalar.node.as_ref() {
            Some(wire::circuit_result_scalar_proto::Node::Apply(apply)) => Some(
                apply
                    .attributes
                    .iter()
                    .filter_map(|attribute| attribute.value.as_ref()),
            ),
            _ => None,
        })
        .flatten()
        .collect::<Vec<_>>();
    collect_result_literals(plan.result.as_ref(), &mut literals);
    validate_literal_references(literals).map_err(ResultPostprocessingError::InvalidPlan)
}

fn collect_result_literals<'a>(
    result: Option<&'a wire::CircuitResultValueProto>,
    literals: &mut Vec<&'a wire::LiteralProto>,
) {
    use wire::circuit_result_value_proto::Value;

    match result.and_then(|result| result.value.as_ref()) {
        Some(Value::Literal(value)) => literals.push(value),
        Some(Value::Aggregate(values)) => {
            for value in &values.values {
                collect_result_literals(Some(value), literals);
            }
        }
        Some(Value::Record(record)) => {
            for field in &record.fields {
                collect_result_literals(field.value.as_ref(), literals);
            }
        }
        Some(Value::Unit(_) | Value::Scalar(_)) | None => {}
    }
}

pub(crate) fn decode_scalar(
    scalar: &wire::CircuitResultScalarProto,
    index: usize,
    measurement_count: u64,
) -> Result<CircuitScalarNode, ResultPostprocessingError> {
    use wire::circuit_result_scalar_proto::Node;

    match scalar.node.as_ref() {
        Some(Node::BoolValue(value)) => Ok(CircuitScalarNode::Bool(*value)),
        Some(Node::IntValue(value)) => Ok(CircuitScalarNode::Int(*value)),
        Some(Node::RealBits(value)) => Ok(CircuitScalarNode::RealBits(*value)),
        Some(Node::Measurement(measurement)) if *measurement < measurement_count => {
            Ok(CircuitScalarNode::Measurement(*measurement))
        }
        Some(Node::Measurement(_)) => Err(ResultPostprocessingError::InvalidPlan(
            "measurement scalar is outside the raw row".to_owned(),
        )),
        Some(Node::Parameter(parameter)) => {
            if parameter.name.is_empty() {
                return Err(ResultPostprocessingError::InvalidPlan(
                    "empty result parameter name".to_owned(),
                ));
            }
            let path = parameter
                .path
                .iter()
                .map(decode_path)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(CircuitScalarNode::Parameter {
                name: parameter.name.clone(),
                path: path.into_boxed_slice(),
                scalar_type: decode_scalar_type(parameter.r#type)?,
            })
        }
        Some(Node::Apply(apply)) => {
            if apply.opcode.is_empty() {
                return Err(ResultPostprocessingError::InvalidPlan(
                    "empty result scalar opcode".to_owned(),
                ));
            }
            let attributes = decode_attributes(&apply.attributes)?;
            let operands = apply
                .operands
                .iter()
                .map(|operand| {
                    usize::try_from(*operand)
                        .is_ok_and(|operand| operand < index)
                        .then_some(CircuitScalarId::from_index(*operand))
                        .ok_or_else(|| {
                            ResultPostprocessingError::InvalidPlan(
                                "non-topological result scalar operand".to_owned(),
                            )
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(CircuitScalarNode::Apply {
                opcode: apply.opcode.clone(),
                result_type: decode_scalar_type(apply.r#type)?,
                attributes: attributes.into_boxed_slice(),
                operands: operands.into_boxed_slice(),
            })
        }
        None => Err(ResultPostprocessingError::InvalidPlan(
            "missing result scalar node".to_owned(),
        )),
    }
}

fn decode_path(
    path: &wire::CircuitResultPathProto,
) -> Result<CircuitParameterPath, ResultPostprocessingError> {
    match path.component.as_ref() {
        Some(wire::circuit_result_path_proto::Component::Index(index)) => {
            Ok(CircuitParameterPath::Index(*index))
        }
        Some(wire::circuit_result_path_proto::Component::Field(field)) if !field.is_empty() => {
            Ok(CircuitParameterPath::Field(field.clone()))
        }
        _ => Err(ResultPostprocessingError::InvalidPlan(
            "invalid result parameter path".to_owned(),
        )),
    }
}

fn decode_attributes(
    attributes: &[wire::AttributeProto],
) -> Result<Vec<CircuitScalarAttribute>, ResultPostprocessingError> {
    if attributes
        .windows(2)
        .any(|pair| pair[0].name >= pair[1].name)
    {
        return Err(ResultPostprocessingError::InvalidPlan(
            "result scalar attributes are not canonically ordered".to_owned(),
        ));
    }
    attributes
        .iter()
        .map(|attribute| {
            if attribute.name.is_empty() {
                return Err(ResultPostprocessingError::InvalidPlan(
                    "invalid result scalar attribute name".to_owned(),
                ));
            }
            let value = decode_attribute_value(attribute.value.as_ref().ok_or_else(|| {
                ResultPostprocessingError::InvalidPlan(
                    "missing result scalar attribute value".to_owned(),
                )
            })?)?;
            Ok(CircuitScalarAttribute::from_parts(
                attribute.name.clone(),
                value,
            ))
        })
        .collect()
}

fn decode_attribute_value(
    value: &wire::LiteralProto,
) -> Result<CircuitScalarAttributeValue, ResultPostprocessingError> {
    use wire::literal_proto::Value;

    match value.value.as_ref() {
        Some(Value::IntValue(value)) => Ok(CircuitScalarAttributeValue::Int(*value)),
        Some(Value::StringValue(value)) => Ok(CircuitScalarAttributeValue::String(value.clone())),
        Some(Value::TupleValue(sequence)) => Ok(CircuitScalarAttributeValue::Strings(
            sequence
                .values
                .iter()
                .map(|value| match value.value.as_ref() {
                    Some(Value::StringValue(value)) => Ok(value.clone()),
                    _ => Err(ResultPostprocessingError::InvalidPlan(
                        "result scalar string-list attribute is malformed".to_owned(),
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_boxed_slice(),
        )),
        _ => Err(ResultPostprocessingError::InvalidPlan(
            "unsupported result scalar attribute".to_owned(),
        )),
    }
}

fn decode_result(
    result: &wire::CircuitResultValueProto,
    scalar_count: usize,
) -> Result<StaticCircuitValue, ResultPostprocessingError> {
    use wire::circuit_result_value_proto::Value;

    match result.value.as_ref() {
        Some(Value::Unit(true)) => Ok(StaticCircuitValue::Unit),
        Some(Value::Unit(false)) => Err(ResultPostprocessingError::InvalidPlan(
            "unit result marker must be true".to_owned(),
        )),
        Some(Value::Literal(value)) => static_values::value_literal(value)
            .map(StaticCircuitValue::Portable)
            .ok_or_else(|| {
                ResultPostprocessingError::InvalidPlan("invalid portable result literal".to_owned())
            }),
        Some(Value::Scalar(identifier)) => usize::try_from(*identifier)
            .is_ok_and(|index| index < scalar_count)
            .then_some(StaticCircuitValue::Scalar(CircuitScalarId::from_index(
                *identifier,
            )))
            .ok_or_else(|| {
                ResultPostprocessingError::InvalidPlan(
                    "result scalar root is outside its arena".to_owned(),
                )
            }),
        Some(Value::Aggregate(sequence)) => Ok(StaticCircuitValue::Aggregate(
            sequence
                .values
                .iter()
                .map(|value| decode_result(value, scalar_count))
                .collect::<Result<Vec<_>, _>>()?
                .into_boxed_slice(),
        )),
        Some(Value::Record(record)) => {
            let mut names = HashSet::new();
            Ok(StaticCircuitValue::Record(
                record
                    .fields
                    .iter()
                    .map(|field| {
                        if field.name.is_empty() || !names.insert(field.name.as_str()) {
                            return Err(ResultPostprocessingError::InvalidPlan(
                                "invalid result record field".to_owned(),
                            ));
                        }
                        Ok((
                            field.name.clone(),
                            decode_result(
                                field.value.as_ref().ok_or_else(|| {
                                    ResultPostprocessingError::InvalidPlan(
                                        "missing result record value".to_owned(),
                                    )
                                })?,
                                scalar_count,
                            )?,
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into_boxed_slice(),
            ))
        }
        None => Err(ResultPostprocessingError::InvalidPlan(
            "missing result value".to_owned(),
        )),
    }
}

const fn encode_scalar_type(value: CircuitScalarType) -> wire::CircuitResultScalarTypeProto {
    match value {
        CircuitScalarType::Bool => wire::CircuitResultScalarTypeProto::CircuitResultBool,
        CircuitScalarType::Int => wire::CircuitResultScalarTypeProto::CircuitResultInt,
        CircuitScalarType::Real => wire::CircuitResultScalarTypeProto::CircuitResultReal,
    }
}

fn decode_scalar_type(value: i32) -> Result<CircuitScalarType, ResultPostprocessingError> {
    match wire::CircuitResultScalarTypeProto::try_from(value) {
        Ok(wire::CircuitResultScalarTypeProto::CircuitResultBool) => Ok(CircuitScalarType::Bool),
        Ok(wire::CircuitResultScalarTypeProto::CircuitResultInt) => Ok(CircuitScalarType::Int),
        Ok(wire::CircuitResultScalarTypeProto::CircuitResultReal) => Ok(CircuitScalarType::Real),
        _ => Err(ResultPostprocessingError::InvalidPlan(
            "invalid result scalar type".to_owned(),
        )),
    }
}

fn preflight_reason(error: budget::PreflightError) -> String {
    match error {
        budget::PreflightError::Budget(reason) | budget::PreflightError::Malformed(reason) => {
            reason
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bool_result_abi() -> Vec<u8> {
        wire::ResultAbiProto {
            r#type: Some(wire::TypeProto {
                kind: wire::type_proto::Kind::Scalar.into(),
                name: "Bool".to_owned(),
                ..Default::default()
            }),
            source_order: vec![0],
            native_width: 1,
        }
        .encode_to_vec()
    }

    fn xor_result_circuit() -> StaticCircuit {
        StaticCircuit::from_result_plan(
            1,
            vec![
                CircuitScalarNode::Measurement(0),
                CircuitScalarNode::Parameter {
                    name: "enabled".to_owned(),
                    path: Box::new([]),
                    scalar_type: CircuitScalarType::Bool,
                },
                CircuitScalarNode::Apply {
                    opcode: "classical.xor".to_owned(),
                    result_type: CircuitScalarType::Bool,
                    attributes: Box::new([]),
                    operands: vec![
                        CircuitScalarId::from_index(0),
                        CircuitScalarId::from_index(1),
                    ]
                    .into_boxed_slice(),
                },
                // This runtime-local node is deliberately unreachable.
                CircuitScalarNode::CallValue {
                    index: 0,
                    scalar_type: CircuitScalarType::Bool,
                },
            ]
            .into_boxed_slice(),
            StaticCircuitValue::Scalar(CircuitScalarId::from_index(2)),
        )
    }

    #[test]
    fn projects_the_complete_static_result_route_sum() {
        let value = project_result_postprocessing(
            &StaticCircuit::from_result_plan(0, Box::new([]), StaticCircuitValue::Unit),
            0,
        )
        .expect("value route");
        assert_eq!(value.route(), StaticCircuitResultRoute::Value);
        assert_eq!(value.into_postprocessing(), None);

        let direct = project_result_postprocessing(
            &StaticCircuit::test_new(1, 1, Vec::new(), Vec::new(), vec![0]),
            1,
        )
        .expect("direct route");
        assert_eq!(direct.route(), StaticCircuitResultRoute::DirectMeasurements);
        assert_eq!(direct.into_postprocessing(), None);

        let postprocessing =
            project_result_postprocessing(&xor_result_circuit(), 1).expect("postprocessing route");
        assert_eq!(
            postprocessing.route(),
            StaticCircuitResultRoute::Postprocessing
        );
        assert!(postprocessing.into_postprocessing().is_some());

        let runtime_local = project_result_postprocessing(
            &StaticCircuit::from_result_plan(
                0,
                vec![CircuitScalarNode::CallValue {
                    index: 0,
                    scalar_type: CircuitScalarType::Bool,
                }]
                .into_boxed_slice(),
                StaticCircuitValue::Scalar(CircuitScalarId::from_index(0)),
            ),
            1,
        )
        .expect("runtime-local route");
        assert_eq!(
            runtime_local.route(),
            StaticCircuitResultRoute::RuntimeLocal
        );
        assert_eq!(runtime_local.into_postprocessing(), None);
    }

    #[test]
    fn rejects_direct_measurements_that_disagree_with_the_result_abi() {
        let circuit = StaticCircuit::test_new(1, 1, Vec::new(), Vec::new(), vec![0]);

        for sampled_width in [0, 2] {
            assert!(matches!(
                project_result_postprocessing(&circuit, sampled_width),
                Err(ResultPostprocessingError::InvalidPlan(_))
            ));
        }
    }

    #[test]
    fn seals_only_reachable_scalars_and_aggregates_backend_rows() {
        let plan = encode_result_postprocessing(&xor_result_circuit())
            .expect("encodable result plan")
            .expect("retained result");
        let decoded = wire::CircuitResultPostprocessingProto::decode(plan.as_slice())
            .expect("canonical result plan");
        assert_eq!(decoded.scalars.len(), 3);
        let counts = vec![
            (vec![0].into_boxed_slice(), 2),
            (vec![1].into_boxed_slice(), 3),
        ];
        let bindings = vec![CircuitLaunchBinding::new(
            "enabled".to_owned(),
            crate::CircuitLaunchValue::Bool(true),
        )];

        let result = postprocess_static_counts(&plan, &counts, &bindings, &bool_result_abi())
            .expect("postprocessed counts");

        assert_eq!(
            result.entries(),
            [
                (vec![0].into_boxed_slice(), 3),
                (vec![1].into_boxed_slice(), 2),
            ]
        );
    }

    #[test]
    fn rejects_non_topological_scalar_dags_before_evaluation() {
        let plan = encode_result_postprocessing(&xor_result_circuit())
            .expect("encodable result plan")
            .expect("retained result");
        let mut decoded = wire::CircuitResultPostprocessingProto::decode(plan.as_slice())
            .expect("canonical result plan");
        let Some(wire::circuit_result_scalar_proto::Node::Apply(apply)) =
            decoded.scalars[2].node.as_mut()
        else {
            panic!("expected apply node");
        };
        apply.operands[0] = 2;

        assert!(matches!(
            postprocess_static_counts(
                &decoded.encode_to_vec(),
                &[(vec![0].into_boxed_slice(), 1)],
                &[CircuitLaunchBinding::new(
                    "enabled".to_owned(),
                    crate::CircuitLaunchValue::Bool(true),
                )],
                &bool_result_abi(),
            ),
            Err(ResultPostprocessingError::InvalidPlan(message))
                if message.contains("non-topological")
        ));
    }

    #[test]
    fn rejects_wrong_width_and_non_bit_backend_rows() {
        let plan = encode_result_postprocessing(&xor_result_circuit())
            .expect("encodable result plan")
            .expect("retained result");
        let bindings = [CircuitLaunchBinding::new(
            "enabled".to_owned(),
            crate::CircuitLaunchValue::Bool(false),
        )];
        for row in [vec![], vec![0, 1], vec![2]] {
            assert!(matches!(
                postprocess_static_counts(
                    &plan,
                    &[(row.into_boxed_slice(), 1)],
                    &bindings,
                    &bool_result_abi(),
                ),
                Err(ResultPostprocessingError::InvalidCounts(_))
            ));
        }
    }

    #[test]
    fn seals_and_evaluates_a_deep_scalar_dag_without_recursion() {
        let mut scalars = vec![CircuitScalarNode::Measurement(0)];
        for index in 1..20_000_u32 {
            scalars.push(CircuitScalarNode::Apply {
                opcode: "classical.not".to_owned(),
                result_type: CircuitScalarType::Bool,
                attributes: Box::new([]),
                operands: Box::new([CircuitScalarId::from_index(index - 1)]),
            });
        }
        let circuit = StaticCircuit::from_result_plan(
            1,
            scalars.into_boxed_slice(),
            StaticCircuitValue::Scalar(CircuitScalarId::from_index(19_999)),
        );
        let plan = encode_result_postprocessing(&circuit)
            .expect("deep plan")
            .expect("retained result");

        let result = postprocess_static_counts(
            &plan,
            &[(vec![0].into_boxed_slice(), 1)],
            &[],
            &bool_result_abi(),
        )
        .expect("deep result evaluation");

        assert_eq!(result.entries(), [(vec![1].into_boxed_slice(), 1)]);
    }
}
