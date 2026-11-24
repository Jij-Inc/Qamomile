//! Canonical sampled-result plans shared by every static backend.
//!
//! Indexed reconstruction borrows each immutable scalar plan and checks its
//! aggregate structural work before evaluating invocation-local result rows.

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
const PLAN_SCHEMA_MINOR: u32 = 3;
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
    if circuit.has_nested_checks() || circuit.has_ordered_binary_checks() {
        return Err(ResultPostprocessingError::InvalidPlan(
            "classical checks require ordered in-job execution".to_owned(),
        ));
    }
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
    if circuit.has_nested_checks() || circuit.has_ordered_binary_checks() {
        return Err(ResultPostprocessingError::InvalidPlan(
            "classical checks require ordered in-job execution".to_owned(),
        ));
    }
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
    let indexed = scalars.iter().any(|node| {
        matches!(
            node,
            CircuitScalarNode::CurrentShot | CircuitScalarNode::Indexed { .. }
        )
    });
    let scalars = scalars
        .iter()
        .map(encode_scalar)
        .collect::<Result<Vec<_>, _>>()?;
    let result = encode_result(&result)?;
    Ok(Some(
        wire::CircuitResultPostprocessingProto {
            schema_major: PLAN_SCHEMA_MAJOR,
            schema_minor: u32::from(circuit.has_terminal_checks()) + if indexed { 2 } else { 0 },
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

/// Evaluate indexed one-shot rows without losing invocation-local coordinates.
///
/// # Errors
/// Rejects excessive aggregate reconstruction work and missing, repeated,
/// reordered, out-of-range, or non-single-shot rows.
pub fn postprocess_indexed_counts(
    payload: &[u8],
    rows: &[(u64, Box<[u8]>, u64)],
    bindings: &[CircuitLaunchBinding],
    shots: u64,
    result_abi: &[u8],
) -> Result<StaticSampleCounts, ResultPostprocessingError> {
    postprocess_indexed_measurements(
        payload,
        IndexedMeasurements::Rows(rows),
        bindings,
        shots,
        result_abi,
    )
}

/// Evaluate measurement-free coordinates using only their invocation cardinality.
///
/// Every logical index still executes its retained result and runtime checks.
/// No per-index input rows are allocated; only distinct logical outputs persist.
///
/// # Errors
/// Rejects plans requiring measurements, invalid cardinality, excessive native
/// work, or a retained expression that fails at any logical index.
pub fn postprocess_unmeasured_indexed_counts(
    payload: &[u8],
    bindings: &[CircuitLaunchBinding],
    shots: u64,
    result_abi: &[u8],
) -> Result<StaticSampleCounts, ResultPostprocessingError> {
    postprocess_indexed_measurements(
        payload,
        IndexedMeasurements::Unmeasured,
        bindings,
        shots,
        result_abi,
    )
}

/// Transport either observed ordered rows or certified measurement-free coverage.
#[derive(Clone, Copy)]
enum IndexedMeasurements<'a> {
    Rows(&'a [(u64, Box<[u8]>, u64)]),
    Unmeasured,
}

fn postprocess_indexed_measurements(
    payload: &[u8],
    measurements: IndexedMeasurements<'_>,
    bindings: &[CircuitLaunchBinding],
    shots: u64,
    result_abi: &[u8],
) -> Result<StaticSampleCounts, ResultPostprocessingError> {
    if shots == 0
        || shots > crate::MAX_INDEXED_SHOTS
        || matches!(&measurements, IndexedMeasurements::Rows(rows) if rows.len() as u64 != shots)
    {
        return Err(ResultPostprocessingError::InvalidCounts(
            "indexed coverage differs from invocation".into(),
        ));
    }
    let decoded = decode_plan(payload)?;
    if matches!(measurements, IndexedMeasurements::Unmeasured)
        && decoded.circuit.measurement_count() != 0
    {
        return Err(ResultPostprocessingError::InvalidCounts(
            "measurement-free coverage disagrees with the plan".into(),
        ));
    }
    let result_abi = checked_result_abi(result_abi)
        .map_err(|error| ResultPostprocessingError::Evaluation(error.to_string()))?;
    if decoded
        .per_row_work(result_abi.logical_width())
        .saturating_mul(u128::from(shots))
        > crate::core::shot_context::MAX_INDEXED_WORK
    {
        return Err(ResultPostprocessingError::Evaluation(
            "indexed result exceeds its native scalar work budget".into(),
        ));
    }
    let mut logical = BTreeMap::<Box<[u8]>, u64>::new();
    for (position, expected) in (0..shots).enumerate() {
        let bits = match &measurements {
            IndexedMeasurements::Unmeasured => &[][..],
            IndexedMeasurements::Rows(rows) => {
                let (index, bits, count) = &rows[position];
                if *index != expected || *count != 1 {
                    return Err(ResultPostprocessingError::InvalidCounts(
                        "indexed rows must be ordered, complete, and single-shot".into(),
                    ));
                }
                bits.as_ref()
            }
        };
        let index = crate::LogicalShotIndex::new(expected, shots)
            .map_err(|error| ResultPostprocessingError::InvalidCounts(error.to_string()))?;
        let mut literal = crate::simulation::evaluate_indexed_retained_result(
            &decoded.circuit,
            bindings,
            bits,
            index,
        )
        .map_err(|error| ResultPostprocessingError::Evaluation(error.to_string()))?;
        if decoded.checked_result {
            let Some(wire::literal_proto::Value::TupleValue(mut tuple)) = literal.value else {
                return Err(ResultPostprocessingError::InvalidPlan(
                    "checked result tuple".into(),
                ));
            };
            literal = tuple.values.pop().ok_or_else(|| {
                ResultPostprocessingError::InvalidPlan("checked result arity".into())
            })?;
        }
        let bits = result_abi
            .encode_literal(&literal)
            .map_err(|error| ResultPostprocessingError::Evaluation(error.to_string()))?
            .into_boxed_slice();
        *logical.entry(bits).or_default() += 1;
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
        CircuitScalarNode::CurrentShot => Node::CurrentShot(true),
        CircuitScalarNode::Indexed {
            index,
            elements,
            scalar_type,
        } => Node::Indexed(wire::CircuitResultIndexedProto {
            index: index.index(),
            elements: elements.iter().map(|element| element.index()).collect(),
            r#type: encode_scalar_type(*scalar_type).into(),
        }),
        CircuitScalarNode::HiddenBinding(_) => {
            return Err(ResultPostprocessingError::InvalidPlan(
                "hidden adapter binding escaped numeric emission".into(),
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

pub(crate) fn encode_result(
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

pub(crate) struct DecodedResultPlan {
    pub(crate) circuit: StaticCircuit,
    checked_result: bool,
    per_row_work: u128,
}

impl DecodedResultPlan {
    /// Return structural, measurement-row, and encoded-result reconstruction work.
    pub(crate) const fn per_row_work(&self, logical_width: usize) -> u128 {
        self.per_row_work.saturating_add(logical_width as u128)
    }
}

pub(crate) fn decode_plan(payload: &[u8]) -> Result<DecodedResultPlan, ResultPostprocessingError> {
    if payload.len() > MAX_PLAN_BYTES {
        return Err(ResultPostprocessingError::InvalidPlan(
            "plan exceeds its byte budget".to_owned(),
        ));
    }
    let structural_work = budget::preflight_result_postprocessing_with_work(payload)
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
    if plan.schema_minor % 2 == 1 {
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
        per_row_work: structural_work.saturating_add(u128::from(plan.measurement_count)),
        circuit: StaticCircuit::from_result_plan(plan.measurement_count, scalars, result),
        checked_result: plan.schema_minor % 2 == 1,
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
        Some(Node::CurrentShot(true)) => Ok(CircuitScalarNode::CurrentShot),
        Some(Node::CurrentShot(false)) => Err(ResultPostprocessingError::InvalidPlan(
            "invalid current-shot marker".into(),
        )),
        Some(Node::Indexed(indexed)) => {
            let checked = |reference: u32| {
                if (reference as usize) < index {
                    Ok(CircuitScalarId::from_index(reference))
                } else {
                    Err(ResultPostprocessingError::InvalidPlan(
                        "non-topological indexed scalar reference".into(),
                    ))
                }
            };
            Ok(CircuitScalarNode::Indexed {
                index: checked(indexed.index)?,
                elements: indexed
                    .elements
                    .iter()
                    .map(|element| checked(*element))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_boxed_slice(),
                scalar_type: decode_scalar_type(indexed.r#type)?,
            })
        }
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

    fn result_plan(scalars: Vec<CircuitScalarNode>, measurement_count: u64) -> Vec<u8> {
        let root = CircuitScalarId::test_new(u32::try_from(scalars.len() - 1).unwrap());
        let circuit = StaticCircuit::from_result_plan(
            measurement_count,
            scalars.into_boxed_slice(),
            StaticCircuitValue::Scalar(root),
        );
        encode_result_postprocessing(&circuit).unwrap().unwrap()
    }

    #[test]
    fn unmeasured_reconstruction_checks_coverage_and_plan_width() {
        let payload = result_plan(vec![CircuitScalarNode::Bool(true)], 0);
        for shots in [0, crate::MAX_INDEXED_SHOTS + 1] {
            assert!(matches!(
                postprocess_unmeasured_indexed_counts(&payload, &[], shots, &bool_result_abi()),
                Err(ResultPostprocessingError::InvalidCounts(_))
            ));
        }
        let measured = result_plan(vec![CircuitScalarNode::Bool(true)], 1);
        assert!(matches!(
            postprocess_unmeasured_indexed_counts(&measured, &[], 2, &bool_result_abi()),
            Err(ResultPostprocessingError::InvalidCounts(reason)) if reason.contains("measurement-free")
        ));
        let counts = postprocess_unmeasured_indexed_counts(
            &payload,
            &[],
            crate::MAX_INDEXED_SHOTS,
            &bool_result_abi(),
        )
        .expect("cardinality-only completion at the supported shot limit");
        assert_eq!(
            counts.entries(),
            &[(vec![1].into_boxed_slice(), crate::MAX_INDEXED_SHOTS)]
        );
    }

    #[test]
    fn indexed_reconstruction_checks_aggregate_work_before_evaluating_rows() {
        let mut scalars = vec![CircuitScalarNode::Parameter {
            name: "missing".into(),
            path: Box::new([]),
            scalar_type: CircuitScalarType::Bool,
        }];
        for index in 0..4096 {
            scalars.push(CircuitScalarNode::Apply {
                opcode: "classical.not".into(),
                result_type: CircuitScalarType::Bool,
                attributes: Box::new([]),
                operands: vec![CircuitScalarId::test_new(index)].into_boxed_slice(),
            });
        }
        let payload = result_plan(scalars, 0);
        assert!(matches!(
            postprocess_unmeasured_indexed_counts(&payload, &[], 4096, &bool_result_abi()),
            Err(ResultPostprocessingError::Evaluation(reason)) if reason.contains("work budget")
        ));
        let rows = (0..4096)
            .map(|index| (index, Box::<[u8]>::default(), 1))
            .collect::<Vec<_>>();
        assert!(matches!(
            postprocess_indexed_counts(
                &payload,
                &rows,
                &[],
                4096,
                &bool_result_abi(),
            ),
            Err(ResultPostprocessingError::Evaluation(reason)) if reason.contains("work budget")
        ));
        assert!(matches!(
            postprocess_indexed_counts(&payload, &rows[..1], &[], 1, &bool_result_abi()),
            Err(ResultPostprocessingError::Evaluation(reason)) if reason.contains("missing")
        ));
    }

    #[test]
    fn indexed_reconstruction_accounts_for_full_measurement_rows() {
        let payload = result_plan(vec![CircuitScalarNode::Bool(true)], 32_768);
        let rows = (0..4096)
            .map(|index| (index, Box::<[u8]>::default(), 1))
            .collect::<Vec<_>>();
        assert!(matches!(
            postprocess_indexed_counts(
                &payload,
                &rows,
                &[],
                4096,
                &bool_result_abi(),
            ),
            Err(ResultPostprocessingError::Evaluation(reason)) if reason.contains("work budget")
        ));
    }

    #[test]
    fn indexed_reconstruction_accounts_for_encoded_result_width() {
        let width = 65_536;
        let result_abi = wire::ResultAbiProto {
            per_shot: false,
            r#type: Some(wire::TypeProto {
                kind: wire::type_proto::Kind::Array.into(),
                name: "UInt".into(),
                arguments: vec![wire::TypeProto {
                    kind: wire::type_proto::Kind::Scalar.into(),
                    name: "Bit".into(),
                    ..Default::default()
                }],
                dimensions: vec![wire::DimensionProto {
                    value: Some(wire::dimension_proto::Value::Fixed(u64::from(width))),
                }],
                ..Default::default()
            }),
            source_order: (0..width).collect(),
            native_width: width,
        }
        .encode_to_vec();
        let payload = result_plan(vec![CircuitScalarNode::Int(0)], 0);
        let rows = (0..4096)
            .map(|index| (index, Box::<[u8]>::default(), 1))
            .collect::<Vec<_>>();
        assert!(matches!(
            postprocess_indexed_counts(&payload, &rows, &[], 4096, &result_abi),
            Err(ResultPostprocessingError::Evaluation(reason)) if reason.contains("work budget")
        ));
        let counts = postprocess_indexed_counts(&payload, &rows[..1], &[], 1, &result_abi)
            .expect("the individual wide word is a valid sampled result");
        assert_eq!(counts.entries().len(), 1);
        assert_eq!(counts.entries()[0].0.len(), width as usize);
        assert!(counts.entries()[0].0.iter().all(|bit| *bit == 0));
    }

    #[test]
    fn indexed_results_borrow_context_and_keep_select_and_index_branches_lazy() {
        let payload = result_plan(
            vec![
                CircuitScalarNode::CurrentShot,
                CircuitScalarNode::Bool(true),
                CircuitScalarNode::Bool(false),
                CircuitScalarNode::Int(1),
                CircuitScalarNode::Int(0),
                CircuitScalarNode::Apply {
                    opcode: "classical.floordiv".into(),
                    result_type: CircuitScalarType::Int,
                    attributes: Box::new([]),
                    operands: vec![CircuitScalarId::test_new(3), CircuitScalarId::test_new(4)]
                        .into_boxed_slice(),
                },
                CircuitScalarNode::Apply {
                    opcode: "classical.not".into(),
                    result_type: CircuitScalarType::Bool,
                    attributes: Box::new([]),
                    operands: vec![CircuitScalarId::test_new(5)].into_boxed_slice(),
                },
                CircuitScalarNode::Indexed {
                    index: CircuitScalarId::test_new(0),
                    elements: vec![
                        CircuitScalarId::test_new(1),
                        CircuitScalarId::test_new(2),
                        CircuitScalarId::test_new(6),
                    ]
                    .into_boxed_slice(),
                    scalar_type: CircuitScalarType::Bool,
                },
                CircuitScalarNode::Apply {
                    opcode: "classical.select".into(),
                    result_type: CircuitScalarType::Bool,
                    attributes: Box::new([]),
                    operands: vec![
                        CircuitScalarId::test_new(1),
                        CircuitScalarId::test_new(7),
                        CircuitScalarId::test_new(6),
                    ]
                    .into_boxed_slice(),
                },
            ],
            0,
        );
        let rows = vec![
            (0, Box::<[u8]>::default(), 1),
            (1, Box::<[u8]>::default(), 1),
        ];
        let counts = postprocess_indexed_counts(&payload, &rows, &[], 2, &bool_result_abi())
            .expect("unselected division by zero must remain lazy");
        assert_eq!(
            counts.entries(),
            &[
                (vec![0].into_boxed_slice(), 1),
                (vec![1].into_boxed_slice(), 1)
            ]
        );
        let counts = postprocess_indexed_counts(&payload, &rows[..1], &[], 1, &bool_result_abi())
            .expect("each invocation starts again at zero");
        assert_eq!(counts.entries(), &[(vec![1].into_boxed_slice(), 1)]);
        let compact = postprocess_unmeasured_indexed_counts(&payload, &[], 2, &bool_result_abi())
            .expect("compact coverage retains lazy indexed evaluation");
        assert_eq!(
            compact.entries(),
            &[
                (vec![0].into_boxed_slice(), 1),
                (vec![1].into_boxed_slice(), 1),
            ]
        );
        assert!(matches!(
            postprocess_unmeasured_indexed_counts(&payload, &[], 3, &bool_result_abi()),
            Err(ResultPostprocessingError::Evaluation(reason)) if reason.contains("zero")
        ));
    }
    fn bool_result_abi() -> Vec<u8> {
        wire::ResultAbiProto {
            per_shot: false,
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
