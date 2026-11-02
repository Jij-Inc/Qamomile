//! Bounded deterministic value evaluation for verified `QCore` programs.
//!
//! This evaluator owns the pure-classical Reference subset. It reuses the
//! compiler's exact portable-value algebra and executes only already-verified
//! calls and structured regions. A typed quantum-effect gap transfers control
//! to the static-circuit machine at the owning `VerifiedProgram` boundary.

use std::{collections::HashMap, sync::Arc};

use num_bigint::BigInt;
use prost::Message;
use thiserror::Error;

use crate::circuit::{CircuitLaunchBinding, CircuitLaunchValue};
use crate::literals::validate_literal_references;
use crate::numeric_values::NumericEvaluationError;
use crate::static_values::{self, StaticScalar, StaticValue};
use crate::{budget, wire};

const MAX_REFERENCE_VALUE_WORK: usize = 10_000_000;
const MAX_REFERENCE_LOOP_ITERATIONS: u128 = 100_000;
const MAX_REFERENCE_CALL_DEPTH: usize = 256;

/// Failure to evaluate one deterministic verified Reference invocation.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ReferenceValueError {
    /// The verified program is outside the migrated pure-classical subset.
    #[error("Q4501: unsupported native Reference value operation: {0}")]
    Unsupported(String),
    /// A selected exact operation is undefined for its runtime operands.
    #[error("Q4502: invalid native Reference value evaluation: {0}")]
    Invalid(String),
    /// Bounded execution exhausted its deterministic work allowance.
    #[error("Q4503: native Reference value execution exceeded its work budget")]
    WorkBudgetExceeded,
}

/// Evaluate one verified pure-classical operation over portable literals.
///
/// This is the shared late-evaluation boundary for targets and the Python
/// reference runtime. The operation has already passed `QCore` type verification;
/// this function owns only deterministic runtime value semantics.
///
/// # Errors
///
/// Returns [`ReferenceValueError`] when an input or attribute is not a closed
/// portable literal, the operation is outside the native classical subset, or
/// the selected operation is undefined for its operands.
pub fn evaluate_classical_literals(
    opcode: &str,
    operands: &[Vec<u8>],
    attributes: &[(String, Vec<u8>)],
) -> Result<Vec<u8>, ReferenceValueError> {
    if !opcode.starts_with("classical.") {
        return Err(ReferenceValueError::Unsupported(opcode.to_owned()));
    }
    let mut environment = HashMap::new();
    for (position, payload) in operands.iter().enumerate() {
        let literal = decode_standalone_literal(payload)?;
        let value = static_values::value_literal(&literal).ok_or_else(|| {
            ReferenceValueError::Invalid("operand is not a closed portable value".to_owned())
        })?;
        let identifier = u64::try_from(position).map_err(|_| {
            ReferenceValueError::Invalid("classical operand count is not representable".to_owned())
        })?;
        environment.insert(identifier, Some(value));
    }
    if attributes
        .windows(2)
        .any(|pair| pair[0].0.as_str() >= pair[1].0.as_str())
    {
        return Err(ReferenceValueError::Invalid(
            "classical attributes must be unique and ordered".to_owned(),
        ));
    }
    let attributes = attributes
        .iter()
        .map(|(name, payload)| {
            Ok(wire::AttributeProto {
                name: name.clone(),
                value: Some(decode_standalone_literal(payload)?),
            })
        })
        .collect::<Result<Vec<_>, ReferenceValueError>>()?;
    let operand_count = u64::try_from(operands.len()).map_err(|_| {
        ReferenceValueError::Invalid("classical operand count is not representable".to_owned())
    })?;
    let operation = wire::OperationProto {
        opcode: opcode.to_owned(),
        operands: (0..operand_count).collect(),
        attributes,
        ..Default::default()
    };
    let value = match static_values::evaluate_value_checked(&operation, &environment) {
        Ok(Some(value)) if value.is_complete() => value,
        Ok(_) => return Err(ReferenceValueError::Unsupported(opcode.to_owned())),
        Err(NumericEvaluationError::Undefined(reason) | NumericEvaluationError::Limit(reason)) => {
            return Err(ReferenceValueError::Invalid(reason.to_owned()));
        }
    };
    static_values::value_to_literal(&value)
        .map(|literal| literal.encode_to_vec())
        .ok_or_else(|| ReferenceValueError::Invalid("result is not portable".to_owned()))
}

fn decode_standalone_literal(payload: &[u8]) -> Result<wire::LiteralProto, ReferenceValueError> {
    budget::preflight_literal(payload).map_err(|_| {
        ReferenceValueError::Invalid("literal exceeds structural budget".to_owned())
    })?;
    let literal = wire::LiteralProto::decode(payload)
        .map_err(|_| ReferenceValueError::Invalid("malformed portable literal".to_owned()))?;
    validate_literal_references(vec![&literal])
        .map_err(|_| ReferenceValueError::Invalid("invalid portable literal".to_owned()))?;
    Ok(literal)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Flow {
    kind: FlowKind,
    values: Vec<StaticValue>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FlowKind {
    Return,
    Yield,
    Continue,
    Break,
}

struct Evaluator<'a> {
    functions: HashMap<&'a str, &'a wire::FunctionProto>,
    work: usize,
}

/// Evaluate one complete pure-classical invocation into a canonical literal.
///
/// The returned bytes are one deterministic `LiteralProto`, not a second
/// runtime-value schema. Binding names, shapes, and scalar kinds are checked
/// against the verified entrypoint ABI before execution.
pub(crate) fn evaluate(
    package: &wire::QCorePackageProto,
    bindings: &[CircuitLaunchBinding],
) -> Result<Vec<u8>, ReferenceValueError> {
    let entry = package
        .functions
        .iter()
        .find(|function| function.symbol == package.entrypoint)
        .ok_or_else(|| ReferenceValueError::Invalid("missing entrypoint".to_owned()))?;
    let supplied = bindings
        .iter()
        .map(|binding| (binding.name(), binding.value()))
        .collect::<HashMap<_, _>>();
    if supplied.len() != bindings.len() {
        return Err(ReferenceValueError::Invalid(
            "duplicate invocation binding".to_owned(),
        ));
    }
    let constants = package
        .const_bindings
        .iter()
        .map(|binding| (binding.name.as_str(), binding.value.as_ref()))
        .collect::<HashMap<_, _>>();
    let expected = entry
        .parameters
        .iter()
        .filter(|parameter| !constants.contains_key(parameter.name.as_str()))
        .map(|parameter| parameter.name.as_str())
        .collect::<std::collections::HashSet<_>>();
    if supplied
        .keys()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        != expected
    {
        return Err(ReferenceValueError::Invalid(
            "invocation bindings disagree with the entrypoint ABI".to_owned(),
        ));
    }
    let arguments = entry
        .parameters
        .iter()
        .map(|parameter| {
            let type_spec = parameter
                .value
                .as_ref()
                .and_then(|value| value.r#type.as_ref())
                .ok_or_else(|| {
                    ReferenceValueError::Invalid(format!(
                        "missing parameter type for {:?}",
                        parameter.name
                    ))
                })?;
            if let Some(value) = constants.get(parameter.name.as_str()).copied().flatten() {
                return static_values::typed_value_literal(value, type_spec).ok_or_else(|| {
                    ReferenceValueError::Invalid(format!(
                        "invalid constant binding for {:?}",
                        parameter.name
                    ))
                });
            }
            supplied
                .get(parameter.name.as_str())
                .copied()
                .map(|value| launch_value(value, type_spec))
                .transpose()?
                .ok_or_else(|| {
                    ReferenceValueError::Invalid(format!(
                        "missing invocation binding for {:?}",
                        parameter.name
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let functions = package
        .functions
        .iter()
        .map(|function| (function.symbol.as_str(), function))
        .collect();
    let mut evaluator = Evaluator { functions, work: 0 };
    let value = evaluator.evaluate_function(entry, arguments, 0)?;
    let literal = static_values::value_to_literal(&value).ok_or_else(|| {
        ReferenceValueError::Invalid("result is not a complete portable value".to_owned())
    })?;
    Ok(literal.encode_to_vec())
}

fn launch_value(
    value: &CircuitLaunchValue,
    type_spec: &wire::TypeProto,
) -> Result<StaticValue, ReferenceValueError> {
    use wire::type_proto::Kind;

    match type_spec.kind() {
        Kind::Staged => launch_value(
            value,
            type_spec
                .arguments
                .first()
                .ok_or_else(|| ReferenceValueError::Invalid("staged invocation type".to_owned()))?,
        ),
        Kind::Scalar => launch_scalar(value, type_spec),
        Kind::Product => launch_product(value, type_spec),
        Kind::Record => launch_record(value, type_spec),
        Kind::Array => launch_array(value, type_spec),
        Kind::Fixed => launch_fixed(value),
        Kind::Quantum | Kind::Function | Kind::Unspecified => Err(
            ReferenceValueError::Unsupported(format!("invocation value type {:?}", type_spec.name)),
        ),
    }
}

fn launch_scalar(
    value: &CircuitLaunchValue,
    type_spec: &wire::TypeProto,
) -> Result<StaticValue, ReferenceValueError> {
    match value {
        CircuitLaunchValue::Bool(value) if matches!(type_spec.name.as_str(), "Bool" | "Bit") => {
            Ok(StaticValue::Scalar(StaticScalar::Bool(*value)))
        }
        CircuitLaunchValue::Int(value) if type_spec.name == "Bit" && matches!(value, 0 | 1) => {
            Ok(StaticValue::Scalar(StaticScalar::Int(*value)))
        }
        CircuitLaunchValue::Int(value)
            if matches!(type_spec.name.as_str(), "Int" | "Real" | "Probability") =>
        {
            Ok(StaticValue::Scalar(StaticScalar::Int(*value)))
        }
        CircuitLaunchValue::BigInt(value) if type_spec.name == "Int" => {
            Ok(StaticValue::BigInt(Arc::from(value.as_ref())))
        }
        CircuitLaunchValue::RealBits(value)
            if matches!(type_spec.name.as_str(), "Real" | "Probability") =>
        {
            Ok(StaticValue::Scalar(StaticScalar::RealBits(*value)))
        }
        CircuitLaunchValue::ComplexBits(real, imaginary) if type_spec.name == "Complex" => {
            Ok(StaticValue::ComplexBits(*real, *imaginary))
        }
        _ if matches!(type_spec.name.as_str(), "Unit" | "Observable") => Err(
            ReferenceValueError::Unsupported(format!("invocation value type {:?}", type_spec.name)),
        ),
        _ => Err(ReferenceValueError::Invalid(format!(
            "invocation value for {:?}",
            type_spec.name
        ))),
    }
}

fn launch_product(
    value: &CircuitLaunchValue,
    type_spec: &wire::TypeProto,
) -> Result<StaticValue, ReferenceValueError> {
    let CircuitLaunchValue::Aggregate(values) = value else {
        return Err(ReferenceValueError::Invalid(
            "product invocation value".to_owned(),
        ));
    };
    if values.len() != type_spec.arguments.len() {
        return Err(ReferenceValueError::Invalid(
            "product invocation arity".to_owned(),
        ));
    }
    Ok(StaticValue::Aggregate(
        values
            .iter()
            .zip(&type_spec.arguments)
            .map(|(value, type_spec)| launch_value(value, type_spec).map(Some))
            .collect::<Result<Vec<_>, _>>()?
            .into(),
    ))
}

fn launch_record(
    value: &CircuitLaunchValue,
    type_spec: &wire::TypeProto,
) -> Result<StaticValue, ReferenceValueError> {
    let CircuitLaunchValue::Record(fields) = value else {
        return Err(ReferenceValueError::Invalid(
            "record invocation value".to_owned(),
        ));
    };
    let fields = fields
        .iter()
        .map(|(name, value)| (name.as_str(), value))
        .collect::<HashMap<_, _>>();
    if fields.len() != type_spec.field_names.len()
        || type_spec.field_names.len() != type_spec.arguments.len()
    {
        return Err(ReferenceValueError::Invalid(
            "record invocation arity".to_owned(),
        ));
    }
    Ok(StaticValue::Record(
        type_spec
            .field_names
            .iter()
            .zip(&type_spec.arguments)
            .map(|(name, type_spec)| {
                let value = fields.get(name.as_str()).ok_or_else(|| {
                    ReferenceValueError::Invalid("record invocation field".to_owned())
                })?;
                Ok((name.as_str().into(), Some(launch_value(value, type_spec)?)))
            })
            .collect::<Result<Vec<_>, ReferenceValueError>>()?
            .into(),
    ))
}

fn launch_fixed(value: &CircuitLaunchValue) -> Result<StaticValue, ReferenceValueError> {
    match value {
        CircuitLaunchValue::Int(value) => Ok(StaticValue::Scalar(StaticScalar::Int(*value))),
        CircuitLaunchValue::BigInt(value) => Ok(StaticValue::BigInt(Arc::from(value.as_ref()))),
        CircuitLaunchValue::RealBits(value) => {
            Ok(StaticValue::Scalar(StaticScalar::RealBits(*value)))
        }
        CircuitLaunchValue::ComplexBits(_, _)
        | CircuitLaunchValue::Bool(_)
        | CircuitLaunchValue::Aggregate(_)
        | CircuitLaunchValue::Record(_) => Err(ReferenceValueError::Invalid(
            "fixed-point invocation value".to_owned(),
        )),
    }
}

fn launch_array(
    value: &CircuitLaunchValue,
    type_spec: &wire::TypeProto,
) -> Result<StaticValue, ReferenceValueError> {
    fn descend(
        value: &CircuitLaunchValue,
        element_type: &wire::TypeProto,
        dimensions: &[wire::DimensionProto],
    ) -> Result<StaticValue, ReferenceValueError> {
        use wire::dimension_proto::Value;

        let CircuitLaunchValue::Aggregate(values) = value else {
            return Err(ReferenceValueError::Invalid(
                "array invocation value".to_owned(),
            ));
        };
        let (dimension, remaining) = dimensions
            .split_first()
            .ok_or_else(|| ReferenceValueError::Invalid("array invocation rank".to_owned()))?;
        let extent = match dimension.value.as_ref() {
            Some(Value::Fixed(extent)) => usize::try_from(*extent).ok(),
            Some(Value::Symbol(_)) | None => None,
        }
        .ok_or_else(|| {
            ReferenceValueError::Unsupported("symbolic invocation array extent".to_owned())
        })?;
        if values.len() != extent {
            return Err(ReferenceValueError::Invalid(
                "array invocation extent".to_owned(),
            ));
        }
        Ok(StaticValue::Aggregate(
            values
                .iter()
                .map(|value| {
                    let value = if remaining.is_empty() {
                        launch_value(value, element_type)?
                    } else {
                        descend(value, element_type, remaining)?
                    };
                    Ok(Some(value))
                })
                .collect::<Result<Vec<_>, ReferenceValueError>>()?
                .into(),
        ))
    }

    if type_spec.name == "UInt" {
        return launch_uint(value, type_spec);
    }
    let element_type = type_spec
        .arguments
        .first()
        .ok_or_else(|| ReferenceValueError::Invalid("array element type".to_owned()))?;
    descend(value, element_type, &type_spec.dimensions)
}

fn launch_uint(
    value: &CircuitLaunchValue,
    type_spec: &wire::TypeProto,
) -> Result<StaticValue, ReferenceValueError> {
    use wire::dimension_proto::Value;

    let [dimension] = type_spec.dimensions.as_slice() else {
        return Err(ReferenceValueError::Invalid(
            "unsigned invocation rank".to_owned(),
        ));
    };
    let Some(Value::Fixed(width)) = dimension.value.as_ref() else {
        return Err(ReferenceValueError::Unsupported(
            "symbolic unsigned invocation width".to_owned(),
        ));
    };
    let integer = match value {
        CircuitLaunchValue::Int(value) => BigInt::from(*value),
        CircuitLaunchValue::BigInt(value) => BigInt::from_signed_bytes_be(value),
        _ => {
            return Err(ReferenceValueError::Invalid(
                "unsigned invocation value".to_owned(),
            ));
        }
    };
    if integer < BigInt::from(0_u8) || integer.bits() > *width {
        return Err(ReferenceValueError::Invalid(
            "unsigned invocation range".to_owned(),
        ));
    }
    Ok(StaticValue::Aggregate(
        (0..*width)
            .map(|index| {
                let shift = *width - index - 1;
                Some(StaticValue::Scalar(StaticScalar::Bool(
                    ((&integer >> shift) & BigInt::from(1_u8)) == BigInt::from(1_u8),
                )))
            })
            .collect::<Vec<_>>()
            .into(),
    ))
}

impl Evaluator<'_> {
    fn consume_work(&mut self) -> Result<(), ReferenceValueError> {
        self.work = self
            .work
            .checked_add(1)
            .ok_or(ReferenceValueError::WorkBudgetExceeded)?;
        if self.work > MAX_REFERENCE_VALUE_WORK {
            return Err(ReferenceValueError::WorkBudgetExceeded);
        }
        Ok(())
    }

    fn evaluate_function(
        &mut self,
        function: &wire::FunctionProto,
        arguments: Vec<StaticValue>,
        depth: usize,
    ) -> Result<StaticValue, ReferenceValueError> {
        if depth >= MAX_REFERENCE_CALL_DEPTH {
            return Err(ReferenceValueError::WorkBudgetExceeded);
        }
        if function.external {
            return Err(ReferenceValueError::Unsupported(format!(
                "external call {:?}",
                function.symbol
            )));
        }
        if arguments.len() != function.parameters.len() {
            return Err(ReferenceValueError::Invalid(format!(
                "call ABI for {:?}",
                function.symbol
            )));
        }
        let body = function.body.as_ref().ok_or_else(|| {
            ReferenceValueError::Invalid(format!("missing body for {:?}", function.symbol))
        })?;
        let mut environment = function
            .parameters
            .iter()
            .zip(arguments)
            .map(|(parameter, value)| {
                parameter
                    .value
                    .as_ref()
                    .map(|definition| (definition.id, Some(value)))
                    .ok_or_else(|| {
                        ReferenceValueError::Invalid(format!(
                            "missing parameter value for {:?}",
                            function.symbol
                        ))
                    })
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        let flow = self.evaluate_block(body, &mut environment, &[], depth)?;
        match flow.values.as_slice() {
            [value] if flow.kind == FlowKind::Return => Ok(value.clone()),
            _ => Err(ReferenceValueError::Invalid(format!(
                "return ABI for {:?}",
                function.symbol
            ))),
        }
    }

    fn evaluate_block(
        &mut self,
        block: &wire::BlockProto,
        outer: &mut HashMap<u64, Option<StaticValue>>,
        arguments: &[StaticValue],
        depth: usize,
    ) -> Result<Flow, ReferenceValueError> {
        if arguments.len() != block.arguments.len() {
            return Err(ReferenceValueError::Invalid(
                "structured region argument ABI".to_owned(),
            ));
        }
        let mut environment = outer.clone();
        for (definition, value) in block.arguments.iter().zip(arguments) {
            environment.insert(definition.id, Some(value.clone()));
        }
        for operation in &block.operations {
            self.consume_work()?;
            let values = match operation.opcode.as_str() {
                "qcore.constant" => vec![Self::constant(operation)?],
                "qcore.stage_lift" | "qcore.reshape" => {
                    vec![operand(&environment, operation, 0)?]
                }
                "tuple" | "array" | "qcore.tuple" => vec![static_values::structural_value(
                    operation.results.first(),
                    operation
                        .operands
                        .iter()
                        .map(|identifier| environment.get(identifier).cloned().flatten())
                        .collect(),
                )],
                "qcore.unpack" => Self::unpack(operation, &environment)?,
                "qcore.call" => vec![self.evaluate_call(operation, &environment, depth)?],
                "qcore.if" => {
                    let flow = self.evaluate_if(operation, &environment, depth)?;
                    if flow.kind == FlowKind::Yield {
                        flow.values
                    } else {
                        return Ok(flow);
                    }
                }
                "qcore.for" | "qcore.while" => {
                    let flow = if operation.opcode == "qcore.while" {
                        self.evaluate_while(operation, &environment, depth)?
                    } else {
                        self.evaluate_for(operation, &environment, depth)?
                    };
                    if flow.kind == FlowKind::Yield {
                        flow.values
                    } else {
                        return Ok(flow);
                    }
                }
                opcode if opcode.starts_with("classical.") => {
                    vec![Self::evaluate_classical(operation, &environment)?]
                }
                opcode => {
                    return Err(ReferenceValueError::Unsupported(opcode.to_owned()));
                }
            };
            if values.len() != operation.results.len() {
                return Err(ReferenceValueError::Invalid(format!(
                    "result ABI for {:?}",
                    operation.opcode
                )));
            }
            environment.extend(
                operation
                    .results
                    .iter()
                    .zip(values)
                    .map(|(result, value)| (result.id, Some(value))),
            );
        }
        let terminator = block
            .terminator
            .as_ref()
            .ok_or_else(|| ReferenceValueError::Invalid("missing block terminator".to_owned()))?;
        if terminator.kind == "unreachable" {
            return Err(ReferenceValueError::Invalid(
                "reached unreachable terminator".to_owned(),
            ));
        }
        let values = terminator
            .operands
            .iter()
            .map(|identifier| exact_value(&environment, *identifier))
            .collect::<Result<_, _>>()?;
        Ok(Flow {
            kind: flow_kind(&terminator.kind)?,
            values,
        })
    }

    fn constant(operation: &wire::OperationProto) -> Result<StaticValue, ReferenceValueError> {
        let literal = attribute(operation, "value")
            .ok_or_else(|| ReferenceValueError::Invalid("constant literal".to_owned()))?;
        let type_spec = operation
            .results
            .first()
            .and_then(|result| result.r#type.as_ref())
            .ok_or_else(|| ReferenceValueError::Invalid("constant result type".to_owned()))?;
        static_values::typed_value_literal(literal, type_spec)
            .ok_or_else(|| ReferenceValueError::Invalid("constant value/type relation".to_owned()))
    }

    fn unpack(
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<StaticValue>>,
    ) -> Result<Vec<StaticValue>, ReferenceValueError> {
        let source = operand(environment, operation, 0)?;
        static_values::structural_components(&source)
            .filter(|values| values.len() == operation.results.len())
            .ok_or_else(|| ReferenceValueError::Invalid("unpack value ABI".to_owned()))?
            .into_iter()
            .map(|value| {
                value.ok_or_else(|| {
                    ReferenceValueError::Invalid("incomplete unpack value".to_owned())
                })
            })
            .collect()
    }

    fn evaluate_call(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<StaticValue>>,
        depth: usize,
    ) -> Result<StaticValue, ReferenceValueError> {
        let callee = string_attribute(operation, "callee")
            .and_then(|symbol| self.functions.get(symbol).copied())
            .ok_or_else(|| ReferenceValueError::Invalid("unresolved callable".to_owned()))?;
        let arguments = operation_values(operation, environment)?;
        self.evaluate_function(callee, arguments, depth + 1)
    }

    fn evaluate_if(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<StaticValue>>,
        depth: usize,
    ) -> Result<Flow, ReferenceValueError> {
        if string_attribute(operation, "body_mode") == Some("adjoint") {
            return Err(ReferenceValueError::Unsupported(
                "adjoint structured if".to_owned(),
            ));
        }
        let condition = truth(&operand(environment, operation, 0)?)?;
        let region = operation
            .regions
            .get(usize::from(!condition))
            .and_then(|region| region.blocks.first())
            .ok_or_else(|| ReferenceValueError::Invalid("if region ABI".to_owned()))?;
        let captures = operation
            .operands
            .iter()
            .skip(1)
            .map(|identifier| exact_value(environment, *identifier))
            .collect::<Result<Vec<_>, _>>()?;
        self.evaluate_block(region, &mut HashMap::new(), &captures, depth)
    }

    fn evaluate_while(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<StaticValue>>,
        depth: usize,
    ) -> Result<Flow, ReferenceValueError> {
        let mut carries = operation_values(operation, environment)?;
        let condition = &operation.regions[0].blocks[0];
        let body = &operation.regions[1].blocks[0];
        let otherwise = &operation.regions[2].blocks[0];
        let mut iterations = 0;
        loop {
            self.consume_work()?;
            let flow = self.evaluate_block(condition, &mut HashMap::new(), &carries, depth)?;
            let test = truth(&flow.values[0])?;
            carries = flow.values[1..].to_vec();
            if !test {
                return self.evaluate_block(otherwise, &mut HashMap::new(), &carries, depth);
            }
            if iterations >= MAX_REFERENCE_LOOP_ITERATIONS {
                return Err(ReferenceValueError::WorkBudgetExceeded);
            }
            iterations += 1;
            let flow = self.evaluate_block(body, &mut HashMap::new(), &carries, depth)?;
            match flow.kind {
                FlowKind::Yield | FlowKind::Continue => carries = flow.values,
                FlowKind::Break => {
                    return Ok(Flow {
                        kind: FlowKind::Yield,
                        values: flow.values,
                    });
                }
                FlowKind::Return => return Ok(flow),
            }
        }
    }

    fn evaluate_for(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<StaticValue>>,
        depth: usize,
    ) -> Result<Flow, ReferenceValueError> {
        if string_attribute(operation, "body_mode") == Some("adjoint") {
            return Err(ReferenceValueError::Unsupported(
                "adjoint structured for".to_owned(),
            ));
        }
        let start = integer(&operand(environment, operation, 0)?)?;
        let upper = integer(&operand(environment, operation, 1)?)?;
        let stride = integer(&operand(environment, operation, 2)?)?;
        let count = static_values::range_len(start, upper, stride).ok_or_else(|| {
            ReferenceValueError::Invalid("for-loop step cannot be zero".to_owned())
        })?;
        if count > MAX_REFERENCE_LOOP_ITERATIONS {
            return Err(ReferenceValueError::WorkBudgetExceeded);
        }
        let mut carries = operation
            .operands
            .iter()
            .skip(3)
            .map(|identifier| exact_value(environment, *identifier))
            .collect::<Result<Vec<_>, _>>()?;
        let body = operation
            .regions
            .first()
            .and_then(|region| region.blocks.first())
            .ok_or_else(|| ReferenceValueError::Invalid("for body ABI".to_owned()))?;
        let reverse = bool_attribute(operation, "reverse").unwrap_or(false);
        let mut broke = false;
        for offset in 0..count {
            self.consume_work()?;
            let offset = if reverse { count - 1 - offset } else { offset };
            let index = i128::from(start)
                .checked_add(
                    i128::from(stride)
                        * i128::try_from(offset)
                            .map_err(|_| ReferenceValueError::WorkBudgetExceeded)?,
                )
                .and_then(|value| i64::try_from(value).ok())
                .ok_or_else(|| {
                    ReferenceValueError::Invalid("for-loop index overflow".to_owned())
                })?;
            let mut arguments = Vec::with_capacity(carries.len() + 1);
            arguments.push(StaticValue::Scalar(StaticScalar::Int(index)));
            arguments.extend(carries);
            let flow = self.evaluate_block(body, &mut HashMap::new(), &arguments, depth)?;
            match flow.kind {
                FlowKind::Yield | FlowKind::Continue => carries = flow.values,
                FlowKind::Break => {
                    carries = flow.values;
                    broke = true;
                    break;
                }
                FlowKind::Return => return Ok(flow),
            }
        }
        if !broke && operation.regions.len() == 2 {
            let otherwise = operation.regions[1]
                .blocks
                .first()
                .ok_or_else(|| ReferenceValueError::Invalid("for-else region ABI".to_owned()))?;
            let flow = self.evaluate_block(otherwise, &mut HashMap::new(), &carries, depth)?;
            match flow.kind {
                FlowKind::Yield => carries = flow.values,
                FlowKind::Return | FlowKind::Break | FlowKind::Continue => return Ok(flow),
            }
        }
        Ok(Flow {
            kind: FlowKind::Yield,
            values: carries,
        })
    }

    fn evaluate_classical(
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<StaticValue>>,
    ) -> Result<StaticValue, ReferenceValueError> {
        match static_values::evaluate_value_checked(operation, environment) {
            Ok(Some(value)) if value.is_complete() => Ok(value),
            Ok(_) => Err(ReferenceValueError::Unsupported(operation.opcode.clone())),
            Err(
                NumericEvaluationError::Undefined(reason) | NumericEvaluationError::Limit(reason),
            ) => Err(ReferenceValueError::Invalid(reason.to_owned())),
        }
    }
}

fn operation_values(
    operation: &wire::OperationProto,
    environment: &HashMap<u64, Option<StaticValue>>,
) -> Result<Vec<StaticValue>, ReferenceValueError> {
    operation
        .operands
        .iter()
        .map(|identifier| exact_value(environment, *identifier))
        .collect()
}

fn operand(
    environment: &HashMap<u64, Option<StaticValue>>,
    operation: &wire::OperationProto,
    index: usize,
) -> Result<StaticValue, ReferenceValueError> {
    operation
        .operands
        .get(index)
        .copied()
        .ok_or_else(|| {
            ReferenceValueError::Invalid(format!("operand ABI for {:?}", operation.opcode))
        })
        .and_then(|identifier| exact_value(environment, identifier))
}

fn exact_value(
    environment: &HashMap<u64, Option<StaticValue>>,
    identifier: u64,
) -> Result<StaticValue, ReferenceValueError> {
    environment
        .get(&identifier)
        .cloned()
        .flatten()
        .filter(StaticValue::is_complete)
        .ok_or_else(|| {
            ReferenceValueError::Invalid(format!("missing exact SSA value %{identifier}"))
        })
}

fn flow_kind(kind: &str) -> Result<FlowKind, ReferenceValueError> {
    match kind {
        "return" => Ok(FlowKind::Return),
        "yield" => Ok(FlowKind::Yield),
        "continue" => Ok(FlowKind::Continue),
        "break" => Ok(FlowKind::Break),
        _ => Err(ReferenceValueError::Invalid(format!(
            "unknown structured exit {kind:?}"
        ))),
    }
}

fn truth(value: &StaticValue) -> Result<bool, ReferenceValueError> {
    value
        .scalar()
        .map(static_values::truth)
        .ok_or_else(|| ReferenceValueError::Invalid("non-scalar condition".to_owned()))
}

fn integer(value: &StaticValue) -> Result<i64, ReferenceValueError> {
    value
        .scalar()
        .and_then(static_values::integer)
        .ok_or_else(|| ReferenceValueError::Invalid("non-integer loop value".to_owned()))
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
    match attribute(operation, name)?.value.as_ref()? {
        wire::literal_proto::Value::StringValue(value) => Some(value),
        _ => None,
    }
}

fn bool_attribute(operation: &wire::OperationProto, name: &str) -> Option<bool> {
    match attribute(operation, name)?.value.as_ref()? {
        wire::literal_proto::Value::BoolValue(value) => Some(*value),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigInt;

    use super::*;

    fn scalar(name: &str) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Scalar.into(),
            name: name.to_owned(),
            ..Default::default()
        }
    }

    fn value(identifier: u64, name: &str) -> wire::ValueProto {
        wire::ValueProto {
            id: identifier,
            r#type: Some(scalar(name)),
            ..Default::default()
        }
    }

    fn int_literal(value: i64) -> wire::LiteralProto {
        wire::LiteralProto {
            value: Some(wire::literal_proto::Value::IntValue(value)),
        }
    }

    fn complex_literal(real: f64, imaginary: f64) -> wire::LiteralProto {
        wire::LiteralProto {
            value: Some(wire::literal_proto::Value::ComplexValue(
                wire::literal_proto::Complex { real, imaginary },
            )),
        }
    }

    fn program_with(operation: wire::OperationProto) -> wire::QCorePackageProto {
        wire::QCorePackageProto {
            entrypoint: "test.main".to_owned(),
            functions: vec![wire::FunctionProto {
                symbol: "test.main".to_owned(),
                parameters: vec![wire::ParameterProto {
                    name: "input".to_owned(),
                    value: Some(value(0, "Int")),
                    ..Default::default()
                }],
                result_type: Some(scalar("Int")),
                body: Some(wire::BlockProto {
                    operations: vec![
                        wire::OperationProto {
                            opcode: "qcore.constant".to_owned(),
                            results: vec![value(1, "Int")],
                            attributes: vec![wire::AttributeProto {
                                name: "value".to_owned(),
                                value: Some(int_literal(2)),
                            }],
                            ..Default::default()
                        },
                        operation,
                    ],
                    terminator: Some(wire::TerminatorProto {
                        kind: "return".to_owned(),
                        operands: vec![2],
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn evaluates_standalone_classical_literals_with_python_division_semantics() {
        let result = evaluate_classical_literals(
            "classical.floordiv",
            &[
                int_literal(-7).encode_to_vec(),
                int_literal(3).encode_to_vec(),
            ],
            &[],
        )
        .expect("supported standalone operation");
        let literal = wire::LiteralProto::decode(result.as_slice()).expect("result literal");

        assert_eq!(literal, int_literal(-3));
    }

    #[test]
    fn evaluates_standalone_immutable_mapping_lookup() {
        use wire::literal_proto::Value;

        let entry = wire::LiteralProto {
            value: Some(Value::MappingValue(wire::literal_proto::Mapping {
                entries: vec![
                    wire::literal_proto::Entry {
                        key: "key".to_owned(),
                        value: Some(int_literal(4)),
                    },
                    wire::literal_proto::Entry {
                        key: "value".to_owned(),
                        value: Some(wire::LiteralProto {
                            value: Some(Value::BoolValue(true)),
                        }),
                    },
                ],
            })),
        };
        let mapping = wire::LiteralProto {
            value: Some(Value::TupleValue(wire::literal_proto::Sequence {
                values: vec![entry],
            })),
        };
        let result = evaluate_classical_literals(
            "classical.map_get",
            &[mapping.encode_to_vec(), int_literal(4).encode_to_vec()],
            &[],
        )
        .expect("supported standalone mapping lookup");
        let literal = wire::LiteralProto::decode(result.as_slice()).expect("result literal");

        assert_eq!(literal.value, Some(Value::BoolValue(true)));
    }

    #[test]
    fn evaluates_one_exact_invocation_into_the_existing_literal_schema() {
        let package = program_with(wire::OperationProto {
            opcode: "classical.add".to_owned(),
            operands: vec![0, 1],
            results: vec![value(2, "Int")],
            ..Default::default()
        });

        let result = evaluate(
            &package,
            &[CircuitLaunchBinding::new(
                "input".to_owned(),
                CircuitLaunchValue::Int(3),
            )],
        )
        .expect("supported exact invocation");
        let literal = wire::LiteralProto::decode(result.as_slice()).expect("result literal");

        assert_eq!(literal, int_literal(5));
    }

    #[test]
    fn evaluates_an_arbitrary_precision_launch_integer_without_narrowing() {
        let package = program_with(wire::OperationProto {
            opcode: "classical.add".to_owned(),
            operands: vec![0, 1],
            results: vec![value(2, "Int")],
            ..Default::default()
        });
        let input = BigInt::from(1_u8) << 4096_u32;

        let result = evaluate(
            &package,
            &[CircuitLaunchBinding::new(
                "input".to_owned(),
                CircuitLaunchValue::BigInt(input.to_signed_bytes_be().into_boxed_slice()),
            )],
        )
        .expect("supported arbitrary-precision invocation");
        let literal = wire::LiteralProto::decode(result.as_slice()).expect("result literal");

        assert_eq!(
            literal.value,
            Some(wire::literal_proto::Value::BigintValue(
                (input + BigInt::from(2_u8)).to_signed_bytes_be()
            ))
        );
    }

    #[test]
    fn evaluates_an_exact_complex_launch_value() {
        let complex = scalar("Complex");
        let package = wire::QCorePackageProto {
            entrypoint: "test.main".to_owned(),
            functions: vec![wire::FunctionProto {
                symbol: "test.main".to_owned(),
                parameters: vec![wire::ParameterProto {
                    name: "input".to_owned(),
                    value: Some(value(0, "Complex")),
                    ..Default::default()
                }],
                result_type: Some(complex.clone()),
                body: Some(wire::BlockProto {
                    operations: vec![
                        wire::OperationProto {
                            opcode: "qcore.constant".to_owned(),
                            results: vec![value(1, "Complex")],
                            attributes: vec![wire::AttributeProto {
                                name: "value".to_owned(),
                                value: Some(complex_literal(1.0, 2.0)),
                            }],
                            ..Default::default()
                        },
                        wire::OperationProto {
                            opcode: "classical.mul".to_owned(),
                            operands: vec![0, 1],
                            results: vec![value(2, "Complex")],
                            ..Default::default()
                        },
                    ],
                    terminator: Some(wire::TerminatorProto {
                        kind: "return".to_owned(),
                        operands: vec![2],
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };

        let result = evaluate(
            &package,
            &[CircuitLaunchBinding::new(
                "input".to_owned(),
                CircuitLaunchValue::ComplexBits(1.5_f64.to_bits(), (-2.25_f64).to_bits()),
            )],
        )
        .expect("supported exact Complex invocation");
        let literal = wire::LiteralProto::decode(result.as_slice()).expect("result literal");

        assert_eq!(literal, complex_literal(6.0, 0.75));
    }

    #[test]
    fn rejects_effectful_execution_as_a_typed_coverage_gap() {
        let package = program_with(wire::OperationProto {
            opcode: "quantum.alloc".to_owned(),
            operands: vec![1],
            results: vec![value(2, "Int")],
            ..Default::default()
        });

        assert_eq!(
            evaluate(
                &package,
                &[CircuitLaunchBinding::new(
                    "input".to_owned(),
                    CircuitLaunchValue::Int(3),
                )],
            ),
            Err(ReferenceValueError::Unsupported("quantum.alloc".to_owned()))
        );
    }
}
