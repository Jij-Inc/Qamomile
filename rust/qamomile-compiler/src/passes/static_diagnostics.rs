//! Bounded diagnostics on the selected source path before algebraic rewriting.
//!
//! Unknown operands, control flow, recursion depth, and analysis exhaustion do
//! not establish an error. This pass neither proves totality nor executes quantum
//! state. Numeric domains and selector validity remain owned by native validators.

use std::collections::HashMap;
use std::sync::Arc;

use crate::numeric_values::NumericEvaluationError;
use crate::operations::{
    LocallyCheckedOperations, SelectorOperand, primitive_gate_spec, selector_operands,
};
use crate::static_values::{self, StaticScalar, StaticValue};
use crate::types::{SemanticType, TypeArena};
use crate::{VerifiedProgram, wire};

/// A selected source operation whose operands establish a domain failure.
#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
#[error("Q7804: compile-time domain error in {callable} at {opcode}: {reason}")]
pub struct StaticDomainError {
    callable: String,
    opcode: String,
    reason: String,
}

const MAX_DIAGNOSTIC_WORK: usize = 10_000;
const MAX_DIAGNOSTIC_DEPTH: usize = 64;

type Environment = HashMap<u64, Fact>;

#[derive(Clone, Default)]
enum Fact {
    #[default]
    Unknown,
    Classical(StaticValue),
    QuantumWidth(usize),
    Product(Arc<[Self]>),
}

impl Fact {
    fn classical(&self) -> Option<StaticValue> {
        match self {
            Self::Classical(value) => Some(value.clone()),
            Self::Product(values) => Some(StaticValue::Aggregate(
                values.iter().map(Self::classical).collect(),
            )),
            Self::Unknown | Self::QuantumWidth(_) => None,
        }
    }

    fn width(&self) -> Option<usize> {
        match self {
            Self::QuantumWidth(width) => Some(*width),
            Self::Product(values) => values
                .iter()
                .try_fold(0_usize, |total, value| total.checked_add(value.width()?)),
            Self::Unknown | Self::Classical(_) => None,
        }
    }

    fn integer(&self) -> Option<i64> {
        self.classical()?.scalar().and_then(static_values::integer)
    }
}

enum Flow {
    Values(Vec<Fact>),
    Return(Vec<Fact>),
    Unknown,
}

pub(super) fn check(program: &VerifiedProgram) -> Result<(), StaticDomainError> {
    let package = &program.local.checked.package;
    let functions = package
        .functions
        .iter()
        .map(|function| (function.symbol.as_str(), function))
        .collect::<HashMap<_, _>>();
    let entry = functions[package.entrypoint.as_str()];
    let mut checker = Checker {
        functions,
        operations: &program.local.checked.operations,
        arena: &program.local.checked.type_arena,
        remaining: MAX_DIAGNOSTIC_WORK,
    };
    let arguments = entry
        .parameters
        .iter()
        .map(|parameter| {
            package
                .const_bindings
                .iter()
                .find(|b| b.name == parameter.name)
                .and_then(|b| b.value.as_ref())
                .and_then(|v| {
                    parameter
                        .value
                        .as_ref()?
                        .r#type
                        .as_ref()
                        .and_then(|t| static_values::typed_value_literal(v, t))
                })
                .map_or_else(
                    || {
                        parameter
                            .value
                            .as_ref()
                            .map_or(Fact::Unknown, |v| checker.typed_fact(&entry.symbol, v.id))
                    },
                    Fact::Classical,
                )
        })
        .collect();
    checker.call(entry, arguments, 0)?;
    Ok(())
}

struct Checker<'a> {
    functions: HashMap<&'a str, &'a wire::FunctionProto>,
    operations: &'a LocallyCheckedOperations,
    arena: &'a TypeArena,
    remaining: usize,
}

impl Checker<'_> {
    fn typed_fact(&self, owner: &str, id: u64) -> Fact {
        let root = self
            .operations
            .value_types(owner)
            .expect("verified function types")[&id];
        let Some(SemanticType::Quantum(payload)) = self.arena.get(self.arena.strip_stage(root))
        else {
            return Fact::Unknown;
        };
        self.arena
            .coherent_payload_width(*payload)
            .ok()
            .flatten()
            .and_then(|width| usize::try_from(width).ok())
            .map_or(Fact::Unknown, Fact::QuantumWidth)
    }

    fn visit(&mut self, depth: usize) -> bool {
        if depth >= MAX_DIAGNOSTIC_DEPTH || self.remaining == 0 {
            return false;
        }
        self.remaining -= 1;
        true
    }

    fn call(
        &mut self,
        function: &wire::FunctionProto,
        arguments: Vec<Fact>,
        depth: usize,
    ) -> Result<Fact, StaticDomainError> {
        if !self.visit(depth) {
            return Ok(Fact::Unknown);
        }
        let Some(body) = &function.body else {
            return Ok(Fact::Unknown);
        };
        let mut environment = function
            .parameters
            .iter()
            .zip(arguments)
            .filter_map(|(p, v)| p.value.as_ref().map(|p| (p.id, v)))
            .collect();
        match self.block(body, &function.symbol, &mut environment, depth + 1)? {
            Flow::Return(mut values) if values.len() == 1 => Ok(values.remove(0)),
            _ => Ok(Fact::Unknown),
        }
    }

    fn block(
        &mut self,
        block: &wire::BlockProto,
        owner: &str,
        environment: &mut Environment,
        depth: usize,
    ) -> Result<Flow, StaticDomainError> {
        for operation in &block.operations {
            if !self.visit(depth) {
                return Ok(Flow::Unknown);
            }
            let arguments = operation
                .operands
                .iter()
                .map(|id| environment.get(id).cloned().unwrap_or_default())
                .collect::<Vec<_>>();
            let flow = self.operation(operation, owner, arguments, depth)?;
            let Flow::Values(values) = flow else {
                return Ok(flow);
            };
            for (result, value) in operation.results.iter().zip(values) {
                environment.insert(
                    result.id,
                    match value {
                        Fact::Unknown => self.typed_fact(owner, result.id),
                        value => value,
                    },
                );
            }
        }
        let Some(terminator) = &block.terminator else {
            return Ok(Flow::Unknown);
        };
        let values = terminator
            .operands
            .iter()
            .map(|id| environment.get(id).cloned().unwrap_or_default())
            .collect();
        Ok(match terminator.kind.as_str() {
            "return" => Flow::Return(values),
            "yield" => Flow::Values(values),
            _ => Flow::Unknown,
        })
    }

    fn region(
        &mut self,
        region: &wire::RegionProto,
        owner: &str,
        arguments: Vec<Fact>,
        depth: usize,
    ) -> Result<Flow, StaticDomainError> {
        let [block] = region.blocks.as_slice() else {
            return Ok(Flow::Unknown);
        };
        let mut environment = block
            .arguments
            .iter()
            .zip(arguments)
            .map(|(v, f)| (v.id, f))
            .collect();
        self.block(block, owner, &mut environment, depth + 1)
    }

    fn operation(
        &mut self,
        operation: &wire::OperationProto,
        owner: &str,
        arguments: Vec<Fact>,
        depth: usize,
    ) -> Result<Flow, StaticDomainError> {
        match operation.opcode.as_str() {
            "qcore.call" | "qcore.adjoint_call" => {
                return self.operation_call(operation, arguments, depth);
            }
            "qcore.if" => {
                let condition = arguments
                    .first()
                    .and_then(Fact::classical)
                    .and_then(|v| v.scalar())
                    .map(static_values::truth);
                let Some(condition) = condition else {
                    return Ok(Flow::Unknown);
                };
                return self.region(
                    &operation.regions[usize::from(!condition)],
                    owner,
                    arguments[1..].to_vec(),
                    depth,
                );
            }
            "qcore.for" => return self.for_loop(operation, owner, &arguments, depth),
            _ if !operation.regions.is_empty() => return Ok(Flow::Unknown),
            _ => {}
        }
        if primitive_gate_spec(&operation.opcode).is_some() {
            check_selectors(operation, owner, &arguments)?;
            return Ok(Flow::Values(vec![
                arguments.first().cloned().unwrap_or_default(),
            ]));
        }
        let values = match operation.opcode.as_str() {
            "qcore.constant" => vec![
                attribute(operation, "value")
                    .and_then(|v| {
                        operation
                            .results
                            .first()?
                            .r#type
                            .as_ref()
                            .and_then(|t| static_values::typed_value_literal(v, t))
                    })
                    .map_or(Fact::Unknown, Fact::Classical),
            ],
            "qcore.stage_lift" => arguments,
            "qcore.tuple" | "tuple" | "array" => vec![Fact::Product(arguments.into())],
            "qcore.unpack" => unpack(&arguments, operation.results.len()),
            "qcore.reshape" => vec![
                arguments
                    .iter()
                    .try_fold(0_usize, |n, f| n.checked_add(f.width()?))
                    .map_or(Fact::Unknown, Fact::QuantumWidth),
            ],
            "quantum.alloc" => {
                let width = arguments.first().and_then(Fact::integer);
                if width.is_some_and(|n| n <= 0) {
                    return Err(invalid(
                        owner,
                        operation,
                        "allocation width must be positive",
                    ));
                }
                vec![
                    width
                        .and_then(|n| usize::try_from(n).ok())
                        .map_or(Fact::Unknown, Fact::QuantumWidth),
                ]
            }
            _ if operation.opcode.starts_with("classical.") => {
                vec![classical(operation, owner, &arguments)?]
            }
            _ => vec![Fact::Unknown; operation.results.len()],
        };
        Ok(Flow::Values(values))
    }

    fn operation_call(
        &mut self,
        operation: &wire::OperationProto,
        arguments: Vec<Fact>,
        depth: usize,
    ) -> Result<Flow, StaticDomainError> {
        let Some(callee) = attribute(operation, "callee").and_then(|v| match v.value.as_ref()? {
            wire::literal_proto::Value::StringValue(name) => {
                self.functions.get(name.as_str()).copied()
            }
            _ => None,
        }) else {
            return Ok(Flow::Unknown);
        };
        let arguments = if operation.opcode == "qcore.adjoint_call" {
            let owners = callee
                .parameters
                .iter()
                .filter(|p| p.value.as_ref().is_some_and(|v| v.has_region))
                .count();
            if owners != 1 || callee.parameters.len() != arguments.len() {
                return Ok(Flow::Values(vec![Fact::Unknown]));
            }
            let mut classical = arguments.iter().skip(1);
            callee
                .parameters
                .iter()
                .map(|p| {
                    if p.value.as_ref().is_some_and(|v| v.has_region) {
                        arguments[0].clone()
                    } else {
                        classical.next().cloned().unwrap_or_default()
                    }
                })
                .collect()
        } else {
            arguments
        };
        Ok(Flow::Values(vec![self.call(
            callee,
            arguments,
            depth + 1,
        )?]))
    }

    fn for_loop(
        &mut self,
        operation: &wire::OperationProto,
        owner: &str,
        arguments: &[Fact],
        depth: usize,
    ) -> Result<Flow, StaticDomainError> {
        let bounds = arguments
            .iter()
            .take(3)
            .map(Fact::integer)
            .collect::<Option<Vec<_>>>();
        let Some(bounds) = bounds else {
            return Ok(Flow::Unknown);
        };
        let [start, stop, stride] = bounds.as_slice() else {
            return Ok(Flow::Unknown);
        };
        if *stride == 0 {
            return Err(invalid(owner, operation, "range step cannot be zero"));
        }
        // Reverse loops have the same domain, but diagnostic ordering must not
        // assume forward induction. Their checked target lowering owns that case.
        if attribute(operation, "reverse")
            .is_some_and(|v| matches!(v.value, Some(wire::literal_proto::Value::BoolValue(true))))
        {
            return Ok(Flow::Unknown);
        }
        let mut index = i128::from(*start);
        let mut carries = arguments[3..].to_vec();
        while if *stride > 0 {
            index < i128::from(*stop)
        } else {
            index > i128::from(*stop)
        } {
            if !self.visit(depth) {
                return Ok(Flow::Unknown);
            }
            let Ok(value) = i64::try_from(index) else {
                return Ok(Flow::Unknown);
            };
            let inputs = std::iter::once(Fact::Classical(StaticValue::Scalar(StaticScalar::Int(
                value,
            ))))
            .chain(carries)
            .collect();
            match self.region(&operation.regions[0], owner, inputs, depth)? {
                Flow::Values(values) => carries = values,
                flow => return Ok(flow),
            }
            index += i128::from(*stride);
        }
        self.region(&operation.regions[1], owner, carries, depth)
    }
}

fn classical(
    operation: &wire::OperationProto,
    owner: &str,
    arguments: &[Fact],
) -> Result<Fact, StaticDomainError> {
    let known = operation
        .operands
        .iter()
        .copied()
        .zip(arguments.iter().map(Fact::classical))
        .collect();
    if let Some(reason) = static_values::known_domain_error(operation, &known) {
        return Err(invalid(owner, operation, reason));
    }
    match static_values::evaluate_value_checked(operation, &known) {
        Ok(value) => Ok(value.map_or(Fact::Unknown, Fact::Classical)),
        Err(NumericEvaluationError::Undefined(reason)) => Err(invalid(owner, operation, reason)),
        // Exhausting a numeric representation budget is not evidence of a
        // source-domain failure, just as an unknown argument is not one.
        Err(NumericEvaluationError::Limit(_)) => Ok(Fact::Unknown),
    }
}

fn check_selectors(
    operation: &wire::OperationProto,
    owner: &str,
    arguments: &[Fact],
) -> Result<(), StaticDomainError> {
    let Some(selectors) = selector_operands(operation) else {
        return Ok(());
    };
    let arguments = operation
        .operands
        .iter()
        .copied()
        .zip(arguments)
        .collect::<HashMap<_, _>>();
    let selectors = selectors
        .into_iter()
        .map(|selector| match selector {
            SelectorOperand::Value(id) => arguments
                .get(&id)
                .and_then(|f| f.integer())
                .map_or(selector, SelectorOperand::Default),
            value @ SelectorOperand::Default(_) => value,
        })
        .collect::<Vec<_>>();
    let known = selectors
        .iter()
        .filter_map(|selector| match selector {
            SelectorOperand::Default(value) => Some(*value),
            SelectorOperand::Value(_) => None,
        })
        .collect::<Vec<_>>();
    let width = operation
        .operands
        .first()
        .and_then(|id| arguments.get(id))
        .and_then(|f| f.width())
        .unwrap_or(usize::MAX);
    crate::validate_circuit_selectors(&operation.opcode, width, &known)
        .map_err(|error| invalid(owner, operation, error.to_string()))?;
    crate::circuit::require_distinct_selectors(&operation.opcode, &selectors)
        .map_err(|error| invalid(owner, operation, error.to_string()))
}

fn unpack(arguments: &[Fact], count: usize) -> Vec<Fact> {
    match arguments.first() {
        Some(Fact::Product(values)) => values.to_vec(),
        Some(Fact::Classical(StaticValue::Aggregate(values))) => values
            .iter()
            .cloned()
            .map(|v| v.map_or(Fact::Unknown, Fact::Classical))
            .collect(),
        _ => vec![Fact::Unknown; count],
    }
}

fn attribute<'a>(
    operation: &'a wire::OperationProto,
    name: &str,
) -> Option<&'a wire::LiteralProto> {
    operation
        .attributes
        .iter()
        .find(|a| a.name == name)?
        .value
        .as_ref()
}

fn invalid(
    owner: &str,
    operation: &wire::OperationProto,
    reason: impl Into<String>,
) -> StaticDomainError {
    StaticDomainError {
        callable: owner.into(),
        opcode: operation.opcode.clone(),
        reason: reason.into(),
    }
}
