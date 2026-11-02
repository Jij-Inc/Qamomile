//! Classical value propagation during symbolic analysis.

use crate::resource_rules::Domain as ResourceDomain;
use crate::static_values::{
    StaticScalar, StaticValue, evaluate_value as evaluate_static_value, value_to_literal,
};
use crate::types::{Dimension, SemanticType};
use crate::wire;
use std::collections::HashMap;
use std::sync::Arc;

use super::{
    ResourceExpressionNode, ResourcePredicateKind, ScalarFact, SymbolicAnalyzer,
    SymbolicFunctionContext, SymbolicResourceError, ValueFact, attribute, bool_attribute,
    exact_static_value, int_fact, project_aggregate, scalar_expression,
};

impl SymbolicAnalyzer<'_> {
    pub(super) fn evaluate_classical(
        &mut self,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ValueFact>>,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<Option<ValueFact>, SymbolicResourceError> {
        let name = operation
            .opcode
            .strip_prefix("classical.")
            .unwrap_or(&operation.opcode);
        if name == "len" && bool_attribute(operation, "shape_only") == Some(true) {
            return self.evaluate_shape_length(operation, context);
        }
        if matches!(name, "index" | "project") {
            return Ok(project_aggregate(operation, environment));
        }
        let exact_environment = operation
            .operands
            .iter()
            .map(|operand| {
                let exact = environment
                    .get(operand)
                    .and_then(Option::as_ref)
                    .and_then(|value| exact_static_value(value, &self.arena));
                (*operand, exact)
            })
            .collect::<HashMap<_, _>>();
        if matches!(name, "slice" | "update") {
            return Ok(structural_aggregate_fact(
                operation,
                environment,
                &exact_environment,
            ));
        }
        let exact = evaluate_static_value(operation, &exact_environment);
        if let Some(value) = exact {
            let literal = value_to_literal(&value).ok_or_else(|| {
                SymbolicResourceError::Missing("static classical result literal".to_owned())
            })?;
            return self.literal_fact(&literal);
        }
        if matches!(name, "ceil" | "log2" | "sqrt") {
            return self.evaluate_structural_math(name, operation, environment);
        }
        if matches!(name, "compare_chain" | "and" | "or" | "not") {
            return self.evaluate_symbolic_predicate(name, operation, environment);
        }
        let Some(values) = operation
            .operands
            .iter()
            .map(|operand| int_fact(environment.get(operand).and_then(Option::as_ref)))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        if values.iter().all(|value| value.exact.is_some()) {
            return Ok(None);
        }
        let expression = match (name, values.as_slice()) {
            ("add", [left, right]) => self.arena.add([left.expression, right.expression])?,
            ("sub", [left, right]) => {
                let negative = self.arena.literal_i64(-1)?;
                let right = self.arena.multiply([negative, right.expression])?;
                self.arena.add([left.expression, right])?
            }
            ("mul", [left, right]) => self.arena.multiply([left.expression, right.expression])?,
            ("floordiv", [left, right]) => self.arena.contracted(
                "qamomile.floor_divide",
                ResourceDomain::Integer,
                [left.expression, right.expression],
            )?,
            ("mod", [left, right]) => self.arena.contracted(
                "qamomile.modulo",
                ResourceDomain::Integer,
                [left.expression, right.expression],
            )?,
            ("pow", [left, right]) => self.arena.power(left.expression, right.expression)?,
            ("lshift", [left, right]) => {
                let two = self.arena.literal_i64(2)?;
                let power = self.arena.power(two, right.expression)?;
                self.arena.multiply([left.expression, power])?
            }
            ("rshift", [left, right]) => {
                let two = self.arena.literal_i64(2)?;
                let power = self.arena.power(two, right.expression)?;
                self.arena.contracted(
                    "qamomile.floor_divide",
                    ResourceDomain::Integer,
                    [left.expression, power],
                )?
            }
            ("pos" | "cast", [value]) => value.expression,
            ("neg", [value]) => {
                let negative = self.arena.literal_i64(-1)?;
                self.arena.multiply([negative, value.expression])?
            }
            _ => return Ok(None),
        };
        Ok(Some(ValueFact::Scalar(ScalarFact::Int {
            expression,
            exact: None,
        })))
    }

    fn evaluate_shape_length(
        &mut self,
        operation: &wire::OperationProto,
        context: SymbolicFunctionContext<'_>,
    ) -> Result<Option<ValueFact>, SymbolicResourceError> {
        let source = operation
            .operands
            .first()
            .and_then(|operand| self.operations.value_types(context.owner)?.get(operand))
            .copied()
            .ok_or_else(|| SymbolicResourceError::Missing("shape-only length type".to_owned()))?;
        let dimension = match self.types.get(self.types.strip_stage(source)) {
            Some(SemanticType::Array { shape, .. }) => shape.first().cloned(),
            Some(SemanticType::Word { width, .. }) => Some(width.clone()),
            Some(SemanticType::Product(elements)) => Some(Dimension::Fixed(
                u64::try_from(elements.len()).map_err(|_| SymbolicResourceError::Overflow)?,
            )),
            Some(SemanticType::Record { fields, .. }) => Some(Dimension::Fixed(
                u64::try_from(fields.len()).map_err(|_| SymbolicResourceError::Overflow)?,
            )),
            _ => None,
        }
        .ok_or_else(|| SymbolicResourceError::Missing("shape-only length extent".to_owned()))?;
        let expression = if let Dimension::Symbol { scope, name } = &dimension
            && name.starts_with("slice(")
        {
            let extent = crate::core::slice_shape::SliceShape::parse(name)
                .and_then(|shape| {
                    shape.evaluate(&|name| {
                        let dimension = Dimension::Symbol {
                            scope: Arc::clone(scope),
                            name: Arc::from(name),
                        };
                        let expression = context.dimensions.values.get(&dimension)?;
                        match self.arena.node(*expression)? {
                            ResourceExpressionNode::Literal {
                                value,
                                domain: ResourceDomain::Natural | ResourceDomain::Integer,
                            } => value.parse::<i64>().ok(),
                            _ => None,
                        }
                    })
                })
                .map_err(|reason| {
                    SymbolicResourceError::Unsupported(format!(
                        "invalid Const slice shape: {reason}"
                    ))
                })?;
            extent
                .map(|extent| self.arena.literal_u128(u128::from(extent)))
                .transpose()?
        } else {
            self.resolve_dimension(&dimension, context.dimensions)?
        };
        let Some(expression) = expression else {
            return Ok(None);
        };
        let exact = match self.arena.node(expression) {
            Some(ResourceExpressionNode::Literal { value, .. }) => value.parse::<i64>().ok(),
            _ => None,
        };
        Ok(Some(ValueFact::Scalar(ScalarFact::Int {
            expression,
            exact,
        })))
    }

    fn evaluate_symbolic_predicate(
        &mut self,
        name: &str,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ValueFact>>,
    ) -> Result<Option<ValueFact>, SymbolicResourceError> {
        let unknown = Some(ValueFact::Scalar(ScalarFact::Bool(None)));
        let mut predicates = Vec::new();
        let kind = if name == "compare_chain" {
            let Some(wire::literal_proto::Value::TupleValue(operators)) =
                attribute(operation, "operators").and_then(|value| value.value.as_ref())
            else {
                return Ok(unknown);
            };
            for (operator, operands) in operators.values.iter().zip(operation.operands.windows(2)) {
                let Some(wire::literal_proto::Value::StringValue(operator)) =
                    operator.value.as_ref()
                else {
                    return Ok(unknown);
                };
                let kind = match operator.as_str() {
                    "eq" => ResourcePredicateKind::Equal,
                    "ne" => ResourcePredicateKind::NotEqual,
                    "lt" => ResourcePredicateKind::Less,
                    "le" => ResourcePredicateKind::LessEqual,
                    "gt" => ResourcePredicateKind::Greater,
                    "ge" => ResourcePredicateKind::GreaterEqual,
                    _ => return Ok(unknown),
                };
                let Some(left) = int_fact(environment.get(&operands[0]).and_then(Option::as_ref))
                else {
                    return Ok(unknown);
                };
                let Some(right) = int_fact(environment.get(&operands[1]).and_then(Option::as_ref))
                else {
                    return Ok(unknown);
                };
                predicates.push(
                    self.arena
                        .predicate_comparison(kind, [left.expression, right.expression])?,
                );
            }
            ResourcePredicateKind::And
        } else {
            for operand in &operation.operands {
                let predicate = match environment.get(operand).and_then(Option::as_ref) {
                    Some(ValueFact::Scalar(ScalarFact::Predicate(predicate))) => *predicate,
                    Some(ValueFact::Scalar(ScalarFact::Bool(Some(value)))) => {
                        self.arena.predicate_literal(*value)?
                    }
                    _ => return Ok(unknown),
                };
                predicates.push(predicate);
            }
            match name {
                "and" => ResourcePredicateKind::And,
                "or" => ResourcePredicateKind::Or,
                _ => ResourcePredicateKind::Not,
            }
        };
        let predicate = self.arena.predicate_connective(kind, predicates)?;
        Ok(Some(ValueFact::Scalar(ScalarFact::Predicate(predicate))))
    }

    pub(super) fn evaluate_structural_math(
        &mut self,
        name: &str,
        operation: &wire::OperationProto,
        environment: &HashMap<u64, Option<ValueFact>>,
    ) -> Result<Option<ValueFact>, SymbolicResourceError> {
        let [operand] = operation.operands.as_slice() else {
            return Ok(None);
        };
        let Some((expression, is_integer)) = environment
            .get(operand)
            .and_then(Option::as_ref)
            .and_then(scalar_expression)
        else {
            return Ok(None);
        };
        if name == "log2" {
            return Ok(Some(ValueFact::Scalar(ScalarFact::Real {
                expression: Some(self.arena.log2(expression)?),
                exact: None,
            })));
        }
        if name == "sqrt" {
            return Ok(Some(ValueFact::Scalar(ScalarFact::Real {
                expression: Some(self.arena.square_root(expression)?),
                exact: None,
            })));
        }
        let expression = if is_integer {
            expression
        } else {
            self.arena.ceiling(expression)?
        };
        Ok(Some(ValueFact::Scalar(ScalarFact::Int {
            expression,
            exact: None,
        })))
    }
}

// Structural operations preserve symbolic and unknown fields by asking the
// native value algebra to route exact positions in the original aggregate.
fn structural_aggregate_fact(
    operation: &wire::OperationProto,
    environment: &HashMap<u64, Option<ValueFact>>,
    exact_environment: &HashMap<u64, Option<StaticValue>>,
) -> Option<ValueFact> {
    let ValueFact::Aggregate(values) = environment.get(operation.operands.first()?)?.as_ref()?
    else {
        return None;
    };
    let mut route_environment = exact_environment.clone();
    let mut route_operation = operation.clone();
    let route = (0..values.len())
        .map(|index| {
            i64::try_from(index)
                .ok()
                .map(|index| Some(StaticValue::Scalar(StaticScalar::Int(index))))
        })
        .collect::<Option<Vec<_>>>()?;
    route_environment.insert(
        operation.operands[0],
        Some(StaticValue::Aggregate(route.into())),
    );
    if operation.opcode == "classical.update" {
        let replacement_id = operation.results.first()?.id;
        *route_operation.operands.get_mut(2)? = replacement_id;
        route_environment.insert(
            replacement_id,
            Some(StaticValue::Scalar(StaticScalar::Int(-1))),
        );
    }
    let StaticValue::Aggregate(route) =
        evaluate_static_value(&route_operation, &route_environment)?
    else {
        return None;
    };
    let projected = route
        .iter()
        .map(
            |position| match position.as_ref().and_then(StaticValue::scalar) {
                Some(StaticScalar::Int(-1)) => operation
                    .operands
                    .get(2)
                    .and_then(|id| environment.get(id))
                    .cloned()
                    .flatten(),
                Some(StaticScalar::Int(index)) => usize::try_from(index)
                    .ok()
                    .and_then(|index| values.get(index))
                    .cloned()
                    .flatten(),
                _ => None,
            },
        )
        .collect::<Vec<_>>();
    Some(ValueFact::Aggregate(projected.into_boxed_slice()))
}
