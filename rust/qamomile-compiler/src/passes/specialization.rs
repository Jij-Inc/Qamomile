//! Native specialization of a selected, fully bound `QCore` entrypoint.
//!
//! The input is an internal specialization request rather than an executable
//! program. This pass substitutes dependent entrypoint dimensions, evaluates
//! exact scalar expressions, and erases selected static branches while
//! preserving canonical SSA identities. Its output must cross the ordinary
//! locally checked program boundary before any later cleanup or target pass.

use std::collections::HashMap;
use std::sync::Arc;

use thiserror::Error;

use crate::numeric_values::NumericEvaluationError;
#[cfg(test)]
use crate::static_values::StaticScalar;
use crate::static_values::{self, StaticValue};
use crate::types::{Dimension, SemanticType, TypeArena, TypeId};
use crate::wire;

/// Failure of native Const specialization.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub(crate) enum SpecializationError {
    /// A bound dimension is not a strictly positive machine-sized extent.
    #[error("Q3007: shape dimension {dimension:?} must be positive")]
    InvalidDimension { dimension: String },
    /// Const bounds select an empty, invalid, or excessive fixed slice.
    #[error("Q3009: invalid Const slice shape: {reason}")]
    InvalidSliceShape { reason: String },
    /// A previously verified semantic type cannot enter the shared type arena.
    #[error("Q4002: verified type could not be specialized: {reason}")]
    MalformedType {
        /// Internal type reconstruction or encoding failure.
        reason: String,
    },
    /// A selected static branch does not satisfy the verified region ABI.
    #[error("Q1101: static-if capture arity mismatch")]
    BranchCaptureArity,
    /// A selected branch yields a different result arity from its operation.
    #[error("Q1102: static-if result arity mismatch")]
    BranchResultArity,
    /// A selected conditional has no canonical then/else region.
    #[error("Q1034: malformed qcore.if operation")]
    MalformedBranch,
    /// A selected allocation has a non-positive exact width.
    #[error("Q1046: allocation width must be positive")]
    InvalidAllocationWidth,
    /// An exact classical expression is undefined on the selected path.
    #[error("Q3008: cannot specialize {opcode}: {reason}")]
    InvalidEvaluation {
        /// Classical operation being evaluated.
        opcode: String,
        /// Stable arithmetic-domain explanation.
        reason: &'static str,
    },
    /// A known portable value lies outside the migrated native evaluator.
    #[error("Q6202: native specialization does not yet evaluate {opcode}")]
    UnsupportedEvaluation {
        /// Classical operation requiring the compatibility evaluator.
        opcode: String,
    },
}

/// Diagnostic work charged by native Const specialization.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SpecializationWork {
    operation_visits: u128,
}

impl SpecializationWork {
    /// Return the number of `QCore` operations visited by specialization.
    ///
    /// This value describes compiler work and is not a logical resource metric.
    #[must_use]
    pub const fn operation_visits(self) -> u128 {
        self.operation_visits
    }

    fn charge_operation(&mut self) {
        self.operation_visits = self.operation_visits.saturating_add(1);
    }
}

/// Specialize one already validated internal request.
///
/// # Errors
///
/// Returns [`SpecializationError`] if a dependent extent is invalid or a
/// supposedly verified static branch cannot be rewritten with its declared
/// region ABI.
pub(crate) fn specialize_with_work(
    package: &wire::QCorePackageProto,
) -> Result<(wire::QCorePackageProto, SpecializationWork), SpecializationError> {
    let mut work = SpecializationWork::default();
    let mut specialized = package.clone();
    let dimensions = integer_bindings(&specialized.const_bindings);
    let Some(entry_index) = specialized
        .functions
        .iter()
        .position(|function| function.symbol == specialized.entrypoint)
    else {
        return Err(SpecializationError::MalformedBranch);
    };
    specialize_function_types(
        &mut specialized.functions[entry_index],
        &dimensions,
        &mut work,
    )?;

    for (index, function) in specialized.functions.iter_mut().enumerate() {
        let mut known = HashMap::new();
        if index == entry_index {
            for parameter in &function.parameters {
                if let Some(binding) = specialized
                    .const_bindings
                    .iter()
                    .find(|binding| binding.name == parameter.name)
                {
                    let fact = binding.value.as_ref().and_then(|value| {
                        parameter
                            .value
                            .as_ref()
                            .and_then(|parameter| parameter.r#type.as_ref())
                            .and_then(|type_spec| {
                                static_values::typed_value_literal(value, type_spec)
                            })
                    });
                    known.insert(parameter.value.as_ref().map_or(0, |value| value.id), fact);
                }
            }
        }
        if let Some(body) = function.body.as_ref() {
            function.body = Some(rewrite_block_with_work(
                body,
                &mut known,
                &HashMap::new(),
                false,
                false,
                &mut work,
            )?);
        }
    }
    Ok((specialized, work))
}

fn integer_bindings(bindings: &[wire::BindingProto]) -> HashMap<String, i64> {
    bindings
        .iter()
        .filter_map(|binding| {
            let value = binding.value.as_ref().and_then(static_values::literal)?;
            static_values::integer(value).map(|value| (binding.name.clone(), value))
        })
        .collect()
}

fn specialize_function_types(
    function: &mut wire::FunctionProto,
    dimensions: &HashMap<String, i64>,
    work: &mut SpecializationWork,
) -> Result<(), SpecializationError> {
    let mut specializer = TypeSpecializer::new(&function.symbol, dimensions);
    if let Some(result) = function.result_type.as_mut() {
        specializer.apply(result)?;
    }
    for parameter in &mut function.parameters {
        if let Some(value) = parameter.value.as_mut() {
            specialize_value(value, &mut specializer)?;
        }
    }
    if let Some(body) = function.body.as_mut() {
        specialize_block_types(body, &mut specializer, work)?;
    }
    Ok(())
}

fn specialize_block_types(
    block: &mut wire::BlockProto,
    specializer: &mut TypeSpecializer<'_>,
    work: &mut SpecializationWork,
) -> Result<(), SpecializationError> {
    for argument in &mut block.arguments {
        specialize_value(argument, specializer)?;
    }
    for operation in &mut block.operations {
        work.charge_operation();
        for result in &mut operation.results {
            specialize_value(result, specializer)?;
        }
        specializer.fold_shape_length(operation)?;
        for region in &mut operation.regions {
            for nested in &mut region.blocks {
                specialize_block_types(nested, specializer, work)?;
            }
        }
    }
    Ok(())
}

fn specialize_value(
    value: &mut wire::ValueProto,
    specializer: &mut TypeSpecializer<'_>,
) -> Result<(), SpecializationError> {
    if let Some(type_spec) = value.r#type.as_mut() {
        let type_id = specializer.apply(type_spec)?;
        specializer.value_types.insert(value.id, type_id);
    }
    Ok(())
}

struct TypeSpecializer<'a> {
    arena: TypeArena,
    scope: Arc<str>,
    dimensions: &'a HashMap<String, i64>,
    substitutions: HashMap<Dimension, Dimension>,
    memo: HashMap<TypeId, TypeId>,
    value_types: HashMap<u64, TypeId>,
}

impl<'a> TypeSpecializer<'a> {
    fn new(scope: &str, dimensions: &'a HashMap<String, i64>) -> Self {
        let mut arena = TypeArena::default();
        let scope = arena.intern_scope(scope);
        Self {
            arena,
            scope,
            dimensions,
            substitutions: HashMap::new(),
            memo: HashMap::new(),
            value_types: HashMap::new(),
        }
    }

    fn apply(&mut self, type_spec: &mut wire::TypeProto) -> Result<TypeId, SpecializationError> {
        let root = self
            .arena
            .resolve_wire_type(type_spec, &self.scope)
            .map_err(|reason| SpecializationError::MalformedType { reason })?;
        for dimension in self.arena.dimensions_in([root]) {
            let Dimension::Symbol { name, .. } = &dimension else {
                continue;
            };
            let fixed = if name.starts_with("slice(") {
                crate::core::slice_shape::SliceShape::parse(name)
                    .and_then(|shape| shape.evaluate(&|name| self.dimensions.get(name).copied()))
                    .map_err(|reason| SpecializationError::InvalidSliceShape { reason })?
            } else {
                let value = crate::types::PolynomialDimension::parse(name)
                    .map_err(|reason| SpecializationError::MalformedType { reason })?
                    .evaluate_signed(&|name| self.dimensions.get(name).copied());
                match value {
                    Some(value) => Some(u64::try_from(value).ok().filter(|n| *n > 0).ok_or_else(
                        || SpecializationError::InvalidDimension {
                            dimension: name.to_string(),
                        },
                    )?),
                    None => None,
                }
            };
            if let Some(fixed) = fixed {
                self.substitutions
                    .insert(dimension, Dimension::Fixed(fixed));
            }
        }
        let substituted = self
            .arena
            .substitute_dimensions_cached(root, &self.substitutions, &mut self.memo)
            .map_err(|reason| SpecializationError::MalformedType { reason })?;
        *type_spec = self
            .arena
            .encode_wire_type(substituted)
            .map_err(|reason| SpecializationError::MalformedType { reason })?;
        Ok(substituted)
    }

    /// Resolve a shape-only length from its type without materializing elements.
    fn fold_shape_length(
        &self,
        operation: &mut wire::OperationProto,
    ) -> Result<(), SpecializationError> {
        if operation.opcode != "classical.len"
            || !operation.attributes.iter().any(|attribute| {
                attribute.name == "shape_only"
                    && matches!(
                        attribute
                            .value
                            .as_ref()
                            .and_then(|value| value.value.as_ref()),
                        Some(wire::literal_proto::Value::BoolValue(true))
                    )
            })
        {
            return Ok(());
        }
        let Some(type_id) = operation
            .operands
            .first()
            .and_then(|operand| self.value_types.get(operand))
        else {
            return Err(SpecializationError::MalformedType {
                reason: "shape-only length operand has no specialized type".to_owned(),
            });
        };
        let base = self.arena.strip_stage(*type_id);
        let extent = match self.arena.get(base) {
            Some(SemanticType::Array { shape, .. }) => match shape.first() {
                Some(Dimension::Fixed(extent)) => Some(*extent),
                _ => None,
            },
            Some(SemanticType::Word {
                width: Dimension::Fixed(extent),
                ..
            }) => Some(*extent),
            _ => self
                .arena
                .aggregate_len(base)
                .and_then(|n| u64::try_from(n).ok()),
        };
        let Some(extent) = extent else {
            return Ok(());
        };
        let extent = i64::try_from(extent).map_err(|_| SpecializationError::InvalidEvaluation {
            opcode: operation.opcode.clone(),
            reason: "iteration extent is not representable",
        })?;
        "qcore.constant".clone_into(&mut operation.opcode);
        operation.operands.clear();
        operation.attributes = vec![wire::AttributeProto {
            name: "value".to_owned(),
            value: Some(wire::LiteralProto {
                value: Some(wire::literal_proto::Value::IntValue(extent)),
            }),
        }];
        Ok(())
    }
}

#[cfg(test)]
fn rewrite_block(
    source: &wire::BlockProto,
    known: &mut HashMap<u64, Option<StaticValue>>,
    initial_remap: &HashMap<u64, u64>,
    inline: bool,
    defer_partial: bool,
) -> Result<wire::BlockProto, SpecializationError> {
    rewrite_block_with_work(
        source,
        known,
        initial_remap,
        inline,
        defer_partial,
        &mut SpecializationWork::default(),
    )
}

fn rewrite_block_with_work(
    source: &wire::BlockProto,
    known: &mut HashMap<u64, Option<StaticValue>>,
    initial_remap: &HashMap<u64, u64>,
    inline: bool,
    defer_partial: bool,
    work: &mut SpecializationWork,
) -> Result<wire::BlockProto, SpecializationError> {
    rewrite_block_with_ids(source, known, initial_remap, inline, defer_partial, work)
}
#[allow(clippy::too_many_lines)]
fn rewrite_block_with_ids(
    source: &wire::BlockProto,
    known: &mut HashMap<u64, Option<StaticValue>>,
    initial_remap: &HashMap<u64, u64>,
    inline: bool,
    defer_partial: bool,
    work: &mut SpecializationWork,
) -> Result<wire::BlockProto, SpecializationError> {
    let mut remap = initial_remap.clone();
    let mut operations = Vec::new();
    let mut terminal_override = None;

    for source_operation in &source.operations {
        work.charge_operation();
        let mut operation = source_operation.clone();
        operation.operands = operation
            .operands
            .iter()
            .map(|operand| resolve(*operand, &remap))
            .collect();

        if operation.opcode == "qcore.constant" && operation.results.len() == 1 {
            let fact = literal_attribute(&operation, "value").and_then(|value| {
                operation.results[0]
                    .r#type
                    .as_ref()
                    .and_then(|type_spec| static_values::typed_value_literal(value, type_spec))
            });
            known.insert(operation.results[0].id, fact);
            operations.push(operation);
            continue;
        }

        if operation.results.len() == 1
            && matches!(operation.opcode.as_str(), "tuple" | "array" | "qcore.tuple")
        {
            let values = operation
                .operands
                .iter()
                .map(|operand| known.get(operand).cloned().flatten())
                .collect::<Vec<_>>();
            let value = static_values::structural_value(operation.results.first(), values);
            known.insert(operation.results[0].id, Some(value));
            operations.push(operation);
            continue;
        }

        if operation.opcode == "qcore.stage_lift" && operation.results.len() == 1 {
            if let Some(fact) = operation
                .operands
                .first()
                .and_then(|operand| known.get(operand))
                .cloned()
            {
                known.insert(operation.results[0].id, fact);
            }
            operations.push(operation);
            continue;
        }

        if operation.opcode == "qcore.unpack" {
            bind_unpack_facts(&operation, known);
            operations.push(operation);
            continue;
        }

        if operation.opcode == "classical.slice" {
            let mut operand = operation.operands.iter().skip(1);
            for name in ["start", "stop", "step"] {
                let attribute = operation
                    .attributes
                    .iter_mut()
                    .find(|a| a.name == name)
                    .expect("verified slice attribute");
                if matches!(
                    attribute.value.as_ref().and_then(|v| v.value.as_ref()),
                    Some(wire::literal_proto::Value::UnitValue(_))
                ) {
                    continue;
                }
                let id = operand.next().expect("verified slice operand");
                if let Some(value) = known
                    .get(id)
                    .and_then(Option::as_ref)
                    .and_then(static_values::value_to_literal)
                {
                    attribute.value = Some(value);
                }
            }
        }
        if operation.opcode.starts_with("classical.") && operation.results.len() == 1 {
            let all_operands_accounted = operation
                .operands
                .iter()
                .all(|operand| known.contains_key(operand));
            let has_unsupported_operand = operation
                .operands
                .iter()
                .any(|operand| known.get(operand).is_some_and(Option::is_none));
            let all_operands_complete = operation.operands.iter().all(|operand| {
                known
                    .get(operand)
                    .and_then(Option::as_ref)
                    .is_some_and(StaticValue::is_complete)
            });
            if let Some(reason) = static_values::known_domain_error(&operation, known)
                && !defer_partial
                // Whole-module specialization cannot assume that an uncalled
                // function with unknown arguments will execute. Partial numeric
                // domain diagnostics belong to the selected-path compiler pass.
                && (all_operands_complete
                    || matches!(operation.opcode.as_str(), "classical.index" | "classical.update"))
            {
                return Err(SpecializationError::InvalidEvaluation {
                    opcode: operation.opcode,
                    reason,
                });
            }
            let folded = match static_values::evaluate_value_checked(&operation, known) {
                Ok(value) => value,
                Err(
                    NumericEvaluationError::Undefined(reason)
                    | NumericEvaluationError::Limit(reason),
                ) if !defer_partial => {
                    return Err(SpecializationError::InvalidEvaluation {
                        opcode: operation.opcode,
                        reason,
                    });
                }
                Err(_) => None,
            };
            if let Some(folded) = folded {
                let literal = static_values::value_to_literal(&folded);
                known.insert(operation.results[0].id, Some(folded));
                if let Some(literal) = literal {
                    "qcore.constant".clone_into(&mut operation.opcode);
                    operation.operands.clear();
                    operation.regions.clear();
                    operation.attributes = vec![wire::AttributeProto {
                        name: "value".to_owned(),
                        value: Some(literal),
                    }];
                }
                operations.push(operation);
                continue;
            }
            if all_operands_accounted
                && (has_unsupported_operand || all_operands_complete)
                && !defer_partial
            {
                return Err(SpecializationError::UnsupportedEvaluation {
                    opcode: operation.opcode,
                });
            }
        }

        if operation.opcode == "quantum.alloc"
            && operation
                .operands
                .first()
                .and_then(|operand| known.get(operand))
                .and_then(Option::as_ref)
                .and_then(StaticValue::scalar)
                .and_then(static_values::integer)
                .is_some_and(|width| width <= 0)
        {
            return Err(SpecializationError::InvalidAllocationWidth);
        }

        let selected_branch = (operation.opcode == "qcore.if")
            .then(|| operation.operands.first().copied())
            .flatten()
            .and_then(|condition| known.get(&condition))
            .and_then(Option::as_ref)
            .and_then(StaticValue::scalar)
            .map(static_values::truth);
        if let Some(condition) = selected_branch {
            let branch_index = usize::from(!condition);
            let region = operation
                .regions
                .get(branch_index)
                .and_then(|region| region.blocks.first())
                .ok_or(SpecializationError::MalformedBranch)?;
            let captures = operation.operands.get(1..).unwrap_or_default();
            if region.arguments.len() != captures.len() {
                return Err(SpecializationError::BranchCaptureArity);
            }
            let nested_remap = region
                .arguments
                .iter()
                .zip(captures)
                .map(|(argument, value)| (argument.id, *value))
                .collect::<HashMap<_, _>>();
            for (argument, value) in region.arguments.iter().zip(captures) {
                if let Some(fact) = known.get(value).cloned() {
                    known.insert(argument.id, fact);
                }
            }
            let rewritten =
                rewrite_block_with_ids(region, known, &nested_remap, true, defer_partial, work)?;
            operations.extend(rewritten.operations);
            let terminator = rewritten
                .terminator
                .ok_or(SpecializationError::MalformedBranch)?;
            if terminator.kind == "yield" {
                if terminator.operands.len() != operation.results.len() {
                    return Err(SpecializationError::BranchResultArity);
                }
                for (result, yielded) in operation.results.iter().zip(&terminator.operands) {
                    remap.insert(result.id, *yielded);
                    if let Some(fact) = known.get(yielded).cloned() {
                        known.insert(result.id, fact);
                    }
                }
                continue;
            }
            terminal_override = Some(terminator);
            break;
        }

        for region in &mut operation.regions {
            for nested in &mut region.blocks {
                *nested =
                    rewrite_block_with_ids(nested, known, &HashMap::new(), false, true, work)?;
            }
        }
        operations.push(operation);
    }

    let mut terminator = terminal_override.or_else(|| source.terminator.clone());
    if let Some(terminator) = terminator.as_mut() {
        terminator.operands = terminator
            .operands
            .iter()
            .map(|operand| resolve(*operand, &remap))
            .collect();
    }
    Ok(wire::BlockProto {
        arguments: if inline {
            Vec::new()
        } else {
            source.arguments.clone()
        },
        operations,
        terminator,
    })
}

fn bind_unpack_facts(
    operation: &wire::OperationProto,
    known: &mut HashMap<u64, Option<StaticValue>>,
) {
    let Some(source) = operation.operands.first() else {
        return;
    };
    let Some(fact) = known.get(source).cloned() else {
        return;
    };
    match fact {
        Some(value) => {
            if let Some(values) = static_values::structural_components(&value)
                .filter(|values| values.len() == operation.results.len())
            {
                for (result, value) in operation.results.iter().zip(values.iter()) {
                    if let Some(value) = value {
                        known.insert(result.id, Some(value.clone()));
                    }
                }
            }
        }
        None => {
            for result in &operation.results {
                known.insert(result.id, None);
            }
        }
    }
}

fn resolve(mut value: u64, remap: &HashMap<u64, u64>) -> u64 {
    while let Some(next) = remap.get(&value).copied() {
        if next == value {
            break;
        }
        value = next;
    }
    value
}

fn literal_attribute<'a>(
    operation: &'a wire::OperationProto,
    name: &str,
) -> Option<&'a wire::LiteralProto> {
    operation
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .and_then(|attribute| attribute.value.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integer(value: i64) -> wire::LiteralProto {
        wire::LiteralProto {
            value: Some(wire::literal_proto::Value::IntValue(value)),
        }
    }

    fn boolean(value: bool) -> wire::LiteralProto {
        wire::LiteralProto {
            value: Some(wire::literal_proto::Value::BoolValue(value)),
        }
    }

    #[test]
    fn substitutes_positive_bound_dimensions_recursively() {
        let mut type_spec = wire::TypeProto {
            kind: wire::type_proto::Kind::Quantum.into(),
            arguments: vec![wire::TypeProto {
                kind: wire::type_proto::Kind::Array.into(),
                name: "Bits".to_owned(),
                arguments: vec![wire::TypeProto {
                    kind: wire::type_proto::Kind::Scalar.into(),
                    name: "Bit".to_owned(),
                    ..Default::default()
                }],
                dimensions: vec![wire::DimensionProto {
                    value: Some(wire::dimension_proto::Value::Symbol("width".to_owned())),
                }],
                ..Default::default()
            }],
            ..Default::default()
        };
        let dimensions = HashMap::from([("width".to_owned(), 5)]);
        TypeSpecializer::new("test", &dimensions)
            .apply(&mut type_spec)
            .expect("positive extent");
        assert_eq!(
            type_spec.arguments[0].dimensions[0].value,
            Some(wire::dimension_proto::Value::Fixed(5))
        );

        let mut invalid = type_spec.clone();
        invalid.arguments[0].dimensions[0].value =
            Some(wire::dimension_proto::Value::Symbol("width".to_owned()));
        let dimensions = HashMap::from([("width".to_owned(), 0)]);
        assert_eq!(
            TypeSpecializer::new("test", &dimensions).apply(&mut invalid),
            Err(SpecializationError::InvalidDimension {
                dimension: "width".to_owned(),
            })
        );

        let mut unbound = invalid;
        let unrelated = HashMap::from([("other".to_owned(), -1)]);
        TypeSpecializer::new("test", &unrelated)
            .apply(&mut unbound)
            .expect("an unrelated integer is not a shape binding");
        assert_eq!(
            unbound.arguments[0].dimensions[0].value,
            Some(wire::dimension_proto::Value::Symbol("width".to_owned()))
        );
    }

    #[test]
    fn folds_shape_only_length_from_specialized_type_without_array_values() {
        let mut source = wire::ValueProto {
            id: 1,
            r#type: Some(wire::TypeProto {
                kind: wire::type_proto::Kind::Staged.into(),
                name: "Param".to_owned(),
                arguments: vec![wire::TypeProto {
                    kind: wire::type_proto::Kind::Array.into(),
                    name: "Array".to_owned(),
                    arguments: vec![wire::TypeProto {
                        kind: wire::type_proto::Kind::Scalar.into(),
                        name: "Real".to_owned(),
                        ..Default::default()
                    }],
                    dimensions: vec![
                        wire::DimensionProto {
                            value: Some(wire::dimension_proto::Value::Symbol(
                                "slice(5,start,stop,step)".to_owned(),
                            )),
                        },
                        wire::DimensionProto {
                            value: Some(wire::dimension_proto::Value::Fixed(1_000_000_000)),
                        },
                    ],
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let dimensions = HashMap::from([
            ("start".to_owned(), -3),
            ("stop".to_owned(), 5),
            ("step".to_owned(), 1),
        ]);
        let mut specializer = TypeSpecializer::new("test", &dimensions);
        specialize_value(&mut source, &mut specializer).expect("specialized array type");
        let mut length = wire::OperationProto {
            opcode: "classical.len".to_owned(),
            operands: vec![source.id],
            attributes: vec![wire::AttributeProto {
                name: "shape_only".to_owned(),
                value: Some(boolean(true)),
            }],
            ..Default::default()
        };
        specializer
            .fold_shape_length(&mut length)
            .expect("known type extent");
        assert_eq!(length.opcode, "qcore.constant");
        assert!(length.operands.is_empty());
        assert_eq!(length.attributes[0].name, "value");
        assert_eq!(length.attributes[0].value, Some(integer(3)));
    }

    #[test]
    fn folds_exact_scalar_operations_to_portable_constants() {
        let result = wire::ValueProto {
            id: 3,
            ..Default::default()
        };
        let source = wire::BlockProto {
            operations: vec![wire::OperationProto {
                opcode: "classical.add".to_owned(),
                operands: vec![1, 2],
                results: vec![result],
                ..Default::default()
            }],
            terminator: Some(wire::TerminatorProto {
                kind: "return".to_owned(),
                operands: vec![3],
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut known = HashMap::from([
            (1, Some(StaticValue::Scalar(StaticScalar::Int(2)))),
            (2, Some(StaticValue::Scalar(StaticScalar::Int(3)))),
        ]);
        let rewritten = rewrite_block(&source, &mut known, &HashMap::new(), false, false)
            .expect("exact addition");
        assert_eq!(rewritten.operations[0].opcode, "qcore.constant");
        assert_eq!(
            rewritten.operations[0].attributes[0].value,
            Some(integer(5))
        );
    }

    #[test]
    fn erases_a_selected_static_branch_and_remaps_its_yield() {
        let branch = |value| wire::RegionProto {
            blocks: vec![wire::BlockProto {
                operations: vec![wire::OperationProto {
                    opcode: "qcore.constant".to_owned(),
                    results: vec![wire::ValueProto {
                        id: value,
                        ..Default::default()
                    }],
                    attributes: vec![wire::AttributeProto {
                        name: "value".to_owned(),
                        value: Some(integer(i64::try_from(value).expect("small test value"))),
                    }],
                    ..Default::default()
                }],
                terminator: Some(wire::TerminatorProto {
                    kind: "yield".to_owned(),
                    operands: vec![value],
                    ..Default::default()
                }),
                ..Default::default()
            }],
        };
        let source = wire::BlockProto {
            operations: vec![wire::OperationProto {
                opcode: "qcore.if".to_owned(),
                operands: vec![1],
                results: vec![wire::ValueProto {
                    id: 4,
                    ..Default::default()
                }],
                regions: vec![branch(2), branch(3)],
                ..Default::default()
            }],
            terminator: Some(wire::TerminatorProto {
                kind: "return".to_owned(),
                operands: vec![4],
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut known = HashMap::from([(1, static_values::value_literal(&boolean(false)))]);
        let rewritten = rewrite_block(&source, &mut known, &HashMap::new(), false, false)
            .expect("selected else branch");
        assert_eq!(rewritten.operations.len(), 1);
        assert_eq!(rewritten.operations[0].results[0].id, 3);
        assert_eq!(rewritten.terminator.expect("return").operands, vec![3]);
    }

    #[test]
    fn folds_exact_aggregate_index_with_python_negative_semantics() {
        let source = wire::BlockProto {
            operations: vec![wire::OperationProto {
                opcode: "classical.index".to_owned(),
                operands: vec![1, 2],
                results: vec![wire::ValueProto {
                    id: 3,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            terminator: Some(wire::TerminatorProto {
                kind: "return".to_owned(),
                operands: vec![3],
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut known = HashMap::from([
            (
                1,
                Some(StaticValue::Aggregate(
                    vec![
                        Some(StaticValue::Scalar(StaticScalar::Int(7))),
                        Some(StaticValue::Scalar(StaticScalar::Int(9))),
                    ]
                    .into(),
                )),
            ),
            (2, Some(StaticValue::Scalar(StaticScalar::Int(-1)))),
        ]);

        let rewritten = rewrite_block(&source, &mut known, &HashMap::new(), false, false)
            .expect("exact aggregate index");
        assert_eq!(rewritten.operations[0].opcode, "qcore.constant");
        assert_eq!(
            rewritten.operations[0].attributes[0].value,
            Some(integer(9))
        );
    }

    #[test]
    fn folds_exact_complex_arithmetic_without_compatibility() {
        let source = wire::BlockProto {
            operations: vec![wire::OperationProto {
                opcode: "classical.add".to_owned(),
                operands: vec![1, 2],
                results: vec![wire::ValueProto {
                    id: 3,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            terminator: Some(wire::TerminatorProto {
                kind: "return".to_owned(),
                operands: vec![3],
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut known = HashMap::from([
            (1, Some(StaticValue::ComplexBits(1.0_f64.to_bits(), 0))),
            (2, Some(StaticValue::ComplexBits(2.0_f64.to_bits(), 0))),
        ]);

        let rewritten = rewrite_block(&source, &mut known, &HashMap::new(), false, false)
            .expect("exact complex addition");
        assert_eq!(rewritten.operations[0].opcode, "qcore.constant");
        assert_eq!(
            rewritten.operations[0].attributes[0].value,
            Some(wire::LiteralProto {
                value: Some(wire::literal_proto::Value::ComplexValue(
                    wire::literal_proto::Complex {
                        real: 3.0,
                        imaginary: 0.0,
                    },
                )),
            })
        );
    }

    #[test]
    fn rejects_an_exact_out_of_bounds_aggregate_index_natively() {
        let source = wire::BlockProto {
            operations: vec![wire::OperationProto {
                opcode: "classical.index".to_owned(),
                operands: vec![1, 2],
                results: vec![wire::ValueProto {
                    id: 3,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            terminator: Some(wire::TerminatorProto {
                kind: "return".to_owned(),
                operands: vec![3],
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut known = HashMap::from([
            (
                1,
                Some(StaticValue::Aggregate(
                    vec![Some(StaticValue::Scalar(StaticScalar::Int(7)))].into(),
                )),
            ),
            (2, Some(StaticValue::Scalar(StaticScalar::Int(1)))),
        ]);

        assert_eq!(
            rewrite_block(&source, &mut known, &HashMap::new(), false, false),
            Err(SpecializationError::InvalidEvaluation {
                opcode: "classical.index".to_owned(),
                reason: "index out of bounds",
            })
        );
    }
}
