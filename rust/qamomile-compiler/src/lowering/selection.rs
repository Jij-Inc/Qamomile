//! Native entrypoint selection, Const materialization, and static linking.
//!
//! A selection request uses the canonical specialization envelope but still
//! contains the complete verified source module, a local or qualified source
//! selector, and only the bindings supplied by the caller. This pass resolves
//! that source-facing request into the fully bound, transitively linked input
//! accepted by the independent specialization boundary.
//! Typed maps share the package byte, node, and nesting budgets enforced before
//! this pass; selection checks key uniqueness without a separate entry-count cap.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use num_bigint::BigInt;
use prost::Message;
use thiserror::Error;

use crate::call_graph::StaticCallGraph;
use crate::equations::literal_conforms;
use crate::operations::LocallyCheckedOperations;
use crate::types::{SemanticType, Stage, TypeArena, TypeId};
use crate::wire;

/// Failure to resolve one source-facing specialization selection.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub(crate) enum SelectionError {
    /// A local or qualified selector is absent or ambiguous.
    #[error("Q3001: expected one function named '{selector}', found {count}")]
    AmbiguousEntrypoint { selector: String, count: usize },
    /// A source selector exceeds the existing canonical string budget.
    #[error("Q3001: function selector exceeds the {maximum}-byte limit: got {actual}")]
    EntrypointTooLarge { actual: usize, maximum: usize },
    /// Required specialization parameters remain unbound after inference.
    #[error("Q3002: missing Const bindings: {parameters}")]
    MissingBindings { parameters: String },
    /// Supplied names do not denote Const parameters of the selected function.
    #[error("Q3003: unknown Const bindings: {bindings}")]
    UnknownBindings { bindings: String },
    /// One supplied or defaulted value is outside its declared type.
    #[error("Q3006: Const binding '{binding}' expects {expected}")]
    InvalidBinding { binding: String, expected: String },
    /// A typed mapping literal contains two structurally equal keys.
    #[error("Q3007: Const mapping '{binding}' contains a duplicate key")]
    DuplicateMappingKey { binding: String },
    /// Aggregate values imply incompatible values for one dependent extent.
    #[error("Q3006: Const dimension '{dimension}' has conflicting extents")]
    ConflictingExtents { dimension: String },
    /// An explicit dimension binding disagrees with an inferred extent.
    #[error("Q3006: Const dimension '{dimension}' expects {expected}, got an incompatible value")]
    ExplicitExtentConflict { dimension: String, expected: usize },
    /// Retained native facts unexpectedly omit a verified type root.
    #[error("Q4002: missing retained declaration type roots for {owner:?}")]
    MissingTypeRoots { owner: String },
    /// The retained closed call graph unexpectedly omits a selected symbol.
    #[error("Q3004: unresolved callable '{symbol}' while linking '{entrypoint}'")]
    UnresolvedCallable { symbol: String, entrypoint: String },
}

/// Resolve, bind, and link one already verified complete source module.
///
/// # Errors
///
/// Returns [`SelectionError`] when entrypoint resolution is ambiguous, Const
/// bindings are incomplete or ill-typed, dependent extents conflict, or the
/// retained closed graph cannot produce the selected transitive closure.
pub(crate) fn materialize(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
    call_graph: &StaticCallGraph,
) -> Result<wire::QCorePackageProto, SelectionError> {
    let (entry_index, entry) = resolve_entrypoint(package, &package.entrypoint)?;
    let values =
        operations
            .value_types(&entry.symbol)
            .ok_or_else(|| SelectionError::MissingTypeRoots {
                owner: entry.symbol.clone(),
            })?;
    let const_parameters = entry
        .parameters
        .iter()
        .filter_map(|parameter| {
            let type_id = parameter
                .value
                .as_ref()
                .and_then(|value| values.get(&value.id))
                .copied()?;
            (arena.stage_of(type_id) == Some(Stage::Const))
                .then_some((parameter.name.as_str(), parameter))
        })
        .collect::<BTreeMap<_, _>>();
    let mut provided = package
        .const_bindings
        .iter()
        .map(|binding| (binding.name.clone(), binding.value.clone()))
        .collect::<BTreeMap<_, _>>();

    infer_extents(&const_parameters, &mut provided)?;

    let missing = const_parameters
        .iter()
        .filter(|(name, parameter)| !parameter.has_default && !provided.contains_key(**name))
        .map(|(name, _)| (*name).to_owned())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(SelectionError::MissingBindings {
            parameters: missing.join(", "),
        });
    }
    let unknown = provided
        .keys()
        .filter(|name| !const_parameters.contains_key(name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        return Err(SelectionError::UnknownBindings {
            bindings: unknown.join(", "),
        });
    }

    let mut resolved = const_parameters
        .iter()
        .filter(|(_, parameter)| parameter.has_default)
        .map(|(name, parameter)| ((*name).to_owned(), parameter.default_value.clone()))
        .collect::<BTreeMap<_, _>>();
    resolved.extend(provided);
    verify_resolved_bindings(&resolved, &const_parameters, values, arena, &entry.symbol)?;

    let reached = reachable(entry_index, call_graph, &entry.symbol)?;
    let mut selected = package.clone();
    selected.entrypoint.clone_from(&entry.symbol);
    selected.const_bindings = resolved
        .into_iter()
        .map(|(name, value)| wire::BindingProto { name, value })
        .collect();
    selected.functions = selected
        .functions
        .into_iter()
        .enumerate()
        .filter_map(|(index, function)| reached.contains(&index).then_some(function))
        .collect();
    Ok(selected)
}

/// Resolve and link one verified symbolic program family without binding Consts.
///
/// # Errors
///
/// Returns [`SelectionError`] when the source selector is absent or ambiguous,
/// or when the retained closed graph cannot produce its transitive closure.
pub(crate) fn select_family(
    package: &wire::QCorePackageProto,
    call_graph: &StaticCallGraph,
) -> Result<wire::QCorePackageProto, SelectionError> {
    let (entry_index, entry) = resolve_entrypoint(package, &package.entrypoint)?;
    let reached = reachable(entry_index, call_graph, &entry.symbol)?;
    let mut selected = package.clone();
    selected.entrypoint.clone_from(&entry.symbol);
    selected.const_bindings.clear();
    selected.functions = selected
        .functions
        .into_iter()
        .enumerate()
        .filter_map(|(index, function)| reached.contains(&index).then_some(function))
        .collect();
    Ok(selected)
}

pub(crate) fn resolve_entrypoint<'a>(
    package: &'a wire::QCorePackageProto,
    selector: &str,
) -> Result<(usize, &'a wire::FunctionProto), SelectionError> {
    if selector.len() > crate::budget::MAX_STRING_BYTES {
        return Err(SelectionError::EntrypointTooLarge {
            actual: selector.len(),
            maximum: crate::budget::MAX_STRING_BYTES,
        });
    }
    let mut matched = None;
    let mut count = 0;
    for (index, function) in package.functions.iter().enumerate() {
        let local = function
            .symbol
            .rsplit_once('.')
            .map_or(function.symbol.as_str(), |(_, local)| local);
        if function.symbol != selector && local != selector {
            continue;
        }
        count += 1;
        if matched.is_none() {
            matched = Some((index, function));
        }
    }
    let Some((entry_index, entry)) = matched.filter(|_| count == 1) else {
        return Err(SelectionError::AmbiguousEntrypoint {
            selector: selector.to_owned(),
            count,
        });
    };
    Ok((entry_index, entry))
}

fn verify_resolved_bindings(
    resolved: &BTreeMap<String, Option<wire::LiteralProto>>,
    const_parameters: &BTreeMap<&str, &wire::ParameterProto>,
    values: &HashMap<u64, crate::types::TypeId>,
    arena: &TypeArena,
    entrypoint: &str,
) -> Result<(), SelectionError> {
    for (name, value) in resolved {
        let parameter = const_parameters
            .get(name.as_str())
            .expect("unknown binding was rejected above");
        let type_id = parameter
            .value
            .as_ref()
            .and_then(|value| values.get(&value.id))
            .copied()
            .ok_or_else(|| SelectionError::MissingTypeRoots {
                owner: entrypoint.to_owned(),
            })?;
        if !value
            .as_ref()
            .is_some_and(|literal| literal_conforms(arena, type_id, literal))
        {
            return Err(SelectionError::InvalidBinding {
                binding: name.clone(),
                expected: parameter
                    .value
                    .as_ref()
                    .and_then(|value| value.r#type.as_ref())
                    .map_or_else(|| "a valid Const value".to_owned(), format_type),
            });
        }
        if is_mapping_type(arena, type_id) {
            verify_mapping_literal(name, value.as_ref().expect("binding was checked"))?;
        }
    }
    Ok(())
}

fn is_mapping_type(arena: &TypeArena, type_id: TypeId) -> bool {
    let type_id = arena.strip_stage(type_id);
    let Some(SemanticType::Array { element, shape }) = arena.get(type_id) else {
        return false;
    };
    matches!(
        arena.get(*element),
        Some(SemanticType::Record { name, fields })
            if shape.len() == 1
                && name == "__qamomile_mapping_entry__"
                && fields.len() == 2
                && fields[0].0 == "key"
                && fields[1].0 == "value"
    )
}

fn verify_mapping_literal(
    binding: &str,
    literal: &wire::LiteralProto,
) -> Result<(), SelectionError> {
    use wire::literal_proto::Value;

    let Some(Value::TupleValue(entries)) = literal.value.as_ref() else {
        return Ok(());
    };
    let mut keys = HashSet::with_capacity(entries.values.len());
    for entry in &entries.values {
        let Some(Value::MappingValue(fields)) = entry.value.as_ref() else {
            continue;
        };
        let Some(key) = fields
            .entries
            .iter()
            .find(|field| field.key == "key")
            .and_then(|field| field.value.as_ref())
        else {
            continue;
        };
        if !keys.insert(key.encode_to_vec()) {
            return Err(SelectionError::DuplicateMappingKey {
                binding: binding.to_owned(),
            });
        }
    }
    Ok(())
}

fn format_type(type_spec: &wire::TypeProto) -> String {
    use wire::type_proto::Kind;

    let arguments = || {
        type_spec
            .arguments
            .iter()
            .map(format_type)
            .collect::<Vec<_>>()
            .join(", ")
    };
    let dimensions = || {
        type_spec
            .dimensions
            .iter()
            .map(|dimension| match dimension.value.as_ref() {
                Some(wire::dimension_proto::Value::Fixed(value)) => value.to_string(),
                Some(wire::dimension_proto::Value::Symbol(value)) => value.clone(),
                None => "?".to_owned(),
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    match Kind::try_from(type_spec.kind).ok() {
        Some(Kind::Scalar | Kind::Record) => type_spec.name.clone(),
        Some(Kind::Array) if matches!(type_spec.name.as_str(), "Bits" | "UInt") => {
            format!("{}[{}]", type_spec.name, dimensions())
        }
        Some(Kind::Array) => {
            let constructor = match type_spec.dimensions.len() {
                1 => "Vector",
                2 => "Matrix",
                _ => "Tensor",
            };
            format!("{constructor}[{}, {}]", arguments(), dimensions())
        }
        Some(Kind::Product) => format!("({})", arguments()),
        Some(Kind::Quantum) => format!("Q[{}]", arguments()),
        Some(Kind::Staged) => format!("{}[{}]", type_spec.name, arguments()),
        Some(Kind::Fixed) => format!("UFixed[{}]", dimensions()),
        Some(Kind::Function) => format!("QKernel[{}]", arguments()),
        Some(Kind::Unspecified) | None => "a valid Const value".to_owned(),
    }
}

fn infer_extents(
    parameters: &BTreeMap<&str, &wire::ParameterProto>,
    provided: &mut BTreeMap<String, Option<wire::LiteralProto>>,
) -> Result<(), SelectionError> {
    let integer_dimensions = parameters
        .iter()
        .filter_map(|(name, parameter)| {
            parameter
                .value
                .as_ref()
                .and_then(|value| value.r#type.as_ref())
                .filter(|type_spec| is_const_int(type_spec))
                .map(|_| (*name).to_owned())
        })
        .collect::<BTreeSet<_>>();
    let mut observed = BTreeMap::new();
    for (name, value) in provided.iter() {
        let Some(parameter) = parameters.get(name.as_str()) else {
            continue;
        };
        let Some((type_spec, literal)) = parameter
            .value
            .as_ref()
            .and_then(|value| value.r#type.as_ref())
            .zip(value.as_ref())
        else {
            continue;
        };
        observe_type_extents(type_spec, literal, &integer_dimensions, &mut observed)?;
    }
    for (name, extent) in observed {
        if let Some(explicit) = provided.get(&name) {
            if explicit.as_ref().and_then(integer_literal) != Some(BigInt::from(extent)) {
                return Err(SelectionError::ExplicitExtentConflict {
                    dimension: name,
                    expected: extent,
                });
            }
        } else {
            let value = i64::try_from(extent).map_err(|_| SelectionError::ConflictingExtents {
                dimension: name.clone(),
            })?;
            provided.insert(
                name,
                Some(wire::LiteralProto {
                    value: Some(wire::literal_proto::Value::IntValue(value)),
                }),
            );
        }
    }
    Ok(())
}

fn observe_type_extents(
    mut type_spec: &wire::TypeProto,
    literal: &wire::LiteralProto,
    integer_dimensions: &BTreeSet<String>,
    observed: &mut BTreeMap<String, usize>,
) -> Result<(), SelectionError> {
    use wire::literal_proto::Value;
    use wire::type_proto::Kind;

    while Kind::try_from(type_spec.kind).ok() == Some(Kind::Staged) {
        let Some(inner) = type_spec.arguments.first() else {
            return Ok(());
        };
        type_spec = inner;
    }
    let Some(Value::TupleValue(sequence)) = literal.value.as_ref() else {
        return Ok(());
    };
    match Kind::try_from(type_spec.kind).ok() {
        Some(Kind::Array) if type_spec.name == "Array" => observe_array_extents(
            &type_spec.dimensions,
            &sequence.values,
            integer_dimensions,
            observed,
        ),
        Some(Kind::Array) if matches!(type_spec.name.as_str(), "Bits" | "UInt") => {
            if let Some(dimension) = type_spec.dimensions.first() {
                observe_dimension(
                    dimension,
                    sequence.values.len(),
                    integer_dimensions,
                    observed,
                )?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn observe_array_extents(
    dimensions: &[wire::DimensionProto],
    values: &[wire::LiteralProto],
    integer_dimensions: &BTreeSet<String>,
    observed: &mut BTreeMap<String, usize>,
) -> Result<(), SelectionError> {
    let Some((dimension, remaining)) = dimensions.split_first() else {
        return Ok(());
    };
    observe_dimension(dimension, values.len(), integer_dimensions, observed)?;
    if !remaining.is_empty() {
        for value in values {
            if let Some(wire::literal_proto::Value::TupleValue(sequence)) = value.value.as_ref() {
                observe_array_extents(remaining, &sequence.values, integer_dimensions, observed)?;
            }
        }
    }
    Ok(())
}

fn observe_dimension(
    dimension: &wire::DimensionProto,
    extent: usize,
    integer_dimensions: &BTreeSet<String>,
    observed: &mut BTreeMap<String, usize>,
) -> Result<(), SelectionError> {
    let Some(wire::dimension_proto::Value::Symbol(name)) = dimension.value.as_ref() else {
        return Ok(());
    };
    if !integer_dimensions.contains(name) {
        return Ok(());
    }
    if observed
        .insert(name.clone(), extent)
        .is_some_and(|prior| prior != extent)
    {
        return Err(SelectionError::ConflictingExtents {
            dimension: name.clone(),
        });
    }
    Ok(())
}

fn is_const_int(mut type_spec: &wire::TypeProto) -> bool {
    use wire::type_proto::Kind;

    if Kind::try_from(type_spec.kind).ok() != Some(Kind::Staged) || type_spec.name != "Const" {
        return false;
    }
    let Some(inner) = type_spec.arguments.first() else {
        return false;
    };
    type_spec = inner;
    Kind::try_from(type_spec.kind).ok() == Some(Kind::Scalar) && type_spec.name == "Int"
}

fn integer_literal(literal: &wire::LiteralProto) -> Option<BigInt> {
    match literal.value.as_ref()? {
        wire::literal_proto::Value::IntValue(value) => Some(BigInt::from(*value)),
        wire::literal_proto::Value::BigintValue(value) => Some(BigInt::from_signed_bytes_be(value)),
        _ => None,
    }
}

fn reachable(
    entry: usize,
    graph: &StaticCallGraph,
    entrypoint: &str,
) -> Result<BTreeSet<usize>, SelectionError> {
    let mut pending = vec![entry];
    let mut reached = BTreeSet::new();
    while let Some(index) = pending.pop() {
        if !reached.insert(index) {
            continue;
        }
        let callees = graph
            .callees(index)
            .ok_or_else(|| SelectionError::UnresolvedCallable {
                symbol: graph.symbol(index).unwrap_or_default().to_owned(),
                entrypoint: entrypoint.to_owned(),
            })?;
        pending.extend(callees.iter().copied());
    }
    Ok(reached)
}
