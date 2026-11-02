//! Native invocation binding against the portable artifact parameter ABI.

use std::collections::{HashMap, HashSet};

use prost::Message;
use thiserror::Error;

use crate::budget;
use crate::equations::literal_conforms;
use crate::literals::validate_literal_references;
use crate::types::{Dimension, ScalarType, SemanticType, TypeArena, TypeId};
use crate::wire;

const MAX_INVOCATION_ABI_BYTES: usize = 16 * 1024 * 1024;
const MAX_INVOCATION_BINDINGS_BYTES: usize = 64 * 1024 * 1024;

/// Maximum scalar leaves exposed by one artifact parameter ABI.
pub const MAX_SCALAR_PARAMETERS: usize = 1_000_000;

/// Maximum aggregate path components materialized by one scalar ABI projection.
pub const MAX_SCALAR_PARAMETER_PATH_COMPONENTS: usize = 10_000_000;

/// A malformed parameter ABI or invocation rejected before endpoint launch.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum InvocationBindingError {
    /// The persistent ABI cannot be decoded canonically.
    #[error("Q8135: invalid native invocation ABI: {0}")]
    InvalidAbi(String),
    /// The portable launch mapping cannot be decoded canonically.
    #[error("Q8135: invalid native invocation values: {0}")]
    InvalidValues(String),
    /// One or more supplied names are absent from the artifact ABI.
    #[error("Q8101: unknown artifact parameters: {0}")]
    Unknown(String),
    /// One or more required slots have no supplied or default value.
    #[error("Q8102: missing artifact parameters: {0}")]
    Missing(String),
    /// A supplied or default value does not inhabit its declared type.
    #[error("Q8109: parameter {name:?} disagrees with its artifact type")]
    TypeMismatch {
        /// Parameter name identifying the rejected value.
        name: String,
    },
}

/// One component of an aggregate parameter's scalar access path.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ScalarParameterPathComponent {
    /// A sequence, word, or positional-product index.
    Index(u64),
    /// A structural record field name.
    Field(String),
}

/// One ordered scalar leaf exposed to circuit SDKs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScalarParameterLeaf {
    /// Artifact parameter slot name.
    name: String,
    /// Row-major aggregate access path from the slot root.
    path: Box<[ScalarParameterPathComponent]>,
    /// Closed semantic scalar type at the path.
    scalar_type: ScalarType,
}

impl ScalarParameterLeaf {
    /// Return the artifact parameter slot name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the row-major aggregate access path from the slot root.
    #[must_use]
    pub fn path(&self) -> &[ScalarParameterPathComponent] {
        &self.path
    }

    /// Return the closed semantic scalar type at the path.
    #[must_use]
    pub const fn scalar_type(&self) -> ScalarType {
        self.scalar_type
    }
}

/// One ordered scalar SDK input paired with its checked portable value.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarParameterBinding {
    /// Scalar parameter identity and aggregate access path.
    parameter: ScalarParameterLeaf,
    /// Canonical standalone `LiteralProto` bytes for the scalar value.
    literal: Vec<u8>,
}

impl ScalarParameterBinding {
    /// Return the checked scalar parameter identity.
    #[must_use]
    pub const fn parameter(&self) -> &ScalarParameterLeaf {
        &self.parameter
    }

    /// Return canonical standalone `LiteralProto` bytes for the scalar value.
    #[must_use]
    pub fn literal(&self) -> &[u8] {
        &self.literal
    }
}

/// Failure to project a portable parameter ABI into bounded scalar leaves.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ScalarParameterProjectionError {
    /// The persistent parameter ABI is malformed or semantically invalid.
    #[error(transparent)]
    InvalidAbi(#[from] InvocationBindingError),
    /// A dependent dimension cannot be expanded for a fixed circuit SDK ABI.
    #[error("Q7242: circuit parameters require fixed shapes")]
    DynamicShape,
    /// A semantic type has no scalar representation in circuit SDK ABIs.
    #[error("Q7245: circuit parameter type has no scalar ABI")]
    UnsupportedType,
    /// The expansion would allocate unreasonable scalar metadata.
    #[error("Q7246: circuit scalar parameter projection exceeds its bounded work budget")]
    WorkBudgetExceeded,
}

/// Failure to bind and flatten portable invocation values for a circuit SDK.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ScalarParameterBindingError {
    /// Invocation names, defaults, or values violate the artifact ABI.
    #[error(transparent)]
    Invocation(#[from] InvocationBindingError),
    /// The artifact ABI cannot be represented as bounded scalar SDK inputs.
    #[error(transparent)]
    Projection(#[from] ScalarParameterProjectionError),
    /// A checked aggregate value disagrees with its projected scalar path.
    #[error("Q8135: native scalar parameter binding is internally inconsistent")]
    InconsistentProjection,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ScalarProjectionSize {
    leaves: usize,
    path_components: usize,
}

struct CheckedParameterSlot {
    slot: wire::ParameterSlotProto,
    type_id: TypeId,
}

struct CheckedParameterAbi {
    arena: TypeArena,
    slots: Vec<CheckedParameterSlot>,
}

/// Bind exact portable launch values to a canonical artifact parameter ABI.
///
/// The returned payload is a canonical `LiteralProto` mapping. Defaults are
/// materialized in Rust, and every value is checked through the same semantic
/// type arena and literal relation used by native program verification.
///
/// # Errors
///
/// Returns [`InvocationBindingError`] when either payload is malformed, names
/// are missing or unknown, defaults are inconsistent, or a value is ill-typed.
pub fn bind_invocation(
    parameter_abi: &[u8],
    supplied_values: &[u8],
) -> Result<Vec<u8>, InvocationBindingError> {
    let checked = check_parameter_abi(parameter_abi)?;
    let supplied = decode_values(supplied_values)?;
    let resolved = bind_checked_invocation(&checked, &supplied)?;
    Ok(encode_bound_values(resolved))
}

fn bind_checked_invocation(
    checked: &CheckedParameterAbi,
    supplied: &[wire::literal_proto::Entry],
) -> Result<Vec<wire::literal_proto::Entry>, InvocationBindingError> {
    let slot_names = checked
        .slots
        .iter()
        .map(|checked_slot| checked_slot.slot.name.as_str())
        .collect::<HashSet<_>>();
    let supplied_by_name = supplied
        .iter()
        .map(|entry| (entry.key.as_str(), entry))
        .collect::<HashMap<_, _>>();
    let mut unknown = supplied_by_name
        .keys()
        .filter(|name| !slot_names.contains(**name))
        .copied()
        .collect::<Vec<_>>();
    unknown.sort_unstable();
    if !unknown.is_empty() {
        return Err(InvocationBindingError::Unknown(unknown.join(", ")));
    }
    let missing = checked
        .slots
        .iter()
        .map(|checked_slot| &checked_slot.slot)
        .filter(|slot| !slot.has_default && !supplied_by_name.contains_key(slot.name.as_str()))
        .map(|slot| slot.name.as_str())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(InvocationBindingError::Missing(missing.join(", ")));
    }

    let mut resolved = Vec::with_capacity(checked.slots.len());
    for checked_slot in &checked.slots {
        let slot = &checked_slot.slot;
        let value = if let Some(entry) = supplied_by_name.get(slot.name.as_str()) {
            entry.value.as_ref()
        } else if slot.has_default {
            slot.default_value.as_ref()
        } else {
            None
        }
        .ok_or_else(|| {
            InvocationBindingError::InvalidAbi(format!(
                "parameter {:?} has no selected default",
                slot.name
            ))
        })?;
        if !literal_conforms(&checked.arena, checked_slot.type_id, value) {
            return Err(InvocationBindingError::TypeMismatch {
                name: slot.name.clone(),
            });
        }
        resolved.push(wire::literal_proto::Entry {
            key: slot.name.clone(),
            value: Some(value.clone()),
        });
    }
    resolved.sort_by(|left, right| left.key.cmp(&right.key));
    Ok(resolved)
}

fn encode_bound_values(entries: Vec<wire::literal_proto::Entry>) -> Vec<u8> {
    wire::LiteralProto {
        value: Some(wire::literal_proto::Value::MappingValue(
            wire::literal_proto::Mapping { entries },
        )),
    }
    .encode_to_vec()
}

/// Expand a canonical artifact parameter ABI into deterministic scalar leaves.
///
/// Arrays use row-major order, words use source bit order, and products and
/// records preserve their semantic field order. The projection is rejected
/// before materialization when either its leaf count or total path metadata
/// exceeds a fixed work budget.
///
/// # Errors
///
/// Returns [`ScalarParameterProjectionError`] when the ABI is malformed, a
/// shape is dependent, a type has no circuit-scalar ABI, or expansion exceeds
/// its bounded work budget.
pub fn project_scalar_parameters(
    parameter_abi: &[u8],
) -> Result<Vec<ScalarParameterLeaf>, ScalarParameterProjectionError> {
    let checked = check_parameter_abi(parameter_abi)?;
    project_checked_scalar_parameters(&checked)
}

fn project_checked_scalar_parameters(
    checked: &CheckedParameterAbi,
) -> Result<Vec<ScalarParameterLeaf>, ScalarParameterProjectionError> {
    let mut total = ScalarProjectionSize::default();
    for slot in &checked.slots {
        total = add_projection_sizes(total, scalar_projection_size(&checked.arena, slot.type_id)?)?;
    }

    let mut leaves = Vec::with_capacity(total.leaves);
    let mut path = Vec::new();
    for slot in &checked.slots {
        collect_scalar_parameters(
            &checked.arena,
            slot.type_id,
            &slot.slot.name,
            &mut path,
            &mut leaves,
        )?;
    }
    debug_assert_eq!(leaves.len(), total.leaves);
    Ok(leaves)
}

/// Bind invocation values and flatten them into deterministic scalar SDK inputs.
///
/// Binding, default materialization, aggregate traversal, and leaf ordering all
/// remain inside the native ABI boundary. Arrays use row-major order, words use
/// source bit order, and products and records retain semantic field order.
///
/// # Errors
///
/// Returns [`ScalarParameterBindingError`] when binding fails, the ABI cannot
/// be projected to bounded scalar inputs, or checked values and paths disagree.
pub fn bind_scalar_parameters(
    parameter_abi: &[u8],
    supplied_values: &[u8],
) -> Result<Vec<ScalarParameterBinding>, ScalarParameterBindingError> {
    let checked = check_parameter_abi(parameter_abi)?;
    let supplied = decode_values(supplied_values)?;
    let bound_entries = bind_checked_invocation(&checked, &supplied)?;
    let values = bound_entries
        .iter()
        .filter_map(|entry| {
            entry
                .value
                .as_ref()
                .map(|value| (entry.key.as_str(), value))
        })
        .collect::<HashMap<_, _>>();
    let parameters = project_checked_scalar_parameters(&checked)?;

    parameters
        .into_iter()
        .map(|parameter| {
            let root = values
                .get(parameter.name.as_str())
                .copied()
                .ok_or(ScalarParameterBindingError::InconsistentProjection)?;
            let value = parameter
                .path
                .iter()
                .try_fold(root, project_literal_component)
                .ok_or(ScalarParameterBindingError::InconsistentProjection)?;
            Ok(ScalarParameterBinding {
                parameter,
                literal: value.encode_to_vec(),
            })
        })
        .collect()
}

pub(crate) fn project_literal_component<'a>(
    value: &'a wire::LiteralProto,
    component: &ScalarParameterPathComponent,
) -> Option<&'a wire::LiteralProto> {
    match (value.value.as_ref()?, component) {
        (
            wire::literal_proto::Value::TupleValue(sequence),
            ScalarParameterPathComponent::Index(index),
        ) => sequence.values.get(usize::try_from(*index).ok()?),
        (
            wire::literal_proto::Value::MappingValue(mapping),
            ScalarParameterPathComponent::Field(field),
        ) => mapping
            .entries
            .binary_search_by(|entry| entry.key.as_str().cmp(field))
            .ok()
            .and_then(|index| mapping.entries[index].value.as_ref()),
        _ => None,
    }
}

fn check_parameter_abi(payload: &[u8]) -> Result<CheckedParameterAbi, InvocationBindingError> {
    let abi = decode_abi(payload)?;
    let mut arena = TypeArena::default();
    let scope = arena.intern_scope("<artifact-invocation>");
    let mut slots = Vec::with_capacity(abi.slots.len());
    let mut names = HashSet::with_capacity(abi.slots.len());
    for slot in abi.slots {
        if slot.name.is_empty() {
            return Err(InvocationBindingError::InvalidAbi(
                "empty parameter name".to_owned(),
            ));
        }
        if !names.insert(slot.name.clone()) {
            return Err(InvocationBindingError::InvalidAbi(
                "duplicate parameter slot".to_owned(),
            ));
        }
        let type_spec = slot.r#type.as_ref().ok_or_else(|| {
            InvocationBindingError::InvalidAbi(format!(
                "parameter {:?} has no semantic type",
                slot.name
            ))
        })?;
        let type_id = arena
            .resolve_wire_type(type_spec, &scope)
            .map_err(InvocationBindingError::InvalidAbi)?;
        crate::artifact_abi::require_classical(&arena, type_id)
            .map_err(|error| InvocationBindingError::InvalidAbi(error.to_string()))?;
        if slot
            .default_value
            .as_ref()
            .is_some_and(|default| !literal_conforms(&arena, type_id, default))
        {
            return Err(InvocationBindingError::InvalidAbi(format!(
                "default for parameter {:?} disagrees with its type",
                slot.name
            )));
        }
        slots.push(CheckedParameterSlot { slot, type_id });
    }
    Ok(CheckedParameterAbi { arena, slots })
}

/// Validate one canonical artifact parameter ABI without binding values.
pub(crate) fn validate_parameter_abi(payload: &[u8]) -> Result<(), InvocationBindingError> {
    check_parameter_abi(payload).map(|_| ())
}

fn scalar_projection_size(
    arena: &TypeArena,
    root: TypeId,
) -> Result<ScalarProjectionSize, ScalarParameterProjectionError> {
    match arena.get(arena.strip_stage(root)) {
        Some(SemanticType::Scalar(_)) => Ok(ScalarProjectionSize {
            leaves: 1,
            path_components: 0,
        }),
        Some(SemanticType::Word {
            width: Dimension::Fixed(width),
            ..
        }) => {
            let width = usize::try_from(*width)
                .map_err(|_| ScalarParameterProjectionError::WorkBudgetExceeded)?;
            bounded_projection_size(width, width)
        }
        Some(SemanticType::Word { .. }) => Err(ScalarParameterProjectionError::DynamicShape),
        Some(SemanticType::Array { element, shape }) => {
            let child = scalar_projection_size(arena, *element)?;
            let mut repetitions = 1_usize;
            for dimension in shape {
                let Dimension::Fixed(extent) = dimension else {
                    return Err(ScalarParameterProjectionError::DynamicShape);
                };
                repetitions = repetitions
                    .checked_mul(
                        usize::try_from(*extent)
                            .map_err(|_| ScalarParameterProjectionError::WorkBudgetExceeded)?,
                    )
                    .ok_or(ScalarParameterProjectionError::WorkBudgetExceeded)?;
                if repetitions > MAX_SCALAR_PARAMETERS {
                    return Err(ScalarParameterProjectionError::WorkBudgetExceeded);
                }
            }
            repeated_projection_size(child, repetitions, shape.len())
        }
        Some(SemanticType::Product(elements)) => {
            elements
                .iter()
                .try_fold(ScalarProjectionSize::default(), |total, element| {
                    add_projection_sizes(
                        total,
                        add_projection_depth(scalar_projection_size(arena, *element)?, 1)?,
                    )
                })
        }
        Some(SemanticType::Record { fields, .. }) => {
            fields
                .iter()
                .try_fold(ScalarProjectionSize::default(), |total, (_, field_type)| {
                    add_projection_sizes(
                        total,
                        add_projection_depth(scalar_projection_size(arena, *field_type)?, 1)?,
                    )
                })
        }
        Some(
            SemanticType::Fixed { .. }
            | SemanticType::Quantum(_)
            | SemanticType::Staged { .. }
            | SemanticType::Function { .. },
        )
        | None => Err(ScalarParameterProjectionError::UnsupportedType),
    }
}

fn bounded_projection_size(
    leaves: usize,
    path_components: usize,
) -> Result<ScalarProjectionSize, ScalarParameterProjectionError> {
    if leaves > MAX_SCALAR_PARAMETERS || path_components > MAX_SCALAR_PARAMETER_PATH_COMPONENTS {
        return Err(ScalarParameterProjectionError::WorkBudgetExceeded);
    }
    Ok(ScalarProjectionSize {
        leaves,
        path_components,
    })
}

fn add_projection_sizes(
    left: ScalarProjectionSize,
    right: ScalarProjectionSize,
) -> Result<ScalarProjectionSize, ScalarParameterProjectionError> {
    bounded_projection_size(
        left.leaves
            .checked_add(right.leaves)
            .ok_or(ScalarParameterProjectionError::WorkBudgetExceeded)?,
        left.path_components
            .checked_add(right.path_components)
            .ok_or(ScalarParameterProjectionError::WorkBudgetExceeded)?,
    )
}

fn add_projection_depth(
    size: ScalarProjectionSize,
    depth: usize,
) -> Result<ScalarProjectionSize, ScalarParameterProjectionError> {
    bounded_projection_size(
        size.leaves,
        size.path_components
            .checked_add(
                size.leaves
                    .checked_mul(depth)
                    .ok_or(ScalarParameterProjectionError::WorkBudgetExceeded)?,
            )
            .ok_or(ScalarParameterProjectionError::WorkBudgetExceeded)?,
    )
}

fn repeated_projection_size(
    size: ScalarProjectionSize,
    repetitions: usize,
    added_depth: usize,
) -> Result<ScalarProjectionSize, ScalarParameterProjectionError> {
    let repeated = ScalarProjectionSize {
        leaves: size
            .leaves
            .checked_mul(repetitions)
            .ok_or(ScalarParameterProjectionError::WorkBudgetExceeded)?,
        path_components: size
            .path_components
            .checked_mul(repetitions)
            .ok_or(ScalarParameterProjectionError::WorkBudgetExceeded)?,
    };
    add_projection_depth(repeated, added_depth)
}

fn collect_scalar_parameters(
    arena: &TypeArena,
    root: TypeId,
    name: &str,
    path: &mut Vec<ScalarParameterPathComponent>,
    output: &mut Vec<ScalarParameterLeaf>,
) -> Result<(), ScalarParameterProjectionError> {
    match arena.get(arena.strip_stage(root)) {
        Some(SemanticType::Scalar(scalar_type)) => output.push(ScalarParameterLeaf {
            name: name.to_owned(),
            path: path.clone().into_boxed_slice(),
            scalar_type: *scalar_type,
        }),
        Some(SemanticType::Word {
            width: Dimension::Fixed(width),
            ..
        }) => {
            for index in 0..*width {
                path.push(ScalarParameterPathComponent::Index(index));
                output.push(ScalarParameterLeaf {
                    name: name.to_owned(),
                    path: path.clone().into_boxed_slice(),
                    scalar_type: ScalarType::Bit,
                });
                path.pop();
            }
        }
        Some(SemanticType::Array { element, shape }) => {
            collect_array_scalar_parameters(arena, *element, shape, 0, name, path, output)?;
        }
        Some(SemanticType::Product(elements)) => {
            for (index, element) in elements.iter().enumerate() {
                path.push(ScalarParameterPathComponent::Index(index as u64));
                collect_scalar_parameters(arena, *element, name, path, output)?;
                path.pop();
            }
        }
        Some(SemanticType::Record { fields, .. }) => {
            for (field, field_type) in fields {
                path.push(ScalarParameterPathComponent::Field(field.clone()));
                collect_scalar_parameters(arena, *field_type, name, path, output)?;
                path.pop();
            }
        }
        Some(
            SemanticType::Word { .. }
            | SemanticType::Fixed { .. }
            | SemanticType::Quantum(_)
            | SemanticType::Staged { .. }
            | SemanticType::Function { .. },
        )
        | None => return Err(ScalarParameterProjectionError::UnsupportedType),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn collect_array_scalar_parameters(
    arena: &TypeArena,
    element: TypeId,
    shape: &[Dimension],
    axis: usize,
    name: &str,
    path: &mut Vec<ScalarParameterPathComponent>,
    output: &mut Vec<ScalarParameterLeaf>,
) -> Result<(), ScalarParameterProjectionError> {
    if axis == shape.len() {
        return collect_scalar_parameters(arena, element, name, path, output);
    }
    let Dimension::Fixed(extent) = shape[axis] else {
        return Err(ScalarParameterProjectionError::DynamicShape);
    };
    for index in 0..extent {
        path.push(ScalarParameterPathComponent::Index(index));
        collect_array_scalar_parameters(arena, element, shape, axis + 1, name, path, output)?;
        path.pop();
    }
    Ok(())
}

fn decode_abi(payload: &[u8]) -> Result<wire::ParameterAbiProto, InvocationBindingError> {
    if payload.len() > MAX_INVOCATION_ABI_BYTES {
        return Err(InvocationBindingError::InvalidAbi(
            "parameter ABI exceeds its byte budget".to_owned(),
        ));
    }
    budget::preflight_parameter_abi(payload)
        .map_err(|error| InvocationBindingError::InvalidAbi(preflight_reason(error)))?;
    let value = wire::ParameterAbiProto::decode(payload)
        .map_err(|error| InvocationBindingError::InvalidAbi(error.to_string()))?;
    if value.encode_to_vec() != payload {
        return Err(InvocationBindingError::InvalidAbi(
            "parameter ABI is not canonical".to_owned(),
        ));
    }
    let defaults = value
        .slots
        .iter()
        .filter_map(|slot| slot.default_value.as_ref())
        .collect::<Vec<_>>();
    validate_literal_references(defaults).map_err(InvocationBindingError::InvalidAbi)?;
    for slot in &value.slots {
        if slot.has_default != slot.default_value.is_some() {
            return Err(InvocationBindingError::InvalidAbi(format!(
                "parameter {:?} has an inconsistent default",
                slot.name
            )));
        }
    }
    Ok(value)
}

fn decode_values(
    payload: &[u8],
) -> Result<Vec<wire::literal_proto::Entry>, InvocationBindingError> {
    if payload.len() > MAX_INVOCATION_BINDINGS_BYTES {
        return Err(InvocationBindingError::InvalidValues(
            "launch values exceed their byte budget".to_owned(),
        ));
    }
    budget::preflight_literal(payload)
        .map_err(|error| InvocationBindingError::InvalidValues(preflight_reason(error)))?;
    let value = wire::LiteralProto::decode(payload)
        .map_err(|error| InvocationBindingError::InvalidValues(error.to_string()))?;
    if value.encode_to_vec() != payload {
        return Err(InvocationBindingError::InvalidValues(
            "launch values are not canonical".to_owned(),
        ));
    }
    validate_literal_references(vec![&value]).map_err(InvocationBindingError::InvalidValues)?;
    let Some(wire::literal_proto::Value::MappingValue(mapping)) = value.value else {
        return Err(InvocationBindingError::InvalidValues(
            "launch values must be a mapping".to_owned(),
        ));
    };
    Ok(mapping.entries)
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

    fn scalar(name: &str) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Scalar.into(),
            name: name.to_owned(),
            ..Default::default()
        }
    }

    fn param(inner: wire::TypeProto) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Staged.into(),
            name: "Param".to_owned(),
            arguments: vec![inner],
            ..Default::default()
        }
    }

    fn fixed_dimension(value: u64) -> wire::DimensionProto {
        wire::DimensionProto {
            value: Some(wire::dimension_proto::Value::Fixed(value)),
        }
    }

    fn array(element: wire::TypeProto, shape: &[u64]) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Array.into(),
            name: "Array".to_owned(),
            arguments: vec![element],
            dimensions: shape.iter().copied().map(fixed_dimension).collect(),
            ..Default::default()
        }
    }

    fn word(width: wire::dimension_proto::Value) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Array.into(),
            name: "Bits".to_owned(),
            arguments: vec![scalar("Bit")],
            dimensions: vec![wire::DimensionProto { value: Some(width) }],
            ..Default::default()
        }
    }

    fn product(elements: Vec<wire::TypeProto>) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Product.into(),
            arguments: elements,
            ..Default::default()
        }
    }

    fn record(name: &str, fields: Vec<(&str, wire::TypeProto)>) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Record.into(),
            name: name.to_owned(),
            field_names: fields
                .iter()
                .map(|(field, _)| (*field).to_owned())
                .collect(),
            arguments: fields
                .into_iter()
                .map(|(_, field_type)| field_type)
                .collect(),
            ..Default::default()
        }
    }

    fn parameter_abi(slots: Vec<(&str, wire::TypeProto)>) -> Vec<u8> {
        wire::ParameterAbiProto {
            slots: slots
                .into_iter()
                .map(|(name, r#type)| wire::ParameterSlotProto {
                    name: name.to_owned(),
                    r#type: Some(param(r#type)),
                    default_value: None,
                    has_default: false,
                })
                .collect(),
        }
        .encode_to_vec()
    }

    fn literal(value: wire::literal_proto::Value) -> wire::LiteralProto {
        wire::LiteralProto { value: Some(value) }
    }

    fn sequence(values: Vec<wire::LiteralProto>) -> wire::LiteralProto {
        literal(wire::literal_proto::Value::TupleValue(
            wire::literal_proto::Sequence { values },
        ))
    }

    fn mapping_literal(entries: Vec<(&str, wire::LiteralProto)>) -> wire::LiteralProto {
        literal(wire::literal_proto::Value::MappingValue(
            wire::literal_proto::Mapping {
                entries: entries
                    .into_iter()
                    .map(|(key, value)| wire::literal_proto::Entry {
                        key: key.to_owned(),
                        value: Some(value),
                    })
                    .collect(),
            },
        ))
    }

    fn mapping(entries: Vec<(&str, wire::LiteralProto)>) -> Vec<u8> {
        mapping_literal(entries).encode_to_vec()
    }

    fn abi() -> Vec<u8> {
        wire::ParameterAbiProto {
            slots: vec![
                wire::ParameterSlotProto {
                    name: "enabled".to_owned(),
                    r#type: Some(param(scalar("Bool"))),
                    default_value: None,
                    has_default: false,
                },
                wire::ParameterSlotProto {
                    name: "theta".to_owned(),
                    r#type: Some(param(scalar("Real"))),
                    default_value: Some(literal(wire::literal_proto::Value::RealValue(0.25))),
                    has_default: true,
                },
            ],
        }
        .encode_to_vec()
    }

    #[test]
    fn binds_defaults_and_returns_a_canonical_mapping() {
        let supplied = mapping(vec![(
            "enabled",
            literal(wire::literal_proto::Value::BoolValue(true)),
        )]);

        let payload = bind_invocation(&abi(), &supplied).expect("valid invocation");
        let decoded = wire::LiteralProto::decode(payload.as_slice()).expect("bound mapping");
        let Some(wire::literal_proto::Value::MappingValue(mapping)) = decoded.value else {
            panic!("bound values must be a mapping");
        };

        assert_eq!(
            mapping
                .entries
                .iter()
                .map(|entry| entry.key.as_str())
                .collect::<Vec<_>>(),
            vec!["enabled", "theta"]
        );
        assert_eq!(
            mapping.entries[1].value,
            Some(literal(wire::literal_proto::Value::RealValue(0.25)))
        );
    }

    #[test]
    fn preserves_the_complete_binary64_real_parameter_domain() {
        let encoded_abi = parameter_abi(vec![("value", scalar("Real"))]);

        for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let supplied = mapping(vec![(
                "value",
                literal(wire::literal_proto::Value::RealValue(value)),
            )]);
            let payload =
                bind_invocation(&encoded_abi, &supplied).expect("Real value is well typed");
            let decoded = wire::LiteralProto::decode(payload.as_slice()).expect("bound mapping");
            let Some(wire::literal_proto::Value::MappingValue(mapping)) = decoded.value else {
                panic!("bound values must be a mapping");
            };
            let Some(wire::LiteralProto {
                value: Some(wire::literal_proto::Value::RealValue(actual)),
            }) = mapping.entries[0].value
            else {
                panic!("bound Real value must retain its wire representation");
            };

            assert_eq!(actual.to_bits(), value.to_bits());
        }
    }

    #[test]
    fn rejects_unknown_and_missing_names_before_type_checking() {
        let unknown = mapping(vec![(
            "ghost",
            literal(wire::literal_proto::Value::IntValue(1)),
        )]);
        assert_eq!(
            bind_invocation(&abi(), &unknown),
            Err(InvocationBindingError::Unknown("ghost".to_owned()))
        );

        let empty = mapping(Vec::new());
        assert_eq!(
            bind_invocation(&abi(), &empty),
            Err(InvocationBindingError::Missing("enabled".to_owned()))
        );
    }

    #[test]
    fn rejects_values_outside_the_native_semantic_type_relation() {
        let supplied = mapping(vec![(
            "enabled",
            literal(wire::literal_proto::Value::IntValue(1)),
        )]);

        assert_eq!(
            bind_invocation(&abi(), &supplied),
            Err(InvocationBindingError::TypeMismatch {
                name: "enabled".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_an_ill_typed_default_even_when_the_slot_is_supplied() {
        let malformed = wire::ParameterAbiProto {
            slots: vec![wire::ParameterSlotProto {
                name: "enabled".to_owned(),
                r#type: Some(param(scalar("Bool"))),
                default_value: Some(literal(wire::literal_proto::Value::IntValue(1))),
                has_default: true,
            }],
        }
        .encode_to_vec();
        let supplied = mapping(vec![(
            "enabled",
            literal(wire::literal_proto::Value::BoolValue(true)),
        )]);

        assert!(matches!(
            bind_invocation(&malformed, &supplied),
            Err(InvocationBindingError::InvalidAbi(message))
                if message.contains("default for parameter \"enabled\"")
        ));
    }

    #[test]
    fn projects_aggregate_parameters_in_semantic_row_major_order() {
        let encoded = parameter_abi(vec![
            ("matrix", array(scalar("Real"), &[2, 2])),
            (
                "options",
                record(
                    "Options",
                    vec![
                        ("enabled", scalar("Bool")),
                        (
                            "payload",
                            product(vec![
                                scalar("Int"),
                                word(wire::dimension_proto::Value::Fixed(2)),
                            ]),
                        ),
                    ],
                ),
            ),
        ]);

        let projected = project_scalar_parameters(&encoded).expect("fixed scalar projection");
        let summary = projected
            .iter()
            .map(|leaf| (leaf.name.as_str(), leaf.path.to_vec(), leaf.scalar_type))
            .collect::<Vec<_>>();

        assert_eq!(
            summary,
            vec![
                (
                    "matrix",
                    vec![
                        ScalarParameterPathComponent::Index(0),
                        ScalarParameterPathComponent::Index(0),
                    ],
                    ScalarType::Real,
                ),
                (
                    "matrix",
                    vec![
                        ScalarParameterPathComponent::Index(0),
                        ScalarParameterPathComponent::Index(1),
                    ],
                    ScalarType::Real,
                ),
                (
                    "matrix",
                    vec![
                        ScalarParameterPathComponent::Index(1),
                        ScalarParameterPathComponent::Index(0),
                    ],
                    ScalarType::Real,
                ),
                (
                    "matrix",
                    vec![
                        ScalarParameterPathComponent::Index(1),
                        ScalarParameterPathComponent::Index(1),
                    ],
                    ScalarType::Real,
                ),
                (
                    "options",
                    vec![ScalarParameterPathComponent::Field("enabled".to_owned())],
                    ScalarType::Bool,
                ),
                (
                    "options",
                    vec![
                        ScalarParameterPathComponent::Field("payload".to_owned()),
                        ScalarParameterPathComponent::Index(0),
                    ],
                    ScalarType::Int,
                ),
                (
                    "options",
                    vec![
                        ScalarParameterPathComponent::Field("payload".to_owned()),
                        ScalarParameterPathComponent::Index(1),
                        ScalarParameterPathComponent::Index(0),
                    ],
                    ScalarType::Bit,
                ),
                (
                    "options",
                    vec![
                        ScalarParameterPathComponent::Field("payload".to_owned()),
                        ScalarParameterPathComponent::Index(1),
                        ScalarParameterPathComponent::Index(1),
                    ],
                    ScalarType::Bit,
                ),
            ]
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn binds_aggregate_scalar_values_in_the_same_native_semantic_order() {
        let options_default = mapping_literal(vec![
            (
                "alpha",
                sequence(vec![
                    literal(wire::literal_proto::Value::IntValue(9)),
                    sequence(vec![
                        literal(wire::literal_proto::Value::BoolValue(false)),
                        literal(wire::literal_proto::Value::BoolValue(true)),
                    ]),
                ]),
            ),
            ("zeta", literal(wire::literal_proto::Value::BoolValue(true))),
        ]);
        let encoded_abi = wire::ParameterAbiProto {
            slots: vec![
                wire::ParameterSlotProto {
                    name: "matrix".to_owned(),
                    r#type: Some(param(array(scalar("Real"), &[2, 2]))),
                    default_value: None,
                    has_default: false,
                },
                wire::ParameterSlotProto {
                    name: "options".to_owned(),
                    r#type: Some(param(record(
                        "Options",
                        vec![
                            ("zeta", scalar("Bool")),
                            (
                                "alpha",
                                product(vec![
                                    scalar("Int"),
                                    word(wire::dimension_proto::Value::Fixed(2)),
                                ]),
                            ),
                        ],
                    ))),
                    default_value: Some(options_default),
                    has_default: true,
                },
            ],
        }
        .encode_to_vec();
        let supplied = mapping(vec![(
            "matrix",
            sequence(vec![
                sequence(vec![
                    literal(wire::literal_proto::Value::RealValue(1.0)),
                    literal(wire::literal_proto::Value::RealValue(2.0)),
                ]),
                sequence(vec![
                    literal(wire::literal_proto::Value::RealValue(3.0)),
                    literal(wire::literal_proto::Value::RealValue(4.0)),
                ]),
            ]),
        )]);

        let bindings =
            bind_scalar_parameters(&encoded_abi, &supplied).expect("valid scalar bindings");
        let summary = bindings
            .iter()
            .map(|binding| {
                let value = wire::LiteralProto::decode(binding.literal.as_slice())
                    .expect("scalar literal")
                    .value
                    .expect("selected scalar value");
                (
                    binding.parameter.name.as_str(),
                    binding.parameter.path.to_vec(),
                    value,
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(
            summary,
            vec![
                (
                    "matrix",
                    vec![
                        ScalarParameterPathComponent::Index(0),
                        ScalarParameterPathComponent::Index(0),
                    ],
                    wire::literal_proto::Value::RealValue(1.0),
                ),
                (
                    "matrix",
                    vec![
                        ScalarParameterPathComponent::Index(0),
                        ScalarParameterPathComponent::Index(1),
                    ],
                    wire::literal_proto::Value::RealValue(2.0),
                ),
                (
                    "matrix",
                    vec![
                        ScalarParameterPathComponent::Index(1),
                        ScalarParameterPathComponent::Index(0),
                    ],
                    wire::literal_proto::Value::RealValue(3.0),
                ),
                (
                    "matrix",
                    vec![
                        ScalarParameterPathComponent::Index(1),
                        ScalarParameterPathComponent::Index(1),
                    ],
                    wire::literal_proto::Value::RealValue(4.0),
                ),
                (
                    "options",
                    vec![ScalarParameterPathComponent::Field("zeta".to_owned())],
                    wire::literal_proto::Value::BoolValue(true),
                ),
                (
                    "options",
                    vec![
                        ScalarParameterPathComponent::Field("alpha".to_owned()),
                        ScalarParameterPathComponent::Index(0),
                    ],
                    wire::literal_proto::Value::IntValue(9),
                ),
                (
                    "options",
                    vec![
                        ScalarParameterPathComponent::Field("alpha".to_owned()),
                        ScalarParameterPathComponent::Index(1),
                        ScalarParameterPathComponent::Index(0),
                    ],
                    wire::literal_proto::Value::BoolValue(false),
                ),
                (
                    "options",
                    vec![
                        ScalarParameterPathComponent::Field("alpha".to_owned()),
                        ScalarParameterPathComponent::Index(1),
                        ScalarParameterPathComponent::Index(1),
                    ],
                    wire::literal_proto::Value::BoolValue(true),
                ),
            ]
        );
    }

    #[test]
    fn rejects_dependent_and_unbounded_scalar_parameter_projections() {
        let dynamic = parameter_abi(vec![(
            "bits",
            word(wire::dimension_proto::Value::Symbol("width".to_owned())),
        )]);
        assert_eq!(
            project_scalar_parameters(&dynamic),
            Err(ScalarParameterProjectionError::DynamicShape)
        );

        let unbounded = parameter_abi(vec![(
            "bits",
            word(wire::dimension_proto::Value::Fixed(
                MAX_SCALAR_PARAMETERS as u64 + 1,
            )),
        )]);
        assert_eq!(
            project_scalar_parameters(&unbounded),
            Err(ScalarParameterProjectionError::WorkBudgetExceeded)
        );
    }
}
