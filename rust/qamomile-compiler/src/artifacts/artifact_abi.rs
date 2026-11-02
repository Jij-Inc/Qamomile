//! Native derivation of the persistent launch and result ABIs.

use std::collections::BTreeSet;

use prost::Message;
use thiserror::Error;

use crate::operations::LocallyCheckedOperations;
use crate::types::{Stage, TypeArena};
use crate::wire;

/// Canonical persistent ABI payloads derived from one verified entrypoint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DerivedArtifactAbi {
    parameter_abi: Vec<u8>,
    result_abi: Vec<u8>,
    sampled_width: u32,
}

impl DerivedArtifactAbi {
    /// Return the canonical `ParameterABIProto` payload.
    #[must_use]
    pub fn parameter_abi(&self) -> &[u8] {
        &self.parameter_abi
    }

    /// Return the canonical `ResultABIProto` payload.
    #[must_use]
    pub fn result_abi(&self) -> &[u8] {
        &self.result_abi
    }

    /// Return the fixed logical sample width, or zero for value execution.
    #[must_use]
    pub const fn sampled_width(&self) -> u32 {
        self.sampled_width
    }
}

/// Failure to derive an executable artifact ABI from verified semantic facts.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ArtifactAbiError {
    /// Quantum ownership cannot cross an execution entry boundary.
    #[error(
        "Q6304: execution entry must have classical inputs and outputs; inspect quantum components as circuits or compose them with measurement"
    )]
    QuantumEntryBoundary,
    /// The closed package does not retain its selected entrypoint declaration.
    #[error("Q4002: missing verified entrypoint declaration")]
    MissingEntrypoint,
    /// A verified entrypoint parameter or result has no retained semantic type.
    #[error("Q4002: missing verified entrypoint type")]
    MissingType,
    /// A result mixes sampled data with a shape that has no fixed bit encoding.
    #[error(
        "Q7011: sampled results must be a pure Product, Record, or tree of fixed-width Bool, Bit, Bits, UInt, or UFixed leaves"
    )]
    InvalidSampledResult,
    /// A fixed sampled result cannot be represented by the persistent ABI width.
    #[error("Q7012: sampled result width exceeds the persistent ABI limit")]
    ResultWidthOverflow,
}

/// Failure to adopt a source and its classical execution layouts.
#[derive(Debug, Error)]
pub enum ExecutionInterfaceError {
    /// The source entry is not an executable signature.
    #[error(transparent)]
    Source(#[from] ArtifactAbiError),
    /// The retained parameter layout is not a classical launch ABI.
    #[error(transparent)]
    Parameters(#[from] crate::InvocationBindingError),
    /// The retained result layout is not a classical result ABI.
    #[error(transparent)]
    Result(#[from] crate::ResultCodecError),
}

pub(crate) fn validate_execution_entry(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
) -> Result<(), ArtifactAbiError> {
    let entry = package
        .functions
        .iter()
        .find(|function| function.symbol == package.entrypoint)
        .ok_or(ArtifactAbiError::MissingEntrypoint)?;
    let value_types = operations
        .value_types(&entry.symbol)
        .ok_or(ArtifactAbiError::MissingType)?;
    for parameter in &entry.parameters {
        let value = parameter
            .value
            .as_ref()
            .ok_or(ArtifactAbiError::MissingType)?;
        let type_id = value_types
            .get(&value.id)
            .ok_or(ArtifactAbiError::MissingType)?;
        require_classical(arena, *type_id)?;
    }
    require_classical(
        arena,
        operations
            .result_type(&entry.symbol)
            .ok_or(ArtifactAbiError::MissingType)?,
    )
}

pub(crate) fn require_classical(
    arena: &TypeArena,
    ty: crate::types::TypeId,
) -> Result<(), ArtifactAbiError> {
    if arena.is_quantum(ty) {
        Err(ArtifactAbiError::QuantumEntryBoundary)
    } else {
        Ok(())
    }
}

pub(crate) fn derive(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
    has_measurement: bool,
) -> Result<DerivedArtifactAbi, ArtifactAbiError> {
    let entry = package
        .functions
        .iter()
        .find(|function| function.symbol == package.entrypoint)
        .ok_or(ArtifactAbiError::MissingEntrypoint)?;
    let value_types = operations
        .value_types(&entry.symbol)
        .ok_or(ArtifactAbiError::MissingType)?;
    let bound = package
        .const_bindings
        .iter()
        .map(|binding| binding.name.as_str())
        .collect::<BTreeSet<_>>();
    let mut slots = Vec::with_capacity(entry.parameters.len());
    for parameter in &entry.parameters {
        let value = parameter
            .value
            .as_ref()
            .ok_or(ArtifactAbiError::MissingType)?;
        let type_id = value_types
            .get(&value.id)
            .ok_or(ArtifactAbiError::MissingType)?;
        if bound.contains(parameter.name.as_str()) || arena.stage_of(*type_id) == Some(Stage::Const)
        {
            continue;
        }
        slots.push(wire::ParameterSlotProto {
            name: parameter.name.clone(),
            r#type: Some(value.r#type.clone().ok_or(ArtifactAbiError::MissingType)?),
            default_value: parameter.default_value.clone(),
            has_default: parameter.has_default,
        });
    }
    let result_type = entry
        .result_type
        .clone()
        .ok_or(ArtifactAbiError::MissingType)?;
    let result_id = operations
        .result_type(&entry.symbol)
        .ok_or(ArtifactAbiError::MissingType)?;
    let logical_width = if has_measurement {
        match arena
            .sampled_width(result_id)
            .map_err(|_| ArtifactAbiError::ResultWidthOverflow)?
        {
            Some(width) => width,
            None if arena.contains_stage(result_id, Stage::Shot) => {
                return Err(ArtifactAbiError::InvalidSampledResult);
            }
            None => 0,
        }
    } else {
        0
    };
    let native_width =
        u32::try_from(logical_width).map_err(|_| ArtifactAbiError::ResultWidthOverflow)?;
    let source_order = (0..native_width).collect();
    Ok(DerivedArtifactAbi {
        parameter_abi: wire::ParameterAbiProto { slots }.encode_to_vec(),
        result_abi: wire::ResultAbiProto {
            r#type: Some(result_type),
            source_order,
            native_width,
        }
        .encode_to_vec(),
        sampled_width: native_width,
    })
}
