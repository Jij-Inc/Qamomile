//! Declaration, portable-value, and specialized entrypoint verification.
//!
//! Types and literal encodings are closed by earlier passes. This pass ties
//! those independent facts together: defaults inhabit their parameter types,
//! module values inhabit their declarations, and a program entrypoint carries
//! exactly one concrete binding for every specialization-time parameter.

use std::collections::{BTreeSet, HashMap, HashSet};

use thiserror::Error;

use crate::equations::literal_conforms;
use crate::operations::LocallyCheckedOperations;
use crate::types::{Dimension, Stage, TypeArena};
use crate::verification::QCoreVerificationWork;
use crate::wire;

/// Failure of one declaration or specialized program ABI.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub(crate) enum DeclarationError {
    /// Two formal parameters expose the same source-level name.
    #[error("Q1017: function {function:?} has duplicate parameter {parameter:?}")]
    DuplicateParameter { function: String, parameter: String },
    /// A parameter default is outside its declared classical type.
    #[error("Q1057: default for parameter {parameter:?} of {function:?} has the wrong type")]
    InvalidDefault { function: String, parameter: String },
    /// A module constant value is outside its declared type.
    #[error("Q1058: module constant {constant:?} has the wrong type")]
    InvalidConstant { constant: String },
    /// Two entrypoint bindings expose the same source-level name.
    #[error("Q1019: duplicate Program Const binding {binding:?}")]
    DuplicateBinding { binding: String },
    /// A binding does not name an outer `Const` entrypoint parameter.
    #[error("Q1020: Program binding {binding:?} is not a Const parameter")]
    NonConstBinding { binding: String },
    /// A binding value is outside its entrypoint parameter type.
    #[error("Q1021: Program binding {binding:?} has the wrong type")]
    InvalidBinding { binding: String },
    /// One specialization-time entrypoint parameter remains unresolved.
    #[error("Q1022: Program has unresolved Const parameter {parameter:?}")]
    MissingBinding { parameter: String },
    /// A selected entrypoint still contains a symbolic shape after binding.
    #[error("Q1059: Program entrypoint retains symbolic dimensions: {dimensions}")]
    SymbolicEntrypoint {
        /// Sorted source spellings of unresolved dependent dimensions.
        dimensions: String,
    },
    /// Retained type roots are unexpectedly incomplete.
    #[error("Q4002: missing retained declaration type roots for {owner:?}")]
    MissingTypeRoots { owner: String },
}

/// Verify declarations and the fully specialized entrypoint ABI.
///
/// # Errors
///
/// Returns [`DeclarationError`] for ambiguous parameter names, ill-typed
/// defaults/constants/bindings, incomplete Const bindings, or symbolic types
/// surviving in the selected concrete entrypoint.
#[cfg(test)]
pub(crate) fn verify(
    package: &wire::QCorePackageProto,
    arena: &mut TypeArena,
    operations: &LocallyCheckedOperations,
) -> Result<(), DeclarationError> {
    let mut work = QCoreVerificationWork::default();
    verify_with_work(package, arena, operations, &mut work)
}

pub(crate) fn verify_with_work(
    package: &wire::QCorePackageProto,
    arena: &mut TypeArena,
    operations: &LocallyCheckedOperations,
    work: &mut QCoreVerificationWork,
) -> Result<(), DeclarationError> {
    verify_module_with_work(package, arena, operations, work)?;
    verify_entrypoint(package, arena, operations, true, true, work)
}

pub(crate) fn verify_module_with_work(
    package: &wire::QCorePackageProto,
    arena: &mut TypeArena,
    operations: &LocallyCheckedOperations,
    work: &mut QCoreVerificationWork,
) -> Result<(), DeclarationError> {
    verify_constants(package, arena, work)?;
    for function in &package.functions {
        verify_parameters(function, arena, operations, work)?;
    }
    Ok(())
}

/// Verify declarations while retaining unbound specialization parameters.
///
/// # Errors
///
/// Returns [`DeclarationError`] for ambiguous parameter names, ill-typed
/// defaults/constants, or malformed partial Const bindings. Symbolic
/// entrypoint dimensions are deliberately retained for analysis.
#[cfg(test)]
pub(crate) fn verify_symbolic(
    package: &wire::QCorePackageProto,
    arena: &mut TypeArena,
    operations: &LocallyCheckedOperations,
) -> Result<(), DeclarationError> {
    let mut work = QCoreVerificationWork::default();
    verify_symbolic_with_work(package, arena, operations, &mut work)
}

pub(crate) fn verify_symbolic_with_work(
    package: &wire::QCorePackageProto,
    arena: &mut TypeArena,
    operations: &LocallyCheckedOperations,
    work: &mut QCoreVerificationWork,
) -> Result<(), DeclarationError> {
    verify_module_with_work(package, arena, operations, work)?;
    verify_entrypoint(package, arena, operations, false, false, work)
}

pub(crate) fn verify_specialization_request_with_work(
    package: &wire::QCorePackageProto,
    arena: &mut TypeArena,
    operations: &LocallyCheckedOperations,
    work: &mut QCoreVerificationWork,
) -> Result<(), DeclarationError> {
    verify_module_with_work(package, arena, operations, work)?;
    verify_entrypoint(package, arena, operations, true, false, work)
}

fn verify_constants(
    package: &wire::QCorePackageProto,
    arena: &mut TypeArena,
    work: &mut QCoreVerificationWork,
) -> Result<(), DeclarationError> {
    let scope = arena.intern_scope(&format!("{}::<constants>", package.module_name));
    for constant in &package.constants {
        work.charge_declarations(1);
        let type_id = arena
            .resolve_wire_type(
                constant
                    .r#type
                    .as_ref()
                    .ok_or_else(|| DeclarationError::MissingTypeRoots {
                        owner: constant.name.clone(),
                    })?,
                &scope,
            )
            .map_err(|_| DeclarationError::MissingTypeRoots {
                owner: constant.name.clone(),
            })?;
        if !constant
            .value
            .as_ref()
            .is_some_and(|value| literal_conforms(arena, type_id, value))
        {
            return Err(DeclarationError::InvalidConstant {
                constant: constant.name.clone(),
            });
        }
    }
    Ok(())
}

fn verify_parameters(
    function: &wire::FunctionProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
    work: &mut QCoreVerificationWork,
) -> Result<(), DeclarationError> {
    work.charge_declarations(1);
    let values = operations.value_types(&function.symbol).ok_or_else(|| {
        DeclarationError::MissingTypeRoots {
            owner: function.symbol.clone(),
        }
    })?;
    let mut names = HashSet::with_capacity(function.parameters.len());
    for parameter in &function.parameters {
        work.charge_declarations(1);
        if !names.insert(parameter.name.as_str()) {
            return Err(DeclarationError::DuplicateParameter {
                function: function.symbol.clone(),
                parameter: parameter.name.clone(),
            });
        }
        if !parameter.has_default {
            continue;
        }
        let type_id = parameter
            .value
            .as_ref()
            .and_then(|value| values.get(&value.id))
            .copied()
            .ok_or_else(|| DeclarationError::MissingTypeRoots {
                owner: function.symbol.clone(),
            })?;
        if !parameter
            .default_value
            .as_ref()
            .is_some_and(|value| literal_conforms(arena, type_id, value))
        {
            return Err(DeclarationError::InvalidDefault {
                function: function.symbol.clone(),
                parameter: parameter.name.clone(),
            });
        }
    }
    Ok(())
}

fn verify_entrypoint(
    package: &wire::QCorePackageProto,
    arena: &TypeArena,
    operations: &LocallyCheckedOperations,
    require_complete_bindings: bool,
    require_concrete_types: bool,
    work: &mut QCoreVerificationWork,
) -> Result<(), DeclarationError> {
    let entry = package
        .functions
        .iter()
        .inspect(|_| work.charge_declarations(1))
        .find(|function| function.symbol == package.entrypoint)
        .ok_or_else(|| DeclarationError::MissingTypeRoots {
            owner: package.entrypoint.clone(),
        })?;
    let values = operations.value_types(&entry.symbol).ok_or_else(|| {
        DeclarationError::MissingTypeRoots {
            owner: entry.symbol.clone(),
        }
    })?;
    work.charge_declarations(entry.parameters.len());
    let parameters = entry
        .parameters
        .iter()
        .map(|parameter| (parameter.name.as_str(), parameter))
        .collect::<HashMap<_, _>>();
    let mut bound = HashSet::with_capacity(package.const_bindings.len());
    for binding in &package.const_bindings {
        work.charge_declarations(1);
        if !bound.insert(binding.name.as_str()) {
            return Err(DeclarationError::DuplicateBinding {
                binding: binding.name.clone(),
            });
        }
        let Some(parameter) = parameters.get(binding.name.as_str()).copied() else {
            return Err(DeclarationError::NonConstBinding {
                binding: binding.name.clone(),
            });
        };
        let type_id = parameter
            .value
            .as_ref()
            .and_then(|value| values.get(&value.id))
            .copied()
            .ok_or_else(|| DeclarationError::MissingTypeRoots {
                owner: entry.symbol.clone(),
            })?;
        if arena.stage_of(type_id) != Some(Stage::Const) {
            return Err(DeclarationError::NonConstBinding {
                binding: binding.name.clone(),
            });
        }
        if !binding
            .value
            .as_ref()
            .is_some_and(|value| literal_conforms(arena, type_id, value))
        {
            return Err(DeclarationError::InvalidBinding {
                binding: binding.name.clone(),
            });
        }
    }
    for parameter in &entry.parameters {
        work.charge_declarations(1);
        let type_id = parameter
            .value
            .as_ref()
            .and_then(|value| values.get(&value.id))
            .copied()
            .ok_or_else(|| DeclarationError::MissingTypeRoots {
                owner: entry.symbol.clone(),
            })?;
        if require_complete_bindings
            && arena.stage_of(type_id) == Some(Stage::Const)
            && !bound.contains(parameter.name.as_str())
        {
            return Err(DeclarationError::MissingBinding {
                parameter: parameter.name.clone(),
            });
        }
    }
    let result = operations.result_type(&entry.symbol).ok_or_else(|| {
        DeclarationError::MissingTypeRoots {
            owner: entry.symbol.clone(),
        }
    })?;
    if require_concrete_types {
        work.charge_declarations(1_usize.saturating_add(values.len()));
        let dimensions = arena
            .symbolic_dimensions_in(std::iter::once(result).chain(values.values().copied()))
            .into_iter()
            .filter_map(|dimension| match dimension {
                Dimension::Symbol { name, .. } => Some(name.to_string()),
                Dimension::Fixed(_) => None,
            })
            .collect::<BTreeSet<_>>();
        if !dimensions.is_empty() {
            return Err(DeclarationError::SymbolicEntrypoint {
                dimensions: dimensions.into_iter().collect::<Vec<_>>().join(", "),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::validate_program_operations_with_types;
    use crate::types::validate_program_types;

    fn scalar(name: &str) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Scalar.into(),
            name: name.to_owned(),
            ..Default::default()
        }
    }

    fn staged(stage: &str, inner: wire::TypeProto) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Staged.into(),
            name: stage.to_owned(),
            arguments: vec![inner],
            ..Default::default()
        }
    }

    fn word(dimension: wire::dimension_proto::Value) -> wire::TypeProto {
        wire::TypeProto {
            kind: wire::type_proto::Kind::Array.into(),
            name: "Bits".to_owned(),
            arguments: vec![scalar("Bit")],
            dimensions: vec![wire::DimensionProto {
                value: Some(dimension),
            }],
            ..Default::default()
        }
    }

    fn value(id: u64, type_spec: wire::TypeProto) -> wire::ValueProto {
        wire::ValueProto {
            id,
            r#type: Some(type_spec),
            ..Default::default()
        }
    }

    fn parameter(
        name: &str,
        value: wire::ValueProto,
        default: Option<wire::LiteralProto>,
    ) -> wire::ParameterProto {
        wire::ParameterProto {
            name: name.to_owned(),
            value: Some(value),
            has_default: default.is_some(),
            default_value: default,
        }
    }

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

    fn function(
        parameters: Vec<wire::ParameterProto>,
        result_type: wire::TypeProto,
        returned: u64,
    ) -> wire::FunctionProto {
        wire::FunctionProto {
            symbol: "test.main".to_owned(),
            parameters,
            result_type: Some(result_type),
            body: Some(wire::BlockProto {
                terminator: Some(wire::TerminatorProto {
                    kind: "return".to_owned(),
                    operands: vec![returned],
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn check(package: &wire::QCorePackageProto) -> Result<(), DeclarationError> {
        let mut arena = validate_program_types(package).expect("test types are valid");
        let operations = validate_program_operations_with_types(package, &mut arena)
            .expect("test operations are locally valid");
        verify(package, &mut arena, &operations)
    }

    fn check_symbolic(package: &wire::QCorePackageProto) -> Result<(), DeclarationError> {
        let mut arena = validate_program_types(package).expect("test types are valid");
        let operations = validate_program_operations_with_types(package, &mut arena)
            .expect("test operations are locally valid");
        verify_symbolic(package, &mut arena, &operations)
    }

    #[test]
    fn rejects_duplicate_parameter_names_and_wrong_defaults() {
        let duplicate = wire::QCorePackageProto {
            functions: vec![function(
                vec![
                    parameter("value", value(1, scalar("Int")), None),
                    parameter("value", value(2, scalar("Int")), None),
                ],
                scalar("Int"),
                1,
            )],
            entrypoint: "test.main".to_owned(),
            ..Default::default()
        };
        assert_eq!(
            check(&duplicate),
            Err(DeclarationError::DuplicateParameter {
                function: "test.main".to_owned(),
                parameter: "value".to_owned(),
            })
        );

        let wrong_default = wire::QCorePackageProto {
            functions: vec![function(
                vec![parameter(
                    "value",
                    value(1, scalar("Int")),
                    Some(boolean(true)),
                )],
                scalar("Int"),
                1,
            )],
            entrypoint: "test.main".to_owned(),
            ..Default::default()
        };
        assert_eq!(
            check(&wrong_default),
            Err(DeclarationError::InvalidDefault {
                function: "test.main".to_owned(),
                parameter: "value".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_a_module_constant_outside_its_declared_type() {
        let package = wire::QCorePackageProto {
            functions: vec![function(
                vec![parameter("value", value(1, scalar("Int")), None)],
                scalar("Int"),
                1,
            )],
            constants: vec![wire::ConstantProto {
                name: "answer".to_owned(),
                r#type: Some(scalar("Int")),
                value: Some(boolean(true)),
            }],
            entrypoint: "test.main".to_owned(),
            ..Default::default()
        };

        assert_eq!(
            check(&package),
            Err(DeclarationError::InvalidConstant {
                constant: "answer".to_owned(),
            })
        );
    }

    #[test]
    fn requires_exact_typed_const_bindings() {
        let const_parameter = parameter("width", value(1, staged("Const", scalar("Int"))), None);
        let mut package = wire::QCorePackageProto {
            functions: vec![function(
                vec![const_parameter],
                staged("Const", scalar("Int")),
                1,
            )],
            entrypoint: "test.main".to_owned(),
            ..Default::default()
        };
        assert_eq!(
            check(&package),
            Err(DeclarationError::MissingBinding {
                parameter: "width".to_owned(),
            })
        );

        package.const_bindings.push(wire::BindingProto {
            name: "width".to_owned(),
            value: Some(boolean(true)),
        });
        assert_eq!(
            check(&package),
            Err(DeclarationError::InvalidBinding {
                binding: "width".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_symbolic_entrypoint_shapes_after_binding() {
        use wire::dimension_proto::Value::Symbol;

        let package = wire::QCorePackageProto {
            functions: vec![function(
                vec![
                    parameter("width", value(1, staged("Const", scalar("Int"))), None),
                    parameter("bits", value(2, word(Symbol("width".to_owned()))), None),
                    parameter("other", value(3, word(Symbol("alpha".to_owned()))), None),
                ],
                word(Symbol("width".to_owned())),
                2,
            )],
            entrypoint: "test.main".to_owned(),
            const_bindings: vec![wire::BindingProto {
                name: "width".to_owned(),
                value: Some(integer(3)),
            }],
            ..Default::default()
        };

        assert_eq!(
            check(&package),
            Err(DeclarationError::SymbolicEntrypoint {
                dimensions: "alpha, width".to_owned(),
            })
        );
    }

    #[test]
    fn symbolic_verification_retains_unbound_const_dimensions() {
        use wire::dimension_proto::Value::Symbol;

        let package = wire::QCorePackageProto {
            functions: vec![function(
                vec![
                    parameter("width", value(1, staged("Const", scalar("Int"))), None),
                    parameter("bits", value(2, word(Symbol("width".to_owned()))), None),
                ],
                word(Symbol("width".to_owned())),
                2,
            )],
            entrypoint: "test.main".to_owned(),
            ..Default::default()
        };

        assert_eq!(
            check(&package),
            Err(DeclarationError::MissingBinding {
                parameter: "width".to_owned(),
            })
        );
        assert_eq!(check_symbolic(&package), Ok(()));
    }
}
